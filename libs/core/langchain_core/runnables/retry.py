"""Runnable that retries a Runnable if it fails.

This module provides RunnableRetry, a wrapper that implements retry logic with
exponential backoff for handling transient failures in LCEL (LangChain Expression
Language) chains. This is particularly useful for network calls to external APIs
that may experience temporary failures due to rate limiting, connection issues,
or server errors.

The retry mechanism is built on the tenacity library (https://tenacity.readthedocs.io),
which provides flexible configuration for retry behavior including:
- Exponential backoff with jitter to prevent thundering herd problems
- Selective retry based on exception types
- Maximum attempt limits
- Configurable backoff parameters (initial delay, maximum delay, exponential base)

Common Use Cases:
- API Rate Limiting: Retry on 429 (Too Many Requests) errors with exponential backoff
- Network Failures: Retry on connection errors and 5xx server errors
- Transient Service Errors: Retry on temporary service unavailability

Best Practices:
- Keep retry scope as small as possible (retry only the failing component)
- Only retry transient errors (avoid retrying validation errors or bad requests)
- Use jitter to prevent synchronized retry storms across multiple clients
- Set reasonable maximum attempt limits to avoid infinite retry loops

Source: libs/core/langchain_core/runnables/retry.py
"""

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
)

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from typing_extensions import TypedDict, override

from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.config import RunnableConfig, patch_config
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )

    T = TypeVar("T", CallbackManagerForChainRun, AsyncCallbackManagerForChainRun)
U = TypeVar("U")


class ExponentialJitterParams(TypedDict, total=False):
    """Parameters for `tenacity.wait_exponential_jitter`.
    
    These parameters control the exponential backoff behavior with jitter.
    The wait time between retries is calculated as:
    wait = min(max, initial * (exp_base ** attempt)) + random.uniform(0, jitter)
    
    Attributes:
        initial: Initial wait time in seconds. Default is 1.0.
        max: Maximum wait time in seconds to cap exponential growth. Default is 10.0.
        exp_base: Base for exponential backoff calculation. Default is 2.0.
        jitter: Random delay added to prevent thundering herd, in seconds. Default is 0.0.
    
    Example configurations for different scenarios:
    
        API Rate Limiting (429 errors):
            # Aggressive backoff to respect rate limits
            {"initial": 2.0, "max": 60.0, "exp_base": 2.0, "jitter": 5.0}
            # Wait times: ~2s, ~4s, ~8s, ~16s, ~32s, ~60s (capped)
        
        Network Failures (connection errors, 5xx errors):
            # Moderate backoff for transient network issues
            {"initial": 1.0, "max": 30.0, "exp_base": 2.0, "jitter": 2.0}
            # Wait times: ~1s, ~2s, ~4s, ~8s, ~16s, ~30s (capped)
        
        Quick Recovery (temporary service blips):
            # Fast retry with minimal backoff
            {"initial": 0.5, "max": 5.0, "exp_base": 1.5, "jitter": 1.0}
            # Wait times: ~0.5s, ~0.75s, ~1.1s, ~1.7s, ~2.5s, ~3.8s, ~5s (capped)
    
    Source: libs/core/langchain_core/runnables/retry.py:35-46
    """

    initial: float
    """Initial wait time in seconds before first retry. Default: 1.0"""
    max: float
    """Maximum wait time in seconds to cap exponential growth. Default: 10.0"""
    exp_base: float
    """Base for exponential backoff (wait = initial * exp_base^attempt). Default: 2.0"""
    jitter: float
    """Random delay in seconds sampled from uniform(0, jitter) to add to wait time. 
    Prevents synchronized retries across multiple clients (thundering herd). Default: 0.0"""


class RunnableRetry(RunnableBindingBase[Input, Output]):  # type: ignore[no-redef]
    """Retry a Runnable if it fails.

    RunnableRetry can be used to add retry logic to any object
    that subclasses the base Runnable.

    Such retries are especially useful for network calls that may fail
    due to transient errors.

    The RunnableRetry is implemented as a RunnableBinding. The easiest
    way to use it is through the `.with_retry()` method on all Runnables.
    
    Retry Mechanism:
        RunnableRetry uses the tenacity library (Retrying/AsyncRetrying) to handle
        retry logic with sophisticated backoff strategies. On each attempt:
        
        1. The wrapped Runnable is invoked with the input
        2. If an exception in retry_exception_types is raised, tenacity catches it
        3. Backoff delay is calculated using exponential_jitter formula
        4. After delay, the next attempt begins (up to max_attempt_number)
        5. If successful, the result is returned
        6. If all attempts fail, the final exception is re-raised
    
    Exception Handling:
        retry_exception_types: Controls which exceptions trigger retry vs immediate re-raise
        - Default: (Exception,) - retries all exceptions
        - Best practice: Only retry transient failures (network errors, rate limits)
        - Non-retryable: Validation errors, authentication failures, bad input
        
        Examples of good retry_exception_types:
        - API rate limiting: (requests.exceptions.HTTPError,) with status code 429 check
        - Network errors: (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
        - Server errors: (requests.exceptions.HTTPError,) with 5xx status codes
    
    Exponential Backoff with Jitter:
        wait_exponential_jitter: Enables exponential backoff to progressively increase
        wait time between retries, with random jitter to prevent thundering herd.
        
        Formula: wait = min(max, initial * (exp_base ** attempt)) + random.uniform(0, jitter)
        
        exponential_jitter_params: Customizes backoff behavior
        - initial: Starting wait time (default 1.0s)
        - max: Maximum wait time cap (default 10.0s)
        - exp_base: Growth rate (default 2.0 for doubling)
        - jitter: Random variance (default 0.0s, recommend 1-5s for distributed systems)
    
    Maximum Attempts:
        max_attempt_number: Limits total retry attempts to prevent infinite loops.
        - Default: 3 attempts (1 initial + 2 retries)
        - Recommendation: 3-5 for API calls, 2-3 for fast-failing operations
    
    Best Practices:
        - Keep retry scope as small as possible (retry only the failing component)
        - Only retry transient errors (avoid retrying validation or logic errors)
        - Use jitter to prevent synchronized retry storms across multiple clients
        - Set reasonable max_attempt_number to avoid excessive delays
        - Monitor retry metrics to detect systemic issues
    
    Type Flow:
        Input → Attempt 1 (exception) → backoff delay → Attempt 2 (exception) → 
        backoff delay → Attempt N (success) → Output
        
        On failure: Input → max_attempt_number attempts → RetryError raised
    
    Callback Integration:
        Each retry attempt creates a child callback manager with tag "retry:attempt:{N}"
        to track individual attempts in tracing systems. The first attempt has no tag,
        subsequent attempts are tagged "retry:attempt:2", "retry:attempt:3", etc.

    Example:
        Here's an example that uses a RunnableLambda to raise an exception

        ```python
        import time


        def foo(input) -> None:
            '''Fake function that raises an exception.'''
            raise ValueError(f"Invoking foo failed. At time {time.time()}")


        runnable = RunnableLambda(foo)

        runnable_with_retries = runnable.with_retry(
            retry_if_exception_type=(ValueError,),  # Retry only on ValueError
            wait_exponential_jitter=True,  # Add jitter to the exponential backoff
            stop_after_attempt=2,  # Try twice
            exponential_jitter_params={"initial": 2},  # if desired, customize backoff
        )

        # The method invocation above is equivalent to the longer form below:

        runnable_with_retries = RunnableRetry(
            bound=runnable,
            retry_exception_types=(ValueError,),
            max_attempt_number=2,
            wait_exponential_jitter=True,
            exponential_jitter_params={"initial": 2},
        )
        ```

    This logic can be used to retry any Runnable, including a chain of Runnables,
    but in general it's best practice to keep the scope of the retry as small as
    possible. For example, if you have a chain of Runnables, you should only retry
    the Runnable that is likely to fail, not the entire chain.

    Example:
        ```python
        from langchain_core.chat_models import ChatOpenAI
        from langchain_core.prompts import PromptTemplate

        template = PromptTemplate.from_template("tell me a joke about {topic}.")
        model = ChatOpenAI(temperature=0.5)

        # Good - only retry the model call
        chain = template | model.with_retry()

        # Bad - retries entire chain including prompt formatting
        chain = template | model
        retryable_chain = chain.with_retry()
        ```
    
    Example - API Rate Limiting:
        ```python
        from langchain_core.chat_models import ChatOpenAI
        from requests.exceptions import HTTPError
        
        # Configure aggressive backoff for rate limiting (429 errors)
        model_with_retry = ChatOpenAI().with_retry(
            retry_if_exception_type=(HTTPError,),
            max_attempt_number=5,
            wait_exponential_jitter=True,
            exponential_jitter_params={
                "initial": 2.0,  # Start with 2 second wait
                "max": 60.0,     # Cap at 60 seconds
                "exp_base": 2.0, # Double each time
                "jitter": 5.0    # Add 0-5 second random jitter
            }
        )
        ```
    
    Example - Network Failures:
        ```python
        from langchain_core.chat_models import ChatAnthropic
        from requests.exceptions import ConnectionError, Timeout
        
        # Moderate backoff for transient network issues
        model_with_retry = ChatAnthropic().with_retry(
            retry_if_exception_type=(ConnectionError, Timeout),
            max_attempt_number=3,
            wait_exponential_jitter=True,
            exponential_jitter_params={
                "initial": 1.0,  # Start with 1 second
                "max": 30.0,     # Cap at 30 seconds
                "jitter": 2.0    # Add 0-2 second jitter
            }
        )
        ```
    
    Source: libs/core/langchain_core/runnables/retry.py:48-112
    """

    retry_exception_types: tuple[type[BaseException], ...] = (Exception,)
    """The exception types to retry on. By default all exceptions are retried.

    In general you should only retry on exceptions that are likely to be
    transient, such as network errors.

    Good exceptions to retry are all server errors (5xx) and selected client
    errors (4xx) such as 429 Too Many Requests.
    """

    wait_exponential_jitter: bool = True
    """Whether to add jitter to the exponential backoff."""

    exponential_jitter_params: ExponentialJitterParams | None = None
    """Parameters for `tenacity.wait_exponential_jitter`. Namely: `initial`,
    `max`, `exp_base`, and `jitter` (all `float` values).
    """

    max_attempt_number: int = 3
    """The maximum number of attempts to retry the Runnable."""

    @property
    def _kwargs_retrying(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}

        if self.max_attempt_number:
            kwargs["stop"] = stop_after_attempt(self.max_attempt_number)

        if self.wait_exponential_jitter:
            kwargs["wait"] = wait_exponential_jitter(
                **(self.exponential_jitter_params or {})
            )

        if self.retry_exception_types:
            kwargs["retry"] = retry_if_exception_type(self.retry_exception_types)

        return kwargs

    def _sync_retrying(self, **kwargs: Any) -> Retrying:
        return Retrying(**self._kwargs_retrying, **kwargs)

    def _async_retrying(self, **kwargs: Any) -> AsyncRetrying:
        return AsyncRetrying(**self._kwargs_retrying, **kwargs)

    @staticmethod
    def _patch_config(
        config: RunnableConfig,
        run_manager: "T",
        retry_state: RetryCallState,
    ) -> RunnableConfig:
        """Create a patched config with child callback manager for retry attempt tracking.
        
        Creates a child callback manager for each retry attempt to enable tracing systems
        to track individual retry attempts separately. The first attempt (attempt 1) uses
        no tag, while subsequent attempts are tagged "retry:attempt:N" where N is the
        attempt number (2, 3, 4, etc.).
        
        This enables observability into retry behavior in tracing systems like LangSmith,
        allowing developers to see exactly how many retries occurred and the latency of
        each attempt.
        
        Args:
            config: The RunnableConfig to patch with new callback manager.
            run_manager: The parent callback manager (CallbackManagerForChainRun or
                AsyncCallbackManagerForChainRun).
            retry_state: The tenacity RetryCallState containing attempt metadata.
        
        Returns:
            A new RunnableConfig with callbacks set to a child callback manager.
            The child manager inherits parent callbacks and adds retry attempt tracking.
        
        Example:
            Internal usage during retry execution:
            
            ```python
            # First attempt (attempt_number=1): No tag added
            config1 = _patch_config(config, run_manager, retry_state)
            # config1.callbacks is child manager without retry tag
            
            # Second attempt (attempt_number=2): Tagged
            config2 = _patch_config(config, run_manager, retry_state)
            # config2.callbacks is child manager with tag "retry:attempt:2"
            ```
        
        Source: libs/core/langchain_core/runnables/retry.py:158-166
        """
        # Extract attempt number from tenacity retry state
        attempt = retry_state.attempt_number
        # Only tag attempts after the first one (2, 3, 4, etc.)
        tag = f"retry:attempt:{attempt}" if attempt > 1 else None
        # Create child callback manager with optional retry tag
        return patch_config(config, callbacks=run_manager.get_child(tag))

    def _patch_config_list(
        self,
        config: list[RunnableConfig],
        run_manager: list["T"],
        retry_state: RetryCallState,
    ) -> list[RunnableConfig]:
        return [
            self._patch_config(c, rm, retry_state)
            for c, rm in zip(config, run_manager, strict=False)
        ]

    def _invoke(
        self,
        input_: Input,
        run_manager: "CallbackManagerForChainRun",
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        """Internal synchronous invoke implementation with retry logic.
        
        Executes the wrapped Runnable with tenacity retry handling. Uses synchronous
        tenacity.Retrying to manage retry attempts with exponential backoff. Each
        attempt is executed within a tenacity attempt context manager, which handles
        exception catching and retry decision logic.
        
        The retry loop continues until either:
        1. An attempt succeeds (no exception raised)
        2. All max_attempt_number attempts are exhausted
        3. An exception not in retry_exception_types is raised (immediately re-raised)
        
        Args:
            input_: The input to pass to the wrapped Runnable.
            run_manager: Callback manager for tracking execution and creating child
                managers for retry attempts.
            config: RunnableConfig with callbacks, tags, metadata for this invocation.
            **kwargs: Additional keyword arguments passed to the wrapped Runnable.
        
        Returns:
            The output from the successful invocation of the wrapped Runnable.
        
        Raises:
            RetryError: If all retry attempts are exhausted and the last attempt failed.
            Exception: If an exception not in retry_exception_types is raised, it is
                immediately re-raised without retry.
        
        Example:
            Internal execution flow:
            
            ```python
            # Attempt 1: Call wrapped runnable with input
            # → Raises ConnectionError → Tenacity catches it
            # → Wait 1 second (exponential backoff)
            
            # Attempt 2: Call wrapped runnable with input (retry:attempt:2 tag)
            # → Raises ConnectionError → Tenacity catches it
            # → Wait 2 seconds (exponential backoff)
            
            # Attempt 3: Call wrapped runnable with input (retry:attempt:3 tag)
            # → Returns result successfully
            # → Return result to caller
            ```
        
        Source: libs/core/langchain_core/runnables/retry.py:179-195
        """
        # Create tenacity Retrying instance with configured parameters (reraise=True
        # means final exception is raised if all attempts fail)
        for attempt in self._sync_retrying(reraise=True):
            with attempt:  # Tenacity attempt context - catches exceptions
                # Invoke wrapped Runnable with patched config (child callback manager)
                result = super().invoke(
                    input_,
                    self._patch_config(config, run_manager, attempt.retry_state),
                    **kwargs,
                )
            # If attempt succeeded (no exception), record result in retry state
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """Invoke the wrapped Runnable with retry logic (synchronous).
        
        Executes the wrapped Runnable with automatic retry on transient failures.
        This is the main entry point for synchronous retry execution. Internally,
        it delegates to _invoke which handles the retry loop using tenacity.
        
        The method will retry failed invocations according to the configured retry
        parameters (max_attempt_number, retry_exception_types, exponential backoff).
        Each retry attempt is tracked with callbacks for observability.
        
        Args:
            input: The input to pass to the wrapped Runnable. Type must match the
                Input type parameter of the wrapped Runnable.
            config: Optional RunnableConfig with callbacks, tags, metadata, and other
                configuration options. If None, uses default configuration.
            **kwargs: Additional keyword arguments passed through to the wrapped
                Runnable's invoke method.
        
        Returns:
            The output from the successful invocation of the wrapped Runnable. Type
            matches the Output type parameter of the wrapped Runnable.
        
        Raises:
            RetryError: If all retry attempts (up to max_attempt_number) are exhausted
                and the operation continues to fail. The original exception is wrapped
                in the RetryError.
            Exception: If an exception not in retry_exception_types is raised, it is
                immediately re-raised without retry attempts.
        
        Example:
            Basic usage with automatic retry:
            
            ```python
            from langchain_core.runnables import RunnableLambda
            import requests
            
            def call_api(topic: str) -> str:
                # May raise ConnectionError on network issues
                response = requests.get(f"https://api.example.com/jokes/{topic}")
                response.raise_for_status()
                return response.json()["joke"]
            
            runnable = RunnableLambda(call_api)
            runnable_with_retry = runnable.with_retry(
                retry_if_exception_type=(requests.exceptions.RequestException,),
                max_attempt_number=3,
                wait_exponential_jitter=True
            )
            
            # Invoke with retry - will automatically retry on connection errors
            result = runnable_with_retry.invoke("python")
            print(result)  # "Why do programmers prefer dark mode? ..."
            ```
        
        Source: libs/core/langchain_core/runnables/retry.py:197-201
        """
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        input_: Input,
        run_manager: "AsyncCallbackManagerForChainRun",
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        """Internal asynchronous invoke implementation with retry logic.
        
        Executes the wrapped Runnable with tenacity async retry handling. Uses async
        tenacity.AsyncRetrying to manage retry attempts with exponential backoff while
        allowing other async tasks to execute during wait periods. Each attempt is
        executed within a tenacity attempt context manager.
        
        The async retry loop continues until either:
        1. An attempt succeeds (no exception raised)
        2. All max_attempt_number attempts are exhausted
        3. An exception not in retry_exception_types is raised (immediately re-raised)
        
        Unlike synchronous retry, async retry properly yields control during backoff
        delays, allowing the event loop to process other tasks.
        
        Args:
            input_: The input to pass to the wrapped Runnable.
            run_manager: Async callback manager for tracking execution and creating
                child managers for retry attempts.
            config: RunnableConfig with callbacks, tags, metadata for this invocation.
            **kwargs: Additional keyword arguments passed to the wrapped Runnable.
        
        Returns:
            The output from the successful invocation of the wrapped Runnable.
        
        Raises:
            RetryError: If all retry attempts are exhausted and the last attempt failed.
            Exception: If an exception not in retry_exception_types is raised, it is
                immediately re-raised without retry.
        
        Example:
            Internal async execution flow:
            
            ```python
            # Attempt 1: Await wrapped runnable with input
            # → Raises asyncio.TimeoutError → Tenacity catches it
            # → Await asyncio.sleep(1.0) - exponential backoff (other tasks can run)
            
            # Attempt 2: Await wrapped runnable with input (retry:attempt:2 tag)
            # → Raises asyncio.TimeoutError → Tenacity catches it
            # → Await asyncio.sleep(2.0) - exponential backoff (other tasks can run)
            
            # Attempt 3: Await wrapped runnable with input (retry:attempt:3 tag)
            # → Returns result successfully
            # → Return result to caller
            ```
        
        Source: libs/core/langchain_core/runnables/retry.py:203-219
        """
        # Create tenacity AsyncRetrying instance with configured parameters
        # (reraise=True means final exception is raised if all attempts fail)
        async for attempt in self._async_retrying(reraise=True):
            with attempt:  # Tenacity attempt context - catches exceptions
                # Await wrapped Runnable with patched config (child callback manager)
                result = await super().ainvoke(
                    input_,
                    self._patch_config(config, run_manager, attempt.retry_state),
                    **kwargs,
                )
            # If attempt succeeded (no exception), record result in retry state
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    @override
    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """Invoke the wrapped Runnable with retry logic (asynchronous).
        
        Executes the wrapped Runnable asynchronously with automatic retry on transient
        failures. This is the main entry point for async retry execution. Internally,
        it delegates to _ainvoke which handles the async retry loop using tenacity's
        AsyncRetrying.
        
        The async implementation properly yields control to the event loop during
        backoff delays, allowing other async tasks to execute concurrently. This is
        particularly important for high-throughput applications where many retries
        may be happening simultaneously.
        
        Args:
            input: The input to pass to the wrapped Runnable. Type must match the
                Input type parameter of the wrapped Runnable.
            config: Optional RunnableConfig with callbacks, tags, metadata, and other
                configuration options. If None, uses default configuration.
            **kwargs: Additional keyword arguments passed through to the wrapped
                Runnable's ainvoke method.
        
        Returns:
            The output from the successful invocation of the wrapped Runnable. Type
            matches the Output type parameter of the wrapped Runnable.
        
        Raises:
            RetryError: If all retry attempts (up to max_attempt_number) are exhausted
                and the operation continues to fail. The original exception is wrapped
                in the RetryError.
            Exception: If an exception not in retry_exception_types is raised, it is
                immediately re-raised without retry attempts.
        
        Example:
            Async usage with automatic retry:
            
            ```python
            import asyncio
            from langchain_core.runnables import RunnableLambda
            import aiohttp
            
            async def call_api_async(topic: str) -> str:
                # May raise asyncio.TimeoutError or aiohttp.ClientError
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://api.example.com/jokes/{topic}",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        return data["joke"]
            
            runnable = RunnableLambda(call_api_async)
            runnable_with_retry = runnable.with_retry(
                retry_if_exception_type=(asyncio.TimeoutError, aiohttp.ClientError),
                max_attempt_number=4,
                wait_exponential_jitter=True,
                exponential_jitter_params={"initial": 1.0, "max": 10.0, "jitter": 2.0}
            )
            
            # Await invoke with retry - retries on timeout/network errors
            result = await runnable_with_retry.ainvoke("python")
            print(result)  # "Why do programmers prefer dark mode? ..."
            ```
        
        Source: libs/core/langchain_core/runnables/retry.py:221-225
        """
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _batch(
        self,
        inputs: list[Input],
        run_manager: list["CallbackManagerForChainRun"],
        config: list[RunnableConfig],
        **kwargs: Any,
    ) -> list[Output | Exception]:
        """Internal synchronous batch implementation with per-item retry logic.
        
        Executes batch processing with intelligent retry handling. Unlike simple batch
        execution, this implementation tracks success/failure per input item and only
        retries the items that failed. This provides efficient retry behavior where
        successful items are preserved across retry attempts.
        
        The retry logic:
        1. Attempt to process all inputs in batch
        2. Track which items succeeded (store in results_map)
        3. If any items failed, retry only those failed items
        4. Repeat until all items succeed or max_attempt_number is reached
        5. Return results in original input order
        
        This per-item retry approach is particularly valuable when processing batches
        of API calls where some may succeed and others fail due to rate limiting or
        transient errors.
        
        Args:
            inputs: List of inputs to process in batch. Each input is passed to the
                wrapped Runnable.
            run_manager: List of callback managers, one per input item, for tracking
                execution of each item.
            config: List of RunnableConfigs, one per input item, with callbacks, tags,
                and metadata for each item's invocation.
            **kwargs: Additional keyword arguments passed to the wrapped Runnable's
                batch method.
        
        Returns:
            List of outputs or exceptions, one per input, in the same order as the
            inputs. Successfully processed items contain Output values, while items
            that failed all retry attempts contain Exception instances.
        
        Raises:
            This method does not raise exceptions directly; exceptions are returned
            in the output list. However, if return_exceptions=False in the public
            batch() method, exceptions will be re-raised there.
        
        Example:
            Internal batch retry flow:
            
            ```python
            # Input: [input1, input2, input3, input4, input5]
            
            # Attempt 1: Process all 5 items
            # → input1: Success (stored in results_map[0])
            # → input2: ConnectionError (needs retry)
            # → input3: Success (stored in results_map[2])
            # → input4: ConnectionError (needs retry)
            # → input5: Success (stored in results_map[4])
            # → Wait 1 second (exponential backoff)
            
            # Attempt 2: Process only failed items [input2, input4]
            # → input2: Success (stored in results_map[1])
            # → input4: ConnectionError (needs retry)
            # → Wait 2 seconds (exponential backoff)
            
            # Attempt 3: Process only failed items [input4]
            # → input4: Success (stored in results_map[3])
            
            # Return: [output1, output2, output3, output4, output5] - all succeeded
            ```
        
        Source: libs/core/langchain_core/runnables/retry.py:227-288
        """
        # Track successful results by original index
        results_map: dict[int, Output] = {}

        # Sentinel value to detect if result was never set
        not_set: list[Output] = []
        result = not_set
        try:
            # Tenacity retry loop (reraise=False for batch - we handle exceptions)
            for attempt in self._sync_retrying():
                with attempt:  # Tenacity attempt context - catches exceptions
                    # Retry for inputs that have not yet succeeded
                    # Determine which original indices remain to be retried
                    remaining_indices = [
                        i for i in range(len(inputs)) if i not in results_map
                    ]
                    if not remaining_indices:
                        break  # All items succeeded, exit retry loop
                    # Extract inputs/configs/managers for items that need retry
                    pending_inputs = [inputs[i] for i in remaining_indices]
                    pending_configs = [config[i] for i in remaining_indices]
                    pending_run_managers = [run_manager[i] for i in remaining_indices]
                    # Invoke underlying batch only on remaining elements
                    # return_exceptions=True ensures exceptions are returned, not raised
                    result = super().batch(
                        pending_inputs,
                        self._patch_config_list(
                            pending_configs, pending_run_managers, attempt.retry_state
                        ),
                        return_exceptions=True,
                        **kwargs,
                    )
                    # Register the results of the inputs that have succeeded, mapping
                    # back to their original indices for proper output ordering
                    first_exception = None
                    for offset, r in enumerate(result):
                        if isinstance(r, Exception):
                            # Track first exception to trigger retry
                            if not first_exception:
                                first_exception = r
                            continue  # Don't store exception in results_map
                        # Map successful result back to original input index
                        orig_idx = remaining_indices[offset]
                        results_map[orig_idx] = r
                    # If any exception occurred, raise it to trigger retry of failed items
                    if first_exception:
                        raise first_exception
                # Record successful batch result in retry state
                if (
                    attempt.retry_state.outcome
                    and not attempt.retry_state.outcome.failed
                ):
                    attempt.retry_state.set_result(result)
        except RetryError as e:
            # All retry attempts exhausted, create exception list for failed items
            if result is not_set:
                result = cast("list[Output]", [e] * len(inputs))

        # Build final output list in original input order
        outputs: list[Output | Exception] = []
        for idx in range(len(inputs)):
            if idx in results_map:
                # Item succeeded at some point, use stored result
                outputs.append(results_map[idx])
            else:
                # Item never succeeded, use exception from final attempt
                outputs.append(result.pop(0))
        return outputs

    @override
    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        """Execute batch processing with intelligent per-item retry logic (synchronous).
        
        Processes multiple inputs in batch with automatic retry for failed items. This
        implementation is more efficient than retrying the entire batch - it tracks
        which individual items succeed and only retries items that failed. This is
        particularly valuable for API rate limiting scenarios where some calls succeed
        while others hit rate limits.
        
        The batch retry behavior:
        - Attempt 1: Process all N inputs
        - Track which items succeeded (preserved for final output)
        - Attempt 2+: Only retry the items that failed
        - Continue until all items succeed or max_attempt_number is reached
        - Return results in original input order
        
        Args:
            inputs: List of inputs to process. Each input is passed to the wrapped
                Runnable. All inputs must be of the Runnable's Input type.
            config: Optional configuration, either a single RunnableConfig applied to
                all inputs, or a list of RunnableConfigs (one per input). If None,
                uses default configuration for all items.
            return_exceptions: If True, exceptions are returned in the output list
                alongside successful results. If False (default), the first exception
                encountered is raised immediately.
            **kwargs: Additional keyword arguments passed to the wrapped Runnable's
                batch method.
        
        Returns:
            List of outputs in the same order as inputs. Each element is either:
            - An Output value if that input was processed successfully
            - An Exception if that input failed all retry attempts (when
              return_exceptions=True)
        
        Raises:
            Exception: If return_exceptions=False and any input fails all retry
                attempts, the first exception is raised. The exception type depends
                on the underlying failure.
        
        Example:
            Batch processing with retry:
            
            ```python
            from langchain_core.runnables import RunnableLambda
            import requests
            
            def fetch_joke(topic: str) -> str:
                response = requests.get(f"https://api.example.com/jokes/{topic}")
                response.raise_for_status()
                return response.json()["joke"]
            
            runnable = RunnableLambda(fetch_joke)
            runnable_with_retry = runnable.with_retry(
                retry_if_exception_type=(requests.exceptions.RequestException,),
                max_attempt_number=3,
                wait_exponential_jitter=True
            )
            
            # Process multiple topics in batch with automatic retry
            topics = ["python", "javascript", "rust", "go", "java"]
            jokes = runnable_with_retry.batch(topics)
            
            # If some topics fail even after retries, get partial results
            jokes_with_errors = runnable_with_retry.batch(
                topics,
                return_exceptions=True  # Returns Exception objects for failures
            )
            
            for topic, joke in zip(topics, jokes_with_errors):
                if isinstance(joke, Exception):
                    print(f"{topic}: Failed - {joke}")
                else:
                    print(f"{topic}: {joke}")
            ```
        
        Example:
            Per-item retry behavior:
            
            ```python
            # Batch of 5 API calls with rate limiting (3 calls/second limit)
            inputs = ["topic1", "topic2", "topic3", "topic4", "topic5"]
            
            # Attempt 1: Process all 5
            # → topic1, topic2, topic3: Success (under rate limit)
            # → topic4, topic5: HTTP 429 (rate limited)
            # → Wait 1 second
            
            # Attempt 2: Retry only topic4, topic5
            # → topic4: Success
            # → topic5: HTTP 429 (still rate limited)
            # → Wait 2 seconds
            
            # Attempt 3: Retry only topic5
            # → topic5: Success
            
            # Final result: All 5 jokes returned in original order
            ```
        
        Source: libs/core/langchain_core/runnables/retry.py:290-301
        """
        return self._batch_with_config(
            self._batch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    async def _abatch(
        self,
        inputs: list[Input],
        run_manager: list["AsyncCallbackManagerForChainRun"],
        config: list[RunnableConfig],
        **kwargs: Any,
    ) -> list[Output | Exception]:
        """Internal asynchronous batch implementation with per-item retry logic.
        
        Executes async batch processing with intelligent retry handling. Like the
        synchronous version, this tracks success/failure per input item and only
        retries items that failed. The async implementation properly yields control
        to the event loop during backoff delays, allowing other async operations to
        continue executing.
        
        The async retry logic:
        1. Attempt to process all inputs in batch (concurrently)
        2. Track which items succeeded (store in results_map)
        3. If any items failed, await backoff delay (non-blocking)
        4. Retry only the failed items
        5. Repeat until all items succeed or max_attempt_number is reached
        6. Return results in original input order
        
        This is particularly efficient for high-throughput async applications where
        many batch operations may be retrying simultaneously without blocking each
        other.
        
        Args:
            inputs: List of inputs to process in batch. Each input is passed to the
                wrapped Runnable's async invocation.
            run_manager: List of async callback managers, one per input item, for
                tracking execution of each item.
            config: List of RunnableConfigs, one per input item, with callbacks, tags,
                and metadata for each item's invocation.
            **kwargs: Additional keyword arguments passed to the wrapped Runnable's
                abatch method.
        
        Returns:
            List of outputs or exceptions, one per input, in the same order as the
            inputs. Successfully processed items contain Output values, while items
            that failed all retry attempts contain Exception instances.
        
        Raises:
            This method does not raise exceptions directly; exceptions are returned
            in the output list. However, if return_exceptions=False in the public
            abatch() method, exceptions will be re-raised there.
        
        Example:
            Internal async batch retry flow:
            
            ```python
            # Input: [input1, input2, input3, input4]
            
            # Attempt 1: Await batch processing of all 4 items (concurrent)
            # → input1: Success (stored in results_map[0])
            # → input2: asyncio.TimeoutError (needs retry)
            # → input3: Success (stored in results_map[2])
            # → input4: asyncio.TimeoutError (needs retry)
            # → Await asyncio.sleep(1.0) - other tasks can run during wait
            
            # Attempt 2: Await batch processing only [input2, input4] (concurrent)
            # → input2: Success (stored in results_map[1])
            # → input4: asyncio.TimeoutError (needs retry)
            # → Await asyncio.sleep(2.0) - other tasks can run during wait
            
            # Attempt 3: Await batch processing only [input4]
            # → input4: Success (stored in results_map[3])
            
            # Return: [output1, output2, output3, output4] - all succeeded
            ```
        
        Source: libs/core/langchain_core/runnables/retry.py:303-363
        """
        # Track successful results by original index
        results_map: dict[int, Output] = {}

        # Sentinel value to detect if result was never set
        not_set: list[Output] = []
        result = not_set
        try:
            # Async tenacity retry loop (reraise=False for batch - we handle exceptions)
            async for attempt in self._async_retrying():
                with attempt:  # Tenacity attempt context - catches exceptions
                    # Retry for inputs that have not yet succeeded
                    # Determine which original indices remain to be retried
                    remaining_indices = [
                        i for i in range(len(inputs)) if i not in results_map
                    ]
                    if not remaining_indices:
                        break  # All items succeeded, exit retry loop
                    # Extract inputs/configs/managers for items that need retry
                    pending_inputs = [inputs[i] for i in remaining_indices]
                    pending_configs = [config[i] for i in remaining_indices]
                    pending_run_managers = [run_manager[i] for i in remaining_indices]
                    # Await underlying batch only on remaining elements
                    # return_exceptions=True ensures exceptions are returned, not raised
                    result = await super().abatch(
                        pending_inputs,
                        self._patch_config_list(
                            pending_configs, pending_run_managers, attempt.retry_state
                        ),
                        return_exceptions=True,
                        **kwargs,
                    )
                    # Register the results of the inputs that have succeeded, mapping
                    # back to their original indices for proper output ordering
                    first_exception = None
                    for offset, r in enumerate(result):
                        if isinstance(r, Exception):
                            # Track first exception to trigger retry
                            if not first_exception:
                                first_exception = r
                            continue  # Don't store exception in results_map
                        # Map successful result back to original input index
                        orig_idx = remaining_indices[offset]
                        results_map[orig_idx] = r
                    # If any exception occurred, raise it to trigger retry of failed items
                    if first_exception:
                        raise first_exception
                # Record successful batch result in retry state
                if (
                    attempt.retry_state.outcome
                    and not attempt.retry_state.outcome.failed
                ):
                    attempt.retry_state.set_result(result)
        except RetryError as e:
            # All retry attempts exhausted, create exception list for failed items
            if result is not_set:
                result = cast("list[Output]", [e] * len(inputs))

        # Build final output list in original input order
        outputs: list[Output | Exception] = []
        for idx in range(len(inputs)):
            if idx in results_map:
                # Item succeeded at some point, use stored result
                outputs.append(results_map[idx])
            else:
                # Item never succeeded, use exception from final attempt
                outputs.append(result.pop(0))
        return outputs

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        """Execute async batch processing with intelligent per-item retry logic.
        
        Processes multiple inputs concurrently in batch with automatic retry for
        failed items. This async implementation combines the efficiency of concurrent
        execution with intelligent retry handling - it tracks which individual items
        succeed and only retries items that failed. During backoff delays, the event
        loop remains free to process other async operations.
        
        The async batch retry behavior:
        - Attempt 1: Process all N inputs concurrently
        - Track which items succeeded (preserved for final output)
        - Attempt 2+: Only retry the items that failed (concurrently)
        - During backoff delays, other async tasks can execute
        - Continue until all items succeed or max_attempt_number is reached
        - Return results in original input order
        
        This is particularly valuable for high-throughput applications making many
        concurrent API calls where some may fail due to rate limiting or transient
        errors while others succeed.
        
        Args:
            inputs: List of inputs to process. Each input is passed to the wrapped
                Runnable asynchronously. All inputs must be of the Runnable's Input type.
            config: Optional configuration, either a single RunnableConfig applied to
                all inputs, or a list of RunnableConfigs (one per input). If None,
                uses default configuration for all items.
            return_exceptions: If True, exceptions are returned in the output list
                alongside successful results. If False (default), the first exception
                encountered is raised immediately.
            **kwargs: Additional keyword arguments passed to the wrapped Runnable's
                abatch method.
        
        Returns:
            List of outputs in the same order as inputs. Each element is either:
            - An Output value if that input was processed successfully
            - An Exception if that input failed all retry attempts (when
              return_exceptions=True)
        
        Raises:
            Exception: If return_exceptions=False and any input fails all retry
                attempts, the first exception is raised. The exception type depends
                on the underlying failure.
        
        Example:
            Async batch processing with retry:
            
            ```python
            import asyncio
            from langchain_core.runnables import RunnableLambda
            import aiohttp
            
            async def fetch_joke_async(topic: str) -> str:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://api.example.com/jokes/{topic}",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        return data["joke"]
            
            runnable = RunnableLambda(fetch_joke_async)
            runnable_with_retry = runnable.with_retry(
                retry_if_exception_type=(
                    asyncio.TimeoutError,
                    aiohttp.ClientError
                ),
                max_attempt_number=3,
                wait_exponential_jitter=True,
                exponential_jitter_params={"initial": 1.0, "jitter": 2.0}
            )
            
            # Process multiple topics concurrently with automatic retry
            topics = ["python", "javascript", "rust", "go", "java"]
            jokes = await runnable_with_retry.abatch(topics)
            
            # If some topics fail even after retries, get partial results
            jokes_with_errors = await runnable_with_retry.abatch(
                topics,
                return_exceptions=True  # Returns Exception objects for failures
            )
            
            for topic, joke in zip(topics, jokes_with_errors):
                if isinstance(joke, Exception):
                    print(f"{topic}: Failed - {joke}")
                else:
                    print(f"{topic}: {joke}")
            ```
        
        Example:
            Concurrent async retry with non-blocking backoff:
            
            ```python
            # Batch of 100 concurrent API calls
            inputs = [f"topic{i}" for i in range(100)]
            
            # Attempt 1: Process all 100 concurrently
            # → 95 succeed immediately
            # → 5 timeout (network issues)
            # → Await asyncio.sleep(1.0) - other coroutines continue running
            
            # Attempt 2: Retry only 5 failed items concurrently
            # → 4 succeed
            # → 1 still times out
            # → Await asyncio.sleep(2.0) - other coroutines continue running
            
            # Attempt 3: Retry only 1 failed item
            # → Succeeds
            
            # Final result: All 100 jokes returned in original order
            # Total time: Much faster than sequential retry
            ```
        
        Source: libs/core/langchain_core/runnables/retry.py:365-376
        """
        return await self._abatch_with_config(
            self._abatch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    # stream() and transform() are not retried because retrying a stream
    # is not very intuitive.
