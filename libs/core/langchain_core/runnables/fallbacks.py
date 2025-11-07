"""Runnable that can fallback to other Runnables if it fails.

This module provides the RunnableWithFallbacks class, which implements a resilient
execution pattern for LangChain Expression Language (LCEL) chains. Fallbacks enable
graceful degradation when external APIs experience failures, rate limiting, or downtime.

Key Use Cases:
    - Multi-provider LLM fallbacks: Try Claude, fallback to GPT-4, then to GPT-3.5
    - API failure recovery: Handle transient network errors or service degradation
    - Hardcoded fallback responses: Provide default responses when all services fail
    - Exception-based routing: Route execution based on specific error types

Type Flow:
    Input → Primary Runnable → (on exception) → Fallback[0] → Fallback[1] → ... → Output
    
    When an exception matching `exceptions_to_handle` occurs, the next fallback in the
    sequence is tried. If all fallbacks fail, the first exception is re-raised.

Source: libs/core/langchain_core/runnables/fallbacks.py:1
"""

import asyncio
import inspect
import typing
from collections.abc import AsyncIterator, Iterator, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
    patch_config,
    set_config_context,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    coro_with_context,
    get_unique_config_specs,
)
from langchain_core.utils.aiter import py_anext

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun


class RunnableWithFallbacks(RunnableSerializable[Input, Output]):
    """Runnable that can fallback to other Runnables if it fails.

    External APIs (e.g., APIs for a language model) may at times experience
    degraded performance or even downtime.

    In these cases, it can be useful to have a fallback Runnable that can be
    used in place of the original Runnable (e.g., fallback to another LLM provider).

    Fallbacks can be defined at the level of a single Runnable, or at the level
    of a chain of Runnables. Fallbacks are tried in order until one succeeds or
    all fail.

    While you can instantiate a `RunnableWithFallbacks` directly, it is usually
    more convenient to use the `with_fallbacks` method on a Runnable.

    Fallback Execution Semantics:
        1. The primary `runnable` is executed first
        2. If it raises an exception matching `exceptions_to_handle`, the first
           fallback is tried
        3. Each subsequent fallback is tried in sequence until one succeeds
        4. If all fallbacks fail, the FIRST exception (from primary runnable) is
           re-raised, not the last
        5. Exceptions NOT in `exceptions_to_handle` are immediately re-raised without
           trying fallbacks

    Exception Handling:
        - `exceptions_to_handle`: Tuple of exception types that trigger fallback logic.
          Default is (Exception,) which catches most errors but not BaseException
          subclasses like KeyboardInterrupt or SystemExit.
        - Exceptions matching this tuple are caught and the next fallback is tried
        - Other exceptions bypass fallback logic and are immediately raised

    Exception Propagation with exception_key:
        When `exception_key` is set to a string, the exception from each failed
        attempt is added to the input dictionary under that key before invoking
        the next fallback. This allows fallback Runnables to access error information
        and make decisions based on what failed.
        
        Requirements when using exception_key:
        - Input MUST be a dictionary (validated at runtime)
        - Primary runnable and all fallbacks must accept dict inputs
        - The exception object is added as: input[exception_key] = caught_exception

    Type Flow:
        Input → runnable.invoke(input) 
             → (on exception in exceptions_to_handle) → fallbacks[0].invoke(input)
             → (on exception) → fallbacks[1].invoke(input)
             → ... 
             → Output or raise first_error

    Use Cases:
        - Multi-provider LLM fallbacks: Primary provider fails → try backup provider
        - Degraded service handling: Full response fails → try summarized response
        - Hardcoded fallback responses: All services down → return static message
        - Custom exception handling: Different fallbacks for different error types

    Example:
        ```python
        from langchain_core.chat_models.openai import ChatOpenAI
        from langchain_core.chat_models.anthropic import ChatAnthropic

        model = ChatAnthropic(model="claude-3-haiku-20240307").with_fallbacks(
            [ChatOpenAI(model="gpt-3.5-turbo-0125")]
        )
        # Will usually use ChatAnthropic, but fallback to ChatOpenAI
        # if ChatAnthropic fails.
        model.invoke("hello")

        # And you can also use fallbacks at the level of a chain.
        # Here if both LLM providers fail, we'll fallback to a good hardcoded
        # response.

        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parser import StrOutputParser
        from langchain_core.runnables import RunnableLambda


        def when_all_is_lost(inputs):
            return (
                "Looks like our LLM providers are down. "
                "Here's a nice 🦜️ emoji for you instead."
            )


        chain_with_fallback = (
            PromptTemplate.from_template("Tell me a joke about {topic}")
            | model
            | StrOutputParser()
        ).with_fallbacks([RunnableLambda(when_all_is_lost)])
        ```

    Source: libs/core/langchain_core/runnables/fallbacks.py:37-87
    """

    runnable: Runnable[Input, Output]
    """The Runnable to run first."""
    fallbacks: Sequence[Runnable[Input, Output]]
    """A sequence of fallbacks to try."""
    exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,)
    """The exceptions on which fallbacks should be tried.

    Any exception that is not a subclass of these exceptions will be raised immediately.
    """
    exception_key: str | None = None
    """If `string` is specified then handled exceptions will be passed to fallbacks as
        part of the input under the specified key. If `None`, exceptions
        will not be passed to fallbacks. If used, the base Runnable and its fallbacks
        must accept a dictionary as input."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    @override
    def InputType(self) -> type[Input]:
        return self.runnable.InputType

    @property
    @override
    def OutputType(self) -> type[Output]:
        return self.runnable.OutputType

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        return self.runnable.get_input_schema(config)

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        return self.runnable.get_output_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec
            for step in [self.runnable, *self.fallbacks]
            for spec in step.config_specs
        )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return True as this class is serializable."""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @property
    def runnables(self) -> Iterator[Runnable[Input, Output]]:
        """Iterator over the Runnable and its fallbacks.

        Yields:
            The Runnable then its fallbacks.
        """
        yield self.runnable
        yield from self.fallbacks

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """Invoke the primary runnable with fallback execution on failure.

        Executes the primary runnable. If it raises an exception in 
        `exceptions_to_handle`, tries each fallback in sequence until one succeeds.
        If all fail, raises the FIRST exception (not the last).

        Args:
            input: The input to pass to the runnable and fallbacks. Must be a
                dictionary if `exception_key` is set.
            config: Optional configuration for callbacks, tags, metadata, and
                runtime settings.
            **kwargs: Additional keyword arguments passed through to the runnable's
                invoke method.

        Returns:
            The output from the first successfully executed runnable (primary or
            fallback).

        Raises:
            ValueError: If `exception_key` is set but input is not a dictionary.
            BaseException: Any exception not in `exceptions_to_handle` is raised
                immediately without trying fallbacks.
            Exception: If all runnables (primary + fallbacks) fail, the FIRST
                exception from the primary runnable is raised, enabling proper
                error diagnostics of the initial failure.

        Example:
            ```python
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic
            
            primary = ChatAnthropic(model="claude-3-haiku-20240307")
            fallback = ChatOpenAI(model="gpt-3.5-turbo")
            model = primary.with_fallbacks([fallback])
            
            # If Anthropic fails, automatically tries OpenAI
            response = model.invoke("Hello!")
            ```

        Note:
            The first exception is raised (not the last) to aid debugging by
            highlighting the primary failure, not the cascade of fallback failures.

        Source: libs/core/langchain_core/runnables/fallbacks.py:164-211
        """
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        # Track first error (for final re-raise) and last error (for exception_key)
        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                # Propagate previous error to fallback via exception_key
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    output = context.run(
                        runnable.invoke,
                        input,
                        config,
                        **kwargs,
                    )
            except self.exceptions_to_handle as e:
                # Store first error for final raise, update last error for next fallback
                if first_error is None:
                    first_error = e
                last_error = e
            except BaseException as e:
                # Exceptions NOT in exceptions_to_handle bypass fallbacks
                run_manager.on_chain_error(e)
                raise
            else:
                # Success! Return immediately without trying remaining fallbacks
                run_manager.on_chain_end(output)
                return output
        # All runnables failed - raise the FIRST error
        if first_error is None:
            msg = "No error stored at end of fallbacks."
            raise ValueError(msg)
        run_manager.on_chain_error(first_error)
        raise first_error

    @override
    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        """Async invoke the primary runnable with fallback execution on failure.

        Asynchronously executes the primary runnable. If it raises an exception in
        `exceptions_to_handle`, tries each fallback in sequence until one succeeds.
        If all fail, raises the FIRST exception (not the last).

        Args:
            input: The input to pass to the runnable and fallbacks. Must be a
                dictionary if `exception_key` is set.
            config: Optional configuration for callbacks, tags, metadata, and
                runtime settings.
            **kwargs: Additional keyword arguments passed through to the runnable's
                ainvoke method.

        Returns:
            The output from the first successfully executed runnable (primary or
            fallback).

        Raises:
            ValueError: If `exception_key` is set but input is not a dictionary.
            BaseException: Any exception not in `exceptions_to_handle` is raised
                immediately without trying fallbacks.
            Exception: If all runnables (primary + fallbacks) fail, the FIRST
                exception from the primary runnable is raised.

        Example:
            ```python
            import asyncio
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic
            
            primary = ChatAnthropic(model="claude-3-haiku-20240307")
            fallback = ChatOpenAI(model="gpt-3.5-turbo")
            model = primary.with_fallbacks([fallback])
            
            # Async execution with fallback handling
            response = await model.ainvoke("Hello!")
            ```

        Note:
            - Fallbacks are tried sequentially (not in parallel) to maintain order
            - Each fallback runs in the same event loop as the caller
            - Callback managers are async-aware and use AsyncCallbackManager
            - The coro_with_context wrapper preserves context vars across awaits

        Source: libs/core/langchain_core/runnables/fallbacks.py:214-261
        """
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # Track first error (for final re-raise) and last error (for exception_key)
        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                # Propagate previous error to fallback via exception_key
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    coro = context.run(runnable.ainvoke, input, config, **kwargs)
                    output = await coro_with_context(coro, context)
            except self.exceptions_to_handle as e:
                # Store first error for final raise, update last error for next fallback
                if first_error is None:
                    first_error = e
                last_error = e
            except BaseException as e:
                # Exceptions NOT in exceptions_to_handle bypass fallbacks
                await run_manager.on_chain_error(e)
                raise
            else:
                # Success! Return immediately without trying remaining fallbacks
                await run_manager.on_chain_end(output)
                return output
        # All runnables failed - raise the FIRST error
        if first_error is None:
            msg = "No error stored at end of fallbacks."
            raise ValueError(msg)
        await run_manager.on_chain_error(first_error)
        raise first_error

    @override
    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        """Batch invoke runnables with per-item fallback handling.

        Executes multiple inputs in batch through the primary runnable. For each
        input that fails with an exception in `exceptions_to_handle`, tries fallbacks
        in sequence. Each input is handled independently with its own fallback chain.

        Args:
            inputs: List of inputs to process. All must be dictionaries if
                `exception_key` is set.
            config: Optional configuration or list of configurations (one per input)
                for callbacks, tags, metadata, and runtime settings.
            return_exceptions: If True, returns exceptions in the output list instead
                of raising them. If False (default), raises the first exception that
                is NOT in `exceptions_to_handle`.
            **kwargs: Additional keyword arguments passed through to the runnable's
                batch method.

        Returns:
            List of outputs, one per input. If `return_exceptions=True`, failed
            inputs return their exception objects. If `return_exceptions=False`,
            only successful outputs are returned (exceptions are raised).

        Raises:
            ValueError: If `exception_key` is set but any input is not a dictionary.
            BaseException: If any input raises an exception NOT in 
                `exceptions_to_handle` and `return_exceptions=False`.
            Exception: If all fallbacks fail for an input and `return_exceptions=False`,
                raises the first exception for that input.

        Example:
            ```python
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic
            
            primary = ChatAnthropic(model="claude-3-haiku-20240307")
            fallback = ChatOpenAI(model="gpt-3.5-turbo")
            model = primary.with_fallbacks([fallback])
            
            # Batch processing with per-item fallback
            inputs = ["Hello", "How are you?", "Tell me a joke"]
            responses = model.batch(inputs)
            
            # With exception handling
            responses = model.batch(inputs, return_exceptions=True)
            # responses may contain Exception objects for failed items
            ```

        Note:
            - Each input maintains its own fallback state independent of others
            - Fallbacks are tried per-item, not globally across the batch
            - Failed items are re-batched with the next fallback until success
            - Callback managers track each input independently for observability

        Source: libs/core/langchain_core/runnables/fallbacks.py:264-357
        """
        if self.exception_key is not None and not all(
            isinstance(input_, dict) for input_ in inputs
        ):
            msg = (
                "If 'exception_key' is specified then inputs must be dictionaries."
                f"However found a type of {type(inputs[0])} for input"
            )
            raise ValueError(msg)

        if not inputs:
            return []

        # setup callbacks
        configs = get_config_list(config, len(inputs))
        callback_managers = [
            CallbackManager.configure(
                inheritable_callbacks=config.get("callbacks"),
                local_callbacks=None,
                verbose=False,
                inheritable_tags=config.get("tags"),
                local_tags=None,
                inheritable_metadata=config.get("metadata"),
                local_metadata=None,
            )
            for config in configs
        ]
        # start the root runs, one per input
        run_managers = [
            cm.on_chain_start(
                None,
                input_ if isinstance(input_, dict) else {"input": input_},
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            for cm, input_, config in zip(
                callback_managers, inputs, configs, strict=False
            )
        ]

        # Track outputs and exception state per input index
        to_return: dict[int, Any] = {}  # Successful outputs by index
        run_again = dict(enumerate(inputs))  # Inputs still needing retry
        handled_exceptions: dict[int, BaseException] = {}  # Exceptions per index
        first_to_raise = None  # First non-handled exception
        
        for runnable in self.runnables:
            # Batch remaining failed inputs with current runnable/fallback
            outputs = runnable.batch(
                [input_ for _, input_ in sorted(run_again.items())],
                [
                    # each step a child run of the corresponding root run
                    patch_config(configs[i], callbacks=run_managers[i].get_child())
                    for i in sorted(run_again)
                ],
                return_exceptions=True,  # Always return exceptions for per-item handling
                **kwargs,
            )
            for (i, input_), output in zip(
                sorted(run_again.copy().items()), outputs, strict=False
            ):
                if isinstance(output, BaseException) and not isinstance(
                    output, self.exceptions_to_handle
                ):
                    # Exception NOT in exceptions_to_handle - don't retry with fallbacks
                    if not return_exceptions:
                        first_to_raise = first_to_raise or output
                    else:
                        handled_exceptions[i] = output
                    run_again.pop(i)  # Remove from retry list
                elif isinstance(output, self.exceptions_to_handle):
                    # Exception matches - propagate to next fallback via exception_key
                    if self.exception_key:
                        input_[self.exception_key] = output  # type: ignore[index]
                    handled_exceptions[i] = output  # Track for final raise if needed
                    # Keep in run_again to retry with next fallback
                else:
                    # Success for this input - store output and remove from retry
                    run_managers[i].on_chain_end(output)
                    to_return[i] = output
                    run_again.pop(i)
                    handled_exceptions.pop(i, None)  # Clear exception tracking
            if first_to_raise:
                raise first_to_raise
            if not run_again:
                break  # All inputs succeeded, no need to try more fallbacks

        sorted_handled_exceptions = sorted(handled_exceptions.items())
        for i, error in sorted_handled_exceptions:
            run_managers[i].on_chain_error(error)
        if not return_exceptions and sorted_handled_exceptions:
            raise sorted_handled_exceptions[0][1]
        to_return.update(handled_exceptions)
        return [output for _, output in sorted(to_return.items())]

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        """Async batch invoke runnables with per-item fallback handling.

        Asynchronously executes multiple inputs in batch through the primary runnable.
        For each input that fails with an exception in `exceptions_to_handle`, tries
        fallbacks in sequence. Each input is handled independently with its own
        fallback chain.

        Args:
            inputs: List of inputs to process. All must be dictionaries if
                `exception_key` is set.
            config: Optional configuration or list of configurations (one per input)
                for callbacks, tags, metadata, and runtime settings.
            return_exceptions: If True, returns exceptions in the output list instead
                of raising them. If False (default), raises the first exception that
                is NOT in `exceptions_to_handle`.
            **kwargs: Additional keyword arguments passed through to the runnable's
                abatch method.

        Returns:
            List of outputs, one per input. If `return_exceptions=True`, failed
            inputs return their exception objects. If `return_exceptions=False`,
            only successful outputs are returned (exceptions are raised).

        Raises:
            ValueError: If `exception_key` is set but any input is not a dictionary.
            BaseException: If any input raises an exception NOT in
                `exceptions_to_handle` and `return_exceptions=False`.
            Exception: If all fallbacks fail for an input and `return_exceptions=False`,
                raises the first exception for that input.

        Example:
            ```python
            import asyncio
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic
            
            primary = ChatAnthropic(model="claude-3-haiku-20240307")
            fallback = ChatOpenAI(model="gpt-3.5-turbo")
            model = primary.with_fallbacks([fallback])
            
            # Async batch processing with per-item fallback
            inputs = ["Hello", "How are you?", "Tell me a joke"]
            responses = await model.abatch(inputs)
            
            # With exception handling
            responses = await model.abatch(inputs, return_exceptions=True)
            # responses may contain Exception objects for failed items
            ```

        Note:
            - Async callbacks are awaited using asyncio.gather for efficiency
            - Each input maintains its own fallback state independent of others
            - Failed items are re-batched with the next fallback until success
            - All async operations run in the same event loop as the caller

        Source: libs/core/langchain_core/runnables/fallbacks.py:360-461
        """
        if self.exception_key is not None and not all(
            isinstance(input_, dict) for input_ in inputs
        ):
            msg = (
                "If 'exception_key' is specified then inputs must be dictionaries."
                f"However found a type of {type(inputs[0])} for input"
            )
            raise ValueError(msg)

        if not inputs:
            return []

        # setup callbacks
        configs = get_config_list(config, len(inputs))
        callback_managers = [
            AsyncCallbackManager.configure(
                inheritable_callbacks=config.get("callbacks"),
                local_callbacks=None,
                verbose=False,
                inheritable_tags=config.get("tags"),
                local_tags=None,
                inheritable_metadata=config.get("metadata"),
                local_metadata=None,
            )
            for config in configs
        ]
        # start the root runs, one per input
        run_managers: list[AsyncCallbackManagerForChainRun] = await asyncio.gather(
            *(
                cm.on_chain_start(
                    None,
                    input_,
                    name=config.get("run_name") or self.get_name(),
                    run_id=config.pop("run_id", None),
                )
                for cm, input_, config in zip(
                    callback_managers, inputs, configs, strict=False
                )
            )
        )

        # Track outputs and exception state per input index
        to_return: dict[int, Output | BaseException] = {}  # Successful outputs by index
        run_again = dict(enumerate(inputs))  # Inputs still needing retry
        handled_exceptions: dict[int, BaseException] = {}  # Exceptions per index
        first_to_raise = None  # First non-handled exception
        
        for runnable in self.runnables:
            # Batch remaining failed inputs with current runnable/fallback
            outputs = await runnable.abatch(
                [input_ for _, input_ in sorted(run_again.items())],
                [
                    # each step a child run of the corresponding root run
                    patch_config(configs[i], callbacks=run_managers[i].get_child())
                    for i in sorted(run_again)
                ],
                return_exceptions=True,  # Always return exceptions for per-item handling
                **kwargs,
            )

            for (i, input_), output in zip(
                sorted(run_again.copy().items()), outputs, strict=False
            ):
                if isinstance(output, BaseException) and not isinstance(
                    output, self.exceptions_to_handle
                ):
                    # Exception NOT in exceptions_to_handle - don't retry with fallbacks
                    if not return_exceptions:
                        first_to_raise = first_to_raise or output
                    else:
                        handled_exceptions[i] = output
                    run_again.pop(i)  # Remove from retry list
                elif isinstance(output, self.exceptions_to_handle):
                    # Exception matches - propagate to next fallback via exception_key
                    if self.exception_key:
                        input_[self.exception_key] = output  # type: ignore[index]
                    handled_exceptions[i] = output  # Track for final raise if needed
                    # Keep in run_again to retry with next fallback
                else:
                    # Success for this input - store output and remove from retry
                    to_return[i] = output
                    await run_managers[i].on_chain_end(output)
                    run_again.pop(i)
                    handled_exceptions.pop(i, None)  # Clear exception tracking

            if first_to_raise:
                raise first_to_raise
            if not run_again:
                break  # All inputs succeeded, no need to try more fallbacks

        sorted_handled_exceptions = sorted(handled_exceptions.items())
        await asyncio.gather(
            *(
                run_managers[i].on_chain_error(error)
                for i, error in sorted_handled_exceptions
            )
        )
        if not return_exceptions and sorted_handled_exceptions:
            raise sorted_handled_exceptions[0][1]
        to_return.update(handled_exceptions)
        return [cast("Output", output) for _, output in sorted(to_return.items())]

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        """Stream outputs from the runnable with fallback on initial failure.

        Attempts to stream from the primary runnable. If the first chunk fails with
        an exception in `exceptions_to_handle`, tries fallbacks until one succeeds.
        Once streaming begins successfully, no fallback occurs for mid-stream errors.

        Args:
            input: The input to pass to the runnable and fallbacks. Must be a
                dictionary if `exception_key` is set.
            config: Optional configuration for callbacks, tags, metadata, and
                runtime settings.
            **kwargs: Additional keyword arguments passed through to the runnable's
                stream method.

        Yields:
            Output chunks from the first successfully started stream.

        Raises:
            ValueError: If `exception_key` is set but input is not a dictionary.
            BaseException: Any exception not in `exceptions_to_handle` is raised
                immediately without trying fallbacks.
            Exception: If all runnables fail to produce the first chunk, raises
                the first exception from the primary runnable.

        Example:
            ```python
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic
            
            primary = ChatAnthropic(model="claude-3-haiku-20240307")
            fallback = ChatOpenAI(model="gpt-3.5-turbo")
            model = primary.with_fallbacks([fallback])
            
            # Stream with fallback on initial failure
            for chunk in model.stream("Tell me a story"):
                print(chunk.content, end="", flush=True)
            ```

        Note:
            - Fallback only applies to the FIRST chunk (stream initialization)
            - Once streaming starts, errors mid-stream are raised without fallback
            - This prevents partial output mixing from different runnables
            - Only the successful runnable's output is streamed

        Source: libs/core/langchain_core/runnables/fallbacks.py:464-525
        """
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        # Track errors during stream initialization (first chunk only)
        first_error = None
        last_error = None
        
        # Try to get the first chunk from primary or fallbacks
        for runnable in self.runnables:
            try:
                # Propagate previous error to fallback via exception_key
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    stream = context.run(
                        runnable.stream,
                        input,
                        **kwargs,
                    )
                    # Critical: Try to get first chunk to detect initialization failures
                    chunk: Output = context.run(next, stream)
            except self.exceptions_to_handle as e:
                # First chunk failed - try next fallback
                first_error = e if first_error is None else first_error
                last_error = e
            except BaseException as e:
                # Unhandled exception - don't try fallbacks
                run_manager.on_chain_error(e)
                raise
            else:
                # Successfully got first chunk - use this stream
                first_error = None
                break
        
        # If all runnables failed to produce first chunk, raise
        if first_error:
            run_manager.on_chain_error(first_error)
            raise first_error

        # Stream is initialized - yield chunks without fallback protection
        yield chunk
        output: Output | None = chunk
        try:
            # Continue streaming remaining chunks (no fallback here)
            for chunk in stream:
                yield chunk
                try:
                    # Attempt to accumulate output for callback
                    output = output + chunk  # type: ignore[operator]
                except TypeError:
                    # Can't accumulate - pass None to callback
                    output = None
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(output)

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        """Async stream outputs from the runnable with fallback on initial failure.

        Asynchronously attempts to stream from the primary runnable. If the first
        chunk fails with an exception in `exceptions_to_handle`, tries fallbacks
        until one succeeds. Once streaming begins successfully, no fallback occurs
        for mid-stream errors.

        Args:
            input: The input to pass to the runnable and fallbacks. Must be a
                dictionary if `exception_key` is set.
            config: Optional configuration for callbacks, tags, metadata, and
                runtime settings.
            **kwargs: Additional keyword arguments passed through to the runnable's
                astream method.

        Yields:
            Output chunks from the first successfully started stream.

        Raises:
            ValueError: If `exception_key` is set but input is not a dictionary.
            BaseException: Any exception not in `exceptions_to_handle` is raised
                immediately without trying fallbacks.
            Exception: If all runnables fail to produce the first chunk, raises
                the first exception from the primary runnable.

        Example:
            ```python
            import asyncio
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic
            
            primary = ChatAnthropic(model="claude-3-haiku-20240307")
            fallback = ChatOpenAI(model="gpt-3.5-turbo")
            model = primary.with_fallbacks([fallback])
            
            # Async stream with fallback on initial failure
            async for chunk in model.astream("Tell me a story"):
                print(chunk.content, end="", flush=True)
            ```

        Note:
            - Fallback only applies to the FIRST chunk (stream initialization)
            - Once streaming starts, errors mid-stream are raised without fallback
            - This prevents partial output mixing from different runnables
            - Stream runs in the same event loop as the caller

        Source: libs/core/langchain_core/runnables/fallbacks.py:528-589
        """
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        # Track errors during stream initialization (first chunk only)
        first_error = None
        last_error = None
        
        # Try to get the first chunk from primary or fallbacks
        for runnable in self.runnables:
            try:
                # Propagate previous error to fallback via exception_key
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    stream = runnable.astream(
                        input,
                        child_config,
                        **kwargs,
                    )
                    # Critical: Try to get first chunk to detect initialization failures
                    chunk = await coro_with_context(py_anext(stream), context)
            except self.exceptions_to_handle as e:
                # First chunk failed - try next fallback
                first_error = e if first_error is None else first_error
                last_error = e
            except BaseException as e:
                # Unhandled exception - don't try fallbacks
                await run_manager.on_chain_error(e)
                raise
            else:
                # Successfully got first chunk - use this stream
                first_error = None
                break
        
        # If all runnables failed to produce first chunk, raise
        if first_error:
            await run_manager.on_chain_error(first_error)
            raise first_error

        # Stream is initialized - yield chunks without fallback protection
        yield chunk
        output: Output | None = chunk
        try:
            # Continue streaming remaining chunks (no fallback here)
            async for chunk in stream:
                yield chunk
                try:
                    # Attempt to accumulate output for callback
                    output = output + chunk  # type: ignore[operator]
                except TypeError:
                    # Can't accumulate - pass None to callback
                    output = None
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(output)

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the wrapped Runnable and its fallbacks.

        Returns:
            If the attribute is anything other than a method that outputs a Runnable,
            returns getattr(self.runnable, name). If the attribute is a method that
            does return a new Runnable (e.g. model.bind_tools([...]) outputs a new
            RunnableBinding) then self.runnable and each of the runnables in
            self.fallbacks is replaced with getattr(x, name).

        Example:
            ```python
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic

            gpt_4o = ChatOpenAI(model="gpt-4o")
            claude_3_sonnet = ChatAnthropic(model="claude-3-7-sonnet-20250219")
            model = gpt_4o.with_fallbacks([claude_3_sonnet])

            model.model_name
            # -> "gpt-4o"

            # .bind_tools() is called on both ChatOpenAI and ChatAnthropic
            # Equivalent to:
            # gpt_4o.bind_tools([...]).with_fallbacks([claude_3_sonnet.bind_tools([...])])
            model.bind_tools([...])
            # -> RunnableWithFallbacks(
                runnable=RunnableBinding(bound=ChatOpenAI(...), kwargs={"tools": [...]}),
                fallbacks=[RunnableBinding(bound=ChatAnthropic(...), kwargs={"tools": [...]})],
            )

            ```
        """  # noqa: E501
        attr = getattr(self.runnable, name)
        if _returns_runnable(attr):

            @wraps(attr)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                new_runnable = attr(*args, **kwargs)
                new_fallbacks = []
                for fallback in self.fallbacks:
                    fallback_attr = getattr(fallback, name)
                    new_fallbacks.append(fallback_attr(*args, **kwargs))

                return self.__class__(
                    **{
                        **self.model_dump(),
                        "runnable": new_runnable,
                        "fallbacks": new_fallbacks,
                    }
                )

            return wrapped

        return attr


def _returns_runnable(attr: Any) -> bool:
    if not callable(attr):
        return False
    return_type = typing.get_type_hints(attr).get("return")
    return bool(return_type and _is_runnable_type(return_type))


def _is_runnable_type(type_: Any) -> bool:
    if inspect.isclass(type_):
        return issubclass(type_, Runnable)
    origin = getattr(type_, "__origin__", None)
    if inspect.isclass(origin):
        return issubclass(origin, Runnable)
    if origin is typing.Union:
        return all(_is_runnable_type(t) for t in type_.__args__)
    return False
