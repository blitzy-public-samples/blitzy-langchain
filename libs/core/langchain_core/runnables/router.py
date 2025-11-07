"""Runnable that routes to a set of Runnables based on a key.

This module provides RouterRunnable, which implements dynamic routing in LCEL
(LangChain Expression Language). It enables key-based selection of Runnables from
a predefined mapping, allowing runtime determination of which Runnable to execute
based on input data.

Key Concepts:
    - **Dynamic Routing**: Select which Runnable to execute based on input['key']
    - **Key-Based Selection**: Uses string keys to map inputs to specific Runnables
    - **Type Preservation**: Maintains type safety with generic Output type
    - **Config Propagation**: Passes RunnableConfig to selected Runnable

Use Cases:
    - Multi-model routing (select LLM based on complexity: GPT-4 vs GPT-3.5)
    - Tool selection (route to different tools based on user intent)
    - Dynamic chain dispatch (select processing pipeline based on document type)
    - Conditional logic (route to different chains based on classification result)

Comparison with RunnableBranch:
    - RouterRunnable: Key-based selection using a string key from input
        * Fast O(1) lookup in dictionary mapping
        * Requires explicit key in input
        * Static mapping defined at construction
    - RunnableBranch: Condition-based selection using predicate functions
        * Evaluates conditions sequentially until one matches
        * More flexible - conditions can use any logic
        * Can include default branch for unmatched conditions

Source: libs/core/langchain_core/runnables/router.py
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from pydantic import ConfigDict
from typing_extensions import TypedDict, override

from langchain_core.runnables.base import (
    Runnable,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    get_config_list,
    get_executor_for_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    gather_with_concurrency,
    get_unique_config_specs,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


class RouterInput(TypedDict):
    """Router input structure for key-based routing.

    This TypedDict defines the required structure for RouterRunnable input.
    It contains two required fields: a routing key and the actual input data.

    Attributes:
        key: String key used to select which Runnable to execute from the
            runnables mapping. Must exactly match a key in the RouterRunnable's
            runnables dictionary.
        input: The actual input data to pass to the selected Runnable. Can be
            any type - the selected Runnable determines the expected input type.

    Examples:
        Simple routing with string input:
            >>> router_input = {"key": "add", "input": 5}

        Routing with dictionary input:
            >>> router_input = {
            ...     "key": "summarize",
            ...     "input": {"text": "Long document...", "max_length": 100}
            ... }

        Multi-model routing based on complexity:
            >>> router_input = {
            ...     "key": "gpt4",  # or "gpt3.5" for simple queries
            ...     "input": "Explain quantum entanglement"
            ... }

        Tool selection based on intent:
            >>> router_input = {
            ...     "key": "calculator",  # or "search", "database", etc.
            ...     "input": {"operation": "multiply", "values": [5, 7]}
            ... }
    """

    key: str
    """The key to route on."""
    input: Any
    """The input to pass to the selected Runnable."""


class RouterRunnable(RunnableSerializable[RouterInput, Output]):
    """Runnable that routes to a set of Runnables based on Input['key'].

    RouterRunnable implements dynamic key-based routing in LCEL, allowing runtime
    selection of which Runnable to execute based on a string key provided in the
    input. This enables flexible chain composition where the execution path is
    determined by input data rather than compile-time structure.

    Routing Mechanism:
        1. Extract 'key' from input dictionary (input['key'])
        2. Validate key exists in runnables mapping
        3. Select corresponding Runnable from mapping
        4. Execute selected Runnable with input['input'] and propagated config
        5. Return output from selected Runnable

    Runnables Mapping:
        The runnables parameter is a Dict[str, Runnable] that maps string keys
        to Runnable instances. Keys must be unique strings. Runnables can be of
        any type as long as they share the same output type (Output generic).

    Error Handling:
        Raises ValueError if:
        - input['key'] is not found in runnables mapping
        - Key validation fails in batch operations (any key not in mapping)

    Type Flow:
        RouterInput (dict with 'key' and 'input') →
        Key extraction and validation →
        Runnable selection from mapping →
        Selected Runnable execution: Input → Output →
        Return Output

    Use Cases:
        - **Multi-model routing**: Select between different LLMs based on query
          complexity (e.g., GPT-4 for complex reasoning, GPT-3.5 for simple tasks)
        - **Tool selection**: Route to different tools based on detected intent
          (calculator, search engine, database query, API call)
        - **Dynamic chain dispatch**: Select processing pipeline based on document
          type, language, or classification result
        - **A/B testing**: Route to different chain variants for experimentation
        - **Load balancing**: Distribute requests across multiple model instances

    Comparison with RunnableBranch:
        - RouterRunnable: Static key-based selection
            * O(1) dictionary lookup performance
            * Requires explicit 'key' field in input
            * Keys defined at construction time
            * Best for: Known set of options, fast routing
        - RunnableBranch: Dynamic condition-based selection
            * Evaluates predicate functions sequentially
            * Conditions can use any logic on input
            * Includes default branch for unmatched cases
            * Best for: Complex conditions, fallback handling

    Examples:
        Basic routing example:
            >>> from langchain_core.runnables.router import RouterRunnable
            >>> from langchain_core.runnables import RunnableLambda
            >>> 
            >>> add = RunnableLambda(func=lambda x: x + 1)
            >>> square = RunnableLambda(func=lambda x: x**2)
            >>> 
            >>> router = RouterRunnable(runnables={"add": add, "square": square})
            >>> router.invoke({"key": "square", "input": 3})  # Returns 9

        Multi-model routing based on complexity:
            >>> from langchain_core.runnables.router import RouterRunnable
            >>> from langchain_openai import ChatOpenAI
            >>> 
            >>> # Define models with different capabilities
            >>> gpt4 = ChatOpenAI(model="gpt-4", temperature=0)
            >>> gpt35 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            >>> 
            >>> # Create router for model selection
            >>> model_router = RouterRunnable(runnables={
            ...     "complex": gpt4,  # For complex reasoning tasks
            ...     "simple": gpt35,  # For straightforward queries
            ... })
            >>> 
            >>> # Route based on detected complexity
            >>> result = model_router.invoke({
            ...     "key": "complex",
            ...     "input": "Explain the implications of quantum entanglement"
            ... })

        Tool selection routing:
            >>> from langchain_core.runnables.router import RouterRunnable
            >>> from langchain_core.tools import tool
            >>> 
            >>> @tool
            >>> def calculator(expression: str) -> float:
            ...     '''Evaluate a mathematical expression'''
            ...     return eval(expression)
            >>> 
            >>> @tool
            >>> def search(query: str) -> str:
            ...     '''Search the web for information'''
            ...     return f"Results for: {query}"
            >>> 
            >>> # Create tool router
            >>> tool_router = RouterRunnable(runnables={
            ...     "calculator": calculator,
            ...     "search": search,
            ... })
            >>> 
            >>> # Route to appropriate tool based on intent
            >>> result = tool_router.invoke({
            ...     "key": "calculator",
            ...     "input": "2 + 2 * 3"
            ... })  # Returns 8.0

        Intent-based chain selection:
            >>> from langchain_core.runnables.router import RouterRunnable
            >>> from langchain_core.prompts import ChatPromptTemplate
            >>> from langchain_openai import ChatOpenAI
            >>> 
            >>> # Define specialized chains for different intents
            >>> summarize_chain = (
            ...     ChatPromptTemplate.from_template("Summarize: {text}")
            ...     | ChatOpenAI()
            ... )
            >>> translate_chain = (
            ...     ChatPromptTemplate.from_template("Translate to {lang}: {text}")
            ...     | ChatOpenAI()
            ... )
            >>> 
            >>> # Create intent router
            >>> intent_router = RouterRunnable(runnables={
            ...     "summarize": summarize_chain,
            ...     "translate": translate_chain,
            ... })
            >>> 
            >>> # Route based on detected user intent
            >>> result = intent_router.invoke({
            ...     "key": "summarize",
            ...     "input": {"text": "Long article content..."}
            ... })
    """

    runnables: Mapping[str, Runnable[Any, Output]]

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec for step in self.runnables.values() for spec in step.config_specs
        )

    def __init__(
        self,
        runnables: Mapping[str, Runnable[Any, Output] | Callable[[Any], Output]],
    ) -> None:
        """Create a RouterRunnable.

        Args:
            runnables: A mapping of keys to Runnables.
        """
        super().__init__(
            runnables={key: coerce_to_runnable(r) for key, r in runnables.items()}
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
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

    @override
    def invoke(
        self, input: RouterInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """Invoke the router to execute the selected Runnable synchronously.

        Extracts the routing key from input, validates it exists in the runnables
        mapping, selects the corresponding Runnable, and executes it with the
        provided input data and configuration.

        Args:
            input: RouterInput dictionary containing:
                - key (str): The routing key to select which Runnable to execute.
                  Must match a key in the runnables mapping.
                - input (Any): The actual input data to pass to the selected
                  Runnable. Type must match what the selected Runnable expects.
            config: Optional RunnableConfig to control execution behavior. This
                config is propagated to the selected Runnable, including callbacks,
                tags, metadata, and other execution parameters.
            **kwargs: Additional keyword arguments passed through to the selected
                Runnable's invoke method.

        Returns:
            Output: The output from the selected Runnable. The return type depends
                on which Runnable was selected and executed.

        Raises:
            ValueError: If input['key'] does not exist in the runnables mapping.
                Error message format: "No runnable associated with key '{key}'"
            KeyError: If input dictionary is missing 'key' or 'input' fields.

        Example:
            >>> from langchain_core.runnables.router import RouterRunnable
            >>> from langchain_core.runnables import RunnableLambda
            >>> 
            >>> router = RouterRunnable(runnables={
            ...     "double": RunnableLambda(func=lambda x: x * 2),
            ...     "triple": RunnableLambda(func=lambda x: x * 3),
            ... })
            >>> 
            >>> result = router.invoke({"key": "double", "input": 5})
            >>> print(result)  # Output: 10
            >>> 
            >>> # With config for callbacks
            >>> from langchain_core.callbacks import StdOutCallbackHandler
            >>> result = router.invoke(
            ...     {"key": "triple", "input": 4},
            ...     config={"callbacks": [StdOutCallbackHandler()]}
            ... )  # Output: 12

        Source: libs/core/langchain_core/runnables/router.py:invoke()
        """
        # Extract routing key from input
        key = input["key"]
        actual_input = input["input"]
        
        # Validate key exists in runnables mapping
        if key not in self.runnables:
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        # Select the Runnable based on the key
        runnable = self.runnables[key]
        
        # Execute selected Runnable with propagated config
        return runnable.invoke(actual_input, config)

    @override
    async def ainvoke(
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        """Invoke the router to execute the selected Runnable asynchronously.

        Asynchronous version of invoke(). Extracts the routing key, validates it,
        selects the corresponding Runnable, and executes it asynchronously using
        the Runnable's ainvoke method.

        Args:
            input: RouterInput dictionary containing:
                - key (str): The routing key to select which Runnable to execute.
                  Must match a key in the runnables mapping.
                - input (Any): The actual input data to pass to the selected
                  Runnable. Type must match what the selected Runnable expects.
            config: Optional RunnableConfig to control execution behavior. This
                config is propagated to the selected Runnable, including async
                callbacks, tags, metadata, and execution parameters.
            **kwargs: Additional keyword arguments passed through to the selected
                Runnable's ainvoke method.

        Returns:
            Output: The output from the selected Runnable's asynchronous execution.
                The return type depends on which Runnable was selected and executed.

        Raises:
            ValueError: If input['key'] does not exist in the runnables mapping.
                Error message format: "No runnable associated with key '{key}'"
            KeyError: If input dictionary is missing 'key' or 'input' fields.

        Example:
            >>> import asyncio
            >>> from langchain_core.runnables.router import RouterRunnable
            >>> from langchain_core.runnables import RunnableLambda
            >>> 
            >>> async def async_double(x):
            ...     await asyncio.sleep(0.1)  # Simulate async work
            ...     return x * 2
            >>> 
            >>> router = RouterRunnable(runnables={
            ...     "double": RunnableLambda(func=async_double),
            ... })
            >>> 
            >>> result = asyncio.run(router.ainvoke({"key": "double", "input": 5}))
            >>> print(result)  # Output: 10

        Source: libs/core/langchain_core/runnables/router.py:ainvoke()
        """
        # Extract routing key from input
        key = input["key"]
        actual_input = input["input"]
        
        # Validate key exists in runnables mapping
        if key not in self.runnables:
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        # Select the Runnable based on the key
        runnable = self.runnables[key]
        
        # Execute selected Runnable asynchronously with propagated config
        return await runnable.ainvoke(actual_input, config)

    @override
    def batch(
        self,
        inputs: list[RouterInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        """Invoke the router on multiple inputs in parallel using a thread pool.

        Executes routing for multiple inputs concurrently using ThreadPoolExecutor.
        Each input is routed independently to its corresponding Runnable based on
        the key, and all executions run in parallel threads for efficiency.

        Args:
            inputs: List of RouterInput dictionaries, each containing:
                - key (str): The routing key for selecting which Runnable to execute
                - input (Any): The input data to pass to the selected Runnable
            config: Optional RunnableConfig or list of configs (one per input).
                If a single config is provided, it's used for all inputs. If a list
                is provided, each config is paired with the corresponding input.
                Configs control execution behavior including callbacks, tags, and
                executor settings.
            return_exceptions: If True, exceptions during execution are returned
                in the output list instead of being raised. If False (default),
                exceptions are raised immediately, stopping batch execution.
            **kwargs: Additional keyword arguments passed to each Runnable's
                invoke method.

        Returns:
            list[Output]: List of outputs from the selected Runnables, one for each
                input in the same order. If return_exceptions=True, list may contain
                Exception objects for failed executions.

        Raises:
            ValueError: If any key in inputs does not exist in the runnables mapping.
                Error message: "One or more keys do not have a corresponding runnable"
            KeyError: If any input dictionary is missing 'key' or 'input' fields.

        Example:
            >>> from langchain_core.runnables.router import RouterRunnable
            >>> from langchain_core.runnables import RunnableLambda
            >>> 
            >>> router = RouterRunnable(runnables={
            ...     "add_one": RunnableLambda(func=lambda x: x + 1),
            ...     "multiply_two": RunnableLambda(func=lambda x: x * 2),
            ... })
            >>> 
            >>> # Batch execute multiple routing operations in parallel
            >>> results = router.batch([
            ...     {"key": "add_one", "input": 5},
            ...     {"key": "multiply_two", "input": 5},
            ...     {"key": "add_one", "input": 10},
            ... ])
            >>> print(results)  # Output: [6, 10, 11]
            >>> 
            >>> # Handle exceptions gracefully
            >>> results = router.batch(
            ...     [{"key": "add_one", "input": 5}, {"key": "invalid", "input": 3}],
            ...     return_exceptions=True
            ... )
            >>> # results[0] is 6, results[1] is ValueError

        Source: libs/core/langchain_core/runnables/router.py:batch()
        """
        # Early return for empty input list
        if not inputs:
            return []

        # Extract keys and actual inputs from all RouterInput dictionaries
        keys = [input_["key"] for input_ in inputs]
        actual_inputs = [input_["input"] for input_ in inputs]
        
        # Validate all keys exist in runnables mapping before execution
        if any(key not in self.runnables for key in keys):
            msg = "One or more keys do not have a corresponding runnable"
            raise ValueError(msg)

        # Define wrapper function for exception handling in parallel execution
        def invoke(
            runnable: Runnable, input_: Input, config: RunnableConfig
        ) -> Output | Exception:
            if return_exceptions:
                try:
                    return runnable.invoke(input_, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return runnable.invoke(input_, config, **kwargs)

        # Select Runnables for each input based on keys
        runnables = [self.runnables[key] for key in keys]
        
        # Prepare configs: expand single config to list or validate list length
        configs = get_config_list(config, len(inputs))
        
        # Execute all invocations in parallel using ThreadPoolExecutor
        with get_executor_for_config(configs[0]) as executor:
            return cast(
                "list[Output]",
                list(executor.map(invoke, runnables, actual_inputs, configs)),
            )

    @override
    async def abatch(
        self,
        inputs: list[RouterInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        """Invoke the router on multiple inputs concurrently using async execution.

        Asynchronous version of batch(). Executes routing for multiple inputs
        concurrently using asyncio.gather with optional concurrency limiting.
        Each input is routed independently to its corresponding Runnable, and all
        executions run concurrently for maximum efficiency.

        Args:
            inputs: List of RouterInput dictionaries, each containing:
                - key (str): The routing key for selecting which Runnable to execute
                - input (Any): The input data to pass to the selected Runnable
            config: Optional RunnableConfig or list of configs (one per input).
                If a single config is provided, it's used for all inputs. If a list
                is provided, each config is paired with the corresponding input.
                Config can include 'max_concurrency' to limit concurrent executions.
            return_exceptions: If True, exceptions during execution are returned
                in the output list instead of being raised. If False (default),
                the first exception stops execution and is raised.
            **kwargs: Additional keyword arguments passed to each Runnable's
                ainvoke method.

        Returns:
            list[Output]: List of outputs from the selected Runnables, one for each
                input in the same order. If return_exceptions=True, list may contain
                Exception objects for failed executions.

        Raises:
            ValueError: If any key in inputs does not exist in the runnables mapping.
                Error message: "One or more keys do not have a corresponding runnable"
            KeyError: If any input dictionary is missing 'key' or 'input' fields.

        Example:
            >>> import asyncio
            >>> from langchain_core.runnables.router import RouterRunnable
            >>> from langchain_core.runnables import RunnableLambda
            >>> 
            >>> async def async_add_one(x):
            ...     await asyncio.sleep(0.1)
            ...     return x + 1
            >>> 
            >>> async def async_multiply_two(x):
            ...     await asyncio.sleep(0.1)
            ...     return x * 2
            >>> 
            >>> router = RouterRunnable(runnables={
            ...     "add_one": RunnableLambda(func=async_add_one),
            ...     "multiply_two": RunnableLambda(func=async_multiply_two),
            ... })
            >>> 
            >>> # Batch execute with concurrent async execution
            >>> results = asyncio.run(router.abatch([
            ...     {"key": "add_one", "input": 5},
            ...     {"key": "multiply_two", "input": 5},
            ...     {"key": "add_one", "input": 10},
            ... ]))
            >>> print(results)  # Output: [6, 10, 11]
            >>> 
            >>> # Limit concurrency to 2 simultaneous operations
            >>> results = asyncio.run(router.abatch(
            ...     [{"key": "add_one", "input": i} for i in range(10)],
            ...     config={"max_concurrency": 2}
            ... ))

        Source: libs/core/langchain_core/runnables/router.py:abatch()
        """
        # Early return for empty input list
        if not inputs:
            return []

        # Extract keys and actual inputs from all RouterInput dictionaries
        keys = [input_["key"] for input_ in inputs]
        actual_inputs = [input_["input"] for input_ in inputs]
        
        # Validate all keys exist in runnables mapping before execution
        if any(key not in self.runnables for key in keys):
            msg = "One or more keys do not have a corresponding runnable"
            raise ValueError(msg)

        # Define async wrapper function for exception handling in concurrent execution
        async def ainvoke(
            runnable: Runnable, input_: Input, config: RunnableConfig
        ) -> Output | Exception:
            if return_exceptions:
                try:
                    return await runnable.ainvoke(input_, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await runnable.ainvoke(input_, config, **kwargs)

        # Select Runnables for each input based on keys
        runnables = [self.runnables[key] for key in keys]
        
        # Prepare configs: expand single config to list or validate list length
        configs = get_config_list(config, len(inputs))
        
        # Execute all invocations concurrently with optional concurrency limit
        return await gather_with_concurrency(
            configs[0].get("max_concurrency"),
            *map(ainvoke, runnables, actual_inputs, configs),
        )

    @override
    def stream(
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        """Stream output from the selected Runnable synchronously.

        Routes to the selected Runnable based on input['key'] and streams its
        output incrementally. This is useful for streaming responses from LLMs
        or other Runnables that support incremental output generation.

        Args:
            input: RouterInput dictionary containing:
                - key (str): The routing key to select which Runnable to execute.
                  Must match a key in the runnables mapping.
                - input (Any): The actual input data to pass to the selected
                  Runnable. Type must match what the selected Runnable expects.
            config: Optional RunnableConfig to control streaming behavior. This
                config is propagated to the selected Runnable, including streaming
                callbacks, tags, and other execution parameters.
            **kwargs: Additional keyword arguments passed through to the selected
                Runnable's stream method.

        Yields:
            Output: Incremental output chunks from the selected Runnable. The exact
                type and granularity of chunks depends on the selected Runnable's
                implementation (e.g., tokens for LLMs, chunks for data processors).

        Raises:
            ValueError: If input['key'] does not exist in the runnables mapping.
                Error message format: "No runnable associated with key '{key}'"
            KeyError: If input dictionary is missing 'key' or 'input' fields.

        Example:
            >>> from langchain_core.runnables.router import RouterRunnable
            >>> from langchain_openai import ChatOpenAI
            >>> from langchain_core.prompts import ChatPromptTemplate
            >>> 
            >>> # Create streaming chains for different models
            >>> gpt4_chain = (
            ...     ChatPromptTemplate.from_template("Answer: {question}")
            ...     | ChatOpenAI(model="gpt-4", streaming=True)
            ... )
            >>> gpt35_chain = (
            ...     ChatPromptTemplate.from_template("Answer: {question}")
            ...     | ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
            ... )
            >>> 
            >>> router = RouterRunnable(runnables={
            ...     "detailed": gpt4_chain,
            ...     "quick": gpt35_chain,
            ... })
            >>> 
            >>> # Stream response token by token
            >>> for chunk in router.stream({
            ...     "key": "detailed",
            ...     "input": {"question": "What is quantum computing?"}
            ... }):
            ...     print(chunk.content, end="", flush=True)

        Source: libs/core/langchain_core/runnables/router.py:stream()
        """
        # Extract routing key from input
        key = input["key"]
        actual_input = input["input"]
        
        # Validate key exists in runnables mapping
        if key not in self.runnables:
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        # Select the Runnable based on the key
        runnable = self.runnables[key]
        
        # Stream output from selected Runnable with propagated config
        yield from runnable.stream(actual_input, config)

    @override
    async def astream(
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        """Stream output from the selected Runnable asynchronously.

        Asynchronous version of stream(). Routes to the selected Runnable based
        on input['key'] and streams its output incrementally using async iteration.
        This is useful for streaming responses from async LLMs or other async
        Runnables that support incremental output generation.

        Args:
            input: RouterInput dictionary containing:
                - key (str): The routing key to select which Runnable to execute.
                  Must match a key in the runnables mapping.
                - input (Any): The actual input data to pass to the selected
                  Runnable. Type must match what the selected Runnable expects.
            config: Optional RunnableConfig to control streaming behavior. This
                config is propagated to the selected Runnable, including async
                streaming callbacks, tags, and execution parameters.
            **kwargs: Additional keyword arguments passed through to the selected
                Runnable's astream method.

        Yields:
            Output: Incremental output chunks from the selected Runnable. The exact
                type and granularity of chunks depends on the selected Runnable's
                implementation (e.g., tokens for LLMs, chunks for data processors).

        Raises:
            ValueError: If input['key'] does not exist in the runnables mapping.
                Error message format: "No runnable associated with key '{key}'"
            KeyError: If input dictionary is missing 'key' or 'input' fields.

        Example:
            >>> import asyncio
            >>> from langchain_core.runnables.router import RouterRunnable
            >>> from langchain_openai import ChatOpenAI
            >>> from langchain_core.prompts import ChatPromptTemplate
            >>> 
            >>> async def stream_example():
            ...     # Create async streaming chains
            ...     gpt4_chain = (
            ...         ChatPromptTemplate.from_template("Answer: {question}")
            ...         | ChatOpenAI(model="gpt-4", streaming=True)
            ...     )
            ...     
            ...     router = RouterRunnable(runnables={"detailed": gpt4_chain})
            ...     
            ...     # Stream response token by token asynchronously
            ...     async for chunk in router.astream({
            ...         "key": "detailed",
            ...         "input": {"question": "What is quantum computing?"}
            ...     }):
            ...         print(chunk.content, end="", flush=True)
            >>> 
            >>> asyncio.run(stream_example())

        Source: libs/core/langchain_core/runnables/router.py:astream()
        """
        # Extract routing key from input
        key = input["key"]
        actual_input = input["input"]
        
        # Validate key exists in runnables mapping
        if key not in self.runnables:
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        # Select the Runnable based on the key
        runnable = self.runnables[key]
        
        # Stream output from selected Runnable asynchronously with propagated config
        async for output in runnable.astream(actual_input, config):
            yield output
