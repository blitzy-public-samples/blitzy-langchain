"""Runnable that selects which branch to run based on a condition.

This module provides conditional branching capabilities for LangChain Expression
Language (LCEL) chains, enabling dynamic routing of inputs to different processing
paths based on runtime conditions.

Key Concepts:
    - **Conditional Routing**: Route inputs to different Runnables based on boolean
      condition evaluation
    - **First-Match Semantics**: Conditions are evaluated in order; the first
      condition that returns True determines which branch executes
    - **Default Fallback**: If no conditions match, a default branch handles the input
    - **Type Safety**: All branches must accept the same input type and return the
      same output type

Use Cases:
    - Route user queries to different LLM prompts based on query type
    - Select different retrieval strategies based on document characteristics
    - Apply different validation logic based on input structure
    - Implement multi-way decision trees in chain workflows

Example:
    ```python
    from langchain_core.runnables import RunnableBranch

    # Route based on input type
    branch = RunnableBranch(
        (lambda x: isinstance(x, str), lambda x: x.upper()),
        (lambda x: isinstance(x, int), lambda x: x + 1),
        lambda x: "default",  # Default branch
    )
    ```

Source: libs/core/langchain_core/runnables/branch.py:1
"""

from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import (
    Any,
    cast,
)

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_core.runnables.base import (
    Runnable,
    RunnableLike,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    get_unique_config_specs,
)


class RunnableBranch(RunnableSerializable[Input, Output]):
    """Runnable that selects which branch to run based on a condition.

    RunnableBranch provides conditional routing capabilities for LCEL chains, allowing
    inputs to be dynamically routed to different processing branches based on runtime
    evaluation of boolean conditions.

    Initialization:
        The Runnable is initialized with a sequence of `(condition, runnable)` pairs
        followed by a default branch. Each condition is a Runnable[Input, bool] or
        callable that returns bool. Each branch runnable must accept the same Input
        type and return the same Output type.

    Execution Semantics:
        **First-Match Wins**: When processing an input:
        1. Conditions are evaluated sequentially in the order provided
        2. The first condition that evaluates to True selects its paired branch
        3. The selected branch runnable is invoked with the original input
        4. Subsequent conditions are not evaluated (short-circuit behavior)
        5. If no conditions match, the default branch executes

    Default Branch Behavior:
        The last argument provided during initialization becomes the default branch.
        This branch must be a Runnable or callable that accepts the input type.
        The default branch executes when:
        - All condition evaluations return False
        - The conditions list is empty (only default provided, though this requires
          at least 2 branches total)

    Type Flow:
        ```
        Input → Condition 1 → bool (False) → Continue
              → Condition 2 → bool (True)  → Branch 2 Runnable → Output
              (Condition 3+ not evaluated due to short-circuit)
        
        Input → All Conditions False → Default Branch → Output
        ```

    Thread Safety:
        RunnableBranch is thread-safe for concurrent invocations. Each invocation
        maintains its own execution context and callback state.

    Examples:
        Basic type-based routing:
        ```python
        from langchain_core.runnables import RunnableBranch

        branch = RunnableBranch(
            (lambda x: isinstance(x, str), lambda x: x.upper()),
            (lambda x: isinstance(x, int), lambda x: x + 1),
            (lambda x: isinstance(x, float), lambda x: x * 2),
            lambda x: "goodbye",  # Default branch
        )

        branch.invoke("hello")  # "HELLO"
        branch.invoke(None)     # "goodbye"
        ```

        Complex multi-branch routing with LCEL composition:
        ```python
        from langchain_core.runnables import RunnableBranch, RunnableLambda
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI()

        # Route based on input characteristics
        branch = RunnableBranch(
            (
                lambda x: len(x.get("query", "")) < 10,
                ChatPromptTemplate.from_template("Short query: {query}") | model
            ),
            (
                lambda x: "code" in x.get("query", "").lower(),
                ChatPromptTemplate.from_template("Code query: {query}") | model
            ),
            (
                lambda x: "?" in x.get("query", ""),
                ChatPromptTemplate.from_template("Question: {query}") | model
            ),
            ChatPromptTemplate.from_template("General query: {query}") | model
        )

        result = branch.invoke({"query": "How do I code this?"})
        # Routes to "Code query" branch (second condition matches first)
        ```

        Integration in larger LCEL chains:
        ```python
        chain = (
            {"query": RunnableLambda(lambda x: x["text"].strip())}
            | branch
            | RunnableLambda(lambda x: x.content)
        )
        ```

    Attributes:
        branches: Sequence of (condition, runnable) pairs evaluated in order
        default: Runnable executed when no conditions match

    Source: libs/core/langchain_core/runnables/branch.py:40-65
    """

    branches: Sequence[tuple[Runnable[Input, bool], Runnable[Input, Output]]]
    """A list of `(condition, Runnable)` pairs."""
    default: Runnable[Input, Output]
    """A `Runnable` to run if no condition is met."""

    def __init__(
        self,
        *branches: tuple[
            Runnable[Input, bool]
            | Callable[[Input], bool]
            | Callable[[Input], Awaitable[bool]],
            RunnableLike,
        ]
        | RunnableLike,
    ) -> None:
        """A `Runnable` that runs one of two branches based on a condition.

        Args:
            *branches: A list of `(condition, Runnable)` pairs.
                Defaults a `Runnable` to run if no condition is met.

        Raises:
            ValueError: If the number of branches is less than 2.
            TypeError: If the default branch is not `Runnable`, `Callable` or `Mapping`.
            TypeError: If a branch is not a tuple or list.
            ValueError: If a branch is not of length 2.
        """
        if len(branches) < 2:
            msg = "RunnableBranch requires at least two branches"
            raise ValueError(msg)

        default = branches[-1]

        if not isinstance(
            default,
            (Runnable, Callable, Mapping),  # type: ignore[arg-type]
        ):
            msg = "RunnableBranch default must be Runnable, callable or mapping."
            raise TypeError(msg)

        default_ = cast(
            "Runnable[Input, Output]", coerce_to_runnable(cast("RunnableLike", default))
        )

        branches_ = []

        for branch in branches[:-1]:
            if not isinstance(branch, (tuple, list)):
                msg = (
                    f"RunnableBranch branches must be "
                    f"tuples or lists, not {type(branch)}"
                )
                raise TypeError(msg)

            if len(branch) != 2:
                msg = (
                    f"RunnableBranch branches must be "
                    f"tuples or lists of length 2, not {len(branch)}"
                )
                raise ValueError(msg)
            condition, runnable = branch
            condition = cast("Runnable[Input, bool]", coerce_to_runnable(condition))
            runnable = coerce_to_runnable(runnable)
            branches_.append((condition, runnable))

        super().__init__(
            branches=branches_,
            default=default_,
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
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
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """Get the input schema by examining all branches and conditions.

        This method determines the input schema for the RunnableBranch by inspecting
        the input schemas of all conditions and branch runnables. It returns the first
        non-empty schema found, ensuring that the input type is well-defined.

        Schema Resolution Order:
            1. Check default branch input schema
            2. Check all branch runnables' input schemas
            3. Check all condition runnables' input schemas
            4. Fall back to parent class schema if all are empty

        The order prioritizes branch runnables over conditions because branches
        typically have more specific input requirements, while conditions often
        accept any input for evaluation.

        Args:
            config: Optional configuration that may affect schema generation. Some
                runnables adjust their input schema based on configuration (e.g.,
                configurable fields that add input parameters).

        Returns:
            A Pydantic BaseModel class representing the input schema. The schema
            defines:
            - Required fields and their types
            - Optional fields with defaults
            - Field descriptions and constraints
            - JSON schema representation for validation

            If no branch provides a specific schema (all return empty/generic schemas),
            returns the default schema from RunnableSerializable.

        Example:
            ```python
            from langchain_core.runnables import RunnableBranch, RunnableLambda
            from pydantic import BaseModel, Field

            class QueryInput(BaseModel):
                text: str = Field(description="Query text")
                category: str = Field(description="Query category")

            branch = RunnableBranch(
                (lambda x: x["category"] == "code", RunnableLambda(lambda x: x)),
                RunnableLambda(lambda x: x),
            )

            schema = branch.get_input_schema()
            print(schema.model_json_schema())
            # Shows schema accepting dict with 'category' key
            ```

        Note:
            - All branches and conditions must accept compatible input types
            - The returned schema represents the union of requirements across branches
            - Type safety: inputs must satisfy the returned schema to avoid runtime
              errors during condition evaluation or branch execution
            - Schema is computed dynamically based on branch composition

        Source: libs/core/langchain_core/runnables/branch.py:157-171
        """
        runnables = (
            [self.default]
            + [r for _, r in self.branches]
            + [r for r, _ in self.branches]
        )

        for runnable in runnables:
            if (
                runnable.get_input_schema(config).model_json_schema().get("type")
                is not None
            ):
                return runnable.get_input_schema(config)

        return super().get_input_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Collect all configurable field specifications from branches and conditions.

        This property aggregates the configurable field specifications from all
        components of the RunnableBranch (default branch, all branch runnables, and
        all condition runnables), removing duplicates to provide a unified view of
        runtime configuration options.

        Configurable fields allow runtime customization of runnable behavior without
        modifying the chain structure. For example, switching between different LLM
        models or adjusting retrieval parameters.

        Aggregation Process:
            1. Collect config_specs from default branch
            2. Collect config_specs from all branch runnables (right side of pairs)
            3. Collect config_specs from all condition runnables (left side of pairs)
            4. Deduplicate specifications with the same id using get_unique_config_specs()
            5. Return merged list of unique specifications

        Returns:
            A list of ConfigurableFieldSpec objects, each defining:
            - id: Unique identifier for the configurable field
            - name: Human-readable name
            - description: Explanation of the field's purpose
            - annotation: Type annotation for the field value
            - default: Optional default value
            - is_shared: Whether the field is shared across multiple components

            Empty list if no components have configurable fields.

        Example:
            ```python
            from langchain_core.runnables import RunnableBranch
            from langchain_openai import ChatOpenAI

            # Create branches with configurable models
            model = ChatOpenAI().configurable_fields(
                temperature=ConfigurableField(
                    id="temperature",
                    name="LLM Temperature",
                    description="Temperature for LLM sampling",
                )
            )

            branch = RunnableBranch(
                (lambda x: len(x) < 10, model),
                model,
            )

            # Get all configurable fields
            specs = branch.config_specs
            print([spec.id for spec in specs])  # ['temperature']

            # Use configuration at runtime
            result = branch.invoke(
                "Hello",
                config={"configurable": {"temperature": 0.9}}
            )
            ```

        Note:
            - Duplicate field specs (same id) are automatically merged
            - Shared fields affect all branches that declare them
            - Configuration is passed down to selected branch during execution
            - Useful for inspecting available runtime customization options
            - config_specs are static (don't change after initialization)

        Source: libs/core/langchain_core/runnables/branch.py:173-184
        """
        return get_unique_config_specs(
            spec
            for step in (
                [self.default]
                + [r for _, r in self.branches]
                + [r for r, _ in self.branches]
            )
            for spec in step.config_specs
        )

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """First evaluates the condition, then delegate to true or false branch.

        Args:
            input: The input to the Runnable.
            config: The configuration for the Runnable.
            **kwargs: Additional keyword arguments to pass to the Runnable.

        Returns:
            The output of the branch that was run.
        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        try:
            # Type flow: Input → Condition evaluation loop (first-match semantics)
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                # Type flow: Input → Runnable[Input, bool] → bool
                # Condition evaluates input to boolean, determining branch selection
                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                # Short-circuit on first True condition
                if expression_value:
                    # Type flow: Input → Runnable[Input, Output] → Output
                    # Selected branch processes original input to produce output
                    output = runnable.invoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                # Type flow: Input → Runnable[Input, Output] → Output (default path)
                # No conditions matched; default branch processes input
                output = self.default.invoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(output)
        return output

    @override
    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """Asynchronously evaluate conditions and execute the first matching branch.

        This method implements async execution of the conditional branching logic,
        evaluating conditions sequentially until one returns True, then executing
        the corresponding branch with async/await semantics.

        Execution Flow:
            1. Initialize async callback manager from config
            2. Trigger on_chain_start callback
            3. For each (condition, runnable) pair in order:
               - Await condition.ainvoke(input) to get boolean result
               - If True: await runnable.ainvoke(input) and return output
               - If False: continue to next condition
            4. If no conditions match: await default.ainvoke(input)
            5. Trigger on_chain_end callback with output

        Args:
            input: The input value to route through the branch logic. Must match
                the Input type parameter of the RunnableBranch.
            config: Optional configuration for this invocation, including:
                - callbacks: Async callbacks to track execution
                - tags: Tags for filtering callback events
                - metadata: Additional metadata for callbacks
                - run_name: Name for this execution run
                - run_id: Unique identifier for this run
            **kwargs: Additional keyword arguments passed to the selected branch's
                ainvoke() method. Common kwargs include model-specific parameters
                or execution options.

        Returns:
            The output from the selected branch (either a matching condition's
            runnable or the default branch). Type matches the Output type parameter
            of the RunnableBranch.

        Raises:
            BaseException: Any exception raised by condition evaluation or branch
                execution is caught, reported to callbacks via on_chain_error, and
                re-raised. Common exceptions include:
                - ValueError: Invalid input structure for condition evaluation
                - TypeError: Type mismatch between input and branch expectations
                - RuntimeError: Branch execution failures (e.g., API errors)

        Example:
            ```python
            import asyncio
            from langchain_core.runnables import RunnableBranch

            branch = RunnableBranch(
                (lambda x: x > 10, lambda x: f"Large: {x}"),
                (lambda x: x > 5, lambda x: f"Medium: {x}"),
                lambda x: f"Small: {x}",
            )

            async def run():
                result = await branch.ainvoke(7)
                print(result)  # "Medium: 7"

            asyncio.run(run())
            ```

        Note:
            - All conditions are evaluated using ainvoke(), even if the condition
              callable is synchronous (automatic async wrapping occurs)
            - Callbacks receive child tags like "condition:1", "branch:2", etc.
            - Short-circuit evaluation: once a condition matches, remaining
              conditions are not evaluated
            - For sync execution, use invoke() instead

        Source: libs/core/langchain_core/runnables/branch.py:246-291
        """
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    output = await runnable.ainvoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = await self.default.ainvoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(output)
        return output

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        """Stream output chunks from the selected branch after condition evaluation.

        This method evaluates conditions sequentially (non-streaming) to determine
        which branch to execute, then streams output chunks from the selected branch's
        stream() method. This enables real-time processing of outputs from branches
        that produce streaming results (e.g., LLM token generation).

        Execution Flow:
            1. Initialize callback manager from config
            2. Trigger on_chain_start callback
            3. For each (condition, runnable) pair in order:
               - Invoke condition.invoke(input) synchronously (non-streaming)
               - If True: stream from runnable.stream(input)
               - If False: continue to next condition
            4. If no conditions match: stream from default.stream(input)
            5. Accumulate chunks for final_output (if chunks support + operator)
            6. Trigger on_chain_end callback with accumulated final_output

        Streaming Behavior:
            - Conditions are evaluated synchronously (no streaming)
            - Only the selected branch's output is streamed
            - Each chunk is yielded immediately as it becomes available
            - Chunks are optionally accumulated via + operator for callbacks
            - If chunks don't support +, final_output is None for callbacks

        Args:
            input: The input value to route through the branch logic. Must match
                the Input type parameter of the RunnableBranch.
            config: Optional configuration for this invocation, including:
                - callbacks: Callbacks to track execution and receive chunks
                - tags: Tags for filtering callback events
                - metadata: Additional metadata for callbacks
                - run_name: Name for this execution run
                - run_id: Unique identifier for this run
            **kwargs: Additional keyword arguments passed to the selected branch's
                stream() method. May include model parameters or execution options.

        Yields:
            Output chunks from the selected branch. The type and structure of chunks
            depend on the branch runnable's implementation. Common patterns:
            - String chunks: Individual tokens or text segments
            - Message chunks: Partial AIMessage objects with incremental content
            - Dict chunks: Partial dictionary outputs with progressive fields

        Raises:
            BaseException: Any exception raised by condition evaluation or branch
                streaming is caught, reported to callbacks via on_chain_error, and
                re-raised. Common exceptions include:
                - ValueError: Invalid input structure for condition evaluation
                - TypeError: Type mismatch in branch execution
                - RuntimeError: Streaming errors from underlying models

        Example:
            ```python
            from langchain_core.runnables import RunnableBranch
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI()

            branch = RunnableBranch(
                (
                    lambda x: len(x) < 20,
                    lambda x: model.stream(f"Short query: {x}")
                ),
                lambda x: model.stream(f"Long query: {x}"),
            )

            # Stream tokens as they're generated
            for chunk in branch.stream("Hello!"):
                print(chunk.content, end="", flush=True)
            ```

        Note:
            - Condition evaluation is NOT streamed (evaluated synchronously first)
            - Only one branch streams output; others are not executed
            - Chunk accumulation attempts to build final_output for callbacks but
              gracefully handles non-addable chunk types
            - For async streaming, use astream() instead

        Source: libs/core/langchain_core/runnables/branch.py:293-375
        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Output | None = None
        final_output_supported = True

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    for chunk in runnable.stream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    ):
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore[operator]
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                for chunk in self.default.stream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore[operator]
                            except TypeError:
                                final_output = None
                                final_output_supported = False
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(final_output)

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        """Asynchronously stream output chunks from selected branch after condition evaluation.

        This method provides async streaming capabilities for conditional branching,
        evaluating conditions asynchronously to determine which branch to execute,
        then async streaming output chunks from the selected branch. This enables
        non-blocking real-time processing of outputs from async branches.

        Execution Flow:
            1. Initialize async callback manager from config
            2. Await on_chain_start callback
            3. For each (condition, runnable) pair in order:
               - Await condition.ainvoke(input) to get boolean result
               - If True: async stream from runnable.astream(input)
               - If False: continue to next condition
            4. If no conditions match: async stream from default.astream(input)
            5. Accumulate chunks for final_output (if chunks support + operator)
            6. Await on_chain_end callback with accumulated final_output

        Async Streaming Behavior:
            - Conditions are evaluated asynchronously using ainvoke()
            - Only the selected branch's output is streamed
            - Each chunk is yielded immediately without blocking
            - Chunks are accumulated via + operator if supported
            - Multiple concurrent astream() calls are independent

        Args:
            input: The input value to route through the branch logic. Must match
                the Input type parameter of the RunnableBranch.
            config: Optional configuration for this invocation, including:
                - callbacks: Async callbacks to track execution and receive chunks
                - tags: Tags for filtering callback events
                - metadata: Additional metadata for callbacks
                - run_name: Name for this execution run
                - run_id: Unique identifier for this run
            **kwargs: Additional keyword arguments passed to the selected branch's
                astream() method. May include model parameters or execution options.

        Yields:
            Output chunks from the selected branch. The type and structure of chunks
            depend on the branch runnable's async streaming implementation. Common
            patterns:
            - String chunks: Individual tokens or text segments
            - Message chunks: Partial AIMessage objects with incremental content
            - Dict chunks: Partial dictionary outputs with progressive fields

        Raises:
            BaseException: Any exception raised by async condition evaluation or
                branch streaming is caught, reported to callbacks via on_chain_error,
                and re-raised. Common exceptions include:
                - ValueError: Invalid input structure for condition evaluation
                - TypeError: Type mismatch in branch execution
                - RuntimeError: Async streaming errors from underlying models
                - asyncio.CancelledError: If streaming is cancelled

        Example:
            ```python
            import asyncio
            from langchain_core.runnables import RunnableBranch
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI()

            branch = RunnableBranch(
                (
                    lambda x: len(x) < 20,
                    lambda x: model.stream(f"Short: {x}")
                ),
                lambda x: model.stream(f"Long: {x}"),
            )

            async def stream_example():
                async for chunk in branch.astream("Hello world!"):
                    print(chunk.content, end="", flush=True)
                print()  # Newline after streaming completes

            asyncio.run(stream_example())
            ```

        Note:
            - All condition evaluations use ainvoke() (async, not streamed)
            - Only the selected branch's output is streamed asynchronously
            - Supports concurrent execution of multiple astream() calls
            - Chunk accumulation handles non-addable types gracefully
            - For sync streaming, use stream() instead
            - Proper async context management ensures cleanup on cancellation

        Source: libs/core/langchain_core/runnables/branch.py:377-459
        """
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Output | None = None
        final_output_supported = True

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    async for chunk in runnable.astream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    ):
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore[operator]
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                async for chunk in self.default.astream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore[operator]
                            except TypeError:
                                final_output = None
                                final_output_supported = False
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(final_output)
