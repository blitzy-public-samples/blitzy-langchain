"""Implementation of RunnablePassthrough and related data manipulation runnables.

This module provides Runnables for common data passthrough and manipulation patterns
in LCEL (LangChain Expression Language) chains:

- **RunnablePassthrough**: Passes input unchanged or optionally applies a side-effect
  function. Useful for preserving original data while computing derived values in
  parallel branches.

- **RunnableAssign**: Adds new keys to dict inputs by running parallel Runnables,
  merging results with the original input. Essential for enriching data with
  computed fields while preserving original keys.

- **RunnablePick**: Extracts specific keys from dict inputs, filtering unnecessary
  data and selecting relevant fields for downstream Runnables.

Common Use Cases:
    - Preserving input data while adding computed values in RunnableParallel
    - Enriching dict inputs with additional fields from parallel computations
    - Filtering dict data to pass only relevant keys to subsequent chain steps
    - Applying side effects (logging, validation) while passing data unchanged

Type Flow Patterns:
    - RunnablePassthrough: Input → (optional func) → Input (unchanged)
    - RunnableAssign: Dict[str, Any] → parallel mappers → Dict with new keys added
    - RunnablePick: Dict[str, Any] → key selection → Dict with only selected keys

Source: libs/core/langchain_core/runnables/passthrough.py:1
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from pydantic import BaseModel, RootModel
from typing_extensions import override

from langchain_core.runnables.base import (
    Other,
    Runnable,
    RunnableParallel,
    RunnableSerializable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
    ensure_config,
    get_executor_for_config,
    patch_config,
)
from langchain_core.runnables.utils import (
    AddableDict,
    ConfigurableFieldSpec,
)
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping

    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
    from langchain_core.runnables.graph import Graph


def identity(x: Other) -> Other:
    """Identity function.

    Args:
        x: input.

    Returns:
        output.
    """
    return x


async def aidentity(x: Other) -> Other:
    """Async identity function.

    Args:
        x: input.

    Returns:
        output.
    """
    return x


class RunnablePassthrough(RunnableSerializable[Other, Other]):
    """Runnable to passthrough inputs unchanged or with additional keys.

    This Runnable behaves almost like the identity function, except that it
    can be configured to add additional keys to the output, if the input is a
    dict.

    **Identity Function Behavior:**
    By default, RunnablePassthrough returns its input unchanged, acting as an
    identity function. This is useful in LCEL chains when you need to preserve
    the original input data while performing other operations in parallel.

    **Optional Transformation with func Parameter:**
    The `func` parameter allows you to specify a transformation function that is
    applied to the input as a side effect, but the original input is still returned
    unchanged. This is useful for logging, validation, or other side effects that
    don't modify the data flow. The function receives the input and optionally a
    RunnableConfig, but its return value is ignored.

    **When to Use RunnablePassthrough vs RunnableLambda:**
    - Use RunnablePassthrough when you want to preserve the input unchanged while
      optionally performing side effects (logging, validation).
    - Use RunnableLambda when you want to transform the input into a different output.
    - RunnablePassthrough is optimized for the passthrough pattern and integrates
      seamlessly with RunnableParallel for preserving original data.

    **Type Flow:**
    - Input → RunnablePassthrough() → Output (same as Input)
    - Input → RunnablePassthrough(func=side_effect) → Output (same as Input, after
      side_effect is called)

    **Common Patterns with RunnableParallel:**
    RunnablePassthrough is frequently used in RunnableParallel to preserve original
    input while computing derived values in other branches. This pattern is essential
    for enriching data without losing the original context.

    The examples below demonstrate this Runnable works using a few simple
    chains. The chains rely on simple lambdas to make the examples easy to execute
    and experiment with.

    Examples:
        ```python
        from langchain_core.runnables import (
            RunnableLambda,
            RunnableParallel,
            RunnablePassthrough,
        )

        runnable = RunnableParallel(
            origin=RunnablePassthrough(), modified=lambda x: x + 1
        )

        runnable.invoke(1)  # {'origin': 1, 'modified': 2}


        def fake_llm(prompt: str) -> str:  # Fake LLM for the example
            return "completion"


        chain = RunnableLambda(fake_llm) | {
            "original": RunnablePassthrough(),  # Original LLM output
            "parsed": lambda text: text[::-1],  # Parsing logic
        }

        chain.invoke("hello")  # {'original': 'completion', 'parsed': 'noitelpmoc'}
        ```

    In some cases, it may be useful to pass the input through while adding some
    keys to the output. In this case, you can use the `assign` method:

        ```python
        from langchain_core.runnables import RunnablePassthrough


        def fake_llm(prompt: str) -> str:  # Fake LLM for the example
            return "completion"


        runnable = {
            "llm1": fake_llm,
            "llm2": fake_llm,
        } | RunnablePassthrough.assign(
            total_chars=lambda inputs: len(inputs["llm1"] + inputs["llm2"])
        )

        runnable.invoke("hello")
        # {'llm1': 'completion', 'llm2': 'completion', 'total_chars': 20}
        ```

    Source: libs/core/langchain_core/runnables/passthrough.py:74
    """

    input_type: type[Other] | None = None

    func: Callable[[Other], None] | Callable[[Other, RunnableConfig], None] | None = (
        None
    )

    afunc: (
        Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None
    ) = None

    @override
    def __repr_args__(self) -> Any:
        # Without this repr(self) raises a RecursionError
        # See https://github.com/pydantic/pydantic/issues/7327
        return []

    def __init__(
        self,
        func: Callable[[Other], None]
        | Callable[[Other, RunnableConfig], None]
        | Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None = None,
        afunc: Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None = None,
        *,
        input_type: type[Other] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create e RunnablePassthrough.

        Args:
            func: Function to be called with the input.
            afunc: Async function to be called with the input.
            input_type: Type of the input.
        """
        if inspect.iscoroutinefunction(func):
            afunc = func
            func = None

        super().__init__(func=func, afunc=afunc, input_type=input_type, **kwargs)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return True as this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @property
    @override
    def InputType(self) -> Any:
        return self.input_type or Any

    @property
    @override
    def OutputType(self) -> Any:
        return self.input_type or Any

    @classmethod
    @override
    def assign(
        cls,
        **kwargs: Runnable[dict[str, Any], Any]
        | Callable[[dict[str, Any]], Any]
        | Mapping[str, Runnable[dict[str, Any], Any] | Callable[[dict[str, Any]], Any]],
    ) -> RunnableAssign:
        """Factory method for creating a RunnableAssign instance.

        This class method creates a RunnableAssign that merges the input dictionary
        with outputs produced by the provided Runnables or Callables. This pattern
        is essential for enriching data with computed fields while preserving all
        original keys.

        The method wraps the provided kwargs into a RunnableParallel, which executes
        all provided Runnables/Callables in parallel, then merges their outputs with
        the original input dictionary.

        Type Flow:
            Dict[str, Any] input → RunnableParallel(kwargs) → merge with input →
            Dict[str, Any] output (with new keys added)

        Args:
            **kwargs: Keyword arguments mapping new key names to Runnables or
                Callables. Each Runnable/Callable receives the input dict and
                produces a value for the corresponding key. Can be:
                - Runnable[dict[str, Any], Any]: A Runnable that processes the input
                - Callable[[dict[str, Any]], Any]: A function that processes the input
                - Mapping[str, Runnable | Callable]: A nested mapping of Runnables/Callables

        Returns:
            RunnableAssign: A Runnable that merges the input dict with the outputs
                produced by the mapping argument. Original keys are preserved,
                and new keys are added from the parallel execution results.

        Raises:
            ValueError: If the input to the returned RunnableAssign is not a dict
                (validated at runtime during invoke/ainvoke).

        Example:
            ```python
            from langchain_core.runnables import RunnablePassthrough

            # Create enrichment chain
            enriched = RunnablePassthrough.assign(
                total_length=lambda x: len(x["text"]),
                word_count=lambda x: len(x["text"].split()),
            )

            # Invoke with dict input
            result = enriched.invoke({"text": "Hello world"})
            # Returns: {'text': 'Hello world', 'total_length': 11, 'word_count': 2}
            ```

        Source: libs/core/langchain_core/runnables/passthrough.py:237
        """
        return RunnableAssign(RunnableParallel[dict[str, Any]](kwargs))

    @override
    def invoke(
        self, input: Other, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Other:
        """Invoke the passthrough on an input, returning the input unchanged.

        If a func was provided during initialization, it will be called with the
        input as a side effect (for logging, validation, etc.), but its return
        value is ignored and the original input is returned.

        Args:
            input: The input to pass through unchanged. Can be any type.
            config: Optional configuration for callbacks, tags, and metadata.
                Used to track execution in LangSmith or custom callback handlers.
            **kwargs: Additional keyword arguments passed to the func if provided.

        Returns:
            Other: The input value, unchanged. Type matches the input type.

        Raises:
            Exception: Any exception raised by the optional func will propagate.
                The identity passthrough itself does not raise exceptions.

        Example:
            ```python
            from langchain_core.runnables import RunnablePassthrough

            # Simple passthrough
            passthrough = RunnablePassthrough()
            result = passthrough.invoke("test")  # Returns: "test"

            # Passthrough with side effect
            def log_input(x):
                print(f"Processing: {x}")

            passthrough_with_logging = RunnablePassthrough(func=log_input)
            result = passthrough_with_logging.invoke("data")
            # Prints "Processing: data", returns "data"
            ```

        Source: libs/core/langchain_core/runnables/passthrough.py:258
        """
        if self.func is not None:
            call_func_with_variable_args(
                self.func, input, ensure_config(config), **kwargs
            )
        return self._call_with_config(identity, input, config)

    @override
    async def ainvoke(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Other:
        """Asynchronously invoke the passthrough, returning the input unchanged.

        If an afunc (async function) was provided during initialization, it will be
        awaited with the input as a side effect. If only func (sync) was provided,
        it will be called synchronously. The return value of either is ignored and
        the original input is returned.

        Args:
            input: The input to pass through unchanged. Can be any type.
            config: Optional configuration for callbacks, tags, and metadata.
                Used to track execution in LangSmith or custom callback handlers.
            **kwargs: Additional keyword arguments passed to afunc/func if provided.

        Returns:
            Other: The input value, unchanged. Type matches the input type.

        Raises:
            Exception: Any exception raised by the optional afunc/func will propagate.
                The identity passthrough itself does not raise exceptions.

        Example:
            ```python
            import asyncio
            from langchain_core.runnables import RunnablePassthrough

            # Simple async passthrough
            passthrough = RunnablePassthrough()
            result = await passthrough.ainvoke("test")  # Returns: "test"

            # Async passthrough with side effect
            async def async_log_input(x):
                await asyncio.sleep(0.1)  # Simulate async operation
                print(f"Processing: {x}")

            passthrough_with_logging = RunnablePassthrough(func=async_log_input)
            result = await passthrough_with_logging.ainvoke("data")
            # Prints "Processing: data" after delay, returns "data"
            ```

        Source: libs/core/langchain_core/runnables/passthrough.py:301
        """
        if self.afunc is not None:
            await acall_func_with_variable_args(
                self.afunc, input, ensure_config(config), **kwargs
            )
        elif self.func is not None:
            call_func_with_variable_args(
                self.func, input, ensure_config(config), **kwargs
            )
        return await self._acall_with_config(aidentity, input, config)

    @override
    def transform(
        self,
        input: Iterator[Other],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Other]:
        """Transform a streaming input by passing chunks through unchanged.

        This method handles streaming inputs (iterators) and passes each chunk
        through unchanged, supporting the streaming/token-by-token pattern in
        LCEL chains.

        **Streaming Behavior:**
        - Each chunk from the input iterator is yielded immediately
        - If func was provided, it is called ONCE with the final aggregated value
          after all chunks have been streamed
        - The func call happens after streaming completes (side effect at the end)

        **Chunk Aggregation Logic (when func is provided):**
        - Attempts to aggregate chunks using the + operator
        - If chunks are not addable (TypeError), uses the last chunk as final value
        - The aggregated/final value is passed to func after streaming

        Args:
            input: Iterator of input chunks to pass through. Each chunk is yielded
                unchanged to support streaming patterns.
            config: Optional configuration for callbacks, tags, and metadata.
            **kwargs: Additional keyword arguments passed to func if provided.

        Yields:
            Other: Each chunk from the input iterator, unchanged. Chunks are
                yielded immediately to support streaming.

        Raises:
            Exception: Any exception raised by the optional func will propagate
                after all chunks have been streamed.

        Example:
            ```python
            from langchain_core.runnables import RunnablePassthrough

            def log_final(x):
                print(f"Final: {x}")

            passthrough = RunnablePassthrough(func=log_final)

            # Stream chunks
            chunks = iter(["Hello", " ", "world"])
            result = list(passthrough.transform(chunks))
            # Yields: "Hello", " ", "world" immediately
            # Then prints: "Final: Hello world" after streaming completes
            ```

        Source: libs/core/langchain_core/runnables/passthrough.py:348
        """
        if self.func is None:
            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk
        else:
            final: Other
            got_first_chunk = False

            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk

                if not got_first_chunk:
                    final = chunk
                    got_first_chunk = True
                else:
                    try:
                        final = final + chunk  # type: ignore[operator]
                    except TypeError:
                        final = chunk

            if got_first_chunk:
                call_func_with_variable_args(
                    self.func, final, ensure_config(config), **kwargs
                )

    @override
    async def atransform(
        self,
        input: AsyncIterator[Other],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Other]:
        """Asynchronously transform streaming input by passing chunks through unchanged.

        This method handles async streaming inputs (async iterators) and passes
        each chunk through unchanged, supporting async streaming patterns in LCEL chains.

        **Async Streaming Behavior:**
        - Each chunk from the async input iterator is yielded immediately
        - If afunc (async) was provided, it is awaited ONCE with the final aggregated
          value after all chunks have been streamed
        - If only func (sync) was provided, it is called synchronously at the end
        - The afunc/func call happens after streaming completes (side effect at the end)

        **Chunk Aggregation Logic (when afunc/func is provided):**
        - Attempts to aggregate chunks using the + operator
        - If chunks are not addable (TypeError), uses the last chunk as final value
        - The aggregated/final value is passed to afunc/func after streaming

        Args:
            input: AsyncIterator of input chunks to pass through. Each chunk is
                yielded unchanged to support async streaming patterns.
            config: Optional configuration for callbacks, tags, and metadata.
            **kwargs: Additional keyword arguments passed to afunc/func if provided.

        Yields:
            Other: Each chunk from the input async iterator, unchanged. Chunks are
                yielded immediately to support async streaming.

        Raises:
            Exception: Any exception raised by the optional afunc/func will propagate
                after all chunks have been streamed.

        Example:
            ```python
            import asyncio
            from langchain_core.runnables import RunnablePassthrough

            async def async_log_final(x):
                await asyncio.sleep(0.1)
                print(f"Final: {x}")

            passthrough = RunnablePassthrough(func=async_log_final)

            # Stream chunks asynchronously
            async def chunk_generator():
                for chunk in ["Hello", " ", "world"]:
                    yield chunk

            result = []
            async for chunk in passthrough.atransform(chunk_generator()):
                result.append(chunk)
            # Yields: "Hello", " ", "world" immediately
            # Then prints: "Final: Hello world" after streaming completes
            ```

        Source: libs/core/langchain_core/runnables/passthrough.py:427
        """
        if self.afunc is None and self.func is None:
            async for chunk in self._atransform_stream_with_config(
                input, identity, config
            ):
                yield chunk
        else:
            got_first_chunk = False

            async for chunk in self._atransform_stream_with_config(
                input, identity, config
            ):
                yield chunk

                # By definitions, a function will operate on the aggregated
                # input. So we'll aggregate the input until we get to the last
                # chunk.
                # If the input is not addable, then we'll assume that we can
                # only operate on the last chunk.
                if not got_first_chunk:
                    final = chunk
                    got_first_chunk = True
                else:
                    try:
                        final = final + chunk  # type: ignore[operator]
                    except TypeError:
                        final = chunk

            if got_first_chunk:
                config = ensure_config(config)
                if self.afunc is not None:
                    await acall_func_with_variable_args(
                        self.afunc, final, config, **kwargs
                    )
                elif self.func is not None:
                    call_func_with_variable_args(self.func, final, config, **kwargs)

    @override
    def stream(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Other]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Other]:
        async def input_aiter() -> AsyncIterator[Other]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


_graph_passthrough: RunnablePassthrough = RunnablePassthrough()


class RunnableAssign(RunnableSerializable[dict[str, Any], dict[str, Any]]):
    """Runnable that assigns key-value pairs to dict[str, Any] inputs.

    The `RunnableAssign` class takes input dictionaries and, through a
    `RunnableParallel` instance, applies transformations, then combines
    these with the original data, introducing new key-value pairs based
    on the mapper's logic.

    **Relationship to RunnablePassthrough.assign():**
    RunnableAssign is typically created via the `RunnablePassthrough.assign()`
    factory method rather than instantiated directly. The factory method provides
    a convenient way to specify field assignments using keyword arguments.

    **Type Flow:**
    Dict[str, Any] input → RunnableParallel mapper (executes in parallel) →
    merge outputs with input → Dict[str, Any] output (original + new keys)

    **Key Preservation and Merging Logic:**
    - ALL original keys from the input dict are preserved in the output
    - New keys are added from the mapper's parallel execution results
    - If a mapper key conflicts with an input key, the mapper's value overwrites it
    - The merge uses Python's dict unpacking: {**input, **mapper_output}

    **Use Cases for Enriching Data:**
    - Adding computed fields (e.g., text length, token count) to existing data
    - Enriching documents with metadata from parallel lookups
    - Augmenting inputs with results from multiple parallel operations
    - Building complex multi-step chains where each step adds context

    **Type Constraints:**
    - Input MUST be a dict[str, Any] - raises ValueError otherwise
    - Output is always dict[str, Any] with original keys + new assigned keys
    - Mapper Runnables/Callables receive the full input dict

    Examples:
        ```python
        # This is a RunnableAssign
        from langchain_core.runnables.passthrough import (
            RunnableAssign,
            RunnableParallel,
        )
        from langchain_core.runnables.base import RunnableLambda


        def add_ten(x: dict[str, int]) -> dict[str, int]:
            return {"added": x["input"] + 10}


        mapper = RunnableParallel(
            {
                "add_step": RunnableLambda(add_ten),
            }
        )

        runnable_assign = RunnableAssign(mapper)

        # Synchronous example
        runnable_assign.invoke({"input": 5})
        # returns {'input': 5, 'add_step': {'added': 15}}

        # Asynchronous example
        await runnable_assign.ainvoke({"input": 5})
        # returns {'input': 5, 'add_step': {'added': 15}}
        ```

    Source: libs/core/langchain_core/runnables/passthrough.py:477
    """

    mapper: RunnableParallel

    def __init__(self, mapper: RunnableParallel[dict[str, Any]], **kwargs: Any) -> None:
        """Create a RunnableAssign.

        Args:
            mapper: A `RunnableParallel` instance that will be used to transform the
                input dictionary.
        """
        super().__init__(mapper=mapper, **kwargs)

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
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        name = (
            name
            or self.name
            or f"RunnableAssign<{','.join(self.mapper.steps__.keys())}>"
        )
        return super().get_name(suffix, name=name)

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        if not issubclass(map_input_schema, RootModel):
            # ie. it's a dict
            return map_input_schema

        return super().get_input_schema(config)

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        map_output_schema = self.mapper.get_output_schema(config)
        if not issubclass(map_input_schema, RootModel) and not issubclass(
            map_output_schema, RootModel
        ):
            fields = {}

            for name, field_info in map_input_schema.model_fields.items():
                fields[name] = (field_info.annotation, field_info.default)

            for name, field_info in map_output_schema.model_fields.items():
                fields[name] = (field_info.annotation, field_info.default)

            return create_model_v2("RunnableAssignOutput", field_definitions=fields)
        if not issubclass(map_output_schema, RootModel):
            # ie. only map output is a dict
            # ie. input type is either unknown or inferred incorrectly
            return map_output_schema

        return super().get_output_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self.mapper.config_specs

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        # get graph from mapper
        graph = self.mapper.get_graph(config)
        # add passthrough node and edges
        input_node = graph.first_node()
        output_node = graph.last_node()
        if input_node is not None and output_node is not None:
            passthrough_node = graph.add_node(_graph_passthrough)
            graph.add_edge(input_node, passthrough_node)
            graph.add_edge(passthrough_node, output_node)
        return graph

    def _invoke(
        self,
        value: dict[str, Any],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Internal method to invoke the mapper and merge results with input.

        This method performs the core assignment logic:
        1. Validates input is a dict
        2. Invokes the mapper RunnableParallel with the input dict
        3. Merges mapper outputs with original input using dict unpacking

        Args:
            value: Input dictionary to enrich with assigned fields.
            run_manager: Callback manager for tracking execution.
            config: Configuration with callbacks, tags, metadata.
            **kwargs: Additional arguments passed to mapper.

        Returns:
            dict[str, Any]: Merged dictionary with original keys + new assigned keys.
                Original input keys are preserved, mapper output keys are added.

        Raises:
            ValueError: If value is not a dict instance.

        Source: libs/core/langchain_core/runnables/passthrough.py:605
        """
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

        return {
            **value,
            **self.mapper.invoke(
                value,
                patch_config(config, callbacks=run_manager.get_child()),
                **kwargs,
            ),
        }

    @override
    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the key picker, extracting specified keys from input dict.

        Extracts only the keys specified during initialization from the input
        dictionary. This is useful for filtering data before passing to downstream
        Runnables.

        Args:
            input: Input dictionary to extract keys from. Must be a dict[str, Any].
            config: Optional configuration for callbacks, tags, and metadata.
                Used to track execution in LangSmith or custom callback handlers.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            dict[str, Any] | Any | None: Extracted value(s) based on keys:
                - If self.keys is a single string: returns the value for that key
                  (or None if key not in input)
                - If self.keys is a list: returns a dict containing only the
                  specified keys that exist in input
                - If none of the specified keys are found: returns None

        Raises:
            ValueError: If input is not a dict instance. RunnablePick requires
                dict inputs to perform key selection.

        Example:
            ```python
            from langchain_core.runnables.passthrough import RunnablePick

            # Pick single key
            picker_single = RunnablePick(keys="name")
            result = picker_single.invoke({"name": "Alice", "age": 30})
            # Returns: "Alice"

            # Pick multiple keys
            picker_multi = RunnablePick(keys=["name", "age"])
            result = picker_multi.invoke({"name": "Alice", "age": 30, "city": "NYC"})
            # Returns: {'name': 'Alice', 'age': 30}

            # Missing keys are omitted
            result = picker_multi.invoke({"name": "Bob"})
            # Returns: {'name': 'Bob'}
            ```

        Source: libs/core/langchain_core/runnables/passthrough.py:1041
        """
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        value: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Internal async method to invoke mapper and merge results with input.

        This async method performs the core assignment logic asynchronously:
        1. Validates input is a dict
        2. Awaits the mapper RunnableParallel ainvoke with the input dict
        3. Merges mapper outputs with original input using dict unpacking

        Args:
            value: Input dictionary to enrich with assigned fields.
            run_manager: Async callback manager for tracking execution.
            config: Configuration with callbacks, tags, metadata.
            **kwargs: Additional arguments passed to mapper.

        Returns:
            dict[str, Any]: Merged dictionary with original keys + new assigned keys.
                Original input keys are preserved, mapper output keys are added.

        Raises:
            ValueError: If value is not a dict instance.

        Source: libs/core/langchain_core/runnables/passthrough.py:690
        """
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

        return {
            **value,
            **await self.mapper.ainvoke(
                value,
                patch_config(config, callbacks=run_manager.get_child()),
                **kwargs,
            ),
        }

    @override
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Asynchronously invoke the key picker, extracting specified keys.

        Asynchronously extracts only the keys specified during initialization from
        the input dictionary. This is useful for filtering data in async chains
        before passing to downstream Runnables.

        Args:
            input: Input dictionary to extract keys from. Must be a dict[str, Any].
            config: Optional configuration for callbacks, tags, and metadata.
                Used to track execution in LangSmith or custom callback handlers.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            dict[str, Any] | Any | None: Extracted value(s) based on keys:
                - If self.keys is a single string: returns the value for that key
                  (or None if key not in input)
                - If self.keys is a list: returns a dict containing only the
                  specified keys that exist in input
                - If none of the specified keys are found: returns None

        Raises:
            ValueError: If input is not a dict instance. RunnablePick requires
                dict inputs to perform key selection.

        Example:
            ```python
            import asyncio
            from langchain_core.runnables.passthrough import RunnablePick

            # Pick single key
            picker_single = RunnablePick(keys="name")
            result = await picker_single.ainvoke({"name": "Alice", "age": 30})
            # Returns: "Alice"

            # Pick multiple keys in async chain
            picker_multi = RunnablePick(keys=["name", "age"])
            result = await picker_multi.ainvoke({
                "name": "Alice",
                "age": 30,
                "city": "NYC"
            })
            # Returns: {'name': 'Alice', 'age': 30}
            ```

        Source: libs/core/langchain_core/runnables/passthrough.py:1101
        """
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        values: Iterator[dict[str, Any]],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Internal method to transform streaming input by merging with mapper output.

        This method handles streaming dict inputs and merges them with mapper outputs.
        
        **Streaming Merge Logic:**
        1. Splits input stream into two: one for passthrough, one for mapper
        2. Starts mapper transformation in background (parallel execution)
        3. Yields passthrough chunks immediately, filtering out keys that will be
           assigned by mapper (to avoid duplication)
        4. After passthrough completes, yields all mapper output chunks
        
        **Key Filtering:** Passthrough chunks have mapper keys removed to prevent
        duplication. Mapper outputs overwrite/add keys after passthrough completes.

        Args:
            values: Iterator of dict chunks to transform and enrich.
            run_manager: Callback manager for tracking execution.
            config: Configuration with callbacks, tags, metadata.
            **kwargs: Additional arguments passed to mapper.

        Yields:
            dict[str, Any]: Chunks with merged data. First yields filtered passthrough
                chunks (original keys except mapper keys), then yields mapper output
                chunks with assigned fields.

        Raises:
            ValueError: If any chunk in values is not a dict instance.

        Source: libs/core/langchain_core/runnables/passthrough.py:767
        """
        # collect mapper keys
        mapper_keys = set(self.mapper.steps__.keys())
        # create two streams, one for the map and one for the passthrough
        for_passthrough, for_map = safetee(values, 2, lock=threading.Lock())

        # create map output stream
        map_output = self.mapper.transform(
            for_map,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
            ),
            **kwargs,
        )

        # get executor to start map output stream in background
        with get_executor_for_config(config) as executor:
            # start map output stream
            first_map_chunk_future = executor.submit(
                next,
                map_output,
                None,
            )
            # consume passthrough stream
            for chunk in for_passthrough:
                if not isinstance(chunk, dict):
                    msg = "The input to RunnablePassthrough.assign() must be a dict."
                    raise ValueError(msg)  # noqa: TRY004
                # remove mapper keys from passthrough chunk, to be overwritten by map
                filtered = AddableDict(
                    {k: v for k, v in chunk.items() if k not in mapper_keys}
                )
                if filtered:
                    yield filtered
            # yield map output
            yield cast("dict[str, Any]", first_map_chunk_future.result())
            for chunk in map_output:
                yield chunk

    @override
    def transform(
        self,
        input: Iterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        values: AsyncIterator[dict[str, Any]],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Internal async method to transform streaming input by merging with mapper.

        This async method handles streaming dict inputs asynchronously and merges
        them with mapper outputs.
        
        **Async Streaming Merge Logic:**
        1. Splits async input stream into two: one for passthrough, one for mapper
        2. Starts mapper atransform as async task (parallel execution)
        3. Yields passthrough chunks immediately, filtering out keys that will be
           assigned by mapper (to avoid duplication)
        4. After passthrough completes, awaits and yields all mapper output chunks
        
        **Key Filtering:** Passthrough chunks have mapper keys removed to prevent
        duplication. Mapper outputs overwrite/add keys after passthrough completes.

        Args:
            values: AsyncIterator of dict chunks to transform and enrich.
            run_manager: Async callback manager for tracking execution.
            config: Configuration with callbacks, tags, metadata.
            **kwargs: Additional arguments passed to mapper.

        Yields:
            dict[str, Any]: Chunks with merged data. First yields filtered passthrough
                chunks (original keys except mapper keys), then yields mapper output
                chunks with assigned fields.

        Raises:
            ValueError: If any chunk in values is not a dict instance.

        Source: libs/core/langchain_core/runnables/passthrough.py:851
        """
        # collect mapper keys
        mapper_keys = set(self.mapper.steps__.keys())
        # create two streams, one for the map and one for the passthrough
        for_passthrough, for_map = atee(values, 2, lock=asyncio.Lock())
        # create map output stream
        map_output = self.mapper.atransform(
            for_map,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
            ),
            **kwargs,
        )
        # start map output stream
        first_map_chunk_task: asyncio.Task = asyncio.create_task(
            py_anext(map_output, None),  # type: ignore[arg-type]
        )
        # consume passthrough stream
        async for chunk in for_passthrough:
            if not isinstance(chunk, dict):
                msg = "The input to RunnablePassthrough.assign() must be a dict."
                raise ValueError(msg)  # noqa: TRY004

            # remove mapper keys from passthrough chunk, to be overwritten by map output
            filtered = AddableDict(
                {k: v for k, v in chunk.items() if k not in mapper_keys}
            )
            if filtered:
                yield filtered
        # yield map output
        yield await first_map_chunk_task
        async for chunk in map_output:
            yield chunk

    @override
    async def atransform(
        self,
        input: AsyncIterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    @override
    def stream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[dict[str, Any]]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


class RunnablePick(RunnableSerializable[dict[str, Any], dict[str, Any]]):
    """Runnable that picks keys from dict[str, Any] inputs.

    RunnablePick class represents a Runnable that selectively picks keys from a
    dictionary input. It allows you to specify one or more keys to extract
    from the input dictionary. It returns a new dictionary containing only
    the selected keys.

    **Purpose:**
    RunnablePick is used to filter dict inputs by extracting only the keys needed
    by downstream Runnables. This reduces data passing overhead and makes chain
    logic clearer by explicitly selecting relevant fields.

    **Type Flow:**
    - Single key: Dict[str, Any] → key selection → Any (single value)
    - Multiple keys: Dict[str, Any] → key selection → Dict[str, Any] (subset)
    
    **Behavior Details:**
    - If keys is a single string, returns the value for that key (or None if missing)
    - If keys is a list of strings, returns a dict with only those keys present
      in the input
    - Missing keys are silently omitted (no KeyError raised)
    - If no specified keys are found in input, returns None

    **Use Cases:**
    - Filtering unnecessary data before passing to LLMs (reduce token count)
    - Selecting relevant fields for specific processing steps
    - Extracting specific outputs from complex chain results
    - Preparing focused inputs for downstream Runnables that expect specific keys

    **Integration in LCEL Chains:**
    RunnablePick is commonly used after operations that produce large dicts
    (like RunnableParallel or document retrievers) to extract only the fields
    needed for subsequent steps.

    Example:
        ```python
        from langchain_core.runnables.passthrough import RunnablePick

        input_data = {
            "name": "John",
            "age": 30,
            "city": "New York",
            "country": "USA",
        }

        runnable = RunnablePick(keys=["name", "age"])

        output_data = runnable.invoke(input_data)

        print(output_data)  # Output: {'name': 'John', 'age': 30}
        ```

    Source: libs/core/langchain_core/runnables/passthrough.py:920
    """

    keys: str | list[str]

    def __init__(self, keys: str | list[str], **kwargs: Any) -> None:
        """Create a RunnablePick.

        Args:
            keys: A single key or a list of keys to pick from the input dictionary.
        """
        super().__init__(keys=keys, **kwargs)

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
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        name = (
            name
            or self.name
            or "RunnablePick"
            f"<{','.join([self.keys] if isinstance(self.keys, str) else self.keys)}>"
        )
        return super().get_name(suffix, name=name)

    def _pick(self, value: dict[str, Any]) -> Any:
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

        if isinstance(self.keys, str):
            return value.get(self.keys)
        picked = {k: value.get(k) for k in self.keys if k in value}
        if picked:
            return AddableDict(picked)
        return None

    def _invoke(
        self,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """Internal method to pick specified keys from the input dict.

        Args:
            value: Input dictionary to pick keys from.

        Returns:
            dict[str, Any] | Any | None: Picked value(s). If keys is a single
                string, returns the value for that key. If keys is a list,
                returns a dict with only those keys. Returns None if no keys found.

        Source: libs/core/langchain_core/runnables/passthrough.py:1029
        """
        return self._pick(value)

    @override
    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the assignment, enriching input dict with new computed fields.

        Executes the mapper RunnableParallel with the input dict, then merges
        the results with the original input. All original keys are preserved,
        and new keys from the mapper outputs are added.

        Args:
            input: Input dictionary to enrich. Must be a dict[str, Any].
                All Runnables/Callables in the mapper receive this full dict.
            config: Optional configuration for callbacks, tags, and metadata.
                Used to track execution in LangSmith or custom callback handlers.
            **kwargs: Additional keyword arguments passed to the mapper.

        Returns:
            dict[str, Any]: Merged dictionary containing all original keys from
                input plus new keys computed by the mapper. If mapper keys conflict
                with input keys, mapper values overwrite input values.

        Raises:
            ValueError: If input is not a dict instance. RunnableAssign requires
                dict inputs to perform key assignment.

        Example:
            ```python
            from langchain_core.runnables import RunnablePassthrough

            enricher = RunnablePassthrough.assign(
                length=lambda x: len(x["text"]),
                uppercase=lambda x: x["text"].upper(),
            )

            result = enricher.invoke({"text": "hello"})
            # Returns: {'text': 'hello', 'length': 5, 'uppercase': 'HELLO'}
            ```

        Source: libs/core/langchain_core/runnables/passthrough.py:651
        """
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """Internal async method to pick specified keys from the input dict.

        Args:
            value: Input dictionary to pick keys from.

        Returns:
            dict[str, Any] | Any | None: Picked value(s). If keys is a single
                string, returns the value for that key. If keys is a list,
                returns a dict with only those keys. Returns None if no keys found.

        Source: libs/core/langchain_core/runnables/passthrough.py:1089
        """
        return self._pick(value)

    @override
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Asynchronously invoke the assignment, enriching input with new fields.

        Asynchronously executes the mapper RunnableParallel with the input dict,
        then merges the results with the original input. All original keys are
        preserved, and new keys from the mapper outputs are added. Mapper
        Runnables execute in parallel when possible.

        Args:
            input: Input dictionary to enrich. Must be a dict[str, Any].
                All Runnables/Callables in the mapper receive this full dict.
            config: Optional configuration for callbacks, tags, and metadata.
                Used to track execution in LangSmith or custom callback handlers.
            **kwargs: Additional keyword arguments passed to the mapper.

        Returns:
            dict[str, Any]: Merged dictionary containing all original keys from
                input plus new keys computed by the mapper. If mapper keys conflict
                with input keys, mapper values overwrite input values.

        Raises:
            ValueError: If input is not a dict instance. RunnableAssign requires
                dict inputs to perform key assignment.

        Example:
            ```python
            import asyncio
            from langchain_core.runnables import RunnablePassthrough

            async def async_compute_length(x):
                await asyncio.sleep(0.1)  # Simulate async operation
                return len(x["text"])

            enricher = RunnablePassthrough.assign(
                length=async_compute_length,
                uppercase=lambda x: x["text"].upper(),
            )

            result = await enricher.ainvoke({"text": "hello"})
            # Returns: {'text': 'hello', 'length': 5, 'uppercase': 'HELLO'}
            ```

        Source: libs/core/langchain_core/runnables/passthrough.py:720
        """
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        chunks: Iterator[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        for chunk in chunks:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    @override
    def transform(
        self,
        input: Iterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        chunks: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in chunks:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    @override
    async def atransform(
        self,
        input: AsyncIterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    @override
    def stream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[dict[str, Any]]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk
