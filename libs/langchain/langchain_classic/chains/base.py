"""Base interface that all chains should implement."""

import builtins
import contextlib
import inspect
import json
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    BaseCallbackManager,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.outputs import RunInfo
from langchain_core.runnables import (
    RunnableConfig,
    RunnableSerializable,
    ensure_config,
    run_in_executor,
)
from langchain_core.utils.pydantic import create_model
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import override

from langchain_classic.base_memory import BaseMemory
from langchain_classic.schema import RUN_KEY

logger = logging.getLogger(__name__)


def _get_verbosity() -> bool:
    from langchain_classic.globals import get_verbose

    return get_verbose()


class Chain(RunnableSerializable[dict[str, Any], dict[str, Any]], ABC):
    """Abstract base class for creating structured sequences of calls to components.

    Chains should be used to encode a sequence of calls to components like
    models, document retrievers, other chains, etc., and provide a simple interface
    to this sequence.

    The Chain interface makes it easy to create apps that are:
        - Stateful: add Memory to any Chain to give it state,
        - Observable: pass Callbacks to a Chain to execute additional functionality,
            like logging, outside the main sequence of component calls,
        - Composable: the Chain API is flexible enough that it is easy to combine
            Chains with other components, including other Chains.

    The main methods exposed by chains are:
        - `__call__`: Chains are callable. The `__call__` method is the primary way to
            execute a Chain. This takes inputs as a dictionary and returns a
            dictionary output.
        - `run`: A convenience method that takes inputs as args/kwargs and returns the
            output as a string or object. This method can only be used for a subset of
            chains and cannot return as rich of an output as `__call__`.

    Chain Execution Lifecycle:
        The complete execution flow follows this sequence:
        1. invoke() receives user input dict and optional RunnableConfig
        2. prep_inputs() merges memory variables via memory.load_memory_variables()
        3. on_chain_start callback fires with complete inputs
        4. _validate_inputs() checks all required input_keys are present
        5. _call() executes the chain's core logic (subclass implementation)
        6. _validate_outputs() checks all required output_keys are present
        7. prep_outputs() saves conversation via memory.save_context()
        8. on_chain_end callback fires with outputs dict
        9. Final outputs dict returned (optionally merged with inputs)

    Memory Integration:
        When self.memory is configured:
        - At execution start: memory.load_memory_variables() adds contextual variables
          (e.g., chat_history) to the inputs dict before _call() execution
        - At execution end: memory.save_context() persists the conversation (inputs
          and outputs) for future retrieval
        This enables conversational chains that maintain context across multiple
        invocations.

    Callback Orchestration:
        Callbacks fire at key lifecycle events:
        - on_chain_start: Fired after prep_inputs, before _call with complete inputs
        - on_chain_end: Fired after successful _call with outputs dict
        - on_chain_error: Fired if _call raises an exception
        Callbacks enable logging, tracing, metrics collection, and custom hooks.
        
    Source: libs/langchain/langchain_classic/chains/base.py:52-73
    """

    memory: BaseMemory | None = None
    """Optional memory object.
    Memory is a class that gets called at the start
    and at the end of every chain. At the start, memory loads variables and passes
    them along in the chain. At the end, it saves any returned variables.
    There are many different types of memory - please see memory docs
    for the full catalog."""
    callbacks: Callbacks = Field(default=None, exclude=True)
    """Optional list of callback handlers (or callback manager).
    Callback handlers are called throughout the lifecycle of a call to a chain,
    starting with on_chain_start, ending with on_chain_end or on_chain_error.
    Each custom chain can optionally call additional callback methods, see Callback docs
    for full details."""
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether or not run in verbose mode. In verbose mode, some intermediate logs
    will be printed to the console. Defaults to the global `verbose` value,
    accessible via `langchain.globals.get_verbose()`."""
    tags: list[str] | None = None
    """Optional list of tags associated with the chain.
    These tags will be associated with each call to this chain,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a chain with its use case.
    """
    metadata: builtins.dict[str, Any] | None = None
    """Optional metadata associated with the chain.
    This metadata will be associated with each call to this chain,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a chain with its use case.
    """
    callback_manager: BaseCallbackManager | None = Field(default=None, exclude=True)
    """[DEPRECATED] Use `callbacks` instead."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @override
    def get_input_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        # This is correct, but pydantic typings/mypy don't think so.
        return create_model("ChainInput", **dict.fromkeys(self.input_keys, (Any, None)))

    @override
    def get_output_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        # This is correct, but pydantic typings/mypy don't think so.
        return create_model(
            "ChainOutput",
            **dict.fromkeys(self.output_keys, (Any, None)),
        )

    @override
    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the chain with synchronous callbacks and memory integration.

        This is the primary user-facing method for chain execution. It orchestrates
        the complete lifecycle: memory loading, callback invocation, input/output
        validation, core chain logic execution, and memory persistence.

        Args:
            input: Dict[str, Any] containing keys from self.input_keys. If self.memory
                is configured, memory variables (e.g., chat_history) are excluded from
                required input keys as they will be loaded automatically.
            config: Optional RunnableConfig with fields:
                - callbacks: Callbacks for this specific run (override instance callbacks)
                - tags: List[str] tags for organizing/filtering runs
                - metadata: Dict[str, Any] arbitrary metadata for this run
                - run_name: str custom name for this run (defaults to class name)
                - run_id: UUID for tracking this specific execution
            **kwargs: Additional execution options:
                - include_run_info (bool): If True, adds RUN_KEY with RunInfo object
                  containing run_id to final outputs. Defaults to False.
                - return_only_outputs (bool): If True, returns only output_keys in
                  result dict. If False (default), merges input and output keys.

        Returns:
            Dict[str, Any] with structure depending on return_only_outputs:
            - If return_only_outputs=False (default): {**inputs, **outputs} containing
              all input keys and all output keys from self.output_keys
            - If return_only_outputs=True: Only keys from self.output_keys
            - If include_run_info=True: Additional RUN_KEY entry with RunInfo(run_id)

        Raises:
            ValueError: If input is missing required keys from self.input_keys (after
                excluding memory variables), or if _call() returns dict missing keys
                from self.output_keys.
            Exception: Any exception raised by subclass _call() implementation
                (e.g., API errors, validation failures, runtime errors). Exception is
                propagated after on_chain_error callback fires.

        Type Flow:
            input dict → prep_inputs (merges memory variables) → validated inputs →
            _call (subclass logic) → outputs dict → prep_outputs (saves to memory) →
            final outputs (optionally merged with inputs)

        Callback Integration:
            1. CallbackManager.configure() merges runtime and instance callbacks
            2. on_chain_start() fires with complete inputs dict, returns run_manager
            3. _call() execution (run_manager passed for nested callbacks)
            4. on_chain_end() fires with outputs dict on success
            5. on_chain_error() fires with exception on failure

        Example:
            >>> from langchain_classic.chains import LLMChain
            >>> from langchain_classic.prompts import PromptTemplate
            >>> chain = LLMChain(
            ...     llm=llm,
            ...     prompt=PromptTemplate.from_template("Tell me about {topic}")
            ... )
            >>> result = chain.invoke({"topic": "langchain"})
            >>> print(result)  # {"topic": "langchain", "text": "LangChain is..."}

        Source: libs/langchain/langchain_classic/chains/base.py:131-184
        """
        config = ensure_config(config)
        callbacks = config.get("callbacks")
        tags = config.get("tags")
        metadata = config.get("metadata")
        run_name = config.get("run_name") or self.get_name()
        run_id = config.get("run_id")
        include_run_info = kwargs.get("include_run_info", False)
        return_only_outputs = kwargs.get("return_only_outputs", False)

        # Merge memory variables into input dict (if memory configured)
        # prep_inputs() calls memory.load_memory_variables() to add context
        inputs = self.prep_inputs(input)
        
        # Configure callback manager by merging runtime and instance callbacks
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")

        # Fire on_chain_start callback with complete inputs dict
        # Returns run_manager for nested callbacks during _call execution
        run_manager = callback_manager.on_chain_start(
            None,
            inputs,
            run_id,
            name=run_name,
        )
        try:
            # Validate all required self.input_keys are present in inputs dict
            self._validate_inputs(inputs)
            
            # Execute core chain logic (abstract method implemented by subclasses)
            # Inputs dict includes original keys + memory variables
            outputs = (
                self._call(inputs, run_manager=run_manager)
                if new_arg_supported
                else self._call(inputs)
            )

            # Validate outputs, save to memory, optionally merge with inputs
            # prep_outputs() calls memory.save_context() to persist conversation
            final_outputs: dict[str, Any] = self.prep_outputs(
                inputs,
                outputs,
                return_only_outputs,
            )
        except BaseException as e:
            # Fire on_chain_error callback with exception before propagating
            run_manager.on_chain_error(e)
            raise
        
        # Fire on_chain_end callback with outputs dict on successful execution
        run_manager.on_chain_end(outputs)

        if include_run_info:
            final_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        return final_outputs

    @override
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the chain asynchronously with async callbacks and memory integration.

        This is the async variant of invoke(). It provides the same lifecycle
        orchestration but uses async callbacks, async memory operations
        (aload_memory_variables, asave_context), and awaits the _acall() method.

        Args:
            input: Dict[str, Any] containing keys from self.input_keys. If self.memory
                is configured, memory variables (e.g., chat_history) are excluded from
                required input keys as they will be loaded automatically via async
                memory.aload_memory_variables().
            config: Optional RunnableConfig with fields:
                - callbacks: Async callbacks for this specific run
                - tags: List[str] tags for organizing/filtering runs
                - metadata: Dict[str, Any] arbitrary metadata for this run
                - run_name: str custom name for this run (defaults to class name)
                - run_id: UUID for tracking this specific execution
            **kwargs: Additional execution options:
                - include_run_info (bool): If True, adds RUN_KEY with RunInfo object
                - return_only_outputs (bool): If True, returns only output_keys

        Returns:
            Dict[str, Any] with same structure as invoke():
            - If return_only_outputs=False (default): {**inputs, **outputs}
            - If return_only_outputs=True: Only keys from self.output_keys
            - If include_run_info=True: Additional RUN_KEY entry with RunInfo

        Raises:
            ValueError: If input is missing required keys or outputs missing expected keys
            Exception: Any exception raised by subclass _acall() implementation

        Async Execution Notes:
            - Event loop requirement: Must be called with await in async context
            - Async callbacks: All callback methods (on_chain_start, on_chain_end,
              on_chain_error) are awaited
            - Async memory: Uses memory.aload_memory_variables() and
              memory.asave_context() which must be awaited
            - _acall() execution: Subclasses should override _acall() for true async
              execution; default implementation runs sync _call() in executor thread

        Type Flow:
            input dict → aprep_inputs (async memory load) → validated inputs →
            _acall (async subclass logic) → outputs dict → aprep_outputs (async memory
            save) → final outputs

        Example:
            >>> async def run_chain():
            ...     chain = LLMChain(llm=async_llm, prompt=prompt)
            ...     result = await chain.ainvoke({"topic": "async langchain"})
            ...     return result

        Source: libs/langchain/langchain_classic/chains/base.py:187-238
        """
        config = ensure_config(config)
        callbacks = config.get("callbacks")
        tags = config.get("tags")
        metadata = config.get("metadata")
        run_name = config.get("run_name") or self.get_name()
        run_id = config.get("run_id")
        include_run_info = kwargs.get("include_run_info", False)
        return_only_outputs = kwargs.get("return_only_outputs", False)

        # Async memory loading: await memory.aload_memory_variables()
        inputs = await self.aprep_inputs(input)
        
        # Configure async callback manager for async callback invocation
        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        new_arg_supported = inspect.signature(self._acall).parameters.get("run_manager")
        
        # Await async on_chain_start callback
        run_manager = await callback_manager.on_chain_start(
            None,
            inputs,
            run_id,
            name=run_name,
        )
        try:
            # Validate all required self.input_keys present (sync validation)
            self._validate_inputs(inputs)
            
            # Await async chain execution (_acall is async variant)
            # Default _acall runs sync _call in executor; subclasses should override
            outputs = (
                await self._acall(inputs, run_manager=run_manager)
                if new_arg_supported
                else await self._acall(inputs)
            )
            
            # Await async output preparation and memory saving
            # aprep_outputs() calls await memory.asave_context()
            final_outputs: dict[str, Any] = await self.aprep_outputs(
                inputs,
                outputs,
                return_only_outputs,
            )
        except BaseException as e:
            # Await async on_chain_error callback
            await run_manager.on_chain_error(e)
            raise
        
        # Await async on_chain_end callback with outputs
        await run_manager.on_chain_end(outputs)

        if include_run_info:
            final_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        return final_outputs

    @property
    def _chain_type(self) -> str:
        msg = "Saving not supported for this chain type."
        raise NotImplementedError(msg)

    @model_validator(mode="before")
    @classmethod
    def raise_callback_manager_deprecation(cls, values: dict) -> Any:
        """Raise deprecation warning if callback_manager is used."""
        if values.get("callback_manager") is not None:
            if values.get("callbacks") is not None:
                msg = (
                    "Cannot specify both callback_manager and callbacks. "
                    "callback_manager is deprecated, callbacks is the preferred "
                    "parameter to pass in."
                )
                raise ValueError(msg)
            warnings.warn(
                "callback_manager is deprecated. Please use callbacks instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            values["callbacks"] = values.pop("callback_manager", None)
        return values

    @field_validator("verbose", mode="before")
    @classmethod
    def set_verbose(
        cls,
        verbose: bool | None,  # noqa: FBT001
    ) -> bool:
        """Set the chain verbosity.

        Defaults to the global setting if not specified by the user.
        """
        if verbose is None:
            return _get_verbosity()
        return verbose

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        """Keys expected to be in the chain input."""

    @property
    @abstractmethod
    def output_keys(self) -> list[str]:
        """Keys expected to be in the chain output."""

    def _validate_inputs(self, inputs: Any) -> None:
        """Check that all inputs are present."""
        if not isinstance(inputs, dict):
            _input_keys = set(self.input_keys)
            if self.memory is not None:
                # If there are multiple input keys, but some get set by memory so that
                # only one is not set, we can still figure out which key it is.
                _input_keys = _input_keys.difference(self.memory.memory_variables)
            if len(_input_keys) != 1:
                msg = (
                    f"A single string input was passed in, but this chain expects "
                    f"multiple inputs ({_input_keys}). When a chain expects "
                    f"multiple inputs, please call it by passing in a dictionary, "
                    "eg `chain({'foo': 1, 'bar': 2})`"
                )
                raise ValueError(msg)

        missing_keys = set(self.input_keys).difference(inputs)
        if missing_keys:
            msg = f"Missing some input keys: {missing_keys}"
            raise ValueError(msg)

    def _validate_outputs(self, outputs: dict[str, Any]) -> None:
        missing_keys = set(self.output_keys).difference(outputs)
        if missing_keys:
            msg = f"Missing some output keys: {missing_keys}"
            raise ValueError(msg)

    @abstractmethod
    def _call(
        self,
        inputs: builtins.dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> builtins.dict[str, Any]:
        """Execute the chain's core logic (abstract method for subclass implementation).

        This is a private method that is not user-facing. It is only called within
        Chain.invoke(), which is the user-facing wrapper method that handles callbacks
        configuration, memory loading/saving, and input/output validation.

        Args:
            inputs: Dict[str, Any] of named inputs to the chain. Contains all keys from
                self.input_keys PLUS any memory variables added by prep_inputs() via
                memory.load_memory_variables(). For example, if self.input_keys is
                ["question"] and memory adds {"chat_history": "..."}, inputs will
                contain both "question" and "chat_history" keys.
            run_manager: Optional CallbackManagerForChainRun for callback instrumentation.
                Subclasses should use run_manager to fire nested callbacks:
                - run_manager.on_llm_start(): Before LLM calls
                - run_manager.on_text(): For streaming text outputs
                - run_manager.on_tool_start(): Before tool execution
                This enables observability and tracing through nested operations.

        Returns:
            Dict[str, Any] containing ALL keys from self.output_keys. Values are
            typically strings but can be Any type. For example, if self.output_keys is
            ["answer"], return {"answer": "The answer is 42"}. The returned dict must
            include all output_keys or _validate_outputs() will raise ValueError.

        Raises:
            ValueError: For input validation failures specific to the chain
                implementation (e.g., malformed prompt variables, invalid parameters).
            RuntimeError: For execution failures during chain logic (e.g., failed
                retries, resource exhaustion).
            Provider-specific exceptions: API errors from LLM providers (e.g.,
                openai.error.RateLimitError, openai.error.AuthenticationError),
                vector store errors, tool execution failures.

        Implementation Contract:
            - Subclasses MUST override this method (abstract)
            - MUST return dict with all keys from self.output_keys
            - SHOULD call run_manager callbacks for observability (on_llm_start, etc.)
            - SHOULD NOT modify inputs dict (treat as read-only)
            - SHOULD handle retries/fallbacks internally or let exceptions propagate

        Note:
            This is a private method. Users should call invoke() instead, which handles
            the complete lifecycle including memory and callbacks.

        Example Implementation:
            >>> def _call(self, inputs, run_manager=None):
            ...     prompt = self.prompt.format(**inputs)
            ...     if run_manager:
            ...         run_manager.on_text(prompt)
            ...     response = self.llm(prompt)
            ...     return {"output": response}

        Source: libs/langchain/langchain_classic/chains/base.py:318-338
        """

    async def _acall(
        self,
        inputs: builtins.dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> builtins.dict[str, Any]:
        """Asynchronously execute the chain's core logic.

        This is a private method that is not user-facing. It is only called within
        Chain.ainvoke(), which is the user-facing wrapper method that handles async
        callbacks configuration, async memory operations, and input/output validation.

        Default Implementation:
            The default implementation runs the sync _call() method in an executor
            thread using run_in_executor(). This provides compatibility for chains that
            haven't implemented true async execution, but incurs thread overhead.

        Args:
            inputs: Dict[str, Any] of named inputs including memory variables, same as
                _call(). Contains all self.input_keys plus memory-added keys.
            run_manager: Optional AsyncCallbackManagerForChainRun for async callbacks.
                The default implementation converts to sync callback manager via
                run_manager.get_sync() for passing to the sync _call() method.

        Returns:
            Dict[str, Any] containing all keys from self.output_keys, same as _call().

        Raises:
            Same exceptions as _call(): ValueError, RuntimeError, provider-specific
            API errors, etc.

        Implementation Recommendation:
            Subclasses SHOULD override this method for true async execution to avoid
            executor thread overhead. Override when:
            - Using async LLM/API clients (e.g., aiohttp, httpx)
            - Performing async I/O operations (database, file system)
            - Calling other async chains or tools
            
            When overriding, use await for async operations and run_manager for async
            callbacks (await run_manager.on_llm_start(), etc.).

        Example Override:
            >>> async def _acall(self, inputs, run_manager=None):
            ...     prompt = self.prompt.format(**inputs)
            ...     if run_manager:
            ...         await run_manager.on_text(prompt)
            ...     response = await self.async_llm.agenerate(prompt)
            ...     return {"output": response}

        Source: libs/langchain/langchain_classic/chains/base.py:340-366
        """
        return await run_in_executor(
            None,
            self._call,
            inputs,
            run_manager.get_sync() if run_manager else None,
        )

    @deprecated("0.1.0", alternative="invoke", removal="1.0")
    def __call__(
        self,
        inputs: dict[str, Any] | Any,
        return_only_outputs: bool = False,  # noqa: FBT001,FBT002
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        include_run_info: bool = False,
    ) -> dict[str, Any]:
        """Execute the chain.

        Args:
            inputs: Dictionary of inputs, or single input if chain expects
                only one param. Should contain all inputs specified in
                `Chain.input_keys` except for inputs that will be set by the chain's
                memory.
            return_only_outputs: Whether to return only outputs in the
                response. If `True`, only new keys generated by this chain will be
                returned. If `False`, both input keys and new keys generated by this
                chain will be returned.
            callbacks: Callbacks to use for this chain run. These will be called in
                addition to callbacks passed to the chain during construction, but only
                these runtime callbacks will propagate to calls to other objects.
            tags: List of string tags to pass to all callbacks. These will be passed in
                addition to tags passed to the chain during construction, but only
                these runtime tags will propagate to calls to other objects.
            metadata: Optional metadata associated with the chain.
            run_name: Optional name for this run of the chain.
            include_run_info: Whether to include run info in the response. Defaults
                to False.

        Returns:
            A dict of named outputs. Should contain all outputs specified in
                `Chain.output_keys`.
        """
        config = {
            "callbacks": callbacks,
            "tags": tags,
            "metadata": metadata,
            "run_name": run_name,
        }

        return self.invoke(
            inputs,
            cast("RunnableConfig", {k: v for k, v in config.items() if v is not None}),
            return_only_outputs=return_only_outputs,
            include_run_info=include_run_info,
        )

    @deprecated("0.1.0", alternative="ainvoke", removal="1.0")
    async def acall(
        self,
        inputs: dict[str, Any] | Any,
        return_only_outputs: bool = False,  # noqa: FBT001,FBT002
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        include_run_info: bool = False,
    ) -> dict[str, Any]:
        """Asynchronously execute the chain.

        Args:
            inputs: Dictionary of inputs, or single input if chain expects
                only one param. Should contain all inputs specified in
                `Chain.input_keys` except for inputs that will be set by the chain's
                memory.
            return_only_outputs: Whether to return only outputs in the
                response. If `True`, only new keys generated by this chain will be
                returned. If `False`, both input keys and new keys generated by this
                chain will be returned.
            callbacks: Callbacks to use for this chain run. These will be called in
                addition to callbacks passed to the chain during construction, but only
                these runtime callbacks will propagate to calls to other objects.
            tags: List of string tags to pass to all callbacks. These will be passed in
                addition to tags passed to the chain during construction, but only
                these runtime tags will propagate to calls to other objects.
            metadata: Optional metadata associated with the chain.
            run_name: Optional name for this run of the chain.
            include_run_info: Whether to include run info in the response. Defaults
                to False.

        Returns:
            A dict of named outputs. Should contain all outputs specified in
                `Chain.output_keys`.
        """
        config = {
            "callbacks": callbacks,
            "tags": tags,
            "metadata": metadata,
            "run_name": run_name,
        }
        return await self.ainvoke(
            inputs,
            cast("RunnableConfig", {k: v for k, v in config.items() if k is not None}),
            return_only_outputs=return_only_outputs,
            include_run_info=include_run_info,
        )

    def prep_outputs(
        self,
        inputs: dict[str, str],
        outputs: dict[str, str],
        return_only_outputs: bool = False,  # noqa: FBT001,FBT002
    ) -> dict[str, str]:
        """Validate and prepare chain outputs, and save info about this run to memory.

        This method validates that outputs contain all required keys, persists the
        conversation to memory, and optionally merges inputs with outputs. Called at
        the end of invoke() after _call() completes successfully.

        Args:
            inputs: Dict[str, str] of chain inputs, including memory variables added by
                prep_inputs(). This is the complete inputs dict passed to _call().
            outputs: Dict[str, str] of initial chain outputs returned by _call(). Must
                contain all keys from self.output_keys or validation will fail.
            return_only_outputs: bool - Whether to return only the chain outputs.
                - If False (default): Returns merged {**inputs, **outputs} dict
                - If True: Returns only outputs dict (inputs excluded)

        Returns:
            Dict[str, str] of final chain outputs:
            - If return_only_outputs=False: {**inputs, **outputs} with all input and
              output keys
            - If return_only_outputs=True: outputs dict only
            
            Example (return_only_outputs=False): 
            {"question": "What is AI?", "chat_history": "...", "answer": "AI is..."}
            
            Example (return_only_outputs=True):
            {"answer": "AI is..."}

        Type Flow:
            outputs dict → _validate_outputs() (checks all output_keys present) →
            memory.save_context() persists conversation → final outputs (optionally
            merged with inputs)

        Memory Saving Lifecycle:
            If self.memory is configured:
            1. _validate_outputs() verifies outputs dict has all required output_keys
            2. memory.save_context(inputs, outputs) persists the conversation
            3. Memory stores both inputs and outputs for future retrieval via
               load_memory_variables()

        Source: libs/langchain/langchain_classic/chains/base.py:471-494
        """
        # Validate outputs dict contains all required self.output_keys
        self._validate_outputs(outputs)
        if self.memory is not None:
            # Persist conversation to memory for future context loading
            # memory.save_context() stores inputs and outputs
            self.memory.save_context(inputs, outputs)
        if return_only_outputs:
            return outputs
        # Merge inputs and outputs for complete result dict
        return {**inputs, **outputs}

    async def aprep_outputs(
        self,
        inputs: dict[str, str],
        outputs: dict[str, str],
        return_only_outputs: bool = False,  # noqa: FBT001,FBT002
    ) -> dict[str, str]:
        """Asynchronously validate and prepare chain outputs, and save to memory.

        This is the async variant of prep_outputs(). It performs the same validation,
        memory saving, and output merging, but uses await for async memory operations.

        Args:
            inputs: Dict[str, str] of chain inputs, including memory variables, same as
                prep_outputs().
            outputs: Dict[str, str] of initial chain outputs from _acall(), must contain
                all keys from self.output_keys.
            return_only_outputs: bool - Whether to return only outputs (True) or merged
                inputs+outputs (False, default).

        Returns:
            Dict[str, str] of final chain outputs:
            - If return_only_outputs=False: {**inputs, **outputs}
            - If return_only_outputs=True: outputs dict only
            Same structure as prep_outputs().

        Async Execution Notes:
            - Event loop requirement: Must be called with await in async context
            - Async memory saving: Uses memory.asave_context() which requires await,
              enabling non-blocking I/O for memory persistence (e.g., to async database)
            - Called by ainvoke(): This method is invoked at the end of ainvoke() after
              _acall() completes successfully

        Example:
            >>> async def run():
            ...     outputs = await chain._acall(inputs, run_manager)
            ...     final = await chain.aprep_outputs(inputs, outputs, False)
            ...     # Memory saved asynchronously

        Source: libs/langchain/langchain_classic/chains/base.py:771-794
        """
        # Validate outputs dict (sync validation)
        self._validate_outputs(outputs)
        if self.memory is not None:
            # Async memory saving: await memory.asave_context()
            # Enables non-blocking persistence of conversation
            await self.memory.asave_context(inputs, outputs)
        if return_only_outputs:
            return outputs
        return {**inputs, **outputs}

    def prep_inputs(self, inputs: dict[str, Any] | Any) -> dict[str, str]:
        """Prepare chain inputs, including adding inputs from memory.

        This method normalizes raw inputs to a dict format and merges in memory
        variables via memory.load_memory_variables(). Called at the beginning of
        invoke() to prepare complete inputs for _call().

        Args:
            inputs: Dict[str, Any] of raw inputs OR a single value. If dict, should
                contain all keys from self.input_keys except those provided by memory.
                If single value (str, int, etc.), it will be auto-wrapped into a dict
                using the first input key name (after excluding memory variables).
                
                Example: If self.input_keys=["question", "context"] and memory provides
                "context", inputs can be just "What is AI?" which becomes
                {"question": "What is AI?"}.

        Returns:
            Dict[str, str] with original inputs PLUS memory variables. The returned dict
            contains:
            - All keys from the original inputs parameter
            - Additional keys from memory.load_memory_variables() (e.g., "chat_history")
            - All keys required for _call() execution
            
            Example return: {"question": "What is AI?", "chat_history": "Human: ...\nAI: ..."}

        Type Flow:
            raw inputs (dict or single value) → dict normalization (if not dict) →
            memory.load_memory_variables(inputs) returns memory dict → merged dict
            with {**original_inputs, **memory_variables}

        Memory Loading Lifecycle:
            If self.memory is configured:
            1. memory.load_memory_variables(inputs) is called with current inputs dict
            2. Memory returns dict of contextual variables (e.g., {"chat_history": "..."})
            3. Memory dict is merged into inputs dict: dict(inputs, **external_context)
            4. Result contains both user inputs and loaded memory context

        Example:
            >>> chain.input_keys = ["question"]
            >>> chain.memory = ConversationBufferMemory()
            >>> # After previous conversation saved to memory
            >>> prepared = chain.prep_inputs({"question": "What is LangChain?"})
            >>> print(prepared)
            >>> # {"question": "What is LangChain?", "chat_history": "Human: ..."}

        Source: libs/langchain/langchain_classic/chains/base.py:521-543
        """
        if not isinstance(inputs, dict):
            _input_keys = set(self.input_keys)
            if self.memory is not None:
                # If there are multiple input keys, but some get set by memory so that
                # only one is not set, we can still figure out which key it is.
                _input_keys = _input_keys.difference(self.memory.memory_variables)
            inputs = {next(iter(_input_keys)): inputs}
        if self.memory is not None:
            # Load memory variables (e.g., chat_history from previous conversations)
            # memory.load_memory_variables() returns dict of contextual variables
            external_context = self.memory.load_memory_variables(inputs)
            # Merge memory variables into inputs dict
            inputs = dict(inputs, **external_context)
        return inputs

    async def aprep_inputs(self, inputs: dict[str, Any] | Any) -> dict[str, str]:
        """Asynchronously prepare chain inputs, including adding inputs from memory.

        This is the async variant of prep_inputs(). It performs the same input
        normalization and memory loading, but uses await for async memory operations.

        Args:
            inputs: Dict[str, Any] of raw inputs OR a single value, same as prep_inputs().
                If dict, should contain all keys from self.input_keys except those
                provided by memory. If single value, will be auto-wrapped into dict.

        Returns:
            Dict[str, str] with original inputs PLUS memory variables loaded via
            memory.aload_memory_variables(). Same structure as prep_inputs().

        Async Execution Notes:
            - Event loop requirement: Must be called with await in async context
            - Async memory loading: Uses memory.aload_memory_variables() which requires
              await, enabling non-blocking I/O for memory retrieval (e.g., from async
              database or API)
            - Called by ainvoke(): This method is invoked at the start of ainvoke()

        Example:
            >>> async def run():
            ...     chain.memory = AsyncConversationMemory()
            ...     prepared = await chain.aprep_inputs({"question": "Hello"})
            ...     # Returns: {"question": "Hello", "chat_history": "..."}

        Source: libs/langchain/langchain_classic/chains/base.py:545-567
        """
        if not isinstance(inputs, dict):
            _input_keys = set(self.input_keys)
            if self.memory is not None:
                # If there are multiple input keys, but some get set by memory so that
                # only one is not set, we can still figure out which key it is.
                _input_keys = _input_keys.difference(self.memory.memory_variables)
            inputs = {next(iter(_input_keys)): inputs}
        if self.memory is not None:
            # Async memory loading: await memory.aload_memory_variables()
            # Enables non-blocking retrieval of contextual variables
            external_context = await self.memory.aload_memory_variables(inputs)
            inputs = dict(inputs, **external_context)
        return inputs

    @property
    def _run_output_key(self) -> str:
        if len(self.output_keys) != 1:
            msg = (
                f"`run` not supported when there is not exactly "
                f"one output key. Got {self.output_keys}."
            )
            raise ValueError(msg)
        return self.output_keys[0]

    @deprecated("0.1.0", alternative="invoke", removal="1.0")
    def run(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Convenience method for executing chain.

        The main difference between this method and `Chain.__call__` is that this
        method expects inputs to be passed directly in as positional arguments or
        keyword arguments, whereas `Chain.__call__` expects a single input dictionary
        with all the inputs

        Args:
            *args: If the chain expects a single input, it can be passed in as the
                sole positional argument.
            callbacks: Callbacks to use for this chain run. These will be called in
                addition to callbacks passed to the chain during construction, but only
                these runtime callbacks will propagate to calls to other objects.
            tags: List of string tags to pass to all callbacks. These will be passed in
                addition to tags passed to the chain during construction, but only
                these runtime tags will propagate to calls to other objects.
            metadata: Optional metadata associated with the chain.
            **kwargs: If the chain expects multiple inputs, they can be passed in
                directly as keyword arguments.

        Returns:
            The chain output.

        Example:
            ```python
            # Suppose we have a single-input chain that takes a 'question' string:
            chain.run("What's the temperature in Boise, Idaho?")
            # -> "The temperature in Boise is..."

            # Suppose we have a multi-input chain that takes a 'question' string
            # and 'context' string:
            question = "What's the temperature in Boise, Idaho?"
            context = "Weather report for Boise, Idaho on 07/03/23..."
            chain.run(question=question, context=context)
            # -> "The temperature in Boise is..."
            ```
        """
        # Run at start to make sure this is possible/defined
        _output_key = self._run_output_key

        if args and not kwargs:
            if len(args) != 1:
                msg = "`run` supports only one positional argument."
                raise ValueError(msg)
            return self(args[0], callbacks=callbacks, tags=tags, metadata=metadata)[
                _output_key
            ]

        if kwargs and not args:
            return self(kwargs, callbacks=callbacks, tags=tags, metadata=metadata)[
                _output_key
            ]

        if not kwargs and not args:
            msg = (
                "`run` supported with either positional arguments or keyword arguments,"
                " but none were provided."
            )
            raise ValueError(msg)
        msg = (
            f"`run` supported with either positional arguments or keyword arguments"
            f" but not both. Got args: {args} and kwargs: {kwargs}."
        )
        raise ValueError(msg)

    @deprecated("0.1.0", alternative="ainvoke", removal="1.0")
    async def arun(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Convenience method for executing chain.

        The main difference between this method and `Chain.__call__` is that this
        method expects inputs to be passed directly in as positional arguments or
        keyword arguments, whereas `Chain.__call__` expects a single input dictionary
        with all the inputs


        Args:
            *args: If the chain expects a single input, it can be passed in as the
                sole positional argument.
            callbacks: Callbacks to use for this chain run. These will be called in
                addition to callbacks passed to the chain during construction, but only
                these runtime callbacks will propagate to calls to other objects.
            tags: List of string tags to pass to all callbacks. These will be passed in
                addition to tags passed to the chain during construction, but only
                these runtime tags will propagate to calls to other objects.
            metadata: Optional metadata associated with the chain.
            **kwargs: If the chain expects multiple inputs, they can be passed in
                directly as keyword arguments.

        Returns:
            The chain output.

        Example:
            ```python
            # Suppose we have a single-input chain that takes a 'question' string:
            await chain.arun("What's the temperature in Boise, Idaho?")
            # -> "The temperature in Boise is..."

            # Suppose we have a multi-input chain that takes a 'question' string
            # and 'context' string:
            question = "What's the temperature in Boise, Idaho?"
            context = "Weather report for Boise, Idaho on 07/03/23..."
            await chain.arun(question=question, context=context)
            # -> "The temperature in Boise is..."
            ```
        """
        if len(self.output_keys) != 1:
            msg = (
                f"`run` not supported when there is not exactly "
                f"one output key. Got {self.output_keys}."
            )
            raise ValueError(msg)
        if args and not kwargs:
            if len(args) != 1:
                msg = "`run` supports only one positional argument."
                raise ValueError(msg)
            return (
                await self.acall(
                    args[0],
                    callbacks=callbacks,
                    tags=tags,
                    metadata=metadata,
                )
            )[self.output_keys[0]]

        if kwargs and not args:
            return (
                await self.acall(
                    kwargs,
                    callbacks=callbacks,
                    tags=tags,
                    metadata=metadata,
                )
            )[self.output_keys[0]]

        msg = (
            f"`run` supported with either positional arguments or keyword arguments"
            f" but not both. Got args: {args} and kwargs: {kwargs}."
        )
        raise ValueError(msg)

    def dict(self, **kwargs: Any) -> dict:
        """Dictionary representation of chain.

        Expects `Chain._chain_type` property to be implemented and for memory to be
            null.

        Args:
            **kwargs: Keyword arguments passed to default `pydantic.BaseModel.dict`
                method.

        Returns:
            A dictionary representation of the chain.

        Example:
            ```python
            chain.model_dump(exclude_unset=True)
            # -> {"_type": "foo", "verbose": False, ...}
            ```
        """
        _dict = super().model_dump(**kwargs)
        with contextlib.suppress(NotImplementedError):
            _dict["_type"] = self._chain_type
        return _dict

    def save(self, file_path: Path | str) -> None:
        """Save the chain.

        Expects `Chain._chain_type` property to be implemented and for memory to be
            null.

        Args:
            file_path: Path to file to save the chain to.

        Example:
            ```python
            chain.save(file_path="path/chain.yaml")
            ```
        """
        if self.memory is not None:
            msg = "Saving of memory is not yet supported."
            raise ValueError(msg)

        # Fetch dictionary to save
        chain_dict = self.model_dump()
        if "_type" not in chain_dict:
            msg = f"Chain {self} does not support saving."
            raise NotImplementedError(msg)

        # Convert file to Path object.
        save_path = Path(file_path) if isinstance(file_path, str) else file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".json":
            with save_path.open("w") as f:
                json.dump(chain_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w") as f:
                yaml.dump(chain_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} must be json or yaml"
            raise ValueError(msg)

    @deprecated("0.1.0", alternative="batch", removal="1.0")
    def apply(
        self,
        input_list: list[builtins.dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> list[builtins.dict[str, str]]:
        """Call the chain on all inputs in the list."""
        return [self(inputs, callbacks=callbacks) for inputs in input_list]
