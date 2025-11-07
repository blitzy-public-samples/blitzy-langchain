"""Configuration utilities for Runnables.

This module provides a comprehensive configuration system for LangChain Runnable execution,
enabling fine-grained control over runtime behavior, callback integration, and distributed
tracing across chain compositions.

Configuration Propagation System
---------------------------------
The RunnableConfig system automatically propagates configuration settings through nested
Runnable execution using Python's contextvars. When a parent Runnable invokes child
Runnables, configuration values (callbacks, tags, metadata, recursion limits) are
inherited unless explicitly overridden.

Context Variable Management
---------------------------
Configuration is propagated through ContextVars (var_child_runnable_config) to ensure
thread-safe inheritance across:
- Sequential chain execution (RunnableSequence)
- Parallel batch operations (batch/abatch methods)
- Nested Runnable compositions (LCEL pipes)
- Async execution contexts (preserving configuration across await boundaries)

Callback Integration
--------------------
The config system integrates with CallbackManager to orchestrate event handling:
- Callbacks specified in config are inherited by all sub-Runnables
- Tags and metadata flow through to all callback handler methods
- Parent-child run relationships are maintained via run_id tracking
- AsyncCallbackManager handles async event dispatch in async contexts

Tracing Context Setup
---------------------
Configuration coordinates with LangSmith tracing infrastructure:
- run_id: Unique identifier for distributed tracing across services
- run_name: Human-readable names for monitoring and debugging
- tags/metadata: Propagated to LangSmith for filtering and analysis
- Tracing context is automatically set/reset via context managers

Key Components
--------------
- RunnableConfig: TypedDict defining all configuration options
- ensure_config(): Merges provided config with context defaults
- patch_config(): Updates specific config fields while preserving others
- merge_configs(): Combines multiple configs with intelligent merging
- set_config_context(): Context manager for scoped config propagation
- get_callback_manager_for_config(): Creates callback managers from config
- get_executor_for_config(): Provides thread pool executors respecting max_concurrency

Source: libs/core/langchain_core/runnables/config.py:1-608
"""

from __future__ import annotations

import asyncio
import uuid
import warnings
from collections.abc import Awaitable, Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import Context, ContextVar, Token, copy_context
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeVar,
    cast,
)

from langsmith.run_helpers import _set_tracing_context, get_tracing_context
from typing_extensions import TypedDict

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain_core.runnables.utils import (
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
)
from langchain_core.tracers.langchain import LangChainTracer

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackManager, Callbacks
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
else:
    # Pydantic validates through typed dicts, but
    # the callbacks need forward refs updated
    Callbacks = list | Any | None


class EmptyDict(TypedDict, total=False):
    """Empty dict type."""


class RunnableConfig(TypedDict, total=False):
    """Configuration for a Runnable.
    
    All fields are optional (total=False), allowing partial configuration updates.
    Configuration values are inherited by child Runnables during nested execution
    and merged with context-level defaults via ensure_config().
    
    Source: libs/core/langchain_core/runnables/config.py:49-98
    """

    tags: list[str]
    """
    Tags for this call and any sub-calls (eg. a Chain calling an LLM).
    
    **Usage Patterns**:
    - Filtering/organizing runs in LangSmith tracing interface
    - Categorizing chains by type (e.g., ["retrieval", "qa"])
    - Environment tagging (e.g., ["production", "experiment-v2"])
    
    **Inheritance Behavior**:
    - Tags are automatically inherited by all child Runnables
    - merge_configs() combines tags from multiple configs and removes duplicates
    - Tags flow through to ALL callback handler methods
    
    **Example**:
        ```python
        config = RunnableConfig(tags=["production", "user-query"])
        result = chain.invoke(input_data, config=config)
        # All nested LLM calls inherit ["production", "user-query"] tags
        ```
    
    Source: libs/core/langchain_core/runnables/config.py:52-56
    """

    metadata: dict[str, Any]
    """
    Metadata for this call and any sub-calls (eg. a Chain calling an LLM).
    
    **JSON-Serializable Constraint**:
    - Keys MUST be strings
    - Values MUST be JSON-serializable (str, int, float, bool, list, dict)
    - Non-serializable values (functions, objects) will cause tracing errors
    
    **Usage in Tracing and Debugging**:
    - Passed to handle*Start callback methods for run context
    - Appears in LangSmith trace metadata for filtering and analysis
    - Useful for tracking user_id, session_id, request_id, etc.
    
    **Inheritance Behavior**:
    - Metadata is inherited by child Runnables
    - merge_configs() performs shallow merge (later configs override earlier)
    - Automatically includes scalar values from 'configurable' field
    
    **Example**:
        ```python
        config = RunnableConfig(metadata={
            "user_id": "user-123",
            "session_id": "sess-456",
            "environment": "production"
        })
        # Metadata flows to all on_chain_start, on_llm_start callbacks
        ```
    
    Source: libs/core/langchain_core/runnables/config.py:58-62
    """

    callbacks: Callbacks
    """
    Callbacks for this call and any sub-calls (eg. a Chain calling an LLM).
    
    **CallbackManager Integration**:
    - Accepts None, list[BaseCallbackHandler], or CallbackManager instance
    - Automatically converted to CallbackManager via get_callback_manager_for_config()
    - Handlers receive events at all stages: on_chain_start → on_llm_start → 
      on_llm_new_token (streaming) → on_llm_end → on_chain_end
    
    **Handler Inheritance**:
    - All handlers inherit tags and metadata from config
    - Child Runnables inherit parent's callback manager
    - merge_configs() intelligently combines handler lists and managers
    
    **Callback Invocation in Chain Execution Lifecycle**:
    1. on_chain_start: Called when Runnable.invoke() begins
    2. on_llm_start: Called when LLM component starts generation
    3. on_llm_new_token: Called for each token (if streaming enabled)
    4. on_llm_end: Called when LLM completes generation
    5. on_chain_end: Called when Runnable.invoke() completes
    6. on_chain_error: Called if any error occurs during execution
    
    **Example**:
        ```python
        from langchain_core.callbacks import StdOutCallbackHandler
        
        config = RunnableConfig(callbacks=[StdOutCallbackHandler()])
        result = chain.invoke(input_data, config=config)
        # All nested calls trigger callback methods with inherited tags/metadata
        ```
    
    Source: libs/core/langchain_core/runnables/config.py:64-68
    """

    run_name: str
    """
    Name for the tracer run for this call. Defaults to the name of the class.
    
    **Tracer Naming Conventions**:
    - Used as human-readable identifier in LangSmith traces
    - Defaults to Runnable class name if not provided
    - Automatically cleared when callbacks are replaced (via patch_config)
    
    **Debugging and Monitoring Use Cases**:
    - Distinguish between multiple invocations of same Runnable type
    - Identify specific chain steps in complex compositions
    - Filter traces by meaningful operation names
    
    **Example**:
        ```python
        config = RunnableConfig(run_name="user_query_summarization")
        result = chain.invoke(input_data, config=config)
        # Appears as "user_query_summarization" in LangSmith trace UI
        ```
    
    Source: libs/core/langchain_core/runnables/config.py:70-73
    """

    max_concurrency: int | None
    """
    Maximum number of parallel calls to make. If not provided, defaults to
    `ThreadPoolExecutor`'s default.
    
    **ThreadPoolExecutor Integration**:
    - Controls max_workers parameter of ContextThreadPoolExecutor
    - Applied in batch() operations to limit concurrent Runnable executions
    - None value uses ThreadPoolExecutor default (typically min(32, os.cpu_count() + 4))
    
    **Parallel Execution Limits in Batch Operations**:
    - batch([input1, input2, input3], config={"max_concurrency": 2})
      executes at most 2 inputs simultaneously
    - Prevents resource exhaustion when processing large input batches
    - Applies to both sync batch() and async abatch() methods
    
    **Use Cases**:
    - Rate limiting to avoid API throttling
    - Memory management for large-scale processing
    - Resource contention prevention in containerized environments
    
    **Example**:
        ```python
        config = RunnableConfig(max_concurrency=5)
        results = chain.batch([input1, input2, ...input20], config=config)
        # Processes 5 inputs in parallel, queues remaining 15
        ```
    
    Source: libs/core/langchain_core/runnables/config.py:75-79
    """

    recursion_limit: int
    """
    Maximum number of times a call can recurse. If not provided, defaults to `25`.
    
    **Recursive Chain Execution Prevention**:
    - Counts depth of nested Runnable.invoke() calls
    - Raises RecursionError when limit exceeded
    - Prevents infinite loops in self-referential chains
    
    **Default Value of 25**:
    - Defined in DEFAULT_RECURSION_LIMIT constant
    - Sufficient for most production chain compositions
    - Can be increased for deeply nested architectures
    
    **Use Cases**:
    - Agent reasoning loops with tool calls
    - Recursive document processing chains
    - Self-improving chains with feedback loops
    
    **Example**:
        ```python
        config = RunnableConfig(recursion_limit=50)
        result = recursive_chain.invoke(input_data, config=config)
        # Allows up to 50 levels of nested invoke() calls
        ```
    
    Source: libs/core/langchain_core/runnables/config.py:81-84
    """

    configurable: dict[str, Any]
    """
    Runtime values for attributes previously made configurable on this `Runnable`,
    or sub-Runnables, through `configurable_fields` or `configurable_alternatives`.
    
    **Runtime Configuration System**:
    - Enables dynamic behavior changes without code modification
    - Accessed via Runnable.configurable_fields() and configurable_alternatives()
    - Keys correspond to field names registered as configurable
    
    **configurable_fields Usage Pattern**:
        ```python
        chain = (prompt | llm).configurable_fields(
            temperature=ConfigurableField(id="llm_temperature")
        )
        config = RunnableConfig(configurable={"llm_temperature": 0.9})
        result = chain.invoke(input_data, config=config)
        # LLM uses temperature=0.9 instead of default
        ```
    
    **configurable_alternatives Usage Pattern**:
        ```python
        chain = prompt.configurable_alternatives(
            ConfigurableField(id="llm_provider"),
            default_key="openai",
            anthropic=anthropic_llm,
            cohere=cohere_llm,
        )
        config = RunnableConfig(configurable={"llm_provider": "anthropic"})
        result = chain.invoke(input_data, config=config)
        # Uses anthropic_llm instead of default openai
        ```
    
    **Field Discovery**:
    - Check Runnable.config_schema() to see available configurable fields
    - Runnable.output_schema() includes configurable field descriptions
    
    **Metadata Integration**:
    - Scalar values (str, int, float, bool) automatically copied to metadata
    - Enables filtering traces by configuration values
    
    Source: libs/core/langchain_core/runnables/config.py:86-92
    """

    run_id: uuid.UUID | None
    """
    Unique identifier for the tracer run for this call. If not provided, a new UUID
    will be generated.
    
    **UUID Tracking for Distributed Tracing**:
    - Globally unique identifier for this specific Runnable execution
    - Auto-generated via uuid.uuid4() if not provided
    - Enables correlation across distributed services and async boundaries
    
    **Parent-Child Relationship in Nested Runnable Execution**:
    - Parent Runnable's run_id becomes parent_run_id for child Runnables
    - Creates hierarchical trace structure in LangSmith
    - CallbackManager tracks parent_run_id for event correlation
    
    **Run ID Propagation Rules**:
    - Automatically cleared when callbacks are replaced (patch_config)
    - Should be unique per invoke() call (Warning issued if reused in batch)
    - Retrieved from callbacks.parent_run_id in _set_config_context()
    
    **Example**:
        ```python
        import uuid
        
        run_id = uuid.uuid4()
        config = RunnableConfig(run_id=run_id)
        result = chain.invoke(input_data, config=config)
        # Trace appears in LangSmith with specified run_id
        # All nested calls have parent_run_id = run_id
        ```
    
    **Batch Operation Warning**:
    - If run_id provided to batch(), only first input uses it
    - RuntimeWarning issued: "Provided run_id will be used only for first element"
    - Subsequent inputs get auto-generated run_ids to maintain uniqueness
    
    Source: libs/core/langchain_core/runnables/config.py:94-98
    """


CONFIG_KEYS = [
    "tags",
    "metadata",
    "callbacks",
    "run_name",
    "max_concurrency",
    "recursion_limit",
    "configurable",
    "run_id",
]

COPIABLE_KEYS = [
    "tags",
    "metadata",
    "callbacks",
    "configurable",
]

DEFAULT_RECURSION_LIMIT = 25


var_child_runnable_config: ContextVar[RunnableConfig | None] = ContextVar(
    "child_runnable_config", default=None
)


# This is imported and used in langgraph, so don't break.
def _set_config_context(
    config: RunnableConfig,
) -> tuple[Token[RunnableConfig | None], dict[str, Any] | None]:
    """Set the child Runnable config + tracing context.
    
    **ContextVar Usage for Thread-Safe Config Propagation**:
    This function stores config in var_child_runnable_config ContextVar, ensuring
    thread-safe propagation through:
    - Nested Runnable invocations (child Runnables inherit parent config)
    - ThreadPoolExecutor parallel execution (each thread gets correct context)
    - Async execution contexts (config preserved across await boundaries)
    
    **LangSmith Tracing Integration Details**:
    Extracts parent run information from CallbackManager to establish trace hierarchy:
    1. Retrieves callbacks from config
    2. Checks if callbacks is a CallbackManager with parent_run_id
    3. Finds LangChainTracer handler in manager's handlers
    4. Looks up parent run in tracer.run_map
    5. Sets tracing context with parent run for LangSmith correlation
    
    **Usage Pattern in Nested Runnable Execution**:
    Called by set_config_context() context manager to establish scoped configuration.
    The returned token must be passed to var_child_runnable_config.reset() to restore
    previous config state when exiting the context.

    Args:
        config: The RunnableConfig to set as child context. Must include callbacks
            field if tracing context setup is required.

    Returns:
        A tuple containing:
        - Token to reset the ContextVar (pass to var_child_runnable_config.reset())
        - Previous tracing context dict (used to restore state after execution)
    
    Example:
        ```python
        config = RunnableConfig(callbacks=callback_manager, tags=["nested"])
        token, prev_context = _set_config_context(config)
        try:
            # Config now available via var_child_runnable_config.get()
            result = child_runnable.invoke(input_data)
        finally:
            var_child_runnable_config.reset(token)
            # Restore previous tracing context
        ```
    
    Note:
        This function is used by langgraph and must maintain API compatibility.
    
    Source: libs/core/langchain_core/runnables/config.py:128-160
    """
    config_token = var_child_runnable_config.set(config)
    current_context = None
    if (
        (callbacks := config.get("callbacks"))
        and (
            parent_run_id := getattr(callbacks, "parent_run_id", None)
        )  # Is callback manager
        and (
            tracer := next(
                (
                    handler
                    for handler in getattr(callbacks, "handlers", [])
                    if isinstance(handler, LangChainTracer)
                ),
                None,
            )
        )
        and (run := tracer.run_map.get(str(parent_run_id)))
    ):
        current_context = get_tracing_context()
        _set_tracing_context({"parent": run})
    return config_token, current_context


@contextmanager
def set_config_context(config: RunnableConfig) -> Generator[Context, None, None]:
    """Set the child Runnable config + tracing context.
    
    Context manager that establishes scoped configuration for nested Runnable execution.
    Automatically handles setup and teardown of both ContextVar and LangSmith tracing state.
    
    **ContextVar Usage for Thread-Safe Config Propagation**:
    - Creates isolated context copy via copy_context()
    - Sets config in var_child_runnable_config within context
    - Child Runnables inherit config via ensure_config() reading ContextVar
    - Automatically resets config on context exit
    
    **LangSmith Tracing Integration Details**:
    - Extracts parent run from callbacks via _set_config_context()
    - Sets tracing context with parent/project/tags/metadata
    - Resets all tracing fields on exit (parent, project_name, tags, metadata, etc.)
    - Ensures clean tracing state for subsequent operations
    
    **Usage Pattern in Nested Runnable Execution**:
    Use when manually managing Runnable invocations that require config inheritance:
        ```python
        parent_config = RunnableConfig(
            tags=["parent"],
            callbacks=callback_manager,
            metadata={"user_id": "123"}
        )
        
        with set_config_context(parent_config) as ctx:
            # Child runnable automatically inherits parent config
            result1 = child_runnable1.invoke(input1)
            result2 = child_runnable2.invoke(input2)
            # Both child invocations see tags=["parent"] and metadata
        
        # Config and tracing context fully reset after context exit
        ```

    Args:
        config: The RunnableConfig to establish as child context. All fields
            (tags, metadata, callbacks, etc.) become defaults for child Runnables.

    Yields:
        The Context object containing the configured state. The context is active
        within the with block and automatically cleaned up on exit.
    
    Note:
        Context is copied via copy_context(), preventing modifications from affecting
        the caller's context. All ContextVars (including var_child_runnable_config)
        are isolated within the yielded context.
    
    Source: libs/core/langchain_core/runnables/config.py:213-240
    """
    ctx = copy_context()
    config_token, _ = ctx.run(_set_config_context, config)
    try:
        yield ctx
    finally:
        ctx.run(var_child_runnable_config.reset, config_token)
        ctx.run(
            _set_tracing_context,
            {
                "parent": None,
                "project_name": None,
                "tags": None,
                "metadata": None,
                "enabled": None,
                "client": None,
            },
        )


def ensure_config(config: RunnableConfig | None = None) -> RunnableConfig:
    """Ensure that a config is a dict with all keys present.
    
    Merges provided config with context-level defaults and ensures all standard keys
    have values. This is the primary function for config normalization throughout
    the Runnable system.
    
    **Config Merge Order (highest to lowest priority)**:
    1. Provided config parameter (explicitly passed values)
    2. Context config from var_child_runnable_config.get() (inherited from parent)
    3. Default values (empty lists/dicts, DEFAULT_RECURSION_LIMIT=25)
    
    **Field Handling Rules**:
    - Standard CONFIG_KEYS: Direct override (later configs replace earlier)
    - COPIABLE_KEYS (tags, metadata, callbacks, configurable): Shallow copied
      to prevent unintended mutations
    - Non-standard keys: Automatically moved to configurable dict
    - Scalar configurable values: Copied to metadata for tracing
    
    **Configurable Field Processing**:
    Any key in config that's not in CONFIG_KEYS is moved to the configurable dict:
        ```python
        config = {"temperature": 0.9, "tags": ["test"]}
        result = ensure_config(config)
        # result["configurable"] = {"temperature": 0.9}
        # result["tags"] = ["test"]
        ```
    
    **Metadata Auto-Population**:
    Scalar values from configurable (str, int, float, bool) are copied to metadata
    unless:
    - Key starts with "__" (private)
    - Key is "api_key" (security)
    - Key already exists in metadata

    Args:
        config: The RunnableConfig to normalize. If None, returns config with defaults.
            Accepts partial config (TypedDict with total=False).

    Returns:
        Complete RunnableConfig with all standard keys present:
        - tags: list[str] (default: [])
        - metadata: dict[str, Any] (default: {})
        - callbacks: Callbacks (default: None)
        - recursion_limit: int (default: 25)
        - configurable: dict[str, Any] (default: {})
        - Plus any other fields from input config or context
    
    Example:
        ```python
        # Merge with context config
        var_child_runnable_config.set(RunnableConfig(tags=["parent"]))
        config = ensure_config(RunnableConfig(tags=["child"], metadata={"key": "value"}))
        # config["tags"] = ["child"]  # Provided config overrides context
        # config["metadata"] = {"key": "value"}
        # config["recursion_limit"] = 25  # Default applied
        ```
    
    Source: libs/core/langchain_core/runnables/config.py:242-292
    """
    empty = RunnableConfig(
        tags=[],
        metadata={},
        callbacks=None,
        recursion_limit=DEFAULT_RECURSION_LIMIT,
        configurable={},
    )
    if var_config := var_child_runnable_config.get():
        empty.update(
            cast(
                "RunnableConfig",
                {
                    k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                    for k, v in var_config.items()
                    if v is not None
                },
            )
        )
    if config is not None:
        empty.update(
            cast(
                "RunnableConfig",
                {
                    k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                    for k, v in config.items()
                    if v is not None and k in CONFIG_KEYS
                },
            )
        )
    if config is not None:
        for k, v in config.items():
            if k not in CONFIG_KEYS and v is not None:
                empty["configurable"][k] = v
    for key, value in empty.get("configurable", {}).items():
        if (
            not key.startswith("__")
            and isinstance(value, (str, int, float, bool))
            and key not in empty["metadata"]
            and key != "api_key"
        ):
            empty["metadata"][key] = value
    return empty


def get_config_list(
    config: RunnableConfig | Sequence[RunnableConfig] | None, length: int
) -> list[RunnableConfig]:
    """Get a list of configs from a single config or a list of configs.

    Converts config input into a list of normalized configs matching the batch size.
    It is useful for subclasses overriding batch() or abatch().
    
    **Config Expansion Patterns**:
    1. Single config → Duplicated to match length, with run_id handling
    2. List of configs → Validated length matches, each config normalized
    3. None → List of default configs (empty values) of specified length
    
    **run_id Special Handling for Batch Operations**:
    If a single config with run_id is provided for length > 1:
    - First element uses the provided run_id
    - Subsequent elements get config with run_id removed (auto-generated)
    - RuntimeWarning issued to alert developer
    
    This prevents run_id reuse across parallel executions, which would break
    distributed tracing parent-child relationships.

    Args:
        config: The config input, which can be:
            - None: Use defaults for all elements
            - Single RunnableConfig: Duplicate for all elements
            - Sequence of RunnableConfig: One config per element
        length: The number of configs needed (must equal batch size).
            Must be >= 0.

    Returns:
        List of normalized RunnableConfig objects, one per batch element.
        Each config is processed through ensure_config().

    Raises:
        ValueError: If length < 0 or if config is a sequence with length != length parameter.
    
    Example:
        ```python
        # Single config expansion
        config = RunnableConfig(tags=["batch"])
        configs = get_config_list(config, length=3)
        # Returns 3 identical configs with tags=["batch"]
        
        # Per-element configs
        configs = [
            RunnableConfig(tags=["item1"]),
            RunnableConfig(tags=["item2"]),
        ]
        result = get_config_list(configs, length=2)
        # Returns normalized versions of provided configs
        
        # run_id warning scenario
        config = RunnableConfig(run_id=uuid.uuid4())
        configs = get_config_list(config, length=3)
        # Warning: "Provided run_id will be used only for first element"
        # configs[0] has run_id, configs[1] and configs[2] don't
        ```

    Source: libs/core/langchain_core/runnables/config.py:295-338
    """
    if length < 0:
        msg = f"length must be >= 0, but got {length}"
        raise ValueError(msg)
    if isinstance(config, Sequence) and len(config) != length:
        msg = (
            f"config must be a list of the same length as inputs, "
            f"but got {len(config)} configs for {length} inputs"
        )
        raise ValueError(msg)

    if isinstance(config, Sequence):
        return list(map(ensure_config, config))
    if length > 1 and isinstance(config, dict) and config.get("run_id") is not None:
        warnings.warn(
            "Provided run_id be used only for the first element of the batch.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        subsequent = cast(
            "RunnableConfig", {k: v for k, v in config.items() if k != "run_id"}
        )
        return [
            ensure_config(subsequent) if i else ensure_config(config)
            for i in range(length)
        ]
    return [ensure_config(config) for i in range(length)]


def patch_config(
    config: RunnableConfig | None,
    *,
    callbacks: BaseCallbackManager | None = None,
    recursion_limit: int | None = None,
    max_concurrency: int | None = None,
    run_name: str | None = None,
    configurable: dict[str, Any] | None = None,
) -> RunnableConfig:
    """Patch a config with new values.
    
    Updates specific fields of a RunnableConfig while preserving all other fields.
    More surgical than merge_configs() - only modifies explicitly provided parameters.
    
    **Callback Replacement Behavior**:
    When callbacks parameter is provided (not None), the function automatically clears
    run_name and run_id fields. This is critical because:
    - run_name and run_id are tied to the original callback manager's trace context
    - Replacing callbacks without clearing these would create inconsistent trace hierarchy
    - New callbacks will generate fresh run_id and use default/new run_name
    
    **Configurable Merge Strategy**:
    The configurable parameter is MERGED with existing configurable dict, not replaced:
        ```python
        original = RunnableConfig(configurable={"temperature": 0.7, "model": "gpt-4"})
        patched = patch_config(original, configurable={"temperature": 0.9})
        # patched["configurable"] = {"temperature": 0.9, "model": "gpt-4"}
        ```
    
    **Typical Usage Patterns**:
    1. Adding child callbacks: patch_config(config, callbacks=run_manager.get_child())
    2. Increasing recursion: patch_config(config, recursion_limit=config["recursion_limit"] + 10)
    3. Updating runtime config: patch_config(config, configurable={"temperature": 0.5})

    Args:
        config: The RunnableConfig to patch. If None, starts with default config
            from ensure_config().
        callbacks: New CallbackManager to replace existing callbacks. When set, also
            clears run_name and run_id to maintain tracing consistency.
        recursion_limit: New recursion limit value. None means keep existing value.
        max_concurrency: New max concurrency value. None means keep existing value.
        run_name: New run name value. None means keep existing value.
        configurable: Dictionary to merge with existing configurable field.
            Uses shallow merge (new keys added, existing keys overridden).

    Returns:
        New RunnableConfig with specified fields updated. All unspecified fields
        preserve their original values from the input config.
    
    Example:
        ```python
        original = RunnableConfig(
            tags=["original"],
            callbacks=parent_manager,
            run_name="parent_run",
            recursion_limit=25
        )
        
        # Create child config with new callbacks
        child_config = patch_config(
            original,
            callbacks=parent_manager.get_child(),
            recursion_limit=24
        )
        # Result:
        # - tags=["original"] (preserved)
        # - callbacks=child_manager (replaced)
        # - run_name deleted (auto-cleared due to callback replacement)
        # - run_id deleted (auto-cleared due to callback replacement)
        # - recursion_limit=24 (updated)
        ```

    Source: libs/core/langchain_core/runnables/config.py:341-381
    """
    config = ensure_config(config)
    if callbacks is not None:
        # If we're replacing callbacks, we need to unset run_name
        # As that should apply only to the same run as the original callbacks
        config["callbacks"] = callbacks
        if "run_name" in config:
            del config["run_name"]
        if "run_id" in config:
            del config["run_id"]
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if run_name is not None:
        config["run_name"] = run_name
    if configurable is not None:
        config["configurable"] = {**config.get("configurable", {}), **configurable}
    return config


def merge_configs(*configs: RunnableConfig | None) -> RunnableConfig:
    """Merge multiple configs into one.
    
    Intelligently combines multiple RunnableConfig objects using field-specific merge
    strategies. Later configs in the argument list take precedence over earlier ones.
    
    **Field-Specific Merge Strategies**:
    
    - **metadata**: Shallow merge (dict.update behavior)
      ```python
      config1 = {"metadata": {"key1": "value1"}}
      config2 = {"metadata": {"key2": "value2"}}
      # Result: {"metadata": {"key1": "value1", "key2": "value2"}}
      ```
    
    - **tags**: Union with deduplication and sorting
      ```python
      config1 = {"tags": ["tag1", "tag2"]}
      config2 = {"tags": ["tag2", "tag3"]}
      # Result: {"tags": ["tag1", "tag2", "tag3"]} (sorted, unique)
      ```
    
    - **configurable**: Shallow merge (dict.update behavior)
      ```python
      config1 = {"configurable": {"temperature": 0.7}}
      config2 = {"configurable": {"max_tokens": 100}}
      # Result: {"configurable": {"temperature": 0.7, "max_tokens": 100}}
      ```
    
    - **callbacks**: Intelligent handler combination
      - None + list: Use list
      - list + list: Concatenate lists
      - list + manager: Add list handlers to manager copy
      - manager + list: Add list handlers to manager copy
      - manager + manager: manager1.merge(manager2)
    
    - **recursion_limit**: Use non-default value (not 25) if present
    
    - **Other COPIABLE_KEYS**: Shallow copy from latest non-None config
    
    - **Remaining fields**: Latest non-None value wins
    
    **Callback Merge Examples**:
    ```python
    # List + List
    config1 = {"callbacks": [handler1, handler2]}
    config2 = {"callbacks": [handler3]}
    # Result: {"callbacks": [handler1, handler2, handler3]}
    
    # Manager + List
    config1 = {"callbacks": callback_manager}
    config2 = {"callbacks": [new_handler]}
    # Result: {"callbacks": manager.copy() with new_handler added}
    ```

    Args:
        *configs: Variable number of RunnableConfig objects to merge. None values
            are skipped. Later configs override earlier configs following the
            field-specific strategies described above.

    Returns:
        New RunnableConfig containing the merged result. All COPIABLE_KEYS
        (tags, metadata, callbacks, configurable) are shallow copied to prevent
        unintended mutations.
    
    Example:
        ```python
        base_config = RunnableConfig(
            tags=["base"],
            metadata={"env": "prod"},
            recursion_limit=25
        )
        
        override_config = RunnableConfig(
            tags=["override"],
            metadata={"user": "123"},
            max_concurrency=5
        )
        
        result = merge_configs(base_config, override_config)
        # result = {
        #     "tags": ["base", "override"],  # Combined and sorted
        #     "metadata": {"env": "prod", "user": "123"},  # Merged
        #     "recursion_limit": 25,  # Kept (default value)
        #     "max_concurrency": 5,  # From override_config
        # }
        ```

    Source: libs/core/langchain_core/runnables/config.py:383-447
    """
    base: RunnableConfig = {}
    # Even though the keys aren't literals, this is correct
    # because both dicts are the same type
    for config in (ensure_config(c) for c in configs if c is not None):
        for key in config:
            if key == "metadata":
                base["metadata"] = {
                    **base.get("metadata", {}),
                    **(config.get("metadata") or {}),
                }
            elif key == "tags":
                base["tags"] = sorted(
                    set(base.get("tags", []) + (config.get("tags") or [])),
                )
            elif key == "configurable":
                base["configurable"] = {
                    **base.get("configurable", {}),
                    **(config.get("configurable") or {}),
                }
            elif key == "callbacks":
                base_callbacks = base.get("callbacks")
                these_callbacks = config["callbacks"]
                # callbacks can be either None, list[handler] or manager
                # so merging two callbacks values has 6 cases
                if isinstance(these_callbacks, list):
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks.copy()
                    elif isinstance(base_callbacks, list):
                        base["callbacks"] = base_callbacks + these_callbacks
                    else:
                        # base_callbacks is a manager
                        mngr = base_callbacks.copy()
                        for callback in these_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                elif these_callbacks is not None:
                    # these_callbacks is a manager
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks.copy()
                    elif isinstance(base_callbacks, list):
                        mngr = these_callbacks.copy()
                        for callback in base_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                    else:
                        # base_callbacks is also a manager
                        base["callbacks"] = base_callbacks.merge(these_callbacks)
            elif key == "recursion_limit":
                if config["recursion_limit"] != DEFAULT_RECURSION_LIMIT:
                    base["recursion_limit"] = config["recursion_limit"]
            elif key in COPIABLE_KEYS and config[key] is not None:  # type: ignore[literal-required]
                base[key] = config[key].copy()  # type: ignore[literal-required]
            else:
                base[key] = config[key] or base.get(key)  # type: ignore[literal-required]
    return base


def call_func_with_variable_args(
    func: Callable[[Input], Output]
    | Callable[[Input, RunnableConfig], Output]
    | Callable[[Input, CallbackManagerForChainRun], Output]
    | Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
    input: Input,
    config: RunnableConfig,
    run_manager: CallbackManagerForChainRun | None = None,
    **kwargs: Any,
) -> Output:
    """Call function that may optionally accept a run_manager and/or config.
    
    Introspects function signature to determine which optional parameters it accepts
    and calls it with the appropriate combination of arguments. This enables flexible
    Runnable implementations that can opt-in to config and/or run_manager parameters.
    
    **Parameter Detection**:
    - Uses accepts_config() to check if func signature includes "config" parameter
    - Uses accepts_run_manager() to check if func signature includes "run_manager" parameter
    - Only passes parameters that the function actually accepts
    
    **Config Patching with Child Callbacks**:
    If both run_manager and config parameter are accepted:
    - Config is patched to use run_manager.get_child() as callbacks
    - This ensures nested operations inherit proper callback hierarchy
    - Maintains parent-child run relationships in tracing

    Args:
        func: The function to call. Can accept any combination of:
            - Just input: (input: Input) -> Output
            - Input + config: (input: Input, config: RunnableConfig) -> Output
            - Input + run_manager: (input: Input, run_manager: CallbackManagerForChainRun) -> Output
            - All three: (input: Input, run_manager: CallbackManagerForChainRun, config: RunnableConfig) -> Output
        input: The input value to pass to the function.
        config: The RunnableConfig. Passed to function if it accepts config parameter.
            Automatically patched with child callbacks if run_manager is also passed.
        run_manager: The CallbackManagerForChainRun. Passed to function if it accepts
            run_manager parameter. If provided, config is patched with its child callbacks.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        The output of the function call.
    
    Example:
        ```python
        # Function accepting only input
        def simple_func(input: str) -> str:
            return input.upper()
        
        result = call_func_with_variable_args(simple_func, "hello", config)
        # Calls: simple_func("hello")
        
        # Function accepting input and config
        def config_func(input: str, config: RunnableConfig) -> str:
            tags = config.get("tags", [])
            return f"{input}: {tags}"
        
        result = call_func_with_variable_args(config_func, "hello", config)
        # Calls: config_func("hello", config=config)
        
        # Function accepting all parameters
        def full_func(input: str, run_manager: CallbackManagerForChainRun, 
                     config: RunnableConfig) -> str:
            run_manager.on_text("Processing")
            return input
        
        result = call_func_with_variable_args(full_func, "hello", config, run_manager)
        # Calls: full_func("hello", run_manager=run_manager, 
        #                  config=patch_config(config, callbacks=run_manager.get_child()))
        ```

    Source: libs/core/langchain_core/runnables/config.py:450-479
    """
    if accepts_config(func):
        if run_manager is not None:
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
        else:
            kwargs["config"] = config
    if run_manager is not None and accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return func(input, **kwargs)  # type: ignore[call-arg]


def acall_func_with_variable_args(
    func: Callable[[Input], Awaitable[Output]]
    | Callable[[Input, RunnableConfig], Awaitable[Output]]
    | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
    | Callable[
        [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
    ],
    input: Input,
    config: RunnableConfig,
    run_manager: AsyncCallbackManagerForChainRun | None = None,
    **kwargs: Any,
) -> Awaitable[Output]:
    """Async call function that may optionally accept a run_manager and/or config.
    
    Async variant of call_func_with_variable_args(). Introspects async function signature
    to determine which optional parameters it accepts and calls it with the appropriate
    combination of arguments.
    
    **Parameter Detection**:
    - Uses accepts_config() to check if func signature includes "config" parameter
    - Uses accepts_run_manager() to check if func signature includes "run_manager" parameter
    - Only passes parameters that the async function actually accepts
    
    **Config Patching with Child Callbacks**:
    If both run_manager and config parameter are accepted:
    - Config is patched to use run_manager.get_child() as callbacks
    - Ensures nested async operations inherit proper callback hierarchy
    - Maintains parent-child run relationships in async tracing
    
    **Async Callback Handling**:
    - run_manager parameter is AsyncCallbackManagerForChainRun (not sync version)
    - Callback events are dispatched asynchronously
    - Config callbacks field should contain AsyncCallbackHandler instances for async chains

    Args:
        func: The async function to call. Can accept any combination of:
            - Just input: async (input: Input) -> Output
            - Input + config: async (input: Input, config: RunnableConfig) -> Output
            - Input + run_manager: async (input: Input, run_manager: AsyncCallbackManagerForChainRun) -> Output
            - All three: async (input: Input, run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig) -> Output
        input: The input value to pass to the async function.
        config: The RunnableConfig. Passed to function if it accepts config parameter.
            Automatically patched with child callbacks if run_manager is also passed.
        run_manager: The AsyncCallbackManagerForChainRun. Passed to function if it accepts
            run_manager parameter. If provided, config is patched with its child callbacks.
        **kwargs: Additional keyword arguments to pass to the async function.

    Returns:
        Awaitable that resolves to the output of the async function call.
    
    Example:
        ```python
        # Async function accepting input and config
        async def async_config_func(input: str, config: RunnableConfig) -> str:
            tags = config.get("tags", [])
            await asyncio.sleep(0.1)  # Simulate async work
            return f"{input}: {tags}"
        
        awaitable = acall_func_with_variable_args(async_config_func, "hello", config)
        result = await awaitable
        # Calls: await async_config_func("hello", config=config)
        
        # Async function with run_manager for callback events
        async def async_full_func(input: str, 
                                  run_manager: AsyncCallbackManagerForChainRun,
                                  config: RunnableConfig) -> str:
            await run_manager.on_text("Processing")
            return input
        
        awaitable = acall_func_with_variable_args(async_full_func, "hello", config, run_manager)
        result = await awaitable
        # Calls: await async_full_func("hello", run_manager=run_manager,
        #                              config=patch_config(config, callbacks=run_manager.get_child()))
        ```

    Source: libs/core/langchain_core/runnables/config.py:532-563
    """
    if accepts_config(func):
        if run_manager is not None:
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
        else:
            kwargs["config"] = config
    if run_manager is not None and accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return func(input, **kwargs)  # type: ignore[call-arg]


def get_callback_manager_for_config(config: RunnableConfig) -> CallbackManager:
    """Get a callback manager for a config.
    
    Creates a CallbackManager instance configured with callbacks, tags, and metadata
    from the provided RunnableConfig. Used at the start of Runnable.invoke() to establish
    callback context for the execution.
    
    **CallbackManager.configure() Behavior**:
    - inheritable_callbacks: Can be None, list[BaseCallbackHandler], or existing CallbackManager
    - inheritable_tags: Passed to all callback handler methods
    - inheritable_metadata: Passed to handle*Start callback methods (on_chain_start, on_llm_start, etc.)
    
    **Inheritance Pattern**:
    The returned CallbackManager will propagate callbacks, tags, and metadata to child
    Runnables via the config system. Child Runnables call run_manager.get_child() to
    obtain their own callback manager with inherited context.

    Args:
        config: RunnableConfig containing callbacks, tags, and metadata fields.
            All fields are optional and retrieved via .get() with None defaults.

    Returns:
        Configured CallbackManager ready for use in synchronous Runnable execution.
        Handlers will receive tags and metadata in all callback method invocations.
    
    Example:
        ```python
        config = RunnableConfig(
            callbacks=[StdOutCallbackHandler()],
            tags=["production", "user-query"],
            metadata={"user_id": "123", "session": "abc"}
        )
        
        manager = get_callback_manager_for_config(config)
        # manager.handlers includes StdOutCallbackHandler
        # manager.tags = ["production", "user-query"]
        # manager.metadata = {"user_id": "123", "session": "abc"}
        
        # In Runnable.invoke():
        with manager.on_chain_start("MyChain", inputs) as run_manager:
            # on_chain_start receives metadata
            # run_manager.get_child() creates child with inherited context
            child_manager = run_manager.get_child()
        ```

    Source: libs/core/langchain_core/runnables/config.py:566-579
    """


def get_async_callback_manager_for_config(
    config: RunnableConfig,
) -> AsyncCallbackManager:
    """Get an async callback manager for a config.
    
    Creates an AsyncCallbackManager instance configured with callbacks, tags, and metadata
    from the provided RunnableConfig. Used at the start of Runnable.ainvoke() to establish
    async callback context for the execution.
    
    **AsyncCallbackManager.configure() Behavior**:
    - inheritable_callbacks: Can be None, list[AsyncCallbackHandler], or existing AsyncCallbackManager
    - inheritable_tags: Passed to all async callback handler methods
    - inheritable_metadata: Passed to async handle*Start callback methods
    
    **Async Event Loop Considerations**:
    - Callback handlers should be AsyncCallbackHandler instances for async chains
    - Events are dispatched asynchronously (await handler.on_chain_start(...))
    - Handlers run in the same event loop as the Runnable execution
    
    **Inheritance Pattern**:
    The returned AsyncCallbackManager will propagate callbacks, tags, and metadata to child
    Runnables via the config system. Child Runnables call run_manager.get_child() to
    obtain their own async callback manager with inherited context.

    Args:
        config: RunnableConfig containing callbacks, tags, and metadata fields.
            All fields are optional and retrieved via .get() with None defaults.

    Returns:
        Configured AsyncCallbackManager ready for use in asynchronous Runnable execution.
        Handlers will receive tags and metadata in all async callback method invocations.
    
    Example:
        ```python
        from langchain_core.callbacks import AsyncCallbackHandler
        
        class MyAsyncHandler(AsyncCallbackHandler):
            async def on_chain_start(self, serialized, inputs, **kwargs):
                print(f"Tags: {kwargs.get('tags')}")
                print(f"Metadata: {kwargs.get('metadata')}")
        
        config = RunnableConfig(
            callbacks=[MyAsyncHandler()],
            tags=["async-chain"],
            metadata={"request_id": "req-123"}
        )
        
        manager = get_async_callback_manager_for_config(config)
        # manager.handlers includes MyAsyncHandler
        
        # In Runnable.ainvoke():
        async with manager.on_chain_start("MyChain", inputs) as run_manager:
            # on_chain_start is awaited with metadata
            child_manager = run_manager.get_child()
            # child_manager inherits async context
        ```

    Source: libs/core/langchain_core/runnables/config.py:622-636
    """


P = ParamSpec("P")
T = TypeVar("T")


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that copies the context to the child thread.
    
    Extends standard ThreadPoolExecutor to preserve Python contextvars (including
    var_child_runnable_config) when executing functions in worker threads. This is
    critical for maintaining RunnableConfig inheritance across parallel batch operations.
    
    **Context Preservation Mechanism**:
    - Captures current context via copy_context() at submission time
    - Wraps function to execute within the captured context in worker thread
    - Ensures child threads see same ContextVar values as parent thread
    
    **Why This Matters for Runnables**:
    Without context copying, worker threads would not see var_child_runnable_config,
    causing ensure_config() to miss inherited configuration. This would break:
    - Tag inheritance in parallel batch() operations
    - Callback propagation to parallel Runnable invocations
    - Metadata flowing through concurrent executions
    
    Source: libs/core/langchain_core/runnables/config.py:654-655
    """

    def submit(  # type: ignore[override]
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[T]:
        """Submit a function to the executor.
        
        Wraps the function to execute within a copy of the current context, ensuring
        ContextVars (including RunnableConfig) are accessible in worker threads.

        Args:
            func: The function to submit to worker thread. Will be executed within
                a copy of the current context.
            *args: Positional arguments to pass to func.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Future[T] that resolves to the function's return value. The function
            executes in a worker thread with the captured context.
        
        Example:
            ```python
            var_child_runnable_config.set(RunnableConfig(tags=["parent"]))
            
            executor = ContextThreadPoolExecutor(max_workers=2)
            
            def worker_func():
                config = var_child_runnable_config.get()
                return config["tags"]  # Returns ["parent"]
            
            future = executor.submit(worker_func)
            result = future.result()  # ["parent"] - context preserved!
            ```

        Source: libs/core/langchain_core/runnables/config.py:657-675
        """
        return super().submit(
            cast("Callable[..., T]", partial(copy_context().run, func, *args, **kwargs))
        )

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        **kwargs: Any,
    ) -> Iterator[T]:
        """Map a function to multiple iterables.
        
        Extends ThreadPoolExecutor.map() to preserve context for each iteration. Creates
        a separate context copy for each element in the iterables to ensure isolated
        ContextVar state per worker invocation.

        Args:
            fn: The function to map over iterables. Will be executed with context
                preservation in worker threads.
            *iterables: Iterables to map over. Each element at index i across all
                iterables is passed to fn as separate arguments.
            **kwargs: Keyword arguments passed to parent ThreadPoolExecutor.map():
                - timeout: Maximum seconds to wait for results
                - chunksize: Number of elements per worker submission

        Returns:
            Iterator yielding results of fn applied to each element set. Results
            are returned in the same order as the input iterables.
        
        Example:
            ```python
            executor = ContextThreadPoolExecutor(max_workers=3)
            
            def process(value: int) -> int:
                config = var_child_runnable_config.get()
                tags = config.get("tags", [])
                return value * len(tags)
            
            var_child_runnable_config.set(RunnableConfig(tags=["a", "b"]))
            results = list(executor.map(process, [1, 2, 3]))
            # results = [2, 4, 6] - each worker sees tags=["a", "b"]
            ```

        Source: libs/core/langchain_core/runnables/config.py:677-703
        """
        contexts = [copy_context() for _ in range(len(iterables[0]))]  # type: ignore[arg-type]

        def _wrapped_fn(*args: Any) -> T:
            return contexts.pop().run(fn, *args)

        return super().map(
            _wrapped_fn,
            *iterables,
            **kwargs,
        )


@contextmanager
def get_executor_for_config(
    config: RunnableConfig | None,
) -> Generator[Executor, None, None]:
    """Get an executor for a config.
    
    Context manager that provides a ContextThreadPoolExecutor configured with the
    max_concurrency setting from RunnableConfig. Automatically manages executor lifecycle
    (creation and cleanup).
    
    **max_workers Configuration**:
    - Reads max_concurrency from config (None uses ThreadPoolExecutor default)
    - Default ThreadPoolExecutor behavior: min(32, os.cpu_count() + 4)
    - Limits concurrent thread execution in batch() operations
    
    **Context Preservation**:
    Uses ContextThreadPoolExecutor (not standard ThreadPoolExecutor) to ensure
    ContextVars including var_child_runnable_config are accessible in worker threads.
    
    **Typical Usage in Batch Operations**:
    Called by Runnable.batch() to obtain executor for parallel input processing:
        ```python
        with get_executor_for_config(config) as executor:
            futures = [executor.submit(self.invoke, input, cfg) for input, cfg in zip(inputs, configs)]
            results = [future.result() for future in futures]
        ```

    Args:
        config: RunnableConfig containing optional max_concurrency field.
            If None or max_concurrency not set, uses ThreadPoolExecutor default workers.

    Yields:
        ContextThreadPoolExecutor instance ready for submitting work. Automatically
        shut down and cleaned up when exiting the context.
    
    Example:
        ```python
        config = RunnableConfig(max_concurrency=5, tags=["batch"])
        
        with get_executor_for_config(config) as executor:
            # Executor has max_workers=5
            futures = []
            for item in items:
                future = executor.submit(process_item, item)
                futures.append(future)
            
            results = [f.result() for f in futures]
        # Executor automatically shut down here
        ```

    Source: libs/core/langchain_core/runnables/config.py:708-723
    """
    config = config or {}
    with ContextThreadPoolExecutor(
        max_workers=config.get("max_concurrency")
    ) as executor:
        yield executor


async def run_in_executor(
    executor_or_config: Executor | RunnableConfig | None,
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run a function in an executor.
    
    Executes a synchronous function in a thread pool executor from async context.
    Handles StopIteration exception conversion and context preservation.
    
    **Executor Selection Logic**:
    - If executor_or_config is None or dict: Uses default event loop executor with context copy
    - If executor_or_config is Executor: Uses provided executor (context NOT automatically copied)
    
    **StopIteration Exception Handling**:
    Catches StopIteration and converts to RuntimeError because StopIteration cannot be set
    on asyncio.Future - it raises TypeError and leaves the Future pending forever. This
    prevents deadlocks when generators raise StopIteration in executor threads.
    
    **Context Preservation in Default Executor**:
    When using default executor (executor_or_config is None or dict), wraps function with
    copy_context().run() to ensure ContextVars are accessible in the executor thread.
    This maintains RunnableConfig inheritance across async→sync boundaries.
    
    **Use Cases**:
    - Running sync-only operations (file I/O, CPU-bound work) from async Runnables
    - Executing blocking LLM API calls from async chains
    - Thread pool execution of batch processing from async context

    Args:
        executor_or_config: Determines execution strategy:
            - None: Use default event loop executor with context copy
            - RunnableConfig (dict): Use default executor with context copy
            - Executor: Use provided executor (caller responsible for context)
        func: Synchronous function to execute in thread pool. Should not be a coroutine.
        *args: Positional arguments to pass to func.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        The return value of func executed in the executor thread.
    
    Raises:
        RuntimeError: If func raises StopIteration (converted to prevent asyncio.Future deadlock).
        Any other exception from func is propagated as-is.
    
    Example:
        ```python
        # Run sync function from async chain
        async def async_chain_step(input: str, config: RunnableConfig):
            def blocking_operation(data: str) -> str:
                # Expensive CPU work or blocking I/O
                time.sleep(1)
                return data.upper()
            
            # Execute in thread pool to avoid blocking event loop
            result = await run_in_executor(config, blocking_operation, input)
            return result
        
        # With custom executor
        executor = ThreadPoolExecutor(max_workers=10)
        result = await run_in_executor(executor, sync_func, "input")
        ```

    Source: libs/core/langchain_core/runnables/config.py:726-759
    """

    def wrapper() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
            # StopIteration can't be set on an asyncio.Future
            # it raises a TypeError and leaves the Future pending forever
            # so we need to convert it to a RuntimeError
            raise RuntimeError from exc

    if executor_or_config is None or isinstance(executor_or_config, dict):
        # Use default executor with context copied from current context
        return await asyncio.get_running_loop().run_in_executor(
            None,
            cast("Callable[..., T]", partial(copy_context().run, wrapper)),
        )

    return await asyncio.get_running_loop().run_in_executor(executor_or_config, wrapper)
