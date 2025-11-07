"""Streaming Event Schema Definitions for LangChain Expression Language (LCEL).

This module defines TypedDict schemas for the streaming event system used by the
LCEL (LangChain Expression Language) streaming API. These type definitions provide
structured access to events emitted during asynchronous execution of Runnable chains.

Overview
--------
The streaming event system enables real-time monitoring of Runnable execution through
structured event objects. Events follow a consistent lifecycle pattern and provide
visibility into inputs, outputs, intermediate results, and errors across all Runnable
types including chains, language models, prompts, tools, and retrievers.

Event Lifecycle
---------------
All Runnable objects emit events following a three-stage lifecycle:

1. **on_[runnable_type]_start**: Emitted when a Runnable begins execution
   - Contains: input data (when available), metadata, tags, run_id
   - Data payload: `{"input": <input_value>}` (if input known at start)

2. **on_[runnable_type]_stream**: Emitted during streaming execution
   - Contains: streaming chunks as they are produced
   - Data payload: `{"chunk": <chunk_value>}`
   - Chunks support addition: sum(chunks) typically equals final output

3. **on_[runnable_type]_end**: Emitted when a Runnable completes execution
   - Contains: final output, complete input (if streamed), or error (if failed)
   - Data payload: `{"output": <final_output>}` or `{"error": <exception>}`

Runnable Types
--------------
Events are categorized by the type of Runnable that generates them:
- **llm**: Non-chat language models
- **chat_model**: Chat-based language models
- **prompt**: Prompt templates (e.g., ChatPromptTemplate, PromptTemplate)
- **tool**: Tools defined via @tool decorator or BaseTool subclasses
- **chain**: General Runnable compositions (most Runnables fall into this category)
- **retriever**: Document retrieval components

Key Components
--------------
- **EventData**: TypedDict defining the data payload structure for streaming events
- **BaseStreamEvent**: Base schema for all streaming events with common fields
- **StandardStreamEvent**: Standard LangChain events following LCEL conventions
- **CustomStreamEvent**: User-defined events for custom instrumentation
- **StreamEvent**: Union type encompassing all event types

Usage with astream_events
--------------------------
These schemas are primarily used with the `Runnable.astream_events()` method,
which provides the most granular streaming API in LangChain:

    ```python
    from langchain_core.runnables import RunnableLambda

    async def process(text: str) -> str:
        return text.upper()

    chain = RunnableLambda(func=process)

    async for event in chain.astream_events("hello"):
        print(f"Event: {event['event']}")
        print(f"Data: {event['data']}")
        # Event types: on_chain_start, on_chain_stream, on_chain_end
    ```

Related Runnable Methods
------------------------
- `Runnable.astream_events()`: Emit structured events during async execution
- `Runnable.astream_log()`: Emit detailed execution logs with event data
- `Runnable.astream()`: Stream output chunks only (simplified API)

See Also
--------
- langchain_core.runnables.base.Runnable: Base Runnable protocol
- langchain_core.callbacks.base: Callback system for event handling

Source
------
Module: libs/core/langchain_core/runnables/schema.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence


class EventData(TypedDict, total=False):
    """Data payload associated with a streaming event.

    This TypedDict defines the structure of the `data` field in streaming events
    emitted by Runnable objects during execution. All fields are optional (total=False)
    because different event types populate different subsets of these fields based on
    the event lifecycle stage (start, stream, or end).

    Field Availability by Event Type
    ---------------------------------
    - **on_*_start events**: May contain `input` (if known at start)
    - **on_*_stream events**: Contains `chunk` (streaming output fragment)
    - **on_*_end events**: Contains `output` (final result) or `error` (if failed),
      plus `input` (if it was streamed and only known at end)

    Notes
    -----
    The `chunk` field supports addition operations, meaning that summing all chunks
    from stream events typically reconstructs the final output. However, some
    Runnables (particularly chat models) may include additional metadata in `output`
    that isn't present in individual chunks.

    Source
    ------
    Class: libs/core/langchain_core/runnables/schema.py:EventData
    """

    input: Any
    """The input passed to the `Runnable` that generated the event.

    Inputs will sometimes be available at the *START* of the `Runnable`, and
    sometimes at the *END* of the `Runnable`.

    If a `Runnable` is able to stream its inputs, then its input by definition
    won't be known until the *END* of the `Runnable` when it has finished streaming
    its inputs.
    """
    error: NotRequired[BaseException]
    """The error that occurred during the execution of the `Runnable`.

    This field is only available if the `Runnable` raised an exception.

    !!! version-added "Added in version 1.0.0"
    """
    output: Any
    """The output of the `Runnable` that generated the event.

    Outputs will only be available at the *END* of the `Runnable`.

    For most `Runnable` objects, this field can be inferred from the `chunk` field,
    though there might be some exceptions for special a cased `Runnable` (e.g., like
    chat models), which may return more information.
    """
    chunk: Any
    """A streaming chunk from the output that generated the event.

    chunks support addition in general, and adding them up should result
    in the output of the `Runnable` that generated the event.
    """


class BaseStreamEvent(TypedDict):
    """Base schema for all streaming events produced by Runnable.astream_events().

    This TypedDict defines the common structure shared by all streaming events emitted
    during Runnable execution. Events provide real-time visibility into chain execution,
    enabling monitoring, debugging, and custom instrumentation.

    Event Naming Convention
    ------------------------
    Event names follow the pattern: `on_[runnable_type]_(start|stream|end)`

    Where:
    - **runnable_type**: The category of Runnable (llm, chat_model, prompt, tool, chain)
    - **lifecycle_stage**: The execution phase (start, stream, end)

    Examples: `on_chain_start`, `on_llm_stream`, `on_tool_end`

    Execution Tracking
    ------------------
    Each event includes a unique `run_id` to track individual Runnable executions.
    Parent-child relationships are captured via `parent_ids`, enabling reconstruction
    of the complete execution tree for nested Runnable compositions.

    Configuration Propagation
    --------------------------
    Tags and metadata flow through the execution hierarchy:
    - Tags are always inherited from parent Runnables
    - Metadata can be bound at construction or passed at runtime
    - Both can be specified via `.with_config()` or passed to execution methods

    Example
    -------
    The following example demonstrates the complete event lifecycle for a simple chain:

        ```python
        from langchain_core.runnables import RunnableLambda


        async def reverse(s: str) -> str:
            return s[::-1]


        chain = RunnableLambda(func=reverse)

        events = [event async for event in chain.astream_events("hello")]

        # Will produce the following events
        # (where some fields have been omitted for brevity):
        [
            {
                "data": {"input": "hello"},
                "event": "on_chain_start",
                "metadata": {},
                "name": "reverse",
                "tags": [],
            },
            {
                "data": {"chunk": "olleh"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "reverse",
                "tags": [],
            },
            {
                "data": {"output": "olleh"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "reverse",
                "tags": [],
            },
        ]
        ```

    See Also
    --------
    - StandardStreamEvent: Standard LangChain events following LCEL conventions
    - CustomStreamEvent: User-defined events for custom instrumentation
    - EventData: Structure of the `data` field payload

    Source
    ------
    Class: libs/core/langchain_core/runnables/schema.py:BaseStreamEvent
    """

    event: str
    """Event names are of the format: `on_[runnable_type]_(start|stream|end)`.

    Runnable types are one of:

    - **llm** - used by non chat models
    - **chat_model** - used by chat models
    - **prompt** --  e.g., `ChatPromptTemplate`
    - **tool** -- from tools defined via `@tool` decorator or inheriting
        from `Tool`/`BaseTool`
    - **chain** - most `Runnable` objects are of this type

    Further, the events are categorized as one of:

    - **start** - when the `Runnable` starts
    - **stream** - when the `Runnable` is streaming
    - **end* - when the `Runnable` ends

    start, stream and end are associated with slightly different `data` payload.

    Please see the documentation for `EventData` for more details.
    """
    run_id: str
    """An randomly generated ID to keep track of the execution of the given `Runnable`.

    Each child `Runnable` that gets invoked as part of the execution of a parent
    `Runnable` is assigned its own unique ID.
    """
    tags: NotRequired[list[str]]
    """Tags associated with the `Runnable` that generated this event.

    Tags are always inherited from parent `Runnable` objects.

    Tags can either be bound to a `Runnable` using `.with_config({"tags":  ["hello"]})`
    or passed at run time using `.astream_events(..., {"tags": ["hello"]})`.
    """
    metadata: NotRequired[dict[str, Any]]
    """Metadata associated with the `Runnable` that generated this event.

    Metadata can either be bound to a `Runnable` using

        `.with_config({"metadata": { "foo": "bar" }})`

    or passed at run time using

        `.astream_events(..., {"metadata": {"foo": "bar"}})`.
    """

    parent_ids: Sequence[str]
    """A list of the parent IDs associated with this event.

    Root Events will have an empty list.

    For example, if a `Runnable` A calls `Runnable` B, then the event generated by
    `Runnable` B will have `Runnable` A's ID in the `parent_ids` field.

    The order of the parent IDs is from the root parent to the immediate parent.

    Only supported as of v2 of the astream events API. v1 will return an empty list.
    """


class StandardStreamEvent(BaseStreamEvent):
    """Standard streaming event following LangChain LCEL conventions.

    This event type represents the standard streaming events automatically emitted by
    all built-in LangChain Runnable objects. The event structure follows LCEL
    conventions with strongly-typed EventData payloads.

    Usage
    -----
    StandardStreamEvent is the primary event type developers will encounter when using
    `astream_events()` with LangChain chains, language models, prompts, tools, and
    other built-in Runnables. The standardized structure enables consistent event
    processing across different Runnable types.

    Data Payload
    ------------
    The `data` field contains an EventData dictionary with fields populated based on
    the event lifecycle stage:
    - Start events: `{"input": <value>}` (when input is known immediately)
    - Stream events: `{"chunk": <value>}` (streaming output fragments)
    - End events: `{"output": <value>}` or `{"error": <exception>}`

    Runnable Identification
    -----------------------
    The `name` field identifies the specific Runnable instance that emitted the event.
    For chains created from functions, the name defaults to the function name. For
    chains with custom names, use `.with_config({"run_name": "custom_name"})`.

    Example
    -------
        ```python
        from langchain_core.runnables import RunnableLambda

        chain = RunnableLambda(lambda x: x.upper())

        async for event in chain.astream_events("hello"):
            if isinstance(event, dict):  # StandardStreamEvent structure
                print(f"{event['name']}: {event['event']}")
                if event['event'] == 'on_chain_end':
                    print(f"Output: {event['data']['output']}")
        ```

    See Also
    --------
    - BaseStreamEvent: Base schema with common fields
    - CustomStreamEvent: User-defined events for custom instrumentation
    - EventData: Detailed documentation of data payload structure

    Source
    ------
    Class: libs/core/langchain_core/runnables/schema.py:StandardStreamEvent
    """

    data: EventData
    """Event data payload following EventData schema.

    The contents of the event data depend on the event type (start, stream, or end).
    See EventData documentation for field availability by event type.
    """
    name: str
    """The name of the Runnable that generated the event.

    Defaults to the function name for function-based Runnables, or can be customized
    using `.with_config({"run_name": "custom_name"})`.
    """


class CustomStreamEvent(BaseStreamEvent):
    """Custom streaming event for user-defined instrumentation.

    This event type enables developers to emit custom events within their Runnable
    implementations for application-specific monitoring, debugging, or instrumentation
    purposes. Unlike StandardStreamEvent which follows LCEL conventions, custom events
    have no constraints on the data payload structure.

    Use Cases
    ---------
    - Application-specific metrics or telemetry
    - Custom debugging checkpoints within complex chains
    - Business logic milestones (e.g., "on_validation_complete", "on_cache_hit")
    - Integration with external monitoring systems

    Event Identification
    --------------------
    All custom events use the fixed event type `"on_custom_event"`. To distinguish
    between different custom events, use the `name` field which is user-defined.

    Data Payload Flexibility
    -------------------------
    The `data` field accepts any value (Any type), providing complete flexibility for
    custom instrumentation. Common patterns include:
    - Dictionaries with custom metrics: `{"latency_ms": 42, "cache_hit": True}`
    - Structured objects: Pydantic models, dataclasses
    - Simple values: strings, numbers, booleans

    Example
    -------
        ```python
        from langchain_core.runnables import RunnableLambda
        from langchain_core.runnables.schema import CustomStreamEvent

        async def instrumented_chain(x: str):
            # Emit custom event for monitoring
            yield {
                "event": "on_custom_event",
                "name": "preprocessing_complete",
                "data": {"input_length": len(x), "timestamp": "2024-01-15T10:30:00Z"},
                "run_id": "...",
                "tags": [],
                "parent_ids": []
            }
            return x.upper()

        # Custom events appear alongside standard events in astream_events()
        async for event in chain.astream_events("hello"):
            if event['event'] == 'on_custom_event':
                print(f"Custom: {event['name']} - {event['data']}")
        ```

    Notes
    -----
    Custom events must be explicitly emitted by user code. They are not automatically
    generated by the LCEL framework.

    See Also
    --------
    - StandardStreamEvent: Standard LCEL events with structured EventData
    - BaseStreamEvent: Common fields shared by all events

    Source
    ------
    Class: libs/core/langchain_core/runnables/schema.py:CustomStreamEvent
    """

    # Overwrite the event field to be more specific.
    event: Literal["on_custom_event"]  # type: ignore[misc]
    """The event type identifier for all custom events.

    Fixed value: "on_custom_event". Use the `name` field to distinguish between
    different custom event types.
    """
    name: str
    """User-defined name identifying the specific custom event.

    Examples: "validation_complete", "cache_miss", "preprocessing_start"
    """
    data: Any
    """Free-form data payload associated with the event.

    No constraints on structure or type. Common patterns include dictionaries with
    metrics, Pydantic models, or simple scalar values.
    """


StreamEvent = StandardStreamEvent | CustomStreamEvent
"""Union type representing all possible streaming event types.

This type alias encompasses both StandardStreamEvent (emitted automatically by LCEL
Runnables) and CustomStreamEvent (emitted explicitly by user code). Use this type
for type hints when processing events from Runnable.astream_events().

Usage
-----
    ```python
    from langchain_core.runnables import Runnable
    from langchain_core.runnables.schema import StreamEvent

    async def process_events(runnable: Runnable) -> None:
        async for event in runnable.astream_events("input"):
            # event is typed as StreamEvent (StandardStreamEvent | CustomStreamEvent)
            match event['event']:
                case 'on_chain_start' | 'on_llm_start':
                    print(f"Starting: {event['name']}")
                case 'on_custom_event':
                    print(f"Custom event: {event['name']}")
    ```

Source
------
Type alias: libs/core/langchain_core/runnables/schema.py:StreamEvent
"""
