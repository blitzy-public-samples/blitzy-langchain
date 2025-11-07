# Runnable Protocol API Reference

Complete API reference for the core `Runnable` protocol and base classes that form the foundation of LangChain Expression Language (LCEL).

**Source**: `libs/core/langchain_core/runnables/base.py:122`

---

## Overview

### What is a Runnable?

`Runnable` is the fundamental abstraction in LangChain for representing a unit of work that can be invoked, batched, streamed, transformed, and composed. It serves as the foundation for LCEL (LangChain Expression Language), enabling declarative composition of complex workflows with automatic support for sync, async, batch, and streaming operations.

**Generic Type Parameters**:
- `Input`: The type of input the Runnable accepts
- `Output`: The type of output the Runnable produces

```python
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")

class Runnable(Generic[Input, Output]):
    ...
```

### Key Characteristics

1. **Invokable**: Transform single inputs to outputs via `invoke()` and `ainvoke()`
2. **Batchable**: Efficiently process multiple inputs via `batch()` and `abatch()`
3. **Streamable**: Stream output as it's produced via `stream()` and `astream()`
4. **Composable**: Combine with other Runnables using the pipe operator (`|`)
5. **Configurable**: Accept runtime configuration for callbacks, tags, metadata, and concurrency control
6. **Introspectable**: Expose input/output schemas via `input_schema` and `output_schema` properties

### LCEL Foundation

The `Runnable` protocol is the cornerstone of LCEL, which allows you to compose complex chains using the pipe operator:

```python
chain = prompt | model | parser
result = chain.invoke({"question": "What is LCEL?"})
```

**Type Flow Example**:
```
Dict[str, str] → PromptTemplate → List[BaseMessage] → ChatModel → AIMessage → StrOutputParser → str
```

---

## Core Execution Methods

### invoke()

Transform a single input into an output synchronously.

**Source**: `libs/core/langchain_core/runnables/base.py:810`

```python
@abstractmethod
def invoke(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Output:
    """Transform a single input into an output."""
```

**Parameters**:

- **input** (`Input`): The input to transform. Type depends on the specific Runnable implementation.
- **config** (`RunnableConfig | None`, optional): Configuration for execution. Supports:
  - `tags` (`list[str]`): Tags for filtering and categorization
  - `metadata` (`dict[str, Any]`): Custom metadata (JSON-serializable values)
  - `callbacks` (`Callbacks`): Callback handlers for tracing and logging
  - `max_concurrency` (`int | None`): Maximum parallel operations limit
  - `recursion_limit` (`int`): Maximum recursion depth (default: 25)
  - `run_name` (`str`): Custom name for tracer run
  - `run_id` (`uuid.UUID | None`): Unique identifier for the run
  - `configurable` (`dict[str, Any]`): Runtime values for configurable fields
- **kwargs** (`Any`): Additional keyword arguments passed to the underlying implementation

**Returns**: `Output` - The transformed output

**Example - Simple Invocation**:
```python
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

runnable = RunnableLambda(add_one)
result = runnable.invoke(5)
print(result)  # Output: 6
```

**Example - With Configuration**:
```python
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers import ConsoleCallbackHandler

runnable = RunnableLambda(lambda x: x * 2)
result = runnable.invoke(
    10,
    config={
        "tags": ["math", "multiplication"],
        "metadata": {"user_id": "123"},
        "callbacks": [ConsoleCallbackHandler()],
    }
)
print(result)  # Output: 20
```

**See Also**: [ainvoke()](#ainvoke), [RunnableConfig](#runnableconfig)

---

### ainvoke()

Transform a single input into an output asynchronously.

**Source**: `libs/core/langchain_core/runnables/base.py:830`

```python
async def ainvoke(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Output:
    """Async transform of a single input into an output."""
```

**Parameters**: Same as [invoke()](#invoke)

**Returns**: `Output` - The transformed output (awaitable)

**Default Behavior**: By default, `ainvoke()` executes the synchronous `invoke()` method in a thread pool executor using `asyncio.run_in_executor()`. Override this method in subclasses for native async implementations.

**Async Considerations**:
- Always use `await` when calling `ainvoke()`
- Runs in asyncio event loop context
- For native async implementations, override to avoid thread pool overhead
- Callback handlers can be async-aware (use `AsyncCallbackHandler`)

**Example - Basic Async Usage**:
```python
import asyncio
from langchain_core.runnables import RunnableLambda

async def async_add_one(x: int) -> int:
    await asyncio.sleep(0.1)  # Simulate async work
    return x + 1

runnable = RunnableLambda(async_add_one)

async def main():
    result = await runnable.ainvoke(5)
    print(result)  # Output: 6

asyncio.run(main())
```

**Example - Concurrent Execution**:
```python
import asyncio
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: x * 2)

async def main():
    results = await asyncio.gather(
        runnable.ainvoke(1),
        runnable.ainvoke(2),
        runnable.ainvoke(3),
    )
    print(results)  # Output: [2, 4, 6]

asyncio.run(main())
```

**See Also**: [invoke()](#invoke), [Async Usage Guide](../../guides/async-usage.md)

---

### batch()

Efficiently transform multiple inputs into outputs in parallel.

**Source**: `libs/core/langchain_core/runnables/base.py:851`

```python
def batch(
    self,
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> list[Output]:
    """Transform multiple inputs into outputs in parallel."""
```

**Parameters**:

- **inputs** (`list[Input]`): List of inputs to process
- **config** (`RunnableConfig | list[RunnableConfig] | None`): Either a single config applied to all inputs, or a list of configs (one per input)
- **return_exceptions** (`bool`, default: `False`): 
  - If `False`: Raises exception on first failure
  - If `True`: Returns exception objects in output list instead of raising
- **kwargs** (`Any`): Additional keyword arguments passed to each invocation

**Returns**: `list[Output]` - List of transformed outputs (or exceptions if `return_exceptions=True`)

**Default Implementation**: Uses thread pool executor to run `invoke()` in parallel. Override in subclasses for more efficient batch processing (e.g., if underlying API supports native batching).

**Parallelization**: Controlled by `max_concurrency` in config. If not specified, uses `ThreadPoolExecutor` default.

**Example - Basic Batch**:
```python
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: x ** 2)
results = runnable.batch([1, 2, 3, 4, 5])
print(results)  # Output: [1, 4, 9, 16, 25]
```

**Example - Per-Input Configuration**:
```python
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: x * 2)
results = runnable.batch(
    [1, 2, 3],
    config=[
        {"tags": ["first"]},
        {"tags": ["second"]},
        {"tags": ["third"]},
    ]
)
print(results)  # Output: [2, 4, 6]
```

**Example - Exception Handling**:
```python
from langchain_core.runnables import RunnableLambda

def sometimes_fail(x: int) -> int:
    if x == 2:
        raise ValueError("Cannot process 2")
    return x * 10

runnable = RunnableLambda(sometimes_fail)

# Return exceptions instead of raising
results = runnable.batch([1, 2, 3], return_exceptions=True)
print(results)  # Output: [10, ValueError("Cannot process 2"), 30]
```

**Performance Tip**: For IO-bound operations, batch() provides automatic parallelization. For CPU-bound operations, consider implementing custom batch logic in your Runnable subclass.

**See Also**: [abatch()](#abatch), [batch_as_completed()](#batch_as_completed)

---

### abatch()

Asynchronously transform multiple inputs into outputs in parallel.

**Source**: `libs/core/langchain_core/runnables/base.py:900+`

```python
async def abatch(
    self,
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> list[Output]:
    """Async transform of multiple inputs into outputs in parallel."""
```

**Parameters**: Same as [batch()](#batch)

**Returns**: `list[Output]` - List of transformed outputs (awaitable)

**Default Behavior**: Uses `asyncio.gather()` to run `ainvoke()` concurrently for all inputs. Override for more efficient async batch processing.

**Example - Basic Async Batch**:
```python
import asyncio
from langchain_core.runnables import RunnableLambda

async def async_square(x: int) -> int:
    await asyncio.sleep(0.1)
    return x ** 2

runnable = RunnableLambda(async_square)

async def main():
    results = await runnable.abatch([1, 2, 3, 4, 5])
    print(results)  # Output: [1, 4, 9, 16, 25]

asyncio.run(main())
```

**See Also**: [batch()](#batch), [ainvoke()](#ainvoke)

---

### stream()

Stream output from a single input as it's produced.

**Source**: `libs/core/langchain_core/runnables/base.py:1107`

```python
def stream(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any | None,
) -> Iterator[Output]:
    """Stream output as it's produced."""
```

**Parameters**: Same as [invoke()](#invoke)

**Returns**: `Iterator[Output]` - Iterator yielding output chunks

**Default Behavior**: Default implementation simply yields the result of `invoke()` (single yield). Override in subclasses to provide true streaming behavior (e.g., LLM token-by-token streaming).

**Example - Basic Stream** (default behavior):
```python
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: x.upper())
for chunk in runnable.stream("hello"):
    print(chunk)  # Output: HELLO (single chunk)
```

**Example - Custom Streaming Runnable**:
```python
from typing import Iterator
from langchain_core.runnables import RunnableGenerator

def generate_chunks(input: str) -> Iterator[str]:
    """Stream output character by character."""
    for char in input:
        yield char

runnable = RunnableGenerator(generate_chunks)
for chunk in runnable.stream("LCEL"):
    print(chunk, end="", flush=True)  # Output: L C E L (streamed)
print()
```

**Use Cases**:
- LLM token streaming for better UX
- Processing large datasets chunk-by-chunk
- Real-time data pipeline processing

**See Also**: [astream()](#astream), [RunnableGenerator](#runnablegenerator)

---

### astream()

Asynchronously stream output from a single input as it's produced.

**Source**: `libs/core/langchain_core/runnables/base.py:1128`

```python
async def astream(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any | None,
) -> AsyncIterator[Output]:
    """Async stream output as it's produced."""
```

**Parameters**: Same as [invoke()](#invoke)

**Returns**: `AsyncIterator[Output]` - Async iterator yielding output chunks

**Default Behavior**: Yields the result of `ainvoke()` as a single chunk. Override for true async streaming.

**Example - Async Stream**:
```python
import asyncio
from typing import AsyncIterator
from langchain_core.runnables import RunnableGenerator

async def async_generate_chunks(input: str) -> AsyncIterator[str]:
    """Stream output character by character asynchronously."""
    for char in input:
        await asyncio.sleep(0.1)  # Simulate async work
        yield char

runnable = RunnableGenerator(async_generate_chunks)

async def main():
    async for chunk in runnable.astream("LCEL"):
        print(chunk, end="", flush=True)  # Output: L C E L (streamed)
    print()

asyncio.run(main())
```

**See Also**: [stream()](#stream), [astream_log()](#astream_log)

---

## Additional Execution Methods

### batch_as_completed()

Process inputs in batch and yield results as they complete (not in original order).

**Source**: `libs/core/langchain_core/runnables/base.py:1050+`

```python
def batch_as_completed(
    self,
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> Iterator[tuple[int, Output]]:
    """Yield (index, output) tuples as each input completes."""
```

**Parameters**: Same as [batch()](#batch)

**Returns**: `Iterator[tuple[int, Output]]` - Yields `(index, output)` tuples as each input completes

**Use Case**: Process results as soon as they're available rather than waiting for all to complete.

**Example**:
```python
from langchain_core.runnables import RunnableLambda
import time

def slow_process(x: int) -> int:
    time.sleep(x * 0.1)  # Simulates variable processing time
    return x * 10

runnable = RunnableLambda(slow_process)

for index, result in runnable.batch_as_completed([5, 1, 3]):
    print(f"Input {index} completed with result: {result}")
# Output order: (1, 10), (2, 30), (0, 50)
```

---

### astream_log()

Stream all output and intermediate results from a Runnable, as reported to the callback system.

**Source**: `libs/core/langchain_core/runnables/base.py:1183`

```python
async def astream_log(
    self,
    input: Any,
    config: RunnableConfig | None = None,
    *,
    diff: bool = True,
    with_streamed_output_list: bool = True,
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]:
    """Stream output and intermediate results."""
```

**Purpose**: Provides detailed visibility into chain execution, including intermediate steps from nested LLMs, retrievers, tools, etc.

**Key Parameters**:
- **diff** (`bool`): If `True`, yields patches (incremental updates); if `False`, yields full state
- **include_names** / **exclude_names**: Filter by Runnable names
- **include_types** / **exclude_types**: Filter by Runnable types (e.g., `"llm"`, `"retriever"`)
- **include_tags** / **exclude_tags**: Filter by tags

**Use Case**: Debugging complex chains, building streaming UIs that show intermediate results.

**See Also**: [astream_events()](#astream_events)

---

### astream_events()

Stream fine-grained events from chain execution for granular observability.

**Purpose**: Provides event-by-event streaming for maximum control over displayed information.

**See Also**: [astream_log()](#astream_log), [Callbacks Guide](../../guides/callbacks.md)

---

## Configuration

### RunnableConfig

TypedDict defining configuration options for Runnable execution.

**Source**: `libs/core/langchain_core/runnables/config.py:49`

```python
class RunnableConfig(TypedDict, total=False):
    tags: list[str]
    metadata: dict[str, Any]
    callbacks: Callbacks
    run_name: str
    max_concurrency: int | None
    recursion_limit: int
    configurable: dict[str, Any]
    run_id: uuid.UUID | None
```

**Configuration Keys**:

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `tags` | `list[str]` | Tags for filtering and categorization in tracing systems | `[]` |
| `metadata` | `dict[str, Any]` | Custom metadata (JSON-serializable); passed to `handle*Start` callbacks | `{}` |
| `callbacks` | `Callbacks` | Callback handlers for tracing, logging, and monitoring | `None` |
| `run_name` | `str` | Custom name for the tracer run; defaults to class name | Class name |
| `max_concurrency` | `int \| None` | Maximum number of parallel operations (for batch processing) | `ThreadPoolExecutor` default |
| `recursion_limit` | `int` | Maximum recursion depth for nested Runnable calls | `25` |
| `configurable` | `dict[str, Any]` | Runtime values for configurable fields (set via `configurable_fields()`) | `{}` |
| `run_id` | `uuid.UUID \| None` | Unique identifier for the run; auto-generated if not provided | Auto-generated |

**Example - Full Configuration**:
```python
import uuid
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers import ConsoleCallbackHandler

runnable = RunnableLambda(lambda x: x * 2)

config = {
    "tags": ["production", "api_v1"],
    "metadata": {"user_id": "user_123", "session_id": "session_456"},
    "callbacks": [ConsoleCallbackHandler()],
    "run_name": "double_operation",
    "max_concurrency": 10,
    "recursion_limit": 50,
    "run_id": uuid.uuid4(),
}

result = runnable.invoke(42, config=config)
```

**See Also**: [with_config()](#with_config)

---

## Schema Introspection

### input_schema

Property that returns a Pydantic model representing the Runnable's input type.

**Source**: `libs/core/langchain_core/runnables/base.py:382`

```python
@property
def input_schema(self) -> type[BaseModel]:
    """Input schema as a Pydantic model."""
    return self.get_input_schema()
```

**Returns**: `type[BaseModel]` - Pydantic model class for input validation

**Example**:
```python
from langchain_core.runnables import RunnableLambda

def add_numbers(data: dict) -> int:
    return data["a"] + data["b"]

runnable = RunnableLambda(add_numbers)
print(runnable.input_schema.model_json_schema())
# Output: JSON schema describing expected input structure
```

---

### output_schema

Property that returns a Pydantic model representing the Runnable's output type.

**Source**: `libs/core/langchain_core/runnables/base.py:433`

```python
@property
def output_schema(self) -> type[BaseModel]:
    """Output schema as a Pydantic model."""
    return self.get_output_schema()
```

**Returns**: `type[BaseModel]` - Pydantic model class for output validation

**Example**:
```python
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: x * 2)
print(runnable.output_schema.model_json_schema())
# Output: JSON schema describing output structure
```

---

### get_input_schema()

Get input schema, optionally specific to a configuration.

**Source**: `libs/core/langchain_core/runnables/base.py:362+`

```python
def get_input_schema(
    self,
    config: RunnableConfig | None = None,
) -> type[BaseModel]:
    """Get Pydantic model for input validation."""
```

**Parameters**:
- **config** (`RunnableConfig | None`): Configuration to use for schema generation (relevant for configurable Runnables)

**Returns**: `type[BaseModel]` - Pydantic model class

---

### get_output_schema()

Get output schema, optionally specific to a configuration.

**Source**: `libs/core/langchain_core/runnables/base.py:440`

```python
def get_output_schema(
    self,
    config: RunnableConfig | None = None,
) -> type[BaseModel]:
    """Get Pydantic model for output validation."""
```

**Parameters**: Same as `get_input_schema()`

**Returns**: `type[BaseModel]` - Pydantic model class

---

### get_input_jsonschema() / get_output_jsonschema()

Get JSON Schema representations of input/output schemas.

**Source**: `libs/core/langchain_core/runnables/base.py:402`, `base.py:480`

```python
def get_input_jsonschema(
    self, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Get JSON schema for input."""

def get_output_jsonschema(
    self, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Get JSON schema for output."""
```

**Returns**: `dict[str, Any]` - JSON Schema as dictionary

**Example**:
```python
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: str(x))
input_schema = runnable.get_input_jsonschema()
output_schema = runnable.get_output_jsonschema()
print(input_schema)
print(output_schema)
```

---

## Composition and Transformation

### Pipe Operator (`|`)

Compose Runnables sequentially using the pipe operator. See [composition.md](./composition.md) for comprehensive documentation.

**Example**:
```python
from langchain_core.runnables import RunnableLambda

add_one = RunnableLambda(lambda x: x + 1)
multiply_two = RunnableLambda(lambda x: x * 2)

chain = add_one | multiply_two
result = chain.invoke(5)  # (5 + 1) * 2 = 12
print(result)  # Output: 12
```

**Type Flow**: `Runnable[A, B] | Runnable[B, C]` → `Runnable[A, C]`

**See Also**: [LCEL Composition Guide](../../guides/lcel-composition.md), [Composition API](./composition.md)

---

### with_retry()

Wrap a Runnable with automatic retry logic on exceptions.

**Source**: `libs/core/langchain_core/runnables/base.py:1825`

```python
def with_retry(
    self,
    *,
    retry_if_exception_type: tuple[type[BaseException], ...] = (Exception,),
    wait_exponential_jitter: bool = True,
    exponential_jitter_params: ExponentialJitterParams | None = None,
    stop_after_attempt: int = 3,
) -> Runnable[Input, Output]:
    """Create a new Runnable with automatic retry on exceptions."""
```

**Parameters**:
- **retry_if_exception_type** (`tuple[type[BaseException], ...]`): Exception types to retry on (default: `(Exception,)`)
- **wait_exponential_jitter** (`bool`): Add jitter to wait time between retries (default: `True`)
- **stop_after_attempt** (`int`): Maximum number of attempts before giving up (default: `3`)
- **exponential_jitter_params** (`ExponentialJitterParams | None`): Custom parameters for exponential backoff (`initial`, `max`, `exp_base`, `jitter`)

**Returns**: `Runnable[Input, Output]` - New Runnable with retry logic

**Example - Basic Retry**:
```python
from langchain_core.runnables import RunnableLambda
import random

def flaky_operation(x: int) -> int:
    if random.random() < 0.7:  # 70% failure rate
        raise ValueError("Random failure")
    return x * 2

runnable = RunnableLambda(flaky_operation).with_retry(
    stop_after_attempt=5,
    retry_if_exception_type=(ValueError,),
)

result = runnable.invoke(10)  # Will retry up to 5 times
print(result)  # Output: 20 (eventually succeeds)
```

**Example - Custom Exponential Backoff**:
```python
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: x).with_retry(
    stop_after_attempt=10,
    wait_exponential_jitter=True,
    exponential_jitter_params={
        "initial": 1.0,   # Start with 1 second
        "max": 60.0,      # Max 60 seconds between retries
        "exp_base": 2.0,  # Double wait time each retry
        "jitter": True,   # Add randomness
    }
)
```

**Use Cases**:
- Handling transient API errors (rate limits, network failures)
- Retrying LLM calls during service degradation
- Resilient data pipeline processing

**See Also**: [Error Handling Guide](../../guides/error-handling.md), [with_fallbacks()](#with_fallbacks)

---

### with_fallbacks()

Add fallback Runnables that are tried in order upon failures.

**Source**: `libs/core/langchain_core/runnables/base.py:1912`

```python
def with_fallbacks(
    self,
    fallbacks: Sequence[Runnable[Input, Output]],
    *,
    exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,),
    exception_key: str | None = None,
) -> RunnableWithFallbacks[Input, Output]:
    """Add fallback Runnables tried in order upon failures."""
```

**Parameters**:
- **fallbacks** (`Sequence[Runnable[Input, Output]]`): Sequence of fallback Runnables to try in order
- **exceptions_to_handle** (`tuple[type[BaseException], ...]`): Exception types that trigger fallbacks (default: `(Exception,)`)
- **exception_key** (`str | None`): If specified, pass caught exception to fallbacks in input dict under this key (default: `None`)

**Returns**: `RunnableWithFallbacks[Input, Output]` - New Runnable with fallback logic

**Example - Basic Fallback**:
```python
from langchain_core.runnables import RunnableLambda

def primary_operation(x: int) -> int:
    raise ValueError("Primary always fails")

def fallback_operation(x: int) -> int:
    return x * 10  # Fallback succeeds

primary = RunnableLambda(primary_operation)
fallback = RunnableLambda(fallback_operation)

runnable = primary.with_fallbacks([fallback])
result = runnable.invoke(5)
print(result)  # Output: 50 (from fallback)
```

**Example - Multiple Fallbacks**:
```python
from langchain_core.runnables import RunnableLambda

def first(x: int) -> int:
    raise ValueError("First fails")

def second(x: int) -> int:
    raise ValueError("Second also fails")

def third(x: int) -> int:
    return x * 100  # Third succeeds

runnable = RunnableLambda(first).with_fallbacks([
    RunnableLambda(second),
    RunnableLambda(third),
])
result = runnable.invoke(1)
print(result)  # Output: 100 (from third fallback)
```

**Example - Exception Passing**:
```python
from langchain_core.runnables import RunnableLambda

def primary(data: dict) -> str:
    raise ValueError("Failed with data")

def fallback_with_error_info(data: dict) -> str:
    error = data.get("error")
    return f"Fallback handled error: {error}"

runnable = RunnableLambda(primary).with_fallbacks(
    [RunnableLambda(fallback_with_error_info)],
    exception_key="error",
)
result = runnable.invoke({"value": 42})
print(result)  # Output: "Fallback handled error: Failed with data"
```

**Use Cases**:
- Primary/secondary LLM model fallbacks (e.g., GPT-4 → GPT-3.5)
- Multiple API endpoint attempts
- Graceful degradation strategies

**See Also**: [Error Handling Guide](../../guides/error-handling.md), [with_retry()](#with_retry)

---

### with_config()

Bind default configuration to a Runnable.

**Source**: `libs/core/langchain_core/runnables/base.py:1615`

```python
def with_config(
    self,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Runnable[Input, Output]:
    """Bind default config to a Runnable."""
```

**Parameters**:
- **config** (`RunnableConfig | None`): Configuration dictionary
- **kwargs**: Additional config keys as keyword arguments

**Returns**: `Runnable[Input, Output]` - New Runnable with bound config

**Example**:
```python
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers import ConsoleCallbackHandler

runnable = RunnableLambda(lambda x: x * 2)

# Bind default configuration
configured_runnable = runnable.with_config(
    tags=["production", "critical"],
    metadata={"version": "1.0"},
    callbacks=[ConsoleCallbackHandler()],
)

# Config is automatically applied on every invocation
result = configured_runnable.invoke(10)
```

**Use Case**: Set default tags, metadata, or callbacks for all invocations of a Runnable.

---

### with_types()

Explicitly set input and output types for a Runnable.

**Source**: `libs/core/langchain_core/runnables/base.py:1803`

```python
def with_types(
    self,
    *,
    input_type: type[Input] | None = None,
    output_type: type[Output] | None = None,
) -> Runnable[Input, Output]:
    """Bind explicit input/output types to a Runnable."""
```

**Parameters**:
- **input_type** (`type[Input] | None`): Explicit input type
- **output_type** (`type[Output] | None`): Explicit output type

**Returns**: `Runnable[Input, Output]` - New Runnable with explicit types

**Use Case**: Override type inference for better schema generation and type checking.

---

### with_listeners()

Bind lifecycle listeners to a Runnable for custom event handling.

**Source**: `libs/core/langchain_core/runnables/base.py:1640`

```python
def with_listeners(
    self,
    *,
    on_start: Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None = None,
    on_end: Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None = None,
    on_error: Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None = None,
) -> Runnable[Input, Output]:
    """Bind lifecycle listeners to a Runnable."""
```

**Parameters**:
- **on_start**: Called before Runnable starts (receives `Run` object with `id`, `type`, `input`)
- **on_end**: Called after Runnable completes (receives `Run` object with `output`, `end_time`)
- **on_error**: Called if Runnable throws an error (receives `Run` object with `error`)

**Returns**: `Runnable[Input, Output]` - New Runnable with listeners

**Example**:
```python
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.schemas import Run

def log_start(run: Run):
    print(f"Starting run {run.id} with input: {run.inputs}")

def log_end(run: Run):
    print(f"Completed run {run.id} with output: {run.outputs}")

def log_error(run: Run):
    print(f"Run {run.id} failed with error: {run.error}")

runnable = RunnableLambda(lambda x: x * 2).with_listeners(
    on_start=log_start,
    on_end=log_end,
    on_error=log_error,
)

result = runnable.invoke(10)
```

**Use Cases**:
- Custom logging and monitoring
- Performance tracking
- Alerting on errors
- Audit trail creation

**See Also**: [with_alisteners()](#with_alisteners) for async variant, [Callbacks Guide](../../guides/callbacks.md)

---

### with_alisteners()

Bind async lifecycle listeners to a Runnable.

**Source**: `libs/core/langchain_core/runnables/base.py:1712`

Similar to `with_listeners()` but listeners are async functions. See [with_listeners()](#with_listeners) for details.

---

## Helper Classes

### RunnableSerializable

Extends `Runnable` with serialization support for saving/loading Runnable configurations.

**Source**: `libs/core/langchain_core/runnables/base.py` (extends Serializable)

**Key Methods**:
- `to_json()`: Serialize Runnable to JSON
- `from_json()`: Deserialize Runnable from JSON

**Use Case**: Persist and restore chain configurations.

---

### RunnableLambda

Wraps a Python function as a Runnable.

**Source**: Referenced in examples throughout `libs/core/langchain_core/runnables/base.py`

```python
from langchain_core.runnables import RunnableLambda

# Wrap a function
def my_function(x: int) -> int:
    return x * 2

runnable = RunnableLambda(my_function)
result = runnable.invoke(5)  # Output: 10
```

**Features**:
- Auto-detects async functions
- Supports both sync and async execution
- Automatically implements all Runnable methods

**See Also**: [@chain decorator](#chain-decorator)

---

### RunnableBinding

Binds partial configuration or kwargs to a Runnable.

**Source**: Created internally by `with_config()`, `bind()` methods

**Use Case**: Pre-configure Runnables with default parameters or config.

---

### RunnableGenerator

Custom streaming implementation for generator functions.

**Source**: Used in streaming examples in `libs/core/langchain_core/runnables/base.py`

```python
from typing import Iterator
from langchain_core.runnables import RunnableGenerator

def my_generator(input: str) -> Iterator[str]:
    for word in input.split():
        yield word

runnable = RunnableGenerator(my_generator)
for chunk in runnable.stream("hello world"):
    print(chunk)  # Outputs: "hello", then "world"
```

**Use Case**: Implement custom streaming logic.

---

## @chain Decorator

Decorator that automatically wraps a function as a Runnable.

**Source**: `libs/core/langchain_core/runnables/base.py` (decorator function)

```python
from langchain_core.runnables import chain

@chain
def my_chain(x: int) -> int:
    return x * 2

# my_chain is now a Runnable
result = my_chain.invoke(5)  # Output: 10
```

**Features**:
- Supports both sync and async functions
- Automatically implements all Runnable methods
- Cleaner syntax than `RunnableLambda`

**Example - Async Chain**:
```python
import asyncio
from langchain_core.runnables import chain

@chain
async def async_chain(x: int) -> int:
    await asyncio.sleep(0.1)
    return x * 3

async def main():
    result = await async_chain.ainvoke(5)
    print(result)  # Output: 15

asyncio.run(main())
```

---

## Type Parameters

### Generic Type Variables

`Runnable` uses Python generics to provide type safety:

```python
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")

class Runnable(Generic[Input, Output]):
    ...
```

**Type Inference in Composition**:

When composing Runnables with the pipe operator, types are automatically inferred:

```python
# Type flow: int → str → bool
runnable1: Runnable[int, str] = ...
runnable2: Runnable[str, bool] = ...

composed: Runnable[int, bool] = runnable1 | runnable2
```

**Type Safety**: The compiler ensures that the output type of `runnable1` (`str`) matches the input type of `runnable2` (`str`).

---

## Complete Examples

### Example 1: Basic Invocation

```python
from langchain_core.runnables import RunnableLambda

def uppercase(text: str) -> str:
    return text.upper()

runnable = RunnableLambda(uppercase)
result = runnable.invoke("hello world")
print(result)  # Output: HELLO WORLD
```

---

### Example 2: Async Invocation

```python
import asyncio
from langchain_core.runnables import RunnableLambda

async def async_uppercase(text: str) -> str:
    await asyncio.sleep(0.1)
    return text.upper()

runnable = RunnableLambda(async_uppercase)

async def main():
    result = await runnable.ainvoke("hello world")
    print(result)  # Output: HELLO WORLD

asyncio.run(main())
```

---

### Example 3: Batch Processing

```python
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: x ** 2)
results = runnable.batch([1, 2, 3, 4, 5])
print(results)  # Output: [1, 4, 9, 16, 25]
```

---

### Example 4: Streaming

```python
from typing import Iterator
from langchain_core.runnables import RunnableGenerator

def stream_words(text: str) -> Iterator[str]:
    for word in text.split():
        yield word + " "

runnable = RunnableGenerator(stream_words)
for chunk in runnable.stream("Hello from LangChain"):
    print(chunk, end="", flush=True)
# Output: Hello from LangChain
```

---

### Example 5: Configuration with Callbacks

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers import ConsoleCallbackHandler

runnable = RunnableLambda(lambda x: x * 2)
result = runnable.invoke(
    42,
    config={
        "tags": ["math", "multiplication"],
        "callbacks": [ConsoleCallbackHandler()],
    }
)
print(result)  # Output: 84 (with console tracing)
```

---

### Example 6: Retry with Exponential Backoff

```python
from langchain_core.runnables import RunnableLambda
import random

def unreliable_service(x: int) -> int:
    if random.random() < 0.8:  # 80% failure rate
        raise ValueError("Service temporarily unavailable")
    return x * 10

runnable = RunnableLambda(unreliable_service).with_retry(
    stop_after_attempt=10,
    retry_if_exception_type=(ValueError,),
    wait_exponential_jitter=True,
)

result = runnable.invoke(5)
print(result)  # Output: 50 (after retries)
```

---

### Example 7: Complete Chain with Prompt | LLM | Parser

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Define components
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# Compose with LCEL
chain = prompt | model | parser

# Type flow: Dict[str, str] → ChatPromptValue → AIMessage → str
result = chain.invoke({"topic": "programming"})
print(result)  # Output: A joke about programming
```

---

## Performance Considerations

### Batch Parallelization

**Default Behavior**: `batch()` uses `ThreadPoolExecutor` to run `invoke()` in parallel for each input.

**Optimization**: For Runnables with native batch APIs (e.g., LLM batch endpoints), override `batch()` to use the batch API directly:

```python
class MyBatchableRunnable(Runnable[Input, Output]):
    def batch(self, inputs: list[Input], ...) -> list[Output]:
        # Use native batch API instead of parallel invoke()
        return self.native_batch_api(inputs)
```

**Concurrency Control**: Use `max_concurrency` in config to limit parallel operations:

```python
results = runnable.batch(
    inputs,
    config={"max_concurrency": 5}  # Max 5 parallel operations
)
```

---

### Async for Concurrency

Use async methods (`ainvoke`, `abatch`, `astream`) for IO-bound operations to maximize concurrency:

```python
import asyncio

results = await asyncio.gather(
    runnable.ainvoke(input1),
    runnable.ainvoke(input2),
    runnable.ainvoke(input3),
)
```

---

### Streaming for Responsiveness

For LLMs and long-running operations, use `stream()` or `astream()` to show progress:

```python
for chunk in runnable.stream(input):
    print(chunk, end="", flush=True)
```

**UX Benefit**: Users see partial results immediately rather than waiting for full completion.

---

## Troubleshooting

### Type Errors in LCEL Composition

**Problem**: Type mismatch when composing Runnables with `|`

```python
runnable1: Runnable[int, str] = ...
runnable2: Runnable[bool, str] = ...  # Input is bool, not str

chain = runnable1 | runnable2  # Type error!
```

**Solution**: Ensure output type of left Runnable matches input type of right Runnable. Use `with_types()` if type inference fails.

---

### Config Conflicts

**Problem**: Multiple configs (from `with_config()`, `invoke()` parameter, etc.) conflict

**Resolution Order**: Configs are merged with this precedence (highest to lowest):
1. `invoke(config=...)` parameter
2. Bound config from `with_config()`
3. Default config

**Solution**: Use `merge_configs()` utility if you need custom merging logic.

---

### Async Event Loop Issues

**Problem**: `RuntimeError: no running event loop` or similar async errors

**Solution**:
1. Always call async methods with `await` inside an async function
2. Use `asyncio.run(main())` to start the event loop
3. For Jupyter notebooks, use `await` directly (Jupyter has a running event loop)

```python
# Wrong
result = runnable.ainvoke(input)  # Missing await!

# Correct
async def main():
    result = await runnable.ainvoke(input)
    return result

asyncio.run(main())
```

---

### Batch Return Exceptions

**Problem**: `batch()` raises exception on first failure, preventing processing of remaining inputs

**Solution**: Use `return_exceptions=True` to return exceptions as values:

```python
results = runnable.batch(inputs, return_exceptions=True)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Input {i} failed: {result}")
    else:
        print(f"Input {i} succeeded: {result}")
```

---

## See Also

- **[LCEL Composition Guide](../../guides/lcel-composition.md)**: Comprehensive guide to composing Runnables with the pipe operator
- **[Async Usage Guide](../../guides/async-usage.md)**: Best practices for async/await patterns in LangChain
- **[Error Handling Guide](../../guides/error-handling.md)**: Retry logic, fallbacks, and error recovery strategies
- **[Composition API](./composition.md)**: Detailed documentation of LCEL operators and composition utilities
- **[RunnableConfig](./config.md)**: Complete configuration reference
- **[Glossary - Runnable](../../glossary.md#runnable)**: Runnable definition and key concepts
- **[Glossary - LCEL](../../glossary.md#lcel)**: LangChain Expression Language overview

---

## API Summary Table

| Method | Sync/Async | Purpose | Returns |
|--------|------------|---------|---------|
| `invoke()` | Sync | Transform single input | `Output` |
| `ainvoke()` | Async | Async transform single input | `Output` (awaitable) |
| `batch()` | Sync | Transform multiple inputs in parallel | `list[Output]` |
| `abatch()` | Async | Async transform multiple inputs | `list[Output]` (awaitable) |
| `stream()` | Sync | Stream output chunks | `Iterator[Output]` |
| `astream()` | Async | Async stream output chunks | `AsyncIterator[Output]` |
| `batch_as_completed()` | Sync | Yield results as they complete | `Iterator[tuple[int, Output]]` |
| `astream_log()` | Async | Stream output + intermediate results | `AsyncIterator[RunLogPatch]` |
| `with_retry()` | Both | Add retry logic | `Runnable[Input, Output]` |
| `with_fallbacks()` | Both | Add fallback logic | `Runnable[Input, Output]` |
| `with_config()` | Both | Bind default config | `Runnable[Input, Output]` |
| `with_listeners()` | Sync | Add lifecycle listeners | `Runnable[Input, Output]` |
| `with_types()` | Both | Set explicit types | `Runnable[Input, Output]` |

---

**Last Updated**: 2025-01-09  
**LangChain Version**: 1.0.1+  
**Maintained By**: Blitzy Documentation Team

