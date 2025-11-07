# LangChain Core Runnables

## Module Overview

The `langchain_core.runnables` module is the foundational runtime package for LangChain, implementing the **Runnable protocol** - the core abstraction that enables composable, type-safe operations throughout the LangChain ecosystem. This module provides:

- **Runnable Protocol**: Generic interface for units of work with automatic sync/async/batch/streaming support
- **LCEL (LangChain Expression Language)**: Declarative composition system for building chains using the pipe operator (`|`)
- **Composition Primitives**: RunnableSequence, RunnableParallel, RunnableBranch, RunnableWithFallbacks, and more
- **Configuration System**: RunnableConfig for execution control, tracing, callbacks, and runtime customization
- **Execution Patterns**: Comprehensive support for single/batch/streaming execution in both sync and async contexts

Every Runnable automatically inherits optimized implementations for parallel batch processing, async execution, token-by-token streaming, and complete traceability - making it the ideal building block for production LangChain applications.

**Source**: `libs/core/langchain_core/runnables/base.py:122-254`

---

## LCEL: LangChain Expression Language

**LCEL** is a declarative way to compose Runnable objects into chains using intuitive syntax. Any chain constructed with LCEL automatically gains:

- ✅ **Sync and Async Support**: Both `invoke()` and `ainvoke()` work without additional code
- ✅ **Batch Processing**: Efficient parallel execution via `batch()` and `abatch()`
- ✅ **Streaming**: Token-by-token output through `stream()` and `astream()`
- ✅ **Type Safety**: Compile-time type checking ensures compatible compositions
- ✅ **Observability**: Automatic tracing, logging, and callback support

### Core Concept

LCEL uses the **pipe operator** (`|`) to chain Runnables together, where the output of one Runnable becomes the input to the next:

```python
chain = step1 | step2 | step3
result = chain.invoke(input)  # Automatically flows through all steps
```

This is similar to Unix pipes or functional composition, but with full type safety and automatic optimization.

**Source**: `libs/core/langchain_core/runnables/base.py:150-167`

---

## Runnable Protocol

The `Runnable` abstract base class defines the contract that all composable LangChain components must follow:

### Type Parameters

```python
Runnable[Input, Output]
```

- **Input**: The type of data this Runnable accepts
- **Output**: The type of data this Runnable produces

Type parameters enable type-safe composition: `Runnable[A, B] | Runnable[B, C]` produces `Runnable[A, C]`.

### Core Methods

| Method | Description | Return Type | Use Case |
|--------|-------------|-------------|----------|
| `invoke(input, config)` | Transform single input synchronously | `Output` | Single request processing |
| `ainvoke(input, config)` | Transform single input asynchronously | `Awaitable[Output]` | Async/await workflows, concurrent operations |
| `batch(inputs, config)` | Transform multiple inputs in parallel | `list[Output]` | Bulk processing, throughput optimization |
| `abatch(inputs, config)` | Transform multiple inputs asynchronously | `Awaitable[list[Output]]` | High-concurrency batch operations |
| `stream(input, config)` | Stream output as it's produced | `Iterator[Output]` | Token-by-token display, real-time UX |
| `astream(input, config)` | Stream output asynchronously | `AsyncIterator[Output]` | Async streaming, SSE/WebSocket responses |

**Source**: `libs/core/langchain_core/runnables/base.py:122-145`

### Schema Introspection

Every Runnable exposes its input and output schemas for validation and documentation:

```python
chain.input_schema  # Pydantic model describing expected input structure
chain.output_schema  # Pydantic model describing output structure
chain.config_schema()  # Description of configurable fields
```

---

## Key Runnable Methods

### invoke(input, config=None) → Output

Synchronously transform a single input into an output.

```python
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: x * 2)
result = runnable.invoke(5)  # 10
```

**When to use**: Synchronous scripts, blocking operations, simple workflows.

**Source**: `libs/core/langchain_core/runnables/base.py:2600-2750`

---

### ainvoke(input, config=None) → Awaitable[Output]

Asynchronously transform a single input into an output. If the Runnable doesn't have a native async implementation, it executes the sync version using `asyncio.to_thread()` or a thread pool.

```python
import asyncio
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: x * 2)
result = await runnable.ainvoke(5)  # 10
```

**When to use**: Async applications, concurrent operations, non-blocking I/O contexts.

**Source**: `libs/core/langchain_core/runnables/base.py:2750-2900`

---

### batch(inputs, config=None, *, max_concurrency=None) → list[Output]

Transform multiple inputs in parallel using a thread pool executor. Significantly faster than sequential invokes for I/O-bound operations.

```python
inputs = [1, 2, 3, 4, 5]
results = runnable.batch(inputs)  # [2, 4, 6, 8, 10]
```

**Optimization**: By default, uses `ThreadPoolExecutor` for parallel execution. Control concurrency with `max_concurrency`:

```python
results = runnable.batch(inputs, config={'max_concurrency': 2})  # Max 2 parallel threads
```

**When to use**: Bulk data processing, throughput-critical operations, I/O-bound tasks.

**Source**: `libs/core/langchain_core/runnables/base.py:3200-3350`

---

### abatch(inputs, config=None, *, max_concurrency=None) → Awaitable[list[Output]]

Asynchronously transform multiple inputs with concurrent execution.

```python
results = await runnable.abatch([1, 2, 3, 4, 5])  # [2, 4, 6, 8, 10]
```

**When to use**: High-throughput async applications, concurrent API calls, async event loops.

**Source**: `libs/core/langchain_core/runnables/base.py:3350-3500`

---

### stream(input, config=None) → Iterator[Output]

Stream output as it's produced, enabling token-by-token display for language models or incremental processing.

```python
for chunk in runnable.stream(input):
    print(chunk, end='', flush=True)  # Display tokens as they arrive
```

**When to use**: Real-time UI updates, progressive output display, streaming responses.

**Source**: `libs/core/langchain_core/runnables/base.py:3700-3850`

---

### astream(input, config=None) → AsyncIterator[Output]

Asynchronously stream output as it's produced.

```python
async for chunk in runnable.astream(input):
    print(chunk, end='', flush=True)
```

**When to use**: Async streaming APIs, SSE/WebSocket endpoints, async UI frameworks.

**Source**: `libs/core/langchain_core/runnables/base.py:3850-4000`

---

## Type-Safe Chain Composition

### The Pipe Operator: `|`

LCEL uses Python's `|` operator (bitwise OR) to compose Runnables sequentially:

```python
chain = runnable1 | runnable2 | runnable3
```

This creates a `RunnableSequence` where:
1. `runnable1` receives the input
2. `runnable2` receives the output of `runnable1`
3. `runnable3` receives the output of `runnable2`
4. The final output of `runnable3` is returned

### Type Flow Verification

The type system ensures that connected Runnables have compatible types:

```python
# Type flow: int → str → int → float
chain = (
    RunnableLambda(lambda x: str(x))       # int → str
    | RunnableLambda(lambda x: len(x))     # str → int
    | RunnableLambda(lambda x: float(x))   # int → float
)
result = chain.invoke(42)  # result: float = 2.0
```

**Type Safety**: If you try to compose incompatible Runnables, type checkers like `mypy` will raise errors:

```python
# Type error: int is not compatible with str
bad_chain = (
    RunnableLambda(lambda x: int(x))   # str → int
    | RunnableLambda(lambda x: x.upper())  # ❌ int has no .upper() method
)
```

### Real-World Type Flow Example

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Type flow visualization:
# Dict[str, str] → ChatPromptTemplate → List[BaseMessage]
#   → ChatOpenAI → AIMessage → StrOutputParser → str

prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

chain = prompt | model | parser
# chain: Runnable[Dict[str, str], str]

result = chain.invoke({"topic": "LCEL"})
# result: str (containing explanation of LCEL)
```

### Automatic Coercion with `coerce_to_runnable`

LCEL automatically converts compatible objects into Runnables:

- **Functions/Lambdas** → `RunnableLambda`
- **Dict literals** → `RunnableParallel`
- **Prompt templates** → `RunnableBinding`
- **Language models** → Already Runnables

```python
# These are equivalent:
chain1 = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
chain2 = (lambda x: x + 1) | (lambda x: x * 2)  # Auto-converted to RunnableLambda
```

**Source**: `libs/core/langchain_core/runnables/base.py:161-167, 4900-5100`

---

## RunnableConfig

`RunnableConfig` is a TypedDict that controls execution behavior, tracing, callbacks, and runtime configuration for all Runnable methods.

### Configuration Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `callbacks` | `list[BaseCallbackHandler]` \| `CallbackManager` \| `None` | Callbacks for tracing and observability | `None` |
| `tags` | `list[str]` | Tags for filtering and organizing runs | `[]` |
| `metadata` | `dict[str, Any]` | JSON-serializable metadata for tracing | `{}` |
| `run_name` | `str` | Human-readable name for the tracer run | Class name |
| `run_id` | `UUID` \| `None` | Unique identifier for distributed tracing | Auto-generated |
| `max_concurrency` | `int` \| `None` | Max parallel calls in batch operations | ThreadPoolExecutor default |
| `recursion_limit` | `int` | Max recursion depth to prevent infinite loops | `25` |
| `configurable` | `dict[str, Any]` | Runtime values for configurable fields | `{}` |

**Source**: `libs/core/langchain_core/runnables/config.py:49-99`

### callbacks: Observability and Tracing

Callbacks enable logging, monitoring, and custom event handling:

```python
from langchain_core.tracers import ConsoleCallbackHandler

chain.invoke(
    {"topic": "AI"},
    config={
        "callbacks": [ConsoleCallbackHandler()],  # Prints execution trace to console
    }
)
```

**Use cases**:
- Logging all LLM calls
- Measuring latency and token usage
- Sending events to observability platforms (LangSmith, Datadog, etc.)
- Debugging chain execution

---

### tags: Filtering and Organization

Tags allow you to categorize and filter runs:

```python
chain.invoke(
    input,
    config={
        "tags": ["production", "user-query", "high-priority"],
    }
)
```

**Use cases**:
- Filtering runs in tracing platforms
- A/B testing different chains
- Cost attribution by team/project
- Rate limiting by category

---

### metadata: Custom Tracing Data

Metadata passes custom JSON-serializable data to callbacks:

```python
chain.invoke(
    input,
    config={
        "metadata": {
            "user_id": "user_12345",
            "session_id": "sess_abc",
            "experiment": "version_2.1",
        },
    }
)
```

**Use cases**:
- User attribution
- Session tracking
- Experiment tracking
- Custom analytics

---

### max_concurrency: Controlling Parallelism

Limit the number of parallel executions in batch operations:

```python
results = chain.batch(
    inputs,
    config={"max_concurrency": 5},  # Max 5 concurrent threads
)
```

**Use cases**:
- Rate limiting API calls
- Preventing resource exhaustion
- Respecting external API quotas
- Controlling memory usage

---

### recursion_limit: Preventing Infinite Loops

Protects against infinite recursion in chains that call themselves:

```python
config = {"recursion_limit": 50}  # Allow up to 50 recursive calls
```

**Default**: 25 levels of recursion

**Use cases**:
- Agents with tool-calling loops
- Self-correcting chains
- Recursive document processing

---

### configurable: Runtime Configuration

Dynamically override Runnable attributes at runtime using `configurable_fields`:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo").configurable_fields(
    model=ConfigurableField(id="model_name")
)

# Override model at runtime
response = model.invoke(
    "Hello",
    config={"configurable": {"model_name": "gpt-4"}}  # Uses GPT-4 instead
)
```

**Use cases**:
- A/B testing different models
- User-selectable options
- Environment-specific configuration
- Dynamic routing

**Source**: `libs/core/langchain_core/runnables/config.py:86-92`

---

### run_name and run_id: Tracing Identifiers

```python
import uuid

config = {
    "run_name": "customer_support_query",  # Descriptive name in traces
    "run_id": uuid.uuid4(),  # Unique ID for correlation
}
```

**Use cases**:
- Debugging specific runs
- Correlating parent-child relationships
- Distributed tracing across services

---

## Chain Composition Patterns

### 1. Simple Sequential Composition

Chain operations that execute one after another:

```python
from langchain_core.runnables import RunnableLambda

# Type flow: int → int+1 → (int+1)*2
add_one = RunnableLambda(lambda x: x + 1)
multiply_two = RunnableLambda(lambda x: x * 2)

chain = add_one | multiply_two
result = chain.invoke(5)  # (5 + 1) * 2 = 12
```

**Source**: `libs/core/langchain_core/runnables/base.py:174-178`

---

### 2. Parallel Composition (RunnableParallel)

Execute multiple Runnables concurrently on the same input:

```python
# Type flow: int → {add: int+10, multiply: int*5}
chain = RunnableLambda(lambda x: x + 1) | {
    'add': RunnableLambda(lambda x: x + 10),
    'multiply': RunnableLambda(lambda x: x * 5)
}

result = chain.invoke(5)
# Output: {'add': 16, 'multiply': 30}
```

**Explanation**:
1. First step: `5 + 1 = 6`
2. Parallel execution:
   - `add`: `6 + 10 = 16`
   - `multiply`: `6 * 5 = 30`

**Use cases**: Feature extraction, multi-model consensus, parallel API calls

**Source**: `libs/core/langchain_core/runnables/base.py:180-186`

---

### 3. Conditional Branching (RunnableBranch)

Route input to different Runnables based on conditions:

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: x > 10, lambda x: f'Large: {x}'),      # Condition 1
    (lambda x: x > 5, lambda x: f'Medium: {x}'),      # Condition 2
    lambda x: f'Small: {x}'                           # Default
)

result1 = branch.invoke(15)  # "Large: 15"
result2 = branch.invoke(7)   # "Medium: 7"
result3 = branch.invoke(2)   # "Small: 2"
```

**Behavior**: Evaluates conditions sequentially and runs the first matching Runnable. If no condition matches, runs the default.

**Use cases**: Dynamic prompt selection, model routing, conditional logic

**Source**: `libs/core/langchain_core/runnables/branch.py:40-65`

---

### 4. Error Handling (RunnableWithFallbacks)

Provide fallback Runnables if the primary Runnable fails:

```python
from langchain_core.runnables import RunnableLambda

# Primary operation that might fail
primary = RunnableLambda(lambda x: 1 / x)  # Fails when x=0

# Fallback operation
fallback = RunnableLambda(lambda x: 'Error: division by zero')

# Create safe chain
safe_chain = primary.with_fallbacks([fallback])

result1 = safe_chain.invoke(5)  # 0.2 (primary succeeds)
result2 = safe_chain.invoke(0)  # "Error: division by zero" (fallback used)
```

**Behavior**: Tries the primary Runnable first. On exception, tries each fallback in order until one succeeds.

**Use cases**:
- Multi-provider LLM fallbacks
- Graceful degradation
- Cache miss fallbacks
- API timeout recovery

**Source**: `libs/core/langchain_core/runnables/fallbacks.py:37-86`

---

### 5. Retry Logic (RunnableRetry)

Automatically retry failed operations with exponential backoff:

```python
import random

def unstable_operation(x):
    if random.random() < 0.7:  # 70% failure rate
        raise ValueError('Transient failure')
    return x * 2

reliable_chain = RunnableLambda(unstable_operation).with_retry(
    retry_if_exception_type=(ValueError,),
    stop_after_attempt=5,
    wait_exponential_jitter=True,
)

result = reliable_chain.invoke(10)  # Retries on ValueError, eventually succeeds
```

**Retry strategies**:
- **stop_after_attempt**: Max number of retries
- **wait_exponential_jitter**: Exponential backoff with jitter (prevents thundering herd)
- **retry_if_exception_type**: Only retry specific exceptions

**Use cases**:
- API rate limiting (retry on 429)
- Transient network failures
- LLM provider throttling
- Database connection timeouts

**Source**: `libs/core/langchain_core/runnables/retry.py:48-112`

---

### 6. Dynamic Routing (RouterRunnable)

Route to different Runnables based on input data:

```python
from langchain_core.runnables.router import RouterRunnable

router = RouterRunnable({
    'add': RunnableLambda(lambda x: x['value'] + 1),
    'multiply': RunnableLambda(lambda x: x['value'] * 2),
    'divide': RunnableLambda(lambda x: x['value'] / 2),
})

result1 = router.invoke({'key': 'multiply', 'value': 5})  # 10
result2 = router.invoke({'key': 'divide', 'value': 20})   # 10.0
```

**Behavior**: Extracts routing key from input and executes the corresponding Runnable.

**Use cases**:
- Multi-agent systems
- Intent-based routing
- Dynamic tool selection
- Conditional chain execution

**Source**: `libs/core/langchain_core/runnables/router.py:46-62`

---

### 7. Data Manipulation (RunnablePassthrough, RunnableAssign)

Preserve input while adding computed fields:

```python
from langchain_core.runnables import RunnablePassthrough

# Start with input dict, add computed fields
chain = RunnablePassthrough.assign(
    doubled=lambda x: x['value'] * 2,
    tripled=lambda x: x['value'] * 3,
    squared=lambda x: x['value'] ** 2,
)

result = chain.invoke({'value': 5})
# Output: {
#     'value': 5,          # Original preserved
#     'doubled': 10,       # Computed field
#     'tripled': 15,       # Computed field
#     'squared': 25,       # Computed field
# }
```

**Behavior**: `RunnablePassthrough.assign()` merges the original input with new computed fields.

**Use cases**:
- Feature engineering
- Context enrichment
- Multi-step transformations
- Preserving original input for debugging

**Source**: `libs/core/langchain_core/runnables/passthrough.py:74-148`

---

### 8. Picking Fields (RunnablePick)

Extract specific fields from a dict output:

```python
from langchain_core.runnables import RunnablePick

chain = {
    'name': lambda x: 'Alice',
    'age': lambda x: 30,
    'city': lambda x: 'NYC',
} | RunnablePick('name')  # Extract only 'name'

result = chain.invoke({})  # "Alice"
```

**Use cases**: Field extraction, output filtering, API response transformation

---

## Practical LCEL Example

Here's a complete, real-world example demonstrating LCEL composition with a language model:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Step 1: Create prompt template
# Type: Runnable[Dict[str, str], List[BaseMessage]]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Tell me a joke about {topic}"),
])

# Step 2: Create language model
# Type: Runnable[List[BaseMessage], AIMessage]
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Step 3: Create output parser
# Type: Runnable[AIMessage, str]
parser = StrOutputParser()

# Step 4: Compose into chain using LCEL
# Type flow:
#   Dict[str, str] → ChatPromptTemplate → List[BaseMessage]
#     → ChatOpenAI → AIMessage → StrOutputParser → str
chain = prompt | model | parser

# Execute the chain
result = chain.invoke({"topic": "programming"})
# result: str containing a programming joke

# All these work automatically:
await chain.ainvoke({"topic": "AI"})                    # Async
chain.batch([{"topic": "robots"}, {"topic": "ML"}])    # Batch
for token in chain.stream({"topic": "Python"}):         # Streaming
    print(token, end='', flush=True)
```

### Why This Works

1. **Type Safety**: Each step's output type matches the next step's input type
2. **Automatic Methods**: `ainvoke`, `batch`, `stream` work without additional code
3. **Traceability**: Every step is automatically traced with callbacks
4. **Composability**: Can extend this chain further: `chain | another_step`

---

## Configuration Examples

### Example 1: Adding Tracing Callbacks

```python
from langchain_core.tracers import ConsoleCallbackHandler

chain.invoke(
    {"topic": "AI"},
    config={
        "callbacks": [ConsoleCallbackHandler()],
        "tags": ["production", "joke-generator"],
        "metadata": {"user_id": "12345", "session": "abc"},
    }
)
```

**Output**: Console logs showing each step of execution with timing information.

---

### Example 2: Controlling Batch Concurrency

```python
inputs = [
    {"topic": "AI"},
    {"topic": "robots"},
    {"topic": "coding"},
    {"topic": "algorithms"},
    {"topic": "databases"},
]

results = chain.batch(
    inputs,
    config={"max_concurrency": 2},  # Process 2 at a time
)
```

**Benefit**: Prevents overwhelming APIs with too many concurrent requests.

---

### Example 3: Dynamic Model Selection

```python
from langchain_core.runnables import ConfigurableField

model = ChatOpenAI(model="gpt-3.5-turbo").configurable_fields(
    model=ConfigurableField(id="model_name"),
    temperature=ConfigurableField(id="temp"),
)

chain = prompt | model | parser

# Use GPT-4 with higher creativity
result = chain.invoke(
    {"topic": "philosophy"},
    config={
        "configurable": {
            "model_name": "gpt-4",
            "temp": 0.9,
        }
    }
)
```

---

## File Reference

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| **base.py** | Core Runnable protocol and LCEL composition | `Runnable`, `RunnableSequence`, `RunnableParallel`, `RunnableLambda`, `RunnableGenerator` |
| **config.py** | Configuration system for execution control | `RunnableConfig`, `ensure_config`, `merge_configs`, `patch_config`, `get_callback_manager_for_config` |
| **schema.py** | Streaming event types and data models | `EventData`, `StreamEvent`, `StandardStreamEvent` |
| **branch.py** | Conditional routing based on predicates | `RunnableBranch` |
| **fallbacks.py** | Error recovery with fallback chains | `RunnableWithFallbacks` |
| **retry.py** | Retry logic with exponential backoff | `RunnableRetry`, retry configuration |
| **router.py** | Dynamic routing based on input data | `RouterRunnable` |
| **passthrough.py** | Data manipulation and field preservation | `RunnablePassthrough`, `RunnableAssign`, `RunnablePick` |
| **history.py** | Message history integration | `RunnableWithMessageHistory` |
| **graph.py** | Graph representation of chains | `Graph`, `Node`, `Edge` |

---

## Source Citations

- **Runnable class docstring**: `libs/core/langchain_core/runnables/base.py:122-254`
- **RunnableConfig definition**: `libs/core/langchain_core/runnables/config.py:49-99`
- **RunnableBranch example**: `libs/core/langchain_core/runnables/branch.py:40-65`
- **RunnableWithFallbacks example**: `libs/core/langchain_core/runnables/fallbacks.py:37-86`
- **RunnableRetry documentation**: `libs/core/langchain_core/runnables/retry.py:48-112`
- **RouterRunnable example**: `libs/core/langchain_core/runnables/router.py:46-62`
- **RunnablePassthrough examples**: `libs/core/langchain_core/runnables/passthrough.py:74-148`

---

## Additional Resources

- **Official LangChain Documentation**: [https://python.langchain.com/docs/expression_language/](https://python.langchain.com/docs/expression_language/)
- **API Reference**: [https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.runnables](https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.runnables)
- **LangChain Expression Language Guide**: [https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel](https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel)
- **LangSmith Tracing**: [https://docs.smith.langchain.com/](https://docs.smith.langchain.com/)

---

## Quick Start

```python
# Install LangChain Core
pip install langchain-core

# Basic LCEL chain
from langchain_core.runnables import RunnableLambda

chain = (
    RunnableLambda(lambda x: x.upper())
    | RunnableLambda(lambda x: f"Hello, {x}!")
)

print(chain.invoke("world"))  # "Hello, WORLD!"
```

For more examples, see the [examples/](../../examples/) directory in the repository.
