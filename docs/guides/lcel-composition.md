# LCEL Composition Guide

This comprehensive guide covers the LangChain Expression Language (LCEL), a declarative syntax for composing type-safe chains using the pipe operator (`|`). LCEL enables you to build complex workflows with automatic support for synchronous, asynchronous, batch, and streaming execution.

**Target Audience**: Developers building LangChain applications who want to understand type-safe composition patterns and master LCEL chain construction.

**Prerequisites**: 
- Basic understanding of Python type hints
- Familiarity with LangChain core concepts (see [Glossary](../glossary.md))
- Python 3.10 or higher

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [The Pipe Operator](#the-pipe-operator)
4. [Type Flow Documentation](#type-flow-documentation)
5. [Simple Composition Patterns](#simple-composition-patterns)
6. [RunnableSequence](#runnablesequence)
7. [RunnableParallel](#runnableparallel)
8. [Advanced Composition Patterns](#advanced-composition-patterns)
9. [Type Annotations Best Practices](#type-annotations-best-practices)
10. [Schema Inspection](#schema-inspection)
11. [Debugging Compositions](#debugging-compositions)
12. [Practical Examples](#practical-examples)
13. [Common Patterns](#common-patterns)
14. [Configuration](#configuration)
15. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is LCEL?

LCEL (LangChain Expression Language) is a declarative way to compose `Runnable` objects into chains. Any chain constructed using LCEL automatically has full support for:

- **Sync execution**: `invoke()` and `batch()`
- **Async execution**: `ainvoke()` and `abatch()`
- **Streaming**: `stream()` and `astream()`
- **Parallel execution**: Automatic parallelization in batch operations
- **Tracing and observability**: Built-in callback integration

**Source**: `libs/core/langchain_core/runnables/base.py:150-156`

### Why Use LCEL?

LCEL provides several key advantages over legacy chain classes:

1. **Type Safety**: Compile-time type checking ensures output type of one runnable matches input type of the next
2. **Explicit Type Flow**: Type transformations are visible and documentable at each composition stage
3. **Automatic Methods**: No need to implement sync/async/batch/stream variants separately
4. **Composability**: Mix and match any runnables without adapter code
5. **Debugging**: Built-in introspection tools (`get_graph()`) for visualizing chain structure

### LCEL vs Legacy Chains

```python
# Legacy approach (LLMChain - deprecated)
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = LLMChain(llm=model, prompt=prompt)
result = chain.invoke({"topic": "programming"})

# Modern LCEL approach (recommended)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | model | StrOutputParser()
result = chain.invoke({"topic": "programming"})
```

**Key Difference**: LCEL makes the data flow explicit with the pipe operator, showing exactly how data transforms at each stage.

---

## Core Concepts

### The Runnable Protocol

At the heart of LCEL is the `Runnable` protocol, which defines a standard interface for components that can process inputs and produce outputs.

**Runnable Generic Type Parameters**:
- `Input`: The type of input the runnable accepts
- `Output`: The type of output the runnable produces

**Source**: `libs/core/langchain_core/runnables/base.py:298-350`

**Core Runnable Methods**:

```python
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")

class Runnable(Generic[Input, Output]):
    """Base protocol for LCEL components."""
    
    def invoke(self, input: Input, config: RunnableConfig | None = None) -> Output:
        """Transform a single input into an output."""
        ...
    
    async def ainvoke(self, input: Input, config: RunnableConfig | None = None) -> Output:
        """Async version of invoke."""
        ...
    
    def batch(self, inputs: list[Input], config: RunnableConfig | None = None) -> list[Output]:
        """Transform multiple inputs into outputs."""
        ...
    
    async def abatch(self, inputs: list[Input], config: RunnableConfig | None = None) -> list[Output]:
        """Async version of batch."""
        ...
    
    def stream(self, input: Input, config: RunnableConfig | None = None) -> Iterator[Output]:
        """Stream output as it's generated."""
        ...
    
    async def astream(self, input: Input, config: RunnableConfig | None = None) -> AsyncIterator[Output]:
        """Async version of stream."""
        ...
```

### Universal Methods

Every component in an LCEL chain automatically supports all execution modes:

| Method | Description | Use Case |
|--------|-------------|----------|
| `invoke(input)` | Process single input synchronously | Single requests, sequential processing |
| `ainvoke(input)` | Process single input asynchronously | Async contexts, concurrent operations |
| `batch(inputs)` | Process multiple inputs in parallel | Bulk processing, batch jobs |
| `abatch(inputs)` | Async batch processing | High-concurrency batch operations |
| `stream(input)` | Stream output incrementally | Real-time responses, token-by-token output |
| `astream(input)` | Async streaming | Async real-time processing |

---

## The Pipe Operator

### How the Pipe Operator Works

The pipe operator (`|`) is implemented via Python's `__or__` method, which creates a `RunnableSequence` that connects two runnables.

**Source**: `libs/core/langchain_core/runnables/base.py:608-627`

```python
def __or__(
    self,
    other: Runnable[Any, Other]
    | Callable[[Iterator[Any]], Iterator[Other]]
    | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
    | Callable[[Any], Other]
    | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
) -> RunnableSerializable[Input, Other]:
    """Runnable 'or' operator.
    
    Compose this Runnable with another object to create a RunnableSequence.
    
    Args:
        other: Another Runnable or a Runnable-like object.
    
    Returns:
        A new Runnable.
    """
    return RunnableSequence(self, coerce_to_runnable(other))
```

### Type Transformation with Pipe Operator

When you compose two runnables with `|`, the types flow as follows:

```python
# Type notation: Runnable[InputType, OutputType]

runnable_a: Runnable[A, B]  # Takes A, produces B
runnable_b: Runnable[B, C]  # Takes B, produces C

# Composition
chain = runnable_a | runnable_b  # Type: Runnable[A, C]

# The output type B of runnable_a must match the input type B of runnable_b
```

**Type Safety**: If types don't align, you'll get type errors with mypy:

```python
from langchain_core.runnables import RunnableLambda

# These types are compatible
add_one: Runnable[int, int] = RunnableLambda(lambda x: x + 1)
multiply_two: Runnable[int, int] = RunnableLambda(lambda x: x * 2)
chain = add_one | multiply_two  # ✓ Works: int → int → int

# These types are incompatible
to_string: Runnable[int, str] = RunnableLambda(lambda x: str(x))
multiply_two: Runnable[int, int] = RunnableLambda(lambda x: x * 2)
chain = to_string | multiply_two  # ✗ Type error: str → int (incompatible)
```

### Equivalent Expressions

The following expressions are equivalent:

```python
from langchain_core.runnables import RunnableSequence

# Using pipe operator (recommended for readability)
chain = runnable_a | runnable_b | runnable_c

# Using pipe() method
chain = runnable_a.pipe(runnable_b, runnable_c)

# Using RunnableSequence constructor
chain = RunnableSequence(runnable_a, runnable_b, runnable_c)
```

**Best Practice**: Use the pipe operator (`|`) for maximum readability and clarity of data flow.

---

## Type Flow Documentation

### Understanding Type Transformations

One of LCEL's key benefits is making type transformations explicit and traceable. Here's how to document type flow through a chain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Define each component with type annotations
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
# Type: Runnable[Dict[str, Any], List[BaseMessage]]

model = ChatOpenAI(model="gpt-4")
# Type: Runnable[List[BaseMessage], AIMessage]

parser = StrOutputParser()
# Type: Runnable[AIMessage, str]

# Compose with inline type flow comments
chain = (
    prompt      # Dict[str, Any] → List[BaseMessage]
    | model     # List[BaseMessage] → AIMessage
    | parser    # AIMessage → str
)
# Final chain type: Runnable[Dict[str, Any], str]

# Invoke with type-checked input
result: str = chain.invoke({"topic": "Python"})
```

### Type Flow Visualization

Here's a complete example showing type transformations at each stage:

```mermaid
graph LR
    A[Dict[str, str]] -->|PromptTemplate| B[List[BaseMessage]]
    B -->|ChatModel| C[AIMessage]
    C -->|StrOutputParser| D[str]
    
    style A fill:#e1f5ff
    style B fill:#fff3e1
    style C fill:#e8f5e9
    style D fill:#f3e5f5
```

**Real-World Type Flow Example**:

```python
# Input: {"question": "What is LCEL?", "context": "..."}
# ↓ Type: Dict[str, str]

# PromptTemplate formats the input into messages
# ↓ Type: List[BaseMessage]
# Example: [SystemMessage("..."), HumanMessage("What is LCEL?")]

# ChatModel processes messages and generates response
# ↓ Type: AIMessage
# Example: AIMessage(content="LCEL is a declarative syntax...")

# StrOutputParser extracts string content
# ↓ Type: str
# Final output: "LCEL is a declarative syntax..."
```

**Source**: Type flow pattern documented in `libs/core/langchain_core/runnables/base.py:150-186`

---

## Simple Composition Patterns

### Pattern 1: Prompt → Model

The most basic LCEL chain connects a prompt template directly to a model:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI(model="gpt-4")

chain = prompt | model

# Invoke the chain
result = chain.invoke({"topic": "programming"})
# Output type: AIMessage
print(result.content)  # Access the message content
```

**Type Flow**: `Dict[str, Any] → List[BaseMessage] → AIMessage`

### Pattern 2: Prompt → Model → Parser

Add an output parser to extract string content from the AI message:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

chain = prompt | model | parser

# Invoke the chain
result = chain.invoke({"topic": "programming"})
# Output type: str
print(result)  # Direct string output
```

**Type Flow**: `Dict[str, Any] → List[BaseMessage] → AIMessage → str`

**When to Use**: Use this pattern when you need simple string output for display or further processing.

### Pattern 3: Prompt → Model → Structured Parser

Parse LLM output into typed Python objects using Pydantic:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Define output structure
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")
    rating: int = Field(description="Humor rating from 1-10")

# Create parser and prompt
parser = PydanticOutputParser(pydantic_object=Joke)
prompt = ChatPromptTemplate.from_template(
    "Tell me a joke about {topic}\n\n{format_instructions}"
)

model = ChatOpenAI(model="gpt-4")

chain = prompt | model | parser

# Invoke with format instructions
result = chain.invoke({
    "topic": "programming",
    "format_instructions": parser.get_format_instructions()
})
# Output type: Joke
print(f"Setup: {result.setup}")
print(f"Punchline: {result.punchline}")
print(f"Rating: {result.rating}/10")
```

**Type Flow**: `Dict[str, Any] → List[BaseMessage] → AIMessage → Joke`

**When to Use**: Use structured parsing when you need type-safe, validated output for downstream processing.

---

## RunnableSequence

### What is RunnableSequence?

`RunnableSequence` is the concrete class created when you compose runnables with the pipe operator. It executes runnables sequentially, passing the output of each step as input to the next.

**Source**: `libs/core/langchain_core/runnables/base.py:160-163`

### Sequential Execution Flow

```python
from langchain_core.runnables import RunnableLambda

# Define individual steps
step1 = RunnableLambda(lambda x: x + 1)       # Add 1
step2 = RunnableLambda(lambda x: x * 2)       # Multiply by 2
step3 = RunnableLambda(lambda x: f"Result: {x}")  # Format as string

# Create sequence
sequence = step1 | step2 | step3

# Execution flow
input_value = 5
# Step 1: 5 → 6
# Step 2: 6 → 12
# Step 3: 12 → "Result: 12"
result = sequence.invoke(input_value)
print(result)  # "Result: 12"
```

### Output-to-Input Mapping

When a step produces a dictionary output and the next step expects specific keys, LCEL automatically handles the mapping:

```python
from langchain_core.runnables import RunnableLambda

# Step 1: Produces dict with multiple keys
step1 = RunnableLambda(lambda x: {
    "value": x * 2,
    "squared": x ** 2,
    "original": x
})

# Step 2: Expects dict with "value" key
step2 = RunnableLambda(lambda d: d["value"] + 10)

# Compose
chain = step1 | step2

result = chain.invoke(5)
# Step 1 output: {"value": 10, "squared": 25, "original": 5}
# Step 2 input: Receives the full dict, accesses d["value"] = 10
# Step 2 output: 20
print(result)  # 20
```

### Multi-Output to Multi-Input

When you need to pass specific outputs to specific inputs, use `RunnableParallel` (covered next) or itemgetter:

```python
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Create multi-output step
analyze = RunnableLambda(lambda x: {
    "summary": f"Summary of {x}",
    "length": len(x),
    "first_char": x[0] if x else ""
})

# Select specific output for next step
extract_summary = itemgetter("summary")

# Compose
chain = analyze | extract_summary | RunnableLambda(lambda s: s.upper())

result = chain.invoke("hello world")
# Step 1: {"summary": "Summary of hello world", "length": 11, "first_char": "h"}
# Step 2: "Summary of hello world"
# Step 3: "SUMMARY OF HELLO WORLD"
print(result)
```

**Source**: Sequential composition behavior in `libs/core/langchain_core/runnables/base.py:650-697`

---

## RunnableParallel

### What is RunnableParallel?

`RunnableParallel` invokes multiple runnables concurrently, providing the same input to each. It's useful for executing independent operations in parallel and collecting their results.

**Source**: `libs/core/langchain_core/runnables/base.py:164-166`

### Creating RunnableParallel with Dict Literal

The most common way to create parallel execution is using a dictionary literal within a sequence:

```python
from langchain_core.runnables import RunnableLambda

# Define parallel operations
analyze_length = RunnableLambda(lambda x: len(x))
analyze_words = RunnableLambda(lambda x: len(x.split()))
analyze_uppercase = RunnableLambda(lambda x: sum(1 for c in x if c.isupper()))

# Create parallel execution using dict literal
chain = {
    "length": analyze_length,
    "word_count": analyze_words,
    "uppercase_count": analyze_uppercase,
}

result = chain.invoke("Hello World from LCEL")
# Output: {
#     "length": 23,
#     "word_count": 4,
#     "uppercase_count": 6
# }
print(result)
```

### Parallel Within Sequential Chains

Combine parallel and sequential execution for complex workflows:

```python
from langchain_core.runnables import RunnableLambda

# Step 1: Preprocess (sequential)
preprocess = RunnableLambda(lambda x: x.strip().lower())

# Step 2: Parallel analysis
analyze = {
    "char_count": RunnableLambda(lambda x: len(x)),
    "word_count": RunnableLambda(lambda x: len(x.split())),
    "vowel_count": RunnableLambda(lambda x: sum(1 for c in x if c in "aeiou"))
}

# Step 3: Combine results (sequential)
summarize = RunnableLambda(lambda d: 
    f"Text has {d['char_count']} chars, {d['word_count']} words, {d['vowel_count']} vowels"
)

# Complete chain
chain = preprocess | analyze | summarize

result = chain.invoke("  Hello LCEL World  ")
# Step 1: "hello lcel world"
# Step 2: {"char_count": 16, "word_count": 3, "vowel_count": 4}
# Step 3: "Text has 16 chars, 3 words, 4 vowels"
print(result)
```

### RunnableParallel Constructor

You can also use the `RunnableParallel` constructor explicitly:

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

parallel = RunnableParallel(
    length=RunnableLambda(lambda x: len(x)),
    uppercase=RunnableLambda(lambda x: x.upper()),
    word_count=RunnableLambda(lambda x: len(x.split()))
)

result = parallel.invoke("hello world")
# Output: {"length": 11, "uppercase": "HELLO WORLD", "word_count": 2}
```

**Best Practice**: Use dict literal syntax for inline parallel execution within chains. Use `RunnableParallel` constructor when you need to name or reuse the parallel component.

### Parallel Execution Benefits

```python
import time
from langchain_core.runnables import RunnableLambda

def slow_operation_1(x):
    time.sleep(1)  # Simulates slow I/O
    return x * 2

def slow_operation_2(x):
    time.sleep(1)  # Simulates slow I/O
    return x + 10

# Sequential execution (2 seconds total)
sequential = (
    RunnableLambda(slow_operation_1) 
    | RunnableLambda(slow_operation_2)
)

# Parallel execution (~1 second total)
parallel = {
    "doubled": RunnableLambda(slow_operation_1),
    "added": RunnableLambda(slow_operation_2)
}
```

**Performance Note**: Parallel execution is especially beneficial for I/O-bound operations like API calls, database queries, or file operations.

---

## Advanced Composition Patterns

### RunnableBranch: Conditional Routing

Route inputs to different runnables based on conditions:

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Define condition functions
def is_long(x: str) -> bool:
    return len(x) > 20

def is_medium(x: str) -> bool:
    return 10 < len(x) <= 20

# Define handlers for each condition
handle_long = RunnableLambda(lambda x: f"Long text: {x[:20]}...")
handle_medium = RunnableLambda(lambda x: f"Medium text: {x}")
handle_short = RunnableLambda(lambda x: f"Short text: {x}")

# Create branch
branch = RunnableBranch(
    (is_long, handle_long),
    (is_medium, handle_medium),
    handle_short  # Default handler (no condition)
)

# Test different inputs
print(branch.invoke("Hi"))  # "Short text: Hi"
print(branch.invoke("Hello world test"))  # "Medium text: Hello world test"
print(branch.invoke("This is a very long text that exceeds twenty characters"))  
# "Long text: This is a very long..."
```

**Use Case**: Dynamic processing based on input characteristics, routing to specialized chains, implementing fallback logic.

### RunnablePassthrough: Preserving Inputs

Pass inputs through unchanged while still participating in composition:

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Scenario: Add metadata to input without modifying it
chain = {
    "original": RunnablePassthrough(),  # Pass input through unchanged
    "length": RunnableLambda(lambda x: len(x)),
    "uppercase": RunnableLambda(lambda x: x.upper())
}

result = chain.invoke("hello")
# Output: {
#     "original": "hello",
#     "length": 5,
#     "uppercase": "HELLO"
# }
```

**Use Case**: Maintaining original input for later stages, adding parallel analysis without losing original data, debugging chains by preserving intermediate values.

### RunnableWithFallbacks: Error Recovery

Provide fallback runnables for error handling:

```python
from langchain_core.runnables import RunnableLambda

def primary_operation(x: int) -> int:
    if x < 0:
        raise ValueError("Negative numbers not allowed")
    return x * 2

def fallback_operation(x: int) -> int:
    return abs(x) * 2  # Handle negative numbers

# Create runnable with fallback
primary = RunnableLambda(primary_operation)
fallback = RunnableLambda(fallback_operation)

chain = primary.with_fallbacks([fallback])

# Test with valid and invalid inputs
print(chain.invoke(5))   # 10 (primary succeeds)
print(chain.invoke(-3))  # 6 (primary fails, fallback handles it)
```

**Reference**: See [Error Handling Guide](error-handling.md) for comprehensive fallback strategies.

---

## Type Annotations Best Practices

### Annotating Custom Chains

When creating custom runnables, provide explicit type annotations for mypy compliance:

```python
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda, Runnable

# Explicitly typed runnable
def process_input(data: Dict[str, Any]) -> str:
    """Process input dict and return formatted string."""
    return f"Processed: {data.get('value', 'N/A')}"

# Create typed runnable
processor: Runnable[Dict[str, Any], str] = RunnableLambda(process_input)

# Compose with type safety
chain: Runnable[Dict[str, Any], str] = processor | RunnableLambda(str.upper)

# Type-checked invocation
result: str = chain.invoke({"value": "test"})
```

### Using Generic Type Variables

For reusable components, use type variables:

```python
from typing import TypeVar, Generic, List
from langchain_core.runnables import Runnable, RunnableLambda

T = TypeVar('T')

def create_list_wrapper(runnable: Runnable[T, T]) -> Runnable[List[T], List[T]]:
    """Wrap a runnable to operate on lists."""
    return RunnableLambda(lambda items: [runnable.invoke(item) for item in items])

# Usage
single_processor: Runnable[str, str] = RunnableLambda(str.upper)
list_processor: Runnable[List[str], List[str]] = create_list_wrapper(single_processor)

result = list_processor.invoke(["hello", "world"])
# Output: ["HELLO", "WORLD"]
```

### Mypy Strict Mode Compliance

Ensure your chains pass mypy `--strict` mode:

```python
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

# Fully type-annotated chain factory
def create_qa_chain() -> Runnable[Dict[str, Any], str]:
    """Create a type-safe question-answering chain.
    
    Returns:
        A runnable that takes a dict with 'question' key and returns a string answer.
    """
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
        "Answer this question: {question}"
    )
    model: ChatOpenAI = ChatOpenAI(model="gpt-4")
    parser: StrOutputParser = StrOutputParser()
    
    chain: Runnable[Dict[str, Any], str] = prompt | model | parser
    return chain

# Type-checked usage
qa_chain = create_qa_chain()
answer: str = qa_chain.invoke({"question": "What is LCEL?"})
```

**Validation Command**:
```bash
mypy --strict your_module.py
```

---

## Schema Inspection

### Understanding Input and Output Schemas

Every runnable exposes `input_schema` and `output_schema` properties for runtime type inspection:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser

# Inspect input schema
print(chain.input_schema.model_json_schema())
# Output: {
#     "title": "PromptInput",
#     "type": "object",
#     "properties": {
#         "topic": {"title": "Topic", "type": "string"}
#     },
#     "required": ["topic"]
# }

# Inspect output schema
print(chain.output_schema.model_json_schema())
# Output: {
#     "title": "StrOutputParserOutput",
#     "type": "string"
# }
```

**Source**: Schema properties defined in `libs/core/langchain_core/runnables/base.py:225-227`

### Using Schemas for Validation

Leverage schemas for input validation before execution:

```python
from pydantic import ValidationError

def safe_invoke(chain, input_data):
    """Invoke chain with input validation."""
    try:
        # Validate input against schema
        validated_input = chain.input_schema(**input_data)
        # Invoke with validated data
        return chain.invoke(validated_input.model_dump())
    except ValidationError as e:
        print(f"Invalid input: {e}")
        return None

# Example usage
result = safe_invoke(chain, {"topic": "Python"})  # ✓ Valid
result = safe_invoke(chain, {"wrong_key": "value"})  # ✗ Validation error
```

### Schema-Driven Documentation

Use schemas to auto-generate API documentation:

```python
def document_chain(chain):
    """Generate documentation for a chain."""
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()
    
    print("Chain Documentation")
    print("=" * 50)
    print("\nInput Requirements:")
    for prop, details in input_schema.get("properties", {}).items():
        required = prop in input_schema.get("required", [])
        print(f"  - {prop}: {details.get('type')} {'(required)' if required else '(optional)'}")
    
    print("\nOutput Type:")
    print(f"  {output_schema.get('type', 'object')}")

document_chain(chain)
```

---

## Debugging Compositions

### Using get_graph() for Visualization

Inspect chain structure programmatically:

```python
from langchain_core.runnables import RunnableLambda

# Create multi-step chain
chain = (
    RunnableLambda(lambda x: x + 1)
    | RunnableLambda(lambda x: x * 2)
    | {
        "doubled": RunnableLambda(lambda x: x * 2),
        "stringified": RunnableLambda(lambda x: str(x))
    }
)

# Get graph representation
graph = chain.get_graph()

# Print nodes
for node_id, node in graph.nodes.items():
    print(f"Node {node_id}: {node.data.get_name()}")

# Print edges showing data flow
for edge in graph.edges:
    print(f"Edge: {edge.source} → {edge.target}")
```

**Output**:
```
Node 0: RunnableLambda
Node 1: RunnableLambda
Node 2: RunnableParallel<doubled,stringified>
Edge: 0 → 1
Edge: 1 → 2
```

### Verbose Mode with Callbacks

Enable detailed logging during execution:

```python
from langchain_core.tracers import ConsoleCallbackHandler

# Create chain with verbose callback
chain = prompt | model | parser

# Invoke with debugging enabled
result = chain.invoke(
    {"topic": "Python"},
    config={"callbacks": [ConsoleCallbackHandler()]}
)

# Console output shows:
# [chain/start] Entering chain
# [prompt/start] Formatting prompt
# [prompt/end] Prompt formatted
# [model/start] Calling model
# [model/end] Model response received
# [parser/start] Parsing output
# [parser/end] Output parsed
# [chain/end] Exiting chain
```

### Global Debug Mode

Enable debug mode globally for all chains:

```python
from langchain_core.globals import set_debug

# Enable global debugging
set_debug(True)

# All chain invocations now print debug information
result = chain.invoke({"topic": "Python"})

# Disable when done
set_debug(False)
```

**Source**: `libs/core/langchain_core/runnables/base.py:236-242`

### Debugging Common Issues

**Issue 1: Type Mismatch**

```python
# Problem: Output type doesn't match next input type
step1 = RunnableLambda(lambda x: str(x))  # Returns str
step2 = RunnableLambda(lambda x: x + 5)   # Expects int

chain = step1 | step2  # Runtime error: can't add int to str

# Solution: Add type conversion
fix_type = RunnableLambda(lambda x: int(x))
chain = step1 | fix_type | step2  # Now works correctly
```

**Issue 2: Missing Dictionary Keys**

```python
# Problem: Next step expects key that doesn't exist
step1 = RunnableLambda(lambda x: {"value": x})
step2 = RunnableLambda(lambda d: d["result"])  # KeyError: "result" doesn't exist

# Solution: Verify keys or use .get() with defaults
step2_fixed = RunnableLambda(lambda d: d.get("value", 0))
chain = step1 | step2_fixed
```

**Issue 3: Schema Validation Failures**

```python
# Problem: Input doesn't match prompt schema
prompt = ChatPromptTemplate.from_template("Question: {question}\nContext: {context}")
chain = prompt | model | parser

# This fails: missing "context" key
result = chain.invoke({"question": "What is LCEL?"})

# Solution: Provide all required keys
result = chain.invoke({
    "question": "What is LCEL?",
    "context": "LCEL is the LangChain Expression Language..."
})
```

### Introspection Tools

Create custom introspection utilities:

```python
def analyze_chain(chain):
    """Analyze chain structure and types."""
    print("Chain Analysis")
    print("=" * 50)
    print(f"Input Type: {chain.InputType}")
    print(f"Output Type: {chain.OutputType}")
    print(f"\nInput Schema:")
    print(chain.input_schema.model_json_schema())
    print(f"\nOutput Schema:")
    print(chain.output_schema.model_json_schema())
    print(f"\nGraph Structure:")
    graph = chain.get_graph()
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")

# Usage
analyze_chain(my_complex_chain)
```

**Reference**: See [Chain Introspection Example](../../examples/debugging/chain_introspection.py) for complete implementation.

---

## Practical Examples

### Example 1: Simple Question-Answering Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Build the chain
template = "You are a helpful assistant. Answer this question: {question}"
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4", temperature=0)
parser = StrOutputParser()

qa_chain = prompt | model | parser

# Use the chain
answer = qa_chain.invoke({"question": "What is the capital of France?"})
print(answer)  # "The capital of France is Paris."

# Type flow: Dict[str, Any] → List[BaseMessage] → AIMessage → str
```

**Complete Example**: See [examples/basic_chains/lcel_basic_composition.py](../../examples/basic_chains/lcel_basic_composition.py)

### Example 2: Multi-Step Reasoning Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Step 1: Generate reasoning steps
reasoning_prompt = ChatPromptTemplate.from_template(
    "Break down this problem into steps: {problem}"
)

# Step 2: Solve using steps
solving_prompt = ChatPromptTemplate.from_template(
    "Given these steps: {steps}\n\nSolve the original problem: {problem}"
)

model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# Build multi-step chain
reasoning_chain = (
    {"problem": RunnablePassthrough()}  # Preserve original problem
    | {
        "steps": reasoning_prompt | model | parser,  # Generate steps
        "problem": lambda x: x["problem"]  # Pass through problem
    }
    | solving_prompt
    | model
    | parser
)

# Use the chain
solution = reasoning_chain.invoke("How many minutes are in a week?")
print(solution)

# Type flow:
# str → Dict[str, str] → Dict[str, str] → List[BaseMessage] → AIMessage → str
```

### Example 3: Retrieval-Augmented Generation (RAG) Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from operator import itemgetter

# Setup vector store (example)
vectorstore = Chroma.from_texts(
    texts=["LCEL is declarative", "Chains use the pipe operator", "Runnables are composable"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# Build RAG chain
template = """Answer based on this context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

rag_chain = (
    {
        "context": itemgetter("question") | retriever | (lambda docs: "\n".join(doc.page_content for doc in docs)),
        "question": itemgetter("question")
    }
    | prompt
    | model
    | parser
)

# Use the chain
answer = rag_chain.invoke({"question": "What is LCEL?"})
print(answer)

# Type flow:
# Dict → Dict[str, str] → List[BaseMessage] → AIMessage → str
```

**Complete Example**: See [examples/advanced_chains/retrieval_qa_chain.py](../../examples/advanced_chains/retrieval_qa_chain.py)

### Example 4: Streaming Chain with Token-by-Token Output

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Write a poem about {topic}")
model = ChatOpenAI(model="gpt-4", streaming=True)
parser = StrOutputParser()

streaming_chain = prompt | model | parser

# Stream output token by token
for chunk in streaming_chain.stream({"topic": "Python programming"}):
    print(chunk, end="", flush=True)

# Output appears incrementally:
# "In" "the" "realm" "of" "code" "," "where" "logic" "reigns" ...
```

**Complete Example**: See [examples/advanced_chains/streaming_responses.py](../../examples/advanced_chains/streaming_responses.py)

### Example 5: Async Chain Execution

```python
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

async def main():
    prompt = ChatPromptTemplate.from_template("Translate to {language}: {text}")
    model = ChatOpenAI(model="gpt-4")
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    
    # Single async invocation
    result = await chain.ainvoke({
        "language": "French",
        "text": "Hello world"
    })
    print(result)  # "Bonjour le monde"
    
    # Batch async invocation
    results = await chain.abatch([
        {"language": "Spanish", "text": "Hello"},
        {"language": "German", "text": "Goodbye"},
        {"language": "Italian", "text": "Thank you"}
    ])
    for translation in results:
        print(translation)

asyncio.run(main())
```

**Complete Example**: See [examples/advanced_chains/async_chain_execution.py](../../examples/advanced_chains/async_chain_execution.py)

---

## Common Patterns

### Pattern: Using `itemgetter` for Key Selection

Extract specific keys from dictionaries:

```python
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

# Create dict with multiple values
create_data = RunnableLambda(lambda x: {
    "name": x,
    "length": len(x),
    "uppercase": x.upper()
})

# Select only "uppercase" for next step
process_uppercase = RunnableLambda(lambda s: f"Processed: {s}")

# Chain with itemgetter
chain = create_data | itemgetter("uppercase") | process_uppercase

result = chain.invoke("hello")
# Step 1: {"name": "hello", "length": 5, "uppercase": "HELLO"}
# Step 2: "HELLO" (extracted via itemgetter)
# Step 3: "Processed: HELLO"
print(result)
```

### Pattern: Dict Unpacking for Multiple Inputs

Pass multiple values to a function expecting multiple arguments:

```python
from langchain_core.runnables import RunnableLambda

# Function expecting two arguments
def combine(text: str, multiplier: int) -> str:
    return text * multiplier

# Create dict with required keys
prepare = RunnableLambda(lambda x: {"text": x, "multiplier": 3})

# Use RunnableLambda with unpacking
process = RunnableLambda(lambda d: combine(d["text"], d["multiplier"]))

chain = prepare | process

result = chain.invoke("Hi ")
# Step 1: {"text": "Hi ", "multiplier": 3}
# Step 2: "Hi Hi Hi "
print(result)
```

### Pattern: RunnableLambda for Custom Transformations

Wrap arbitrary functions as runnables:

```python
from langchain_core.runnables import RunnableLambda

def complex_transformation(data):
    """Custom business logic."""
    # Arbitrary processing
    processed = data.upper().replace(" ", "_")
    return {"result": processed, "length": len(processed)}

# Wrap as runnable
transformer = RunnableLambda(complex_transformation)

# Compose with other runnables
chain = transformer | RunnableLambda(lambda d: d["result"])

result = chain.invoke("hello world")
print(result)  # "HELLO_WORLD"
```

### Pattern: Assigning Additional Context

Add extra information to the data flow:

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Add metadata without losing original input
chain = {
    "original": RunnablePassthrough(),
    "timestamp": RunnableLambda(lambda x: "2024-01-01"),
    "processed": RunnableLambda(lambda x: x.upper())
}

result = chain.invoke("hello")
# Output: {
#     "original": "hello",
#     "timestamp": "2024-01-01",
#     "processed": "HELLO"
# }
```

### Pattern: Fallback Chains

Handle errors gracefully with fallbacks:

```python
from langchain_core.runnables import RunnableLambda

def risky_operation(x):
    if x < 0:
        raise ValueError("Negative not allowed")
    return x ** 2

def safe_fallback(x):
    return abs(x) ** 2  # Always works

primary = RunnableLambda(risky_operation)
fallback = RunnableLambda(safe_fallback)

chain = primary.with_fallbacks([fallback])

print(chain.invoke(5))   # 25 (primary)
print(chain.invoke(-3))  # 9 (fallback)
```

**Reference**: See [Error Handling Guide](error-handling.md) for comprehensive fallback patterns.

---

## Configuration

### Using RunnableConfig

Pass configuration options to chains for callbacks, tags, and metadata:

```python
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import ConsoleCallbackHandler

config = RunnableConfig(
    callbacks=[ConsoleCallbackHandler()],
    tags=["production", "qa-chain"],
    metadata={"user_id": "12345", "session": "abc"}
)

result = chain.invoke({"question": "What is LCEL?"}, config=config)
```

**Configuration Options**:

| Option | Type | Description |
|--------|------|-------------|
| `callbacks` | List[BaseCallbackHandler] | Callback handlers for tracing/logging |
| `tags` | List[str] | Tags for categorizing execution |
| `metadata` | Dict[str, Any] | Custom metadata attached to trace |
| `max_concurrency` | int | Maximum concurrent operations in parallel execution |
| `recursion_limit` | int | Maximum recursion depth for nested chains |

**Source**: Configuration handling in `libs/core/langchain_core/runnables/config.py`

### Configurable Chains

Make chains configurable at runtime:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField

# Create model with configurable temperature
model = ChatOpenAI(model="gpt-4").configurable_fields(
    temperature=ConfigurableField(
        id="model_temperature",
        name="Model Temperature",
        description="Sampling temperature for generation"
    )
)

prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
chain = prompt | model | StrOutputParser()

# Use with different configurations
creative_result = chain.invoke(
    {"topic": "space"},
    config={"configurable": {"model_temperature": 0.9}}
)

factual_result = chain.invoke(
    {"topic": "space"},
    config={"configurable": {"model_temperature": 0.1}}
)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Type Mismatch Between Steps

**Symptom**: Runtime error when composing runnables, unexpected type errors

**Cause**: Output type of one step doesn't match input type of the next

**Solution**:
```python
# Add explicit type conversion step
from langchain_core.runnables import RunnableLambda

# Problem chain
step1 = RunnableLambda(lambda x: str(x))  # Returns str
step2 = RunnableLambda(lambda x: x * 2)   # Expects numeric

# Solution: Insert conversion
convert = RunnableLambda(lambda x: int(x))
chain = step1 | convert | step2  # Now: int → str → int → int
```

#### Issue: Missing Dictionary Keys

**Symptom**: `KeyError` when accessing dictionary keys in subsequent steps

**Cause**: Previous step doesn't produce expected keys

**Solution**:
```python
# Use .get() with defaults
step = RunnableLambda(lambda d: d.get("expected_key", "default_value"))

# Or validate keys before access
def safe_access(d):
    if "expected_key" not in d:
        raise ValueError(f"Missing key 'expected_key'. Available: {list(d.keys())}")
    return d["expected_key"]

step = RunnableLambda(safe_access)
```

#### Issue: Schema Validation Failures

**Symptom**: Pydantic validation errors when invoking chains

**Cause**: Input doesn't match the expected schema

**Solution**:
```python
# Inspect schema to see requirements
print(chain.input_schema.model_json_schema())

# Ensure all required fields are provided
required_input = {
    "field1": "value1",
    "field2": "value2",
    # Add all required fields shown in schema
}
result = chain.invoke(required_input)
```

#### Issue: Async/Sync Mixing

**Symptom**: Runtime warnings or errors about event loops

**Cause**: Mixing async and sync calls incorrectly

**Solution**:
```python
import asyncio

# Use async methods consistently in async context
async def async_workflow():
    result = await chain.ainvoke(input_data)
    return result

# Use sync methods in sync context
def sync_workflow():
    result = chain.invoke(input_data)
    return result

# Don't mix: chain.invoke() inside async function
# Do: await chain.ainvoke() inside async function
```

#### Issue: Streaming Not Working

**Symptom**: Stream returns empty or blocks unexpectedly

**Cause**: Model not configured for streaming, or parser doesn't support streaming

**Solution**:
```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Enable streaming on model
model = ChatOpenAI(model="gpt-4", streaming=True)  # Must set streaming=True

# Use stream-compatible parser
parser = StrOutputParser()  # Supports streaming

chain = prompt | model | parser

# Stream output
for chunk in chain.stream(input_data):
    print(chunk, end="", flush=True)
```

### Debugging Checklist

When a chain isn't working as expected:

- [ ] Inspect input schema: `print(chain.input_schema.model_json_schema())`
- [ ] Inspect output schema: `print(chain.output_schema.model_json_schema())`
- [ ] Visualize graph: `print(chain.get_graph())`
- [ ] Enable verbose mode: `config={"callbacks": [ConsoleCallbackHandler()]}`
- [ ] Check type annotations with mypy: `mypy --strict your_file.py`
- [ ] Validate each step independently before composing
- [ ] Test with minimal input first, then increase complexity
- [ ] Review logs for intermediate values and error messages

---

## Summary

### Key Takeaways

1. **LCEL Fundamentals**:
   - Use the pipe operator (`|`) to compose runnables sequentially
   - LCEL provides automatic sync/async/batch/stream support
   - Type safety ensures output of step N matches input of step N+1

2. **Composition Patterns**:
   - **Sequential**: Use `|` for step-by-step transformations
   - **Parallel**: Use dict literals for concurrent execution
   - **Conditional**: Use `RunnableBranch` for routing logic
   - **Error Handling**: Use `.with_fallbacks()` for resilience

3. **Type Flow**:
   - Document type transformations at each stage
   - Use explicit type annotations for mypy compliance
   - Leverage `input_schema` and `output_schema` for runtime validation

4. **Best Practices**:
   - Start simple, then add complexity incrementally
   - Use inline comments to document type flow
   - Test each component independently
   - Enable verbose mode for debugging
   - Wrap custom logic in `RunnableLambda`

5. **Common Patterns**:
   - `itemgetter` for key selection
   - `RunnablePassthrough` for preserving inputs
   - Dict unpacking for multi-argument functions
   - Parallel branches for independent operations

### Next Steps

- **Explore API Reference**: [Runnables Base API](../api-reference/runnables/base.md)
- **Study Examples**: [Basic Chains](../../examples/basic_chains/), [Advanced Chains](../../examples/advanced_chains/)
- **Learn Type Patterns**: [Type Annotation Examples](../../examples/type_patterns/)
- **Master Debugging**: [Debugging Guide](../debugging/common-issues.md)
- **Production Deployment**: [Production Guide](production-deployment.md)

### Additional Resources

- [LangChain Official Documentation](https://docs.langchain.com/)
- [LCEL Type System Architecture](../architecture/lcel-type-system.md)
- [Callback System Guide](callbacks.md)
- [Error Handling Guide](error-handling.md)
- [Glossary](../glossary.md)

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-01  
**Source References**:
- `libs/core/langchain_core/runnables/base.py:150-697`
- `libs/core/langchain_core/prompts/chat.py:55-100`
- `libs/core/langchain_core/output_parsers/string.py:1-38`

**Maintainer**: LangChain Documentation Team  
**Feedback**: Please report issues or suggestions via GitHub Issues

