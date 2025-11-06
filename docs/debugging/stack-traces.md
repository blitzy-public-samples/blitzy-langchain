# Interpreting LangChain Stack Traces

## Overview

LangChain stack traces can be challenging to interpret due to multiple layers of abstraction including LCEL composition, callback managers, async event loops, and Pydantic validation. This guide provides strategies for navigating these traces to quickly identify root causes of failures.

## Anatomy of a LangChain Stack Trace

A typical LangChain stack trace follows this pattern:

```
1. Your Application Code (entry point)
   ↓
2. Runnable.invoke() / Chain.invoke() (execution layer)
   ↓
3. CallbackManager.on_chain_start() (callback layer)
   ↓
4. Internal Execution (_call, _run methods)
   ↓
5. Composed Runnables (__or__, __ror__ operators)
   ↓
6. Pydantic Validation (if using structured outputs)
   ↓
7. External API Calls (LLM providers, vector stores)
   ↓
8. Exception Origin (actual error location)
```

Each layer adds frames to the stack trace, making it crucial to understand which frames represent framework infrastructure versus your actual code.

## Key Stack Frame Patterns

### Framework Infrastructure Frames

These frames are part of LangChain's internal operation and typically don't contain your bugs:

- **`langchain_core/runnables/base.py`** - Runnable protocol implementation
- **`langchain_core/callbacks/manager.py`** - Callback orchestration
- **`langchain_core/runnables/config.py`** - Configuration handling
- **`pydantic/main.py`** - Model validation (unless you're debugging schema issues)
- **`asyncio/*.py`** - Event loop management (unless async-specific issue)

### User Code Frames

These frames likely contain your application logic:

- Your application files (e.g., `app.py`, `chains.py`, `agents.py`)
- Custom chain implementations
- Custom tool implementations
- Custom callback handlers
- Lambda functions in LCEL compositions

## Common Stack Trace Patterns

### Pattern 1: LCEL Pipe Operator Type Mismatch

**Symptom**: Error occurs during chain composition with `__or__` or `__ror__` in the stack trace.

**Example Stack Trace**:

```python
Traceback (most recent call last):
  File "app.py", line 23, in main
    result = chain.invoke({"topic": "AI"})
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 2826, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 3064, in invoke
    return self.last.invoke(
           ^^^^^^^^^^^^^^^^
  File "langchain_core/output_parsers/string.py", line 63, in invoke
    return self._call_with_config(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 1923, in _call_with_config
    output = context.run(
             ^^^^^^^^^^^
  File "langchain_core/output_parsers/string.py", line 77, in parse
    text = parsed_result if isinstance(parsed_result, str) else str(parsed_result)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: __str__ returned non-string (type dict)
```

**Analysis**:
1. **Entry Point** (Line 1-3): Your code at `app.py:23` invoked the chain
2. **Runnable Sequence** (Lines 4-9): Shows execution through composed runnables
3. **Parser Failure** (Lines 10-15): `StrOutputParser` expected string but received dict

**Root Cause**: The previous step in your LCEL chain returned a dictionary, but you piped it to `StrOutputParser` which expects a string.

**Source Code Reference**: `libs/core/langchain_core/runnables/base.py:608-627`

**Fix Strategy**: 
- Check the output type of your chain's previous step
- Use appropriate output parser (e.g., `JsonOutputParser` for dict)
- Add transformation step: `chain | RunnableLambda(lambda x: x["key"]) | StrOutputParser()`

### Pattern 2: Callback Invocation Sequence Error

**Symptom**: Error occurs during callback execution with `on_chain_start`, `on_llm_start`, or similar callback methods in stack.

**Example Stack Trace**:

```python
Traceback (most recent call last):
  File "app.py", line 45, in run_chain
    response = llm_chain.invoke({"question": query})
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_classic/chains/base.py", line 158, in invoke
    run_manager = callback_manager.on_chain_start(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/callbacks/manager.py", line 1544, in on_chain_start
    for handler in self.handlers:
  File "langchain_core/callbacks/manager.py", line 1546, in on_chain_start
        handler.on_chain_start(serialized, inputs, run_id=run_id, **kwargs)
  File "custom_callbacks.py", line 28, in on_chain_start
    self.log_data["runs"][str(run_id)]["start_time"] = time.time()
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: '7a3f2c1b-9e5d-4f8a-b2c7-8e9f1a2b3c4d'
```

**Analysis**:
1. **Entry Point** (Lines 1-3): Chain invocation at `app.py:45`
2. **Callback Setup** (Lines 4-6): `Chain.invoke()` calls `on_chain_start()` callback
3. **Callback Manager** (Lines 7-10): Iterating through registered handlers
4. **Custom Handler Error** (Lines 11-14): Your custom callback handler failed

**Root Cause**: Custom callback handler assumed `run_id` key existed in `log_data["runs"]` dictionary before accessing it.

**Source Code Reference**: `libs/langchain/langchain_classic/chains/base.py:158-163`

**Fix Strategy**:
- Initialize data structures in callback handler's `__init__` or before first access
- Use `.get()` with defaults: `self.log_data["runs"].get(str(run_id), {}).get("start_time")`
- Add defensive checks: `if str(run_id) not in self.log_data["runs"]: ...`

### Pattern 3: Async Event Loop and Coroutine Errors

**Symptom**: Stack trace includes asyncio frames, mentions of coroutines, or "RuntimeError: This event loop is already running".

**Example Stack Trace**:

```python
Traceback (most recent call last):
  File "app.py", line 67, in async_main
    results = await chain.abatch(inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 1142, in abatch
    return await gather_with_concurrency(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/utils.py", line 578, in gather_with_concurrency
    coros = [self._transform(c) for c in coros]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<ipython-input-12-abc123>", line 4, in <listcomp>
    coros = [self._transform(c) for c in coros]
            ^^^^^^^^^
  File "langchain_core/runnables/utils.py", line 572, in _transform
    return asyncio.create_task(coro)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "asyncio/tasks.py", line 381, in create_task
    task = loop.create_task(coro)
           ^^^^^^^^^^^^^^^^^^^^^
RuntimeError: no running event loop
```

**Analysis**:
1. **Entry Point** (Lines 1-3): Async chain invocation at `app.py:67`
2. **Batch Processing** (Lines 4-6): `abatch()` uses concurrent execution
3. **Asyncio Utilities** (Lines 7-9): Gathering coroutines with concurrency control
4. **Event Loop Error** (Lines 10-15): Attempted to create task without running event loop

**Root Cause**: Called async method from synchronous context or tried to nest event loops.

**Source Code Reference**: `libs/core/langchain_core/runnables/base.py:1142` (abatch method)

**Fix Strategy**:
- Ensure you're in an async context: `asyncio.run(async_main())`
- Don't mix sync and async: Use `ainvoke()` not `invoke()` in async functions
- For Jupyter notebooks, use: `await chain.abatch(inputs)` directly
- Avoid: `asyncio.run()` inside already-running event loop

### Pattern 4: Memory Loading Failures

**Symptom**: Error during memory operations with `BaseMemory.load_memory_variables()` or `save_context()` in stack trace.

**Example Stack Trace**:

```python
Traceback (most recent call last):
  File "app.py", line 89, in chat_loop
    response = conversation.invoke({"input": user_message})
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_classic/chains/base.py", line 146, in invoke
    inputs = self.prep_inputs(input)
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_classic/chains/base.py", line 354, in prep_inputs
    memory_variables = self.memory.load_memory_variables(inputs)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_classic/memory/buffer.py", line 78, in load_memory_variables
    return {self.memory_key: self.buffer_as_str}
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_classic/memory/buffer.py", line 54, in buffer_as_str
    return get_buffer_string(
           ^^^^^^^^^^^^^^^^^
  File "langchain_core/messages/utils.py", line 324, in get_buffer_string
    for m in messages:
  File "langchain_classic/memory/chat_memory.py", line 42, in chat_memory
    return self.chat_memory.messages
           ^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'messages'
```

**Analysis**:
1. **Entry Point** (Lines 1-3): Conversational chain invoked at `app.py:89`
2. **Input Preparation** (Lines 4-6): `prep_inputs()` loads memory before execution
3. **Memory Loading** (Lines 7-9): `load_memory_variables()` accesses memory buffer
4. **Buffer Access** (Lines 10-18): Attempted to access uninitialized `chat_memory`

**Root Cause**: ConversationBufferMemory's `chat_memory` was not initialized before use.

**Source Code Reference**: `libs/langchain/langchain_classic/chains/base.py:146` (prep_inputs call)

**Fix Strategy**:
- Initialize memory correctly: `ConversationBufferMemory(chat_memory=ChatMessageHistory())`
- Check memory setup before first use
- Verify memory key matches chain's expected input variables
- For custom memory: Ensure all required attributes are initialized in `__init__`

### Pattern 5: Pydantic Validation Errors

**Symptom**: Stack trace includes `pydantic/main.py` or `pydantic/validators.py` with ValidationError.

**Example Stack Trace**:

```python
Traceback (most recent call last):
  File "app.py", line 112, in process_document
    result = structured_chain.invoke({"text": doc_text})
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 2826, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/output_parsers/pydantic.py", line 98, in invoke
    return self._call_with_config(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 1923, in _call_with_config
    output = context.run(
             ^^^^^^^^^^^
  File "langchain_core/output_parsers/pydantic.py", line 124, in parse
    return self.pydantic_object.model_validate_json(text, strict=True)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pydantic/main.py", line 580, in model_validate_json
    return cls.__pydantic_validator__.validate_json(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pydantic_core/_pydantic_core.pyx", line 1121, in validate_json
pydantic_core._pydantic_core.ValidationError: 2 validation errors for DocumentSummary
title
  Field required [type=missing, input_value={'summary': 'This docu...trends.', 'tags': []}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.5/v/missing
summary
  String should have at least 10 characters [type=string_too_short, input_value='Short', input_type=str]
    For further information visit https://errors.pydantic.dev/2.5/v/string_too_short
```

**Analysis**:
1. **Entry Point** (Lines 1-3): Structured output chain invoked at `app.py:112`
2. **Runnable Execution** (Lines 4-6): Chain steps executing
3. **Pydantic Parser** (Lines 7-12): `PydanticOutputParser` validating LLM output
4. **Validation Failure** (Lines 13-22): Two validation errors in `DocumentSummary` model

**Root Cause**: LLM output didn't match the expected Pydantic schema:
- Missing required field `title`
- Field `summary` too short (< 10 characters)

**Source Code Reference**: `libs/core/langchain_core/output_parsers/pydantic.py:98-124`

**Fix Strategy**:
- Review your Pydantic model: Are all fields truly required?
- Improve LLM prompt to include clear schema instructions
- Use `Optional[str]` for fields that might be missing
- Add retry logic with `RunnableWithFallbacks`
- Use `.with_structured_output()` method for better reliability
- Validate LLM output format before Pydantic parsing

### Pattern 6: RunnableSequence Composition Errors

**Symptom**: Error in `RunnableSequence.invoke()` with multiple chained operations failing.

**Example Stack Trace**:

```python
Traceback (most recent call last):
  File "app.py", line 134, in pipeline
    output = (prompt | model | parser).invoke({"query": question})
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 2826, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 3064, in invoke
    return self.last.invoke(
           ^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 2806, in invoke
    for step in self.steps[:-1]:
        ^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 2810, in invoke
        input = context.run(step.invoke, input, config)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/prompts/chat.py", line 189, in invoke
    prompt_value = self.format_prompt(**input)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/prompts/chat.py", line 178, in format_prompt
    return ChatPromptValue(messages=messages)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/prompts/base.py", line 456, in format_prompt
    kwargs = self._merge_partial_and_user_variables(**kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'context'
```

**Analysis**:
1. **Entry Point** (Lines 1-3): LCEL chain `(prompt | model | parser)` invoked at `app.py:134`
2. **Sequence Execution** (Lines 4-9): RunnableSequence executing steps sequentially
3. **Prompt Formatting** (Lines 10-15): ChatPromptTemplate trying to format with inputs
4. **Missing Variable** (Lines 16-18): Required template variable `context` not provided

**Root Cause**: Your prompt template expects a `context` variable, but you only provided `query` in the invoke call.

**Source Code Reference**: `libs/core/langchain_core/runnables/base.py:608-627` (__or__ operator)

**Fix Strategy**:
- Check template's `input_variables`: `print(prompt.input_variables)`
- Provide all required variables: `invoke({"query": q, "context": ctx})`
- Use `.partial()` for constant values: `prompt.partial(context="...")`
- Add variable extraction step before prompt: `RunnableLambda(lambda x: {...}) | prompt`

## Distinguishing Framework vs User Code

### Strategy 1: Source File Path Analysis

**Framework Code Paths**:
- `site-packages/langchain_core/` - Core abstractions
- `site-packages/langchain_classic/` - Classic chains (framework logic)
- `site-packages/pydantic/` - Validation framework
- `site-packages/asyncio/` - Python async infrastructure

**User Code Paths**:
- Your project directory (e.g., `/app/`, `/src/`, `/chains/`)
- Custom modules you've written
- Lambda functions (shown as `<lambda>` or `<listcomp>`)

**Example Frame Analysis**:

```python
# FRAMEWORK FRAME (skip when debugging your logic)
File "langchain_core/runnables/base.py", line 2826, in invoke
  input = context.run(step.invoke, input, config, **kwargs)

# USER CODE FRAME (investigate this)
File "app.py", line 134, in pipeline
  output = (prompt | model | parser).invoke({"query": question})

# FRAMEWORK FRAME (skip)
File "langchain_core/prompts/chat.py", line 189, in invoke
  prompt_value = self.format_prompt(**input)
```

Focus your debugging on the USER CODE FRAME at `app.py:134`.

### Strategy 2: Exception Message Analysis

The exception message often reveals whether the issue is in your code or the framework:

**User Code Issues** (fix in your application):
- `KeyError: 'context'` - You forgot to provide a required input variable
- `TypeError: 'NoneType' object is not subscriptable` - Your data is None
- `AttributeError: 'dict' object has no attribute 'text'` - Wrong data structure

**Configuration Issues** (fix your setup):
- `ValidationError: 2 validation errors for Model` - Schema mismatch
- `ValueError: Could not parse LLM output` - Parser incompatible with output format
- `RuntimeError: no running event loop` - Async/sync context mismatch

**External/Framework Issues** (check dependencies or report bug):
- `ImportError: cannot import name 'X'` - Version mismatch or missing dependency
- `AssertionError` in framework code - Potential framework bug
- Network errors from external APIs - Provider issue

### Strategy 3: Reading Bottom-Up

Start from the **bottom** of the stack trace (the actual exception) and work **upward** to find the first frame in your code:

```python
Traceback (most recent call last):                    ← START HERE
  File "app.py", line 45, in main                     ← YOUR CODE (likely cause)
    result = chain.invoke(inputs)
  File "langchain_core/runnables/base.py", line 2826  ← FRAMEWORK
    input = context.run(step.invoke, ...)
  File "langchain_core/prompts/chat.py", line 189     ← FRAMEWORK
    prompt_value = self.format_prompt(**input)
KeyError: 'required_field'                            ← ACTUAL ERROR (read this first)
```

**Reading Order**:
1. **Last line**: What error occurred? (`KeyError: 'required_field'`)
2. **Second to last frame**: Where in framework did it fail? (prompt formatting)
3. **Work upward**: Find first frame in your code (`app.py:45`)
4. **Conclusion**: Your code at `app.py:45` didn't provide `required_field` to the prompt

## Advanced Techniques

### Technique 1: Identifying LCEL Composition Boundaries

LCEL chains use `__or__` and `RunnableSequence`. Look for these patterns:

```python
# Stack trace snippet showing LCEL boundary
File "langchain_core/runnables/base.py", line 608, in __or__
  return RunnableSequence(self, coerce_to_runnable(other))
                          ^^^^
File "langchain_core/runnables/base.py", line 2806, in invoke
  for step in self.steps[:-1]:
      ^^^^ Each iteration processes one step in: (prompt | model | parser)
```

Count the steps to identify which component failed:
- Step 0: `prompt` (ChatPromptTemplate)
- Step 1: `model` (ChatOpenAI)
- Step 2: `parser` (StrOutputParser)

### Technique 2: Callback Event Sequence Tracking

Callbacks follow a predictable sequence. In the stack, you'll see:

```python
# Normal execution
on_chain_start → on_llm_start → on_llm_end → on_chain_end

# With error
on_chain_start → on_llm_start → <exception> → on_chain_error
```

**Source Code Reference**: `libs/langchain/langchain_classic/chains/base.py:158-180`

Trace which callback frame failed to identify the execution stage:
- `on_chain_start`: Error before execution (setup issue)
- `on_llm_start`: Error before LLM call (input preparation issue)
- Between `on_llm_start` and `on_llm_end`: LLM API error
- `on_llm_end`: Error processing LLM response
- `on_chain_error`: Your error handler itself has a bug

### Technique 3: Async Stack Trace Navigation

Async stack traces include extra asyncio frames. Filter them mentally:

```python
# Actual execution flow
File "app.py", line 67, in async_main              ← YOUR CODE
  results = await chain.abatch(inputs)
File "langchain_core/runnables/base.py", line 1142 ← FRAMEWORK (async entry)
  return await gather_with_concurrency(...)

# Asyncio infrastructure (can usually skip)
File "asyncio/tasks.py", line 381
File "asyncio/base_events.py", line 1890

# Back to relevant code
File "langchain_core/runnables/utils.py", line 578 ← FRAMEWORK (gathering coroutines)
```

Focus on frames with meaningful method names (`abatch`, `gather_with_concurrency`) and skip generic asyncio frames (`tasks.py`, `base_events.py`).

## Practical Debugging Workflow

### Step-by-Step Debugging Process

1. **Read the Exception Message First**
   ```python
   KeyError: 'context'
   ```
   This tells you what went wrong at a high level.

2. **Identify the Exception Location**
   Find the last frame before the exception was raised:
   ```python
   File "langchain_core/prompts/chat.py", line 189, in invoke
     prompt_value = self.format_prompt(**input)
   ```
   The error occurred during prompt formatting.

3. **Find Your Code in the Stack**
   Work backward to the first frame in your application:
   ```python
   File "app.py", line 134, in pipeline
     output = (prompt | model | parser).invoke({"query": question})
   ```
   Your invocation at `app.py:134` is the entry point.

4. **Analyze the Data Flow**
   Trace how data flows from your code to the error:
   - You called: `invoke({"query": question})`
   - Prompt expected: `{"query": ..., "context": ...}`
   - Missing: `context` key

5. **Formulate a Fix**
   Based on the analysis:
   ```python
   # Before (error)
   output = (prompt | model | parser).invoke({"query": question})
   
   # After (fixed)
   output = (prompt | model | parser).invoke({
       "query": question,
       "context": retrieved_context  # Added missing variable
   })
   ```

### Debugging Tools and Tricks

#### Enable Verbose Logging

Set environment variable to see execution flow:
```bash
export LANGCHAIN_VERBOSE=true
export LANGCHAIN_TRACING_V2=true
```

This will print callbacks and help identify which step in the chain failed.

#### Use Python Debugger

Insert a breakpoint before the failing line:
```python
import pdb; pdb.set_trace()
result = chain.invoke(inputs)  # Debugger stops here
```

Commands in debugger:
- `n` - Next line
- `s` - Step into function
- `c` - Continue execution
- `p variable` - Print variable value
- `pp variable` - Pretty-print variable

#### Inspect Chain Structure

Before invoking, inspect what the chain expects:
```python
# For prompts
print("Input variables:", prompt.input_variables)
print("Template:", prompt.template)

# For chains
print("Input keys:", chain.input_keys)
print("Output keys:", chain.output_keys)

# For LCEL sequences
print("Steps:", chain.steps if hasattr(chain, 'steps') else "N/A")
```

#### Catch and Log Errors

Wrap invocations with detailed error logging:
```python
try:
    result = chain.invoke(inputs)
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print(f"Inputs provided: {inputs}")
    import traceback
    traceback.print_exc()
    raise
```

## Complete Example: Debugging a Real Stack Trace

### The Failing Code

```python
# app.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Create a RAG chain
template = """Answer the question based on the context:

Context: {context}
Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

rag_chain = prompt | model | parser

# Execute (THIS FAILS)
result = rag_chain.invoke({"question": "What is LangChain?"})
```

### The Stack Trace

```python
Traceback (most recent call last):
  File "app.py", line 18, in <module>
    result = rag_chain.invoke({"question": "What is LangChain?"})
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 2826, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 3064, in invoke
    return self.last.invoke(
           ^^^^^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 2806, in invoke
    for step in self.steps[:-1]:
        ^^^^^^^^^^^^
  File "langchain_core/runnables/base.py", line 2810, in invoke
        input = context.run(step.invoke, input, config)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/prompts/chat.py", line 189, in invoke
    prompt_value = self.format_prompt(**input)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/prompts/chat.py", line 178, in format_prompt
    return ChatPromptValue(messages=messages)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "langchain_core/prompts/base.py", line 456, in format_prompt
    kwargs = self._merge_partial_and_user_variables(**kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'context'
```

### Step-by-Step Analysis

**Step 1: Read Exception**
```
KeyError: 'context'
```
A required key named `context` is missing.

**Step 2: Locate Error in Stack**
```python
File "langchain_core/prompts/base.py", line 456, in format_prompt
  kwargs = self._merge_partial_and_user_variables(**kwargs)
KeyError: 'context'
```
Error occurred during prompt variable merging. The prompt template is missing the `context` variable.

**Step 3: Find User Code**
```python
File "app.py", line 18, in <module>
  result = rag_chain.invoke({"question": "What is LangChain?"})
```
Our code only provided `{"question": ...}` but template needs both `question` and `context`.

**Step 4: Trace Execution Path**
```
app.py:18 (invoke with {"question": ...})
  ↓
RunnableSequence.invoke (LCEL chain execution)
  ↓
Step 1: prompt.invoke (ChatPromptTemplate)
  ↓
format_prompt (trying to format template)
  ↓
ERROR: KeyError 'context' (missing required variable)
```

**Step 5: Understand Template Requirements**
Looking at our template:
```python
template = """Answer the question based on the context:

Context: {context}      ← REQUIRES 'context' variable
Question: {question}    ← REQUIRES 'question' variable

Answer:"""
```

We need BOTH `context` and `question`, but only provided `question`.

**Step 6: Implement Fix**

Option 1: Provide context in invoke
```python
# Retrieve context from vector store
docs = retriever.get_relevant_documents("What is LangChain?")
context = "\n".join([doc.page_content for doc in docs])

# Now provide both variables
result = rag_chain.invoke({
    "question": "What is LangChain?",
    "context": context
})
```

Option 2: Add retrieval step to chain
```python
from langchain_core.runnables import RunnablePassthrough

# Chain now handles retrieval automatically
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt 
    | model 
    | parser
)

# Invoke with just the question
result = rag_chain.invoke("What is LangChain?")
```

Option 3: Use partial variables for static context
```python
# If context is static, use partial
prompt = prompt.partial(context="LangChain is a framework...")

# Now only question is required
result = rag_chain.invoke({"question": "What is LangChain?"})
```

## Quick Reference: Error Type → Likely Cause

| Error Type | Likely Cause | Where to Look |
|------------|--------------|---------------|
| `KeyError` | Missing input variable for prompt/chain | Check `invoke()` arguments vs `input_variables` |
| `TypeError: __str__ returned non-string` | Wrong output type for parser | Check previous step's output type |
| `AttributeError: 'NoneType' object has no attribute` | Uninitialized component | Check object initialization |
| `ValidationError` | Pydantic schema mismatch | Check LLM output vs expected schema |
| `RuntimeError: no running event loop` | Async/sync mismatch | Use `asyncio.run()` or `await` properly |
| `ImportError` | Missing or mismatched dependency | Check package versions |
| Network errors (timeout, connection) | External API issue | Check API status, rate limits, credentials |

## When to Report a Bug

If after thorough analysis you determine the issue is in LangChain itself:

1. **Verify it's truly a framework bug**:
   - Works in one version but not another (regression)
   - Framework code raises `AssertionError` or unexpected exception
   - Documented behavior doesn't match actual behavior

2. **Create a minimal reproduction**:
   ```python
   # Simplest code that reproduces the issue
   from langchain_core.runnables import RunnableLambda
   
   # This should work but raises AssertionError
   chain = RunnableLambda(lambda x: x) | RunnableLambda(lambda x: x)
   result = chain.invoke("test")
   ```

3. **Report with details**:
   - Full stack trace
   - LangChain version: `langchain --version`
   - Python version: `python --version`
   - Minimal reproduction code
   - Expected vs actual behavior

## Related Resources

- [Common LangChain Issues](common-issues.md) - Top 10 failure modes and solutions
- [Logging Guide](logging.md) - Configure detailed execution logging
- [Troubleshooting Decision Tree](troubleshooting.md) - Systematic problem diagnosis

## Source Code References

All frame patterns documented in this guide can be verified against:

- `libs/core/langchain_core/runnables/base.py` - Runnable protocol and LCEL composition
- `libs/core/langchain_core/callbacks/manager.py` - Callback orchestration and event flow
- `libs/langchain/langchain_classic/chains/base.py` - Chain base class with callback integration
- `libs/core/langchain_core/prompts/chat.py` - Prompt template formatting and variable handling
- `libs/core/langchain_core/output_parsers/` - Output parser implementations and validation

For the most current implementation details, always refer to the source code in your installed version of LangChain.
