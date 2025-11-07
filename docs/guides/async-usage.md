# Async Usage Guide

Comprehensive guide for using async/await patterns with LangChain, including event loop management, async callbacks, concurrent execution, and common pitfalls.

---

## Introduction

### Why Use Async with LangChain?

Asynchronous programming provides significant benefits when working with LangChain chains, particularly for I/O-bound operations like API calls and database queries:

**Performance Benefits**:
- **Non-blocking I/O**: Make concurrent API calls without blocking the event loop
- **Better Throughput**: Process multiple requests simultaneously, reducing total latency
- **Resource Efficiency**: Handle more concurrent operations with fewer system resources
- **Improved Responsiveness**: Keep applications responsive during long-running operations

**Example Performance Impact**:
```
Synchronous: 10 API calls × 1 second each = 10 seconds total
Asynchronous: 10 API calls × 1 second each = ~1 second total (with concurrency)
```

### LangChain Async Support

All `Runnable` objects in LangChain have built-in async support with three primary async methods:

- **`ainvoke()`**: Transform a single input asynchronously
- **`abatch()`**: Process multiple inputs concurrently
- **`astream()`**: Stream output asynchronously token-by-token

**Source**: `libs/core/langchain_core/runnables/base.py:122-254`

**Key Feature**: When you compose chains using LCEL (LangChain Expression Language), the resulting chain automatically supports both sync and async execution without additional code.

---

## Async Methods Overview

### `ainvoke()` - Async Single Input Transformation

The `ainvoke()` method processes a single input asynchronously and returns a complete output.

**Signature**:
```python
async def ainvoke(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Output:
    """Transform a single input into an output asynchronously."""
```

**Source**: `libs/core/langchain_core/runnables/base.py:830-849`

**Usage Example**:
```python
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Create a simple chain
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI()
chain = prompt | model

# Async invocation
async def process():
    result = await chain.ainvoke({"topic": "programming"})
    print(result.content)

# Run the async function
asyncio.run(process())
```

**When to Use `ainvoke()` vs `invoke()`**:

| Scenario | Use `ainvoke()` | Use `invoke()` |
|----------|----------------|----------------|
| Web application with async framework (FastAPI, aiohttp) | ✅ Yes | ❌ No |
| Processing multiple requests concurrently | ✅ Yes | ❌ No |
| I/O-bound operations (API calls, database queries) | ✅ Yes | ⚠️ Maybe |
| CLI scripts with simple sequential processing | ⚠️ Maybe | ✅ Yes |
| CPU-bound operations | ❌ No | ✅ Yes |
| Existing sync codebase | ❌ No | ✅ Yes |

### `abatch()` - Concurrent Processing of Multiple Inputs

The `abatch()` method processes multiple inputs concurrently using `asyncio.gather()` internally, providing significant performance improvements over sequential processing.

**Signature**:
```python
async def abatch(
    self,
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> list[Output]:
    """Process multiple inputs concurrently."""
```

**Source**: `libs/core/langchain_core/runnables/base.py:983-1027`

**Implementation Detail**: The default implementation runs `ainvoke()` on each input concurrently using `gather_with_concurrency()`, which respects the `max_concurrency` setting in the config.

**Usage Example**:
```python
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Summarize: {text}")
model = ChatOpenAI()
chain = prompt | model

async def process_batch():
    inputs = [
        {"text": "Article 1 content..."},
        {"text": "Article 2 content..."},
        {"text": "Article 3 content..."},
    ]
    
    # Process all inputs concurrently
    results = await chain.abatch(inputs)
    
    for i, result in enumerate(results):
        print(f"Summary {i+1}: {result.content}")

asyncio.run(process_batch())
```

**Performance Comparison**:
```python
import time
import asyncio

# Sequential processing with sync invoke()
start = time.time()
results = [chain.invoke(inp) for inp in inputs]  # ~3 seconds for 3 inputs
print(f"Sync: {time.time() - start:.2f}s")

# Concurrent processing with async abatch()
start = time.time()
results = await chain.abatch(inputs)  # ~1 second for 3 inputs
print(f"Async: {time.time() - start:.2f}s")
```

**Controlling Concurrency**:
```python
# Limit concurrent operations to avoid rate limits
config = {"max_concurrency": 5}
results = await chain.abatch(inputs, config=config)
```

### `astream()` - Async Streaming Output

The `astream()` method streams output as it's produced, enabling real-time display of responses and better user experience for long-running operations.

**Signature**:
```python
async def astream(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any | None,
) -> AsyncIterator[Output]:
    """Stream output asynchronously as it's produced."""
```

**Source**: `libs/core/langchain_core/runnables/base.py:1128-1147`

**Usage Example**:
```python
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Write a story about {topic}")
model = ChatOpenAI()
chain = prompt | model

async def stream_response():
    async for chunk in chain.astream({"topic": "space exploration"}):
        print(chunk.content, end="", flush=True)
    print()  # New line at end

asyncio.run(stream_response())
```

**Streaming to WebSocket** (FastAPI example):
```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    user_input = await websocket.receive_text()
    
    # Stream response to client
    async for chunk in chain.astream({"input": user_input}):
        await websocket.send_text(chunk.content)
```

---

## Async Context: When to Use Async

### When Async Methods Are Beneficial

**1. Web Applications with Async Frameworks**

Modern Python web frameworks like FastAPI, Sanic, and aiohttp use async/await for handling concurrent requests:

```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()
chain = ChatPromptTemplate.from_template("Answer: {question}") | ChatOpenAI()

@app.post("/ask")
async def ask_question(question: str):
    # FastAPI manages the event loop - just use await
    response = await chain.ainvoke({"question": question})
    return {"answer": response.content}
```

**2. Concurrent Processing of Multiple Requests**

When processing multiple independent requests, async enables true concurrency:

```python
async def process_documents(documents: list[str]):
    # Process all documents concurrently
    results = await chain.abatch([{"text": doc} for doc in documents])
    return results
```

**3. I/O-Bound Operations**

Operations that spend most time waiting for external resources:
- API calls to LLM providers
- Database queries
- File I/O operations
- Network requests

**4. Long-Running Chains with Streaming**

For better user experience, stream tokens as they're generated:

```python
async def generate_with_streaming():
    async for token in model.astream("Write a long essay"):
        yield token  # Display immediately
```

### When Sync Methods Are Sufficient

**1. Simple CLI Scripts**

For straightforward command-line tools without concurrency needs:

```python
# Simple script - sync is fine
chain = prompt | model
result = chain.invoke({"input": user_input})
print(result)
```

**2. Sequential Processing Requirements**

When operations must happen in strict sequence:

```python
# Each step depends on the previous
result1 = chain1.invoke(input1)
result2 = chain2.invoke(result1)  # Must wait for result1
result3 = chain3.invoke(result2)  # Must wait for result2
```

**3. CPU-Bound Operations**

For computationally intensive tasks (not typical with LLM chains):

```python
# CPU-bound processing
result = expensive_computation(data)  # No I/O waiting
```

**4. Existing Synchronous Codebase**

When integrating with existing sync code without refactoring needs:

```python
def existing_sync_function():
    result = chain.invoke(input)  # Sync interface
    return process_result(result)
```

---

## Event Loop Fundamentals

### Understanding the Asyncio Event Loop

The event loop is the core of Python's asyncio framework. It manages and executes asynchronous tasks, handles I/O operations, and schedules callbacks.

**Key Concepts**:
- **Event Loop**: Central execution controller that runs coroutines and manages I/O
- **Coroutine**: Function defined with `async def` that can be paused and resumed
- **Task**: Wrapper around a coroutine scheduled for execution
- **await**: Pauses execution until the awaited coroutine completes

**Event Loop Lifecycle**:
```python
import asyncio

# 1. Create event loop (usually automatic)
loop = asyncio.new_event_loop()

# 2. Run tasks
loop.run_until_complete(my_coroutine())

# 3. Close loop (cleanup)
loop.close()
```

### `asyncio.run()` for Simple Scripts

For standalone scripts, `asyncio.run()` handles loop creation and cleanup automatically:

```python
import asyncio
from langchain_openai import ChatOpenAI

model = ChatOpenAI()

async def main():
    response = await model.ainvoke("Hello!")
    print(response.content)

# asyncio.run() creates loop, runs main(), and cleans up
asyncio.run(main())
```

**What `asyncio.run()` Does**:
1. Creates a new event loop
2. Runs the provided coroutine
3. Closes the loop when complete
4. Cleans up resources

**Important**: Only use `asyncio.run()` at the top level. Never call it from within an async function:

```python
# ❌ WRONG - Don't call asyncio.run() inside async function
async def process():
    asyncio.run(other_async_function())  # ERROR!

# ✅ CORRECT - Use await instead
async def process():
    await other_async_function()
```

### Existing Event Loops in Web Frameworks

Web frameworks like FastAPI, Sanic, and aiohttp create and manage their own event loops. You don't need (and shouldn't try) to create your own.

**FastAPI Example**:
```python
from fastapi import FastAPI

app = FastAPI()

# FastAPI's event loop is already running
@app.get("/generate")
async def generate(prompt: str):
    # Just use await - no asyncio.run()!
    result = await chain.ainvoke({"prompt": prompt})
    return {"result": result.content}
```

**Rule of Thumb**: If you're inside an `async def` function, use `await`. If you're at the top level of a script, use `asyncio.run()`.

### Jupyter Notebooks and IPython

Jupyter notebooks and IPython already have a running event loop, which requires special handling:

**Problem**: Jupyter has a running event loop, and `asyncio.run()` tries to create a new one:

```python
# In Jupyter notebook
async def query():
    return await chain.ainvoke({"input": "test"})

# ❌ ERROR: RuntimeError: asyncio.run() cannot be called from a running event loop
asyncio.run(query())
```

**Solution 1**: Use `await` directly (Jupyter allows top-level await):

```python
# ✅ CORRECT - Jupyter supports top-level await
response = await chain.ainvoke({"input": "test"})
print(response)
```

**Solution 2**: Use `nest_asyncio` for complex scenarios:

```python
import nest_asyncio
nest_asyncio.apply()

# Now asyncio.run() works in Jupyter
asyncio.run(query())
```

**When to Use Each Approach**:
- **Top-level await**: Simple single-async-call scenarios
- **nest_asyncio**: When you need to run existing code that uses `asyncio.run()`

---

## Async Chain Composition with LCEL

### Automatic Async Support in LCEL

One of the most powerful features of LCEL is that chains composed with the pipe operator (`|`) automatically support both sync and async execution:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Create chain using LCEL
chain = ChatPromptTemplate.from_template("Explain {topic}") | ChatOpenAI() | StrOutputParser()

# Works with sync
result = chain.invoke({"topic": "async"})

# Also works with async - no changes needed!
result = await chain.ainvoke({"topic": "async"})
```

**Key Insight**: You define the chain once, and LCEL automatically provides both `invoke()` and `ainvoke()`, `batch()` and `abatch()`, `stream()` and `astream()`.

**Source**: `libs/core/langchain_core/runnables/base.py:150-186`

### Type Flow in Async LCEL Chains

The type transformations in async chains are identical to sync chains:

```
Dict[str, str] → PromptTemplate → List[BaseMessage] → ChatModel → AIMessage → StrOutputParser → str
```

**Example with Explicit Types**:
```python
from typing import Dict
from langchain_core.messages import BaseMessage, AIMessage

# Input type
input_data: Dict[str, str] = {"topic": "python"}

# Async execution with type flow
prompt_template = ChatPromptTemplate.from_template("Explain {topic}")
messages: List[BaseMessage] = await prompt_template.ainvoke(input_data)

model = ChatOpenAI()
ai_message: AIMessage = await model.ainvoke(messages)

parser = StrOutputParser()
result: str = await parser.ainvoke(ai_message)
```

### Streaming Async LCEL Chains

LCEL chains support streaming with automatic async support:

```python
import asyncio

# Same chain definition
chain = prompt | model | parser

async def stream_example():
    # Stream tokens as they're generated
    async for chunk in chain.astream({"topic": "quantum physics"}):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_example())
```

**Streaming Intermediate Steps**:

You can also stream intermediate results from each component:

```python
async def stream_with_details():
    async for event in chain.astream_events(
        {"topic": "machine learning"},
        version="v2"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            print(content, end="", flush=True)

asyncio.run(stream_with_details())
```

### Complex Async Chain Patterns

**Parallel Execution with RunnableParallel**:

```python
from langchain_core.runnables import RunnableParallel

# Execute multiple chains in parallel
chain = RunnableParallel({
    "summary": summary_chain,
    "sentiment": sentiment_chain,
    "keywords": keyword_chain,
})

# All three chains execute concurrently
results = await chain.ainvoke({"text": document_text})
# Results: {"summary": "...", "sentiment": "...", "keywords": [...]}
```

**Conditional Branching with RunnableBranch**:

```python
from langchain_core.runnables import RunnableBranch

# Route to different chains based on condition
branch = RunnableBranch(
    (lambda x: x["type"] == "question", question_chain),
    (lambda x: x["type"] == "summary", summary_chain),
    default_chain,  # Fallback
)

result = await branch.ainvoke({"type": "question", "content": "What is async?"})
```

---

## Async Callbacks

### AsyncCallbackHandler Overview

Async callbacks allow you to hook into chain execution lifecycle events asynchronously, enabling non-blocking logging, monitoring, and custom processing.

**Source**: `libs/core/langchain_core/callbacks/base.py:478-580`

**Key Async Callback Methods**:

```python
from langchain_core.callbacks import AsyncCallbackHandler
from uuid import UUID
from typing import Any

class CustomAsyncHandler(AsyncCallbackHandler):
    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts."""
        print(f"LLM starting with prompts: {prompts[:50]}...")
    
    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when LLM completes."""
        print(f"LLM completed")
    
    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts."""
        print(f"Chain starting with inputs: {inputs}")
    
    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when chain completes."""
        print(f"Chain completed with outputs: {outputs}")
    
    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called for each new token during streaming."""
        print(token, end="", flush=True)
```

### Thread Safety and Event Loop Considerations

**Critical Rules for Async Callbacks**:

1. **All callbacks run in the same event loop**: Callbacks execute in the same event loop as your chain, ensuring thread safety.

2. **Avoid blocking operations**: Never use blocking sync calls inside async callbacks:

```python
# ❌ WRONG - Blocking operations in async callback
class BadAsyncHandler(AsyncCallbackHandler):
    async def on_llm_end(self, response, **kwargs):
        # This blocks the event loop!
        import time
        time.sleep(1)  # BAD!
        
        # This also blocks!
        import requests
        requests.get("https://api.example.com")  # BAD!

# ✅ CORRECT - Use async equivalents
class GoodAsyncHandler(AsyncCallbackHandler):
    async def on_llm_end(self, response, **kwargs):
        # Use asyncio.sleep()
        await asyncio.sleep(1)
        
        # Use async HTTP client
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.example.com") as resp:
                data = await resp.text()
```

3. **Use async libraries**: Always use async-compatible libraries:
   - ✅ `aiohttp` instead of `requests`
   - ✅ `asyncpg` instead of `psycopg2`
   - ✅ `aiomysql` instead of `pymysql`
   - ✅ `motor` instead of `pymongo`
   - ✅ `aiofiles` instead of standard `open()`

### Practical Async Callback Example

**Async Logging to Database**:

```python
import asyncio
from datetime import datetime
from langchain_core.callbacks import AsyncCallbackHandler
import aiopg  # Async PostgreSQL driver

class DatabaseLoggerCallback(AsyncCallbackHandler):
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    async def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        """Log chain start to database."""
        async with self.db_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO chain_logs (run_id, event, timestamp, data) "
                    "VALUES (%s, %s, %s, %s)",
                    (str(run_id), "start", datetime.now(), str(inputs))
                )
    
    async def on_chain_end(self, outputs, *, run_id, **kwargs):
        """Log chain completion to database."""
        async with self.db_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO chain_logs (run_id, event, timestamp, data) "
                    "VALUES (%s, %s, %s, %s)",
                    (str(run_id), "end", datetime.now(), str(outputs))
                )

# Usage
async def main():
    # Create database connection pool
    db_pool = await aiopg.create_pool("dbname=mydb user=myuser")
    
    # Create callback
    logger = DatabaseLoggerCallback(db_pool)
    
    # Use with chain
    result = await chain.ainvoke(
        {"input": "test"},
        config={"callbacks": [logger]}
    )
```

**Async Monitoring Callback**:

```python
import aiohttp
from langchain_core.callbacks import AsyncCallbackHandler

class MonitoringCallback(AsyncCallbackHandler):
    def __init__(self, metrics_endpoint: str):
        self.metrics_endpoint = metrics_endpoint
        self.session = None
    
    async def on_llm_end(self, response, *, run_id, **kwargs):
        """Send metrics to monitoring service."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        # Extract metrics
        tokens = len(response.llm_output.get("token_usage", {}))
        
        # Send async POST request
        await self.session.post(
            self.metrics_endpoint,
            json={
                "run_id": str(run_id),
                "tokens": tokens,
                "timestamp": datetime.now().isoformat(),
            }
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

# Usage with context manager
async def main():
    async with MonitoringCallback("https://metrics.example.com/api") as callback:
        result = await chain.ainvoke(
            {"input": "test"},
            config={"callbacks": [callback]}
        )
```

### Callback Configuration with Async Chains

Pass async callbacks through the config parameter:

```python
# Single callback
handler = CustomAsyncHandler()
result = await chain.ainvoke(
    {"input": "test"},
    config={"callbacks": [handler]}
)

# Multiple callbacks
handlers = [
    DatabaseLoggerCallback(db_pool),
    MonitoringCallback(metrics_url),
    DebugPrintCallback(),
]
result = await chain.ainvoke(
    {"input": "test"},
    config={"callbacks": handlers}
)
```

---

## Async Tools and Agents

### Implementing Async Tools

Custom tools can implement async methods for non-blocking external API calls or database queries.

**Source**: `libs/core/langchain_core/tools/base.py`

**Basic Async Tool Structure**:

```python
from langchain_core.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(description="The search query")

class AsyncSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information"
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str) -> str:
        """Sync implementation (fallback)."""
        raise NotImplementedError("Use async version")
    
    async def _arun(self, query: str) -> str:
        """Async implementation using aiohttp."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.search.com/search",
                params={"q": query}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["results"][0]["snippet"]
                else:
                    return f"Search failed with status {response.status}"
```

**Database Async Tool**:

```python
import asyncpg
from langchain_core.tools import BaseTool

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "Query the database for information"
    db_pool: asyncpg.Pool
    
    def _run(self, query: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, query: str) -> str:
        """Execute database query asynchronously."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query)
            return str(rows)
```

### Async Agent Execution

Agents execute tools asynchronously when using `ainvoke()`:

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Create async tools
search_tool = AsyncSearchTool()
db_tool = DatabaseQueryTool(db_pool=pool)

# Create agent
llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_tools_agent(llm, [search_tool, db_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[search_tool, db_tool])

# Execute agent asynchronously
async def run_agent():
    result = await agent_executor.ainvoke({
        "input": "Search for recent Python news and save to database"
    })
    print(result["output"])

asyncio.run(run_agent())
```

**Tool Execution Flow**:
```
User Input → Agent Decision (LLM) → Tool Selection → Async Tool Execution → Result Processing → Final Answer
```

---

## Concurrent Execution Patterns

### Using `abatch()` for Concurrent Processing

The `abatch()` method is the primary way to process multiple inputs concurrently:

**Basic Concurrent Processing**:

```python
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chain = ChatPromptTemplate.from_template("Analyze sentiment: {text}") | ChatOpenAI()

async def analyze_batch():
    reviews = [
        {"text": "This product is amazing!"},
        {"text": "Terrible experience, very disappointed."},
        {"text": "It's okay, nothing special."},
        {"text": "Exceeded all my expectations!"},
        {"text": "Complete waste of money."},
    ]
    
    # Process all reviews concurrently
    results = await chain.abatch(reviews)
    
    for i, result in enumerate(results):
        print(f"Review {i+1}: {result.content}")

asyncio.run(analyze_batch())
```

**Performance Impact**:
```python
import time

# Sequential processing (sync)
start = time.time()
results = [chain.invoke(review) for review in reviews]
sync_time = time.time() - start
print(f"Sequential: {sync_time:.2f}s")

# Concurrent processing (async)
start = time.time()
results = await chain.abatch(reviews)
async_time = time.time() - start
print(f"Concurrent: {async_time:.2f}s")
print(f"Speedup: {sync_time / async_time:.2f}x")

# Example output:
# Sequential: 5.23s
# Concurrent: 1.15s
# Speedup: 4.55x
```

### Using `asyncio.gather()` for Multiple Chains

For running multiple different chains in parallel:

```python
async def process_document(document: str):
    # Run multiple analyses concurrently
    summary, keywords, sentiment = await asyncio.gather(
        summary_chain.ainvoke({"text": document}),
        keyword_chain.ainvoke({"text": document}),
        sentiment_chain.ainvoke({"text": document}),
    )
    
    return {
        "summary": summary.content,
        "keywords": keywords.content,
        "sentiment": sentiment.content,
    }
```

**With Error Handling**:

```python
async def process_with_fallback(document: str):
    # Continue even if some chains fail
    results = await asyncio.gather(
        summary_chain.ainvoke({"text": document}),
        keyword_chain.ainvoke({"text": document}),
        sentiment_chain.ainvoke({"text": document}),
        return_exceptions=True,  # Don't fail on first error
    )
    
    # Handle results and exceptions
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(f"Error: {result}")
        else:
            processed_results.append(result.content)
    
    return processed_results
```

### Rate Limiting with Semaphores

Control concurrency to avoid rate limits or overwhelming external services:

```python
import asyncio
from langchain_openai import ChatOpenAI

async def process_with_rate_limit(items: list[str], max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_one(item: str):
        async with semaphore:  # Limit concurrent executions
            result = await chain.ainvoke({"input": item})
            return result
    
    # Create all tasks
    tasks = [process_one(item) for item in items]
    
    # Execute with concurrency limit
    results = await asyncio.gather(*tasks)
    return results

# Process 100 items with max 5 concurrent requests
results = await process_with_rate_limit(large_item_list, max_concurrent=5)
```

**Using RunnableConfig max_concurrency**:

```python
# LangChain's built-in concurrency control
config = {"max_concurrency": 10}
results = await chain.abatch(inputs, config=config)
```

### Concurrent Document Processing Example

**Real-World Scenario**: Process 100 documents with multiple analysis chains:

```python
import asyncio
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Create analysis chains
summarize = ChatPromptTemplate.from_template("Summarize: {text}") | ChatOpenAI()
extract_entities = ChatPromptTemplate.from_template("Extract entities: {text}") | ChatOpenAI()
categorize = ChatPromptTemplate.from_template("Categorize: {text}") | ChatOpenAI()

async def analyze_document(doc: str, semaphore: asyncio.Semaphore) -> Dict:
    """Analyze a single document with multiple chains concurrently."""
    async with semaphore:
        # Run all analyses for this document in parallel
        summary, entities, category = await asyncio.gather(
            summarize.ainvoke({"text": doc}),
            extract_entities.ainvoke({"text": doc}),
            categorize.ainvoke({"text": doc}),
        )
        
        return {
            "summary": summary.content,
            "entities": entities.content,
            "category": category.content,
        }

async def process_corpus(documents: List[str], max_concurrent: int = 10):
    """Process entire corpus with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = [analyze_document(doc, semaphore) for doc in documents]
    results = await asyncio.gather(*tasks)
    
    return results

# Process 100 documents
documents = [f"Document {i} content..." for i in range(100)]
results = await process_corpus(documents, max_concurrent=10)
```

**Performance Characteristics**:
- Sequential: 100 docs × 3 chains × 1 sec = 300 seconds
- Concurrent (10 parallel): ~30 seconds (10x speedup)

---

## Event Loop Management

### Simple Scripts with `asyncio.run()`

For standalone Python scripts, use `asyncio.run()` to handle event loop management:

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

async def main():
    """Main async entry point."""
    chain = ChatPromptTemplate.from_template("Explain {topic}") | ChatOpenAI()
    
    result = await chain.ainvoke({"topic": "asyncio"})
    print(result.content)

if __name__ == "__main__":
    # asyncio.run() handles event loop lifecycle
    asyncio.run(main())
```

**What `asyncio.run()` Provides**:
- Creates fresh event loop
- Runs your coroutine
- Cancels remaining tasks
- Closes the loop
- Cleans up resources

### Web Applications (FastAPI)

FastAPI and similar frameworks manage the event loop for you. Simply use `await` in your route handlers:

```python
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Create chain once at startup
chain = ChatPromptTemplate.from_template("Answer: {question}") | ChatOpenAI()

@app.post("/ask")
async def ask_question(question: str):
    """FastAPI manages the event loop - just use await."""
    try:
        response = await chain.ainvoke({"question": question})
        return {"answer": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_questions(questions: List[str]):
    """Process multiple questions concurrently."""
    inputs = [{"question": q} for q in questions]
    responses = await chain.abatch(inputs)
    return {"answers": [r.content for r in responses]}

# Run with: uvicorn app:app --reload
```

**Key Points**:
- **No `asyncio.run()`**: FastAPI's event loop is already running
- **Use `await` directly**: In async route handlers
- **Automatic concurrency**: FastAPI handles concurrent requests
- **Lifespan events**: Use for setup/cleanup:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup and cleanup."""
    # Startup: Initialize resources
    print("Starting up...")
    yield
    # Shutdown: Cleanup resources
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)
```

### Jupyter Notebooks

Jupyter and IPython have a running event loop, enabling top-level `await`:

```python
# In Jupyter cell - works directly!
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
response = await model.ainvoke("What is machine learning?")
print(response.content)
```

**Streaming in Jupyter**:

```python
# Stream tokens in notebook
async for chunk in model.astream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

**Handling "Event Loop Already Running" Errors**:

If you have code that uses `asyncio.run()` and need to run it in Jupyter:

```python
import nest_asyncio
nest_asyncio.apply()

# Now asyncio.run() works in Jupyter
import asyncio

async def my_function():
    return await model.ainvoke("test")

result = asyncio.run(my_function())  # Works with nest_asyncio
```

**Best Practice for Jupyter**: Write code to use `await` directly rather than `asyncio.run()`, making it compatible with both Jupyter and regular Python scripts.

### Multiple Event Loops (Advanced)

**Rule**: Avoid multiple event loops in the same thread. Python's asyncio is designed for one loop per thread.

**Anti-Pattern** (Don't Do This):

```python
# ❌ BAD - Multiple loops in same thread
async def task1():
    await chain1.ainvoke(input1)

async def task2():
    await chain2.ainvoke(input2)

# Creating separate loops - DON'T DO THIS
loop1 = asyncio.new_event_loop()
loop1.run_until_complete(task1())

loop2 = asyncio.new_event_loop()
loop2.run_until_complete(task2())
```

**Correct Pattern** (Use Single Loop):

```python
# ✅ GOOD - Single loop, concurrent tasks
async def main():
    # Run both tasks concurrently in same loop
    result1, result2 = await asyncio.gather(
        chain1.ainvoke(input1),
        chain2.ainvoke(input2),
    )

asyncio.run(main())
```

**Threading with Async** (If You Really Need It):

```python
import threading
import asyncio

def run_in_thread(coro):
    """Run async function in separate thread with its own loop."""
    def target():
        asyncio.run(coro)
    
    thread = threading.Thread(target=target)
    thread.start()
    return thread

# Run async tasks in different threads
thread1 = run_in_thread(task1())
thread2 = run_in_thread(task2())

thread1.join()
thread2.join()
```

**When This Might Be Useful**:
- Integrating async code into sync frameworks
- Running async operations in background threads
- Testing async code

---

## Resource Management with Async

### Async Context Managers

Use async context managers (`async with`) for proper resource cleanup:

**Database Connections**:

```python
import asyncpg

async def query_with_connection():
    # Context manager ensures connection is closed
    async with asyncpg.create_pool("postgresql://...") as pool:
        async with pool.acquire() as conn:
            result = await conn.fetch("SELECT * FROM users")
            return result
    # Pool automatically closed here
```

**HTTP Sessions**:

```python
import aiohttp

async def fetch_data():
    # Context manager ensures session is closed
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com/data") as response:
            data = await response.json()
            return data
    # Session automatically closed here
```

### Managing Resources in LangChain Chains

**Custom Callback with Resource Management**:

```python
from langchain_core.callbacks import AsyncCallbackHandler
import aiohttp

class APIMonitoringCallback(AsyncCallbackHandler):
    """Callback that manages HTTP session lifecycle."""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session = None
    
    async def __aenter__(self):
        """Setup: Create HTTP session."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup: Close HTTP session."""
        if self.session:
            await self.session.close()
    
    async def on_chain_end(self, outputs, **kwargs):
        """Send data to monitoring API."""
        if self.session:
            await self.session.post(self.api_url, json=outputs)

# Usage with context manager
async def main():
    async with APIMonitoringCallback("https://monitor.example.com") as callback:
        result = await chain.ainvoke(
            {"input": "test"},
            config={"callbacks": [callback]}
        )
    # Session automatically cleaned up here
```

### Properly Closing Async Resources

**Problem**: Resources not properly closed can lead to warnings:

```python
# ❌ BAD - ResourceWarning: unclosed <socket.socket...>
async def bad_example():
    session = aiohttp.ClientSession()
    async with session.get("https://api.example.com") as resp:
        data = await resp.text()
    # Session never closed!
    return data
```

**Solution 1**: Use context manager:

```python
# ✅ GOOD - Session closed automatically
async def good_example():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as resp:
            data = await resp.text()
    return data  # Session closed here
```

**Solution 2**: Explicit close:

```python
# ✅ GOOD - Explicit close with try/finally
async def explicit_close():
    session = aiohttp.ClientSession()
    try:
        async with session.get("https://api.example.com") as resp:
            data = await resp.text()
        return data
    finally:
        await session.close()  # Always closes
```

### Long-Lived Connections in Applications

**FastAPI Example with Connection Pool**:

```python
from fastapi import FastAPI
import asyncpg

app = FastAPI()

# Global connection pool
db_pool = None

@app.on_event("startup")
async def startup():
    """Create connection pool on startup."""
    global db_pool
    db_pool = await asyncpg.create_pool(
        "postgresql://user:password@localhost/dbname",
        min_size=10,
        max_size=50,
    )

@app.on_event("shutdown")
async def shutdown():
    """Close pool on shutdown."""
    global db_pool
    if db_pool:
        await db_pool.close()

@app.get("/query")
async def execute_query():
    """Use pool from requests."""
    async with db_pool.acquire() as conn:
        result = await conn.fetch("SELECT * FROM data")
        return {"data": result}
```

---

## Performance Optimization

### Benefits of Async for LangChain

**Scenario**: Processing 10 independent LLM requests

**Sequential (Sync) Execution**:
```
Request 1 → Response 1 (1s)
Request 2 → Response 2 (1s)
...
Request 10 → Response 10 (1s)
Total Time: 10 seconds
```

**Concurrent (Async) Execution**:
```
Requests 1-10 → (all sent simultaneously)
Responses 1-10 ← (received as they complete)
Total Time: ~1 second (limited by slowest request)
```

**Performance Measurement**:

```python
import asyncio
import time
from langchain_openai import ChatOpenAI

model = ChatOpenAI()

async def measure_performance():
    inputs = [f"Explain concept {i}" for i in range(10)]
    
    # Sequential sync execution
    start = time.time()
    sync_results = [model.invoke(inp) for inp in inputs]
    sync_time = time.time() - start
    
    # Concurrent async execution
    start = time.time()
    async_results = await model.abatch(inputs)
    async_time = time.time() - start
    
    print(f"Sync time: {sync_time:.2f}s")
    print(f"Async time: {async_time:.2f}s")
    print(f"Speedup: {sync_time / async_time:.2f}x")

# Example output:
# Sync time: 12.43s
# Async time: 1.87s
# Speedup: 6.65x
```

### Batching Strategies

**Optimal Batch Sizes**:

```python
async def process_large_dataset(items: list[str], batch_size: int = 50):
    """Process items in optimal batch sizes."""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await chain.abatch(batch)
        results.extend(batch_results)
        
        # Optional: Add delay to avoid rate limits
        await asyncio.sleep(1)
    
    return results

# Process 1000 items in batches of 50
all_results = await process_large_dataset(large_list, batch_size=50)
```

**Dynamic Batching Based on Rate Limits**:

```python
import asyncio
from datetime import datetime, timedelta

class RateLimitedBatcher:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
    
    async def process_batch(self, items: list):
        """Process batch while respecting rate limits."""
        # Remove old request times
        cutoff = datetime.now() - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        # Calculate how many requests we can make
        available = self.requests_per_minute - len(self.request_times)
        
        if available <= 0:
            # Wait until we can make more requests
            wait_time = (self.request_times[0] - cutoff).total_seconds()
            await asyncio.sleep(wait_time)
            return await self.process_batch(items)
        
        # Process up to available slots
        batch = items[:available]
        results = await chain.abatch(batch)
        
        # Record request times
        now = datetime.now()
        self.request_times.extend([now] * len(batch))
        
        return results

# Usage
batcher = RateLimitedBatcher(requests_per_minute=60)
results = await batcher.process_batch(items)
```

### Streaming for Better User Experience

Streaming provides immediate feedback, improving perceived performance:

**CLI with Streaming**:

```python
async def stream_to_console():
    """Stream response with immediate display."""
    print("AI: ", end="", flush=True)
    
    async for chunk in chain.astream({"input": "Write a poem"}):
        print(chunk.content, end="", flush=True)
    
    print()  # New line at end

asyncio.run(stream_to_console())
```

**Web Application with Server-Sent Events (SSE)**:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
async def stream_response(prompt: str):
    """Stream LLM response using SSE."""
    async def generate():
        async for chunk in chain.astream({"input": prompt}):
            # SSE format
            yield f"data: {chunk.content}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

**WebSocket Streaming**:

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        # Receive message
        data = await websocket.receive_text()
        
        # Stream response
        async for chunk in chain.astream({"input": data}):
            await websocket.send_text(chunk.content)
        
        # Send completion marker
        await websocket.send_text("[DONE]")
```

### Concurrent API Calls

**Multiple Model Providers Concurrently**:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

async def compare_models(prompt: str):
    """Get responses from multiple models concurrently."""
    openai_model = ChatOpenAI()
    anthropic_model = ChatAnthropic()
    
    # Run both models concurrently
    openai_response, anthropic_response = await asyncio.gather(
        openai_model.ainvoke(prompt),
        anthropic_model.ainvoke(prompt),
    )
    
    return {
        "openai": openai_response.content,
        "anthropic": anthropic_response.content,
    }

# Get responses from both models in parallel
results = await compare_models("What is quantum computing?")
```

---

## Common Pitfalls

### 1. Blocking Sync Calls in Async Context

**Problem**: Using blocking synchronous operations inside async functions blocks the entire event loop:

```python
# ❌ WRONG - Blocks event loop
async def bad_callback():
    import requests
    response = requests.get("https://api.example.com")  # Blocks!
    
    import time
    time.sleep(1)  # Blocks!
    
    with open("file.txt") as f:  # Blocks!
        data = f.read()
```

**Solution**: Use async equivalents:

```python
# ✅ CORRECT - Non-blocking
async def good_callback():
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as resp:
            data = await resp.json()  # Doesn't block
    
    await asyncio.sleep(1)  # Doesn't block
    
    import aiofiles
    async with aiofiles.open("file.txt") as f:
        data = await f.read()  # Doesn't block
```

**Library Replacements**:
- `requests` → `aiohttp` or `httpx`
- `time.sleep()` → `asyncio.sleep()`
- `open()` → `aiofiles.open()`
- `psycopg2` → `asyncpg`
- `pymongo` → `motor`
- `pymysql` → `aiomysql`

### 2. Event Loop Conflicts

**Problem 1**: Calling `asyncio.run()` from within async function:

```python
# ❌ WRONG
async def outer():
    async def inner():
        return "result"
    
    result = asyncio.run(inner())  # ERROR: Can't call asyncio.run() here
    return result
```

**Solution**: Use `await` instead:

```python
# ✅ CORRECT
async def outer():
    async def inner():
        return "result"
    
    result = await inner()  # Use await
    return result
```

**Problem 2**: Jupyter/IPython event loop conflict:

```python
# ❌ ERROR in Jupyter
async def my_func():
    return await chain.ainvoke(input)

asyncio.run(my_func())  # RuntimeError: This event loop is already running
```

**Solution**: Use top-level await in Jupyter:

```python
# ✅ CORRECT in Jupyter
async def my_func():
    return await chain.ainvoke(input)

result = await my_func()  # Works in Jupyter
```

### 3. Callback Thread Safety Issues

**Problem**: Sharing mutable state between callbacks without synchronization:

```python
# ❌ POTENTIALLY UNSAFE
class StatefulCallback(AsyncCallbackHandler):
    def __init__(self):
        self.token_count = 0  # Shared state
    
    async def on_llm_new_token(self, token, **kwargs):
        self.token_count += 1  # Race condition with concurrent calls
```

**Solution**: Use asyncio locks:

```python
# ✅ CORRECT - Thread-safe
class SafeStatefulCallback(AsyncCallbackHandler):
    def __init__(self):
        self.token_count = 0
        self.lock = asyncio.Lock()
    
    async def on_llm_new_token(self, token, **kwargs):
        async with self.lock:
            self.token_count += 1  # Protected by lock
```

### 4. Forgotten `await`

**Problem**: Forgetting to `await` async functions:

```python
# ❌ WRONG - Returns coroutine object, doesn't execute
async def process():
    result = chain.ainvoke({"input": "test"})  # Missing await!
    return result  # Returns coroutine, not actual result

# Warning: coroutine 'ainvoke' was never awaited
```

**Solution**: Always `await` async calls:

```python
# ✅ CORRECT
async def process():
    result = await chain.ainvoke({"input": "test"})  # Await it!
    return result  # Returns actual result
```

**Detection**: Python will warn about unawaited coroutines. Enable warnings:

```python
import warnings
warnings.simplefilter('always', RuntimeWarning)
```

### 5. Mixing Sync and Async

**Problem**: Calling sync methods from async functions loses concurrency benefits:

```python
# ⚠️ WORKS BUT SUBOPTIMAL
async def mixed_approach():
    # These run sequentially, not concurrently
    result1 = chain.invoke(input1)  # Sync call - blocks
    result2 = chain.invoke(input2)  # Sync call - blocks
    return result1, result2
```

**Solution**: Use async methods for concurrency:

```python
# ✅ CORRECT - Truly concurrent
async def async_approach():
    # These run concurrently
    result1, result2 = await asyncio.gather(
        chain.ainvoke(input1),  # Async call
        chain.ainvoke(input2),  # Async call
    )
    return result1, result2
```

---

## Troubleshooting Async Issues

### "RuntimeError: Event loop is closed"

**Cause**: Trying to use an event loop after it's been closed.

**Common Scenario**:
```python
# ❌ PROBLEM
loop = asyncio.get_event_loop()
result = loop.run_until_complete(chain.ainvoke(input))
loop.close()  # Close the loop

# Try to reuse
result = loop.run_until_complete(chain.ainvoke(input))  # ERROR!
```

**Solution 1**: Use `asyncio.run()` (creates new loop each time):

```python
# ✅ CORRECT
result = asyncio.run(chain.ainvoke(input))
# Loop is created and closed automatically

result = asyncio.run(chain.ainvoke(input))
# New loop created for second call
```

**Solution 2**: Keep loop open or create new one:

```python
# ✅ CORRECT
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

try:
    result = loop.run_until_complete(chain.ainvoke(input))
finally:
    loop.close()
```

### "RuntimeError: This event loop is already running"

**Cause**: Calling `asyncio.run()` when an event loop is already running (common in Jupyter/IPython).

**Scenario**:
```python
# In Jupyter notebook
async def my_function():
    return await chain.ainvoke(input)

asyncio.run(my_function())  # ERROR: Loop already running
```

**Solution 1**: Use top-level await in Jupyter:

```python
# ✅ CORRECT in Jupyter
async def my_function():
    return await chain.ainvoke(input)

result = await my_function()  # Works!
```

**Solution 2**: Use `nest_asyncio`:

```python
# ✅ CORRECT - Allows nested event loops
import nest_asyncio
nest_asyncio.apply()

asyncio.run(my_function())  # Now works in Jupyter
```

**Solution 3**: FastAPI/existing loop - use `await`:

```python
# ✅ CORRECT in FastAPI
@app.get("/process")
async def process():
    # Don't use asyncio.run() here!
    result = await chain.ainvoke(input)  # Use await directly
    return result
```

### "Coroutine 'ainvoke' was never awaited"

**Cause**: Calling async function without `await`.

**Problem**:
```python
# ❌ WRONG
async def process():
    result = chain.ainvoke({"input": "test"})  # Missing await!
    print(result)  # Prints: <coroutine object ainvoke at 0x...>

# RuntimeWarning: coroutine 'ainvoke' was never awaited
```

**Solution**: Add `await`:

```python
# ✅ CORRECT
async def process():
    result = await chain.ainvoke({"input": "test"})  # Add await!
    print(result)  # Prints actual result
```

**Detection Tips**:
- Look for type hints: Functions returning coroutines should be awaited
- Check return type: If it says `<coroutine ...>`, you forgot `await`
- IDE warnings: Modern IDEs highlight unawaited coroutines

### Slow Async Performance

**Symptom**: Async code runs as slow as sync code.

**Cause 1**: Blocking calls in async code:

```python
# ❌ PROBLEM - requests.get() blocks the event loop
async def slow():
    import requests
    result = requests.get("https://api.example.com")  # Blocks!
    return result
```

**Solution**: Use async HTTP client:

```python
# ✅ FIX
async def fast():
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as resp:
            return await resp.text()
```

**Cause 2**: Sequential processing instead of concurrent:

```python
# ❌ PROBLEM - Sequential, no concurrency
async def slow_batch():
    results = []
    for item in items:
        result = await chain.ainvoke(item)  # One at a time
        results.append(result)
    return results
```

**Solution**: Use `abatch()` or `gather()`:

```python
# ✅ FIX - Concurrent processing
async def fast_batch():
    results = await chain.abatch(items)  # All concurrent
    return results
```

**Cause 3**: CPU-bound operations:

```python
# ⚠️ ASYNC DOESN'T HELP - CPU-bound
async def compute_heavy():
    result = expensive_calculation()  # Uses CPU, not I/O
    return result
```

**Solution**: Use thread/process pool for CPU-bound work:

```python
# ✅ FIX - Use thread pool
import concurrent.futures

async def compute_with_executor():
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, expensive_calculation)
    return result
```

**Profiling Async Code**:

```python
import asyncio
import time

async def profile_performance():
    start = time.time()
    
    result = await chain.ainvoke(input)
    
    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f}s")

# Enable asyncio debug mode
asyncio.run(profile_performance(), debug=True)
```

### Callbacks Not Firing

**Symptom**: Async callbacks don't execute.

**Cause 1**: Using `BaseCallbackHandler` instead of `AsyncCallbackHandler`:

```python
# ❌ PROBLEM - Wrong base class
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):  # Should be AsyncCallbackHandler
    async def on_chain_start(self, **kwargs):
        print("Started")  # Won't be called for async chains
```

**Solution**: Use `AsyncCallbackHandler`:

```python
# ✅ FIX
from langchain_core.callbacks import AsyncCallbackHandler

class MyCallback(AsyncCallbackHandler):  # Correct base class
    async def on_chain_start(self, **kwargs):
        print("Started")  # Now works!
```

**Cause 2**: Not passing callbacks in config:

```python
# ❌ PROBLEM - Callback not passed
callback = MyCallback()
result = await chain.ainvoke(input)  # Callback not used
```

**Solution**: Pass via config:

```python
# ✅ FIX
callback = MyCallback()
result = await chain.ainvoke(
    input,
    config={"callbacks": [callback]}  # Pass callbacks
)
```

---

## Practical Examples

### Example 1: FastAPI Endpoint with Async Chain

Complete working example of FastAPI integration:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="LangChain Async API")

# Create chain once at module level
chain = (
    ChatPromptTemplate.from_template("Answer this question: {question}")
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | StrOutputParser()
)

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@app.post("/ask", response_model=Answer)
async def ask_question(q: Question):
    """Process single question asynchronously."""
    try:
        answer = await chain.ainvoke({"question": q.question})
        return Answer(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-batch", response_model=list[Answer])
async def ask_batch(questions: list[Question]):
    """Process multiple questions concurrently."""
    try:
        inputs = [{"question": q.question} for q in questions]
        answers = await chain.abatch(inputs)
        return [Answer(answer=a) for a in answers]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn app:app --reload
```

### Example 2: Jupyter Notebook Async Usage

```python
# Jupyter Notebook Cell 1: Setup
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Explain {topic} in simple terms")
    | ChatOpenAI()
    | StrOutputParser()
)

# Cell 2: Single async call (top-level await works in Jupyter)
result = await chain.ainvoke({"topic": "quantum entanglement"})
print(result)

# Cell 3: Streaming response
print("AI: ", end="")
async for chunk in chain.astream({"topic": "black holes"}):
    print(chunk, end="", flush=True)
print()

# Cell 4: Batch processing
topics = ["dark matter", "string theory", "multiverse"]
inputs = [{"topic": t} for t in topics]
results = await chain.abatch(inputs)

for topic, result in zip(topics, results):
    print(f"\n{topic.upper()}:")
    print(result[:200] + "...")
```

### Example 3: Concurrent Batch Processing

Processing 100 documents with progress tracking:

```python
import asyncio
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

async def process_documents_with_progress(
    documents: List[str],
    batch_size: int = 10
) -> List[str]:
    """Process documents in batches with progress tracking."""
    chain = (
        ChatPromptTemplate.from_template("Summarize: {text}")
        | ChatOpenAI()
    )
    
    total = len(documents)
    results = []
    
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        batch_inputs = [{"text": doc} for doc in batch]
        
        print(f"Processing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}...")
        
        batch_results = await chain.abatch(batch_inputs)
        results.extend([r.content for r in batch_results])
        
        print(f"Completed {min(i + batch_size, total)}/{total} documents")
    
    return results

# Usage
async def main():
    documents = [f"Document {i} content..." for i in range(100)]
    summaries = await process_documents_with_progress(documents, batch_size=10)
    print(f"\nProcessed {len(summaries)} documents successfully")

asyncio.run(main())
```

### Example 4: Async Tool with API Calls

Custom async tool using aiohttp:

```python
import aiohttp
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class WeatherInput(BaseModel):
    location: str = Field(description="City name or location")

class AsyncWeatherTool(BaseTool):
    name: str = "weather"
    description: str = "Get current weather for a location"
    args_schema: Type[BaseModel] = WeatherInput
    api_key: str
    
    def _run(self, location: str) -> str:
        """Sync version not implemented."""
        raise NotImplementedError("Use async version")
    
    async def _arun(self, location: str) -> str:
        """Async implementation using aiohttp."""
        async with aiohttp.ClientSession() as session:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    temp = data["main"]["temp"]
                    description = data["weather"][0]["description"]
                    return f"Weather in {location}: {temp}°C, {description}"
                else:
                    return f"Error: Could not fetch weather for {location}"

# Usage with agent
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

async def run_weather_agent():
    tool = AsyncWeatherTool(api_key="your-api-key")
    
    llm = ChatOpenAI(model="gpt-4")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can check weather."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_openai_tools_agent(llm, [tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[tool])
    
    result = await agent_executor.ainvoke({
        "input": "What's the weather like in Paris and London?"
    })
    
    print(result["output"])

asyncio.run(run_weather_agent())
```

### Example 5: Async Callback for Monitoring

Complete async callback with database logging:

```python
import asyncio
from datetime import datetime
from langchain_core.callbacks import AsyncCallbackHandler
from typing import Any, Dict
from uuid import UUID

class AsyncDatabaseLogger(AsyncCallbackHandler):
    """Logs chain execution to database asynchronously."""
    
    def __init__(self, db_connection_string: str):
        self.db_string = db_connection_string
        self.pool = None
    
    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log chain start."""
        if not self.pool:
            import asyncpg
            self.pool = await asyncpg.create_pool(self.db_string)
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chain_logs (run_id, event_type, timestamp, data)
                VALUES ($1, $2, $3, $4)
                """,
                str(run_id),
                "chain_start",
                datetime.now(),
                str(inputs)
            )
    
    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log chain completion."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chain_logs (run_id, event_type, timestamp, data)
                VALUES ($1, $2, $3, $4)
                """,
                str(run_id),
                "chain_end",
                datetime.now(),
                str(outputs)
            )
    
    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log chain errors."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chain_logs (run_id, event_type, timestamp, data)
                VALUES ($1, $2, $3, $4)
                """,
                str(run_id),
                "chain_error",
                datetime.now(),
                str(error)
            )
    
    async def cleanup(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()

# Usage
async def main():
    logger = AsyncDatabaseLogger("postgresql://localhost/langchain_logs")
    
    try:
        result = await chain.ainvoke(
            {"input": "test"},
            config={"callbacks": [logger]}
        )
        print(result)
    finally:
        await logger.cleanup()

asyncio.run(main())
```

---

## Best Practices

### 1. Use Async Throughout the Stack

For maximum benefit, use async from top to bottom:

```python
# ✅ FULL ASYNC STACK
# - Async web framework (FastAPI)
# - Async chains (ainvoke, abatch, astream)
# - Async callbacks (AsyncCallbackHandler)
# - Async tools (async _arun methods)
# - Async database/HTTP clients (asyncpg, aiohttp)

@app.post("/process")
async def endpoint():
    async with AsyncCallbackHandler() as callback:
        async with aiohttp.ClientSession() as session:
            result = await chain.ainvoke(
                input,
                config={"callbacks": [callback]}
            )
    return result
```

### 2. Choose Async Drivers

Always use async-compatible libraries for I/O operations:

| Operation | Sync Library | Async Alternative |
|-----------|--------------|-------------------|
| HTTP requests | `requests` | `aiohttp`, `httpx` |
| PostgreSQL | `psycopg2` | `asyncpg` |
| MySQL | `pymysql` | `aiomysql` |
| MongoDB | `pymongo` | `motor` |
| File I/O | `open()` | `aiofiles` |
| Sleep | `time.sleep()` | `asyncio.sleep()` |

### 3. Limit Concurrency with Semaphores

Protect against rate limits and resource exhaustion:

```python
# ✅ GOOD - Controlled concurrency
config = {"max_concurrency": 10}
results = await chain.abatch(inputs, config=config)

# Or use semaphore directly
semaphore = asyncio.Semaphore(10)

async def process_with_limit(item):
    async with semaphore:
        return await chain.ainvoke(item)

results = await asyncio.gather(*[process_with_limit(i) for i in inputs])
```

### 4. Stream Long-Running Chains

Provide immediate feedback for better UX:

```python
# ✅ GOOD - Streaming for long responses
async def stream_to_user():
    async for chunk in chain.astream(input):
        yield chunk.content  # Immediate display
        
# Instead of
async def wait_for_complete():
    result = await chain.ainvoke(input)  # User waits entire time
    return result
```

### 5. Test Async Code with pytest-asyncio

```python
# test_chains.py
import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

@pytest.mark.asyncio
async def test_chain_async():
    """Test async chain execution."""
    chain = ChatPromptTemplate.from_template("Echo: {text}") | ChatOpenAI()
    
    result = await chain.ainvoke({"text": "hello"})
    
    assert result.content
    assert isinstance(result.content, str)

@pytest.mark.asyncio
async def test_batch_async():
    """Test async batch processing."""
    chain = ChatPromptTemplate.from_template("Echo: {text}") | ChatOpenAI()
    
    inputs = [{"text": f"test{i}"} for i in range(5)]
    results = await chain.abatch(inputs)
    
    assert len(results) == 5
    for result in results:
        assert result.content
```

### 6. Profile with Asyncio Debug Mode

Enable debug mode to catch common issues:

```python
import asyncio
import logging

# Enable debug mode
asyncio.run(main(), debug=True)

# Or set environment variable
# PYTHONASYNCIODEBUG=1 python script.py

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Cross-References

### Related Documentation

- **[Runnable Protocol API Reference](../api-reference/runnables/base.md)**: Complete API documentation for `ainvoke()`, `abatch()`, `astream()` methods
- **[Callbacks Guide](./callbacks.md)**: Detailed callback system documentation including async callback patterns
- **[Agent Development Guide](./agent-development.md)**: Agent implementation with async tool execution
- **[Glossary](../glossary.md)**: Definitions of key terms used in this guide

### Example Code

- **[examples/advanced_chains/async_chain_execution.py](../../examples/advanced_chains/async_chain_execution.py)**: Complete executable examples demonstrating async patterns

### External Resources

- **Python asyncio Documentation**: https://docs.python.org/3/library/asyncio.html
- **FastAPI Async Documentation**: https://fastapi.tiangolo.com/async/
- **aiohttp Documentation**: https://docs.aiohttp.org/
- **pytest-asyncio**: https://pytest-asyncio.readthedocs.io/

---

**Source References**:
- `libs/core/langchain_core/runnables/base.py:122-254` - Runnable class with async methods
- `libs/core/langchain_core/runnables/base.py:830-849` - `ainvoke()` implementation
- `libs/core/langchain_core/runnables/base.py:983-1027` - `abatch()` implementation
- `libs/core/langchain_core/runnables/base.py:1128-1147` - `astream()` implementation
- `libs/core/langchain_core/callbacks/base.py:478-580` - `AsyncCallbackHandler` class
- `libs/langchain/langchain_classic/chains/base.py:187` - Chain `ainvoke()` method

---

*Last Updated: 2024*
*Documentation Version: 1.0*
