"""
Async Chain Execution Example

This example demonstrates comprehensive async/await patterns for LangChain chain execution.
It covers async invocation, batch processing, streaming, concurrent execution, and proper
event loop management per Agent Action Plan sections 0.3.1, 0.4.1, and 0.9.2.

Key Concepts Demonstrated:
- ainvoke() vs invoke() - When to use async methods
- abatch() for concurrent batch processing
- astream() for real-time token streaming
- asyncio.gather() and create_task() for parallel execution
- Async callback handler implementation
- Proper event loop management
- Error handling in async contexts
- Performance comparison: async vs sync execution

Prerequisites:
- Python >=3.10
- OPENAI_API_KEY environment variable (or will use mock mode)
- Dependencies: langchain-core, langchain-openai, python-dotenv

Source: examples/advanced_chains/async_chain_execution.py
Based on patterns from: libs/core/tests/unit_tests/runnables/test_runnable.py
"""

import asyncio
import os
import time
from typing import Any, Dict, List
from collections.abc import AsyncIterator

# LangChain Core imports for LCEL composition
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun

# Environment setup for API keys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if API key is available
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))


# ============================================================================
# Mock LLM for demonstration when API key not available
# ============================================================================

class MockAsyncLLM(Runnable[str, str]):
    """
    Mock async LLM for demonstration purposes when API key is not available.
    
    This mock LLM simulates async behavior including artificial delays to
    demonstrate async patterns without requiring actual API calls.
    """
    
    def __init__(self, response_delay: float = 0.1):
        """
        Initialize mock LLM.
        
        Args:
            response_delay: Simulated API latency in seconds
        """
        self.response_delay = response_delay
        self._call_count = 0
    
    def invoke(self, input: str, config: Dict[str, Any] | None = None) -> str:
        """Sync invoke - simulates blocking API call."""
        time.sleep(self.response_delay)
        self._call_count += 1
        return f"Mock response #{self._call_count} to: {input[:50]}"
    
    async def ainvoke(
        self, 
        input: str, 
        config: Dict[str, Any] | None = None
    ) -> str:
        """
        Async invoke - simulates non-blocking API call.
        
        This demonstrates the key difference: async operations don't block
        the event loop, allowing other tasks to run concurrently.
        """
        await asyncio.sleep(self.response_delay)
        self._call_count += 1
        return f"Mock async response #{self._call_count} to: {input[:50]}"
    
    async def astream(
        self, 
        input: str, 
        config: Dict[str, Any] | None = None
    ) -> AsyncIterator[str]:
        """Async stream - simulates token-by-token streaming."""
        response = await self.ainvoke(input, config)
        words = response.split()
        for word in words:
            await asyncio.sleep(0.01)  # Simulate streaming delay
            yield word + " "


# ============================================================================
# Async Callback Handler Implementation
# ============================================================================

class DetailedAsyncCallbackHandler(AsyncCallbackHandler):
    """
    Custom async callback handler for detailed chain execution monitoring.
    
    This demonstrates how to implement async callbacks for logging,
    monitoring, or debugging chain execution. All callback methods are
    async and don't block the event loop.
    
    Source: libs/core/langchain_core/callbacks/base.py
    """
    
    def __init__(self):
        """Initialize callback handler with tracking state."""
        self.events: List[Dict[str, Any]] = []
        self.start_times: Dict[str, float] = {}
    
    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """
        Called when LLM starts processing.
        
        Args:
            serialized: Serialized LLM information
            prompts: Input prompts being processed
            **kwargs: Additional context (run_id, parent_run_id, tags, etc.)
        """
        run_id = kwargs.get("run_id", "unknown")
        self.start_times[str(run_id)] = time.time()
        self.events.append({
            "event": "llm_start",
            "run_id": str(run_id),
            "prompts": [p[:50] + "..." if len(p) > 50 else p for p in prompts],
            "timestamp": time.time()
        })
        print(f"🚀 LLM Start [Run: {str(run_id)[:8]}] - Processing {len(prompts)} prompt(s)")
    
    async def on_llm_end(
        self,
        response: Any,
        **kwargs: Any,
    ) -> None:
        """
        Called when LLM finishes processing.
        
        Args:
            response: LLM output result
            **kwargs: Additional context including run_id
        """
        run_id = kwargs.get("run_id", "unknown")
        run_id_str = str(run_id)
        duration = time.time() - self.start_times.get(run_id_str, time.time())
        
        self.events.append({
            "event": "llm_end",
            "run_id": run_id_str,
            "duration": duration,
            "timestamp": time.time()
        })
        print(f"✅ LLM End [Run: {run_id_str[:8]}] - Completed in {duration:.3f}s")
    
    async def on_llm_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """
        Called when LLM encounters an error.
        
        Args:
            error: Exception that occurred
            **kwargs: Additional context including run_id
        """
        run_id = kwargs.get("run_id", "unknown")
        self.events.append({
            "event": "llm_error",
            "run_id": str(run_id),
            "error": str(error),
            "timestamp": time.time()
        })
        print(f"❌ LLM Error [Run: {str(run_id)[:8]}] - {error}")


# ============================================================================
# Example 1: Basic Async Invocation
# ============================================================================

async def example_basic_async_invoke():
    """
    Demonstrate basic async chain invocation with ainvoke().
    
    Key Concepts:
    - Use ainvoke() instead of invoke() for async operation
    - await keyword for non-blocking execution
    - Proper asyncio.run() in main() function
    
    When to use async:
    - IO-bound operations (API calls, database queries)
    - Need to run multiple operations concurrently
    - Building async web applications (FastAPI, async frameworks)
    """
    print("\n" + "="*70)
    print("Example 1: Basic Async Invocation")
    print("="*70)
    
    # Create a simple LCEL chain
    prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
    
    if HAS_OPENAI_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    else:
        llm = MockAsyncLLM(response_delay=0.2)
    
    parser = StrOutputParser()
    
    # LCEL composition: prompt | llm | parser
    # This creates a Runnable chain supporting both sync and async
    chain = prompt | llm | parser
    
    # Async invocation - non-blocking
    print("\n📝 Invoking chain asynchronously...")
    result = await chain.ainvoke({"topic": "programming"})
    print(f"Result: {result}\n")
    
    return result


# ============================================================================
# Example 2: Async Batch Processing
# ============================================================================

async def example_async_batch():
    """
    Demonstrate async batch processing with abatch().
    
    Key Concepts:
    - abatch() processes multiple inputs concurrently
    - Much faster than sequential processing for IO-bound operations
    - All requests execute in parallel (up to rate limits)
    
    Performance benefit: N requests take ~same time as 1 request
    (vs N * request_time for sequential processing)
    """
    print("\n" + "="*70)
    print("Example 2: Async Batch Processing")
    print("="*70)
    
    prompt = PromptTemplate.from_template("What is {number} + {number}?")
    
    if HAS_OPENAI_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo")
    else:
        llm = MockAsyncLLM(response_delay=0.2)
    
    chain = prompt | llm | StrOutputParser()
    
    # Prepare batch inputs
    inputs = [
        {"number": i}
        for i in range(1, 6)
    ]
    
    print(f"\n📦 Processing {len(inputs)} inputs with abatch()...")
    start_time = time.time()
    
    # Async batch - processes concurrently
    results = await chain.abatch(inputs)
    
    batch_duration = time.time() - start_time
    print(f"✅ Batch completed in {batch_duration:.3f}s")
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result[:80]}")
    
    # Compare with synchronous batch (would take N * request_time)
    print(f"\n💡 Async benefit: {len(inputs)} requests in {batch_duration:.3f}s")
    print(f"   Sequential would take ~{len(inputs) * 0.2:.3f}s (estimated)")
    
    return results


# ============================================================================
# Example 3: Async Streaming
# ============================================================================

async def example_async_streaming():
    """
    Demonstrate async streaming with astream().
    
    Key Concepts:
    - astream() returns async iterator for token-by-token streaming
    - Use 'async for' to consume stream
    - Enables real-time UI updates in chat applications
    - Non-blocking: other tasks can run while streaming
    """
    print("\n" + "="*70)
    print("Example 3: Async Streaming")
    print("="*70)
    
    prompt = PromptTemplate.from_template("Write a haiku about {subject}")
    
    if HAS_OPENAI_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
    else:
        llm = MockAsyncLLM(response_delay=0.1)
    
    chain = prompt | llm | StrOutputParser()
    
    print("\n🌊 Streaming response token-by-token...")
    print("Output: ", end="", flush=True)
    
    full_response = ""
    async for chunk in chain.astream({"subject": "async code"}):
        print(chunk, end="", flush=True)
        full_response += chunk
        # In a real application, you'd send each chunk to the client
        # for real-time UI updates
    
    print("\n")
    return full_response


# ============================================================================
# Example 4: Concurrent Chain Execution
# ============================================================================

async def example_concurrent_execution():
    """
    Demonstrate concurrent execution with asyncio.gather() and create_task().
    
    Key Concepts:
    - asyncio.gather() runs multiple coroutines concurrently
    - asyncio.create_task() schedules tasks in background
    - Proper task management and error handling
    - Waiting for all tasks to complete
    
    Use Case: Run multiple independent chains in parallel
    """
    print("\n" + "="*70)
    print("Example 4: Concurrent Chain Execution")
    print("="*70)
    
    if HAS_OPENAI_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo")
    else:
        llm = MockAsyncLLM(response_delay=0.2)
    
    # Create three different chains
    chain1 = PromptTemplate.from_template("Translate '{text}' to French") | llm | StrOutputParser()
    chain2 = PromptTemplate.from_template("Translate '{text}' to Spanish") | llm | StrOutputParser()
    chain3 = PromptTemplate.from_template("Translate '{text}' to German") | llm | StrOutputParser()
    
    print("\n🔀 Running three chains concurrently...")
    start_time = time.time()
    
    # Method 1: asyncio.gather() - run all concurrently and wait for all
    results = await asyncio.gather(
        chain1.ainvoke({"text": "Hello"}),
        chain2.ainvoke({"text": "Hello"}),
        chain3.ainvoke({"text": "Hello"}),
        return_exceptions=True  # Don't fail all if one fails
    )
    
    duration = time.time() - start_time
    print(f"✅ All chains completed in {duration:.3f}s")
    
    languages = ["French", "Spanish", "German"]
    for lang, result in zip(languages, results):
        if isinstance(result, Exception):
            print(f"  ❌ {lang}: Error - {result}")
        else:
            print(f"  ✅ {lang}: {result[:60]}")
    
    # Method 2: create_task() for background execution
    print("\n🎯 Using create_task() for background execution...")
    task1 = asyncio.create_task(chain1.ainvoke({"text": "Goodbye"}))
    task2 = asyncio.create_task(chain2.ainvoke({"text": "Goodbye"}))
    task3 = asyncio.create_task(chain3.ainvoke({"text": "Goodbye"}))
    
    # Tasks are now running in background
    # We can do other work here
    print("   Tasks scheduled in background...")
    
    # Wait for all tasks to complete
    await task1
    await task2
    await task3
    
    print("   All background tasks completed!")
    
    return results


# ============================================================================
# Example 5: Async Callback Integration
# ============================================================================

async def example_async_callbacks():
    """
    Demonstrate async callback handler integration.
    
    Key Concepts:
    - Pass callbacks in RunnableConfig
    - Async callbacks don't block chain execution
    - Useful for logging, monitoring, debugging
    - Event sequence: on_llm_start → on_llm_end (or on_llm_error)
    """
    print("\n" + "="*70)
    print("Example 5: Async Callback Integration")
    print("="*70)
    
    if HAS_OPENAI_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo")
    else:
        llm = MockAsyncLLM(response_delay=0.15)
    
    prompt = PromptTemplate.from_template("Summarize: {text}")
    chain = prompt | llm | StrOutputParser()
    
    # Create async callback handler
    callback = DetailedAsyncCallbackHandler()
    
    print("\n📊 Executing chain with async callback monitoring...")
    
    # Pass callbacks in config
    result = await chain.ainvoke(
        {"text": "LangChain is a framework for building LLM applications"},
        config={"callbacks": [callback]}
    )
    
    print(f"\nResult: {result[:100]}")
    print(f"\n📈 Callback Events Recorded: {len(callback.events)}")
    for event in callback.events:
        print(f"  - {event['event']} at {event.get('duration', 0):.3f}s")
    
    return result


# ============================================================================
# Example 6: Error Handling in Async Context
# ============================================================================

async def example_async_error_handling():
    """
    Demonstrate proper error handling for async operations.
    
    Key Concepts:
    - Try/except blocks work normally with await
    - asyncio.gather() with return_exceptions=True
    - Timeout handling with asyncio.wait_for()
    - Graceful degradation patterns
    """
    print("\n" + "="*70)
    print("Example 6: Async Error Handling")
    print("="*70)
    
    if HAS_OPENAI_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo")
    else:
        llm = MockAsyncLLM(response_delay=0.1)
    
    prompt = PromptTemplate.from_template("Process: {input}")
    chain = prompt | llm | StrOutputParser()
    
    # Example 1: Basic try/except
    print("\n1️⃣ Basic try/except pattern:")
    try:
        result = await chain.ainvoke({"input": "test"})
        print(f"   ✅ Success: {result[:50]}")
    except Exception as e:
        print(f"   ❌ Error caught: {type(e).__name__}: {e}")
    
    # Example 2: Timeout handling
    print("\n2️⃣ Timeout handling with asyncio.wait_for():")
    try:
        result = await asyncio.wait_for(
            chain.ainvoke({"input": "test with timeout"}),
            timeout=5.0  # 5 second timeout
        )
        print(f"   ✅ Completed within timeout: {result[:50]}")
    except asyncio.TimeoutError:
        print("   ⏱️  Operation timed out")
    except Exception as e:
        print(f"   ❌ Other error: {e}")
    
    # Example 3: Multiple chains with partial failure handling
    print("\n3️⃣ Multiple chains with return_exceptions=True:")
    
    async def failing_chain():
        """Simulated failing chain."""
        await asyncio.sleep(0.1)
        raise ValueError("Simulated API error")
    
    results = await asyncio.gather(
        chain.ainvoke({"input": "success 1"}),
        failing_chain(),
        chain.ainvoke({"input": "success 2"}),
        return_exceptions=True
    )
    
    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"   ❌ Chain {i}: {type(result).__name__}: {result}")
        else:
            print(f"   ✅ Chain {i}: {result[:40]}")
    
    return "Error handling demonstrated"


# ============================================================================
# Example 7: Performance Comparison - Async vs Sync
# ============================================================================

async def example_performance_comparison():
    """
    Compare async vs sync execution performance.
    
    Demonstrates the dramatic performance benefit of async execution
    for IO-bound operations like LLM API calls.
    """
    print("\n" + "="*70)
    print("Example 7: Performance Comparison - Async vs Sync")
    print("="*70)
    
    if HAS_OPENAI_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo")
    else:
        llm = MockAsyncLLM(response_delay=0.2)
    
    prompt = PromptTemplate.from_template("Count to {number}")
    chain = prompt | llm | StrOutputParser()
    
    inputs = [{"number": i} for i in range(1, 6)]
    
    # Synchronous execution
    print("\n⏱️  Synchronous execution (sequential)...")
    sync_start = time.time()
    sync_results = []
    for input_data in inputs:
        result = chain.invoke(input_data)
        sync_results.append(result)
    sync_duration = time.time() - sync_start
    print(f"   Completed in {sync_duration:.3f}s")
    
    # Asynchronous execution
    print("\n⚡ Asynchronous execution (concurrent)...")
    async_start = time.time()
    async_results = await chain.abatch(inputs)
    async_duration = time.time() - async_start
    print(f"   Completed in {async_duration:.3f}s")
    
    # Show improvement
    speedup = sync_duration / async_duration if async_duration > 0 else 1
    print(f"\n📊 Performance Summary:")
    print(f"   Sync:  {sync_duration:.3f}s")
    print(f"   Async: {async_duration:.3f}s")
    print(f"   Speedup: {speedup:.2f}x faster")
    print(f"   Time saved: {sync_duration - async_duration:.3f}s ({((sync_duration - async_duration) / sync_duration * 100):.1f}%)")
    
    return {"sync": sync_duration, "async": async_duration, "speedup": speedup}


# ============================================================================
# When to Use Async vs Sync - Decision Guide
# ============================================================================

def print_async_vs_sync_guide():
    """
    Print guidance on when to use async vs sync chain APIs.
    
    Source: Agent Action Plan section 0.3.2 - async/sync dual API patterns
    """
    print("\n" + "="*70)
    print("📚 When to Use Async vs Sync - Decision Guide")
    print("="*70)
    
    guide = """
    Use ASYNC (ainvoke, abatch, astream) when:
    ✅ Building async web applications (FastAPI, async frameworks)
    ✅ Need to run multiple chains concurrently
    ✅ Handling multiple users/requests simultaneously
    ✅ Want to avoid blocking the event loop
    ✅ Already in an async context (async def functions)
    ✅ Working with other async libraries (httpx, databases)
    
    Use SYNC (invoke, batch, stream) when:
    ✅ Simple scripts or notebooks (Jupyter)
    ✅ Synchronous web frameworks (Flask, Django)
    ✅ Not dealing with concurrency
    ✅ Easier to understand for beginners
    ✅ No event loop available
    ✅ Sequential processing is acceptable
    
    ⚠️  Important Considerations:
    - Never mix sync calls in async contexts (will block event loop)
    - Can't use await in sync functions (will cause SyntaxError)
    - Async provides no benefit for CPU-bound operations
    - Async adds complexity - only use when benefits justify it
    
    🎯 Rule of Thumb:
    If you're making multiple LLM calls and speed matters, use async.
    If you're making one call or simplicity matters, use sync.
    """
    
    print(guide)


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """
    Main async function demonstrating all async chain patterns.
    
    This is the proper way to run async code: define an async main()
    function and execute it with asyncio.run(main()).
    
    Event Loop Considerations:
    - asyncio.run() creates a new event loop and runs main()
    - Only call asyncio.run() once at the entry point
    - All async functions must be awaited
    - Use asyncio.create_task() or gather() for concurrent execution
    """
    print("\n" + "="*70)
    print("🚀 LangChain Async Chain Execution Examples")
    print("="*70)
    
    if not HAS_OPENAI_KEY:
        print("\n⚠️  No OPENAI_API_KEY found - using mock LLM for demonstration")
        print("   Set OPENAI_API_KEY in .env file for actual API calls\n")
    else:
        print("\n✅ Using OpenAI API\n")
    
    try:
        # Run all examples
        await example_basic_async_invoke()
        await example_async_batch()
        await example_async_streaming()
        await example_concurrent_execution()
        await example_async_callbacks()
        await example_async_error_handling()
        await example_performance_comparison()
        
        # Print decision guide
        print_async_vs_sync_guide()
        
        print("\n" + "="*70)
        print("✅ All async examples completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    """
    Entry point for the script.
    
    Proper async execution pattern:
    1. Define async functions with 'async def'
    2. Create an async main() function
    3. Call asyncio.run(main()) at module level
    
    This ensures proper event loop management and cleanup.
    """
    asyncio.run(main())

