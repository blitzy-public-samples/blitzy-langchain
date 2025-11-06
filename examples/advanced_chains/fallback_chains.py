#!/usr/bin/env python3
"""
Fallback Chains Example - Comprehensive Error Recovery Patterns

This example demonstrates production-ready error recovery strategies using
RunnableWithFallbacks from langchain_core. It shows how to build resilient
chains that gracefully degrade when primary operations fail.

Key Patterns Demonstrated:
1. Basic fallback: Primary chain → Fallback chain
2. Multi-level fallbacks: Primary → Fallback 1 → Fallback 2 → Fallback 3
3. Exception key passing: Forward exception info to fallback chains
4. Selective exception handling: Handle only specific exception types
5. Production patterns: Expensive/accurate → Cheap/faster
6. Async fallback execution
7. Comprehensive logging showing which fallback was used
8. Different failure scenarios: Rate limiting, timeouts, invalid outputs

Requirements:
    - langchain-core>=1.0.0
    - python-dotenv (optional, for API key management)

Source: libs/core/tests/unit_tests/runnables/test_fallbacks.py
Agent Action Plan Reference: Section 0.4.1 (Error Recovery), Section 0.3.1
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)

# Configure logging to show which fallback is being used
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Simulated LLM Classes for Demonstration
# ============================================================================
# Note: In production, replace these with actual LLM providers like
# OpenAI, Anthropic, etc. These simulate different failure scenarios.


class SimulatedLLM(Runnable):
    """
    Simulated LLM that can be configured to fail or succeed.
    
    This class simulates different LLM behaviors for demonstration purposes:
    - Success cases (normal operation)
    - Rate limiting errors (429 responses)
    - Timeout errors (slow responses)
    - Invalid output format errors
    
    Args:
        name: Identifier for this LLM (used in logging)
        should_fail: Whether this LLM should raise an exception
        failure_type: Type of failure to simulate ('rate_limit', 'timeout', 
                     'invalid_output')
        response: Response to return on success
    """
    
    def __init__(
        self,
        name: str,
        should_fail: bool = False,
        failure_type: str = "rate_limit",
        response: str = "Success"
    ):
        self.name = name
        self.should_fail = should_fail
        self.failure_type = failure_type
        self.response = response
        super().__init__()
    
    def invoke(self, input: Any, config: Any = None) -> str:
        """
        Invoke the simulated LLM.
        
        Args:
            input: Input prompt or message
            config: Optional configuration
            
        Returns:
            String response from the LLM
            
        Raises:
            ValueError: For rate limiting or invalid output errors
            TimeoutError: For timeout errors
        """
        logger.info(f"Attempting to invoke {self.name}")
        
        if self.should_fail:
            if self.failure_type == "rate_limit":
                logger.error(f"{self.name} failed: Rate limit exceeded")
                raise ValueError(f"Rate limit exceeded for {self.name}")
            elif self.failure_type == "timeout":
                logger.error(f"{self.name} failed: Request timeout")
                raise TimeoutError(f"Request timeout for {self.name}")
            elif self.failure_type == "invalid_output":
                logger.error(f"{self.name} failed: Invalid output format")
                raise ValueError(f"Invalid output format from {self.name}")
        
        logger.info(f"{self.name} succeeded with response: {self.response}")
        return self.response
    
    async def ainvoke(self, input: Any, config: Any = None) -> str:
        """
        Async invoke the simulated LLM.
        
        Args:
            input: Input prompt or message
            config: Optional configuration
            
        Returns:
            String response from the LLM
            
        Raises:
            ValueError: For rate limiting or invalid output errors
            TimeoutError: For timeout errors
        """
        logger.info(f"Async: Attempting to invoke {self.name}")
        
        # Simulate async operation
        await asyncio.sleep(0.1)
        
        if self.should_fail:
            if self.failure_type == "rate_limit":
                logger.error(f"Async: {self.name} failed: Rate limit exceeded")
                raise ValueError(f"Rate limit exceeded for {self.name}")
            elif self.failure_type == "timeout":
                logger.error(f"Async: {self.name} failed: Request timeout")
                raise TimeoutError(f"Request timeout for {self.name}")
            elif self.failure_type == "invalid_output":
                logger.error(f"Async: {self.name} failed: Invalid output format")
                raise ValueError(f"Invalid output format from {self.name}")
        
        logger.info(f"Async: {self.name} succeeded with response: {self.response}")
        return self.response


# ============================================================================
# Example 1: Basic Fallback Pattern
# ============================================================================

def example_basic_fallback():
    """
    Demonstrates basic fallback: primary chain fails → fallback chain succeeds.
    
    Pattern: Expensive/accurate model → Cheaper/faster model
    Use Case: Cost optimization while maintaining availability
    
    Source: libs/core/tests/unit_tests/runnables/test_fallbacks.py:34-38
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Fallback Pattern")
    print("=" * 70)
    
    # Primary LLM (configured to fail to demonstrate fallback)
    primary_llm = SimulatedLLM(
        name="GPT-4",
        should_fail=True,
        failure_type="rate_limit"
    )
    
    # Fallback LLM (simpler, more reliable)
    fallback_llm = SimulatedLLM(
        name="GPT-3.5-Turbo",
        should_fail=False,
        response="Response from fallback model"
    )
    
    # Create chain with fallback using .with_fallbacks() method
    # Type Flow: Runnable[Input, Output].with_fallbacks([Runnable[Input, Output]])
    #           → RunnableWithFallbacks[Input, Output]
    chain_with_fallback = primary_llm.with_fallbacks([fallback_llm])
    
    # Execute the chain - primary will fail, fallback will succeed
    try:
        result = chain_with_fallback.invoke("What is the capital of France?")
        print(f"✓ Final Result: {result}")
        print(f"✓ Fallback mechanism worked: Primary failed → Fallback succeeded")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


# ============================================================================
# Example 2: Multi-Level Fallbacks
# ============================================================================

def example_multi_level_fallbacks():
    """
    Demonstrates cascading fallbacks with 3+ levels.
    
    Pattern: Primary → Fallback 1 → Fallback 2 → Fallback 3
    Use Case: Maximum resilience with progressive degradation
    
    Fallback Strategy:
    - Level 0 (Primary): GPT-4 (most capable, highest cost)
    - Level 1: Claude-3 (alternative provider)
    - Level 2: GPT-3.5 (cheaper, faster)
    - Level 3: Cached response (last resort)
    
    Source: libs/core/tests/unit_tests/runnables/test_fallbacks.py:42-47
    """
    print("\n" + "=" * 70)
    print("Example 2: Multi-Level Fallback Pattern")
    print("=" * 70)
    
    # Level 0: Primary LLM (fails)
    primary_llm = SimulatedLLM(
        name="GPT-4",
        should_fail=True,
        failure_type="rate_limit"
    )
    
    # Level 1: First fallback (also fails)
    fallback_1 = SimulatedLLM(
        name="Claude-3",
        should_fail=True,
        failure_type="timeout"
    )
    
    # Level 2: Second fallback (also fails)
    fallback_2 = SimulatedLLM(
        name="GPT-3.5-Turbo",
        should_fail=True,
        failure_type="invalid_output"
    )
    
    # Level 3: Final fallback (cached/default response - always succeeds)
    def cached_response(input: Any) -> str:
        """Last resort: return cached or default response."""
        logger.info("Using cached response (final fallback)")
        return "I'm currently experiencing high demand. Please try again shortly."
    
    fallback_3 = RunnableLambda(cached_response)
    
    # Create multi-level fallback chain
    # Each fallback is tried in order until one succeeds
    chain_with_multi_fallback = primary_llm.with_fallbacks([
        fallback_1,
        fallback_2,
        fallback_3
    ])
    
    try:
        result = chain_with_multi_fallback.invoke("Explain quantum computing")
        print(f"✓ Final Result: {result}")
        print(f"✓ Multi-level fallback executed: Tried 4 levels, succeeded at level 3")
    except Exception as e:
        print(f"✗ All fallbacks failed: {e}")


# ============================================================================
# Example 3: Exception Key - Passing Exception Info to Fallbacks
# ============================================================================

def example_exception_key():
    """
    Demonstrates exception_key parameter for context-aware fallbacks.
    
    The exception_key parameter allows fallback chains to receive information
    about the exception that occurred, enabling intelligent error recovery.
    
    Pattern: Fallback receives exception info and can make recovery decisions
    Use Case: Adaptive fallback behavior based on error type
    
    Source: libs/core/tests/unit_tests/runnables/test_fallbacks.py:125-143
    """
    print("\n" + "=" * 70)
    print("Example 3: Exception Key Pattern")
    print("=" * 70)
    
    def primary_function(inputs: Dict[str, Any]) -> str:
        """Primary function that may fail."""
        logger.info("Executing primary function")
        # Simulate rate limit error
        raise ValueError("Rate limit exceeded (simulated)")
    
    def fallback_with_exception_handling(inputs: Dict[str, Any]) -> str:
        """
        Fallback that receives exception info and adapts behavior.
        
        The 'exception' key is automatically added by RunnableWithFallbacks
        when exception_key parameter is specified.
        """
        if "exception" in inputs:
            exception = inputs["exception"]
            logger.info(f"Fallback received exception info: {type(exception).__name__}")
            
            # Adapt behavior based on exception type
            if isinstance(exception, ValueError):
                if "rate limit" in str(exception).lower():
                    logger.info("Detected rate limit error - using cached response")
                    return "Using cached response due to rate limiting"
                else:
                    logger.info("Detected value error - using alternative approach")
                    return "Using alternative processing due to validation error"
            elif isinstance(exception, TimeoutError):
                logger.info("Detected timeout - using quick response")
                return "Quick response due to timeout"
            else:
                return "Generic fallback response"
        
        # This shouldn't happen with exception_key set, but handle gracefully
        logger.warning("Fallback called without exception info")
        return "Fallback executed without context"
    
    # Create runnables
    primary = RunnableLambda(primary_function)
    fallback = RunnableLambda(fallback_with_exception_handling)
    
    # Configure fallback with exception_key parameter
    # This tells RunnableWithFallbacks to pass exception info to fallback chains
    chain = {"text": RunnablePassthrough()} | primary.with_fallbacks(
        [fallback],
        exception_key="exception"  # Exception will be passed as inputs["exception"]
    )
    
    try:
        result = chain.invoke("test input")
        print(f"✓ Final Result: {result}")
        print(f"✓ Fallback received exception context and adapted behavior")
    except Exception as e:
        print(f"✗ Error: {e}")


# ============================================================================
# Example 4: Selective Exception Handling
# ============================================================================

def example_selective_exceptions():
    """
    Demonstrates exceptions_to_handle parameter for selective fallback.
    
    By default, RunnableWithFallbacks catches all exceptions. The
    exceptions_to_handle parameter allows you to specify which exception
    types should trigger fallback behavior.
    
    Pattern: Only certain exceptions trigger fallback, others propagate
    Use Case: Distinguish between recoverable and non-recoverable errors
    
    Source: libs/core/tests/unit_tests/runnables/test_fallbacks.py:197-207
    """
    print("\n" + "=" * 70)
    print("Example 4: Selective Exception Handling")
    print("=" * 70)
    
    # Test Case 1: ValueError (should trigger fallback)
    primary_1 = SimulatedLLM(
        name="Primary-ValueError",
        should_fail=True,
        failure_type="rate_limit"  # Raises ValueError
    )
    fallback_1 = SimulatedLLM(
        name="Fallback-1",
        should_fail=False,
        response="Fallback handled ValueError"
    )
    
    # Only handle ValueError - other exceptions will propagate
    chain_selective = primary_1.with_fallbacks(
        [fallback_1],
        exceptions_to_handle=(ValueError,)
    )
    
    try:
        result = chain_selective.invoke("test")
        print(f"✓ ValueError Test Result: {result}")
        print(f"✓ ValueError was caught and fallback executed")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print()
    
    # Test Case 2: TimeoutError (should NOT trigger fallback, propagates)
    primary_2 = SimulatedLLM(
        name="Primary-TimeoutError",
        should_fail=True,
        failure_type="timeout"  # Raises TimeoutError
    )
    fallback_2 = SimulatedLLM(
        name="Fallback-2",
        should_fail=False,
        response="This should not be reached"
    )
    
    # Still only handling ValueError, so TimeoutError will propagate
    chain_selective_2 = primary_2.with_fallbacks(
        [fallback_2],
        exceptions_to_handle=(ValueError,)  # TimeoutError NOT in this tuple
    )
    
    try:
        result = chain_selective_2.invoke("test")
        print(f"✗ This should not print: {result}")
    except TimeoutError as e:
        print(f"✓ TimeoutError Test: Exception propagated as expected")
        print(f"✓ TimeoutError was NOT caught (not in exceptions_to_handle)")
        print(f"  Exception: {e}")


# ============================================================================
# Example 5: Production Pattern - Cost Optimization
# ============================================================================

def example_production_pattern():
    """
    Demonstrates production-ready fallback strategy for cost optimization.
    
    Pattern: Expensive/accurate → Cheap/faster → Cached
    Use Case: Real-world production deployment with cost considerations
    
    Strategy:
    1. Try expensive, high-quality model (GPT-4)
    2. Fall back to cheaper, faster model (GPT-3.5)
    3. Final fallback to cached response
    
    This ensures maximum quality when possible, while maintaining availability
    and controlling costs during high-load or error conditions.
    """
    print("\n" + "=" * 70)
    print("Example 5: Production Cost Optimization Pattern")
    print("=" * 70)
    
    # Create prompt template
    # In production, this would be more complex
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question concisely: {question}"
    )
    
    # Primary: High-quality, expensive model
    primary_llm = SimulatedLLM(
        name="GPT-4-Turbo (Premium)",
        should_fail=False,  # Set to True to test fallback
        response="Detailed, high-quality answer from premium model"
    )
    
    # Fallback 1: Cheaper, faster model
    fallback_llm = SimulatedLLM(
        name="GPT-3.5-Turbo (Standard)",
        should_fail=False,
        response="Good quality answer from standard model"
    )
    
    # Fallback 2: Cached response function
    def get_cached_response(input: Any) -> str:
        """
        Retrieve cached response or return generic message.
        
        In production, this would:
        - Check Redis/Memcached for cached responses
        - Use semantic search to find similar questions
        - Return appropriate cached answer
        """
        logger.info("Retrieving cached response (cost: $0.00)")
        return "This is a cached response. For real-time information, please try again later."
    
    cached_fallback = RunnableLambda(get_cached_response)
    
    # Build production chain with multi-level fallbacks
    # Note: In production, replace SimulatedLLM with actual LLM providers
    production_chain = (
        prompt 
        | primary_llm.with_fallbacks([fallback_llm, cached_fallback])
        | StrOutputParser()
    )
    
    # Test the production chain
    questions = [
        "What is machine learning?",
        "Explain neural networks",
    ]
    
    for question in questions:
        try:
            print(f"\nQuestion: {question}")
            result = production_chain.invoke({"question": question})
            print(f"Answer: {result}")
        except Exception as e:
            print(f"✗ Error (all fallbacks exhausted): {e}")


# ============================================================================
# Example 6: Async Fallback Execution
# ============================================================================

async def example_async_fallbacks():
    """
    Demonstrates async fallback patterns with ainvoke().
    
    Pattern: Async primary → Async fallback
    Use Case: Non-blocking error recovery in async applications
    
    Important: Async fallbacks use the same .with_fallbacks() method,
    but are invoked with .ainvoke() instead of .invoke()
    
    Source: libs/core/tests/unit_tests/runnables/test_fallbacks.py:97-101
    """
    print("\n" + "=" * 70)
    print("Example 6: Async Fallback Pattern")
    print("=" * 70)
    
    # Primary async LLM (fails)
    primary_llm = SimulatedLLM(
        name="Async-Primary",
        should_fail=True,
        failure_type="timeout"
    )
    
    # Fallback async LLM (succeeds)
    fallback_llm = SimulatedLLM(
        name="Async-Fallback",
        should_fail=False,
        response="Async fallback response"
    )
    
    # Create async chain with fallback
    async_chain = primary_llm.with_fallbacks([fallback_llm])
    
    try:
        # Use ainvoke for async execution
        result = await async_chain.ainvoke("Async test input")
        print(f"✓ Async Result: {result}")
        print(f"✓ Async fallback executed successfully")
    except Exception as e:
        print(f"✗ Async error: {e}")


# ============================================================================
# Example 7: Complex Chain with Fallbacks
# ============================================================================

def example_complex_chain_fallbacks():
    """
    Demonstrates fallbacks in complex multi-stage chains.
    
    Pattern: (Prompt | LLM | Parser).with_fallbacks([...])
    Use Case: Production chains with preprocessing, processing, and postprocessing
    
    This shows that fallbacks can be applied to entire chain compositions,
    not just individual runnables.
    
    Source: libs/core/tests/unit_tests/runnables/test_fallbacks.py:51-58
    """
    print("\n" + "=" * 70)
    print("Example 7: Complex Chain with Fallbacks")
    print("=" * 70)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("Process this: {input}")
    
    # Primary chain: prompt | llm | parser
    primary_llm = SimulatedLLM(
        name="Primary-Chain-LLM",
        should_fail=True,
        failure_type="rate_limit"
    )
    primary_chain = prompt | primary_llm | StrOutputParser()
    
    # Fallback chain: same structure, different LLM
    fallback_llm = SimulatedLLM(
        name="Fallback-Chain-LLM",
        should_fail=False,
        response="Processed by fallback chain"
    )
    fallback_chain = prompt | fallback_llm | StrOutputParser()
    
    # Apply fallback to the entire chain
    # Type Flow: RunnableSequence[Input, Output].with_fallbacks([RunnableSequence[Input, Output]])
    #           → RunnableWithFallbacks[Input, Output]
    full_chain_with_fallback = primary_chain.with_fallbacks([fallback_chain])
    
    try:
        result = full_chain_with_fallback.invoke({"input": "test data"})
        print(f"✓ Chain Result: {result}")
        print(f"✓ Entire chain fallback executed: Primary chain failed → Fallback chain succeeded")
    except Exception as e:
        print(f"✗ Error: {e}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Execute all fallback chain examples.
    
    This demonstrates comprehensive error recovery patterns for production
    LangChain applications, showing how to build resilient systems that
    gracefully handle failures.
    """
    print("\n" + "=" * 70)
    print("LANGCHAIN FALLBACK CHAINS - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate production-ready error recovery patterns")
    print("using RunnableWithFallbacks from langchain_core.")
    print("\nNote: All examples use simulated LLMs. In production, replace")
    print("SimulatedLLM with actual LLM providers (OpenAI, Anthropic, etc.)")
    
    # Execute synchronous examples
    example_basic_fallback()
    example_multi_level_fallbacks()
    example_exception_key()
    example_selective_exceptions()
    example_production_pattern()
    example_complex_chain_fallbacks()
    
    # Execute async example
    print("\n" + "=" * 70)
    print("Running Async Example...")
    print("=" * 70)
    asyncio.run(example_async_fallbacks())
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Key Takeaways")
    print("=" * 70)
    print("""
1. Basic Fallback: Use .with_fallbacks([fallback]) for simple error recovery

2. Multi-Level: Chain multiple fallbacks for maximum resilience
   primary.with_fallbacks([fallback1, fallback2, fallback3])

3. Exception Key: Pass exception info to fallbacks for context-aware recovery
   .with_fallbacks([fallback], exception_key="exception")

4. Selective Handling: Only catch specific exception types
   .with_fallbacks([fallback], exceptions_to_handle=(ValueError,))

5. Production Pattern: Expensive/accurate → Cheap/faster → Cached
   Balances quality, cost, and availability

6. Async Support: Same .with_fallbacks() API, use .ainvoke() for execution

7. Chain Fallbacks: Apply fallbacks to entire chain compositions
   (prompt | llm | parser).with_fallbacks([fallback_chain])

Best Practices:
- Always have at least one guaranteed-to-succeed fallback
- Log which fallback was used for monitoring and debugging
- Use exceptions_to_handle to distinguish recoverable vs. fatal errors
- Consider cost/performance tradeoffs in fallback strategy
- Test fallback behavior under realistic failure conditions
""")
    
    print("\nFor more information, see:")
    print("- langchain_core.runnables.RunnableWithFallbacks documentation")
    print("- Agent Action Plan Section 0.4.1 (Error Recovery)")
    print("- Source: libs/core/tests/unit_tests/runnables/test_fallbacks.py")


if __name__ == "__main__":
    main()
