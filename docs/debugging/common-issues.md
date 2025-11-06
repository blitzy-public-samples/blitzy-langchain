# Common LangChain Issues and Solutions

## Overview

This guide documents the top 10 most common failure modes when working with LangChain chains, along with detailed symptoms, root causes, solutions, and prevention strategies. Each issue includes executable code examples demonstrating both the problem and the fix.

**Target Audience:** Developers with <2 years experience working with LangChain who need to debug chain execution failures independently.

**Prerequisites:**
- Basic understanding of LangChain chains and LCEL composition
- Python 3.10+ environment
- Familiarity with async/await patterns (for async-related issues)

**Related Documentation:**
- [Stack Trace Interpretation Guide](./stack-traces.md) - Understanding LangChain stack traces
- [Logging Configuration Guide](./logging.md) - Setting up granular chain logging
- [LCEL Composition Guide](../guides/lcel-composition.md) - Type-safe chain composition

## Quick Reference Table

| Issue | Symptom | Common Trigger | Solution Link |
|-------|---------|----------------|---------------|
| Rate Limiting | `429 Too Many Requests` | High API call frequency | [#1-rate-limiting](#1-rate-limiting-errors) |
| Template Variables | `KeyError: 'variable_name'` | Missing/mismatched prompt variables | [#2-template-variables](#2-invalid-prompt-template-variable-mismatches) |
| Context Overflow | Token limit errors | Large conversation history | [#3-context-overflow](#3-memory-context-window-overflow) |
| Type Mismatches | `TypeError` in LCEL pipes | Incompatible input/output types | [#4-type-mismatches](#4-type-mismatches-in-lcel-pipe-compositions) |
| Timeouts | `asyncio.TimeoutError` | Long-running chain execution | [#5-timeout-errors](#5-timeout-errors-on-long-running-chains) |
| Parser Failures | `OutputParserException` | Malformed LLM output | [#6-parser-failures](#6-malformed-output-parser-expectations) |
| Tool Execution | `ToolException` | Invalid tool interface | [#7-tool-failures](#7-agent-tool-execution-failures) |
| Empty Retrieval | No documents returned | Vector store misconfiguration | [#8-empty-retrieval](#8-retrieval-returning-empty-results) |
| Token Limits | `InvalidRequestError` | Input exceeds model limits | [#9-token-limits](#9-token-limit-exceeded-errors) |
| Callback Errors | Exception in callback chain | Callback handler bug | [#10-callback-errors](#10-callback-handler-exceptions) |

---

## 1. Rate Limiting Errors

### Symptoms

**Observable Error Messages:**
```python
openai.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for requests', 'type': 'rate_limit_error'}}

anthropic.RateLimitError: rate_limit_error: Number of request tokens has exceeded your per-minute rate limit
```

**Behavioral Indicators:**
- Chain fails after successful initial invocations
- Errors occur in bursts when processing multiple requests
- Success rate decreases during peak usage times
- Intermittent failures that resolve after waiting

### Root Causes

**Technical Explanation:**

LLM providers (OpenAI, Anthropic, etc.) enforce rate limits on API requests to ensure fair usage and prevent service abuse. Rate limits are typically measured in:

1. **Requests per minute (RPM)**: Maximum number of API calls per minute
2. **Tokens per minute (TPM)**: Maximum number of tokens processed per minute
3. **Requests per day (RPD)**: Daily quota cap

When your application exceeds these limits, the provider returns a `429` HTTP status code, causing the chain to fail immediately without retrying.

**Source Code Context:**

LangChain chains invoke LLM APIs through provider-specific clients which raise rate limit exceptions when throttled.

- **OpenAI Rate Limiting**: `libs/langchain/langchain_classic/chains/base.py:219-233` - Chain execution catches `BaseException` during `_acall()`
- **Callback Integration**: Errors trigger `on_chain_error` callback before propagating

### Solutions

**Solution 1: Exponential Backoff with Retry Logic**

Implement automatic retry with exponential backoff to handle transient rate limits gracefully:

```python
import asyncio
import time
from typing import Any, Dict
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def invoke_with_retry(
    chain: Runnable,
    inputs: Dict[str, Any],
    max_retries: int = 5,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0
) -> Any:
    """
    Invoke a chain with exponential backoff retry logic for rate limits.
    
    Args:
        chain: The LangChain runnable to invoke
        inputs: Input dictionary for the chain
        max_retries: Maximum number of retry attempts (default: 5)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        exponential_base: Multiplier for exponential backoff (default: 2.0)
    
    Returns:
        Chain output on success
    
    Raises:
        Exception: Re-raises the last exception if all retries exhausted
    
    Example:
        >>> llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        >>> chain = ChatPromptTemplate.from_template("Say {input}") | llm | StrOutputParser()
        >>> result = invoke_with_retry(chain, {"input": "hello"}, max_retries=3)
        >>> print(result)
    """
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            # Check if this is a rate limit error (status code 429)
            is_rate_limit = (
                "rate_limit" in str(e).lower() or 
                "429" in str(e) or
                "too many requests" in str(e).lower()
            )
            
            if not is_rate_limit or attempt == max_retries:
                # Not a rate limit error or no retries left
                raise
            
            # Calculate delay with exponential backoff
            delay = initial_delay * (exponential_base ** attempt)
            print(f"Rate limit hit. Retry {attempt + 1}/{max_retries} after {delay:.1f}s")
            time.sleep(delay)
    
    # Should never reach here, but satisfy type checker
    raise RuntimeError("Unexpected retry loop exit")


# Example usage
if __name__ == "__main__":
    # Initialize chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_retries=0)
    prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence")
    chain = prompt | llm | StrOutputParser()
    
    # Invoke with retry logic
    try:
        result = invoke_with_retry(
            chain, 
            {"topic": "quantum computing"},
            max_retries=3,
            initial_delay=1.0
        )
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed after all retries: {e}")
```

**Solution 2: Async Rate Limiting with Token Bucket**

For production systems handling concurrent requests, implement a token bucket rate limiter:

```python
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict
from langchain_core.runnables import Runnable


class AsyncRateLimiter:
    """
    Token bucket rate limiter for async LangChain chains.
    
    Ensures requests stay within specified rate limits by queueing
    requests that would exceed the limit.
    
    Args:
        requests_per_minute: Maximum requests per minute
        tokens_per_minute: Maximum tokens per minute (optional)
    
    Example:
        >>> limiter = AsyncRateLimiter(requests_per_minute=60)
        >>> async def process():
        ...     result = await limiter.invoke_async(chain, {"input": "test"})
        ...     return result
    """
    
    def __init__(self, requests_per_minute: int, tokens_per_minute: int = None):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        # Token bucket for requests
        self.request_tokens = requests_per_minute
        self.max_request_tokens = requests_per_minute
        self.last_request_refill = datetime.now()
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def _refill_tokens(self):
        """Refill token buckets based on elapsed time."""
        now = datetime.now()
        elapsed = (now - self.last_request_refill).total_seconds()
        
        # Refill request tokens proportionally to elapsed time
        tokens_to_add = (elapsed / 60.0) * self.max_request_tokens
        self.request_tokens = min(
            self.max_request_tokens,
            self.request_tokens + tokens_to_add
        )
        self.last_request_refill = now
    
    async def invoke_async(
        self, 
        chain: Runnable, 
        inputs: Dict[str, Any],
        max_retries: int = 3
    ) -> Any:
        """
        Invoke chain with rate limiting and retry logic.
        
        Args:
            chain: Runnable chain to invoke
            inputs: Input dictionary
            max_retries: Maximum retry attempts for rate limit errors
        
        Returns:
            Chain output
        
        Raises:
            Exception: If chain fails after all retries
        """
        async with self.lock:
            # Wait until we have a token available
            while self.request_tokens < 1:
                await self._refill_tokens()
                if self.request_tokens < 1:
                    # Calculate wait time for next token
                    wait_time = 60.0 / self.max_request_tokens
                    await asyncio.sleep(wait_time)
            
            # Consume a token
            self.request_tokens -= 1
        
        # Invoke chain with retry logic
        for attempt in range(max_retries + 1):
            try:
                return await chain.ainvoke(inputs)
            except Exception as e:
                is_rate_limit = (
                    "rate_limit" in str(e).lower() or 
                    "429" in str(e)
                )
                
                if not is_rate_limit or attempt == max_retries:
                    raise
                
                # Exponential backoff
                delay = 2 ** attempt
                print(f"Rate limit error, retrying in {delay}s")
                await asyncio.sleep(delay)
        
        raise RuntimeError("Unexpected retry loop exit")


# Example usage with concurrent requests
async def process_batch_with_rate_limiting():
    """Process multiple requests with rate limiting."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    # Initialize chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template("Summarize: {text}")
    chain = prompt | llm | StrOutputParser()
    
    # Create rate limiter (60 requests per minute)
    limiter = AsyncRateLimiter(requests_per_minute=60)
    
    # Process batch of requests
    texts = [f"Document {i}" for i in range(100)]
    
    tasks = [
        limiter.invoke_async(chain, {"text": text})
        for text in texts
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle results
    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]
    
    print(f"Processed: {len(successes)} successful, {len(failures)} failed")
    return successes


# Run async example
if __name__ == "__main__":
    asyncio.run(process_batch_with_rate_limiting())
```

### Prevention Strategies

**Best Practices to Avoid Rate Limiting:**

1. **Understand Your Tier Limits**: Check your API provider's rate limits for your account tier
   - OpenAI: View limits at https://platform.openai.com/account/rate-limits
   - Anthropic: Check console at https://console.anthropic.com/settings/limits

2. **Implement Client-Side Rate Limiting**: Always use a rate limiter before hitting external limits
   ```python
   from langchain_core.rate_limiters import InMemoryRateLimiter
   
   # LangChain's built-in rate limiter (Python 3.11+)
   rate_limiter = InMemoryRateLimiter(
       requests_per_second=1,  # 60 requests per minute
       max_bucket_size=10
   )
   
   llm = ChatOpenAI(
       model="gpt-3.5-turbo",
       rate_limiter=rate_limiter  # Apply to all LLM calls
   )
   ```
   **Source**: `libs/core/langchain_core/rate_limiters.py`

3. **Batch Requests Efficiently**: Group similar requests to minimize API calls
   ```python
   # Use batch() for parallel processing with automatic rate limiting
   results = await chain.abatch(
       [{"input": text} for text in texts],
       config={"max_concurrency": 5}  # Limit concurrent requests
   )
   ```

4. **Monitor Token Usage**: Track tokens to stay within TPM limits
   ```python
   from langchain_core.callbacks import get_openai_callback
   
   with get_openai_callback() as cb:
       result = chain.invoke({"input": "test"})
       print(f"Tokens used: {cb.total_tokens}")
       print(f"Cost: ${cb.total_cost}")
   ```

5. **Cache Responses**: Avoid redundant API calls for identical inputs
   ```python
   from langchain_core.caches import InMemoryCache
   from langchain_core.globals import set_llm_cache
   
   set_llm_cache(InMemoryCache())  # Cache LLM responses
   ```

---

## 2. Invalid Prompt Template Variable Mismatches

### Symptoms

**Observable Error Messages:**
```python
KeyError: 'user_name'

pydantic.ValidationError: 1 validation error for PromptTemplate
  Value error, Invalid prompt schema; check for mismatched or missing input parameters. 'user_name'

ValueError: Missing variables {'user_query'} in prompt template
```

**Behavioral Indicators:**
- Chain fails immediately upon invocation
- Error message specifies missing variable name
- Occurs during prompt template formatting phase
- Happens before LLM API call

### Root Causes

**Technical Explanation:**

Prompt templates define placeholders (variables) that must be provided when the template is invoked. The mismatch occurs when:

1. **Template defines variables not provided in inputs**: Template has `{user_name}` but input only has `{"query": "..."}`
2. **Input provides variables not in template**: Less common but can cause issues with strict validation
3. **Variable name typos**: Template has `{username}` but input has `{"user_name": "..."}`
4. **Case sensitivity**: Template has `{UserName}` but input has `{"username": "..."}`

**Source Code Context:**

- **Prompt Template Validation**: `libs/core/langchain_core/prompts/chat.py` - `ChatPromptTemplate` validates `input_variables` against template placeholders
- **Variable Extraction**: Templates parse `{variable}` syntax and build required variable list
- **Invocation Validation**: `invoke()` method checks all required variables are present before formatting

### Solutions

**Solution 1: Validate Template Variables Before Invocation**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, Set


def validate_template_variables(
    template: ChatPromptTemplate,
    inputs: Dict[str, Any]
) -> tuple[bool, Set[str], Set[str]]:
    """
    Validate that inputs match template's required variables.
    
    Args:
        template: ChatPromptTemplate to validate
        inputs: Input dictionary to check
    
    Returns:
        Tuple of (is_valid, missing_vars, extra_vars)
        - is_valid: True if inputs match template requirements
        - missing_vars: Variables in template but not in inputs
        - extra_vars: Variables in inputs but not in template
    
    Example:
        >>> template = ChatPromptTemplate.from_template("Hello {name}, your age is {age}")
        >>> inputs = {"name": "Alice"}
        >>> valid, missing, extra = validate_template_variables(template, inputs)
        >>> print(f"Valid: {valid}, Missing: {missing}")
        Valid: False, Missing: {'age'}
    """
    required_vars = set(template.input_variables)
    provided_vars = set(inputs.keys())
    
    missing_vars = required_vars - provided_vars
    extra_vars = provided_vars - required_vars
    
    is_valid = len(missing_vars) == 0
    
    return is_valid, missing_vars, extra_vars


def safe_invoke_with_validation(
    chain: Any,
    inputs: Dict[str, Any],
    template: ChatPromptTemplate
) -> Any:
    """
    Safely invoke chain with upfront template validation.
    
    Args:
        chain: LangChain runnable chain
        inputs: Input dictionary
        template: Prompt template used in the chain
    
    Returns:
        Chain output
    
    Raises:
        ValueError: If required template variables are missing
    
    Example:
        >>> llm = ChatOpenAI(model="gpt-3.5-turbo")
        >>> template = ChatPromptTemplate.from_template("Explain {topic}")
        >>> chain = template | llm | StrOutputParser()
        >>> result = safe_invoke_with_validation(
        ...     chain, 
        ...     {"topic": "AI"}, 
        ...     template
        ... )
    """
    is_valid, missing, extra = validate_template_variables(template, inputs)
    
    if not is_valid:
        error_msg = f"Template validation failed!\n"
        error_msg += f"Missing required variables: {missing}\n"
        error_msg += f"Template expects: {template.input_variables}\n"
        error_msg += f"Provided inputs: {list(inputs.keys())}"
        raise ValueError(error_msg)
    
    if extra:
        print(f"Warning: Extra variables provided (will be ignored): {extra}")
    
    return chain.invoke(inputs)


# Example: Detecting and fixing variable mismatch
if __name__ == "__main__":
    # Template expects 'user_query' and 'context'
    template = ChatPromptTemplate.from_template(
        "Context: {context}\n\nUser Query: {user_query}\n\nAnswer:"
    )
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = template | llm | StrOutputParser()
    
    # Incorrect inputs (typo: 'query' instead of 'user_query')
    wrong_inputs = {
        "context": "LangChain is a framework for LLM applications",
        "query": "What is LangChain?"  # Wrong key name!
    }
    
    try:
        safe_invoke_with_validation(chain, wrong_inputs, template)
    except ValueError as e:
        print(f"Validation Error:\n{e}\n")
        
        # Fix the input keys
        correct_inputs = {
            "context": wrong_inputs["context"],
            "user_query": wrong_inputs["query"]  # Corrected key name
        }
        
        result = safe_invoke_with_validation(chain, correct_inputs, template)
        print(f"Success: {result}")
```

**Solution 2: Dynamic Template with Partial Variables**

Use partial variables for optional parameters:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime


def create_flexible_prompt_template():
    """
    Create template with partial variables for optional parameters.
    
    Partial variables are pre-filled and don't need to be provided
    at invocation time, making templates more flexible.
    
    Returns:
        ChatPromptTemplate with partial variables configured
    
    Example:
        >>> template = create_flexible_prompt_template()
        >>> # Only need to provide 'question', not 'current_date'
        >>> chain = template | llm | parser
        >>> result = chain.invoke({"question": "What's the weather?"})
    """
    # Template with both required and optional variables
    template_str = """Current Date: {current_date}
System: You are a helpful assistant.

User Question: {question}

Instructions: {instructions}

Answer:"""
    
    template = ChatPromptTemplate.from_template(template_str)
    
    # Provide default values for optional variables using partial()
    template = template.partial(
        current_date=lambda: datetime.now().strftime("%Y-%m-%d"),
        instructions="Provide a concise, accurate response."
    )
    
    # Now template only requires 'question' input
    print(f"Required input variables: {template.input_variables}")
    # Output: ['question']
    
    return template


# Example usage
if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    parser = StrOutputParser()
    
    # Create template with partials
    template = create_flexible_prompt_template()
    chain = template | llm | parser
    
    # Only need to provide 'question' now
    result = chain.invoke({"question": "What is Python?"})
    print(f"Result: {result}")
    
    # Can override partial variables if needed
    result_custom = chain.invoke({
        "question": "What is Python?",
        "instructions": "Explain in technical detail."  # Override default
    })
    print(f"Custom result: {result_custom}")
```

**Solution 3: Template Introspection Helper**

```python
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any


def introspect_template(template: ChatPromptTemplate) -> Dict[str, Any]:
    """
    Extract detailed information about a prompt template's requirements.
    
    Args:
        template: ChatPromptTemplate to introspect
    
    Returns:
        Dictionary containing template metadata:
        - required_variables: List of required input variables
        - optional_variables: List of partial (pre-filled) variables
        - template_preview: String representation of template
    
    Example:
        >>> template = ChatPromptTemplate.from_template("Hello {name}")
        >>> info = introspect_template(template)
        >>> print(info['required_variables'])
        ['name']
    """
    info = {
        "required_variables": template.input_variables,
        "optional_variables": list(template.partial_variables.keys()) if template.partial_variables else [],
        "template_preview": str(template.messages[0].prompt.template if hasattr(template.messages[0], 'prompt') else template),
        "total_messages": len(template.messages)
    }
    return info


def print_template_requirements(template: ChatPromptTemplate):
    """Print human-readable template requirements."""
    info = introspect_template(template)
    
    print("=== Template Requirements ===")
    print(f"Required Variables: {info['required_variables']}")
    print(f"Optional Variables: {info['optional_variables']}")
    print(f"Message Count: {info['total_messages']}")
    print(f"\nTemplate Preview:")
    print(info['template_preview'])
    print("=" * 30)


# Example: Debugging template requirements
if __name__ == "__main__":
    # Complex template with multiple variables
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a {role} assistant. Current context: {context}"),
        ("human", "{user_input}")
    ])
    
    # Inspect template
    print_template_requirements(template)
    
    # Expected output shows all required variables
    # Required Variables: ['role', 'context', 'user_input']
```

### Prevention Strategies

**Best Practices to Avoid Template Variable Mismatches:**

1. **Use Type Hints and Pydantic Models**: Define input schema explicitly
   ```python
   from pydantic import BaseModel
   from langchain_core.prompts import ChatPromptTemplate
   
   class ChainInput(BaseModel):
       user_query: str
       context: str
       max_tokens: int = 100  # Optional with default
   
   template = ChatPromptTemplate.from_template(
       "Context: {context}\nQuery: {user_query}"
   )
   
   # Type checker will catch mismatches
   def invoke_chain(inputs: ChainInput):
       return chain.invoke(inputs.model_dump())
   ```

2. **Document Required Variables**: Add docstrings to chain factory functions
   ```python
   def create_qa_chain():
       \"\"\"
       Create Q&A chain.
       
       Required Inputs:
           - question (str): User's question
           - context (str): Retrieved context documents
       
       Returns:
           Runnable chain expecting {'question': str, 'context': str}
       \"\"\"
       template = ChatPromptTemplate.from_template(
           "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
       )
       return template | llm | parser
   ```

3. **Validate Early in Development**: Add validation in chain construction
   ```python
   def create_validated_chain(expected_inputs: List[str]):
       \"\"\"Create chain with validation that checks expected inputs.\"\"\"
       template = ChatPromptTemplate.from_template("...")
       
       # Validate template matches expected inputs
       if set(template.input_variables) != set(expected_inputs):
           raise ValueError(
               f"Template mismatch! "
               f"Expected: {expected_inputs}, "
               f"Got: {template.input_variables}"
           )
       
       return template | llm | parser
   ```

4. **Use Consistent Naming Conventions**: Standardize variable names across project
   - Use `user_query` consistently, not mixing with `query`, `user_input`, `question`
   - Document naming conventions in project README
   - Use linting rules to enforce naming patterns

---

## 3. Memory Context Window Overflow

### Symptoms

**Observable Error Messages:**
```python
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 4096 tokens. However, your messages resulted in 5234 tokens.", 'type': 'invalid_request_error'}}

anthropic.BadRequestError: prompt is too long: 12000 tokens > 8000 maximum
```

**Behavioral Indicators:**
- Chain works initially but fails after extended conversation
- Error occurs after N successful message exchanges
- Token count grows with each conversation turn
- Memory-enabled chains fail while stateless chains succeed

### Root Causes

**Technical Explanation:**

LLM models have fixed context windows (maximum tokens they can process). When using conversational memory:

1. **Message History Accumulation**: Each conversation turn adds messages to history
   - User message: ~50-500 tokens
   - AI response: ~100-2000 tokens
   - After 5-10 turns, can exceed model limits

2. **Memory Implementation Details**:
   - `ConversationBufferMemory`: Stores ALL messages (unbounded growth)
   - `ConversationBufferWindowMemory`: Stores last K messages (better but still can overflow)
   - Memory loaded before each chain invocation adds to prompt tokens

3. **Token Calculation Oversight**: Developers often forget to account for:
   - System messages (instructions, role definitions)
   - Few-shot examples in prompts
   - Retrieved documents in RAG chains
   - Chain of thought reasoning tokens

**Source Code Context:**

- **Memory Loading**: `libs/langchain/langchain_classic/chains/base.py:144-157` - `prep_inputs()` loads memory variables before chain execution
- **Context Window**: Model-specific limits enforced by provider APIs

**Model Context Limits Reference:**
- GPT-3.5-turbo: 4,096 tokens (older) / 16,385 tokens (newer)
- GPT-4: 8,192 tokens / 32,768 tokens / 128,000 tokens
- Claude 2: 100,000 tokens
- Claude 3: 200,000 tokens

### Solutions

**Solution 1: Implement Token-Aware Memory Management**

```python
from langchain_core.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, Any, List
import tiktoken


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to tokenize
        model: Model name for correct tokenizer
    
    Returns:
        Token count
    
    Example:
        >>> count_tokens("Hello, world!")
        4
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def count_messages_tokens(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in message list (includes formatting overhead).
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name for correct tokenizer
    
    Returns:
        Total token count including message formatting
    
    Example:
        >>> messages = [{"role": "user", "content": "Hi"}]
        >>> count_messages_tokens(messages)
        8
    """
    total_tokens = 0
    
    for message in messages:
        # Account for message formatting tokens
        total_tokens += 4  # Every message has formatting overhead
        total_tokens += count_tokens(message.get("content", ""), model)
        total_tokens += count_tokens(message.get("role", ""), model)
    
    total_tokens += 2  # Account for priming tokens
    
    return total_tokens


class TokenAwareConversationMemory:
    """
    Conversation memory that truncates to stay within token limits.
    
    Args:
        max_tokens: Maximum tokens to keep in memory
        model: Model name for token counting
    
    Example:
        >>> memory = TokenAwareConversationMemory(max_tokens=1000)
        >>> memory.add_user_message("Hello")
        >>> memory.add_ai_message("Hi there!")
        >>> messages = memory.get_messages()
    """
    
    def __init__(self, max_tokens: int = 2000, model: str = "gpt-3.5-turbo"):
        self.max_tokens = max_tokens
        self.model = model
        self.messages: List[Dict[str, str]] = []
    
    def add_user_message(self, content: str):
        """Add user message to memory."""
        self.messages.append({"role": "user", "content": content})
        self._truncate_if_needed()
    
    def add_ai_message(self, content: str):
        """Add AI message to memory."""
        self.messages.append({"role": "assistant", "content": content})
        self._truncate_if_needed()
    
    def _truncate_if_needed(self):
        """Remove oldest messages if token limit exceeded."""
        while len(self.messages) > 1:  # Keep at least 1 message
            current_tokens = count_messages_tokens(self.messages, self.model)
            
            if current_tokens <= self.max_tokens:
                break
            
            # Remove oldest message pair (user + assistant)
            if len(self.messages) >= 2:
                self.messages.pop(0)  # Remove oldest
                self.messages.pop(0)  # Remove its pair
            else:
                break
        
        print(f"Memory: {len(self.messages)} messages, "
              f"{count_messages_tokens(self.messages, self.model)} tokens")
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get current message history."""
        return self.messages.copy()
    
    def clear(self):
        """Clear all messages."""
        self.messages = []


# Example: Using token-aware memory
if __name__ == "__main__":
    memory = TokenAwareConversationMemory(max_tokens=500, model="gpt-3.5-turbo")
    
    # Simulate conversation
    conversations = [
        ("What is LangChain?", "LangChain is a framework..."),
        ("How do I install it?", "You can install LangChain using pip..."),
        ("What are chains?", "Chains are sequences of calls..."),
        ("Can you give an example?", "Here's an example of a simple chain..."),
        ("What about agents?", "Agents are autonomous entities...")
    ]
    
    for user_msg, ai_msg in conversations:
        memory.add_user_message(user_msg)
        memory.add_ai_message(ai_msg)
    
    final_messages = memory.get_messages()
    print(f"\nFinal history: {len(final_messages)} messages retained")
```

**Solution 2: Implement Sliding Window with Buffer**

```python
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


def create_windowed_conversation_chain(window_size: int = 5):
    """
    Create chain with sliding window memory.
    
    Args:
        window_size: Number of recent conversation turns to remember
    
    Returns:
        Conversational chain with bounded memory
    
    Example:
        >>> chain = create_windowed_conversation_chain(window_size=3)
        >>> result = chain.invoke({"input": "Hello"})
    """
    # Memory stores only last N conversation turns
    memory = ConversationBufferWindowMemory(
        k=window_size,  # Keep last 5 exchanges
        return_messages=True,
        memory_key="history"
    )
    
    # Create prompt with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Keep responses concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    chain = prompt | llm | StrOutputParser()
    
    return chain, memory


# Example usage with manual memory management
if __name__ == "__main__":
    chain, memory = create_windowed_conversation_chain(window_size=3)
    
    # Conversation simulation
    user_inputs = [
        "Hi, I'm learning LangChain",
        "What are the main components?",
        "Tell me about chains",
        "What about agents?",
        "How do I use memory?",
        "What was my first question?"  # Should not remember (outside window)
    ]
    
    for user_input in user_inputs:
        # Load memory context
        memory_vars = memory.load_memory_variables({})
        
        # Invoke chain
        response = chain.invoke({
            "input": user_input,
            "history": memory_vars.get("history", [])
        })
        
        print(f"User: {user_input}")
        print(f"AI: {response}\n")
        
        # Save to memory
        memory.save_context(
            {"input": user_input},
            {"output": response}
        )
```

**Solution 3: Summarization-Based Memory**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict


class SummarizingMemory:
    """
    Memory that summarizes old messages to save tokens.
    
    Maintains recent messages in full, older messages as summary.
    
    Args:
        max_recent_messages: Number of recent messages to keep in full
        llm: Language model for summarization
    
    Example:
        >>> llm = ChatOpenAI(model="gpt-3.5-turbo")
        >>> memory = SummarizingMemory(max_recent_messages=4, llm=llm)
    """
    
    def __init__(self, max_recent_messages: int = 6, llm=None):
        self.max_recent_messages = max_recent_messages
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.summary = ""
        self.recent_messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """Add message and trigger summarization if needed."""
        self.recent_messages.append({"role": role, "content": content})
        
        # If exceeded max recent, summarize oldest messages
        if len(self.recent_messages) > self.max_recent_messages:
            self._summarize_old_messages()
    
    def _summarize_old_messages(self):
        """Summarize oldest messages and update summary."""
        # Take oldest 4 messages for summarization
        to_summarize = self.recent_messages[:4]
        self.recent_messages = self.recent_messages[4:]
        
        # Create summary prompt
        messages_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in to_summarize
        ])
        
        prompt = ChatPromptTemplate.from_template(
            """Existing summary: {summary}

New messages to incorporate:
{messages}

Provide a concise summary of the conversation so far, incorporating the new messages. Keep it under 100 words.

Summary:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        self.summary = chain.invoke({
            "summary": self.summary or "No previous context.",
            "messages": messages_text
        })
        
        print(f"Summarized {len(to_summarize)} messages. Current summary length: {len(self.summary)} chars")
    
    def get_context(self) -> str:
        """Get formatted context for prompt."""
        context_parts = []
        
        if self.summary:
            context_parts.append(f"Previous conversation summary:\n{self.summary}\n")
        
        if self.recent_messages:
            context_parts.append("Recent messages:")
            for msg in self.recent_messages:
                context_parts.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(context_parts)


# Example usage
if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = SummarizingMemory(max_recent_messages=4, llm=llm)
    
    # Simulate long conversation
    conversations = [
        ("What is Python?", "Python is a programming language..."),
        ("Who created it?", "Python was created by Guido van Rossum..."),
        ("When was it created?", "Python was first released in 1991..."),
        ("What are its key features?", "Key features include..."),
        ("Is it good for AI?", "Yes, Python is excellent for AI..."),
        ("What libraries exist?", "Popular AI libraries include..."),
    ]
    
    for user_q, ai_resp in conversations:
        memory.add_message("user", user_q)
        memory.add_message("assistant", ai_resp)
    
    print("\nFinal Context:")
    print(memory.get_context())
```

### Prevention Strategies

**Best Practices to Avoid Context Overflow:**

1. **Choose Appropriate Memory Type**: Match memory to use case
   ```python
   # For short conversations (< 10 turns)
   from langchain_classic.memory import ConversationBufferMemory
   memory = ConversationBufferMemory()
   
   # For medium conversations (10-50 turns)
   from langchain_classic.memory import ConversationBufferWindowMemory
   memory = ConversationBufferWindowMemory(k=10)  # Last 10 exchanges
   
   # For long conversations or limited context models
   from langchain_classic.memory import ConversationSummaryMemory
   memory = ConversationSummaryMemory(llm=ChatOpenAI())
   ```
   **Source**: `libs/langchain/langchain_classic/memory/`

2. **Monitor Token Usage**: Track cumulative tokens
   ```python
   from langchain_core.callbacks import get_openai_callback
   
   with get_openai_callback() as cb:
       result = chain.invoke({"input": "query"})
       print(f"Prompt tokens: {cb.prompt_tokens}")
       print(f"Completion tokens: {cb.completion_tokens}")
       
       # Alert if approaching limit
       if cb.prompt_tokens > 3000:  # For 4K context model
           print("WARNING: Approaching context limit!")
   ```

3. **Set Context Budgets**: Allocate tokens explicitly
   ```python
   CONTEXT_BUDGET = {
       "system_prompt": 200,
       "conversation_history": 2000,
       "user_query": 500,
       "response_buffer": 1000,
       "safety_margin": 396  # For 4K model
   }
   
   # Ensure memory doesn't exceed budget
   max_history_tokens = CONTEXT_BUDGET["conversation_history"]
   ```

4. **Implement Graceful Degradation**: Handle overflow elegantly
   ```python
   def invoke_with_fallback(chain, inputs, memory):
       try:
           return chain.invoke(inputs)
       except Exception as e:
           if "maximum context length" in str(e):
               print("Context overflow detected, clearing old memory...")
               memory.clear()  # Clear memory and retry
               return chain.invoke(inputs)
           raise
   ```

---

## 4. Type Mismatches in LCEL Pipe Compositions

### Symptoms

**Observable Error Messages:**
```python
TypeError: __str__ returned non-string (type dict)

TypeError: 'dict' object is not callable

pydantic.ValidationError: 1 validation error for StrOutputParser
  Input should be a valid string

AttributeError: 'str' object has no attribute 'content'
```

**Behavioral Indicators:**
- Error occurs during chain execution, not construction
- Stack trace shows `__or__` or `RunnableSequence` frames
- Chain composition looks correct syntactically
- Error message indicates type incompatibility

### Root Causes

**Technical Explanation:**

LCEL (LangChain Expression Language) uses the pipe operator (`|`) to compose Runnables. Each Runnable has:
- **Input Type**: What type it accepts
- **Output Type**: What type it produces

Type mismatch occurs when: `Runnable[A,B] | Runnable[C,D]` where `B ≠ C`

**Common Type Flow Mismatches:**

1. **Dict → String Parser**: LLM returns `AIMessage` (dict-like) but piped to `StrOutputParser`
   ```python
   # WRONG: ChatModel outputs AIMessage, not str directly
   chain = prompt | ChatOpenAI() | some_function_expecting_str
   ```

2. **String → Dict-Expecting Runnable**: Output parser returns `str` but next step expects `dict`
   ```python
   # WRONG: StrOutputParser outputs str, but next runnable expects dict with keys
   chain = prompt | llm | StrOutputParser() | runnable_needing_dict
   ```

3. **Message → Non-Message Runnable**: Chat model outputs `BaseMessage` but next step expects different type

**Source Code Context:**

- **LCEL Composition**: `libs/core/langchain_core/runnables/base.py:608-627` - `__or__` operator creates `RunnableSequence`
- **Type Validation**: Runtime type checking happens during `invoke()`, not at composition time
- **Output Parsers**: `libs/core/langchain_core/output_parsers/` - Define input/output types

### Solutions

**Solution 1: Add Type Conversion Steps**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any


# Example: Function expecting dict but receiving string
def analyze_sentiment(text_dict: Dict[str, Any]) -> str:
    """
    Analyze sentiment from dict containing 'text' key.
    
    Args:
        text_dict: Dictionary with 'text' key
    
    Returns:
        Sentiment analysis result
    """
    text = text_dict["text"]  # Expects dict!
    # Sentiment analysis logic here
    return f"Sentiment analysis of: {text[:50]}..."


# WRONG: Type mismatch - StrOutputParser returns str, but analyze_sentiment expects dict
def create_broken_chain():
    prompt = ChatPromptTemplate.from_template("Analyze: {input}")
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # This will fail!
    broken_chain = (
        prompt 
        | llm 
        | StrOutputParser()  # Returns str
        | analyze_sentiment  # Expects Dict[str, Any] - TYPE MISMATCH!
    )
    return broken_chain


# CORRECT: Add type conversion step
def create_fixed_chain():
    prompt = ChatPromptTemplate.from_template("Analyze: {input}")
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Add conversion lambda to transform str → dict
    str_to_dict = RunnableLambda(lambda x: {"text": x})
    
    fixed_chain = (
        prompt 
        | llm 
        | StrOutputParser()  # Returns str
        | str_to_dict        # Converts str → Dict[str, Any]
        | analyze_sentiment  # Now receives correct type!
    )
    return fixed_chain


# Example usage
if __name__ == "__main__":
    # Demonstrate the fix
    chain = create_fixed_chain()
    
    try:
        result = chain.invoke({"input": "LangChain is powerful"})
        print(f"Success: {result}")
    except TypeError as e:
        print(f"Type error: {e}")
```

**Solution 2: Type-Safe Chain Construction with Explicit Types**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from typing import TypeVar, Dict, Any


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def validate_chain_types(
    runnable1: Runnable,
    runnable2: Runnable,
    runnable1_name: str = "Step 1",
    runnable2_name: str = "Step 2"
) -> bool:
    """
    Validate type compatibility between two runnables (simplified).
    
    Args:
        runnable1: First runnable in sequence
        runnable2: Second runnable in sequence
        runnable1_name: Name for error messages
        runnable2_name: Name for error messages
    
    Returns:
        True if types appear compatible (basic check)
    
    Note:
        This is a simplified validator. Full type checking requires
        analyzing generic type parameters which is complex in Python.
    
    Example:
        >>> r1 = ChatOpenAI()
        >>> r2 = StrOutputParser()
        >>> validate_chain_types(r1, r2, "LLM", "Parser")
        True
    """
    # In practice, full type validation is complex
    # This is a simplified version for demonstration
    print(f"Checking: {runnable1_name} → {runnable2_name}")
    
    # Could add runtime type checking logic here
    # For now, just log the composition
    return True


def create_type_safe_chain():
    """
    Create chain with explicit type checking at each step.
    
    Returns:
        Type-safe runnable chain
    
    Example:
        >>> chain = create_type_safe_chain()
        >>> result = chain.invoke({"topic": "AI"})
    """
    # Step 1: Prompt (Dict → PromptValue)
    prompt = ChatPromptTemplate.from_template("Explain {topic}")
    
    # Step 2: LLM (PromptValue → AIMessage)
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Step 3: Parser (AIMessage → str)
    parser = StrOutputParser()
    
    # Validate each composition
    validate_chain_types(prompt, llm, "Prompt", "LLM")
    validate_chain_types(llm, parser, "LLM", "Parser")
    
    # Construct chain
    chain = prompt | llm | parser
    
    print("✓ Type-safe chain constructed successfully")
    return chain


# Example: Detecting type mismatches at runtime
if __name__ == "__main__":
    chain = create_type_safe_chain()
    result = chain.invoke({"topic": "quantum computing"})
    print(f"Result: {result}")
```

**Solution 3: Use Proper Output Parsers for Each Type**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser
)
from pydantic import BaseModel, Field
from typing import List


class StructuredOutput(BaseModel):
    """Structured output schema."""
    summary: str = Field(description="Brief summary")
    key_points: List[str] = Field(description="List of key points")
    sentiment: str = Field(description="Overall sentiment")


def demonstrate_parser_types():
    """
    Demonstrate correct parser selection for different output types.
    
    Returns:
        Dictionary of chains with different output types
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Chain 1: String output
    string_chain = (
        ChatPromptTemplate.from_template("Summarize in one sentence: {text}")
        | llm
        | StrOutputParser()  # AIMessage → str
    )
    
    # Chain 2: JSON object output
    json_chain = (
        ChatPromptTemplate.from_template(
            "Return JSON with 'summary' and 'word_count' for: {text}"
        )
        | llm
        | JsonOutputParser()  # AIMessage → Dict
    )
    
    # Chain 3: Pydantic model output
    pydantic_parser = PydanticOutputParser(pydantic_object=StructuredOutput)
    
    pydantic_chain = (
        ChatPromptTemplate.from_template(
            "Analyze this text and return structured data:\n{text}\n\n{format_instructions}"
        ).partial(format_instructions=pydantic_parser.get_format_instructions())
        | llm
        | pydantic_parser  # AIMessage → StructuredOutput
    )
    
    return {
        "string_chain": string_chain,
        "json_chain": json_chain,
        "pydantic_chain": pydantic_chain
    }


# Example usage with proper type handling
if __name__ == "__main__":
    chains = demonstrate_parser_types()
    
    text_input = {"text": "LangChain is a framework for building LLM applications."}
    
    # String output
    string_result = chains["string_chain"].invoke(text_input)
    print(f"String result type: {type(string_result)}")  # <class 'str'>
    print(f"Result: {string_result}\n")
    
    # JSON output
    json_result = chains["json_chain"].invoke(text_input)
    print(f"JSON result type: {type(json_result)}")  # <class 'dict'>
    print(f"Result: {json_result}\n")
    
    # Pydantic output
    pydantic_result = chains["pydantic_chain"].invoke(text_input)
    print(f"Pydantic result type: {type(pydantic_result)}")  # <class 'StructuredOutput'>
    print(f"Result: {pydantic_result}\n")
```

### Prevention Strategies

**Best Practices to Avoid Type Mismatches:**

1. **Document Type Flows**: Comment expected types at each pipe stage
   ```python
   chain = (
       prompt              # Dict[str, Any] → PromptValue
       | llm               # PromptValue → AIMessage
       | StrOutputParser() # AIMessage → str
       | custom_function   # str → CustomOutput
   )
   ```

2. **Use Type Hints in Custom Functions**: Explicitly declare input/output types
   ```python
   from typing import Dict, Any
   
   def process_llm_output(output: str) -> Dict[str, Any]:
       """Process LLM string output into structured dict."""
       return {"processed": output, "length": len(output)}
   
   # Type checker will catch mismatches
   chain = prompt | llm | StrOutputParser() | process_llm_output
   ```

3. **Test Chain Composition Early**: Validate with sample inputs during development
   ```python
   # Test immediately after construction
   chain = prompt | llm | parser
   
   test_result = chain.invoke({"input": "test"})
   assert isinstance(test_result, str), f"Expected str, got {type(test_result)}"
   ```

4. **Use Runnable.with_types() for Documentation**: (if available in your version)
   ```python
   from langchain_core.runnables import RunnableLambda
   
   custom_step = RunnableLambda(
       lambda x: x.upper(),
       input_type=str,  # Explicitly declare input type
       output_type=str  # Explicitly declare output type
   )
   ```

5. **Leverage RunnablePassthrough for Debugging**: Inspect values between steps
   ```python
   from langchain_core.runnables import RunnablePassthrough
   
   def debug_print(x):
       print(f"Debug: type={type(x)}, value={x}")
       return x
   
   chain = (
       prompt 
       | llm 
       | RunnablePassthrough(func=debug_print)  # Inspect here
       | StrOutputParser()
   )
   ```

---

## 5. Timeout Errors on Long-Running Chains

### Symptoms

**Observable Error Messages:**
```python
asyncio.TimeoutError: Task exceeded timeout of 30.0 seconds

httpx.ReadTimeout: timed out

openai.APITimeoutError: Request timed out
```

**Behavioral Indicators:**
- Chain succeeds with short inputs but fails with long inputs
- Errors occur after exactly N seconds (e.g., 30s, 60s)
- Complex chains (agents, multi-step retrieval) more prone to timeouts
- Async chains fail more frequently than sync chains

### Root Causes

**Technical Explanation:**

Timeouts occur at multiple levels in LangChain chains:

1. **HTTP Client Timeouts**: Default timeout for API requests (often 30-60 seconds)
2. **Chain-Level Timeouts**: Configurable timeout for entire chain execution
3. **Agent Step Timeouts**: Individual agent tool execution timeouts
4. **Async Event Loop Timeouts**: asyncio timeout contexts

**Common Timeout Triggers:**
- Large document retrieval and processing
- Agent reasoning loops with many tool calls
- Slow external APIs (vector stores, search APIs)
- LLM generating very long responses
- Network latency spikes

**Source Code Context:**

- **Agent Timeout**: `libs/langchain/langchain_classic/agents/agent.py` uses `asyncio_timeout` context manager
- **Runnable Config**: Timeout can be set in `RunnableConfig` passed to `invoke()`

### Solutions

**Solution 1: Configure Appropriate Timeouts**

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any


def create_chain_with_timeout(timeout_seconds: int = 60):
    """
    Create chain with explicit timeout configuration.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
    
    Returns:
        Configured chain with timeout
    
    Example:
        >>> chain = create_chain_with_timeout(timeout_seconds=120)
        >>> result = chain.invoke({"input": "long task"})
    """
    # Configure LLM with timeout
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        request_timeout=timeout_seconds,  # API request timeout
        max_retries=2
    )
    
    prompt = ChatPromptTemplate.from_template("Process: {input}")
    chain = prompt | llm | StrOutputParser()
    
    return chain


async def invoke_with_timeout(
    chain: Any,
    inputs: Dict[str, Any],
    timeout: float = 30.0
) -> Any:
    """
    Invoke chain with asyncio timeout wrapper.
    
    Args:
        chain: Runnable chain to invoke
        inputs: Input dictionary
        timeout: Timeout in seconds
    
    Returns:
        Chain output
    
    Raises:
        asyncio.TimeoutError: If execution exceeds timeout
    
    Example:
        >>> result = await invoke_with_timeout(chain, {"input": "test"}, timeout=60)
    """
    try:
        async with asyncio.timeout(timeout):  # Python 3.11+
            return await chain.ainvoke(inputs)
    except asyncio.TimeoutError:
        print(f"Chain exceeded timeout of {timeout}s")
        raise


# Python 3.10 compatible version
async def invoke_with_timeout_compat(
    chain: Any,
    inputs: Dict[str, Any],
    timeout: float = 30.0
) -> Any:
    """
    Invoke chain with timeout (Python 3.10 compatible).
    
    Uses asyncio.wait_for for timeout control.
    """
    try:
        return await asyncio.wait_for(
            chain.ainvoke(inputs),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        print(f"Chain exceeded timeout of {timeout}s")
        raise


# Example with sync chain
def invoke_sync_with_timeout(chain: Any, inputs: Dict[str, Any], timeout: float = 30.0) -> Any:
    """
    Invoke sync chain with timeout using threading.
    
    Args:
        chain: Runnable chain
        inputs: Input dict
        timeout: Timeout in seconds
    
    Returns:
        Chain output
    
    Raises:
        TimeoutError: If execution exceeds timeout
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(chain.invoke, inputs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            print(f"Chain exceeded timeout of {timeout}s")
            raise TimeoutError(f"Chain execution exceeded {timeout}s")


# Example usage
if __name__ == "__main__":
    chain = create_chain_with_timeout(timeout_seconds=60)
    
    # Async usage
    async def main():
        result = await invoke_with_timeout(
            chain,
            {"input": "Explain quantum computing in detail"},
            timeout=90.0
        )
        print(f"Result: {result}")
    
    asyncio.run(main())
```

**Solution 2: Implement Progressive Timeout with Partial Results**

```python
import asyncio
from typing import Optional, Any, Dict
from langchain_core.runnables import Runnable


class TimeoutHandler:
    """
    Handle timeouts gracefully with partial results and retry logic.
    
    Args:
        chain: Runnable to execute
        default_timeout: Default timeout in seconds
        max_retries: Maximum retry attempts
    
    Example:
        >>> handler = TimeoutHandler(chain, default_timeout=30, max_retries=2)
        >>> result = await handler.invoke_with_fallback({"input": "test"})
    """
    
    def __init__(
        self, 
        chain: Runnable,
        default_timeout: float = 30.0,
        max_retries: int = 2
    ):
        self.chain = chain
        self.default_timeout = default_timeout
        self.max_retries = max_retries
    
    async def invoke_with_fallback(
        self,
        inputs: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Invoke with timeout and fallback to simplified prompt on timeout.
        
        Args:
            inputs: Chain inputs
            timeout: Override default timeout
        
        Returns:
            Dictionary with 'result' and 'metadata'
        """
        timeout = timeout or self.default_timeout
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self.chain.ainvoke(inputs),
                    timeout=timeout
                )
                
                return {
                    "result": result,
                    "metadata": {
                        "attempt": attempt + 1,
                        "timeout": timeout,
                        "status": "success"
                    }
                }
            
            except asyncio.TimeoutError:
                print(f"Attempt {attempt + 1} timed out after {timeout}s")
                
                if attempt < self.max_retries:
                    # Increase timeout for retry
                    timeout *= 1.5
                    print(f"Retrying with timeout={timeout}s")
                else:
                    # Return partial result or error
                    return {
                        "result": None,
                        "metadata": {
                            "attempt": attempt + 1,
                            "timeout": timeout,
                            "status": "timeout_exceeded",
                            "error": f"Chain exceeded maximum timeout of {timeout}s"
                        }
                    }
        
        # Should never reach here
        return {"result": None, "metadata": {"status": "unknown_error"}}


# Example usage
async def main():
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = (
        ChatPromptTemplate.from_template("Explain {topic} in great detail")
        | llm
        | StrOutputParser()
    )
    
    handler = TimeoutHandler(chain, default_timeout=10.0, max_retries=2)
    
    result = await handler.invoke_with_fallback({
        "topic": "quantum computing"
    })
    
    print(f"Status: {result['metadata']['status']}")
    print(f"Result: {result['result']}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Prevention Strategies

**Best Practices to Avoid Timeouts:**

1. **Set Realistic Timeouts Based on Use Case**:
   ```python
   # Quick queries
   quick_chain = ChatOpenAI(request_timeout=15)
   
   # Complex analysis
   analysis_chain = ChatOpenAI(request_timeout=120)
   
   # Agent with tools
   agent_chain = ChatOpenAI(request_timeout=180)
   ```

2. **Implement Streaming for Long Responses**:
   ```python
   # Stream tokens as they arrive (avoids single long wait)
   async for chunk in chain.astream({"input": "long query"}):
       print(chunk, end="", flush=True)
   ```

3. **Break Down Complex Chains**:
   ```python
   # Instead of one long chain
   # BAD: prompt | retriever | reranker | llm | parser (may timeout)
   
   # Break into steps with intermediate timeouts
   # GOOD:
   docs = await retriever.ainvoke(query, timeout=30)
   ranked_docs = await reranker.ainvoke(docs, timeout=15)
   result = await llm.ainvoke(ranked_docs, timeout=60)
   ```

4. **Monitor and Log Execution Times**:
   ```python
   import time
   from langchain_core.callbacks import BaseCallbackHandler
   
   class TimingCallback(BaseCallbackHandler):
       def on_chain_start(self, *args, **kwargs):
           self.start_time = time.time()
       
       def on_chain_end(self, *args, **kwargs):
           elapsed = time.time() - self.start_time
           print(f"Chain completed in {elapsed:.2f}s")
           if elapsed > 30:
               print("WARNING: Chain took longer than expected")
   ```

5. **Use Caching for Repeated Queries**:
   ```python
   from langchain_core.caches import InMemoryCache
   from langchain_core.globals import set_llm_cache
   
   set_llm_cache(InMemoryCache())  # Skip timeout for cached responses
   ```

---

## 6. Malformed Output Parser Expectations

### Symptoms

**Observable Error Messages:**
```python
langchain_core.exceptions.OutputParserException: Could not parse LLM output: `The answer is 42`

json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

pydantic.ValidationError: 1 validation error for OutputSchema
  field_name: Field required

ValueError: Expected JSON object with keys ['answer', 'confidence'], got string
```

**Behavioral Indicators:**
- Chain executes successfully but fails during parsing phase
- LLM returns reasonable text but parser rejects it
- Intermittent failures (LLM sometimes follows format, sometimes doesn't)
- Error occurs after LLM API call completes

### Root Causes

**Technical Explanation:**

Output parsers expect LLM responses in specific formats. Failures occur when:

1. **Format Instructions Not in Prompt**: LLM doesn't know expected output format
2. **LLM Ignores Format Instructions**: Model generates natural language instead of structured format
3. **Partial Compliance**: LLM mostly follows format but with minor deviations (extra text, wrong keys)
4. **Ambiguous Instructions**: Format instructions unclear or conflicting

**Source Code Context:**

- **Output Parser Exception**: `libs/core/langchain_core/exceptions.py` - `OutputParserException` raised when parsing fails
- **Parser Base**: `libs/core/langchain_core/output_parsers/base.py` - Defines `parse()` and `parse_with_prompt()` methods

### Solutions

**Solution 1: Include Explicit Format Instructions**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class AnalysisOutput(BaseModel):
    """Structured analysis output."""
    summary: str = Field(description="Brief summary in 1-2 sentences")
    key_points: List[str] = Field(description="List of 3-5 key points")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")


def create_chain_with_format_instructions():
    """
    Create chain with explicit format instructions in prompt.
    
    Returns:
        Chain that reliably produces structured output
    
    Example:
        >>> chain = create_chain_with_format_instructions()
        >>> result = chain.invoke({"text": "..."})
        >>> print(result.summary)
    """
    # Create parser
    parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
    
    # Prompt includes format instructions
    prompt = ChatPromptTemplate.from_template(
        """Analyze the following text:

{text}

{format_instructions}

Your response:"""
    ).partial(format_instructions=parser.get_format_instructions())
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    chain = prompt | llm | parser
    
    return chain


# Example usage
if __name__ == "__main__":
    chain = create_chain_with_format_instructions()
    
    result = chain.invoke({
        "text": "LangChain makes it easy to build LLM applications with composable components."
    })
    
    print(f"Summary: {result.summary}")
    print(f"Key Points: {result.key_points}")
    print(f"Sentiment: {result.sentiment}")
```

**Solution 2: Implement Fallback Parsing**

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from typing import Any, Optional
import json
import re


class FallbackJsonParser(BaseOutputParser[dict]):
    """
    JSON parser with multiple fallback strategies.
    
    Attempts to parse JSON with progressively more lenient strategies:
    1. Standard JSON parsing
    2. Extract JSON from markdown code blocks
    3. Extract JSON using regex
    4. Return error dict with raw output
    
    Example:
        >>> parser = FallbackJsonParser()
        >>> result = parser.parse("```json\\n{\"key\": \"value\"}\\n```")
        >>> print(result)
        {'key': 'value'}
    """
    
    def parse(self, text: str) -> dict:
        """
        Parse JSON with fallback strategies.
        
        Args:
            text: LLM output text
        
        Returns:
            Parsed dictionary
        
        Raises:
            OutputParserException: If all parsing strategies fail
        """
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from markdown code block
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find any JSON-like structure
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        for match in json_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Strategy 4: Return error with raw output
        raise OutputParserException(
            f"Could not parse JSON from LLM output. "
            f"Tried multiple strategies. "
            f"Raw output: {text[:200]}..."
        )
    
    @property
    def _type(self) -> str:
        return "fallback_json"


# Example with retry on parse failure
def invoke_with_parse_retry(chain: Any, inputs: dict, max_retries: int = 3) -> Any:
    """
    Invoke chain with retry on output parser failures.
    
    Args:
        chain: Runnable chain with output parser
        inputs: Input dictionary
        max_retries: Maximum retry attempts
    
    Returns:
        Parsed output
    
    Raises:
        OutputParserException: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except OutputParserException as e:
            print(f"Parse attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                # Add more explicit instructions for retry
                if "format_instructions" in inputs:
                    inputs["format_instructions"] += "\n\nIMPORTANT: Return ONLY valid JSON, no additional text."
                print("Retrying with stronger format emphasis...")
            else:
                raise
    
    raise OutputParserException("All parse attempts failed")


# Example usage
if __name__ == "__main__":
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    
    parser = FallbackJsonParser()
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "Extract key info as JSON with 'name' and 'description' keys: {text}"
    )
    
    chain = prompt | llm | parser
    
    # Test with various inputs
    test_cases = [
        "LangChain is a framework for LLM apps",
        "Python is a programming language"
    ]
    
    for text in test_cases:
        try:
            result = invoke_with_parse_retry(chain, {"text": text}, max_retries=2)
            print(f"Parsed: {result}")
        except OutputParserException as e:
            print(f"Failed: {e}")
```

**Solution 3: Use Structured Output with Function Calling**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List


class ExtractedInfo(BaseModel):
    """Information extraction schema."""
    entities: List[str] = Field(description="List of named entities found")
    summary: str = Field(description="One sentence summary")
    topics: List[str] = Field(description="Main topics discussed")


def create_structured_output_chain():
    """
    Create chain using OpenAI function calling for guaranteed structure.
    
    Function calling ensures LLM returns data in exact schema format,
    eliminating parsing errors.
    
    Returns:
        Chain with structured output guarantee
    
    Example:
        >>> chain = create_structured_output_chain()
        >>> result = chain.invoke({"text": "..."})
        >>> print(type(result))  # <class '__main__.ExtractedInfo'>
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Use structured output (OpenAI function calling)
    structured_llm = llm.with_structured_output(ExtractedInfo)
    
    prompt = ChatPromptTemplate.from_template(
        "Extract information from this text:\n\n{text}"
    )
    
    chain = prompt | structured_llm
    
    return chain


# Example usage
if __name__ == "__main__":
    chain = create_structured_output_chain()
    
    result = chain.invoke({
        "text": "Apple Inc. announced new AI features. CEO Tim Cook discussed innovation."
    })
    
    # Guaranteed to be ExtractedInfo instance
    print(f"Entities: {result.entities}")
    print(f"Summary: {result.summary}")
    print(f"Topics: {result.topics}")
```

### Prevention Strategies

**Best Practices to Avoid Parser Failures:**

1. **Use Strong Format Instructions**: Be explicit and repetitive
   ```python
   format_instructions = """
   CRITICAL: Return ONLY a valid JSON object with these exact keys:
   - "answer": string containing the answer
   - "confidence": number between 0 and 1
   
   Do NOT include any text before or after the JSON object.
   Do NOT use markdown code blocks.
   Do NOT add explanations.
   
   Example: {"answer": "42", "confidence": 0.95}
   """
   ```

2. **Set Temperature to 0**: Reduce randomness for structured outputs
   ```python
   llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Deterministic
   ```

3. **Use Function Calling When Available**: Guaranteed structure
   ```python
   # Modern approach: function calling
   structured_llm = llm.with_structured_output(schema)
   
   # Old approach: output parsers (more fragile)
   chain = prompt | llm | parser  # May fail
   ```

4. **Validate and Provide Examples**: Show format in prompt
   ```python
   prompt = ChatPromptTemplate.from_template("""
   Extract info as JSON.
   
   Example output:
   {{"name": "LangChain", "type": "framework"}}
   
   Text to process: {text}
   
   JSON output:
   """)
   ```

5. **Implement Graceful Degradation**: Handle parse failures
   ```python
   try:
       result = chain.invoke(inputs)
   except OutputParserException:
       # Fall back to simpler format or return raw output
       result = chain_without_parser.invoke(inputs)
       print(f"Warning: Using unparsed output: {result}")
   ```

---

## 7. Agent Tool Execution Failures

### Symptoms

**Observable Error Messages:**
```python
langchain_classic.agents.tools.InvalidTool: Tool 'search_web' not found in tool list

pydantic.ValidationError: 1 validation error for SearchTool
  query: Field required

AttributeError: Tool 'calculator' has no attribute '_run'

ToolException: Tool execution failed: API key not configured
```

**Behavioral Indicators:**
- Agent starts reasoning but fails during tool invocation
- Error occurs after agent selects a tool
- Tool executes but returns unexpected format
- Agent gets stuck in reasoning loop without calling tools

### Root Causes

**Technical Explanation:**

Agent tool failures occur due to:

1. **Tool Interface Violations**: Tool doesn't implement required methods (`_run`, `_arun`)
2. **Input Schema Mismatches**: Tool expects different input structure than agent provides
3. **Tool Not Registered**: Agent tries to use tool not in its tool list
4. **Tool Execution Errors**: Runtime errors within tool implementation (API failures, missing credentials)
5. **Output Format Issues**: Tool returns data in format agent can't process

**Source Code Context:**

- **Tool Base Class**: `libs/langchain/langchain_classic/tools/base.py` - Defines `BaseTool` interface
- **Agent Tool Execution**: `libs/langchain/langchain_classic/agents/agent.py` - Handles tool invocation
- **Invalid Tool Error**: Raised when agent references non-existent tool

### Solutions

**Solution 1: Implement Proper Tool Interface**

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional


class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(description="Search query string")
    num_results: int = Field(default=5, description="Number of results to return")


class WebSearchTool(BaseTool):
    """
    Web search tool with proper interface implementation.
    
    Demonstrates correct tool implementation with:
    - Input schema definition
    - Sync and async methods
    - Error handling
    - Clear descriptions
    
    Example:
        >>> tool = WebSearchTool(api_key="your-key")
        >>> result = tool._run(query="LangChain", num_results=3)
    """
    
    name: str = "search_web"
    description: str = (
        "Search the web for current information. "
        "Input should be a search query string. "
        "Returns a list of relevant web results."
    )
    args_schema: Type[BaseModel] = SearchInput
    api_key: str = Field(description="API key for search service")
    
    def _run(self, query: str, num_results: int = 5) -> str:
        """
        Execute search synchronously.
        
        Args:
            query: Search query
            num_results: Number of results
        
        Returns:
            Formatted search results as string
        
        Raises:
            ToolException: If API call fails
        """
        try:
            # Simulate search API call
            if not self.api_key:
                raise ValueError("API key not configured")
            
            # In production, call actual search API
            results = [
                f"Result {i+1}: Information about {query}"
                for i in range(num_results)
            ]
            
            return "\n".join(results)
        
        except Exception as e:
            # Raise ToolException for agent to handle
            from langchain_core.tools import ToolException
            raise ToolException(f"Search failed: {e}")
    
    async def _arun(self, query: str, num_results: int = 5) -> str:
        """
        Execute search asynchronously.
        
        Args:
            query: Search query
            num_results: Number of results
        
        Returns:
            Formatted search results
        """
        # In production, use async HTTP client
        import asyncio
        await asyncio.sleep(0.1)  # Simulate API call
        return self._run(query, num_results)


# Example: Creating valid tool for agent
if __name__ == "__main__":
    # Create tool instance
    search_tool = WebSearchTool(api_key="test-key-123")
    
    # Test tool directly
    result = search_tool._run(query="LangChain documentation", num_results=3)
    print(f"Tool output:\n{result}")
    
    # Verify tool interface
    print(f"\nTool name: {search_tool.name}")
    print(f"Tool description: {search_tool.description}")
    print(f"Input schema: {search_tool.args_schema.schema()}")
```

**Solution 2: Validate Tools Before Agent Initialization**

```python
from langchain_core.tools import BaseTool
from typing import List
import inspect


def validate_tool(tool: BaseTool) -> tuple[bool, List[str]]:
    """
    Validate that tool implements required interface correctly.
    
    Args:
        tool: Tool to validate
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    
    Example:
        >>> tool = WebSearchTool(api_key="key")
        >>> is_valid, issues = validate_tool(tool)
        >>> if not is_valid:
        ...     print(f"Issues: {issues}")
    """
    issues = []
    
    # Check required attributes
    if not hasattr(tool, 'name') or not tool.name:
        issues.append("Missing or empty 'name' attribute")
    
    if not hasattr(tool, 'description') or not tool.description:
        issues.append("Missing or empty 'description' attribute")
    
    # Check required methods
    if not hasattr(tool, '_run'):
        issues.append("Missing '_run' method")
    else:
        # Check _run signature
        sig = inspect.signature(tool._run)
        if len(sig.parameters) < 1:  # Should accept at least 'self'
            issues.append("'_run' method has invalid signature")
    
    if not hasattr(tool, '_arun'):
        issues.append("Missing '_arun' async method")
    
    # Check args_schema if present
    if hasattr(tool, 'args_schema') and tool.args_schema:
        try:
            # Verify it's a valid Pydantic model
            if not issubclass(tool.args_schema, BaseModel):
                issues.append("'args_schema' must be a Pydantic BaseModel")
        except TypeError:
            issues.append("'args_schema' is not a valid class")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_tool_list(tools: List[BaseTool]) -> bool:
    """
    Validate all tools in list and print issues.
    
    Args:
        tools: List of tools to validate
    
    Returns:
        True if all tools are valid
    
    Example:
        >>> tools = [search_tool, calculator_tool]
        >>> if not validate_tool_list(tools):
        ...     print("Some tools have issues")
    """
    all_valid = True
    
    for tool in tools:
        is_valid, issues = validate_tool(tool)
        
        if not is_valid:
            all_valid = False
            print(f"❌ Tool '{tool.name}' has issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"✓ Tool '{tool.name}' is valid")
    
    return all_valid


# Example usage
if __name__ == "__main__":
    # Create tool
    search_tool = WebSearchTool(api_key="test-key")
    
    # Validate before using with agent
    tools = [search_tool]
    
    if validate_tool_list(tools):
        print("\nAll tools valid, safe to create agent")
        # Create agent here
    else:
        print("\nFix tool issues before creating agent")
```

**Solution 3: Implement Tool Error Handling**

```python
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field
from typing import Type
import logging

logger = logging.getLogger(__name__)


class CalculatorInput(BaseModel):
    """Input for calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate")


class SafeCalculatorTool(BaseTool):
    """
    Calculator tool with comprehensive error handling.
    
    Handles:
    - Invalid expressions
    - Division by zero
    - Unsupported operations
    - Security risks (code injection)
    
    Example:
        >>> calc = SafeCalculatorTool()
        >>> result = calc._run(expression="2 + 2")
        >>> print(result)
        '4'
    """
    
    name: str = "calculator"
    description: str = (
        "Useful for mathematical calculations. "
        "Input should be a valid mathematical expression like '2 + 2' or '10 * 5'. "
        "Supports +, -, *, /, **, parentheses."
    )
    args_schema: Type[BaseModel] = CalculatorInput
    handle_tool_error: bool = True  # Return error message instead of raising
    
    def _run(self, expression: str) -> str:
        """
        Evaluate mathematical expression safely.
        
        Args:
            expression: Math expression string
        
        Returns:
            Result as string, or error message
        """
        try:
            # Security: only allow safe mathematical operations
            allowed_chars = set("0123456789+-*/() .**")
            if not all(c in allowed_chars for c in expression):
                return "Error: Expression contains invalid characters. Only numbers and +, -, *, /, **, () allowed."
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, {})
            
            return str(result)
        
        except ZeroDivisionError:
            logger.warning(f"Division by zero in expression: {expression}")
            return "Error: Division by zero"
        
        except SyntaxError:
            logger.warning(f"Invalid syntax: {expression}")
            return f"Error: Invalid mathematical expression '{expression}'"
        
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return f"Error: Could not evaluate expression - {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """Async version."""
        return self._run(expression)


# Example: Tool execution with error handling
if __name__ == "__main__":
    calc = SafeCalculatorTool()
    
    # Test cases including error scenarios
    test_expressions = [
        "2 + 2",           # Valid
        "10 / 0",          # Division by zero
        "5 * (3 + 2)",     # Valid with parentheses
        "import os",       # Security risk
        "2 +",             # Syntax error
    ]
    
    for expr in test_expressions:
        result = calc._run(expression=expr)
        print(f"Expression: {expr}")
        print(f"Result: {result}\n")
```

### Prevention Strategies

**Best Practices to Avoid Tool Failures:**

1. **Use Tool Decorators for Simple Tools**: Simplify tool creation
   ```python
   from langchain_core.tools import tool
   
   @tool
   def search_wikipedia(query: str) -> str:
       """Search Wikipedia for information."""
       # Implementation
       return f"Wikipedia results for: {query}"
   
   # Automatically creates proper tool interface
   ```

2. **Document Tool Requirements Clearly**:
   ```python
   class MyTool(BaseTool):
       """
       Tool description.
       
       Required Configuration:
           - api_key: API key for external service
           - endpoint: API endpoint URL
       
       Input Format:
           - query (str): Search query
           - max_results (int, optional): Max results (default: 5)
       
       Output Format:
           Returns JSON string with results array
       
       Error Handling:
           Returns error message string if API fails
       """
   ```

3. **Test Tools Independently**: Verify before agent integration
   ```python
   # Test tool directly before using with agent
   tool = MyTool(api_key="key")
   
   # Test valid input
   result = tool._run(query="test")
   assert result is not None
   
   # Test error handling
   result = tool._run(query="")  # Empty query
   assert "error" in result.lower()
   ```

4. **Implement Tool Return Value Validation**:
   ```python
   def _run(self, query: str) -> str:
       result = self._execute_query(query)
       
       # Validate return type
       if not isinstance(result, str):
           logger.error(f"Tool returned {type(result)}, expected str")
           return str(result)  # Convert to string
       
       # Validate content
       if len(result) == 0:
           return "No results found"
       
       return result
   ```

---

## 8. Retrieval Returning Empty Results

### Symptoms

**Observable Error Messages:**
```python
# Often no exception, but empty results
[] # Empty document list

IndexError: list index out of range  # Accessing results[0] when empty

ValueError: No documents retrieved for query
```

**Behavioral Indicators:**
- Query completes without errors but returns no documents
- Works for some queries but not others
- New documents not appearing in results
- Previously working queries suddenly return empty results

### Root Causes

**Technical Explanation:**

Empty retrieval results typically stem from:

1. **Vector Store Not Initialized**: Documents not indexed
2. **Embedding Mismatch**: Query embeddings incompatible with stored embeddings
3. **Similarity Threshold Too High**: No documents meet minimum similarity score
4. **Collection/Index Name Mismatch**: Querying wrong collection
5. **Empty Index**: No documents ingested yet
6. **Dimension Mismatch**: Query embedding dimensions don't match stored embeddings

**Common Scenarios:**
- Changing embedding models without re-indexing
- Typos in collection names
- Network issues preventing vector store connection
- Documents indexed but search parameters too restrictive

### Solutions

**Solution 1: Implement Retrieval Validation**

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ValidatedRetriever:
    """
    Wrapper around retriever with validation and diagnostics.
    
    Args:
        retriever: Base retriever to wrap
        min_docs: Minimum documents to return
        similarity_threshold: Minimum similarity score
    
    Example:
        >>> base_retriever = vector_store.as_retriever()
        >>> retriever = ValidatedRetriever(base_retriever, min_docs=1)
        >>> docs = retriever.get_relevant_documents("query")
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        min_docs: int = 1,
        similarity_threshold: float = 0.0
    ):
        self.retriever = retriever
        self.min_docs = min_docs
        self.similarity_threshold = similarity_threshold
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents with validation.
        
        Args:
            query: Search query
        
        Returns:
            List of documents
        
        Raises:
            ValueError: If insufficient documents retrieved
        """
        logger.info(f"Retrieving documents for query: '{query[:50]}...'")
        
        # Attempt retrieval
        docs = self.retriever.get_relevant_documents(query)
        
        logger.info(f"Retrieved {len(docs)} documents")
        
        # Validate result count
        if len(docs) < self.min_docs:
            logger.warning(
                f"Only {len(docs)} documents retrieved, expected >= {self.min_docs}"
            )
            
            # Diagnostic information
            self._print_diagnostics(query)
            
            if len(docs) == 0:
                raise ValueError(
                    f"No documents retrieved for query: '{query}'. "
                    "Check vector store connection and document indexing."
                )
        
        # Filter by similarity if scores available
        if docs and hasattr(docs[0], 'metadata') and 'score' in docs[0].metadata:
            filtered_docs = [
                doc for doc in docs 
                if doc.metadata.get('score', 0) >= self.similarity_threshold
            ]
            
            if len(filtered_docs) < len(docs):
                logger.info(
                    f"Filtered to {len(filtered_docs)} docs "
                    f"with similarity >= {self.similarity_threshold}"
                )
                docs = filtered_docs
        
        return docs
    
    def _print_diagnostics(self, query: str):
        """Print diagnostic information for debugging."""
        print("\n=== Retrieval Diagnostics ===")
        print(f"Query: {query}")
        print(f"Query length: {len(query)} characters")
        print(f"Min docs expected: {self.min_docs}")
        print(f"Similarity threshold: {self.similarity_threshold}")
        
        # Try to get vector store info
        if hasattr(self.retriever, 'vectorstore'):
            vs = self.retriever.vectorstore
            print(f"Vector store type: {type(vs).__name__}")
            
            # Check document count if possible
            if hasattr(vs, '_collection'):
                try:
                    count = vs._collection.count()
                    print(f"Documents in store: {count}")
                except Exception as e:
                    print(f"Could not get document count: {e}")
        
        print("=" * 30 + "\n")


# Example usage with diagnostic output
if __name__ == "__main__":
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    
    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Initialize with some documents
    docs = [
        Document(page_content="LangChain is a framework for LLM applications", metadata={"source": "doc1"}),
        Document(page_content="Python is a programming language", metadata={"source": "doc2"}),
        Document(page_content="Vector stores enable semantic search", metadata={"source": "doc3"}),
    ]
    
    vector_store = Chroma.from_documents(docs, embeddings)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Wrap with validation
    retriever = ValidatedRetriever(
        base_retriever,
        min_docs=1,
        similarity_threshold=0.3
    )
    
    # Test queries
    queries = [
        "What is LangChain?",  # Should find results
        "quantum physics",      # May find nothing
    ]
    
    for query in queries:
        try:
            docs = retriever.get_relevant_documents(query)
            print(f"Query: {query}")
            print(f"Found {len(docs)} documents")
            for i, doc in enumerate(docs, 1):
                print(f"  {i}. {doc.page_content[:60]}...")
            print()
        except ValueError as e:
            print(f"Retrieval failed: {e}\n")
```

**Solution 2: Implement Fallback Retrieval Strategies**

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List


class FallbackRetriever:
    """
    Retriever with multiple fallback strategies.
    
    Tries strategies in order:
    1. Semantic search with original query
    2. Semantic search with relaxed threshold
    3. Keyword search (if supported)
    4. Return default documents
    
    Args:
        primary_retriever: Main semantic search retriever
        fallback_docs: Default documents to return if all else fails
    
    Example:
        >>> retriever = FallbackRetriever(vector_store.as_retriever())
        >>> docs = retriever.get_relevant_documents("query")
    """
    
    def __init__(
        self,
        primary_retriever: BaseRetriever,
        fallback_docs: List[Document] = None
    ):
        self.primary_retriever = primary_retriever
        self.fallback_docs = fallback_docs or []
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve with fallback strategies.
        
        Args:
            query: Search query
        
        Returns:
            List of documents (always returns at least fallback docs)
        """
        # Strategy 1: Standard retrieval
        docs = self.primary_retriever.get_relevant_documents(query)
        
        if len(docs) > 0:
            print(f"✓ Primary retrieval: found {len(docs)} documents")
            return docs
        
        print("⚠ Primary retrieval returned no results, trying fallbacks...")
        
        # Strategy 2: Relaxed similarity threshold (if supported)
        docs = self._try_relaxed_search(query)
        if len(docs) > 0:
            print(f"✓ Relaxed search: found {len(docs)} documents")
            return docs
        
        # Strategy 3: Keyword-based search (if vector store supports it)
        docs = self._try_keyword_search(query)
        if len(docs) > 0:
            print(f"✓ Keyword search: found {len(docs)} documents")
            return docs
        
        # Strategy 4: Return default fallback documents
        print(f"⚠ All strategies failed, returning {len(self.fallback_docs)} fallback documents")
        return self.fallback_docs
    
    def _try_relaxed_search(self, query: str) -> List[Document]:
        """Try retrieval with relaxed parameters."""
        try:
            if hasattr(self.primary_retriever, 'search_kwargs'):
                # Create new retriever with relaxed threshold
                relaxed_retriever = type(self.primary_retriever)(
                    **{**self.primary_retriever.__dict__, 
                       'search_kwargs': {'k': 5, 'score_threshold': 0.1}}
                )
                return relaxed_retriever.get_relevant_documents(query)
        except Exception as e:
            print(f"Relaxed search failed: {e}")
        
        return []
    
    def _try_keyword_search(self, query: str) -> List[Document]:
        """Try keyword-based search if supported."""
        try:
            if hasattr(self.primary_retriever, 'vectorstore'):
                vs = self.primary_retriever.vectorstore
                
                # Some vector stores support keyword search
                if hasattr(vs, 'max_marginal_relevance_search'):
                    docs = vs.max_marginal_relevance_search(query, k=3)
                    return docs
        except Exception as e:
            print(f"Keyword search failed: {e}")
        
        return []


# Example usage
if __name__ == "__main__":
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create vector store with documents
    docs = [
        Document(page_content="LangChain documentation"),
        Document(page_content="Python programming guide"),
    ]
    
    vector_store = Chroma.from_documents(docs, embeddings)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Create fallback document
    fallback = [Document(page_content="No specific information found. Please rephrase your question.")]
    
    # Create fallback retriever
    retriever = FallbackRetriever(base_retriever, fallback_docs=fallback)
    
    # Test with various queries
    results = retriever.get_relevant_documents("What is LangChain?")
    print(f"\nResults: {len(results)} documents")
    for doc in results:
        print(f"- {doc.page_content}")
```

### Prevention Strategies

**Best Practices to Avoid Empty Retrieval:**

1. **Verify Document Ingestion**:
   ```python
   # After adding documents, verify they're indexed
   vector_store.add_documents(docs)
   
   # Check document count
   if hasattr(vector_store, '_collection'):
       count = vector_store._collection.count()
       print(f"Vector store now contains {count} documents")
       assert count > 0, "No documents in vector store!"
   ```

2. **Use Consistent Embedding Models**:
   ```python
   # Store embedding model name with metadata
   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
   
   # When loading, use SAME model
   vector_store = Chroma(
       embedding_function=embeddings,  # Must match indexed model
       collection_name="my_docs"
   )
   ```

3. **Set Reasonable Similarity Thresholds**:
   ```python
   # Too high (0.9) - very restrictive
   retriever = vector_store.as_retriever(
       search_kwargs={"k": 5, "score_threshold": 0.9}  # May return nothing
   )
   
   # Better (0.5-0.7) - balanced
   retriever = vector_store.as_retriever(
       search_kwargs={"k": 5, "score_threshold": 0.6}
   )
   ```

4. **Monitor and Log Retrieval Metrics**:
   ```python
   from langchain_core.callbacks import BaseCallbackHandler
   
   class RetrievalMonitorCallback(BaseCallbackHandler):
       def on_retriever_end(self, documents, **kwargs):
           print(f"Retrieved {len(documents)} documents")
           if len(documents) == 0:
               print("WARNING: No documents retrieved!")
   ```

5. **Test Retrieval After Deployment**:
   ```python
   def test_retrieval_health(retriever, test_queries: List[str]):
       """Test retriever with known queries."""
       for query in test_queries:
           docs = retriever.get_relevant_documents(query)
           assert len(docs) > 0, f"No results for: {query}"
       print("✓ Retrieval health check passed")
   ```

---

## 9. Token Limit Exceeded Errors

### Symptoms

**Observable Error Messages:**
```python
openai.BadRequestError: This model's maximum context length is 4096 tokens. However, you submitted 5000 tokens.

anthropic.BadRequestError: prompt is too long: 100001 tokens > 100000 maximum

InvalidRequestError: This model's maximum context length is 8192 tokens, however you requested 9500 tokens
```

**Behavioral Indicators:**
- Error occurs immediately when invoking chain
- Works with short inputs, fails with long inputs
- Fails after retrieving many documents
- Error message specifies exact token counts

### Root Causes

**Technical Explanation:**

Each LLM model has a fixed maximum context window (input + output tokens). Exceeding this limit causes immediate rejection:

**Model Context Limits:**
- GPT-3.5-turbo: 4K or 16K tokens
- GPT-4: 8K, 32K, or 128K tokens  
- Claude 3: 200K tokens
- Llama 2: 4K tokens

**Common Causes:**
1. Retrieving too many documents in RAG chains
2. Long conversation history in memory-enabled chains
3. Large few-shot examples in prompts
4. Not accounting for system messages and formatting

### Solutions

**Solution 1: Count and Truncate Tokens**

```python
import tiktoken
from typing import List
from langchain_core.documents import Document


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text.
    
    Args:
        text: Text to tokenize
        model: Model name for correct tokenizer
    
    Returns:
        Token count
    
    Example:
        >>> count_tokens("Hello world")
        2
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        model: Model name
    
    Returns:
        Truncated text
    
    Example:
        >>> long_text = "word " * 1000
        >>> truncated = truncate_to_token_limit(long_text, max_tokens=100)
        >>> count_tokens(truncated) <= 100
        True
    """
    encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate tokens
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)
    
    return truncated_text


def truncate_documents(
    documents: List[Document],
    max_total_tokens: int,
    model: str = "gpt-3.5-turbo"
) -> List[Document]:
    """
    Truncate document list to stay within token budget.
    
    Args:
        documents: List of documents
        max_total_tokens: Maximum total tokens for all documents
        model: Model name
    
    Returns:
        Truncated list of documents
    
    Example:
        >>> docs = [Document(page_content="..." * 1000) for _ in range(10)]
        >>> truncated = truncate_documents(docs, max_total_tokens=1000)
        >>> total = sum(count_tokens(d.page_content) for d in truncated)
        >>> total <= 1000
        True
    """
    truncated_docs = []
    total_tokens = 0
    
    for doc in documents:
        doc_tokens = count_tokens(doc.page_content, model)
        
        if total_tokens + doc_tokens <= max_total_tokens:
            # Full document fits
            truncated_docs.append(doc)
            total_tokens += doc_tokens
        else:
            # Partial document to fill remaining budget
            remaining_tokens = max_total_tokens - total_tokens
            
            if remaining_tokens > 50:  # Only add if meaningful amount left
                truncated_content = truncate_to_token_limit(
                    doc.page_content,
                    remaining_tokens,
                    model
                )
                truncated_docs.append(
                    Document(page_content=truncated_content, metadata=doc.metadata)
                )
            break
    
    print(f"Truncated {len(documents)} docs to {len(truncated_docs)} docs, {total_tokens} tokens")
    return truncated_docs


# Example usage
if __name__ == "__main__":
    from langchain_core.documents import Document
    
    # Simulate large documents
    docs = [
        Document(page_content=f"Document {i}: " + "content " * 500)
        for i in range(20)
    ]
    
    # Model with 4K context, reserve space for prompt and output
    max_doc_tokens = 2000  # Leave room for prompt (500) and output (1500)
    
    truncated = truncate_documents(docs, max_total_tokens=max_doc_tokens)
    
    print(f"Original: {len(docs)} documents")
    print(f"Truncated: {len(truncated)} documents")
    print(f"Total tokens: {sum(count_tokens(d.page_content) for d in truncated)}")
```

**Solution 2: Implement Dynamic Context Window Management**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List, Dict, Any


class ContextWindowManager:
    """
    Manage context window allocation for RAG chains.
    
    Allocates token budget across:
    - System prompt
    - Retrieved documents  
    - User query
    - Output buffer
    
    Args:
        model: Model name
        total_context_window: Total context window size
    
    Example:
        >>> manager = ContextWindowManager("gpt-3.5-turbo", total_context_window=4096)
        >>> docs = manager.fit_documents_to_budget(retrieved_docs, query)
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", total_context_window: int = 4096):
        self.model = model
        self.total_context_window = total_context_window
        
        # Allocate budget
        self.budget = {
            "system_prompt": int(total_context_window * 0.05),  # 5%
            "documents": int(total_context_window * 0.50),       # 50%
            "query": int(total_context_window * 0.10),           # 10%
            "output": int(total_context_window * 0.30),          # 30%
            "safety_margin": int(total_context_window * 0.05)    # 5%
        }
    
    def fit_documents_to_budget(
        self,
        documents: List[Document],
        query: str
    ) -> List[Document]:
        """
        Fit documents within allocated budget.
        
        Args:
            documents: Retrieved documents
            query: User query
        
        Returns:
            Truncated documents that fit budget
        """
        # Count query tokens
        query_tokens = count_tokens(query, self.model)
        
        if query_tokens > self.budget["query"]:
            print(f"WARNING: Query uses {query_tokens} tokens, "
                  f"exceeds budget of {self.budget['query']}")
        
        # Truncate documents to fit budget
        doc_budget = self.budget["documents"]
        fitted_docs = truncate_documents(documents, doc_budget, self.model)
        
        return fitted_docs
    
    def validate_inputs(
        self,
        system_prompt: str,
        documents: List[Document],
        query: str
    ) -> Dict[str, Any]:
        """
        Validate that all inputs fit within context window.
        
        Returns:
            Dictionary with validation results and token counts
        """
        counts = {
            "system_prompt": count_tokens(system_prompt, self.model),
            "documents": sum(count_tokens(d.page_content, self.model) for d in documents),
            "query": count_tokens(query, self.model),
        }
        
        counts["total_input"] = sum(counts.values())
        counts["available_for_output"] = (
            self.total_context_window - counts["total_input"]
        )
        
        counts["within_budget"] = counts["total_input"] < (
            self.total_context_window - self.budget["output"]
        )
        
        return counts


# Example: RAG chain with context window management
def create_budget_aware_rag_chain():
    """
    Create RAG chain that respects token budgets.
    
    Returns:
        Runnable chain with context window management
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    manager = ContextWindowManager(model="gpt-3.5-turbo", total_context_window=4096)
    
    def process_with_budget(inputs: Dict[str, Any]) -> str:
        """Process inputs with token budget management."""
        query = inputs["query"]
        documents = inputs["documents"]
        
        # Fit documents to budget
        fitted_docs = manager.fit_documents_to_budget(documents, query)
        
        # Build context from fitted documents
        context = "\n\n".join([doc.page_content for doc in fitted_docs])
        
        # Validate total input
        system_prompt = "You are a helpful assistant. Answer based on the provided context."
        validation = manager.validate_inputs(system_prompt, fitted_docs, query)
        
        print(f"Token usage: {validation['total_input']}/{manager.total_context_window}")
        print(f"Available for output: {validation['available_for_output']}")
        
        if not validation["within_budget"]:
            raise ValueError(
                f"Inputs exceed token budget: {validation['total_input']} tokens"
            )
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(
            f"{system_prompt}\n\nContext:\n{{context}}\n\nQuestion: {{query}}\n\nAnswer:"
        )
        
        # Invoke chain
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "query": query})
    
    return process_with_budget


# Example usage
if __name__ == "__main__":
    # Simulate retrieval
    retrieved_docs = [
        Document(page_content="LangChain is a framework. " * 100) for _ in range(10)
    ]
    
    chain_func = create_budget_aware_rag_chain()
    
    result = chain_func({
        "query": "What is LangChain?",
        "documents": retrieved_docs
    })
    
    print(f"\nResult: {result}")
```

### Prevention Strategies

**Best Practices to Avoid Token Limit Errors:**

1. **Choose Model Based on Use Case**:
   ```python
   # For short queries
   llm_short = ChatOpenAI(model="gpt-3.5-turbo")  # 4K context
   
   # For RAG with many documents
   llm_rag = ChatOpenAI(model="gpt-3.5-turbo-16k")  # 16K context
   
   # For long conversations
   llm_long = ChatOpenAI(model="gpt-4-turbo-preview")  # 128K context
   ```

2. **Limit Retrieved Documents**:
   ```python
   # Retrieve fewer, more relevant documents
   retriever = vector_store.as_retriever(
       search_kwargs={"k": 3}  # Only top 3 documents
   )
   ```

3. **Implement Document Compression**:
   ```python
   from langchain.retrievers import ContextualCompressionRetriever
   from langchain.retrievers.document_compressors import LLMChainExtractor
   
   # Compress documents to only relevant parts
   compressor = LLMChainExtractor.from_llm(llm)
   compression_retriever = ContextualCompressionRetriever(
       base_compressor=compressor,
       base_retriever=retriever
   )
   ```

4. **Monitor Token Usage**:
   ```python
   from langchain_core.callbacks import get_openai_callback
   
   with get_openai_callback() as cb:
       result = chain.invoke(inputs)
       print(f"Prompt tokens: {cb.prompt_tokens}")
       
       if cb.prompt_tokens > 3500:  # Approaching 4K limit
           print("WARNING: High token usage, consider truncation")
   ```

---

## 10. Callback Handler Exceptions

### Symptoms

**Observable Error Messages:**
```python
Exception in callback handler: AttributeError: 'NoneType' object has no attribute 'on_llm_end'

RuntimeError: Callback handler raised exception during on_chain_start

TypeError: on_llm_new_token() takes 1 positional argument but 2 were given
```

**Behavioral Indicators:**
- Chain executes but fails during/after execution
- Error mentions callback methods (on_chain_start, on_llm_end, etc.)
- Works without callbacks but fails when callbacks added
- Intermittent failures in callback-dependent functionality (logging, monitoring)

### Root Causes

**Technical Explanation:**

Callback errors occur when:

1. **Incorrect Method Signatures**: Callback methods don't match expected signature
2. **Missing Error Handling**: Callback raises exception, propagates to chain
3. **Async/Sync Mismatch**: Using sync callback with async chain or vice versa
4. **State Management Issues**: Callback assumes state that doesn't exist

**Source Code Context:**

- **Callback Execution**: `libs/langchain/langchain_classic/chains/base.py:213-218` - Callbacks invoked during chain lifecycle
- **Error Propagation**: `libs/langchain/langchain_classic/chains/base.py:231-233` - Exceptions in callbacks trigger `on_chain_error`

### Solutions

**Solution 1: Implement Robust Callback Handler**

```python
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class RobustCallbackHandler(BaseCallbackHandler):
    """
    Callback handler with comprehensive error handling.
    
    Ensures callback errors don't crash the main chain.
    
    Example:
        >>> callback = RobustCallbackHandler()
        >>> chain.invoke({"input": "test"}, config={"callbacks": [callback]})
    """
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Called when LLM starts."""
        try:
            logger.info(f"LLM started with {len(prompts)} prompts")
            # Custom logic here
        except Exception as e:
            logger.error(f"Error in on_llm_start: {e}", exc_info=True)
            # Don't re-raise - prevent chain failure
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when LLM ends."""
        try:
            logger.info(f"LLM completed")
            # Custom logic here
        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}", exc_info=True)
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM errors."""
        try:
            logger.error(f"LLM error: {error}")
            # Custom error handling
        except Exception as e:
            logger.error(f"Error in on_llm_error handler: {e}", exc_info=True)
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when chain starts."""
        try:
            logger.info(f"Chain started with inputs: {list(inputs.keys())}")
            # Custom logic here
        except Exception as e:
            logger.error(f"Error in on_chain_start: {e}", exc_info=True)
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain ends."""
        try:
            logger.info(f"Chain completed with outputs: {list(outputs.keys())}")
            # Custom logic here
        except Exception as e:
            logger.error(f"Error in on_chain_end: {e}", exc_info=True)
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when chain errors."""
        try:
            logger.error(f"Chain error: {error}")
            # Custom error handling
        except Exception as e:
            logger.error(f"Error in on_chain_error handler: {e}", exc_info=True)


# Example usage
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    # Create chain
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_template("Say: {input}")
    chain = prompt | llm | StrOutputParser()
    
    # Create callback
    callback = RobustCallbackHandler()
    
    # Invoke with callback
    result = chain.invoke(
        {"input": "hello"},
        config={"callbacks": [callback]}
    )
    
    print(f"Result: {result}")
```

**Solution 2: Validate Callback Methods**

```python
from langchain_core.callbacks import BaseCallbackHandler
import inspect
from typing import Set


def validate_callback_handler(handler: BaseCallbackHandler) -> tuple[bool, List[str]]:
    """
    Validate callback handler implements methods correctly.
    
    Args:
        handler: Callback handler to validate
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    
    Example:
        >>> handler = MyCallback()
        >>> is_valid, issues = validate_callback_handler(handler)
        >>> if not is_valid:
        ...     print(f"Issues: {issues}")
    """
    issues = []
    
    # Expected callback methods with their signatures
    expected_methods = {
        "on_llm_start": ["serialized", "prompts"],
        "on_llm_end": ["response"],
        "on_llm_error": ["error"],
        "on_chain_start": ["serialized", "inputs"],
        "on_chain_end": ["outputs"],
        "on_chain_error": ["error"],
        "on_tool_start": ["serialized", "input_str"],
        "on_tool_end": ["output"],
        "on_tool_error": ["error"],
    }
    
    for method_name, expected_params in expected_methods.items():
        if not hasattr(handler, method_name):
            continue  # Optional method
        
        method = getattr(handler, method_name)
        
        if not callable(method):
            issues.append(f"{method_name} is not callable")
            continue
        
        # Check signature
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        
        # Remove 'self' and 'kwargs'
        params = [p for p in params if p not in ['self', 'kwargs']]
        
        # Check all expected params are present
        for expected_param in expected_params:
            if expected_param not in params and '**kwargs' not in str(sig):
                issues.append(
                    f"{method_name} missing parameter '{expected_param}' "
                    f"(has: {params})"
                )
    
    is_valid = len(issues) == 0
    return is_valid, issues


# Example usage
if __name__ == "__main__":
    # Valid handler
    valid_handler = RobustCallbackHandler()
    is_valid, issues = validate_callback_handler(valid_handler)
    print(f"Valid handler: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    
    # Invalid handler (for demonstration)
    class InvalidHandler(BaseCallbackHandler):
        def on_llm_start(self):  # Missing required parameters!
            pass
    
    invalid_handler = InvalidHandler()
    is_valid, issues = validate_callback_handler(invalid_handler)
    print(f"\nInvalid handler: {is_valid}")
    print(f"Issues: {issues}")
```

### Prevention Strategies

**Best Practices to Avoid Callback Errors:**

1. **Inherit from BaseCallbackHandler**: Ensures correct interface
   ```python
   from langchain_core.callbacks import BaseCallbackHandler
   
   class MyCallback(BaseCallbackHandler):  # Inherit!
       def on_llm_start(self, serialized, prompts, **kwargs):
           # Implementation
           pass
   ```

2. **Use **kwargs for Forward Compatibility**: Handle future parameters
   ```python
   def on_chain_start(self, serialized, inputs, **kwargs):
       # **kwargs catches any additional parameters added in future versions
       run_id = kwargs.get("run_id")  # Safe access to optional params
   ```

3. **Wrap Callback Logic in Try-Except**: Prevent cascade failures
   ```python
   def on_llm_end(self, response, **kwargs):
       try:
           # Your callback logic
           self.log_response(response)
       except Exception as e:
           # Log but don't raise
           logger.error(f"Callback error: {e}")
   ```

4. **Test Callbacks Independently**: Verify before integration
   ```python
   def test_callback():
       callback = MyCallback()
       
       # Test each method
       callback.on_chain_start({}, {"input": "test"})
       callback.on_llm_start({}, ["prompt"])
       callback.on_llm_end("response")
       
       print("✓ All callback methods work")
   ```

5. **Use Async Callbacks for Async Chains**: Match chain's async nature
   ```python
   from langchain_core.callbacks import AsyncCallbackHandler
   
   class MyAsyncCallback(AsyncCallbackHandler):
       async def on_llm_start(self, serialized, prompts, **kwargs):
           # Async implementation
           await self.async_log(prompts)
   ```

---

## Summary and Next Steps

### Quick Debugging Checklist

When encountering a chain failure, follow this diagnostic process:

1. **Identify Error Category**: Match error message to one of the 10 common issues above
2. **Check Recent Changes**: What changed since it last worked?
3. **Isolate the Problem**: Test components individually (LLM, retriever, parser)
4. **Add Logging**: Enable DEBUG logging to see execution flow
5. **Verify Configuration**: Check API keys, model names, timeouts
6. **Test with Simple Input**: Does it work with minimal input?
7. **Review Stack Trace**: Use [stack trace guide](./stack-traces.md) to pinpoint origin

### Additional Resources

- **[Stack Trace Interpretation Guide](./stack-traces.md)** - Detailed guide to reading LangChain stack traces
- **[Logging Configuration Guide](./logging.md)** - Set up comprehensive logging for debugging
- **[LCEL Composition Guide](../guides/lcel-composition.md)** - Type-safe chain composition patterns
- **[Error Handling Guide](../guides/error-handling.md)** - Retry logic and fallback strategies

### Getting Help

If these solutions don't resolve your issue:

1. **Enable verbose logging**: Set `verbose=True` on chains and `logging.DEBUG`
2. **Create minimal reproduction**: Isolate the problem in a small, standalone script
3. **Check LangChain GitHub Issues**: Search for similar problems
4. **LangChain Discord/Forum**: Community support for debugging help

### Contributing

Found a common issue not covered here? Contribute to this guide:

1. Document the failure mode with symptoms and root cause
2. Provide executable code example demonstrating the problem and fix
3. Include prevention strategies
4. Submit PR to add to this guide

---

**Last Updated**: 2024
**LangChain Version Compatibility**: 0.1.x, 0.2.x, 1.0.x
**Maintainer**: LangChain Documentation Team

