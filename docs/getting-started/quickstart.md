# Quickstart

Get up and running with LangChain in 5 minutes by building your first chain—a simple yet powerful pattern for combining prompts with language models.

## Introduction

**Why use chains?** Chains are the fundamental building blocks of LangChain applications. They allow you to connect multiple components (prompts, models, output parsers) into a single, reusable unit of work. By composing chains, you can:

- **Standardize prompt patterns** across your application
- **Swap LLM providers** without rewriting application logic
- **Add preprocessing and postprocessing** steps declaratively
- **Monitor and debug** with built-in observability hooks
- **Test components independently** before integrating them

This quickstart demonstrates the simplest yet most common chain pattern: combining a prompt template with a language model using **LCEL** (LangChain Expression Language), the declarative syntax for composing chains.

**Source**: Based on chain pattern from `libs/cli/langchain_cli/package_template/package_template/chain.py:6-19`

## Prerequisites

Before building your first chain, ensure you have:

### 1. Python Installation

LangChain requires **Python 3.10 or higher**. Verify your version:

```bash
python --version
```

If you need to install or upgrade Python, see the [Installation Guide](installation.md#prerequisites).

### 2. LangChain Packages

Install the core package and an LLM provider integration:

```bash
# Using uv (recommended for speed)
uv pip install langchain-core langchain-openai python-dotenv

# Or using pip
pip install langchain-core langchain-openai python-dotenv
```

**What each package provides**:
- `langchain-core`: Core abstractions including Runnables, LCEL, and message types
- `langchain-openai`: OpenAI GPT model integration
- `python-dotenv`: Environment variable management for API keys

For detailed installation options (poetry, virtual environments, etc.), see the complete [Installation Guide](installation.md).

### 3. API Key Configuration

This quickstart uses OpenAI's GPT models. You'll need an OpenAI API key:

**Step 1**: Obtain an API key from [OpenAI Platform](https://platform.openai.com/api-keys)

**Step 2**: Create a `.env` file in your project directory:

```bash
# .env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
```

**Step 3**: Add `.env` to your `.gitignore` to avoid committing secrets:

```bash
echo ".env" >> .gitignore
```

For comprehensive API key setup, security best practices, and alternative providers (Anthropic, Google, etc.), see the [Configuration Guide](configuration.md).

## Your First Chain: Hello World

Let's build a complete working chain that combines a prompt template with a language model.

### Complete Example Code

Create a new file `hello_chain.py`:

```python
"""
Hello World Chain Example

Demonstrates the simplest LCEL pattern: prompt | model
This creates a chain that takes user input, formats it with a prompt template,
and passes it to an LLM for completion.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Step 1: Create a prompt template
# ChatPromptTemplate structures the conversation with the LLM by defining
# system instructions and placeholders for user input
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant who provides clear, concise answers."
    ),
    (
        "human",
        "{user_input}"  # Placeholder for user's question
    ),
])

# Step 2: Initialize the language model
# ChatOpenAI connects to OpenAI's GPT models (uses OPENAI_API_KEY env var)
model = ChatOpenAI(
    model="gpt-4o-mini",  # Cost-effective model for most tasks
    temperature=0.7       # Controls randomness (0=deterministic, 1=creative)
)

# Step 3: Compose the chain using LCEL pipe operator (|)
# The | operator chains components left-to-right:
#   1. prompt receives input dict {"user_input": "..."}
#   2. prompt produces List[BaseMessage] (system + human messages)
#   3. model receives messages and produces AIMessage response
chain = prompt | model

# Step 4: Invoke the chain with user input
if __name__ == "__main__":
    # Example 1: Simple invocation with dict input
    response = chain.invoke({"user_input": "What is LangChain?"})
    
    print("Response:")
    print(response.content)
    print("\n" + "="*60 + "\n")
    
    # Example 2: Another question
    response2 = chain.invoke({
        "user_input": "Explain what a chain is in one sentence."
    })
    
    print("Response 2:")
    print(response2.content)
```

### Running the Example

Save the code above as `hello_chain.py` and run it:

```bash
python hello_chain.py
```

**Expected Output:**

```
Response:
LangChain is a framework for building applications powered by large language
models (LLMs), providing tools and abstractions to compose prompts, models,
memory, and external data sources into cohesive workflows.

============================================================

Response 2:
A chain is a composable unit that connects multiple components like prompts,
models, and output parsers into a single executable workflow.
```

## Understanding the Components

Let's break down each part of the chain:

### 1. ChatPromptTemplate - Structuring LLM Input

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant..."),
    ("human", "{user_input}"),
])
```

**What it does:**
- Defines the conversation structure using message types (`system`, `human`, `ai`)
- `system` message sets the LLM's behavior and role
- `{user_input}` is a template variable that gets replaced with actual input
- Produces `List[BaseMessage]` when invoked

**Type flow:** `Dict[str, str]` → `List[BaseMessage]`

**Source**: `libs/core/langchain_core/prompts/chat.py:273-298`

### 2. ChatOpenAI - The Language Model

```python
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
```

**What it does:**
- Connects to OpenAI's API using the `OPENAI_API_KEY` environment variable
- Accepts chat messages and generates AI responses
- `model`: Specifies which GPT variant to use (`gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`, etc.)
- `temperature`: Controls output randomness (0.0 = deterministic, 1.0 = creative)

**Type flow:** `List[BaseMessage]` → `AIMessage`

**Source**: Integration package `langchain-openai`

### 3. LCEL Pipe Operator (|) - Chain Composition

```python
chain = prompt | model
```

**What it does:**
- The `|` operator chains Runnables together left-to-right
- Output of the left component becomes input to the right component
- Creates a new `Runnable` that executes the entire sequence
- Type-safe: The output type of `prompt` must match the input type of `model`

**Type flow:** `Runnable[Dict, List[BaseMessage]]` | `Runnable[List[BaseMessage], AIMessage]` → `Runnable[Dict, AIMessage]`

**Benefits:**
- **Declarative syntax**: Easy to read and understand data flow
- **Composability**: Chains can be further composed with other chains
- **Observability**: Built-in callback support for debugging

**Source**: `libs/core/langchain_core/runnables/base.py:1519-1561` (\_\_or\_\_ method)

### 4. Invoking the Chain

```python
response = chain.invoke({"user_input": "What is LangChain?"})
print(response.content)
```

**What it does:**
- `invoke()` executes the chain synchronously with a single input
- Input must be a dictionary with keys matching template variables
- Returns an `AIMessage` object containing the LLM's response
- Access the response text via `.content` attribute

**Alternative invocation patterns:**

```python
# Pattern 1: Structured dict input (recommended)
response = chain.invoke({"user_input": "Hello!"})

# Pattern 2: Accessing full message object
response = chain.invoke({"user_input": "Tell me a joke"})
print(f"Content: {response.content}")
print(f"Response metadata: {response.response_metadata}")

# Pattern 3: Batch processing multiple inputs
responses = chain.batch([
    {"user_input": "What is AI?"},
    {"user_input": "What is ML?"},
    {"user_input": "What is DL?"}
])
for resp in responses:
    print(resp.content)
```

## Expected Outputs and Behavior

### What You Should See

When you run the example, you'll receive natural language responses from the GPT model. The exact wording will vary due to the model's generative nature, but responses should:

- **Be relevant** to your question
- **Follow the system message instructions** (clear and concise)
- **Complete within 1-5 seconds** for simple queries

### Response Object Structure

The `AIMessage` object returned by the chain contains:

```python
AIMessage(
    content="LangChain is a framework for...",  # The actual response text
    response_metadata={
        'token_usage': {'prompt_tokens': 25, 'completion_tokens': 42, ...},
        'model_name': 'gpt-4o-mini',
        'finish_reason': 'stop',
    }
)
```

**Key attributes:**
- `.content`: The response text (string)
- `.response_metadata`: Token usage, model info, finish reason
- `.additional_kwargs`: Provider-specific extra data

## Troubleshooting

### Common Issues and Solutions

#### Issue: `openai.AuthenticationError: Incorrect API key`

**Cause:** The `OPENAI_API_KEY` environment variable is not set or is invalid.

**Solutions:**
1. Verify your `.env` file exists in the same directory as your script
2. Ensure `load_dotenv()` is called before initializing `ChatOpenAI`
3. Check that your API key is valid at [OpenAI Platform](https://platform.openai.com/api-keys)
4. Verify the environment variable is loaded:
   ```python
   import os
   print(f"API Key loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
   ```

**Reference:** See [Configuration Guide](configuration.md#api-key-configuration) for detailed setup.

#### Issue: `ModuleNotFoundError: No module named 'langchain_openai'`

**Cause:** The `langchain-openai` package is not installed.

**Solution:**
```bash
uv pip install langchain-openai
# or
pip install langchain-openai
```

**Reference:** See [Installation Guide](installation.md#using-uv-recommended) for package installation.

#### Issue: `ValidationError: 1 validation error for ChatPromptTemplate`

**Cause:** Input dictionary doesn't contain all required template variables.

**Solution:** Ensure your `invoke()` dict includes all placeholders from the template:
```python
# Template has {user_input} placeholder
prompt = ChatPromptTemplate.from_messages([
    ("human", "{user_input}")
])

# ✅ Correct: Dict contains 'user_input' key
chain.invoke({"user_input": "Hello"})

# ❌ Wrong: Missing required key
chain.invoke({"question": "Hello"})  # Raises ValidationError
```

#### Issue: `openai.RateLimitError: Rate limit exceeded`

**Cause:** You've exceeded your OpenAI API usage limits.

**Solutions:**
1. Check your usage at [OpenAI Usage Dashboard](https://platform.openai.com/usage)
2. Add retry logic with exponential backoff
3. Switch to a different model with higher limits
4. Upgrade your OpenAI plan if needed

**Example with retry:**
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    max_retries=3,  # Automatically retry on rate limits
)
```

#### Issue: Chain execution is slow

**Cause:** Network latency or model processing time.

**Solutions:**
1. Use a faster model like `gpt-4o-mini` instead of `gpt-4`
2. Reduce `max_tokens` if you don't need long responses
3. Consider streaming for perceived performance:
   ```python
   for chunk in chain.stream({"user_input": "Tell me a story"}):
       print(chunk.content, end="", flush=True)
   ```

**Reference:** See [Async Usage Guide](../guides/async-usage.md) for async patterns.

#### Issue: `ImportError: cannot import name 'ChatPromptTemplate'`

**Cause:** Using incorrect import path.

**Solution:** Ensure you're importing from `langchain_core.prompts`:
```python
# ✅ Correct import
from langchain_core.prompts import ChatPromptTemplate

# ❌ Incorrect imports (old patterns)
from langchain.prompts import ChatPromptTemplate  # May work but not recommended
```

## Next Steps

Congratulations! You've built your first LangChain chain. Here's what to explore next:

### 1. Understand Core Concepts

Read the [Core Concepts Guide](concepts.md) to learn about:
- **Runnables**: The standard interface powering all LangChain components
- **LCEL**: Advanced composition patterns (branching, fallbacks, parallelization)
- **Messages**: Different message types and when to use them
- **Callbacks**: Debugging and monitoring your chains

### 2. Explore More Complex Chains

- **Add Output Parsing**: Convert LLM responses to structured data (JSON, Pydantic models)
  ```python
  from langchain_core.output_parsers import StrOutputParser
  
  chain = prompt | model | StrOutputParser()  # Returns string instead of AIMessage
  ```

- **Sequential Chains**: Chain multiple prompts together where one output feeds the next
  
- **Retrieval Chains**: Combine chains with vector stores for RAG (Retrieval-Augmented Generation) applications

### 3. Try Different LLM Providers

Swap out OpenAI for other providers while keeping the same chain structure:

```python
# Anthropic Claude
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
chain = prompt | model  # Same chain pattern!

# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
chain = prompt | model  # Same chain pattern!
```

**Reference:** See [Configuration Guide](configuration.md#api-key-configuration) for provider-specific setup.

### 4. Add Memory for Conversations

Enable your chain to remember previous interactions:

```python
from langchain.memory import ConversationBufferMemory
# See Memory Integration Guide for full examples
```

### 5. Explore Example Code

Browse complete, executable examples in the repository:
- `examples/basic_chains/` - Simple chain patterns
- `examples/advanced_chains/` - RAG, agents, streaming, async
- `examples/type_patterns/` - Type-safe chain implementations

### 6. Learn Debugging Techniques

- Enable LangSmith tracing for visual debugging
- Use callback handlers to log chain execution
- See [Debugging Guide](../debugging/common-issues.md) for common failure modes

## Additional Resources

- **[Installation Guide](installation.md)**: Detailed package installation and dependency management
- **[Configuration Guide](configuration.md)**: API keys, environment variables, and LangSmith setup
- **[Core Concepts](concepts.md)**: Deep dive into Runnables, LCEL, Messages, and Callbacks
- **[API Reference](../api-reference/prompts/templates.md)**: Complete API documentation for prompts, models, and chains
- **[LangChain Documentation](https://docs.langchain.com/)**: Official comprehensive documentation

## Summary

You've learned how to:
- ✅ Install LangChain packages and configure API keys
- ✅ Create a prompt template with system and user messages
- ✅ Initialize a language model (ChatOpenAI)
- ✅ Compose a chain using the LCEL pipe operator (`|`)
- ✅ Invoke the chain and access responses
- ✅ Troubleshoot common issues

This simple pattern—`prompt | model`—is the foundation for more complex LangChain applications. As you progress, you'll add output parsers, retrievers, memory, agents, and custom components, all using the same composable LCEL syntax.

**Happy building with LangChain!** 🦜⛓️
