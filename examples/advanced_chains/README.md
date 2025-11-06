# Advanced LangChain Chain Examples

This directory contains production-ready examples demonstrating advanced LangChain patterns and techniques. These examples build upon the foundations covered in `basic_chains/` and showcase real-world patterns including retrieval-augmented generation (RAG), conversational memory, agent systems, streaming responses, asynchronous execution, and error recovery strategies.

## 🎯 Overview

The advanced examples in this directory are designed for developers who:

- Have completed the `basic_chains/` examples and understand LCEL composition fundamentals
- Are building production applications requiring robust error handling and state management
- Need to implement retrieval-augmented generation with vector stores
- Want to integrate custom tools with agent-based reasoning systems
- Require streaming responses for real-time user experiences
- Need to optimize performance with asynchronous execution patterns
- Are implementing fault-tolerant systems with fallback strategies

**Target Audience**: Developers with basic LangChain experience who are ready to implement production-grade patterns for complex use cases.

## ✅ Prerequisites

Before running these examples, ensure you have:

### Required
- **Python >=3.10** - These examples use modern Python type hints and async features
- **Completed `basic_chains/` examples** - Understanding of prompt templates, LCEL pipe operators, and basic chain composition
- **OpenAI API key** - Required for all examples (obtain from https://platform.openai.com/api-keys)
- **LCEL composition fundamentals** - Knowledge of how `prompt | llm | parser` chains work

### Recommended
- **Anthropic API key** - Optional, for fallback examples demonstrating multi-provider strategies
- **Understanding of Python asyncio** - Helpful for async execution examples
- **Vector store concepts** - Basic knowledge of embeddings and similarity search for RAG examples
- **Familiarity with agent patterns** - Understanding of tool-calling and reasoning loops

### Optional Dependencies
- **Chroma vector database** - Required only for `retrieval_qa_chain.py` example
- **pytest and pytest-asyncio** - Required only if running validation tests

## 📦 Installation

### Step 1: Install Dependencies

Install all required packages using pip:

```bash
cd examples/advanced_chains
pip install -r requirements.txt
```

For optimal dependency management, we recommend using `uv`:

```bash
uv pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required for all examples
OPENAI_API_KEY=your-openai-api-key-here

# Optional: for fallback examples
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: for debugging with LangSmith
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your-langsmith-api-key-here
```

**Security Note**: Never commit your `.env` file to version control. It is already included in `.gitignore`.

### Step 3: Install Optional Dependencies

If you plan to run the RAG example with Chroma vector store:

```bash
pip install chromadb>=0.4.22
```

For Python 3.13+, numpy 2.x is automatically installed. For Python <3.13, numpy 1.26.x will be used.

## 📁 Examples Overview

This directory contains the following advanced pattern examples:

| Example File | Pattern | Description | Key Concepts |
|-------------|---------|-------------|--------------|
| `retrieval_qa_chain.py` | RAG | Question answering with vector store retrieval | Document embeddings, similarity search, context injection |
| `memory_enabled_chain.py` | Conversational State | Multi-turn conversations with message history | ConversationBufferMemory, context management, state persistence |
| `agent_with_tools.py` | Agent Systems | Custom tool implementation with reasoning loop | Tool interface, @tool decorator, agent decision-making |
| `streaming_responses.py` | Streaming | Token-by-token response generation | .stream() method, real-time output, callback handlers |
| `async_chain_execution.py` | Async/Await | Non-blocking concurrent chain execution | ainvoke(), abatch(), astream(), asyncio patterns |
| `fallback_chains.py` | Error Recovery | Graceful degradation with fallback strategies | RunnableWithFallbacks, exception handling, retry logic |

## 🚀 Usage Instructions

### Running Individual Examples

Each example is a self-contained Python script that can be run directly:

```bash
# Run RAG example
python retrieval_qa_chain.py

# Run memory-enabled conversation
python memory_enabled_chain.py

# Run agent with custom tools
python agent_with_tools.py

# Run streaming example
python streaming_responses.py

# Run async execution example
python async_chain_execution.py

# Run fallback chains example
python fallback_chains.py
```

### Expected Output

Each example produces informative output showing:

- **Input**: The question or prompt being processed
- **Intermediate Steps**: Chain execution stages (retrievals, tool calls, reasoning steps)
- **Final Output**: The generated response or result
- **Timing Information**: Execution time for performance comparison (where applicable)

### Environment Variable Requirements

| Example | Required Variables | Optional Variables |
|---------|-------------------|-------------------|
| `retrieval_qa_chain.py` | `OPENAI_API_KEY` | `LANGCHAIN_TRACING_V2` |
| `memory_enabled_chain.py` | `OPENAI_API_KEY` | `LANGCHAIN_TRACING_V2` |
| `agent_with_tools.py` | `OPENAI_API_KEY` | `LANGCHAIN_TRACING_V2` |
| `streaming_responses.py` | `OPENAI_API_KEY` | `LANGCHAIN_TRACING_V2` |
| `async_chain_execution.py` | `OPENAI_API_KEY` | `LANGCHAIN_TRACING_V2` |
| `fallback_chains.py` | `OPENAI_API_KEY` | `ANTHROPIC_API_KEY`, `LANGCHAIN_TRACING_V2` |

## 📚 Key Concepts Explained

### Retrieval-Augmented Generation (RAG)

RAG combines large language models with external knowledge retrieval to provide accurate, context-aware responses. The pattern involves:

1. **Document Ingestion**: Loading and splitting documents into chunks
2. **Embedding Generation**: Converting text chunks into vector embeddings
3. **Vector Storage**: Storing embeddings in a searchable database (Chroma, FAISS, etc.)
4. **Similarity Search**: Finding relevant documents for a given query
5. **Context Injection**: Passing retrieved documents to the LLM as context
6. **Answer Generation**: LLM generates response based on retrieved context

**Example**: `retrieval_qa_chain.py` demonstrates a complete RAG workflow with in-memory vector storage.

### Memory Management in Conversations

Conversational memory enables chains to maintain context across multiple interactions:

- **ConversationBufferMemory**: Stores all messages in the conversation
- **ConversationBufferWindowMemory**: Stores only the last N messages
- **ConversationSummaryMemory**: Stores a summary of the conversation
- **Memory Lifecycle**: load_memory_variables() → chain execution → save_context()

**Example**: `memory_enabled_chain.py` shows how to integrate ConversationBufferMemory into LCEL chains.

### Agent Decision Loops

Agents use LLMs for reasoning about which actions to take:

1. **Observation**: Receive user input or previous tool output
2. **Thought**: LLM reasons about what action to take
3. **Action**: Agent selects and executes a tool
4. **Observation**: Tool returns result
5. **Repeat** or **Final Answer**: Continue loop or return final response

**Tool Interface Requirements**:
- `name`: Unique identifier for the tool
- `description`: Clear explanation of what the tool does (LLM uses this to decide when to call it)
- `args_schema`: Pydantic model defining expected parameters with validation

**Example**: `agent_with_tools.py` demonstrates custom tool implementation with proper type validation.

### Streaming vs Batch Processing

**Batch Processing** (`.invoke()`):
- Waits for complete response before returning
- Simpler programming model
- Better for non-interactive use cases

**Streaming** (`.stream()`):
- Yields tokens as they're generated
- Provides real-time feedback to users
- Better user experience for chat interfaces
- Slightly more complex to implement

**Example**: `streaming_responses.py` shows both patterns with timing comparisons.

### Async Benefits and Considerations

**When to Use Async**:
- Making multiple LLM calls concurrently
- Building web applications with async frameworks (FastAPI, aiohttp)
- Improving throughput for batch processing
- Non-blocking I/O operations

**Key Methods**:
- `ainvoke()`: Async single invocation
- `abatch()`: Async batch processing with concurrency
- `astream()`: Async streaming responses

**Event Loop Considerations**:
- Use `asyncio.run()` for standalone scripts
- Use existing event loop in async frameworks
- Avoid mixing blocking sync calls in async contexts

**Example**: `async_chain_execution.py` demonstrates proper async patterns with performance comparisons.

### Fallback Chains for Reliability

Fallback strategies enable graceful degradation when primary systems fail:

**Fallback Patterns**:
- **Model Fallback**: GPT-4 → GPT-3.5 (cost/speed tradeoff)
- **Provider Fallback**: OpenAI → Anthropic (multi-provider resilience)
- **Cached Fallback**: LLM → cached response (when API unavailable)
- **Simplified Fallback**: Complex chain → simpler logic (guaranteed response)

**Configuration**:
```python
chain_with_fallbacks = primary_chain.with_fallbacks(
    [fallback1, fallback2],
    exception_key="exception"  # Pass exception info to fallbacks
)
```

**Example**: `fallback_chains.py` demonstrates multi-level fallback strategies with different failure scenarios.

## 🐛 Troubleshooting

### Missing API Key Errors

**Symptom**:
```
Error: openai.AuthenticationError: No API key provided
```

**Solution**:
1. Verify `.env` file exists in `examples/advanced_chains/` directory
2. Confirm `OPENAI_API_KEY` is set in `.env`
3. Ensure no extra spaces around the API key value
4. Verify the API key is valid at https://platform.openai.com/api-keys

### Import Errors for Optional Dependencies

**Symptom**:
```
ImportError: No module named 'chromadb'
```

**Solution**:
Install the optional dependency:
```bash
pip install chromadb>=0.4.22
```

For the fallback example requiring Anthropic:
```bash
pip install langchain-anthropic>=0.3.22
```

### Rate Limiting Errors

**Symptom**:
```
openai.error.RateLimitError: Rate limit exceeded
```

**Solution**:
1. Check your OpenAI usage limits at https://platform.openai.com/usage
2. Implement retry logic with exponential backoff (see `fallback_chains.py`)
3. Consider upgrading your OpenAI plan for higher rate limits
4. Use `asyncio.Semaphore` to limit concurrent requests in async examples

### Memory Context Window Overflow

**Symptom**:
```
openai.error.InvalidRequestError: maximum context length exceeded
```

**Solution**:
1. Use `ConversationBufferWindowMemory` instead of `ConversationBufferMemory` to limit message history
2. Implement conversation summarization with `ConversationSummaryMemory`
3. Manually trim conversation history with `memory.clear()`
4. Use a model with larger context window (e.g., GPT-4-turbo with 128k tokens)

### Agent Tool Execution Failures

**Symptom**:
Agent gets stuck in a loop or fails to execute tools correctly

**Solution**:
1. Ensure tool descriptions are clear and specific
2. Validate tool `args_schema` matches actual implementation
3. Add error handling to tool `_run()` methods
4. Review agent prompts to ensure proper instruction formatting
5. Enable LangSmith tracing (`LANGCHAIN_TRACING_V2=true`) to inspect agent reasoning

### Async Event Loop Errors

**Symptom**:
```
RuntimeError: asyncio.run() cannot be called from a running event loop
```

**Solution**:
- In standalone scripts: Use `asyncio.run(main())`
- In Jupyter notebooks: Use `await main()` (notebook already has event loop)
- In async frameworks: Use existing event loop, don't call `asyncio.run()`

### Common Failure Modes

For comprehensive debugging guidance covering the top 10 LangChain failure modes, see:
- `../debugging/` - Debugging utilities and callback handlers
- `../../docs/debugging/common-issues.md` - Detailed troubleshooting guide (if available)
- LangSmith tracing: Set `LANGCHAIN_TRACING_V2=true` for detailed execution traces

## 🔗 References and Related Documentation

### Foundation Concepts
- **Basic Chains**: `../basic_chains/README.md` - Start here if you're new to LangChain
- **Type Patterns**: `../type_patterns/README.md` - Type annotation patterns for custom chains

### Debugging and Troubleshooting
- **Debugging Examples**: `../debugging/README.md` - Logging wrappers and inspection tools
- **Debugging Guide**: `../../docs/debugging/common-issues.md` - Common failure modes and solutions (if available)

### API Reference Documentation
- **LangChain API Reference**: https://reference.langchain.com/python/langchain_classic/
- **LangChain Core API**: https://reference.langchain.com/python/langchain_core/
- **Official Documentation**: https://docs.langchain.com/

### LangChain Modules
- **Chains**: `libs/langchain/langchain_classic/chains/` - Chain implementations
- **Agents**: `libs/langchain/langchain_classic/agents/` - Agent and tool classes
- **Memory**: `libs/langchain/langchain_classic/memory/` - Memory implementations
- **Runnables**: `libs/core/langchain_core/runnables/` - Core LCEL abstractions

### External Resources
- **OpenAI API Documentation**: https://platform.openai.com/docs/
- **Anthropic API Documentation**: https://docs.anthropic.com/
- **Chroma Vector Database**: https://docs.trychroma.com/
- **LangSmith Tracing**: https://smith.langchain.com/

## 📝 Notes

- All examples use modern LCEL (LangChain Expression Language) composition patterns
- Examples are designed to be self-contained and copy-paste ready
- Each example includes comprehensive error handling and graceful degradation
- Production deployment should include additional monitoring, logging, and rate limiting
- For production use, consider implementing retry logic, caching, and fallback strategies from these examples

## 🤝 Contributing

Found an issue or want to improve these examples? See the main repository contribution guidelines:
- `../../CONTRIBUTING.md` - General contribution guide (if available)
- `../../AGENTS.md` - Developer guidelines for API stability and testing

## 📄 License

These examples are part of the LangChain project and follow the same MIT License as the main repository.
