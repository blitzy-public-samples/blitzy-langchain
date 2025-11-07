# LangChain Classic Memory Module

## Overview

The **Memory** subsystem maintains Chain state by incorporating context from past conversation runs. Memory enables chains to maintain conversation history across multiple interactions, making it essential for building chatbots, Q&A systems, conversational agents, and context-aware applications.

> **⚠️ DEPRECATION NOTICE**
>
> All memory classes in this module are deprecated as of version 0.3.1 and will be removed in version 1.0.0.
>
> **Migration Guide**: https://python.langchain.com/docs/versions/migrating_memory/
>
> For new implementations, refer to the modern approaches documented in the migration guide, particularly LangGraph's built-in memory management capabilities.

## Purpose

Memory solves the stateless nature of language models by:

- **Storing conversation history**: Preserves the context of previous interactions
- **Managing context windows**: Provides strategies to fit history within model token limits
- **Enabling continuity**: Allows models to reference earlier parts of conversations
- **Supporting personalization**: Maintains user-specific context across sessions

### Use Cases

- **Chatbots**: Multi-turn conversations that reference earlier exchanges
- **Q&A Systems**: Answer follow-up questions using previous context
- **Conversational Agents**: Tool-using agents that maintain task context
- **Context-Aware Applications**: Any application requiring stateful interactions

## Memory Types Overview

### ConversationBufferMemory

**Description**: Stores ALL messages without truncation or summarization.

**Source**: `libs/langchain/langchain_classic/memory/buffer.py:21-92`

**Best For**:
- Short conversations where full history is manageable
- Debugging and development scenarios
- When complete conversation context is critical

**Limitations**:
- **Grows unbounded**: Memory size increases with every interaction
- **May exceed context windows**: Long conversations can surpass model token limits
- **No automatic pruning**: Requires manual clearing or migration to windowed memory

**Key Parameters**:
- `memory_key` (str, default: "history"): Key name for memory in chain inputs
- `return_messages` (bool, default: False): Return List[BaseMessage] if True, formatted string if False
- `human_prefix` (str, default: "Human"): Prefix for user messages in string format
- `ai_prefix` (str, default: "AI"): Prefix for AI messages in string format

---

### ConversationStringBufferMemory

**Description**: String-optimized variant of buffer memory for non-chat models. Stores formatted string instead of message objects.

**Source**: `libs/langchain/langchain_classic/memory/buffer.py:102-180`

**Best For**:
- LLMs (non-chat models) requiring string-formatted history
- Legacy chains expecting string inputs
- Simpler use cases without message metadata

**Limitations**:
- **Cannot use return_messages=True**: Raises ValueError if configured
- **Loses message structure**: No access to message types or metadata
- **Same unbounded growth**: No automatic size management

**Key Parameters**:
- `buffer` (str, default: ""): Internal string buffer
- `input_key` (str | None): Which input dict key contains user message
- `output_key` (str | None): Which output dict key contains AI response
- `memory_key` (str, default: "history"): Key name for memory in chain inputs

---

### ConversationBufferWindowMemory

**Description**: Stores last `k` conversation turns (k×2 messages, one human + one AI per turn).

**Source**: `libs/langchain/langchain_classic/memory/buffer_window.py:18-63`

**Best For**:
- Limiting context to recent interactions
- Predictable memory size management
- Applications where older context becomes irrelevant

**Limitations**:
- **Abrupt context loss**: Oldest messages dropped immediately when k exceeded
- **Fixed window**: Cannot adapt to varying message lengths
- **May lose critical context**: Important early information can be discarded

**Key Parameters**:
- `k` (int, default: 5): Number of conversation turns to retain
  - `k=5` keeps last 10 messages (5 human + 5 AI)
  - `k=0` keeps no messages
- `memory_key` (str, default: "history"): Key name for memory in chain inputs
- `return_messages` (bool, default: False): Return format

**Window Calculation**: `messages[-k * 2:]` selects the last `k` turns from the full history.

---

### ConversationSummaryMemory

**Description**: Uses LLM to create a rolling summary of the conversation, replacing full history with condensed version.

**Source**: `libs/langchain/langchain_classic/memory/summary.py:91-120`

**Best For**:
- Long conversations where full history exceeds context windows
- When gist of conversation is sufficient
- Reducing token usage in prompts

**Limitations**:
- **LLM calls add latency**: Each save_context triggers summarization
- **Increased cost**: Summary generation consumes tokens
- **Potential information loss**: Nuanced details may be omitted
- **Requires LLM parameter**: Must provide BaseLanguageModel instance

**Key Parameters**:
- `llm` (BaseLanguageModel, required): Model used for generating summaries
- `prompt` (BasePromptTemplate, default: SUMMARY_PROMPT): Template for summarization
- `memory_key` (str, default: "history"): Key name for memory in chain inputs
- `summary_message_cls` (type[BaseMessage], default: SystemMessage): Message class for summary

**Trade-off**: Summary reduces prompt tokens but adds generation cost and latency.

---

### ConversationSummaryBufferMemory

**Description**: Hybrid approach keeping recent messages verbatim plus summary of older messages.

**Source**: `libs/langchain/langchain_classic/memory/summary_buffer.py:20-120`

**Best For**:
- Balancing detail and context length
- When recent exchanges need full fidelity
- Long conversations requiring historical context

**Limitations**:
- **Increased complexity**: More sophisticated than simple buffer or summary
- **Still requires LLM**: Inherits summarization costs
- **Token limit management**: Must monitor max_token_limit

**Key Parameters**:
- `max_token_limit` (int, default: 2000): Maximum tokens before summarization triggers
- `moving_summary_buffer` (str, default: ""): Internal summary storage
- `llm` (BaseLanguageModel, required): Model for summarization
- `memory_key` (str, default: "history"): Key name for memory in chain inputs

**Behavior**: When total tokens exceed `max_token_limit`, oldest messages are summarized and removed, summary prepended to remaining messages.

---

### ConversationTokenBufferMemory

**Description**: Prunes messages based on token count using `llm.get_num_tokens_from_messages()`.

**Source**: `libs/langchain/langchain_classic/memory/token_buffer.py:19-75`

**Best For**:
- Strict token limit enforcement
- Precise control over context window usage
- When token counting accuracy is critical

**Limitations**:
- **Token counting overhead**: Calculates tokens on every save_context call
- **Requires LLM with token counting**: Must provide model with get_num_tokens_from_messages method
- **FIFO pruning**: Oldest messages dropped first, no semantic relevance

**Key Parameters**:
- `llm` (BaseLanguageModel, required): Model for token counting
- `max_token_limit` (int, default: 2000): Maximum tokens to retain
- `memory_key` (str, default: "history"): Key name for memory in chain inputs

**Pruning Logic**: After each `save_context`, if `curr_buffer_length > max_token_limit`, removes oldest messages until under limit.

---

### VectorStoreRetrieverMemory

**Description**: Stores messages in vector database, retrieves semantically relevant context based on current input.

**Source**: `libs/langchain/langchain_classic/memory/vectorstore.py:23-120`

**Best For**:
- Very long conversations spanning many sessions
- Selective context retrieval based on relevance
- When semantic similarity matters more than recency

**Limitations**:
- **Requires vector store**: Must configure and manage separate VectorStoreRetriever
- **Retrieval accuracy varies**: Quality depends on embeddings and query
- **Complexity**: More infrastructure than simple buffer
- **No guaranteed completeness**: May miss relevant but dissimilar context

**Key Parameters**:
- `retriever` (VectorStoreRetriever, required): Configured retriever instance
- `memory_key` (str, default: "history"): Key name for memory in chain inputs
- `input_key` (str | None): Which input dict key to use for retrieval query
- `return_docs` (bool, default: False): Return Document objects if True, concatenated strings if False

**Retrieval Mechanism**: On `load_memory_variables`, queries vector store with current input, returns top-k most similar past interactions.

---

### ConversationVectorStoreTokenBufferMemory

**Description**: Combines token buffer with vector store persistence for overflow handling.

**Source**: `libs/langchain/langchain_classic/memory/vectorstore_token_buffer_memory.py`

**Best For**:
- Long conversations requiring both recency and semantic retrieval
- When recent context must be complete but older context can be retrieved
- Advanced memory management scenarios

**Key Features**:
- Keeps recent messages under token limit
- Persists older messages to vector store
- Retrieves relevant historical context semantically

## Key Interfaces

All memory classes implement the `BaseMemory` interface with these core methods:

### load_memory_variables()

**Signature**: `load_memory_variables(inputs: Dict[str, Any]) -> Dict[str, Any]`

**Purpose**: Retrieves conversation context as dictionary for chain execution.

**Source**: `libs/langchain/langchain_classic/memory/chat_memory.py:74-96`

**Behavior**:
- Called automatically by Chain before execution (`Chain.prep_inputs()`)
- Returns dictionary with `{memory_key: context}` where context format depends on `return_messages` flag
- For BaseChatMemory: Context is `List[BaseMessage]` if `return_messages=True`, formatted string if `False`

**Parameters**:
- `inputs` (Dict[str, Any]): Current chain inputs (may be used for retrieval query in VectorStoreRetrieverMemory)

**Returns**:
- Dict[str, Any]: Dictionary with memory_key mapping to conversation history

**Example**:
```python
memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "Hello"}, {"output": "Hi there!"})

# Retrieve memory
context = memory.load_memory_variables({})
# Returns: {"history": [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]}
```

**Async Variant**: `aload_memory_variables(inputs: Dict[str, Any]) -> Dict[str, Any]`

---

### save_context()

**Signature**: `save_context(inputs: Dict[str, Any], outputs: Dict[str, str]) -> None`

**Purpose**: Persists interaction after chain execution.

**Source**: `libs/langchain/langchain_classic/memory/chat_memory.py:74-96`

**Behavior**:
- Called automatically by Chain after execution (`Chain._call()` completion)
- Extracts user input and AI output from dictionaries using `input_key`/`output_key`
- Stores as HumanMessage and AIMessage in chat_memory
- Some implementations (TokenBufferMemory, SummaryMemory) trigger additional processing

**Parameters**:
- `inputs` (Dict[str, Any]): Chain inputs containing user message
  - If `input_key` is None, uses `get_prompt_input_key()` to infer key
  - If `input_key` specified, uses that key
- `outputs` (Dict[str, str]): Chain outputs containing AI response
  - If `output_key` is None, infers from outputs (expects single key or "output" key)
  - If `output_key` specified, uses that key

**Returns**: None (mutates internal state)

**Example**:
```python
memory = ConversationBufferMemory(
    input_key="user_input",
    output_key="bot_response"
)

# Save interaction
memory.save_context(
    {"user_input": "What's the weather?", "metadata": "extra"},
    {"bot_response": "Sunny!", "confidence": 0.9}
)
# Stores: HumanMessage("What's the weather?"), AIMessage("Sunny!")
```

**Key Extraction Logic** (Source: `libs/langchain/langchain_classic/memory/chat_memory.py:43-72`):
- **input_key determination**: Uses `input_key` parameter if set, otherwise calls `get_prompt_input_key()`
- **output_key determination**: Uses `output_key` if set, otherwise:
  - If outputs has 1 key: uses that key
  - If outputs has multiple keys and "output" exists: uses "output" (with warning)
  - Otherwise: raises ValueError

**Async Variant**: `asave_context(inputs: Dict[str, Any], outputs: Dict[str, str]) -> None`

---

### clear()

**Signature**: `clear() -> None`

**Purpose**: Resets memory to empty state.

**Source**: Multiple implementations (e.g., `libs/langchain/langchain_classic/memory/buffer.py:173-175`)

**Behavior**:
- Removes all stored conversation history
- For BaseChatMemory: Calls `chat_memory.clear()`
- For ConversationStringBufferMemory: Resets `buffer = ""`
- Useful for starting new conversation sessions

**Parameters**: None

**Returns**: None

**Example**:
```python
memory = ConversationBufferMemory()
memory.save_context({"input": "Hello"}, {"output": "Hi!"})
print(len(memory.chat_memory.messages))  # 2

memory.clear()
print(len(memory.chat_memory.messages))  # 0
```

**Async Variant**: `aclear() -> None`

---

### Additional Interface Methods

**memory_variables** (property):
- **Type**: `List[str]`
- **Purpose**: Returns list of keys this memory will add to chain inputs
- **Typical Value**: `[self.memory_key]` (usually `["history"]`)

**buffer** (property, BaseChatMemory implementations):
- **Type**: `List[BaseMessage] | str`
- **Purpose**: Direct access to memory content
- **Return**: Depends on `return_messages` flag

## Integration Patterns

### With Chains

Memory integrates with Chain classes through the `memory` parameter:

```python
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Create memory instance
memory = ConversationBufferMemory(return_messages=True)

# Pass to chain constructor
llm = ChatOpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

# Memory is automatically loaded and saved during chain execution
response = conversation.invoke({"input": "Hello"})
# Chain calls: memory.load_memory_variables() → llm.invoke() → memory.save_context()
```

**Chain Execution Flow**:
1. **Before LLM call**: Chain calls `memory.load_memory_variables(inputs)` in `prep_inputs()`
2. **Memory added to inputs**: Chain adds `{memory_key: context}` to inputs dictionary
3. **LLM receives context**: Prompt template can reference `{history}` variable
4. **After LLM call**: Chain calls `memory.save_context(inputs, outputs)` to persist interaction

---

### memory_key Usage

The `memory_key` parameter determines the dict key used in chain inputs:

```python
# Default memory_key is "history"
memory = ConversationBufferMemory(memory_key="history")
context = memory.load_memory_variables({})
# Returns: {"history": [...]}

# Custom memory_key
memory = ConversationBufferMemory(memory_key="chat_context")
context = memory.load_memory_variables({})
# Returns: {"chat_context": [...]}
```

**Prompt Template Integration**:
```python
from langchain_core.prompts import PromptTemplate

template = """The following is a conversation history:
{history}

Current question: {input}
Answer:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],  # Must match memory_key
    template=template
)

memory = ConversationBufferMemory(memory_key="history")
# Chain will inject memory at "history" key
```

---

### input_key and output_key Configuration

Specify which chain input/output keys contain messages:

```python
from langchain_classic.memory import ConversationBufferMemory

# Chain with multiple inputs/outputs
memory = ConversationBufferMemory(
    input_key="user_message",    # Extract user input from inputs["user_message"]
    output_key="assistant_reply" # Extract AI output from outputs["assistant_reply"]
)

# save_context uses specified keys
memory.save_context(
    {
        "user_message": "Hello",
        "metadata": {"timestamp": "2024-01-01"},
        "session_id": "abc123"
    },
    {
        "assistant_reply": "Hi there!",
        "confidence": 0.95,
        "model": "gpt-4"
    }
)
# Stores only: HumanMessage("Hello"), AIMessage("Hi there!")
```

**Default Behavior** (when keys are None):
- **input_key=None**: Uses `get_prompt_input_key()` to find key (excludes memory_variables)
- **output_key=None**: 
  - Single output key: uses that key
  - Multiple keys with "output": uses "output" (warns)
  - Multiple keys without "output": raises ValueError

---

### return_messages Flag

Controls the format of memory context:

**return_messages=False** (default):
```python
memory = ConversationBufferMemory(return_messages=False)
memory.save_context({"input": "Hi"}, {"output": "Hello!"})

context = memory.load_memory_variables({})
# Returns: {"history": "\nHuman: Hi\nAI: Hello!"}
# Type: str - formatted string suitable for LLMs
```

**return_messages=True**:
```python
memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "Hi"}, {"output": "Hello!"})

context = memory.load_memory_variables({})
# Returns: {"history": [HumanMessage(content="Hi"), AIMessage(content="Hello!")]}
# Type: List[BaseMessage] - suitable for chat models
```

**When to Use Each**:
- **return_messages=False**: For traditional LLMs expecting string context
- **return_messages=True**: For chat models (ChatOpenAI, ChatAnthropic) expecting message lists

---

### Integration with BaseChatMemory

All chat-based memory classes use `BaseChatMemory` which provides:

**Source**: `libs/langchain/langchain_classic/memory/chat_memory.py:25-100`

**chat_memory Field**:
- **Type**: `BaseChatMessageHistory`
- **Default**: `InMemoryChatMessageHistory()` (ephemeral, in-memory storage)
- **Purpose**: Backend storage for messages

**Persistent Backends**:

To use persistent storage, provide custom `chat_memory`:

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory

# Persistent Redis storage
memory = ConversationBufferMemory(
    chat_memory=RedisChatMessageHistory(
        url="redis://localhost:6379",
        session_id="user_123"
    ),
    return_messages=True
)

# Messages persist across application restarts
```

**Available Backends** (via langchain_community):
- **RedisChatMessageHistory**: Redis storage
- **PostgresChatMessageHistory**: PostgreSQL storage
- **FileChatMessageHistory**: File-based storage
- **MongoDBChatMessageHistory**: MongoDB storage
- **DynamoDBChatMessageHistory**: AWS DynamoDB storage
- And many more (see `langchain_community.chat_message_histories`)

## State Management

### Non-Persistent by Default

**Default Behavior**: Memory classes use **in-memory storage** which is lost on process restart.

**Source**: `libs/langchain/langchain_classic/memory/chat_memory.py:36-38`

```python
chat_memory: BaseChatMessageHistory = Field(
    default_factory=InMemoryChatMessageHistory,
)
```

**Implications**:
- Memory exists only during application runtime
- Process restart = conversation history lost
- Not suitable for production multi-session applications
- Acceptable for development, testing, single-session scripts

---

### Persistence Options

To persist memory across sessions, use `chat_message_histories` backends:

#### File-Based Persistence

```python
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("conversation_history.json"),
    return_messages=True
)

# Messages written to conversation_history.json
memory.save_context({"input": "Hello"}, {"output": "Hi!"})

# Survives application restart
# Load from same file to resume conversation
```

#### Database Persistence

```python
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    chat_memory=PostgresChatMessageHistory(
        connection_string="postgresql://user:pass@localhost/dbname",
        session_id="user_session_123"
    ),
    return_messages=True
)

# Messages stored in PostgreSQL table
# Can query, analyze, backup conversation data
```

---

### Thread Safety

**⚠️ WARNING**: Default memory implementations are **NOT thread-safe**.

**Implications**:
- Concurrent access from multiple threads can corrupt memory state
- Race conditions in `save_context()` calls
- Message ordering may be incorrect under concurrency

**Recommendations**:
1. **Separate memory instances per conversation**: Don't share memory objects across threads
2. **Use session identifiers**: Create per-user/per-session memory instances
3. **External synchronization**: Use locks if sharing is unavoidable
4. **Async-safe backends**: Some chat_message_histories backends (Redis, databases) handle concurrency better

**Example - Per-Session Memory**:
```python
from langchain_classic.memory import ConversationBufferMemory

# BAD: Shared memory across threads (unsafe)
global_memory = ConversationBufferMemory()

def handle_request(user_id):
    conversation = ConversationChain(llm=llm, memory=global_memory)  # UNSAFE

# GOOD: Per-session memory
def handle_request(user_id):
    session_memory = ConversationBufferMemory(
        chat_memory=get_chat_history_for_user(user_id)  # Per-user backend
    )
    conversation = ConversationChain(llm=llm, memory=session_memory)  # SAFE
```

---

### Serialization

**Memory State is NOT automatically serialized**. To persist memory across process boundaries:

**Manual Serialization Approach**:

```python
import json
from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "Hello"}, {"output": "Hi!"})

# Manual serialization
messages = memory.chat_memory.messages
serialized = [
    {"type": m.__class__.__name__, "content": m.content}
    for m in messages
]
json.dump(serialized, open("memory_state.json", "w"))

# Manual deserialization
loaded_data = json.load(open("memory_state.json", "r"))
# Reconstruct memory from loaded_data
```

**Better Approach**: Use persistent chat_message_histories backends which handle serialization automatically.

## Complete Executable Examples

### Example 1: Basic ConversationBufferMemory

**Source**: `libs/langchain/langchain_classic/memory/buffer.py:21-92`

```python
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Create chain with memory
llm = ChatOpenAI(temperature=0)
conversation = ConversationChain(llm=llm, memory=memory)

# First interaction
response1 = conversation.invoke({"input": "Hi, my name is Alice"})
print("AI:", response1["response"])
# Output: AI: Hello Alice! How can I help you today?

# Second interaction - memory provides context
response2 = conversation.invoke({"input": "What's my name?"})
print("AI:", response2["response"])
# Output: AI: Your name is Alice.

# Inspect memory
print("\nMemory content:")
history = memory.load_memory_variables({})
print(history["history"])
# Output: [HumanMessage(content="Hi, my name is Alice"), 
#          AIMessage(content="Hello Alice!..."), 
#          HumanMessage(content="What's my name?"),
#          AIMessage(content="Your name is Alice.")]

# Clear memory
memory.clear()
print("\nAfter clear:", len(memory.chat_memory.messages))
# Output: After clear: 0
```

---

### Example 2: ConversationBufferWindowMemory with k=2

**Source**: `libs/langchain/langchain_classic/memory/buffer_window.py:18-63`

```python
from langchain_classic.memory import ConversationBufferWindowMemory

# Keep only last 2 conversation turns (4 messages)
memory = ConversationBufferWindowMemory(k=2, return_messages=False)

# Simulate multiple interactions
memory.save_context({"input": "Hi"}, {"output": "Hello!"})
memory.save_context({"input": "How are you?"}, {"output": "I'm good, thanks!"})
memory.save_context({"input": "What's the weather?"}, {"output": "It's sunny!"})
memory.save_context({"input": "Any recommendations?"}, {"output": "Enjoy the outdoors!"})

# Only last 2 turns retained (4 messages)
history = memory.load_memory_variables({})
print(history["history"])
# Output:
# Human: What's the weather?
# AI: It's sunny!
# Human: Any recommendations?
# AI: Enjoy the outdoors!

# First two exchanges ("Hi" and "How are you?") are dropped

# Verify message count
print("Total messages stored:", len(memory.chat_memory.messages))
# Output: Total messages stored: 4 (even though we saved 8 messages)
```

**Window Behavior**:
- `k=2` means last 2 turns = last 4 messages (2 human + 2 AI)
- Window: `messages[-k * 2:]` = `messages[-4:]`
- Older messages automatically pruned on each `save_context()`

---

### Example 3: ConversationStringBufferMemory

**Source**: `libs/langchain/langchain_classic/memory/buffer.py:102-180`

```python
from langchain_classic.memory import ConversationStringBufferMemory

# String-formatted memory for non-chat models
memory = ConversationStringBufferMemory()

memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
memory.save_context({"input": "How's it going?"}, {"output": "Great, thanks for asking!"})
memory.save_context({"input": "Goodbye"}, {"output": "See you later!"})

# Access string buffer directly
print("String buffer:")
print(memory.buffer)
# Output:
# 
# Human: Hello
# AI: Hi there!
# Human: How's it going?
# AI: Great, thanks for asking!
# Human: Goodbye
# AI: See you later!

# Load via interface
history = memory.load_memory_variables({})
print("\nLoaded history:")
print(history["history"])
# Same formatted string output

# Clear resets string to empty
memory.clear()
print("\nAfter clear:", repr(memory.buffer))
# Output: After clear: ''
```

---

### Example 4: Custom memory_key and input/output Keys

**Source**: `libs/langchain/langchain_classic/memory/chat_memory.py:25-100`

```python
from langchain_classic.memory import ConversationBufferMemory

# Configure custom keys for complex chains
memory = ConversationBufferMemory(
    memory_key="chat_history",     # Custom key in chain inputs
    input_key="user_input",         # Which input dict key has user message
    output_key="bot_response",      # Which output dict key has AI message
    return_messages=True
)

# save_context extracts from specified keys
memory.save_context(
    {
        "user_input": "What's 2+2?",
        "timestamp": "2024-01-01T10:00:00",
        "session_id": "abc123"
    },
    {
        "bot_response": "2+2 equals 4.",
        "confidence": 0.99,
        "model_used": "gpt-4"
    }
)

# Only extracts "user_input" and "bot_response" keys
history = memory.load_memory_variables({})
print(history["chat_history"])
# Output: [HumanMessage(content="What's 2+2?"), 
#          AIMessage(content="2+2 equals 4.")]

# Metadata and extra keys are ignored
```

---

### Example 5: ConversationTokenBufferMemory

**Source**: `libs/langchain/langchain_classic/memory/token_buffer.py:19-75`

```python
from langchain_classic.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI

# Initialize with token limit
llm = ChatOpenAI(temperature=0)
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=100,  # Strict limit
    return_messages=True
)

# Add messages - automatic pruning when limit exceeded
memory.save_context(
    {"input": "Tell me about Python programming language"},
    {"output": "Python is a high-level, interpreted programming language..."}
)

memory.save_context(
    {"input": "What about JavaScript?"},
    {"output": "JavaScript is a versatile language primarily used for web development..."}
)

memory.save_context(
    {"input": "And TypeScript?"},
    {"output": "TypeScript is a superset of JavaScript that adds static typing..."}
)

# Check what's retained
history = memory.load_memory_variables({})
messages = history["history"]
print(f"Retained {len(messages)} messages")

# Oldest messages pruned to stay under max_token_limit
# Token count checked using: llm.get_num_tokens_from_messages(buffer)
```

**Token Pruning Logic** (Source: Lines 64-74):
1. After each `save_context`, calculates: `curr_buffer_length = llm.get_num_tokens_from_messages(buffer)`
2. If `curr_buffer_length > max_token_limit`:
   - Removes oldest message: `buffer.pop(0)`
   - Recalculates token count
   - Repeats until under limit

---

### Example 6: ConversationSummaryBufferMemory

**Source**: `libs/langchain/langchain_classic/memory/summary_buffer.py:20-120`

```python
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

# Hybrid: recent messages + summary of older
llm = ChatOpenAI(temperature=0)
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200,
    return_messages=False
)

# Add several interactions
memory.save_context(
    {"input": "What's the capital of France?"},
    {"output": "The capital of France is Paris."}
)

memory.save_context(
    {"input": "What's the population?"},
    {"output": "Paris has approximately 2.2 million residents."}
)

memory.save_context(
    {"input": "What's it famous for?"},
    {"output": "Paris is famous for the Eiffel Tower, Louvre Museum, and cuisine."}
)

# Load memory - includes summary + recent messages
history = memory.load_memory_variables({})
print(history["history"])
# Output may include:
# System: [Summary of early conversation about Paris]
# Human: What's it famous for?
# AI: Paris is famous for...

# moving_summary_buffer contains older conversation summary
# Recent messages kept verbatim
```

**Summary Behavior**:
- Monitors total token count
- When exceeding `max_token_limit`, triggers summarization of oldest messages
- Summary stored in `moving_summary_buffer`
- Prepended as SystemMessage when loading memory

---

### Example 7: VectorStoreRetrieverMemory

**Source**: `libs/langchain/langchain_classic/memory/vectorstore.py:23-120`

```python
from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Setup vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts([], embeddings)  # Start empty

# Create memory with retriever
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_docs=False
)

# Save multiple conversations
memory.save_context(
    {"input": "I love machine learning"},
    {"output": "ML is fascinating! What interests you most?"}
)

memory.save_context(
    {"input": "What's for dinner?"},
    {"output": "How about pasta tonight?"}
)

memory.save_context(
    {"input": "Tell me about neural networks"},
    {"output": "Neural networks are inspired by biological neurons..."}
)

# Retrieve semantically relevant context
# Query about ML retrieves ML-related conversations
history = memory.load_memory_variables({"input": "Explain deep learning"})
print(history["history"])
# Output: Retrieved context about ML and neural networks
# (not dinner conversation - semantic filtering)
```

**Retrieval Mechanism** (Source: Lines 67-85):
1. Extracts query from inputs using `input_key`
2. Calls `retriever.invoke(query)` to get top-k relevant documents
3. Returns concatenated page_content if `return_docs=False`
4. Returns Document objects if `return_docs=True`

## Context Window Management Strategies

Different memory types offer distinct strategies for managing limited context windows:

### Strategy 1: Fixed Window (ConversationBufferWindowMemory)

**Mechanism**: Keep exactly last `k` conversation turns.

**Source**: `libs/langchain/langchain_classic/memory/buffer_window.py:39`

**Implementation**: `messages[-k * 2:]` slicing

**Pros**:
- **Simple**: Easy to understand and predict
- **Predictable size**: Memory footprint is constant
- **No computation overhead**: Pure slicing operation

**Cons**:
- **Abrupt context loss**: Oldest messages dropped immediately when k exceeded
- **Fixed behavior**: Cannot adapt to varying message lengths
- **May lose critical context**: Important setup information can be discarded

**Best For**:
- Chatbots with short-term context needs
- Applications where recent exchanges matter most
- When computational efficiency is critical

**Configuration**:
```python
memory = ConversationBufferWindowMemory(
    k=5,  # Keep last 5 turns = 10 messages
    return_messages=True
)
```

**Typical k Values**:
- `k=3-5`: Very short-term context (6-10 messages)
- `k=5-10`: Standard conversational context (10-20 messages)
- `k=10-20`: Extended context (20-40 messages)

---

### Strategy 2: Token-Based Truncation (ConversationTokenBufferMemory)

**Mechanism**: Prune oldest messages to stay under `max_token_limit`.

**Source**: `libs/langchain/langchain_classic/memory/token_buffer.py:64-74`

**Implementation**: Uses `llm.get_num_tokens_from_messages()` to count tokens, removes oldest until under limit

**Pros**:
- **Precise token control**: Exactly manage context window usage
- **Adaptive**: Handles varying message lengths
- **Model-aware**: Uses actual tokenization from LLM

**Cons**:
- **Token counting overhead**: Adds computation on every `save_context()`
- **Requires LLM dependency**: Must provide model with token counting capability
- **FIFO pruning**: No consideration of message importance

**Best For**:
- Applications with strict token budget
- When using models with known context limits (e.g., 4K, 8K, 16K tokens)
- Production systems requiring precise control

**Configuration**:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")  # 4K context
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000,  # Reserve 2K for history, 2K for prompt+response
    return_messages=True
)
```

**Typical Limits by Model**:
- GPT-3.5-turbo (4K): `max_token_limit=1500-2000`
- GPT-4 (8K): `max_token_limit=4000-5000`
- GPT-4-turbo (128K): `max_token_limit=60000-80000`
- Claude-2 (100K): `max_token_limit=50000-70000`

---

### Strategy 3: Summarization (ConversationSummaryMemory)

**Mechanism**: Use LLM to create rolling summary, replacing full history with condensed version.

**Source**: `libs/langchain/langchain_classic/memory/summary.py:36-57`

**Implementation**: Calls LLM with SUMMARY_PROMPT to generate summary from new messages + existing summary

**Pros**:
- **Preserves gist**: Maintains essence of long conversations
- **Unbounded conversations**: Can handle arbitrarily long histories
- **Semantic compression**: Intelligently condenses information

**Cons**:
- **LLM calls add latency**: Each `save_context` triggers summarization
- **Increased cost**: Summary generation consumes tokens
- **Potential information loss**: Nuanced details and exact phrasing lost
- **Quality dependent**: Summary quality depends on LLM capability

**Best For**:
- Long customer support conversations
- Multi-session interactions
- When gist is sufficient (vs exact wording)

**Configuration**:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.3)  # Slightly creative for summaries
memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=False
)
```

**Cost Consideration**:
- Each summary call: ~100-300 tokens for summarization prompt + existing content
- Trade-off: Reduced prompt tokens in main chain vs summarization cost
- Break-even: Typically after 5-10 conversation turns

---

### Strategy 4: Hybrid (ConversationSummaryBufferMemory)

**Mechanism**: Keep recent messages verbatim + summary of older messages.

**Source**: `libs/langchain/langchain_classic/memory/summary_buffer.py:51-67`

**Implementation**: Monitors token count; when exceeded, summarizes oldest messages, keeps recent ones

**Pros**:
- **Detailed recent context**: Latest exchanges preserved exactly
- **Historical awareness**: Older context available as summary
- **Balanced approach**: Combines benefits of buffer and summary

**Cons**:
- **Increased complexity**: More sophisticated than single-strategy approaches
- **Still requires LLM**: Inherits summarization costs
- **Token management**: Must carefully tune `max_token_limit`

**Best For**:
- Production applications balancing detail and length
- When recent exchanges need high fidelity
- Multi-turn task completion scenarios

**Configuration**:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000,  # Total tokens for summary + recent messages
    return_messages=True
)
```

**Behavior**:
1. Tracks total token count of `moving_summary_buffer + chat_memory.messages`
2. When exceeding `max_token_limit`:
   - Summarizes oldest messages
   - Appends to `moving_summary_buffer`
   - Removes those messages from chat_memory
3. On load: Prepends summary as SystemMessage, followed by recent messages

---

### Strategy 5: Retrieval (VectorStoreRetrieverMemory)

**Mechanism**: Store all messages in vector database, retrieve semantically relevant subset.

**Source**: `libs/langchain/langchain_classic/memory/vectorstore.py:67-85`

**Implementation**: Embeds all interactions, queries vector store with current input, returns top-k similar

**Pros**:
- **Semantic relevance**: Retrieves contextually related messages, not just recent
- **Unbounded storage**: No size limit on total history
- **Selective context**: Only includes relevant past interactions

**Cons**:
- **Infrastructure dependency**: Requires vector store setup and maintenance
- **Retrieval accuracy**: Quality depends on embeddings and similarity metrics
- **Complexity**: More moving parts than simple buffer
- **No guaranteed completeness**: May miss relevant but dissimilar context

**Best For**:
- Very long conversations (100+ turns)
- When semantic relevance > recency
- Customer support with varied topics
- Knowledge-intensive applications

**Configuration**:
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts([], embeddings)

memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Top 3 relevant
    return_docs=False
)
```

**Retrieval Parameters**:
- `k`: Number of relevant past interactions to retrieve (typical: 2-5)
- `score_threshold`: Minimum similarity score (if supported by vector store)
- `fetch_k`: Number of candidates to fetch before filtering

---

### Strategy Comparison Table

| Strategy | Memory Type | Pros | Cons | Best Use Case |
|----------|-------------|------|------|---------------|
| **Fixed Window** | BufferWindowMemory | Simple, predictable | Abrupt loss, fixed size | Short-term context |
| **Token-Based** | TokenBufferMemory | Precise control, adaptive | Overhead, FIFO | Strict token budgets |
| **Summarization** | SummaryMemory | Preserves gist, unbounded | Cost, latency | Long conversations |
| **Hybrid** | SummaryBufferMemory | Balanced, detailed recent | Complex, still costly | Production apps |
| **Retrieval** | VectorStoreRetrieverMemory | Semantic, unbounded | Infrastructure, accuracy | Very long histories |

---

### Recommendation by Use Case

**For Chatbots (10-50 turns)**:
- Start with `ConversationBufferWindowMemory(k=10)`
- Upgrade to `ConversationTokenBufferMemory` if token management critical
- Consider `ConversationSummaryBufferMemory` if conversations exceed 50 turns

**For Customer Support (50+ turns, multi-session)**:
- Use `ConversationSummaryBufferMemory(max_token_limit=3000)` for balance
- Or `VectorStoreRetrieverMemory` for very long histories with topic switching

**For Task Completion (focused, <20 turns)**:
- Use `ConversationBufferMemory` (unbounded) for full context
- Clear memory after task completion

**For Production (cost-sensitive)**:
- Use `ConversationTokenBufferMemory` for precise cost control
- Monitor token usage, adjust `max_token_limit` based on metrics

## Troubleshooting

### Issue: Memory Not Persisting Across Runs

**Symptoms**:
- Conversation history lost after application restart
- New process has no memory of previous interactions

**Root Cause**: Default memory uses `InMemoryChatMessageHistory`, which is ephemeral.

**Source**: `libs/langchain/langchain_classic/memory/chat_memory.py:36-38`

**Solution**: Use persistent chat_message_histories backend:

```python
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory

# Persistent file-based storage
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("history.json"),
    return_messages=True
)
```

**Alternative Solutions**:
- Redis: `RedisChatMessageHistory(session_id="user_123")`
- PostgreSQL: `PostgresChatMessageHistory(connection_string=...)`
- DynamoDB: `DynamoDBChatMessageHistory(table_name=...)`

---

### Issue: Context Window Exceeded Error

**Symptoms**:
- `InvalidRequestError`: "This model's maximum context length is X tokens"
- Chain fails during LLM call with token limit error

**Root Cause**: Memory + prompt + response exceeds model's context window.

**Solution 1**: Switch to windowed memory:

```python
from langchain_classic.memory import ConversationBufferWindowMemory

# Limit to last 5 turns
memory = ConversationBufferWindowMemory(k=5, return_messages=True)
```

**Solution 2**: Use token-based memory:

```python
from langchain_classic.memory import ConversationTokenBufferMemory

# Reserve 2K tokens for memory (out of 4K total)
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000,
    return_messages=True
)
```

**Solution 3**: Use summarization:

```python
from langchain_classic.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1500,
    return_messages=True
)
```

**Prevention**:
- Monitor token usage: `llm.get_num_tokens_from_messages(memory.chat_memory.messages)`
- Choose memory type based on expected conversation length
- Set appropriate limits based on model capacity

---

### Issue: Memory Returns Wrong Format

**Symptoms**:
- Expected List[BaseMessage] but got string
- Expected string but got List[BaseMessage]
- Type errors in prompt template or chain

**Root Cause**: Mismatch between `return_messages` flag and consumer expectations.

**Source**: `libs/langchain/langchain_classic/memory/buffer.py:38`

**Solution**: Match `return_messages` to your use case:

**For Chat Models (ChatOpenAI, ChatAnthropic)**:
```python
memory = ConversationBufferMemory(return_messages=True)  # List[BaseMessage]
```

**For LLMs (OpenAI, Anthropic text models)**:
```python
memory = ConversationBufferMemory(return_messages=False)  # str
```

**Verification**:
```python
memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "test"}, {"output": "response"})

context = memory.load_memory_variables({})
print(type(context["history"]))  # Should be <class 'list'>
```

---

### Issue: save_context Raises ValueError

**Symptoms**:
- `ValueError: One output key expected, got {...}`
- `ValueError: Got multiple output keys: {...}, cannot determine which to store`

**Root Cause**: Ambiguous output_key when chain returns multiple outputs.

**Source**: `libs/langchain/langchain_classic/memory/chat_memory.py:52-69`

**Solution**: Explicitly set `output_key`:

```python
from langchain_classic.memory import ConversationBufferMemory

# Chain returns: {"answer": "...", "sources": "...", "confidence": 0.9}
memory = ConversationBufferMemory(
    output_key="answer",  # Specify which key contains AI response
    return_messages=True
)

# Now save_context works correctly
memory.save_context(
    {"input": "question"},
    {"answer": "response", "sources": "docs", "confidence": 0.9}
)
```

**Similarly for input_key**:
```python
memory = ConversationBufferMemory(
    input_key="user_query",
    output_key="assistant_answer",
    return_messages=True
)
```

---

### Issue: Memory Shows Stale/Incorrect Data

**Symptoms**:
- Memory contains messages from different conversation
- Historical data appears to be corrupted or mixed

**Root Cause**: Shared memory instance across multiple conversations/users.

**Solution**: Create per-session memory instances:

```python
# BAD: Global shared memory
global_memory = ConversationBufferMemory()

def handle_user_request(user_id):
    chain = ConversationChain(llm=llm, memory=global_memory)  # WRONG
    # All users share same memory!

# GOOD: Per-user memory
def handle_user_request(user_id):
    user_memory = ConversationBufferMemory(
        chat_memory=get_user_history(user_id)  # Separate backend per user
    )
    chain = ConversationChain(llm=llm, memory=user_memory)  # CORRECT
```

**With Session IDs**:
```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def create_user_memory(user_id: str):
    return ConversationBufferMemory(
        chat_memory=RedisChatMessageHistory(
            url="redis://localhost:6379",
            session_id=f"user_{user_id}"  # Unique per user
        ),
        return_messages=True
    )
```

---

### Issue: Async Methods Not Working

**Symptoms**:
- `RuntimeError: no running event loop`
- Async methods (`aload_memory_variables`, `asave_context`) fail

**Root Cause**: Calling async methods without event loop or mixing sync/async incorrectly.

**Solution**: Ensure proper async context:

```python
import asyncio
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI

async def run_conversation():
    memory = ConversationBufferMemory(return_messages=True)
    llm = ChatOpenAI(temperature=0)
    chain = ConversationChain(llm=llm, memory=memory)
    
    # Use ainvoke for async
    response = await chain.ainvoke({"input": "Hello"})
    print(response["response"])

# Run in event loop
asyncio.run(run_conversation())
```

**Note**: Memory's async methods are called automatically by chain's async methods.

---

### Issue: ConversationStringBufferMemory Raises ValueError with return_messages=True

**Symptoms**:
- `ValueError: return_messages must be False for ConversationStringBufferMemory`

**Root Cause**: ConversationStringBufferMemory only supports string format.

**Source**: `libs/langchain/langchain_classic/memory/buffer.py:124-128`

**Solution**: Use ConversationBufferMemory instead:

```python
# WRONG
memory = ConversationStringBufferMemory(return_messages=True)  # Raises ValueError

# RIGHT: Use ConversationBufferMemory for message format
memory = ConversationBufferMemory(return_messages=True)

# OR: Keep ConversationStringBufferMemory with string format
memory = ConversationStringBufferMemory(return_messages=False)
```

---

### Issue: Token Count Incorrect or Memory Not Pruning

**Symptoms**:
- ConversationTokenBufferMemory exceeds expected size
- Token limit not being enforced

**Root Cause**: LLM's `get_num_tokens_from_messages()` method may not be accurate or available.

**Solution**: Ensure LLM supports token counting:

```python
from langchain_openai import ChatOpenAI

# Use OpenAI models which have accurate token counting
llm = ChatOpenAI(model="gpt-3.5-turbo")

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000,
    return_messages=True
)

# Verify token counting works
messages = memory.chat_memory.messages
token_count = llm.get_num_tokens_from_messages(messages)
print(f"Current tokens: {token_count}")
```

**Note**: Not all LLM implementations provide accurate token counting. OpenAI models are recommended for TokenBufferMemory.

## References

### Migration Guide

**Deprecation Notice**: All memory classes are deprecated as of v0.3.1.

**Official Migration Guide**: https://python.langchain.com/docs/versions/migrating_memory/

The migration guide covers:
- Modern alternatives to memory classes
- LangGraph's built-in state management
- RunnableWithMessageHistory for stateful chains
- Best practices for production memory management

---

### Related Modules

**Chat Message Histories** (persistent backends):
- Location: `libs/langchain/langchain_classic/memory/chat_message_histories/`
- Community implementations: `langchain_community.chat_message_histories`
- Includes: Redis, PostgreSQL, MongoDB, DynamoDB, File-based, and more

**Base Memory Interface**:
- Location: `libs/langchain/langchain_classic/base_memory.py`
- Defines: `BaseMemory` abstract class with core interface

**LangChain Core Messages**:
- Location: `libs/core/langchain_core/messages/`
- Defines: `BaseMessage`, `HumanMessage`, `AIMessage`, `SystemMessage`, `FunctionMessage`

**Chat History Core**:
- Location: `libs/core/langchain_core/chat_history.py`
- Defines: `BaseChatMessageHistory`, `InMemoryChatMessageHistory`

---

### Source Files

| File | Description | Key Classes |
|------|-------------|-------------|
| `libs/langchain/langchain_classic/memory/__init__.py` | Module exports | All memory classes |
| `libs/langchain/langchain_classic/memory/buffer.py` | Buffer memory implementations | ConversationBufferMemory, ConversationStringBufferMemory |
| `libs/langchain/langchain_classic/memory/buffer_window.py` | Windowed buffer | ConversationBufferWindowMemory |
| `libs/langchain/langchain_classic/memory/chat_memory.py` | Base chat memory | BaseChatMemory (abstract base) |
| `libs/langchain/langchain_classic/memory/summary.py` | Summary memory | ConversationSummaryMemory, SummarizerMixin |
| `libs/langchain/langchain_classic/memory/summary_buffer.py` | Hybrid memory | ConversationSummaryBufferMemory |
| `libs/langchain/langchain_classic/memory/token_buffer.py` | Token-based memory | ConversationTokenBufferMemory |
| `libs/langchain/langchain_classic/memory/vectorstore.py` | Vector-based memory | VectorStoreRetrieverMemory |

---

### Additional Resources

**LangChain Documentation**: https://python.langchain.com/docs/

**LangChain API Reference**: https://api.python.langchain.com/

**Community Integrations**: https://python.langchain.com/docs/integrations/memory/

**GitHub Repository**: https://github.com/langchain-ai/langchain

---

## Quick Reference

### Memory Type Selection Flowchart

```
Start: What are your requirements?
│
├─ Short conversations (<20 turns)?
│  └─→ Use ConversationBufferMemory
│
├─ Need to limit by conversation turns?
│  └─→ Use ConversationBufferWindowMemory(k=5-10)
│
├─ Need strict token limit?
│  └─→ Use ConversationTokenBufferMemory(max_token_limit=2000)
│
├─ Long conversations (50+ turns)?
│  ├─ Cost-sensitive?
│  │  └─→ Use ConversationSummaryBufferMemory
│  └─ Quality-focused?
│     └─→ Use VectorStoreRetrieverMemory
│
└─ Very long, multi-topic conversations?
   └─→ Use VectorStoreRetrieverMemory
```

### Common Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `memory_key` | str | "history" | Key name in chain inputs |
| `return_messages` | bool | False | Return List[BaseMessage] (True) or str (False) |
| `input_key` | str \| None | None | Which input dict key contains user message |
| `output_key` | str \| None | None | Which output dict key contains AI response |
| `human_prefix` | str | "Human" | Prefix for user messages in string format |
| `ai_prefix` | str | "AI" | Prefix for AI messages in string format |
| `k` | int | 5 | (BufferWindowMemory) Number of turns to retain |
| `max_token_limit` | int | 2000 | (TokenBuffer, SummaryBuffer) Maximum tokens |
| `llm` | BaseLanguageModel | required* | (Summary, TokenBuffer) LLM for summarization/counting |

*Required for ConversationSummaryMemory, ConversationSummaryBufferMemory, ConversationTokenBufferMemory

---

**End of Memory Module Documentation**

For questions, issues, or contributions, please refer to the main LangChain repository and the official migration guide linked above.
