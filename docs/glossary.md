# LangChain Integration Glossary

This glossary defines key terms and concepts used throughout the LangChain integration documentation. Each entry includes a clear definition, usage context, and links to detailed documentation.

---

## Agent

**Definition**: An autonomous system that uses Large Language Models (LLMs) to make decisions and take actions to accomplish goals. Agents can reason about problems, select appropriate tools to use, and execute multi-step workflows.

**Usage Context**: Agents are used when you need dynamic decision-making rather than predefined chains. They can access external tools (like search engines, calculators, or databases) and decide which tool to use based on the input and current state.

**Key Characteristics**:
- Autonomous decision-making using LLM reasoning
- Access to a set of tools that can be called dynamically
- Iterative execution loop: observation → thought → action → observation
- Requires tool descriptions and input schemas for proper operation

**Source**: `libs/langchain/langchain_classic/agents/agent.py`

**See Also**:
- [Agent Development Guide](guides/agent-development.md)
- [Agent Types API Reference](api-reference/agents/agent-types.md)
- [Tool Definition](#tool)

---

## Callback

**Definition**: Event handlers that are invoked at specific lifecycle points during chain execution, allowing you to hook into the execution flow for logging, monitoring, debugging, or custom processing.

**Usage Context**: Callbacks enable observability and extensibility without modifying core chain logic. Common use cases include logging intermediate results, tracking token usage, streaming outputs to external systems, or implementing custom tracing.

**Key Lifecycle Events**:
- `on_chain_start`: Invoked when a chain begins execution
- `on_chain_end`: Invoked when a chain completes successfully
- `on_chain_error`: Invoked when a chain raises an exception
- `on_llm_start`: Invoked when an LLM call begins
- `on_llm_end`: Invoked when an LLM call completes
- `on_llm_new_token`: Invoked for each token during streaming

**Source**: `libs/core/langchain_core/callbacks/base.py`

**See Also**:
- [Callbacks Guide](guides/callbacks.md)
- [Callback Handlers API Reference](api-reference/callbacks/handlers.md)
- [Callback System Architecture](architecture/callback-system.md)

---

## Chain

**Definition**: An abstract base class that represents structured sequences of calls to components like models, document retrievers, other chains, or utilities. Chains provide a consistent interface for executing multi-step workflows with built-in support for state management, observability, and composition.

**Usage Context**: Chains are the foundational building blocks for creating predictable, repeatable workflows. Use chains when you have a well-defined sequence of operations, such as loading context, formatting prompts, calling an LLM, and parsing the output.

**Key Features**:
- **Stateful**: Can integrate memory to maintain conversation context
- **Observable**: Accepts callbacks for logging and monitoring
- **Composable**: Can be combined with other chains and components
- **Structured**: Enforces a consistent input/output interface (typically dictionaries)

**Core Methods**:
- `invoke()`: Execute the chain with a single input
- `ainvoke()`: Async version of invoke
- `batch()`: Process multiple inputs efficiently
- `stream()`: Stream output as it's generated

**Source**: `libs/langchain/langchain_classic/chains/base.py:52`

**See Also**:
- [Chain Types Guide](guides/chain-types.md)
- [Chain Base API Reference](api-reference/chains/base.md)
- [Chain Execution Lifecycle](architecture/chain-lifecycle.md)

---

## LCEL (LangChain Expression Language)

**Definition**: A declarative syntax for composing Runnable objects into chains using the pipe operator (`|`). LCEL provides type-safe composition with automatic support for synchronous, asynchronous, batch, and streaming execution.

**Usage Context**: LCEL is the modern, recommended way to build LangChain applications. It replaces legacy chain classes with a more flexible, composable approach that makes type transformations explicit and enables powerful composition patterns.

**Key Concepts**:
- **Pipe Operator (`|`)**: Connects runnables sequentially (output of left becomes input of right)
- **Type Safety**: Automatically validates that output type of one runnable matches input type of the next
- **Automatic Parallelization**: Batch operations run efficiently without manual threading
- **Universal Methods**: All LCEL chains support `invoke`, `ainvoke`, `batch`, `abatch`, `stream`, `astream`

**Example Type Flow**:
```
Dict[str, str] → PromptTemplate → List[BaseMessage] → ChatModel → AIMessage → StrOutputParser → str
```

**Source**: `libs/core/langchain_core/runnables/base.py:150-167`

**See Also**:
- [LCEL Composition Guide](guides/lcel-composition.md)
- [Runnables API Reference](api-reference/runnables/base.md)
- [LCEL Type System](architecture/lcel-type-system.md)
- [Runnable Definition](#runnable)

---

## Memory

**Definition**: Components that manage conversation state and context across multiple chain invocations. Memory systems store and retrieve relevant information to maintain coherent, context-aware interactions.

**Usage Context**: Use memory when building conversational applications, chatbots, or any system that needs to reference previous interactions. Memory types vary in complexity from simple buffer storage to sophisticated retrieval-augmented systems.

**Common Memory Types**:
- **ConversationBufferMemory**: Stores complete conversation history
- **ConversationBufferWindowMemory**: Keeps only the last N messages
- **ConversationSummaryMemory**: Maintains a running summary of the conversation
- **VectorStoreMemory**: Uses semantic search to retrieve relevant past interactions

**Key Methods**:
- `load_memory_variables()`: Retrieve relevant context for current input
- `save_context()`: Store new inputs and outputs after chain execution

**Source**: `libs/langchain/langchain_classic/memory/buffer.py`

**See Also**:
- [Memory Integration Guide](guides/memory-integration.md)
- [Memory API Reference](api-reference/memory/buffer-memory.md)

---

## Message Types

**Definition**: Structured representations of different roles in a conversation (human, AI, system). Message types provide type safety and semantic clarity when building chat-based applications.

**Usage Context**: Message types are the standard way to represent conversational turns in LangChain. They're used extensively with chat models, prompt templates, and memory systems to maintain proper conversation structure.

**Core Message Types**:
- **HumanMessage**: Represents input from a human user
- **AIMessage**: Represents output from an AI/LLM
- **SystemMessage**: Represents system-level instructions or context
- **FunctionMessage**: Represents results from function/tool calls
- **ToolMessage**: Represents tool execution results in agent workflows

**Common Conversions**:
- String → HumanMessage (implicit in many APIs)
- AIMessage → String (via content property or output parsers)
- Message lists → Formatted prompt strings (via chat templates)

**Source**: `libs/core/langchain_core/messages/`

**See Also**:
- [Message Types API Reference](api-reference/prompts/message-types.md)
- [Message Flow Architecture](architecture/message-flow.md)

---

## Output Parser

**Definition**: Components that transform unstructured LLM text output into structured, typed data formats like JSON objects, Pydantic models, lists, or custom types.

**Usage Context**: Use output parsers to reliably extract structured information from LLM responses. Instead of manually parsing text, output parsers handle the transformation and provide error handling for malformed outputs.

**Common Parser Types**:
- **StrOutputParser**: Extracts plain text string from AI messages
- **JsonOutputParser**: Parses JSON from LLM output into Python dictionaries
- **PydanticOutputParser**: Validates and parses output into Pydantic models
- **ListOutputParser**: Extracts lists or sequences from text
- **DatetimeOutputParser**: Parses date/time strings into Python datetime objects

**Key Methods**:
- `parse()`: Transform LLM output string into structured format
- `parse_with_prompt()`: Parse with access to the original prompt for context
- `get_format_instructions()`: Generate instructions to include in prompts for proper formatting

**Source**: `libs/core/langchain_core/output_parsers/`

**See Also**:
- [Output Parsers API Reference](api-reference/utilities/output-parsers.md)
- [LCEL Type Flow Examples](guides/lcel-composition.md#output-parsers)

---

## Prompt Template

**Definition**: Reusable prompt structures with variable placeholders that enable dynamic prompt generation. Prompt templates separate prompt engineering from application logic and support consistent formatting.

**Usage Context**: Use prompt templates to create maintainable, testable prompts that can be reused across different inputs. Templates support variable substitution, conditional sections, and validation of required variables.

**Common Template Types**:
- **PromptTemplate**: Simple string-based templates with variable substitution
- **ChatPromptTemplate**: Templates for multi-message chat conversations
- **FewShotPromptTemplate**: Templates that include example inputs/outputs
- **MessagePromptTemplate**: Templates for individual messages in a chat sequence

**Key Features**:
- Variable substitution using `{variable_name}` syntax
- Type validation for input variables
- Partial variable application for reusable templates
- Automatic formatting for different model types

**Example**:
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Tell me about {topic}")
])

prompt = template.invoke({"topic": "Python"})
```

**Source**: `libs/core/langchain_core/prompts/chat.py`

**See Also**:
- [Prompt Templates API Reference](api-reference/prompts/templates.md)
- [LCEL Composition with Prompts](guides/lcel-composition.md#prompt-templates)

---

## Runnable

**Definition**: The core protocol and base class for composable units of work in LangChain. Runnables define a standard interface for components that can be invoked, batched, streamed, transformed, and composed using LCEL.

**Usage Context**: Runnable is the fundamental abstraction that enables LCEL composition. Almost all LangChain components (models, prompts, parsers, chains, retrievers) implement the Runnable interface, making them interoperable and composable.

**Key Interface Methods**:
- `invoke(input)`: Transform a single input into an output (synchronous)
- `ainvoke(input)`: Async version of invoke
- `batch(inputs)`: Efficiently process multiple inputs (parallelized by default)
- `abatch(inputs)`: Async version of batch
- `stream(input)`: Stream output as it's produced
- `astream(input)`: Async version of stream

**Type Parameters**:
- `Input`: The type of input the runnable accepts
- `Output`: The type of output the runnable produces
- Type-safe composition: `Runnable[A,B] | Runnable[B,C]` → `Runnable[A,C]`

**Built-in Optimizations**:
- Parallel batch execution using thread pools
- Automatic async support for synchronous implementations
- Configurable execution with callbacks, tags, and metadata

**Source**: `libs/core/langchain_core/runnables/base.py:122`

**See Also**:
- [Runnable Protocol API Reference](api-reference/runnables/base.md)
- [LCEL Composition Guide](guides/lcel-composition.md)
- [LCEL Definition](#lcel-langchain-expression-language)

---

## Tool

**Definition**: Functions that agents can call to perform specific actions or retrieve information. Tools have defined names, descriptions, and input schemas that agents use to determine when and how to invoke them.

**Usage Context**: Tools extend agent capabilities beyond pure language generation. They enable agents to perform calculations, search databases, call APIs, or execute any programmatic logic that would be difficult or impossible for an LLM alone.

**Required Components**:
- **Name**: Unique identifier for the tool (e.g., "calculator", "web_search")
- **Description**: Natural language explanation of what the tool does and when to use it
- **Args Schema**: Pydantic model defining the tool's input parameters and types
- **Implementation**: The actual function logic (_run for sync, _arun for async)

**Tool Definition Methods**:
- **BaseTool class**: Inherit and implement _run() and _arun() methods
- **StructuredTool**: Create tools from existing functions with structured inputs
- **@tool decorator**: Simplest method - decorate a function to create a tool

**Example**:
```python
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the product database for items matching the query.
    
    Args:
        query: The search term to look for
        
    Returns:
        Matching product information
    """
    # Implementation here
    return results
```

**Source**: `libs/langchain/langchain_classic/tools/base.py`

**See Also**:
- [Tools API Reference](api-reference/agents/tools.md)
- [Agent Development Guide](guides/agent-development.md)
- [Agent Definition](#agent)

---

## Additional Resources

For comprehensive coverage of these concepts, please refer to:

- **[Getting Started Guide](getting-started/quickstart.md)**: Quick introduction to core concepts
- **[API Reference](api-reference/)**: Detailed API documentation for all components
- **[User Guides](guides/)**: In-depth tutorials and best practices
- **[Architecture Documentation](architecture/)**: System design and data flow diagrams
- **[Debugging Guide](debugging/common-issues.md)**: Troubleshooting and common issues

---

**Last Updated**: Documentation reflects LangChain v1.0.0+ API

**Note**: This glossary is maintained alongside the codebase. Source code citations link to specific files for verification and deeper exploration.
