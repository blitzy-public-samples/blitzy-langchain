# API Index

Alphabetical index of all public LangChain integration APIs. Each entry links to detailed API reference documentation with complete type signatures, docstrings, and usage examples.

---

## A

**Agent** - Base class for agents that use an LLM to choose a sequence of actions → [Agent API Reference](api-reference/agents/agent-types.md#agent)

**AgentExecutor** - Runtime for executing agent decision loops with tool invocation → [AgentExecutor API Reference](api-reference/agents/agent-types.md#agentexecutor)

**AgentOutputParser** - Parser for converting agent LLM outputs into structured actions → [AgentOutputParser API Reference](api-reference/agents/agent-types.md#agentoutputparser)

**AgentType** - Enum defining supported agent types (ZERO_SHOT_REACT_DESCRIPTION, CHAT_CONVERSATIONAL_REACT_DESCRIPTION, etc.) → [AgentType API Reference](api-reference/agents/agent-types.md#agenttype)

**AIMessage** - Message type representing AI/assistant responses in conversations → [AIMessage API Reference](api-reference/prompts/message-types.md#aimessage)

**AIMessagePromptTemplate** - Template for constructing AI message prompts with variable substitution → [AIMessagePromptTemplate API Reference](api-reference/prompts/templates.md#aimessageprompttemplate)

**AnalyzeDocumentChain** - Chain for analyzing single documents by splitting and processing chunks → [AnalyzeDocumentChain API Reference](api-reference/chains/specialized/analyze-document.md)

**APIChain** - Chain for making API requests based on LLM-generated queries → [APIChain API Reference](api-reference/chains/specialized/api-chain.md)

**AsyncCallbackHandler** - Async base class for implementing custom callback handlers → [AsyncCallbackHandler API Reference](api-reference/callbacks/handlers.md#asynccallbackhandler)

## B

**BaseChatPromptTemplate** - Abstract base class for chat-based prompt templates → [BaseChatPromptTemplate API Reference](api-reference/prompts/templates.md#basechatprompttemplate)

**BaseCallbackHandler** - Base class for implementing custom callback handlers → [BaseCallbackHandler API Reference](api-reference/callbacks/handlers.md#basecallbackhandler)

**BaseMultiActionAgent** - Base class for agents that can take multiple actions per step → [BaseMultiActionAgent API Reference](api-reference/agents/agent-types.md#basemultiactionagent)

**BasePromptTemplate** - Abstract base class for all prompt templates → [BasePromptTemplate API Reference](api-reference/prompts/templates.md#baseprompttemplate)

**BaseSingleActionAgent** - Base class for agents that take one action per step → [BaseSingleActionAgent API Reference](api-reference/agents/agent-types.md#basesingleactionagent)

**BaseTool** - Base class for implementing custom tools with type validation → [BaseTool API Reference](api-reference/agents/tools.md#basetool)

## C

**chain** - Decorator for converting functions into Runnable chains → [chain decorator API Reference](api-reference/runnables/utilities.md#chain)

**ChatMessagePromptTemplate** - Template for arbitrary role chat messages → [ChatMessagePromptTemplate API Reference](api-reference/prompts/templates.md#chatmessageprompttemplate)

**ChatPromptTemplate** - Template for constructing multi-message chat prompts → [ChatPromptTemplate API Reference](api-reference/prompts/templates.md#chatprompttemplate)

**ChatVectorDBChain** - (Deprecated) Chain for conversational retrieval from vector databases → [ChatVectorDBChain API Reference](api-reference/chains/specialized/chat-vector-db.md)

**ConstitutionalChain** - Chain implementing constitutional AI principles for output validation → [ConstitutionalChain API Reference](api-reference/chains/specialized/constitutional.md)

**ConversationChain** - Simple chain for managing conversational interactions with memory → [ConversationChain API Reference](api-reference/chains/specialized/conversation.md)

**ConversationalAgent** - Agent optimized for conversational interactions → [ConversationalAgent API Reference](api-reference/agents/agent-types.md#conversationalagent)

**ConversationalChatAgent** - Chat-optimized conversational agent → [ConversationalChatAgent API Reference](api-reference/agents/agent-types.md#conversationalchatagent)

**ConversationalRetrievalChain** - Chain combining retrieval and conversation with context → [ConversationalRetrievalChain API Reference](api-reference/chains/specialized/conversational-retrieval.md)

**ConversationBufferMemory** - Memory that stores conversation history in a buffer → [ConversationBufferMemory API Reference](api-reference/memory/buffer-memory.md#conversationbuffermemory)

**ConversationBufferWindowMemory** - Memory that stores only the last K conversation turns → [ConversationBufferWindowMemory API Reference](api-reference/memory/buffer-memory.md#conversationbufferwindowmemory)

**create_history_aware_retriever** - Factory function creating retrievers that incorporate conversation history → [create_history_aware_retriever API Reference](api-reference/chains/retrieval.md#create_history_aware_retriever)

**create_json_chat_agent** - Factory function creating agents for JSON-based chat interactions → [create_json_chat_agent API Reference](api-reference/agents/agent-types.md#create_json_chat_agent)

**create_openai_functions_agent** - Factory function creating agents using OpenAI function calling → [create_openai_functions_agent API Reference](api-reference/agents/agent-types.md#create_openai_functions_agent)

**create_openai_tools_agent** - Factory function creating agents using OpenAI tools API → [create_openai_tools_agent API Reference](api-reference/agents/agent-types.md#create_openai_tools_agent)

**create_react_agent** - Factory function creating ReAct pattern agents → [create_react_agent API Reference](api-reference/agents/agent-types.md#create_react_agent)

**create_retrieval_chain** - Factory function creating retrieval-augmented generation chains → [create_retrieval_chain API Reference](api-reference/chains/retrieval.md#create_retrieval_chain)

**create_self_ask_with_search_agent** - Factory function creating self-ask agents with search capability → [create_self_ask_with_search_agent API Reference](api-reference/agents/agent-types.md#create_self_ask_with_search_agent)

**create_structured_chat_agent** - Factory function creating agents for structured chat interactions → [create_structured_chat_agent API Reference](api-reference/agents/agent-types.md#create_structured_chat_agent)

**create_tool_calling_agent** - Factory function creating tool-calling agents → [create_tool_calling_agent API Reference](api-reference/agents/agent-types.md#create_tool_calling_agent)

**create_vectorstore_agent** - Factory function creating agents for vector store operations → [create_vectorstore_agent API Reference](api-reference/agents/agent-types.md#create_vectorstore_agent)

**create_xml_agent** - Factory function creating XML-format agents → [create_xml_agent API Reference](api-reference/agents/agent-types.md#create_xml_agent)

## F

**FewShotChatMessagePromptTemplate** - Template for few-shot learning with chat messages → [FewShotChatMessagePromptTemplate API Reference](api-reference/prompts/templates.md#fewshotchatmessageprompttemplate)

**FewShotPromptTemplate** - Template for few-shot learning with examples → [FewShotPromptTemplate API Reference](api-reference/prompts/templates.md#fewshotprompttemplate)

**FlareChain** - Chain implementing Forward-Looking Active REtrieval pattern → [FlareChain API Reference](api-reference/chains/specialized/flare.md)

## H

**HumanMessage** - Message type representing human/user inputs in conversations → [HumanMessage API Reference](api-reference/prompts/message-types.md#humanmessage)

**HumanMessagePromptTemplate** - Template for constructing human message prompts with variable substitution → [HumanMessagePromptTemplate API Reference](api-reference/prompts/templates.md#humanmessageprompttemplate)

**HypotheticalDocumentEmbedder** - Chain for HyDE (Hypothetical Document Embeddings) pattern → [HypotheticalDocumentEmbedder API Reference](api-reference/chains/specialized/hyde.md)

## I

**initialize_agent** - Factory function for initializing agents with specified type and tools → [initialize_agent API Reference](api-reference/agents/agent-types.md#initialize_agent)

## J

**JsonOutputParser** - Output parser for extracting structured JSON from LLM responses → [JsonOutputParser API Reference](api-reference/utilities/output-parsers.md#jsonoutputparser)

## L

**LLMChain** - (Deprecated) Basic chain for LLM invocation with prompt template → [LLMChain API Reference](api-reference/chains/llm-chain.md)

**LLMCheckerChain** - Chain that verifies LLM outputs for factual accuracy → [LLMCheckerChain API Reference](api-reference/chains/specialized/llm-checker.md)

**LLMMathChain** - Chain for performing mathematical calculations via LLM → [LLMMathChain API Reference](api-reference/chains/specialized/llm-math.md)

**LLMSingleActionAgent** - Agent that uses LLM for single action selection → [LLMSingleActionAgent API Reference](api-reference/agents/agent-types.md#llmsingleactionagent)

**load_agent** - Function for loading serialized agent configurations → [load_agent API Reference](api-reference/agents/agent-types.md#load_agent)

**load_chain** - Function for loading serialized chain configurations → [load_chain API Reference](api-reference/chains/base.md#load_chain)

**load_prompt** - Function for loading serialized prompt templates → [load_prompt API Reference](api-reference/prompts/templates.md#load_prompt)

## M

**MapReduceChain** - Chain implementing map-reduce pattern for parallel processing → [MapReduceChain API Reference](api-reference/chains/sequential.md#mapreducechain)

**MapReduceDocumentsChain** - Chain for processing multiple documents with map-reduce → [MapReduceDocumentsChain API Reference](api-reference/chains/specialized/combine-documents.md#mapreducedocumentschain)

**MessagesPlaceholder** - Placeholder for message lists in chat prompt templates → [MessagesPlaceholder API Reference](api-reference/prompts/templates.md#messagesplaceholder)

**MRKLChain** - Modular Reasoning, Knowledge and Language chain → [MRKLChain API Reference](api-reference/agents/agent-types.md#mrklchain)

## O

**OpenAIFunctionsAgent** - Agent using OpenAI function calling API → [OpenAIFunctionsAgent API Reference](api-reference/agents/agent-types.md#openaifunctionsagent)

**OpenAIModerationChain** - Chain for content moderation using OpenAI moderation API → [OpenAIModerationChain API Reference](api-reference/chains/specialized/moderation.md)

**OpenAIMultiFunctionsAgent** - Agent supporting multiple OpenAI function calls → [OpenAIMultiFunctionsAgent API Reference](api-reference/agents/agent-types.md#openaimultifunctionsagent)

## P

**PromptTemplate** - Basic template for text prompts with variable substitution → [PromptTemplate API Reference](api-reference/prompts/templates.md#prompttemplate)

## Q

**QAGenerationChain** - Chain for generating question-answer pairs from documents → [QAGenerationChain API Reference](api-reference/chains/specialized/qa-generation.md)

**QAWithSourcesChain** - Question-answering chain that cites source documents → [QAWithSourcesChain API Reference](api-reference/chains/specialized/qa-with-sources.md)

## R

**ReActChain** - Chain implementing ReAct (Reasoning + Acting) pattern → [ReActChain API Reference](api-reference/agents/agent-types.md#reactchain)

**ReActTextWorldAgent** - ReAct agent for text-based environments → [ReActTextWorldAgent API Reference](api-reference/agents/agent-types.md#reacttextworldagent)

**ReduceDocumentsChain** - Chain for reducing multiple documents into single output → [ReduceDocumentsChain API Reference](api-reference/chains/specialized/combine-documents.md#reducedocumentschain)

**RefineDocumentsChain** - Chain for iteratively refining outputs across documents → [RefineDocumentsChain API Reference](api-reference/chains/specialized/combine-documents.md#refinedocumentschain)

**RetrievalQA** - Question-answering chain with document retrieval → [RetrievalQA API Reference](api-reference/chains/specialized/retrieval-qa.md)

**RetrievalQAWithSourcesChain** - Retrieval QA chain that cites sources → [RetrievalQAWithSourcesChain API Reference](api-reference/chains/specialized/qa-with-sources.md#retrievalqawithsourceschain)

**RouterChain** - Chain that routes inputs to different sub-chains → [RouterChain API Reference](api-reference/chains/specialized/router.md)

**Runnable** - Core protocol for composable LangChain components → [Runnable API Reference](api-reference/runnables/base.md#runnable)

**RunnableBinding** - Runnable with bound configuration or kwargs → [RunnableBinding API Reference](api-reference/runnables/base.md#runnablebinding)

**RunnableBranch** - Runnable that conditionally routes to different branches → [RunnableBranch API Reference](api-reference/runnables/utilities.md#runnablebranch)

**RunnableConfig** - Configuration object for Runnable execution (callbacks, tags, metadata) → [RunnableConfig API Reference](api-reference/runnables/composition.md#runnableconfig)

**RunnableGenerator** - Runnable created from a generator function → [RunnableGenerator API Reference](api-reference/runnables/base.md#runnablegenerator)

**RunnableLambda** - Runnable created from a simple function → [RunnableLambda API Reference](api-reference/runnables/base.md#runnablelambda)

**RunnableMap** - Runnable that executes multiple runnables in parallel and returns dict → [RunnableMap API Reference](api-reference/runnables/base.md#runnablemap)

**RunnableParallel** - Runnable for parallel execution of multiple runnables → [RunnableParallel API Reference](api-reference/runnables/base.md#runnableparallel)

**RunnablePassthrough** - Runnable that passes inputs through unchanged → [RunnablePassthrough API Reference](api-reference/runnables/utilities.md#runnablepassthrough)

**RunnableSequence** - Runnable representing sequential composition via pipe operator → [RunnableSequence API Reference](api-reference/runnables/base.md#runnablesequence)

**RunnableSerializable** - Runnable with serialization support → [RunnableSerializable API Reference](api-reference/runnables/base.md#runnableserializable)

**RunnableWithFallbacks** - Runnable with fallback behavior on errors → [RunnableWithFallbacks API Reference](api-reference/runnables/utilities.md#runnablewithfallbacks)

**RunnableWithMessageHistory** - Runnable with automatic message history management → [RunnableWithMessageHistory API Reference](api-reference/runnables/utilities.md#runnablewithmessagehistory)

## S

**SelfAskWithSearchChain** - Chain implementing self-ask with search pattern → [SelfAskWithSearchChain API Reference](api-reference/agents/agent-types.md#selfaskwithsearchchain)

**SequentialChain** - Chain that executes multiple chains in sequence → [SequentialChain API Reference](api-reference/chains/sequential.md#sequentialchain)

**SimpleSequentialChain** - Simplified sequential chain with single input/output per step → [SimpleSequentialChain API Reference](api-reference/chains/sequential.md#simplesequentialchain)

**StrOutputParser** - Output parser for extracting string content from LLM responses → [StrOutputParser API Reference](api-reference/utilities/output-parsers.md#stroutputparser)

**StringPromptTemplate** - Base class for string-based prompt templates → [StringPromptTemplate API Reference](api-reference/prompts/templates.md#stringprompttemplate)

**StructuredChatAgent** - Agent for structured chat interactions with tools → [StructuredChatAgent API Reference](api-reference/agents/agent-types.md#structuredchatagent)

**StructuredTool** - Tool implementation with Pydantic schema validation → [StructuredTool API Reference](api-reference/agents/tools.md#structuredtool)

**StuffDocumentsChain** - Chain that stuffs all documents into a single prompt → [StuffDocumentsChain API Reference](api-reference/chains/specialized/combine-documents.md#stuffdocumentschain)

**SystemMessage** - Message type representing system instructions in conversations → [SystemMessage API Reference](api-reference/prompts/message-types.md#systemmessage)

**SystemMessagePromptTemplate** - Template for constructing system message prompts → [SystemMessagePromptTemplate API Reference](api-reference/prompts/templates.md#systemmessageprompttemplate)

## T

**tool** - Decorator for converting functions into Tool objects → [tool decorator API Reference](api-reference/agents/tools.md#tool)

**Tool** - Simple tool implementation for agent actions → [Tool API Reference](api-reference/agents/tools.md#tool-class)

**TransformChain** - Chain that applies transformation function to inputs → [TransformChain API Reference](api-reference/chains/specialized/transform.md)

## V

**VectorDBQA** - (Deprecated) Question-answering over vector database → [VectorDBQA API Reference](api-reference/chains/specialized/vector-db-qa.md)

## X

**XMLAgent** - Agent that uses XML format for structured outputs → [XMLAgent API Reference](api-reference/agents/agent-types.md#xmlagent)

## Z

**ZeroShotAgent** - Zero-shot learning agent using ReAct pattern → [ZeroShotAgent API Reference](api-reference/agents/agent-types.md#zeroshotagent)

---

## API Categories

For easier navigation, APIs are also organized by category:

### Chains
[Chain Base Classes](api-reference/chains/base.md) | [LLMChain](api-reference/chains/llm-chain.md) | [Sequential Chains](api-reference/chains/sequential.md) | [Retrieval Chains](api-reference/chains/retrieval.md) | [Specialized Chains](api-reference/chains/specialized/)

### Agents
[Agent Types](api-reference/agents/agent-types.md) | [Tools](api-reference/agents/tools.md)

### Runnables (LCEL)
[Base Runnables](api-reference/runnables/base.md) | [Composition](api-reference/runnables/composition.md) | [Utilities](api-reference/runnables/utilities.md)

### Prompts
[Prompt Templates](api-reference/prompts/templates.md) | [Message Types](api-reference/prompts/message-types.md)

### Memory
[Buffer Memory](api-reference/memory/buffer-memory.md)

### Callbacks
[Callback Handlers](api-reference/callbacks/handlers.md)

### Utilities
[Output Parsers](api-reference/utilities/output-parsers.md)

---

## Quick Links

- [Getting Started Guide](getting-started/quickstart.md)
- [LCEL Composition Guide](guides/lcel-composition.md)
- [Architecture Documentation](architecture/overview.md)
- [Debugging Guide](debugging/common-issues.md)
- [Examples](../examples/README.md)

## Source References

This API index is generated from the following source files:

- `libs/langchain/langchain_classic/chains/__init__.py` - Chain exports
- `libs/langchain/langchain_classic/agents/__init__.py` - Agent exports  
- `libs/core/langchain_core/runnables/__init__.py` - Runnable exports
- `libs/core/langchain_core/prompts/__init__.py` - Prompt template exports

For complete implementation details, refer to the detailed API reference documentation linked for each API.
