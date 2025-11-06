"""
Memory-Enabled Conversational Chain Example

This example demonstrates how to create a conversational chain with memory
using ConversationBufferMemory from LangChain. Memory allows the chain to
maintain context across multiple interactions, enabling natural conversations
where the model remembers previous exchanges.

Key Concepts Demonstrated:
- ConversationBufferMemory initialization and configuration
- Memory integration with LCEL (LangChain Expression Language) chains
- Automatic memory persistence across conversation turns
- Manual memory management (save_context, load_memory_variables, clear)
- Proper prompt template setup with memory placeholders
- Error handling for missing API keys and LLM failures

Source: Agent Action Plan section 0.5.1 - Advanced chains examples
"""

import os
import sys
from typing import Dict, Any

# Core LangChain imports for memory and prompts
try:
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI
except ImportError as e:
    print(f"Error: Missing required package. Please install dependencies:")
    print(f"  pip install langchain-core langchain-classic langchain-openai")
    print(f"Details: {e}")
    sys.exit(1)

# Import dotenv for environment variable management
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Ensure OPENAI_API_KEY is set manually.")


def setup_memory() -> ConversationBufferMemory:
    """
    Initialize ConversationBufferMemory with proper configuration.
    
    ConversationBufferMemory stores the complete conversation history as a buffer,
    maintaining all previous human inputs and AI responses. This enables the chain
    to reference earlier parts of the conversation.
    
    Returns:
        ConversationBufferMemory: Configured memory instance with:
            - memory_key: "chat_history" - key used to store/retrieve conversation
            - return_messages: True - returns structured Message objects vs strings
            - input_key: "input" - key for human input in save_context
            - output_key: "output" - key for AI output in save_context
    
    Source: libs/langchain/tests/unit_tests/chains/test_memory.py:26
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",      # Key for accessing history in prompt template
        return_messages=True,             # Return as Message objects for ChatPromptTemplate
        input_key="input",                # Expected key for human input
        output_key="output"               # Expected key for AI response
    )
    return memory


def create_conversational_prompt() -> ChatPromptTemplate:
    """
    Create a conversational prompt template with memory placeholder.
    
    The prompt template includes:
    1. System message defining the assistant's behavior
    2. MessagesPlaceholder for injecting conversation history from memory
    3. Human message placeholder for the current user input
    
    Returns:
        ChatPromptTemplate: Template configured for memory-enabled conversations
    
    Source: libs/core/langchain_core/prompts/chat.py
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. You remember previous parts "
                   "of our conversation and use that context to provide relevant responses."),
        # MessagesPlaceholder injects the chat history from memory
        # The variable name must match the memory_key in ConversationBufferMemory
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),  # Current user input
    ])
    return prompt


def build_memory_chain(memory: ConversationBufferMemory) -> Any:
    """
    Build an LCEL chain with integrated memory.
    
    This function demonstrates memory integration with LCEL composition:
    1. Load memory variables using RunnablePassthrough
    2. Pass through the original input alongside memory
    3. Apply the prompt template (merges input + chat_history)
    4. Invoke the LLM
    5. Parse the output to string
    
    Type Flow:
    Dict[str, str] → (+ memory) → Dict[str, Any] → ChatPromptTemplate 
    → List[BaseMessage] → ChatOpenAI → AIMessage → StrOutputParser → str
    
    Args:
        memory: ConversationBufferMemory instance to integrate
    
    Returns:
        Runnable chain that maintains conversation context via memory
    
    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    
    Source: libs/core/langchain_core/runnables/base.py
    """
    # Validate API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment:\n"
            "  export OPENAI_API_KEY='your-api-key-here'\n"
            "Or copy .env.example to .env and add your key."
        )
    
    # Initialize components
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,  # Slightly creative for natural conversation
    )
    prompt = create_conversational_prompt()
    output_parser = StrOutputParser()
    
    # Build LCEL chain with memory integration
    # RunnablePassthrough.assign() loads memory variables and adds them to the input dict
    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: memory.load_memory_variables(x)["chat_history"]
        )
        | prompt
        | llm
        | output_parser
    )
    
    return chain


def demonstrate_automatic_memory() -> None:
    """
    Demonstrate automatic memory management in a conversation loop.
    
    This example shows how memory automatically persists context across
    multiple conversation turns. The chain remembers previous exchanges
    and can reference them in responses.
    
    Memory Lifecycle (Automatic):
    1. User provides input
    2. Chain loads chat_history from memory
    3. Chain invokes with input + history
    4. Chain produces response
    5. We manually save the exchange to memory with save_context()
    
    Source: libs/langchain/tests/unit_tests/memory/test_combined_memory.py:23-32
    """
    print("=" * 70)
    print("DEMONSTRATION 1: Automatic Memory Management")
    print("=" * 70)
    print("This example shows a multi-turn conversation with memory.\n")
    
    try:
        # Initialize memory and chain
        memory = setup_memory()
        chain = build_memory_chain(memory)
        
        # Conversation turns demonstrating memory
        conversations = [
            "Hello! My name is Alice.",
            "What is my name?",
            "I work as a software engineer. What do I do for a living?",
        ]
        
        for i, user_input in enumerate(conversations, 1):
            print(f"\nTurn {i}")
            print(f"Human: {user_input}")
            
            # Invoke chain with user input
            # The chain automatically loads chat_history from memory
            response = chain.invoke({"input": user_input})
            
            print(f"AI: {response}")
            
            # Save this exchange to memory for future turns
            # This is the manual step required after each interaction
            memory.save_context(
                {"input": user_input},
                {"output": response}
            )
        
        print("\n" + "-" * 70)
        print("Notice how the AI remembered:")
        print("- Turn 2: The AI recalled the name 'Alice' from Turn 1")
        print("- Turn 3: The AI recalled the profession from earlier in Turn 3")
        print("-" * 70)
        
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("Please ensure OPENAI_API_KEY is set and try again.")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        print("Please check your configuration and try again.")


def demonstrate_manual_memory_management() -> None:
    """
    Demonstrate manual memory inspection and manipulation.
    
    This example shows how to:
    1. Manually add conversations to memory with save_context()
    2. Inspect memory state with load_memory_variables()
    3. Clear memory with clear()
    
    These operations are useful for:
    - Debugging conversation state
    - Implementing custom memory persistence
    - Testing conversation flows
    - Resetting conversations
    
    Source: libs/langchain/tests/unit_tests/memory/test_combined_memory.py:18-32
    """
    print("\n\n" + "=" * 70)
    print("DEMONSTRATION 2: Manual Memory Management")
    print("=" * 70)
    print("This example shows how to manually inspect and manipulate memory.\n")
    
    try:
        # Initialize memory
        memory = setup_memory()
        
        # Manually add conversations to memory
        print("Step 1: Adding conversations to memory manually...")
        memory.save_context(
            {"input": "What is the capital of France?"},
            {"output": "The capital of France is Paris."}
        )
        memory.save_context(
            {"input": "What is the population of that city?"},
            {"output": "Paris has a population of approximately 2.2 million people "
                      "in the city proper, and about 12 million in the metropolitan area."}
        )
        print("✓ Added 2 conversation exchanges to memory")
        
        # Inspect memory state
        print("\nStep 2: Inspecting memory contents...")
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars["chat_history"]
        
        print(f"✓ Memory contains {len(chat_history)} messages:")
        for i, message in enumerate(chat_history, 1):
            message_type = message.__class__.__name__
            content_preview = message.content[:60] + "..." if len(message.content) > 60 else message.content
            print(f"  {i}. {message_type}: {content_preview}")
        
        # Demonstrate memory clear
        print("\nStep 3: Clearing memory...")
        memory.clear()
        print("✓ Memory cleared")
        
        # Verify memory is empty
        memory_vars_after = memory.load_memory_variables({})
        chat_history_after = memory_vars_after["chat_history"]
        print(f"✓ Memory now contains {len(chat_history_after)} messages")
        
        print("\n" + "-" * 70)
        print("Key Takeaways:")
        print("- save_context(): Manually add exchanges to memory")
        print("- load_memory_variables(): Inspect current memory state")
        print("- clear(): Reset conversation history")
        print("-" * 70)
        
    except Exception as e:
        print(f"\nError during manual memory management: {e}")


def demonstrate_memory_with_context_window() -> None:
    """
    Demonstrate memory behavior with longer conversations.
    
    Note: ConversationBufferMemory stores ALL conversation history.
    For production applications with long conversations, consider:
    - ConversationBufferWindowMemory: Keep only last K exchanges
    - ConversationSummaryMemory: Summarize old exchanges
    - ConversationSummaryBufferMemory: Hybrid approach
    
    This helps manage token limits for LLM context windows.
    
    Source: libs/langchain/tests/unit_tests/chains/test_memory.py:23-29
    """
    print("\n\n" + "=" * 70)
    print("DEMONSTRATION 3: Memory and Context Windows")
    print("=" * 70)
    print("This example discusses memory types for different use cases.\n")
    
    print("ConversationBufferMemory (used in this example):")
    print("  ✓ Stores complete conversation history")
    print("  ✓ Simple and preserves all context")
    print("  ✗ Can exceed LLM token limits in long conversations")
    print("  Use case: Short to medium conversations\n")
    
    print("Alternative Memory Types:")
    print("\n1. ConversationBufferWindowMemory:")
    print("     - Keeps only last K conversation turns")
    print("     - Example: memory_key='chat_history', k=5")
    print("     - Use case: Long conversations, recent context matters most")
    
    print("\n2. ConversationSummaryMemory:")
    print("     - Summarizes old conversation turns")
    print("     - Requires LLM for summarization")
    print("     - Use case: Very long conversations, need context compression")
    
    print("\n3. ConversationSummaryBufferMemory:")
    print("     - Keeps recent messages + summary of older ones")
    print("     - Best of both approaches")
    print("     - Use case: Production applications with variable conversation lengths")
    
    print("\n" + "-" * 70)
    print("Production Recommendation:")
    print("- Start with ConversationBufferMemory for prototyping")
    print("- Monitor token usage in production")
    print("- Switch to windowed or summary memory if hitting limits")
    print("-" * 70)


def main() -> None:
    """
    Main execution function demonstrating all memory patterns.
    
    This function runs three demonstrations:
    1. Automatic memory management in conversation loops
    2. Manual memory inspection and manipulation
    3. Discussion of memory types and context windows
    
    Prerequisites:
    - OPENAI_API_KEY environment variable set
    - Dependencies installed: langchain-core, langchain-classic, langchain-openai
    
    Example:
        $ export OPENAI_API_KEY='your-key-here'
        $ python memory_enabled_chain.py
    """
    print("\n" + "=" * 70)
    print("MEMORY-ENABLED CONVERSATIONAL CHAIN EXAMPLE")
    print("=" * 70)
    print("\nThis example demonstrates how to build conversational chains")
    print("with memory using LangChain's ConversationBufferMemory.")
    print("\nMemory enables the chain to:")
    print("  • Remember previous exchanges in the conversation")
    print("  • Reference earlier context in responses")
    print("  • Maintain state across multiple interactions")
    
    # Run demonstrations
    demonstrate_automatic_memory()
    demonstrate_manual_memory_management()
    demonstrate_memory_with_context_window()
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nNext Steps:")
    print("  • Modify the conversation examples to test different scenarios")
    print("  • Try ConversationBufferWindowMemory for limited history")
    print("  • Integrate memory with your own custom chains")
    print("  • Explore memory persistence to databases for production use")
    print("\nFor more examples, see:")
    print("  • examples/basic_chains/ - Basic chain patterns")
    print("  • examples/advanced_chains/agent_with_tools.py - Agent memory")
    print("  • docs/guides/memory-integration.md - Comprehensive memory guide")
    print()


if __name__ == "__main__":
    """
    Standalone execution block.
    
    This allows the example to be run directly:
        python memory_enabled_chain.py
    
    Or imported as a module for testing:
        from memory_enabled_chain import setup_memory, build_memory_chain
    """
    main()



