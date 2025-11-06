"""Agent with Custom Tools Example.

This example demonstrates comprehensive agent implementation with custom tool
integration, showing the complete agent lifecycle including tool definition,
validation, execution, and error handling.

Features demonstrated:
- Custom tool creation using @tool decorator
- StructuredTool with complex Pydantic validation
- Agent initialization and configuration
- Agent decision loop: observation → thought → action → observation
- Tool execution error handling
- Type-safe tool interfaces

Source: Agent Action Plan section 0.5.1
"""

import json
import os
from typing import Optional

from pydantic import BaseModel, Field


# Conditional imports with graceful degradation for missing API keys
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: langchain-openai not installed. Install with: "
          "pip install langchain-openai")

try:
    from langchain_core.tools import tool, StructuredTool
    from langchain_core.messages import HumanMessage
    from langchain_classic.agents import (
        AgentExecutor,
        create_tool_calling_agent,
    )
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"Error: Required LangChain packages not available: {e}")
    print("Install with: pip install langchain-core langchain-classic")


# =============================================================================
# TOOL DEFINITIONS - Using @tool decorator for simple tools
# =============================================================================

@tool
def calculator(operation: str, x: float, y: float) -> str:
    """Perform basic arithmetic operations.
    
    This tool demonstrates simple tool creation using the @tool decorator
    with type annotations for automatic input validation.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        x: First number
        y: Second number
    
    Returns:
        Result of the calculation as a string
        
    Raises:
        ValueError: If operation is not recognized or division by zero
    """
    operations = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else None,
    }
    
    if operation not in operations:
        raise ValueError(
            f"Unknown operation: {operation}. "
            f"Valid operations: {', '.join(operations.keys())}"
        )
    
    result = operations[operation](x, y)
    
    if result is None:
        raise ValueError("Cannot divide by zero")
    
    return f"{x} {operation} {y} = {result}"


@tool
def web_search_simulator(query: str) -> str:
    """Simulate a web search (mock tool for demonstration).
    
    In production, this would integrate with a real search API.
    This demonstrates tool output formatting for agent consumption.
    
    Args:
        query: Search query string
        
    Returns:
        Simulated search results as formatted string
    """
    # Mock search results based on common queries
    mock_results = {
        "weather": "Current weather: Sunny, 72°F with light clouds",
        "langchain": "LangChain is a framework for developing applications "
                    "powered by large language models",
        "python": "Python is a high-level programming language known for "
                 "readability and versatility",
    }
    
    # Simple keyword matching for demo purposes
    for keyword, result in mock_results.items():
        if keyword.lower() in query.lower():
            return f"Search results for '{query}':\n{result}"
    
    return f"Search results for '{query}':\nNo specific results found. " \
           f"This is a simulated search tool."


# =============================================================================
# COMPLEX TOOL WITH PYDANTIC VALIDATION - Using StructuredTool
# =============================================================================

class DataLookupInput(BaseModel):
    """Input schema for data lookup tool with validation.
    
    This demonstrates complex tool input validation using Pydantic models,
    ensuring type safety and providing clear error messages.
    """
    
    dataset: str = Field(
        description="Name of the dataset to query (users, products, orders)"
    )
    record_id: int = Field(
        description="ID of the record to retrieve",
        ge=1,  # Greater than or equal to 1
    )
    include_metadata: bool = Field(
        default=False,
        description="Whether to include metadata in the response"
    )


def data_lookup_function(
    dataset: str,
    record_id: int,
    include_metadata: bool = False
) -> str:
    """Look up data from simulated database.
    
    This function demonstrates the implementation of a complex tool
    with multiple parameters and validation logic.
    
    Args:
        dataset: Dataset name (users, products, orders)
        record_id: Record identifier
        include_metadata: Whether to include metadata
        
    Returns:
        JSON-formatted record data
        
    Raises:
        ValueError: If dataset is not recognized or record not found
    """
    # Mock database
    mock_data = {
        "users": {
            1: {"name": "Alice Johnson", "email": "alice@example.com"},
            2: {"name": "Bob Smith", "email": "bob@example.com"},
        },
        "products": {
            1: {"name": "Widget", "price": 29.99, "stock": 100},
            2: {"name": "Gadget", "price": 49.99, "stock": 50},
        },
        "orders": {
            1: {"user_id": 1, "product_id": 1, "quantity": 2, "total": 59.98},
            2: {"user_id": 2, "product_id": 2, "quantity": 1, "total": 49.99},
        },
    }
    
    # Validate dataset
    if dataset not in mock_data:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available datasets: {', '.join(mock_data.keys())}"
        )
    
    # Retrieve record
    dataset_records = mock_data[dataset]
    if record_id not in dataset_records:
        raise ValueError(
            f"Record {record_id} not found in dataset '{dataset}'. "
            f"Available IDs: {', '.join(map(str, dataset_records.keys()))}"
        )
    
    record = dataset_records[record_id].copy()
    
    # Add metadata if requested
    if include_metadata:
        record["_metadata"] = {
            "dataset": dataset,
            "record_id": record_id,
            "timestamp": "2024-01-15T10:30:00Z",
        }
    
    return json.dumps(record, indent=2)


# Create StructuredTool with explicit schema
data_lookup_tool = StructuredTool.from_function(
    func=data_lookup_function,
    name="data_lookup",
    description="Look up records from datasets (users, products, orders). "
                "Use this tool when you need to retrieve specific data by ID.",
    args_schema=DataLookupInput,
)


# =============================================================================
# AGENT CONFIGURATION AND EXECUTION
# =============================================================================

def create_agent_with_tools() -> Optional[AgentExecutor]:
    """Create and configure an agent with custom tools.
    
    This function demonstrates the complete agent setup process including:
    - Tool registration
    - LLM configuration
    - Prompt template setup
    - AgentExecutor creation with error handling
    
    Returns:
        Configured AgentExecutor or None if setup fails
    """
    if not LANGCHAIN_AVAILABLE:
        print("Error: LangChain packages not available")
        return None
    
    if not OPENAI_AVAILABLE:
        print("Error: OpenAI API key not found. Set OPENAI_API_KEY environment "
              "variable.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return None
    
    # Initialize chat model with tool calling capability
    # Tool calling is required for the agent to use structured tools
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,  # Deterministic responses for consistent tool usage
    )
    
    # Gather all tools in a list
    tools = [
        calculator,
        web_search_simulator,
        data_lookup_tool,
    ]
    
    # Define agent prompt template
    # MessagesPlaceholder allows agent to maintain conversation history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to various tools. "
                  "Use the tools when needed to answer questions accurately. "
                  "Think step by step about which tool to use."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    try:
        # Create the agent using tool calling pattern
        # This agent will automatically format tool calls for the LLM
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Create AgentExecutor with configuration
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # Enable detailed logging of agent reasoning
            handle_parsing_errors=True,  # Gracefully handle malformed tool calls
            max_iterations=5,  # Prevent infinite loops
            max_execution_time=60,  # Timeout after 60 seconds
        )
        
        print("✓ Agent successfully configured with tools:")
        for tool_item in tools:
            tool_name = getattr(tool_item, 'name', str(tool_item))
            print(f"  - {tool_name}")
        print()
        
        return agent_executor
        
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None


def demonstrate_agent_with_tools(agent_executor: AgentExecutor) -> None:
    """Demonstrate agent execution with various tool usage scenarios.
    
    This function shows:
    - Single tool usage
    - Multi-step reasoning requiring multiple tools
    - Error handling for invalid tool inputs
    - Agent decision loop visualization
    
    Args:
        agent_executor: Configured AgentExecutor instance
    """
    print("=" * 80)
    print("AGENT WITH TOOLS DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Example 1: Simple calculation using calculator tool
    print("Example 1: Basic Calculator Usage")
    print("-" * 80)
    try:
        result = agent_executor.invoke({
            "input": "What is 234 multiplied by 67?"
        })
        print(f"\nFinal Answer: {result['output']}")
        print()
    except Exception as e:
        print(f"Error in Example 1: {e}")
        print()
    
    # Example 2: Information retrieval using search simulator
    print("Example 2: Web Search Simulation")
    print("-" * 80)
    try:
        result = agent_executor.invoke({
            "input": "Search for information about LangChain"
        })
        print(f"\nFinal Answer: {result['output']}")
        print()
    except Exception as e:
        print(f"Error in Example 2: {e}")
        print()
    
    # Example 3: Data lookup with complex parameters
    print("Example 3: Structured Data Lookup")
    print("-" * 80)
    try:
        result = agent_executor.invoke({
            "input": "Look up user with ID 1 from the users dataset and "
                    "include metadata"
        })
        print(f"\nFinal Answer: {result['output']}")
        print()
    except Exception as e:
        print(f"Error in Example 3: {e}")
        print()
    
    # Example 4: Multi-step reasoning requiring multiple tools
    print("Example 4: Multi-Step Reasoning")
    print("-" * 80)
    try:
        result = agent_executor.invoke({
            "input": "First, calculate 15 + 25. Then search for information "
                    "about Python."
        })
        print(f"\nFinal Answer: {result['output']}")
        print()
    except Exception as e:
        print(f"Error in Example 4: {e}")
        print()
    
    # Example 5: Error handling - invalid tool input
    print("Example 5: Error Handling (Division by Zero)")
    print("-" * 80)
    try:
        result = agent_executor.invoke({
            "input": "What is 10 divided by 0?"
        })
        print(f"\nFinal Answer: {result['output']}")
        print()
    except Exception as e:
        print(f"Error in Example 5: {e}")
        print()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


def explain_agent_lifecycle() -> None:
    """Explain the agent decision loop and tool execution lifecycle.
    
    This function provides educational context about how agents work,
    documenting the key phases per Agent Action Plan section 0.3.1.
    """
    print("\n")
    print("=" * 80)
    print("AGENT LIFECYCLE EXPLANATION")
    print("=" * 80)
    print("""
The agent follows this decision loop for each query:

1. OBSERVATION: Agent receives the user's input/question
   
2. THOUGHT: Agent analyzes what information or tools are needed
   - Reviews available tools and their descriptions
   - Determines which tool(s) are most relevant
   
3. ACTION: Agent decides to use a specific tool
   - Formats the tool call with appropriate parameters
   - Executes the tool function
   
4. OBSERVATION: Agent receives the tool's output
   - Reviews the result from the tool
   - Determines if more information is needed
   
5. THOUGHT: Agent decides next steps
   - If sufficient information: Formulate final answer
   - If more needed: Return to step 3 with different tool
   
6. FINAL ANSWER: Agent provides the complete response

Tool Interface Contract (per Agent Action Plan section 0.3.1):
- name: Unique identifier for the tool
- description: Clear explanation of tool's purpose (agent uses this!)
- args_schema: Pydantic model defining input parameters with validation
- func: The actual function implementation

Error Handling:
- Tool execution errors are caught and logged
- Agent can recover from errors and try alternative approaches
- Max iterations prevent infinite loops
- Timeout prevents hung executions
""")
    print("=" * 80)
    print()


def main() -> None:
    """Main execution function demonstrating agent with tools.
    
    This is the entry point for the example, following the pattern
    required by Agent Action Plan section 0.9.2.
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "AGENT WITH TOOLS EXAMPLE" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Check prerequisites
    if not LANGCHAIN_AVAILABLE:
        print("ERROR: Required packages not installed.")
        print("Install with: pip install langchain-core langchain-classic")
        return
    
    # Explain the agent lifecycle first
    explain_agent_lifecycle()
    
    # Create the agent
    print("Setting up agent with custom tools...")
    print()
    agent_executor = create_agent_with_tools()
    
    if agent_executor is None:
        print("\nFailed to create agent. Please check:")
        print("1. All required packages are installed")
        print("2. OPENAI_API_KEY environment variable is set")
        print("\nTo set API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Run demonstrations
    demonstrate_agent_with_tools(agent_executor)
    
    print("\nExample completed successfully!")
    print("\nKey Takeaways:")
    print("• Tools are defined with clear interfaces using @tool decorator")
    print("• Complex tools use Pydantic models for input validation")
    print("• Agent automatically decides when and how to use tools")
    print("• Error handling ensures robust execution")
    print("• Verbose mode shows the agent's reasoning process")


if __name__ == "__main__":
    main()
    
    
    
