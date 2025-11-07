"""
LCEL (LangChain Expression Language) Basic Composition Example

This example demonstrates the fundamental concepts of LCEL (LangChain Expression Language),
which is the modern, recommended way to compose LangChain components into chains.

Key Concepts:
-------------
- LCEL: LangChain Expression Language - a declarative syntax for building chains
- Pipe Operator (|): Connects Runnable components to create sequences
- Runnable Protocol: All LCEL components implement Runnable[InputType, OutputType]
- Type Safety: Each stage in the pipe has clearly defined input/output types
- Automatic Support: Chains built with LCEL automatically get sync, async, batch, 
  and streaming support

Type Flow in LCEL:
------------------
When you compose components with the pipe operator (|), data flows through 
transformations:

  Dict[str, str] → ChatPromptTemplate → ChatPromptValue (List[BaseMessage]) 
    → ChatOpenAI → AIMessage → StrOutputParser → str

Each component is a Runnable that transforms its input type to its output type.

Runnable Protocol Methods:
---------------------------
- invoke(input): Transform a single input into an output (synchronous)
- ainvoke(input): Transform a single input into an output (asynchronous)
- batch(inputs): Efficiently transform multiple inputs into outputs
- abatch(inputs): Async version of batch
- stream(input): Stream output from a single input as it's produced
- astream(input): Async version of stream

Source References:
------------------
- Runnable protocol: libs/core/langchain_core/runnables/base.py
- LCEL composition: libs/core/langchain_core/runnables/base.py:608-627 (__or__ method)
- Type flow examples: Agent Action Plan section 0.3.1
"""

import os
import sys
from typing import Dict, Any, List

# LangChain imports for LCEL composition
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Environment and utilities
from dotenv import load_dotenv


def setup_environment() -> str:
    """
    Load environment variables and validate API key availability.
    
    Returns:
        str: The OpenAI API key from environment
    
    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment
    """
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables.\n"
            "Please create a .env file in examples/basic_chains/ with:\n"
            "  OPENAI_API_KEY=your-api-key-here\n"
            "Get your API key from: https://platform.openai.com/api-keys"
        )
    
    return api_key


def demonstrate_basic_lcel_chain():
    """
    Demonstrate basic LCEL chain composition using the pipe operator (|).
    
    This function shows:
    1. Creating individual LCEL components (prompt, model, parser)
    2. Composing them with the pipe operator
    3. Understanding type transformations at each stage
    4. Invoking the composed chain
    """
    print("\n" + "="*80)
    print("BASIC LCEL CHAIN COMPOSITION")
    print("="*80)
    
    # Stage 1: Create ChatPromptTemplate
    # ----------------------------------
    # Input Type: Dict[str, str] with "topic" key
    # Output Type: ChatPromptValue (which wraps List[BaseMessage])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains technical concepts."),
        ("human", "Explain {topic} in simple terms.")
    ])
    
    print("\n✓ Created ChatPromptTemplate")
    print(f"  Input variables: {prompt.input_variables}")
    print(f"  Template messages: {len(prompt.messages)} messages (system + human)")
    
    # Stage 2: Create ChatOpenAI model
    # --------------------------------
    # Input Type: ChatPromptValue (List[BaseMessage])
    # Output Type: AIMessage (contains the LLM's response)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    print("\n✓ Created ChatOpenAI model")
    print(f"  Model: {model.model_name}")
    print(f"  Temperature: {model.temperature}")
    
    # Stage 3: Create StrOutputParser
    # -------------------------------
    # Input Type: AIMessage
    # Output Type: str (extracts the text content from AIMessage)
    parser = StrOutputParser()
    
    print("\n✓ Created StrOutputParser")
    print("  Extracts string content from AIMessage")
    
    # Stage 4: Compose the chain with LCEL pipe operator (|)
    # -------------------------------------------------------
    # The pipe operator creates a RunnableSequence that:
    # 1. Takes the output of the left operand
    # 2. Passes it as input to the right operand
    # 3. Returns a new Runnable with combined behavior
    #
    # Type Flow:
    #   Input: Dict[str, str]
    #   → prompt: Dict → ChatPromptValue
    #   → model: ChatPromptValue → AIMessage
    #   → parser: AIMessage → str
    #   Final Output: str
    chain = prompt | model | parser
    
    print("\n✓ Composed LCEL chain: prompt | model | parser")
    print("\n  Type Flow Diagram:")
    print("  ┌─────────────────┐")
    print("  │ Dict[str, str]  │  Input: {'topic': '...'}") 
    print("  └────────┬────────┘")
    print("           │ prompt")
    print("           ↓")
    print("  ┌─────────────────┐")
    print("  │ ChatPromptValue │  List[SystemMessage, HumanMessage]")
    print("  └────────┬────────┘")
    print("           │ model")
    print("           ↓")
    print("  ┌─────────────────┐")
    print("  │   AIMessage     │  LLM response with metadata")
    print("  └────────┬────────┘")
    print("           │ parser")
    print("           ↓")
    print("  ┌─────────────────┐")
    print("  │      str        │  Final output: extracted text")
    print("  └─────────────────┘")
    
    # Stage 5: Invoke the chain
    # -------------------------
    # The invoke() method is the synchronous execution method for Runnables
    print("\n" + "-"*80)
    print("INVOKING THE CHAIN")
    print("-"*80)
    
    try:
        input_data = {"topic": "machine learning"}
        print(f"\nInput: {input_data}")
        
        result = chain.invoke(input_data)
        
        print(f"\nOutput (type={type(result).__name__}):")
        print(f"{result}")
        
    except Exception as e:
        print(f"\n✗ Error invoking chain: {e}")
        print("  Check your API key and network connection")


def demonstrate_batch_invocation():
    """
    Demonstrate batch invocation for processing multiple inputs efficiently.
    
    The batch() method processes multiple inputs in parallel when possible,
    which is more efficient than calling invoke() multiple times.
    """
    print("\n" + "="*80)
    print("BATCH INVOCATION")
    print("="*80)
    
    # Create the same chain as before
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a concise assistant. Answer in one sentence."),
            ("human", "What is {topic}?")
        ])
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
        | StrOutputParser()
    )
    
    print("\n✓ Created chain for batch processing")
    
    try:
        # Batch accepts a list of inputs
        # Each input must match the chain's expected input type (Dict[str, str])
        inputs = [
            {"topic": "Python"},
            {"topic": "LCEL"},
            {"topic": "Runnables"}
        ]
        
        print(f"\nProcessing {len(inputs)} inputs in batch...")
        print(f"Inputs: {[inp['topic'] for inp in inputs]}")
        
        # batch() returns a list of outputs, one for each input
        results = chain.batch(inputs)
        
        print(f"\nResults (type={type(results).__name__}, length={len(results)}):")
        for i, (inp, result) in enumerate(zip(inputs, results), 1):
            print(f"\n  {i}. Topic: {inp['topic']}")
            print(f"     Answer: {result}")
            
    except Exception as e:
        print(f"\n✗ Error in batch invocation: {e}")


def demonstrate_streaming():
    """
    Demonstrate streaming output for token-by-token generation.
    
    The stream() method yields output as it's produced, which is useful for:
    - Displaying partial results to users
    - Processing large outputs incrementally
    - Providing immediate feedback
    """
    print("\n" + "="*80)
    print("STREAMING OUTPUT")
    print("="*80)
    
    # Create chain with streaming enabled
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a storyteller. Tell a very short story."),
            ("human", "Tell a story about {subject}.")
        ])
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, streaming=True)
        | StrOutputParser()
    )
    
    print("\n✓ Created streaming chain")
    
    try:
        input_data = {"subject": "a curious robot"}
        print(f"\nInput: {input_data}")
        print("\nStreaming output (token-by-token):")
        print("-" * 60)
        
        # stream() returns an iterator that yields chunks as they arrive
        # For StrOutputParser, each chunk is a string token
        for chunk in chain.stream(input_data):
            print(chunk, end="", flush=True)
        
        print("\n" + "-" * 60)
        print("✓ Streaming complete")
        
    except Exception as e:
        print(f"\n✗ Error in streaming: {e}")


def demonstrate_runnable_passthrough():
    """
    Demonstrate RunnablePassthrough for passing data through chains unchanged.
    
    RunnablePassthrough is useful for:
    - Preserving original input alongside transformations
    - Adding computed fields to input dictionaries
    - Creating parallel branches in chains
    """
    print("\n" + "="*80)
    print("RUNNABLEPASSTHROUGH")
    print("="*80)
    
    print("\n✓ RunnablePassthrough passes input through unchanged")
    print("  Useful for preserving original data in chain pipelines")
    
    # Example 1: Simple passthrough
    # -----------------------------
    passthrough = RunnablePassthrough()
    
    try:
        test_input = {"message": "Hello, LCEL!"}
        result = passthrough.invoke(test_input)
        
        print(f"\nExample 1 - Simple Passthrough:")
        print(f"  Input:  {test_input}")
        print(f"  Output: {result}")
        print(f"  → Input and output are identical")
        
    except Exception as e:
        print(f"\n✗ Error in passthrough: {e}")
    
    # Example 2: RunnablePassthrough.assign() to add fields
    # -----------------------------------------------------
    # assign() creates a new Runnable that:
    # 1. Passes through all input fields
    # 2. Adds new computed fields based on the input
    
    print(f"\n✓ RunnablePassthrough.assign() adds computed fields")
    
    # Create a simple chain that adds a generated field
    chain_with_assign = (
        RunnablePassthrough.assign(
            # Add a 'summary' field by calling a chain on the 'text' field
            summary=(
                ChatPromptTemplate.from_template("Summarize in 5 words: {text}")
                | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                | StrOutputParser()
            )
        )
    )
    
    try:
        test_input_2 = {"text": "LangChain Expression Language enables composable AI"}
        
        print(f"\nExample 2 - Assign with transformation:")
        print(f"  Input:  {test_input_2}")
        
        result_2 = chain_with_assign.invoke(test_input_2)
        
        print(f"  Output: {result_2}")
        print(f"  → Original 'text' field preserved")
        print(f"  → New 'summary' field added via chain execution")
        
    except Exception as e:
        print(f"\n✗ Error in assign: {e}")


def demonstrate_chain_introspection():
    """
    Demonstrate chain introspection to understand chain structure and schemas.
    
    Runnables expose information about:
    - Input schema: What type of input does the chain expect?
    - Output schema: What type of output does the chain produce?
    - Chain structure: How are components connected?
    """
    print("\n" + "="*80)
    print("CHAIN INTROSPECTION")
    print("="*80)
    
    # Create a sample chain
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{query}")
        ])
        | ChatOpenAI(model="gpt-3.5-turbo")
        | StrOutputParser()
    )
    
    print("\n✓ Created chain for introspection")
    
    # Inspect input schema
    # -------------------
    # The input_schema property shows what structure the chain expects
    print("\nInput Schema:")
    try:
        input_schema = chain.input_schema.model_json_schema()
        print(f"  Type: {input_schema.get('type', 'unknown')}")
        if 'properties' in input_schema:
            print(f"  Required fields: {input_schema.get('required', [])}")
            print(f"  Properties:")
            for prop_name, prop_info in input_schema['properties'].items():
                prop_type = prop_info.get('type', 'unknown')
                print(f"    - {prop_name}: {prop_type}")
    except Exception as e:
        print(f"  (Schema inspection not available: {e})")
    
    # Inspect output schema
    # --------------------
    # The output_schema property shows what structure the chain produces
    print("\nOutput Schema:")
    try:
        output_schema = chain.output_schema.model_json_schema()
        print(f"  Type: {output_schema.get('type', 'unknown')}")
        if 'description' in output_schema:
            print(f"  Description: {output_schema['description']}")
    except Exception as e:
        print(f"  (Schema inspection not available: {e})")
    
    # Inspect chain structure
    # ----------------------
    # Understanding how components are connected
    print("\nChain Structure:")
    print(f"  Type: {type(chain).__name__}")
    print(f"  Components: prompt | model | parser")
    print(f"  Execution flow: Sequential (left to right)")


def demonstrate_error_handling():
    """
    Demonstrate error handling in LCEL chains.
    
    Common error scenarios:
    - Missing required input fields
    - Invalid input types
    - API errors (rate limits, auth failures)
    - Network errors
    """
    print("\n" + "="*80)
    print("ERROR HANDLING")
    print("="*80)
    
    chain = (
        ChatPromptTemplate.from_template("Translate to French: {text}")
        | ChatOpenAI(model="gpt-3.5-turbo")
        | StrOutputParser()
    )
    
    print("\n✓ Created chain for error handling demonstration")
    
    # Error Case 1: Missing required field
    # -----------------------------------
    print("\nError Case 1: Missing required input field")
    try:
        # Chain expects {"text": "..."} but we provide empty dict
        result = chain.invoke({})
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  ✗ Expected error caught: {type(e).__name__}")
        print(f"  Message: {str(e)[:100]}...")
        print(f"  → Ensure all required input variables are provided")
    
    # Error Case 2: Wrong input type
    # -----------------------------
    print("\nError Case 2: Wrong input type (string instead of dict)")
    try:
        # Chain expects dict but we provide string
        result = chain.invoke("Hello")  # type: ignore
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  ✗ Expected error caught: {type(e).__name__}")
        print(f"  Message: {str(e)[:100]}...")
        print(f"  → Provide input in the correct format (dict for ChatPromptTemplate)")
    
    # Success Case: Proper invocation
    # ------------------------------
    print("\nSuccess Case: Proper invocation with correct input")
    try:
        result = chain.invoke({"text": "Hello, world!"})
        print(f"  ✓ Success! Result: {result}")
    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    """
    Main execution function demonstrating all LCEL composition patterns.
    
    This example covers:
    1. Basic LCEL chain composition (prompt | model | parser)
    2. Type flow understanding through chain stages
    3. Runnable protocol methods (invoke, batch, stream)
    4. RunnablePassthrough for data preservation
    5. Chain introspection capabilities
    6. Error handling patterns
    
    Expected Output:
    ---------------
    - Successful chain execution showing type transformations
    - Batch processing results for multiple inputs
    - Streaming token generation
    - Passthrough and assign operations
    - Schema information for input/output types
    - Error handling demonstrations
    
    Prerequisites:
    -------------
    - OPENAI_API_KEY environment variable set
    - Internet connection for API calls
    - Valid OpenAI API credits
    """
    print("\n" + "="*80)
    print("LCEL BASIC COMPOSITION - COMPREHENSIVE EXAMPLE")
    print("="*80)
    print("\nThis example demonstrates the fundamental concepts of LCEL")
    print("(LangChain Expression Language) for building composable AI chains.")
    
    try:
        # Setup environment and validate API key
        setup_environment()
        print("\n✓ Environment setup successful")
        print("  API key loaded from environment")
        
        # Demonstrate each LCEL pattern
        demonstrate_basic_lcel_chain()
        demonstrate_batch_invocation()
        demonstrate_streaming()
        demonstrate_runnable_passthrough()
        demonstrate_chain_introspection()
        demonstrate_error_handling()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("\nKey Takeaways:")
        print("  1. LCEL uses the pipe operator (|) to compose Runnable components")
        print("  2. Each stage transforms input to output with clear type flow")
        print("  3. Chains automatically support invoke, batch, and stream methods")
        print("  4. RunnablePassthrough preserves data through transformations")
        print("  5. Chain introspection reveals input/output schemas")
        print("  6. Proper error handling ensures robust chain execution")
        print("\nType Flow: Dict[str,str] → ChatPromptValue → AIMessage → str")
        print("\nFor more examples, see:")
        print("  - prompt_template_chain.py: Advanced prompt templates")
        print("  - sequential_chain.py: Multi-step chain composition")
        print("  - ../advanced_chains/: Production-ready patterns")
        print("\n✓ All demonstrations completed successfully!")
        
    except ValueError as e:
        print(f"\n✗ Configuration Error: {e}")
        print("\nPlease ensure:")
        print("  1. You have created a .env file in examples/basic_chains/")
        print("  2. The .env file contains: OPENAI_API_KEY=your-key-here")
        print("  3. Your API key is valid and has available credits")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Unexpected Error: {type(e).__name__}")
        print(f"  Message: {e}")
        print("\nFor troubleshooting, check:")
        print("  - Network connection")
        print("  - API key validity")
        print("  - Python version (requires >=3.10)")
        print("  - Dependencies installed (pip install -r requirements.txt)")
        sys.exit(1)


if __name__ == "__main__":
    main()
