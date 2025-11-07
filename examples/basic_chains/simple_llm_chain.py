"""Simple LLMChain Example - Basic LangChain Usage Pattern

This example demonstrates the basic LLMChain pattern for combining a prompt template
with a language model. LLMChain provides a simple wrapper around prompt templates
and LLMs, handling the formatting and invocation.

**DEPRECATION NOTICE**: LLMChain is deprecated since LangChain 0.1.17 and will be
removed in version 1.0. This example is provided for educational purposes and to
help developers understand the migration to modern LCEL (LangChain Expression
Language) syntax.

**Modern Alternative**: Use LCEL composition with the pipe operator (|) instead:
    chain = prompt | model | StrOutputParser()

This example is designed for developers new to LangChain, demonstrating:
- Basic prompt template creation with input variables
- OpenAI model initialization
- Chain composition using LLMChain (deprecated pattern)
- Chain invocation with input data
- Comparison with modern LCEL approach

Prerequisites:
- Python >=3.10
- OpenAI API key set in .env file
- Dependencies installed from requirements.txt

Source: examples/basic_chains/simple_llm_chain.py
Based on: libs/langchain/langchain_classic/chains/llm.py
"""

import os
import sys
from typing import Dict, Any

# LangChain core imports for prompt templates
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# OpenAI integration for LLM access
from langchain_openai import OpenAI

# Classic LLMChain (deprecated but shown for educational purposes)
from langchain_classic.chains import LLMChain

# Environment variable management
from dotenv import load_dotenv


def setup_environment() -> str:
    """Set up environment and validate required API keys.
    
    Loads environment variables from .env file and validates that the
    required OPENAI_API_KEY is present.
    
    Returns:
        str: The OpenAI API key from environment variables.
        
    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment.
        
    Example:
        >>> api_key = setup_environment()
        >>> print(f"API key loaded: {api_key[:8]}...")
        API key loaded: sk-proj-...
    """
    # Load environment variables from .env file in current directory
    # This looks for a .env file and loads all KEY=value pairs into os.environ
    load_dotenv()
    
    # Retrieve the OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Validate that the API key exists
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables.\n"
            "Please create a .env file with your OpenAI API key:\n"
            "  1. Copy .env.example to .env\n"
            "  2. Add your OpenAI API key: OPENAI_API_KEY=sk-...\n"
            "  3. Get your API key from https://platform.openai.com/api-keys"
        )
    
    return api_key


def demonstrate_llm_chain() -> None:
    """Demonstrate basic LLMChain usage (deprecated pattern).
    
    This function shows the classic LLMChain pattern for combining a prompt
    template with an LLM. While this pattern is deprecated, understanding it
    helps when migrating legacy code or reading older LangChain examples.
    
    Type Flow (LLMChain pattern):
        1. Input: Dict[str, Any] with keys matching prompt variables
        2. PromptTemplate formats the input into a prompt string
        3. LLM processes the prompt string
        4. Output: Dict[str, str] with "text" key containing the result
    
    Raises:
        ValueError: If API key is missing or invalid.
        Exception: If API call fails (rate limits, network issues, etc.).
        
    Example:
        >>> demonstrate_llm_chain()
        
        === Using LLMChain (Deprecated Pattern) ===
        Input: {'adjective': 'funny'}
        Output: Why did the chicken cross the road? To get to the other side!
    """
    try:
        # Step 1: Set up environment and validate API key
        print("\n" + "="*70)
        print("Simple LLMChain Example - Basic LangChain Pattern")
        print("="*70 + "\n")
        
        api_key = setup_environment()
        print(f"✓ Environment configured (API key: {api_key[:8]}...)")
        
        # Step 2: Create a prompt template with input variables
        # The template string uses {variable} syntax for placeholders
        # input_variables must match the placeholders in the template
        prompt_template = "Tell me a {adjective} joke"
        prompt = PromptTemplate(
            input_variables=["adjective"],  # List of variable names to substitute
            template=prompt_template         # Template string with {variable} syntax
        )
        print(f"✓ Prompt template created: '{prompt_template}'")
        
        # Step 3: Initialize the OpenAI language model
        # The model processes prompts and generates text completions
        # Default model is text-davinci-003 (can be changed via model_name parameter)
        model = OpenAI(
            temperature=0.7,  # Controls randomness (0=deterministic, 1=creative)
            openai_api_key=api_key
        )
        print(f"✓ OpenAI model initialized (temperature=0.7)")
        
        # Step 4: Create LLMChain combining prompt and model
        # DEPRECATION NOTE: LLMChain is deprecated since LangChain 0.1.17
        # This pattern wraps the prompt and LLM for convenient invocation
        # Migration path: Use LCEL syntax (prompt | model) instead
        chain = LLMChain(
            llm=model,      # The language model to use
            prompt=prompt,  # The prompt template to format inputs
            verbose=False   # Set True to see intermediate steps
        )
        print(f"✓ LLMChain created (deprecated pattern)")
        
        # Step 5: Invoke the chain with input data
        # Input must be a dictionary with keys matching input_variables
        input_data = {"adjective": "funny"}
        print(f"\n📝 Invoking chain with input: {input_data}")
        
        result = chain.invoke(input_data)
        
        # Step 6: Display the output
        # LLMChain.invoke() returns a dict with "text" key containing the result
        print(f"\n✅ LLMChain Output:")
        print(f"   {result['text']}")
        
        # Step 7: Demonstrate the modern LCEL alternative
        print("\n" + "="*70)
        print("Modern LCEL Alternative (Recommended)")
        print("="*70 + "\n")
        
        # LCEL uses the pipe operator (|) to compose Runnable components
        # Type Flow: Dict[str, str] → PromptTemplate → str → LLM → str
        modern_chain = prompt | model | StrOutputParser()
        print("✓ Modern chain created: prompt | model | StrOutputParser()")
        
        print(f"\n📝 Invoking modern chain with input: {input_data}")
        modern_result = modern_chain.invoke(input_data)
        
        print(f"\n✅ LCEL Output:")
        print(f"   {modern_result}")
        
        # Step 8: Explain the differences
        print("\n" + "="*70)
        print("Key Differences")
        print("="*70)
        print("\nLLMChain (Deprecated):")
        print("  - Returns: Dict[str, str] with 'text' key")
        print("  - Explicit wrapper class required")
        print("  - Less flexible for composition")
        print("  - Deprecated since LangChain 0.1.17")
        
        print("\nLCEL (Modern):")
        print("  - Returns: str directly (due to StrOutputParser)")
        print("  - Uses pipe operator (|) for composition")
        print("  - More flexible and composable")
        print("  - Recommended for all new development")
        print("  - Type safe: Runnable[InputT, OutputT] protocol")
        
        print("\n" + "="*70)
        print("Example completed successfully!")
        print("="*70 + "\n")
        
    except ValueError as e:
        # Handle missing API key or configuration errors
        print(f"\n❌ Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        # Handle API call failures, network errors, rate limits, etc.
        print(f"\n❌ Execution Error: {e}", file=sys.stderr)
        print("\nCommon issues:", file=sys.stderr)
        print("  - Rate limit exceeded: Wait a few minutes or upgrade your OpenAI plan", 
              file=sys.stderr)
        print("  - Invalid API key: Check your OPENAI_API_KEY in .env file", 
              file=sys.stderr)
        print("  - Network error: Check your internet connection", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the example.
    
    Executes the LLMChain demonstration with comprehensive error handling.
    
    Expected Output:
        ✓ Environment configured
        ✓ Prompt template created
        ✓ OpenAI model initialized
        ✓ LLMChain created
        📝 Invoking chain...
        ✅ Output: [Generated joke text]
        
        [Modern LCEL alternative demonstration]
        [Comparison of patterns]
    """
    demonstrate_llm_chain()


if __name__ == "__main__":
    # Execute the example when run as a script
    # This block is not executed when importing this module
    main()
