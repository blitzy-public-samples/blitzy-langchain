"""PromptTemplate Composition with LLM using LCEL

This example demonstrates how to use PromptTemplate with multiple input variables
and compose it with a language model using modern LCEL (LangChain Expression
Language) syntax. It shows the complete workflow including:

- Creating multi-variable prompt templates with {variable} placeholders
- Partial template application for fixing specific variables
- Type-safe LCEL pipe operator composition (prompt | model | parser)
- Multiple invocation patterns (single, batch)
- Prompt inspection and debugging techniques

The example follows LCEL best practices for type-safe chain composition as
outlined in Agent Action Plan section 0.3.1 and 0.5.1.

Source: Agent Action Plan section 0.1.1 - Complete executable examples
"""

import os
import sys
from typing import Dict, List, Any

# LangChain imports for prompt templates and LCEL composition
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser

# Environment variable management
from dotenv import load_dotenv


def setup_environment() -> str:
    """Set up environment and validate API key.
    
    Loads environment variables from .env file and validates that the
    OpenAI API key is configured properly.
    
    Returns:
        str: The OpenAI API key from environment variables.
        
    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Validate API key exists
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables.\n"
            "Please create a .env file with: OPENAI_API_KEY=your-key-here\n"
            "Or set the environment variable directly."
        )
    
    return api_key


def demonstrate_basic_prompt_template() -> None:
    """Demonstrate basic PromptTemplate usage with multiple variables.
    
    Shows how to create a prompt template with multiple input variables,
    format it with different inputs, and inspect its properties.
    """
    print("\n" + "="*80)
    print("1. BASIC PROMPT TEMPLATE WITH MULTIPLE VARIABLES")
    print("="*80)
    
    # Create a PromptTemplate with multiple input variables
    # The template uses {variable} syntax for variable substitution
    template = """Write a {length} article about {topic} for {audience} audience.
The tone should be {tone} and include practical examples."""
    
    prompt = PromptTemplate(
        input_variables=["topic", "audience", "tone", "length"],
        template=template
    )
    
    # Alternative: Use from_template() which auto-detects variables
    # prompt = PromptTemplate.from_template(template)
    
    print(f"\nTemplate string:\n{template}")
    print(f"\nDetected input variables: {prompt.input_variables}")
    
    # Demonstrate prompt formatting (preview before sending to LLM)
    formatted = prompt.format(
        topic="artificial intelligence",
        audience="business executives",
        tone="professional",
        length="short"
    )
    print(f"\nFormatted prompt:\n{formatted}")


def demonstrate_partial_templates() -> PromptTemplate:
    """Demonstrate partial template application.
    
    Shows how to create a partially applied template where some variables
    are fixed, allowing reuse with only the remaining variables.
    
    Returns:
        PromptTemplate: A partially applied template with 'tone' and
            'audience' fixed.
    """
    print("\n" + "="*80)
    print("2. PARTIAL TEMPLATE APPLICATION")
    print("="*80)
    
    # Create base template with four variables
    base_template = (
        "Write a {length} article about {topic} for {audience} audience. "
        "The tone should be {tone}."
    )
    base_prompt = PromptTemplate.from_template(base_template)
    
    print(f"\nBase template variables: {base_prompt.input_variables}")
    
    # Apply partial() to fix some variables (tone and audience)
    # This creates a new template that only needs topic and length
    partial_prompt = base_prompt.partial(
        tone="professional",
        audience="software developers"
    )
    
    print(f"After partial application: {partial_prompt.input_variables}")
    print("\nNow we only need to provide 'topic' and 'length' when invoking!")
    
    return partial_prompt


def demonstrate_lcel_composition() -> None:
    """Demonstrate LCEL pipe operator composition.
    
    Shows how to compose a complete chain using the LCEL pipe operator (|),
    including detailed type flow documentation at each stage of composition.
    
    Type Flow Visualization:
    ========================
    Dict[str, str] → PromptTemplate → StringPromptValue → OpenAI
        → str (from LLM) → StrOutputParser → str (final output)
    """
    print("\n" + "="*80)
    print("3. LCEL COMPOSITION WITH PIPE OPERATOR")
    print("="*80)
    
    # Create prompt template (this will be the start of our chain)
    template = "Write a {length} summary about {topic}."
    prompt = PromptTemplate.from_template(template)
    
    # Initialize OpenAI model
    # Note: Using gpt-3.5-turbo-instruct which is a completion model
    model = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=0.7,
        max_tokens=200
    )
    
    # Create output parser to extract string from LLMResult
    # StrOutputParser converts the complex LLMResult object to a simple string
    parser = StrOutputParser()
    
    # LCEL Composition using pipe operator |
    # This creates a RunnableSequence that connects the components
    # 
    # Type flow through the chain:
    # 1. Input: {"topic": "...", "length": "..."} - Dict[str, str]
    # 2. prompt processes input → StringPromptValue (formatted prompt text)
    # 3. model processes StringPromptValue → str (LLM generated text)
    # 4. parser processes str → str (extracts clean string output)
    #
    # The | operator type checks at composition time to ensure:
    # - Output type of stage N matches input type of stage N+1
    # - Runnable[A,B] | Runnable[B,C] produces Runnable[A,C]
    chain = prompt | model | parser
    
    print("\nChain composition: prompt | model | parser")
    print("Type flow: Dict → StringPromptValue → str → str")
    
    # Single invocation with .invoke()
    print("\n--- Single Invocation ---")
    result = chain.invoke({
        "topic": "quantum computing",
        "length": "brief"
    })
    print(f"\nInput: {{'topic': 'quantum computing', 'length': 'brief'}}")
    print(f"Output:\n{result}")
    
    # Batch invocation with .batch() for multiple inputs
    print("\n--- Batch Invocation ---")
    inputs = [
        {"topic": "machine learning", "length": "concise"},
        {"topic": "blockchain", "length": "short"},
        {"topic": "cloud computing", "length": "brief"}
    ]
    results = chain.batch(inputs)
    
    print(f"\nProcessed {len(inputs)} inputs in batch:")
    for i, (inp, out) in enumerate(zip(inputs, results), 1):
        print(f"\n{i}. Topic: {inp['topic']}")
        print(f"   Output: {out[:100]}..." if len(out) > 100 else f"   Output: {out}")


def demonstrate_advanced_patterns() -> None:
    """Demonstrate advanced prompt template patterns.
    
    Shows additional features like:
    - Combining partial application with LCEL
    - Reusing templates with different models
    - Inspecting chain structure
    """
    print("\n" + "="*80)
    print("4. ADVANCED PATTERNS")
    print("="*80)
    
    # Create a template with partial application
    template = "Explain {concept} to a {audience} using {style} language."
    prompt = PromptTemplate.from_template(template).partial(
        style="simple, non-technical"
    )
    
    print(f"\nTemplate with partial: {prompt.template}")
    print(f"Required variables: {prompt.input_variables}")
    
    # Create chain with partialed prompt
    model = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.5, max_tokens=150)
    chain = prompt | model | StrOutputParser()
    
    # Invoke with remaining variables
    result = chain.invoke({
        "concept": "neural networks",
        "audience": "high school student"
    })
    
    print(f"\nInput: concept='neural networks', audience='high school student'")
    print(f"Output:\n{result[:200]}...")
    
    # Demonstrate prompt inspection
    print("\n--- Prompt Inspection ---")
    print(f"Template format: {prompt.template_format}")
    print(f"Input variables: {prompt.input_variables}")
    print(f"Partial variables: {list(prompt.partial_variables.keys())}")


def main() -> None:
    """Main execution function demonstrating all PromptTemplate patterns.
    
    Orchestrates all demonstration functions in a logical learning sequence:
    1. Basic multi-variable templates
    2. Partial template application
    3. LCEL composition with pipe operator
    4. Advanced patterns and techniques
    
    Includes comprehensive error handling for common failure modes:
    - Missing API key
    - API call failures
    - Rate limiting
    """
    try:
        # Setup environment and validate API key
        print("Setting up environment...")
        api_key = setup_environment()
        print("✓ Environment configured successfully")
        
        # Execute demonstrations
        demonstrate_basic_prompt_template()
        demonstrate_partial_templates()
        demonstrate_lcel_composition()
        demonstrate_advanced_patterns()
        
        # Success message
        print("\n" + "="*80)
        print("✓ All demonstrations completed successfully!")
        print("="*80)
        print("\nKey Takeaways:")
        print("1. PromptTemplate supports multiple input variables with {var} syntax")
        print("2. Use .partial() to fix some variables and create reusable templates")
        print("3. LCEL pipe operator (|) creates type-safe chain compositions")
        print("4. Type flow: Dict → PromptTemplate → StringPromptValue → LLM → str")
        print("5. Use .invoke() for single calls, .batch() for multiple inputs")
        print("\nNext Steps:")
        print("- Try lcel_basic_composition.py for more LCEL patterns")
        print("- Explore sequential_chain.py for multi-step workflows")
        
    except ValueError as e:
        # Handle missing API key or configuration errors
        print(f"\n❌ Configuration Error: {e}", file=sys.stderr)
        print("\nTroubleshooting:", file=sys.stderr)
        print("1. Create .env file in examples/basic_chains/", file=sys.stderr)
        print("2. Add: OPENAI_API_KEY=your-key-here", file=sys.stderr)
        print("3. Get your key from: https://platform.openai.com/api-keys",
              file=sys.stderr)
        sys.exit(1)
        
    except ImportError as e:
        # Handle missing dependencies
        print(f"\n❌ Import Error: {e}", file=sys.stderr)
        print("\nInstall required dependencies:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        print("  or", file=sys.stderr)
        print("  uv pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        # Handle API errors, rate limits, and other runtime errors
        print(f"\n❌ Runtime Error: {e}", file=sys.stderr)
        print("\nCommon issues:", file=sys.stderr)
        print("- Rate limit exceeded: Wait a few minutes or upgrade OpenAI plan",
              file=sys.stderr)
        print("- Invalid API key: Check your OPENAI_API_KEY is correct",
              file=sys.stderr)
        print("- Network issues: Check internet connection", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    """
    Execute the example when run as a script.
    
    Expected output:
    - Demonstration of basic prompt templates with multiple variables
    - Examples of partial template application
    - LCEL chain composition with type flow documentation
    - Advanced patterns combining features
    - Generated text outputs from OpenAI model
    
    Requirements:
    - Python >=3.10
    - OpenAI API key set in .env file
    - Dependencies installed from requirements.txt
    
    Usage:
        python prompt_template_chain.py
    """
    main()
