"""Sequential Chain Example - Multi-Step Processing Pipeline

This example demonstrates the SequentialChain pattern in LangChain, which allows
chaining multiple processing steps where the output of step N becomes available
as input to step N+1.

Concept:
    SequentialChain composes multiple chains in sequence, maintaining a shared
    state dictionary that accumulates outputs from each step. This enables
    complex multi-stage processing pipelines where each step builds on previous
    results.

Use Cases:
    - Multi-stage text processing (e.g., generate → refine → summarize)
    - Data transformation pipelines with dependent steps
    - Complex workflows requiring intermediate results
    - Building upon previous outputs in a structured way

Input/Output Variable Mapping:
    - input_variables: Variables required at the start (provided by user)
    - output_variables: Variables to return at the end (selected from accumulated state)
    - Each chain adds its output_key to the shared state
    - Subsequent chains can access any variables in the accumulated state

Type Flow Through Sequential Chain:
    Initial Input:  {"topic": "space exploration"}
    After Chain 1:  {"topic": "space exploration", "outline": "..."}
    After Chain 2:  {"topic": "space exploration", "outline": "...", "story": "..."}
    After Chain 3:  {"topic": "...", "outline": "...", "story": "...", "summary": "..."}
    Final Output:   {"outline": "...", "story": "...", "summary": "..."}

Source: libs/langchain/langchain_classic/chains/sequential.py
"""

import os
import sys
from typing import Dict, Any

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import OpenAI
    from langchain_classic.chains import LLMChain, SequentialChain
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r requirements.txt")
    print("  or")
    print("  uv pip install -r requirements.txt")
    sys.exit(1)


def setup_environment() -> str:
    """Setup environment and validate API key.
    
    Loads environment variables from .env file and validates that the
    required OPENAI_API_KEY is present.
    
    Returns:
        str: The OpenAI API key.
    
    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Validate API key exists
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment.\n"
            "Please create a .env file with your OpenAI API key:\n"
            "  1. Copy .env.example to .env\n"
            "  2. Add your OpenAI API key: OPENAI_API_KEY=your-key-here\n"
            "  3. Get an API key from: https://platform.openai.com/api-keys"
        )
    
    return api_key


def main() -> None:
    """Demonstrate SequentialChain with 3-step story generation pipeline.
    
    This function creates a sequential chain with three steps:
    1. Generate a story outline from a topic
    2. Expand the outline into a full story
    3. Summarize the story
    
    Each step builds upon the outputs of previous steps, demonstrating
    how SequentialChain manages variable flow through the pipeline.
    """
    print("=" * 80)
    print("Sequential Chain Example: Multi-Step Story Generation")
    print("=" * 80)
    print()
    
    try:
        # Setup: Load environment and initialize model
        api_key = setup_environment()
        print("✓ Environment setup complete")
        
        # Initialize OpenAI model
        # Using temperature=0.7 for creative text generation
        model = OpenAI(
            api_key=api_key,
            temperature=0.7,
            max_tokens=500
        )
        print("✓ OpenAI model initialized")
        print()
        
        # =====================================================================
        # STEP 1: Create Chain to Generate Story Outline
        # =====================================================================
        # Input:  {"topic": <user input>}
        # Output: {"outline": <generated outline>}
        
        prompt_1 = PromptTemplate(
            input_variables=["topic"],
            template=(
                "Create a brief story outline (3-5 bullet points) about {topic}. "
                "Include main plot points and characters."
            )
        )
        
        # output_key specifies what key this chain's output will be stored under
        # This becomes available to subsequent chains in the sequence
        chain_1 = LLMChain(
            llm=model,
            prompt=prompt_1,
            output_key="outline"  # Output stored as {"outline": "..."}
        )
        print("✓ Chain 1 created: Topic → Outline")
        
        # =====================================================================
        # STEP 2: Create Chain to Expand Outline into Full Story
        # =====================================================================
        # Input:  {"outline": <from chain_1>, "topic": <from original input>}
        # Output: {"story": <generated story>}
        
        prompt_2 = PromptTemplate(
            input_variables=["outline", "topic"],
            template=(
                "Write a short story based on this outline:\n\n"
                "{outline}\n\n"
                "The story should be about {topic} and follow the outline above. "
                "Keep it to 2-3 paragraphs."
            )
        )
        
        # This chain requires TWO inputs: "outline" (from chain_1) and "topic" 
        # (from original input). SequentialChain validates all inputs are available.
        chain_2 = LLMChain(
            llm=model,
            prompt=prompt_2,
            output_key="story"  # Output stored as {"story": "..."}
        )
        print("✓ Chain 2 created: Outline + Topic → Story")
        
        # =====================================================================
        # STEP 3: Create Chain to Summarize the Story
        # =====================================================================
        # Input:  {"story": <from chain_2>}
        # Output: {"summary": <generated summary>}
        
        prompt_3 = PromptTemplate(
            input_variables=["story"],
            template=(
                "Provide a one-sentence summary of this story:\n\n"
                "{story}"
            )
        )
        
        chain_3 = LLMChain(
            llm=model,
            prompt=prompt_3,
            output_key="summary"  # Output stored as {"summary": "..."}
        )
        print("✓ Chain 3 created: Story → Summary")
        print()
        
        # =====================================================================
        # Compose SequentialChain: Connecting All Steps
        # =====================================================================
        # The SequentialChain orchestrates the execution of all chains in order,
        # managing the flow of data between them.
        
        overall_chain = SequentialChain(
            chains=[chain_1, chain_2, chain_3],
            
            # input_variables: What the user must provide at the start
            input_variables=["topic"],
            
            # output_variables: What gets returned from the complete chain
            # We select specific outputs to return (could also use return_all=True)
            output_variables=["outline", "story", "summary"],
            
            # verbose=True shows the execution flow and intermediate results
            verbose=True
        )
        print("✓ Sequential chain composed with 3 steps")
        print()
        
        # =====================================================================
        # Variable Flow Validation by SequentialChain:
        # =====================================================================
        # SequentialChain validates at initialization that:
        # 1. Chain 1 requirements satisfied:
        #    - Needs: ["topic"] 
        #    - Has: ["topic"] from input_variables ✓
        # 2. Chain 2 requirements satisfied:
        #    - Needs: ["outline", "topic"]
        #    - Has: ["topic"] (input) + ["outline"] (from chain_1) ✓
        # 3. Chain 3 requirements satisfied:
        #    - Needs: ["story"]
        #    - Has: ["story"] (from chain_2) ✓
        # 4. Output variables valid:
        #    - Wants: ["outline", "story", "summary"]
        #    - Available: All three keys exist in accumulated state ✓
        
        # =====================================================================
        # Execute the Sequential Chain
        # =====================================================================
        
        print("Executing sequential chain with topic: 'space exploration'")
        print("-" * 80)
        print()
        
        # Invoke the chain with initial input
        # Type: Dict[str, str] → Dict[str, str]
        result = overall_chain.invoke({"topic": "space exploration"})
        
        print()
        print("-" * 80)
        print("Sequential chain execution complete!")
        print("=" * 80)
        print()
        
        # =====================================================================
        # Display Results with Type Flow Explanation
        # =====================================================================
        
        print("RESULTS:")
        print()
        
        print("1. OUTLINE (from Chain 1):")
        print("-" * 80)
        print(result["outline"])
        print()
        
        print("2. FULL STORY (from Chain 2):")
        print("-" * 80)
        print(result["story"])
        print()
        
        print("3. SUMMARY (from Chain 3):")
        print("-" * 80)
        print(result["summary"])
        print()
        
        # =====================================================================
        # Data Flow Diagram (as executed):
        # =====================================================================
        print("=" * 80)
        print("DATA FLOW THROUGH SEQUENTIAL CHAIN:")
        print("=" * 80)
        print()
        print("Initial Input:")
        print("  {'topic': 'space exploration'}")
        print()
        print("After Chain 1 (Generate Outline):")
        print("  {'topic': 'space exploration', 'outline': '<outline text>'}")
        print()
        print("After Chain 2 (Write Story):")
        print("  {'topic': 'space exploration', 'outline': '<outline>', 'story': '<story>'}")
        print()
        print("After Chain 3 (Summarize):")
        print("  {'topic': '...', 'outline': '...', 'story': '...', 'summary': '<summary>'}")
        print()
        print("Final Output (selected output_variables):")
        print("  {'outline': '<outline>', 'story': '<story>', 'summary': '<summary>'}")
        print()
        
    except ValueError as e:
        # Handles environment setup errors (missing API key)
        print(f"\n❌ Configuration Error: {e}")
        sys.exit(1)
        
    except ImportError as e:
        # Handles missing dependencies
        print(f"\n❌ Import Error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        # Handles unexpected errors during chain execution
        print(f"\n❌ Unexpected Error: {e}")
        print("\nCommon issues and solutions:")
        print("  1. API rate limits: Wait a moment and try again")
        print("  2. Network issues: Check your internet connection")
        print("  3. Invalid API key: Verify your OPENAI_API_KEY in .env")
        sys.exit(1)


# =============================================================================
# COMMON TROUBLESHOOTING ISSUES WITH SEQUENTIAL CHAINS
# =============================================================================
#
# Issue 1: Variable Name Mismatches
# ----------------------------------
# Problem: ValueError: Missing required input keys
# Cause: Chain N requires a variable that no previous chain produces
# Solution: Ensure output_key of chain N-1 matches input_variables of chain N
#
# Example Error:
#   chain_1 has output_key="summary"
#   chain_2 needs input_variables=["outline"]  # Mismatch!
#
# Fix:
#   chain_1 should have output_key="outline"
#
#
# Issue 2: Missing input_variables
# ---------------------------------
# Problem: ValueError: Missing required input keys
# Cause: Chain requires variable not in input_variables or previous outputs
# Solution: Add missing variable to input_variables list
#
# Example Error:
#   input_variables=["topic"]
#   chain_2 needs ["topic", "style"]  # "style" not available!
#
# Fix:
#   input_variables=["topic", "style"]
#
#
# Issue 3: Output Key Conflicts
# ------------------------------
# Problem: ValueError: Chain returned keys that already exist
# Cause: Chain tries to output a key that's already in the state
# Solution: Use unique output_key for each chain
#
# Example Error:
#   chain_1 has output_key="result"
#   chain_2 has output_key="result"  # Conflict!
#
# Fix:
#   chain_2 should have output_key="result_2" or similar
#
#
# Issue 4: Incorrect output_variables
# ------------------------------------
# Problem: ValueError: Expected output variables that were not found
# Cause: output_variables requests keys that no chain produces
# Solution: Only request keys that exist after all chains run
#
# Example Error:
#   Chains produce: ["outline", "story"]
#   output_variables=["outline", "story", "summary"]  # "summary" doesn't exist!
#
# Fix:
#   output_variables=["outline", "story"]
#
#
# Issue 5: Chain Order Matters
# -----------------------------
# Problem: Chain fails because it runs before its dependencies
# Cause: Chains are not ordered correctly in the chains list
# Solution: Arrange chains so each chain comes after chains it depends on
#
# Example Error:
#   chains=[chain_2, chain_1]  # chain_2 needs output from chain_1!
#
# Fix:
#   chains=[chain_1, chain_2]  # chain_1 first, then chain_2
#
# =============================================================================


if __name__ == "__main__":
    """Entry point when script is run directly.
    
    This block executes the main() function when the script is run from
    the command line:
    
        python sequential_chain.py
    
    Expected Output:
        - Environment validation messages
        - Chain initialization confirmations
        - Verbose execution logs showing each chain's input/output
        - Final results displaying outline, story, and summary
        - Data flow diagram explaining variable progression
    
    Execution Time:
        Approximately 15-30 seconds depending on:
        - OpenAI API response time
        - Network latency
        - Token generation length
    """
    main()
