#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic Model Integration with LangChain Chains.

This example demonstrates how to integrate Pydantic models with LangChain chains
for type-safe input/output validation and structured data handling. It showcases:

- Complete type annotation coverage (100%) per Agent Action Plan section 0.1.1
- Pydantic BaseModel for structured input/output validation
- Type-safe chain composition with validated data structures
- Runtime validation with automatic error reporting
- Field validators for custom validation logic
- JSON schema generation for API documentation
- Integration with PydanticOutputParser for structured LLM outputs

This example addresses type opacity issues by making data structures explicit
and ensuring type safety throughout the chain execution lifecycle.

Source: Agent Action Plan section 0.5.1 - Type Patterns Examples
"""

import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator


# ============================================================================
# Pydantic Input Model Definition
# ============================================================================

class PersonQuery(BaseModel):
    """Input model for person information query.
    
    This Pydantic model demonstrates:
    - Field validation with constraints (age bounds)
    - Optional fields with default values
    - List fields with default factories
    - Custom field validators
    - Self-documenting field descriptions
    
    The model provides runtime validation and automatic JSON schema generation,
    making it ideal for API contracts and type-safe chain inputs.
    """
    
    name: str = Field(
        ...,
        description="Person's full name (required)",
        min_length=1,
        max_length=100
    )
    
    age: int = Field(
        ...,
        description="Person's age in years",
        ge=0,
        le=150
    )
    
    interests: List[str] = Field(
        default_factory=list,
        description="List of personal interests or hobbies"
    )
    
    location: Optional[str] = Field(
        None,
        description="Current location (city, country)",
        max_length=200
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name contains only alphabetic characters and spaces.
        
        Args:
            v: The name value to validate
            
        Returns:
            The validated name
            
        Raises:
            ValueError: If name contains invalid characters
        """
        if not all(c.isalpha() or c.isspace() for c in v):
            raise ValueError(
                'Name must contain only alphabetic characters and spaces'
            )
        return v.strip()
    
    @field_validator('interests')
    @classmethod
    def validate_interests(cls, v: List[str]) -> List[str]:
        """Validate that interests list is not empty if provided.
        
        Args:
            v: The interests list to validate
            
        Returns:
            The validated interests list
            
        Raises:
            ValueError: If any interest is empty
        """
        if v:
            for interest in v:
                if not interest.strip():
                    raise ValueError('Each interest must be non-empty')
        return [interest.strip() for interest in v]


# ============================================================================
# Pydantic Output Model Definition
# ============================================================================

class PersonBio(BaseModel):
    """Output model for generated person biography.
    
    This model demonstrates:
    - Structured LLM output parsing
    - Float constraints for confidence scores
    - Required list fields
    - Type-safe output validation
    
    The PydanticOutputParser will validate LLM output against this schema,
    ensuring the response matches expected structure and types.
    """
    
    summary: str = Field(
        ...,
        description="Brief biographical summary (2-3 sentences)",
        min_length=10
    )
    
    personality_traits: List[str] = Field(
        ...,
        description="3-5 key personality traits inferred from the person",
        min_items=3,
        max_items=5
    )
    
    career_suggestion: str = Field(
        ...,
        description="Career path recommendation based on interests and traits",
        min_length=10
    )
    
    confidence_score: float = Field(
        ...,
        description="Confidence score for the bio (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    @field_validator('personality_traits')
    @classmethod
    def validate_traits(cls, v: List[str]) -> List[str]:
        """Ensure each trait is non-empty.
        
        Args:
            v: The list of personality traits
            
        Returns:
            The validated traits list
            
        Raises:
            ValueError: If any trait is empty
        """
        for trait in v:
            if not trait.strip():
                raise ValueError('Each personality trait must be non-empty')
        return [trait.strip() for trait in v]


# ============================================================================
# Type-Safe Chain Construction Functions
# ============================================================================

def create_pydantic_chain() -> Runnable[Dict[str, Any], PersonBio]:
    """Create a type-safe chain using Pydantic models for input/output.
    
    This function demonstrates:
    - PydanticOutputParser for structured output validation
    - Prompt template with format instructions
    - Type flow: Dict[str, Any] → ChatPromptTemplate → ChatOpenAI → 
                 PydanticOutputParser → PersonBio
    - Complete chain type signature: Runnable[Dict[str, Any], PersonBio]
    
    Returns:
        A fully composed chain with Pydantic input/output validation
        
    Raises:
        ValueError: If OpenAI API key is not configured
    """
    # Verify API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set. "
            "Create a .env file with your API key or set it in environment."
        )
    
    # Stage 1: Create PydanticOutputParser for PersonBio
    # This parser validates LLM output against the PersonBio schema
    parser: PydanticOutputParser[PersonBio] = PydanticOutputParser(
        pydantic_object=PersonBio
    )
    
    # Stage 2: Create prompt template with format instructions
    # The parser provides format instructions for the LLM
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert biographer and career counselor. "
            "Based on the person's information, create a detailed bio.\n\n"
            "{format_instructions}"
        )),
        ("human", (
            "Create a biography for:\n"
            "Name: {name}\n"
            "Age: {age}\n"
            "Interests: {interests}\n"
            "Location: {location}\n"
        ))
    ])
    
    # Stage 3: Create LLM instance
    # Type: ChatOpenAI transforms ChatPromptValue → AIMessage
    model: ChatOpenAI = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Stage 4: Compose the chain with explicit type flow
    # Type Flow Documentation:
    #   Input: Dict[str, Any] with keys: name, age, interests, location, 
    #          format_instructions
    #   → ChatPromptTemplate: Dict → ChatPromptValue (List[BaseMessage])
    #   → ChatOpenAI: ChatPromptValue → AIMessage
    #   → PydanticOutputParser: AIMessage → PersonBio (validated)
    #   Output: PersonBio
    #
    # Complete chain signature: Runnable[Dict[str, Any], PersonBio]
    chain: Runnable[Dict[str, Any], PersonBio] = (
        prompt | model | parser
    )
    
    return chain


def create_input_transformer() -> RunnableLambda[PersonQuery, Dict[str, Any]]:
    """Create a typed lambda to transform PersonQuery to dict for chain input.
    
    This demonstrates:
    - RunnableLambda with explicit type annotations
    - Type-safe data transformation
    - Integration of Pydantic models with LCEL chains
    
    Returns:
        A RunnableLambda that transforms PersonQuery → Dict[str, Any]
    """
    def transform_input(query: PersonQuery) -> Dict[str, Any]:
        """Transform PersonQuery Pydantic model to dictionary.
        
        Args:
            query: Validated PersonQuery instance
            
        Returns:
            Dictionary with keys required by the prompt template
        """
        # Get format instructions from the parser
        parser = PydanticOutputParser(pydantic_object=PersonBio)
        
        return {
            "name": query.name,
            "age": str(query.age),
            "interests": ", ".join(query.interests) if query.interests else "None",
            "location": query.location or "Not specified",
            "format_instructions": parser.get_format_instructions()
        }
    
    # Create typed RunnableLambda
    # Type signature: RunnableLambda[PersonQuery, Dict[str, Any]]
    return RunnableLambda(transform_input)


# ============================================================================
# Example Usage and Demonstration
# ============================================================================

def demonstrate_pydantic_validation() -> None:
    """Demonstrate Pydantic validation features.
    
    Shows:
    - Successful validation with valid data
    - Validation errors with detailed messages
    - Field-level error reporting
    """
    print("=" * 80)
    print("Pydantic Validation Demonstration")
    print("=" * 80)
    
    # Example 1: Valid input
    print("\n1. Valid Input:")
    try:
        valid_query = PersonQuery(
            name="Alice Johnson",
            age=28,
            interests=["reading", "hiking", "photography"],
            location="San Francisco, CA"
        )
        print(f"✓ Successfully validated: {valid_query.name}")
        print(f"  Model dump: {valid_query.model_dump()}")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    # Example 2: Invalid age
    print("\n2. Invalid Age (negative):")
    try:
        invalid_query = PersonQuery(
            name="Bob Smith",
            age=-5,  # Invalid: negative age
            interests=["coding"]
        )
        print(f"✓ Validated: {invalid_query.name}")
    except ValidationError as e:
        print(f"✗ Validation failed (expected):")
        for error in e.errors():
            print(f"  Field: {error['loc']}, Error: {error['msg']}")
    
    # Example 3: Invalid name
    print("\n3. Invalid Name (contains numbers):")
    try:
        invalid_query = PersonQuery(
            name="Charlie123",  # Invalid: contains numbers
            age=30,
            interests=["sports"]
        )
        print(f"✓ Validated: {invalid_query.name}")
    except ValidationError as e:
        print(f"✗ Validation failed (expected):")
        for error in e.errors():
            print(f"  Field: {error['loc']}, Error: {error['msg']}")
    
    # Example 4: Missing required field
    print("\n4. Missing Required Field (name):")
    try:
        invalid_query = PersonQuery(
            age=25,
            interests=["music"]
        )
        print(f"✓ Validated")
    except ValidationError as e:
        print(f"✗ Validation failed (expected):")
        for error in e.errors():
            print(f"  Field: {error['loc']}, Error: {error['msg']}")


def demonstrate_json_schema_generation() -> None:
    """Demonstrate automatic JSON schema generation from Pydantic models.
    
    This is useful for:
    - API documentation
    - Client code generation
    - OpenAPI specification
    - Type documentation
    """
    print("\n" + "=" * 80)
    print("JSON Schema Generation")
    print("=" * 80)
    
    # Generate schema for input model
    print("\nPersonQuery JSON Schema:")
    print("-" * 80)
    input_schema = PersonQuery.model_json_schema()
    for key, value in input_schema.items():
        print(f"{key}: {value}")
    
    # Generate schema for output model
    print("\nPersonBio JSON Schema:")
    print("-" * 80)
    output_schema = PersonBio.model_json_schema()
    for key, value in output_schema.items():
        print(f"{key}: {value}")


def demonstrate_full_chain_execution() -> None:
    """Demonstrate complete chain execution with Pydantic models.
    
    Shows:
    - End-to-end type-safe chain execution
    - Input validation
    - LLM invocation
    - Output validation
    - Error handling
    """
    print("\n" + "=" * 80)
    print("Full Chain Execution with Pydantic Models")
    print("=" * 80)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ OPENAI_API_KEY not set in environment.")
        print("To run this example with actual LLM:")
        print("1. Create a .env file with: OPENAI_API_KEY=your-key-here")
        print("2. Or set environment variable: export OPENAI_API_KEY=your-key")
        print("\nSkipping LLM execution (demonstration mode only)...")
        return
    
    try:
        # Step 1: Create and validate input
        print("\nStep 1: Creating validated input...")
        query = PersonQuery(
            name="Emily Chen",
            age=32,
            interests=["artificial intelligence", "rock climbing", "cooking"],
            location="Seattle, WA"
        )
        print(f"✓ Input validated: {query.name}, age {query.age}")
        
        # Step 2: Create transformer and chain
        print("\nStep 2: Building type-safe chain...")
        transformer = create_input_transformer()
        chain = create_pydantic_chain()
        
        # Step 3: Transform input to dict
        print("\nStep 3: Transforming Pydantic model to dict...")
        input_dict = transformer.invoke(query)
        print(f"✓ Transformed input keys: {list(input_dict.keys())}")
        
        # Step 4: Execute chain
        print("\nStep 4: Executing chain (calling LLM)...")
        result: PersonBio = chain.invoke(input_dict)
        
        # Step 5: Display validated output
        print("\nStep 5: Chain execution complete!")
        print("=" * 80)
        print("Validated Output (PersonBio):")
        print("-" * 80)
        print(f"Summary: {result.summary}")
        print(f"\nPersonality Traits:")
        for trait in result.personality_traits:
            print(f"  - {trait}")
        print(f"\nCareer Suggestion: {result.career_suggestion}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print("=" * 80)
        
        # Step 6: Demonstrate model introspection
        print("\nStep 6: Model Introspection:")
        print(f"Output as dict: {result.model_dump()}")
        print(f"Output as JSON: {result.model_dump_json(indent=2)}")
        
    except ValidationError as e:
        print(f"\n✗ Validation Error:")
        for error in e.errors():
            print(f"  Field: {error['loc']}, Error: {error['msg']}")
    except Exception as e:
        print(f"\n✗ Execution Error: {type(e).__name__}: {e}")


def demonstrate_composed_chain() -> None:
    """Demonstrate composed chain with Pydantic transformation.
    
    Shows:
    - Full type-safe pipeline: PersonQuery → Dict → PersonBio
    - Chain composition with type preservation
    - Complete type flow documentation
    """
    print("\n" + "=" * 80)
    print("Composed Chain: PersonQuery → Dict → PersonBio")
    print("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ OPENAI_API_KEY not set. Skipping composed chain demo...")
        return
    
    try:
        # Create composed chain
        # Type Flow:
        #   PersonQuery (Pydantic input)
        #   → RunnableLambda: PersonQuery → Dict[str, Any]
        #   → ChatPromptTemplate: Dict → ChatPromptValue
        #   → ChatOpenAI: ChatPromptValue → AIMessage
        #   → PydanticOutputParser: AIMessage → PersonBio (Pydantic output)
        #
        # Complete chain signature: Runnable[PersonQuery, PersonBio]
        transformer = create_input_transformer()
        main_chain = create_pydantic_chain()
        
        full_chain: Runnable[PersonQuery, PersonBio] = transformer | main_chain
        
        # Execute with Pydantic input directly
        print("\nExecuting composed chain with Pydantic input...")
        query = PersonQuery(
            name="David Martinez",
            age=45,
            interests=["teaching", "writing", "gardening"],
            location="Austin, TX"
        )
        
        result: PersonBio = full_chain.invoke(query)
        
        print("\n✓ Chain execution successful!")
        print(f"Input type: {type(query).__name__}")
        print(f"Output type: {type(result).__name__}")
        print(f"\nGenerated summary: {result.summary[:100]}...")
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    """Main function demonstrating Pydantic model integration patterns.
    
    Executes all demonstration functions showing:
    - Pydantic validation features
    - JSON schema generation
    - Type-safe chain execution
    - Error handling patterns
    """
    print("\n" + "=" * 80)
    print("Pydantic Model Integration with LangChain Chains")
    print("Type-Safe Input/Output Validation Examples")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    
    # Run demonstrations
    demonstrate_pydantic_validation()
    demonstrate_json_schema_generation()
    demonstrate_full_chain_execution()
    demonstrate_composed_chain()
    
    # Summary
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
1. Pydantic Models provide runtime validation and type safety
   - Field constraints (min, max, regex patterns)
   - Custom validators for complex logic
   - Automatic error reporting

2. PydanticOutputParser ensures structured LLM outputs
   - Validates LLM responses against schema
   - Provides format instructions to LLM
   - Type-safe output guarantees

3. Type Flow is explicit and documented
   - PersonQuery → Dict → ChatPromptValue → AIMessage → PersonBio
   - Each stage has clear input/output types
   - Runnable[Input, Output] protocol throughout

4. Benefits for Production Code:
   - Early error detection (validation at boundaries)
   - Self-documenting schemas (JSON schema generation)
   - IDE support (autocomplete, type checking)
   - API contract enforcement

5. Integration with mypy --strict
   - Complete type annotation coverage
   - Static type checking validates chain composition
   - Prevents type-related runtime errors

For more information:
- Agent Action Plan section 0.1.1: Type annotation requirements
- Agent Action Plan section 0.5.1: Type patterns implementation
- Pydantic docs: https://docs.pydantic.dev/
- LangChain docs: https://python.langchain.com/docs/
    """)
    print("=" * 80)


if __name__ == "__main__":
    main()
