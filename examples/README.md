# LangChain Executable Examples

Welcome to the LangChain executable examples directory! This collection provides complete, runnable code examples demonstrating how to use LangChain integration code in this repository.

## Overview

These examples are designed to be **self-contained, executable, and production-ready**. Each example file includes:

- **Complete imports**: All necessary dependencies explicitly imported
- **Sample data**: Real-world example inputs (or graceful mocks when API keys unavailable)
- **Error handling**: Best practices for handling common failure modes
- **Clear outputs**: Informative print statements showing execution results
- **Inline documentation**: Comments explaining LangChain-specific concepts

**Target Audience**: These examples enable developers with <2 years of experience to independently implement LangChain chains, following the junior developer enablement goal from the documentation initiative.

**Philosophy**: Every example can be copied, pasted, and run without modification (given proper environment setup). No placeholders, no TODOs, no incomplete implementations.

## Prerequisites

### Python Version

- **Required**: Python >=3.10.0, <4.0.0
- **Tested**: Python 3.10, 3.13
- **Recommended**: Python 3.13.8 (latest tested version)

### Package Managers

Choose one of the following tools to install dependencies:

| Tool | Installation Command | Documentation |
|------|---------------------|---------------|
| **uv** (recommended) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | [uv docs](https://github.com/astral-sh/uv) |
| **pip** | Included with Python | [pip docs](https://pip.pypa.io/) |
| **poetry** | `curl -sSL https://install.python-poetry.org \| python3 -` | [poetry docs](https://python-poetry.org/) |

**Why uv?** Fast, reliable Python package installer with excellent dependency resolution. Used throughout LangChain monorepo development.

### API Keys

Most examples require API keys from LLM providers. Obtain keys from:

- **OpenAI**: [platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
- **Anthropic**: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

## Installation

### Quick Install (All Examples)

Install all dependencies needed for every example category:

```bash
# Using uv (recommended)
uv pip install langchain-core langchain-classic langchain-openai python-dotenv

# Using pip
pip install langchain-core langchain-classic langchain-openai python-dotenv
```

### Category-Specific Installation

Each example category includes its own `requirements.txt` file for targeted installation:

```bash
# Install dependencies for basic_chains only
uv pip install -r examples/basic_chains/requirements.txt

# Install dependencies for advanced_chains (includes vector stores, etc.)
uv pip install -r examples/advanced_chains/requirements.txt

# Install dependencies for type_patterns
uv pip install -r examples/type_patterns/requirements.txt

# Install dependencies for debugging utilities
uv pip install -r examples/debugging/requirements.txt
```

### Development Installation

For contributors working on examples:

```bash
# Install with development tools (ruff, pytest, mypy)
uv pip install langchain-core langchain-classic langchain-openai python-dotenv ruff pytest mypy
```

## Environment Setup

### Configuration Files

Each example category contains a `.env.example` file showing required environment variables:

```bash
# Copy example environment file to create your local configuration
cp examples/basic_chains/.env.example examples/basic_chains/.env

# Edit with your actual API keys
nano examples/basic_chains/.env
```

### Required Environment Variables

Minimum configuration for most examples:

```bash
# .env file format
OPENAI_API_KEY=your_openai_api_key_here
```

Extended configuration for advanced examples:

```bash
# Advanced examples may require additional keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Environment Loading

Examples use `python-dotenv` to automatically load `.env` files:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access API keys
api_key = os.getenv("OPENAI_API_KEY")
```

## Directory Structure

The examples directory is organized into four categories, progressing from simple to complex:

### 📁 basic_chains/

**Purpose**: Foundational chain patterns for beginners

**Target Audience**: Developers new to LangChain, learning core concepts

**Contents**:
- `simple_llm_chain.py` - Minimal LLMChain example showing basic prompt → LLM → response flow
- `prompt_template_chain.py` - PromptTemplate usage with variable substitution
- `lcel_basic_composition.py` - Introduction to LCEL (LangChain Expression Language) pipe operators
- `sequential_chain.py` - SequentialChain demonstrating multi-step chain composition

**Key Concepts Covered**: Chains, Prompts, LCEL operators (|), Sequential composition

**Prerequisites**: OpenAI API key, basic Python knowledge

### 📁 advanced_chains/

**Purpose**: Production-ready patterns for real-world applications

**Target Audience**: Developers building production systems, requiring advanced features

**Contents**:
- `retrieval_qa_chain.py` - Retrieval-Augmented Generation (RAG) with vector store integration
- `memory_enabled_chain.py` - Conversational chains using ConversationBufferMemory
- `agent_with_tools.py` - AgentExecutor with custom tool implementation and validation
- `streaming_responses.py` - Token-by-token streaming for real-time user feedback
- `async_chain_execution.py` - Async/await patterns with asyncio for concurrent execution
- `fallback_chains.py` - RunnableWithFallbacks for error recovery and resilience

**Key Concepts Covered**: RAG, Memory systems, Agent loops, Streaming, Async execution, Error recovery

**Prerequisites**: Understanding of basic chains, additional API keys for advanced features

### 📁 type_patterns/

**Purpose**: Type-safe LangChain development with comprehensive type annotations

**Target Audience**: Developers requiring type safety, teams using mypy strict mode

**Contents**:
- `lcel_type_annotations.py` - LCEL composition with explicit type flow documentation
- `custom_chain_typing.py` - Building type-safe custom chains with proper generic types
- `pydantic_model_chain.py` - Chain with Pydantic model input/output validation

**Key Concepts Covered**: Type hints, Pydantic models, Generic types, Type flow through LCEL pipes

**Prerequisites**: Understanding of Python type hints and Pydantic

### 📁 debugging/

**Purpose**: Utilities and patterns for debugging LangChain chains

**Target Audience**: Developers troubleshooting chain failures, implementing observability

**Contents**:
- `logging_wrapper.py` - Reusable decorator for granular chain execution logging
- `callback_debugger.py` - Custom callback handler for debugging chain events
- `chain_introspection.py` - Utility to inspect and visualize chain structure

**Key Concepts Covered**: Callbacks, Logging strategies, Chain introspection, Debugging techniques

**Prerequisites**: Familiarity with Python decorators and logging

## Quick Start

### Running Your First Example

1. **Install dependencies**:
   ```bash
   uv pip install -r examples/basic_chains/requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp examples/basic_chains/.env.example examples/basic_chains/.env
   # Edit .env with your OPENAI_API_KEY
   ```

3. **Run the simplest example**:
   ```bash
   cd examples/basic_chains
   python simple_llm_chain.py
   ```

4. **Expected output**:
   ```
   ========================================
   Simple LLM Chain Example
   ========================================
   
   Input: Tell me a joke about programming
   
   Response: Why do programmers prefer dark mode?
   Because light attracts bugs!
   
   ========================================
   Example completed successfully!
   ========================================
   ```

### Running Examples Without API Keys

Examples gracefully handle missing API keys by either:
- **Skipping**: Exit with informative message
- **Mocking**: Use mock responses for demonstration (where applicable)

Example output without API key:
```
ERROR: OPENAI_API_KEY environment variable not set
Please set your API key: export OPENAI_API_KEY=your-key-here
Or configure .env file in examples/basic_chains/.env
```

### Running All Examples in a Category

Use pytest to execute all examples as tests:

```bash
# Run all basic_chains examples
pytest examples/basic_chains/ -v

# Run all examples in all categories
pytest examples/ -v

# Run with detailed output
pytest examples/ -v --tb=short
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: `ImportError: No module named 'langchain_core'`

**Cause**: Dependencies not installed

**Solution**:
```bash
uv pip install langchain-core langchain-classic
```

#### Issue: `ValueError: OPENAI_API_KEY not set`

**Cause**: Missing API key configuration

**Solution**:
```bash
# Option 1: Export environment variable
export OPENAI_API_KEY=your-key-here

# Option 2: Create .env file
echo "OPENAI_API_KEY=your-key-here" > examples/basic_chains/.env
```

#### Issue: `ModuleNotFoundError: No module named 'langchain_openai'`

**Cause**: Provider-specific package not installed

**Solution**:
```bash
uv pip install langchain-openai
```

#### Issue: Example hangs or times out

**Cause**: Network issues, API rate limits, or long-running LLM calls

**Solution**:
- Check internet connection
- Verify API key is valid and has credits
- Try again after a few seconds (rate limit cooldown)
- Check example code for timeout parameters

#### Issue: `TypeError: 'dict' object is not callable`

**Cause**: Incorrect LCEL composition or parameter passing

**Solution**:
- Review the specific example's inline comments
- Check type flow documentation in type_patterns/ examples
- Verify you're using LCEL pipe operator (|) correctly

#### Issue: Pydantic validation errors

**Cause**: Input data doesn't match expected Pydantic model schema

**Solution**:
- Review the Pydantic model definition in the example
- Check required fields vs optional fields
- Examine the error message for specific field issues

### Getting Help

If you encounter issues not covered here:

1. **Check the documentation**: Refer to comprehensive guides in `docs/` directory
   - `docs/debugging/common-issues.md` - Top 10 LangChain failure modes
   - `docs/debugging/troubleshooting.md` - Debugging decision tree
   - `docs/guides/error-handling.md` - Error recovery patterns

2. **Review API reference**: Consult detailed API documentation
   - `docs/api-reference/chains/` - Chain class documentation
   - `docs/api-reference/runnables/` - Runnable protocol documentation

3. **Inspect example source**: All examples include inline comments explaining behavior

4. **Check LangChain official docs**: [https://docs.langchain.com/](https://docs.langchain.com/)

## Example Execution Checklist

Before running any example, verify:

- [ ] Python >=3.10 installed (`python --version`)
- [ ] Dependencies installed (`uv pip install -r requirements.txt`)
- [ ] Environment variables configured (`.env` file with API keys)
- [ ] Working directory is correct (`cd examples/category_name/`)
- [ ] Network connectivity available (for API calls)

## References

### Main Documentation

Complete documentation for this repository:

- **Getting Started**: `docs/getting-started/` - Installation, quickstart, configuration, core concepts
- **User Guides**: `docs/guides/` - LCEL composition, memory integration, agent development, async patterns
- **API Reference**: `docs/api-reference/` - Complete API documentation for all public interfaces
- **Architecture**: `docs/architecture/` - System architecture, chain lifecycle, callback system
- **Debugging**: `docs/debugging/` - Common issues, logging strategies, stack trace interpretation

### LangChain Resources

- **Official Documentation**: [https://docs.langchain.com/](https://docs.langchain.com/)
- **API Reference**: [https://reference.langchain.com/python/langchain_classic/](https://reference.langchain.com/python/langchain_classic/)
- **LangChain GitHub**: [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

### Type Annotation Resources

- **PEP 484 - Type Hints**: [https://peps.python.org/pep-0484/](https://peps.python.org/pep-0484/)
- **Pydantic Documentation**: [https://docs.pydantic.dev/](https://docs.pydantic.dev/)
- **mypy Documentation**: [https://mypy.readthedocs.io/](https://mypy.readthedocs.io/)

## Contributing

### Adding New Examples

When contributing new examples, ensure:

1. **Complete and executable**: Example runs without modification (given proper environment)
2. **Self-contained**: All imports explicit, no external file dependencies
3. **Well-documented**: Inline comments explaining LangChain concepts
4. **Error handling**: Graceful handling of missing API keys and common failures
5. **Type annotated**: Full type hints on functions (see `type_patterns/` for guidance)
6. **Tested**: Example executes successfully via pytest

### Example File Template

```python
"""
Brief description of what this example demonstrates.

Prerequisites:
    - langchain-core>=1.0.0
    - langchain-openai
    - OPENAI_API_KEY environment variable

Demonstrates:
    - Key concept 1
    - Key concept 2
    - Key concept 3
"""

from dotenv import load_dotenv
import os
from typing import Dict, Any

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def main() -> None:
    """Main execution function."""
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    # Your example implementation here
    print("Example output...")


if __name__ == "__main__":
    main()
```

### Testing Examples

Run validation before submitting:

```bash
# Type check
mypy --strict examples/your_example.py

# Style check
ruff check examples/your_example.py

# Execute example
python examples/your_example.py

# Test with pytest
pytest examples/your_example.py -v
```

## License

This documentation and examples are part of the LangChain monorepo and follow the same MIT License.

---

**Documentation Version**: 1.0.0  
**Last Updated**: 2024  
**Maintainer**: LangChain Documentation Team  
**LangChain Version Compatibility**: langchain-core>=1.0.0, langchain-classic>=1.0.0
