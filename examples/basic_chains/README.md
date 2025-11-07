# Basic Chains - Foundational LangChain Examples

Welcome to the basic chains examples! This directory contains foundational LangChain patterns designed for developers new to the framework.

## Overview

### Purpose

These examples demonstrate **core LangChain chain concepts** using simple, clear implementations. Each example is self-contained, executable, and progressively introduces new concepts.

### Target Audience

- **Primary**: Developers with <2 years of Python experience learning LangChain
- **Secondary**: Experienced developers new to LangChain who want quick orientation
- **No prior LangChain experience required**

### Learning Objectives

By completing these examples, you will understand:

1. **Chains**: How to compose LLM calls into reusable components
2. **Prompts**: Using PromptTemplate for dynamic input formatting
3. **LCEL**: LangChain Expression Language pipe operator (`|`) for composition
4. **Sequential Processing**: Building multi-step workflows where outputs feed to next steps

### Time Commitment

- **Individual examples**: 5-10 minutes each
- **Complete category**: 30-60 minutes for all examples

## Prerequisites

### Required Knowledge

- **Python Basics**: Functions, classes, dictionaries, string formatting
- **Command Line**: Running Python scripts, activating virtual environments
- **Environment Variables**: Understanding of environment variable configuration

### Python Version

- **Required**: Python >=3.10.0, <4.0.0
- **Tested**: Python 3.10, 3.13
- **Recommended**: Python 3.13 (latest tested version)

Verify your Python version:
```bash
python --version
```

### API Access

**OpenAI API Account**: Required for running examples

1. Create account at [platform.openai.com](https://platform.openai.com/)
2. Generate API key at [platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
3. Note: Free tier has rate limits; upgrade if hitting limits frequently

**Cost Estimate**: Running all basic examples typically costs $0.01-0.05 USD with GPT-3.5-turbo

## Installation Instructions

### Step 1: Navigate to Examples Directory

```bash
# If you cloned the repository
cd examples/basic_chains

# If working from repository root
cd examples/basic_chains
```

### Step 2: Create Virtual Environment

**Why virtual environment?** Isolates dependencies, prevents conflicts with system packages.

```bash
# Create virtual environment
python -m venv venv

# Verify creation
ls venv/
```

### Step 3: Activate Virtual Environment

**Linux/macOS**:
```bash
source venv/bin/activate
```

**Windows**:
```bash
venv\Scripts\activate
```

**Verification**: Your prompt should show `(venv)` prefix:
```
(venv) user@machine:~/examples/basic_chains$
```

### Step 4: Install Dependencies

Choose your preferred package manager:

**Using uv (recommended - fastest)**:
```bash
uv pip install -r requirements.txt
```

**Using pip (standard)**:
```bash
pip install -r requirements.txt
```

**Verification**:
```bash
python -c "import langchain_core; print('LangChain installed successfully!')"
```

### Step 5: Configure Environment Variables

**Copy template**:
```bash
cp .env.example .env
```

**Edit with your API key**:
```bash
# Linux/macOS
nano .env

# Windows
notepad .env
```

**Required content** in `.env`:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Security Note**: Never commit `.env` files to version control. The `.gitignore` file already excludes `.env` files.

### Step 6: Verify Installation

Test your setup:
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key configured!' if os.getenv('OPENAI_API_KEY') else 'API Key missing!')"
```

## Examples Directory Structure

This directory contains the following files:

```
basic_chains/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                # Environment variable template
├── simple_llm_chain.py         # Example 1: Basic LLMChain usage
├── prompt_template_chain.py    # Example 2: PromptTemplate with variables
├── lcel_basic_composition.py   # Example 3: LCEL pipe operator fundamentals
└── sequential_chain.py         # Example 4: Multi-step chain pipeline
```

### Example Files Description

| File | Purpose | Key Concepts | Difficulty |
|------|---------|--------------|------------|
| `simple_llm_chain.py` | Introduction to basic chain structure with LLMChain (deprecated pattern) and modern LCEL alternative | Chain execution, prompt → LLM → response flow | ⭐ Beginner |
| `prompt_template_chain.py` | Using PromptTemplate with multiple input variables and LCEL composition | Dynamic prompts, variable substitution, template formatting | ⭐ Beginner |
| `lcel_basic_composition.py` | Modern LCEL pipe operator syntax with explicit type flow documentation | LCEL operators (`|`), chain composition, type transformations | ⭐⭐ Intermediate |
| `sequential_chain.py` | Multi-step workflows where outputs from step N become inputs to step N+1 | Sequential processing, variable mapping, multi-stage pipelines | ⭐⭐ Intermediate |

### Supporting Files

| File | Description |
|------|-------------|
| `requirements.txt` | All Python package dependencies with version constraints |
| `.env.example` | Template showing required environment variables (copy to `.env`) |

## Recommended Learning Path

Follow this sequence for optimal learning progression:

### 1️⃣ Start with `simple_llm_chain.py`

**Why first**: Introduces fundamental chain concepts without complexity.

**Run it**:
```bash
python simple_llm_chain.py
```

**What you'll learn**:
- Basic chain structure
- LLMChain pattern (deprecated but instructive)
- Modern LCEL alternative
- Simple prompt → LLM → response flow

**Time**: 5 minutes

---

### 2️⃣ Progress to `prompt_template_chain.py`

**Why second**: Builds on chain basics, adds dynamic prompt formatting.

**Run it**:
```bash
python prompt_template_chain.py
```

**What you'll learn**:
- PromptTemplate for variable substitution
- Using multiple input variables
- Template formatting with LCEL
- Reusable prompt patterns

**Time**: 10 minutes

---

### 3️⃣ Learn `lcel_basic_composition.py`

**Why third**: Introduces modern LCEL syntax, the foundation for advanced patterns.

**Run it**:
```bash
python lcel_basic_composition.py
```

**What you'll learn**:
- LCEL pipe operator (`|`)
- Type flow through chain stages
- Composing prompt → model → parser
- Modern LangChain patterns

**Time**: 15 minutes

---

### 4️⃣ Master `sequential_chain.py`

**Why fourth**: Combines concepts into multi-step workflows.

**Run it**:
```bash
python sequential_chain.py
```

**What you'll learn**:
- Multi-step chain pipelines
- Variable mapping between steps
- Sequential processing patterns
- Output-to-input transformations

**Time**: 15 minutes

---

### Learning Path Summary Table

| Example | Difficulty | Concepts | Estimated Time |
|---------|-----------|----------|----------------|
| `simple_llm_chain.py` | ⭐ Beginner | Basic chains, LLM invocation | 5 minutes |
| `prompt_template_chain.py` | ⭐ Beginner | Templates, variables, formatting | 10 minutes |
| `lcel_basic_composition.py` | ⭐⭐ Intermediate | LCEL operators, composition, type flow | 15 minutes |
| `sequential_chain.py` | ⭐⭐ Intermediate | Multi-step pipelines, variable mapping | 15 minutes |
| **Total** | | **All basic concepts** | **45 minutes** |

## Usage Instructions

### Running Individual Examples

Execute any example directly with Python:

```bash
# General pattern
python <example_name>.py

# Specific examples
python simple_llm_chain.py
python prompt_template_chain.py
python lcel_basic_composition.py
python sequential_chain.py
```

### Expected Output Format

All examples follow this output structure:

```
========================================
[Example Title]
========================================

[Description of what's happening]

Input: [User input or parameters]

[Processing information]

Output: [Final result]

========================================
Example completed successfully!
========================================
```

### Example-Specific Expected Outputs

**simple_llm_chain.py**:
```
Response: [A joke about programming]
Modern LCEL Result: [Same joke using LCEL syntax]
```

**prompt_template_chain.py**:
```
Topic: artificial intelligence
Generated Output: [Essay or content about AI]
```

**lcel_basic_composition.py**:
```
Type Flow Demonstration:
  Input (dict) → PromptTemplate → ChatPromptValue → ChatModel → AIMessage → StrOutputParser → str
Final Result: [Parsed string output]
```

**sequential_chain.py**:
```
Step 1 Output: [First transformation result]
Step 2 Output: [Second transformation using step 1 output]
Final Result: [Complete pipeline result]
```

### Running All Examples

Use pytest to execute all examples as validation tests:

```bash
# Run all basic chain examples with verbose output
pytest . -v

# Run with detailed error information
pytest . -v --tb=short

# Run specific example as test
pytest simple_llm_chain.py -v
```

### Modifying Examples for Your Use Case

All examples are designed to be modified:

1. **Copy the example file**: Don't modify originals
2. **Change input data**: Replace example inputs with your data
3. **Adjust prompts**: Customize prompt templates for your use case
4. **Add error handling**: Extend with your specific error scenarios

Example modification:
```python
# Original
topic = "artificial intelligence"

# Your modification
topic = "your custom topic"
```

## Key Concepts Explained

### What is a Chain?

**Definition**: A Chain is a sequence of calls to an LLM or other components, encapsulating logic for processing inputs and producing outputs.

**Analogy**: Like a factory assembly line where each station performs a specific transformation.

**Core Characteristics**:
- **Reusable**: Define once, use many times
- **Composable**: Combine multiple chains into larger workflows
- **Configurable**: Parameterize behavior with inputs

**Example**:
```python
# Chain = Prompt Template + LLM + Output Processing
chain = prompt_template | llm | output_parser
result = chain.invoke({"topic": "AI"})
```

---

### What is LLMChain? (Deprecated Pattern)

**Definition**: LLMChain is a classic wrapper class combining a PromptTemplate and an LLM.

**Status**: **DEPRECATED** - Replaced by modern LCEL syntax

**Why learn it?**: Understanding legacy patterns helps when:
- Reading older LangChain code
- Migrating existing applications
- Understanding LangChain's evolution

**Modern Alternative**: Use LCEL pipe operator instead
```python
# Old (deprecated)
from langchain_classic.chains import LLMChain
chain = LLMChain(llm=model, prompt=prompt)

# New (recommended)
chain = prompt | model
```

**Source**: `libs/langchain/langchain_classic/chains/llm.py:15-25`

---

### What is PromptTemplate?

**Definition**: A template for formatting inputs with dynamic variables using Python string formatting syntax.

**Purpose**: Create reusable prompt patterns with variable substitution.

**Key Features**:
- **Variable placeholders**: `{variable_name}` syntax
- **Multiple variables**: Support any number of input variables
- **Type safety**: Validates that all variables are provided
- **Template validation**: Checks template syntax at creation

**Example**:
```python
from langchain_core.prompts import PromptTemplate

template = "Write a {length} essay about {topic}"
prompt = PromptTemplate.from_template(template)

# Variable substitution
formatted = prompt.format(length="short", topic="AI")
# Result: "Write a short essay about AI"
```

**Input Requirements**:
- All template variables must be provided
- Variables must be string-serializable
- Invalid variables raise `KeyError`

**Source**: `libs/core/langchain_core/prompts/prompt.py:50-75`

---

### What is LCEL (LangChain Expression Language)?

**Definition**: LCEL is a declarative syntax for composing LangChain components using the pipe operator (`|`).

**Philosophy**: "Unix pipes for LLM chains" - compose simple components into complex workflows.

**Operator**: `|` (pipe operator)

**Type Flow**: Each component in a pipe transforms input to output type
```
Dict[str, str] | PromptTemplate → ChatPromptValue
ChatPromptValue | ChatModel → AIMessage  
AIMessage | StrOutputParser → str
```

**Complete Example**:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Compose with pipe operator
chain = (
    ChatPromptTemplate.from_template("Tell me about {topic}")
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

# Execute
result = chain.invoke({"topic": "Python"})  # Returns: str
```

**Benefits**:
- **Clarity**: Visual pipeline structure
- **Type safety**: Type checking through composition
- **Streaming**: Automatic streaming support
- **Concurrency**: Built-in parallelization

**Source**: `libs/core/langchain_core/runnables/base.py:2100-2150`

---

### What is a Runnable?

**Definition**: The base protocol that all LCEL-compatible components implement.

**Key Methods**:
- `invoke(input)`: Synchronous execution
- `ainvoke(input)`: Asynchronous execution
- `batch(inputs)`: Process multiple inputs
- `stream(input)`: Stream output tokens

**All These Are Runnables**:
- PromptTemplate
- ChatModel (LLM)
- OutputParser
- Custom chains
- Any component using `|` operator

**Significance**: Runnables are **composable** - any Runnable can connect to any other Runnable with compatible types.

**Source**: `libs/core/langchain_core/runnables/base.py:1-50`

---

### What is SequentialChain?

**Definition**: A chain that executes multiple sub-chains in sequence, passing outputs from one chain as inputs to the next.

**Pattern**: Chain₁ → Chain₂ → Chain₃ → ... → Final Output

**Key Characteristics**:
- **Variable mapping**: Explicit mapping of output keys to input keys
- **Multi-step processing**: Each step can perform different transformations
- **Linear flow**: Steps execute in order, no branching

**Example Use Cases**:
- Generate content, then summarize it
- Analyze data, then create visualization description
- Translate text, then extract entities from translation

**Variable Mapping**:
```python
# Step 1 outputs: {"summary": "..."}
# Step 2 needs input: {"text": "..."}
# Mapping: summary → text
```

**Modern Alternative**: Use LCEL with explicit variable transformations
```python
chain = chain1 | transform_vars | chain2
```

**Source**: `libs/langchain/langchain_classic/chains/sequential.py:20-50`

---

### Concept Relationships Diagram

```
Chain (Abstract Concept)
├── LLMChain (Deprecated - Specific Implementation)
├── SequentialChain (Multi-step Pattern)
└── LCEL Composition (Modern Approach)
    ├── PromptTemplate (Runnable)
    ├── ChatModel (Runnable)
    └── OutputParser (Runnable)
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: `ModuleNotFoundError: No module named 'langchain_core'`

**Symptom**: Import fails when running examples

**Cause**: Dependencies not installed or virtual environment not activated

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import langchain_core; print('Success!')"
```

---

#### Issue: `ValueError: OPENAI_API_KEY not found`

**Symptom**: Example exits immediately with API key error

**Cause**: Environment variable not set or `.env` file missing

**Solution**:
```bash
# Check if .env file exists
ls -la .env

# If missing, copy from example
cp .env.example .env

# Edit with your API key
nano .env  # or use any text editor

# Verify environment variable loads
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY', 'NOT FOUND'))"
```

**Alternative**: Export directly in terminal session
```bash
export OPENAI_API_KEY=sk-your-key-here
python simple_llm_chain.py
```

---

#### Issue: `openai.RateLimitError: Rate limit exceeded`

**Symptom**: API call fails with rate limit message

**Cause**: Too many requests to OpenAI API in short time period

**Solutions**:

**Option 1 - Wait and Retry**:
```bash
# Wait 60 seconds, then retry
sleep 60
python simple_llm_chain.py
```

**Option 2 - Upgrade OpenAI Plan**: Visit [platform.openai.com/account/billing](https://platform.openai.com/account/billing)

**Option 3 - Add Retry Logic**: (Advanced users)
```python
# Add to your code
import time
from openai import RateLimitError

try:
    result = chain.invoke(input_data)
except RateLimitError:
    time.sleep(60)
    result = chain.invoke(input_data)
```

---

#### Issue: `ImportError: cannot import name 'LLMChain' from 'langchain_classic.chains'`

**Symptom**: Import fails for LLMChain class

**Cause**: Incorrect package or old LangChain version

**Solution**:
```bash
# Update langchain-classic package
pip install --upgrade langchain-classic

# Verify version (should be >=1.0.0)
python -c "import langchain_classic; print(langchain_classic.__version__)"
```

---

#### Issue: Python version error `requires Python >=3.10`

**Symptom**: Installation fails with Python version requirement error

**Cause**: Using Python 3.9 or earlier

**Solution**:
```bash
# Check current version
python --version

# Install Python 3.10+ using your system's package manager
# Ubuntu/Debian
sudo apt install python3.10 python3.10-venv

# macOS with Homebrew
brew install python@3.10

# Create virtual environment with correct Python version
python3.10 -m venv venv
source venv/bin/activate
```

---

#### Issue: `TypeError: argument of type 'NoneType' is not iterable`

**Symptom**: Cryptic error during chain execution

**Cause**: Missing required input variables in prompt template

**Solution**:
```python
# Check template variables
template = prompt.input_variables
print(f"Required variables: {template}")

# Ensure all variables provided
chain.invoke({"var1": "value1", "var2": "value2"})
```

**Prevention**: Always check `prompt.input_variables` before invoking

---

#### Issue: Examples run but produce no output

**Symptom**: Script completes but prints nothing

**Cause**: API key invalid or expired

**Solution**:
```bash
# Test API key manually
python -c "from openai import OpenAI; client = OpenAI(); print(client.models.list())"

# If error, regenerate API key at platform.openai.com
# Update .env file with new key
```

---

### Getting More Help

If issues persist after trying these solutions:

1. **Check LangChain Documentation**: [docs.langchain.com](https://docs.langchain.com/)
2. **Review API Reference**: See `docs/api-reference/` in repository
3. **Consult Debugging Guide**: See `examples/debugging/` for diagnostic tools
4. **Verify Environment**: Run `python --version` and `pip list` to check setup

## Next Steps

### After Completing Basic Chains

You're ready to explore more advanced patterns:

#### 1. Advanced Chains (`examples/advanced_chains/`)

Build production-ready patterns including:
- **Retrieval-Augmented Generation (RAG)**: Combine LLMs with document retrieval
- **Memory-Enabled Chains**: Add conversation history and context
- **Agent Systems**: LLMs that use tools and make decisions
- **Streaming**: Real-time token-by-token output
- **Async Execution**: Concurrent chain execution for performance
- **Error Recovery**: Fallback strategies for resilience

**When ready**: After mastering LCEL composition and sequential chains

**Path**: `cd ../advanced_chains && python retrieval_qa_chain.py`

---

#### 2. Type Patterns (`examples/type_patterns/`)

Learn type-safe LangChain development:
- **LCEL Type Annotations**: Explicit type flow documentation
- **Custom Chain Typing**: Building type-safe custom chains
- **Pydantic Integration**: Chain input/output validation

**When ready**: If using mypy or requiring type safety

**Path**: `cd ../type_patterns && python lcel_type_annotations.py`

---

#### 3. Debugging Utilities (`examples/debugging/`)

Master debugging techniques:
- **Logging Wrappers**: Granular execution monitoring
- **Callback Debuggers**: Inspect chain events
- **Chain Introspection**: Visualize chain structure

**When ready**: When troubleshooting complex chains

**Path**: `cd ../debugging && python logging_wrapper.py`

---

### Recommended Documentation

Deepen your understanding with comprehensive guides:

| Documentation | Path | Topics |
|--------------|------|--------|
| **LCEL Composition Guide** | `docs/guides/lcel-composition.md` | Advanced LCEL patterns, type safety, streaming |
| **Chain Types Guide** | `docs/guides/chain-types.md` | Choosing appropriate chain types for use cases |
| **API Reference - Chains** | `docs/api-reference/chains/` | Complete API documentation for all chain classes |
| **API Reference - Runnables** | `docs/api-reference/runnables/` | Runnable protocol and composition operators |
| **Architecture - Chain Lifecycle** | `docs/architecture/chain-lifecycle.md` | Deep dive into chain execution flow |

---

### Building Your First Real Chain

Apply what you've learned:

1. **Define your use case**: What problem are you solving?
2. **Choose chain pattern**: Simple (LCEL) or complex (Sequential)?
3. **Design prompt**: Create effective prompt templates
4. **Add error handling**: Implement retry and fallback logic
5. **Test thoroughly**: Validate with various inputs
6. **Monitor in production**: Add logging and observability

**Template to start**:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Your custom chain
chain = (
    ChatPromptTemplate.from_template("Your template: {input}")
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

# Execute
result = chain.invoke({"input": "your data"})
print(result)
```

## References

### Official Documentation

- **LangChain Official Docs**: [docs.langchain.com](https://docs.langchain.com/)
- **LangChain Python API Reference**: [reference.langchain.com/python](https://reference.langchain.com/python/langchain_classic/)
- **OpenAI API Documentation**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **OpenAI API Keys**: [platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)

### Repository Documentation

- **LCEL Guide**: `docs/guides/lcel-composition.md` - Comprehensive LCEL composition patterns
- **Chain API Reference**: `docs/api-reference/chains/` - Complete chain class documentation
- **Runnable API Reference**: `docs/api-reference/runnables/` - Runnable protocol details
- **Debugging Guide**: `docs/debugging/common-issues.md` - Top 10 failure modes and solutions

### Source Code References

Key source files for understanding basic chains:

- **Chain Base Class**: `libs/langchain/langchain_classic/chains/base.py:1-150`
- **LLMChain (Deprecated)**: `libs/langchain/langchain_classic/chains/llm.py:15-100`
- **SequentialChain**: `libs/langchain/langchain_classic/chains/sequential.py:20-150`
- **Runnable Protocol**: `libs/core/langchain_core/runnables/base.py:1-100`
- **PromptTemplate**: `libs/core/langchain_core/prompts/prompt.py:50-200`
- **LCEL Composition**: `libs/core/langchain_core/runnables/base.py:2100-2200`

### Learning Resources

- **LangChain Concepts**: `docs/getting-started/concepts.md` - Core concepts overview
- **Installation Guide**: `docs/getting-started/installation.md` - Detailed installation instructions
- **Configuration Guide**: `docs/getting-started/configuration.md` - Environment setup patterns
- **Main Examples README**: `examples/README.md` - Overview of all example categories

### Community and Support

- **GitHub Issues**: [github.com/langchain-ai/langchain/issues](https://github.com/langchain-ai/langchain/issues)
- **LangChain Discord**: Community support and discussions
- **Twitter**: [@LangChainAI](https://twitter.com/langchainai)

---

**Questions or Issues?** 
- Check the [Troubleshooting](#troubleshooting) section above
- Review `docs/debugging/common-issues.md` for detailed failure modes
- Consult the debugging utilities in `examples/debugging/`

**Ready to build?** Start with `python simple_llm_chain.py` and progress through the examples!
