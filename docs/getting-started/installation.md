# Installation

This guide covers everything you need to install LangChain and get started building LLM-powered applications.

## Introduction

LangChain follows a modular package structure that allows you to install only what you need:

- **`langchain-core`** - Contains base abstractions that power the LangChain ecosystem, including the Runnable protocol, LCEL (LangChain Expression Language), message types, and core interfaces. This is the foundation package required by all other LangChain packages.

- **`langchain-classic`** - Provides legacy chains, agents, memory implementations, and compatibility layers for existing applications. Contains the traditional Chain abstractions and pre-built components.

- **Provider integrations** - Separate packages for each LLM provider or service (e.g., `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`), allowing you to install only the integrations you need.

This modular structure is reflected in the monorepo's `libs/` directory structure, where each package is independently versioned and maintained.

**Source**: Based on repository structure in libs/core/ and libs/langchain/

## Prerequisites

### Python Version Requirements

LangChain requires **Python 3.10.0 or higher**. Both `langchain-core` and `langchain-classic` specify this requirement:

```toml
requires-python = ">=3.10.0,<4.0.0"
```

**Source**: libs/langchain/pyproject.toml:8 and libs/core/pyproject.toml:8

To check your current Python version:

```bash
python --version
```

If you need to upgrade Python, download the latest version from the [official Python downloads page](https://www.python.org/downloads/).

## Installation Methods

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is the fastest Python package installer and resolver, offering significantly better performance than pip while maintaining compatibility.

**Installing uv:**

```bash
# Install via pip
pip install uv

# Or install via shell script (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Installing LangChain packages with uv:**

```bash
# Install core package and OpenAI integration
uv pip install langchain-core langchain-openai

# Install classic package with additional integrations
uv pip install langchain-classic langchain-anthropic langchain-openai
```

**Benefits of using uv:**
- **Speed**: 10-100x faster package installation and dependency resolution
- **Reliability**: Better conflict resolution and dependency management
- **Compatibility**: Drop-in replacement for pip commands

**Source**: Agent Action Plan section 0.9.1 validation commands demonstrate uv usage patterns

### Using pip

pip is the standard Python package installer included with Python installations.

**Installing LangChain packages:**

```bash
# Install core package only
pip install langchain-core

# Install with OpenAI integration
pip install langchain-core langchain-openai

# Install classic package
pip install langchain-classic
```

**When to use pip vs uv:**
- Use **pip** if you're working in an environment where uv isn't available or your team uses pip exclusively
- Use **uv** for faster installations, especially in CI/CD pipelines or when managing complex dependency trees

**Upgrading packages:**

```bash
pip install --upgrade langchain-core langchain-openai
```

### Using poetry

Poetry is a dependency management and packaging tool ideal for projects with complex dependency requirements and reproducible builds.

**Adding LangChain to your project:**

```bash
# Add core and provider packages
poetry add langchain-core langchain-openai

# Add classic package
poetry add langchain-classic

# Add multiple integrations at once
poetry add langchain-core langchain-openai langchain-anthropic
```

**Integration with pyproject.toml:**

Poetry automatically adds dependencies to your `pyproject.toml`:

```toml
[tool.poetry.dependencies]
python = "^3.10"
langchain-core = "^1.0.0"
langchain-openai = "^0.3.0"
```

**Installing all dependencies:**

```bash
# Install from pyproject.toml and poetry.lock
poetry install

# Update dependencies
poetry update
```

## Core Packages

### langchain-core

The foundation package containing base abstractions used throughout the LangChain ecosystem.

**Installation:**

```bash
pip install langchain-core
```

**What's included:**
- **Runnable protocol** - The base interface for all LangChain components
- **LCEL (LangChain Expression Language)** - Declarative syntax for composing chains using the pipe operator (`|`)
- **Message types** - HumanMessage, AIMessage, SystemMessage, and related classes
- **Prompt templates** - PromptTemplate, ChatPromptTemplate, and template utilities
- **Output parsers** - Base classes for parsing LLM outputs
- **Callbacks** - Callback system for monitoring and debugging
- **Base interfaces** - Language model, embeddings, retriever, and vector store abstractions

**Version requirement**: `>=1.0.0,<2.0.0` (from libs/langchain/pyproject.toml:10)

**Source**: libs/core/README.md describes core abstractions

### langchain-classic

Legacy chains, agents, memory implementations, and the traditional Chain abstraction layer.

**Installation:**

```bash
pip install langchain-classic
```

**What's included:**
- **Classic chains** - LLMChain, SequentialChain, and specialized chain types
- **Agents** - Agent executors and pre-built agent implementations
- **Memory** - ConversationBufferMemory, ConversationBufferWindowMemory, and memory interfaces
- **Tools** - Base tool classes and tool integration utilities
- **Legacy components** - Deprecated functionality maintained for backward compatibility

**Note**: In most modern applications, you should use the main `langchain` package and LCEL patterns. `langchain-classic` is primarily for maintaining existing applications using legacy chain patterns.

**Version**: 1.0.0 (from libs/langchain/pyproject.toml:20)

**Source**: libs/langchain/README.md describes classic package contents

## Provider Integrations

LangChain uses separate packages for each LLM provider or service integration. Install only the providers you need:

### OpenAI (GPT Models)

```bash
pip install langchain-openai
```

Provides integrations for OpenAI's GPT models, embeddings, and chat models.

### Anthropic (Claude Models)

```bash
pip install langchain-anthropic
```

Provides integrations for Anthropic's Claude models.

### Google AI

```bash
# Google Generative AI (Gemini)
pip install langchain-google-genai

# Google Vertex AI
pip install langchain-google-vertexai
```

### Other Popular Providers

```bash
# Cohere
pip install langchain-cohere

# Hugging Face
pip install langchain-huggingface

# Ollama (local models)
pip install langchain-ollama

# Groq
pip install langchain-groq

# Fireworks
pip install langchain-fireworks

# Together AI
pip install langchain-together

# Mistral AI
pip install langchain-mistralai

# AWS Bedrock
pip install langchain-aws

# DeepSeek
pip install langchain-deepseek

# xAI (Grok)
pip install langchain-xai

# Perplexity
pip install langchain-perplexity
```

**Complete provider list**: See libs/partners/ directory structure for all available integrations

**Source**: libs/langchain/pyproject.toml:24-41 lists optional provider dependencies

## Additional Components

### Text Splitters

For document chunking and text processing:

```bash
pip install langchain-text-splitters
```

Required dependency for `langchain-classic`: `>=1.0.0,<2.0.0` (from libs/langchain/pyproject.toml:11)

### Vector Store Integrations

Install vector store integrations as needed:

```bash
# Chroma
pip install langchain-chroma

# Pinecone
pip install langchain-pinecone

# Weaviate
pip install langchain-weaviate

# Qdrant
pip install langchain-qdrant

# FAISS
pip install langchain-community  # Includes FAISS integration
```

### Environment Variable Management

For managing API keys and configuration:

```bash
pip install python-dotenv
```

**Usage pattern:**

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file
api_key = os.getenv("OPENAI_API_KEY")
```

## Verification

After installation, verify that LangChain is installed correctly:

### Check Installed Version

```bash
python -c "import langchain_core; print(langchain_core.__version__)"
```

**Expected output format**: `1.0.1` (or your installed version)

### Test Core Imports

```bash
python -c "from langchain_core.runnables import Runnable; print('LCEL import successful')"
```

**Expected output**: `LCEL import successful`

### Verify Provider Integration

```bash
# For OpenAI
python -c "from langchain_openai import ChatOpenAI; print('OpenAI integration available')"

# For Anthropic
python -c "from langchain_anthropic import ChatAnthropic; print('Anthropic integration available')"
```

## Version Management

### Checking Installed Versions

List all installed LangChain packages and their versions:

```bash
pip list | grep langchain
```

**Example output:**
```
langchain-anthropic    0.3.22
langchain-classic      1.0.0
langchain-core         1.0.1
langchain-openai       0.3.35
langchain-text-splitters 1.0.0
```

### Semantic Versioning

LangChain packages follow [semantic versioning](https://semver.org/) with the format `MAJOR.MINOR.PATCH`:

- **MAJOR** version changes indicate breaking API changes
- **MINOR** version changes add functionality in a backward-compatible manner
- **PATCH** version changes include backward-compatible bug fixes

**Important**: Breaking changes only occur in major version bumps (e.g., 1.x.x → 2.0.0). The LangChain team communicates these changes with advance notice.

**Source**: Referenced in libs/langchain/README.md:30-31 and libs/core/README.md:39-40 versioning policies

### Pinning Versions for Production

For production deployments, pin exact versions in your requirements file to ensure reproducibility:

```txt
# requirements.txt
langchain-core==1.0.1
langchain-openai==0.3.35
langchain-anthropic==0.3.22
python-dotenv==1.0.0
```

Install pinned versions:

```bash
pip install -r requirements.txt
```

### Using uv.lock for Reproducible Installs

If using `uv`, the `uv.lock` file ensures exact reproducibility across all environments:

```bash
# Generate uv.lock (if not present)
uv pip compile requirements.txt -o uv.lock

# Install from lock file
uv pip sync uv.lock
```

The monorepo includes a `uv.lock` file at the root for development consistency.

**Source**: uv.lock file presence in repository root

## Troubleshooting

### Python Version Mismatch

**Problem**: Error message indicating Python version is too old.

```
ERROR: Package requires Python '>=3.10.0' but the running Python is 3.9.x
```

**Solution**: Upgrade to Python 3.10 or higher:

1. Download Python from [python.org/downloads](https://www.python.org/downloads/)
2. Install the new version
3. Create a new virtual environment with the updated Python:

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Conflicting Dependencies

**Problem**: pip reports conflicting dependency requirements.

```
ERROR: Cannot install langchain-core and package-x because these package versions have conflicting dependencies.
```

**Solution**: Use a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows

# Install packages in isolated environment
pip install langchain-core langchain-openai
```

**Alternative**: Use `uv` for better dependency resolution:

```bash
uv pip install langchain-core langchain-openai
```

### Import Errors for Provider Packages

**Problem**: `ModuleNotFoundError` when trying to import a provider integration.

```python
>>> from langchain_openai import ChatOpenAI
ModuleNotFoundError: No module named 'langchain_openai'
```

**Solution**: Install the missing provider integration:

```bash
pip install langchain-openai
```

**Common provider packages**:
- `langchain-openai` for OpenAI models
- `langchain-anthropic` for Claude models
- `langchain-google-genai` for Gemini models

### Permission Errors on System Python

**Problem**: Permission denied when installing packages.

```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied
```

**Solution**: Use a virtual environment (recommended) or install with `--user` flag:

```bash
# Recommended: Use virtual environment
python -m venv .venv
source .venv/bin/activate
pip install langchain-core

# Alternative: Install to user directory
pip install --user langchain-core
```

**Never use `sudo pip install`** - this can break your system Python installation.

### Package Installation Hangs

**Problem**: `pip install` appears to hang or takes extremely long.

**Solution**: Use `uv` for faster installation:

```bash
pip install uv
uv pip install langchain-core langchain-openai
```

Or increase pip's timeout and use verbose mode to diagnose:

```bash
pip install --timeout 300 -v langchain-core
```

## Next Steps

Now that you have LangChain installed, continue with:

- **[Configuration](configuration.md)** - Set up API keys and environment variables
- **[Quickstart](quickstart.md)** - Build your first LangChain application in 5 minutes
- **[Core Concepts](concepts.md)** - Understand Chains, Runnables, LCEL, Messages, and Callbacks

## Additional Resources

- **Official Documentation**: [https://docs.langchain.com/](https://docs.langchain.com/)
- **API Reference (langchain-core)**: [https://reference.langchain.com/python/langchain_core/](https://reference.langchain.com/python/langchain_core/)
- **API Reference (langchain-classic)**: [https://reference.langchain.com/python/langchain_classic/](https://reference.langchain.com/python/langchain_classic/)
- **GitHub Repository**: [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- **Community Forum**: [https://forum.langchain.com](https://forum.langchain.com)
- **LangSmith** (Observability & Testing): [https://smith.langchain.com](https://smith.langchain.com)
