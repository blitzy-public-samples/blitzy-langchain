# Configuration

This guide covers how to configure LangChain applications, including environment variable setup for API keys, optional dependency configuration, runtime settings, and observability tools like LangSmith tracing.

## Introduction

Proper configuration is essential for LangChain applications to function correctly. Configuration manages:

- **API Authentication**: Credentials for LLM providers (OpenAI, Anthropic, Google, etc.)
- **Optional Features**: Integration-specific settings for vector stores, embeddings, and specialized tools
- **Observability**: LangSmith tracing for debugging and monitoring chain execution
- **Runtime Behavior**: Logging levels, timeout settings, and other operational parameters

LangChain automatically detects environment variables for common settings, allowing you to configure applications without hardcoding sensitive credentials. This approach enables secure configuration management across development, testing, and production environments.

**Source**: Configuration patterns based on libs/langchain/pyproject.toml and libs/core/pyproject.toml dependency structure

## API Key Configuration

LangChain integrations require API keys to authenticate with LLM providers. The framework automatically detects these keys from environment variables when initializing model clients.

### Required Environment Variables

The specific environment variables you need depend on which LLM providers you're using:

| Provider | Environment Variable | Integration Package |
|----------|---------------------|---------------------|
| OpenAI | `OPENAI_API_KEY` | `langchain-openai` |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | `langchain-anthropic` |
| Google Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS` | `langchain-google-vertexai` |
| Google Generative AI | `GOOGLE_API_KEY` | `langchain-google-genai` |
| Fireworks AI | `FIREWORKS_API_KEY` | `langchain-fireworks` |
| Together AI | `TOGETHER_API_KEY` | `langchain-together` |
| Mistral AI | `MISTRAL_API_KEY` | `langchain-mistralai` |
| Groq | `GROQ_API_KEY` | `langchain-groq` |
| DeepSeek | `DEEPSEEK_API_KEY` | `langchain-deepseek` |
| xAI | `XAI_API_KEY` | `langchain-xai` |
| Perplexity AI | `PERPLEXITY_API_KEY` | `langchain-perplexity` |

**Source**: Provider integrations listed in libs/langchain/pyproject.toml:24-41

### How LangChain Detects Environment Variables

When you instantiate a model client, LangChain integration packages automatically check for the corresponding environment variable:

```python
from langchain_openai import ChatOpenAI

# Automatically uses OPENAI_API_KEY from environment
llm = ChatOpenAI(model="gpt-4")
```

The environment variable detection happens transparently through the integration package. You can also explicitly pass API keys:

```python
# Explicitly pass API key (overrides environment variable)
llm = ChatOpenAI(model="gpt-4", api_key="your-api-key-here")
```

**Best Practice**: Use environment variables for API keys rather than hardcoding them, especially when committing code to version control.

### Creating a .env File

The recommended approach for managing environment variables during development is using a `.env` file in your project root:

```bash
# .env file example
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: LangSmith configuration (see LangSmith section below)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=my-project-name
```

**Important Security Practices**:
- Never commit `.env` files to version control
- Add `.env` to your `.gitignore` file
- Use `.env.example` (without actual secrets) to document required variables
- Rotate API keys regularly, especially if exposed

Example `.env.example` file for your repository:

```bash
# Required: OpenAI API key for GPT models
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Anthropic API key for Claude models
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: LangSmith tracing (recommended for debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=your-project-name
```

## Using python-dotenv

The `python-dotenv` package loads environment variables from `.env` files into your Python environment, making configuration management seamless during development.

### Installation

```bash
# Using uv (recommended)
uv pip install python-dotenv

# Using pip
pip install python-dotenv
```

**Source**: python-dotenv listed in test_integration dependencies in libs/langchain/pyproject.toml:84

### Loading Environment Variables

Add this code at the top of your application entry point (before importing LangChain):

```python
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now LangChain can access the variables
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")  # Uses OPENAI_API_KEY from .env
```

### .env File Placement

Place your `.env` file in the project root directory (the directory where you run your Python scripts). The `load_dotenv()` function automatically searches:

1. Current working directory for `.env`
2. Parent directories if not found in current directory
3. You can specify an explicit path: `load_dotenv(".env.production")`

### Security Reminder

Always add `.env` to your `.gitignore` file:

```bash
# .gitignore
.env
.env.local
.env.*.local
```

This prevents accidentally committing sensitive credentials to version control.

## LangSmith Configuration

[LangSmith](https://smith.langchain.com/) is LangChain's observability platform for debugging, testing, and monitoring LLM applications. It provides detailed traces of chain execution, including inputs, outputs, token usage, and latency for every step.

### Why Use LangSmith

LangSmith helps you:
- **Debug complex chains**: See exactly what happens at each step of chain execution
- **Monitor production applications**: Track performance metrics and identify bottlenecks
- **Test and evaluate**: Compare different prompts, models, and chain configurations
- **Understand costs**: Monitor token usage across different models and chains

### Enabling LangSmith Tracing

Configure LangSmith using environment variables:

```bash
# Required: Enable tracing
LANGCHAIN_TRACING_V2=true

# Required: Your LangSmith API key
LANGCHAIN_API_KEY=ls__xxxxxxxxxxxxxxxxxxxxxxxx

# Optional: Organize traces by project
LANGCHAIN_PROJECT=my-application-name

# Optional: Set custom endpoint (defaults to LangSmith cloud)
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Getting a LangSmith API Key

1. Sign up for a free account at [smith.langchain.com](https://smith.langchain.com/)
2. Navigate to Settings → API Keys
3. Create a new API key
4. Add the key to your `.env` file as `LANGCHAIN_API_KEY`

### Using LangSmith in Your Code

Once configured, LangSmith automatically traces all chain executions:

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Loads LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY

# Create a simple chain
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | StrOutputParser()

# This invocation is automatically traced in LangSmith
result = chain.invoke({"topic": "programming"})
print(result)
```

Visit the LangSmith dashboard to see the trace with:
- Input parameters
- Prompt template rendering
- LLM request and response
- Output parsing
- Timing and token usage

**Source**: LangSmith observability mentioned in repository documentation

## Optional Dependencies

LangChain uses a modular package structure where you install only the integrations you need. This keeps installations lightweight and allows precise dependency management.

### Common Optional Packages

Based on your use case, you may need to install additional packages:

#### LLM Provider Integrations

```bash
# OpenAI (GPT-3.5, GPT-4, embeddings)
uv pip install langchain-openai

# Anthropic (Claude models)
uv pip install langchain-anthropic

# Google Vertex AI
uv pip install langchain-google-vertexai

# Google Generative AI (Gemini)
uv pip install langchain-google-genai

# Fireworks AI
uv pip install langchain-fireworks

# Ollama (local models)
uv pip install langchain-ollama

# Together AI
uv pip install langchain-together

# Mistral AI
uv pip install langchain-mistralai

# Hugging Face
uv pip install langchain-huggingface

# Groq
uv pip install langchain-groq

# AWS Bedrock
uv pip install langchain-aws

# DeepSeek
uv pip install langchain-deepseek

# xAI
uv pip install langchain-xai

# Perplexity AI
uv pip install langchain-perplexity
```

**Source**: Optional dependencies in libs/langchain/pyproject.toml:24-41

#### Vector Store Integrations

Vector stores are commonly used for retrieval-augmented generation (RAG):

```bash
# Chroma (lightweight, local vector store)
uv pip install langchain-chroma chromadb

# Pinecone (managed cloud vector database)
uv pip install langchain-pinecone

# FAISS (Facebook AI Similarity Search)
uv pip install langchain-community faiss-cpu
# Or for GPU support: faiss-gpu

# Weaviate (open-source vector database)
uv pip install langchain-weaviate

# Qdrant (vector search engine)
uv pip install langchain-qdrant
```

#### Document Loaders and Utilities

```bash
# PDF document loading
uv pip install pypdf

# Web scraping and HTML parsing
uv pip install beautifulsoup4 requests

# Markdown parsing
uv pip install markdown

# Environment variable management
uv pip install python-dotenv
```

### When Each Package Is Needed

| Use Case | Required Packages |
|----------|------------------|
| Basic chain with OpenAI | `langchain-core`, `langchain-openai` |
| RAG with Chroma vector store | `langchain-core`, `langchain-openai`, `langchain-chroma`, `chromadb` |
| Multi-provider support | `langchain-core`, `langchain-openai`, `langchain-anthropic` |
| Document Q&A from PDFs | `langchain-core`, `langchain-openai`, `langchain-chroma`, `pypdf` |
| Legacy chains and agents | `langchain-classic`, `langchain-openai` |

### Core vs Classic Packages

- **`langchain-core`**: Modern LCEL-based chains using the pipe operator (`|`)
- **`langchain-classic`**: Legacy Chain classes (LLMChain, SequentialChain, etc.)

Most new projects should use `langchain-core` with LCEL composition. Use `langchain-classic` only for:
- Maintaining existing codebases using legacy Chain classes
- Accessing pre-built chains and agents not yet ported to LCEL
- Compatibility with older tutorials and documentation

**Reference**: See [installation.md](installation.md) for detailed package installation instructions

## Configuration Loading Patterns

### Direct Environment Variable Access

You can access environment variables directly in your code using Python's `os` module:

```python
import os
from langchain_openai import ChatOpenAI

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Use the API key
llm = ChatOpenAI(model="gpt-4", api_key=api_key)
```

### Passing API Keys to Model Constructors

LangChain model classes accept API keys as constructor parameters:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Explicit API key (overrides environment variable)
openai_llm = ChatOpenAI(
    model="gpt-4",
    api_key="your-openai-key"
)

anthropic_llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key="your-anthropic-key"
)
```

### Configuration Precedence

LangChain follows this priority order for configuration:

1. **Explicit constructor parameters** (highest priority)
2. **Environment variables** (automatic detection)
3. **Default values** (lowest priority)

Example demonstrating precedence:

```python
import os
from langchain_openai import ChatOpenAI

# Environment variable
os.environ["OPENAI_API_KEY"] = "env-key"

# This uses the environment variable
llm1 = ChatOpenAI(model="gpt-4")

# This overrides the environment variable
llm2 = ChatOpenAI(model="gpt-4", api_key="explicit-key")
```

### Dynamic Configuration

For applications that need to switch between configurations:

```python
import os
from langchain_openai import ChatOpenAI

class LLMFactory:
    """Factory for creating LLMs with different configurations."""
    
    @staticmethod
    def create_llm(environment="development"):
        """Create an LLM based on environment.
        
        Args:
            environment: One of "development", "staging", "production"
            
        Returns:
            Configured ChatOpenAI instance
        """
        if environment == "production":
            # Production uses GPT-4 with stricter settings
            return ChatOpenAI(
                model="gpt-4",
                temperature=0.3,
                max_tokens=2000,
                api_key=os.getenv("OPENAI_API_KEY_PROD")
            )
        elif environment == "staging":
            # Staging uses GPT-3.5 for cost savings
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.5,
                api_key=os.getenv("OPENAI_API_KEY_STAGING")
            )
        else:
            # Development uses GPT-3.5 with higher temperature
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY_DEV")
            )

# Usage
llm = LLMFactory.create_llm(environment="production")
```

## Development vs Production

Configuration requirements differ between development and production environments. Follow these best practices for secure and maintainable configuration management.

### Development Environment

**Best Practices**:

1. **Use `.env` files with `python-dotenv`**:
   ```python
   from dotenv import load_dotenv
   
   load_dotenv()  # Load from .env file
   ```

2. **Keep `.env` files local** (never commit):
   ```bash
   # .gitignore
   .env
   .env.local
   ```

3. **Provide `.env.example` template**:
   ```bash
   # .env.example (committed to repo)
   OPENAI_API_KEY=your-key-here
   ANTHROPIC_API_KEY=your-key-here
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-langsmith-key-here
   ```

4. **Use LangSmith tracing** for debugging:
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=dev-my-app
   ```

5. **Enable verbose logging**:
   ```python
   import logging
   
   logging.basicConfig(level=logging.DEBUG)
   ```

### Production Environment

**Best Practices**:

1. **Use platform-native secret management**:
   - **Kubernetes**: Use Secrets and ConfigMaps
   - **AWS**: Use Systems Manager Parameter Store or Secrets Manager
   - **Azure**: Use Key Vault
   - **Google Cloud**: Use Secret Manager
   - **Docker**: Use Docker secrets
   - **Heroku**: Use Config Vars

2. **Never hardcode credentials**:
   ```python
   # ❌ NEVER do this
   llm = ChatOpenAI(api_key="sk-proj-hardcoded-key")
   
   # ✅ DO this
   llm = ChatOpenAI()  # Uses environment variable
   ```

3. **Set environment variables at deployment time**:
   ```bash
   # Kubernetes example
   kubectl create secret generic langchain-secrets \
     --from-literal=OPENAI_API_KEY=your-key \
     --from-literal=ANTHROPIC_API_KEY=your-key
   ```

4. **Minimize LangSmith tracing** (or sample requests):
   ```bash
   # Only trace errors or sample 10% of requests
   LANGCHAIN_TRACING_V2=false  # Enable selectively
   ```

5. **Use read-only API keys** where possible

6. **Implement key rotation procedures**

7. **Monitor for unauthorized API usage**

### Environment-Specific Configuration Example

```python
import os
from langchain_openai import ChatOpenAI

# Determine environment
ENV = os.getenv("ENVIRONMENT", "development")

# Configure based on environment
if ENV == "production":
    # Production: Disable tracing, use production keys
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.3,
        max_retries=3,
        request_timeout=30
    )
elif ENV == "staging":
    # Staging: Enable tracing, use staging keys
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "staging-app"
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_retries=2
    )
else:
    # Development: Enable tracing, verbose logging
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "dev-app"
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_retries=1
    )
```

### Security Checklist

- [ ] API keys stored in environment variables or secret management system
- [ ] `.env` files added to `.gitignore`
- [ ] No hardcoded credentials in code
- [ ] API keys rotated regularly (quarterly minimum)
- [ ] Production keys separate from development/staging keys
- [ ] Monitoring for unusual API usage patterns
- [ ] Access to production secrets restricted to necessary personnel
- [ ] Secrets encrypted at rest and in transit

## Complete .env.example Template

Copy this template to create your own `.env` file:

```bash
################################################################################
# LangChain Configuration Template
# 
# Usage:
#   1. Copy this file to .env: cp .env.example .env
#   2. Fill in your actual API keys and configuration values
#   3. Never commit the .env file to version control
#
# Important: Add .env to your .gitignore file
################################################################################

################################################################################
# LLM Provider API Keys
# Obtain keys from provider dashboards:
# - OpenAI: https://platform.openai.com/api-keys
# - Anthropic: https://console.anthropic.com/settings/keys
# - Google: https://console.cloud.google.com/
################################################################################

# OpenAI API key (required for GPT models)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx

# Anthropic API key (required for Claude models)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx

# Google API key (for Gemini models)
GOOGLE_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxx

# Google Application Credentials (for Vertex AI)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Other LLM Providers (uncomment if needed)
# FIREWORKS_API_KEY=your-fireworks-key
# TOGETHER_API_KEY=your-together-key
# MISTRAL_API_KEY=your-mistral-key
# GROQ_API_KEY=your-groq-key
# DEEPSEEK_API_KEY=your-deepseek-key
# XAI_API_KEY=your-xai-key
# PERPLEXITY_API_KEY=your-perplexity-key

################################################################################
# LangSmith Configuration (Observability)
# Sign up for free at: https://smith.langchain.com/
################################################################################

# Enable LangSmith tracing (true/false)
LANGCHAIN_TRACING_V2=true

# LangSmith API key
LANGCHAIN_API_KEY=ls__xxxxxxxxxxxxxxxxxxxxxxxx

# LangSmith project name (organizes traces)
LANGCHAIN_PROJECT=my-project-name

# LangSmith endpoint (optional, defaults to cloud)
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

################################################################################
# Vector Store Configuration (if using)
################################################################################

# Pinecone
# PINECONE_API_KEY=your-pinecone-key
# PINECONE_ENVIRONMENT=your-environment

# Weaviate
# WEAVIATE_URL=https://your-instance.weaviate.network
# WEAVIATE_API_KEY=your-weaviate-key

# Qdrant
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=your-qdrant-key

################################################################################
# Application Configuration
################################################################################

# Environment (development, staging, production)
ENVIRONMENT=development

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Application-specific settings
# APP_NAME=my-langchain-app
# MAX_RETRIES=3
# REQUEST_TIMEOUT=30
```

## Troubleshooting

### ImportError: Missing Optional Dependencies

**Symptom**: Import errors when trying to use specific integrations:

```python
ImportError: langchain-openai is not installed. 
Please install it with `pip install langchain-openai`
```

**Solution**: Install the required integration package:

```bash
# Using uv
uv pip install langchain-openai

# Using pip
pip install langchain-openai
```

**Diagnostic Steps**:
1. Check installed packages: `pip list | grep langchain`
2. Verify package versions: `pip show langchain-openai`
3. Install missing packages from [Optional Dependencies](#optional-dependencies) section

### Authentication Errors: Missing or Invalid API Keys

**Symptom**: Authentication failures when initializing model clients:

```python
AuthenticationError: Incorrect API key provided
```

or

```python
ValueError: OPENAI_API_KEY not found in environment
```

**Solution**: Verify environment variables are set:

```bash
# Check if variable is set
echo $OPENAI_API_KEY

# Or in Python
import os
print(os.getenv("OPENAI_API_KEY"))
```

**Diagnostic Steps**:

1. **Verify .env file exists** in project root
2. **Check .env file is loaded**:
   ```python
   from dotenv import load_dotenv
   
   # Returns True if .env file was found and loaded
   loaded = load_dotenv()
   print(f"Loaded .env: {loaded}")
   ```

3. **Verify API key format**:
   - OpenAI keys start with `sk-proj-` or `sk-`
   - Anthropic keys start with `sk-ant-`
   - Check for extra whitespace or quotes

4. **Test API key validity** with a simple request:
   ```python
   from langchain_openai import ChatOpenAI
   
   try:
       llm = ChatOpenAI(model="gpt-3.5-turbo")
       response = llm.invoke("Say hello")
       print("API key valid:", response.content)
   except Exception as e:
       print(f"API key invalid: {e}")
   ```

5. **Regenerate API key** if necessary from provider dashboard

### LangSmith Tracing Not Working

**Symptom**: Traces don't appear in LangSmith dashboard

**Solution**: Verify LangSmith configuration:

```python
import os

# Check LangSmith environment variables
print("Tracing enabled:", os.getenv("LANGCHAIN_TRACING_V2"))
print("API key set:", bool(os.getenv("LANGCHAIN_API_KEY")))
print("Project:", os.getenv("LANGCHAIN_PROJECT"))
```

**Diagnostic Steps**:

1. **Confirm LANGCHAIN_TRACING_V2=true** (not "True" or "1")
2. **Verify LANGCHAIN_API_KEY** is set and valid
3. **Check network connectivity** to LangSmith API
4. **Wait a few seconds** for traces to appear in dashboard
5. **Check LangSmith status page** for service issues

### Environment Variables Not Loading

**Symptom**: Environment variables work in shell but not in Python

**Solution**: Ensure `load_dotenv()` is called before importing LangChain:

```python
# ✅ Correct order
from dotenv import load_dotenv
load_dotenv()  # Must be BEFORE LangChain imports

from langchain_openai import ChatOpenAI

# ❌ Wrong order
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()  # Too late, already imported
```

**Diagnostic Steps**:

1. **Verify .env file location** (should be in project root)
2. **Check file permissions** (must be readable)
3. **Use absolute path** if needed:
   ```python
   from pathlib import Path
   env_path = Path(__file__).parent / '.env'
   load_dotenv(dotenv_path=env_path)
   ```
4. **Check for syntax errors** in .env file (no quotes needed for values)

### Different Behavior Between Environments

**Symptom**: Code works locally but fails in production

**Solution**: Ensure consistent environment variable naming and values:

1. **List all LangChain-related variables**:
   ```python
   import os
   
   langchain_vars = {k: v for k, v in os.environ.items() 
                     if 'LANGCHAIN' in k or 'API_KEY' in k}
   print(langchain_vars)
   ```

2. **Verify production environment variables** are set in deployment platform
3. **Check for variable name typos** (e.g., `OPEN_AI_KEY` vs `OPENAI_API_KEY`)
4. **Ensure secrets are properly encoded** (no newlines, special characters)

## Next Steps

Now that you've configured your LangChain environment:

1. **[Quickstart Guide](quickstart.md)**: Build your first LangChain application in 5 minutes
2. **[LCEL Composition Guide](../guides/lcel-composition.md)**: Learn to compose chains using LangChain Expression Language
3. **[API Reference](../api-reference/chains/base.md)**: Explore detailed API documentation for chains and runnables

## Additional Resources

- **[Installation Guide](installation.md)**: Complete package installation instructions
- **[LangSmith Documentation](https://docs.smith.langchain.com/)**: Learn more about observability and tracing
- **[Security Best Practices](https://docs.langchain.com/security)**: Official security guidelines
- **[Environment Variables Reference](https://docs.langchain.com/configuration)**: Complete list of supported environment variables
