# Logging Configuration Guide

## Overview

Effective logging is critical for debugging LangChain chains, monitoring production systems, and understanding chain execution behavior. This guide covers logging strategies for both development and production environments, including granular execution monitoring through LangChain's callback system.

**Key Topics Covered:**
- Logging levels and when to use each
- Development vs production logging strategies
- Module-specific logger configuration
- Callback-based logging for granular execution monitoring
- Structured logging patterns
- Integration with observability platforms
- Performance considerations

**Prerequisites:**
- Understanding of Python's `logging` module
- Familiarity with LangChain chains and callbacks
- Basic knowledge of log aggregation tools (optional)

## Logging Levels and Use Cases

LangChain integrations support standard Python logging levels. Understanding when to use each level is essential for effective debugging and monitoring.

### DEBUG Level

**When to Use:**
- Development and debugging
- Step-by-step chain execution tracing
- Investigating type flow through LCEL pipes
- Troubleshooting unexpected behavior

**What Gets Logged:**
- Chain inputs with full data structures
- Intermediate outputs between chain steps
- Type transformations in LCEL compositions
- Callback event details (on_chain_start, on_llm_start, etc.)
- Memory load/save operations
- Variable substitutions in prompt templates

**Example Log Entry:**
```
DEBUG:langchain_classic.chains.base:Chain.__call__ inputs: {'input': 'What is AI?', 'history': []}
DEBUG:langchain_core.prompts.chat:Rendering template with variables: {'input': 'What is AI?'}
DEBUG:langchain_classic.chains.base:Chain.__call__ outputs: {'output': 'Artificial intelligence...'}
```

**Source:** `libs/langchain/langchain_classic/chains/base.py:43`

### INFO Level

**When to Use:**
- Production monitoring
- High-level operation tracking
- Performance metrics collection
- Audit trails

**What Gets Logged:**
- Chain invocation start/end
- LLM API calls (without full payloads)
- Retrieval operations
- Agent decisions (tool selection)
- Execution timings
- Token usage statistics

**Example Log Entry:**
```
INFO:langchain_classic.chains.base:Chain invoked: run_id=abc123, tags=['qa_chain']
INFO:langchain_classic.agents.agent:Agent selected tool: search, reason="Need current information"
INFO:langchain_classic.chains.base:Chain completed in 2.3s, tokens_used=450
```

### WARNING Level

**When to Use:**
- Retry/fallback scenarios
- Deprecated API usage
- Rate limiting encountered
- Context window approaching limits
- Recoverable errors

**What Gets Logged:**
- Retry attempts with backoff information
- Fallback chain activations
- API rate limit warnings
- Token count approaching model limits
- Deprecated feature usage

**Example Log Entry:**
```
WARNING:langchain_core.runnables.base:Retry attempt 2/3 for LLM call, backoff=4s
WARNING:langchain_classic.chains.llm:LLMChain is deprecated, use LCEL: prompt | llm | parser
WARNING:langchain_openai.chat_models:Rate limit warning: 80% of quota used
```

### ERROR Level

**When to Use:**
- Chain execution failures
- Unhandled exceptions
- API errors
- Validation failures
- Configuration issues

**What Gets Logged:**
- Full exception stack traces
- Chain execution context at failure
- Input data that caused failure
- API error responses
- Configuration details for troubleshooting

**Example Log Entry:**
```
ERROR:langchain_classic.chains.base:Chain execution failed: run_id=abc123
ERROR:langchain_classic.chains.base:Exception: ValidationError
ERROR:langchain_classic.chains.base:Input: {'query': None}
ERROR:langchain_classic.chains.base:Traceback: ...
```

**Source:** `libs/langchain/langchain_classic/chains/base.py` (error handling patterns)

## Development Configuration

Development logging should prioritize visibility and detail over performance. The goal is to understand exactly what's happening during chain execution.

### Basic Development Setup

**Configuration Approach:**
- Set DEBUG level for LangChain modules
- Use console handler with detailed formatting
- Include timestamp, logger name, level, and message
- Log full input/output structures

**Python Code Example:**
```python
import logging

# Configure root logger for LangChain modules
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set specific levels for LangChain components
logging.getLogger('langchain_classic').setLevel(logging.DEBUG)
logging.getLogger('langchain_core').setLevel(logging.DEBUG)
logging.getLogger('langchain_openai').setLevel(logging.INFO)  # Less verbose for API client
```

### Advanced Development Configuration

**Using dictConfig for Fine-Grained Control:**

```python
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'langchain_debug.log',
            'mode': 'a'
        }
    },
    'loggers': {
        'langchain_classic': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'langchain_core': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'langchain_classic.chains': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'langchain_core.runnables': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

# Apply configuration
logging.config.dictConfig(LOGGING_CONFIG)
```

### Logging Chain Inputs and Outputs

**Pattern for Logging Chain Data:**

```python
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def log_chain_execution(chain_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """Log chain execution details for debugging.
    
    Args:
        chain_name: Name of the chain being executed
        inputs: Input dictionary passed to the chain
        outputs: Output dictionary returned by the chain
    """
    logger.debug(f"Chain '{chain_name}' execution:")
    logger.debug(f"  Inputs: {inputs}")
    logger.debug(f"  Outputs: {outputs}")
    
    # Log specific keys if present
    if 'input' in inputs:
        logger.debug(f"  Query: {inputs['input'][:100]}...")  # Truncate long inputs
    if 'output' in outputs:
        logger.debug(f"  Response: {outputs['output'][:100]}...")
```

### Type Flow Logging for LCEL

**Logging Type Transformations in LCEL Pipes:**

```python
import logging
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

def log_type_flow(step_name: str):
    """Create a logging runnable for LCEL type flow visualization.
    
    Args:
        step_name: Name of the pipeline step for logging
    
    Returns:
        RunnableLambda that logs data and passes it through
    """
    def _log_and_pass(data):
        logger.debug(f"LCEL Step '{step_name}':")
        logger.debug(f"  Type: {type(data).__name__}")
        logger.debug(f"  Value: {str(data)[:200]}...")  # Truncate for readability
        return data
    
    return RunnableLambda(_log_and_pass)

# Usage in LCEL chain:
# chain = prompt | log_type_flow("after_prompt") | llm | log_type_flow("after_llm") | parser
```

**Source:** Development patterns derived from `libs/core/langchain_core/runnables/base.py`

## Production Configuration

Production logging must balance observability with performance. Key considerations include log volume management, structured logging for parsing, sensitive data filtering, and async handlers to prevent blocking.

### Production Logging Best Practices

**Key Principles:**
- Use INFO level for most modules (WARNING for low-priority components)
- Implement structured logging (JSON format) for log aggregation tools
- Filter sensitive data (API keys, PII) before logging
- Use async or buffered handlers to avoid blocking chain execution
- Implement log rotation to manage disk space
- Add contextual information (request_id, user_id, session_id)

### Production Configuration Template

**Optimized Production dictConfig:**

```python
import logging.config
import os

PRODUCTION_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s %(run_id)s'
        },
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'filters': {
        'sensitive_data_filter': {
            '()': 'langchain_logging.filters.SensitiveDataFilter',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'WARNING',  # Only warnings and errors to console
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file_json': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json',
            'filename': '/var/log/langchain/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10,
            'filters': ['sensitive_data_filter']
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'json',
            'filename': '/var/log/langchain/errors.log',
            'maxBytes': 10485760,
            'backupCount': 5,
            'filters': ['sensitive_data_filter']
        }
    },
    'loggers': {
        'langchain_classic': {
            'level': 'INFO',
            'handlers': ['file_json', 'error_file', 'console'],
            'propagate': False
        },
        'langchain_core': {
            'level': 'INFO',
            'handlers': ['file_json', 'error_file', 'console'],
            'propagate': False
        },
        'langchain_classic.chains': {
            'level': 'INFO',
            'handlers': ['file_json', 'error_file'],
            'propagate': False
        },
        'langchain_classic.agents': {
            'level': 'INFO',
            'handlers': ['file_json', 'error_file'],
            'propagate': False
        },
        'langchain_openai': {
            'level': 'WARNING',  # Only warnings/errors from API client
            'handlers': ['file_json', 'error_file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

# Apply production configuration
logging.config.dictConfig(PRODUCTION_LOGGING_CONFIG)
```

### Sensitive Data Filtering

**Custom Filter for Redacting Sensitive Information:**

```python
import logging
import re
from typing import Pattern

class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from log records."""
    
    # Patterns for sensitive data
    API_KEY_PATTERN: Pattern = re.compile(r'(api[_-]?key|token|secret)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]+)', re.IGNORECASE)
    EMAIL_PATTERN: Pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True (always allow record, just modify it)
        """
        # Redact API keys and tokens
        record.msg = self.API_KEY_PATTERN.sub(r'\1=***REDACTED***', str(record.msg))
        
        # Redact email addresses
        record.msg = self.EMAIL_PATTERN.sub('***EMAIL***', record.msg)
        
        # Redact from args if present
        if record.args:
            record.args = tuple(
                self._redact_string(str(arg)) for arg in record.args
            )
        
        return True
    
    def _redact_string(self, text: str) -> str:
        """Redact sensitive patterns from string.
        
        Args:
            text: String to redact
            
        Returns:
            Redacted string
        """
        text = self.API_KEY_PATTERN.sub(r'\1=***REDACTED***', text)
        text = self.EMAIL_PATTERN.sub('***EMAIL***', text)
        return text
```

### Structured Logging with Context

**Adding Request Context to All Log Entries:**

```python
import logging
from contextvars import ContextVar
from typing import Any, Dict

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
run_id_var: ContextVar[str] = ContextVar('run_id', default='')

class ContextFilter(logging.Filter):
    """Add context variables to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record.
        
        Args:
            record: Log record to enhance
            
        Returns:
            True (always allow record)
        """
        record.request_id = request_id_var.get()
        record.run_id = run_id_var.get()
        return True

# Usage in chain execution
def execute_chain_with_context(chain, inputs: Dict[str, Any], request_id: str):
    """Execute chain with logging context.
    
    Args:
        chain: LangChain chain to execute
        inputs: Chain inputs
        request_id: Request identifier for tracing
    """
    # Set context
    request_id_var.set(request_id)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Chain execution started", extra={'request_id': request_id})
    
    try:
        result = chain.invoke(inputs)
        logger.info(f"Chain execution completed", extra={'request_id': request_id})
        return result
    except Exception as e:
        logger.error(f"Chain execution failed: {e}", extra={'request_id': request_id}, exc_info=True)
        raise
```

## Module-Specific Loggers

LangChain uses hierarchical logger names following Python's standard `package.module` convention. Understanding the logger hierarchy enables fine-grained control over logging verbosity.

### Logger Hierarchy

**LangChain Logger Structure:**

```
langchain_classic/                      # Root for classic LangChain package
├── langchain_classic.chains/          # All chain-related logging
│   ├── langchain_classic.chains.base  # Base Chain class
│   ├── langchain_classic.chains.llm   # LLMChain (deprecated)
│   └── langchain_classic.chains.retrieval  # Retrieval chains
├── langchain_classic.agents/          # Agent execution logging
│   ├── langchain_classic.agents.agent
│   └── langchain_classic.agents.executor
├── langchain_classic.memory/          # Memory operations
├── langchain_classic.tools/           # Tool execution
└── langchain_classic.callbacks/       # Callback system

langchain_core/                         # Root for core abstractions
├── langchain_core.runnables/          # LCEL and Runnable protocol
│   ├── langchain_core.runnables.base
│   └── langchain_core.runnables.branch
├── langchain_core.prompts/            # Prompt templates
├── langchain_core.callbacks/          # Core callback handlers
├── langchain_core.output_parsers/     # Output parsing
└── langchain_core.messages/           # Message types
```

**Source:** Logger names follow the module structure in `libs/langchain/langchain_classic/` and `libs/core/langchain_core/`

### Configuring Specific Loggers

**Example: Debug Chains, Info for Everything Else:**

```python
import logging

# Get specific loggers
chain_logger = logging.getLogger('langchain_classic.chains')
runnable_logger = logging.getLogger('langchain_core.runnables')
agent_logger = logging.getLogger('langchain_classic.agents')
llm_logger = logging.getLogger('langchain_openai')

# Set different levels
chain_logger.setLevel(logging.DEBUG)      # Detailed chain execution
runnable_logger.setLevel(logging.DEBUG)   # LCEL composition details
agent_logger.setLevel(logging.INFO)       # High-level agent actions only
llm_logger.setLevel(logging.WARNING)      # Only warnings from LLM client

# Add handlers
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

for logger in [chain_logger, runnable_logger, agent_logger, llm_logger]:
    logger.addHandler(console_handler)
    logger.propagate = False  # Don't propagate to root logger
```

### Logger Configuration by Use Case

**Debugging Specific Chain Issues:**

```python
import logging

def debug_specific_chain():
    """Enable detailed logging for specific chain debugging."""
    # Only debug the chain module you're investigating
    logging.getLogger('langchain_classic.chains.retrieval').setLevel(logging.DEBUG)
    
    # Keep others at INFO to reduce noise
    logging.getLogger('langchain_classic').setLevel(logging.INFO)
    logging.getLogger('langchain_core').setLevel(logging.INFO)
```

**Monitoring Agent Decisions:**

```python
import logging

def monitor_agent_execution():
    """Enable logging for agent decision monitoring."""
    # Detailed agent logging
    logging.getLogger('langchain_classic.agents').setLevel(logging.DEBUG)
    
    # Detailed tool execution
    logging.getLogger('langchain_classic.tools').setLevel(logging.DEBUG)
    
    # Less verbose for other components
    logging.getLogger('langchain_classic.chains').setLevel(logging.INFO)
    logging.getLogger('langchain_core').setLevel(logging.WARNING)
```

**Investigating LCEL Composition Issues:**

```python
import logging

def debug_lcel_composition():
    """Enable detailed logging for LCEL debugging."""
    # Debug runnables and composition
    logging.getLogger('langchain_core.runnables').setLevel(logging.DEBUG)
    
    # Debug prompts to see template rendering
    logging.getLogger('langchain_core.prompts').setLevel(logging.DEBUG)
    
    # Debug output parsing
    logging.getLogger('langchain_core.output_parsers').setLevel(logging.DEBUG)
    
    # Reduce noise from chains
    logging.getLogger('langchain_classic.chains').setLevel(logging.WARNING)
```

### Logger Propagation Control

**Understanding Propagation:**

By default, log messages propagate up the logger hierarchy. This can cause duplicate log entries if both parent and child loggers have handlers.

**Controlling Propagation:**

```python
import logging

# Get logger
logger = logging.getLogger('langchain_classic.chains.llm')

# Add handler
handler = logging.StreamHandler()
logger.addHandler(handler)

# Disable propagation to prevent duplicate logs
logger.propagate = False

# Now logs from langchain_classic.chains.llm won't propagate to:
# - langchain_classic.chains
# - langchain_classic
# - root logger
```

**Best Practice:**
- Set `propagate = False` for loggers with their own handlers
- Use propagation when you want centralized handling at root logger
- In production, typically disable propagation for LangChain loggers to control output precisely

## Callback-Based Logging

LangChain's callback system provides the most granular approach to logging chain execution. Custom callback handlers can capture detailed execution events that standard logging might miss.

### Understanding Callback Events

**Callback Event Sequence for Chain Execution:**

```
1. on_chain_start     → Chain begins execution
2. on_llm_start       → LLM invocation starts (if chain uses LLM)
3. on_llm_new_token   → Each token generated (streaming only)
4. on_llm_end         → LLM completes
5. on_chain_end       → Chain completes successfully
   OR
   on_chain_error     → Chain fails with exception
```

**Source:** `libs/core/langchain_core/callbacks/base.py:121-191`

### Custom Logging Callback Handler

**Complete Logging Callback Implementation:**

```python
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class DetailedLoggingCallback(BaseCallbackHandler):
    """Callback handler for detailed chain execution logging.
    
    Logs all major events in chain execution with timing information,
    input/output data, and token usage statistics.
    """
    
    def __init__(self):
        """Initialize callback with timing tracking."""
        self.start_times: Dict[UUID, float] = {}
        self.token_counts: Dict[UUID, int] = {}
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Log when chain starts.
        
        Args:
            serialized: Serialized chain information
            inputs: Chain inputs
            run_id: Unique identifier for this run
            parent_run_id: Parent run identifier if nested
            tags: Tags associated with this run
            metadata: Metadata for this run
            **kwargs: Additional arguments
        """
        self.start_times[run_id] = time.time()
        
        chain_name = serialized.get('name', 'Unknown')
        logger.info(
            f"Chain started: {chain_name}",
            extra={
                'run_id': str(run_id),
                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                'tags': tags,
                'event': 'chain_start'
            }
        )
        logger.debug(f"Chain inputs: {inputs}", extra={'run_id': str(run_id)})
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Log when chain completes successfully.
        
        Args:
            outputs: Chain outputs
            run_id: Unique identifier for this run
            parent_run_id: Parent run identifier if nested
            **kwargs: Additional arguments
        """
        duration = time.time() - self.start_times.get(run_id, time.time())
        tokens = self.token_counts.get(run_id, 0)
        
        logger.info(
            f"Chain completed: duration={duration:.2f}s, tokens={tokens}",
            extra={
                'run_id': str(run_id),
                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                'duration_seconds': duration,
                'token_count': tokens,
                'event': 'chain_end'
            }
        )
        logger.debug(f"Chain outputs: {outputs}", extra={'run_id': str(run_id)})
        
        # Cleanup
        self.start_times.pop(run_id, None)
        self.token_counts.pop(run_id, None)
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Log when chain encounters an error.
        
        Args:
            error: Exception that occurred
            run_id: Unique identifier for this run
            parent_run_id: Parent run identifier if nested
            **kwargs: Additional arguments
        """
        duration = time.time() - self.start_times.get(run_id, time.time())
        
        logger.error(
            f"Chain error after {duration:.2f}s: {type(error).__name__}: {error}",
            extra={
                'run_id': str(run_id),
                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                'duration_seconds': duration,
                'error_type': type(error).__name__,
                'event': 'chain_error'
            },
            exc_info=True
        )
        
        # Cleanup
        self.start_times.pop(run_id, None)
        self.token_counts.pop(run_id, None)
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Log when LLM invocation starts.
        
        Args:
            serialized: Serialized LLM information
            prompts: List of prompts being sent
            run_id: Unique identifier for this run
            parent_run_id: Parent run identifier
            **kwargs: Additional arguments
        """
        llm_name = serialized.get('name', 'Unknown')
        logger.info(
            f"LLM call started: {llm_name}, {len(prompts)} prompt(s)",
            extra={
                'run_id': str(run_id),
                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                'llm_name': llm_name,
                'prompt_count': len(prompts),
                'event': 'llm_start'
            }
        )
        
        # Log prompts at debug level (can be verbose)
        for i, prompt in enumerate(prompts):
            logger.debug(
                f"LLM prompt {i+1}: {prompt[:200]}...",
                extra={'run_id': str(run_id)}
            )
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Log when LLM invocation completes.
        
        Args:
            response: LLM response
            run_id: Unique identifier for this run
            parent_run_id: Parent run identifier
            **kwargs: Additional arguments
        """
        # Extract token usage if available
        tokens = 0
        if response.llm_output and 'token_usage' in response.llm_output:
            token_usage = response.llm_output['token_usage']
            tokens = token_usage.get('total_tokens', 0)
            
            # Track tokens for parent chain
            if parent_run_id:
                self.token_counts[parent_run_id] = self.token_counts.get(parent_run_id, 0) + tokens
        
        logger.info(
            f"LLM call completed: {len(response.generations)} generation(s), {tokens} tokens",
            extra={
                'run_id': str(run_id),
                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                'generation_count': len(response.generations),
                'token_count': tokens,
                'event': 'llm_end'
            }
        )
        
        # Log responses at debug level
        for i, generation_list in enumerate(response.generations):
            for j, generation in enumerate(generation_list):
                logger.debug(
                    f"LLM generation {i+1}.{j+1}: {generation.text[:200]}...",
                    extra={'run_id': str(run_id)}
                )
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Log when LLM encounters an error.
        
        Args:
            error: Exception that occurred
            run_id: Unique identifier for this run
            parent_run_id: Parent run identifier
            **kwargs: Additional arguments
        """
        logger.error(
            f"LLM error: {type(error).__name__}: {error}",
            extra={
                'run_id': str(run_id),
                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                'error_type': type(error).__name__,
                'event': 'llm_error'
            },
            exc_info=True
        )
    
    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Log each new token during streaming (optional, can be very verbose).
        
        Args:
            token: New token generated
            run_id: Unique identifier for this run
            parent_run_id: Parent run identifier
            **kwargs: Additional arguments
        """
        # Only log at debug level - streaming tokens are very verbose
        logger.debug(
            f"New token: {repr(token)}",
            extra={
                'run_id': str(run_id),
                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                'event': 'llm_new_token'
            }
        )
```

**Usage Example:**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)

# Create callback
logging_callback = DetailedLoggingCallback()

# Create chain with callback
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI(temperature=0)
chain = prompt | llm

# Execute with logging callback
result = chain.invoke(
    {"topic": "artificial intelligence"},
    config={"callbacks": [logging_callback]}
)
```

## Performance Considerations

Logging can impact chain execution performance if not configured properly. Understanding the performance implications helps balance observability with speed.

### Log Volume Management

**High-Volume Scenarios:**
- Streaming LLM responses (on_llm_new_token fires for each token)
- Long sequential chains (multiple on_chain_start/end events)
- Agent loops (many tool invocations)
- Batch processing (parallel chain executions)

**Strategies for Reducing Log Volume:**

```python
import logging

# 1. Use appropriate log levels
logger = logging.getLogger('langchain_classic')
logger.setLevel(logging.INFO)  # Skip DEBUG in production

# 2. Disable verbose loggers
logging.getLogger('langchain_core.runnables').setLevel(logging.WARNING)

# 3. Conditional logging based on sampling
import random

class SampledLoggingCallback(BaseCallbackHandler):
    """Callback that logs only a sample of events."""
    
    def __init__(self, sample_rate: float = 0.1):
        """Initialize with sampling rate.
        
        Args:
            sample_rate: Fraction of events to log (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
    
    def on_llm_new_token(self, token: str, *, run_id: UUID, **kwargs: Any) -> Any:
        """Log tokens with sampling to reduce volume.
        
        Args:
            token: New token generated
            run_id: Run identifier
            **kwargs: Additional arguments
        """
        if random.random() < self.sample_rate:
            logger.debug(f"Token sample: {repr(token)}", extra={'run_id': str(run_id)})
```

### Async Logging

**Non-Blocking Logging for Production:**

```python
from logging.handlers import QueueHandler, QueueListener
import logging
import queue

def setup_async_logging():
    """Configure async logging to prevent blocking chain execution."""
    # Create a queue for log records
    log_queue = queue.Queue(-1)  # No size limit
    
    # Create actual file handler
    file_handler = logging.FileHandler('langchain_async.log')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Create queue listener (processes logs in background thread)
    queue_listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
    queue_listener.start()
    
    # Create queue handler (non-blocking)
    queue_handler = QueueHandler(log_queue)
    
    # Configure LangChain loggers to use queue handler
    langchain_logger = logging.getLogger('langchain_classic')
    langchain_logger.addHandler(queue_handler)
    langchain_logger.setLevel(logging.INFO)
    
    return queue_listener  # Keep reference to prevent garbage collection

# Usage
listener = setup_async_logging()

# ... run chains ...

# Cleanup on shutdown
# listener.stop()
```

### Memory-Efficient Logging

**Avoiding Memory Leaks in Long-Running Services:**

```python
class MemoryEfficientLoggingCallback(BaseCallbackHandler):
    """Callback that doesn't accumulate run data indefinitely."""
    
    def __init__(self, max_tracked_runs: int = 1000):
        """Initialize with maximum run tracking limit.
        
        Args:
            max_tracked_runs: Maximum number of runs to track simultaneously
        """
        self.start_times: Dict[UUID, float] = {}
        self.max_tracked_runs = max_tracked_runs
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], 
                       *, run_id: UUID, **kwargs: Any) -> Any:
        """Track chain start with memory limit.
        
        Args:
            serialized: Chain information
            inputs: Chain inputs
            run_id: Run identifier
            **kwargs: Additional arguments
        """
        # Cleanup old runs if exceeding limit
        if len(self.start_times) >= self.max_tracked_runs:
            # Remove oldest entries
            oldest_runs = sorted(self.start_times.items(), key=lambda x: x[1])[:100]
            for old_run_id, _ in oldest_runs:
                self.start_times.pop(old_run_id, None)
        
        self.start_times[run_id] = time.time()
        logger.info(f"Chain started", extra={'run_id': str(run_id)})
    
    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any) -> Any:
        """Log and cleanup.
        
        Args:
            outputs: Chain outputs
            run_id: Run identifier
            **kwargs: Additional arguments
        """
        duration = time.time() - self.start_times.get(run_id, time.time())
        logger.info(f"Chain completed: {duration:.2f}s", extra={'run_id': str(run_id)})
        
        # Always cleanup
        self.start_times.pop(run_id, None)
```

## Integration with Observability Platforms

### LangSmith Integration

**LangSmith Automatic Tracing:**

LangSmith provides built-in tracing for LangChain applications. Enable it with environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-langsmith-api-key
export LANGCHAIN_PROJECT=your-project-name
```

**Combining Standard Logging with LangSmith:**

```python
import os
import logging

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "production-app"

# Also configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now chains will send traces to LangSmith AND log to standard output
```

### Custom Observability Integration

**Sending Logs to External Systems:**

```python
import logging
import requests
from typing import Any, Dict

class ObservabilityLoggingHandler(logging.Handler):
    """Custom handler that sends logs to observability platform."""
    
    def __init__(self, endpoint: str, api_key: str):
        """Initialize handler with endpoint configuration.
        
        Args:
            endpoint: API endpoint for log ingestion
            api_key: Authentication key
        """
        super().__init__()
        self.endpoint = endpoint
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def emit(self, record: logging.LogRecord) -> None:
        """Send log record to observability platform.
        
        Args:
            record: Log record to send
        """
        try:
            log_entry = {
                'timestamp': record.created,
                'level': record.levelname,
                'logger': record.name,
                'message': self.format(record),
                'run_id': getattr(record, 'run_id', None),
                'metadata': {
                    'filename': record.filename,
                    'lineno': record.lineno,
                    'funcName': record.funcName,
                }
            }
            
            # Non-blocking send (fire and forget)
            self.session.post(self.endpoint, json=log_entry, timeout=1)
        except Exception:
            # Don't let logging errors break chain execution
            self.handleError(record)

# Usage
observability_handler = ObservabilityLoggingHandler(
    endpoint='https://logs.example.com/api/v1/logs',
    api_key='your-api-key'
)
observability_handler.setFormatter(
    logging.Formatter('%(message)s')
)

logger = logging.getLogger('langchain_classic')
logger.addHandler(observability_handler)
```

## Executable Examples

### Example 1: Basic Logging Setup

**File:** `examples/debugging/basic_logging_setup.py`

```python
"""Basic logging configuration for LangChain development.

This example demonstrates how to configure logging for debugging
LangChain chains with appropriate levels and formatting.
"""

import logging
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Configure logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set specific levels for different components
logging.getLogger('langchain_classic').setLevel(logging.DEBUG)
logging.getLogger('langchain_core').setLevel(logging.DEBUG)
logging.getLogger('langchain_openai').setLevel(logging.INFO)

# Reduce noise from HTTP libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main():
    """Run example chain with logging."""
    logger.info("Starting LangChain logging example")
    
    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Create a simple chain
        logger.debug("Creating prompt template")
        prompt = ChatPromptTemplate.from_template("Tell me a short fact about {topic}")
        
        logger.debug("Initializing ChatOpenAI")
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        
        logger.debug("Composing chain")
        chain = prompt | llm
        
        # Execute chain
        logger.info("Invoking chain with topic='Python programming'")
        result = chain.invoke({"topic": "Python programming"})
        
        logger.info(f"Chain completed successfully")
        logger.debug(f"Result type: {type(result)}")
        logger.debug(f"Result content: {result.content}")
        
        print(f"\nResult: {result.content}")
        
    except Exception as e:
        logger.error(f"Chain execution failed: {type(e).__name__}: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
```

**Expected Output:**
```
2024-01-15 10:30:45 - __main__ - INFO - Starting LangChain logging example
2024-01-15 10:30:45 - __main__ - DEBUG - Creating prompt template
2024-01-15 10:30:45 - langchain_core.prompts.chat - DEBUG - Creating ChatPromptTemplate
2024-01-15 10:30:45 - __main__ - DEBUG - Initializing ChatOpenAI
2024-01-15 10:30:45 - __main__ - DEBUG - Composing chain
2024-01-15 10:30:45 - __main__ - INFO - Invoking chain with topic='Python programming'
2024-01-15 10:30:46 - langchain_openai.chat_models - INFO - LLM call completed
2024-01-15 10:30:46 - __main__ - INFO - Chain completed successfully
2024-01-15 10:30:46 - __main__ - DEBUG - Result type: <class 'AIMessage'>
2024-01-15 10:30:46 - __main__ - DEBUG - Result content: Python is a high-level...

Result: Python is a high-level programming language known for its simplicity and readability.
```

### Example 2: Custom Logging Callback

**File:** `examples/debugging/logging_callback_example.py`

```python
"""Custom callback handler for detailed chain execution logging.

This example implements a complete logging callback that captures
all chain execution events with timing and token tracking.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ExecutionLoggingCallback(BaseCallbackHandler):
    """Callback handler that logs detailed chain execution information."""
    
    def __init__(self):
        """Initialize callback with timing tracking."""
        self.start_times: Dict[UUID, float] = {}
        self.token_counts: Dict[UUID, int] = {}
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        """Log chain start event."""
        self.start_times[run_id] = time.time()
        chain_name = serialized.get('name', 'Unknown')
        
        logger.info(f"[{run_id}] Chain started: {chain_name}")
        logger.info(f"[{run_id}] Inputs: {inputs}")
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        """Log chain completion event."""
        duration = time.time() - self.start_times.get(run_id, time.time())
        tokens = self.token_counts.get(run_id, 0)
        
        logger.info(f"[{run_id}] Chain completed in {duration:.2f}s")
        logger.info(f"[{run_id}] Total tokens: {tokens}")
        logger.info(f"[{run_id}] Outputs: {outputs}")
        
        # Cleanup
        self.start_times.pop(run_id, None)
        self.token_counts.pop(run_id, None)
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        """Log chain error event."""
        duration = time.time() - self.start_times.get(run_id, time.time())
        
        logger.error(
            f"[{run_id}] Chain failed after {duration:.2f}s: "
            f"{type(error).__name__}: {error}"
        )
        
        # Cleanup
        self.start_times.pop(run_id, None)
        self.token_counts.pop(run_id, None)
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        """Log LLM invocation start."""
        llm_name = serialized.get('name', 'Unknown')
        logger.info(f"[{run_id}] LLM call started: {llm_name}")
        logger.debug(f"[{run_id}] Prompt: {prompts[0][:100]}...")
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Log LLM completion event."""
        # Extract token usage
        tokens = 0
        if response.llm_output and 'token_usage' in response.llm_output:
            token_usage = response.llm_output['token_usage']
            tokens = token_usage.get('total_tokens', 0)
            
            if parent_run_id:
                self.token_counts[parent_run_id] = (
                    self.token_counts.get(parent_run_id, 0) + tokens
                )
        
        logger.info(f"[{run_id}] LLM call completed: {tokens} tokens")
        
        # Log response
        if response.generations:
            text = response.generations[0][0].text
            logger.debug(f"[{run_id}] Response: {text[:100]}...")


def main():
    """Run example with logging callback."""
    logger.info("=== Custom Logging Callback Example ===")
    
    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY not set")
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Create callback
    logging_callback = ExecutionLoggingCallback()
    
    try:
        # Create chain
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer this question: {question}"
        )
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        chain = prompt | llm
        
        # Execute with callback
        logger.info("Executing chain with logging callback...")
        result = chain.invoke(
            {"question": "What is the capital of France?"},
            config={"callbacks": [logging_callback]}
        )
        
        print(f"\n✓ Chain execution successful!")
        print(f"Answer: {result.content}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
```

**Expected Output:**
```
2024-01-15 10:35:12 - __main__ - INFO - === Custom Logging Callback Example ===
2024-01-15 10:35:12 - __main__ - INFO - Executing chain with logging callback...
2024-01-15 10:35:12 - __main__ - INFO - [abc-123] Chain started: RunnableSequence
2024-01-15 10:35:12 - __main__ - INFO - [abc-123] Inputs: {'question': 'What is the capital of France?'}
2024-01-15 10:35:12 - __main__ - INFO - [def-456] LLM call started: ChatOpenAI
2024-01-15 10:35:13 - __main__ - INFO - [def-456] LLM call completed: 45 tokens
2024-01-15 10:35:13 - __main__ - INFO - [abc-123] Chain completed in 1.23s
2024-01-15 10:35:13 - __main__ - INFO - [abc-123] Total tokens: 45
2024-01-15 10:35:13 - __main__ - INFO - [abc-123] Outputs: {...}

✓ Chain execution successful!
Answer: The capital of France is Paris.
```

### Example 3: Structured Logging with Context

**File:** `examples/debugging/structured_logging_example.py`

```python
"""Structured logging with context for production monitoring.

This example demonstrates production-ready structured logging with
request context, JSON formatting, and sensitive data filtering.
"""

import json
import logging
import os
import re
from contextvars import ContextVar
from typing import Any, Dict
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format record as JSON string.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted string
        """
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'request_id': request_id_var.get(),
            'user_id': user_id_var.get(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'run_id'):
            log_data['run_id'] = record.run_id
        if hasattr(record, 'duration_seconds'):
            log_data['duration_seconds'] = record.duration_seconds
        if hasattr(record, 'token_count'):
            log_data['token_count'] = record.token_count
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from logs."""
    
    API_KEY_PATTERN = re.compile(
        r'(api[_-]?key|token|secret)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]+)',
        re.IGNORECASE
    )
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True (always allow, just modify)
        """
        # Redact sensitive patterns
        record.msg = self.API_KEY_PATTERN.sub(r'\1=***REDACTED***', str(record.msg))
        return True


def setup_structured_logging():
    """Configure structured JSON logging with filtering."""
    # Create formatter
    json_formatter = JSONFormatter()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    console_handler.addFilter(SensitiveDataFilter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # Configure LangChain loggers
    for logger_name in ['langchain_classic', 'langchain_core', 'langchain_openai']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = True  # Use root handler


def process_request(user_query: str, user_id: str = "anonymous"):
    """Process a user request with full context logging.
    
    Args:
        user_query: User's question
        user_id: User identifier for tracking
    """
    # Generate request ID
    request_id = str(uuid4())
    
    # Set context
    request_id_var.set(request_id)
    user_id_var.set(user_id)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing request: {user_query[:50]}...")
    
    try:
        # Create chain
        prompt = ChatPromptTemplate.from_template("Answer briefly: {question}")
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        chain = prompt | llm
        
        # Execute
        result = chain.invoke({"question": user_query})
        
        logger.info("Request completed successfully")
        return result.content
        
    except Exception as e:
        logger.error(f"Request failed: {type(e).__name__}: {e}", exc_info=True)
        raise


def main():
    """Run structured logging example."""
    print("=== Structured Logging Example ===\n")
    
    # Setup logging
    setup_structured_logging()
    
    # Check API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Process multiple requests with different contexts
    requests = [
        ("What is machine learning?", "user123"),
        ("Explain neural networks", "user456"),
    ]
    
    for query, user_id in requests:
        try:
            print(f"\nProcessing: {query}")
            answer = process_request(query, user_id)
            print(f"Answer: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
```

**Expected Output (JSON formatted):**
```json
{"timestamp": "2024-01-15 10:40:00", "level": "INFO", "logger": "__main__", "message": "Processing request: What is machine learning?...", "request_id": "abc-123-def", "user_id": "user123"}
{"timestamp": "2024-01-15 10:40:01", "level": "INFO", "logger": "__main__", "message": "Request completed successfully", "request_id": "abc-123-def", "user_id": "user123"}
{"timestamp": "2024-01-15 10:40:02", "level": "INFO", "logger": "__main__", "message": "Processing request: Explain neural networks...", "request_id": "xyz-789-ghi", "user_id": "user456"}
{"timestamp": "2024-01-15 10:40:03", "level": "INFO", "logger": "__main__", "message": "Request completed successfully", "request_id": "xyz-789-ghi", "user_id": "user456"}
```

## Summary and Best Practices

### Key Takeaways

1. **Use Appropriate Log Levels:**
   - DEBUG: Development debugging with full details
   - INFO: Production monitoring and high-level operations
   - WARNING: Retry scenarios and recoverable issues
   - ERROR: Failures requiring attention

2. **Development vs Production:**
   - Development: Verbose console logging with full data
   - Production: Structured JSON logging with filtering and rotation

3. **Module-Specific Configuration:**
   - Configure loggers hierarchically (langchain_classic.chains, langchain_core.runnables)
   - Use propagation control to prevent duplicate logs
   - Adjust levels per module based on debugging needs

4. **Callback-Based Logging:**
   - Implement BaseCallbackHandler for granular execution monitoring
   - Track timing, token usage, and execution flow
   - Use callbacks for production observability

5. **Performance Optimization:**
   - Use async handlers to prevent blocking
   - Implement sampling for high-volume scenarios
   - Manage memory with run tracking limits
   - Disable verbose logging in production

6. **Security:**
   - Always filter sensitive data (API keys, PII)
   - Use structured logging for parsing and analysis
   - Implement log rotation for disk space management

### Quick Reference

**Enable Debug Logging:**
```python
import logging
logging.getLogger('langchain_classic').setLevel(logging.DEBUG)
logging.getLogger('langchain_core').setLevel(logging.DEBUG)
```

**Production Configuration:**
```python
import logging.config
logging.config.dictConfig(PRODUCTION_LOGGING_CONFIG)  # See section above
```

**Custom Callback:**
```python
from langchain_core.callbacks import BaseCallbackHandler
callback = DetailedLoggingCallback()
chain.invoke(inputs, config={"callbacks": [callback]})
```

**Structured Logging:**
```python
# Use JSONFormatter for structured output
handler.setFormatter(JSONFormatter())
```

### Related Documentation

- [Common Issues Guide](./common-issues.md) - Debugging specific error patterns
- [Stack Traces Guide](./stack-traces.md) - Interpreting LangChain exceptions
- [Troubleshooting Guide](./troubleshooting.md) - Systematic debugging approach
- [Callback System Architecture](../architecture/callback-system.md) - Callback event flow
- [Production Deployment Guide](../guides/production-deployment.md) - Production best practices

### Additional Resources

- **LangSmith Documentation**: https://docs.smith.langchain.com/ - LangChain's observability platform
- **Python Logging Documentation**: https://docs.python.org/3/library/logging.html - Standard library reference
- **Structured Logging**: https://github.com/madzak/python-json-logger - JSON formatter library

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-15  
**Source References**: 
- `libs/core/langchain_core/callbacks/base.py` - Callback handler interfaces
- `libs/langchain/langchain_classic/chains/base.py` - Chain execution and logging patterns
