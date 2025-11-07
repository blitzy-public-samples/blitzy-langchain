# LangChain Debugging Utilities

## Overview

This directory contains specialized debugging utilities designed to help developers diagnose, troubleshoot, and understand the execution behavior of LangChain chains. These tools address common debugging challenges when working with LangChain's abstraction layers, including:

- **Execution Visibility**: Understanding what happens inside chain compositions and LCEL pipes
- **Event Sequence Tracking**: Monitoring callback events and their order during execution
- **Chain Structure Analysis**: Inspecting the composition and configuration of complex chains
- **Granular Logging**: Capturing detailed execution traces for debugging and monitoring

These utilities are particularly valuable for:
- Junior developers learning LangChain patterns and debugging chain failures
- Debugging complex multi-stage chains where failures are difficult to isolate
- Production troubleshooting where detailed execution logs are needed
- Development workflows requiring comprehensive chain execution visibility

## Available Debugging Utilities

### 1. logging_wrapper.py - Chain Logging Decorator

**Purpose**: Provides a decorator for adding comprehensive logging to any LangChain chain execution, capturing inputs, outputs, execution time, and errors.

**Key Features**:
- Wraps chain invoke() and ainvoke() methods with detailed logging
- Logs input parameters, output results, and execution duration
- Captures and logs exceptions with full stack traces
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Thread-safe logging for async chain execution
- Supports both synchronous and asynchronous chain methods

**Use Cases**:
- Development: Understanding chain behavior during implementation
- Debugging: Identifying where in a chain sequence failures occur
- Production: Monitoring chain performance and reliability
- Testing: Verifying chain inputs and outputs match expectations

### 2. callback_debugger.py - Debug Callback Handler

**Purpose**: Implements a comprehensive callback handler that logs all LangChain callback events with detailed context, enabling visibility into the internal execution flow.

**Key Features**:
- Logs all callback events: on_chain_start, on_llm_start, on_llm_new_token, on_llm_end, on_chain_end, on_chain_error
- Captures event timing and duration for performance analysis
- Records input/output data at each callback stage
- Tracks callback hierarchy for nested chain execution
- Provides event sequence visualization
- Supports both sync and async callback methods

**Use Cases**:
- Debugging LCEL compositions to see type transformations at each pipe stage
- Understanding callback event ordering in complex chains
- Identifying performance bottlenecks in multi-stage chains
- Troubleshooting callback-related errors
- Learning LangChain's internal execution model

### 3. chain_introspection.py - Chain Structure Inspector

**Purpose**: Utility functions for analyzing and visualizing the structure of LangChain chains, including component inspection, type analysis, and composition mapping.

**Key Features**:
- Inspects chain structure and component types
- Extracts configuration parameters from chains
- Visualizes chain composition (sequential, branching, etc.)
- Identifies input/output schemas for each chain component
- Detects LCEL pipe compositions and displays type flow
- Generates human-readable chain structure representations

**Use Cases**:
- Understanding the composition of complex chains
- Debugging type mismatches in LCEL compositions
- Documenting chain architectures
- Validating chain construction before execution
- Comparing expected vs actual chain structures

## Prerequisites

### Python Version
- **Required**: Python >= 3.10.0, < 4.0.0
- **Tested**: Python 3.10, 3.13

### Core Dependencies
These utilities require LangChain core packages:
- `langchain-core >= 1.0.1` - Core LangChain abstractions and interfaces
- `langchain-classic >= 1.0.0` - LangChain classic chains (optional, for legacy chain debugging)

### Optional Dependencies
For enhanced functionality:
- `rich >= 13.0.0` - Pretty terminal output and formatting (recommended for chain_introspection.py)
- `colorlog >= 6.7.0` - Colored log output (recommended for logging_wrapper.py)

## Installation

### Quick Setup

1. **Navigate to the debugging examples directory**:
   ```bash
   cd examples/debugging
   ```

2. **Install dependencies using uv** (recommended):
   ```bash
   uv pip install -r requirements.txt
   ```

   **Alternative: Install using pip**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "from langchain_core.runnables import Runnable; print('LangChain installed successfully')"
   ```

### Environment Configuration

These debugging utilities do not require API keys for basic functionality. However, if you want to test them with actual LLM chains, create a `.env` file:

```bash
# Optional: For testing with real LLM providers
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

## Usage Examples

### Using logging_wrapper.py

**Basic Usage - Decorating a Chain Function**:

```python
from logging_wrapper import log_chain_execution
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

@log_chain_execution(log_level="INFO")
def create_summarization_chain():
    prompt = ChatPromptTemplate.from_template("Summarize: {text}")
    model = ChatOpenAI(model="gpt-3.5-turbo")
    parser = StrOutputParser()
    return prompt | model | parser

# Execute the chain - logging happens automatically
chain = create_summarization_chain()
result = chain.invoke({"text": "LangChain is a framework for developing applications powered by language models."})
print(result)
```

**Expected Console Output**:
```
[INFO] Chain execution started: create_summarization_chain
[INFO] Input: {"text": "LangChain is a framework..."}
[INFO] Chain execution completed in 1.23s
[INFO] Output: "LangChain enables developers to build LLM applications..."
```

**Advanced Usage - Wrapping Existing Chains**:

```python
from logging_wrapper import wrap_chain_with_logging
from langchain.chains import LLMChain

# Wrap an existing chain instance
existing_chain = LLMChain(llm=model, prompt=prompt)
logged_chain = wrap_chain_with_logging(existing_chain, name="MyChain", log_level="DEBUG")

# All invocations now include detailed logging
result = logged_chain.invoke({"input": "test"})
```

### Using callback_debugger.py

**Basic Usage - Adding Debug Callbacks to a Chain**:

```python
from callback_debugger import DebugCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Create the debug callback handler
debug_handler = DebugCallbackHandler(verbose=True, include_timestamps=True)

# Build your chain
prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
model = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()
chain = prompt | model | parser

# Execute with debug callbacks
result = chain.invoke(
    {"text": "Hello, how are you?"},
    config={"callbacks": [debug_handler]}
)

print(result)
```

**Expected Console Output**:
```
[CALLBACK] on_chain_start: RunnableSequence (id: abc123) at 2024-01-15 10:30:45.123
[CALLBACK]   Input: {"text": "Hello, how are you?"}
[CALLBACK] on_llm_start: ChatOpenAI (id: def456) at 2024-01-15 10:30:45.234
[CALLBACK]   Prompts: ["Translate to French: Hello, how are you?"]
[CALLBACK] on_llm_end: ChatOpenAI (id: def456) at 2024-01-15 10:30:46.567
[CALLBACK]   Duration: 1.33s
[CALLBACK]   Response: AIMessage(content="Bonjour, comment allez-vous?")
[CALLBACK] on_chain_end: RunnableSequence (id: abc123) at 2024-01-15 10:30:46.678
[CALLBACK]   Duration: 1.55s
[CALLBACK]   Output: "Bonjour, comment allez-vous?"
```

**Advanced Usage - Analyzing Event Sequences**:

```python
from callback_debugger import DebugCallbackHandler

# Create handler with event sequence tracking
debug_handler = DebugCallbackHandler(
    verbose=True,
    track_event_sequence=True,
    output_file="chain_execution_log.json"
)

# After execution, analyze the event sequence
chain.invoke(input_data, config={"callbacks": [debug_handler]})

# Get the recorded event sequence
event_sequence = debug_handler.get_event_sequence()
print(f"Total events: {len(event_sequence)}")
print(f"Total execution time: {debug_handler.get_total_duration()}s")
```

### Using chain_introspection.py

**Basic Usage - Inspecting Chain Structure**:

```python
from chain_introspection import inspect_chain, print_chain_structure
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Build a chain
prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms")
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
parser = StrOutputParser()
chain = prompt | model | parser

# Inspect the chain structure
chain_info = inspect_chain(chain)
print(chain_info)

# Pretty-print the structure
print_chain_structure(chain)
```

**Expected Console Output**:
```
Chain Structure Analysis
========================
Type: RunnableSequence
Components: 3

Component 1: ChatPromptTemplate
  - Type: PromptTemplate
  - Input Variables: ['topic']
  - Template: "Explain {topic} in simple terms"

Component 2: ChatOpenAI
  - Type: ChatModel
  - Model: gpt-3.5-turbo
  - Temperature: 0.7
  - Max Tokens: None

Component 3: StrOutputParser
  - Type: OutputParser
  - Input Type: AIMessage
  - Output Type: str

Type Flow: Dict[str, str] → List[BaseMessage] → AIMessage → str
```

**Advanced Usage - Validating Chain Composition**:

```python
from chain_introspection import validate_chain_types, get_chain_schema

# Validate type compatibility in LCEL composition
is_valid, error_msg = validate_chain_types(chain)
if not is_valid:
    print(f"Chain type error: {error_msg}")
else:
    print("Chain type composition is valid")

# Extract input/output schemas
schema = get_chain_schema(chain)
print(f"Expected input schema: {schema['input']}")
print(f"Output schema: {schema['output']}")
```

## Tips and Best Practices

### Effective Debugging Workflow

1. **Start with Chain Introspection**: Before debugging execution issues, use `chain_introspection.py` to verify your chain structure matches expectations
2. **Add Callback Debugger for Execution Flow**: Use `callback_debugger.py` to understand the event sequence and identify which component is failing
3. **Enable Logging Wrapper for Detailed Traces**: Use `logging_wrapper.py` to capture complete input/output logs for reproduction and analysis

### Performance Considerations

- **Disable in Production**: These debugging utilities add overhead. Disable or reduce verbosity in production environments
- **Use Log Levels Appropriately**: Set `log_level="ERROR"` for production, `"DEBUG"` for development
- **Limit Callback Event Logging**: For high-throughput applications, consider logging only on_chain_start and on_chain_end events

### Common Debugging Scenarios

**Scenario 1: Chain Returns Unexpected Output**
```python
# Use logging_wrapper to capture exact inputs and outputs
from logging_wrapper import log_chain_execution

@log_chain_execution(log_level="DEBUG", log_inputs=True, log_outputs=True)
def my_chain():
    return prompt | model | parser

# Examine logged inputs/outputs to identify transformation issues
```

**Scenario 2: Chain Fails with Cryptic Error**
```python
# Use callback_debugger to see which component fails
from callback_debugger import DebugCallbackHandler

debug_handler = DebugCallbackHandler(verbose=True, log_errors=True)
try:
    chain.invoke(input, config={"callbacks": [debug_handler]})
except Exception as e:
    # Check debug_handler logs to see which callback raised the error
    print(f"Failed at: {debug_handler.get_last_event()}")
```

**Scenario 3: Type Mismatch in LCEL Composition**
```python
# Use chain_introspection to validate type flow
from chain_introspection import validate_chain_types, print_type_flow

is_valid, error = validate_chain_types(chain)
if not is_valid:
    print(f"Type error: {error}")
    print_type_flow(chain)  # Visualize where types don't align
```

## Troubleshooting

### Import Errors

**Problem**: `ImportError: cannot import name 'Runnable' from 'langchain_core.runnables'`

**Solution**: Ensure you have the correct version of langchain-core installed:
```bash
uv pip install "langchain-core>=1.0.1"
```

### Logging Not Appearing

**Problem**: Logging output not visible in console

**Solution**: Configure logging level for your application:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Callback Events Not Firing

**Problem**: DebugCallbackHandler not logging any events

**Solution**: Ensure callbacks are passed in the config parameter:
```python
# Correct
chain.invoke(input, config={"callbacks": [debug_handler]})

# Incorrect
chain.invoke(input, callbacks=[debug_handler])  # Wrong parameter location
```

## Related Documentation

For more information about LangChain debugging and troubleshooting:

- **Debugging Guide**: `../../docs/debugging/common-issues.md` - Top 10 LangChain failure modes and solutions
- **Logging Configuration Guide**: `../../docs/debugging/logging.md` - Comprehensive logging setup for LangChain applications
- **Stack Trace Interpretation**: `../../docs/debugging/stack-traces.md` - Understanding LangChain error traces
- **Callback System Architecture**: `../../docs/architecture/callback-system.md` - Deep dive into LangChain callback events
- **Chain Lifecycle Documentation**: `../../docs/architecture/chain-lifecycle.md` - Understanding chain execution flow

## Contributing

Found a bug or have a suggestion for additional debugging utilities? Please refer to the main repository contribution guidelines at `../../CONTRIBUTING.md`.

## License

These debugging utilities are part of the LangChain monorepo and are subject to the same MIT license. See the root `LICENSE` file for details.
