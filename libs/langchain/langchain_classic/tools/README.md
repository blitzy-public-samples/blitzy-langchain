# LangChain Tools Module

## Overview

The `tools` module provides the foundational abstractions for creating tools that LangChain agents can invoke to perform specific actions. Tools are callable components that encapsulate discrete functionality, such as searching Wikipedia, querying databases, performing calculations, or making API calls.

**Module Purpose**: Provides base classes and utilities for defining tools with structured input schemas, type validation, and error handling that integrate seamlessly with LangChain's agent execution framework.

**Module Responsibility**: Defines the tool interface contracts and provides multiple implementation patterns (decorator, factory, and subclassing) to accommodate different levels of complexity and customization needs.

**Compatibility Note**: This module re-exports core tool implementations from `langchain_core.tools`, maintaining backward compatibility while leveraging the standardized core abstractions.

Source: `libs/langchain/langchain_classic/tools/base.py:1-20`

---

## Key Classes and Components

### BaseTool

**Abstract base class** requiring implementation of `_run()` (synchronous) and optionally `_arun()` (asynchronous) methods.

**Required Attributes**:
- `name` (str): Unique identifier for the tool, used by agents to reference and select the tool
- `description` (str): Natural language description explaining when and how to use the tool; agents use this to determine tool applicability

**Optional Attributes**:
- `args_schema` (Type[BaseModel] | None): Pydantic model defining input argument structure and validation rules
- `return_direct` (bool): When `True`, stops the agent loop immediately after tool execution (default: `False`)
- `verbose` (bool): Enables detailed logging of tool execution (default: `False`)
- `callbacks` (Callbacks): Callback handlers for monitoring tool execution events
- `tags` (List[str] | None): Tags for categorizing and tracking tool usage
- `metadata` (Dict[str, Any] | None): Additional metadata passed to callback handlers
- `handle_tool_error` (bool | str | Callable): Configuration for handling `ToolException` errors
- `handle_validation_error` (bool | str | Callable): Configuration for handling Pydantic `ValidationError`
- `response_format` (Literal["content", "content_and_artifact"]): Defines tool output structure

**Use Case**: Custom tool implementations requiring full control over execution logic, complex validation, or specialized error handling.

Source: `libs/core/langchain_core/tools/base.py:390-850`

### StructuredTool

**Factory class** for creating tools with multiple structured arguments validated by a Pydantic schema.

**Key Features**:
- Supports multiple named parameters with comprehensive type validation
- Automatically generates input schema from function signatures
- Handles both synchronous and asynchronous execution
- Ideal for tools requiring complex input structures

**Use Case**: Tools that accept multiple parameters with specific types, constraints, or nested structures (e.g., database queries with connection parameters, API calls with authentication and request options).

Source: `libs/core/langchain_core/tools/structured.py:36-120`

### Tool

**Simple tool wrapper** for single-input functions that accept a single string parameter.

**Key Features**:
- Simplified interface for functions with single string input
- Backward-compatible with legacy tool patterns
- Automatic handling of string input conversion

**Use Case**: Basic tools with single parameter, such as simple text transformations, single-query searches, or straightforward calculations.

Source: `libs/core/langchain_core/tools/simple.py:30-150`

### @tool Decorator

**Convenience decorator** for converting functions into tools with automatic schema inference.

**Key Features**:
- Automatically infers input schema from function signature and type hints
- Supports both synchronous and asynchronous functions
- Optional manual schema override via `args_schema` parameter
- Docstring parsing for automatic description generation

**Use Case**: Quick tool creation from existing functions without manual schema definition, ideal for prototyping and simple integrations.

Source: `libs/core/langchain_core/tools/convert.py:72-300`

---

## Tool Interface Contracts

### Required Fields

#### name: str
- **Purpose**: Unique identifier for the tool
- **Usage**: Agents use this to reference and select tools during execution
- **Constraints**: Must be unique within an agent's tool set; use descriptive, action-oriented names (e.g., `search_wikipedia`, `calculate_sum`)
- **Example**: `"database_query"`, `"weather_lookup"`, `"send_email"`

#### description: str
- **Purpose**: Natural language explanation of tool's purpose and usage
- **Usage**: LLMs use this description to determine when and how to invoke the tool
- **Best Practices**:
  - Clearly state what the tool does
  - Specify input requirements and format
  - Include use case examples or constraints
  - Can include few-shot examples for complex tools
- **Example**: 
  ```python
  description = """Search Wikipedia for information about a topic.
  Input should be a search query string.
  Returns a summary of the most relevant Wikipedia article."""
  ```

Source: `libs/core/langchain_core/tools/base.py:430-436`

### Optional Fields

#### args_schema: Type[BaseModel] | None
- **Purpose**: Pydantic model defining input argument structure and validation rules
- **Type**: Subclass of `pydantic.BaseModel` or `pydantic.v1.BaseModel` (for Pydantic V2)
- **Validation**: Inputs are validated against this schema before `_run()` is called
- **Error Handling**: ValidationError is raised for invalid inputs (customizable via `handle_validation_error`)
- **Example**:
  ```python
  from pydantic import BaseModel, Field
  
  class SearchInput(BaseModel):
      query: str = Field(description="Search query string")
      max_results: int = Field(default=5, ge=1, le=50, description="Maximum results")
  ```

Source: `libs/core/langchain_core/tools/base.py:438-448`

#### return_direct: bool
- **Purpose**: Controls agent execution flow after tool invocation
- **Behavior**: When `True`, agent stops iteration and returns tool output immediately
- **Use Case**: Final-step tools like "submit_answer" or "send_message" that conclude the agent's task
- **Default**: `False`

Source: `libs/core/langchain_core/tools/base.py:449-454`

#### handle_tool_error: bool | str | Callable[[ToolException], str]
- **Purpose**: Configures error handling for `ToolException` raised during execution
- **Options**:
  - `False` (default): Raise exception to stop execution
  - `True`: Return error message as tool output (agent continues)
  - `str`: Return custom error message
  - `Callable`: Custom error handler function
- **Example**:
  ```python
  handle_tool_error = "Tool execution failed. Please try a different approach."
  ```

Source: `libs/core/langchain_core/tools/base.py:474-475`

#### handle_validation_error: bool | str | Callable[[ValidationError], str]
- **Purpose**: Configures error handling for Pydantic ValidationError
- **Options**: Same as `handle_tool_error`
- **Use Case**: Gracefully handle malformed inputs and provide guidance to the agent

Source: `libs/core/langchain_core/tools/base.py:477-480`

### Method Contracts

#### _run(*args, **kwargs) -> Any
**Synchronous tool execution method** (abstract - must be implemented by subclasses)

**Parameters**:
- `*args`: Positional arguments derived from tool input
- `**kwargs`: Keyword arguments derived from tool input (typically from dict)
- `run_manager` (Optional[CallbackManagerForToolRun]): Optional callback manager for monitoring and tracing

**Return Type**: Any (typically `str`, `dict`, or domain-specific types)

**Behavior**: 
- Called when tool is invoked synchronously via `tool.invoke()` or `tool.run()`
- Receives validated input (after `args_schema` validation)
- Should perform the tool's core functionality
- May use `run_manager` to emit progress updates

**Example Implementation**:
```python
def _run(
    self,
    query: str,
    run_manager: Optional[CallbackManagerForToolRun] = None
) -> str:
    """Execute search query."""
    if run_manager:
        run_manager.on_text(f"Searching for: {query}\n")
    
    # Perform search
    results = search_api(query)
    
    return results
```

Source: `libs/core/langchain_core/tools/base.py:684-693`

#### _arun(*args, **kwargs) -> Any
**Asynchronous tool execution method** (optional - default implementation runs `_run()` in executor)

**Parameters**: Same as `_run()` but with `AsyncCallbackManagerForToolRun`

**Return Type**: Awaitable returning same type as `_run()`

**Behavior**:
- Called when tool is invoked asynchronously via `tool.ainvoke()` or `tool.arun()`
- Default implementation wraps `_run()` in an executor thread
- Override for truly asynchronous operations (e.g., async HTTP requests)

**Example Implementation**:
```python
async def _arun(
    self,
    query: str,
    run_manager: Optional[AsyncCallbackManagerForToolRun] = None
) -> str:
    """Execute search query asynchronously."""
    if run_manager:
        await run_manager.on_text(f"Searching for: {query}\n")
    
    # Perform async search
    results = await async_search_api(query)
    
    return results
```

Source: `libs/core/langchain_core/tools/base.py:695-708`

### Exception Handling

#### ToolException
**Custom exception** for tool-specific errors that should be handled gracefully.

**Usage**: Raise when tool encounters expected failure conditions (e.g., API rate limits, resource not found, invalid state)

**Handling**: Controlled by `handle_tool_error` configuration

**Example**:
```python
from langchain_core.tools import ToolException

def _run(self, resource_id: str) -> str:
    resource = fetch_resource(resource_id)
    if resource is None:
        raise ToolException(f"Resource {resource_id} not found")
    return resource.data
```

Source: `libs/core/langchain_core/tools/base.py:84-86`

#### ValidationError
**Pydantic validation exception** raised when input doesn't match `args_schema`.

**Timing**: Raised before `_run()` is called during input parsing

**Handling**: Controlled by `handle_validation_error` configuration

**Common Causes**:
- Missing required fields
- Type mismatches (e.g., string provided for int field)
- Constraint violations (e.g., value outside Field constraints)
- Invalid nested structures

Source: `libs/core/langchain_core/tools/base.py:477-480`

---

## Usage Patterns with Executable Examples

### Pattern 1: Using @tool Decorator (Simplest)

**Best for**: Quick tool creation from existing functions, prototyping, simple single-purpose tools

```python
from langchain_classic.tools import tool
from typing import Annotated

@tool
def search_wikipedia(
    query: Annotated[str, "The search query for Wikipedia"]
) -> str:
    """Search Wikipedia for information about a topic.
    
    Args:
        query: The topic to search for
    
    Returns:
        Summary of the Wikipedia article
    """
    # In real implementation, this would call Wikipedia API
    # For demonstration, returning mock response
    return f"Wikipedia summary for '{query}': [Article content would appear here]"

# Usage in agent context
if __name__ == "__main__":
    # Direct invocation
    result = search_wikipedia.invoke("Python programming")
    print(f"Tool Name: {search_wikipedia.name}")
    print(f"Description: {search_wikipedia.description}")
    print(f"Result: {result}")
    
    # Expected Output:
    # Tool Name: search_wikipedia
    # Description: Search Wikipedia for information about a topic...
    # Result: Wikipedia summary for 'Python programming': [Article content would appear here]
```

**Key Features**:
- Automatic name inference from function name
- Automatic description from docstring
- Type hints converted to schema automatically
- Supports `Annotated` for parameter descriptions

Source: `libs/core/langchain_core/tools/convert.py:72-150`

### Pattern 2: Using StructuredTool for Multi-Argument Tools

**Best for**: Tools requiring multiple parameters with validation, complex input structures, tools with optional parameters

```python
from langchain_classic.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Literal

class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="Arithmetic operation to perform"
    )

def calculate(a: float, b: float, operation: str) -> float:
    """Perform arithmetic operations on two numbers.
    
    Args:
        a: First number
        b: Second number
        operation: Operation type (add, subtract, multiply, divide)
    
    Returns:
        Result of the arithmetic operation
    
    Raises:
        ValueError: If operation is not supported or division by zero
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }
    
    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return operations[operation](a, b)

# Create structured tool
calculator_tool = StructuredTool.from_function(
    func=calculate,
    name="Calculator",
    description="Perform arithmetic operations on two numbers. Supports add, subtract, multiply, and divide.",
    args_schema=CalculatorInput,
)

# Usage
if __name__ == "__main__":
    # Invoke with dictionary input
    result = calculator_tool.invoke({
        "a": 10,
        "b": 5,
        "operation": "multiply"
    })
    print(f"Result: {result}")  # Output: Result: 50.0
    
    # Invoke with different operation
    result = calculator_tool.invoke({
        "a": 100,
        "b": 25,
        "operation": "divide"
    })
    print(f"Result: {result}")  # Output: Result: 4.0
    
    # Schema validation prevents invalid inputs
    try:
        result = calculator_tool.invoke({
            "a": "not a number",  # Invalid type
            "b": 5,
            "operation": "add"
        })
    except Exception as e:
        print(f"Validation Error: {type(e).__name__}")
```

**Key Features**:
- Multiple parameters with type validation
- Literal types for enumerated options
- Pydantic Field constraints (ge, le, regex, etc.)
- Clear error messages for validation failures

Source: `libs/core/langchain_core/tools/structured.py:36-120`

### Pattern 3: Subclassing BaseTool for Complex Custom Tools

**Best for**: Tools with complex state, custom initialization, advanced error handling, or integration with external systems

```python
from langchain_classic.tools import BaseTool, ToolException
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
import asyncio

class DatabaseQueryInput(BaseModel):
    """Input schema for database query tool."""
    query: str = Field(description="SQL query to execute")
    database: str = Field(description="Database name to query")
    timeout: int = Field(default=30, ge=1, le=300, description="Query timeout in seconds")

class DatabaseTool(BaseTool):
    """Tool for executing SQL queries on specified databases."""
    
    name: str = "database_query"
    description: str = """Execute SQL queries on specified databases.
    Use this tool to retrieve data from production or staging databases.
    Input should include the SQL query and target database name."""
    args_schema: Type[BaseModel] = DatabaseQueryInput
    
    # Custom tool state
    connection_pool: dict = {}  # In real implementation, maintain connection pools
    
    def _run(
        self,
        query: str,
        database: str,
        timeout: int = 30,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the database query synchronously.
        
        Args:
            query: SQL query string to execute
            database: Target database name
            timeout: Query timeout in seconds
            run_manager: Optional callback manager for monitoring
        
        Returns:
            Query results formatted as string
        
        Raises:
            ToolException: If query execution fails or times out
        """
        try:
            # Emit start event
            if run_manager:
                run_manager.on_text(f"Executing query on {database}...\n")
            
            # Simulate database query execution
            # In real implementation: execute against actual database
            result = f"Executed '{query}' on {database} (timeout: {timeout}s)"
            
            # Emit completion event
            if run_manager:
                run_manager.on_text(f"Query completed successfully\n")
            
            return result
            
        except TimeoutError as e:
            raise ToolException(f"Query timed out after {timeout}s: {str(e)}")
        except Exception as e:
            raise ToolException(f"Database query failed: {str(e)}")
    
    async def _arun(
        self,
        query: str,
        database: str,
        timeout: int = 30,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the database query asynchronously.
        
        Args:
            query: SQL query string to execute
            database: Target database name
            timeout: Query timeout in seconds
            run_manager: Optional async callback manager
        
        Returns:
            Query results formatted as string
        """
        # Implement true async execution for better performance
        # In real implementation: use async database driver
        return await asyncio.to_thread(self._run, query, database, timeout, run_manager)

# Usage
if __name__ == "__main__":
    db_tool = DatabaseTool()
    
    # Synchronous invocation
    result = db_tool.invoke({
        "query": "SELECT * FROM users WHERE active = true",
        "database": "production"
    })
    print(f"Query result: {result}")
    
    # With custom timeout
    result = db_tool.invoke({
        "query": "SELECT COUNT(*) FROM transactions",
        "database": "analytics",
        "timeout": 60
    })
    print(f"Query result: {result}")
    
    # Async invocation
    async def run_async_query():
        result = await db_tool.ainvoke({
            "query": "SELECT * FROM logs ORDER BY timestamp DESC LIMIT 100",
            "database": "staging"
        })
        print(f"Async result: {result}")
    
    # asyncio.run(run_async_query())
```

**Key Features**:
- Full control over tool lifecycle and state
- Custom initialization with configuration
- Explicit callback integration for monitoring
- Both sync and async implementations
- Type-safe with Pydantic validation

Source: `libs/core/langchain_core/tools/base.py:390-850`

---

## Type Validation Patterns

### Pydantic Model Integration

Tools automatically validate inputs against `args_schema` (if provided) before execution. Validation occurs in the following sequence:

1. **Input Parsing**: Tool input (string or dict) is parsed and matched to schema
2. **Type Validation**: Pydantic validates types match field definitions
3. **Constraint Validation**: Field constraints (ge, le, regex, etc.) are checked
4. **Custom Validators**: Any `@validator` methods are executed
5. **Execution**: If validation passes, `_run()` is called with validated inputs

**ValidationError Handling**: Controlled by `handle_validation_error`:
- `False` (default): Exception propagates, stopping execution
- `True`: Error message returned as tool output
- `str`: Custom error message returned
- `Callable`: Custom error handler processes the error

### Type Annotation Best Practices

```python
from typing import Annotated, List, Optional
from pydantic import BaseModel, Field, validator
import re

class AdvancedToolInput(BaseModel):
    """Schema demonstrating advanced validation patterns."""
    
    # Email with custom validation
    email: str = Field(description="User email address")
    
    # Numeric constraints
    age: int = Field(description="User age", ge=0, le=150)
    
    # Optional with default
    preferences: List[str] = Field(
        default_factory=list,
        description="User preferences"
    )
    
    # String pattern matching
    phone: Optional[str] = Field(
        default=None,
        description="Phone number in format XXX-XXX-XXXX",
        pattern=r"^\d{3}-\d{3}-\d{4}$"
    )
    
    # Custom validator
    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if '@' not in v or '.' not in v.split('@')[1]:
            raise ValueError('Invalid email format')
        return v.lower()  # Normalize to lowercase
    
    # Dependent field validation
    @validator('preferences')
    def validate_preferences(cls, v):
        """Ensure preferences are unique."""
        if len(v) != len(set(v)):
            raise ValueError('Preferences must be unique')
        return v

# Usage with advanced validation
from langchain_classic.tools import StructuredTool

def process_user_data(
    email: str,
    age: int,
    preferences: List[str],
    phone: Optional[str] = None
) -> str:
    """Process user data with validation."""
    return f"Processed user: {email}, age {age}, {len(preferences)} preferences"

user_tool = StructuredTool.from_function(
    func=process_user_data,
    name="ProcessUser",
    description="Process user data with validation",
    args_schema=AdvancedToolInput
)

# Valid input
result = user_tool.invoke({
    "email": "User@Example.com",  # Will be normalized to lowercase
    "age": 25,
    "preferences": ["dark_mode", "notifications"]
})
print(result)  # Output: Processed user: user@example.com, age 25, 2 preferences

# Invalid input - age constraint
try:
    result = user_tool.invoke({
        "email": "user@example.com",
        "age": 200,  # Exceeds maximum
        "preferences": []
    })
except Exception as e:
    print(f"Validation failed: {e}")
```

**Key Validation Features**:
- **Field Constraints**: `ge`, `le`, `gt`, `lt`, `min_length`, `max_length`, `pattern`
- **Custom Validators**: `@validator` decorator for complex validation logic
- **Type Coercion**: Pydantic attempts type conversion (e.g., "123" → 123)
- **Nested Models**: Support for complex nested structures
- **Default Values**: `default`, `default_factory` for optional fields

Source: `libs/core/langchain_core/tools/base.py:438-448`

---

## Error Handling Best Practices

### Exception Types and Handling Strategies

#### ToolException

**Purpose**: Signals expected tool-specific errors that can be handled gracefully

**When to Use**:
- API rate limits or quota exceeded
- Resource not found (404 errors)
- Invalid state or preconditions not met
- Transient failures that agents might retry

**Handling Configuration**:
```python
from langchain_classic.tools import tool, ToolException

@tool
def risky_operation(input_data: str) -> str:
    """Tool that may fail and needs error handling."""
    if not input_data:
        raise ToolException("Input data cannot be empty")
    if len(input_data) > 1000:
        raise ToolException("Input data exceeds maximum length of 1000 characters")
    return f"Processed: {input_data}"

# Configure error handling - return custom message
safe_tool = risky_operation.copy()
safe_tool.handle_tool_error = "An error occurred while processing your request. Please provide different input."

# This will return error message instead of raising exception
result = safe_tool.invoke("")
print(result)  # Output: "An error occurred while processing your request..."

# Configure error handling - custom handler function
def custom_error_handler(error: ToolException) -> str:
    """Custom error handling with logging."""
    print(f"Tool error logged: {str(error)}")
    return f"Tool failed: {str(error)}. Attempting alternative approach recommended."

risky_operation.handle_tool_error = custom_error_handler
result = risky_operation.invoke("")
print(result)  # Output: "Tool failed: Input data cannot be empty..."
```

Source: `libs/core/langchain_core/tools/base.py:474-475`, `libs/core/langchain_core/tools/base.py:845-850`

#### ValidationError

**Purpose**: Signals input validation failures against `args_schema`

**Automatic Handling**: Raised before `_run()` is called during input parsing

**Common Scenarios**:
1. **Missing Required Fields**:
   ```python
   # Schema requires 'query' field
   tool.invoke({"database": "prod"})  # ValidationError: query field required
   ```

2. **Type Mismatches**:
   ```python
   # Schema expects int for 'age' field
   tool.invoke({"age": "twenty-five"})  # ValidationError: value is not a valid integer
   ```

3. **Constraint Violations**:
   ```python
   # Schema has Field(ge=0, le=100)
   tool.invoke({"score": 150})  # ValidationError: value must be <= 100
   ```

**Handling Configuration**:
```python
from langchain_classic.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field, ValidationError

class StrictInput(BaseModel):
    value: int = Field(ge=0, le=100, description="Value between 0 and 100")

def process_value(value: int) -> str:
    return f"Processed value: {value}"

strict_tool = StructuredTool.from_function(
    func=process_value,
    name="ProcessValue",
    description="Process a value with strict validation",
    args_schema=StrictInput,
    handle_validation_error=True  # Return error message instead of raising
)

# Invalid input - handled gracefully
result = strict_tool.invoke({"value": 150})
print(result)  # Output: "Validation error: [error details]"

# Custom validation error handler
def validation_error_handler(error: ValidationError) -> str:
    errors = error.errors()
    error_msg = "; ".join([f"{e['loc'][0]}: {e['msg']}" for e in errors])
    return f"Input validation failed: {error_msg}. Please correct your input and try again."

strict_tool.handle_validation_error = validation_error_handler
result = strict_tool.invoke({"value": "not a number"})
print(result)  # Output: "Input validation failed: value: value is not a valid integer..."
```

Source: `libs/core/langchain_core/tools/base.py:477-480`

### Error Handling Pattern Examples

#### Pattern: Graceful Degradation with Fallback

```python
from langchain_classic.tools import BaseTool, ToolException
from typing import Optional

class RobustSearchTool(BaseTool):
    """Search tool with fallback strategies."""
    
    name = "robust_search"
    description = "Search with automatic fallback to alternative sources"
    
    def _run(self, query: str) -> str:
        """Execute search with fallback logic."""
        # Try primary search
        try:
            return self._search_primary(query)
        except ToolException as e:
            # Log and try fallback
            print(f"Primary search failed: {e}, trying fallback...")
            try:
                return self._search_fallback(query)
            except ToolException:
                # Both failed - return informative message
                return f"Search unavailable for query '{query}'. Please try again later."
    
    def _search_primary(self, query: str) -> str:
        """Primary search implementation."""
        # Simulate API call that might fail
        if len(query) < 3:
            raise ToolException("Query too short for primary search")
        return f"Primary results for: {query}"
    
    def _search_fallback(self, query: str) -> str:
        """Fallback search implementation."""
        return f"Fallback results for: {query}"
```

#### Pattern: Retry Logic with Exponential Backoff

```python
import time
from langchain_classic.tools import BaseTool, ToolException

class RetryableTool(BaseTool):
    """Tool with automatic retry logic."""
    
    name = "retryable_api"
    description = "API call with automatic retry on transient failures"
    max_retries: int = 3
    
    def _run(self, request: str) -> str:
        """Execute with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self._make_api_call(request)
            except ToolException as e:
                last_error = e
                if "rate limit" in str(e).lower() and attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    print(f"Rate limited, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Non-retryable error or max retries reached
                    break
        
        # All retries exhausted
        raise ToolException(f"Failed after {self.max_retries} attempts: {last_error}")
    
    def _make_api_call(self, request: str) -> str:
        """Simulate API call that might fail transiently."""
        # Real implementation would make actual API call
        return f"API response for: {request}"
```

---

## Common Patterns and Troubleshooting

### Pattern: Tool with Callbacks for Monitoring

```python
from langchain_classic.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from typing import Optional

class MonitoredTool(BaseTool):
    """Tool with comprehensive monitoring and logging."""
    
    name = "monitored_tool"
    description = "Tool with detailed execution monitoring"
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute with detailed monitoring."""
        
        # Log start
        if run_manager:
            run_manager.on_text(f"Starting tool execution with query: {query}\n")
        
        # Processing step 1
        if run_manager:
            run_manager.on_text("Step 1: Validating input...\n")
        validated_query = query.strip()
        
        # Processing step 2
        if run_manager:
            run_manager.on_text("Step 2: Processing query...\n")
        result = f"Processed: {validated_query}"
        
        # Log completion
        if run_manager:
            run_manager.on_text(f"Completed successfully. Result length: {len(result)}\n")
        
        return result

# Usage with callbacks
from langchain_core.callbacks import StdOutCallbackHandler

tool = MonitoredTool()
result = tool.invoke(
    "test query",
    config={"callbacks": [StdOutCallbackHandler()]}
)
# Output will include all monitoring messages
```

Source: `libs/core/langchain_core/tools/base.py:684-708`

### Common Issues and Solutions

#### Issue 1: ValidationError with Dictionary Inputs

**Symptom**: Tool raises ValidationError when invoked with dict input

**Cause**: Dict structure doesn't match `args_schema` fields

**Solution**:
```python
# Problem: Missing required field
tool.invoke({"query": "test"})  # Missing 'database' field

# Solution: Provide all required fields
tool.invoke({"query": "test", "database": "prod"})

# Or: Make fields optional with defaults
class FlexibleInput(BaseModel):
    query: str
    database: str = Field(default="default_db")
```

#### Issue 2: Tool Not Accepting Multiple Arguments

**Symptom**: ToolException: "Too many arguments to single-input tool"

**Cause**: Using `Tool` class (single-input) instead of `StructuredTool` (multi-input)

**Solution**:
```python
# Problem: Tool expects single string input
simple_tool = Tool(
    name="simple",
    description="Single input tool",
    func=lambda x: x
)
simple_tool.invoke({"arg1": "a", "arg2": "b"})  # Error!

# Solution: Use StructuredTool for multiple arguments
from pydantic import BaseModel

class MultiInput(BaseModel):
    arg1: str
    arg2: str

structured_tool = StructuredTool.from_function(
    func=lambda arg1, arg2: f"{arg1}, {arg2}",
    name="structured",
    description="Multi-input tool",
    args_schema=MultiInput
)
structured_tool.invoke({"arg1": "a", "arg2": "b"})  # Works!
```

Source: `libs/core/langchain_core/tools/simple.py:86-94`

#### Issue 3: Async Tool Not Executing Asynchronously

**Symptom**: Async tool blocks execution instead of running concurrently

**Cause**: Default `_arun()` implementation wraps `_run()` in executor (still blocks)

**Solution**:
```python
# Problem: Relying on default _arun() implementation
class SlowTool(BaseTool):
    name = "slow"
    
    def _run(self, query: str) -> str:
        time.sleep(5)  # Blocking operation
        return "done"
    # No _arun() - default will still block!

# Solution: Implement true async _arun()
import asyncio

class FastAsyncTool(BaseTool):
    name = "fast_async"
    
    def _run(self, query: str) -> str:
        time.sleep(5)
        return "done"
    
    async def _arun(self, query: str) -> str:
        await asyncio.sleep(5)  # Non-blocking async operation
        return "done"
```

Source: `libs/core/langchain_core/tools/base.py:695-708`

#### Issue 4: Agent Not Selecting Tool

**Symptom**: Agent doesn't invoke tool even when appropriate

**Cause**: Tool description is unclear or missing key information

**Solution**:
```python
# Problem: Vague description
bad_tool = tool(
    func=search_function,
    name="search",
    description="Search for things"  # Too vague!
)

# Solution: Detailed description with examples
good_tool = tool(
    func=search_function,
    name="search_wikipedia",
    description="""Search Wikipedia for information about a specific topic.

Input: A search query string (e.g., "Python programming", "Albert Einstein")
Output: Summary of the most relevant Wikipedia article

Use this tool when you need:
- Factual information about people, places, concepts
- Historical context or background information
- General knowledge verification

Do NOT use for:
- Recent news or current events
- Opinions or subjective information
- Real-time data (stock prices, weather)"""
)
```

Source: `libs/core/langchain_core/tools/base.py:432-436`

---

## Source File Citations

**Re-export Compatibility Layer**:
- `libs/langchain/langchain_classic/tools/base.py:1-20` - Module exports and compatibility

**Core Implementations**:
- `libs/core/langchain_core/tools/base.py:390-850` - BaseTool abstract base class
- `libs/core/langchain_core/tools/structured.py:36-120` - StructuredTool implementation
- `libs/core/langchain_core/tools/simple.py:30-150` - Tool class for single-input tools
- `libs/core/langchain_core/tools/convert.py:72-300` - @tool decorator and conversion utilities

**Supporting Utilities**:
- `libs/core/langchain_core/tools/base.py:84-86` - SchemaAnnotationError and ToolException
- `libs/core/langchain_core/tools/base.py:684-708` - _run() and _arun() method contracts
- `libs/core/langchain_core/tools/base.py:438-448` - args_schema definition and validation
- `libs/core/langchain_core/tools/base.py:474-480` - Error handling configuration

---

## Additional Resources

**Related Documentation**:
- LangChain Agents: `libs/langchain/langchain_classic/agents/README.md`
- Callbacks System: `libs/core/langchain_core/callbacks/base.py`
- Pydantic Models: https://docs.pydantic.dev/

**External References**:
- LangChain Tools Documentation: https://docs.langchain.com/docs/components/tools
- LangChain Expression Language (LCEL): https://docs.langchain.com/docs/expression-language

---

**Note**: This is documentation-only content. All code examples are illustrative and demonstrate patterns for creating tools. No production code has been modified per the Agent Action Plan preservation requirements.
