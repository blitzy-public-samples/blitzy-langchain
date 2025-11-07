"""LangChain Classic Tools Compatibility Layer.

This module provides a compatibility layer that re-exports core tool classes and utilities
from langchain_core.tools, maintaining backward compatibility for legacy langchain_classic
imports while directing users to the authoritative implementations in langchain_core.

Re-exported Symbols:
    BaseTool: Abstract base class for creating custom tools with full control over
        execution logic, input schema validation, and error handling.
    
    StructuredTool: Tool implementation for functions requiring multiple named arguments,
        automatically generating input schemas from function signatures or Pydantic models.
    
    Tool: Simple tool wrapper for functions accepting a single string input, ideal for
        basic tool implementations with minimal configuration.
    
    ToolException: Exception class for tool execution errors, providing structured error
        handling within agent tool invocations.
    
    SchemaAnnotationError: Exception raised when tool input schema annotations are invalid
        or incompatible with the expected Pydantic model structure.
    
    create_schema_from_function: Utility function that generates Pydantic input schemas
        from Python function signatures, enabling automatic schema creation for tools.
    
    tool: Decorator for converting standard Python functions into LangChain tools,
        providing the most convenient method for creating simple tools.

Usage Guidance:
    - Use BaseTool when you need full control over tool implementation, custom validation
      logic, or complex execution patterns (e.g., async operations, stateful tools).
    
    - Use StructuredTool when your tool function requires multiple named arguments or
      when you have an existing function you want to wrap with explicit schema control.
    
    - Use Tool for simple single-input tools where the input is a string and minimal
      configuration is needed (e.g., search tools, simple API wrappers).
    
    - Use @tool decorator for the most convenient tool creation from existing functions,
      automatically handling schema generation and tool registration.

Source:
    All implementations are defined in langchain_core.tools. This module serves only as
    a compatibility re-export layer.
    
    Authoritative Source: langchain_core.tools
    Documentation: https://python.langchain.com/docs/modules/agents/tools/

Migration Note:
    This module is part of the langchain_classic compatibility surface. For new
    development, prefer importing directly from langchain_core.tools to access the
    canonical implementations and latest features. This re-export layer is maintained
    for backward compatibility with existing code using langchain_classic imports.

Example:
    Basic tool creation using the decorator pattern::

        from langchain_classic.tools import tool

        @tool
        def search_tool(query: str) -> str:
            '''Search for information about a query.'''
            # Implementation here
            return f"Results for: {query}"

    Creating a structured tool with multiple inputs::

        from langchain_classic.tools import StructuredTool

        def complex_tool(param1: str, param2: int) -> str:
            return f"Processed {param1} with {param2}"

        tool = StructuredTool.from_function(
            func=complex_tool,
            name="complex_tool",
            description="A tool with multiple parameters"
        )
"""
from langchain_core.tools import (
    BaseTool,
    SchemaAnnotationError,
    StructuredTool,
    Tool,
    ToolException,
    create_schema_from_function,
    tool,
)

__all__ = [
    "BaseTool",
    "SchemaAnnotationError",
    "StructuredTool",
    "Tool",
    "ToolException",
    "create_schema_from_function",
    "tool",
]
