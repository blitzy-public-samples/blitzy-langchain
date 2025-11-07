"""Logging wrapper decorator for granular LangChain chain execution monitoring.

This module provides a reusable decorator for wrapping LangChain chain objects
to add comprehensive execution logging including inputs, outputs, execution time,
and error tracking with configurable log levels.

Example:
    Basic usage with a simple chain::

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        
        chain = ChatPromptTemplate.from_template("Hello {name}") | ChatOpenAI()
        logged_chain = log_chain_execution(chain, log_level="INFO")
        result = logged_chain.invoke({"name": "World"})

Source: examples/debugging/logging_wrapper.py
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar, cast

# Configure logging format for detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Type variables for generic function signatures
T = TypeVar('T')
ChainT = TypeVar('ChainT')


def log_chain_execution(
    chain: Any,
    log_level: str = "INFO",
    logger_name: str | None = None,
    include_inputs: bool = True,
    include_outputs: bool = True,
    include_timing: bool = True,
) -> Any:
    """Wrap a LangChain chain to add comprehensive execution logging.

    This decorator wraps the invoke() and ainvoke() methods of a LangChain chain
    to provide granular logging of execution details including inputs, outputs,
    execution time, and error information.

    Args:
        chain: The LangChain chain object to wrap (must have invoke/ainvoke methods).
        log_level: Logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to "INFO".
        logger_name: Custom logger name. If None, uses the chain's class name.
            Defaults to None.
        include_inputs: Whether to log input values. Useful to disable for
            sensitive data. Defaults to True.
        include_outputs: Whether to log output values. Useful to disable for
            large outputs. Defaults to True.
        include_timing: Whether to log execution time. Defaults to True.

    Returns:
        The wrapped chain object with logging functionality added to invoke
        and ainvoke methods.

    Raises:
        AttributeError: If the chain object does not have invoke or ainvoke methods.
        ValueError: If log_level is not a valid logging level.

    Example:
        Wrapping a simple chain::

            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_openai import ChatOpenAI

            # Create a simple chain
            prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
            model = ChatOpenAI(model="gpt-3.5-turbo")
            parser = StrOutputParser()
            chain = prompt | model | parser

            # Wrap with logging
            logged_chain = log_chain_execution(
                chain,
                log_level="DEBUG",
                include_inputs=True,
                include_outputs=True
            )

            # Execute - all details will be logged
            result = logged_chain.invoke({"topic": "programming"})
            print(result)
    """
    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create logger
    logger_name = logger_name or chain.__class__.__name__
    logger = logging.getLogger(f"chain_logger.{logger_name}")
    logger.setLevel(numeric_level)

    # Store original methods
    original_invoke = getattr(chain, 'invoke', None)
    original_ainvoke = getattr(chain, 'ainvoke', None)

    if not original_invoke and not original_ainvoke:
        raise AttributeError(
            f"Chain object {chain.__class__.__name__} must have "
            "invoke or ainvoke method"
        )

    def _log_execution_start(method_name: str, inputs: Any) -> None:
        """Log the start of chain execution.

        Args:
            method_name: Name of the method being executed (invoke/ainvoke).
            inputs: Input parameters to the chain.
        """
        logger.log(numeric_level, f"=== Starting {method_name} ===")
        if include_inputs:
            logger.log(numeric_level, f"Inputs: {_format_value(inputs)}")

    def _log_execution_end(
        method_name: str,
        outputs: Any,
        elapsed_time: float
    ) -> None:
        """Log the successful completion of chain execution.

        Args:
            method_name: Name of the method that was executed.
            outputs: Output results from the chain.
            elapsed_time: Time taken for execution in seconds.
        """
        if include_outputs:
            logger.log(numeric_level, f"Outputs: {_format_value(outputs)}")
        if include_timing:
            logger.log(
                numeric_level,
                f"Execution time: {elapsed_time:.4f} seconds"
            )
        logger.log(numeric_level, f"=== Completed {method_name} ===\n")

    def _log_execution_error(
        method_name: str,
        error: Exception,
        elapsed_time: float
    ) -> None:
        """Log an error that occurred during chain execution.

        Args:
            method_name: Name of the method that failed.
            error: The exception that was raised.
            elapsed_time: Time elapsed before the error in seconds.
        """
        logger.error(
            f"Error in {method_name} after {elapsed_time:.4f}s: "
            f"{error.__class__.__name__}: {str(error)}"
        )
        logger.error(f"=== Failed {method_name} ===\n")

    def _format_value(value: Any, max_length: int = 500) -> str:
        """Format a value for logging with length truncation.

        Args:
            value: The value to format.
            max_length: Maximum string length before truncation. Defaults to 500.

        Returns:
            Formatted string representation of the value.
        """
        str_value = str(value)
        if len(str_value) > max_length:
            return f"{str_value[:max_length]}... (truncated)"
        return str_value

    if original_invoke:
        @functools.wraps(original_invoke)
        def logged_invoke(*args: Any, **kwargs: Any) -> Any:
            """Wrapped invoke method with logging.

            Args:
                *args: Positional arguments to pass to the original invoke.
                **kwargs: Keyword arguments to pass to the original invoke.

            Returns:
                The result from the original invoke method.

            Raises:
                Exception: Any exception raised by the original invoke method.
            """
            # Extract input for logging (first positional arg or 'input' kwarg)
            input_value = args[0] if args else kwargs.get('input', kwargs)
            
            _log_execution_start("invoke", input_value)
            start_time = time.time()
            
            try:
                result = original_invoke(*args, **kwargs)
                elapsed = time.time() - start_time
                _log_execution_end("invoke", result, elapsed)
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                _log_execution_error("invoke", e, elapsed)
                raise

        chain.invoke = logged_invoke

    if original_ainvoke:
        @functools.wraps(original_ainvoke)
        async def logged_ainvoke(*args: Any, **kwargs: Any) -> Any:
            """Wrapped ainvoke method with logging.

            Args:
                *args: Positional arguments to pass to the original ainvoke.
                **kwargs: Keyword arguments to pass to the original ainvoke.

            Returns:
                The result from the original ainvoke method.

            Raises:
                Exception: Any exception raised by the original ainvoke method.
            """
            # Extract input for logging
            input_value = args[0] if args else kwargs.get('input', kwargs)
            
            _log_execution_start("ainvoke", input_value)
            start_time = time.time()
            
            try:
                result = await original_ainvoke(*args, **kwargs)
                elapsed = time.time() - start_time
                _log_execution_end("ainvoke", result, elapsed)
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                _log_execution_error("ainvoke", e, elapsed)
                raise

        chain.ainvoke = logged_ainvoke

    return chain


def create_logging_wrapper(
    log_level: str = "INFO",
    logger_name: str | None = None,
    **wrapper_kwargs: Any
) -> Callable[[Any], Any]:
    """Create a reusable logging wrapper with pre-configured settings.

    This factory function creates a logging wrapper with specific configuration
    that can be applied to multiple chains consistently.

    Args:
        log_level: Default logging level for all wrapped chains. Defaults to "INFO".
        logger_name: Base logger name for all wrapped chains. Defaults to None.
        **wrapper_kwargs: Additional keyword arguments to pass to log_chain_execution.

    Returns:
        A function that wraps chains with the specified logging configuration.

    Example:
        Creating a reusable wrapper::

            # Create wrapper with DEBUG level
            debug_wrapper = create_logging_wrapper(
                log_level="DEBUG",
                include_timing=True
            )

            # Apply to multiple chains
            chain1 = debug_wrapper(my_chain_1)
            chain2 = debug_wrapper(my_chain_2)
            chain3 = debug_wrapper(my_chain_3)
    """
    def wrapper(chain: Any) -> Any:
        """Wrap a chain with the pre-configured logging settings.

        Args:
            chain: The chain to wrap.

        Returns:
            The wrapped chain.
        """
        return log_chain_execution(
            chain,
            log_level=log_level,
            logger_name=logger_name,
            **wrapper_kwargs
        )
    
    return wrapper


if __name__ == "__main__":
    """Example usage demonstrating the logging wrapper with a mock chain."""
    
    print("=" * 70)
    print("LangChain Logging Wrapper Example")
    print("=" * 70)
    print()
    
    # Example 1: Mock chain for demonstration without API keys
    print("Example 1: Basic logging wrapper usage with mock chain")
    print("-" * 70)
    
    class MockChain:
        """Mock chain for demonstration purposes."""
        
        def invoke(self, input_data: dict[str, Any]) -> dict[str, Any]:
            """Mock invoke method that processes input.

            Args:
                input_data: Input dictionary with a 'query' key.

            Returns:
                Dictionary with processed result.
            """
            time.sleep(0.1)  # Simulate processing time
            return {
                "result": f"Processed: {input_data.get('query', 'no query')}",
                "status": "success"
            }
        
        async def ainvoke(self, input_data: dict[str, Any]) -> dict[str, Any]:
            """Mock async invoke method.

            Args:
                input_data: Input dictionary with a 'query' key.

            Returns:
                Dictionary with processed result.
            """
            import asyncio
            await asyncio.sleep(0.1)  # Simulate async processing
            return {
                "result": f"Async processed: {input_data.get('query', 'no query')}",
                "status": "success"
            }
    
    # Create mock chain
    mock_chain = MockChain()
    
    # Wrap with logging
    logged_chain = log_chain_execution(
        mock_chain,
        log_level="INFO",
        include_inputs=True,
        include_outputs=True,
        include_timing=True
    )
    
    # Execute and see logging output
    print("\nExecuting logged chain...")
    result = logged_chain.invoke({"query": "What is LangChain?"})
    print(f"Final result: {result}\n")
    
    # Example 2: Using create_logging_wrapper for multiple chains
    print("\nExample 2: Using factory function for consistent logging")
    print("-" * 70)
    
    # Create reusable wrapper
    debug_wrapper = create_logging_wrapper(
        log_level="DEBUG",
        include_inputs=True,
        include_outputs=True
    )
    
    # Apply to new chain
    mock_chain_2 = MockChain()
    logged_chain_2 = debug_wrapper(mock_chain_2)
    
    print("\nExecuting second logged chain...")
    result_2 = logged_chain_2.invoke({"query": "How does logging work?"})
    print(f"Final result: {result_2}\n")
    
    # Example 3: Error handling demonstration
    print("\nExample 3: Error handling and logging")
    print("-" * 70)
    
    class ErrorChain:
        """Mock chain that raises an error for demonstration."""
        
        def invoke(self, input_data: dict[str, Any]) -> dict[str, Any]:
            """Mock invoke that raises an error.

            Args:
                input_data: Input data.

            Returns:
                Never returns, always raises ValueError.

            Raises:
                ValueError: Always raised for demonstration.
            """
            time.sleep(0.05)
            raise ValueError("Simulated chain execution error")
    
    error_chain = ErrorChain()
    logged_error_chain = log_chain_execution(error_chain, log_level="ERROR")
    
    print("\nExecuting chain that will fail...")
    try:
        logged_error_chain.invoke({"query": "This will fail"})
    except ValueError as e:
        print(f"Caught exception as expected: {e}\n")
    
    # Example 4: Async execution demonstration
    print("\nExample 4: Async chain execution with logging")
    print("-" * 70)
    
    import asyncio
    
    async def async_example() -> None:
        """Demonstrate async chain logging."""
        mock_async_chain = MockChain()
        logged_async_chain = log_chain_execution(
            mock_async_chain,
            log_level="INFO"
        )
        
        print("\nExecuting async logged chain...")
        result = await logged_async_chain.ainvoke(
            {"query": "Async query example"}
        )
        print(f"Final async result: {result}\n")
    
    # Run async example
    asyncio.run(async_example())
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print()
    print("Usage Summary:")
    print("- Use log_chain_execution() to wrap individual chains")
    print("- Use create_logging_wrapper() for consistent configuration")
    print("- Configure log_level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    print("- Enable/disable inputs, outputs, or timing as needed")
    print("- Works with both sync (invoke) and async (ainvoke) methods")
