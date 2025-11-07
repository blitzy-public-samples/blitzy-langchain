"""Debug Callback Handler for LangChain Chain Execution Event Inspection.

This module implements a comprehensive debug callback handler that provides detailed
logging of all LangChain chain execution events. It helps developers troubleshoot
chain execution issues by capturing timestamps, run IDs, inputs, outputs, and errors.

Example:
    Basic usage with a simple chain::

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from callback_debugger import DebugCallbackHandler
        
        # Initialize debug handler
        debug_handler = DebugCallbackHandler()
        
        # Create a simple chain
        prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        chain = prompt | StrOutputParser()
        
        # Execute with debug handler
        result = chain.invoke(
            {"topic": "programming"},
            config={"callbacks": [debug_handler]}
        )

Source: examples/debugging/callback_debugger.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.base import BaseCallbackHandler

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult


# Configure module logger
_LOGGER = logging.getLogger(__name__)


class DebugCallbackHandler(BaseCallbackHandler):
    """Debug callback handler for detailed LangChain execution event logging.
    
    This handler logs all major LangChain execution events including chain start/end,
    LLM interactions, tool usage, and errors. Each event is logged with:
    - Timestamp (ISO 8601 format)
    - Event type (chain, llm, tool, agent)
    - Run ID (unique identifier for this execution)
    - Parent Run ID (identifier for parent execution, if nested)
    - Event-specific data (inputs, outputs, errors, etc.)
    
    The handler is particularly useful for:
    - Understanding chain execution flow
    - Debugging nested chain compositions
    - Tracking execution time between events
    - Identifying error sources in complex chains
    - Monitoring LLM token generation in streaming mode
    
    Attributes:
        log_level: Logging level for events (default: logging.INFO)
        include_timestamps: Whether to include timestamps in log messages
        include_run_ids: Whether to include run IDs in log messages
        pretty_print: Whether to pretty-print JSON data
        
    Example:
        Configure custom logging behavior::
        
            handler = DebugCallbackHandler(
                log_level=logging.DEBUG,
                include_timestamps=True,
                pretty_print=True
            )
            
            # Use with any Runnable
            result = chain.invoke(
                inputs,
                config={"callbacks": [handler]}
            )
    
    Note:
        This handler is designed for development and debugging. For production
        use, consider implementing a custom callback handler that integrates
        with your observability infrastructure.
    """

    def __init__(
        self,
        log_level: int = logging.INFO,
        include_timestamps: bool = True,
        include_run_ids: bool = True,
        pretty_print: bool = False,
    ) -> None:
        """Initialize the debug callback handler.
        
        Args:
            log_level: Logging level for events (e.g., logging.DEBUG, logging.INFO).
                Default is logging.INFO.
            include_timestamps: If True, include ISO 8601 timestamps in log messages.
                Default is True.
            include_run_ids: If True, include run IDs and parent run IDs in log
                messages. Default is True.
            pretty_print: If True, pretty-print JSON data with indentation.
                Default is False for more compact output.
        
        Example:
            Create handler with debug-level logging::
            
                handler = DebugCallbackHandler(
                    log_level=logging.DEBUG,
                    pretty_print=True
                )
        """
        super().__init__()
        self.log_level = log_level
        self.include_timestamps = include_timestamps
        self.include_run_ids = include_run_ids
        self.pretty_print = pretty_print
        
        # Configure logger for this handler
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        # Ensure handler is attached if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _format_message(
        self,
        event_type: str,
        event_name: str,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Format a log message with optional timestamp and run ID information.
        
        Args:
            event_type: Type of event (e.g., 'chain', 'llm', 'tool', 'agent')
            event_name: Specific event name (e.g., 'start', 'end', 'error')
            run_id: Unique identifier for this execution
            parent_run_id: Identifier for parent execution, if nested
            data: Additional event data to include in the message
        
        Returns:
            Formatted log message string
        """
        parts = []
        
        # Add timestamp if enabled
        if self.include_timestamps:
            timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            parts.append(f"[{timestamp}]")
        
        # Add event type and name
        parts.append(f"{event_type.upper()}_{event_name.upper()}")
        
        # Add run ID information if enabled
        if self.include_run_ids:
            parts.append(f"run_id={run_id}")
            if parent_run_id:
                parts.append(f"parent_run_id={parent_run_id}")
        
        # Join parts into main message
        message = " ".join(parts)
        
        # Add data if provided
        if data:
            if self.pretty_print:
                data_str = json.dumps(data, indent=2, default=str)
                message += f"\n{data_str}"
            else:
                data_str = json.dumps(data, default=str)
                message += f" | {data_str}"
        
        return message

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chain starts running.
        
        Logs the chain name (if available), input data, run identifiers, and
        any associated tags or metadata.
        
        Args:
            serialized: The serialized chain, containing type and name information
            inputs: The input dictionary passed to the chain
            run_id: Unique identifier for this chain execution
            parent_run_id: Identifier for parent execution, if this is a nested chain
            tags: Optional list of tags associated with this execution
            metadata: Optional metadata dictionary for this execution
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called automatically when a chain starts::
            
                # When this executes:
                chain.invoke({"input": "test"}, config={"callbacks": [handler]})
                
                # Logs something like:
                # [2024-01-15T10:30:00.123Z] CHAIN_START run_id=abc-123 |
                # {"name": "MyChain", "inputs": {"input": "test"}}
        """
        # Extract chain name from serialized data or kwargs
        chain_name = kwargs.get("name")
        if not chain_name and serialized:
            chain_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        if not chain_name:
            chain_name = "<unknown>"
        
        # Prepare data for logging
        data = {
            "name": chain_name,
            "inputs": inputs,
        }
        
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata
        
        # Log the event
        message = self._format_message(
            event_type="chain",
            event_name="start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.log(self.log_level, message)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chain ends running.
        
        Logs the chain output data and run identifiers. Useful for tracking
        what data is produced by each chain in a composition.
        
        Args:
            outputs: The output dictionary produced by the chain
            run_id: Unique identifier for this chain execution
            parent_run_id: Identifier for parent execution, if this is a nested chain
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called automatically when a chain completes::
            
                # After chain execution completes
                # Logs something like:
                # [2024-01-15T10:30:01.456Z] CHAIN_END run_id=abc-123 |
                # {"outputs": {"result": "Chain completed successfully"}}
        """
        data = {"outputs": outputs}
        
        message = self._format_message(
            event_type="chain",
            event_name="end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.log(self.log_level, message)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chain errors.
        
        Logs the error type, message, and run identifiers. Critical for
        debugging chain failures and understanding error propagation in
        nested chains.
        
        Args:
            error: The exception that occurred during chain execution
            run_id: Unique identifier for this chain execution
            parent_run_id: Identifier for parent execution, if this is a nested chain
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called automatically when a chain raises an exception::
            
                # If chain execution fails
                # Logs something like:
                # [2024-01-15T10:30:01.789Z] CHAIN_ERROR run_id=abc-123 |
                # {"error_type": "ValueError", "error_message": "Invalid input"}
        """
        data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        
        message = self._format_message(
            event_type="chain",
            event_name="error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.error(message)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running.
        
        Logs the LLM model name (if available), prompts being sent, and
        execution context. Note: For chat models, on_chat_model_start is
        called instead.
        
        Args:
            serialized: The serialized LLM configuration
            prompts: List of prompt strings being sent to the LLM
            run_id: Unique identifier for this LLM call
            parent_run_id: Identifier for parent execution (e.g., the chain)
            tags: Optional list of tags associated with this execution
            metadata: Optional metadata dictionary for this execution
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called automatically when an LLM is invoked::
            
                # When LLM is called:
                llm.invoke("What is 2+2?", config={"callbacks": [handler]})
                
                # Logs something like:
                # [2024-01-15T10:30:02.000Z] LLM_START run_id=def-456
                # parent_run_id=abc-123 |
                # {"model": "gpt-3.5-turbo", "prompts": ["What is 2+2?"]}
        """
        # Extract model name if available
        model_name = None
        if serialized:
            model_name = serialized.get("name") or serialized.get("id", ["<unknown>"])[-1]
        
        data = {
            "model": model_name,
            "prompts": prompts,
            "num_prompts": len(prompts),
        }
        
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata
        
        message = self._format_message(
            event_type="llm",
            event_name="start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.log(self.log_level, message)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running.
        
        Logs the chat model name (if available), messages being sent, and
        execution context. This is the chat model equivalent of on_llm_start.
        
        Args:
            serialized: The serialized chat model configuration
            messages: List of message lists being sent to the chat model
            run_id: Unique identifier for this chat model call
            parent_run_id: Identifier for parent execution (e.g., the chain)
            tags: Optional list of tags associated with this execution
            metadata: Optional metadata dictionary for this execution
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called automatically when a chat model is invoked::
            
                # When chat model is called:
                chat_model.invoke(messages, config={"callbacks": [handler]})
                
                # Logs something like:
                # [2024-01-15T10:30:02.100Z] LLM_START run_id=def-456 |
                # {"model": "gpt-4", "num_message_lists": 1, "total_messages": 2}
        """
        # Extract model name if available
        model_name = None
        if serialized:
            model_name = serialized.get("name") or serialized.get("id", ["<unknown>"])[-1]
        
        # Count total messages across all lists
        total_messages = sum(len(msg_list) for msg_list in messages)
        
        data = {
            "model": model_name,
            "num_message_lists": len(messages),
            "total_messages": total_messages,
        }
        
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata
        
        message = self._format_message(
            event_type="llm",
            event_name="start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.log(self.log_level, message)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running.
        
        Logs the LLM response including generated text and token usage
        information (if available). Useful for monitoring LLM costs and
        performance.
        
        Args:
            response: The LLMResult containing generations and metadata
            run_id: Unique identifier for this LLM call
            parent_run_id: Identifier for parent execution
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called automatically when an LLM completes::
            
                # After LLM completes
                # Logs something like:
                # [2024-01-15T10:30:03.000Z] LLM_END run_id=def-456 |
                # {"num_generations": 1, "llm_output": {"token_usage": {...}}}
        """
        data = {
            "num_generations": len(response.generations),
        }
        
        # Include token usage if available
        if response.llm_output:
            data["llm_output"] = response.llm_output
        
        message = self._format_message(
            event_type="llm",
            event_name="end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.log(self.log_level, message)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors.
        
        Logs the error type and message when an LLM call fails. Common errors
        include rate limits, authentication failures, and timeout errors.
        
        Args:
            error: The exception that occurred during LLM execution
            run_id: Unique identifier for this LLM call
            parent_run_id: Identifier for parent execution
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called automatically when an LLM call fails::
            
                # If LLM call fails
                # Logs something like:
                # [2024-01-15T10:30:03.500Z] LLM_ERROR run_id=def-456 |
                # {"error_type": "RateLimitError", "error_message": "Rate limit exceeded"}
        """
        data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        
        message = self._format_message(
            event_type="llm",
            event_name="error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.error(message)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a tool starts running.
        
        Logs the tool name and input string. Useful for tracking agent tool
        usage and debugging tool execution flows.
        
        Args:
            serialized: The serialized tool configuration
            input_str: The input string passed to the tool
            run_id: Unique identifier for this tool execution
            parent_run_id: Identifier for parent execution (e.g., the agent)
            tags: Optional list of tags associated with this execution
            metadata: Optional metadata dictionary for this execution
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called automatically when an agent uses a tool::
            
                # When tool is invoked:
                # Logs something like:
                # [2024-01-15T10:30:04.000Z] TOOL_START run_id=ghi-789
                # parent_run_id=abc-123 |
                # {"tool": "Calculator", "input": "2+2"}
        """
        # Extract tool name if available
        tool_name = None
        if serialized:
            tool_name = serialized.get("name") or serialized.get("id", ["<unknown>"])[-1]
        
        data = {
            "tool": tool_name,
            "input": input_str,
        }
        
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata
        
        message = self._format_message(
            event_type="tool",
            event_name="start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.log(self.log_level, message)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a tool ends running.
        
        Logs the tool output. Useful for verifying tool results and debugging
        tool behavior in agent workflows.
        
        Args:
            output: The output returned by the tool
            run_id: Unique identifier for this tool execution
            parent_run_id: Identifier for parent execution
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called automatically when a tool completes::
            
                # After tool execution completes
                # Logs something like:
                # [2024-01-15T10:30:04.500Z] TOOL_END run_id=ghi-789 |
                # {"output": "4", "output_type": "str"}
        """
        data = {
            "output": str(output),
            "output_type": type(output).__name__,
        }
        
        message = self._format_message(
            event_type="tool",
            event_name="end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.log(self.log_level, message)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a tool errors.
        
        Logs the error type and message when a tool execution fails. Critical
        for debugging agent tool failures and understanding why agent workflows
        fail.
        
        Args:
            error: The exception that occurred during tool execution
            run_id: Unique identifier for this tool execution
            parent_run_id: Identifier for parent execution
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called automatically when a tool raises an exception::
            
                # If tool execution fails
                # Logs something like:
                # [2024-01-15T10:30:04.800Z] TOOL_ERROR run_id=ghi-789 |
                # {"error_type": "ValueError", "error_message": "Invalid calculation"}
        """
        data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        
        message = self._format_message(
            event_type="tool",
            event_name="error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.error(message)

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action.
        
        Logs the agent's decision including tool choice, tool input, and
        reasoning log. Essential for understanding agent decision-making.
        
        Args:
            action: The AgentAction containing tool, input, and log
            run_id: Unique identifier for this agent action
            parent_run_id: Identifier for parent execution
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called when an agent decides to use a tool::
            
                # When agent chooses action
                # Logs something like:
                # [2024-01-15T10:30:05.000Z] AGENT_ACTION run_id=jkl-012 |
                # {"tool": "Calculator", "tool_input": "2+2", "log": "I need to..."}
        """
        data = {
            "tool": action.tool,
            "tool_input": str(action.tool_input),
            "log": action.log,
        }
        
        message = self._format_message(
            event_type="agent",
            event_name="action",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.log(self.log_level, message)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent finish.
        
        Logs the agent's final output and reasoning log. Marks the end of
        an agent's execution loop.
        
        Args:
            finish: The AgentFinish containing return values and log
            run_id: Unique identifier for this agent execution
            parent_run_id: Identifier for parent execution
            **kwargs: Additional keyword arguments
        
        Example:
            This method is called when an agent completes its task::
            
                # When agent finishes
                # Logs something like:
                # [2024-01-15T10:30:06.000Z] AGENT_FINISH run_id=jkl-012 |
                # {"return_values": {"output": "The answer is 4"}, "log": "..."}
        """
        data = {
            "return_values": finish.return_values,
            "log": finish.log,
        }
        
        message = self._format_message(
            event_type="agent",
            event_name="finish",
            run_id=run_id,
            parent_run_id=parent_run_id,
            data=data,
        )
        self.logger.log(self.log_level, message)


def main() -> None:
    """Demonstrate the DebugCallbackHandler with a complete example.
    
    This example shows how to use the debug callback handler to inspect
    the execution flow of a simple LangChain chain. It demonstrates:
    1. Handler initialization with custom settings
    2. Chain creation using LCEL
    3. Chain execution with the debug handler
    4. Event sequence logging
    
    The example uses a mock chat model to avoid requiring API keys,
    making it immediately runnable for demonstration purposes.
    """
    import sys
    from typing import Any
    
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    # Configure logging to see debug output
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        stream=sys.stdout
    )
    
    # Define a mock chat model for demonstration (no API key required)
    class MockChatModel(BaseChatModel):
        """Mock chat model that returns predefined responses."""
        
        @property
        def _llm_type(self) -> str:
            """Return identifier for this LLM type."""
            return "mock"
        
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Generate a mock response."""
            # Extract the last message for context
            last_message = messages[-1].content if messages else ""
            
            # Generate a simple mock response
            response_text = f"Mock response to: {last_message}"
            message = AIMessage(content=response_text)
            generation = ChatGeneration(message=message)
            
            return ChatResult(
                generations=[generation],
                llm_output={"token_usage": {"total_tokens": 10}}
            )
    
    print("=" * 70)
    print("DebugCallbackHandler Example")
    print("=" * 70)
    print()
    
    # Initialize the debug callback handler with custom settings
    print("1. Initializing DebugCallbackHandler...")
    debug_handler = DebugCallbackHandler(
        log_level=logging.INFO,
        include_timestamps=True,
        include_run_ids=True,
        pretty_print=True
    )
    print("   ✓ Handler initialized with pretty printing enabled")
    print()
    
    # Create a simple chain using LCEL
    print("2. Creating a simple LCEL chain (Prompt | Model | Parser)...")
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer this question: {question}"
    )
    model = MockChatModel()
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    print("   ✓ Chain created successfully")
    print()
    
    # Execute the chain with the debug handler
    print("3. Executing chain with debug callback handler...")
    print("-" * 70)
    
    try:
        result = chain.invoke(
            {"question": "What is the capital of France?"},
            config={"callbacks": [debug_handler]}
        )
        
        print("-" * 70)
        print()
        print("4. Chain execution completed successfully!")
        print(f"   Result: {result}")
        print()
        
        print("5. Event Sequence Summary:")
        print("   The debug handler logged the following events:")
        print("   a. CHAIN_START - Chain begins execution with inputs")
        print("   b. CHAIN_START - Prompt template formatting (nested)")
        print("   c. CHAIN_END   - Prompt template produces formatted messages")
        print("   d. LLM_START   - Chat model receives messages")
        print("   e. LLM_END     - Chat model returns response")
        print("   f. CHAIN_START - Output parser begins processing")
        print("   g. CHAIN_END   - Output parser extracts string")
        print("   h. CHAIN_END   - Main chain completes with final output")
        print()
        
    except Exception as e:
        print("-" * 70)
        print()
        print(f"4. Chain execution failed: {e}")
        print("   The debug handler logged error events showing where failure occurred")
        print()
    
    print("=" * 70)
    print("Example Complete")
    print("=" * 70)
    print()
    print("Usage Tips:")
    print("- Use this handler during development to understand chain behavior")
    print("- Enable pretty_print for easier reading of complex data structures")
    print("- Check parent_run_id to understand nested chain relationships")
    print("- Monitor timestamps to identify performance bottlenecks")
    print("- For production, implement a custom handler for your observability stack")
    print()


if __name__ == "__main__":
    main()

