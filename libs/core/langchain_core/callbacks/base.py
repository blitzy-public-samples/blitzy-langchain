"""Base callback handler and manager classes for LangChain observability.

This module provides the core callback system for monitoring and tracing
execution across LangChain components including chains, LLMs, agents, tools,
and retrievers. The callback architecture uses a mixin pattern to organize
callback methods by component type, enabling flexible event handling for
logging, metrics, tracing, and custom observability needs.

Key Classes:
    - BaseCallbackHandler: Synchronous base class for custom callback handlers
    - AsyncCallbackHandler: Async variant for async chain execution
    - BaseCallbackManager: Manages multiple handlers and callback lifecycle

See Also:
    langchain_core.callbacks.manager: Higher-level callback management
    langchain_core.callbacks.stdout: Built-in stdout handlers
    langchain_core.callbacks.streaming_stdout: Built-in streaming handlers

Source: libs/core/langchain_core/callbacks/base.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from tenacity import RetryCallState

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

_LOGGER = logging.getLogger(__name__)


class RetrieverManagerMixin:
    """Mixin for Retriever callbacks."""

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever errors during document retrieval.

        This callback is invoked when an exception occurs during retriever
        execution, such as connection failures, timeout errors, or document
        processing errors. Use this to log retrieval failures, implement
        fallback strategies, or track error metrics.

        Args:
            error: The exception that occurred during retrieval. Common types
                include TimeoutError, ConnectionError, or ValueError from
                document processing failures.
            run_id: The run ID. This is the unique identifier for the current
                retriever operation.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                or component that initiated this retrieval, if any. Use this to
                trace nested executions.
            **kwargs: Additional keyword arguments. May include implementation-
                specific context about the retrieval attempt.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging

            logger = logging.getLogger(__name__)

            class RetrieverErrorHandler(BaseCallbackHandler):
                def on_retriever_error(
                    self,
                    error: BaseException,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    logger.error(
                        f"Retrieval failed (run_id={run_id}): {type(error).__name__}: {error}",
                        exc_info=True
                    )
                    if isinstance(error, TimeoutError):
                        logger.warning("Consider increasing retriever timeout")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:45
        """

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever completes successfully.

        This callback is invoked after retriever execution completes with
        retrieved documents. Use this to log retrieval results, track metrics
        like document count and relevance scores, or perform post-processing.

        Args:
            documents: The documents retrieved by the retriever. Each document
                contains page_content (str) and metadata (dict). May be empty if
                no relevant documents were found.
            run_id: The run ID. This is the unique identifier for the current
                retriever operation.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                or component that initiated this retrieval, if any. Use this to
                trace nested executions.
            **kwargs: Additional keyword arguments. May include query information
                or retriever-specific metadata.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from langchain_core.documents import Document
            from collections.abc import Sequence
            from uuid import UUID
            from typing import Any
            import logging

            logger = logging.getLogger(__name__)

            class RetrieverMetricsHandler(BaseCallbackHandler):
                def on_retriever_end(
                    self,
                    documents: Sequence[Document],
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    doc_count = len(documents)
                    logger.info(f"Retrieved {doc_count} documents (run_id={run_id})")
                    
                    if doc_count == 0:
                        logger.warning("No documents retrieved - query may be too specific")
                    
                    # Log document metadata
                    for i, doc in enumerate(documents[:3]):  # First 3 docs
                        logger.debug(
                            f"Doc {i+1}: {len(doc.page_content)} chars, "
                            f"metadata={doc.metadata}"
                        )
            ```

        Source: libs/core/langchain_core/callbacks/base.py:62
        """


class LLMManagerMixin:
    """Mixin for LLM callbacks."""

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on new output token during streaming LLM generation.

        This callback is invoked for each token generated during streaming mode.
        Only available when streaming is enabled on the LLM. Fired for both chat
        models and legacy LLMs. Use this to implement real-time token display,
        streaming responses to users, or token-level processing.

        Args:
            token: The new token string generated by the model. This is the raw
                text content of the token.
            chunk: The complete generation chunk object containing the token plus
                metadata. For ChatGenerationChunk includes message type and role.
                For GenerationChunk includes generation info. May be None for
                simple streaming implementations.
            run_id: The run ID. This is the unique identifier for the current
                LLM generation operation.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                or component that initiated this LLM call, if any. Use this to
                trace nested executions.
            **kwargs: Additional keyword arguments. May include model-specific
                metadata like finish_reason or token logprobs.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code and interrupt streaming if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from langchain_core.outputs import GenerationChunk, ChatGenerationChunk
            from uuid import UUID
            from typing import Any
            import sys

            class StreamingHandler(BaseCallbackHandler):
                \"\"\"Handler that displays tokens in real-time.\"\"\"
                
                def on_llm_new_token(
                    self,
                    token: str,
                    *,
                    chunk: GenerationChunk | ChatGenerationChunk | None = None,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    # Display token immediately (no newline)
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    
                    # Log chunk metadata if available
                    if chunk and hasattr(chunk, 'generation_info'):
                        info = chunk.generation_info
                        if info and 'finish_reason' in info:
                            print(f"\\n[Finished: {info['finish_reason']}]")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:83
        """

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM completes generation successfully.

        This callback is invoked after LLM generation completes, receiving the
        full response with all generations and metadata. Use this to log outputs,
        track token usage and costs, collect performance metrics, or implement
        response validation.

        Args:
            response: The complete LLM response object containing:
                - generations: List of lists of Generation objects with text outputs
                - llm_output: Dict with model metadata (token_usage, model_name, etc.)
                - run: Optional run information
            run_id: The run ID. This is the unique identifier for the current
                LLM generation operation.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                or component that initiated this LLM call, if any. Use this to
                trace nested executions.
            **kwargs: Additional keyword arguments. May include prompt information
                or model-specific context.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from langchain_core.outputs import LLMResult
            from uuid import UUID
            from typing import Any
            import logging

            logger = logging.getLogger(__name__)

            class LLMMetricsHandler(BaseCallbackHandler):
                \"\"\"Handler that tracks LLM usage and costs.\"\"\"
                
                def __init__(self):
                    self.total_tokens = 0
                    self.total_cost = 0.0
                
                def on_llm_end(
                    self,
                    response: LLMResult,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    # Extract token usage
                    if response.llm_output and 'token_usage' in response.llm_output:
                        usage = response.llm_output['token_usage']
                        total_tokens = usage.get('total_tokens', 0)
                        self.total_tokens += total_tokens
                        
                        # Estimate cost (example for GPT-3.5)
                        cost = total_tokens * 0.000002  # $0.002 per 1K tokens
                        self.total_cost += cost
                        
                        logger.info(
                            f"LLM completed: {total_tokens} tokens, "
                            f"${cost:.4f} (run_id={run_id})"
                        )
                    
                    # Log first generation
                    if response.generations and response.generations[0]:
                        first_gen = response.generations[0][0]
                        logger.debug(f"Generated: {first_gen.text[:100]}...")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:122
        """

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM generation fails with an error.

        This callback is invoked when an exception occurs during LLM execution,
        such as API errors, rate limiting, timeout errors, or invalid requests.
        Use this to log failures, implement retry logic, track error metrics,
        or send alerts.

        Args:
            error: The exception that occurred during LLM generation. Common
                types include:
                - RateLimitError: API rate limit exceeded
                - AuthenticationError: Invalid API key or credentials
                - TimeoutError: Request timeout
                - ValueError: Invalid request parameters
                - ConnectionError: Network connectivity issues
            run_id: The run ID. This is the unique identifier for the current
                LLM generation operation.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                or component that initiated this LLM call, if any. Use this to
                trace nested executions.
            **kwargs: Additional keyword arguments. May include:
                - prompts: The input prompts that caused the error
                - invocation_params: Model parameters used in the failed call

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging
            import time

            logger = logging.getLogger(__name__)

            class LLMErrorHandler(BaseCallbackHandler):
                \"\"\"Handler that categorizes and logs LLM errors.\"\"\"
                
                def __init__(self):
                    self.error_counts = {}
                
                def on_llm_error(
                    self,
                    error: BaseException,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    error_type = type(error).__name__
                    self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                    
                    logger.error(
                        f"LLM error ({error_type}): {error} (run_id={run_id})",
                        exc_info=True
                    )
                    
                    # Provide actionable guidance
                    if "rate" in str(error).lower():
                        logger.warning("Rate limit hit - consider implementing backoff")
                    elif "auth" in str(error).lower():
                        logger.error("Authentication failed - check API key")
                    elif "timeout" in str(error).lower():
                        logger.warning("Request timeout - consider increasing timeout or reducing prompt size")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:176
        """


class ChainManagerMixin:
    """Mixin for chain callbacks."""

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain completes execution successfully.

        This callback is invoked after a chain completes with output results.
        Use this to log chain results, track execution metrics, validate outputs,
        or trigger downstream processing. For nested chains, this is called for
        each chain in the hierarchy.

        Args:
            outputs: The output dictionary from the chain. Structure depends on
                the chain type:
                - Simple chains: {"output": result}
                - Retrieval chains: {"answer": text, "context": documents}
                - Sequential chains: Contains all output keys from final chain
                Keys and structure vary by chain implementation.
            run_id: The run ID. This is the unique identifier for the current
                chain execution.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                if this is a nested chain execution, or None for top-level chains.
                Use this to trace chain hierarchies.
            **kwargs: Additional keyword arguments. May include execution metadata
                or chain-specific context.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging
            import json

            logger = logging.getLogger(__name__)

            class ChainOutputHandler(BaseCallbackHandler):
                \"\"\"Handler that logs and validates chain outputs.\"\"\"
                
                def on_chain_end(
                    self,
                    outputs: dict[str, Any],
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    # Log output summary
                    output_keys = list(outputs.keys())
                    logger.info(
                        f"Chain completed with keys: {output_keys} (run_id={run_id})"
                    )
                    
                    # Log full output at debug level
                    logger.debug(f"Full output: {json.dumps(outputs, indent=2)}")
                    
                    # Validate expected structure
                    if 'answer' in outputs:
                        answer = outputs['answer']
                        if not answer or len(answer) < 10:
                            logger.warning(f"Suspiciously short answer: {answer}")
                    
                    # Track nesting
                    if parent_run_id:
                        logger.debug(f"Nested chain (parent={parent_run_id})")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:216
        """

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain execution fails with an error.

        This callback is invoked when a chain raises an exception during execution.
        Errors can originate from LLMs, tools, retrievers, or chain logic itself.
        Use this to log failures, implement error recovery, track error patterns,
        or send alerts for production issues.

        Args:
            error: The exception that occurred during chain execution. Common
                types include:
                - ValueError: Invalid inputs or chain configuration
                - KeyError: Missing required input or output keys
                - LLM errors: Rate limits, authentication failures
                - Tool errors: Tool execution failures
                - TimeoutError: Chain execution timeout
            run_id: The run ID. This is the unique identifier for the current
                chain execution.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                if this is a nested chain execution, or None for top-level chains.
                Use this to trace error propagation through chain hierarchies.
            **kwargs: Additional keyword arguments. May include:
                - inputs: The inputs that caused the error
                - Execution context and state at time of failure

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging

            logger = logging.getLogger(__name__)

            class ChainErrorHandler(BaseCallbackHandler):
                \"\"\"Handler that tracks and categorizes chain errors.\"\"\"
                
                def __init__(self):
                    self.error_log = []
                
                def on_chain_error(
                    self,
                    error: BaseException,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    error_info = {
                        'type': type(error).__name__,
                        'message': str(error),
                        'run_id': str(run_id),
                        'parent_run_id': str(parent_run_id) if parent_run_id else None
                    }
                    self.error_log.append(error_info)
                    
                    logger.error(
                        f"Chain failed ({error_info['type']}): {error_info['message']} "
                        f"(run_id={run_id})",
                        exc_info=True
                    )
                    
                    # Provide context-specific guidance
                    if isinstance(error, KeyError):
                        logger.error(f"Missing key: {error}. Check chain input/output configuration")
                    elif isinstance(error, ValueError):
                        logger.error("Invalid value. Verify input format and constraints")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:280
        """

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when agent decides to take an action.

        This callback is invoked when an agent decides on an action to execute,
        such as calling a tool or returning a final answer. This occurs in the
        agent decision loop after the LLM generates a plan but before tool
        execution. Use this to log agent reasoning, track tool usage patterns,
        or implement action validation.

        Args:
            action: The agent action object containing:
                - tool: The name of the tool to execute (str)
                - tool_input: The input to pass to the tool (str or dict)
                - log: The agent's reasoning log explaining the action (str)
            run_id: The run ID. This is the unique identifier for the current
                agent execution.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                that contains this agent, if any. Use this to trace agent
                execution within larger workflows.
            **kwargs: Additional keyword arguments. May include agent state or
                intermediate steps.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from langchain_core.agents import AgentAction
            from uuid import UUID
            from typing import Any
            import logging

            logger = logging.getLogger(__name__)

            class AgentActionHandler(BaseCallbackHandler):
                \"\"\"Handler that logs agent decision-making process.\"\"\"
                
                def __init__(self):
                    self.action_count = 0
                    self.tool_usage = {}
                
                def on_agent_action(
                    self,
                    action: AgentAction,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    self.action_count += 1
                    tool_name = action.tool
                    self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
                    
                    logger.info(
                        f"Agent action #{self.action_count}: {tool_name} "
                        f"(run_id={run_id})"
                    )
                    logger.debug(f"Tool input: {action.tool_input}")
                    logger.debug(f"Agent reasoning: {action.log}")
                    
                    # Warn on repeated tool usage (potential loop)
                    if self.tool_usage[tool_name] > 3:
                        logger.warning(
                            f"Tool '{tool_name}' used {self.tool_usage[tool_name]} times - "
                            "possible agent loop"
                        )
            ```

        Source: libs/core/langchain_core/callbacks/base.py:350
        """

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when agent completes execution with a final answer.

        This callback is invoked when an agent determines it has sufficient
        information to return a final answer, ending the agent decision loop.
        This marks the completion of agent reasoning. Use this to log final
        outputs, track agent performance metrics, or validate agent conclusions.

        Args:
            finish: The agent finish object containing:
                - return_values: Dict with the final answer and any additional
                  output values (typically {"output": final_answer})
                - log: The agent's final reasoning log (str)
            run_id: The run ID. This is the unique identifier for the current
                agent execution.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                that contains this agent, if any. Use this to trace agent
                execution within larger workflows.
            **kwargs: Additional keyword arguments. May include intermediate
                steps or full agent execution history.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from langchain_core.agents import AgentFinish
            from uuid import UUID
            from typing import Any
            import logging

            logger = logging.getLogger(__name__)

            class AgentFinishHandler(BaseCallbackHandler):
                \"\"\"Handler that logs agent completion and validates outputs.\"\"\"
                
                def on_agent_finish(
                    self,
                    finish: AgentFinish,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    logger.info(f"Agent completed (run_id={run_id})")
                    
                    # Extract and log final answer
                    return_values = finish.return_values
                    final_output = return_values.get('output', 'No output')
                    logger.info(f"Final answer: {final_output[:200]}...")
                    
                    # Log agent reasoning
                    logger.debug(f"Final reasoning: {finish.log}")
                    
                    # Validate output quality
                    if len(final_output) < 20:
                        logger.warning("Agent output is very short - may be incomplete")
                    
                    # Log intermediate steps if available
                    if 'intermediate_steps' in kwargs:
                        steps = kwargs['intermediate_steps']
                        logger.info(f"Agent took {len(steps)} intermediate steps")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:419
        """


class ToolManagerMixin:
    """Mixin for tool callbacks."""

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool execution completes successfully.

        This callback is invoked after a tool finishes execution with a result.
        Tools are typically called by agents to perform specific actions like
        searching, calculations, or API calls. Use this to log tool results,
        track tool performance, or validate tool outputs.

        Args:
            output: The output returned by the tool. Type and structure depend
                on the specific tool implementation. Common types include:
                - str: Text response from search or API tools
                - dict: Structured data from database or API tools
                - list: Multiple results from batch operations
                May be None for tools with side effects only.
            run_id: The run ID. This is the unique identifier for the current
                tool execution.
            parent_run_id: The parent run ID. This is typically the ID of the
                agent or chain that invoked this tool. Use this to trace tool
                usage within agent workflows.
            **kwargs: Additional keyword arguments. May include tool name,
                input parameters, or execution metadata.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging

            logger = logging.getLogger(__name__)

            class ToolMetricsHandler(BaseCallbackHandler):
                \"\"\"Handler that tracks tool execution metrics.\"\"\"
                
                def __init__(self):
                    self.tool_calls = {}
                
                def on_tool_end(
                    self,
                    output: Any,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    # Extract tool name if available
                    tool_name = kwargs.get('name', 'unknown')
                    self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
                    
                    logger.info(
                        f"Tool '{tool_name}' completed (run_id={run_id}, "
                        f"agent={parent_run_id})"
                    )
                    
                    # Log output summary
                    output_str = str(output)
                    logger.debug(f"Tool output: {output_str[:200]}...")
                    
                    # Validate output
                    if output is None or output == "":
                        logger.warning(f"Tool '{tool_name}' returned empty output")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:493
        """

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool execution fails with an error.

        This callback is invoked when a tool raises an exception during execution.
        Common causes include invalid inputs, API failures, permission errors, or
        tool-specific failures. Use this to log tool failures, implement fallback
        strategies, track error patterns, or help agents recover gracefully.

        Args:
            error: The exception that occurred during tool execution. Common
                types include:
                - ValueError: Invalid tool input parameters
                - ConnectionError: API or network failures
                - TimeoutError: Tool execution timeout
                - PermissionError: Insufficient permissions
                - Tool-specific exceptions from tool implementation
            run_id: The run ID. This is the unique identifier for the current
                tool execution.
            parent_run_id: The parent run ID. This is typically the ID of the
                agent or chain that invoked this tool. Use this to trace tool
                failures within agent workflows.
            **kwargs: Additional keyword arguments. May include:
                - name: Tool name that failed
                - input_str: Input that caused the error
                - Tool-specific error context

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging

            logger = logging.getLogger(__name__)

            class ToolErrorHandler(BaseCallbackHandler):
                \"\"\"Handler that tracks and categorizes tool failures.\"\"\"
                
                def __init__(self):
                    self.tool_errors = {}
                
                def on_tool_error(
                    self,
                    error: BaseException,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    tool_name = kwargs.get('name', 'unknown')
                    error_type = type(error).__name__
                    
                    # Track error stats
                    key = f"{tool_name}:{error_type}"
                    self.tool_errors[key] = self.tool_errors.get(key, 0) + 1
                    
                    logger.error(
                        f"Tool '{tool_name}' failed ({error_type}): {error} "
                        f"(run_id={run_id})",
                        exc_info=True
                    )
                    
                    # Provide actionable guidance
                    if isinstance(error, ValueError):
                        logger.error(
                            f"Invalid input to '{tool_name}'. "
                            f"Input was: {kwargs.get('input_str', 'N/A')}"
                        )
                    elif isinstance(error, ConnectionError):
                        logger.warning(f"Network issue calling '{tool_name}' - consider retry")
                    elif isinstance(error, TimeoutError):
                        logger.warning(f"'{tool_name}' timeout - consider increasing limit")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:568
        """


class CallbackManagerMixin:
    """Mixin for callback manager."""

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
        """Run when LLM starts generating (non-chat models only).

        This callback is invoked when a legacy LLM (non-chat model) begins
        generation. For chat models, use on_chat_model_start instead. This fires
        before the model receives prompts. Use this to log prompts, track LLM
        invocations, start timing measurements, or implement pre-processing.

        !!! warning
            This method is called for non-chat models (regular LLMs). If you're
            implementing a handler for a chat model, you should use
            `on_chat_model_start` instead.

        Args:
            serialized: The serialized LLM representation containing:
                - name: Model class name (e.g., "OpenAI", "Anthropic")
                - id: List with model identifier path
                - Model configuration parameters
                Used for logging and model identification.
            prompts: The list of prompt strings being sent to the LLM. Each
                string is a complete prompt. Multiple prompts indicate batch
                generation.
            run_id: The run ID. This is the unique identifier for the current
                LLM generation operation.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                or component that initiated this LLM call, if any. Use this to
                trace nested executions.
            tags: Optional list of tags associated with this LLM call. Tags can
                be used for filtering, grouping, or categorizing executions.
            metadata: Optional metadata dict associated with this LLM call.
                Metadata can include custom context, user IDs, or execution
                parameters.
            **kwargs: Additional keyword arguments. May include:
                - invocation_params: Model parameters (temperature, max_tokens, etc.)
                - options: Provider-specific options

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code and prevent LLM execution if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging
            import time

            logger = logging.getLogger(__name__)

            class LLMTimingHandler(BaseCallbackHandler):
                \"\"\"Handler that times LLM calls and logs prompts.\"\"\"
                
                def __init__(self):
                    self.start_times = {}
                
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
                    # Start timing
                    self.start_times[run_id] = time.time()
                    
                    # Log model and prompt info
                    model_name = serialized.get('name', 'Unknown')
                    logger.info(
                        f"LLM start: {model_name} with {len(prompts)} prompt(s) "
                        f"(run_id={run_id}, tags={tags})"
                    )
                    
                    # Log first prompt (preview)
                    if prompts:
                        logger.debug(f"First prompt: {prompts[0][:200]}...")
                    
                    # Log invocation parameters
                    if 'invocation_params' in kwargs:
                        params = kwargs['invocation_params']
                        logger.debug(f"Parameters: {params}")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:658
        """

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
        """Run when a chat model starts generating (chat models only).

        This callback is invoked when a chat model begins generation. For legacy
        LLMs (non-chat models), on_llm_start is called instead. This fires before
        the model receives messages. Use this to log chat inputs, track chat model
        invocations, start timing measurements, or implement pre-processing.

        By default, this raises NotImplementedError to allow fallback to
        on_llm_start for handlers that don't distinguish between chat and
        non-chat models. Override this method to handle chat-specific logic.

        !!! warning
            This method is called for chat models. If you're implementing a handler for
            a non-chat model, you should use `on_llm_start` instead.

        Args:
            serialized: The serialized chat model representation containing:
                - name: Model class name (e.g., "ChatOpenAI", "ChatAnthropic")
                - id: List with model identifier path
                - Model configuration parameters
                Used for logging and model identification.
            messages: List of message lists being sent to the chat model. Each
                inner list represents a conversation with BaseMessage objects
                (SystemMessage, HumanMessage, AIMessage). Multiple lists indicate
                batch generation.
            run_id: The run ID. This is the unique identifier for the current
                chat model generation operation.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                or component that initiated this chat model call, if any. Use
                this to trace nested executions.
            tags: Optional list of tags associated with this chat model call.
                Tags can be used for filtering, grouping, or categorizing
                executions.
            metadata: Optional metadata dict associated with this chat model call.
                Metadata can include custom context, user IDs, or execution
                parameters.
            **kwargs: Additional keyword arguments. May include:
                - invocation_params: Model parameters (temperature, max_tokens, etc.)
                - options: Provider-specific options

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            NotImplementedError: Raised by default to trigger fallback to
                on_llm_start. Override this method to implement chat-specific
                handling without raising this exception.
            
            If raise_error attribute is True and another exception is raised
            within your implementation, it will propagate to the calling code
            and prevent chat model execution.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from langchain_core.messages import BaseMessage
            from uuid import UUID
            from typing import Any
            import logging

            logger = logging.getLogger(__name__)

            class ChatModelHandler(BaseCallbackHandler):
                \"\"\"Handler that logs chat model interactions.\"\"\"
                
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
                    # Don't raise NotImplementedError - implement chat handling
                    model_name = serialized.get('name', 'Unknown')
                    logger.info(
                        f"Chat model start: {model_name} with {len(messages)} "
                        f"conversation(s) (run_id={run_id})"
                    )
                    
                    # Log first conversation
                    if messages and messages[0]:
                        first_conv = messages[0]
                        logger.debug(f"First conversation has {len(first_conv)} messages")
                        for msg in first_conv:
                            role = msg.__class__.__name__
                            content = msg.content[:100] if hasattr(msg, 'content') else 'N/A'
                            logger.debug(f"  {role}: {content}...")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:758
        """
        # NotImplementedError is thrown intentionally
        # Callback handler will fall back to on_llm_start if this is exception is thrown
        msg = f"{self.__class__.__name__} does not implement `on_chat_model_start`"
        raise NotImplementedError(msg)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever starts document retrieval.

        This callback is invoked when a retriever begins searching for relevant
        documents based on a query. This fires before retrieval execution. Use
        this to log queries, track retrieval patterns, start timing measurements,
        or implement query preprocessing.

        Args:
            serialized: The serialized Retriever representation containing:
                - name: Retriever class name (e.g., "VectorStoreRetriever")
                - id: List with retriever identifier path
                - Retriever configuration parameters
                Used for logging and retriever identification.
            query: The search query string used to retrieve relevant documents.
                This is the text input that will be used for similarity search
                or other retrieval methods.
            run_id: The run ID. This is the unique identifier for the current
                retrieval operation.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                or component that initiated this retrieval, if any. Use this to
                trace nested executions.
            tags: Optional list of tags associated with this retrieval. Tags can
                be used for filtering, grouping, or categorizing executions.
            metadata: Optional metadata dict associated with this retrieval.
                Metadata can include custom context, user IDs, or execution
                parameters.
            **kwargs: Additional keyword arguments. May include retriever-specific
                parameters or configuration.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code and prevent retrieval execution if raise_error
            is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging
            import time

            logger = logging.getLogger(__name__)

            class RetrieverTrackingHandler(BaseCallbackHandler):
                \"\"\"Handler that tracks retrieval queries and timing.\"\"\"
                
                def __init__(self):
                    self.retrieval_times = {}
                    self.queries = []
                
                def on_retriever_start(
                    self,
                    serialized: dict[str, Any],
                    query: str,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> Any:
                    # Start timing
                    self.retrieval_times[run_id] = time.time()
                    self.queries.append(query)
                    
                    # Log retrieval start
                    retriever_name = serialized.get('name', 'Unknown')
                    logger.info(
                        f"Retrieval start: {retriever_name} (run_id={run_id})"
                    )
                    logger.debug(f"Query: {query}")
                    
                    # Track query patterns
                    if len(query) < 10:
                        logger.warning("Very short query - results may be broad")
                    elif len(query) > 500:
                        logger.warning("Very long query - consider summarizing")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:875
        """

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
        """Run when chain starts executing.

        This callback is invoked when a chain begins execution with inputs. This
        is the entry point for chain execution tracking. Use this to log inputs,
        track chain invocations, start timing measurements, or implement input
        validation. For nested chains, this is called for each chain in the
        hierarchy.

        Args:
            serialized: The serialized chain representation containing:
                - name: Chain class name (e.g., "LLMChain", "SequentialChain")
                - id: List with chain identifier path
                - Chain configuration parameters
                Used for logging and chain identification.
            inputs: The input dictionary passed to the chain. Structure depends
                on the chain type:
                - Simple chains: {"input": user_input}
                - Retrieval chains: {"input": query}
                - Sequential chains: Contains all required input keys
                Keys and structure vary by chain implementation.
            run_id: The run ID. This is the unique identifier for the current
                chain execution.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                if this is a nested chain execution, or None for top-level chains.
                Use this to trace chain hierarchies.
            tags: Optional list of tags associated with this chain execution.
                Tags can be used for filtering, grouping, or categorizing
                executions.
            metadata: Optional metadata dict associated with this chain execution.
                Metadata can include custom context, user IDs, or execution
                parameters.
            **kwargs: Additional keyword arguments. May include chain-specific
                configuration or execution context.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code and prevent chain execution if raise_error is
            True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging
            import time
            import json

            logger = logging.getLogger(__name__)

            class ChainExecutionHandler(BaseCallbackHandler):
                \"\"\"Handler that tracks chain execution flow.\"\"\"
                
                def __init__(self):
                    self.execution_times = {}
                    self.chain_depth = {}
                
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
                    # Start timing
                    self.execution_times[run_id] = time.time()
                    
                    # Track nesting depth
                    depth = 0
                    if parent_run_id and parent_run_id in self.chain_depth:
                        depth = self.chain_depth[parent_run_id] + 1
                    self.chain_depth[run_id] = depth
                    
                    # Log chain start with indentation for nested chains
                    indent = "  " * depth
                    chain_name = serialized.get('name', 'Unknown')
                    logger.info(
                        f"{indent}Chain start: {chain_name} (run_id={run_id}, "
                        f"parent={parent_run_id}, depth={depth})"
                    )
                    logger.debug(f"{indent}Inputs: {json.dumps(inputs, indent=2)}")
                    
                    if tags:
                        logger.debug(f"{indent}Tags: {tags}")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:975
        """

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool starts executing.

        This callback is invoked when a tool begins execution, typically called
        by an agent. This fires before tool execution. Use this to log tool
        invocations, track tool usage patterns, start timing measurements, or
        implement input validation.

        Args:
            serialized: The serialized tool representation containing:
                - name: Tool name (e.g., "Calculator", "Search", "WikipediaAPI")
                - id: List with tool identifier path
                - Tool description and configuration
                Used for logging and tool identification.
            input_str: The input string passed to the tool. This is the primary
                input parameter as a string representation. For structured tools,
                this may be a JSON-encoded representation of the input dict.
            run_id: The run ID. This is the unique identifier for the current
                tool execution.
            parent_run_id: The parent run ID. This is typically the ID of the
                agent or chain that invoked this tool. Use this to trace tool
                usage within agent workflows.
            tags: Optional list of tags associated with this tool execution.
                Tags can be used for filtering, grouping, or categorizing
                executions.
            metadata: Optional metadata dict associated with this tool execution.
                Metadata can include custom context, user IDs, or execution
                parameters.
            inputs: Optional structured input dict for tools that accept
                structured inputs (StructuredTool). Contains parameter names as
                keys and values as passed by the agent.
            **kwargs: Additional keyword arguments. May include tool-specific
                configuration or execution context.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code and prevent tool execution if raise_error is
            True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging
            import time

            logger = logging.getLogger(__name__)

            class ToolExecutionHandler(BaseCallbackHandler):
                \"\"\"Handler that tracks tool execution patterns.\"\"\"
                
                def __init__(self):
                    self.tool_times = {}
                    self.tool_call_count = {}
                
                def on_tool_start(
                    self,
                    serialized: dict[str, Any],
                    input_str: str,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    inputs: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> Any:
                    # Start timing
                    self.tool_times[run_id] = time.time()
                    
                    # Track tool usage
                    tool_name = serialized.get('name', 'Unknown')
                    self.tool_call_count[tool_name] = self.tool_call_count.get(tool_name, 0) + 1
                    
                    logger.info(
                        f"Tool start: {tool_name} (run_id={run_id}, "
                        f"agent={parent_run_id})"
                    )
                    logger.debug(f"Input string: {input_str[:200]}...")
                    
                    # Log structured inputs if available
                    if inputs:
                        logger.debug(f"Structured inputs: {inputs}")
                    
                    # Warn on repeated tool calls (potential loop)
                    if self.tool_call_count[tool_name] > 5:
                        logger.warning(
                            f"Tool '{tool_name}' called {self.tool_call_count[tool_name]} times - "
                            "possible agent loop"
                        )
            ```

        Source: libs/core/langchain_core/callbacks/base.py:1085
        """


class RunManagerMixin:
    """Mixin for run manager."""

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on arbitrary intermediate text output.

        This callback is invoked when a component produces intermediate text output
        that doesn't fit into other specific callback types. This is a catch-all
        for text logging during execution, such as agent thoughts, intermediate
        chain results, or debug output. Use this to capture and log miscellaneous
        text during execution.

        Args:
            text: The arbitrary text produced during execution. This could be agent
                reasoning steps, intermediate chain outputs, debug messages, or any
                other text output that components emit during execution. The format
                and content depend on the component producing the text.
            run_id: The run ID. This is the unique identifier for the current
                execution that produced this text.
            parent_run_id: The parent run ID. This is the ID of the parent chain,
                agent, or tool execution, if any. Use this to trace text output
                within execution hierarchies.
            **kwargs: Additional keyword arguments. May include component-specific
                context or metadata about the text output.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging

            logger = logging.getLogger(__name__)

            class VerboseOutputHandler(BaseCallbackHandler):
                \"\"\"Handler that captures all intermediate text output.\"\"\"
                
                def __init__(self):
                    self.text_outputs = []
                
                def on_text(
                    self,
                    text: str,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    # Store all text outputs
                    self.text_outputs.append({
                        'run_id': str(run_id),
                        'parent_run_id': str(parent_run_id) if parent_run_id else None,
                        'text': text,
                        'kwargs': kwargs
                    })
                    
                    # Log intermediate output
                    logger.info(f"Intermediate text (run_id={run_id}): {text[:100]}...")
                    
                    # Track verbose output patterns
                    if kwargs.get('verbose'):
                        logger.debug(f"Verbose mode output: {text}")
                
                def get_execution_transcript(self) -> str:
                    \"\"\"Get full transcript of all text outputs.\"\"\"
                    return "\\n".join(item['text'] for item in self.text_outputs)
            ```

        Source: libs/core/langchain_core/callbacks/base.py:1498
        """

    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a retry event occurs.

        This callback is invoked when a component operation is retried due to a
        transient failure (e.g., rate limits, temporary API errors, network issues).
        This is triggered by retry decorators (using tenacity library) on LLM calls
        or other retryable operations. Use this to track retry patterns, implement
        exponential backoff logging, or alert on excessive retries.

        Args:
            retry_state: The retry state object from tenacity containing:
                - attempt_number: Current retry attempt number (1-indexed)
                - outcome: The result of the previous attempt (exception or value)
                - seconds_since_start: Total elapsed time since first attempt
                - next_action: Planned next action (sleep time before retry)
                - args/kwargs: Arguments passed to the retried function
                Access via retry_state.attempt_number, retry_state.outcome, etc.
            run_id: The run ID. This is the unique identifier for the execution
                being retried.
            parent_run_id: The parent run ID. This is the ID of the parent chain,
                LLM call, or component that triggered the retryable operation.
            **kwargs: Additional keyword arguments. May include retry configuration
                or component-specific context.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from tenacity import RetryCallState
            from uuid import UUID
            from typing import Any
            import logging

            logger = logging.getLogger(__name__)

            class RetryMonitorHandler(BaseCallbackHandler):
                \"\"\"Handler that monitors retry patterns and alerts on issues.\"\"\"
                
                def __init__(self, max_retries_alert: int = 3):
                    self.retry_counts = {}
                    self.max_retries_alert = max_retries_alert
                
                def on_retry(
                    self,
                    retry_state: RetryCallState,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    # Track retry attempts
                    if run_id not in self.retry_counts:
                        self.retry_counts[run_id] = 0
                    self.retry_counts[run_id] += 1
                    
                    # Log retry event
                    attempt = retry_state.attempt_number
                    elapsed = retry_state.seconds_since_start
                    logger.warning(
                        f"Retry event: attempt {attempt} after {elapsed:.2f}s "
                        f"(run_id={run_id})"
                    )
                    
                    # Log the exception that triggered retry
                    if retry_state.outcome and retry_state.outcome.failed:
                        exception = retry_state.outcome.exception()
                        logger.error(f"Retry triggered by: {type(exception).__name__}: {exception}")
                    
                    # Alert on excessive retries
                    if attempt >= self.max_retries_alert:
                        logger.critical(
                            f"ALERT: {attempt} retry attempts detected - "
                            f"possible persistent failure (run_id={run_id})"
                        )
                    
                    # Log next retry wait time
                    if retry_state.next_action:
                        wait_time = getattr(retry_state.next_action, 'sleep', None)
                        if wait_time:
                            logger.info(f"Waiting {wait_time:.2f}s before next retry")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:1565
        """

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Override to define a handler for custom user-defined events.

        This callback enables users to emit and handle custom application-specific
        events during chain execution that don't fit into standard callback types.
        Custom events can be dispatched from within chains, tools, or other
        components using the callback manager's dispatch_custom_event method. Use
        this to implement domain-specific logging, metrics, or workflow triggers.

        Args:
            name: The name of the custom event. This is a user-defined string
                identifying the event type (e.g., "user_feedback", "cache_hit",
                "validation_failed", "metric_update"). Use consistent naming
                conventions for event types in your application.
            data: The data payload for the custom event. Format and structure are
                completely user-defined and should match the event type. Can be
                any JSON-serializable data: dict, list, string, number, etc.
                Examples:
                - {"score": 0.95, "metric": "relevance"} for metrics
                - {"error": "validation failed", "field": "email"} for validation
                - "cache_key_xyz" for cache events
            run_id: The run ID. This is the unique identifier for the execution
                context that emitted this custom event. Links the event to the
                specific chain, agent, or tool execution.
            tags: Optional list of tags associated with the custom event. Includes
                tags inherited from the parent execution context plus any tags
                specified when dispatching the event. Use tags for filtering and
                categorizing custom events.
            metadata: Optional metadata dict associated with the custom event.
                Includes metadata inherited from the parent execution context plus
                any metadata specified when dispatching the event. Use metadata
                for additional context like user IDs, session IDs, or application
                state.
            **kwargs: Additional keyword arguments. May include event-specific
                parameters or application context.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import BaseCallbackHandler
            from uuid import UUID
            from typing import Any
            import logging
            import json

            logger = logging.getLogger(__name__)

            class CustomEventHandler(BaseCallbackHandler):
                \"\"\"Handler for application-specific custom events.\"\"\"
                
                def __init__(self):
                    self.event_log = []
                    self.metrics = {}
                
                def on_custom_event(
                    self,
                    name: str,
                    data: Any,
                    *,
                    run_id: UUID,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> Any:
                    # Log all custom events
                    event_record = {
                        'name': name,
                        'data': data,
                        'run_id': str(run_id),
                        'tags': tags,
                        'metadata': metadata
                    }
                    self.event_log.append(event_record)
                    
                    logger.info(
                        f"Custom event '{name}' (run_id={run_id}): "
                        f"{json.dumps(data, default=str)[:200]}"
                    )
                    
                    # Handle specific event types
                    if name == "metric_update":
                        # Track metrics
                        if isinstance(data, dict) and 'metric' in data and 'value' in data:
                            metric_name = data['metric']
                            metric_value = data['value']
                            self.metrics[metric_name] = metric_value
                            logger.info(f"Metric updated: {metric_name}={metric_value}")
                    
                    elif name == "user_feedback":
                        # Log user feedback events
                        if isinstance(data, dict):
                            score = data.get('score', 'N/A')
                            comment = data.get('comment', '')
                            logger.info(
                                f"User feedback received: score={score}, "
                                f"comment={comment[:100]}"
                            )
                    
                    elif name == "cache_hit":
                        # Track cache performance
                        logger.debug(f"Cache hit: {data}")
                    
                    elif name == "validation_error":
                        # Alert on validation errors
                        logger.error(f"Validation error: {data}")
                    
                    # Filter by tags
                    if tags and "production" in tags:
                        logger.info(f"Production event: {name}")
            
            # Usage in a custom chain:
            # await callback_manager.dispatch_custom_event(
            #     "metric_update",
            #     {"metric": "relevance_score", "value": 0.95},
            #     tags=["production"],
            #     metadata={"user_id": "user123"}
            # )
            ```

        Source: libs/core/langchain_core/callbacks/base.py:1632
        """


class BaseCallbackHandler(
    LLMManagerMixin,
    ChainManagerMixin,
    ToolManagerMixin,
    RetrieverManagerMixin,
    CallbackManagerMixin,
    RunManagerMixin,
):
    """Base callback handler for LangChain.

This module provides the core callback system architecture for observability,
logging, and tracing across LangChain components. The callback system enables
monitoring of chain execution, LLM calls, agent actions, tool usage, and
retriever operations through a comprehensive event-driven interface.

Architecture Overview:
    The callback system uses a mixin pattern to organize callback methods by
    component type:
    
    - RetrieverManagerMixin: Callbacks for document retrieval operations
    - LLMManagerMixin: Callbacks for language model invocations
    - ChainManagerMixin: Callbacks for chain and agent execution
    - ToolManagerMixin: Callbacks for tool usage
    - CallbackManagerMixin: Callbacks for component lifecycle events
    - RunManagerMixin: Callbacks for general text and retry events

Classes:
    BaseCallbackHandler: Synchronous base class for implementing custom
        callback handlers. Subclass this to create handlers for logging,
        metrics collection, persistence, or other observability needs.
    
    AsyncCallbackHandler: Async variant of BaseCallbackHandler for use with
        async chain execution. Use this when working with async/await patterns.
    
    BaseCallbackManager: Manager class for coordinating multiple callback
        handlers and managing callback lifecycle.

Usage Patterns:
    Create a custom handler by subclassing BaseCallbackHandler or
    AsyncCallbackHandler and overriding the specific callback methods you need:
    
    ```python
    from langchain_core.callbacks.base import BaseCallbackHandler
    from uuid import UUID
    from typing import Any
    
    class MyCustomHandler(BaseCallbackHandler):
        def on_chain_start(
            self,
            serialized: dict[str, Any],
            inputs: dict[str, Any],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> Any:
            print(f"Chain started: {serialized.get('name', 'unknown')}")
        
        def on_chain_end(
            self,
            outputs: dict[str, Any],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> Any:
            print(f"Chain completed with outputs: {outputs}")
    ```

Event Sequence:
    Callbacks are invoked in a predictable sequence during chain execution.
    For a typical chain calling an LLM:
    
    1. on_chain_start - Chain execution begins
    2. on_llm_start or on_chat_model_start - LLM call begins
    3. on_llm_new_token (if streaming) - Token received
    4. on_llm_end - LLM call completes
    5. on_chain_end - Chain execution completes
    
    Error path:
    1. on_chain_start
    2. on_llm_start
    3. on_llm_error - LLM call fails
    4. on_chain_error - Chain execution fails

Nested Execution:
    For nested chains or tool calls, the parent_run_id parameter tracks
    relationships between callback events, enabling hierarchical tracing.

Source: libs/core/langchain_core/callbacks/base.py
"""

    raise_error: bool = False
    """Whether to raise an error if an exception occurs."""

    run_inline: bool = False
    """Whether to run the callback inline."""

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False

    @property
    def ignore_retry(self) -> bool:
        """Whether to ignore retry callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return False

    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return False

    @property
    def ignore_custom_event(self) -> bool:
        """Ignore custom event."""
        return False


class AsyncCallbackHandler(BaseCallbackHandler):
    """Async callback handler for LangChain async chain execution.

    This class extends BaseCallbackHandler to provide async/await support for
    callback methods, enabling non-blocking callback execution in async chain
    workflows. Use AsyncCallbackHandler when working with async chains, async
    LLM calls, or when callback operations involve async I/O (e.g., async
    database writes, async API calls for logging/metrics).

    Key Differences from BaseCallbackHandler:
        - All callback methods are async (async def) instead of synchronous
        - Callback methods can use await for async operations
        - Requires async runtime environment (asyncio event loop)
        - Callback execution doesn't block the async chain execution flow
        - Ideal for async database writes, async metrics APIs, async logging

    When to Use AsyncCallbackHandler:
        - When working with async chains (using ainvoke, astream, etc.)
        - When callback logic involves async I/O operations
        - When you need non-blocking callback execution
        - When integrating with async observability platforms

    When to Use BaseCallbackHandler:
        - When working with synchronous chains (using invoke, stream, etc.)
        - When callback logic is purely synchronous
        - When simpler synchronous patterns are sufficient

    Event Loop Considerations:
        - AsyncCallbackHandler methods run in the same event loop as the chain
        - Avoid blocking operations in async callbacks (use await for I/O)
        - Long-running sync operations should be wrapped with
          asyncio.to_thread() or run_in_executor()
        - Exceptions in async callbacks propagate according to raise_error flag

    Usage Pattern:
        ```python
        from langchain_core.callbacks.base import AsyncCallbackHandler
        from uuid import UUID
        from typing import Any
        import aiohttp
        import logging

        logger = logging.getLogger(__name__)

        class AsyncMetricsHandler(AsyncCallbackHandler):
            \"\"\"Async handler that sends metrics to an API.\"\"\"
            
            def __init__(self, metrics_url: str):
                self.metrics_url = metrics_url
                self.session = None
            
            async def on_chain_start(
                self,
                serialized: dict[str, Any],
                inputs: dict[str, Any],
                *,
                run_id: UUID,
                **kwargs: Any,
            ) -> Any:
                # Async HTTP request doesn't block chain execution
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                try:
                    async with self.session.post(
                        f"{self.metrics_url}/chain_start",
                        json={
                            'chain_name': serialized.get('name'),
                            'run_id': str(run_id),
                            'inputs': inputs
                        }
                    ) as resp:
                        await resp.text()
                        logger.info(f"Metrics sent for run_id={run_id}")
                except Exception as e:
                    logger.error(f"Failed to send metrics: {e}")
            
            async def on_chain_end(
                self,
                outputs: dict[str, Any],
                *,
                run_id: UUID,
                **kwargs: Any,
            ) -> Any:
                # Async database write
                if self.session:
                    try:
                        async with self.session.post(
                            f"{self.metrics_url}/chain_end",
                            json={'run_id': str(run_id), 'outputs': outputs}
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to send completion metrics: {e}")
            
            async def __aenter__(self):
                self.session = aiohttp.ClientSession()
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.session:
                    await self.session.close()
        
        # Usage with async chain:
        # async with AsyncMetricsHandler("https://metrics.example.com") as handler:
        #     result = await chain.ainvoke(
        #         {"input": "query"},
        #         config={"callbacks": [handler]}
        #     )
        ```

    Inheritance:
        AsyncCallbackHandler inherits from BaseCallbackHandler, so all
        properties (raise_error, run_inline) and ignore flags (ignore_llm,
        ignore_chain, etc.) are available. Override async versions of callback
        methods to implement custom async behavior.

    Source: libs/core/langchain_core/callbacks/base.py:1954
    """

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts generating (async version).

        This is the async version of the synchronous on_llm_start callback. It is
        invoked when a legacy LLM (non-chat model) begins generation. This fires
        before the LLM API call is made. Use this for async logging, async metrics
        collection, or async database writes without blocking chain execution.

        !!! warning
            This method is called for non-chat models (regular LLMs). If you're
            implementing a handler for a chat model, you should use
            `on_chat_model_start` instead.

        Args:
            serialized: The serialized LLM representation containing:
                - name: LLM class name (e.g., "OpenAI", "HuggingFaceHub")
                - id: List with LLM identifier path
                - LLM configuration (temperature, max_tokens, etc.)
                Used for logging and LLM identification.
            prompts: The list of prompt strings that will be sent to the LLM. Each
                prompt is a complete text input. For batch requests, this list
                contains multiple prompts.
            run_id: The run ID. This is the unique identifier for the current LLM
                call.
            parent_run_id: The parent run ID. This is the ID of the parent chain or
                component that triggered this LLM call, if any. Use this to trace
                nested executions.
            tags: Optional list of tags associated with this LLM call. Tags can be
                used for filtering, grouping, or categorizing executions.
            metadata: Optional metadata dict associated with this LLM call. Metadata
                can include custom context, user IDs, or execution parameters.
            **kwargs: Additional keyword arguments. May include invocation_params
                dict with model parameters (temperature, max_tokens, etc.).

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code and prevent LLM execution if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from uuid import UUID
            from typing import Any
            import aiohttp
            import logging

            logger = logging.getLogger(__name__)

            class AsyncLLMMetricsHandler(AsyncCallbackHandler):
                \"\"\"Async handler that logs LLM calls to an API.\"\"\"
                
                def __init__(self, api_url: str):
                    self.api_url = api_url
                    self.session = None
                
                async def on_llm_start(
                    self,
                    serialized: dict[str, Any],
                    prompts: list[str],
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Async HTTP request doesn't block
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    llm_name = serialized.get('name', 'Unknown')
                    invocation_params = kwargs.get('invocation_params', {})
                    
                    try:
                        async with self.session.post(
                            f"{self.api_url}/llm_start",
                            json={
                                'llm_name': llm_name,
                                'run_id': str(run_id),
                                'num_prompts': len(prompts),
                                'params': invocation_params,
                                'tags': tags
                            }
                        ) as resp:
                            await resp.text()
                            logger.info(f"LLM start logged: {llm_name} (run_id={run_id})")
                    except Exception as e:
                        logger.error(f"Failed to log LLM start: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:2074
        """

    async def on_chat_model_start(
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
        """Run when chat model starts generating (async version).

        This is the async version of the synchronous on_chat_model_start callback.
        It is invoked when a chat model (e.g., ChatOpenAI, ChatAnthropic) begins
        generation. This fires before the chat model API call is made. Use this for
        async logging of chat conversations, async metrics collection, or async
        database writes without blocking chain execution.

        !!! warning
            This method is called for chat models. If you're implementing a handler for
            a non-chat model, you should use `on_llm_start` instead.

        Args:
            serialized: The serialized chat model representation containing:
                - name: Chat model class name (e.g., "ChatOpenAI", "ChatAnthropic")
                - id: List with chat model identifier path
                - Model configuration (temperature, max_tokens, etc.)
                Used for logging and model identification.
            messages: List of conversations, where each conversation is a list of
                BaseMessage objects (HumanMessage, AIMessage, SystemMessage, etc.).
                For single conversations, this is a list with one element. For batch
                requests, contains multiple conversation lists. Each message has
                content and role information.
            run_id: The run ID. This is the unique identifier for the current chat
                model call.
            parent_run_id: The parent run ID. This is the ID of the parent chain or
                component that triggered this chat model call, if any. Use this to
                trace nested executions.
            tags: Optional list of tags associated with this chat model call. Tags
                can be used for filtering, grouping, or categorizing executions.
            metadata: Optional metadata dict associated with this chat model call.
                Metadata can include custom context, user IDs, or execution
                parameters.
            **kwargs: Additional keyword arguments. May include invocation_params
                dict with model parameters (temperature, max_tokens, etc.).

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            NotImplementedError: Raised by default to trigger fallback to
                on_llm_start. Override this method to implement chat-specific
                handling without raising this exception.
            
            If raise_error attribute is True and another exception is raised
            within your implementation, it will propagate to the calling code
            and prevent chat model execution.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from langchain_core.messages import BaseMessage
            from uuid import UUID
            from typing import Any
            import aiohttp
            import logging

            logger = logging.getLogger(__name__)

            class AsyncChatModelHandler(AsyncCallbackHandler):
                \"\"\"Async handler that logs chat model interactions.\"\"\"
                
                def __init__(self, api_url: str):
                    self.api_url = api_url
                    self.session = None
                
                async def on_chat_model_start(
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
                    # Don't raise NotImplementedError - implement chat handling
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    model_name = serialized.get('name', 'Unknown')
                    
                    # Log conversation details asynchronously
                    conversation_data = []
                    for conversation in messages:
                        conv_messages = []
                        for msg in conversation:
                            conv_messages.append({
                                'role': msg.__class__.__name__,
                                'content': msg.content[:200] if hasattr(msg, 'content') else 'N/A'
                            })
                        conversation_data.append(conv_messages)
                    
                    try:
                        async with self.session.post(
                            f"{self.api_url}/chat_start",
                            json={
                                'model': model_name,
                                'run_id': str(run_id),
                                'num_conversations': len(messages),
                                'conversations': conversation_data,
                                'tags': tags
                            }
                        ) as resp:
                            await resp.text()
                            logger.info(
                                f"Chat model start logged: {model_name} "
                                f"with {len(messages)} conversation(s)"
                            )
                    except Exception as e:
                        logger.error(f"Failed to log chat model start: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:2190
        """
        # NotImplementedError is thrown intentionally
        # Callback handler will fall back to on_llm_start if this is exception is thrown
        msg = f"{self.__class__.__name__} does not implement `on_chat_model_start`"
        raise NotImplementedError(msg)

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run on new LLM output token during streaming (async version).

        This is the async version of the synchronous on_llm_new_token callback. It
        is invoked for each token generated during streaming LLM/chat model
        responses. Only available when streaming is enabled on the model. Use this
        for async real-time processing of tokens, async token buffering, or async
        display updates.

        For both chat models and non-chat models (legacy LLMs).

        Args:
            token: The new token string generated by the model. This is the text
                content of the token, which can be a word fragment, punctuation, or
                whitespace. Tokens are emitted sequentially as the model generates
                them.
            chunk: Optional generation chunk object containing the token plus
                additional metadata:
                - text: The token text (same as token parameter)
                - generation_info: Model-specific generation metadata
                - message: For chat models, the partial message being constructed
                For chat models, this is ChatGenerationChunk; for LLMs, this is
                GenerationChunk. May be None for some streaming implementations.
            run_id: The run ID. This is the unique identifier for the current
                streaming LLM call.
            parent_run_id: The parent run ID. This is the ID of the parent chain or
                component, if any. Use this to trace nested streaming executions.
            tags: Optional list of tags associated with this streaming call. Tags
                can be used for filtering or categorizing streaming events.
            **kwargs: Additional keyword arguments. May include streaming-specific
                parameters or model metadata.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code and may interrupt streaming if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from langchain_core.outputs import GenerationChunk, ChatGenerationChunk
            from uuid import UUID
            from typing import Any
            import asyncio
            import logging

            logger = logging.getLogger(__name__)

            class AsyncStreamingHandler(AsyncCallbackHandler):
                \"\"\"Async handler for real-time streaming token processing.\"\"\"
                
                def __init__(self):
                    self.token_buffer = []
                    self.token_count = 0
                
                async def on_llm_new_token(
                    self,
                    token: str,
                    *,
                    chunk: GenerationChunk | ChatGenerationChunk | None = None,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Process token asynchronously
                    self.token_count += 1
                    self.token_buffer.append(token)
                    
                    # Async I/O doesn't block streaming
                    logger.info(f"Token {self.token_count}: '{token}' (run_id={run_id})")
                    
                    # Simulate async processing (e.g., async display update)
                    await asyncio.sleep(0.001)
                    
                    # Buffer management
                    if len(self.token_buffer) > 100:
                        # Async flush to storage
                        buffered_text = ''.join(self.token_buffer)
                        logger.debug(f"Buffer flushed: {len(buffered_text)} chars")
                        self.token_buffer = []
                    
                    # Extract chunk metadata if available
                    if chunk and hasattr(chunk, 'generation_info'):
                        gen_info = chunk.generation_info
                        if gen_info:
                            logger.debug(f"Generation info: {gen_info}")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:2250
        """

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM generation completes successfully (async version).

        This is the async version of the synchronous on_llm_end callback. It is
        invoked when an LLM or chat model successfully completes generation. This
        fires after the complete response is received. Use this for async logging
        of results, async metrics calculation, async database writes, or async
        post-processing without blocking chain execution.

        Args:
            response: The LLMResult object containing the complete generation:
                - generations: List of lists of Generation objects, one list per
                  input prompt. Each Generation contains:
                  - text: The generated text
                  - generation_info: Model-specific metadata (finish_reason, etc.)
                - llm_output: Dictionary with model-level output info (token_usage,
                  model_name, etc.)
                Access token counts via response.llm_output.get('token_usage', {}).
            run_id: The run ID. This is the unique identifier for the LLM call that
                just completed.
            parent_run_id: The parent run ID. This is the ID of the parent chain or
                component, if any. Use this to trace nested executions.
            tags: Optional list of tags associated with this LLM call. Tags can be
                used for filtering or categorizing executions.
            **kwargs: Additional keyword arguments. May include response metadata or
                execution context.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from langchain_core.outputs import LLMResult
            from uuid import UUID
            from typing import Any
            import aiohttp
            import logging

            logger = logging.getLogger(__name__)

            class AsyncLLMCompletionHandler(AsyncCallbackHandler):
                \"\"\"Async handler that logs LLM completion metrics.\"\"\"
                
                def __init__(self, metrics_url: str):
                    self.metrics_url = metrics_url
                    self.session = None
                
                async def on_llm_end(
                    self,
                    response: LLMResult,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Extract token usage
                    token_usage = response.llm_output.get('token_usage', {}) if response.llm_output else {}
                    prompt_tokens = token_usage.get('prompt_tokens', 0)
                    completion_tokens = token_usage.get('completion_tokens', 0)
                    total_tokens = token_usage.get('total_tokens', 0)
                    
                    # Extract generation info
                    num_outputs = len(response.generations)
                    first_output = ''
                    if response.generations and response.generations[0]:
                        first_output = response.generations[0][0].text[:200]
                    
                    logger.info(
                        f"LLM end (run_id={run_id}): "
                        f"{total_tokens} tokens, {num_outputs} outputs"
                    )
                    
                    # Async metrics upload
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    try:
                        async with self.session.post(
                            f"{self.metrics_url}/llm_end",
                            json={
                                'run_id': str(run_id),
                                'prompt_tokens': prompt_tokens,
                                'completion_tokens': completion_tokens,
                                'total_tokens': total_tokens,
                                'num_outputs': num_outputs,
                                'first_output_preview': first_output,
                                'tags': tags
                            }
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to upload metrics: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:2350
        """

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM generation fails (async version).

        This is the async version of the synchronous on_llm_error callback. It is
        invoked when an LLM or chat model call fails with an exception. This fires
        instead of on_llm_end when an error occurs. Use this for async error
        logging, async alerting on failures, async error tracking, or async
        fallback logic without blocking chain error handling.

        Args:
            error: The exception that caused the LLM call to fail. Common error
                types include:
                - Rate limit errors (e.g., openai.RateLimitError)
                - API errors (e.g., openai.APIError)
                - Timeout errors
                - Authentication errors
                - Invalid request errors
                Check error type and message for debugging information.
            run_id: The run ID. This is the unique identifier for the LLM call that
                failed.
            parent_run_id: The parent run ID. This is the ID of the parent chain or
                component, if any. Use this to trace failed nested executions.
            tags: Optional list of tags associated with this LLM call. Tags can be
                used for filtering or categorizing error events.
            **kwargs: Additional keyword arguments. May include:
                - response (LLMResult): Partial response if the error occurred
                  during streaming after some tokens were generated.
                - Other error context or execution metadata.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from uuid import UUID
            from typing import Any
            import aiohttp
            import logging

            logger = logging.getLogger(__name__)

            class AsyncLLMErrorHandler(AsyncCallbackHandler):
                \"\"\"Async handler that tracks and alerts on LLM errors.\"\"\"
                
                def __init__(self, alert_url: str):
                    self.alert_url = alert_url
                    self.error_counts = {}
                    self.session = None
                
                async def on_llm_error(
                    self,
                    error: BaseException,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    error_type = type(error).__name__
                    error_message = str(error)
                    
                    # Track error frequency
                    self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                    
                    logger.error(
                        f"LLM error (run_id={run_id}): {error_type}: {error_message[:200]}"
                    )
                    
                    # Check for partial response
                    partial_response = kwargs.get('response')
                    if partial_response:
                        logger.info("Partial response available before error occurred")
                    
                    # Async alert on critical errors
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    # Alert on rate limits or repeated failures
                    if "rate" in error_message.lower() or self.error_counts[error_type] > 5:
                        try:
                            async with self.session.post(
                                f"{self.alert_url}/llm_error",
                                json={
                                    'run_id': str(run_id),
                                    'error_type': error_type,
                                    'error_message': error_message[:500],
                                    'error_count': self.error_counts[error_type],
                                    'tags': tags,
                                    'severity': 'high' if self.error_counts[error_type] > 5 else 'medium'
                                }
                            ) as resp:
                                await resp.text()
                                logger.info(f"Error alert sent: {error_type}")
                        except Exception as e:
                            logger.error(f"Failed to send error alert: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:2450
        """

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when async chain starts executing (async version).

        This is the async version of the synchronous on_chain_start callback. It
        is invoked when an async chain begins execution using ainvoke, astream, or
        abatch. This is the entry point for async chain execution tracking. Use
        this for async logging, async metrics, async database writes, or async
        input validation without blocking chain execution.

        Args:
            serialized: The serialized chain representation containing:
                - name: Chain class name (e.g., "LLMChain", "SequentialChain")
                - id: List with chain identifier path
                - Chain configuration parameters
                Used for logging and chain identification.
            inputs: The input dictionary passed to the chain. Structure depends
                on the chain type:
                - Simple chains: {"input": user_input}
                - Retrieval chains: {"input": query}
                - Sequential chains: Contains all required input keys
                Keys and structure vary by chain implementation.
            run_id: The run ID. This is the unique identifier for the current
                async chain execution.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                if this is a nested chain execution, or None for top-level chains.
                Use this to trace chain hierarchies.
            tags: Optional list of tags associated with this chain execution.
                Tags can be used for filtering, grouping, or categorizing
                executions.
            metadata: Optional metadata dict associated with this chain execution.
                Metadata can include custom context, user IDs, or execution
                parameters.
            **kwargs: Additional keyword arguments. May include chain-specific
                configuration or execution context.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code and prevent chain execution if raise_error is
            True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from uuid import UUID
            from typing import Any
            import aiohttp
            import asyncio
            import logging

            logger = logging.getLogger(__name__)

            class AsyncChainTrackingHandler(AsyncCallbackHandler):
                \"\"\"Async handler that tracks chain execution flow.\"\"\"
                
                def __init__(self, tracking_url: str):
                    self.tracking_url = tracking_url
                    self.chain_times = {}
                    self.session = None
                
                async def on_chain_start(
                    self,
                    serialized: dict[str, Any],
                    inputs: dict[str, Any],
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Start timing
                    self.chain_times[run_id] = asyncio.get_event_loop().time()
                    
                    chain_name = serialized.get('name', 'Unknown')
                    logger.info(
                        f"Async chain start: {chain_name} (run_id={run_id}, "
                        f"parent={parent_run_id})"
                    )
                    
                    # Async tracking update
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    try:
                        async with self.session.post(
                            f"{self.tracking_url}/chain_start",
                            json={
                                'chain_name': chain_name,
                                'run_id': str(run_id),
                                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                                'inputs': inputs,
                                'tags': tags,
                                'metadata': metadata
                            }
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to track chain start: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:2550
        """

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when async chain execution completes successfully (async version).

        This is the async version of the synchronous on_chain_end callback. It is
        invoked when an async chain successfully completes execution. This fires
        after all chain processing is complete and outputs are ready. Use this for
        async result logging, async metrics calculation, async output validation,
        or async post-processing without blocking chain execution.

        Args:
            outputs: The output dictionary produced by the chain. Structure depends
                on the chain type:
                - Simple chains: {"output": result_text}
                - Retrieval chains: {"answer": answer, "context": documents}
                - Sequential chains: Contains all output keys from final chain
                - Transform chains: Custom output structure
                Keys and structure vary by chain implementation.
            run_id: The run ID. This is the unique identifier for the async chain
                execution that just completed.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                if this was a nested chain execution, or None for top-level chains.
                Use this to trace chain hierarchies.
            tags: Optional list of tags associated with this chain execution. Tags
                can be used for filtering or categorizing executions.
            **kwargs: Additional keyword arguments. May include execution metadata
                or chain-specific output information.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from uuid import UUID
            from typing import Any
            import aiohttp
            import asyncio
            import logging

            logger = logging.getLogger(__name__)

            class AsyncChainMetricsHandler(AsyncCallbackHandler):
                \"\"\"Async handler that calculates chain execution metrics.\"\"\"
                
                def __init__(self, metrics_url: str):
                    self.metrics_url = metrics_url
                    self.chain_times = {}
                    self.session = None
                
                async def on_chain_start(
                    self,
                    serialized: dict[str, Any],
                    inputs: dict[str, Any],
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Record start time
                    self.chain_times[run_id] = asyncio.get_event_loop().time()
                
                async def on_chain_end(
                    self,
                    outputs: dict[str, Any],
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Calculate execution time
                    start_time = self.chain_times.get(run_id)
                    duration = None
                    if start_time:
                        duration = asyncio.get_event_loop().time() - start_time
                        del self.chain_times[run_id]
                    
                    logger.info(
                        f"Async chain end (run_id={run_id}): "
                        f"duration={duration:.2f}s if duration else 'N/A'"
                    )
                    
                    # Extract output metrics
                    output_keys = list(outputs.keys())
                    output_size = sum(len(str(v)) for v in outputs.values())
                    
                    # Async metrics upload
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    try:
                        async with self.session.post(
                            f"{self.metrics_url}/chain_end",
                            json={
                                'run_id': str(run_id),
                                'duration': duration,
                                'output_keys': output_keys,
                                'output_size': output_size,
                                'tags': tags
                            }
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to upload chain metrics: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:2650
        """

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when async chain execution fails (async version).

        This is the async version of the synchronous on_chain_error callback. It
        is invoked when an async chain execution fails with an exception. This
        fires instead of on_chain_end when an error occurs during chain execution.
        Use this for async error logging, async alerting on chain failures, async
        error tracking, or async error recovery without blocking error propagation.

        Args:
            error: The exception that caused the chain execution to fail. Common
                error types include:
                - ValueError: Invalid inputs or configuration
                - KeyError: Missing required input/output keys
                - LLM errors propagated from nested LLM calls
                - Tool execution errors from agent chains
                - Validation errors from output parsers
                Check error type and message for debugging information.
            run_id: The run ID. This is the unique identifier for the async chain
                execution that failed.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                if this was a nested chain execution, or None for top-level chains.
                Use this to trace failed nested executions.
            tags: Optional list of tags associated with this chain execution. Tags
                can be used for filtering or categorizing error events.
            **kwargs: Additional keyword arguments. May include partial outputs,
                error context, or execution state at the time of failure.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from uuid import UUID
            from typing import Any
            import aiohttp
            import asyncio
            import logging

            logger = logging.getLogger(__name__)

            class AsyncChainErrorHandler(AsyncCallbackHandler):
                \"\"\"Async handler that tracks and alerts on chain errors.\"\"\"
                
                def __init__(self, alert_url: str):
                    self.alert_url = alert_url
                    self.chain_times = {}
                    self.error_counts = {}
                    self.session = None
                
                async def on_chain_start(
                    self,
                    serialized: dict[str, Any],
                    inputs: dict[str, Any],
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    self.chain_times[run_id] = {
                        'start': asyncio.get_event_loop().time(),
                        'chain_name': serialized.get('name', 'Unknown')
                    }
                
                async def on_chain_error(
                    self,
                    error: BaseException,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    error_type = type(error).__name__
                    error_message = str(error)
                    
                    # Calculate execution time before failure
                    chain_info = self.chain_times.get(run_id)
                    duration = None
                    chain_name = 'Unknown'
                    if chain_info:
                        duration = asyncio.get_event_loop().time() - chain_info['start']
                        chain_name = chain_info['chain_name']
                        del self.chain_times[run_id]
                    
                    # Track error frequency
                    self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                    
                    logger.error(
                        f"Async chain error in {chain_name} (run_id={run_id}): "
                        f"{error_type}: {error_message[:200]}"
                    )
                    
                    if duration:
                        logger.error(f"Failed after {duration:.2f}s")
                    
                    # Async alert on critical errors
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    # Alert on repeated failures
                    if self.error_counts[error_type] > 3:
                        try:
                            async with self.session.post(
                                f"{self.alert_url}/chain_error",
                                json={
                                    'chain_name': chain_name,
                                    'run_id': str(run_id),
                                    'parent_run_id': str(parent_run_id) if parent_run_id else None,
                                    'error_type': error_type,
                                    'error_message': error_message[:500],
                                    'error_count': self.error_counts[error_type],
                                    'duration': duration,
                                    'tags': tags,
                                    'severity': 'high'
                                }
                            ) as resp:
                                await resp.text()
                                logger.info(f"Chain error alert sent: {error_type}")
                        except Exception as e:
                            logger.error(f"Failed to send error alert: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:2933
        """

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when async tool starts executing (async version).

        This is the async version of the synchronous on_tool_start callback. It is
        invoked when an async tool begins execution, typically called by an agent.
        This fires before async tool execution. Use this for async logging of tool
        invocations, async input validation, async metrics tracking, or async
        database writes without blocking tool execution.

        Args:
            serialized: The serialized tool representation containing:
                - name: Tool name (e.g., "Calculator", "Search", "WikipediaAPI")
                - id: List with tool identifier path
                - Tool description and configuration
                Used for logging and tool identification.
            input_str: The input string passed to the tool. This is the primary
                input parameter as a string representation. For structured tools,
                this may be a JSON-encoded representation of the input dict.
            run_id: The run ID. This is the unique identifier for the current
                async tool execution.
            parent_run_id: The parent run ID. This is typically the ID of the
                agent or chain that invoked this tool. Use this to trace tool
                usage within async agent workflows.
            tags: Optional list of tags associated with this tool execution.
                Tags can be used for filtering, grouping, or categorizing
                executions.
            metadata: Optional metadata dict associated with this tool execution.
                Metadata can include custom context, user IDs, or execution
                parameters.
            inputs: Optional structured input dict for tools that accept
                structured inputs (StructuredTool). Contains parameter names as
                keys and values as passed by the agent.
            **kwargs: Additional keyword arguments. May include tool-specific
                configuration or execution context.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code and prevent tool execution if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from uuid import UUID
            from typing import Any
            import aiohttp
            import asyncio
            import logging

            logger = logging.getLogger(__name__)

            class AsyncToolTrackingHandler(AsyncCallbackHandler):
                \"\"\"Async handler that tracks tool execution patterns.\"\"\"
                
                def __init__(self, tracking_url: str):
                    self.tracking_url = tracking_url
                    self.tool_times = {}
                    self.tool_call_count = {}
                    self.session = None
                
                async def on_tool_start(
                    self,
                    serialized: dict[str, Any],
                    input_str: str,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    inputs: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Start timing
                    self.tool_times[run_id] = asyncio.get_event_loop().time()
                    
                    # Track tool usage
                    tool_name = serialized.get('name', 'Unknown')
                    self.tool_call_count[tool_name] = self.tool_call_count.get(tool_name, 0) + 1
                    
                    logger.info(
                        f"Async tool start: {tool_name} (run_id={run_id}, "
                        f"agent={parent_run_id})"
                    )
                    
                    # Async tracking update
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    try:
                        async with self.session.post(
                            f"{self.tracking_url}/tool_start",
                            json={
                                'tool_name': tool_name,
                                'run_id': str(run_id),
                                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                                'input_str': input_str[:200],
                                'inputs': inputs,
                                'tags': tags,
                                'call_count': self.tool_call_count[tool_name]
                            }
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to track tool start: {e}")
                    
                    # Warn on repeated tool calls (potential loop)
                    if self.tool_call_count[tool_name] > 5:
                        logger.warning(
                            f"Tool '{tool_name}' called {self.tool_call_count[tool_name]} times - "
                            "possible agent loop"
                        )
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:3080
        """

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when async tool execution completes successfully (async version).

        This is the async version of the synchronous on_tool_end callback. It is
        invoked when an async tool successfully completes execution. This fires
        after the tool has produced its output. Use this for async result logging,
        async metrics calculation, async output validation, or async post-processing
        without blocking tool execution.

        Args:
            output: The output produced by the tool. Type and structure depend on
                the tool implementation:
                - String tools: Simple text output
                - Search tools: JSON or formatted search results
                - API tools: Structured data from API responses
                - Calculator tools: Numeric results
                The output can be any type: str, int, dict, list, or custom objects.
            run_id: The run ID. This is the unique identifier for the async tool
                execution that just completed.
            parent_run_id: The parent run ID. This is typically the ID of the agent
                or chain that invoked this tool. Use this to trace tool results
                within async agent workflows.
            tags: Optional list of tags associated with this tool execution. Tags
                can be used for filtering or categorizing executions.
            **kwargs: Additional keyword arguments. May include execution metadata
                or tool-specific output information.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from uuid import UUID
            from typing import Any
            import aiohttp
            import asyncio
            import logging

            logger = logging.getLogger(__name__)

            class AsyncToolMetricsHandler(AsyncCallbackHandler):
                \"\"\"Async handler that calculates tool execution metrics.\"\"\"
                
                def __init__(self, metrics_url: str):
                    self.metrics_url = metrics_url
                    self.tool_times = {}
                    self.session = None
                
                async def on_tool_start(
                    self,
                    serialized: dict[str, Any],
                    input_str: str,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    inputs: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    self.tool_times[run_id] = {
                        'start': asyncio.get_event_loop().time(),
                        'tool_name': serialized.get('name', 'Unknown')
                    }
                
                async def on_tool_end(
                    self,
                    output: Any,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Calculate execution time
                    tool_info = self.tool_times.get(run_id)
                    duration = None
                    tool_name = 'Unknown'
                    if tool_info:
                        duration = asyncio.get_event_loop().time() - tool_info['start']
                        tool_name = tool_info['tool_name']
                        del self.tool_times[run_id]
                    
                    logger.info(
                        f"Async tool end: {tool_name} (run_id={run_id})"
                        f"{f', duration={duration:.2f}s' if duration else ''}"
                    )
                    
                    # Measure output size
                    output_size = len(str(output))
                    output_type = type(output).__name__
                    
                    # Async metrics upload
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    try:
                        async with self.session.post(
                            f"{self.metrics_url}/tool_end",
                            json={
                                'tool_name': tool_name,
                                'run_id': str(run_id),
                                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                                'duration': duration,
                                'output_size': output_size,
                                'output_type': output_type,
                                'output_preview': str(output)[:200],
                                'tags': tags
                            }
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to upload tool metrics: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:3180
        """

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when async tool execution fails (async version).

        This is the async version of the synchronous on_tool_error callback. It is
        invoked when an async tool execution fails with an exception. This fires
        instead of on_tool_end when an error occurs during tool execution. Use this
        for async error logging, async alerting on tool failures, async error
        tracking, or async fallback logic without blocking error propagation.

        Args:
            error: The exception that caused the tool execution to fail. Common
                error types include:
                - ValueError: Invalid tool inputs
                - ConnectionError: Network failures for API tools
                - TimeoutError: Tool execution timeout
                - ImportError: Missing dependencies for tool
                - Tool-specific exceptions from the tool implementation
                Check error type and message for debugging information.
            run_id: The run ID. This is the unique identifier for the async tool
                execution that failed.
            parent_run_id: The parent run ID. This is typically the ID of the agent
                or chain that invoked this tool. Use this to trace failed tool
                executions within async agent workflows.
            tags: Optional list of tags associated with this tool execution. Tags
                can be used for filtering or categorizing error events.
            **kwargs: Additional keyword arguments. May include partial output,
                error context, or execution state at the time of failure.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from uuid import UUID
            from typing import Any
            import aiohttp
            import asyncio
            import logging

            logger = logging.getLogger(__name__)

            class AsyncToolErrorHandler(AsyncCallbackHandler):
                \"\"\"Async handler that tracks and alerts on tool errors.\"\"\"
                
                def __init__(self, alert_url: str):
                    self.alert_url = alert_url
                    self.tool_times = {}
                    self.error_counts = {}
                    self.session = None
                
                async def on_tool_start(
                    self,
                    serialized: dict[str, Any],
                    input_str: str,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    inputs: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    self.tool_times[run_id] = {
                        'start': asyncio.get_event_loop().time(),
                        'tool_name': serialized.get('name', 'Unknown')
                    }
                
                async def on_tool_error(
                    self,
                    error: BaseException,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    error_type = type(error).__name__
                    error_message = str(error)
                    
                    # Calculate execution time before failure
                    tool_info = self.tool_times.get(run_id)
                    duration = None
                    tool_name = 'Unknown'
                    if tool_info:
                        duration = asyncio.get_event_loop().time() - tool_info['start']
                        tool_name = tool_info['tool_name']
                        del self.tool_times[run_id]
                    
                    # Track error frequency
                    tool_error_key = f"{tool_name}:{error_type}"
                    self.error_counts[tool_error_key] = self.error_counts.get(tool_error_key, 0) + 1
                    
                    logger.error(
                        f"Async tool error in {tool_name} (run_id={run_id}): "
                        f"{error_type}: {error_message[:200]}"
                    )
                    
                    if duration:
                        logger.error(f"Failed after {duration:.2f}s")
                    
                    # Async alert on critical errors or repeated failures
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    if self.error_counts[tool_error_key] > 2:
                        try:
                            async with self.session.post(
                                f"{self.alert_url}/tool_error",
                                json={
                                    'tool_name': tool_name,
                                    'run_id': str(run_id),
                                    'parent_run_id': str(parent_run_id) if parent_run_id else None,
                                    'error_type': error_type,
                                    'error_message': error_message[:500],
                                    'error_count': self.error_counts[tool_error_key],
                                    'duration': duration,
                                    'tags': tags,
                                    'severity': 'high' if self.error_counts[tool_error_key] > 5 else 'medium'
                                }
                            ) as resp:
                                await resp.text()
                                logger.info(f"Tool error alert sent: {tool_name} - {error_type}")
                        except Exception as e:
                            logger.error(f"Failed to send tool error alert: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:3280
        """

    async def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run on arbitrary intermediate text output (async version).

        This is the async version of the synchronous on_text callback. It is
        invoked when an async component produces intermediate text output that
        doesn't fit into other specific callback types. This is a catch-all for
        text logging during async execution, such as agent thoughts, intermediate
        chain results, or debug output. Use this for async capture and logging of
        miscellaneous text without blocking execution.

        Args:
            text: The arbitrary text produced during async execution. This could
                be agent reasoning steps, intermediate chain outputs, debug
                messages, or any other text output that components emit during
                execution. The format and content depend on the component producing
                the text.
            run_id: The run ID. This is the unique identifier for the current
                async execution that produced this text.
            parent_run_id: The parent run ID. This is the ID of the parent chain,
                agent, or tool execution, if any. Use this to trace text output
                within async execution hierarchies.
            tags: Optional list of tags associated with this execution. Tags can
                be used for filtering or categorizing text output events.
            **kwargs: Additional keyword arguments. May include component-specific
                context or metadata about the text output.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from uuid import UUID
            from typing import Any
            import aiohttp
            import logging

            logger = logging.getLogger(__name__)

            class AsyncVerboseOutputHandler(AsyncCallbackHandler):
                \"\"\"Async handler that captures all intermediate text output.\"\"\"
                
                def __init__(self, storage_url: str):
                    self.storage_url = storage_url
                    self.text_outputs = []
                    self.session = None
                
                async def on_text(
                    self,
                    text: str,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Store all text outputs
                    self.text_outputs.append({
                        'run_id': str(run_id),
                        'parent_run_id': str(parent_run_id) if parent_run_id else None,
                        'text': text,
                        'tags': tags,
                        'kwargs': kwargs
                    })
                    
                    # Log intermediate output
                    logger.info(f"Async intermediate text (run_id={run_id}): {text[:100]}...")
                    
                    # Async storage update every 10 outputs
                    if len(self.text_outputs) % 10 == 0:
                        if not self.session:
                            self.session = aiohttp.ClientSession()
                        
                        try:
                            async with self.session.post(
                                f"{self.storage_url}/text_outputs",
                                json={
                                    'outputs': self.text_outputs[-10:],
                                    'total_count': len(self.text_outputs)
                                }
                            ) as resp:
                                await resp.text()
                        except Exception as e:
                            logger.error(f"Failed to store text outputs: {e}")
                
                def get_execution_transcript(self) -> str:
                    \"\"\"Get full transcript of all text outputs.\"\"\"
                    return "\\n".join(item['text'] for item in self.text_outputs)
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:3380
        """

    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when an async retry event occurs (async version).

        This is the async version of the synchronous on_retry callback. It is
        invoked when an async component operation is retried due to a transient
        failure (e.g., rate limits, temporary API errors, network issues). This is
        triggered by retry decorators (using tenacity library) on async LLM calls
        or other retryable async operations. Use this for async retry tracking,
        async exponential backoff logging, or async alerting without blocking.

        Args:
            retry_state: The retry state object from tenacity containing:
                - attempt_number: Current retry attempt number (1-indexed)
                - outcome: The result of the previous attempt (exception or value)
                - seconds_since_start: Total elapsed time since first attempt
                - next_action: Planned next action (sleep time before retry)
                - args/kwargs: Arguments passed to the retried function
                Access via retry_state.attempt_number, retry_state.outcome, etc.
            run_id: The run ID. This is the unique identifier for the async
                execution being retried.
            parent_run_id: The parent run ID. This is the ID of the parent chain,
                LLM call, or component that triggered the retryable async operation.
            **kwargs: Additional keyword arguments. May include retry configuration
                or component-specific context.

        Returns:
            Any: Implementation-defined return value. Return value is typically
            ignored by the callback system.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from tenacity import RetryCallState
            from uuid import UUID
            from typing import Any
            import aiohttp
            import logging

            logger = logging.getLogger(__name__)

            class AsyncRetryMonitorHandler(AsyncCallbackHandler):
                \"\"\"Async handler that monitors retry patterns and alerts on issues.\"\"\"
                
                def __init__(self, alert_url: str, max_retries_alert: int = 3):
                    self.alert_url = alert_url
                    self.retry_counts = {}
                    self.max_retries_alert = max_retries_alert
                    self.session = None
                
                async def on_retry(
                    self,
                    retry_state: RetryCallState,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    **kwargs: Any,
                ) -> Any:
                    # Track retry attempts
                    if run_id not in self.retry_counts:
                        self.retry_counts[run_id] = 0
                    self.retry_counts[run_id] += 1
                    
                    # Log retry event
                    attempt = retry_state.attempt_number
                    elapsed = retry_state.seconds_since_start
                    logger.warning(
                        f"Async retry event: attempt {attempt} after {elapsed:.2f}s "
                        f"(run_id={run_id})"
                    )
                    
                    # Log the exception that triggered retry
                    if retry_state.outcome and retry_state.outcome.failed:
                        exception = retry_state.outcome.exception()
                        logger.error(f"Retry triggered by: {type(exception).__name__}: {exception}")
                    
                    # Async alert on excessive retries
                    if attempt >= self.max_retries_alert:
                        if not self.session:
                            self.session = aiohttp.ClientSession()
                        
                        try:
                            async with self.session.post(
                                f"{self.alert_url}/retry_alert",
                                json={
                                    'run_id': str(run_id),
                                    'parent_run_id': str(parent_run_id) if parent_run_id else None,
                                    'attempt_number': attempt,
                                    'elapsed_seconds': elapsed,
                                    'severity': 'critical' if attempt >= self.max_retries_alert + 2 else 'high'
                                }
                            ) as resp:
                                await resp.text()
                                logger.critical(
                                    f"ALERT: {attempt} retry attempts detected - "
                                    f"possible persistent failure (run_id={run_id})"
                                )
                        except Exception as e:
                            logger.error(f"Failed to send retry alert: {e}")
                    
                    # Log next retry wait time
                    if retry_state.next_action:
                        wait_time = getattr(retry_state.next_action, 'sleep', None)
                        if wait_time:
                            logger.info(f"Waiting {wait_time:.2f}s before next async retry")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:3480
        """

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when async agent takes an action (async version).

        This is the async version of the synchronous on_agent_action callback. It
        is invoked when an async agent decides to take an action (typically calling
        a tool). This fires after the agent's reasoning step but before tool
        execution. Use this for async logging of agent decisions, async validation
        of agent actions, or async metrics tracking without blocking agent execution.

        Args:
            action: The AgentAction object containing:
                - tool: Name of the tool the agent decided to use (str)
                - tool_input: Input arguments for the tool (str or dict)
                - log: The agent's reasoning log/thought process (str)
                This represents the agent's decision about what action to take next.
            run_id: The run ID. This is the unique identifier for the current
                async agent action decision.
            parent_run_id: The parent run ID. This is the ID of the parent agent
                executor or chain that is running this agent, if any. Use this to
                trace agent decisions within async execution hierarchies.
            tags: Optional list of tags associated with this agent execution. Tags
                can be used for filtering, grouping, or categorizing agent actions.
            **kwargs: Additional keyword arguments. May include agent-specific
                context or execution metadata.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code and may interrupt agent execution if raise_error
            is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from langchain_core.agents import AgentAction
            from uuid import UUID
            from typing import Any
            import aiohttp
            import logging

            logger = logging.getLogger(__name__)

            class AsyncAgentActionHandler(AsyncCallbackHandler):
                \"\"\"Async handler that tracks agent decision patterns.\"\"\"
                
                def __init__(self, tracking_url: str):
                    self.tracking_url = tracking_url
                    self.action_counts = {}
                    self.session = None
                
                async def on_agent_action(
                    self,
                    action: AgentAction,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Track tool usage patterns
                    tool_name = action.tool
                    self.action_counts[tool_name] = self.action_counts.get(tool_name, 0) + 1
                    
                    logger.info(
                        f"Async agent action: {tool_name} "
                        f"(run_id={run_id}, count={self.action_counts[tool_name]})"
                    )
                    logger.debug(f"Tool input: {action.tool_input}")
                    logger.debug(f"Agent reasoning: {action.log[:200]}...")
                    
                    # Async tracking update
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    try:
                        async with self.session.post(
                            f"{self.tracking_url}/agent_action",
                            json={
                                'run_id': str(run_id),
                                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                                'tool': tool_name,
                                'tool_input': str(action.tool_input)[:500],
                                'reasoning': action.log[:500],
                                'action_count': self.action_counts[tool_name],
                                'tags': tags
                            }
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to track agent action: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:3755
        """

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when async agent completes execution (async version).

        This is the async version of the synchronous on_agent_finish callback. It
        is invoked when an async agent completes its decision-making loop and
        produces a final answer. This fires after the agent has determined it has
        enough information to answer. Use this for async logging of agent results,
        async metrics calculation, or async result validation without blocking
        agent execution.

        Args:
            finish: The AgentFinish object containing:
                - return_values: Dictionary with agent's final outputs, typically
                  includes "output" key with the final answer (dict)
                - log: The agent's final reasoning log explaining why it finished (str)
                This represents the agent's final conclusion and answer.
            run_id: The run ID. This is the unique identifier for the async agent
                execution that just completed.
            parent_run_id: The parent run ID. This is the ID of the parent agent
                executor or chain that was running this agent, if any. Use this to
                trace agent completion within async execution hierarchies.
            tags: Optional list of tags associated with this agent execution. Tags
                can be used for filtering or categorizing agent completions.
            **kwargs: Additional keyword arguments. May include agent-specific
                execution metadata or context.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from langchain_core.agents import AgentFinish
            from uuid import UUID
            from typing import Any
            import aiohttp
            import logging

            logger = logging.getLogger(__name__)

            class AsyncAgentFinishHandler(AsyncCallbackHandler):
                \"\"\"Async handler that tracks agent completion metrics.\"\"\"
                
                def __init__(self, metrics_url: str):
                    self.metrics_url = metrics_url
                    self.agent_start_times = {}
                    self.session = None
                
                async def on_agent_action(
                    self,
                    action: AgentAction,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Track agent start time on first action
                    if parent_run_id and parent_run_id not in self.agent_start_times:
                        import asyncio
                        self.agent_start_times[parent_run_id] = asyncio.get_event_loop().time()
                
                async def on_agent_finish(
                    self,
                    finish: AgentFinish,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Calculate total agent execution time
                    import asyncio
                    start_time = self.agent_start_times.get(run_id)
                    duration = None
                    if start_time:
                        duration = asyncio.get_event_loop().time() - start_time
                        del self.agent_start_times[run_id]
                    
                    # Extract final output
                    final_output = finish.return_values.get('output', 'N/A')
                    
                    logger.info(
                        f"Async agent finish (run_id={run_id})"
                        f"{f', duration={duration:.2f}s' if duration else ''}"
                    )
                    logger.info(f"Final output: {str(final_output)[:200]}...")
                    logger.debug(f"Final reasoning: {finish.log[:200]}...")
                    
                    # Async metrics upload
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    try:
                        async with self.session.post(
                            f"{self.metrics_url}/agent_finish",
                            json={
                                'run_id': str(run_id),
                                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                                'duration': duration,
                                'output': str(final_output)[:500],
                                'output_keys': list(finish.return_values.keys()),
                                'reasoning': finish.log[:500],
                                'tags': tags
                            }
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to upload agent metrics: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:3860
        """

    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when async retriever starts retrieving documents (async version).

        This is the async version of the synchronous on_retriever_start callback.
        It is invoked when an async retriever begins document retrieval based on a
        query. This fires before document search/retrieval occurs. Use this for
        async logging of retrieval requests, async query preprocessing, or async
        metrics tracking without blocking retrieval execution.

        Args:
            serialized: The serialized retriever representation containing:
                - name: Retriever class name (e.g., "VectorStoreRetriever")
                - id: List with retriever identifier path
                - Retriever configuration (e.g., search_type, search_kwargs)
                Used for logging and retriever identification.
            query: The query string being used to retrieve documents. This is the
                search query that will be embedded or processed to find relevant
                documents. For semantic retrieval, this query will be embedded and
                compared to document embeddings.
            run_id: The run ID. This is the unique identifier for the current
                async retriever execution.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                or component that invoked this retriever, if any. Use this to trace
                async retrieval within execution hierarchies.
            tags: Optional list of tags associated with this retrieval. Tags can
                be used for filtering, grouping, or categorizing retrieval
                operations.
            metadata: Optional metadata dict associated with this retrieval.
                Metadata can include search configuration, user context, or
                retrieval parameters.
            **kwargs: Additional keyword arguments. May include retriever-specific
                configuration or execution context.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code and prevent retrieval if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from uuid import UUID
            from typing import Any
            import aiohttp
            import asyncio
            import logging

            logger = logging.getLogger(__name__)

            class AsyncRetrievalTrackingHandler(AsyncCallbackHandler):
                \"\"\"Async handler that tracks retrieval query patterns.\"\"\"
                
                def __init__(self, analytics_url: str):
                    self.analytics_url = analytics_url
                    self.retrieval_times = {}
                    self.query_history = []
                    self.session = None
                
                async def on_retriever_start(
                    self,
                    serialized: dict[str, Any],
                    query: str,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Start timing
                    self.retrieval_times[run_id] = asyncio.get_event_loop().time()
                    self.query_history.append(query)
                    
                    retriever_name = serialized.get('name', 'Unknown')
                    search_type = serialized.get('search_type', 'similarity')
                    
                    logger.info(
                        f"Async retriever start: {retriever_name} "
                        f"(type={search_type}, run_id={run_id})"
                    )
                    logger.debug(f"Query: {query}")
                    
                    # Async analytics update
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    try:
                        async with self.session.post(
                            f"{self.analytics_url}/retrieval_start",
                            json={
                                'run_id': str(run_id),
                                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                                'retriever_name': retriever_name,
                                'query': query,
                                'query_length': len(query),
                                'search_type': search_type,
                                'tags': tags,
                                'metadata': metadata
                            }
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to track retrieval start: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:4004
        """

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when async retriever completes document retrieval (async version).

        This is the async version of the synchronous on_retriever_end callback. It
        is invoked when an async retriever successfully completes document
        retrieval. This fires after documents have been searched and retrieved. Use
        this for async logging of retrieval results, async document post-processing,
        async metrics calculation, or async result validation without blocking the
        calling code.

        Args:
            documents: The sequence of Document objects retrieved. Each Document
                contains:
                - page_content: The actual text content of the document (str)
                - metadata: Dict with document metadata (source, page, score, etc.)
                Empty sequence indicates no documents found matching the query.
            run_id: The run ID. This is the unique identifier for the async
                retriever execution that just completed.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                or component that invoked this retriever, if any. Use this to trace
                async retrieval within execution hierarchies.
            tags: Optional list of tags associated with this retrieval. Tags can
                be used for filtering or categorizing retrieval operations.
            **kwargs: Additional keyword arguments. May include retriever-specific
                results metadata or execution context.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from langchain_core.documents import Document
            from collections.abc import Sequence
            from uuid import UUID
            from typing import Any
            import aiohttp
            import asyncio
            import logging

            logger = logging.getLogger(__name__)

            class AsyncRetrievalResultsHandler(AsyncCallbackHandler):
                \"\"\"Async handler that analyzes retrieval results.\"\"\"
                
                def __init__(self, analytics_url: str):
                    self.analytics_url = analytics_url
                    self.retrieval_times = {}
                    self.session = None
                
                async def on_retriever_start(
                    self,
                    serialized: dict[str, Any],
                    query: str,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Start timing
                    self.retrieval_times[run_id] = asyncio.get_event_loop().time()
                
                async def on_retriever_end(
                    self,
                    documents: Sequence[Document],
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Calculate retrieval latency
                    start_time = self.retrieval_times.get(run_id)
                    duration = None
                    if start_time:
                        duration = asyncio.get_event_loop().time() - start_time
                        del self.retrieval_times[run_id]
                    
                    # Analyze retrieved documents
                    doc_count = len(documents)
                    total_chars = sum(len(doc.page_content) for doc in documents)
                    sources = [doc.metadata.get('source', 'unknown') for doc in documents]
                    
                    logger.info(
                        f"Async retriever end: {doc_count} documents retrieved "
                        f"({total_chars} total chars)"
                        f"{f', duration={duration:.2f}s' if duration else ''}"
                    )
                    
                    if doc_count == 0:
                        logger.warning("No documents retrieved!")
                    else:
                        logger.debug(f"Sources: {', '.join(set(sources))}")
                        # Log first doc preview
                        preview = documents[0].page_content[:100]
                        logger.debug(f"First doc preview: {preview}...")
                    
                    # Async analytics update
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    try:
                        async with self.session.post(
                            f"{self.analytics_url}/retrieval_end",
                            json={
                                'run_id': str(run_id),
                                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                                'doc_count': doc_count,
                                'total_chars': total_chars,
                                'duration': duration,
                                'sources': list(set(sources)),
                                'tags': tags
                            }
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to track retrieval end: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:4127
        """

    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when async retriever encounters an error (async version).

        This is the async version of the synchronous on_retriever_error callback.
        It is invoked when an async retriever encounters an error during document
        retrieval. This fires when retrieval fails due to connection issues, query
        errors, or other exceptions. Use this for async error logging, async error
        tracking, async alerting, or async fallback triggering without blocking
        error handling.

        Args:
            error: The exception that occurred during retrieval. Common error types:
                - ConnectionError: Vector store or database connection failed
                - ValueError: Invalid query format or parameters
                - TimeoutError: Retrieval operation timed out
                - KeyError: Missing required configuration or metadata
                - Exception subclasses from specific vector store implementations
            run_id: The run ID. This is the unique identifier for the async
                retriever execution that failed.
            parent_run_id: The parent run ID. This is the ID of the parent chain
                or component that invoked this retriever, if any. Use this to trace
                async retrieval errors within execution hierarchies.
            tags: Optional list of tags associated with this retrieval. Tags can
                be used for filtering or categorizing retrieval errors.
            **kwargs: Additional keyword arguments. May include retriever-specific
                error context or execution metadata.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True. The original error is
            propagated regardless.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from uuid import UUID
            from typing import Any
            import aiohttp
            import asyncio
            import logging
            import traceback

            logger = logging.getLogger(__name__)

            class AsyncRetrievalErrorHandler(AsyncCallbackHandler):
                \"\"\"Async handler that tracks and alerts on retrieval errors.\"\"\"
                
                def __init__(self, error_tracking_url: str):
                    self.error_tracking_url = error_tracking_url
                    self.retrieval_times = {}
                    self.error_counts = {}
                    self.session = None
                
                async def on_retriever_start(
                    self,
                    serialized: dict[str, Any],
                    query: str,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    self.retrieval_times[run_id] = asyncio.get_event_loop().time()
                
                async def on_retriever_error(
                    self,
                    error: BaseException,
                    *,
                    run_id: UUID,
                    parent_run_id: UUID | None = None,
                    tags: list[str] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Track error frequency by type
                    error_type = type(error).__name__
                    self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                    
                    # Calculate time to failure
                    start_time = self.retrieval_times.get(run_id)
                    time_to_error = None
                    if start_time:
                        time_to_error = asyncio.get_event_loop().time() - start_time
                        del self.retrieval_times[run_id]
                    
                    logger.error(
                        f"Async retriever error: {error_type} "
                        f"(run_id={run_id}, count={self.error_counts[error_type]})"
                        f"{f', time_to_error={time_to_error:.2f}s' if time_to_error else ''}"
                    )
                    logger.error(f"Error message: {str(error)}")
                    logger.debug(f"Full traceback:\\n{traceback.format_exc()}")
                    
                    # Async error tracking
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    try:
                        async with self.session.post(
                            f"{self.error_tracking_url}/retrieval_error",
                            json={
                                'run_id': str(run_id),
                                'parent_run_id': str(parent_run_id) if parent_run_id else None,
                                'error_type': error_type,
                                'error_message': str(error)[:500],
                                'error_count': self.error_counts[error_type],
                                'time_to_error': time_to_error,
                                'tags': tags,
                                'traceback': traceback.format_exc()[:1000]
                            }
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to track retrieval error: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()
            ```

        Source: libs/core/langchain_core/callbacks/base.py:4269
        """

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Override to define handler for custom events (async version).

        This is the async version for handling custom events dispatched by
        Runnable.dispatch_custom_event(). Custom events allow you to emit
        application-specific events during chain execution that don't fit the
        standard callback lifecycle (on_chain_start, on_llm_start, etc.). Use this
        for async tracking of custom metrics, async logging of domain-specific
        events, or async coordination between components without blocking execution.

        Args:
            name: The name of the custom event. This is a user-defined string
                identifier for the event type. Convention is to use namespaced
                names like "myapp:validation_complete" or "retrieval:cache_hit" to
                avoid collisions with other components.
            data: The data payload for the custom event. Format and structure are
                completely user-defined and must match what the event emitter
                provides. Can be any type: dict, list, str, int, Pydantic model,
                or custom objects. Handler should validate data format.
            run_id: The run ID. This is the unique identifier for the execution
                context where the custom event was dispatched.
            tags: Optional list of tags associated with this custom event. Includes
                tags inherited from parent chains/runnables. Tags can be used for
                filtering or routing events.
            metadata: Optional metadata dict associated with this custom event.
                Includes metadata inherited from parent chains/runnables. Metadata
                can include execution context, user IDs, or event-specific
                parameters.
            **kwargs: Additional keyword arguments. May include event-specific
                context or execution metadata.

        Returns:
            None: This async method returns None.

        Raises:
            This method should not raise exceptions unless the handler's
            raise_error attribute is True. Exceptions raised here will propagate
            to the calling code if raise_error is True.

        Example:
            ```python
            from langchain_core.callbacks.base import AsyncCallbackHandler
            from langchain_core.runnables import RunnableLambda
            from uuid import UUID
            from typing import Any
            import aiohttp
            import logging

            logger = logging.getLogger(__name__)

            class AsyncCustomEventHandler(AsyncCallbackHandler):
                \"\"\"Async handler for application-specific custom events.\"\"\"
                
                def __init__(self, events_url: str):
                    self.events_url = events_url
                    self.event_counts = {}
                    self.session = None
                
                async def on_custom_event(
                    self,
                    name: str,
                    data: Any,
                    *,
                    run_id: UUID,
                    tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    # Track event frequency
                    self.event_counts[name] = self.event_counts.get(name, 0) + 1
                    
                    logger.info(
                        f"Custom event: {name} "
                        f"(run_id={run_id}, count={self.event_counts[name]})"
                    )
                    logger.debug(f"Event data: {data}")
                    logger.debug(f"Event tags: {tags}")
                    
                    # Handle specific event types
                    if name == "myapp:validation_complete":
                        validation_result = data.get('is_valid', False)
                        logger.info(f"Validation result: {validation_result}")
                    
                    elif name == "retrieval:cache_hit":
                        cache_key = data.get('cache_key')
                        logger.info(f"Cache hit for key: {cache_key}")
                    
                    # Async event forwarding
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    
                    try:
                        async with self.session.post(
                            f"{self.events_url}/custom_event",
                            json={
                                'event_name': name,
                                'run_id': str(run_id),
                                'data': str(data)[:1000],  # Truncate large data
                                'event_count': self.event_counts[name],
                                'tags': tags,
                                'metadata': metadata
                            }
                        ) as resp:
                            await resp.text()
                    except Exception as e:
                        logger.error(f"Failed to forward custom event: {e}")
                
                async def __aenter__(self):
                    self.session = aiohttp.ClientSession()
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.session:
                        await self.session.close()

            # Example usage: Dispatch custom events from a Runnable
            async def process_with_events(input_data: str) -> str:
                \"\"\"Process data and emit custom events.\"\"\"
                # Validation step
                from langchain_core.callbacks.manager import adispatch_custom_event
                
                is_valid = len(input_data) > 0
                await adispatch_custom_event(
                    "myapp:validation_complete",
                    {"input": input_data, "is_valid": is_valid}
                )
                
                # Check cache
                cache_key = f"cache:{hash(input_data)}"
                await adispatch_custom_event(
                    "retrieval:cache_hit",
                    {"cache_key": cache_key}
                )
                
                return input_data.upper()

            # Create chain with custom event handler
            async def run_with_custom_events():
                handler = AsyncCustomEventHandler(events_url="http://localhost:8000")
                runnable = RunnableLambda(process_with_events)
                
                async with handler:
                    result = await runnable.ainvoke(
                        "test input",
                        config={"callbacks": [handler]}
                    )
                    print(f"Result: {result}")
            ```

        Source: libs/core/langchain_core/callbacks/base.py:4418
        """


class BaseCallbackManager(CallbackManagerMixin):
    """Base callback manager for LangChain."""

    def __init__(
        self,
        handlers: list[BaseCallbackHandler],
        inheritable_handlers: list[BaseCallbackHandler] | None = None,
        parent_run_id: UUID | None = None,
        *,
        tags: list[str] | None = None,
        inheritable_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize callback manager.

        Args:
            handlers: The handlers.
            inheritable_handlers: The inheritable handlers.
            parent_run_id: The parent run ID.
            tags: The tags.
            inheritable_tags: The inheritable tags.
            metadata: The metadata.
            inheritable_metadata: The inheritable metadata.
        """
        self.handlers: list[BaseCallbackHandler] = handlers
        self.inheritable_handlers: list[BaseCallbackHandler] = (
            inheritable_handlers or []
        )
        self.parent_run_id: UUID | None = parent_run_id
        self.tags = tags or []
        self.inheritable_tags = inheritable_tags or []
        self.metadata = metadata or {}
        self.inheritable_metadata = inheritable_metadata or {}

    def copy(self) -> Self:
        """Return a copy of the callback manager."""
        return self.__class__(
            handlers=self.handlers.copy(),
            inheritable_handlers=self.inheritable_handlers.copy(),
            parent_run_id=self.parent_run_id,
            tags=self.tags.copy(),
            inheritable_tags=self.inheritable_tags.copy(),
            metadata=self.metadata.copy(),
            inheritable_metadata=self.inheritable_metadata.copy(),
        )

    def merge(self, other: BaseCallbackManager) -> Self:
        """Merge the callback manager with another callback manager.

        May be overwritten in subclasses. Primarily used internally
        within merge_configs.

        Returns:
            The merged callback manager of the same type as the current object.

        Example: Merging two callback managers.

            ```python
            from langchain_core.callbacks.manager import (
                CallbackManager,
                trace_as_chain_group,
            )
            from langchain_core.callbacks.stdout import StdOutCallbackHandler

            manager = CallbackManager(handlers=[StdOutCallbackHandler()], tags=["tag2"])
            with trace_as_chain_group("My Group Name", tags=["tag1"]) as group_manager:
                merged_manager = group_manager.merge(manager)
                print(merged_manager.handlers)
                # [
                #    <langchain_core.callbacks.stdout.StdOutCallbackHandler object at ...>,
                #    <langchain_core.callbacks.streaming_stdout.StreamingStdOutCallbackHandler object at ...>,
                # ]

                print(merged_manager.tags)
                #    ['tag2', 'tag1']
            ```
        """  # noqa: E501
        manager = self.__class__(
            parent_run_id=self.parent_run_id or other.parent_run_id,
            handlers=[],
            inheritable_handlers=[],
            tags=list(set(self.tags + other.tags)),
            inheritable_tags=list(set(self.inheritable_tags + other.inheritable_tags)),
            metadata={
                **self.metadata,
                **other.metadata,
            },
        )

        handlers = self.handlers + other.handlers
        inheritable_handlers = self.inheritable_handlers + other.inheritable_handlers

        for handler in handlers:
            manager.add_handler(handler)

        for handler in inheritable_handlers:
            manager.add_handler(handler, inherit=True)
        return manager

    @property
    def is_async(self) -> bool:
        """Whether the callback manager is async."""
        return False

    def add_handler(
        self,
        handler: BaseCallbackHandler,
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Add a handler to the callback manager.

        Args:
            handler: The handler to add.
            inherit: Whether to inherit the handler.
        """
        if handler not in self.handlers:
            self.handlers.append(handler)
        if inherit and handler not in self.inheritable_handlers:
            self.inheritable_handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager.

        Args:
            handler: The handler to remove.
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
        if handler in self.inheritable_handlers:
            self.inheritable_handlers.remove(handler)

    def set_handlers(
        self,
        handlers: list[BaseCallbackHandler],
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Set handlers as the only handlers on the callback manager.

        Args:
            handlers: The handlers to set.
            inherit: Whether to inherit the handlers.
        """
        self.handlers = []
        self.inheritable_handlers = []
        for handler in handlers:
            self.add_handler(handler, inherit=inherit)

    def set_handler(
        self,
        handler: BaseCallbackHandler,
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Set handler as the only handler on the callback manager.

        Args:
            handler: The handler to set.
            inherit: Whether to inherit the handler.
        """
        self.set_handlers([handler], inherit=inherit)

    def add_tags(
        self,
        tags: list[str],
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Add tags to the callback manager.

        Args:
            tags: The tags to add.
            inherit: Whether to inherit the tags.
        """
        for tag in tags:
            if tag in self.tags:
                self.remove_tags([tag])
        self.tags.extend(tags)
        if inherit:
            self.inheritable_tags.extend(tags)

    def remove_tags(self, tags: list[str]) -> None:
        """Remove tags from the callback manager.

        Args:
            tags: The tags to remove.
        """
        for tag in tags:
            if tag in self.tags:
                self.tags.remove(tag)
            if tag in self.inheritable_tags:
                self.inheritable_tags.remove(tag)

    def add_metadata(
        self,
        metadata: dict[str, Any],
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Add metadata to the callback manager.

        Args:
            metadata: The metadata to add.
            inherit: Whether to inherit the metadata.
        """
        self.metadata.update(metadata)
        if inherit:
            self.inheritable_metadata.update(metadata)

    def remove_metadata(self, keys: list[str]) -> None:
        """Remove metadata from the callback manager.

        Args:
            keys: The keys to remove.
        """
        for key in keys:
            self.metadata.pop(key, None)
            self.inheritable_metadata.pop(key, None)


Callbacks = list[BaseCallbackHandler] | BaseCallbackManager | None
