"""Run managers."""

from __future__ import annotations

import asyncio
import atexit
import functools
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import copy_context
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from langsmith.run_helpers import get_tracing_context
from typing_extensions import Self, override

from langchain_core.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    Callbacks,
    ChainManagerMixin,
    LLMManagerMixin,
    RetrieverManagerMixin,
    RunManagerMixin,
    ToolManagerMixin,
)
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.globals import get_debug
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.tracers.context import (
    _configure_hooks,
    _get_trace_callbacks,
    _get_tracer_project,
    _tracing_v2_is_enabled,
    tracing_v2_callback_var,
)
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.schemas import Run
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.utils.env import env_var_is_set

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Generator, Sequence

    from tenacity import RetryCallState

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
    from langchain_core.runnables.config import RunnableConfig

logger = logging.getLogger(__name__)


def _get_debug() -> bool:
    return get_debug()


@contextmanager
def trace_as_chain_group(
    group_name: str,
    callback_manager: CallbackManager | None = None,
    *,
    inputs: dict[str, Any] | None = None,
    project_name: str | None = None,
    example_id: str | UUID | None = None,
    run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[CallbackManagerForChainGroup, None, None]:
    """Get a callback manager for a chain group in a context manager.

    Useful for grouping different calls together as a single run even if
    they aren't composed in a single chain.

    Args:
        group_name: The name of the chain group.
        callback_manager: The callback manager to use.
        inputs: The inputs to the chain group.
        project_name: The name of the project.
        example_id: The ID of the example.
        run_id: The ID of the run.
        tags: The inheritable tags to apply to all runs.
        metadata: The metadata to apply to all runs.

    !!! note
        Must have `LANGCHAIN_TRACING_V2` env var set to true to see the trace in
        LangSmith.

    Yields:
        The callback manager for the chain group.

    Example:
        ```python
        llm_input = "Foo"
        with trace_as_chain_group("group_name", inputs={"input": llm_input}) as manager:
            # Use the callback manager for the chain group
            res = llm.invoke(llm_input, {"callbacks": manager})
            manager.on_chain_end({"output": res})
        ```
    """
    cb = _get_trace_callbacks(
        project_name, example_id, callback_manager=callback_manager
    )
    cm = CallbackManager.configure(
        inheritable_callbacks=cb,
        inheritable_tags=tags,
        inheritable_metadata=metadata,
    )

    run_manager = cm.on_chain_start({"name": group_name}, inputs or {}, run_id=run_id)
    child_cm = run_manager.get_child()
    group_cm = CallbackManagerForChainGroup(
        child_cm.handlers,
        child_cm.inheritable_handlers,
        child_cm.parent_run_id,
        parent_run_manager=run_manager,
        tags=child_cm.tags,
        inheritable_tags=child_cm.inheritable_tags,
        metadata=child_cm.metadata,
        inheritable_metadata=child_cm.inheritable_metadata,
    )
    try:
        yield group_cm
    except Exception as e:
        if not group_cm.ended:
            run_manager.on_chain_error(e)
        raise
    else:
        if not group_cm.ended:
            run_manager.on_chain_end({})


@asynccontextmanager
async def atrace_as_chain_group(
    group_name: str,
    callback_manager: AsyncCallbackManager | None = None,
    *,
    inputs: dict[str, Any] | None = None,
    project_name: str | None = None,
    example_id: str | UUID | None = None,
    run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncGenerator[AsyncCallbackManagerForChainGroup, None]:
    """Get an async callback manager for a chain group in a context manager.

    Useful for grouping different async calls together as a single run even if
    they aren't composed in a single chain.

    Args:
        group_name: The name of the chain group.
        callback_manager: The async callback manager to use,
            which manages tracing and other callback behavior.
        inputs: The inputs to the chain group.
        project_name: The name of the project.
        example_id: The ID of the example.
        run_id: The ID of the run.
        tags: The inheritable tags to apply to all runs.
        metadata: The metadata to apply to all runs.

    Yields:
        The async callback manager for the chain group.

    !!! note
        Must have `LANGCHAIN_TRACING_V2` env var set to true to see the trace in
        LangSmith.

    Example:
        ```python
        llm_input = "Foo"
        async with atrace_as_chain_group(
            "group_name", inputs={"input": llm_input}
        ) as manager:
            # Use the async callback manager for the chain group
            res = await llm.ainvoke(llm_input, {"callbacks": manager})
            await manager.on_chain_end({"output": res})
        ```
    """
    cb = _get_trace_callbacks(
        project_name, example_id, callback_manager=callback_manager
    )
    cm = AsyncCallbackManager.configure(
        inheritable_callbacks=cb, inheritable_tags=tags, inheritable_metadata=metadata
    )

    run_manager = await cm.on_chain_start(
        {"name": group_name}, inputs or {}, run_id=run_id
    )
    child_cm = run_manager.get_child()
    group_cm = AsyncCallbackManagerForChainGroup(
        child_cm.handlers,
        child_cm.inheritable_handlers,
        child_cm.parent_run_id,
        parent_run_manager=run_manager,
        tags=child_cm.tags,
        inheritable_tags=child_cm.inheritable_tags,
        metadata=child_cm.metadata,
        inheritable_metadata=child_cm.inheritable_metadata,
    )
    try:
        yield group_cm
    except Exception as e:
        if not group_cm.ended:
            await run_manager.on_chain_error(e)
        raise
    else:
        if not group_cm.ended:
            await run_manager.on_chain_end({})


Func = TypeVar("Func", bound=Callable)


def shielded(func: Func) -> Func:
    """Makes so an awaitable method is always shielded from cancellation.

    Args:
        func: The function to shield.

    Returns:
        The shielded function

    """

    @functools.wraps(func)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.shield(func(*args, **kwargs))

    return cast("Func", wrapped)


def handle_event(
    handlers: list[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: str | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Generic synchronous event dispatcher for callback handlers.
    
    Dispatches events to all registered callback handlers, handling both synchronous
    and asynchronous handlers appropriately. For async handlers called from sync
    context, this function manages the async/sync bridging using ThreadPoolExecutor
    to avoid deadlocks when an event loop is already running.

    !!! note "LangServe Integration"
        This function is used by `LangServe` to handle events in server contexts.

    Args:
        handlers: The list of callback handlers that will receive the event. Each
            handler's corresponding event method will be invoked if the handler
            passes the ignore condition check.
        event_name: The name of the event method to invoke on each handler
            (e.g., `'on_llm_start'`, `'on_chain_end'`, `'on_tool_error'`).
        ignore_condition_name: Name of the boolean attribute on the handler that,
            if True, causes the handler to be skipped for this event. For example,
            `'ignore_llm'` skips handlers with `handler.ignore_llm == True`. Pass
            None to invoke all handlers regardless.
        *args: Positional arguments to pass to each handler's event method.
        **kwargs: Keyword arguments to pass to each handler's event method.
    
    !!! warning "Async Handler Execution"
        When a handler's event method is a coroutine, it is collected and executed
        using the async/sync bridging mechanism. If an event loop is already running,
        coroutines are submitted to a ThreadPoolExecutor to avoid deadlocks. Otherwise,
        they are executed in a new event loop via `_run_coros()`.
    
    !!! info "Error Handling"
        - **NotImplementedError**: Special handling for `on_chat_model_start` falls
          back to `on_llm_start` with converted message strings.
        - **Other Exceptions**: Logged as warnings and raised only if the handler's
          `raise_error` attribute is True, allowing other handlers to continue.
    
    Example:
        ```python
        from langchain_core.callbacks.stdout import StdOutCallbackHandler
        
        handlers = [StdOutCallbackHandler()]
        handle_event(
            handlers,
            "on_llm_start",
            "ignore_llm",
            {"name": "OpenAI"},
            ["What is AI?"],
            run_id=uuid.uuid4(),
            tags=["question"]
        )
        ```
    
    Source: libs/core/langchain_core/callbacks/manager.py:237
    """
    coros: list[Coroutine[Any, Any, Any]] = []

    try:
        message_strings: list[str] | None = None
        for handler in handlers:
            try:
                if ignore_condition_name is None or not getattr(
                    handler, ignore_condition_name
                ):
                    event = getattr(handler, event_name)(*args, **kwargs)
                    if asyncio.iscoroutine(event):
                        coros.append(event)
            except NotImplementedError as e:
                if event_name == "on_chat_model_start":
                    if message_strings is None:
                        message_strings = [get_buffer_string(m) for m in args[1]]
                    handle_event(
                        [handler],
                        "on_llm_start",
                        "ignore_llm",
                        args[0],
                        message_strings,
                        *args[2:],
                        **kwargs,
                    )
                else:
                    handler_name = handler.__class__.__name__
                    logger.warning(
                        "NotImplementedError in %s.%s callback: %s",
                        handler_name,
                        event_name,
                        repr(e),
                    )
            except Exception as e:
                logger.warning(
                    "Error in %s.%s callback: %s",
                    handler.__class__.__name__,
                    event_name,
                    repr(e),
                )
                if handler.raise_error:
                    raise
    finally:
        if coros:
            try:
                # Raises RuntimeError if there is no current event loop.
                asyncio.get_running_loop()
                loop_running = True
            except RuntimeError:
                loop_running = False

            if loop_running:
                # If we try to submit this coroutine to the running loop
                # we end up in a deadlock, as we'd have gotten here from a
                # running coroutine, which we cannot interrupt to run this one.
                # The solution is to run the synchronous function on the globally shared
                # thread pool executor to avoid blocking the main event loop.
                _executor().submit(
                    cast("Callable", copy_context().run), _run_coros, coros
                ).result()
            else:
                # If there's no running loop, we can run the coroutines directly.
                _run_coros(coros)


def _run_coros(coros: list[Coroutine[Any, Any, Any]]) -> None:
    """Execute async callback coroutines in a new event loop from sync context.
    
    This function handles the async/sync bridging pattern used when async callback
    handlers need to be invoked from synchronous callback manager methods. It uses
    different strategies depending on Python version for optimal event loop management.
    
    Args:
        coros: List of coroutines to execute. These are typically async callback
            handler methods that need to run even when called from sync code.
    
    !!! note "Python Version Compatibility"
        - **Python 3.11+**: Uses `asyncio.Runner` API which properly manages
          signal handlers, pending tasks, asyncgens, and executors in a clean
          event loop lifecycle.
        - **Python <3.11**: Falls back to running each coroutine individually
          with `asyncio.run()`, creating and tearing down a new event loop
          for each coroutine.
    
    !!! warning "Error Handling"
        Exceptions raised by individual coroutines are caught and logged as warnings
        but do not prevent other coroutines from executing. This ensures that a
        failing callback handler does not break the entire callback chain.
    
    Source: libs/core/langchain_core/callbacks/manager.py:323
    """
    if hasattr(asyncio, "Runner"):
        # Python 3.11+
        # Run the coroutines in a new event loop, taking care to
        # - install signal handlers
        # - run pending tasks scheduled by `coros`
        # - close asyncgens and executors
        # - close the loop
        with asyncio.Runner() as runner:
            # Run the coroutine, get the result
            for coro in coros:
                try:
                    runner.run(coro)
                except Exception as e:
                    logger.warning("Error in callback coroutine: %s", repr(e))

            # Run pending tasks scheduled by coros until they are all done
            while pending := asyncio.all_tasks(runner.get_loop()):
                runner.run(asyncio.wait(pending))
    else:
        # Before Python 3.11 we need to run each coroutine in a new event loop
        # as the Runner api is not available.
        for coro in coros:
            try:
                asyncio.run(coro)
            except Exception as e:
                logger.warning("Error in callback coroutine: %s", repr(e))


async def _ahandle_event_for_handler(
    handler: BaseCallbackHandler,
    event_name: str,
    ignore_condition_name: str | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    try:
        if ignore_condition_name is None or not getattr(handler, ignore_condition_name):
            event = getattr(handler, event_name)
            if asyncio.iscoroutinefunction(event):
                await event(*args, **kwargs)
            elif handler.run_inline:
                event(*args, **kwargs)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    cast(
                        "Callable",
                        functools.partial(copy_context().run, event, *args, **kwargs),
                    ),
                )
    except NotImplementedError as e:
        if event_name == "on_chat_model_start":
            message_strings = [get_buffer_string(m) for m in args[1]]
            await _ahandle_event_for_handler(
                handler,
                "on_llm_start",
                "ignore_llm",
                args[0],
                message_strings,
                *args[2:],
                **kwargs,
            )
        else:
            logger.warning(
                "NotImplementedError in %s.%s callback: %s",
                handler.__class__.__name__,
                event_name,
                repr(e),
            )
    except Exception as e:
        logger.warning(
            "Error in %s.%s callback: %s",
            handler.__class__.__name__,
            event_name,
            repr(e),
        )
        if handler.raise_error:
            raise


async def ahandle_event(
    handlers: list[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: str | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Async event dispatcher for callback handlers with inline execution control.
    
    Dispatches events to all registered callback handlers in an async context,
    with special handling for handlers marked as `run_inline`. Inline handlers
    execute sequentially to maintain order, while other handlers execute
    concurrently via asyncio.gather() for optimal performance.

    !!! note "LangServe Integration"
        This function is used by `LangServe` to handle events in async server contexts.

    Args:
        handlers: The list of callback handlers that will receive the event. Handlers
            are automatically separated into inline and concurrent groups based on
            their `run_inline` attribute.
        event_name: The name of the event method to invoke on each handler
            (e.g., `'on_llm_start'`, `'on_chain_end'`, `'on_tool_error'`).
        ignore_condition_name: Name of the boolean attribute on the handler that,
            if True, causes the handler to be skipped for this event. For example,
            `'ignore_llm'` skips handlers with `handler.ignore_llm == True`. Pass
            None to invoke all handlers regardless.
        *args: Positional arguments to pass to each handler's event method.
        **kwargs: Keyword arguments to pass to each handler's event method.
    
    !!! info "Execution Order"
        - **Inline Handlers** (`run_inline=True`): Executed sequentially in order
          to preserve execution dependencies and allow handlers to affect subsequent
          handler behavior.
        - **Concurrent Handlers** (`run_inline=False`): Executed concurrently via
          `asyncio.gather()` for better performance when order independence exists.
    
    !!! info "Sync Handler Support"
        Non-async handler methods are supported. Sync handlers marked with `run_inline=True`
        execute inline, while others run in the executor pool to avoid blocking the
        event loop.
    
    Example:
        ```python
        from langchain_core.callbacks.stdout import StdOutCallbackHandler
        
        handlers = [StdOutCallbackHandler()]
        await ahandle_event(
            handlers,
            "on_chain_start",
            "ignore_chain",
            {"name": "MyChain"},
            {"input": "data"},
            run_id=uuid.uuid4()
        )
        ```
    
    Source: libs/core/langchain_core/callbacks/manager.py:468
    """
    for handler in [h for h in handlers if h.run_inline]:
        await _ahandle_event_for_handler(
            handler, event_name, ignore_condition_name, *args, **kwargs
        )
    await asyncio.gather(
        *(
            _ahandle_event_for_handler(
                handler,
                event_name,
                ignore_condition_name,
                *args,
                **kwargs,
            )
            for handler in handlers
            if not handler.run_inline
        )
    )


class BaseRunManager(RunManagerMixin):
    """Base class for run manager (a bound callback manager)."""

    def __init__(
        self,
        *,
        run_id: UUID,
        handlers: list[BaseCallbackHandler],
        inheritable_handlers: list[BaseCallbackHandler],
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        inheritable_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the run manager.

        Args:
            run_id: The ID of the run.
            handlers: The list of handlers.
            inheritable_handlers: The list of inheritable handlers.
            parent_run_id: The ID of the parent run.
            tags: The list of tags.
            inheritable_tags: The list of inheritable tags.
            metadata: The metadata.
            inheritable_metadata: The inheritable metadata.

        """
        self.run_id = run_id
        self.handlers = handlers
        self.inheritable_handlers = inheritable_handlers
        self.parent_run_id = parent_run_id
        self.tags = tags or []
        self.inheritable_tags = inheritable_tags or []
        self.metadata = metadata or {}
        self.inheritable_metadata = inheritable_metadata or {}

    @classmethod
    def get_noop_manager(cls) -> Self:
        """Return a manager that doesn't perform any operations.

        Returns:
            The noop manager.

        """
        return cls(
            run_id=uuid.uuid4(),
            handlers=[],
            inheritable_handlers=[],
            tags=[],
            inheritable_tags=[],
            metadata={},
            inheritable_metadata={},
        )


class RunManager(BaseRunManager):
    """Sync Run Manager."""

    def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        """Run when a text is received.

        Args:
            text: The received text.
            **kwargs: Additional keyword arguments.
        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_retry(
        self,
        retry_state: RetryCallState,
        **kwargs: Any,
    ) -> None:
        """Run when a retry is received.

        Args:
            retry_state: The retry state.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_retry",
            "ignore_retry",
            retry_state,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class ParentRunManager(RunManager):
    """Sync Parent Run Manager."""

    def get_child(self, tag: str | None = None) -> CallbackManager:
        """Get a child callback manager with inherited handlers, tags, and metadata.
        
        Creates a new callback manager instance that inherits all inheritable handlers,
        tags, and metadata from this parent manager, establishing a parent-child run
        relationship. This is used when a chain or tool creates nested executions that
        should be tracked as children in the run hierarchy.

        Args:
            tag: Optional tag to add to the child manager (not inherited by its children).
                Useful for labeling specific sub-operations within a chain execution.

        Returns:
            A new CallbackManager instance configured as a child of this manager, with:
            - parent_run_id set to this manager's run_id
            - All inheritable_handlers copied (but not local handlers)
            - All inheritable_tags copied (but not local tags)
            - All inheritable_metadata copied (but not local metadata)
            - Optional tag added as a local tag if provided
        
        !!! note "Run Hierarchy"
            The parent-child relationship is established via parent_run_id, allowing
            tracing systems like LangSmith to visualize nested execution trees.
        
        Example:
            ```python
            # Parent manager for outer chain
            parent_manager = CallbackManager.configure(
                inheritable_callbacks=[handler],
                inheritable_tags=["outer-chain"]
            )
            
            # Child manager for nested tool execution
            child_manager = parent_manager.get_child(tag="tool-execution")
            # child_manager inherits handler and "outer-chain" tag
            # child_manager also has local "tool-execution" tag
            ```
        
        Source: libs/core/langchain_core/callbacks/manager.py:556
        """
        manager = CallbackManager(handlers=[], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        manager.add_tags(self.inheritable_tags)
        manager.add_metadata(self.inheritable_metadata)
        if tag is not None:
            manager.add_tags([tag], inherit=False)
        return manager


class AsyncRunManager(BaseRunManager, ABC):
    """Async Run Manager."""

    @abstractmethod
    def get_sync(self) -> RunManager:
        """Get the equivalent sync RunManager.

        Returns:
            The sync RunManager.

        """

    async def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        """Run when a text is received.

        Args:
            text: The received text.
            **kwargs: Additional keyword arguments.
        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_retry(
        self,
        retry_state: RetryCallState,
        **kwargs: Any,
    ) -> None:
        """Async run when a retry is received.

        Args:
            retry_state: The retry state.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_retry",
            "ignore_retry",
            retry_state,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncParentRunManager(AsyncRunManager):
    """Async Parent Run Manager."""

    def get_child(self, tag: str | None = None) -> AsyncCallbackManager:
        """Get a child async callback manager with inherited handlers, tags, and metadata.
        
        Creates a new async callback manager instance that inherits all inheritable
        handlers, tags, and metadata from this parent manager, establishing a parent-child
        run relationship in async execution contexts.

        Args:
            tag: Optional tag to add to the child manager (not inherited by its children).
                Useful for labeling specific sub-operations within an async chain execution.

        Returns:
            A new AsyncCallbackManager instance configured as a child of this manager, with:
            - parent_run_id set to this manager's run_id
            - All inheritable_handlers copied (but not local handlers)
            - All inheritable_tags copied (but not local tags)
            - All inheritable_metadata copied (but not local metadata)
            - Optional tag added as a local tag if provided
        
        !!! note "Async Run Hierarchy"
            The parent-child relationship is established via parent_run_id, allowing
            tracing systems like LangSmith to visualize nested async execution trees.
        
        Example:
            ```python
            # Parent manager for outer async chain
            parent_manager = AsyncCallbackManager.configure(
                inheritable_callbacks=[handler],
                inheritable_tags=["async-outer-chain"]
            )
            
            # Child manager for nested async tool execution
            child_manager = parent_manager.get_child(tag="async-tool")
            # child_manager inherits handler and "async-outer-chain" tag
            # child_manager also has local "async-tool" tag
            ```
        
        Source: libs/core/langchain_core/callbacks/manager.py:640
        """
        manager = AsyncCallbackManager(handlers=[], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        manager.add_tags(self.inheritable_tags)
        manager.add_metadata(self.inheritable_metadata)
        if tag is not None:
            manager.add_tags([tag], inherit=False)
        return manager


class CallbackManagerForLLMRun(RunManager, LLMManagerMixin):
    """Callback manager for the lifecycle of a single LLM invocation.
    
    This run manager is created when an LLM starts execution and provides methods
    to dispatch lifecycle events (token generation, completion, errors) to all
    registered callback handlers. It maintains the run context (run_id, tags,
    metadata) throughout the LLM execution.
    
    Lifecycle Methods:
    - `on_llm_new_token()`: Called for each token generated (streaming mode)
    - `on_llm_end()`: Called when LLM completes successfully
    - `on_llm_error()`: Called if LLM raises an exception
    
    !!! note "Run Context"
        The run manager inherits handlers, tags, and metadata from the parent
        callback manager and associates them with a unique run_id for this
        specific LLM invocation.
    
    !!! info "Streaming Support"
        The `on_llm_new_token()` method enables real-time streaming responses,
        allowing handlers to process tokens as they are generated rather than
        waiting for the complete response.
    
    Source: libs/core/langchain_core/callbacks/manager.py:659
    """

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM generates a new token.

        Args:
            token: The new token.
            chunk: The chunk.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token=token,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            chunk=chunk,
            **kwargs,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        Args:
            response: The LLM result.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_llm_end",
            "ignore_llm",
            response,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors.

        Args:
            error: The error.
            **kwargs: Additional keyword arguments.
                - response (LLMResult): The response which was generated before
                    the error occurred.
        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_llm_error",
            "ignore_llm",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForLLMRun(AsyncRunManager, LLMManagerMixin):
    """Async callback manager for the lifecycle of a single LLM invocation.
    
    This async run manager is created when an LLM starts execution in an async
    context and provides async methods to dispatch lifecycle events (token generation,
    completion, errors) to all registered callback handlers. It maintains the run
    context (run_id, tags, metadata) throughout the async LLM execution.
    
    Lifecycle Methods:
    - `on_llm_new_token()`: Called for each token generated (async streaming mode)
    - `on_llm_end()`: Called when LLM completes successfully
    - `on_llm_error()`: Called if LLM raises an exception
    
    !!! note "Async Context"
        All lifecycle methods are async and should be awaited. The manager uses
        `ahandle_event()` for proper async event dispatching to both sync and
        async callback handlers.
    
    !!! info "Streaming Support"
        The `on_llm_new_token()` method enables real-time async streaming responses,
        allowing handlers to process tokens as they are generated without blocking
        the event loop.
    
    Source: libs/core/langchain_core/callbacks/manager.py:739
    """

    def get_sync(self) -> CallbackManagerForLLMRun:
        """Get the equivalent sync RunManager.

        Returns:
            The sync RunManager.

        """
        return CallbackManagerForLLMRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM generates a new token.

        Args:
            token: The new token.
            chunk: The chunk.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token,
            chunk=chunk,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    @shielded
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        Args:
            response: The LLM result.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_llm_end",
            "ignore_llm",
            response,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    @shielded
    async def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors.

        Args:
            error: The error.
            **kwargs: Additional keyword arguments.
                - response (LLMResult): The response which was generated before
                    the error occurred.



        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_llm_error",
            "ignore_llm",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManagerForChainRun(ParentRunManager, ChainManagerMixin):
    """Callback manager for the lifecycle of a single chain execution.
    
    This run manager is created when a chain starts execution and provides methods
    to dispatch lifecycle events (completion, errors, agent actions) to all registered
    callback handlers. As a ParentRunManager, it can create child callback managers
    for nested operations (sub-chains, LLM calls, tool executions).
    
    Lifecycle Methods:
    - `on_chain_end()`: Called when chain completes successfully
    - `on_chain_error()`: Called if chain raises an exception
    - `on_agent_action()`: Called when an agent takes an action
    - `on_agent_finish()`: Called when an agent completes
    
    !!! note "Parent-Child Relationships"
        This manager can spawn child run managers for nested operations via inherited
        ParentRunManager methods, establishing a hierarchical run tree for tracing.
    
    !!! info "Agent Integration"
        Includes agent-specific lifecycle methods (on_agent_action, on_agent_finish)
        for tracking agent reasoning loops within chain execution.
    
    Source: libs/core/langchain_core/callbacks/manager.py:842
    """

    def on_chain_end(self, outputs: dict[str, Any] | Any, **kwargs: Any) -> None:
        """Run when chain ends running.

        Args:
            outputs: The outputs of the chain.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_chain_end",
            "ignore_chain",
            outputs,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error: The error.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_chain_error",
            "ignore_chain",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Run when agent action is received.

        Args:
            action: The agent action.
            **kwargs: Additional keyword arguments.
        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_agent_action",
            "ignore_agent",
            action,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent finish is received.

        Args:
            finish: The agent finish.
            **kwargs: Additional keyword arguments.
        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_agent_finish",
            "ignore_agent",
            finish,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForChainRun(AsyncParentRunManager, ChainManagerMixin):
    """Async callback manager for the lifecycle of a single chain execution.
    
    This async run manager is created when a chain starts execution in an async
    context and provides async methods to dispatch lifecycle events (completion,
    errors, agent actions) to all registered callback handlers. As an
    AsyncParentRunManager, it can create child async callback managers for nested
    operations.
    
    Lifecycle Methods:
    - `on_chain_end()`: Async method called when chain completes successfully
    - `on_chain_error()`: Async method called if chain raises an exception
    - `on_agent_action()`: Async method called when an agent takes an action
    - `on_agent_finish()`: Async method called when an agent completes
    
    !!! note "Shielded Operations"
        End and error methods are decorated with @shielded to ensure they complete
        even if the parent task is cancelled, guaranteeing proper cleanup and
        tracing data is recorded.
    
    !!! info "Parent-Child Relationships"
        This manager can spawn child async run managers for nested operations via
        inherited AsyncParentRunManager methods, establishing a hierarchical async
        run tree for tracing.
    
    Source: libs/core/langchain_core/callbacks/manager.py:932
    """

    def get_sync(self) -> CallbackManagerForChainRun:
        """Get the equivalent sync RunManager.

        Returns:
            The sync RunManager.
        """
        return CallbackManagerForChainRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @shielded
    async def on_chain_end(self, outputs: dict[str, Any] | Any, **kwargs: Any) -> None:
        """Run when a chain ends running.

        Args:
            outputs: The outputs of the chain.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_chain_end",
            "ignore_chain",
            outputs,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    @shielded
    async def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error: The error.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_chain_error",
            "ignore_chain",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Run when agent action is received.

        Args:
            action: The agent action.
            **kwargs: Additional keyword arguments.
        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_agent_action",
            "ignore_agent",
            action,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent finish is received.

        Args:
            finish: The agent finish.
            **kwargs: Additional keyword arguments.
        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_agent_finish",
            "ignore_agent",
            finish,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManagerForToolRun(ParentRunManager, ToolManagerMixin):
    """Callback manager for the lifecycle of a single tool execution.
    
    This run manager is created when a tool (typically used by an agent) starts
    execution and provides methods to dispatch lifecycle events (completion, errors)
    to all registered callback handlers. As a ParentRunManager, it can create child
    callback managers for nested operations within tool execution.
    
    Lifecycle Methods:
    - `on_tool_end()`: Called when tool completes successfully
    - `on_tool_error()`: Called if tool raises an exception
    
    !!! note "Agent Integration"
        Tools are primarily used by agents to perform actions. This manager tracks
        individual tool invocations as part of the agent's reasoning loop, allowing
        tracing systems to visualize tool usage patterns.
    
    !!! info "Output Flexibility"
        The `on_tool_end()` method accepts Any type for output, as tools can return
        various data types (strings, dicts, objects) depending on their implementation.
    
    Source: libs/core/langchain_core/callbacks/manager.py:1041
    """

    def on_tool_end(
        self,
        output: Any,
        **kwargs: Any,
    ) -> None:
        """Run when the tool ends running.

        Args:
            output: The output of the tool.
            **kwargs: The keyword arguments to pass to the event handler

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_tool_end",
            "ignore_agent",
            output,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_tool_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        Args:
            error: The error.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_tool_error",
            "ignore_agent",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForToolRun(AsyncParentRunManager, ToolManagerMixin):
    """Async callback manager for the lifecycle of a single tool execution.
    
    This async run manager is created when a tool starts execution in an async
    context and provides async methods to dispatch lifecycle events (completion,
    errors) to all registered callback handlers. Used by async agents for non-blocking
    tool invocations.
    
    Lifecycle Methods:
    - `on_tool_end()`: Async method called when tool completes successfully
    - `on_tool_error()`: Async method called if tool raises an exception
    
    !!! note "Async Agent Integration"
        Async tools enable agents to perform I/O-bound operations (API calls,
        database queries) without blocking the event loop, improving agent
        performance and responsiveness.
    
    !!! info "Output Flexibility"
        The `on_tool_end()` method accepts Any type for output, as async tools can
        return various data types depending on their implementation.
    
    Source: libs/core/langchain_core/callbacks/manager.py:1095
    """

    def get_sync(self) -> CallbackManagerForToolRun:
        """Get the equivalent sync RunManager.

        Returns:
            The sync RunManager.
        """
        return CallbackManagerForToolRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Async run when the tool ends running.

        Args:
            output: The output of the tool.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_tool_end",
            "ignore_agent",
            output,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_tool_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        Args:
            error: The error.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_tool_error",
            "ignore_agent",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManagerForRetrieverRun(ParentRunManager, RetrieverManagerMixin):
    """Callback manager for the lifecycle of a single retriever execution.
    
    This run manager is created when a retriever (used in RAG patterns) starts
    execution and provides methods to dispatch lifecycle events (document retrieval,
    errors) to all registered callback handlers. Retrievers fetch relevant documents
    from vector stores or other sources based on a query.
    
    Lifecycle Methods:
    - `on_retriever_end()`: Called when retriever completes successfully with documents
    - `on_retriever_error()`: Called if retriever raises an exception
    
    !!! note "RAG Pattern Integration"
        Retrievers are essential in Retrieval-Augmented Generation (RAG) patterns,
        fetching context documents that are passed to LLMs. This manager enables
        tracing and monitoring of the retrieval step.
    
    !!! info "Document Output"
        The `on_retriever_end()` method receives a Sequence[Document] containing
        the retrieved documents, allowing handlers to inspect what context was
        fetched for the query.
    
    Source: libs/core/langchain_core/callbacks/manager.py:1162
    """

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> None:
        """Run when retriever ends running.

        Args:
            documents: The retrieved documents.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_retriever_end",
            "ignore_retriever",
            documents,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_retriever_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when retriever errors.

        Args:
            error: The error.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_retriever_error",
            "ignore_retriever",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForRetrieverRun(
    AsyncParentRunManager,
    RetrieverManagerMixin,
):
    """Async callback manager for the lifecycle of a single retriever execution.
    
    This async run manager is created when a retriever starts execution in an async
    context and provides async methods to dispatch lifecycle events (document
    retrieval, errors) to all registered callback handlers. Used in async RAG
    patterns for non-blocking document fetching.
    
    Lifecycle Methods:
    - `on_retriever_end()`: Async method called when retriever completes with documents
    - `on_retriever_error()`: Async method called if retriever raises an exception
    
    !!! note "Async RAG Patterns"
        Async retrievers enable non-blocking document fetching in async chain
        execution, improving performance when retrieval involves network calls
        or database queries.
    
    !!! info "Shielded Operations"
        End and error methods are decorated with @shielded to ensure they complete
        even if the parent task is cancelled, guaranteeing retrieval metrics are
        recorded.
    
    Source: libs/core/langchain_core/callbacks/manager.py:1216
    """

    def get_sync(self) -> CallbackManagerForRetrieverRun:
        """Get the equivalent sync RunManager.

        Returns:
            The sync RunManager.

        """
        return CallbackManagerForRetrieverRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @shielded
    async def on_retriever_end(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> None:
        """Run when the retriever ends running.

        Args:
            documents: The retrieved documents.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_retriever_end",
            "ignore_retriever",
            documents,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    @shielded
    async def on_retriever_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when retriever errors.

        Args:
            error: The error.
            **kwargs: Additional keyword arguments.

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_retriever_error",
            "ignore_retriever",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManager(BaseCallbackManager):
    """Callback manager for LangChain."""

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> list[CallbackManagerForLLMRun]:
        """Run when LLM starts running.

        Args:
            serialized: The serialized LLM.
            prompts: The list of prompts.
            run_id: The ID of the run.
            **kwargs: Additional keyword arguments.

        Returns:
            A callback manager for each prompt as an LLM run.

        """
        managers = []
        for i, prompt in enumerate(prompts):
            # Can't have duplicate runs with the same run ID (if provided)
            run_id_ = run_id if i == 0 and run_id is not None else uuid.uuid4()
            handle_event(
                self.handlers,
                "on_llm_start",
                "ignore_llm",
                serialized,
                [prompt],
                run_id=run_id_,
                parent_run_id=self.parent_run_id,
                tags=self.tags,
                metadata=self.metadata,
                **kwargs,
            )

            managers.append(
                CallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

        return managers

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> list[CallbackManagerForLLMRun]:
        """Run when chat model starts running.

        Args:
            serialized: The serialized LLM.
            messages: The list of messages.
            run_id: The ID of the run.
            **kwargs: Additional keyword arguments.

        Returns:
            A callback manager for each list of messages as an LLM run.

        """
        managers = []
        for message_list in messages:
            if run_id is not None:
                run_id_ = run_id
                run_id = None
            else:
                run_id_ = uuid.uuid4()
            handle_event(
                self.handlers,
                "on_chat_model_start",
                "ignore_chat_model",
                serialized,
                [message_list],
                run_id=run_id_,
                parent_run_id=self.parent_run_id,
                tags=self.tags,
                metadata=self.metadata,
                **kwargs,
            )

            managers.append(
                CallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

        return managers

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | Any,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> CallbackManagerForChainRun:
        """Run when chain starts running.

        Args:
            serialized: The serialized chain.
            inputs: The inputs to the chain.
            run_id: The ID of the run.
            **kwargs: Additional keyword arguments.

        Returns:
            The callback manager for the chain run.

        """
        if run_id is None:
            run_id = uuid.uuid4()
        handle_event(
            self.handlers,
            "on_chain_start",
            "ignore_chain",
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return CallbackManagerForChainRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @override
    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CallbackManagerForToolRun:
        """Run when tool starts running.

        Args:
            serialized: Serialized representation of the tool.
            input_str: The  input to the tool as a string.
                Non-string inputs are cast to strings.
            run_id: ID for the run.
            parent_run_id: The ID of the parent run.
            inputs: The original input to the tool if provided.
                Recommended for usage instead of input_str when the original
                input is needed.
                If provided, the inputs are expected to be formatted as a dict.
                The keys will correspond to the named-arguments in the tool.
            **kwargs: The keyword arguments to pass to the event handler

        Returns:
            The callback manager for the tool run.

        """
        if run_id is None:
            run_id = uuid.uuid4()

        handle_event(
            self.handlers,
            "on_tool_start",
            "ignore_agent",
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            inputs=inputs,
            **kwargs,
        )

        return CallbackManagerForToolRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @override
    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> CallbackManagerForRetrieverRun:
        """Run when the retriever starts running.

        Args:
            serialized: The serialized retriever.
            query: The query.
            run_id: The ID of the run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.

        Returns:
            The callback manager for the retriever run.
        """
        if run_id is None:
            run_id = uuid.uuid4()

        handle_event(
            self.handlers,
            "on_retriever_start",
            "ignore_retriever",
            serialized,
            query,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return CallbackManagerForRetrieverRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    def on_custom_event(
        self,
        name: str,
        data: Any,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Dispatch an adhoc event to the handlers (async version).

        This event should NOT be used in any internal LangChain code. The event
        is meant specifically for users of the library to dispatch custom
        events that are tailored to their application.

        Args:
            name: The name of the adhoc event.
            data: The data for the adhoc event.
            run_id: The ID of the run.

        Raises:
            ValueError: If additional keyword arguments are passed.
        """
        if not self.handlers:
            return
        if kwargs:
            msg = (
                "The dispatcher API does not accept additional keyword arguments."
                "Please do not pass any additional keyword arguments, instead "
                "include them in the data field."
            )
            raise ValueError(msg)
        if run_id is None:
            run_id = uuid.uuid4()

        handle_event(
            self.handlers,
            "on_custom_event",
            "ignore_custom_event",
            name,
            data,
            run_id=run_id,
            tags=self.tags,
            metadata=self.metadata,
        )

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Callbacks = None,
        local_callbacks: Callbacks = None,
        verbose: bool = False,  # noqa: FBT001,FBT002
        inheritable_tags: list[str] | None = None,
        local_tags: list[str] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
        local_metadata: dict[str, Any] | None = None,
    ) -> CallbackManager:
        """Configure the callback manager with handlers, tags, and metadata propagation.
        
        This method creates a new callback manager instance with the specified handlers
        and configuration, properly managing the distinction between inheritable and
        local attributes. Inheritable attributes are passed down to child callback
        managers created during chain execution, while local attributes only apply to
        the current manager instance.

        Args:
            inheritable_callbacks: Callback handlers that will be propagated to child
                callback managers. Can be a list of handlers or another CallbackManager
                instance to copy configuration from.
            local_callbacks: Callback handlers that apply only to this manager instance
                and are not inherited by child managers. Can be a list of handlers or
                another CallbackManager instance.
            verbose: Whether to enable verbose mode, which adds StdOutCallbackHandler
                for printing execution details to stdout.
            inheritable_tags: Tags that will be propagated to child callback managers,
                useful for hierarchical run organization in tracing systems.
            local_tags: Tags that apply only to this manager instance and are not
                inherited by child managers.
            inheritable_metadata: Metadata dict that will be propagated to child
                callback managers, useful for passing context through execution chains.
            local_metadata: Metadata dict that applies only to this manager instance
                and is not inherited by child managers.

        Returns:
            A configured CallbackManager instance with all handlers, tags, and metadata
            properly registered and ready for use in chain execution. The manager will
            also include any handlers configured via environment variables (e.g.,
            LANGCHAIN_TRACING_V2 for LangSmith tracing).
        
        !!! note "Handler Propagation"
            When a callback manager creates child managers (e.g., when a chain invokes
            a nested chain), only handlers and attributes marked as "inheritable" are
            passed down. This allows for fine-grained control over callback scope.
        
        Example:
            ```python
            from langchain_core.callbacks import CallbackManager
            from langchain_core.callbacks.stdout import StdOutCallbackHandler
            
            # Create manager with inheritable handler for nested chains
            manager = CallbackManager.configure(
                inheritable_callbacks=[StdOutCallbackHandler()],
                inheritable_tags=["main-chain"],
                verbose=True
            )
            
            # Child managers will inherit the handler and tags
            child_manager = manager.on_chain_start({"name": "child"}, {})
            ```
        
        Source: libs/core/langchain_core/callbacks/manager.py:1619
        """
        return _configure(
            cls,
            inheritable_callbacks,
            local_callbacks,
            inheritable_tags,
            local_tags,
            inheritable_metadata,
            local_metadata,
            verbose=verbose,
        )


class CallbackManagerForChainGroup(CallbackManager):
    """Callback manager for grouping multiple operations under a single parent run.
    
    This specialized callback manager is used with the `trace_as_chain_group` context
    manager to group logically related operations (that aren't necessarily composed
    in a single chain) under a common parent run in tracing systems. This enables
    organizing complex workflows into hierarchical traces.
    
    Key Features:
    - Maintains reference to parent_run_manager for automatic cleanup
    - Tracks 'ended' state to prevent duplicate end events
    - Overrides merge() to preserve parent_run_manager during configuration merging
    
    !!! note "Automatic Cleanup"
        When exiting the `trace_as_chain_group` context, this manager automatically
        calls `parent_run_manager.on_chain_end()` if not already ended, ensuring
        the parent run is properly closed even if individual operations fail.
    
    !!! info "Use Case"
        Useful when multiple independent chain invocations should be logically
        grouped for analysis, such as parallel tool calls or multi-step workflows
        that don't fit into a single chain composition.
    
    Example:
        ```python
        from langchain_core.callbacks.manager import trace_as_chain_group
        
        with trace_as_chain_group("data-processing", tags=["batch-1"]) as manager:
            # All operations within this context share the same parent run
            result1 = chain1.invoke(input1, {"callbacks": manager})
            result2 = chain2.invoke(input2, {"callbacks": manager})
            # Parent run automatically closed on context exit
        ```
    
    Source: libs/core/langchain_core/callbacks/manager.py:1630
    """

    def __init__(
        self,
        handlers: list[BaseCallbackHandler],
        inheritable_handlers: list[BaseCallbackHandler] | None = None,
        parent_run_id: UUID | None = None,
        *,
        parent_run_manager: CallbackManagerForChainRun,
        **kwargs: Any,
    ) -> None:
        """Initialize the callback manager.

        Args:
            handlers: The list of handlers.
            inheritable_handlers: The list of inheritable handlers.
            parent_run_id: The ID of the parent run.
            parent_run_manager: The parent run manager.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(
            handlers,
            inheritable_handlers,
            parent_run_id,
            **kwargs,
        )
        self.parent_run_manager = parent_run_manager
        self.ended = False

    @override
    def copy(self) -> CallbackManagerForChainGroup:
        return self.__class__(
            handlers=self.handlers.copy(),
            inheritable_handlers=self.inheritable_handlers.copy(),
            parent_run_id=self.parent_run_id,
            tags=self.tags.copy(),
            inheritable_tags=self.inheritable_tags.copy(),
            metadata=self.metadata.copy(),
            inheritable_metadata=self.inheritable_metadata.copy(),
            parent_run_manager=self.parent_run_manager,
        )

    def merge(
        self: CallbackManagerForChainGroup, other: BaseCallbackManager
    ) -> CallbackManagerForChainGroup:
        """Merge the group callback manager with another callback manager.

        Overwrites the merge method in the base class to ensure that the
        parent run manager is preserved. Keeps the parent_run_manager
        from the current object.

        Returns:
            A copy of the current object with the handlers, tags, and other attributes
            merged from the other object.

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
                print(type(merged_manager))
                # <class 'langchain_core.callbacks.manager.CallbackManagerForChainGroup'>

                print(merged_manager.handlers)
                # [
                #    <langchain_core.callbacks.stdout.LangChainTracer object at ...>,
                #    <langchain_core.callbacks.streaming_stdout.StdOutCallbackHandler object at ...>,
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
            parent_run_manager=self.parent_run_manager,
        )

        handlers = self.handlers + other.handlers
        inheritable_handlers = self.inheritable_handlers + other.inheritable_handlers

        for handler in handlers:
            manager.add_handler(handler)

        for handler in inheritable_handlers:
            manager.add_handler(handler, inherit=True)
        return manager

    def on_chain_end(self, outputs: dict[str, Any] | Any, **kwargs: Any) -> None:
        """Run when traced chain group ends.

        Args:
            outputs: The outputs of the chain.
            **kwargs: Additional keyword arguments.

        """
        self.ended = True
        return self.parent_run_manager.on_chain_end(outputs, **kwargs)

    def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error: The error.
            **kwargs: Additional keyword arguments.

        """
        self.ended = True
        return self.parent_run_manager.on_chain_error(error, **kwargs)


class AsyncCallbackManager(BaseCallbackManager):
    """Async callback manager that handles callbacks from LangChain."""

    @property
    def is_async(self) -> bool:
        """Return whether the handler is async."""
        return True

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> list[AsyncCallbackManagerForLLMRun]:
        """Run when LLM starts running.

        Args:
            serialized: The serialized LLM.
            prompts: The list of prompts.
            run_id: The ID of the run.
            **kwargs: Additional keyword arguments.

        Returns:
            The list of async callback managers, one for each LLM Run corresponding to
            each prompt.
        """
        inline_tasks = []
        non_inline_tasks = []
        inline_handlers = [handler for handler in self.handlers if handler.run_inline]
        non_inline_handlers = [
            handler for handler in self.handlers if not handler.run_inline
        ]
        managers = []

        for prompt in prompts:
            if run_id is not None:
                run_id_ = run_id
                run_id = None
            else:
                run_id_ = uuid.uuid4()

            if inline_handlers:
                inline_tasks.append(
                    ahandle_event(
                        inline_handlers,
                        "on_llm_start",
                        "ignore_llm",
                        serialized,
                        [prompt],
                        run_id=run_id_,
                        parent_run_id=self.parent_run_id,
                        tags=self.tags,
                        metadata=self.metadata,
                        **kwargs,
                    )
                )
            else:
                non_inline_tasks.append(
                    ahandle_event(
                        non_inline_handlers,
                        "on_llm_start",
                        "ignore_llm",
                        serialized,
                        [prompt],
                        run_id=run_id_,
                        parent_run_id=self.parent_run_id,
                        tags=self.tags,
                        metadata=self.metadata,
                        **kwargs,
                    )
                )

            managers.append(
                AsyncCallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

        # Run inline tasks sequentially
        for inline_task in inline_tasks:
            await inline_task

        # Run non-inline tasks concurrently
        if non_inline_tasks:
            await asyncio.gather(*non_inline_tasks)

        return managers

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> list[AsyncCallbackManagerForLLMRun]:
        """Async run when LLM starts running.

        Args:
            serialized: The serialized LLM.
            messages: The list of messages.
            run_id: The ID of the run.
            **kwargs: Additional keyword arguments.

        Returns:
            The list of async callback managers, one for each LLM Run corresponding to
            each inner  message list.
        """
        inline_tasks = []
        non_inline_tasks = []
        managers = []

        for message_list in messages:
            if run_id is not None:
                run_id_ = run_id
                run_id = None
            else:
                run_id_ = uuid.uuid4()

            for handler in self.handlers:
                task = ahandle_event(
                    [handler],
                    "on_chat_model_start",
                    "ignore_chat_model",
                    serialized,
                    [message_list],
                    run_id=run_id_,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    metadata=self.metadata,
                    **kwargs,
                )
                if handler.run_inline:
                    inline_tasks.append(task)
                else:
                    non_inline_tasks.append(task)

            managers.append(
                AsyncCallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

        # Run inline tasks sequentially
        for task in inline_tasks:
            await task

        # Run non-inline tasks concurrently
        if non_inline_tasks:
            await asyncio.gather(*non_inline_tasks)

        return managers

    async def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | Any,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForChainRun:
        """Async run when chain starts running.

        Args:
            serialized: The serialized chain.
            inputs: The inputs to the chain.
            run_id: The ID of the run.
            **kwargs: Additional keyword arguments.

        Returns:
            The async callback manager for the chain run.
        """
        if run_id is None:
            run_id = uuid.uuid4()

        await ahandle_event(
            self.handlers,
            "on_chain_start",
            "ignore_chain",
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return AsyncCallbackManagerForChainRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @override
    async def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForToolRun:
        """Run when the tool starts running.

        Args:
            serialized: The serialized tool.
            input_str: The input to the tool.
            run_id: The ID of the run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.

        Returns:
            The async callback manager for the tool run.
        """
        if run_id is None:
            run_id = uuid.uuid4()

        await ahandle_event(
            self.handlers,
            "on_tool_start",
            "ignore_agent",
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return AsyncCallbackManagerForToolRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Dispatch an adhoc event to the handlers (async version).

        This event should NOT be used in any internal LangChain code. The event
        is meant specifically for users of the library to dispatch custom
        events that are tailored to their application.

        Args:
            name: The name of the adhoc event.
            data: The data for the adhoc event.
            run_id: The ID of the run.

        Raises:
            ValueError: If additional keyword arguments are passed.
        """
        if not self.handlers:
            return
        if run_id is None:
            run_id = uuid.uuid4()

        if kwargs:
            msg = (
                "The dispatcher API does not accept additional keyword arguments."
                "Please do not pass any additional keyword arguments, instead "
                "include them in the data field."
            )
            raise ValueError(msg)
        await ahandle_event(
            self.handlers,
            "on_custom_event",
            "ignore_custom_event",
            name,
            data,
            run_id=run_id,
            tags=self.tags,
            metadata=self.metadata,
        )

    @override
    async def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForRetrieverRun:
        """Run when the retriever starts running.

        Args:
            serialized: The serialized retriever.
            query: The query.
            run_id: The ID of the run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.

        Returns:
            The async callback manager for the retriever run.
        """
        if run_id is None:
            run_id = uuid.uuid4()

        await ahandle_event(
            self.handlers,
            "on_retriever_start",
            "ignore_retriever",
            serialized,
            query,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return AsyncCallbackManagerForRetrieverRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Callbacks = None,
        local_callbacks: Callbacks = None,
        verbose: bool = False,  # noqa: FBT001,FBT002
        inheritable_tags: list[str] | None = None,
        local_tags: list[str] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
        local_metadata: dict[str, Any] | None = None,
    ) -> AsyncCallbackManager:
        """Configure the async callback manager with handlers, tags, and metadata propagation.
        
        This method creates a new async callback manager instance with the specified
        handlers and configuration, properly managing the distinction between inheritable
        and local attributes. This is the async equivalent of CallbackManager.configure(),
        designed for use in async chain execution contexts.

        Args:
            inheritable_callbacks: Callback handlers that will be propagated to child
                async callback managers. Can be a list of handlers or another
                AsyncCallbackManager instance to copy configuration from.
            local_callbacks: Callback handlers that apply only to this manager instance
                and are not inherited by child managers. Can be a list of handlers or
                another AsyncCallbackManager instance.
            verbose: Whether to enable verbose mode, which adds StdOutCallbackHandler
                for printing execution details to stdout in async contexts.
            inheritable_tags: Tags that will be propagated to child async callback
                managers, useful for hierarchical run organization in tracing systems.
            local_tags: Tags that apply only to this manager instance and are not
                inherited by child managers.
            inheritable_metadata: Metadata dict that will be propagated to child
                async callback managers, useful for passing context through async
                execution chains.
            local_metadata: Metadata dict that applies only to this manager instance
                and is not inherited by child managers.

        Returns:
            A configured AsyncCallbackManager instance with all handlers, tags, and
            metadata properly registered and ready for use in async chain execution.
            The manager will also include any handlers configured via environment
            variables (e.g., LANGCHAIN_TRACING_V2 for LangSmith tracing).
        
        !!! note "Async Context Awareness"
            This manager is designed for async contexts and will use `await` when
            invoking async callback handler methods. Use `CallbackManager.configure()`
            for synchronous contexts instead.
        
        Example:
            ```python
            from langchain_core.callbacks import AsyncCallbackManager
            from langchain_core.callbacks.stdout import StdOutCallbackHandler
            
            # Create async manager with inheritable handler for nested chains
            manager = AsyncCallbackManager.configure(
                inheritable_callbacks=[StdOutCallbackHandler()],
                inheritable_tags=["async-chain"],
                verbose=True
            )
            
            # Use in async chain execution
            async def run_chain():
                child_manager = await manager.on_chain_start({"name": "child"}, {})
            ```
        
        Source: libs/core/langchain_core/callbacks/manager.py:2115
        """
        return _configure(
            cls,
            inheritable_callbacks,
            local_callbacks,
            inheritable_tags,
            local_tags,
            inheritable_metadata,
            local_metadata,
            verbose=verbose,
        )


class AsyncCallbackManagerForChainGroup(AsyncCallbackManager):
    """Async callback manager for grouping multiple async operations under a single parent run.
    
    This specialized async callback manager is used with the `atrace_as_chain_group`
    context manager to group logically related async operations (that aren't necessarily
    composed in a single chain) under a common parent run in tracing systems. This
    enables organizing complex async workflows into hierarchical traces.
    
    Key Features:
    - Maintains reference to parent_run_manager for automatic async cleanup
    - Tracks 'ended' state to prevent duplicate end events
    - Overrides merge() to preserve parent_run_manager during configuration merging
    
    !!! note "Automatic Async Cleanup"
        When exiting the `atrace_as_chain_group` async context, this manager
        automatically awaits `parent_run_manager.on_chain_end()` if not already
        ended, ensuring the parent run is properly closed even if individual
        operations fail.
    
    !!! info "Use Case"
        Useful when multiple independent async chain invocations should be logically
        grouped for analysis, such as concurrent tool calls or multi-step async
        workflows that don't fit into a single chain composition.
    
    Example:
        ```python
        from langchain_core.callbacks.manager import atrace_as_chain_group
        
        async with atrace_as_chain_group(
            "async-processing", tags=["batch-1"]
        ) as manager:
            # All async operations share the same parent run
            result1 = await chain1.ainvoke(input1, {"callbacks": manager})
            result2 = await chain2.ainvoke(input2, {"callbacks": manager})
            # Parent run automatically closed on context exit
        ```
    
    Source: libs/core/langchain_core/callbacks/manager.py:2151
    """

    def __init__(
        self,
        handlers: list[BaseCallbackHandler],
        inheritable_handlers: list[BaseCallbackHandler] | None = None,
        parent_run_id: UUID | None = None,
        *,
        parent_run_manager: AsyncCallbackManagerForChainRun,
        **kwargs: Any,
    ) -> None:
        """Initialize the async callback manager.

        Args:
            handlers: The list of handlers.
            inheritable_handlers: The list of inheritable handlers.
            parent_run_id: The ID of the parent run.
            parent_run_manager: The parent run manager.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            handlers,
            inheritable_handlers,
            parent_run_id,
            **kwargs,
        )
        self.parent_run_manager = parent_run_manager
        self.ended = False

    def copy(self) -> AsyncCallbackManagerForChainGroup:
        """Return a copy the async callback manager."""
        return self.__class__(
            handlers=self.handlers.copy(),
            inheritable_handlers=self.inheritable_handlers.copy(),
            parent_run_id=self.parent_run_id,
            tags=self.tags.copy(),
            inheritable_tags=self.inheritable_tags.copy(),
            metadata=self.metadata.copy(),
            inheritable_metadata=self.inheritable_metadata.copy(),
            parent_run_manager=self.parent_run_manager,
        )

    def merge(
        self: AsyncCallbackManagerForChainGroup, other: BaseCallbackManager
    ) -> AsyncCallbackManagerForChainGroup:
        """Merge the group callback manager with another callback manager.

        Overwrites the merge method in the base class to ensure that the
        parent run manager is preserved. Keeps the parent_run_manager
        from the current object.

        Returns:
            A copy of the current AsyncCallbackManagerForChainGroup
            with the handlers, tags, etc. of the other callback manager merged in.

        Example: Merging two callback managers.

            ```python
            from langchain_core.callbacks.manager import (
                CallbackManager,
                atrace_as_chain_group,
            )
            from langchain_core.callbacks.stdout import StdOutCallbackHandler

            manager = CallbackManager(handlers=[StdOutCallbackHandler()], tags=["tag2"])
            async with atrace_as_chain_group(
                "My Group Name", tags=["tag1"]
            ) as group_manager:
                merged_manager = group_manager.merge(manager)
                print(type(merged_manager))
                # <class 'langchain_core.callbacks.manager.AsyncCallbackManagerForChainGroup'>

                print(merged_manager.handlers)
                # [
                #    <langchain_core.callbacks.stdout.LangChainTracer object at ...>,
                #    <langchain_core.callbacks.streaming_stdout.StdOutCallbackHandler object at ...>,
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
            parent_run_manager=self.parent_run_manager,
        )

        handlers = self.handlers + other.handlers
        inheritable_handlers = self.inheritable_handlers + other.inheritable_handlers

        for handler in handlers:
            manager.add_handler(handler)

        for handler in inheritable_handlers:
            manager.add_handler(handler, inherit=True)
        return manager

    async def on_chain_end(self, outputs: dict[str, Any] | Any, **kwargs: Any) -> None:
        """Run when traced chain group ends.

        Args:
            outputs: The outputs of the chain.
            **kwargs: Additional keyword arguments.
        """
        self.ended = True
        await self.parent_run_manager.on_chain_end(outputs, **kwargs)

    async def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error: The error.
            **kwargs: Additional keyword arguments.
        """
        self.ended = True
        await self.parent_run_manager.on_chain_error(error, **kwargs)


T = TypeVar("T", CallbackManager, AsyncCallbackManager)


def _configure(
    callback_manager_cls: type[T],
    inheritable_callbacks: Callbacks = None,
    local_callbacks: Callbacks = None,
    inheritable_tags: list[str] | None = None,
    local_tags: list[str] | None = None,
    inheritable_metadata: dict[str, Any] | None = None,
    local_metadata: dict[str, Any] | None = None,
    *,
    verbose: bool = False,
) -> T:
    """Central configuration logic for creating callback manager instances.
    
    This function is the core implementation shared by CallbackManager.configure()
    and AsyncCallbackManager.configure(). It handles complex configuration scenarios
    including:
    - Inheriting handlers/tags/metadata from parent managers
    - Merging with LangSmith tracing context (parent runs, metadata, tags)
    - Auto-enabling handlers based on environment variables
    - Resolving conflicts between external tracing context and inherited context
    
    Configuration Flow:
    1. Retrieve LangSmith tracing context (parent run, tags, metadata)
    2. Create base manager with parent_run_id from tracing context
    3. Configure inheritable and local callbacks/handlers
    4. Merge inheritable and local tags
    5. Merge inheritable and local metadata
    6. Apply tracing context metadata and tags
    7. Auto-enable handlers based on environment variables:
       - LANGCHAIN_TRACING_V2 → LangChainTracer
       - LANGCHAIN_DEBUG → ConsoleCallbackHandler
       - verbose=True → StdOutCallbackHandler
    8. Configure custom hooks from _configure_hooks registry

    Args:
        callback_manager_cls: The callback manager class to instantiate (either
            CallbackManager or AsyncCallbackManager).
        inheritable_callbacks: Callback handlers that will be propagated to child
            managers. Can be a list of handlers or another callback manager instance.
        local_callbacks: Callback handlers that apply only to this manager instance.
            Can be a list of handlers or another callback manager instance.
        inheritable_tags: Tags that will be propagated to child callback managers.
        local_tags: Tags that apply only to this manager instance.
        inheritable_metadata: Metadata dict that will be propagated to child managers.
        local_metadata: Metadata dict that applies only to this manager instance.
        verbose: If True, adds StdOutCallbackHandler for printing execution details.

    Raises:
        RuntimeError: If deprecated `LANGCHAIN_TRACING` environment variable is set
            without `LANGCHAIN_TRACING_V2`. The V1 tracer is no longer supported.

    Returns:
        A fully configured callback manager instance of type callback_manager_cls,
        with all handlers, tags, metadata, and environment-based configurations applied.
    
    !!! note "Parent Run Context Resolution"
        When inheritable_callbacks contains a parent_run_id AND a LangSmith tracing
        context exists with a different parent, this function intelligently resolves
        which parent to use by checking if the LC parent is already reflected in the
        run tree's dotted_order. This prevents duplicate parent tracking.
    
    !!! warning "Handler Deduplication"
        The function ensures handlers are not added multiple times by checking for
        existing instances of each handler class before adding environment-configured
        handlers. This prevents duplicate logging and tracing.
    
    Example:
        ```python
        from langchain_core.callbacks.manager import CallbackManager, _configure
        
        # Create manager with tracing enabled via environment
        # Assumes LANGCHAIN_TRACING_V2=true is set
        manager = _configure(
            CallbackManager,
            inheritable_tags=["experiment-1"],
            verbose=True
        )
        # Result: manager with LangChainTracer + StdOutCallbackHandler
        ```
    
    Source: libs/core/langchain_core/callbacks/manager.py:2285
    """
    tracing_context = get_tracing_context()
    tracing_metadata = tracing_context["metadata"]
    tracing_tags = tracing_context["tags"]
    run_tree: Run | None = tracing_context["parent"]
    parent_run_id = None if run_tree is None else run_tree.id
    callback_manager = callback_manager_cls(
        handlers=[],
        parent_run_id=parent_run_id,
    )
    if inheritable_callbacks or local_callbacks:
        if isinstance(inheritable_callbacks, list) or inheritable_callbacks is None:
            inheritable_callbacks_ = inheritable_callbacks or []
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks_.copy(),
                inheritable_handlers=inheritable_callbacks_.copy(),
                parent_run_id=parent_run_id,
            )
        else:
            parent_run_id_ = inheritable_callbacks.parent_run_id
            # Break ties between the external tracing context and inherited context
            if parent_run_id is not None and (
                parent_run_id_ is None
                # If the LC parent has already been reflected
                # in the run tree, we know the run_tree is either the
                # same parent or a child of the parent.
                or (run_tree and str(parent_run_id_) in run_tree.dotted_order)
            ):
                parent_run_id_ = parent_run_id
                # Otherwise, we assume the LC context has progressed
                # beyond the run tree and we should not inherit the parent.
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks.handlers.copy(),
                inheritable_handlers=inheritable_callbacks.inheritable_handlers.copy(),
                parent_run_id=parent_run_id_,
                tags=inheritable_callbacks.tags.copy(),
                inheritable_tags=inheritable_callbacks.inheritable_tags.copy(),
                metadata=inheritable_callbacks.metadata.copy(),
                inheritable_metadata=inheritable_callbacks.inheritable_metadata.copy(),
            )
        local_handlers_ = (
            local_callbacks
            if isinstance(local_callbacks, list)
            else (local_callbacks.handlers if local_callbacks else [])
        )
        for handler in local_handlers_:
            callback_manager.add_handler(handler, inherit=False)
    if inheritable_tags or local_tags:
        callback_manager.add_tags(inheritable_tags or [])
        callback_manager.add_tags(local_tags or [], inherit=False)
    if inheritable_metadata or local_metadata:
        callback_manager.add_metadata(inheritable_metadata or {})
        callback_manager.add_metadata(local_metadata or {}, inherit=False)
    if tracing_metadata:
        callback_manager.add_metadata(tracing_metadata.copy())
    if tracing_tags:
        callback_manager.add_tags(tracing_tags.copy())

    v1_tracing_enabled_ = env_var_is_set("LANGCHAIN_TRACING") or env_var_is_set(
        "LANGCHAIN_HANDLER"
    )

    tracer_v2 = tracing_v2_callback_var.get()
    tracing_v2_enabled_ = _tracing_v2_is_enabled()

    if v1_tracing_enabled_ and not tracing_v2_enabled_:
        # if both are enabled, can silently ignore the v1 tracer
        msg = (
            "Tracing using LangChainTracerV1 is no longer supported. "
            "Please set the LANGCHAIN_TRACING_V2 environment variable to enable "
            "tracing instead."
        )
        raise RuntimeError(msg)

    tracer_project = _get_tracer_project()
    debug = _get_debug()
    if verbose or debug or tracing_v2_enabled_:
        if verbose and not any(
            isinstance(handler, StdOutCallbackHandler)
            for handler in callback_manager.handlers
        ):
            if debug:
                pass
            else:
                callback_manager.add_handler(StdOutCallbackHandler(), inherit=False)
        if debug and not any(
            isinstance(handler, ConsoleCallbackHandler)
            for handler in callback_manager.handlers
        ):
            callback_manager.add_handler(ConsoleCallbackHandler())
        if tracing_v2_enabled_ and not any(
            isinstance(handler, LangChainTracer)
            for handler in callback_manager.handlers
        ):
            if tracer_v2:
                callback_manager.add_handler(tracer_v2)
            else:
                try:
                    handler = LangChainTracer(
                        project_name=tracer_project,
                        client=(
                            run_tree.client
                            if run_tree is not None
                            else tracing_context["client"]
                        ),
                        tags=tracing_tags,
                    )
                    callback_manager.add_handler(handler)
                except Exception as e:
                    logger.warning(
                        "Unable to load requested LangChainTracer."
                        " To disable this warning,"
                        " unset the LANGCHAIN_TRACING_V2 environment variables.\n"
                        "%s",
                        repr(e),
                    )
        if run_tree is not None:
            for handler in callback_manager.handlers:
                if isinstance(handler, LangChainTracer):
                    handler.order_map[run_tree.id] = (
                        run_tree.trace_id,
                        run_tree.dotted_order,
                    )
                    handler.run_map[str(run_tree.id)] = run_tree
    for var, inheritable, handler_class, env_var in _configure_hooks:
        create_one = (
            env_var is not None
            and env_var_is_set(env_var)
            and handler_class is not None
        )
        if var.get() is not None or create_one:
            var_handler = (
                var.get() or cast("type[BaseCallbackHandler]", handler_class)()
            )
            if handler_class is None:
                if not any(
                    handler is var_handler  # direct pointer comparison
                    for handler in callback_manager.handlers
                ):
                    callback_manager.add_handler(var_handler, inheritable)
            elif not any(
                isinstance(handler, handler_class)
                for handler in callback_manager.handlers
            ):
                callback_manager.add_handler(var_handler, inheritable)
    return callback_manager


async def adispatch_custom_event(
    name: str, data: Any, *, config: RunnableConfig | None = None
) -> None:
    """Dispatch an adhoc event to the handlers.

    Args:
        name: The name of the adhoc event.
        data: The data for the adhoc event. Free form data. Ideally should be
            JSON serializable to avoid serialization issues downstream, but
            this is not enforced.
        config: Optional config object. Mirrors the async API but not strictly needed.

    Raises:
        RuntimeError: If there is no parent run ID available to associate
            the event with.

    Example:
        ```python
        from langchain_core.callbacks import (
            AsyncCallbackHandler,
            adispatch_custom_event
        )
        from langchain_core.runnable import RunnableLambda

        class CustomCallbackManager(AsyncCallbackHandler):
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
                print(f"Received custom event: {name} with data: {data}")

        callback = CustomCallbackManager()

        async def foo(inputs):
            await adispatch_custom_event("my_event", {"bar": "buzz})
            return inputs

        foo_ = RunnableLambda(foo)
        await foo_.ainvoke({"a": "1"}, {"callbacks": [CustomCallbackManager()]})
        ```

    Example: Use with astream events

        ```python
        from langchain_core.callbacks import (
            AsyncCallbackHandler,
            adispatch_custom_event
        )
        from langchain_core.runnable import RunnableLambda

        class CustomCallbackManager(AsyncCallbackHandler):
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
                print(f"Received custom event: {name} with data: {data}")

        callback = CustomCallbackManager()

        async def foo(inputs):
            await adispatch_custom_event("event_type_1", {"bar": "buzz})
            await adispatch_custom_event("event_type_2", 5)
            return inputs

        foo_ = RunnableLambda(foo)

        async for event in foo_.ainvoke_stream(
            {"a": "1"},
            version="v2",
            config={"callbacks": [CustomCallbackManager()]}
        ):
            print(event)
        ```

    !!! warning
        If using python <= 3.10 and async, you MUST
        specify the `config` parameter or the function will raise an error.
        This is due to a limitation in asyncio for python <= 3.10 that prevents
        LangChain from automatically propagating the config object on the user's
        behalf.
    """
    # Import locally to prevent circular imports.
    from langchain_core.runnables.config import (  # noqa: PLC0415
        ensure_config,
        get_async_callback_manager_for_config,
    )

    config = ensure_config(config)
    callback_manager = get_async_callback_manager_for_config(config)
    # We want to get the callback manager for the parent run.
    # This is a work-around for now to be able to dispatch adhoc events from
    # within a tool or a lambda and have the metadata events associated
    # with the parent run rather than have a new run id generated for each.
    if callback_manager.parent_run_id is None:
        msg = (
            "Unable to dispatch an adhoc event without a parent run id."
            "This function can only be called from within an existing run (e.g.,"
            "inside a tool or a RunnableLambda or a RunnableGenerator.)"
            "If you are doing that and still seeing this error, try explicitly"
            "passing the config parameter to this function."
        )
        raise RuntimeError(msg)

    await callback_manager.on_custom_event(
        name,
        data,
        run_id=callback_manager.parent_run_id,
    )


def dispatch_custom_event(
    name: str, data: Any, *, config: RunnableConfig | None = None
) -> None:
    """Dispatch an adhoc event.

    Args:
        name: The name of the adhoc event.
        data: The data for the adhoc event. Free form data. Ideally should be
            JSON serializable to avoid serialization issues downstream, but
            this is not enforced.
        config: Optional config object. Mirrors the async API but not strictly needed.

    Raises:
        RuntimeError: If there is no parent run ID available to associate
            the event with.

    Example:
        ```python
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.callbacks import dispatch_custom_event
        from langchain_core.runnable import RunnableLambda

        class CustomCallbackManager(BaseCallbackHandler):
            def on_custom_event(
                self,
                name: str,
                data: Any,
                *,
                run_id: UUID,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                print(f"Received custom event: {name} with data: {data}")

        def foo(inputs):
            dispatch_custom_event("my_event", {"bar": "buzz})
            return inputs

        foo_ = RunnableLambda(foo)
        foo_.invoke({"a": "1"}, {"callbacks": [CustomCallbackManager()]})
        ```
    """
    # Import locally to prevent circular imports.
    from langchain_core.runnables.config import (  # noqa: PLC0415
        ensure_config,
        get_callback_manager_for_config,
    )

    config = ensure_config(config)
    callback_manager = get_callback_manager_for_config(config)
    # We want to get the callback manager for the parent run.
    # This is a work-around for now to be able to dispatch adhoc events from
    # within a tool or a lambda and have the metadata events associated
    # with the parent run rather than have a new run id generated for each.
    if callback_manager.parent_run_id is None:
        msg = (
            "Unable to dispatch an adhoc event without a parent run id."
            "This function can only be called from within an existing run (e.g.,"
            "inside a tool or a RunnableLambda or a RunnableGenerator.)"
            "If you are doing that and still seeing this error, try explicitly"
            "passing the config parameter to this function."
        )
        raise RuntimeError(msg)
    callback_manager.on_custom_event(
        name,
        data,
        run_id=callback_manager.parent_run_id,
    )


@functools.lru_cache(maxsize=1)
def _executor() -> ThreadPoolExecutor:
    # If the user is specifying ASYNC callback handlers to be run from a
    # SYNC context, and an event loop is already running,
    # we cannot submit the coroutine to the running loop, because it
    # would result in a deadlock. Instead we have to schedule them
    # on a background thread. To avoid creating & shutting down
    # a new executor every time, we use a lazily-created, shared
    # executor. If you're using regular langgchain parallelism (batch, etc.)
    # you'd only ever need 1 worker, but we permit more for now to reduce the chance
    # of slowdown if you are mixing with your own executor.
    cutie = ThreadPoolExecutor(max_workers=10)
    atexit.register(cutie.shutdown, wait=True)
    return cutie
