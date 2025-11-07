"""Chain that takes in an input and produces an action and action input."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import logging
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import (
    Any,
    cast,
)

import yaml
from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    BaseCallbackManager,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self, override

from langchain_classic._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain_classic.agents.agent_iterator import AgentExecutorIterator
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.tools import InvalidTool
from langchain_classic.chains.base import Chain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.utilities.asyncio import asyncio_timeout

logger = logging.getLogger(__name__)


class BaseSingleActionAgent(BaseModel):
    """Base Single Action Agent class.
    
    Abstract base for agents that return a single action per planning step.
    Subclasses must implement plan(), aplan(), and input_keys property to define
    agent behavior.
    
    **Usage:**
    Extend this class to create custom single-action agents. The agent is called
    repeatedly by AgentExecutor until it returns AgentFinish instead of AgentAction.
    
    **Subclass Requirements:**
    - Implement plan(): Synchronous method returning AgentAction or AgentFinish
    - Implement aplan(): Async version of plan()
    - Implement input_keys property: List of required input key names
    
    **Single Action vs Multi Action:**
    BaseSingleActionAgent returns one AgentAction per iteration, suitable for
    sequential tool execution. For parallel tool execution, use BaseMultiActionAgent.
    
    Example:
        ```python
        from langchain_core.agents import AgentAction, AgentFinish
        from langchain_classic.agents.agent import BaseSingleActionAgent
        
        class CustomAgent(BaseSingleActionAgent):
            @property
            def input_keys(self) -> list[str]:
                return ["input"]
            
            def plan(self, intermediate_steps, callbacks=None, **kwargs):
                # Custom planning logic
                if len(intermediate_steps) >= 3:
                    return AgentFinish({"output": "Done"}, "")
                return AgentAction("search", "query", "Searching...")
            
            async def aplan(self, intermediate_steps, callbacks=None, **kwargs):
                return self.plan(intermediate_steps, callbacks, **kwargs)
        ```
    
    Source: libs/langchain/langchain_classic/agents/agent.py:55
    """

    @property
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return ["output"]

    def get_allowed_tools(self) -> list[str] | None:
        """Get allowed tools."""
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Async given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        """Return the input keys.

        :meta private:
        """

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],  # noqa: ARG002
        **_: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations.

        Args:
            early_stopping_method: Method to use for early stopping.
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.

        Returns:
            Agent finish object.

        Raises:
            ValueError: If `early_stopping_method` is not supported.
        """
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."},
                "",
            )
        msg = f"Got unsupported early_stopping_method `{early_stopping_method}`"
        raise ValueError(msg)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: BaseCallbackManager | None = None,
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        """Construct an agent from an LLM and tools.

        Args:
            llm: Language model to use.
            tools: Tools to use.
            callback_manager: Callback manager to use.
            kwargs: Additional arguments.

        Returns:
            Agent object.
        """
        raise NotImplementedError

    @property
    def _agent_type(self) -> str:
        """Return Identifier of an agent type."""
        raise NotImplementedError

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """Return dictionary representation of agent.

        Returns:
            Dictionary representation of agent.
        """
        _dict = super().model_dump()
        try:
            _type = self._agent_type
        except NotImplementedError:
            _type = None
        if isinstance(_type, AgentType):
            _dict["_type"] = str(_type.value)
        elif _type is not None:
            _dict["_type"] = _type
        return _dict

    def save(self, file_path: Path | str) -> None:
        """Save the agent.

        Args:
            file_path: Path to file to save the agent to.

        Example:
        ```python
        # If working with agent executor
        agent.agent.save(file_path="path/agent.yaml")
        ```
        """
        # Convert file to Path object.
        save_path = Path(file_path) if isinstance(file_path, str) else file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        agent_dict = self.dict()
        if "_type" not in agent_dict:
            msg = f"Agent {self} does not support saving"
            raise NotImplementedError(msg)

        if save_path.suffix == ".json":
            with save_path.open("w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} must be json or yaml"
            raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """Return logging kwargs for tool run."""
        return {}


class BaseMultiActionAgent(BaseModel):
    """Base Multi Action Agent class.
    
    Abstract base for agents that can return multiple actions per planning step,
    enabling parallel tool execution. Subclasses must implement plan(), aplan(),
    and input_keys property.
    
    **Difference from BaseSingleActionAgent:**
    While BaseSingleActionAgent returns one AgentAction per iteration,
    BaseMultiActionAgent returns list[AgentAction], allowing the executor to
    run multiple tools in parallel for efficiency.
    
    **Use Case:**
    Appropriate for scenarios where multiple independent tools can be executed
    simultaneously, such as:
    - Gathering data from multiple sources concurrently
    - Running multiple validation checks in parallel
    - Executing complementary research queries together
    
    **Subclass Requirements:**
    - Implement plan(): Return list[AgentAction] | AgentFinish
    - Implement aplan(): Async version of plan()
    - Implement input_keys property: List of required input key names
    
    Example:
        ```python
        from langchain_core.agents import AgentAction, AgentFinish
        from langchain_classic.agents.agent import BaseMultiActionAgent
        
        class ParallelSearchAgent(BaseMultiActionAgent):
            @property
            def input_keys(self) -> list[str]:
                return ["query"]
            
            def plan(self, intermediate_steps, callbacks=None, **kwargs):
                if intermediate_steps:
                    return AgentFinish({"output": "Complete"}, "")
                # Return multiple actions for parallel execution
                return [
                    AgentAction("web_search", kwargs["query"], "Searching web"),
                    AgentAction("wiki_search", kwargs["query"], "Searching wiki")
                ]
            
            async def aplan(self, intermediate_steps, callbacks=None, **kwargs):
                return self.plan(intermediate_steps, callbacks, **kwargs)
        ```
    
    Source: libs/langchain/langchain_classic/agents/agent.py:224
    """

    @property
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return ["output"]

    def get_allowed_tools(self) -> list[str] | None:
        """Get allowed tools.

        Returns:
            Allowed tools.
        """
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Actions specifying what tool to use.
        """

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """Async given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Actions specifying what tool to use.
        """

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        """Return the input keys.

        :meta private:
        """

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],  # noqa: ARG002
        **_: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations.

        Args:
            early_stopping_method: Method to use for early stopping.
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.

        Returns:
            Agent finish object.

        Raises:
            ValueError: If `early_stopping_method` is not supported.
        """
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish({"output": "Agent stopped due to max iterations."}, "")
        msg = f"Got unsupported early_stopping_method `{early_stopping_method}`"
        raise ValueError(msg)

    @property
    def _agent_type(self) -> str:
        """Return Identifier of an agent type."""
        raise NotImplementedError

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """Return dictionary representation of agent."""
        _dict = super().model_dump()
        with contextlib.suppress(NotImplementedError):
            _dict["_type"] = str(self._agent_type)
        return _dict

    def save(self, file_path: Path | str) -> None:
        """Save the agent.

        Args:
            file_path: Path to file to save the agent to.

        Raises:
            NotImplementedError: If agent does not support saving.
            ValueError: If `file_path` is not json or yaml.

        Example:
        ```python
        # If working with agent executor
        agent.agent.save(file_path="path/agent.yaml")
        ```
        """
        # Convert file to Path object.
        save_path = Path(file_path) if isinstance(file_path, str) else file_path

        # Fetch dictionary to save
        agent_dict = self.dict()
        if "_type" not in agent_dict:
            msg = f"Agent {self} does not support saving."
            raise NotImplementedError(msg)

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".json":
            with save_path.open("w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} must be json or yaml"
            raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """Return logging kwargs for tool run."""
        return {}


class AgentOutputParser(BaseOutputParser[AgentAction | AgentFinish]):
    """Base class for parsing agent output into agent action/finish."""

    @abstractmethod
    def parse(self, text: str) -> AgentAction | AgentFinish:
        """Parse text into agent action/finish."""


class MultiActionAgentOutputParser(
    BaseOutputParser[list[AgentAction] | AgentFinish],
):
    """Base class for parsing agent output into agent actions/finish.

    This is used for agents that can return multiple actions.
    """

    @abstractmethod
    def parse(self, text: str) -> list[AgentAction] | AgentFinish:
        """Parse text into agent actions/finish.

        Args:
            text: Text to parse.

        Returns:
            List of agent actions or agent finish.
        """


class RunnableAgent(BaseSingleActionAgent):
    """Agent powered by Runnables.
    
    Adapter wrapping a Runnable as BaseSingleActionAgent, enabling LCEL-based
    agent construction. The wrapped Runnable must accept dict input and return
    AgentAction or AgentFinish.
    
    **Runnable Contract:**
    The wrapped Runnable must:
    - Accept dict input with 'intermediate_steps' key plus any agent-specific keys
    - Return AgentAction (continue execution) or AgentFinish (complete)
    - Handle the agent reasoning logic internally
    
    **Streaming Behavior:**
    The stream_runnable parameter controls LLM invocation mode:
    - True (default): Invokes via .stream() for token-level access in stream_log
    - False: Invokes via .invoke() without token streaming
    
    When streaming is enabled, the agent accumulates chunks into final output,
    making individual LLM tokens available through AgentExecutor.stream_log().
    
    Args:
        runnable: Runnable that implements agent planning logic. Input dict must
            contain 'intermediate_steps' key. Output must be AgentAction or
            AgentFinish.
        input_keys_arg: List of input keys required by agent (excluding
            'intermediate_steps').
        return_keys_arg: List of keys in final output dict.
        stream_runnable: Whether to invoke runnable with .stream() (True) or
            .invoke() (False). Streaming enables token access but may be slower.
    
    Example:
        ```python
        from langchain_core.agents import AgentAction, AgentFinish
        from langchain_core.runnables import RunnableLambda
        from langchain_classic.agents import RunnableAgent, AgentExecutor
        
        def agent_logic(inputs: dict) -> AgentAction | AgentFinish:
            if len(inputs.get("intermediate_steps", [])) >= 2:
                return AgentFinish({"output": "Done"}, "")
            return AgentAction("search", inputs["input"], "Searching")
        
        runnable = RunnableLambda(agent_logic)
        agent = RunnableAgent(
            runnable=runnable,
            input_keys_arg=["input"],
            stream_runnable=False
        )
        executor = AgentExecutor(agent=agent, tools=tools)
        ```
    
    Source: libs/langchain/langchain_classic/agents/agent.py:395
    """

    runnable: Runnable[dict, AgentAction | AgentFinish]
    """Runnable to call to get agent action."""
    input_keys_arg: list[str] = []
    return_keys_arg: list[str] = []
    stream_runnable: bool = True
    """Whether to stream from the runnable or not.

    If `True` then underlying LLM is invoked in a streaming fashion to make it possible
        to get access to the individual LLM tokens when using stream_log with the Agent
        Executor. If `False` then LLM is invoked in a non-streaming fashion and
        individual LLM tokens will not be available in stream_log.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return self.return_keys_arg

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys."""
        return self.input_keys_arg

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # Use streaming to make sure that the underlying LLM is invoked in a
            # streaming
            # fashion to make it possible to get access to the individual LLM tokens
            # when using stream_log with the Agent Executor.
            # Because the response from the plan is not a generator, we need to
            # accumulate the output into final output and return that.
            for chunk in self.runnable.stream(inputs, config={"callbacks": callbacks}):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = self.runnable.invoke(inputs, config={"callbacks": callbacks})

        return final_output

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Async based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # Use streaming to make sure that the underlying LLM is invoked in a
            # streaming
            # fashion to make it possible to get access to the individual LLM tokens
            # when using stream_log with the Agent Executor.
            # Because the response from the plan is not a generator, we need to
            # accumulate the output into final output and return that.
            async for chunk in self.runnable.astream(
                inputs,
                config={"callbacks": callbacks},
            ):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = await self.runnable.ainvoke(
                inputs,
                config={"callbacks": callbacks},
            )
        return final_output


class RunnableMultiActionAgent(BaseMultiActionAgent):
    """Agent powered by Runnables for multi-action scenarios.
    
    Adapter wrapping a Runnable as BaseMultiActionAgent, enabling LCEL-based
    agent construction with parallel action support. The wrapped Runnable must
    accept dict input and return list[AgentAction] or AgentFinish.
    
    **Runnable Contract:**
    The wrapped Runnable must:
    - Accept dict input with 'intermediate_steps' key plus any agent-specific keys
    - Return list[AgentAction] (execute multiple tools) or AgentFinish (complete)
    - Handle multi-action agent reasoning logic internally
    
    **Difference from RunnableAgent:**
    While RunnableAgent returns single AgentAction per iteration,
    RunnableMultiActionAgent returns list[AgentAction], enabling AgentExecutor
    to execute multiple tools in parallel for improved efficiency.
    
    **Streaming Behavior:**
    Same as RunnableAgent - stream_runnable parameter controls whether to use
    .stream() (True, token access) or .invoke() (False, faster) for LLM calls.
    
    Args:
        runnable: Runnable implementing multi-action planning logic. Input dict
            must contain 'intermediate_steps'. Output must be list[AgentAction]
            or AgentFinish.
        input_keys_arg: List of input keys required by agent (excluding
            'intermediate_steps').
        return_keys_arg: List of keys in final output dict.
        stream_runnable: Whether to invoke runnable with .stream() (True) or
            .invoke() (False). Streaming enables token access through stream_log.
    
    Example:
        ```python
        from langchain_core.agents import AgentAction, AgentFinish
        from langchain_core.runnables import RunnableLambda
        from langchain_classic.agents import RunnableMultiActionAgent
        
        def parallel_agent_logic(inputs: dict):
            if inputs.get("intermediate_steps"):
                return AgentFinish({"output": "Complete"}, "")
            # Return multiple actions for parallel execution
            return [
                AgentAction("search_web", inputs["query"], "Web search"),
                AgentAction("search_docs", inputs["query"], "Doc search")
            ]
        
        runnable = RunnableLambda(parallel_agent_logic)
        agent = RunnableMultiActionAgent(
            runnable=runnable,
            input_keys_arg=["query"]
        )
        ```
    
    Source: libs/langchain/langchain_classic/agents/agent.py:503
    """

    runnable: Runnable[dict, list[AgentAction] | AgentFinish]
    """Runnable to call to get agent actions."""
    input_keys_arg: list[str] = []
    return_keys_arg: list[str] = []
    stream_runnable: bool = True
    """Whether to stream from the runnable or not.

    If `True` then underlying LLM is invoked in a streaming fashion to make it possible
        to get access to the individual LLM tokens when using stream_log with the Agent
        Executor. If `False` then LLM is invoked in a non-streaming fashion and
        individual LLM tokens will not be available in stream_log.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return self.return_keys_arg

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys.

        Returns:
            List of input keys.
        """
        return self.input_keys_arg

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """Based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # Use streaming to make sure that the underlying LLM is invoked in a
            # streaming
            # fashion to make it possible to get access to the individual LLM tokens
            # when using stream_log with the Agent Executor.
            # Because the response from the plan is not a generator, we need to
            # accumulate the output into final output and return that.
            for chunk in self.runnable.stream(inputs, config={"callbacks": callbacks}):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = self.runnable.invoke(inputs, config={"callbacks": callbacks})

        return final_output

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """Async based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # Use streaming to make sure that the underlying LLM is invoked in a
            # streaming
            # fashion to make it possible to get access to the individual LLM tokens
            # when using stream_log with the Agent Executor.
            # Because the response from the plan is not a generator, we need to
            # accumulate the output into final output and return that.
            async for chunk in self.runnable.astream(
                inputs,
                config={"callbacks": callbacks},
            ):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = await self.runnable.ainvoke(
                inputs,
                config={"callbacks": callbacks},
            )

        return final_output


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class LLMSingleActionAgent(BaseSingleActionAgent):
    """Base class for single action agents."""

    llm_chain: LLMChain
    """LLMChain to use for agent."""
    output_parser: AgentOutputParser
    """Output parser to use for agent."""
    stop: list[str]
    """List of strings to stop on."""

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys.

        Returns:
            List of input keys.
        """
        return list(set(self.llm_chain.input_keys) - {"intermediate_steps"})

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Async given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        output = await self.llm_chain.arun(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """Return logging kwargs for tool run."""
        return {
            "llm_prefix": "",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class Agent(BaseSingleActionAgent):
    """Legacy agent that calls the language model and decides the action.
    
    **DEPRECATED:** This class is deprecated as of version 0.1.0 and will be
    removed in version 1.0. Use RunnableAgent with LCEL-based agent construction
    (create_react_agent, create_openai_functions_agent, etc.) instead.
    
    **Legacy Architecture:**
    This agent is driven by an LLMChain. The prompt in the LLMChain MUST include
    a variable called "agent_scratchpad" where the agent accumulates its
    intermediary work (previous actions and observations).
    
    **Relationship to BaseSingleActionAgent:**
    Agent extends BaseSingleActionAgent, implementing plan() by:
    1. Constructing scratchpad from intermediate_steps
    2. Calling llm_chain.predict() with inputs + scratchpad
    3. Parsing LLM output with output_parser to get AgentAction or AgentFinish
    
    **Migration Guidance:**
    Instead of Agent with LLMChain, use modern Runnable-based agents:
    
    ```python
    # Old (deprecated):
    from langchain_classic.agents import Agent, AgentExecutor
    agent = Agent(llm_chain=llm_chain, output_parser=parser)
    
    # New (recommended):
    from langchain_classic.agents import create_react_agent, AgentExecutor
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    ```
    
    Args:
        llm_chain: LLMChain to use for agent reasoning. Prompt must include
            'agent_scratchpad' variable for accumulating trajectory.
        output_parser: AgentOutputParser to parse LLM output into AgentAction
            or AgentFinish.
        allowed_tools: Optional list of allowed tool names. If None, all tools
            provided to AgentExecutor are allowed.
    
    Source: libs/langchain/langchain_classic/agents/agent.py:710
    """

    llm_chain: LLMChain
    """LLMChain to use for agent."""
    output_parser: AgentOutputParser
    """Output parser to use for agent."""
    allowed_tools: list[str] | None = None
    """Allowed tools for the agent. If `None`, all tools are allowed."""

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def get_allowed_tools(self) -> list[str] | None:
        """Get allowed tools."""
        return self.allowed_tools

    @property
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return ["output"]

    @property
    def _stop(self) -> list[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]

    def _construct_scratchpad(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
    ) -> str | list[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        return thoughts

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Async given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = await self.llm_chain.apredict(callbacks=callbacks, **full_inputs)
        return await self.output_parser.aparse(full_output)

    def get_full_inputs(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> builtins.dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            **kwargs: User inputs.

        Returns:
            Full inputs for the LLMChain.
        """
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        return {**kwargs, **new_inputs}

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys.

        :meta private:
        """
        return list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        """Validate that prompt matches format.

        Args:
            values: Values to validate.

        Returns:
            Validated values.

        Raises:
            ValueError: If `agent_scratchpad` is not in prompt.input_variables
             and prompt is not a FewShotPromptTemplate or a PromptTemplate.
        """
        prompt = self.llm_chain.prompt
        if "agent_scratchpad" not in prompt.input_variables:
            logger.warning(
                "`agent_scratchpad` should be a variable in prompt.input_variables."
                " Did not find it, so adding it at the end.",
            )
            prompt.input_variables.append("agent_scratchpad")
            if isinstance(prompt, PromptTemplate):
                prompt.template += "\n{agent_scratchpad}"
            elif isinstance(prompt, FewShotPromptTemplate):
                prompt.suffix += "\n{agent_scratchpad}"
            else:
                msg = f"Got unexpected prompt type {type(prompt)}"
                raise ValueError(msg)
        return self

    @property
    @abstractmethod
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""

    @property
    @abstractmethod
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""

    @classmethod
    @abstractmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Create a prompt for this class.

        Args:
            tools: Tools to use.

        Returns:
            Prompt template.
        """

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        """Validate that appropriate tools are passed in.

        Args:
            tools: Tools to use.
        """

    @classmethod
    @abstractmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        """Get default output parser for this class."""

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: BaseCallbackManager | None = None,
        output_parser: AgentOutputParser | None = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools.

        Args:
            llm: Language model to use.
            tools: Tools to use.
            callback_manager: Callback manager to use.
            output_parser: Output parser to use.
            kwargs: Additional arguments.

        Returns:
            Agent object.
        """
        cls._validate_tools(tools)
        llm_chain = LLMChain(
            llm=llm,
            prompt=cls.create_prompt(tools),
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser()
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations.

        Args:
            early_stopping_method: Method to use for early stopping.
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            **kwargs: User inputs.

        Returns:
            Agent finish object.

        Raises:
            ValueError: If `early_stopping_method` is not in ['force', 'generate'].
        """
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."},
                "",
            )
        if early_stopping_method == "generate":
            # Generate does one final forward pass
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += (
                    f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
                )
            # Adding to the previous steps, we now tell the LLM to make a final pred
            thoughts += (
                "\n\nI now need to return a final answer based on the previous steps:"
            )
            new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
            full_inputs = {**kwargs, **new_inputs}
            full_output = self.llm_chain.predict(**full_inputs)
            # We try to extract a final answer
            parsed_output = self.output_parser.parse(full_output)
            if isinstance(parsed_output, AgentFinish):
                # If we can extract, we send the correct stuff
                return parsed_output
            # If we can extract, but the tool is not the final tool,
            # we just return the full output
            return AgentFinish({"output": full_output}, full_output)
        msg = (
            "early_stopping_method should be one of `force` or `generate`, "
            f"got {early_stopping_method}"
        )
        raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """Return logging kwargs for tool run."""
        return {
            "llm_prefix": self.llm_prefix,
            "observation_prefix": self.observation_prefix,
        }


class ExceptionTool(BaseTool):
    """Fallback tool for handling parsing errors.
    
    Internal tool used by AgentExecutor when handle_parsing_errors is enabled
    and agent output cannot be parsed into valid AgentAction. Simply returns
    the input string as observation, allowing the error message to be passed
    back to the agent for recovery.
    
    **When Used:**
    Activated automatically by AgentExecutor when:
    1. OutputParserException is raised during agent.plan()
    2. handle_parsing_errors is configured (True, string, or callable)
    3. Error observation needs to be returned to agent
    
    **Purpose:**
    Enables graceful error recovery by treating parsing errors as tool execution
    results, giving the agent another chance to generate valid output based on
    the error feedback.
    
    Args:
        name: Tool name, always "_Exception" for internal identification.
        description: Tool description, always "Exception tool".
    
    Source: libs/langchain/langchain_classic/agents/agent.py:992
    """

    name: str = "_Exception"
    """Name of the tool."""
    description: str = "Exception tool"
    """Description of the tool."""

    @override
    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return query

    @override
    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return query


NextStepOutput = list[AgentFinish | AgentAction | AgentStep]
RunnableAgentType = RunnableAgent | RunnableMultiActionAgent


class AgentExecutor(Chain):
    """Orchestrates agent reasoning loop with tool execution.
    
    AgentExecutor wraps a BaseSingleActionAgent or BaseMultiActionAgent with tools,
    executing the iterative thought-action-observation loop until the agent returns
    a final answer or reaches configured limits.
    
    **Architecture:**
    The executor manages the agent lifecycle by:
    1. Invoking agent.plan() to select next action(s)
    2. Executing selected tool(s) and collecting observations
    3. Passing observations back to agent for next planning iteration
    4. Repeating until AgentFinish is returned or limits are reached
    
    **Agent Decision Loop:**
    ```mermaid
    graph TD
        A[Start: User Input] --> B[Agent Planning: plan]
        B --> C{AgentFinish?}
        C -->|Yes| D[Return Output]
        C -->|No| E[AgentAction Selected]
        E --> F[Execute Tool]
        F --> G[Observation Result]
        G --> H{Max Iterations or Time?}
        H -->|Yes| I[Early Stopping]
        H -->|No| B
        I --> D
    ```
    
    **Tool Interface Contract:**
    Tools provided to the agent must implement the BaseTool interface with:
    - name (str, required): Unique tool identifier used in agent reasoning and selection
    - description (str, required): Natural language description for agent context,
      critical for the agent to understand when and how to use the tool
    - args_schema (Pydantic BaseModel, required): Structured input validation schema
      defining expected arguments with types and constraints
    
    **Callback Integration:**
    The executor triggers callbacks at key lifecycle points:
    - on_agent_action: Called before tool execution with AgentAction
    - on_agent_finish: Called when agent completes with AgentFinish
    - on_tool_start: Called before tool.run() with tool name and inputs
    - on_tool_end: Called after tool.run() with observation
    - on_tool_error: Called when tool execution raises exception
    
    **Error Handling Patterns:**
    - Tool execution failures: ToolException caught, observation returned to agent
    - Parsing errors: Controlled by handle_parsing_errors (bool/str/Callable)
    - Max iterations: Triggers early_stopping_method ('force' or 'generate')
    - Timeout: Triggers early stopping based on max_execution_time
    
    Args:
        agent: The agent for planning actions. Can be BaseSingleActionAgent (returns
            single action per step), BaseMultiActionAgent (returns multiple actions),
            or Runnable (automatically wrapped in RunnableAgent/RunnableMultiActionAgent
            based on output type).
        tools: Valid tools the agent can call. Each tool must have unique name,
            description, and args_schema. Agent's allowed_tools (if set) must match
            provided tool names exactly.
        return_intermediate_steps: Whether to return agent trajectory
            (intermediate_steps) in final output dict. Useful for debugging and
            transparency. Defaults to False.
        max_iterations: Maximum planning steps before early stopping. Setting to None
            could lead to infinite loop if agent never returns AgentFinish. Defaults
            to 15.
        max_execution_time: Maximum wall clock time (seconds) for execution loop.
            None means no time limit. Defaults to None.
        early_stopping_method: Strategy when max_iterations or max_execution_time
            reached. 'force' returns constant message; 'generate' calls agent LLM
            one final time for answer based on trajectory. Defaults to 'force'.
        handle_parsing_errors: Strategy for OutputParserException from agent output.
            False (default) raises error. True sends error back to agent as
            observation. String value sends that string as observation. Callable
            receives exception and returns observation string.
        trim_intermediate_steps: How to trim trajectory before returning. -1 (default)
            means no trimming. Positive int keeps last N steps. Callable receives
            full trajectory and returns trimmed version.
    
    Returns:
        Final output dict with keys:
        - 'output': Agent's final answer (always present)
        - 'intermediate_steps': List of (AgentAction, observation) tuples
          (only if return_intermediate_steps=True)
    
    Raises:
        ValueError: When tools incompatible with agent.get_allowed_tools()
        ValueError: When parsing errors occur and handle_parsing_errors=False
        OutputParserException: When agent output cannot be parsed and
            handle_parsing_errors=False
    
    Example:
        ```python
        from langchain_classic.agents import AgentExecutor, create_react_agent
        from langchain_classic.tools import Tool
        from langchain_openai import ChatOpenAI
        
        # Define custom tool
        def search_tool(query: str) -> str:
            \"\"\"Search for information.\"\"\"
            return f"Results for: {query}"
        
        tools = [Tool(
            name="search",
            func=search_tool,
            description="Useful for searching information"
        )]
        
        llm = ChatOpenAI(temperature=0)
        agent = create_react_agent(llm, tools)
        
        # Create executor with configuration
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=10,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        
        # Execute agent
        result = agent_executor.invoke({"input": "What is LangChain?"})
        print(result["output"])
        print(f"Steps taken: {len(result.get('intermediate_steps', []))}")
        ```
    
    Source: libs/langchain/langchain_classic/agents/agent.py:1021
    """

    agent: BaseSingleActionAgent | BaseMultiActionAgent | Runnable
    """The agent to run for creating a plan and determining actions
    to take at each step of the execution loop."""
    tools: Sequence[BaseTool]
    """The valid tools the agent can call."""
    return_intermediate_steps: bool = False
    """Whether to return the agent's trajectory of intermediate steps
    at the end in addition to the final output."""
    max_iterations: int | None = 15
    """The maximum number of steps to take before ending the execution
    loop.

    Setting to 'None' could lead to an infinite loop."""
    max_execution_time: float | None = None
    """The maximum amount of wall clock time to spend in the execution
    loop.
    """
    early_stopping_method: str = "force"
    """The method to use for early stopping if the agent never
    returns `AgentFinish`. Either 'force' or 'generate'.

    `"force"` returns a string saying that it stopped because it met a
        time or iteration limit.

    `"generate"` calls the agent's LLM Chain one final time to generate
        a final answer based on the previous steps.
    """
    handle_parsing_errors: bool | str | Callable[[OutputParserException], str] = False
    """How to handle errors raised by the agent's output parser.
    Defaults to `False`, which raises the error.
    If `true`, the error will be sent back to the LLM as an observation.
    If a string, the string itself will be sent to the LLM as an observation.
    If a callable function, the function will be called with the exception as an
    argument, and the result of that function will be passed to the agent as an
    observation.
    """
    trim_intermediate_steps: (
        int | Callable[[list[tuple[AgentAction, str]]], list[tuple[AgentAction, str]]]
    ) = -1
    """How to trim the intermediate steps before returning them.
    Defaults to -1, which means no trimming.
    """

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: BaseSingleActionAgent | BaseMultiActionAgent | Runnable,
        tools: Sequence[BaseTool],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Create AgentExecutor from agent and tools.
        
        Factory method providing convenient construction of AgentExecutor with
        required agent and tools, plus optional configuration.

        Args:
            agent: Agent instance for planning. Can be BaseSingleActionAgent,
                BaseMultiActionAgent, or Runnable (automatically wrapped).
            tools: Sequence of BaseTool instances available to agent. Tool names
                must be unique and match agent.get_allowed_tools() if specified.
            callbacks: Optional callbacks for chain execution events.
            **kwargs: Additional AgentExecutor configuration (max_iterations,
                max_execution_time, return_intermediate_steps,
                handle_parsing_errors, etc.).

        Returns:
            Configured AgentExecutor instance ready for invocation.
        
        Example:
            ```python
            from langchain_classic.agents import AgentExecutor, create_react_agent
            from langchain_classic.tools import Tool
            
            agent = create_react_agent(llm, tools)
            executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                max_iterations=10,
                verbose=True
            )
            ```
        
        Source: libs/langchain/langchain_classic/agents/agent.py:1068
        """
        return cls(
            agent=agent,
            tools=tools,
            callbacks=callbacks,
            **kwargs,
        )

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        """Validate that tools are compatible with agent.

        Args:
            values: Values to validate.

        Returns:
            Validated values.

        Raises:
            ValueError: If allowed tools are different than provided tools.
        """
        agent = self.agent
        tools = self.tools
        allowed_tools = agent.get_allowed_tools()  # type: ignore[union-attr]
        if allowed_tools is not None and set(allowed_tools) != {
            tool.name for tool in tools
        }:
            msg = (
                f"Allowed tools ({allowed_tools}) different than "
                f"provided tools ({[tool.name for tool in tools]})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_runnable_agent(cls, values: dict) -> Any:
        """Convert runnable to agent if passed in.

        Args:
            values: Values to validate.

        Returns:
            Validated values.
        """
        agent = values.get("agent")
        if agent and isinstance(agent, Runnable):
            try:
                output_type = agent.OutputType
            except TypeError:
                multi_action = False
            except Exception:
                logger.exception("Unexpected error getting OutputType from agent")
                multi_action = False
            else:
                multi_action = output_type == list[AgentAction] | AgentFinish

            stream_runnable = values.pop("stream_runnable", True)
            if multi_action:
                values["agent"] = RunnableMultiActionAgent(
                    runnable=agent,
                    stream_runnable=stream_runnable,
                )
            else:
                values["agent"] = RunnableAgent(
                    runnable=agent,
                    stream_runnable=stream_runnable,
                )
        return values

    @property
    def _action_agent(self) -> BaseSingleActionAgent | BaseMultiActionAgent:
        """Type cast self.agent.

        If the `agent` attribute is a Runnable, it will be converted one of
        RunnableAgentType in the validate_runnable_agent root_validator.

        To support instantiating with a Runnable, here we explicitly cast the type
        to reflect the changes made in the root_validator.
        """
        if isinstance(self.agent, Runnable):
            return cast("RunnableAgentType", self.agent)
        return self.agent

    @override
    def save(self, file_path: Path | str) -> None:
        """Raise error - saving not supported for Agent Executors.

        Args:
            file_path: Path to save to.

        Raises:
            ValueError: Saving not supported for agent executors.
        """
        msg = (
            "Saving not supported for agent executors. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )
        raise ValueError(msg)

    def save_agent(self, file_path: Path | str) -> None:
        """Save the underlying agent.

        Args:
            file_path: Path to save to.
        """
        return self._action_agent.save(file_path)

    def iter(
        self,
        inputs: Any,
        callbacks: Callbacks = None,
        *,
        include_run_info: bool = False,
        async_: bool = False,  # noqa: ARG002 arg kept for backwards compat, but ignored
    ) -> AgentExecutorIterator:
        """Enables iteration over steps taken to reach final output.

        Args:
            inputs: Inputs to the agent.
            callbacks: Callbacks to run.
            include_run_info: Whether to include run info.
            async_: Whether to run async. (Ignored)

        Returns:
            Agent executor iterator object.
        """
        return AgentExecutorIterator(
            self,
            inputs,
            callbacks,
            tags=self.tags,
            include_run_info=include_run_info,
        )

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys.

        :meta private:
        """
        return self._action_agent.input_keys

    @property
    def output_keys(self) -> list[str]:
        """Return the singular output key.

        :meta private:
        """
        if self.return_intermediate_steps:
            return [*self._action_agent.return_values, "intermediate_steps"]
        return self._action_agent.return_values

    def lookup_tool(self, name: str) -> BaseTool:
        """Lookup tool by name.

        Args:
            name: Name of tool.

        Returns:
            Tool object.
        """
        return {tool.name: tool for tool in self.tools}[name]

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        return self.max_execution_time is None or time_elapsed < self.max_execution_time

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    async def _areturn(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        if run_manager:
            await run_manager.on_agent_finish(
                output,
                color="green",
                verbose=self.verbose,
            )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _consume_next_step(
        self,
        values: NextStepOutput,
    ) -> AgentFinish | list[tuple[AgentAction, str]]:
        if isinstance(values[-1], AgentFinish):
            if len(values) != 1:
                msg = "Expected a single AgentFinish output, but got multiple values."
                raise ValueError(msg)
            return values[-1]
        return [(a.action, a.observation) for a in values if isinstance(a, AgentStep)]

    def _take_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> AgentFinish | list[tuple[AgentAction, str]]:
        return self._consume_next_step(
            list(
                self._iter_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager,
                ),
            ),
        )

    def _iter_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> Iterator[AgentFinish | AgentAction | AgentStep]:
        """Take a single step in the thought-action-observation loop.
        
        Implements the core agent reasoning cycle: calls agent.plan() to get next
        action(s), executes tool(s), and yields results. Handles parsing errors
        according to handle_parsing_errors configuration.
        
        Type flow: intermediate_steps → agent.plan() → AgentAction/AgentFinish →
        tool execution → observation string → AgentStep
        
        Override this to take control of how the agent makes and acts on choices.
        
        Args:
            name_to_tool_map: Mapping of tool name to BaseTool instance for lookup.
                Constructed from self.tools at start of execution loop.
            color_mapping: Mapping of tool name to color string for logging output.
                Used to visually distinguish different tools in verbose mode.
            inputs: User inputs dict passed to agent.plan(). Contains original query
                and any additional context.
            intermediate_steps: Trajectory of (AgentAction, observation) tuples
                from previous iterations. Passed to agent for context.
            run_manager: Optional callback manager for triggering on_agent_action,
                on_tool_start, on_tool_end callbacks.
        
        Yields:
            - AgentFinish: When agent determines task is complete (ends iteration)
            - AgentAction: Action(s) to execute (before tool execution)
            - AgentStep: Result after tool execution (action + observation)
        
        Raises:
            ValueError: When OutputParserException occurs and handle_parsing_errors
                is False or invalid type
        
        Source: libs/langchain/langchain_classic/agents/agent.py:1316
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Type flow: intermediate_steps (list[tuple[AgentAction, str]]) 
            # → agent.plan() → output (AgentAction | AgentFinish)
            # Call the LLM to see what to do.
            output = self._action_agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                msg = (
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {e!s}"
                )
                raise ValueError(msg) from e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                msg = "Got unexpected type of `handle_parsing_errors`"  # type: ignore[unreachable]
                raise ValueError(msg) from e  # noqa: TRY004
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: list[AgentAction]
        actions = [output] if isinstance(output, AgentAction) else output
        for agent_action in actions:
            yield agent_action
        for agent_action in actions:
            yield self._perform_agent_action(
                name_to_tool_map,
                color_mapping,
                agent_action,
                run_manager,
            )

    def _perform_agent_action(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        agent_action: AgentAction,
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> AgentStep:
        """Execute a single agent action and return observation.
        
        Looks up requested tool, executes it with agent_action.tool_input, and
        wraps result in AgentStep. If tool not found, executes InvalidTool with
        available tool names.
        
        Args:
            name_to_tool_map: Mapping of tool name to BaseTool instance.
            color_mapping: Mapping of tool name to color for logging.
            agent_action: Action selected by agent containing tool name and input.
            run_manager: Optional callback manager for on_agent_action callback.
        
        Returns:
            AgentStep containing the executed action and observation string from
            tool execution.
        
        Source: libs/langchain/langchain_classic/agents/agent.py:1395
        """
        if run_manager:
            run_manager.on_agent_action(agent_action, color="green")
        # Otherwise we lookup the tool
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # Type flow: AgentAction.tool_input (str | dict) 
            # → tool.run() → observation (str)
            # We then call the tool on the tool input to get an observation
            observation = tool.run(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = InvalidTool().run(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    async def _atake_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> AgentFinish | list[tuple[AgentAction, str]]:
        return self._consume_next_step(
            [
                a
                async for a in self._aiter_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager,
                )
            ],
        )

    async def _aiter_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> AsyncIterator[AgentFinish | AgentAction | AgentStep]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = await self._action_agent.aplan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                msg = (
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {e!s}"
                )
                raise ValueError(msg) from e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                msg = "Got unexpected type of `handle_parsing_errors`"  # type: ignore[unreachable]
                raise ValueError(msg) from e  # noqa: TRY004
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: list[AgentAction]
        actions = [output] if isinstance(output, AgentAction) else output
        for agent_action in actions:
            yield agent_action

        # Use asyncio.gather to run multiple tool.arun() calls concurrently
        result = await asyncio.gather(
            *[
                self._aperform_agent_action(
                    name_to_tool_map,
                    color_mapping,
                    agent_action,
                    run_manager,
                )
                for agent_action in actions
            ],
        )

        # TODO: This could yield each result as it becomes available
        for chunk in result:
            yield chunk

    async def _aperform_agent_action(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        agent_action: AgentAction,
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> AgentStep:
        if run_manager:
            await run_manager.on_agent_action(
                agent_action,
                verbose=self.verbose,
                color="green",
            )
        # Otherwise we lookup the tool
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            observation = await tool.arun(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = await InvalidTool().arun(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Run text through and get agent response.
        
        Executes the agent reasoning loop synchronously, iterating through
        thought-action-observation cycles until the agent returns a final answer
        or configured limits are reached.
        
        Type flow: Dict[str,str] inputs → agent planning loop → tool observations
        → Dict[str,Any] final output
        
        Args:
            inputs: User inputs as dict with string keys and values. Must contain
                keys required by agent.input_keys. Common keys include 'input',
                'question', or 'query' depending on agent type.
            run_manager: Optional callback manager for chain run. Used to trigger
                on_agent_action, on_agent_finish, and tool lifecycle callbacks.
        
        Returns:
            Dict containing:
            - 'output' (str): Agent's final answer
            - 'intermediate_steps' (list[tuple[AgentAction, str]]): Trajectory of
              actions and observations (only if return_intermediate_steps=True)
        
        Raises:
            ValueError: When max_iterations or max_execution_time limit reached
                and early_stopping_method fails
            OutputParserException: When agent output cannot be parsed and
                handle_parsing_errors=False
        
        Source: libs/langchain/langchain_classic/agents/agent.py:1585
        """
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools],
            excluded_colors=["green", "red"],
        )
        intermediate_steps: list[tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output,
                    intermediate_steps,
                    run_manager=run_manager,
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self._action_agent.return_stopped_response(
            self.early_stopping_method,
            intermediate_steps,
            **inputs,
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    async def _acall(
        self,
        inputs: dict[str, str],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """Async run text through and get agent response.
        
        Executes the agent reasoning loop asynchronously with async/await support,
        iterating through thought-action-observation cycles until the agent returns
        a final answer or configured limits are reached.
        
        Lifecycle identical to _call but with async tool execution and proper
        timeout handling via asyncio_timeout context manager.
        
        Type flow: Dict[str,str] inputs → async agent planning loop → async tool
        observations → Dict[str,Any] final output
        
        Args:
            inputs: User inputs as dict with string keys and values. Must contain
                keys required by agent.input_keys. Common keys include 'input',
                'question', or 'query' depending on agent type.
            run_manager: Optional async callback manager for chain run. Used to
                trigger async on_agent_action, on_agent_finish, and tool lifecycle
                callbacks in async context.
        
        Returns:
            Dict containing:
            - 'output' (str): Agent's final answer
            - 'intermediate_steps' (list[tuple[AgentAction, str]]): Trajectory of
              actions and observations (only if return_intermediate_steps=True)
        
        Raises:
            TimeoutError: When max_execution_time exceeded (asyncio timeout)
            ValueError: When max_iterations limit reached and early_stopping_method
                fails
            OutputParserException: When agent output cannot be parsed and
                handle_parsing_errors=False
        
        Source: libs/langchain/langchain_classic/agents/agent.py:1639
        """
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools],
            excluded_colors=["green"],
        )
        intermediate_steps: list[tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        try:
            async with asyncio_timeout(self.max_execution_time):
                while self._should_continue(iterations, time_elapsed):
                    next_step_output = await self._atake_next_step(
                        name_to_tool_map,
                        color_mapping,
                        inputs,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
                    if isinstance(next_step_output, AgentFinish):
                        return await self._areturn(
                            next_step_output,
                            intermediate_steps,
                            run_manager=run_manager,
                        )

                    intermediate_steps.extend(next_step_output)
                    if len(next_step_output) == 1:
                        next_step_action = next_step_output[0]
                        # See if tool should return directly
                        tool_return = self._get_tool_return(next_step_action)
                        if tool_return is not None:
                            return await self._areturn(
                                tool_return,
                                intermediate_steps,
                                run_manager=run_manager,
                            )

                    iterations += 1
                    time_elapsed = time.time() - start_time
                output = self._action_agent.return_stopped_response(
                    self.early_stopping_method,
                    intermediate_steps,
                    **inputs,
                )
                return await self._areturn(
                    output,
                    intermediate_steps,
                    run_manager=run_manager,
                )
        except (TimeoutError, asyncio.TimeoutError):
            # stop early when interrupted by the async timeout
            output = self._action_agent.return_stopped_response(
                self.early_stopping_method,
                intermediate_steps,
                **inputs,
            )
            return await self._areturn(
                output,
                intermediate_steps,
                run_manager=run_manager,
            )

    def _get_tool_return(
        self,
        next_step_output: tuple[AgentAction, str],
    ) -> AgentFinish | None:
        """Check if the tool is a returning tool."""
        agent_action, observation = next_step_output
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        return_value_key = "output"
        if len(self._action_agent.return_values) > 0:
            return_value_key = self._action_agent.return_values[0]
        # Invalid tools won't be in the map, so we return False.
        if (
            agent_action.tool in name_to_tool_map
            and name_to_tool_map[agent_action.tool].return_direct
        ):
            return AgentFinish(
                {return_value_key: observation},
                "",
            )
        return None

    def _prepare_intermediate_steps(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
    ) -> list[tuple[AgentAction, str]]:
        if (
            isinstance(self.trim_intermediate_steps, int)
            and self.trim_intermediate_steps > 0
        ):
            return intermediate_steps[-self.trim_intermediate_steps :]
        if callable(self.trim_intermediate_steps):
            return self.trim_intermediate_steps(intermediate_steps)
        return intermediate_steps

    @override
    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[AddableDict]:
        """Enables streaming over steps taken to reach final output.

        Args:
            input: Input to the agent.
            config: Config to use.
            kwargs: Additional arguments.

        Yields:
            Addable dictionary.
        """
        config = ensure_config(config)
        iterator = AgentExecutorIterator(
            self,
            input,
            config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.get("run_id"),
            yield_actions=True,
            **kwargs,
        )
        yield from iterator

    @override
    async def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AddableDict]:
        """Async enables streaming over steps taken to reach final output.

        Args:
            input: Input to the agent.
            config: Config to use.
            kwargs: Additional arguments.

        Yields:
            Addable dictionary.
        """
        config = ensure_config(config)
        iterator = AgentExecutorIterator(
            self,
            input,
            config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.get("run_id"),
            yield_actions=True,
            **kwargs,
        )
        async for step in iterator:
            yield step
