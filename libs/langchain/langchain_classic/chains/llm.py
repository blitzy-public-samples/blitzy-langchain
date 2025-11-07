"""Chain that just formats a prompt and calls an LLM."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, cast

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.language_models import (
    BaseLanguageModel,
    LanguageModelInput,
)
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseLLMOutputParser, StrOutputParser
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableBranch,
    RunnableWithFallbacks,
)
from langchain_core.runnables.configurable import DynamicRunnable
from langchain_core.utils.input import get_colored_text
from pydantic import ConfigDict, Field
from typing_extensions import override

from langchain_classic.chains.base import Chain


@deprecated(
    since="0.1.17",
    alternative="RunnableSequence, e.g., `prompt | llm`",
    removal="1.0",
)
class LLMChain(Chain):
    """Chain to run queries against LLMs.

    ⚠️ **DEPRECATION WARNING**: This class is deprecated and will be removed in
    version 1.0. Use LCEL (LangChain Expression Language) composition instead.

    **Why LCEL is Better**:
    - **Type Safety**: LCEL provides explicit type annotations and better IDE support
    - **Easier Composition**: Natural pipe operator (|) for intuitive chain building
    - **Streaming Support**: First-class support for streaming responses
    - **Async-First Design**: Built with async/await patterns from the ground up
    - **Better Error Messages**: Clear type mismatches and composition errors

    **Migration Guide - Side-by-Side Comparison**:

    1. **Simple Prompt + LLM Composition**:
       ```python
       # DEPRECATED: LLMChain approach
       from langchain_classic.chains import LLMChain
       from langchain_core.prompts import PromptTemplate
       from langchain_openai import OpenAI

       prompt = PromptTemplate.from_template("Tell me a {adjective} joke")
       chain = LLMChain(llm=OpenAI(), prompt=prompt)
       result = chain.invoke({"adjective": "funny"})["text"]

       # MODERN: LCEL approach (RECOMMENDED)
       from langchain_core.prompts import PromptTemplate
       from langchain_openai import OpenAI

       prompt = PromptTemplate.from_template("Tell me a {adjective} joke")
       chain = prompt | OpenAI()
       result = chain.invoke({"adjective": "funny"})
       ```

    2. **With Output Parser**:
       ```python
       # DEPRECATED: LLMChain with output parser
       from langchain_classic.chains import LLMChain
       from langchain_core.output_parsers import StrOutputParser

       chain = LLMChain(
           llm=OpenAI(),
           prompt=prompt,
           output_parser=StrOutputParser()
       )

       # MODERN: LCEL with explicit parser (RECOMMENDED)
       from langchain_core.output_parsers import StrOutputParser

       chain = prompt | OpenAI() | StrOutputParser()
       ```

    3. **With Memory (Conversational)**:
       ```python
       # DEPRECATED: LLMChain with memory
       from langchain_classic.chains import LLMChain
       from langchain_classic.memory import ConversationBufferMemory

       memory = ConversationBufferMemory()
       chain = LLMChain(llm=OpenAI(), prompt=prompt, memory=memory)

       # MODERN: LCEL with manual memory management (RECOMMENDED)
       from langchain_core.runnables import RunnablePassthrough

       def load_memory(input_dict):
           input_dict["history"] = memory.load_memory_variables({})
           return input_dict

       chain = RunnablePassthrough.assign(history=load_memory) | prompt | OpenAI()
       ```

    **Type Flow Documentation**:
    The complete execution flow through LLMChain:
        Dict[input_vars] → prep_inputs (validate)
        → prompt.format_prompt → PromptValue
        → llm.generate_prompt/batch → LLMResult
        → output_parser.parse_result → Dict[output_key: str]

    **Callback Integration**:
        on_chain_start → on_llm_start → on_llm_new_token (optional)
        → on_llm_end → on_chain_end

    Legacy Example (for reference only):
        ```python
        from langchain_classic.chains import LLMChain
        from langchain_community.llms import OpenAI
        from langchain_core.prompts import PromptTemplate

        prompt_template = "Tell me a {adjective} joke"
        prompt = PromptTemplate(input_variables=["adjective"], template=prompt_template)
        model = LLMChain(llm=OpenAI(), prompt=prompt)
        ```
    """

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: Runnable[LanguageModelInput, str] | Runnable[LanguageModelInput, BaseMessage]
    """Language model to call.

    Accepts Union type with the following options:
    - Runnable[LanguageModelInput, str]: For completion models that return strings
    - Runnable[LanguageModelInput, BaseMessage]: For chat models that return messages

    Where LanguageModelInput can be:
    - str: Plain text input
    - List[BaseMessage]: Sequence of chat messages (HumanMessage, AIMessage, etc.)
    - PromptValue: Formatted prompt value from prompt templates

    Legacy BaseLanguageModel instances are also supported for backwards compatibility.

    Implementation Requirements:
    - Must implement either generate_prompt() method (BaseLanguageModel interface)
      OR batch() method (Runnable interface)
    - For Runnable implementations, batch() will be called with prompts and config
    - Results are normalized to LLMResult format regardless of implementation type

    Source: libs/core/langchain_core/language_models/base.py:98
    """
    output_key: str = "text"  #: :meta private:
    output_parser: BaseLLMOutputParser = Field(default_factory=StrOutputParser)
    """Output parser to use.
    Defaults to one that takes the most likely string but does not change it
    otherwise."""
    return_final_only: bool = True
    """Whether to return only the final parsed result.
    If `False`, will return a bunch of extra information about the generation."""
    llm_kwargs: dict = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> list[str]:
        """Will always return text key.

        :meta private:
        """
        if self.return_final_only:
            return [self.output_key]
        return [self.output_key, "full_generation"]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        # Type flow: Dict[str, Any] → generate() batches single input
        # → LLMResult with generations → create_outputs() → Dict[output_key: str]
        response = self.generate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def generate(
        self,
        input_list: list[dict[str, Any]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        callbacks = run_manager.get_child() if run_manager else None
        
        # Code path 1: Legacy BaseLanguageModel interface
        # Uses generate_prompt() which directly returns LLMResult
        if isinstance(self.llm, BaseLanguageModel):
            return self.llm.generate_prompt(
                prompts,
                stop,
                callbacks=callbacks,
                **self.llm_kwargs,
            )
        
        # Code path 2: Modern Runnable interface
        # Uses batch() and normalizes results to LLMResult format
        # Type flow: List[PromptValue] → llm.batch() → List[str | BaseMessage]
        # → normalize to LLMResult(generations=List[List[Generation]])
        results = self.llm.bind(stop=stop, **self.llm_kwargs).batch(
            cast("list", prompts),
            {"callbacks": callbacks},
        )
        generations: list[list[Generation]] = []
        for res in results:
            # Normalize BaseMessage responses to ChatGeneration
            if isinstance(res, BaseMessage):
                generations.append([ChatGeneration(message=res)])
            # Normalize string responses to Generation
            else:
                generations.append([Generation(text=res)])
        return LLMResult(generations=generations)

    async def agenerate(
        self,
        input_list: list[dict[str, Any]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
        callbacks = run_manager.get_child() if run_manager else None
        if isinstance(self.llm, BaseLanguageModel):
            return await self.llm.agenerate_prompt(
                prompts,
                stop,
                callbacks=callbacks,
                **self.llm_kwargs,
            )
        results = await self.llm.bind(stop=stop, **self.llm_kwargs).abatch(
            cast("list", prompts),
            {"callbacks": callbacks},
        )
        generations: list[list[Generation]] = []
        for res in results:
            if isinstance(res, BaseMessage):
                generations.append([ChatGeneration(message=res)])
            else:
                generations.append([Generation(text=res)])
        return LLMResult(generations=generations)

    def prep_prompts(
        self,
        input_list: list[dict[str, Any]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> tuple[list[PromptValue], list[str] | None]:
        """Prepare prompts from inputs."""
        stop = None
        if len(input_list) == 0:
            return [], stop
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            # Type flow: Dict[str, Any] → prompt.format_prompt() → PromptValue
            # PromptValue is the standardized format that can be passed to any LLM
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if run_manager:
                run_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                msg = "If `stop` is present in any inputs, should be present in all."
                raise ValueError(msg)
            prompts.append(prompt)
        return prompts, stop

    async def aprep_prompts(
        self,
        input_list: list[dict[str, Any]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> tuple[list[PromptValue], list[str] | None]:
        """Prepare prompts from inputs."""
        stop = None
        if len(input_list) == 0:
            return [], stop
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if run_manager:
                await run_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                msg = "If `stop` is present in any inputs, should be present in all."
                raise ValueError(msg)
            prompts.append(prompt)
        return prompts, stop

    def apply(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> list[dict[str, str]]:
        """Utilize the LLM generate method for batch processing speed gains.

        Process multiple inputs in a single batch, which is more efficient than
        calling predict() multiple times. The LLM's generate() method is called
        once with all inputs, reducing network overhead and taking advantage of
        batch processing optimizations.

        Args:
            input_list: List of input dictionaries where each dict contains the
                prompt input variables. Keys in each dict must match the
                prompt.input_variables. Example: [{"adjective": "funny"},
                {"adjective": "sad"}]
            callbacks: Optional callback handlers or callback manager for chain
                execution tracing. These callbacks will receive events like
                on_chain_start, on_llm_start, on_llm_end, and on_chain_end.

        Returns:
            List of output dictionaries where each dict contains:
            - self.output_key (default "text"): The parsed LLM output string
            If return_final_only is False, each dict also contains:
            - "full_generation": The complete Generation object with metadata

        Raises:
            KeyError: If any input dict is missing required prompt input_variables
            ValueError: If prompt formatting fails (e.g., inconsistent "stop" values)
            ValidationError: If Pydantic validation fails on prompt inputs
            LLMError: Various LLM-specific errors (rate limits, API errors, etc.)

        Type Flow:
            List[Dict] → List[PromptValue] → llm.generate()
            → LLMResult → List[Dict[output_key: str]]

        Batch Processing Advantage:
            This method makes a single LLM.generate() call for all inputs, which is
            significantly more efficient than calling predict() in a loop. Benefits:
            - Reduced network latency (one request vs N requests)
            - Lower API overhead
            - Potential cost savings with batch pricing
            - Better throughput for high-volume processing

        Example:
            ```python
            from langchain_classic.chains import LLMChain
            from langchain_core.prompts import PromptTemplate
            from langchain_openai import OpenAI

            prompt = PromptTemplate.from_template("Tell me a {adjective} joke")
            chain = LLMChain(llm=OpenAI(), prompt=prompt)

            inputs = [
                {"adjective": "funny"},
                {"adjective": "sad"},
                {"adjective": "clever"}
            ]
            results = chain.apply(inputs)
            # Returns: [
            #     {"text": "Why did the... [funny joke]"},
            #     {"text": "What do you... [sad joke]"},
            #     {"text": "A photon... [clever joke]"}
            # ]
            ```

        Source: libs/langchain/langchain_classic/chains/llm.py:230
        """
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
        )
        run_manager = callback_manager.on_chain_start(
            None,
            {"input_list": input_list},
            name=self.get_name(),
        )
        try:
            response = self.generate(input_list, run_manager=run_manager)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        outputs = self.create_outputs(response)
        run_manager.on_chain_end({"outputs": outputs})
        return outputs

    async def aapply(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> list[dict[str, str]]:
        """Async version: Utilize the LLM generate method for batch processing speed gains.

        Asynchronous variant of apply() that processes multiple inputs in a single
        batch. This method must be awaited and runs in an async event loop context.
        Provides the same batch processing advantages as apply() while supporting
        concurrent execution patterns.

        Args:
            input_list: List of input dictionaries where each dict contains the
                prompt input variables. Keys in each dict must match the
                prompt.input_variables. Example: [{"adjective": "funny"},
                {"adjective": "sad"}]
            callbacks: Optional callback handlers or callback manager for chain
                execution tracing. Async callbacks will be properly awaited during
                execution (on_chain_start, on_llm_start, on_llm_end, on_chain_end).

        Returns:
            List of output dictionaries where each dict contains:
            - self.output_key (default "text"): The parsed LLM output string
            If return_final_only is False, each dict also contains:
            - "full_generation": The complete Generation object with metadata

        Raises:
            KeyError: If any input dict is missing required prompt input_variables
            ValueError: If prompt formatting fails (e.g., inconsistent "stop" values)
            ValidationError: If Pydantic validation fails on prompt inputs
            LLMError: Various LLM-specific errors (rate limits, API errors, etc.)

        Async Execution Notes:
            - Requires await: Must be called with await keyword
            - Event Loop: Runs in the current async event loop context
            - Callback Handling: All callback methods are awaited if they are async
            - Concurrency: For concurrent batch processing, consider asyncio.gather()
              with multiple aapply() calls

        Type Flow:
            List[Dict] → List[PromptValue] → llm.agenerate()
            → LLMResult → List[Dict[output_key: str]]

        Example:
            ```python
            import asyncio
            from langchain_classic.chains import LLMChain
            from langchain_core.prompts import PromptTemplate
            from langchain_openai import OpenAI

            async def process_batch():
                prompt = PromptTemplate.from_template("Tell me a {adjective} joke")
                chain = LLMChain(llm=OpenAI(), prompt=prompt)

                inputs = [
                    {"adjective": "funny"},
                    {"adjective": "sad"},
                    {"adjective": "clever"}
                ]
                results = await chain.aapply(inputs)
                return results

            # Run in event loop
            results = asyncio.run(process_batch())
            ```

        Source: libs/langchain/langchain_classic/chains/llm.py:255
        """
        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
        )
        run_manager = await callback_manager.on_chain_start(
            None,
            {"input_list": input_list},
            name=self.get_name(),
        )
        try:
            response = await self.agenerate(input_list, run_manager=run_manager)
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        outputs = self.create_outputs(response)
        await run_manager.on_chain_end({"outputs": outputs})
        return outputs

    @property
    def _run_output_key(self) -> str:
        return self.output_key

    def create_outputs(self, llm_result: LLMResult) -> list[dict[str, Any]]:
        """Create outputs from response."""
        result = [
            # Get the text of the top generated string.
            {
                self.output_key: self.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result.generations
        ]
        if self.return_final_only:
            result = [{self.output_key: r[self.output_key]} for r in result]
        return result

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        response = await self.agenerate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def predict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.

        Convenience method that formats the prompt template with provided keyword
        arguments and returns just the LLM output string (not the full dict).

        Args:
            callbacks: Optional callback handlers (or callback manager) for chain
                execution tracing. Callbacks receive events throughout the chain
                lifecycle: on_chain_start, on_llm_start, on_llm_new_token (if
                streaming), on_llm_end, and on_chain_end.
            **kwargs: Keyword arguments where keys must match prompt.input_variables
                exactly. Values are typically strings but can be any type accepted
                by the prompt template. Example: adjective="funny", topic="cats"

        Returns:
            Parsed LLM output string. Specifically returns the value of
            self.output_key (default "text") from the result dictionary. The
            output_parser is applied to the raw LLM response before returning.

        Raises:
            KeyError: If kwargs is missing any required prompt input_variables
            ValidationError: If prompt template formatting fails (e.g., invalid
                template variables or formatting syntax errors)
            LLMError: Various LLM-specific errors including rate limiting,
                authentication failures, or API errors

        Type Flow:
            kwargs → prompt.format() → PromptValue → llm.generate()
            → LLMResult → output_parser.parse() → str

        Example:
            ```python
            from langchain_classic.chains import LLMChain
            from langchain_core.prompts import PromptTemplate
            from langchain_openai import OpenAI

            prompt = PromptTemplate(
                input_variables=["adjective", "topic"],
                template="Tell me a {adjective} joke about {topic}"
            )
            chain = LLMChain(llm=OpenAI(), prompt=prompt)

            # Direct keyword arguments matching input_variables
            result = chain.predict(adjective="funny", topic="programming")
            # Returns: "Why do programmers prefer dark mode?..."
            ```

        Source: libs/langchain/langchain_classic/chains/llm.py:306
        """
        return self(kwargs, callbacks=callbacks)[self.output_key]

    async def apredict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Async version: Format prompt with kwargs and pass to LLM.

        Asynchronous variant of predict() that must be awaited. Formats the prompt
        template with provided keyword arguments and returns just the LLM output
        string. Use this method when working in async contexts to avoid blocking.

        Args:
            callbacks: Optional callback handlers (or callback manager) for chain
                execution tracing. Async callbacks will be properly awaited during
                execution. Events include: on_chain_start, on_llm_start,
                on_llm_new_token (if streaming), on_llm_end, and on_chain_end.
            **kwargs: Keyword arguments where keys must match prompt.input_variables
                exactly. Values are typically strings but can be any type accepted
                by the prompt template. Example: adjective="funny", topic="cats"

        Returns:
            Parsed LLM output string. Specifically returns the value of
            self.output_key (default "text") from the result dictionary. The
            output_parser is applied to the raw LLM response before returning.

        Raises:
            KeyError: If kwargs is missing any required prompt input_variables
            ValidationError: If prompt template formatting fails (e.g., invalid
                template variables or formatting syntax errors)
            LLMError: Various LLM-specific errors including rate limiting,
                authentication failures, or API errors

        Async Execution Notes:
            - Requires await: Must be called with await keyword in async function
            - Event Loop: Executes in the current async event loop context
            - Callback Handling: All callback methods are properly awaited if async
            - Non-Blocking: Does not block the event loop during LLM API calls

        Type Flow:
            kwargs → prompt.format() → PromptValue → llm.agenerate()
            → LLMResult → output_parser.parse() → str

        Example:
            ```python
            import asyncio
            from langchain_classic.chains import LLMChain
            from langchain_core.prompts import PromptTemplate
            from langchain_openai import OpenAI

            async def get_joke():
                prompt = PromptTemplate(
                    input_variables=["adjective", "topic"],
                    template="Tell me a {adjective} joke about {topic}"
                )
                chain = LLMChain(llm=OpenAI(), prompt=prompt)

                # Must await the async method
                result = await chain.apredict(
                    adjective="funny",
                    topic="programming"
                )
                return result

            # Run in event loop
            joke = asyncio.run(get_joke())
            # Returns: "Why do programmers prefer dark mode?..."
            ```

        Source: libs/langchain/langchain_classic/chains/llm.py:323
        """
        return (await self.acall(kwargs, callbacks=callbacks))[self.output_key]

    def predict_and_parse(
        self,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> str | list[str] | dict[str, Any]:
        """Call predict and then parse the results."""
        warnings.warn(
            "The predict_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain.",
            stacklevel=2,
        )
        result = self.predict(callbacks=callbacks, **kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        return result

    async def apredict_and_parse(
        self,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> str | list[str] | dict[str, str]:
        """Call apredict and then parse the results."""
        warnings.warn(
            "The apredict_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain.",
            stacklevel=2,
        )
        result = await self.apredict(callbacks=callbacks, **kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        return result

    def apply_and_parse(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> Sequence[str | list[str] | dict[str, str]]:
        """Call apply and then parse the results."""
        warnings.warn(
            "The apply_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain.",
            stacklevel=2,
        )
        result = self.apply(input_list, callbacks=callbacks)
        return self._parse_generation(result)

    def _parse_generation(
        self,
        generation: list[dict[str, str]],
    ) -> Sequence[str | list[str] | dict[str, str]]:
        if self.prompt.output_parser is not None:
            return [
                self.prompt.output_parser.parse(res[self.output_key])
                for res in generation
            ]
        return generation

    async def aapply_and_parse(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> Sequence[str | list[str] | dict[str, str]]:
        """Call apply and then parse the results."""
        warnings.warn(
            "The aapply_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain.",
            stacklevel=2,
        )
        result = await self.aapply(input_list, callbacks=callbacks)
        return self._parse_generation(result)

    @property
    def _chain_type(self) -> str:
        return "llm_chain"

    @classmethod
    def from_string(cls, llm: BaseLanguageModel, template: str) -> LLMChain:
        """Create LLMChain from LLM and template."""
        prompt_template = PromptTemplate.from_template(template)
        return cls(llm=llm, prompt=prompt_template)

    def _get_num_tokens(self, text: str) -> int:
        return _get_language_model(self.llm).get_num_tokens(text)


def _get_language_model(llm_like: Runnable) -> BaseLanguageModel:
    if isinstance(llm_like, BaseLanguageModel):
        return llm_like
    if isinstance(llm_like, RunnableBinding):
        return _get_language_model(llm_like.bound)
    if isinstance(llm_like, RunnableWithFallbacks):
        return _get_language_model(llm_like.runnable)
    if isinstance(llm_like, (RunnableBranch, DynamicRunnable)):
        return _get_language_model(llm_like.default)
    msg = (
        f"Unable to extract BaseLanguageModel from llm_like object of type "
        f"{type(llm_like)}"
    )
    raise ValueError(msg)
