"""Base class for prompt templates."""

from __future__ import annotations

import contextlib
import json
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core.exceptions import ErrorCode, create_message
from langchain_core.load import dumpd
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompt_values import (
    ChatPromptValueConcrete,
    PromptValue,
    StringPromptValue,
)
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from langchain_core.documents import Document


FormatOutputType = TypeVar("FormatOutputType")


class BasePromptTemplate(
    RunnableSerializable[dict, PromptValue], ABC, Generic[FormatOutputType]
):
    """Base class for all prompt templates, returning a prompt.
    
    BasePromptTemplate serves as the abstract foundation for all prompt template
    implementations in LangChain. It integrates with the Runnable protocol to enable
    LCEL (LangChain Expression Language) composition, allowing prompt templates to be
    seamlessly chained with language models, output parsers, and other components.
    
    As a Runnable, BasePromptTemplate accepts a dictionary of input variables and
    produces a PromptValue, which can be converted to either string format for LLMs
    or message format for chat models.
    
    Key capabilities:
    - Variable substitution with validation (input_variables)
    - Partial variable application for reusable templates (partial_variables)
    - Optional variables for flexible template composition (optional_variables)
    - Output parsing integration (output_parser)
    - Metadata and tag propagation for tracing (metadata, tags)
    - Synchronous and asynchronous execution (invoke/ainvoke)
    - Serialization and persistence (save/load)
    
    Type Flow in LCEL Composition:
        dict[str, Any] → BasePromptTemplate → PromptValue → LanguageModel
    
    Subclasses must implement:
        - format(): Convert inputs to FormatOutputType (str or list[BaseMessage])
        - format_prompt(): Convert inputs to PromptValue wrapper
        - _prompt_type property: Unique identifier for serialization
    
    Source: libs/core/langchain_core/prompts/base.py:42-44
    """

    input_variables: list[str]
    """A list of the names of the variables whose values are required as inputs to the
    prompt."""
    optional_variables: list[str] = Field(default=[])
    """optional_variables: A list of the names of the variables for placeholder
       or MessagePlaceholder that are optional. These variables are auto inferred
       from the prompt and user need not provide them."""
    input_types: typing.Dict[str, Any] = Field(default_factory=dict, exclude=True)  # noqa: UP006
    """A dictionary of the types of the variables the prompt template expects.
    If not provided, all variables are assumed to be strings."""
    output_parser: BaseOutputParser | None = None
    """How to parse the output of calling an LLM on this formatted prompt."""
    partial_variables: Mapping[str, Any] = Field(default_factory=dict)
    """A dictionary of the partial variables the prompt template carries.

    Partial variables populate the template so that you don't need to
    pass them in every time you call the prompt."""
    metadata: typing.Dict[str, Any] | None = None  # noqa: UP006
    """Metadata to be used for tracing."""
    tags: list[str] | None = None
    """Tags to be used for tracing."""

    @model_validator(mode="after")
    def validate_variable_names(self) -> Self:
        """Validate variable names do not include restricted names."""
        if "stop" in self.input_variables:
            msg = (
                "Cannot have an input variable named 'stop', as it is used internally,"
                " please rename."
            )
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )
        if "stop" in self.partial_variables:
            msg = (
                "Cannot have an partial variable named 'stop', as it is used "
                "internally, please rename."
            )
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )

        overall = set(self.input_variables).intersection(self.partial_variables)
        if overall:
            msg = f"Found overlapping input and partial variables: {overall}"
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )
        return self

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "prompt_template"]`
        """
        return ["langchain", "schema", "prompt_template"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return True as this class is serializable."""
        return True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @cached_property
    def _serialized(self) -> dict[str, Any]:
        return dumpd(self)

    @property
    @override
    def OutputType(self) -> Any:
        """Return the output type of the prompt."""
        return StringPromptValue | ChatPromptValueConcrete

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """Get the input schema for the prompt.

        Args:
            config: configuration for the prompt.

        Returns:
            The input schema for the prompt.
        """
        # This is correct, but pydantic typings/mypy don't think so.
        required_input_variables = {
            k: (self.input_types.get(k, str), ...) for k in self.input_variables
        }
        optional_input_variables = {
            k: (self.input_types.get(k, str), None) for k in self.optional_variables
        }
        return create_model_v2(
            "PromptInput",
            field_definitions={**required_input_variables, **optional_input_variables},
        )

    def _validate_input(self, inner_input: Any) -> dict:
        """Validate and normalize input to prompt template.
        
        This method handles two input patterns:
        1. Single-arg shorthand: If the prompt has exactly one input variable,
           a non-dict value is automatically wrapped as {variable_name: value}
        2. Mapping requirement: If the prompt has multiple variables, input must
           be a dictionary mapping variable names to values
        
        Args:
            inner_input: Raw input to the prompt template. Can be:
                - dict: Mapping of variable names to values (always accepted)
                - Any other type: Only accepted if prompt has single input variable
        
        Returns:
            Dictionary mapping all required input variable names to their values.
            Guaranteed to contain all keys from self.input_variables.
        
        Raises:
            TypeError: If input is not a dict and prompt has multiple variables.
                Error code: ErrorCode.INVALID_PROMPT_INPUT
            KeyError: If required input variables are missing from the input dict.
                Error code: ErrorCode.INVALID_PROMPT_INPUT
                Includes helpful escape hint for variables intended as literal text.
        
        Example:
            ```python
            # Single variable prompt - shorthand accepted
            template = PromptTemplate(template="Hello {name}", input_variables=["name"])
            template._validate_input("World")  # Returns: {"name": "World"}
            
            # Multiple variable prompt - dict required
            template = PromptTemplate(
                template="Hello {name}, you are {age} years old",
                input_variables=["name", "age"]
            )
            template._validate_input({"name": "Alice", "age": 30})  # Valid
            template._validate_input("Alice")  # Raises TypeError
            ```
        
        Source: libs/core/langchain_core/prompts/base.py:147-179
        """
        if not isinstance(inner_input, dict):
            if len(self.input_variables) == 1:
                var_name = self.input_variables[0]
                inner_input = {var_name: inner_input}

            else:
                msg = (
                    f"Expected mapping type as input to {self.__class__.__name__}. "
                    f"Received {type(inner_input)}."
                )
                raise TypeError(
                    create_message(
                        message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT
                    )
                )
        missing = set(self.input_variables).difference(inner_input)
        if missing:
            msg = (
                f"Input to {self.__class__.__name__} is missing variables {missing}. "
                f" Expected: {self.input_variables}"
                f" Received: {list(inner_input.keys())}"
            )
            example_key = missing.pop()
            msg += (
                f"\nNote: if you intended {{{example_key}}} to be part of the string"
                " and not a variable, please escape it with double curly braces like: "
                f"'{{{{{example_key}}}}}'."
            )
            raise KeyError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )
        return inner_input

    def _format_prompt_with_error_handling(self, inner_input: dict) -> PromptValue:
        inner_input_ = self._validate_input(inner_input)
        return self.format_prompt(**inner_input_)

    async def _aformat_prompt_with_error_handling(
        self, inner_input: dict
    ) -> PromptValue:
        inner_input_ = self._validate_input(inner_input)
        return await self.aformat_prompt(**inner_input_)

    @override
    def invoke(
        self, input: dict, config: RunnableConfig | None = None, **kwargs: Any
    ) -> PromptValue:
        """Invoke the prompt template with input variables to produce a PromptValue.
        
        This is the primary synchronous entry point for prompt execution in LCEL chains.
        It handles input validation, metadata/tag propagation, callback orchestration,
        and produces a PromptValue that can be passed to language models.

        Args:
            input: Dictionary mapping input variable names to their values.
                Must contain all keys specified in self.input_variables.
                For single-variable prompts, non-dict values are auto-wrapped.
            config: Runnable configuration controlling execution behavior.
                Key configuration options:
                - callbacks: List of callback handlers for tracing/logging
                - metadata: Dict merged with prompt's self.metadata
                - tags: List extended with prompt's self.tags
                - run_name: Custom name for this execution in traces
                If None, default configuration is used.
            **kwargs: Additional keyword arguments (currently unused, reserved for
                future extensibility).

        Returns:
            PromptValue wrapping the formatted prompt. The PromptValue can be:
            - StringPromptValue: For text-based prompts (LLM input)
            - ChatPromptValueConcrete: For message-based prompts (chat model input)

        Raises:
            TypeError: If input is not a dict and multiple variables are required.
            KeyError: If required input variables are missing from input dict.

        Example:
            ```python
            from langchain_core.prompts import PromptTemplate
            
            prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
            
            # Basic invocation
            prompt_value = prompt.invoke({"topic": "cats"})
            print(prompt_value.to_string())  # "Tell me a joke about cats"
            
            # With callbacks and metadata
            from langchain_core.callbacks import StdOutCallbackHandler
            
            prompt_value = prompt.invoke(
                {"topic": "dogs"},
                config={
                    "callbacks": [StdOutCallbackHandler()],
                    "metadata": {"user_id": "123"},
                    "tags": ["production"]
                }
            )
            ```
        
        Note:
            The metadata and tags from self.metadata and self.tags are merged into
            the config, enabling automatic propagation through LCEL chains for
            comprehensive tracing.
        
        Source: libs/core/langchain_core/prompts/base.py:192-215
        """
        config = ensure_config(config)
        if self.metadata:
            config["metadata"] = {**config["metadata"], **self.metadata}
        if self.tags:
            config["tags"] += self.tags
        return self._call_with_config(
            self._format_prompt_with_error_handling,
            input,
            config,
            run_type="prompt",
            serialized=self._serialized,
        )

    @override
    async def ainvoke(
        self, input: dict, config: RunnableConfig | None = None, **kwargs: Any
    ) -> PromptValue:
        """Asynchronously invoke the prompt template to produce a PromptValue.
        
        This is the asynchronous counterpart to invoke(), enabling non-blocking
        prompt execution in async contexts. Most prompt templates execute synchronously
        internally, but this method integrates properly with async callback handlers
        and maintains consistency with the Runnable async protocol.

        Args:
            input: Dictionary mapping input variable names to their values.
                Must contain all keys specified in self.input_variables.
                For single-variable prompts, non-dict values are auto-wrapped.
            config: Runnable configuration controlling execution behavior.
                Key configuration options:
                - callbacks: List of callback handlers (async handlers supported)
                - metadata: Dict merged with prompt's self.metadata
                - tags: List extended with prompt's self.tags
                - run_name: Custom name for this execution in traces
                If None, default configuration is used.
            **kwargs: Additional keyword arguments (currently unused, reserved for
                future extensibility).

        Returns:
            PromptValue wrapping the formatted prompt. The PromptValue can be:
            - StringPromptValue: For text-based prompts (LLM input)
            - ChatPromptValueConcrete: For message-based prompts (chat model input)

        Raises:
            TypeError: If input is not a dict and multiple variables are required.
            KeyError: If required input variables are missing from input dict.

        Example:
            ```python
            import asyncio
            from langchain_core.prompts import PromptTemplate
            
            prompt = PromptTemplate.from_template("Translate to {language}: {text}")
            
            async def translate_async():
                prompt_value = await prompt.ainvoke({
                    "language": "French",
                    "text": "Hello, world!"
                })
                return prompt_value.to_string()
            
            result = asyncio.run(translate_async())
            # "Translate to French: Hello, world!"
            ```
        
        Note:
            - Use ainvoke() when integrating with async LCEL chains or when using
              async callback handlers for non-blocking tracing/logging
            - For most prompt templates, the formatting itself is synchronous, but
              async callbacks are properly awaited during execution
            - Metadata and tags are extended (not replaced) to preserve tracing context
        
        Source: libs/core/langchain_core/prompts/base.py:218-241
        """
        config = ensure_config(config)
        if self.metadata:
            config["metadata"].update(self.metadata)
        if self.tags:
            config["tags"].extend(self.tags)
        return await self._acall_with_config(
            self._aformat_prompt_with_error_handling,
            input,
            config,
            run_type="prompt",
            serialized=self._serialized,
        )

    @abstractmethod
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Create a PromptValue from input variables (abstract method).
        
        This is the core formatting method that subclasses must implement to convert
        input variables into a PromptValue wrapper. The PromptValue provides a unified
        interface for passing prompts to both LLMs (via to_string()) and chat models
        (via to_messages()).
        
        Type Flow:
            kwargs (dict) → format_prompt() → PromptValue → to_string() or to_messages()

        Args:
            **kwargs: Keyword arguments providing values for template variables.
                Keys must correspond to variable names in self.input_variables and
                self.partial_variables. Partial variables are automatically merged
                with provided kwargs before formatting.

        Returns:
            PromptValue wrapping the formatted prompt in one of two forms:
            - StringPromptValue: Contains a single text string for LLM completion
            - ChatPromptValueConcrete: Contains a sequence of BaseMessage objects
              for chat model APIs
        
        Raises:
            KeyError: If required input variables are missing from kwargs.
            ValueError: If template formatting fails (e.g., invalid template syntax).
        
        Example:
            ```python
            # Subclass implementation example (PromptTemplate)
            def format_prompt(self, **kwargs: Any) -> PromptValue:
                text = self.format(**kwargs)  # Get formatted string
                return StringPromptValue(text=text)  # Wrap in PromptValue
            
            # Usage
            from langchain_core.prompts import PromptTemplate
            
            prompt = PromptTemplate.from_template("Question: {question}")
            prompt_value = prompt.format_prompt(question="What is AI?")
            
            # For LLM input
            llm_input = prompt_value.to_string()  # "Question: What is AI?"
            
            # For chat model input
            messages = prompt_value.to_messages()  # [HumanMessage(content="...")]
            ```
        
        Note:
            This method is called internally by invoke() after input validation and
            partial variable merging. Subclasses should focus on the formatting logic
            while BasePromptTemplate handles validation and config management.
        
        Source: libs/core/langchain_core/prompts/base.py:243-252
        """

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """Asynchronously create a PromptValue from input variables.
        
        This is the async counterpart to format_prompt(), enabling integration with
        async LCEL chains and async callback handlers. The default implementation
        delegates to the synchronous format_prompt() method, as most template
        formatting operations are CPU-bound and inherently synchronous.
        
        Subclasses can override this method if they require true async operations,
        such as fetching template content from external sources or performing
        async variable transformations.

        Args:
            **kwargs: Keyword arguments providing values for template variables.
                Keys must correspond to variable names in self.input_variables and
                self.partial_variables. Partial variables are automatically merged
                with provided kwargs before formatting.

        Returns:
            PromptValue wrapping the formatted prompt in one of two forms:
            - StringPromptValue: Contains a single text string for LLM completion
            - ChatPromptValueConcrete: Contains a sequence of BaseMessage objects
              for chat model APIs

        Raises:
            KeyError: If required input variables are missing from kwargs.
            ValueError: If template formatting fails (e.g., invalid template syntax).

        Example:
            ```python
            import asyncio
            from langchain_core.prompts import PromptTemplate
            
            prompt = PromptTemplate.from_template("Summarize: {text}")
            
            async def process_async():
                prompt_value = await prompt.aformat_prompt(
                    text="Long document content..."
                )
                return prompt_value.to_string()
            
            result = asyncio.run(process_async())
            ```
        
        Note:
            - Default implementation calls format_prompt() synchronously
            - Override in subclasses if true async behavior is needed
            - Called internally by ainvoke() for async chain execution
            - Most prompt templates do not require async formatting
        
        Source: libs/core/langchain_core/prompts/base.py:254-263
        """
        return self.format_prompt(**kwargs)

    def partial(self, **kwargs: str | Callable[[], str]) -> BasePromptTemplate:
        """Create a new prompt template with partial variables pre-filled.
        
        Partial application allows you to create reusable prompt templates with some
        variables fixed while leaving others to be specified at invocation time. This
        is useful for creating specialized versions of general templates or for injecting
        context that's available at template creation time but not at runtime.
        
        Partial variables support two forms:
        1. Static values: Pre-filled string constants
        2. Dynamic values: Zero-argument callables invoked at format time

        Args:
            **kwargs: Variables to partially apply. Each keyword argument can be:
                - str: A static value that will be fixed in the returned template
                - Callable[[], str]: A function that will be called (with no arguments)
                  each time the template is formatted, enabling dynamic values like
                  timestamps or environment variables

        Returns:
            New BasePromptTemplate instance (same type as self) with:
            - input_variables: Reduced by the variables provided in kwargs
            - partial_variables: Extended with the provided kwargs
            - All other attributes: Copied from the original template

        Raises:
            ValueError: If kwargs contains variables that overlap with existing
                partial_variables (validated by model_validator).

        Example:
            ```python
            from langchain_core.prompts import PromptTemplate
            from datetime import datetime
            
            # Base template with 3 variables
            template = PromptTemplate.from_template(
                "On {date}, user {user} asked: {question}"
            )
            
            # Static partial application
            user_template = template.partial(user="Alice")
            # Now only requires: date, question
            
            # Dynamic partial application with callable
            dated_template = template.partial(
                date=lambda: datetime.now().strftime("%Y-%m-%d"),
                user="Bob"
            )
            # Callable is invoked on each format()
            prompt1 = dated_template.format(question="What is AI?")
            # date is current date at format time
            
            # Chain multiple partial applications
            specific_template = template.partial(user="Charlie").partial(
                date="2024-01-01"
            )
            # Only question remains as input variable
            ```
        
        Note:
            - Partial application is immutable: Returns a new template instance
            - Callable partials are evaluated at format time, not at partial() time
            - Useful for dependency injection and template specialization patterns
            - Variables in partial_variables are merged with runtime kwargs in
              _merge_partial_and_user_variables()
        
        Source: libs/core/langchain_core/prompts/base.py:265-279
        """
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> dict[str, Any]:
        # Get partial params:
        partial_kwargs = {
            k: v if not callable(v) else v() for k, v in self.partial_variables.items()
        }
        return {**partial_kwargs, **kwargs}

    @abstractmethod
    def format(self, **kwargs: Any) -> FormatOutputType:
        """Format the prompt with input variables (abstract method).
        
        This is the core formatting method that subclasses must implement to convert
        input variables into the final prompt output. Unlike format_prompt() which
        returns a PromptValue wrapper, this method returns the raw formatted content.
        
        The return type (FormatOutputType) is generic and depends on the subclass:
        - PromptTemplate: Returns str (single text string)
        - ChatPromptTemplate: Returns list[BaseMessage] (message sequence)

        Args:
            **kwargs: Keyword arguments providing values for template variables.
                Keys must correspond to variable names in self.input_variables.
                Partial variables from self.partial_variables are automatically
                merged before this method is called (via
                _merge_partial_and_user_variables()).

        Returns:
            FormatOutputType: The formatted prompt in subclass-specific format.
            - For text-based prompts (PromptTemplate): str
            - For chat-based prompts (ChatPromptTemplate): list[BaseMessage]

        Raises:
            KeyError: If required input variables are missing from kwargs.
            ValueError: If template syntax is invalid or formatting fails.

        Example:
            ```python
            from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
            from langchain_core.messages import SystemMessage, HumanMessagePromptTemplate
            
            # PromptTemplate returns str
            text_prompt = PromptTemplate.from_template("Hello {name}!")
            result = text_prompt.format(name="World")
            print(result)  # "Hello World!"
            print(type(result))  # <class 'str'>
            
            # ChatPromptTemplate returns list[BaseMessage]
            chat_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful assistant"),
                HumanMessagePromptTemplate.from_template("Question: {question}")
            ])
            messages = chat_prompt.format(question="What is AI?")
            print(type(messages))  # <class 'list'>
            print(type(messages[0]))  # <class 'SystemMessage'>
            ```
        
        Note:
            - This method is called by format_prompt() to generate the content that
              gets wrapped in a PromptValue
            - Subclasses should implement the actual template formatting logic here
            - Partial variables are pre-merged by the caller; implementers don't need
              to handle partial variable merging
            - For async operations, implement aformat() (which by default delegates
              to format())
        
        Source: libs/core/langchain_core/prompts/base.py:288-302
        """

    async def aformat(self, **kwargs: Any) -> FormatOutputType:
        """Asynchronously format the prompt with input variables.
        
        This is the async counterpart to format(), enabling integration with async
        LCEL chains and non-blocking template formatting. The default implementation
        delegates to the synchronous format() method, as most template string
        operations are CPU-bound and inherently synchronous.
        
        Subclasses should override this method if they require true async operations,
        such as fetching template fragments from external APIs, performing async
        variable transformations, or streaming large template content.

        Args:
            **kwargs: Keyword arguments providing values for template variables.
                Keys must correspond to variable names in self.input_variables.
                Partial variables from self.partial_variables are automatically
                merged before this method is called.

        Returns:
            FormatOutputType: The formatted prompt in subclass-specific format.
            - For text-based prompts (PromptTemplate): str
            - For chat-based prompts (ChatPromptTemplate): list[BaseMessage]

        Raises:
            KeyError: If required input variables are missing from kwargs.
            ValueError: If template syntax is invalid or formatting fails.

        Example:
            ```python
            import asyncio
            from langchain_core.prompts import PromptTemplate
            
            prompt = PromptTemplate.from_template(
                "Translate to {language}: {text}"
            )
            
            async def translate_prompt():
                result = await prompt.aformat(
                    language="Spanish",
                    text="Hello, world!"
                )
                return result
            
            formatted = asyncio.run(translate_prompt())
            print(formatted)  # "Translate to Spanish: Hello, world!"
            ```
        
        Note:
            - Default implementation calls format() synchronously
            - Override in subclasses if true async behavior is needed
            - Called internally by aformat_prompt() for async PromptValue creation
            - Most prompt templates do not require async formatting
            - For async variable resolution, consider using async partial callables
              or overriding _merge_partial_and_user_variables()
        
        Source: libs/core/langchain_core/prompts/base.py:304-318
        """
        return self.format(**kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> dict:
        """Return dictionary representation of prompt.

        Args:
            **kwargs: Any additional arguments to pass to the dictionary.

        Returns:
            Dictionary representation of the prompt.
        """
        prompt_dict = super().model_dump(**kwargs)
        with contextlib.suppress(NotImplementedError):
            prompt_dict["_type"] = self._prompt_type
        return prompt_dict

    def save(self, file_path: Path | str) -> None:
        """Save the prompt template to a file in JSON or YAML format.
        
        This method serializes the prompt template configuration (template string,
        input variables, metadata, etc.) to a file for version control, sharing, or
        later loading. The serialization format is determined by the file extension.
        
        File Format Specifications:
        - JSON (.json): Indented (4 spaces), UTF-8 encoding
        - YAML (.yaml, .yml): Default flow style disabled (block style), UTF-8 encoding
        
        Both formats include:
        - _type: Prompt type identifier from _prompt_type property
        - input_variables: List of required variable names
        - template or messages: Template content (format depends on subclass)
        - optional_variables: List of optional variable names (if any)
        - metadata: Metadata dictionary (if set)
        - output_parser: Output parser configuration (if set)
        
        Validation Requirements:
        - Prompt must not have partial variables (they cannot be serialized)
        - Subclass must implement _prompt_type property
        - File extension must be .json, .yaml, or .yml

        Args:
            file_path: Target file path for saving the prompt template.
                Can be string or Path object. Parent directories are created
                automatically if they don't exist.

        Raises:
            ValueError: If the prompt has partial_variables set. Partial variables
                cannot be serialized because they may contain callables or runtime-
                specific values. Create a non-partial version before saving.
            ValueError: If file_path extension is not .json, .yaml, or .yml.
                Only these formats are supported for prompt serialization.
            NotImplementedError: If the subclass does not implement _prompt_type
                property. The _type field is required for deserialization routing.

        Example:
            ```python
            from langchain_core.prompts import PromptTemplate
            from pathlib import Path
            
            # Create and save a prompt template
            prompt = PromptTemplate(
                template="Answer the following question: {question}",
                input_variables=["question"],
                metadata={"version": "1.0", "author": "team"}
            )
            
            # Save as YAML (recommended for readability)
            prompt.save("prompts/qa_prompt.yaml")
            
            # Save as JSON (useful for programmatic processing)
            prompt.save("prompts/qa_prompt.json")
            
            # Using Path object
            prompt.save(Path("prompts/subfolder/qa_prompt.yml"))
            
            # This will fail - partial variables cannot be saved
            partial_prompt = prompt.partial(question="What is AI?")
            # partial_prompt.save("file.yaml")  # Raises ValueError
            ```
        
        Note:
            - Parent directories are created automatically with parents=True
            - Existing files are overwritten without warning
            - Prompts can be loaded back using load_prompt() function
            - Partial variables must be removed before saving (by creating a new
              template without partials)
            - Callable partial variables especially cannot be serialized
        
        Source: libs/core/langchain_core/prompts/base.py:339-379
        """
        if self.partial_variables:
            msg = "Cannot save prompt with partial variables."
            raise ValueError(msg)

        # Fetch dictionary to save
        prompt_dict = self.dict()
        if "_type" not in prompt_dict:
            msg = f"Prompt {self} does not support saving."
            raise NotImplementedError(msg)

        # Convert file to Path object.
        save_path = Path(file_path)

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".json":
            with save_path.open("w", encoding="utf-8") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w", encoding="utf-8") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} must be json or yaml"
            raise ValueError(msg)


def _get_document_info(doc: Document, prompt: BasePromptTemplate[str]) -> dict:
    base_info = {"page_content": doc.page_content, **doc.metadata}
    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        required_metadata = [
            iv for iv in prompt.input_variables if iv != "page_content"
        ]
        msg = (
            f"Document prompt requires documents to have metadata variables: "
            f"{required_metadata}. Received document with missing metadata: "
            f"{list(missing_metadata)}."
        )
        raise ValueError(
            create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
        )
    return {k: base_info[k] for k in prompt.input_variables}


def format_document(doc: Document, prompt: BasePromptTemplate[str]) -> str:
    """Format a document into a string based on a prompt template.

    First, this pulls information from the document from two sources:

    1. page_content:
        This takes the information from the `document.page_content`
        and assigns it to a variable named `page_content`.
    2. metadata:
        This takes information from `document.metadata` and assigns
        it to variables of the same name.

    Those variables are then passed into the `prompt` to produce a formatted string.

    Args:
        doc: Document, the page_content and metadata will be used to create
            the final string.
        prompt: BasePromptTemplate, will be used to format the page_content
            and metadata into the final string.

    Returns:
        string of the document formatted.

    Example:
        ```python
        from langchain_core.documents import Document
        from langchain_core.prompts import PromptTemplate

        doc = Document(page_content="This is a joke", metadata={"page": "1"})
        prompt = PromptTemplate.from_template("Page {page}: {page_content}")
        format_document(doc, prompt)
        >>> "Page 1: This is a joke"

        ```
    """
    return prompt.format(**_get_document_info(doc, prompt))


async def aformat_document(doc: Document, prompt: BasePromptTemplate[str]) -> str:
    """Asynchronously format a document into a string based on a prompt template.

    This is the async counterpart to format_document(), enabling non-blocking
    document formatting in async contexts. It extracts document information and
    applies a prompt template asynchronously.
    
    First, this pulls information from the document from two sources:

    1. page_content:
        This takes the information from the `document.page_content`
        and assigns it to a variable named `page_content`.
    2. metadata:
        This takes information from `document.metadata` and assigns
        it to variables of the same name.

    Those variables are then passed into the `prompt` to produce a formatted string.

    Args:
        doc: Document whose page_content and metadata will be used to create
            the final string. All required template variables must be present
            either as 'page_content' or in the metadata dictionary.
        prompt: BasePromptTemplate[str] that will format the document information.
            Must be a string-returning prompt template (not chat template).
            The template's input_variables must match available document fields.

    Returns:
        Formatted string with document information inserted into the template.

    Raises:
        ValueError: If the document is missing required metadata variables specified
            in the prompt's input_variables. Error includes details about which
            variables are missing.
            Error code: ErrorCode.INVALID_PROMPT_INPUT

    Example:
        ```python
        import asyncio
        from langchain_core.documents import Document
        from langchain_core.prompts import PromptTemplate
        
        doc = Document(
            page_content="LangChain is a framework for LLM applications",
            metadata={"source": "docs.langchain.com", "page": "1"}
        )
        
        prompt = PromptTemplate.from_template(
            "Source: {source} (Page {page})\\n\\nContent: {page_content}"
        )
        
        async def format_async():
            return await aformat_document(doc, prompt)
        
        result = asyncio.run(format_async())
        # Output:
        # Source: docs.langchain.com (Page 1)
        #
        # Content: LangChain is a framework for LLM applications
        ```
    
    Note:
        - Commonly used in retrieval chains to format retrieved documents
        - The document's page_content is automatically mapped to {page_content}
        - All metadata fields become available as template variables
        - For batch document formatting, use asyncio.gather() with multiple calls
        - Most prompt templates format synchronously internally; async is for
          integration consistency and callback handling
    
    Source: libs/core/langchain_core/prompts/base.py:438-461
    """
    return await prompt.aformat(**_get_document_info(doc, prompt))
