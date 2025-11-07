"""Chat prompt template."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

from pydantic import (
    Field,
    PositiveInt,
    SkipValidation,
    model_validator,
)
from typing_extensions import Self, override

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    convert_to_messages,
)
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompt_values import ChatPromptValue, ImageURL
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.dict import DictPromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts.message import (
    BaseMessagePromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
    PromptTemplateFormat,
    StringPromptTemplate,
    get_template_variables,
)
from langchain_core.utils import get_colored_text
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from collections.abc import Sequence


class MessagesPlaceholder(BaseMessagePromptTemplate):
    """Prompt template that assumes variable is already list of messages.

    A placeholder which can be used to pass in a list of messages during prompt
    formatting. This enables dynamic message insertion, allowing variable-length
    message histories or conversation contexts to be injected into chat prompt templates.

    **Purpose:**
        Enable flexible message list insertion in chat prompts without hardcoding
        message count or content. Commonly used for chat history, few-shot examples,
        or any scenario requiring dynamic message sequences.

    **Key Features:**
        - Dynamic message list insertion at specified position in template
        - Optional parameter: allows placeholder to be omitted without error
        - Message count limiting: restrict to last N messages with n_messages
        - Automatic message conversion: accepts tuple/dict formats, converts to BaseMessage

    **Usage Patterns:**
        - Chat history: Insert conversation history between system and current message
        - Few-shot examples: Dynamically include example message pairs
        - Context injection: Add retrieved or computed messages at runtime

    **Optional Parameter Behavior:**
        When optional=False (default):
            - Variable MUST be provided in format_messages() kwargs
            - Omitting the variable raises KeyError
            - Empty list [] is valid input
        
        When optional=True:
            - Variable can be omitted from format_messages() kwargs
            - Omission results in empty list [] inserted
            - Useful for optional chat history in templates

    **n_messages Parameter:**
        Limits the number of messages inserted to the last N messages:
        - If n_messages=3, only last 3 messages from list are used
        - Useful for managing context window size
        - Prevents token limit overflow with long histories
        - Applied after message list is retrieved, before formatting

    **Integration with ChatPromptTemplate:**
        MessagesPlaceholder variables are automatically:
        - Detected during template construction
        - Added to optional_variables if optional=True
        - Typed as list[AnyMessage] in input_types
        - Merged with partial_variables if optional

    Direct usage:

        ```python
        from langchain_core.prompts import MessagesPlaceholder

        # Non-optional placeholder (default behavior)
        prompt = MessagesPlaceholder("history")
        prompt.format_messages()  # raises KeyError: 'history'

        # Optional placeholder
        prompt = MessagesPlaceholder("history", optional=True)
        prompt.format_messages()  # returns empty list []

        # Providing message list
        prompt.format_messages(
            history=[
                ("system", "You are an AI assistant."),
                ("human", "Hello!"),
            ]
        )
        # -> [
        #     SystemMessage(content="You are an AI assistant."),
        #     HumanMessage(content="Hello!"),
        # ]
        ```

    Building a prompt with chat history:

        ```python
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        prompt.invoke(
            {
                "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
                "question": "now multiply that by 4",
            }
        )
        # -> ChatPromptValue(messages=[
        #     SystemMessage(content="You are a helpful assistant."),
        #     HumanMessage(content="what's 5 + 2"),
        #     AIMessage(content="5 + 2 is 7"),
        #     HumanMessage(content="now multiply that by 4"),
        # ])
        ```

    Limiting the number of messages:

        ```python
        from langchain_core.prompts import MessagesPlaceholder

        # Only include last N messages to manage context window
        prompt = MessagesPlaceholder("history", n_messages=1)

        prompt.format_messages(
            history=[
                ("system", "You are an AI assistant."),
                ("human", "Hello!"),
            ]
        )
        # -> [
        #     HumanMessage(content="Hello!"),
        # ]
        # Note: Only last 1 message included
        ```

    Optional placeholder in chat template:

        ```python
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        # Chat history is optional
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are helpful."),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}")
        ])

        # Can invoke without history
        prompt.invoke({"input": "Hello"})
        # -> Only system and human messages

        # Or invoke with history
        prompt.invoke({
            "history": [("human", "Hi"), ("ai", "Hello!")],
            "input": "How are you?"
        })
        # -> Includes history messages
        ```

    Source: libs/core/langchain_core/prompts/chat.py:55
    """

    variable_name: str
    """Name of variable to use as messages."""

    optional: bool = False
    """If `True` format_messages can be called with no arguments and will return an
        empty list. If `False` then a named argument with name `variable_name` must be
        passed in, even if the value is an empty list."""

    n_messages: PositiveInt | None = None
    """Maximum number of messages to include. If `None`, then will include all.
    """

    def __init__(
        self, variable_name: str, *, optional: bool = False, **kwargs: Any
    ) -> None:
        """Create a messages placeholder.

        Args:
            variable_name: Name of variable to use as messages.
            optional: If `True` format_messages can be called with no arguments and will
                return an empty list. If `False` then a named argument with name
                `variable_name` must be passed in, even if the value is an empty list.
        """
        # mypy can't detect the init which is defined in the parent class
        # b/c these are BaseModel classes.
        super().__init__(variable_name=variable_name, optional=optional, **kwargs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessage.

        Raises:
            ValueError: If variable is not a list of messages.
        """
        value = (
            kwargs.get(self.variable_name, [])
            if self.optional
            else kwargs[self.variable_name]
        )
        if not isinstance(value, list):
            msg = (
                f"variable {self.variable_name} should be a list of base messages, "
                f"got {value} of type {type(value)}"
            )
            raise ValueError(msg)  # noqa: TRY004
        value = convert_to_messages(value)
        if self.n_messages:
            value = value[-self.n_messages :]
        return value

    @property
    def input_variables(self) -> list[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        return [self.variable_name] if not self.optional else []

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML.

        Returns:
            Human-readable representation.
        """
        var = "{" + self.variable_name + "}"
        if html:
            title = get_msg_title_repr("Messages Placeholder", bold=True)
            var = get_colored_text(var, "yellow")
        else:
            title = get_msg_title_repr("Messages Placeholder")
        return f"{title}\n\n{var}"


MessagePromptTemplateT = TypeVar(
    "MessagePromptTemplateT", bound="BaseStringMessagePromptTemplate"
)
"""Type variable for message prompt templates."""


class BaseStringMessagePromptTemplate(BaseMessagePromptTemplate, ABC):
    """Base class for message prompt templates that use a string prompt template."""

    prompt: StringPromptTemplate
    """String prompt template."""
    additional_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the prompt template."""

    @classmethod
    def from_template(
        cls,
        template: str,
        template_format: PromptTemplateFormat = "f-string",
        partial_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a class from a string template.

        Args:
            template: a template.
            template_format: format of the template.
            partial_variables: A dictionary of variables that can be used to partially
                fill in the template. For example, if the template is
                `"{variable1} {variable2}"`, and `partial_variables` is
                `{"variable1": "foo"}`, then the final prompt will be
                `"foo {variable2}"`.

            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        prompt = PromptTemplate.from_template(
            template,
            template_format=template_format,
            partial_variables=partial_variables,
        )
        return cls(prompt=prompt, **kwargs)

    @classmethod
    def from_template_file(
        cls,
        template_file: str | Path,
        **kwargs: Any,
    ) -> Self:
        """Create a class from a template file.

        Args:
            template_file: path to a template file. String or Path.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        prompt = PromptTemplate.from_file(template_file)
        return cls(prompt=prompt, **kwargs)

    @abstractmethod
    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """Async format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        return self.format(**kwargs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> list[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        return self.prompt.input_variables

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML.

        Returns:
            Human-readable representation.
        """
        # TODO: Handle partials
        title = self.__class__.__name__.replace("MessagePromptTemplate", " Message")
        title = get_msg_title_repr(title, bold=html)
        return f"{title}\n\n{self.prompt.pretty_repr(html=html)}"


class ChatMessagePromptTemplate(BaseStringMessagePromptTemplate):
    """Chat message prompt template."""

    role: str
    """Role of the message."""

    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        text = self.prompt.format(**kwargs)
        return ChatMessage(
            content=text, role=self.role, additional_kwargs=self.additional_kwargs
        )

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """Async format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        text = await self.prompt.aformat(**kwargs)
        return ChatMessage(
            content=text, role=self.role, additional_kwargs=self.additional_kwargs
        )


class _TextTemplateParam(TypedDict, total=False):
    text: str | dict


class _ImageTemplateParam(TypedDict, total=False):
    image_url: str | dict


class _StringImageMessagePromptTemplate(BaseMessagePromptTemplate):
    """Human message prompt template. This is a message sent from the user."""

    prompt: (
        StringPromptTemplate
        | list[StringPromptTemplate | ImagePromptTemplate | DictPromptTemplate]
    )
    """Prompt template."""
    additional_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the prompt template."""

    _msg_class: type[BaseMessage]

    @classmethod
    def from_template(
        cls: type[Self],
        template: str
        | list[str | _TextTemplateParam | _ImageTemplateParam | dict[str, Any]],
        template_format: PromptTemplateFormat = "f-string",
        *,
        partial_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a class from a string template.

        Args:
            template: a template.
            template_format: format of the template.
                Options are: 'f-string', 'mustache', 'jinja2'.
            partial_variables: A dictionary of variables that can be used too partially.

            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.

        Raises:
            ValueError: If the template is not a string or list of strings.
        """
        if isinstance(template, str):
            prompt: StringPromptTemplate | list = PromptTemplate.from_template(
                template,
                template_format=template_format,
                partial_variables=partial_variables,
            )
            return cls(prompt=prompt, **kwargs)
        if isinstance(template, list):
            if (partial_variables is not None) and len(partial_variables) > 0:
                msg = "Partial variables are not supported for list of templates."
                raise ValueError(msg)
            prompt = []
            for tmpl in template:
                if isinstance(tmpl, str) or (
                    isinstance(tmpl, dict)
                    and "text" in tmpl
                    and set(tmpl.keys()) <= {"type", "text"}
                ):
                    if isinstance(tmpl, str):
                        text: str = tmpl
                    else:
                        text = cast("_TextTemplateParam", tmpl)["text"]  # type: ignore[assignment]
                    prompt.append(
                        PromptTemplate.from_template(
                            text, template_format=template_format
                        )
                    )
                elif (
                    isinstance(tmpl, dict)
                    and "image_url" in tmpl
                    and set(tmpl.keys())
                    <= {
                        "type",
                        "image_url",
                    }
                ):
                    img_template = cast("_ImageTemplateParam", tmpl)["image_url"]
                    input_variables = []
                    if isinstance(img_template, str):
                        variables = get_template_variables(
                            img_template, template_format
                        )
                        if variables:
                            if len(variables) > 1:
                                msg = (
                                    "Only one format variable allowed per image"
                                    f" template.\nGot: {variables}"
                                    f"\nFrom: {tmpl}"
                                )
                                raise ValueError(msg)
                            input_variables = [variables[0]]
                        img_template = {"url": img_template}
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables,
                            template=img_template,
                            template_format=template_format,
                        )
                    elif isinstance(img_template, dict):
                        img_template = dict(img_template)
                        for key in ["url", "path", "detail"]:
                            if key in img_template:
                                input_variables.extend(
                                    get_template_variables(
                                        img_template[key], template_format
                                    )
                                )
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables,
                            template=img_template,
                            template_format=template_format,
                        )
                    else:
                        msg = f"Invalid image template: {tmpl}"
                        raise ValueError(msg)
                    prompt.append(img_template_obj)
                elif isinstance(tmpl, dict):
                    if template_format == "jinja2":
                        msg = (
                            "jinja2 is unsafe and is not supported for templates "
                            "expressed as dicts. Please use 'f-string' or 'mustache' "
                            "format."
                        )
                        raise ValueError(msg)
                    data_template_obj = DictPromptTemplate(
                        template=cast("dict[str, Any]", tmpl),
                        template_format=template_format,
                    )
                    prompt.append(data_template_obj)
                else:
                    msg = f"Invalid template: {tmpl}"
                    raise ValueError(msg)
            return cls(prompt=prompt, **kwargs)
        msg = f"Invalid template: {template}"
        raise ValueError(msg)

    @classmethod
    def from_template_file(
        cls: type[Self],
        template_file: str | Path,
        input_variables: list[str],
        **kwargs: Any,
    ) -> Self:
        """Create a class from a template file.

        Args:
            template_file: path to a template file. String or Path.
            input_variables: list of input variables.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        template = Path(template_file).read_text(encoding="utf-8")
        return cls.from_template(template, input_variables=input_variables, **kwargs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> list[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        return [iv for prompt in prompts for iv in prompt.input_variables]

    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = self.prompt.format(**kwargs)
            return self._msg_class(
                content=text, additional_kwargs=self.additional_kwargs
            )
        content: list = []
        for prompt in self.prompt:
            inputs = {var: kwargs[var] for var in prompt.input_variables}
            if isinstance(prompt, StringPromptTemplate):
                formatted: str | ImageURL | dict[str, Any] = prompt.format(**inputs)
                content.append({"type": "text", "text": formatted})
            elif isinstance(prompt, ImagePromptTemplate):
                formatted = prompt.format(**inputs)
                content.append({"type": "image_url", "image_url": formatted})
            elif isinstance(prompt, DictPromptTemplate):
                formatted = prompt.format(**inputs)
                content.append(formatted)
        return self._msg_class(
            content=content, additional_kwargs=self.additional_kwargs
        )

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """Async format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = await self.prompt.aformat(**kwargs)
            return self._msg_class(
                content=text, additional_kwargs=self.additional_kwargs
            )
        content: list = []
        for prompt in self.prompt:
            inputs = {var: kwargs[var] for var in prompt.input_variables}
            if isinstance(prompt, StringPromptTemplate):
                formatted: str | ImageURL | dict[str, Any] = await prompt.aformat(
                    **inputs
                )
                content.append({"type": "text", "text": formatted})
            elif isinstance(prompt, ImagePromptTemplate):
                formatted = await prompt.aformat(**inputs)
                content.append({"type": "image_url", "image_url": formatted})
            elif isinstance(prompt, DictPromptTemplate):
                formatted = prompt.format(**inputs)
                content.append(formatted)
        return self._msg_class(
            content=content, additional_kwargs=self.additional_kwargs
        )

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML.

        Returns:
            Human-readable representation.
        """
        # TODO: Handle partials
        title = self.__class__.__name__.replace("MessagePromptTemplate", " Message")
        title = get_msg_title_repr(title, bold=html)
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        prompt_reprs = "\n\n".join(prompt.pretty_repr(html=html) for prompt in prompts)
        return f"{title}\n\n{prompt_reprs}"


class HumanMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """Human message prompt template. This is a message sent from the user."""

    _msg_class: type[BaseMessage] = HumanMessage


class AIMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """AI message prompt template. This is a message sent from the AI."""

    _msg_class: type[BaseMessage] = AIMessage


class SystemMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """System message prompt template.

    This is a message that is not sent to the user.
    """

    _msg_class: type[BaseMessage] = SystemMessage


class BaseChatPromptTemplate(BasePromptTemplate, ABC):
    """Base class for chat prompt templates."""

    @property
    @override
    def lc_attributes(self) -> dict:
        return {"input_variables": self.input_variables}

    def format(self, **kwargs: Any) -> str:
        """Format the chat template into a string.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                in all the template messages in this chat template.

        Returns:
            formatted string.
        """
        return self.format_prompt(**kwargs).to_string()

    async def aformat(self, **kwargs: Any) -> str:
        """Async format the chat template into a string.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                in all the template messages in this chat template.

        Returns:
            formatted string.
        """
        return (await self.aformat_prompt(**kwargs)).to_string()

    def format_prompt(self, **kwargs: Any) -> ChatPromptValue:
        """Format prompt. Should return a ChatPromptValue.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            ChatPromptValue.
        """
        messages = self.format_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    async def aformat_prompt(self, **kwargs: Any) -> ChatPromptValue:
        """Async format prompt. Should return a ChatPromptValue.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            PromptValue.
        """
        messages = await self.aformat_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format kwargs into a list of messages.

        Returns:
            List of messages.
        """

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format kwargs into a list of messages.

        Returns:
            List of messages.
        """
        return self.format_messages(**kwargs)

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML.

        Returns:
            Human-readable representation.
        """
        raise NotImplementedError

    def pretty_print(self) -> None:
        """Print a human-readable representation."""
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


MessageLike = BaseMessagePromptTemplate | BaseMessage | BaseChatPromptTemplate

MessageLikeRepresentation = (
    MessageLike
    | tuple[str | type, str | list[dict] | list[object]]
    | str
    | dict[str, Any]
)


class ChatPromptTemplate(BaseChatPromptTemplate):
    """Prompt template for chat models.

    Use to create flexible templated prompts for chat models. This class enables
    composition of message sequences with template variables, supporting multiple
    message types (System, Human, AI) and dynamic message insertion via 
    MessagesPlaceholder.

    **Purpose:**
        Create structured chat prompts with variable substitution, allowing reusable
        prompt templates for chat-based language models. Supports flexible message
        composition patterns including static messages, templated messages, and 
        dynamic message lists.

    **Usage Patterns:**
        - Simple system + human message prompts with variables
        - Multi-turn conversation templates with message history
        - Few-shot examples with system/human/AI message patterns
        - Complex prompts with MessagesPlaceholder for dynamic message insertion

    **Message Composition:**
        Messages can be constructed from multiple formats:
        - BaseMessagePromptTemplate objects: Pre-constructed message templates
        - BaseMessage objects: Direct message instances (SystemMessage, HumanMessage, AIMessage)
        - 2-tuple (message_type, template): e.g., ("human", "{user_input}")
        - 2-tuple (message_class, template): e.g., (HumanMessage, "Hello")
        - String shorthand: Interpreted as human message template
        - MessagesPlaceholder: For dynamic message list insertion

    **Input Variable Substitution:**
        Template variables are automatically extracted from message templates.
        Variables are specified using f-string format: "{variable_name}".
        All variables must be provided at format time unless marked as optional
        via MessagesPlaceholder(optional=True) or partial variables.

    **Validation:**
        - Template variables are validated during formatting
        - Missing required variables raise KeyError
        - Optional MessagesPlaceholder variables can be omitted
        - Extra variables provided at format time are ignored
        - Partial variables are merged with runtime variables before formatting

    Source: libs/core/langchain_core/prompts/chat.py:774

    Examples:
        !!! warning "Behavior changed in 0.2.24"
            You can pass any Message-like formats supported by
            `ChatPromptTemplate.from_messages()` directly to `ChatPromptTemplate()`
            init.

        ```python
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            [
                ("system", "You are a helpful AI bot. Your name is {name}."),
                ("human", "Hello, how are you doing?"),
                ("ai", "I'm doing well, thanks!"),
                ("human", "{user_input}"),
            ]
        )

        prompt_value = template.invoke(
            {
                "name": "Bob",
                "user_input": "What is your name?",
            }
        )
        # Output:
        # ChatPromptValue(
        #    messages=[
        #        SystemMessage(content='You are a helpful AI bot. Your name is Bob.'),
        #        HumanMessage(content='Hello, how are you doing?'),
        #        AIMessage(content="I'm doing well, thanks!"),
        #        HumanMessage(content='What is your name?')
        #    ]
        # )
        ```

    Messages Placeholder:

        ```python
        # In addition to Human/AI/Tool/Function messages,
        # you can initialize the template with a MessagesPlaceholder
        # either using the class directly or with the shorthand tuple syntax:

        template = ChatPromptTemplate(
            [
                ("system", "You are a helpful AI bot."),
                # Means the template will receive an optional list of messages under
                # the "conversation" key
                ("placeholder", "{conversation}"),
                # Equivalently:
                # MessagesPlaceholder(variable_name="conversation", optional=True)
            ]
        )

        prompt_value = template.invoke(
            {
                "conversation": [
                    ("human", "Hi!"),
                    ("ai", "How can I assist you today?"),
                    ("human", "Can you make me an ice cream sundae?"),
                    ("ai", "No."),
                ]
            }
        )

        # Output:
        # ChatPromptValue(
        #    messages=[
        #        SystemMessage(content='You are a helpful AI bot.'),
        #        HumanMessage(content='Hi!'),
        #        AIMessage(content='How can I assist you today?'),
        #        HumanMessage(content='Can you make me an ice cream sundae?'),
        #        AIMessage(content='No.'),
        #    ]
        # )
        ```

    Single-variable template:

        If your prompt has only a single input variable (i.e., 1 instance of "{variable_nams}"),
        and you invoke the template with a non-dict object, the prompt template will
        inject the provided argument into that variable location.


        ```python
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            [
                ("system", "You are a helpful AI bot. Your name is Carl."),
                ("human", "{user_input}"),
            ]
        )

        prompt_value = template.invoke("Hello, there!")
        # Equivalent to
        # prompt_value = template.invoke({"user_input": "Hello, there!"})

        # Output:
        #  ChatPromptValue(
        #     messages=[
        #         SystemMessage(content='You are a helpful AI bot. Your name is Carl.'),
        #         HumanMessage(content='Hello, there!'),
        #     ]
        # )
        ```
    """  # noqa: E501

    messages: Annotated[list[MessageLike], SkipValidation()]
    """List of messages consisting of either message prompt templates or messages.
    
    Type: list[MessageLike]
        where MessageLike = BaseMessagePromptTemplate | BaseMessage | BaseChatPromptTemplate
    
    This field accepts a heterogeneous list of message representations:
        - BaseMessagePromptTemplate: Templates like HumanMessagePromptTemplate that
          require variable substitution
        - BaseMessage: Concrete message instances like SystemMessage, HumanMessage,
          AIMessage that are included as-is
        - BaseChatPromptTemplate: Nested chat prompt templates for composition
        - MessagesPlaceholder: Special template for dynamic message list insertion
    
    The messages are automatically converted during __init__ from MessageLikeRepresentation
    formats (tuples, strings, dicts) into appropriate template objects. During formatting,
    templates are resolved to BaseMessage instances with variables substituted.
    """
    
    validate_template: bool = False
    """Whether or not to try validating the template.
    
    When True, validates that provided input_variables match those extracted from
    message templates. When False (default), input_variables are automatically
    inferred from templates. Validation is useful for catching configuration errors
    but is typically disabled for flexibility.
    """

    def __init__(
        self,
        messages: Sequence[MessageLikeRepresentation],
        *,
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
    ) -> None:
        """Create a chat prompt template from a variety of message formats.

        Constructs a ChatPromptTemplate by converting message representations into
        template objects and automatically extracting input variables. This is the
        primary constructor, though from_messages() classmethod is often preferred
        for its clearer interface.

        **Automatic Variable Extraction:**
            Input variables are automatically detected from all message templates:
            - Scans each BaseMessagePromptTemplate for template variables
            - Creates sorted list of unique variable names
            - Sets as input_variables unless explicitly overridden
            - Optional MessagesPlaceholder variables tracked separately

        **Message Conversion Process:**
            Each message in the sequence is converted via _convert_to_message_template():
            - Tuples converted to appropriate MessagePromptTemplate types
            - Strings converted to HumanMessagePromptTemplate
            - BaseMessage instances included directly
            - MessagesPlaceholder instances configured with optional settings

        Args:
            messages: Sequence of message representations. Each element can be:
                
                (1) **BaseMessagePromptTemplate**: Pre-constructed template instances
                    - HumanMessagePromptTemplate, AIMessagePromptTemplate, etc.
                    - MessagesPlaceholder for dynamic message insertion
                    
                (2) **BaseMessage**: Concrete message instances (no variable substitution)
                    - SystemMessage(content="You are helpful")
                    - HumanMessage(content="Hello")
                    - AIMessage(content="Hi there")
                    
                (3) **2-tuple of (message_type, template)**: String type with template
                    - ("system", "You are {name}")
                    - ("human", "{user_input}")
                    - ("ai", "{response}")
                    - ("placeholder", "{conversation}") for MessagesPlaceholder
                    
                (4) **2-tuple of (message_class, template)**: Class with template
                    - (HumanMessage, "Hello {name}")
                    - (SystemMessage, "Instructions: {instructions}")
                    
                (5) **String**: Shorthand for human message template
                    - "{user_input}" → ("human", "{user_input}")
                    
                (6) **Dict with 'role' and 'content'**: Dictionary representation
                    - {"role": "human", "content": "Hello {name}"}
                    
            template_format: Format for template string parsing. Options:
                - "f-string" (default): Python f-string format with {variable}
                - "jinja2": Jinja2 template syntax with {{ variable }}
                - "mustache": Mustache template syntax with {{variable}}
                
                Affects how variables are extracted and substituted.
                
            **kwargs: Additional configuration parameters passed to parent BasePromptTemplate:
            
                input_variables: list[str] (optional)
                    Explicit list of required input variable names.
                    If not provided, automatically inferred from message templates.
                    Validation only occurs if validate_template=True.
                    
                optional_variables: list[str] (optional)
                    List of optional variable names (typically from MessagesPlaceholder
                    with optional=True). Auto-inferred from templates if not provided.
                    
                partial_variables: dict[str, Any] (optional)
                    Variables pre-filled with values. These don't need to be provided
                    at format time. Useful for constants or computed values.
                    Example: {"date": "2024-01-01", "version": "1.0"}
                    
                validate_template: bool (optional, default=False)
                    If True, validates that provided input_variables match extracted
                    variables. Raises ValueError on mismatch. If False, uses inferred
                    variables without validation.
                    
                input_types: dict[str, Any] (optional)
                    Type specifications for input variables. Used for type checking
                    and validation. If not provided, variables assumed to be strings
                    except MessagesPlaceholder which is typed as list[AnyMessage].

        Examples:
            Instantiation from tuple list:

            ```python
            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate(
                [
                    ("system", "You are a {role}"),
                    ("human", "Hello, how are you?"),
                    ("ai", "I'm doing well, thanks!"),
                    ("human", "{user_input}"),
                ]
            )
            # input_variables automatically set to ["role", "user_input"]
            ```

            Mixed message formats:

            ```python
            from langchain_core.messages import SystemMessage

            template = ChatPromptTemplate(
                [
                    SystemMessage(content="You are helpful"),  # No substitution
                    ("human", "Hello, {name}!"),  # Template with variable
                ]
            )
            # input_variables automatically set to ["name"]
            ```

            With partial variables:

            ```python
            template = ChatPromptTemplate(
                [
                    ("system", "Version: {version}, User: {user}"),
                    ("human", "{input}")
                ],
                partial_variables={"version": "1.0"}
            )
            # Only need to provide "user" and "input" at format time
            ```

            With MessagesPlaceholder:

            ```python
            from langchain_core.prompts import MessagesPlaceholder

            template = ChatPromptTemplate(
                [
                    ("system", "You are helpful"),
                    MessagesPlaceholder("history", optional=True),
                    ("human", "{question}")
                ]
            )
            # input_variables: ["question"]
            # optional_variables: ["history"]
            ```

        Source: libs/core/langchain_core/prompts/chat.py:891
        """
        messages_ = [
            _convert_to_message_template(message, template_format)
            for message in messages
        ]

        # Automatically infer input variables from messages
        input_vars: set[str] = set()
        optional_variables: set[str] = set()
        partial_vars: dict[str, Any] = {}
        for _message in messages_:
            if isinstance(_message, MessagesPlaceholder) and _message.optional:
                partial_vars[_message.variable_name] = []
                optional_variables.add(_message.variable_name)
            elif isinstance(
                _message, (BaseChatPromptTemplate, BaseMessagePromptTemplate)
            ):
                input_vars.update(_message.input_variables)

        kwargs = {
            "input_variables": sorted(input_vars),
            "optional_variables": sorted(optional_variables),
            "partial_variables": partial_vars,
            **kwargs,
        }
        cast("type[ChatPromptTemplate]", super()).__init__(messages=messages_, **kwargs)

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "prompts", "chat"]`
        """
        return ["langchain", "prompts", "chat"]

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """Combine two prompt templates.

        Args:
            other: Another prompt template.

        Returns:
            Combined prompt template.
        """
        partials = {**self.partial_variables}

        # Need to check that other has partial variables since it may not be
        # a ChatPromptTemplate.
        if hasattr(other, "partial_variables") and other.partial_variables:
            partials.update(other.partial_variables)

        # Allow for easy combining
        if isinstance(other, ChatPromptTemplate):
            return ChatPromptTemplate(messages=self.messages + other.messages).partial(
                **partials
            )
        if isinstance(
            other, (BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate)
        ):
            return ChatPromptTemplate(messages=[*self.messages, other]).partial(
                **partials
            )
        if isinstance(other, (list, tuple)):
            other_ = ChatPromptTemplate.from_messages(other)
            return ChatPromptTemplate(messages=self.messages + other_.messages).partial(
                **partials
            )
        if isinstance(other, str):
            prompt = HumanMessagePromptTemplate.from_template(other)
            return ChatPromptTemplate(messages=[*self.messages, prompt]).partial(
                **partials
            )
        msg = f"Unsupported operand type for +: {type(other)}"
        raise NotImplementedError(msg)

    @model_validator(mode="before")
    @classmethod
    def validate_input_variables(cls, values: dict) -> Any:
        """Validate input variables.

        If input_variables is not set, it will be set to the union of
        all input variables in the messages.

        Args:
            values: values to validate.

        Returns:
            Validated values.

        Raises:
            ValueError: If input variables do not match.
        """
        messages = values["messages"]
        input_vars: set = set()
        optional_variables = set()
        input_types: dict[str, Any] = values.get("input_types", {})
        for message in messages:
            if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
                input_vars.update(message.input_variables)
            if isinstance(message, MessagesPlaceholder):
                if "partial_variables" not in values:
                    values["partial_variables"] = {}
                if (
                    message.optional
                    and message.variable_name not in values["partial_variables"]
                ):
                    values["partial_variables"][message.variable_name] = []
                    optional_variables.add(message.variable_name)
                if message.variable_name not in input_types:
                    input_types[message.variable_name] = list[AnyMessage]
        if "partial_variables" in values:
            input_vars -= set(values["partial_variables"])
        if optional_variables:
            input_vars -= optional_variables
        if "input_variables" in values and values.get("validate_template"):
            if input_vars != set(values["input_variables"]):
                msg = (
                    "Got mismatched input_variables. "
                    f"Expected: {input_vars}. "
                    f"Got: {values['input_variables']}"
                )
                raise ValueError(msg)
        else:
            values["input_variables"] = sorted(input_vars)
        if optional_variables:
            values["optional_variables"] = sorted(optional_variables)
        values["input_types"] = input_types
        return values

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> ChatPromptTemplate:
        """Create a chat prompt template from a template string.

        Convenience method for creating a simple chat template consisting of a single
        human message. This is a shorthand for creating a ChatPromptTemplate with
        one HumanMessagePromptTemplate. Useful for simple single-turn interactions.

        Args:
            template: Template string with f-string variable placeholders.
                Variables should be specified using curly braces: "{variable_name}".
                Example: "Hello {name}, how can I help you with {topic}?"
                
                The template string will be converted to a HumanMessagePromptTemplate,
                representing a message from the user to the chat model.
                
            **kwargs: Additional keyword arguments passed to the PromptTemplate constructor.
                Common options include:
                - template_format: Format of template ("f-string", "jinja2", "mustache")
                - partial_variables: Dict of variables to pre-fill
                - input_variables: Explicit list of variable names (auto-detected if omitted)

        Returns:
            ChatPromptTemplate: A new ChatPromptTemplate instance containing a single
                HumanMessagePromptTemplate with the specified template string.
                The returned template can be invoked with a dict of variable values
                or directly with a single value if only one variable exists.

        Examples:
            Simple single-variable template:

            ```python
            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate.from_template("Hello {name}!")
            result = template.invoke({"name": "Alice"})
            # Returns: ChatPromptValue with HumanMessage(content="Hello Alice!")
            
            # Single-variable templates support direct value passing:
            result = template.invoke("Alice")
            # Same result as above
            ```

            Multi-variable template:

            ```python
            template = ChatPromptTemplate.from_template(
                "Tell me about {topic} in {language}"
            )
            result = template.invoke({
                "topic": "Python",
                "language": "English"
            })
            # Returns: ChatPromptValue with HumanMessage(content="Tell me about Python in English")
            ```

            Template with partial variables:

            ```python
            template = ChatPromptTemplate.from_template(
                "Hello {name}, the date is {date}",
                partial_variables={"date": "2024-01-01"}
            )
            # Only need to provide 'name' at invoke time
            result = template.invoke({"name": "Bob"})
            ```

        Note:
            For multi-turn conversations or templates requiring system messages,
            use ChatPromptTemplate.from_messages() instead, which provides more
            flexibility in message composition.

        Source: libs/core/langchain_core/prompts/chat.py:1075
        """
        prompt_template = PromptTemplate.from_template(template, **kwargs)
        message = HumanMessagePromptTemplate(prompt=prompt_template)
        return cls.from_messages([message])

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[MessageLikeRepresentation],
        template_format: PromptTemplateFormat = "f-string",
    ) -> ChatPromptTemplate:
        """Create a chat prompt template from a variety of message formats.

        Factory method providing a flexible interface for creating chat prompt templates
        from diverse message representations. Automatically extracts input variables from
        templates and handles message type conversions.

        Args:
            messages: Sequence of message representations. Each message can be represented
                in one of the following formats:
                
                (1) **BaseMessagePromptTemplate objects**: Pre-constructed message templates
                    like HumanMessagePromptTemplate, AIMessagePromptTemplate, or 
                    SystemMessagePromptTemplate instances.
                    
                (2) **BaseMessage objects**: Direct message instances including:
                    - SystemMessage: System-level instructions
                    - HumanMessage: User messages
                    - AIMessage: Assistant responses
                    - ChatMessage: Generic messages with custom roles
                    
                (3) **2-tuple of (message_type, template)**: String type identifier with template.
                    - message_type: One of "system", "human", "user", "ai", "assistant", or "placeholder"
                    - template: String template with f-string variables like "{variable_name}"
                    - Example: ("human", "{user_input}")
                    - Example: ("system", "You are a helpful assistant named {name}")
                    
                (4) **2-tuple of (message_class, template)**: Message class with template string.
                    - message_class: A BaseMessage subclass like HumanMessage or SystemMessage
                    - template: String template for the message content
                    - Example: (HumanMessage, "Hello {name}")
                    
                (5) **String shorthand**: Plain string interpreted as human message template.
                    - Example: "{user_input}" is equivalent to ("human", "{user_input}")
                    
                (6) **MessagesPlaceholder**: For dynamic message list insertion.
                    - Allows inserting a variable-length list of messages at runtime
                    - Example: MessagesPlaceholder("chat_history")
                    - Example using tuple: ("placeholder", "{conversation}")
                    
                (7) **Dict with 'role' and 'content' keys**: Dictionary message representation.
                    - Must have exactly two keys: "role" and "content"
                    - Example: {"role": "human", "content": "Hello {name}"}

            template_format: Format of the template strings. Options are:
                - "f-string" (default): Python f-string formatting using {variable}
                - "jinja2": Jinja2 template syntax
                - "mustache": Mustache template syntax
                
                Note: Variable extraction and substitution behavior depends on the format.
                f-string is recommended for most use cases.

        Returns:
            ChatPromptTemplate: A new ChatPromptTemplate instance with:
                - messages: Converted list of message templates
                - input_variables: Automatically extracted from templates (sorted list)
                - optional_variables: Variables from optional MessagesPlaceholder instances
                - partial_variables: Pre-filled variables (initially empty unless set)

        Examples:
            Simple system + human message template:

            ```python
            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant named {name}."),
                ("human", "{user_input}")
            ])
            
            result = template.invoke({"name": "Alice", "user_input": "Hello!"})
            # Returns: ChatPromptValue with SystemMessage and HumanMessage
            ```

            Multi-turn conversation with MessagesPlaceholder:

            ```python
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}")
            ])
            
            result = template.invoke({
                "chat_history": [
                    ("human", "What's 2+2?"),
                    ("ai", "2+2 equals 4."),
                ],
                "question": "What about 3+3?"
            })
            # MessagesPlaceholder inserts the chat history messages
            ```

            Mixed message formats:

            ```python
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.messages import SystemMessage

            template = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful assistant."),
                ("human", "Hello, how are you?"),
                ("ai", "I'm doing well, thanks!"),
                ("human", "{user_input}"),
            ])
            ```

            Few-shot examples with AI messages:

            ```python
            template = ChatPromptTemplate.from_messages([
                ("system", "Translate English to French."),
                ("human", "Hello"),
                ("ai", "Bonjour"),
                ("human", "Goodbye"),
                ("ai", "Au revoir"),
                ("human", "{input}")
            ])
            ```

        Source: libs/core/langchain_core/prompts/chat.py:1092
        """
        return cls(messages, template_format=template_format)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format the chat template into a list of finalized messages.

        Merges partial variables with provided kwargs, then iterates through each
        message template to produce concrete BaseMessage instances with variables
        substituted.

        **Validation Rules:**
            - All required input_variables must be provided in kwargs
            - Missing required variables raise KeyError
            - Optional MessagesPlaceholder variables can be omitted
            - Extra variables in kwargs are ignored (no error)
            - Partial variables are automatically merged before formatting
            - Message list values for MessagesPlaceholder must be lists

        **Variable Substitution Process:**
            1. Merge partial_variables with provided kwargs
            2. For each message template in self.messages:
               - If BaseMessage: Include directly without substitution
               - If BaseMessagePromptTemplate or BaseChatPromptTemplate: 
                 Call format_messages() with merged kwargs
               - If MessagesPlaceholder: Retrieve variable list and convert to messages
            3. Flatten all resulting messages into single list

        **Type Constraints:**
            - All template variables after substitution must be strings
            - MessagesPlaceholder variables must be list types
            - Message content must be string or list of content blocks

        Args:
            **kwargs: Keyword arguments providing values for template variables.
                Keys should match input_variables extracted from templates.
                For MessagesPlaceholder, provide list of message-like objects:
                - List of tuples: [("human", "text"), ("ai", "response")]
                - List of BaseMessage instances
                - Mixed formats supported
                
                Example:
                    {
                        "name": "Alice",
                        "user_input": "Hello",
                        "chat_history": [
                            ("human", "Previous message"),
                            ("ai", "Previous response")
                        ]
                    }

        Returns:
            list[BaseMessage]: List of formatted, concrete BaseMessage instances
                (SystemMessage, HumanMessage, AIMessage, etc.) with all template
                variables substituted. Ready to be sent to a chat model.

        Raises:
            KeyError: If required input variable is missing from kwargs.
                Optional MessagesPlaceholder variables with optional=True do not
                raise KeyError when omitted.
                
            ValueError: If message template is of unexpected type or if
                MessagesPlaceholder variable is not a list.

        Examples:
            Basic variable substitution:

            ```python
            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate.from_messages([
                ("system", "You are {name}"),
                ("human", "{input}")
            ])
            
            messages = template.format_messages(name="Alice", input="Hello")
            # -> [
            #     SystemMessage(content="You are Alice"),
            #     HumanMessage(content="Hello")
            # ]
            ```

            With MessagesPlaceholder:

            ```python
            template = ChatPromptTemplate.from_messages([
                ("system", "You are helpful"),
                MessagesPlaceholder("history"),
                ("human", "{input}")
            ])
            
            messages = template.format_messages(
                history=[("human", "Hi"), ("ai", "Hello!")],
                input="How are you?"
            )
            # -> [SystemMessage, HumanMessage("Hi"), AIMessage("Hello!"), HumanMessage("How are you?")]
            ```

        Source: libs/core/langchain_core/prompts/chat.py:1138
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                result.extend([message_template])
            elif isinstance(
                message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)
            ):
                message = message_template.format_messages(**kwargs)
                result.extend(message)
            else:
                msg = f"Unexpected input: {message_template}"
                raise ValueError(msg)  # noqa: TRY004
        return result

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format the chat template into a list of finalized messages.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                in all the template messages in this chat template.

        Returns:
            list of formatted messages.

        Raises:
            ValueError: If unexpected input.
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                result.extend([message_template])
            elif isinstance(
                message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)
            ):
                message = await message_template.aformat_messages(**kwargs)
                result.extend(message)
            else:
                msg = f"Unexpected input: {message_template}"
                raise ValueError(msg)  # noqa:TRY004
        return result

    def partial(self, **kwargs: Any) -> ChatPromptTemplate:
        """Get a new ChatPromptTemplate with some input variables already filled in.

        Args:
            **kwargs: keyword arguments to use for filling in template variables. Ought
                        to be a subset of the input variables.

        Returns:
            A new ChatPromptTemplate.


        Example:
            ```python
            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are an AI assistant named {name}."),
                    ("human", "Hi I'm {user}"),
                    ("ai", "Hi there, {user}, I'm {name}."),
                    ("human", "{input}"),
                ]
            )
            template2 = template.partial(user="Lucy", name="R2D2")

            template2.format_messages(input="hello")
            ```
        """
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def append(self, message: MessageLikeRepresentation) -> None:
        """Append a message to the end of the chat template.

        Args:
            message: representation of a message to append.
        """
        self.messages.append(_convert_to_message_template(message))

    def extend(self, messages: Sequence[MessageLikeRepresentation]) -> None:
        """Extend the chat template with a sequence of messages.

        Args:
            messages: sequence of message representations to append.
        """
        self.messages.extend(
            [_convert_to_message_template(message) for message in messages]
        )

    @overload
    def __getitem__(self, index: int) -> MessageLike: ...

    @overload
    def __getitem__(self, index: slice) -> ChatPromptTemplate: ...

    def __getitem__(self, index: int | slice) -> MessageLike | ChatPromptTemplate:
        """Use to index into the chat template.

        Returns:
            If index is an int, returns the message at that index.
            If index is a slice, returns a new `ChatPromptTemplate`
            containing the messages in that slice.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.messages))
            messages = self.messages[start:stop:step]
            return ChatPromptTemplate.from_messages(messages)
        return self.messages[index]

    def __len__(self) -> int:
        """Return the length of the chat template."""
        return len(self.messages)

    @property
    def _prompt_type(self) -> str:
        """Name of prompt type. Used for serialization."""
        return "chat"

    def save(self, file_path: Path | str) -> None:
        """Save prompt to file.

        Args:
            file_path: path to file.
        """
        raise NotImplementedError

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML.

        Returns:
            Human-readable representation.
        """
        # TODO: handle partials
        return "\n\n".join(msg.pretty_repr(html=html) for msg in self.messages)


def _create_template_from_message_type(
    message_type: str,
    template: str | list,
    template_format: PromptTemplateFormat = "f-string",
) -> BaseMessagePromptTemplate:
    """Create a message prompt template from a message type and template string.

    Args:
        message_type: str the type of the message template (e.g., "human", "ai", etc.)
        template: str the template string.
        template_format: format of the template.

    Returns:
        a message prompt template of the appropriate type.

    Raises:
        ValueError: If unexpected message type.
    """
    if message_type in {"human", "user"}:
        message: BaseMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
            template, template_format=template_format
        )
    elif message_type in {"ai", "assistant"}:
        message = AIMessagePromptTemplate.from_template(
            cast("str", template), template_format=template_format
        )
    elif message_type == "system":
        message = SystemMessagePromptTemplate.from_template(
            cast("str", template), template_format=template_format
        )
    elif message_type == "placeholder":
        if isinstance(template, str):
            if template[0] != "{" or template[-1] != "}":
                msg = (
                    f"Invalid placeholder template: {template}."
                    " Expected a variable name surrounded by curly braces."
                )
                raise ValueError(msg)
            var_name = template[1:-1]
            message = MessagesPlaceholder(variable_name=var_name, optional=True)
        elif len(template) == 2 and isinstance(template[1], bool):
            var_name_wrapped, is_optional = template
            if not isinstance(var_name_wrapped, str):
                msg = f"Expected variable name to be a string. Got: {var_name_wrapped}"
                raise ValueError(msg)  # noqa:TRY004
            if var_name_wrapped[0] != "{" or var_name_wrapped[-1] != "}":
                msg = (
                    f"Invalid placeholder template: {var_name_wrapped}."
                    " Expected a variable name surrounded by curly braces."
                )
                raise ValueError(msg)
            var_name = var_name_wrapped[1:-1]

            message = MessagesPlaceholder(variable_name=var_name, optional=is_optional)
        else:
            msg = (
                "Unexpected arguments for placeholder message type."
                " Expected either a single string variable name"
                " or a list of [variable_name: str, is_optional: bool]."
                f" Got: {template}"
            )
            raise ValueError(msg)
    else:
        msg = (
            f"Unexpected message type: {message_type}. Use one of 'human',"
            f" 'user', 'ai', 'assistant', or 'system'."
        )
        raise ValueError(msg)
    return message


def _convert_to_message_template(
    message: MessageLikeRepresentation,
    template_format: PromptTemplateFormat = "f-string",
) -> BaseMessage | BaseMessagePromptTemplate | BaseChatPromptTemplate:
    """Instantiate a message from a variety of message formats.

    The message format can be one of the following:

    - BaseMessagePromptTemplate
    - BaseMessage
    - 2-tuple of (role string, template); e.g., ("human", "{user_input}")
    - 2-tuple of (message class, template)
    - string: shorthand for ("human", template); e.g., "{user_input}"

    Args:
        message: a representation of a message in one of the supported formats.
        template_format: format of the template.

    Returns:
        an instance of a message or a message template.

    Raises:
        ValueError: If unexpected message type.
        ValueError: If 2-tuple does not have 2 elements.
    """
    if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
        message_: BaseMessage | BaseMessagePromptTemplate | BaseChatPromptTemplate = (
            message
        )
    elif isinstance(message, BaseMessage):
        message_ = message
    elif isinstance(message, str):
        message_ = _create_template_from_message_type(
            "human", message, template_format=template_format
        )
    elif isinstance(message, (tuple, dict)):
        if isinstance(message, dict):
            if set(message.keys()) != {"content", "role"}:
                msg = (
                    "Expected dict to have exact keys 'role' and 'content'."
                    f" Got: {message}"
                )
                raise ValueError(msg)
            message = (message["role"], message["content"])
        if len(message) != 2:
            msg = f"Expected 2-tuple of (role, template), got {message}"
            raise ValueError(msg)
        message_type_str, template = message
        if isinstance(message_type_str, str):
            message_ = _create_template_from_message_type(
                message_type_str, template, template_format=template_format
            )
        else:
            message_ = message_type_str(
                prompt=PromptTemplate.from_template(
                    cast("str", template), template_format=template_format
                )
            )
    else:
        msg = f"Unsupported message type: {type(message)}"
        raise NotImplementedError(msg)

    return message_


# For backwards compat:
_convert_to_message = _convert_to_message_template
