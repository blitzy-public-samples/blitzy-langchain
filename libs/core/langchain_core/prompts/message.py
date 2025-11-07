"""Message prompt templates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langchain_core.load import Serializable
from langchain_core.messages import BaseMessage
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from langchain_core.prompts.chat import ChatPromptTemplate


class BaseMessagePromptTemplate(Serializable, ABC):
    """Base class for message prompt templates.
    
    This abstract base class defines the interface for all message prompt 
    templates in LangChain. Message prompt templates are used to construct 
    BaseMessage objects dynamically from input variables, enabling flexible 
    and reusable prompt composition patterns.
    
    Subclasses must implement:
        - format_messages(): Synchronous method to format messages from kwargs
        - input_variables: Property listing required input variable names
    
    Key Features:
        - Serializable: Supports serialization/deserialization for persistence
        - Composable: Can be combined with other templates using + operator
        - Type-safe: Returns strongly-typed list[BaseMessage]
        - Async support: Provides async formatting via aformat_messages()
    
    Common implementations include:
        - MessagesPlaceholder: Pass through list of pre-constructed messages
        - HumanMessagePromptTemplate: Format human messages with variables
        - SystemMessagePromptTemplate: Format system messages with variables
        - AIMessagePromptTemplate: Format AI messages with variables
    
    Usage Pattern:
        1. Create template instance with input_variables
        2. Call format_messages(**kwargs) with variable values
        3. Receive list[BaseMessage] suitable for chat model input
    
    Integration:
        - Typically used within ChatPromptTemplate compositions
        - Can be combined with other templates using __add__ operator
        - Integrates with LCEL pipe operator for chain composition
    
    Source: libs/core/langchain_core/prompts/message.py:16-17
    
    Example:
        ```python
        from langchain_core.prompts import HumanMessagePromptTemplate
        
        # Create a template (concrete subclass implementation)
        template = HumanMessagePromptTemplate.from_template(
            "Tell me about {topic}"
        )
        
        # Format messages with input variables
        messages = template.format_messages(topic="Python")
        # Returns: [HumanMessage(content="Tell me about Python")]
        ```
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return True as this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "prompts", "chat"]`
        """
        return ["langchain", "prompts", "chat"]

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format messages from input variables.
        
        This abstract method must be implemented by all subclasses to convert
        keyword arguments containing template variables into a list of formatted
        BaseMessage objects ready for chat model consumption.
        
        The method performs template variable substitution, type conversion, and
        message construction according to the specific template implementation.
        All input variables declared in input_variables property must be provided
        in kwargs, unless marked as optional in the template configuration.

        Args:
            **kwargs: Keyword arguments providing values for template variables.
                Each key should correspond to a variable name in input_variables.
                Values can be strings, lists, or other types depending on the
                specific template implementation requirements.
                
                Common patterns:
                - String values: Substituted into template text
                - List[MessageLike]: Passed through for MessagesPlaceholder
                - Dict values: Used for structured message content
                
                Missing required variables will raise KeyError.
                Extra variables not in input_variables are typically ignored.

        Returns:
            list[BaseMessage]: Formatted messages ready for chat model input.
                Typically contains one or more of:
                - HumanMessage: User input messages
                - AIMessage: Assistant response messages  
                - SystemMessage: System instruction messages
                - ChatMessage: Generic role-based messages
                
                Empty list may be returned for optional placeholders with no input.
                Order of messages matches template definition order.

        Raises:
            KeyError: If required input variable is missing from kwargs and not
                marked as optional in the template configuration.
            ValueError: If provided variable value has incorrect type or format
                for the template's requirements.
            TypeError: If kwargs contains non-serializable values that cannot
                be processed by the template formatter.
        
        Source: libs/core/langchain_core/prompts/message.py:34-42
        
        Example:
            ```python
            # Subclass implementation example
            class CustomMessageTemplate(BaseMessagePromptTemplate):
                template_text: str
                
                def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
                    formatted_text = self.template_text.format(**kwargs)
                    return [HumanMessage(content=formatted_text)]
                
                @property
                def input_variables(self) -> list[str]:
                    return ["name", "question"]
            
            # Usage
            template = CustomMessageTemplate(
                template_text="Hi {name}, {question}"
            )
            messages = template.format_messages(
                name="Alice",
                question="how are you?"
            )
            # Returns: [HumanMessage(content="Hi Alice, how are you?")]
            ```
        """

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format messages from input variables.
        
        Asynchronous version of format_messages() that allows message formatting
        within async contexts and event loops. The default implementation calls
        the synchronous format_messages() method, but subclasses can override
        to provide truly asynchronous formatting if needed (e.g., async template
        loading, async variable validation, or async content generation).
        
        Use this method when:
        - Operating within an async function or coroutine
        - Building async chains with LCEL async operators (ainvoke, astream)
        - Formatting messages as part of async agent workflows
        - Need to avoid blocking the event loop during formatting
        
        The method signature and behavior matches format_messages() exactly,
        with the only difference being async/await compatibility.

        Args:
            **kwargs: Keyword arguments providing values for template variables.
                Identical to format_messages() - each key corresponds to a 
                variable name in input_variables property.
                
                All type requirements, constraints, and validation rules from
                format_messages() apply here as well.
                
                Missing required variables will raise KeyError.
                Invalid types or formats will raise ValueError or TypeError.

        Returns:
            list[BaseMessage]: Formatted messages ready for chat model input.
                Return structure identical to format_messages():
                - Contains BaseMessage subclass instances
                - Order matches template definition
                - May be empty list for optional placeholders
                
                Can be directly passed to chat model ainvoke() or astream().

        Raises:
            KeyError: If required input variable missing from kwargs.
            ValueError: If variable value has incorrect type or format.
            TypeError: If kwargs contains non-serializable values.
            
            Note: Subclasses implementing truly async formatting may raise
            additional exceptions related to async operations (e.g., 
            asyncio.TimeoutError, network errors during async content loading).
        
        Source: libs/core/langchain_core/prompts/message.py:44-53
        
        Example:
            ```python
            import asyncio
            from langchain_core.prompts import ChatPromptTemplate
            
            async def format_async_prompt():
                template = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant"),
                    ("human", "{user_input}")
                ])
                
                # Use aformat_messages in async context
                messages = await template.aformat_messages(
                    user_input="What is async/await?"
                )
                return messages
            
            # Run in event loop
            messages = asyncio.run(format_async_prompt())
            # Returns: [SystemMessage(...), HumanMessage(...)]
            ```
        """
        return self.format_messages(**kwargs)

    @property
    @abstractmethod
    def input_variables(self) -> list[str]:
        """Input variables required by this prompt template.
        
        Returns the list of variable names that must (or may) be provided when
        calling format_messages() or aformat_messages(). These variable names
        define the template's input schema and are used for:
        
        - Validation: Checking that required variables are provided
        - Type hints: Documenting expected input structure
        - Chain composition: Determining input/output compatibility in LCEL
        - Serialization: Persisting template configuration
        
        Requirements:
            - Must return list[str] with variable names as strings
            - Variable names should be valid Python identifiers
            - List should be deterministic (same order each call)
            - Empty list is valid for templates with no input variables
            - Optional variables may be included or documented separately
        
        Validation Behavior:
            - When format_messages() is called, presence of these variables
              in kwargs is typically checked (unless marked optional)
            - Missing required variables raise KeyError
            - Extra variables in kwargs are usually ignored
            - Some template types allow partial variable sets
        
        The property is used by:
            - ChatPromptTemplate for aggregating variables from all message
              templates in the composition
            - LCEL chain validation to ensure input/output compatibility
            - Serialization systems to persist template configuration
            - IDE type checkers and documentation generators

        Returns:
            list[str]: Variable names required for message formatting.
                Each string is a variable name that corresponds to a key
                expected in the kwargs dict when format_messages() is called.
                
                Examples:
                - ["user_input"] - Single required variable
                - ["name", "question", "context"] - Multiple variables
                - [] - No input variables (static template)
                - ["history"] - Common for MessagesPlaceholder templates
        
        Source: libs/core/langchain_core/prompts/message.py:55-62
        
        Example:
            ```python
            from langchain_core.prompts import HumanMessagePromptTemplate
            
            # Create template with variables in template string
            template = HumanMessagePromptTemplate.from_template(
                "Hello {name}, you asked: {question}"
            )
            
            # Check required input variables
            variables = template.input_variables
            print(variables)  # ["name", "question"]
            
            # Must provide all variables when formatting
            messages = template.format_messages(
                name="Alice",
                question="What is AI?"
            )
            # Success - all required variables provided
            
            # Missing variables raises error
            try:
                template.format_messages(name="Alice")
            except KeyError as e:
                print(f"Missing variable: {e}")
            ```
        """

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

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """Combine this message template with another to create a ChatPromptTemplate.
        
        Enables intuitive prompt composition using the + operator, allowing you to
        build complex multi-message prompt sequences by chaining message templates
        together. The operation creates a ChatPromptTemplate containing both
        templates' messages in order.
        
        This operator overload supports building prompts incrementally:
            template1 + template2 + template3 → ChatPromptTemplate with all messages
        
        Combination Rules:
            - Self is wrapped in ChatPromptTemplate as first message
            - Other is added to the ChatPromptTemplate (delegated to ChatPromptTemplate.__add__)
            - Result always returns ChatPromptTemplate, never BaseMessagePromptTemplate
            - Message order is preserved (self appears before other)
            - Input variables are merged from both templates
        
        Type Flexibility:
            The 'other' parameter accepts multiple types through ChatPromptTemplate's
            __add__ implementation:
            - BaseMessagePromptTemplate: Another message template
            - ChatPromptTemplate: Existing multi-message template
            - list[MessageLike]: List of message tuples or objects
            - str: Converted to HumanMessage automatically
        
        Common Usage Patterns:
            - System + Human: system_template + human_template
            - Multi-turn: system + human + ai + human
            - With placeholders: system + MessagesPlaceholder("history") + human

        Args:
            other: Template or message to combine with this template.
                Accepted types (handled by ChatPromptTemplate.__add__):
                - BaseMessagePromptTemplate: Adds another template's messages
                - ChatPromptTemplate: Merges all messages from the template
                - list: Converts list of message-like objects to templates
                - str: Converts string to HumanMessagePromptTemplate
                - tuple: Converts (role, content) to appropriate message type
                
                Type compatibility is validated by ChatPromptTemplate.

        Returns:
            ChatPromptTemplate: A new ChatPromptTemplate instance containing:
                - This template's message(s) as the first message(s)
                - Other's message(s) appended after
                - Merged input_variables from both templates (deduplicated)
                - Combined message formatting behavior
                
                The returned ChatPromptTemplate can be:
                - Used directly with format_messages() / aformat_messages()
                - Combined with more templates using additional + operations
                - Composed in LCEL chains with | pipe operator
                - Serialized and persisted

        Raises:
            TypeError: If 'other' has a type that ChatPromptTemplate.__add__ 
                cannot handle (rare, as ChatPromptTemplate is flexible).
            ValueError: If combining templates results in conflicting 
                input variable definitions or incompatible message structures.
        
        Source: libs/core/langchain_core/prompts/message.py:82-95
        
        Example:
            ```python
            from langchain_core.prompts import (
                SystemMessagePromptTemplate,
                HumanMessagePromptTemplate,
                MessagesPlaceholder,
            )
            
            # Create individual templates
            system = SystemMessagePromptTemplate.from_template(
                "You are a helpful {role}"
            )
            history = MessagesPlaceholder("chat_history", optional=True)
            human = HumanMessagePromptTemplate.from_template(
                "{user_input}"
            )
            
            # Combine with + operator
            full_prompt = system + history + human
            # Type: ChatPromptTemplate with 3 message templates
            
            # Format the combined template
            messages = full_prompt.format_messages(
                role="assistant",
                chat_history=[
                    ("human", "Hi!"),
                    ("ai", "Hello! How can I help?")
                ],
                user_input="Tell me about Python"
            )
            # Returns list with SystemMessage, 2 history messages, HumanMessage
            
            # Can continue adding
            followup = HumanMessagePromptTemplate.from_template("{followup}")
            extended = full_prompt + followup
            # Now has 4 message templates
            ```
        """
        # Import locally to avoid circular import.
        from langchain_core.prompts.chat import ChatPromptTemplate  # noqa: PLC0415

        prompt = ChatPromptTemplate(messages=[self])
        return prompt + other
