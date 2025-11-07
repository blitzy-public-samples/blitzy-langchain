from typing import Any

from langchain_core._api import deprecated
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils import pre_init
from typing_extensions import override

from langchain_classic.base_memory import BaseMemory
from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_classic.memory.utils import get_prompt_input_key


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationBufferMemory(BaseChatMemory):
    """Buffer for storing ALL conversation history without summarization or truncation.

    This memory class stores the complete conversation history by maintaining a list
    of messages (HumanMessage and AIMessage objects) in an in-memory buffer. Unlike
    memory types that summarize or truncate, ConversationBufferMemory preserves every
    message from the conversation, making it ideal for short conversations or when
    complete context is required.

    **Storage Behavior:**
        - Stores ALL messages without any summarization or truncation
        - Each interaction adds both a HumanMessage and AIMessage to the buffer
        - Messages persist only in memory (non-persistent across sessions)
        - NOT thread-safe for concurrent access

    **Memory Management:**
        The returned memory can be formatted in two ways based on the return_messages flag:
        - return_messages=False (default): Returns formatted string with human_prefix/ai_prefix
        - return_messages=True: Returns List[BaseMessage] for chat model chains

    **Context Window Warning:**
        This memory stores unbounded conversation history. For long conversations, the
        accumulated history may exceed the model's context window, causing failures.
        Consider using ConversationBufferWindowMemory or ConversationSummaryMemory
        for long-running conversations.

    Attributes:
        memory_key (str): Key name for injecting memory into chain inputs. Default: "history"
        human_prefix (str): Prefix for human messages in string format. Default: "Human"
        ai_prefix (str): Prefix for AI messages in string format. Default: "AI"
        return_messages (bool): If False, returns string; if True, returns List[BaseMessage].
            Default: False (inherited from BaseChatMemory)
        chat_memory (BaseChatMessageHistory): Internal message storage, defaults to
            InMemoryChatMessageHistory (inherited from BaseChatMemory)
        input_key (str | None): Specific key to extract from chain inputs. If None, auto-detected
            (inherited from BaseChatMemory)
        output_key (str | None): Specific key to extract from chain outputs. If None, auto-detected
            (inherited from BaseChatMemory)

    **Deprecation Notice:**
        This class is deprecated as of version 0.3.1 and will be removed in version 1.0.0.
        Please migrate to modern memory patterns. See migration guide:
        https://python.langchain.com/docs/versions/migrating_memory/

    Examples:
        Basic usage with string format (return_messages=False):
            >>> from langchain_classic.memory.buffer import ConversationBufferMemory
            >>> memory = ConversationBufferMemory()
            >>> memory.save_context({"input": "Hi there!"}, {"output": "Hello! How can I help?"})
            >>> memory.load_memory_variables({})
            {'history': 'Human: Hi there!\\nAI: Hello! How can I help?'}

        Usage with message list format (return_messages=True):
            >>> memory = ConversationBufferMemory(return_messages=True)
            >>> memory.save_context({"input": "Hi"}, {"output": "Hello"})
            >>> result = memory.load_memory_variables({})
            >>> result['history']  # Returns List[BaseMessage]
            [HumanMessage(content='Hi'), AIMessage(content='Hello')]

        Integration with chains:
            >>> from langchain_classic.chains import LLMChain
            >>> from langchain_classic.prompts import PromptTemplate
            >>> # Assuming llm is configured
            >>> template = "Context: {history}\\nHuman: {input}\\nAI:"
            >>> prompt = PromptTemplate(input_variables=["history", "input"], template=template)
            >>> chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
            >>> # Memory automatically injected into chain inputs

    Source: libs/langchain/langchain_classic/memory/buffer.py:21-29
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:

    # NOTE: save_context() method is inherited from BaseChatMemory
    # 
    # The inherited save_context(inputs, outputs) method:
    # - Extracts user input and AI output from the provided dictionaries
    # - Uses input_key/output_key for extraction (auto-detected if not set)
    # - Creates HumanMessage from inputs and AIMessage from outputs
    # - Appends both messages to chat_memory.messages list
    # - Called automatically after chain execution to persist conversation
    #
    # Args:
    #     inputs (dict[str, Any]): Chain inputs containing user message.
    #         If input_key is None, the key is auto-detected using get_prompt_input_key()
    #         which identifies the non-memory input key.
    #     outputs (dict[str, str]): Chain outputs containing AI response.
    #         If output_key is None and outputs has one key, that key is used.
    #         If multiple keys exist, raises ValueError unless output_key is specified.
    #
    # Raises:
    #     ValueError: If output_key is ambiguous (multiple output keys without
    #         explicit output_key configuration).
    #
    # Example:
    #     >>> memory = ConversationBufferMemory()
    #     >>> memory.save_context(
    #     ...     inputs={"question": "What is AI?"},
    #     ...     outputs={"answer": "AI is artificial intelligence"}
    #     ... )
    #     >>> # Internally creates and stores:
    #     >>> # HumanMessage(content="What is AI?")
    #     >>> # AIMessage(content="AI is artificial intelligence")
    #
    # Source: libs/langchain/langchain_classic/memory/chat_memory.py:74-82

    @property
    def buffer(self) -> Any:
        """Returns conversation buffer in the format specified by return_messages flag.

        Returns:
            Union[str, List[BaseMessage]]: If return_messages=False, returns formatted
                string with human_prefix and ai_prefix (e.g., "Human: Hi\\nAI: Hello").
                If return_messages=True, returns List[BaseMessage] containing HumanMessage
                and AIMessage objects in chronological order.

        Examples:
            String format (return_messages=False):
                >>> memory = ConversationBufferMemory()
                >>> memory.save_context({"input": "test"}, {"output": "response"})
                >>> memory.buffer
                'Human: test\\nAI: response'

            Message list format (return_messages=True):
                >>> memory = ConversationBufferMemory(return_messages=True)
                >>> memory.save_context({"input": "test"}, {"output": "response"})
                >>> memory.buffer
                [HumanMessage(content='test'), AIMessage(content='response')]

        Source: libs/langchain/langchain_classic/memory/buffer.py:35-38
        """
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    async def abuffer(self) -> Any:
        """String buffer of memory."""
        return (
            await self.abuffer_as_messages()
            if self.return_messages
            else await self.abuffer_as_str()
        )

    def _buffer_as_str(self, messages: list[BaseMessage]) -> str:
        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is True."""
        return self._buffer_as_str(self.chat_memory.messages)

    async def abuffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is True."""
        messages = await self.chat_memory.aget_messages()
        return self._buffer_as_str(messages)

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return self.chat_memory.messages

    async def abuffer_as_messages(self) -> list[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return await self.chat_memory.aget_messages()

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load conversation history from memory buffer for injection into chain inputs.

        This method retrieves the stored conversation history and returns it in a
        dictionary format that will be merged into the chain's input variables at
        execution time. The inputs parameter is required by the BaseMemory interface
        but is not used by this implementation.

        Args:
            inputs (dict[str, Any]): Input dictionary from the chain. This parameter is
                not used by ConversationBufferMemory but is required to conform to the
                BaseMemory interface. Memory loading is independent of current inputs.

        Returns:
            dict[str, Any]: Dictionary with a single key (memory_key, default "history")
                mapping to the conversation buffer. The buffer format depends on the
                return_messages setting:
                - If return_messages=False: {memory_key: str} with formatted conversation
                - If return_messages=True: {memory_key: List[BaseMessage]} with message objects

        Raises:
            No exceptions are raised by this method.

        Examples:
            With string format (return_messages=False):
                >>> memory = ConversationBufferMemory()
                >>> memory.save_context({"input": "What is 2+2?"}, {"output": "4"})
                >>> memory.load_memory_variables({})
                {'history': 'Human: What is 2+2?\\nAI: 4'}

            With message list format (return_messages=True):
                >>> memory = ConversationBufferMemory(return_messages=True)
                >>> memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
                >>> result = memory.load_memory_variables({})
                >>> result['history']
                [HumanMessage(content='Hello'), AIMessage(content='Hi there!')]

            Usage in chain execution context:
                >>> # Chain calls load_memory_variables before invoke
                >>> chain_inputs = {"question": "Follow-up query"}
                >>> memory_vars = memory.load_memory_variables(chain_inputs)
                >>> # Chain merges: {**chain_inputs, **memory_vars}
                >>> # Result: {"question": "Follow-up query", "history": "..."}

        Source: libs/langchain/langchain_classic/memory/buffer.py:82-85
        """
        return {self.memory_key: self.buffer}

    @override
    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Asynchronously load conversation history from memory buffer.

        This is the async counterpart to load_memory_variables(). It retrieves the
        conversation history asynchronously via the abuffer() method, which fetches
        messages from the chat_memory using aget_messages(). Use this method when
        working with async chains or when the memory backend supports async operations.

        Args:
            inputs (dict[str, Any]): Input dictionary from the chain. Not used by this
                implementation but required by the BaseMemory interface.

        Returns:
            dict[str, Any]: Dictionary with memory_key mapping to the buffer:
                - If return_messages=False: {memory_key: str}
                - If return_messages=True: {memory_key: List[BaseMessage]}

        Raises:
            No exceptions are raised by this method.

        Examples:
            Async usage with string format:
                >>> import asyncio
                >>> async def example():
                ...     memory = ConversationBufferMemory()
                ...     memory.save_context({"input": "Hi"}, {"output": "Hello"})
                ...     result = await memory.aload_memory_variables({})
                ...     return result
                >>> asyncio.run(example())
                {'history': 'Human: Hi\\nAI: Hello'}

            Usage with async chain execution:
                >>> # Async chain calls aload_memory_variables during ainvoke
                >>> async def run_async_chain(chain, memory):
                ...     memory_vars = await memory.aload_memory_variables({})
                ...     # Chain merges memory into inputs asynchronously

        Source: libs/langchain/langchain_classic/memory/buffer.py:87-91
        """
        buffer = await self.abuffer()
        return {self.memory_key: buffer}


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationStringBufferMemory(BaseMemory):
    """String-based memory buffer for storing conversation history as formatted text.

    This memory class stores conversation history as a single concatenated string,
    optimized for string-based conversations and traditional LLM chains (non-chat models).
    Unlike ConversationBufferMemory which stores structured message objects, this class
    maintains a simple string buffer with human/AI prefixes for each turn.

    **Key Differences from ConversationBufferMemory:**
        - Storage format: Single string vs List[BaseMessage]
        - Target use case: String-based LLM chains vs Chat model chains
        - Memory structure: Concatenated text vs Structured message objects
        - return_messages: Must be False (enforced by validator)

    **Storage Behavior:**
        - Accumulates conversation as newline-separated string
        - Format: "\\n{human_prefix}: {input}\\n{ai_prefix}: {output}"
        - Each interaction appends to the buffer string
        - Non-persistent (in-memory only, not saved across sessions)
        - NOT thread-safe for concurrent access

    **String Accumulation Pattern:**
        Initial: buffer = ""
        After turn 1: "\\nHuman: Hi\\nAI: Hello"
        After turn 2: "\\nHuman: Hi\\nAI: Hello\\nHuman: Bye\\nAI: Goodbye"

    Attributes:
        buffer (str): The accumulated conversation string. Default: ""
        memory_key (str): Key for injecting memory into chain inputs. Default: "history"
        human_prefix (str): Prefix for human messages. Default: "Human"
        ai_prefix (str): Prefix for AI responses. Default: "AI"
        input_key (str | None): Specific key to extract from chain inputs. If None,
            auto-detected using get_prompt_input_key(). Default: None
        output_key (str | None): Specific key to extract from chain outputs. If None
            and outputs has one key, that key is used. If multiple keys exist without
            explicit output_key, raises ValueError. Default: None

    **Deprecation Notice:**
        This class is deprecated as of version 0.3.1 and will be removed in version 1.0.0.
        Please migrate to modern memory patterns. See migration guide:
        https://python.langchain.com/docs/versions/migrating_memory/

    Examples:
        Basic usage with string accumulation:
            >>> from langchain_classic.memory.buffer import ConversationStringBufferMemory
            >>> memory = ConversationStringBufferMemory()
            >>> memory.save_context({"input": "Hi there"}, {"output": "Hello"})
            >>> memory.buffer
            '\\nHuman: Hi there\\nAI: Hello'
            >>> memory.save_context({"input": "How are you?"}, {"output": "I'm good"})
            >>> memory.buffer
            '\\nHuman: Hi there\\nAI: Hello\\nHuman: How are you?\\nAI: I\\'m good'

        Custom prefixes:
            >>> memory = ConversationStringBufferMemory(
            ...     human_prefix="User",
            ...     ai_prefix="Assistant"
            ... )
            >>> memory.save_context({"query": "test"}, {"response": "result"})
            >>> memory.buffer
            '\\nUser: test\\nAssistant: result'

        Explicit key configuration (for chains with multiple outputs):
            >>> memory = ConversationStringBufferMemory(
            ...     input_key="question",
            ...     output_key="answer"
            ... )
            >>> memory.save_context(
            ...     {"question": "Q1", "context": "..."},
            ...     {"answer": "A1", "sources": "..."}
            ... )
            >>> # Only "question" and "answer" keys are used

    Source: libs/langchain/langchain_classic/memory/buffer.py:102-113
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    buffer: str = ""
    output_key: str | None = None
    input_key: str | None = None
    memory_key: str = "history"  #: :meta private:

    @pre_init
    def validate_chains(cls, values: dict) -> dict:
        """Validate that return_messages is False for string-based memory.

        ConversationStringBufferMemory only supports string output format and cannot
        return structured message objects. This validator enforces that constraint
        during initialization.

        Args:
            cls: The class being initialized (ConversationStringBufferMemory)
            values (dict): Dictionary of field values provided during initialization

        Returns:
            dict: The validated values dictionary, unchanged if validation passes

        Raises:
            ValueError: If return_messages is set to True. ConversationStringBufferMemory
                requires return_messages=False because it stores conversation as a string,
                not as message objects. Use ConversationBufferMemory instead if you need
                return_messages=True functionality.

        Example:
            >>> # This will raise ValueError
            >>> try:
            ...     memory = ConversationStringBufferMemory(return_messages=True)
            ... except ValueError as e:
            ...     print(e)
            'return_messages must be False for ConversationStringBufferMemory'

        Source: libs/langchain/langchain_classic/memory/buffer.py:123-129
        """
        if values.get("return_messages", False):
            msg = "return_messages must be False for ConversationStringBufferMemory"
            raise ValueError(msg)
        return values

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Load conversation history string for injection into chain inputs.

        Returns the accumulated conversation buffer as a dictionary that will be merged
        into chain inputs. The buffer is always returned as a string (never as message
        objects) because return_messages is enforced to be False.

        Args:
            inputs (dict[str, Any]): Input dictionary from the chain. Not used by this
                implementation but required by the BaseMemory interface.

        Returns:
            dict[str, str]: Dictionary with single key (memory_key) mapping to the
                conversation buffer string. Structure: {memory_key: buffer_string}
                The buffer string contains the full conversation history formatted as:
                "\\n{human_prefix}: {input}\\n{ai_prefix}: {output}\\n..."

        Raises:
            No exceptions are raised by this method.

        Examples:
            Basic retrieval:
                >>> memory = ConversationStringBufferMemory()
                >>> memory.save_context({"input": "Hello"}, {"output": "Hi"})
                >>> memory.load_memory_variables({})
                {'history': '\\nHuman: Hello\\nAI: Hi'}

            After multiple interactions:
                >>> memory.save_context({"input": "How are you?"}, {"output": "Great!"})
                >>> result = memory.load_memory_variables({})
                >>> print(result['history'])
                
                Human: Hello
                AI: Hi
                Human: How are you?
                AI: Great!

            Integration with chain:
                >>> # Chain automatically calls load_memory_variables
                >>> chain_inputs = {"question": "What's next?"}
                >>> memory_vars = memory.load_memory_variables(chain_inputs)
                >>> # Merged inputs: {"question": "What's next?", "history": "..."}

        Source: libs/langchain/langchain_classic/memory/buffer.py:139-142
        """
        return {self.memory_key: self.buffer}

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Return history buffer."""
        return self.load_memory_variables(inputs)

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Append conversation turn to the string buffer with formatted prefixes.

        Extracts the user input and AI output from the provided dictionaries, formats
        them with human_prefix and ai_prefix, and appends the formatted strings to the
        internal buffer. This method is automatically called after chain execution to
        persist the conversation turn.

        **Key Resolution Logic:**
            1. Input key: Uses input_key if set, otherwise auto-detects using
               get_prompt_input_key() which finds the non-memory input key
            2. Output key: Uses output_key if set, otherwise expects exactly one output
               key (raises ValueError if multiple keys without explicit output_key)

        Args:
            inputs (dict[str, Any]): Chain inputs containing the user message. The method
                extracts the value from the key specified by input_key, or auto-detects
                the input key if input_key is None. Auto-detection excludes memory_variables
                and "stop" key.
            outputs (dict[str, str]): Chain outputs containing the AI response. If output_key
                is not specified and outputs contains multiple keys, raises ValueError.

        Returns:
            None: Modifies the internal buffer state in-place by appending the new
                conversation turn.

        Raises:
            ValueError: If output_key is None and outputs dictionary has multiple keys.
                Error message format: "One output key expected, got {outputs.keys()}"
            KeyError: If the specified input_key or output_key does not exist in the
                respective dictionaries.

        Examples:
            Basic usage with single input/output keys:
                >>> memory = ConversationStringBufferMemory()
                >>> memory.save_context(
                ...     inputs={"input": "What is AI?"},
                ...     outputs={"output": "AI is artificial intelligence"}
                ... )
                >>> memory.buffer
                '\\nHuman: What is AI?\\nAI: AI is artificial intelligence'

            Multiple interactions showing buffer growth:
                >>> memory = ConversationStringBufferMemory()
                >>> memory.save_context({"input": "Hi"}, {"output": "Hello"})
                >>> print(f"After turn 1: {repr(memory.buffer)}")
                After turn 1: '\\nHuman: Hi\\nAI: Hello'
                >>> memory.save_context({"input": "Bye"}, {"output": "Goodbye"})
                >>> print(f"After turn 2: {repr(memory.buffer)}")
                After turn 2: '\\nHuman: Hi\\nAI: Hello\\nHuman: Bye\\nAI: Goodbye'

            Explicit key configuration for multi-key dictionaries:
                >>> memory = ConversationStringBufferMemory(
                ...     input_key="question",
                ...     output_key="answer"
                ... )
                >>> memory.save_context(
                ...     inputs={"question": "Q?", "context": "extra data"},
                ...     outputs={"answer": "A", "sources": "refs"}
                ... )
                >>> memory.buffer  # Only question and answer keys used
                '\\nHuman: Q?\\nAI: A'

            Error case - ambiguous output key:
                >>> memory = ConversationStringBufferMemory()
                >>> try:
                ...     memory.save_context(
                ...         {"input": "test"},
                ...         {"output1": "A", "output2": "B"}
                ...     )
                ... except ValueError as e:
                ...     print(e)
                One output key expected, got dict_keys(['output1', 'output2'])

        Source: libs/langchain/langchain_classic/memory/buffer.py:148-163
        """
        # Determine input key: use configured input_key or auto-detect from inputs
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        
        # Determine output key: use configured output_key or validate single output
        if self.output_key is None:
            if len(outputs) != 1:
                msg = f"One output key expected, got {outputs.keys()}"
                raise ValueError(msg)
            output_key = next(iter(outputs.keys()))
        else:
            output_key = self.output_key
        
        # Format and append conversation turn to buffer
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        self.buffer += f"\n{human}\n{ai}"

    async def asave_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """Save context from this conversation to buffer."""
        return self.save_context(inputs, outputs)

    def clear(self) -> None:
        """Reset the conversation buffer to empty string.

        Clears all accumulated conversation history by resetting the buffer to an
        empty string. This is useful for starting a new conversation context or
        implementing conversation reset functionality.

        Returns:
            None: Modifies the internal buffer state in-place.

        Examples:
            Basic usage:
                >>> memory = ConversationStringBufferMemory()
                >>> memory.save_context({"input": "Hi"}, {"output": "Hello"})
                >>> memory.buffer
                '\\nHuman: Hi\\nAI: Hello'
                >>> memory.clear()
                >>> memory.buffer
                ''

            Starting fresh conversation:
                >>> memory.save_context({"input": "First chat"}, {"output": "Response"})
                >>> # ... conversation continues ...
                >>> memory.clear()  # Start new conversation
                >>> memory.save_context({"input": "New chat"}, {"output": "New response"})
                >>> memory.buffer
                '\\nHuman: New chat\\nAI: New response'

        Source: libs/langchain/langchain_classic/memory/buffer.py:173-175
        """
        self.buffer = ""

    @override
    async def aclear(self) -> None:
        self.clear()
