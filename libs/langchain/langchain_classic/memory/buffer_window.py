from typing import Any

from langchain_core._api import deprecated
from langchain_core.messages import BaseMessage, get_buffer_string
from typing_extensions import override

from langchain_classic.memory.chat_memory import BaseChatMemory


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationBufferWindowMemory(BaseChatMemory):
    """Memory that maintains a sliding window of the last k conversation turns.

    This memory implementation stores only the most recent k conversation turns,
    where each turn consists of a human message followed by an AI message. The
    total number of messages stored is k * 2 (k human messages + k AI messages).
    
    When the conversation exceeds k turns, the oldest messages are automatically
    dropped from the buffer, maintaining a fixed-size window of recent context.
    This is useful for limiting memory usage and context length in long-running
    conversations.

    **DEPRECATED**: This class is deprecated as of version 0.3.1 and will be
    removed in version 1.0.0. Please see the migration guide at:
    https://python.langchain.com/docs/versions/migrating_memory/

    Attributes:
        k: Number of conversation turns to keep in the buffer. Each turn contains
            one human message and one AI message, so the total message count is k*2.
            Default is 5 turns (10 messages).
        human_prefix: Prefix used when formatting human messages as strings.
            Default is "Human".
        ai_prefix: Prefix used when formatting AI messages as strings.
            Default is "AI".
        memory_key: Key name used to store the conversation history in the
            returned dictionary. Default is "history".
        return_messages: If True, returns conversation history as List[BaseMessage].
            If False, returns as formatted string. Inherited from BaseChatMemory.

    Note:
        This memory is non-persistent (stored in-memory only) and non-thread-safe.
        Messages are dropped silently when the window size is exceeded.

    Example:
        >>> from langchain_classic.memory import ConversationBufferWindowMemory
        >>> # Keep only the last 2 conversation turns (4 messages)
        >>> memory = ConversationBufferWindowMemory(k=2)
        >>> memory.save_context({"input": "Hi there!"}, {"output": "Hello!"})
        >>> memory.save_context({"input": "How are you?"}, {"output": "I'm good!"})
        >>> memory.save_context({"input": "What's your name?"}, {"output": "I'm an AI"})
        >>> # Only the last 2 turns are retained
        >>> memory.load_memory_variables({})
        {'history': 'Human: How are you?\\nAI: I\\'m good!\\nHuman: What\\'s your name?\\nAI: I\\'m an AI'}

        >>> # Using with return_messages=True
        >>> memory = ConversationBufferWindowMemory(k=3, return_messages=True)
        >>> memory.save_context({"input": "Hello"}, {"output": "Hi!"})
        >>> result = memory.load_memory_variables({})
        >>> # Returns List[BaseMessage] instead of string
        >>> len(result['history'])  # 2 messages (1 turn)
        2

    Source: libs/langchain/langchain_classic/memory/buffer_window.py:18
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:
    k: int = 5
    """Number of messages to store in buffer."""

    @property
    def buffer(self) -> str | list[BaseMessage]:
        """Return the conversation buffer in the appropriate format.

        Returns the conversation history as either a formatted string or a list
        of BaseMessage objects, depending on the return_messages attribute.

        Returns:
            Union[str, List[BaseMessage]]: The conversation buffer containing the
                last k*2 messages (k conversation turns). Format depends on
                return_messages setting:
                - If return_messages is False: Returns formatted string with
                  human_prefix and ai_prefix labels
                - If return_messages is True: Returns List[BaseMessage] objects

        Example:
            >>> memory = ConversationBufferWindowMemory(k=2, return_messages=False)
            >>> memory.save_context({"input": "Hi"}, {"output": "Hello"})
            >>> buffer = memory.buffer
            >>> isinstance(buffer, str)
            True
            >>> memory.return_messages = True
            >>> buffer = memory.buffer
            >>> isinstance(buffer, list)
            True

        Source: libs/langchain/langchain_classic/memory/buffer_window.py:31
        """
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """Return the conversation buffer as a formatted string.

        Retrieves the last k*2 messages from chat_memory and formats them as a
        string with human_prefix and ai_prefix labels. This property is used
        internally when return_messages is False.

        The window calculation uses messages[-k*2:] to select the most recent
        k conversation turns (each turn = 1 human message + 1 AI message).

        Returns:
            str: Formatted conversation history with each message prefixed by
                human_prefix or ai_prefix, separated by newlines. Returns empty
                string if k=0 or no messages exist.

        Example:
            >>> memory = ConversationBufferWindowMemory(k=2)
            >>> memory.save_context({"input": "Hi"}, {"output": "Hello"})
            >>> memory.save_context({"input": "Bye"}, {"output": "Goodbye"})
            >>> memory.buffer_as_str
            'Human: Hi\\nAI: Hello\\nHuman: Bye\\nAI: Goodbye'

            >>> # With k=0, returns empty string
            >>> memory_empty = ConversationBufferWindowMemory(k=0)
            >>> memory_empty.save_context({"input": "Hi"}, {"output": "Hello"})
            >>> memory_empty.buffer_as_str
            ''

        Source: libs/langchain/langchain_classic/memory/buffer_window.py:36
        """
        # Select last k*2 messages (k conversation turns) from the sliding window
        messages = self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []
        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """Return the conversation buffer as a list of BaseMessage objects.

        Retrieves the last k*2 messages from chat_memory as BaseMessage objects.
        This property is used internally when return_messages is True. The sliding
        window behavior ensures only the most recent k conversation turns are retained.

        Returns:
            List[BaseMessage]: List of the most recent k*2 messages (k conversation
                turns), where each turn contains one HumanMessage followed by one
                AIMessage. Returns empty list if k=0 or no messages exist.

        Example:
            >>> from langchain_core.messages import HumanMessage, AIMessage
            >>> memory = ConversationBufferWindowMemory(k=2, return_messages=True)
            >>> memory.save_context({"input": "Hi"}, {"output": "Hello"})
            >>> memory.save_context({"input": "How are you?"}, {"output": "Good"})
            >>> messages = memory.buffer_as_messages
            >>> len(messages)
            4
            >>> isinstance(messages[0], HumanMessage)
            True
            >>> isinstance(messages[1], AIMessage)
            True

            >>> # With k=0, window size is zero
            >>> memory_empty = ConversationBufferWindowMemory(k=0)
            >>> memory_empty.save_context({"input": "Hi"}, {"output": "Hello"})
            >>> memory_empty.buffer_as_messages
            []

        Source: libs/langchain/langchain_classic/memory/buffer_window.py:46
        """
        # Apply sliding window: select last k*2 messages (k conversation turns)
        return self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load conversation history from the sliding window buffer.

        Retrieves the last k conversation turns from memory and returns them in
        a dictionary format suitable for use in chain inputs. The buffer contains
        at most k*2 messages (k human messages + k AI messages).

        Args:
            inputs: Input dictionary from the chain. This parameter is required by
                the BaseMemory interface but is not used by this implementation.
                The sliding window memory always returns the same buffer regardless
                of inputs.
                Type: Dict[str, Any]

        Returns:
            Dict[str, Any]: Dictionary containing the conversation history with
                structure:
                {
                    memory_key: buffer
                }
                Where:
                - memory_key (str): The key name for storing history (default "history")
                - buffer (Union[str, List[BaseMessage]]): The conversation buffer
                  containing the last k*2 messages. Format depends on return_messages:
                  * If return_messages=False: Formatted string with prefixes
                  * If return_messages=True: List[BaseMessage] objects

        Raises:
            No exceptions are raised by this method. Even if k=0 or no messages
            exist, returns a valid dictionary with an empty buffer.

        Example:
            >>> memory = ConversationBufferWindowMemory(k=2)
            >>> memory.save_context({"input": "Hi there"}, {"output": "Hello"})
            >>> memory.save_context({"input": "How are you?"}, {"output": "Great!"})
            >>> # Load memory - inputs parameter is ignored
            >>> result = memory.load_memory_variables({})
            >>> result
            {'history': 'Human: Hi there\\nAI: Hello\\nHuman: How are you?\\nAI: Great!'}

            >>> # Using with a chain - typical usage pattern
            >>> from langchain_classic.chains import LLMChain
            >>> memory_with_messages = ConversationBufferWindowMemory(
            ...     k=3, return_messages=True, memory_key="chat_history"
            ... )
            >>> memory_with_messages.save_context(
            ...     {"input": "What is 2+2?"}, {"output": "4"}
            ... )
            >>> result = memory_with_messages.load_memory_variables({"input": "new query"})
            >>> # Returns BaseMessage list under 'chat_history' key
            >>> isinstance(result['chat_history'], list)
            True
            >>> len(result['chat_history'])  # 2 messages (1 turn of human + AI)
            2

            >>> # With k=0, returns empty buffer
            >>> memory_zero = ConversationBufferWindowMemory(k=0)
            >>> memory_zero.save_context({"input": "test"}, {"output": "response"})
            >>> memory_zero.load_memory_variables({})
            {'history': ''}

        Note:
            Type flow: inputs (Dict[str, Any]) → [ignored] → self.buffer retrieval
            → Dict[str, Union[str, List[BaseMessage]]] output

        Source: libs/langchain/langchain_classic/memory/buffer_window.py:59
        """
        # Retrieve buffer (last k*2 messages) and return in dictionary format
        return {self.memory_key: self.buffer}
