# Prompt Templates API Reference

## Overview

Prompt templates provide a structured way to format inputs for language models. LangChain supports multiple template types for different use cases:

- **PromptTemplate**: String-based templates for text completion models
- **ChatPromptTemplate**: Message-based templates for chat models
- **MessagesPlaceholder**: Dynamic placeholder for inserting message lists
- **BasePromptTemplate**: Abstract base class for all prompt templates

Templates support variable substitution, partial application, and LCEL composition patterns for building complex chains.

Source: `libs/core/langchain_core/prompts/`

---

## PromptTemplate

```python
class PromptTemplate(StringPromptTemplate)
```

String-based prompt template for language models. Accepts a template string with variable placeholders and formats it with provided values.

**Source**: `libs/core/langchain_core/prompts/prompt.py:24`

### Fields

#### template
```python
template: str
```
The prompt template string containing placeholders for variables.

#### template_format
```python
template_format: PromptTemplateFormat = "f-string"
```
Format of the prompt template. Options:
- `"f-string"` (default): Python f-string format using `{variable}` syntax
- `"jinja2"`: Jinja2 template format (⚠️ Security Warning: See below)
- `"mustache"`: Mustache template format using `{{variable}}` syntax

**Security Warning for Jinja2**:
Prefer `template_format="f-string"` instead of `template_format="jinja2"`. Never accept jinja2 templates from untrusted sources as they may lead to arbitrary Python code execution. As of LangChain 0.0.329, Jinja2 templates use SandboxedEnvironment by default, but this should be treated as best-effort security, not a guarantee.

#### input_variables
```python
input_variables: list[str]
```
List of variable names required as inputs. Auto-extracted from template if not explicitly provided.

#### partial_variables
```python
partial_variables: dict[str, Any] = {}
```
Dictionary of variables that are pre-filled. Useful for creating reusable templates with some values already set.

#### validate_template
```python
validate_template: bool = False
```
Whether to validate the template format and variables during initialization.

### Factory Methods

#### from_template (Recommended)
```python
@classmethod
def from_template(
    cls,
    template: str,
    *,
    template_format: PromptTemplateFormat = "f-string",
    partial_variables: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PromptTemplate
```

Create a prompt template from a template string. This is the recommended factory method.

**Args**:
- `template` (str): The template string with variable placeholders
- `template_format` (PromptTemplateFormat): Format of template - `"f-string"`, `"jinja2"`, or `"mustache"`. Defaults to `"f-string"`
- `partial_variables` (dict[str, Any] | None): Variables to partially fill in the template. For example, if template is `"{variable1} {variable2}"` and `partial_variables` is `{"variable1": "foo"}`, final prompt becomes `"foo {variable2}"`
- `**kwargs`: Additional arguments passed to constructor

**Returns**:
- `PromptTemplate`: Configured prompt template instance

**Example**:
```python
from langchain_core.prompts import PromptTemplate

# Basic usage with f-string (recommended)
prompt = PromptTemplate.from_template("Say {foo}")
formatted = prompt.format(foo="bar")
# Output: "Say bar"

# With multiple variables
prompt = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}"
)
formatted = prompt.format(adjective="funny", content="chickens")
# Output: "Tell me a funny joke about chickens"
```

**Source**: `libs/core/langchain_core/prompts/prompt.py:251`

#### from_examples
```python
@classmethod
def from_examples(
    cls,
    examples: list[str],
    suffix: str,
    input_variables: list[str],
    example_separator: str = "\n\n",
    prefix: str = "",
    **kwargs: Any,
) -> PromptTemplate
```

Create a prompt from a list of examples with prefix and suffix.

**Args**:
- `examples` (list[str]): List of example strings to include in prompt
- `suffix` (str): String after examples, typically sets up user input
- `input_variables` (list[str]): Variable names the final prompt expects
- `example_separator` (str): Separator between examples. Defaults to `"\n\n"`
- `prefix` (str): String before examples. Defaults to empty string
- `**kwargs`: Additional constructor arguments

**Returns**:
- `PromptTemplate`: Template with examples, prefix, and suffix combined

**Example**:
```python
examples = [
    "Input: What is 2+2? Output: 4",
    "Input: What is 3+3? Output: 6",
]
prompt = PromptTemplate.from_examples(
    examples=examples,
    suffix="Input: {question} Output:",
    input_variables=["question"],
    prefix="Solve these math problems:",
)
formatted = prompt.format(question="What is 5+5?")
# Output:
# Solve these math problems:
# 
# Input: What is 2+2? Output: 4
# 
# Input: What is 3+3? Output: 6
# 
# Input: What is 5+5? Output:
```

**Source**: `libs/core/langchain_core/prompts/prompt.py:201`

#### from_file
```python
@classmethod
def from_file(
    cls,
    template_file: str | Path,
    encoding: str | None = None,
    **kwargs: Any,
) -> PromptTemplate
```

Load a prompt template from a file.

**Args**:
- `template_file` (str | Path): Path to file containing prompt template
- `encoding` (str | None): Text encoding for reading file. Uses OS default if not provided
- `**kwargs`: Additional constructor arguments

**Returns**:
- `PromptTemplate`: Template loaded from file contents

**Example**:
```python
from pathlib import Path

# File: prompts/greeting.txt contains "Hello {name}, welcome to {place}!"
prompt = PromptTemplate.from_file("prompts/greeting.txt")
formatted = prompt.format(name="Alice", place="Wonderland")
# Output: "Hello Alice, welcome to Wonderland!"
```

**Source**: `libs/core/langchain_core/prompts/prompt.py:231`

### Instance Methods

#### format
```python
def format(self, **kwargs: Any) -> str
```

Format the prompt template with provided variable values.

**Args**:
- `**kwargs`: Variable name-value pairs for template substitution

**Returns**:
- `str`: Formatted prompt string

**Raises**:
- `KeyError`: If required input variables are missing
- `ValueError`: If template format is invalid

**Example**:
```python
prompt = PromptTemplate.from_template("The {animal} says {sound}")
result = prompt.format(animal="dog", sound="woof")
# Output: "The dog says woof"
```

**Source**: `libs/core/langchain_core/prompts/prompt.py:188`

---

## ChatPromptTemplate

```python
class ChatPromptTemplate(BaseChatPromptTemplate)
```

Chat prompt template for chat models. Constructs prompts as sequences of messages (system, human, AI, etc.) rather than single strings.

**Source**: `libs/core/langchain_core/prompts/chat.py:730`

### Fields

#### messages
```python
messages: list[MessageLike]
```
List of messages consisting of message prompt templates or message objects. Each message can be:
1. `BaseMessagePromptTemplate` - A template that produces messages
2. `BaseMessage` - A concrete message (SystemMessage, HumanMessage, AIMessage, etc.)
3. `tuple` - Shorthand format: `("role", "template")`, e.g., `("human", "{input}")`

#### validate_template
```python
validate_template: bool = False
```
Whether to validate template variables during initialization.

### Factory Methods

#### from_messages (Recommended)
```python
@classmethod
def from_messages(
    cls,
    messages: Sequence[MessageLikeRepresentation],
    template_format: PromptTemplateFormat = "f-string",
) -> ChatPromptTemplate
```

Create a chat prompt template from a variety of message formats. This is the recommended factory method.

**Args**:
- `messages` (Sequence[MessageLikeRepresentation]): Sequence of message representations. Each message can be:
  - `BaseMessagePromptTemplate`: Message template object
  - `BaseMessage`: Concrete message instance
  - `tuple[str, str]`: `(message_type, template)`, e.g., `("human", "{user_input}")`
  - `tuple[type, str]`: `(MessageClass, template)`, e.g., `(HumanMessage, "Hello")`
  - `str`: Shorthand for `("human", template)`
- `template_format` (PromptTemplateFormat): Template format - `"f-string"`, `"jinja2"`, or `"mustache"`. Defaults to `"f-string"`

**Returns**:
- `ChatPromptTemplate`: Configured chat prompt template

**Message Type Shortcuts**:
- `"system"`: SystemMessage
- `"human"` or `"user"`: HumanMessage
- `"ai"` or `"assistant"`: AIMessage
- `"placeholder"`: MessagesPlaceholder (for dynamic message lists)

**Example**:
```python
from langchain_core.prompts import ChatPromptTemplate

# Basic usage with tuple format
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named {name}."),
    ("human", "{user_input}"),
])
prompt_value = template.invoke({
    "name": "Bob",
    "user_input": "What is your name?"
})
# Output: ChatPromptValue with:
#   SystemMessage(content="You are a helpful assistant named Bob.")
#   HumanMessage(content="What is your name?")

# Multi-turn conversation template
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])
prompt_value = template.invoke({
    "name": "Bob",
    "user_input": "What is your name?"
})
# Output: ChatPromptValue with 4 messages
```

**Source**: `libs/core/langchain_core/prompts/chat.py:1092`

#### from_template
```python
@classmethod
def from_template(cls, template: str, **kwargs: Any) -> ChatPromptTemplate
```

Create a chat template from a single template string (assumed to be human message).

**Args**:
- `template` (str): Template string for human message
- `**kwargs`: Additional constructor arguments

**Returns**:
- `ChatPromptTemplate`: Template with single human message

**Example**:
```python
template = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
prompt_value = template.invoke({"topic": "chickens"})
# Output: ChatPromptValue with:
#   HumanMessage(content="Tell me a joke about chickens")
```

**Source**: `libs/core/langchain_core/prompts/chat.py:1074`

### Instance Methods

#### format_messages
```python
def format_messages(self, **kwargs: Any) -> list[BaseMessage]
```

Format the chat template into a list of finalized messages.

**Args**:
- `**kwargs`: Variable name-value pairs for template substitution

**Returns**:
- `list[BaseMessage]`: List of formatted message objects

**Raises**:
- `ValueError`: If messages are of unexpected types or required variables missing

**Example**:
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are {role}"),
    ("human", "{question}"),
])
messages = template.format_messages(role="a teacher", question="What is 2+2?")
# Output: [
#   SystemMessage(content="You are a teacher"),
#   HumanMessage(content="What is 2+2?"),
# ]
```

**Source**: `libs/core/langchain_core/prompts/chat.py:1138`

#### partial
```python
def partial(self, **kwargs: Any) -> ChatPromptTemplate
```

Create a new ChatPromptTemplate with some input variables pre-filled.

**Args**:
- `**kwargs`: Variable name-value pairs to pre-fill. Must be subset of input_variables

**Returns**:
- `ChatPromptTemplate`: New template with partial variables set

**Example**:
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant named {name}."),
    ("human", "Hi I'm {user}"),
    ("ai", "Hi there, {user}, I'm {name}."),
    ("human", "{input}"),
])
# Pre-fill user and name
template2 = template.partial(user="Lucy", name="R2D2")

# Now only need to provide 'input'
prompt_value = template2.invoke({"input": "hello"})
# Output: ChatPromptValue with:
#   SystemMessage(content="You are an AI assistant named R2D2.")
#   HumanMessage(content="Hi I'm Lucy")
#   AIMessage(content="Hi there, Lucy, I'm R2D2.")
#   HumanMessage(content="hello")
```

**Source**: `libs/core/langchain_core/prompts/chat.py:1194`

#### append
```python
def append(self, message: MessageLikeRepresentation) -> None
```

Append a message to the end of the chat template.

**Args**:
- `message` (MessageLikeRepresentation): Message to append in any supported format

**Example**:
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful"),
])
template.append(("human", "{question}"))
# Template now has 2 messages
```

**Source**: `libs/core/langchain_core/prompts/chat.py:1229`

#### extend
```python
def extend(self, messages: Sequence[MessageLikeRepresentation]) -> None
```

Extend the chat template with a sequence of messages.

**Args**:
- `messages` (Sequence[MessageLikeRepresentation]): Sequence of messages to append

**Example**:
```python
template = ChatPromptTemplate.from_messages([("system", "You are helpful")])
template.extend([
    ("human", "Question: {question}"),
    ("ai", "Let me help with that."),
])
# Template now has 3 messages
```

**Source**: `libs/core/langchain_core/prompts/chat.py:1237`

---

## MessagesPlaceholder

```python
class MessagesPlaceholder(BaseMessagePromptTemplate)
```

Placeholder for a dynamic list of messages. Used to insert variable-length message sequences into chat templates, such as conversation history.

**Source**: `libs/core/langchain_core/prompts/chat.py:55`

### Fields

#### variable_name
```python
variable_name: str
```
Name of the variable containing the message list.

#### optional
```python
optional: bool = False
```
If `True`, `format_messages()` can be called without providing this variable (returns empty list). If `False`, the variable must be provided even if it's an empty list.

#### n_messages
```python
n_messages: PositiveInt | None = None
```
Maximum number of messages to include. If `None`, includes all messages. If set, takes the last `n_messages` from the list.

### Constructor

```python
def __init__(
    self, 
    variable_name: str, 
    *, 
    optional: bool = False, 
    **kwargs: Any
) -> None
```

**Args**:
- `variable_name` (str): Name of variable containing messages
- `optional` (bool): Whether variable is optional. Defaults to `False`
- `**kwargs`: Additional arguments

**Example**:
```python
from langchain_core.prompts import MessagesPlaceholder

# Required placeholder
placeholder = MessagesPlaceholder("history")
messages = placeholder.format_messages(history=[
    ("human", "Hi"),
    ("ai", "Hello!"),
])
# Output: [HumanMessage(content="Hi"), AIMessage(content="Hello!")]

# Optional placeholder
placeholder = MessagesPlaceholder("history", optional=True)
messages = placeholder.format_messages()  # No error, returns []

# Limiting number of messages
placeholder = MessagesPlaceholder("history", n_messages=2)
messages = placeholder.format_messages(history=[
    ("system", "You are helpful"),
    ("human", "Hi"),
    ("ai", "Hello!"),
    ("human", "How are you?"),
])
# Output: Last 2 messages only:
#   [AIMessage(content="Hello!"), HumanMessage(content="How are you?")]
```

**Source**: `libs/core/langchain_core/prompts/chat.py:140`

### Usage in ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Using MessagesPlaceholder for conversation history
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

prompt_value = template.invoke({
    "history": [
        ("human", "what's 5 + 2"),
        ("ai", "5 + 2 is 7"),
    ],
    "question": "now multiply that by 4",
})
# Output: ChatPromptValue with:
#   SystemMessage(content="You are a helpful assistant.")
#   HumanMessage(content="what's 5 + 2")
#   AIMessage(content="5 + 2 is 7")
#   HumanMessage(content="now multiply that by 4")

# Shorthand tuple syntax for optional placeholder
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("placeholder", "{conversation}"),  # Equivalent to MessagesPlaceholder
])
```

**Source**: `libs/core/langchain_core/prompts/chat.py:55-126`

---

## BasePromptTemplate

```python
class BasePromptTemplate(RunnableSerializable[dict, PromptValue], ABC)
```

Abstract base class for all prompt templates. Provides common functionality including variable validation, partial application, invoke/ainvoke methods, and LCEL integration.

**Source**: `libs/core/langchain_core/prompts/base.py:42`

### Fields

#### input_variables
```python
input_variables: list[str]
```
List of variable names required as inputs to the prompt. These must be provided when invoking the prompt.

#### optional_variables
```python
optional_variables: list[str] = []
```
List of variable names that are optional (typically from optional MessagesPlaceholder). These are auto-inferred and need not be provided.

#### partial_variables
```python
partial_variables: Mapping[str, Any] = {}
```
Dictionary of partially filled variables. These populate the template so you don't need to pass them every time.

#### output_parser
```python
output_parser: BaseOutputParser | None = None
```
Output parser to apply to LLM response. If set, the prompt can be piped to an LLM and automatically parse the output.

### Core Methods

#### invoke
```python
def invoke(
    self, 
    input: dict, 
    config: RunnableConfig | None = None, 
    **kwargs: Any
) -> PromptValue
```

Invoke the prompt template with input variables.

**Args**:
- `input` (dict): Dictionary of variable name-value pairs. Must contain all required `input_variables`
- `config` (RunnableConfig | None): Configuration for runnable execution (callbacks, tags, metadata)
- `**kwargs`: Additional arguments

**Returns**:
- `PromptValue`: Formatted prompt value (StringPromptValue or ChatPromptValue)

**Raises**:
- `KeyError`: If required input variables are missing
- `TypeError`: If input is not a dict (unless template has single variable)

**Example**:
```python
prompt = PromptTemplate.from_template("Tell me about {topic}")
prompt_value = prompt.invoke({"topic": "Python"})
# Output: StringPromptValue(text="Tell me about Python")
```

**Source**: `libs/core/langchain_core/prompts/base.py:191`

#### ainvoke
```python
async def ainvoke(
    self, 
    input: dict, 
    config: RunnableConfig | None = None, 
    **kwargs: Any
) -> PromptValue
```

Async version of invoke.

**Args**:
- `input` (dict): Dictionary of input variables
- `config` (RunnableConfig | None): Runnable configuration
- `**kwargs`: Additional arguments

**Returns**:
- `PromptValue`: Formatted prompt value

**Example**:
```python
prompt = PromptTemplate.from_template("Translate {text} to {language}")
prompt_value = await prompt.ainvoke({
    "text": "Hello",
    "language": "Spanish"
})
```

**Source**: `libs/core/langchain_core/prompts/base.py:217`

#### partial
```python
def partial(self, **kwargs: str | Callable[[], str]) -> BasePromptTemplate
```

Return a new template with some input variables partially filled.

**Args**:
- `**kwargs`: Variable name-value pairs or callable factories for partial variables

**Returns**:
- `BasePromptTemplate`: New template instance with partial variables set

**Example**:
```python
prompt = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}"
)
# Partially fill 'adjective'
partial_prompt = prompt.partial(adjective="funny")

# Now only need to provide 'content'
result = partial_prompt.format(content="chickens")
# Output: "Tell me a funny joke about chickens"

# Can also use callables for dynamic values
import datetime
partial_prompt = prompt.partial(
    adjective=lambda: "funny",
    content=lambda: datetime.datetime.now().strftime("%A")
)
```

**Source**: `libs/core/langchain_core/prompts/base.py:265`

#### format_prompt
```python
@abstractmethod
def format_prompt(self, **kwargs: Any) -> PromptValue
```

Abstract method to create PromptValue from inputs. Implemented by subclasses.

**Args**:
- `**kwargs`: Variable name-value pairs

**Returns**:
- `PromptValue`: Formatted prompt value

**Source**: `libs/core/langchain_core/prompts/base.py:243`

#### format
```python
@abstractmethod
def format(self, **kwargs: Any) -> FormatOutputType
```

Abstract method to format prompt as string or messages. Implemented by subclasses.

**Args**:
- `**kwargs`: Variable name-value pairs

**Returns**:
- `FormatOutputType`: Formatted output (str for PromptTemplate, list[BaseMessage] for ChatPromptTemplate)

**Source**: `libs/core/langchain_core/prompts/base.py:288`

---

## Type Flow Documentation

### PromptTemplate Type Flow

```
Input: dict[str, Any]
  ↓
PromptTemplate.invoke()
  ↓
Variable substitution and formatting
  ↓
StringPromptValue
  ↓ (when piped to LLM)
str
```

**Example**:
```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Type flow: dict → PromptTemplate → StringPromptValue → LLM → str
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI()

# Dict[str, str] → PromptTemplate
chain = prompt | model
# chain is Runnable[Dict[str, str], AIMessage]

result = chain.invoke({"topic": "chickens"})
# result is AIMessage
```

### ChatPromptTemplate Type Flow

```
Input: dict[str, Any]
  ↓
ChatPromptTemplate.invoke()
  ↓
Message formatting and variable substitution
  ↓
ChatPromptValue (contains list[BaseMessage])
  ↓ (when piped to chat model)
AIMessage
  ↓ (with output parser)
Parsed output (str, dict, etc.)
```

**Example with LCEL Composition**:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Complete type flow through LCEL pipe
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{question}"),
])
model = ChatOpenAI()
output_parser = StrOutputParser()

# Compose: Dict → ChatPromptTemplate → ChatPromptValue → ChatOpenAI → AIMessage → StrOutputParser → str
chain = prompt | model | output_parser

# Type at each stage:
# prompt: Runnable[Dict[str, Any], ChatPromptValue]
# model: Runnable[ChatPromptValue, AIMessage]
# output_parser: Runnable[AIMessage, str]
# chain: Runnable[Dict[str, Any], str]

result = chain.invoke({"question": "What is 2+2?"})
# result: str = "2 + 2 equals 4."
```

---

## Input Validation Rules

### Variable Requirements

1. **Required Variables**: All variables in `input_variables` must be provided when invoking
2. **Optional Variables**: Variables in `optional_variables` need not be provided
3. **Partial Variables**: Variables in `partial_variables` are pre-filled and should not be provided again
4. **Auto-extraction**: `input_variables` are automatically extracted from template if not explicitly set

**Example**:
```python
prompt = PromptTemplate.from_template("Hello {name}, you are in {place}")
# input_variables automatically set to ["name", "place"]

# ✓ Valid - all variables provided
prompt.invoke({"name": "Alice", "place": "Wonderland"})

# ✗ Invalid - missing 'place'
prompt.invoke({"name": "Alice"})  # Raises KeyError

# ✓ Valid with partial
partial = prompt.partial(place="Wonderland")
partial.invoke({"name": "Alice"})  # Only need 'name' now
```

### Single Variable Shorthand

If a prompt has exactly one input variable, you can invoke it with a non-dict object:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful"),
    ("human", "{user_input}"),
])

# Both are equivalent:
prompt_value = template.invoke("Hello there!")
prompt_value = template.invoke({"user_input": "Hello there!"})
```

**Source**: `libs/core/langchain_core/prompts/base.py:147`

---

## Template Format Options

### f-string Format (Recommended)

Uses Python f-string syntax with `{variable}` placeholders.

```python
prompt = PromptTemplate.from_template(
    "Hello {name}, today is {day}",
    template_format="f-string"
)
```

**Advantages**:
- Most secure option
- Familiar Python syntax
- Default format

### Jinja2 Format (⚠️ Use with Caution)

Uses Jinja2 template syntax with `{{ variable }}` placeholders and control structures.

```python
prompt = PromptTemplate.from_template(
    "Hello {{ name }}, {% if urgent %}URGENT{% endif %}",
    template_format="jinja2"
)
```

**Security Warning**: Never use jinja2 templates with untrusted input. Even with SandboxedEnvironment, this should be treated as best-effort security, not a guarantee. Prefer f-string format.

**Source**: `libs/core/langchain_core/prompts/string.py:30`

### Mustache Format

Uses Mustache syntax with `{{variable}}` placeholders.

```python
prompt = PromptTemplate.from_template(
    "Hello {{name}}, welcome to {{place}}",
    template_format="mustache"
)
```

**Note**: Mustache templates cannot be validated during initialization.

---

## LCEL Composition Patterns

Prompt templates integrate seamlessly with LangChain Expression Language (LCEL) for building chains.

### Basic Chain Pattern

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}"),
    ("human", "{input}"),
])
model = ChatOpenAI()
output_parser = StrOutputParser()

# LCEL pipe operator creates a chain
chain = prompt | model | output_parser

# Invoke the entire chain
result = chain.invoke({
    "role": "helpful assistant",
    "input": "What is LangChain?"
})
# result is a string
```

### Sequential Chain Pattern

```python
# First chain: Generate a topic
topic_prompt = ChatPromptTemplate.from_template(
    "Generate a random topic about {category}"
)
topic_chain = topic_prompt | model | StrOutputParser()

# Second chain: Use the topic
joke_prompt = ChatPromptTemplate.from_template(
    "Tell me a joke about {topic}"
)
joke_chain = joke_prompt | model | StrOutputParser()

# Can't directly pipe because output of first doesn't match input of second
# Use RunnablePassthrough or lambda to adapt
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"category": RunnablePassthrough()}
    | topic_chain
    | (lambda topic: {"topic": topic})
    | joke_chain
)

result = chain.invoke("science")
```

### Parallel Chain Pattern

```python
from langchain_core.runnables import RunnableParallel

# Create multiple chains that run in parallel
summary_prompt = ChatPromptTemplate.from_template("Summarize: {text}")
sentiment_prompt = ChatPromptTemplate.from_template(
    "What is the sentiment of: {text}"
)

summary_chain = summary_prompt | model | StrOutputParser()
sentiment_chain = sentiment_prompt | model | StrOutputParser()

# Run both chains in parallel
parallel_chain = RunnableParallel(
    summary=summary_chain,
    sentiment=sentiment_chain,
)

result = parallel_chain.invoke({"text": "LangChain is amazing!"})
# result = {
#     "summary": "...",
#     "sentiment": "positive"
# }
```

---

## Common Usage Patterns

### Reusable Templates with Partials

```python
# Create a base template
base_template = ChatPromptTemplate.from_messages([
    ("system", "You are {role}. You speak in {style}."),
    ("human", "{input}"),
])

# Create specialized versions with partials
teacher_template = base_template.partial(
    role="a teacher",
    style="simple terms"
)
poet_template = base_template.partial(
    role="a poet",
    style="verse"
)

# Now only need to provide 'input'
teacher_response = teacher_template.invoke({"input": "Explain gravity"})
poet_response = poet_template.invoke({"input": "Explain gravity"})
```

### Dynamic Few-Shot Examples

```python
from langchain_core.prompts import FewShotPromptTemplate

example_prompt = PromptTemplate.from_template(
    "Input: {input}\nOutput: {output}"
)

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3+3", "output": "6"},
]

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

result = few_shot_prompt.format(input="5+5")
# Output:
# Input: 2+2
# Output: 4
#
# Input: 3+3
# Output: 6
#
# Input: 5+5
# Output:
```

### Conversation with Memory

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
])

# First turn - no history
prompt1 = template.invoke({
    "input": "Hi, I'm Alice",
})

# Second turn - with history
prompt2 = template.invoke({
    "chat_history": [
        ("human", "Hi, I'm Alice"),
        ("ai", "Hello Alice! How can I help you today?"),
    ],
    "input": "What's my name?",
})
```

---

## Related APIs

- **Message Types**: See [Message Types API](./message-types.md) for HumanMessage, AIMessage, SystemMessage, etc.
- **Output Parsers**: See [Output Parsers API](../utilities/output-parsers.md) for parsing LLM outputs
- **Runnables**: See [Runnables API](../runnables/base.md) for LCEL composition details
- **Few-Shot Prompts**: See `FewShotPromptTemplate` for dynamic example selection

---

## Source Code References

- `BasePromptTemplate`: `libs/core/langchain_core/prompts/base.py:42`
- `PromptTemplate`: `libs/core/langchain_core/prompts/prompt.py:24`
- `ChatPromptTemplate`: `libs/core/langchain_core/prompts/chat.py:730`
- `MessagesPlaceholder`: `libs/core/langchain_core/prompts/chat.py:55`
- `StringPromptTemplate`: `libs/core/langchain_core/prompts/string.py`
- Template Format Types: `libs/core/langchain_core/prompts/string.py:27`

---

*Last Updated*: 2024
*LangChain Version*: 1.0+
