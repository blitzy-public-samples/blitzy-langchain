# LangChain Core Prompts Module

## Module Overview

The `langchain_core.prompts` module provides a comprehensive framework for creating **reusable, parameterized prompts** for language models with **type safety** and **composability**. Prompt templates enable developers to build dynamic prompts by separating the template structure from the runtime variables, making it easier to maintain, test, and reuse prompts across different contexts.

**Key Benefits**:
- **Reusability**: Define prompt structure once, reuse with different inputs
- **Type Safety**: Validate input variables and enforce schema compliance
- **Composability**: Combine prompts using operators for complex workflows
- **Multi-Modal Support**: Handle text, images, and structured data
- **Security**: Built-in protection against template injection attacks

**Source**: `libs/core/langchain_core/prompts/`

---

## Quick Start

Get started in 30 seconds with this minimal working example:

```python
from langchain_core.prompts import ChatPromptTemplate

# Create a simple chat template
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

# Format the prompt with variables
prompt_value = template.invoke({"question": "What is 2+2?"})
# -> ChatPromptValue(messages=[
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="What is 2+2?")
# ])
```

---

## Template Types Guide

### ChatPromptTemplate

**Purpose**: Primary template for modern chat-based LLMs that work with message sequences.

**When to Use**: 
- Chat models (OpenAI ChatGPT, Anthropic Claude, etc.)
- Conversational AI applications
- Multi-turn dialogues with system instructions

**Key Features**:
- Supports SystemMessage, HumanMessage, AIMessage types
- Tuple shorthand syntax for quick composition
- MessagesPlaceholder integration for dynamic history
- Composable with `+` operator

**Source**: `libs/core/langchain_core/prompts/chat.py`

**Example - Basic Chat Template**:
```python
from langchain_core.prompts import ChatPromptTemplate

# Using tuple shorthand (recommended)
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

prompt_value = template.invoke({
    "name": "Bob",
    "user_input": "What is your name?",
})
# -> ChatPromptValue with 4 messages: System, Human, AI, Human
```

**Example - With MessagesPlaceholder**:
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("conversation_history"),
    ("human", "{question}"),
])

prompt_value = template.invoke({
    "conversation_history": [
        ("human", "Hi!"),
        ("ai", "Hello! How can I help you?"),
    ],
    "question": "What's the weather like?",
})
# -> ChatPromptValue with 4 messages: System, Human, AI, Human
```

---

### PromptTemplate

**Purpose**: Simple string-based prompts with variable substitution for traditional text completion models.

**When to Use**:
- Legacy LLM integrations (text-in, text-out)
- Single-turn prompts without conversation context
- Simple use cases without message structure requirements

**Key Features**:
- Lightweight and straightforward
- Supports f-string, jinja2, mustache formatting
- Partial variable application
- File-based template loading

**Source**: `libs/core/langchain_core/prompts/prompt.py`

**Example - Basic String Template**:
```python
from langchain_core.prompts import PromptTemplate

# Using from_template (recommended)
prompt = PromptTemplate.from_template("Say {foo}")
result = prompt.format(foo="bar")
# -> "Say bar"

# Using initializer
prompt = PromptTemplate(
    template="Tell me a {adjective} joke about {content}.",
    input_variables=["adjective", "content"]
)
result = prompt.format(adjective="funny", content="chickens")
# -> "Tell me a funny joke about chickens."
```

**Example - Partial Variables**:
```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

prompt = PromptTemplate(
    template="Tell me a {adjective} joke about the day {date}",
    input_variables=["adjective", "date"]
)

# Apply partial variables
partial_prompt = prompt.partial(date=datetime.now().strftime("%B %d"))
result = partial_prompt.format(adjective="funny")
# -> "Tell me a funny joke about the day November 07"
```

---

### FewShotPromptTemplate

**Purpose**: For few-shot learning by including example sets in prompts to help models learn from demonstrations.

**When to Use**:
- Teaching models new tasks through examples
- Improving accuracy on specific patterns
- Demonstrations of desired output format

**Key Features**:
- Static examples list or dynamic ExampleSelector
- Customizable example formatting
- Prefix and suffix text support

**Source**: `libs/core/langchain_core/prompts/few_shot.py`

**Example - Few-Shot Learning**:
```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Define examples
examples = [
    {"question": "Who lived longer, Mozart or Bach?", 
     "answer": "Bach lived longer than Mozart."},
    {"question": "Who was born first, Einstein or Newton?", 
     "answer": "Newton was born first."},
]

# Define how each example should be formatted
example_prompt = PromptTemplate(
    template="Question: {question}\nAnswer: {answer}",
    input_variables=["question", "answer"]
)

# Create few-shot template
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Answer the following questions:",
    suffix="Question: {input}\nAnswer:",
    input_variables=["input"],
)

result = prompt.format(input="Who lived longer, Mozart or Beethoven?")
# -> Formatted string with examples followed by the new question
```

---

### MessagesPlaceholder

**Purpose**: Dynamic message insertion in chat templates to enable conversation history integration.

**When to Use**:
- Chat applications requiring conversation history
- Agent systems with multi-turn interactions
- Variable-length message sequences

**Key Features**:
- Optional placeholder (no error if missing)
- Message count limiting with `n_messages`
- Automatically converts tuple format to Message objects

**Source**: `libs/core/langchain_core/prompts/chat.py:55`

**Example - Basic Placeholder**:
```python
from langchain_core.prompts import MessagesPlaceholder

# Required placeholder
prompt = MessagesPlaceholder("history")
messages = prompt.format_messages(
    history=[
        ("system", "You are an AI assistant."),
        ("human", "Hello!"),
    ]
)
# -> [SystemMessage(...), HumanMessage(...)]

# Optional placeholder (no error if omitted)
prompt = MessagesPlaceholder("history", optional=True)
messages = prompt.format_messages()  # No error, returns []
```

**Example - Limiting Message Count**:
```python
from langchain_core.prompts import MessagesPlaceholder

# Keep only last N messages
prompt = MessagesPlaceholder("history", n_messages=2)

messages = prompt.format_messages(
    history=[
        ("system", "You are an AI assistant."),
        ("human", "Message 1"),
        ("ai", "Response 1"),
        ("human", "Message 2"),
    ]
)
# -> Only last 2 messages: [AIMessage("Response 1"), HumanMessage("Message 2")]
```

---

### DictPromptTemplate

**Purpose**: For dictionary-based prompt construction with structured data templates.

**When to Use**:
- Structured input/output requirements
- Nested data with multiple fields
- API request formatting

**Key Features**:
- Recursive variable substitution in dict values
- Supports f-string and mustache formats
- Does NOT recognize variables in dict keys (only values)

**Source**: `libs/core/langchain_core/prompts/dict.py`

**Example - Dict Template**:
```python
from langchain_core.prompts.dict import DictPromptTemplate

template = DictPromptTemplate(
    template={
        "name": "{person_name}",
        "greeting": "Hello {person_name}!",
        "details": {
            "age": "{age}",
            "location": "Lives in {city}"
        }
    },
    template_format="f-string"
)

result = template.format(
    person_name="Alice",
    age="30",
    city="New York"
)
# -> {
#     "name": "Alice",
#     "greeting": "Hello Alice!",
#     "details": {"age": "30", "location": "Lives in New York"}
# }
```

---

### ImagePromptTemplate

**Purpose**: For multimodal prompts with images supporting vision-enabled models.

**When to Use**:
- Vision models (GPT-4 Vision, Claude with vision, etc.)
- Image analysis and description tasks
- Multimodal AI applications

**Key Features**:
- Image URL, path, or base64 support
- Image detail level control
- Integration with ChatPromptTemplate for multimodal conversations

**Source**: `libs/core/langchain_core/prompts/image.py`

**Example - Image Prompt**:
```python
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# Using image in chat template with HumanMessagePromptTemplate
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful vision AI assistant."),
    (
        "human",
        [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": "{image_url}"},
        ],
    ),
])

prompt_value = template.invoke({
    "image_url": "https://example.com/image.jpg"
})
# -> ChatPromptValue with text and image content
```

---

## Message Construction Patterns

### Message Types

LangChain defines three primary message types for chat-based interactions:

| Message Type | Purpose | Usage |
|-------------|---------|-------|
| **SystemMessage** | System-level instructions and context | Set AI behavior, capabilities, constraints |
| **HumanMessage** | User inputs and queries | Questions, commands, user requests |
| **AIMessage** | Assistant responses and outputs | Model responses, agent actions |

**Source**: `libs/core/langchain_core/messages/`

### Tuple Shorthand Syntax

The most convenient way to create messages is using tuple shorthand:

```python
# Tuple format: (role, content)
messages = [
    ("system", "You are a helpful assistant."),  # -> SystemMessage
    ("human", "{user_input}"),                   # -> HumanMessage with variable
    ("ai", "I'm here to help!"),                 # -> AIMessage
]
```

### Message Composition and Ordering

**Best Practices**:
1. **System Message First**: Always place system instructions at the beginning
2. **Alternating Turns**: Alternate between human and AI messages for natural conversation
3. **Context Before Query**: Provide context messages before the final user question

```python
from langchain_core.prompts import ChatPromptTemplate

# Proper message ordering
template = ChatPromptTemplate.from_messages([
    # 1. System context
    ("system", "You are an expert in {domain}."),
    
    # 2. Example conversation (optional)
    ("human", "What is your expertise?"),
    ("ai", "I specialize in {domain}."),
    
    # 3. Current query
    ("human", "{current_question}"),
])
```

---

## Input Variable Type Requirements

### Type Constraints

**Critical Rule**: All template variables must ultimately resolve to **strings** after substitution.

**Valid Variable Types**:
- `str` - Direct string values
- Any type with `__str__()` method - Will be converted to string
- For MessagesPlaceholder: `list[tuple]` or `list[BaseMessage]`

**Source**: `libs/core/langchain_core/prompts/base.py:47-56`

### Validation Rules

1. **Required Variables**: All variables in template must be provided unless marked optional
2. **No Reserved Names**: Cannot use variable name `"stop"` (used internally)
3. **No Overlap**: Partial variables cannot overlap with input variables
4. **Type Consistency**: Variables must match expected types for each template format

**Example - Variable Validation**:
```python
from langchain_core.prompts import PromptTemplate

# This will raise ValueError - missing variable
prompt = PromptTemplate.from_template("Hello {name}, welcome to {place}!")
# prompt.format(name="Alice")  # ERROR: Missing 'place' variable

# Correct usage
result = prompt.format(name="Alice", place="Wonderland")
# -> "Hello Alice, welcome to Wonderland!"
```

### Partial Variable Application

Apply some variables early, others later:

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Answer the question about {topic}: {question}",
    input_variables=["topic", "question"]
)

# Partially apply 'topic'
science_prompt = prompt.partial(topic="science")

# Later, only provide 'question'
result = science_prompt.format(question="What is gravity?")
# -> "Answer the question about science: What is gravity?"
```

---

## Template Formatting Options

### Format Comparison

| Format | Syntax | Security | Complexity | Use Case |
|--------|--------|----------|------------|----------|
| **f-string** | `{variable}` | ✅ Safe | Simple | **Recommended default** |
| **mustache** | `{{variable}}` | ✅ Safe | Simple | Logic-less templates |
| **jinja2** | `{{ variable }}` with loops/conditionals | ⚠️ Risky | Complex | Advanced formatting only |

**Source**: `libs/core/langchain_core/prompts/string.py:27`

---

### f-string Format (Default - Recommended)

**Syntax**: Standard Python f-string syntax with `{variable_name}`

**Benefits**:
- ✅ **Most secure** - No code execution risk
- ✅ Simple and familiar to Python developers
- ✅ Fast performance
- ✅ Type-safe variable substitution

**Example**:
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Translate the following {source_lang} to {target_lang}: {text}",
    template_format="f-string"  # This is the default
)

result = prompt.format(
    source_lang="English",
    target_lang="French",
    text="Hello, world!"
)
# -> "Translate the following English to French: Hello, world!"
```

---

### mustache Format

**Syntax**: Logic-less templates with `{{variable_name}}`

**Benefits**:
- ✅ **Safe** - No code execution
- ✅ Cross-language compatibility
- ✅ Simple conditional logic with sections

**Example**:
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Hello {{name}}! Welcome to {{place}}.",
    input_variables=["name", "place"],
    template_format="mustache"
)

result = prompt.format(name="Alice", place="Wonderland")
# -> "Hello Alice! Welcome to Wonderland."
```

---

### jinja2 Format (Use With Caution)

**⚠️ SECURITY WARNING**: Jinja2 templates can lead to **arbitrary code execution** if templates come from untrusted sources. Only use jinja2 with templates YOU control, never with user-provided templates.

**Syntax**: Jinja2 template syntax with `{{ variable }}`, loops, conditionals

**Benefits**:
- Powerful control flow (if/else, for loops, filters)
- Complex formatting logic
- Template inheritance

**When to Use**: Only when you need advanced template features AND you control the template source

**Source**: `libs/core/langchain_core/prompts/string.py:30-69`

**Example - Safe Usage**:
```python
from langchain_core.prompts import PromptTemplate

# SAFE: Template is hardcoded by developer
prompt = PromptTemplate(
    template="""
    Generate a list of {{ count }} items:
    {% for i in range(count) %}
    {{ i + 1 }}. Item {{ i + 1 }}
    {% endfor %}
    """,
    input_variables=["count"],
    template_format="jinja2"
)

result = prompt.format(count=3)
# -> Multi-line list with 3 items
```

**Example - UNSAFE (Never Do This)**:
```python
# DANGEROUS: Never accept jinja2 templates from users
user_provided_template = request.form.get("template")  # ❌ NEVER DO THIS
prompt = PromptTemplate(
    template=user_provided_template,
    input_variables=["data"],
    template_format="jinja2"  # ❌ SECURITY RISK
)
```

**Security Note**: LangChain uses `SandboxedEnvironment` by default for jinja2, but this should be treated as best-effort, not a security guarantee.

---

## Practical Examples (Simple to Complex)

### Example 1: Simple String Template (Beginner)

```python
from langchain_core.prompts import PromptTemplate

# Most basic prompt template
prompt = PromptTemplate.from_template("What is the capital of {country}?")
result = prompt.format(country="France")

print(result)
# Output: "What is the capital of France?"
```

---

### Example 2: Chat Template with System Message

```python
from langchain_core.prompts import ChatPromptTemplate

# Chat template with system instructions
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {domain}."),
    ("human", "{user_question}")
])

prompt_value = template.invoke({
    "domain": "geography",
    "user_question": "What is the capital of France?"
})

# Output: ChatPromptValue with 2 messages
print(prompt_value.to_messages())
# [
#   SystemMessage(content="You are a helpful assistant specialized in geography."),
#   HumanMessage(content="What is the capital of France?")
# ]
```

---

### Example 3: Few-Shot Template with Examples

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

# Define few-shot examples
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "5-3", "output": "2"},
]

# Example prompt showing how to format each example
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# Few-shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# Final template combining examples and new question
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a calculator. Answer with only the number."),
    few_shot_prompt,
    ("human", "{question}"),
])

prompt_value = final_prompt.invoke({"question": "10+5"})
# Output: System message + 2 example pairs + new question = 6 messages total
```

---

### Example 4: Complex Multi-Message Template with MessagesPlaceholder

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Advanced template with conversation history
template = ChatPromptTemplate.from_messages([
    ("system", """You are {agent_name}, a {role}. 
    Current date: {date}
    
    Guidelines:
    - Be helpful and concise
    - Use {tone} tone
    """),
    MessagesPlaceholder("conversation_history", optional=True),
    ("human", "{current_question}"),
])

# Invoke with full context
prompt_value = template.invoke({
    "agent_name": "Alex",
    "role": "customer support agent",
    "date": "2024-11-07",
    "tone": "friendly",
    "conversation_history": [
        ("human", "Hi, I need help with my account."),
        ("ai", "Hello! I'd be happy to help with your account. What do you need?"),
        ("human", "I can't log in."),
        ("ai", "Let me help you troubleshoot the login issue."),
    ],
    "current_question": "I tried resetting my password but didn't receive an email."
})

# Output: 7 messages total (system + 4 history + 1 current question)
```

---

### Example 5: Partial Variable Application

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

# Template with both fixed and runtime variables
template = PromptTemplate(
    template="""Date: {date}
    From: {sender}
    To: {recipient}
    Subject: {subject}
    
    {body}
    """,
    input_variables=["date", "sender", "recipient", "subject", "body"]
)

# Partially apply date and sender (fixed values)
email_template = template.partial(
    date=datetime.now().strftime("%Y-%m-%d"),
    sender="system@company.com"
)

# Now only provide runtime variables
result = email_template.format(
    recipient="user@example.com",
    subject="Welcome to Our Service",
    body="Thank you for signing up!"
)

print(result)
# Output: Fully formatted email with current date
```

---

### Example 6: Template Combination with + Operator

```python
from langchain_core.prompts import ChatPromptTemplate

# Create reusable prompt components
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role}.")
])

context_prompt = ChatPromptTemplate.from_messages([
    ("system", "Context: {context}")
])

query_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}")
])

# Combine prompts with + operator
combined_prompt = system_prompt + context_prompt + query_prompt

# Use combined template
prompt_value = combined_prompt.invoke({
    "role": "data analyst",
    "context": "Customer data shows 20% growth in Q4",
    "question": "What trends should we focus on?"
})

# Output: 3 messages (system role + system context + human question)
```

---

## Best Practices

### Security Considerations

1. **Always Use f-string Format**: Default to f-string unless you have specific needs
2. **Never Accept jinja2 from Users**: User-provided templates with jinja2 = code execution risk
3. **Validate Input Variables**: Check variable types before formatting
4. **Sanitize User Inputs**: Clean user-provided data before substitution

```python
# ✅ GOOD: Safe default format
prompt = PromptTemplate.from_template("Hello {name}")  # f-string default

# ⚠️ RISKY: Only if template is hardcoded by you
prompt = PromptTemplate(
    template="Hello {{ name | upper }}",
    input_variables=["name"],
    template_format="jinja2"  # Use only with your templates
)

# ❌ DANGEROUS: Never do this
user_template = get_user_input()  # From untrusted source
prompt = PromptTemplate(template=user_template, template_format="jinja2")
```

---

### Performance Tips

1. **Reuse Templates**: Create template once, invoke many times
2. **Use Partial Variables**: Pre-fill static values to reduce runtime overhead
3. **Limit Message History**: Use `MessagesPlaceholder(n_messages=N)` to control context size
4. **Cache Compiled Templates**: Templates are cached automatically after first use

```python
# ✅ GOOD: Create once, reuse many times
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

for question in user_questions:
    prompt_value = template.invoke({"question": question})  # Fast reuse
```

---

### Common Patterns

**Pattern 1: Context-Augmented Query**
```python
template = ChatPromptTemplate.from_messages([
    ("system", "Use the following context to answer questions:\n{context}"),
    ("human", "{question}")
])
```

**Pattern 2: Conversation with Memory**
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
```

**Pattern 3: Multi-Step Reasoning**
```python
template = ChatPromptTemplate.from_messages([
    ("system", "Break down the problem into steps."),
    ("human", """Problem: {problem}
    
    Step 1: Understand the problem
    Step 2: Identify key information
    Step 3: Propose solution
    Step 4: Verify answer
    """)
])
```

---

## Troubleshooting

### Common Issues

**Issue 1: Missing Variable Error**
```python
# Problem
prompt = PromptTemplate.from_template("Hello {name}, you are {age} years old.")
prompt.format(name="Alice")  # ❌ KeyError: 'age'

# Solution
prompt.format(name="Alice", age=30)  # ✅ Provide all variables
```

**Issue 2: MessagesPlaceholder Not Optional**
```python
# Problem
template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"),
    ("human", "{input}")
])
template.invoke({"input": "Hi"})  # ❌ Error: 'history' required

# Solution: Make it optional
template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history", optional=True),  # ✅ Optional
    ("human", "{input}")
])
template.invoke({"input": "Hi"})  # ✅ Works without history
```

**Issue 3: Invalid Message Format**
```python
# Problem
MessagesPlaceholder("history").format_messages(
    history="Not a list"  # ❌ Must be list
)

# Solution
MessagesPlaceholder("history").format_messages(
    history=[("human", "Hello")]  # ✅ List of tuples or messages
)
```

---

## Additional Resources

### Related Modules

- **langchain_core.messages**: Message type definitions (SystemMessage, HumanMessage, AIMessage)
- **langchain_core.runnables**: Runnable protocol for LCEL composition
- **langchain_core.output_parsers**: Parse LLM outputs into structured formats
- **langchain_core.example_selectors**: Dynamic example selection for few-shot prompts

### External Documentation

- [LangChain Official Docs](https://docs.langchain.com/) - Comprehensive guides and tutorials
- [LangChain API Reference](https://reference.langchain.com/python/langchain_core/) - Full API documentation
- [Prompt Engineering Guide](https://www.promptingguide.ai/) - Best practices for prompt design

### Source Code References

All implementation details and behaviors documented here are derived from:

- `libs/core/langchain_core/prompts/base.py` - Base template abstractions
- `libs/core/langchain_core/prompts/chat.py` - Chat templates and message handling
- `libs/core/langchain_core/prompts/prompt.py` - String-based PromptTemplate
- `libs/core/langchain_core/prompts/string.py` - Template format handlers
- `libs/core/langchain_core/prompts/few_shot.py` - Few-shot learning templates
- `libs/core/langchain_core/prompts/message.py` - Message prompt template base
- `libs/core/langchain_core/prompts/dict.py` - Dictionary-based templates
- `libs/core/langchain_core/prompts/image.py` - Multimodal image templates

---

## Quick Reference

### Template Selection Decision Tree

```
Need images/multimodal? 
├─ Yes → ImagePromptTemplate + ChatPromptTemplate
└─ No → Continue

Need conversation format?
├─ Yes → ChatPromptTemplate
└─ No → Continue

Need few-shot examples?
├─ Yes → FewShotPromptTemplate or FewShotChatMessagePromptTemplate
└─ No → Continue

Need structured data?
├─ Yes → DictPromptTemplate
└─ No → PromptTemplate (simple string)
```

### Method Quick Reference

| Method | Purpose | Returns |
|--------|---------|---------|
| `from_template()` | Create template from string | Template instance |
| `from_messages()` | Create chat template from message list | ChatPromptTemplate |
| `format()` | Fill variables, return string | `str` |
| `format_messages()` | Fill variables, return messages | `list[BaseMessage]` |
| `invoke()` | Fill variables, return prompt value (LCEL) | `PromptValue` |
| `partial()` | Partially apply variables | Template instance |

---

**Last Updated**: 2024-11-07  
**LangChain Version Compatibility**: langchain-core >= 1.0.0  
**Python Version**: >= 3.10

