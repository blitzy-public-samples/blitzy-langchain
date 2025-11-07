"""Prompt schema definition."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, model_validator
from typing_extensions import override

from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    PromptTemplateFormat,
    StringPromptTemplate,
    check_valid_template,
    get_template_variables,
    mustache_schema,
)

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


class PromptTemplate(StringPromptTemplate):
    """Prompt template for a language model with flexible formatting options.

    A prompt template consists of a string template with variable placeholders that
    accepts parameters from the user to generate formatted prompts for language models.
    PromptTemplate provides the core abstraction for dynamic prompt construction,
    enabling reusable prompt logic across different use cases and models.

    The template can be formatted using three distinct template systems:
    
    - **f-string format** (default, recommended): Uses Python f-string syntax with
      curly braces for variables: `"Hello {name}, you are {age} years old"`
    - **jinja2 format**: Uses Jinja2 templating with full template logic support:
      `"Hello {{ name }}, {% if age %}you are {{ age }} years old{% endif %}"`
    - **mustache format**: Uses Mustache syntax for logic-less templates:
      `"Hello {{name}}, you are {{age}} years old"`

    **Usage Patterns**:
    
    - **Simple variable substitution**: Use f-string format for straightforward
      variable replacement without complex logic
    - **Conditional rendering**: Use jinja2 format when prompts require conditional
      sections, loops, or filters
    - **Cross-language templates**: Use mustache format for templates shared across
      multiple programming languages
    - **Partial variable application**: Pre-fill some variables using partial_variables
      to create reusable prompt templates with fixed context

    **Security Considerations**:
    
    *Critical Security Warning*:
        Prefer using `template_format="f-string"` (default) instead of
        `template_format="jinja2"`. NEVER accept jinja2 templates from untrusted
        sources as they may lead to arbitrary Python code execution.

        As of LangChain 0.0.329, Jinja2 templates are rendered using Jinja2's
        SandboxedEnvironment by default. However, this sandboxing should be treated
        as a best-effort security measure rather than a guarantee, as it uses an
        opt-out rather than opt-in approach.

        Despite the sandboxing, we strongly recommend never using jinja2 templates
        from untrusted or user-controlled sources. Use f-string format for
        user-provided templates.

    **Template Format Characteristics**:
    
    - **f-string**: Fast, secure, supports basic variable substitution only. No
      template logic or filters. Variables use single curly braces: {variable}
    - **jinja2**: Full-featured templating with conditionals, loops, filters, and
      inheritance. Requires jinja2 package. Variables use double curly braces:
      {{variable}}. Security risk with untrusted input.
    - **mustache**: Logic-less templates with sections and partials. Variables use
      double curly braces: {{variable}}. Supports nested objects and iteration.

    Example:
        ```python
        from langchain_core.prompts import PromptTemplate

        # Instantiation using from_template (recommended)
        prompt = PromptTemplate.from_template("Say {foo}")
        result = prompt.format(foo="bar")
        # result: "Say bar"

        # Direct instantiation
        prompt = PromptTemplate(
            template="Tell me a {adjective} joke about {topic}",
            input_variables=["adjective", "topic"]
        )
        result = prompt.format(adjective="funny", topic="chickens")
        # result: "Tell me a funny joke about chickens"

        # Using jinja2 format with conditional logic
        prompt = PromptTemplate.from_template(
            "Hello {{ name }}{% if age %}, you are {{ age }} years old{% endif %}",
            template_format="jinja2"
        )
        result = prompt.format(name="Alice", age=30)
        # result: "Hello Alice, you are 30 years old"

        # Using mustache format
        prompt = PromptTemplate.from_template(
            "Hello {{name}}, you are {{age}} years old",
            template_format="mustache"
        )
        result = prompt.format(name="Bob", age=25)
        # result: "Hello Bob, you are 25 years old"

        # Using partial variables for reusable templates
        prompt = PromptTemplate.from_template(
            "System context: {context}\\nUser query: {query}",
            partial_variables={"context": "You are a helpful assistant"}
        )
        result = prompt.format(query="What is Python?")
        # result: "System context: You are a helpful assistant\\nUser query: What is Python?"
        ```

    Source: libs/core/langchain_core/prompts/prompt.py:24-57
    """

    @property
    @override
    def lc_attributes(self) -> dict[str, Any]:
        return {
            "template_format": self.template_format,
        }

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "prompts", "prompt"]`
        """
        return ["langchain", "prompts", "prompt"]

    template: str
    """The prompt template."""

    template_format: PromptTemplateFormat = "f-string"
    """The format of the prompt template. Determines how variables are substituted.
    
    Options are:
    - 'f-string' (default, recommended): Python f-string syntax using single curly
      braces {variable}. Fast, secure, no template logic. Example: "Hello {name}"
    - 'mustache': Mustache syntax using double curly braces {{variable}}. Logic-less
      templates with sections and partials. Example: "Hello {{name}}"
    - 'jinja2': Jinja2 syntax using double curly braces {{variable}} with full
      template logic support (conditionals, loops, filters). Requires jinja2 package.
      Security warning: never use with untrusted input. Example: "Hello {{ name }}"
    
    Characteristics by format:
    - f-string: No dependencies, fastest, most secure, no template logic
    - mustache: Cross-language compatible, supports nested objects, logic-less
    - jinja2: Most powerful, supports complex logic, security risk with untrusted input
    
    Source: libs/core/langchain_core/prompts/prompt.py:79-81
    """

    validate_template: bool = False
    """Whether or not to try validating the template."""

    @model_validator(mode="before")
    @classmethod
    def pre_init_validation(cls, values: dict) -> Any:
        """Check that template and input variables are consistent."""
        if values.get("template") is None:
            # Will let pydantic fail with a ValidationError if template
            # is not provided.
            return values

        # Set some default values based on the field defaults
        values.setdefault("template_format", "f-string")
        values.setdefault("partial_variables", {})

        if values.get("validate_template"):
            if values["template_format"] == "mustache":
                msg = "Mustache templates cannot be validated."
                raise ValueError(msg)

            if "input_variables" not in values:
                msg = "Input variables must be provided to validate the template."
                raise ValueError(msg)

            all_inputs = values["input_variables"] + list(values["partial_variables"])
            check_valid_template(
                values["template"], values["template_format"], all_inputs
            )

        if values["template_format"]:
            values["input_variables"] = [
                var
                for var in get_template_variables(
                    values["template"], values["template_format"]
                )
                if var not in values["partial_variables"]
            ]

        return values

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """Get the input schema for the prompt.

        Args:
            config: The runnable configuration.

        Returns:
            The input schema for the prompt.
        """
        if self.template_format != "mustache":
            return super().get_input_schema(config)

        return mustache_schema(self.template)

    def __add__(self, other: Any) -> PromptTemplate:
        """Override the + operator to allow for combining prompt templates.

        This operator enables intuitive template composition by concatenating template
        strings and merging their configurations. When adding two PromptTemplate instances
        or adding a string to a PromptTemplate, the result is a new PromptTemplate with
        combined template content and unified variable requirements.

        **Combination Rules**:
        
        - **Template concatenation**: Template strings are joined directly without separators
        - **Variable merging**: input_variables from both templates are combined (union of sets)
        - **Format consistency**: Both templates must use the same template_format
        - **Partial variable merging**: partial_variables are merged, duplicates raise ValueError
        - **Validation inheritance**: New template validates only if both originals validate

        **String Addition Behavior**:
        
        When adding a string to a PromptTemplate, the string is first converted to a
        PromptTemplate using from_template() with the same template_format as the original,
        then the two PromptTemplates are combined using standard addition rules.

        Args:
            other: The template or string to combine with this template. Supported types:
                - PromptTemplate: Another prompt template to concatenate. Must have the
                  same template_format as this template.
                - str: A template string that will be converted to a PromptTemplate with
                  matching template_format before combination.

        Returns:
            PromptTemplate: A new PromptTemplate instance representing the combination with:
                - template: Concatenation of self.template + other.template
                - input_variables: Union of variables from both templates (no duplicates)
                - partial_variables: Merged partial_variables from both templates
                - template_format: Same format as the original templates
                - validate_template: True only if both original templates validate

        Raises:
            ValueError: If attempting to combine templates with different template_format
                values. Templates must have matching formats (both "f-string", both "jinja2",
                or both "mustache") to be compatible for addition.
            ValueError: If both templates define the same partial variable with potentially
                different values. Each partial variable name must be unique across the
                combined templates to prevent ambiguous variable resolution.
            NotImplementedError: If the other operand is not a PromptTemplate or str.
                Only these types are supported for template combination.

        Example:
            ```python
            from langchain_core.prompts import PromptTemplate

            # Combining two PromptTemplate instances
            system_prompt = PromptTemplate.from_template("System: {system_context}\\n")
            user_prompt = PromptTemplate.from_template("User: {user_query}\\n")
            combined = system_prompt + user_prompt
            result = combined.format(
                system_context="You are helpful",
                user_query="What is AI?"
            )
            # result: "System: You are helpful\\nUser: What is AI?\\n"

            # Adding a string to a PromptTemplate
            prompt = PromptTemplate.from_template("Question: {question}\\n")
            full_prompt = prompt + "Answer:"
            result = full_prompt.format(question="What is Python?")
            # result: "Question: What is Python?\\nAnswer:"

            # Combining templates with merged variables
            intro = PromptTemplate.from_template("Hello {name}, ")
            body = PromptTemplate.from_template("you are {age} years old")
            greeting = intro + body
            # greeting.input_variables == ["name", "age"]
            result = greeting.format(name="Alice", age=30)
            # result: "Hello Alice, you are 30 years old"

            # Error case: mismatched template formats
            f_string_prompt = PromptTemplate.from_template("Hello {name}")
            jinja_prompt = PromptTemplate.from_template(
                "Goodbye {{ name }}",
                template_format="jinja2"
            )
            # combined = f_string_prompt + jinja_prompt  # Raises ValueError

            # Error case: conflicting partial variables
            prompt1 = PromptTemplate.from_template(
                "Context: {context}\\n{query}",
                partial_variables={"context": "Version 1"}
            )
            prompt2 = PromptTemplate.from_template(
                "More context: {context}\\n{response}",
                partial_variables={"context": "Version 2"}  # Duplicate key
            )
            # combined = prompt1 + prompt2  # Raises ValueError

            # Combining templates with partial variables (non-conflicting)
            prompt1 = PromptTemplate.from_template(
                "Role: {role}\\n",
                partial_variables={"role": "assistant"}
            )
            prompt2 = PromptTemplate.from_template(
                "Task: {task}",
                partial_variables={"task": "help user"}
            )
            combined = prompt1 + prompt2
            result = combined.format()  # No additional variables needed
            # result: "Role: assistant\\nTask: help user"
            ```

        Source: libs/core/langchain_core/prompts/prompt.py:139-181
        """
        # Allow for easy combining
        if isinstance(other, PromptTemplate):
            if self.template_format != other.template_format:
                msg = "Cannot add templates of different formats"
                raise ValueError(msg)
            input_variables = list(
                set(self.input_variables) | set(other.input_variables)
            )
            template = self.template + other.template
            # If any do not want to validate, then don't
            validate_template = self.validate_template and other.validate_template
            partial_variables = dict(self.partial_variables.items())
            for k, v in other.partial_variables.items():
                if k in partial_variables:
                    msg = "Cannot have same variable partialed twice."
                    raise ValueError(msg)
                partial_variables[k] = v
            return PromptTemplate(
                template=template,
                input_variables=input_variables,
                partial_variables=partial_variables,
                template_format=self.template_format,
                validate_template=validate_template,
            )
        if isinstance(other, str):
            prompt = PromptTemplate.from_template(
                other,
                template_format=self.template_format,
            )
            return self + prompt
        msg = f"Unsupported operand type for +: {type(other)}"
        raise NotImplementedError(msg)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "prompt"

    def format(self, **kwargs: Any) -> str:
        """Format the prompt template by substituting variables with provided values.

        This method performs template variable substitution using the template_format
        specified during template creation. It merges user-provided variables with
        partial_variables (with user values taking precedence), then applies the
        appropriate formatter to generate the final prompt string.

        The formatting behavior varies by template_format:
        - f-string: Direct string formatting with Python's format() method
        - jinja2: Template rendering using Jinja2's SandboxedEnvironment
        - mustache: Logic-less template rendering using mustache library

        **Variable Resolution Order**:
        1. Partial variables (pre-filled during template creation)
        2. User-provided kwargs (override partial variables if duplicate keys exist)

        **Type Handling**:
        - String values: Used directly in template substitution
        - Non-string values: Automatically converted to strings via str() before substitution
        - Nested objects: For mustache format, supports dot notation (e.g., {{user.name}})
        - List values: For jinja2 format, can be iterated with {% for %} loops

        Args:
            **kwargs: Keyword arguments providing values for template variables. Each key
                should correspond to a variable name in the template. Missing required
                variables will cause formatting errors.
                - For f-string templates: Keys must match variable names exactly
                - For jinja2 templates: Keys become template context variables
                - For mustache templates: Keys support nested object access via dot notation
                
                All values are converted to strings during substitution. Complex objects
                should implement __str__() for meaningful string representation.

        Returns:
            str: The formatted prompt string with all variable placeholders replaced by
                their corresponding values from kwargs and partial_variables. The returned
                string is ready for consumption by language models.

        Raises:
            KeyError: If a required template variable is not provided in kwargs or
                partial_variables (f-string and jinja2 formats)
            ValueError: If template variable names in kwargs conflict with reserved names
            TypeError: If kwargs contains non-serializable values that cannot be converted
                to strings for template substitution

        Example:
            ```python
            from langchain_core.prompts import PromptTemplate

            # Basic variable substitution with f-string format
            prompt = PromptTemplate.from_template(
                "Write a {length} story about {topic}"
            )
            result = prompt.format(length="short", topic="space exploration")
            # result: "Write a short story about space exploration"

            # Jinja2 format with conditional logic
            prompt = PromptTemplate.from_template(
                "{% if formal %}Dear{% else %}Hi{% endif %} {{ name }},\\n{{ message }}",
                template_format="jinja2"
            )
            result = prompt.format(formal=True, name="Dr. Smith", message="Thank you")
            # result: "Dear Dr. Smith,\\nThank you"

            # Mustache format with nested objects
            prompt = PromptTemplate.from_template(
                "Name: {{person.name}}\\nAge: {{person.age}}",
                template_format="mustache"
            )
            result = prompt.format(person={"name": "Alice", "age": 30})
            # result: "Name: Alice\\nAge: 30"

            # Using partial variables with user overrides
            prompt = PromptTemplate.from_template(
                "Role: {role}\\nTask: {task}",
                partial_variables={"role": "assistant"}
            )
            # User value overrides partial variable
            result = prompt.format(role="expert", task="code review")
            # result: "Role: expert\\nTask: code review"

            # Non-string values automatically converted
            prompt = PromptTemplate.from_template(
                "Temperature: {temp}°C, Count: {count} items"
            )
            result = prompt.format(temp=23.5, count=42)
            # result: "Temperature: 23.5°C, Count: 42 items"

            # Multi-line formatted prompts
            prompt = PromptTemplate.from_template(
                "System: {system}\\n\\nUser: {user}\\n\\nAssistant:"
            )
            result = prompt.format(
                system="You are a helpful assistant",
                user="Explain quantum computing"
            )
            # result: "System: You are a helpful assistant\\n\\nUser: Explain quantum computing\\n\\nAssistant:"
            ```

        Source: libs/core/langchain_core/prompts/prompt.py:188-198
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        return DEFAULT_FORMATTER_MAPPING[self.template_format](self.template, **kwargs)

    @classmethod
    def from_examples(
        cls,
        examples: list[str],
        suffix: str,
        input_variables: list[str],
        example_separator: str = "\n\n",
        prefix: str = "",
        **kwargs: Any,
    ) -> PromptTemplate:
        """Take examples in list format with prefix and suffix to create a prompt.

        Intended to be used as a way to dynamically create a prompt from examples.

        Args:
            examples: List of examples to use in the prompt.
            suffix: String to go after the list of examples. Should generally
                set up the user's input.
            input_variables: A list of variable names the final prompt template
                will expect.
            example_separator: The separator to use in between examples. Defaults
                to two new line characters.
            prefix: String that should go before any examples. Generally includes
                examples.

        Returns:
            The final prompt generated.
        """
        template = example_separator.join([prefix, *examples, suffix])
        return cls(input_variables=input_variables, template=template, **kwargs)

    @classmethod
    def from_file(
        cls,
        template_file: str | Path,
        encoding: str | None = None,
        **kwargs: Any,
    ) -> PromptTemplate:
        """Load a prompt template from a file with automatic variable extraction.

        This factory method reads a template string from a file and delegates to
        from_template() for prompt construction. It provides a convenient way to
        manage prompt templates as separate files for better organization and version
        control, especially for long or complex prompts.

        The file should contain a plain text template string with variable placeholders
        matching the specified template_format. The method reads the entire file content
        as the template string, so the file should not contain YAML or JSON structure
        unless that structure itself is part of the desired template output.

        **Supported File Formats**:
        
        - **Plain text files** (.txt): Template string with variables, no structure
        - **Markdown files** (.md): Template with markdown formatting preserved
        - **Custom extensions**: Any text file readable with the specified encoding
        
        **File Content Format**:
        
        The file should contain only the template string itself, not a structured
        format like YAML or JSON. For structured prompt files, use the legacy
        load_prompt() function from langchain.prompts instead.

        Args:
            template_file: The path to the file containing the prompt template. Can be
                either a string path or a pathlib.Path object. Path can be relative to
                current working directory or absolute.
                Example: "prompts/qa_template.txt" or Path("templates/greeting.md")
            encoding: The character encoding for reading the template file. If None,
                uses the operating system's default encoding (typically 'utf-8' on
                Unix/Linux/Mac, 'cp1252' on Windows). Specify encoding explicitly for
                cross-platform compatibility.
                Common values: "utf-8", "ascii", "latin-1", "cp1252"
                Default: None (OS default)
            **kwargs: Additional arguments passed to from_template() and subsequently
                to the PromptTemplate constructor. Commonly used arguments:
                - template_format (str): "f-string", "jinja2", or "mustache"
                - partial_variables (dict): Pre-filled template variables
                - validate_template (bool): Whether to validate template syntax
                - output_parser (BaseOutputParser): Parser for LLM output

        Returns:
            PromptTemplate: A new PromptTemplate instance with the template content
                loaded from the file, input_variables automatically extracted, and
                any additional configuration from **kwargs applied.

        Raises:
            FileNotFoundError: If the specified template_file does not exist
            PermissionError: If the file cannot be read due to insufficient permissions
            UnicodeDecodeError: If the file encoding does not match the specified encoding
            ValueError: If template_format in kwargs is invalid or template validation fails
            ImportError: If template_format is "jinja2" but jinja2 package is not installed

        Example:
            ```python
            from pathlib import Path
            from langchain_core.prompts import PromptTemplate

            # Create a template file first
            Path("qa_prompt.txt").write_text(
                "Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer:"
            )

            # Load template from file
            prompt = PromptTemplate.from_file("qa_prompt.txt")
            result = prompt.format(
                context="Python is a programming language",
                question="What is Python?"
            )
            # result: "Context: Python is a programming language\\n\\nQuestion: What is Python?\\n\\nAnswer:"

            # Load with specific encoding
            prompt = PromptTemplate.from_file(
                "templates/greeting.txt",
                encoding="utf-8"
            )

            # Load jinja2 template from file
            Path("jinja_prompt.txt").write_text(
                "{% if show_greeting %}Hello {{ name }}!{% endif %}\\n{{ message }}"
            )
            prompt = PromptTemplate.from_file(
                "jinja_prompt.txt",
                template_format="jinja2"
            )
            result = prompt.format(show_greeting=True, name="Alice", message="Welcome")
            # result: "Hello Alice!\\nWelcome"

            # Load with partial variables
            prompt = PromptTemplate.from_file(
                "system_prompt.txt",
                partial_variables={"role": "helpful assistant"}
            )
            ```

        Source: libs/core/langchain_core/prompts/prompt.py:231-249
        """
        template = Path(template_file).read_text(encoding=encoding)
        return cls.from_template(template=template, **kwargs)

    @classmethod
    def from_template(
        cls,
        template: str,
        *,
        template_format: PromptTemplateFormat = "f-string",
        partial_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> PromptTemplate:
        """Load a prompt template from a template string with automatic variable extraction.

        This factory method is the recommended way to create PromptTemplate instances.
        It automatically extracts input variables from the template string based on the
        specified format, eliminating the need to manually declare input_variables.

        The method parses the template to identify variable placeholders, filters out
        any variables specified in partial_variables, and constructs a PromptTemplate
        with the remaining variables as required inputs.

        *Security warning*:
            Prefer using `template_format="f-string"` (default) instead of
            `template_format="jinja2"`. NEVER accept jinja2 templates from untrusted
            sources as they may lead to arbitrary Python code execution.

            As of LangChain 0.0.329, Jinja2 templates are rendered using Jinja2's
            SandboxedEnvironment by default. However, this sandboxing should be treated
            as a best-effort security measure rather than a guarantee, as it uses an
            opt-out rather than opt-in approach.

            Despite the sandboxing, we strongly recommend never using jinja2 templates
            from untrusted or user-controlled sources. Always use f-string format for
            templates from untrusted sources.

        Args:
            template: The template string with variable placeholders. Variable syntax
                depends on template_format:
                - f-string: Use single curly braces like "Hello {name}, you are {age}"
                - jinja2: Use double curly braces like "Hello {{ name }}, age: {{ age }}"
                - mustache: Use double curly braces like "Hello {{name}}, age: {{age}}"
                Must be a non-empty string containing at least one variable placeholder
                unless partial_variables fully satisfies all template variables.
            template_format: The format of the template. Determines variable extraction
                and substitution behavior. Options:
                - "f-string" (default, recommended): Python f-string syntax. Fast, secure,
                  no template logic. Example: "Answer: {response}"
                - "jinja2": Jinja2 templating with full logic support (conditionals, loops,
                  filters). Requires jinja2 package installation. Security risk with
                  untrusted input. Example: "{% if show %}Answer: {{ response }}{% endif %}"
                - "mustache": Mustache syntax for logic-less templates. Cross-language
                  compatible. Example: "Answer: {{response}}"
            partial_variables: A dictionary of variables to pre-fill in the template,
                reducing the required inputs for format(). Variables specified here will
                be applied during template creation and removed from input_variables list.
                Useful for setting constant context or default values.
                Example: If template is "{context}\nQuery: {query}" and partial_variables
                is {"context": "You are a helpful assistant"}, then only {query} will be
                required when calling format().
                Default: None (no partial variables)
            **kwargs: Additional arguments passed to the PromptTemplate constructor,
                such as:
                - validate_template (bool): Whether to validate template variable consistency
                - output_parser (BaseOutputParser): Parser for LLM output processing
                - metadata (dict): Metadata for tracing
                - tags (list[str]): Tags for tracing

        Returns:
            PromptTemplate: A new PromptTemplate instance with:
                - template: The original template string
                - input_variables: List of required variable names (excluding partial_variables)
                - template_format: The specified template format
                - partial_variables: The provided partial variables dictionary
                - Additional attributes from **kwargs

        Raises:
            ValueError: If template_format is not one of "f-string", "jinja2", or "mustache"
            ImportError: If template_format is "jinja2" but jinja2 package is not installed
            ValueError: If validate_template=True and template contains invalid variable
                references or mismatched brackets

        Example:
            ```python
            from langchain_core.prompts import PromptTemplate

            # Basic f-string template (recommended)
            prompt = PromptTemplate.from_template(
                "Tell me a {adjective} joke about {topic}"
            )
            result = prompt.format(adjective="funny", topic="chickens")
            # result: "Tell me a funny joke about chickens"

            # Jinja2 template with conditional logic
            prompt = PromptTemplate.from_template(
                "Hello {{ name }}{% if age %}, you are {{ age }} years old{% endif %}",
                template_format="jinja2"
            )
            result = prompt.format(name="Alice", age=30)
            # result: "Hello Alice, you are 30 years old"

            # Mustache template with nested variables
            prompt = PromptTemplate.from_template(
                "User: {{user.name}}, Email: {{user.email}}",
                template_format="mustache"
            )
            result = prompt.format(user={"name": "Bob", "email": "bob@example.com"})
            # result: "User: Bob, Email: bob@example.com"

            # Template with partial variables for reusable context
            prompt = PromptTemplate.from_template(
                "System: {system_context}\\n\\nUser: {user_input}\\n\\nAssistant:",
                partial_variables={"system_context": "You are a helpful AI assistant"}
            )
            # Only user_input is required now
            result = prompt.format(user_input="What is Python?")
            # result: "System: You are a helpful AI assistant\\n\\nUser: What is Python?\\n\\nAssistant:"

            # Template with validation enabled
            prompt = PromptTemplate.from_template(
                "Question: {question}\\nContext: {context}",
                validate_template=True
            )
            # Validation ensures template syntax is correct at creation time
            ```

        Source: libs/core/langchain_core/prompts/prompt.py:251-303
        """
        input_variables = get_template_variables(template, template_format)
        partial_variables_ = partial_variables or {}

        if partial_variables_:
            input_variables = [
                var for var in input_variables if var not in partial_variables_
            ]

        return cls(
            input_variables=input_variables,
            template=template,
            template_format=template_format,
            partial_variables=partial_variables_,
            **kwargs,
        )
