"""BasePrompt schema definition."""

from __future__ import annotations

import warnings
from abc import ABC
from collections.abc import Callable, Sequence
from string import Formatter
from typing import Any, Literal

from pydantic import BaseModel, create_model

from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.utils import get_colored_text, mustache
from langchain_core.utils.formatting import formatter
from langchain_core.utils.interactive_env import is_interactive_env

try:
    from jinja2 import Environment, meta
    from jinja2.sandbox import SandboxedEnvironment

    _HAS_JINJA2 = True
except ImportError:
    _HAS_JINJA2 = False

PromptTemplateFormat = Literal["f-string", "mustache", "jinja2"]


def jinja2_formatter(template: str, /, **kwargs: Any) -> str:
    """Format a template using jinja2.

    This function uses Jinja2's templating engine to render templates with variables.
    Jinja2 supports advanced features like loops, conditionals, filters, and more.

    *Security warning*:
        As of LangChain 0.0.329, this method uses Jinja2's
        SandboxedEnvironment by default. However, this sand-boxing should
        be treated as a best-effort approach rather than a guarantee of security.
        Do not accept jinja2 templates from untrusted sources as they may lead
        to arbitrary Python code execution.

        **CRITICAL**: Never use jinja2 templates with user-controlled or unverified
        inputs. Even with sandboxing, malicious templates can potentially execute
        arbitrary Python code. Always validate and sanitize template sources.

        https://jinja.palletsprojects.com/en/3.1.x/sandbox/

    Args:
        template: The template string with Jinja2 syntax (e.g., "Hello {{ name }}!").
        **kwargs: The variables to format the template with.

    Returns:
        The formatted string with all template variables replaced.

    Raises:
        ImportError: If jinja2 is not installed.
        jinja2.TemplateSyntaxError: If the template contains invalid Jinja2 syntax.
        jinja2.UndefinedError: If a required template variable is missing from kwargs.

    Example:
        Basic variable substitution:
        >>> jinja2_formatter("Hello {{ name }}!", name="World")
        'Hello World!'

        Using filters:
        >>> jinja2_formatter("{{ name|upper }}", name="world")
        'WORLD'

        Using conditionals:
        >>> jinja2_formatter("{% if admin %}Admin{% else %}User{% endif %}", admin=True)
        'Admin'

        Using loops:
        >>> jinja2_formatter("{% for item in items %}{{ item }},{% endfor %}", items=["a", "b"])
        'a,b,'

    Source: libs/core/langchain_core/prompts/string.py:30
    """
    if not _HAS_JINJA2:
        msg = (
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
            "Please be cautious when using jinja2 templates. "
            "Do not expand jinja2 templates using unverified or user-controlled "
            "inputs as that can result in arbitrary Python code execution."
        )
        raise ImportError(msg)

    # This uses a sandboxed environment to prevent arbitrary code execution.
    # Jinja2 uses an opt-out rather than opt-in approach for sand-boxing.
    # Please treat this sand-boxing as a best-effort approach rather than
    # a guarantee of security.
    # We recommend to never use jinja2 templates with untrusted inputs.
    # https://jinja.palletsprojects.com/en/3.1.x/sandbox/
    # approach not a guarantee of security.
    return SandboxedEnvironment().from_string(template).render(**kwargs)


def validate_jinja2(template: str, input_variables: list[str]) -> None:
    """Validate that the input variables are valid for the template.

    Issues a warning if missing or extra variables are found.

    Args:
        template: The template string.
        input_variables: The input variables.
    """
    input_variables_set = set(input_variables)
    valid_variables = _get_jinja2_variables_from_template(template)
    missing_variables = valid_variables - input_variables_set
    extra_variables = input_variables_set - valid_variables

    warning_message = ""
    if missing_variables:
        warning_message += f"Missing variables: {missing_variables} "

    if extra_variables:
        warning_message += f"Extra variables: {extra_variables}"

    if warning_message:
        warnings.warn(warning_message.strip(), stacklevel=7)


def _get_jinja2_variables_from_template(template: str) -> set[str]:
    if not _HAS_JINJA2:
        msg = (
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
        )
        raise ImportError(msg)
    env = Environment()  # noqa: S701
    ast = env.parse(template)
    return meta.find_undeclared_variables(ast)


def mustache_formatter(template: str, /, **kwargs: Any) -> str:
    """Format a template using mustache.

    Mustache is a logic-less templating system that uses double curly braces {{}}
    for variable substitution. It supports simple variables, sections (loops),
    inverted sections, and nested object access using dot notation.

    Args:
        template: The template string with Mustache syntax (e.g., "Hello {{name}}!").
        **kwargs: The variables to format the template with. Supports nested dicts
            for dot notation access (e.g., {{person.name}}).

    Returns:
        The formatted string with all template variables replaced.

    Example:
        Basic variable substitution:
        >>> mustache_formatter("Hello {{name}}!", name="World")
        'Hello World!'

        Nested object access:
        >>> mustache_formatter("{{person.name}} is {{person.age}}", 
        ...                    person={"name": "Alice", "age": 30})
        'Alice is 30'

        Sections (loops over lists):
        >>> mustache_formatter("{{#items}}{{.}},{{/items}}", items=["a", "b", "c"])
        'a,b,c,'

        Inverted sections (render if false/empty):
        >>> mustache_formatter("{{^items}}No items{{/items}}", items=[])
        'No items'

    Source: libs/core/langchain_core/prompts/string.py:109
    """
    return mustache.render(template, kwargs)


def mustache_template_vars(
    template: str,
) -> set[str]:
    """Get the top-level variables from a mustache template.

    For nested variables like `{{person.name}}`, only the top-level
    key (`person`) is returned.

    Args:
        template: The template string.

    Returns:
       The top-level variables from the template.
    """
    variables: set[str] = set()
    section_depth = 0
    for type_, key in mustache.tokenize(template):
        if type_ == "end":
            section_depth -= 1
        elif (
            type_ in {"variable", "section", "inverted section", "no escape"}
            and key != "."
            and section_depth == 0
        ):
            variables.add(key.split(".")[0])
        if type_ in {"section", "inverted section"}:
            section_depth += 1
    return variables


Defs = dict[str, "Defs"]


def mustache_schema(template: str) -> type[BaseModel]:
    """Get the variables from a mustache template as a Pydantic model.

    This function analyzes a Mustache template and automatically generates a Pydantic
    model class that represents the expected input structure. It parses the template
    to identify variables, sections, and nested structures, then creates a typed
    model with the appropriate hierarchy.

    The generated model can be used for:
    - Type validation of template inputs
    - Auto-generating input schemas for prompts
    - Documentation of expected template structure
    - IDE autocompletion when building template inputs

    Template parsing rules:
    - Simple variables {{name}} become string fields
    - Sections {{#person}}...{{/person}} become nested model fields
    - Nested variables {{person.name}} create hierarchical models
    - Inverted sections {{^items}}...{{/items}} are treated like regular sections

    Args:
        template: The Mustache template string to analyze. Should contain valid
            Mustache syntax with {{variable}}, {{#section}}...{{/section}}, and/or
            {{^inverted}}...{{/inverted}} blocks.

    Returns:
        A dynamically created Pydantic BaseModel subclass (named "PromptInput") that
        represents the template's input structure. The model's fields correspond to
        the template's variables:
        - Top-level variables become direct fields
        - Sections become nested model fields
        - Nested variables (dot notation) create model hierarchies

    Example:
        Simple variables:
        >>> schema = mustache_schema("Hello {{name}}!")
        >>> schema.__fields__.keys()
        dict_keys(['name'])

        Nested variables:
        >>> schema = mustache_schema("{{person.name}} is {{person.age}}")
        >>> hasattr(schema.__fields__['person'].annotation, '__fields__')
        True

        Sections:
        >>> schema = mustache_schema("{{#users}}{{name}}{{/users}}")
        >>> 'users' in schema.__fields__
        True

        Complex nested structure:
        >>> template = "{{#company}}{{name}} - {{#employees}}{{firstName}}{{/employees}}{{/company}}"
        >>> schema = mustache_schema(template)
        >>> 'company' in schema.__fields__
        True

    Note:
        The generated model uses None as default for all fields, making them optional.
        Leaf nodes (simple variables) are typed as str, while nested structures are
        typed as nested Pydantic models.

    Source: libs/core/langchain_core/prompts/string.py:207
    """
    fields = {}
    prefix: tuple[str, ...] = ()
    section_stack: list[tuple[str, ...]] = []
    for type_, key in mustache.tokenize(template):
        if key == ".":
            continue
        if type_ == "end":
            if section_stack:
                prefix = section_stack.pop()
        elif type_ in {"section", "inverted section"}:
            section_stack.append(prefix)
            prefix += tuple(key.split("."))
            fields[prefix] = False
        elif type_ in {"variable", "no escape"}:
            fields[prefix + tuple(key.split("."))] = True

    for fkey, fval in fields.items():
        fields[fkey] = fval and not any(
            is_subsequence(fkey, k) for k in fields if k != fkey
        )
    defs: Defs = {}  # None means leaf node
    while fields:
        field, is_leaf = fields.popitem()
        current = defs
        for part in field[:-1]:
            current = current.setdefault(part, {})
        current.setdefault(field[-1], "" if is_leaf else {})  # type: ignore[arg-type]
    return _create_model_recursive("PromptInput", defs)


def _create_model_recursive(name: str, defs: Defs) -> type:
    return create_model(  # type: ignore[call-overload]
        name,
        **{
            k: (_create_model_recursive(k, v), None) if v else (type(v), None)
            for k, v in defs.items()
        },
    )


DEFAULT_FORMATTER_MAPPING: dict[str, Callable] = {
    "f-string": formatter.format,
    "mustache": mustache_formatter,
    "jinja2": jinja2_formatter,
}
"""Default mapping of template format names to formatting functions.

This dictionary maps template format identifiers to their corresponding formatting
functions. It is used by prompt templates to render templates based on the specified
format type.

Supported formats:
    - "f-string": Python f-string style formatting using {variable} syntax.
      Uses the standard formatter.format function.
    - "mustache": Mustache templating using {{variable}} syntax with support for
      sections and nested objects. Uses mustache_formatter function.
    - "jinja2": Jinja2 templating with advanced features like filters, loops, and
      conditionals using {{ variable }} syntax. Uses jinja2_formatter function.

Usage:
    formatter_func = DEFAULT_FORMATTER_MAPPING["f-string"]
    result = formatter_func("Hello {name}", name="World")

Source: libs/core/langchain_core/prompts/string.py:204
"""

DEFAULT_VALIDATOR_MAPPING: dict[str, Callable] = {
    "f-string": formatter.validate_input_variables,
    "jinja2": validate_jinja2,
}
"""Default mapping of template format names to validation functions.

This dictionary maps template format identifiers to their corresponding validation
functions. Validators check that the provided input variables match the template's
requirements before formatting.

Supported validators:
    - "f-string": Validates that all placeholder variables in the f-string template
      are provided in input_variables. Uses formatter.validate_input_variables.
    - "jinja2": Validates Jinja2 template variables and issues warnings for missing
      or extra variables. Uses validate_jinja2 function.

Note: Mustache format does not have a validator in this mapping as it handles
missing variables gracefully by rendering them as empty strings.

Usage:
    validator_func = DEFAULT_VALIDATOR_MAPPING["f-string"]
    validator_func(template="Hello {name}", input_variables=["name"])

Source: libs/core/langchain_core/prompts/string.py:229
"""


def check_valid_template(
    template: str, template_format: str, input_variables: list[str]
) -> None:
    """Check that template string is valid.

    This function validates that a template string conforms to the specified format
    and that all required input variables are properly defined. It performs two levels
    of validation:
    
    1. Format validation: Ensures the template_format is supported
    2. Variable validation: Ensures input_variables match template requirements

    Args:
        template: The template string to validate. Format depends on template_format.
        template_format: The template format. Should be one of "f-string", "jinja2",
            or "mustache". Note that only "f-string" and "jinja2" have validators;
            mustache templates are not validated by this function.
        input_variables: List of variable names that will be provided when formatting
            the template. Must match the variables referenced in the template.

    Raises:
        ValueError: If the template_format is not one of the supported formats in
            DEFAULT_VALIDATOR_MAPPING. Error message includes the list of valid formats.
        ValueError: If the template contains variable mismatches. This can occur when:
            - Required template variables are missing from input_variables
            - Template references undefined variables
            - Variable names in template don't match input_variables
            The error message will indicate "Invalid prompt schema" and suggest checking
            for mismatched or missing input parameters.
        KeyError: Indirectly raised and caught when template variable lookup fails
            during validation. Converted to ValueError with descriptive message.
        IndexError: Indirectly raised and caught when template parsing encounters
            structural errors. Converted to ValueError with descriptive message.

    Example:
        Valid f-string template:
        >>> check_valid_template("Hello {name}!", "f-string", ["name"])
        # No exception raised

        Invalid - missing variable:
        >>> check_valid_template("Hello {name}!", "f-string", [])
        # Raises ValueError: Invalid prompt schema; check for mismatched or missing...

        Invalid format:
        >>> check_valid_template("Hello {{name}}", "invalid", ["name"])
        # Raises ValueError: Invalid template format 'invalid'...

        Valid jinja2 template:
        >>> check_valid_template("Hello {{ name }}!", "jinja2", ["name"])
        # No exception raised (may issue warning if variables mismatch)

    Source: libs/core/langchain_core/prompts/string.py:268
    """
    try:
        validator_func = DEFAULT_VALIDATOR_MAPPING[template_format]
    except KeyError as exc:
        msg = (
            f"Invalid template format {template_format!r}, should be one of"
            f" {list(DEFAULT_FORMATTER_MAPPING)}."
        )
        raise ValueError(msg) from exc
    try:
        validator_func(template, input_variables)
    except (KeyError, IndexError) as exc:
        msg = (
            "Invalid prompt schema; check for mismatched or missing input parameters"
            f" from {input_variables}."
        )
        raise ValueError(msg) from exc


def get_template_variables(template: str, template_format: str) -> list[str]:
    """Get the variables from the template.

    This function extracts all variable names from a template string based on the
    specified format. It parses the template and identifies placeholders that need
    to be filled when the template is rendered. The extraction method varies by format:

    - **f-string**: Uses Python's string.Formatter to parse {variable} placeholders
    - **jinja2**: Uses Jinja2's AST parser to find {{ variable }} references and
      undeclared variables
    - **mustache**: Uses mustache tokenizer to find {{variable}} placeholders,
      returning only top-level keys for nested variables (e.g., "person" from
      "{{person.name}}")

    Args:
        template: The template string to analyze. The expected syntax depends on
            template_format (e.g., "{name}" for f-string, "{{ name }}" for jinja2,
            "{{name}}" for mustache).
        template_format: The template format identifier. Must be one of:
            - "f-string": Python f-string style with {variable} syntax
            - "jinja2": Jinja2 template style with {{ variable }} syntax
            - "mustache": Mustache template style with {{variable}} syntax

    Returns:
        A sorted list of unique variable names found in the template. Variables are
        returned in alphabetical order for consistency. For mustache templates with
        nested variables like {{person.name}}, only the top-level key ("person") is
        returned.

    Raises:
        ValueError: If template_format is not one of the supported formats
            ("f-string", "jinja2", "mustache"). Error message includes the
            unsupported format name.
        ImportError: If template_format is "jinja2" but jinja2 package is not
            installed. Raised by _get_jinja2_variables_from_template.

    Example:
        Extract variables from f-string template:
        >>> get_template_variables("Hello {name}, you are {age} years old", "f-string")
        ['age', 'name']

        Extract variables from jinja2 template:
        >>> get_template_variables("Hello {{ name }}!", "jinja2")
        ['name']

        Extract variables from mustache template:
        >>> get_template_variables("Hello {{name}}!", "mustache")
        ['name']

        Nested mustache variables return only top-level key:
        >>> get_template_variables("{{person.name}} is {{person.age}}", "mustache")
        ['person']

        Multiple variables are sorted alphabetically:
        >>> get_template_variables("{zebra} {apple} {banana}", "f-string")
        ['apple', 'banana', 'zebra']

    Source: libs/core/langchain_core/prompts/string.py:301
    """
    if template_format == "jinja2":
        # Get the variables for the template
        input_variables = _get_jinja2_variables_from_template(template)
    elif template_format == "f-string":
        input_variables = {
            v for _, v, _, _ in Formatter().parse(template) if v is not None
        }
    elif template_format == "mustache":
        input_variables = mustache_template_vars(template)
    else:
        msg = f"Unsupported template format: {template_format}"
        raise ValueError(msg)

    return sorted(input_variables)


class StringPromptTemplate(BasePromptTemplate, ABC):
    """Base class for all string-based prompt templates.

    StringPromptTemplate serves as the foundation for prompt templates that produce
    string outputs. It extends BasePromptTemplate with string-specific formatting
    capabilities and provides both synchronous and asynchronous formatting methods.

    This class is abstract and should be subclassed to implement specific prompt
    template types. Subclasses must implement the abstract methods from
    BasePromptTemplate, particularly the format() method which defines how the
    template is rendered into a string.

    Key features:
    - Converts formatted strings into StringPromptValue objects for LLM consumption
    - Supports both sync (format) and async (aformat) formatting workflows
    - Provides pretty printing capabilities for debugging and visualization
    - Maintains compatibility with LangChain's Runnable interface for LCEL chains

    Common subclasses:
    - PromptTemplate: Simple f-string or Jinja2 based templates
    - FewShotPromptTemplate: Templates with example-based few-shot learning
    - Custom templates: User-defined templates with specialized formatting logic

    Attributes:
        input_variables: List of variable names required by the template (inherited)
        partial_variables: Pre-filled variables that don't need to be provided each
            time (inherited)

    Methods:
        format_prompt(**kwargs): Format the template and return a StringPromptValue
        aformat_prompt(**kwargs): Async version of format_prompt
        pretty_repr(html=False): Get a visual representation with variable placeholders
        pretty_print(): Print the template with highlighted variables

    Example:
        Creating a custom StringPromptTemplate subclass:
        >>> from langchain_core.prompts.string import StringPromptTemplate
        >>> class CustomPrompt(StringPromptTemplate):
        ...     def format(self, **kwargs) -> str:
        ...         return f"Custom: {kwargs.get('text', '')}"
        ...
        >>> prompt = CustomPrompt(input_variables=["text"])
        >>> prompt.format_prompt(text="Hello")
        StringPromptValue(text='Custom: Hello')

    Source: libs/core/langchain_core/prompts/string.py:329
    """

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "prompts", "base"]`
        """
        return ["langchain", "prompts", "base"]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt with the inputs.

        Args:
            **kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return StringPromptValue(text=self.format(**kwargs))

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """Async format the prompt with the inputs.

        Args:
            **kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return StringPromptValue(text=await self.aformat(**kwargs))

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """Get a pretty representation of the prompt.

        Args:
            html: Whether to return an HTML-formatted string.

        Returns:
            A pretty representation of the prompt.
        """
        # TODO: handle partials
        dummy_vars = {
            input_var: "{" + f"{input_var}" + "}" for input_var in self.input_variables
        }
        if html:
            dummy_vars = {
                k: get_colored_text(v, "yellow") for k, v in dummy_vars.items()
            }
        return self.format(**dummy_vars)

    def pretty_print(self) -> None:
        """Print a pretty representation of the prompt."""
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


def is_subsequence(child: Sequence, parent: Sequence) -> bool:
    """Return True if child is subsequence of parent."""
    if len(child) == 0 or len(parent) == 0:
        return False
    if len(parent) < len(child):
        return False
    return all(child[i] == parent[i] for i in range(len(child)))
