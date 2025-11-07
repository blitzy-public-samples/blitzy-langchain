# Type Annotation Patterns - Type-Safe LangChain Development

Welcome to the type annotation patterns examples! This directory demonstrates comprehensive type-safety techniques for LangChain development, helping you achieve 100% type annotation coverage and resolve the "type opacity" challenge in LCEL (LangChain Expression Language) compositions.

## 🎯 Overview

### Purpose

These examples demonstrate **type annotation best practices** for building type-safe LangChain applications with complete IDE support, early error detection, and self-documenting code. Each example is executable, mypy-compliant, and progressively introduces advanced type patterns.

**Key Challenge Addressed**: LangChain's LCEL pipe operators (`|`) perform implicit type transformations that hide the actual data flow (e.g., `dict → PromptValue → AIMessage → str`). These examples make those transformations explicit through comprehensive type annotations.

### Target Audience

- **Primary**: Developers building production LangChain applications requiring maximum type safety
- **Secondary**: Teams adopting mypy strict mode for LangChain projects
- **Developers with <2 years experience**: Clear type hints enable faster onboarding and reduce runtime errors
- **Basic LangChain knowledge recommended**: Completion of `basic_chains/` examples helpful but not required

### Learning Objectives

By completing these examples, you will master:

1. **LCEL Type Flows**: Document type transformations through pipe operator compositions
2. **Pydantic Integration**: Structure input/output validation with runtime and static type checking
3. **Custom Runnable Implementation**: Build type-safe custom chains using `Runnable[Input, Output]` protocol
4. **mypy Compliance**: Achieve 100% type annotation coverage passing `mypy --strict`
5. **Type Aliases**: Improve code readability with clear type definitions
6. **Generic Type Parameters**: Create flexible, reusable components with `Generic[T]`

### Why Type Annotations Matter

**Benefits for Development**:
- ✅ **IDE Support**: Autocomplete, inline documentation, intelligent refactoring
- ✅ **Early Error Detection**: Catch type mismatches before runtime
- ✅ **Self-Documenting Code**: Type hints serve as always-accurate documentation
- ✅ **Refactoring Safety**: Type checker validates breaking changes
- ✅ **Junior Developer Enablement**: Clear APIs accelerate learning (per Agent Action Plan §0.1.1)
- ✅ **Production Reliability**: Fewer runtime type errors in deployed systems

### Time Commitment

- **Individual examples**: 15-30 minutes each
- **Complete category**: 60-90 minutes for all examples
- **Additional time**: 15-20 minutes for mypy setup and validation

## ✅ Prerequisites

### Required Knowledge

- **Python Type Hints**: Understanding of `typing` module, `List`, `Dict`, `Optional`, `Union`
- **Python Generics**: Familiarity with `Generic[T]` and type parameters
- **Protocol Classes**: Basic understanding of structural typing (duck typing)
- **Command Line**: Running Python scripts, pip/uv package management
- **Recommended**: Completed `basic_chains/` examples for LangChain foundation

### Python Version

- **Required**: Python >=3.10.0, <4.0.0
- **Tested**: Python 3.10, 3.13
- **Recommended**: Python 3.13 (latest tested version with full typing features)

Verify your Python version:
```bash
python --version
```

### Pydantic Knowledge (Recommended)

- **Pydantic v2** experience helpful for `pydantic_model_chain.py`
- Understanding of `BaseModel`, `Field`, validators
- Not required: Examples include comprehensive inline explanations

### mypy Installation Required

These examples require mypy for type checking validation:

```bash
# Verify mypy installation
mypy --version

# If not installed, install with:
pip install mypy>=1.18.2
```

Expected output: `mypy 1.18.2` or higher

## 📦 Installation Instructions

### Step 1: Navigate to Type Patterns Directory

```bash
# From repository root
cd examples/type_patterns

# Verify location
pwd
```

Expected output: `*/examples/type_patterns`

### Step 2: Create Virtual Environment

**Why virtual environment?** Isolates dependencies, prevents version conflicts, enables clean mypy checking.

```bash
# Create virtual environment
python -m venv venv

# Verify creation
ls venv/
```

You should see: `bin/` (or `Scripts/` on Windows), `lib/`, `pydantic.cfg`

### Step 3: Activate Virtual Environment

**Linux/macOS**:
```bash
source venv/bin/activate
```

**Windows**:
```bash
venv\Scripts\activate
```

**Verification**: Your prompt should show `(venv)` prefix:
```
(venv) user@machine:~/examples/type_patterns$
```

### Step 4: Install Dependencies

Choose your preferred package manager:

**Using uv (recommended - fastest)**:
```bash
uv pip install -r requirements.txt
```

**Using pip (standard)**:
```bash
pip install -r requirements.txt
```

**Installation includes**:
- `langchain-core>=1.0.1` - Core abstractions with Runnable protocol
- `langchain-openai>=0.3.35` - OpenAI integration for examples
- `pydantic>=2.12.3` - Data validation with type annotations
- `mypy>=1.18.2` - Static type checker for validation
- Additional dependencies listed in `requirements.txt`

**Verification**:
```bash
python -c "import langchain_core; import mypy; print('All dependencies installed!')"
```

### Step 5: Verify mypy Installation

Confirm mypy is available and working:

```bash
mypy --version
```

Expected: `mypy 1.18.2` or higher

**Test mypy on a simple file**:
```bash
echo "def greet(name: str) -> str: return f'Hello {name}'" > test_type.py
mypy --strict test_type.py
rm test_type.py
```

Expected: `Success: no issues found in 1 source file`

### Step 6: Set Up Environment Variables (Optional)

Most type pattern examples work without API keys. For examples requiring OpenAI:

```bash
# Copy template
cp .env.example .env 2>/dev/null || echo "OPENAI_API_KEY=your-key-here" > .env

# Edit with your API key (optional)
nano .env  # Linux/macOS
notepad .env  # Windows
```

**Note**: `custom_chain_typing.py` and `lcel_type_annotations.py` can run without API keys using mock mode.

## 📁 Examples Directory Structure

This directory contains the following files:

```
type_patterns/
├── README.md                    # This file - comprehensive guide
├── requirements.txt             # Python dependencies including mypy
├── pydantic_model_chain.py     # Example 1: Pydantic model integration
├── lcel_type_annotations.py    # Example 2: LCEL type flow documentation
└── custom_chain_typing.py      # Example 3: Custom Runnable implementation
```

### Example Files Detailed Description

#### 1️⃣ `pydantic_model_chain.py` - Structured Data Validation

**Purpose**: Demonstrates Pydantic model integration for type-safe input/output validation.

**Key Concepts**:
- `BaseModel` for structured data with field validation
- `PydanticOutputParser` for validated LLM outputs
- Runtime validation combined with static type checking
- Automatic JSON schema generation
- Field validators and constraints (`ge`, `le`, `Field(...)`)

**Type Patterns Demonstrated**:
- Pydantic `BaseModel` with typed fields
- `Field` with validation constraints
- `field_validator` decorators for custom validation
- Integration with LangChain `PydanticOutputParser`
- Type flow: `InputModel → Dict → Prompt → LLM → Parser → OutputModel`

**Complexity**: ⭐ Beginner-Intermediate

**Estimated Time**: 15-20 minutes

**Prerequisites**: Basic Pydantic knowledge helpful

---

#### 2️⃣ `lcel_type_annotations.py` - LCEL Type Flow Documentation

**Purpose**: Resolves LCEL "type opacity" by documenting explicit type transformations through pipe operators.

**Key Concepts**:
- `Runnable[Input, Output]` protocol for type specification
- `TypeAlias` for complex type definitions
- Documenting implicit type conversions in LCEL chains
- Type flow through: `Dict → ChatPromptValue → AIMessage → str`
- Making invisible type transformations visible

**Type Patterns Demonstrated**:
- `TypeAlias` for readable type definitions
- `Runnable[InputType, OutputType]` explicit annotations
- Type flow documentation with inline comments
- Composition type safety: `Runnable[A,B] | Runnable[B,C] → Runnable[A,C]`
- Union types for flexible inputs

**Complexity**: ⭐⭐ Intermediate

**Estimated Time**: 20-30 minutes

**Prerequisites**: Understanding of LCEL pipe operators (`|`) from `basic_chains/lcel_basic_composition.py`

---

#### 3️⃣ `custom_chain_typing.py` - Type-Safe Custom Chains

**Purpose**: Shows how to implement custom chains with complete type safety using the Runnable protocol.

**Key Concepts**:
- Implementing `Runnable[Input, Output]` protocol
- `Generic[Input, Output]` for flexible type parameters
- Method overrides with `@override` decorator
- Complete type annotations on `invoke`, `ainvoke`, `batch`, `stream`
- Type-safe chain composition with custom chains

**Type Patterns Demonstrated**:
- `Runnable[Input, Output]` protocol implementation
- `Generic[T]` for flexible type parameters
- `@override` for type-safe method overriding
- Method signature compatibility with parent class
- Composing custom chains with LCEL: `CustomChain | Prompt | LLM`

**Complexity**: ⭐⭐⭐ Advanced

**Estimated Time**: 25-30 minutes

**Prerequisites**: Understanding of Python protocols, generics, and method overrides

---

### Supporting Files

| File | Description |
|------|-------------|
| `requirements.txt` | All Python package dependencies with version constraints, includes mypy for type checking |

## 🛤️ Recommended Learning Path

Follow this sequence for optimal learning progression:

### Path 1: Start with `pydantic_model_chain.py`

**Why first**: Introduces structured data validation with runtime and static type checking.

**Run it**:
```bash
python pydantic_model_chain.py
```

**Check types**:
```bash
mypy --strict pydantic_model_chain.py
```

**What you'll learn**:
- Pydantic model definition with type annotations
- Field-level validation with `Field()` constraints
- `PydanticOutputParser` for structured LLM outputs
- Type-safe input/output handling
- Validation error handling patterns

**Time**: 15-20 minutes

---

### Path 2: Progress to `lcel_type_annotations.py`

**Why second**: Builds on type fundamentals, introduces LCEL type flow documentation.

**Run it**:
```bash
python lcel_type_annotations.py
```

**Check types**:
```bash
mypy --strict lcel_type_annotations.py
```

**What you'll learn**:
- `TypeAlias` for complex type definitions
- `Runnable[Input, Output]` explicit typing
- Documenting type transformations through pipe operators
- Making implicit LCEL conversions explicit
- Type flow patterns: `Dict → Prompt → LLM → Parser → str`

**Time**: 20-30 minutes

---

### Path 3: Master `custom_chain_typing.py`

**Why third**: Advanced patterns requiring understanding of protocols and generics.

**Run it**:
```bash
python custom_chain_typing.py
```

**Check types**:
```bash
mypy --strict custom_chain_typing.py
```

**What you'll learn**:
- Implementing `Runnable[Input, Output]` protocol
- Generic type parameters for flexible chains
- Method overrides with `@override` decorator
- Complete type safety in custom chain implementations
- Composing custom chains with existing LangChain components

**Time**: 25-30 minutes

---

### Quick Reference Table

| Example | Difficulty | Key Type Patterns | Time | Prerequisites |
|---------|-----------|-------------------|------|---------------|
| `pydantic_model_chain.py` | ⭐ Beginner-Intermediate | `BaseModel`, `Field`, validators | 15-20 min | Basic Pydantic |
| `lcel_type_annotations.py` | ⭐⭐ Intermediate | `TypeAlias`, `Runnable[I,O]` | 20-30 min | LCEL basics |
| `custom_chain_typing.py` | ⭐⭐⭐ Advanced | `Generic[T]`, `@override`, protocol | 25-30 min | Protocols, generics |

**Total estimated time**: 60-90 minutes for complete mastery

## 📖 Type Annotation Concepts Explained

### Type Opacity Problem

**Definition**: Hidden type transformations in LCEL compositions where the actual data types flowing through pipe operators are invisible to developers and type checkers.

**Example of Type Opacity**:
```python
# What developers see:
chain = prompt | model | parser

# What actually happens (hidden):
# Dict[str, str] → ChatPromptTemplate.invoke() → ChatPromptValue
# ChatPromptValue → ChatOpenAI.invoke() → AIMessage  
# AIMessage → StrOutputParser.invoke() → str
```

**Solution**: Explicit type annotations with `Runnable[Input, Output]` and inline type flow comments.

### Runnable Protocol

**Definition**: Core LangChain interface using `Generic[Input, Output]` for composable components.

**Signature**:
```python
class Runnable(Generic[Input, Output]):
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        ...
```

**Key Benefits**:
- Type-safe composition: `Runnable[A,B] | Runnable[B,C]` → `Runnable[A,C]`
- Compile-time type checking with mypy
- Clear API contracts for custom chains

### Type Aliases

**Purpose**: Improve readability by naming complex type definitions.

**Example**:
```python
from typing_extensions import TypeAlias

# Without TypeAlias (harder to read)
def process(data: Dict[str, Union[str, int, List[str]]]) -> Dict[str, Any]:
    ...

# With TypeAlias (clearer)
InputDict: TypeAlias = Dict[str, Union[str, int, List[str]]]
OutputDict: TypeAlias = Dict[str, Any]

def process(data: InputDict) -> OutputDict:
    ...
```

### Pydantic Integration

**Purpose**: Combine runtime validation with static type checking.

**Pattern**:
```python
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=150)

# Provides:
# 1. Static type checking (mypy sees types)
# 2. Runtime validation (invalid data raises ValidationError)
# 3. Automatic JSON schema generation
# 4. IDE autocomplete and inline docs
```

### Generic Types

**Purpose**: Create flexible, reusable components that work with multiple types.

**Pattern**:
```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
    
    def get(self) -> T:
        return self.value

# Usage:
int_container: Container[int] = Container(42)
str_container: Container[str] = Container("hello")
```

### Method Overrides

**Purpose**: Ensure type safety when subclassing with method overrides.

**Pattern**:
```python
from typing_extensions import override

class BaseChain:
    def invoke(self, input: str) -> str:
        ...

class CustomChain(BaseChain):
    @override  # Type checker validates signature matches parent
    def invoke(self, input: str) -> str:
        return input.upper()
```

### Type Flow

**Definition**: Documenting type transformations at each stage of chain composition.

**Best Practice**:
```python
# Type Flow Documentation:
# Stage 1: Dict[str, str] (input)
# Stage 2: Dict[str, str] → ChatPromptTemplate → ChatPromptValue
# Stage 3: ChatPromptValue → ChatOpenAI → AIMessage
# Stage 4: AIMessage → StrOutputParser → str (output)
# Complete: Runnable[Dict[str, str], str]

chain: Runnable[Dict[str, str], str] = prompt | model | parser
```

## 🏆 Type Safety Best Practices

### 1. Always Use Explicit Type Annotations

**❌ Avoid**:
```python
def create_chain():
    prompt = ChatPromptTemplate.from_messages([...])
    model = ChatOpenAI()
    return prompt | model
```

**✅ Prefer**:
```python
def create_chain() -> Runnable[Dict[str, str], AIMessage]:
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([...])
    model: ChatOpenAI = ChatOpenAI()
    return prompt | model
```

### 2. Leverage Runnable[Input, Output] for Custom Chains

**❌ Avoid**:
```python
class MyChain:
    def invoke(self, input):
        return self.process(input)
```

**✅ Prefer**:
```python
class MyChain(Runnable[str, int]):
    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> int:
        return len(input)
```

### 3. Document Type Transformations with Inline Comments

**❌ Avoid**:
```python
chain = prompt | model | parser
```

**✅ Prefer**:
```python
# Type flow: Dict[str, str] → ChatPromptValue → AIMessage → str
chain: Runnable[Dict[str, str], str] = prompt | model | parser
```

### 4. Use Pydantic Models for Complex Input/Output

**❌ Avoid**:
```python
def process(data: Dict[str, Any]) -> Dict[str, Any]:
    # Unclear what keys are required/expected
    ...
```

**✅ Prefer**:
```python
class Input(BaseModel):
    query: str
    filters: List[str]

class Output(BaseModel):
    result: str
    confidence: float

def process(data: Input) -> Output:
    # Clear contract, validated at runtime
    ...
```

### 5. Use TypeAlias for Improved Readability

**❌ Avoid**:
```python
def transform(
    data: Dict[str, Union[str, int, List[Dict[str, Any]]]]
) -> List[Dict[str, Union[str, float]]]:
    ...
```

**✅ Prefer**:
```python
InputData: TypeAlias = Dict[str, Union[str, int, List[Dict[str, Any]]]]
OutputData: TypeAlias = List[Dict[str, Union[str, float]]]

def transform(data: InputData) -> OutputData:
    ...
```

### 6. Enable mypy --strict Mode

**Configuration** (`mypy.ini` or `pyproject.toml`):
```ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_generics = True
disallow_subclassing_any = True
disallow_untyped_calls = True
disallow_incomplete_defs = True
check_untyped_defs = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
strict = True
```

**Benefit**: Maximum type safety, catches errors early (per Agent Action Plan §0.1.1)

### 7. Add Type Flow Comments Per Agent Action Plan

**Purpose**: Make implicit LCEL transformations explicit for junior developers (§0.1.1)

**Pattern**:
```python
# LCEL Type Flow Documentation:
# Input:  Dict[str, str] with keys: {"question": str}
# Step 1: ChatPromptTemplate.invoke() → ChatPromptValue (List[BaseMessage])
# Step 2: ChatOpenAI.invoke() → AIMessage
# Step 3: StrOutputParser.invoke() → str
# Output: str (extracted message content)

chain: Runnable[Dict[str, str], str] = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)
```

## 🔍 Type Checking with mypy

### Installation

If not already installed:

```bash
pip install mypy>=1.18.2
```

Verify installation:

```bash
mypy --version
```

Expected: `mypy 1.18.2` or higher

### Basic Type Checking

Check a single file:

```bash
mypy pydantic_model_chain.py
```

Expected output (if types correct):
```
Success: no issues found in 1 source file
```

### Strict Mode (Recommended)

Run with maximum type safety (per Agent Action Plan §0.1.1):

```bash
mypy --strict pydantic_model_chain.py
```

**All examples in this directory should pass `mypy --strict`**

### Check Multiple Files

```bash
mypy *.py
```

### Common mypy Options

```bash
# Show error codes
mypy --show-error-codes pydantic_model_chain.py

# Verbose output
mypy --verbose pydantic_model_chain.py

# Generate HTML report
mypy --html-report mypy-report/ pydantic_model_chain.py
```

### Expected Behavior

**All examples should produce**:
```
Success: no issues found in X source file(s)
```

If you see errors, refer to "Common Type Issues and Solutions" section below.

### mypy Configuration

Create `mypy.ini` in your project root for consistent checking:

```ini
[mypy]
python_version = 3.10
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

Or add to `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
```

## 🚀 Usage Instructions

### Running Examples

Each example is self-contained and executable:

```bash
# Run Pydantic model integration example
python pydantic_model_chain.py

# Run LCEL type annotations example
python lcel_type_annotations.py

# Run custom chain typing example
python custom_chain_typing.py
```

### Validating Types

After running, validate type correctness:

```bash
# Check single file
mypy --strict pydantic_model_chain.py

# Check all examples
mypy --strict *.py

# Generate detailed report
mypy --strict --show-error-codes --pretty *.py
```

### Expected Outputs

Each example produces:

#### `pydantic_model_chain.py`:
```
=== Pydantic Model Chain Example ===

Input Model (validated):
{
  "name": "Alice Johnson",
  "age": 28,
  "interests": ["reading", "hiking", "coding"],
  "location": "San Francisco"
}

Type Flow:
PersonQuery (Pydantic) → dict → ChatPromptValue → AIMessage → PersonBio (validated)

Output Model:
{
  "summary": "Alice is a 28-year-old from San Francisco...",
  "personality_traits": ["curious", "adventurous", "analytical"],
  "career_suggestion": "Software Engineer or Data Scientist",
  "confidence_score": 0.85
}
```

#### `lcel_type_annotations.py`:
```
=== LCEL Type Annotations Example ===

Type Flow Documentation:
Stage 1: Dict[str, str] {"topic": "Python"}
Stage 2: Dict[str, str] → ChatPromptTemplate → ChatPromptValue
Stage 3: ChatPromptValue → ChatOpenAI → AIMessage
Stage 4: AIMessage → StrOutputParser → str

Result: "Python is a high-level programming language..."
```

#### `custom_chain_typing.py`:
```
=== Custom Chain Typing Example ===

Custom Chain: TextAnalysisChain
Type Signature: Runnable[TextAnalysisInput, TextAnalysisOutput]

Input: {"text": "Sample text", "options": {...}}

Type Flow:
TextAnalysisInput → [internal processing] → TextAnalysisOutput

Output:
{
  "word_count": 42,
  "sentiment": "positive",
  "keywords": ["sample", "text", "analysis"]
}
```

### Environment Variables

Most examples work without API keys. For examples requiring OpenAI:

| Example | Required Variables | Fallback Mode |
|---------|-------------------|---------------|
| `pydantic_model_chain.py` | `OPENAI_API_KEY` (optional) | Uses mock data if missing |
| `lcel_type_annotations.py` | `OPENAI_API_KEY` (optional) | Demonstrates types without API calls |
| `custom_chain_typing.py` | None | Fully self-contained |

## 📋 Key Type Patterns Reference

### Common Type Patterns

| Pattern | Example | Use Case |
|---------|---------|----------|
| `Runnable[I, O]` | `Runnable[str, int]` | Custom chain type signature |
| `TypeAlias` | `InputDict: TypeAlias = Dict[str, str]` | Readable type definitions |
| `Generic[T]` | `class Chain(Generic[T])` | Flexible type parameters |
| `Pydantic BaseModel` | `class Input(BaseModel)` | Structured validation |
| `Optional[T]` | `Optional[str]` | Nullable values |
| `Union[A, B]` | `Union[str, int]` | Multiple allowed types |
| `List[T]` | `List[str]` | Homogeneous lists |
| `Dict[K, V]` | `Dict[str, Any]` | Dictionary with typed keys/values |
| `Callable[[A], B]` | `Callable[[str], int]` | Function type signatures |
| `Literal["a", "b"]` | `Literal["debug", "info"]` | Fixed value sets |

### LangChain-Specific Type Patterns

| Pattern | Example | Description |
|---------|---------|-------------|
| `Runnable[Input, Output]` | `Runnable[Dict[str, str], str]` | Base protocol for all chains |
| `RunnableSerializable[I, O]` | `RunnableSerializable[str, str]` | Serializable chains |
| `ChatPromptTemplate` | Type: `Runnable[Dict, ChatPromptValue]` | Prompt with messages |
| `ChatOpenAI` | Type: `Runnable[ChatPromptValue, AIMessage]` | LLM invocation |
| `StrOutputParser` | Type: `Runnable[AIMessage, str]` | Extract message content |
| `PydanticOutputParser[T]` | `PydanticOutputParser[PersonBio]` | Parse to Pydantic model |
| `RunnablePassthrough` | Type: `Runnable[T, T]` | Pass data unchanged |
| `RunnableLambda[I, O]` | `RunnableLambda[str, int]` | Custom transform function |

## 🔄 Type Flow Documentation Examples

### Standard LCEL Type Flows

#### Pattern 1: Simple Prompt → LLM → Parser

```python
# Type Flow:
# Dict[str, str] → ChatPromptTemplate → ChatPromptValue
# ChatPromptValue → ChatOpenAI → AIMessage
# AIMessage → StrOutputParser → str
# Complete: Runnable[Dict[str, str], str]

chain: Runnable[Dict[str, str], str] = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)
```

#### Pattern 2: Pydantic Input → Validated Output

```python
# Type Flow:
# InputModel (Pydantic) → dict → ChatPromptValue
# ChatPromptValue → AIMessage → PydanticOutputParser
# PydanticOutputParser → OutputModel (validated)
# Complete: Runnable[InputModel, OutputModel]

chain: Runnable[InputModel, OutputModel] = (
    RunnableLambda(lambda x: x.model_dump())
    | prompt
    | model
    | PydanticOutputParser(pydantic_object=OutputModel)
)
```

#### Pattern 3: Custom Chain Composition

```python
# Type Flow:
# CustomInput → CustomChain.invoke() → CustomOutput
# CustomOutput → ChatPromptTemplate → ChatPromptValue
# ChatPromptValue → ChatOpenAI → AIMessage
# Complete: Runnable[CustomInput, AIMessage]

custom_chain: Runnable[CustomInput, CustomOutput] = CustomChain()
full_chain: Runnable[CustomInput, AIMessage] = (
    custom_chain
    | prompt_template
    | model
)
```

#### Pattern 4: Parallel Execution with Type Merging

```python
# Type Flow:
# str → RunnableParallel → {"summary": str, "keywords": List[str]}
# Parallel branches:
#   - Branch 1: str → summarize_chain → str
#   - Branch 2: str → extract_keywords → List[str]
# Complete: Runnable[str, Dict[str, Union[str, List[str]]]]

from langchain_core.runnables import RunnableParallel

ResultDict: TypeAlias = Dict[str, Union[str, List[str]]]

chain: Runnable[str, ResultDict] = RunnableParallel({
    "summary": summarize_chain,
    "keywords": extract_keywords_chain
})
```

### Visual Type Flow Diagram

```
Input Type (Dict[str, str])
    ↓
ChatPromptTemplate.invoke()
    ↓
ChatPromptValue (List[BaseMessage])
    ↓
ChatOpenAI.invoke()
    ↓
AIMessage (content: str, metadata: dict)
    ↓
StrOutputParser.invoke()
    ↓
Output Type (str)
```

## ⚠️ Common Type Issues and Solutions

### Issue 1: Type 'dict' is not assignable to type 'Dict[str, str]'

**Error**:
```
error: Incompatible types in assignment (expression has type "dict[str, str]", 
variable has type "Dict[str, str]")
```

**Cause**: Using lowercase `dict` instead of capitalized `Dict` from typing module.

**Solution**:
```python
# ❌ Wrong
from typing import Dict

data: Dict[str, str] = dict(key="value")  # Error with mypy --strict

# ✅ Correct
from typing import Dict

data: Dict[str, str] = {"key": "value"}  # OK

# Or use Python 3.9+ syntax:
data: dict[str, str] = {"key": "value"}  # OK in Python 3.9+
```

### Issue 2: Cannot infer type of Runnable

**Error**:
```
error: Need type annotation for "chain" (hint: "chain: Runnable[<type>, <type>] = ...")
```

**Cause**: Missing explicit type annotations on custom chain instances.

**Solution**:
```python
# ❌ Wrong
chain = prompt | model | parser  # Type cannot be inferred

# ✅ Correct
chain: Runnable[Dict[str, str], str] = prompt | model | parser
```

### Issue 3: Incompatible types in pipe operator

**Error**:
```
error: Argument 1 has incompatible type "Runnable[AIMessage, str]"; 
expected "Runnable[ChatPromptValue, str]"
```

**Cause**: Output type of first chain doesn't match input type of second chain.

**Solution**:
```python
# ❌ Wrong - Type mismatch
prompt: Runnable[Dict[str, str], ChatPromptValue] = ...
parser: Runnable[AIMessage, str] = ...
chain = prompt | parser  # Error: ChatPromptValue ≠ AIMessage

# ✅ Correct - Add intermediate step
model: Runnable[ChatPromptValue, AIMessage] = ChatOpenAI()
chain = prompt | model | parser  # OK: types align
```

### Issue 4: mypy errors with 'Any' type

**Error**:
```
error: Returning Any from function declared to return "str"
```

**Cause**: Using `Any` type instead of specific types in strict mode.

**Solution**:
```python
# ❌ Wrong
from typing import Any

def process(data: Any) -> Any:
    return data.upper()

# ✅ Correct
def process(data: str) -> str:
    return data.upper()
```

### Issue 5: Untyped function definition

**Error**:
```
error: Function is missing a type annotation
```

**Cause**: Missing return type or parameter types.

**Solution**:
```python
# ❌ Wrong
def create_chain():
    return prompt | model

# ✅ Correct
def create_chain() -> Runnable[Dict[str, str], str]:
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([...])
    model: ChatOpenAI = ChatOpenAI()
    return prompt | model
```

### Issue 6: Pydantic field validation type mismatch

**Error**:
```
error: Argument "default" to "Field" has incompatible type "int"; expected "str"
```

**Cause**: Field default value doesn't match declared type.

**Solution**:
```python
# ❌ Wrong
class Model(BaseModel):
    count: str = Field(default=0)  # Type mismatch

# ✅ Correct
class Model(BaseModel):
    count: int = Field(default=0)  # Correct type
```

### Issue 7: Missing @override decorator

**Error**:
```
error: Signature of "invoke" incompatible with supertype "Runnable"
```

**Cause**: Method override signature doesn't match parent class.

**Solution**:
```python
from typing_extensions import override

# ❌ Wrong
class CustomChain(Runnable[str, int]):
    def invoke(self, input: str) -> int:  # Missing @override
        return len(input)

# ✅ Correct
class CustomChain(Runnable[str, int]):
    @override
    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> int:
        return len(input)
```

## 🎁 Benefits of Type Annotations

### 1. IDE Support

**Autocomplete**:
```python
# With type hints, IDE knows exact methods available
chain: Runnable[str, int] = ...
result = chain.  # IDE shows: invoke, ainvoke, batch, stream, etc.
```

**Inline Documentation**:
- Hover over variables to see types
- Navigate to type definitions
- View parameter and return types instantly

**Refactoring**:
- Rename variables/functions safely across entire codebase
- Find all usages with type information
- Automated import organization

### 2. Early Error Detection

**Catch errors before runtime**:
```python
# mypy catches this at development time:
chain: Runnable[Dict[str, str], str] = prompt | model | parser
result: int = chain.invoke({"q": "test"})  # Error: str ≠ int
```

**Benefit**: Fix issues in seconds vs. hours debugging production failures.

### 3. Self-Documenting Code

**Type hints as documentation**:
```python
def create_qa_chain(
    llm: ChatOpenAI,
    retriever: BaseRetriever,
    memory: Optional[ConversationBufferMemory] = None
) -> Runnable[Dict[str, str], Dict[str, Any]]:
    """
    No need to document parameter types - they're in the signature!
    """
    ...
```

### 4. Refactoring Safety

**Type checker validates breaking changes**:
```python
# Change function signature
def old_process(data: str) -> str: ...
def new_process(data: Dict[str, str]) -> str: ...  # Changed input type

# mypy reports ALL call sites that need updating
result = new_process("test")  # Error: str ≠ Dict[str, str]
```

### 5. Junior Developer Enablement

**Clear APIs accelerate learning** (per Agent Action Plan §0.1.1):

- Type hints show exactly what inputs are expected
- IDE autocomplete guides correct usage
- Errors caught immediately with helpful messages
- No need to read extensive documentation for basic usage

**Impact**: Developers with <2 years experience can independently implement LangChain chains.

### 6. Production Reliability

**Fewer runtime type errors**:
- Type checking catches mismatches before deployment
- Reduces exception rates in production
- Improves system stability and uptime
- Decreases debugging time for type-related issues

## 🔗 Integration with LangChain Components

### Type-Safe Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

# Explicit type: Runnable[Dict[str, str], ChatPromptValue]
prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

# Type checker ensures correct input
result = prompt.invoke({"question": "What is Python?"})  # ✅ OK
result = prompt.invoke("What is Python?")  # ❌ Error: str ≠ Dict
```

### Type-Annotated LLM Wrappers

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

# Explicit type: Runnable[ChatPromptValue, AIMessage]
model: ChatOpenAI = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Type-safe invocation
message: AIMessage = model.invoke(prompt_value)
```

### Typed Output Parsers

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel

# String parser: Runnable[AIMessage, str]
str_parser: StrOutputParser = StrOutputParser()

# JSON parser: Runnable[AIMessage, Dict[str, Any]]
json_parser: JsonOutputParser = JsonOutputParser()

# Pydantic parser: Runnable[AIMessage, OutputModel]
class OutputModel(BaseModel):
    answer: str
    confidence: float

pydantic_parser: PydanticOutputParser[OutputModel] = (
    PydanticOutputParser(pydantic_object=OutputModel)
)
```

### Type-Safe Callbacks

```python
from langchain_core.callbacks import BaseCallbackHandler
from typing_extensions import override

class TypedCallbackHandler(BaseCallbackHandler):
    @override
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        print(f"Chain started with inputs: {inputs}")
    
    @override
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        print(f"Chain completed with outputs: {outputs}")
```

### Composition Example

```python
from typing_extensions import TypeAlias

# Define clear type flow
InputType: TypeAlias = Dict[str, str]
PromptType: TypeAlias = ChatPromptValue
LLMType: TypeAlias = AIMessage
OutputType: TypeAlias = str

# Build type-safe chain
prompt: Runnable[InputType, PromptType] = ChatPromptTemplate.from_messages([...])
model: Runnable[PromptType, LLMType] = ChatOpenAI()
parser: Runnable[LLMType, OutputType] = StrOutputParser()

# Compose with type checking
chain: Runnable[InputType, OutputType] = prompt | model | parser
```

## 🚀 Advanced Type Patterns

### Conditional Types with Union

```python
from typing import Union

# Accept multiple input types
def process(data: Union[str, Dict[str, str]]) -> str:
    if isinstance(data, str):
        return data
    else:
        return data.get("text", "")

# Or use type narrowing
InputData: TypeAlias = Union[str, Dict[str, str]]

def process_v2(data: InputData) -> str:
    match data:
        case str():
            return data
        case dict():
            return data.get("text", "")
```

### Recursive Types for Nested Structures

```python
from typing import Dict, List, Union

# Recursive JSON-like structure
JsonValue: TypeAlias = Union[
    str,
    int,
    float,
    bool,
    None,
    List['JsonValue'],
    Dict[str, 'JsonValue']
]

def process_json(data: JsonValue) -> str:
    # Type checker understands recursive structure
    ...
```

### Protocol Classes for Duck Typing

```python
from typing import Protocol

class Retrievable(Protocol):
    def retrieve(self, query: str) -> List[str]:
        ...

# Any class with retrieve() method is compatible
def search(retriever: Retrievable, query: str) -> List[str]:
    return retriever.retrieve(query)
```

### TypeVar for Generic Constraints

```python
from typing import TypeVar, Sequence

T = TypeVar('T', bound=BaseModel)

def validate_batch(items: Sequence[T]) -> List[T]:
    """Works with any Pydantic model."""
    return [item for item in items if item]

# Usage
class User(BaseModel):
    name: str

users: List[User] = validate_batch([User(name="Alice"), User(name="Bob")])
```

### Literal Types for Fixed Values

```python
from typing import Literal

# Restrict to specific values
LogLevel: TypeAlias = Literal["debug", "info", "warning", "error"]

def log(message: str, level: LogLevel) -> None:
    print(f"[{level.upper()}] {message}")

log("Starting", "info")  # ✅ OK
log("Error", "critical")  # ❌ Error: "critical" not in Literal
```

### Callable Types for Functions

```python
from typing import Callable

# Function that takes str and returns int
TransformFunc: TypeAlias = Callable[[str], int]

def apply_transform(data: str, func: TransformFunc) -> int:
    return func(data)

# Usage
def count_chars(text: str) -> int:
    return len(text)

result: int = apply_transform("hello", count_chars)
```

## 📚 Next Steps

### 1. Explore Debugging Examples

Learn how to troubleshoot type-safe chains:
```bash
cd ../debugging
cat README.md
```

**Topics covered**:
- Logging configuration for typed chains
- Debugging type errors with callback handlers
- Chain introspection utilities

### 2. Review API Reference Documentation

Detailed type documentation for all LangChain components:
- [Runnables API Reference](../../docs/api-reference/runnables/base.md)
- [Prompts API Reference](../../docs/api-reference/prompts/templates.md)
- [Output Parsers API Reference](../../docs/api-reference/utilities/output-parsers.md)

### 3. Production Deployment Guide

Learn production-ready patterns:
- [Production Deployment Guide](../../docs/guides/production-deployment.md)
- Type-safe error handling strategies
- Monitoring and observability for typed chains

### 4. Explore Async Type Patterns

Type annotations for async/await chains:
```bash
cd ../advanced_chains
python async_chain_execution.py
```

**Key concepts**:
- `async def ainvoke(input: T) -> U`
- `AsyncIterator[T]` for streaming
- Type-safe async context managers

### 5. Configure mypy for Your Project

Set up project-wide type checking:

**Create `mypy.ini`**:
```ini
[mypy]
python_version = 3.10
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_generics = True

[mypy-langchain.*]
ignore_missing_imports = True

[mypy-openai.*]
ignore_missing_imports = True
```

**Or use `pyproject.toml`**:
```toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true

[[tool.mypy.overrides]]
module = ["langchain.*", "openai.*"]
ignore_missing_imports = true
```

**Add to CI/CD**:
```yaml
# .github/workflows/type-check.yml
- name: Type check with mypy
  run: mypy --strict src/
```

## 📖 References

### Python Typing Documentation

- **Official Python typing module**: https://docs.python.org/3/library/typing.html
- **PEP 484 - Type Hints**: https://www.python.org/dev/peps/pep-0484/
- **PEP 585 - Type Hinting Generics**: https://www.python.org/dev/peps/pep-0585/
- **typing_extensions**: https://pypi.org/project/typing-extensions/

### Pydantic Documentation

- **Pydantic v2 Documentation**: https://docs.pydantic.dev/
- **Field Validation**: https://docs.pydantic.dev/latest/concepts/fields/
- **Model Configuration**: https://docs.pydantic.dev/latest/concepts/models/
- **JSON Schema**: https://docs.pydantic.dev/latest/concepts/json_schema/

### mypy Documentation

- **mypy Documentation**: https://mypy.readthedocs.io/
- **mypy Strict Mode**: https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-strict
- **Type Checking Tips**: https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
- **Common Issues**: https://mypy.readthedocs.io/en/stable/common_issues.html

### LangChain Type Documentation

- **Runnable Protocol**: [docs/api-reference/runnables/base.md](../../docs/api-reference/runnables/base.md)
- **LCEL Guide**: [docs/guides/lcel-composition.md](../../docs/guides/lcel-composition.md)
- **Type Safety Guide**: [docs/guides/type-annotations.md](../../docs/guides/type-annotations.md) (if exists)

### Agent Action Plan References

- **Type Annotation Requirements**: Agent Action Plan §0.1.1
- **100% Type Coverage Goal**: Agent Action Plan §0.1.1
- **Type Opacity Resolution**: Agent Action Plan §0.1.1
- **Junior Developer Enablement**: Agent Action Plan §0.1.1
- **Documentation Format Standards**: Agent Action Plan §0.9.2

## 🛠️ Troubleshooting

### mypy Installation Issues

**Problem**: `mypy: command not found`

**Solution**:
```bash
# Ensure pip is up to date
pip install --upgrade pip

# Install mypy
pip install mypy>=1.18.2

# Verify
mypy --version
```

**Problem**: `ModuleNotFoundError: No module named 'mypy'`

**Solution**:
```bash
# Ensure you're in the correct virtual environment
which python  # Should show venv/bin/python

# Reinstall in current environment
pip install --force-reinstall mypy
```

### Type Checking Performance Concerns

**Problem**: `mypy` takes a long time to check files

**Solutions**:

1. **Use incremental mode** (default):
   ```bash
   mypy --incremental pydantic_model_chain.py
   ```

2. **Check specific modules only**:
   ```bash
   mypy --follow-imports=skip pydantic_model_chain.py
   ```

3. **Use daemon mode** for faster repeated checks:
   ```bash
   dmypy run -- pydantic_model_chain.py
   ```

4. **Disable strict mode temporarily**:
   ```bash
   mypy pydantic_model_chain.py  # Without --strict
   ```

### Integration with IDEs

#### VS Code

**Install Python extension**: `ms-python.python`

**Configure mypy in `settings.json`**:
```json
{
    "python.linting.mypyEnabled": true,
    "python.linting.mypyArgs": [
        "--strict",
        "--ignore-missing-imports"
    ]
}
```

**Enable type checking on save**:
```json
{
    "python.analysis.typeCheckingMode": "strict"
}
```

#### PyCharm

**Enable mypy**:
1. Open Settings → Tools → External Tools
2. Add New Tool:
   - Name: mypy
   - Program: `$ProjectFileDir$/venv/bin/mypy`
   - Arguments: `--strict $FilePath$`
3. Assign keyboard shortcut in Keymap settings

**Built-in type checking**:
- PyCharm has built-in type checking (no mypy needed)
- Configure in Settings → Editor → Inspections → Python → Type Checker

### Type Stub Files for External Packages

**Problem**: `error: Skipping analyzing "langchain_core": module is installed, but missing library stubs or py.typed marker`

**Solution**:

1. **Install type stubs if available**:
   ```bash
   pip install types-langchain-core  # If available
   ```

2. **Or ignore missing imports** in `mypy.ini`:
   ```ini
   [mypy-langchain_core.*]
   ignore_missing_imports = True
   ```

3. **Or create stub files** (advanced):
   ```bash
   stubgen -p langchain_core -o stubs/
   ```

### Example Not Running

**Problem**: `ModuleNotFoundError: No module named 'langchain_core'`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt

# Verify
python -c "import langchain_core; print('OK')"
```

**Problem**: `openai.AuthenticationError: No API key provided`

**Solution**:
```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Or export temporarily
export OPENAI_API_KEY="your-key-here"

# Run example
python pydantic_model_chain.py
```

## 🔗 Quick Links to Examples

- **[Pydantic Model Chain](./pydantic_model_chain.py)** - Structured data validation with `BaseModel` (⭐ Beginner-Intermediate)
- **[LCEL Type Annotations](./lcel_type_annotations.py)** - Type flow documentation patterns (⭐⭐ Intermediate)
- **[Custom Chain Typing](./custom_chain_typing.py)** - Type-safe custom Runnable implementation (⭐⭐⭐ Advanced)

---

## 📝 Summary

This directory provides comprehensive examples for achieving **100% type annotation coverage** in LangChain applications. By following these patterns, you will:

✅ **Resolve type opacity** in LCEL compositions  
✅ **Enable IDE autocomplete** and inline documentation  
✅ **Catch type errors early** with mypy --strict  
✅ **Create self-documenting code** with clear type hints  
✅ **Build production-ready** type-safe chains  
✅ **Accelerate junior developer onboarding** with clear APIs  

**Start with**: `pydantic_model_chain.py` → `lcel_type_annotations.py` → `custom_chain_typing.py`

**Time investment**: 60-90 minutes for complete mastery

**Result**: Type-safe LangChain development with maximum IDE support and early error detection per Agent Action Plan §0.1.1.

---

**Questions or Issues?** Review the Troubleshooting section or check the [LangChain documentation](https://docs.langchain.com/).

