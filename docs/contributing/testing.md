# Testing Documentation Changes

This guide provides complete validation command reference and integrated CI pipeline instructions for testing LangChain integration documentation changes. Following these procedures ensures 100% type annotation coverage, executable example validation, and documentation build integrity per Agent Action Plan requirements.

## Overview

Documentation validation encompasses multiple layers of testing to ensure quality, accuracy, and maintainability:

- **Type Checking**: Verify 100% type annotation coverage with zero type errors using mypy strict mode
- **Example Validation**: Ensure all executable example files run successfully without modification
- **Docstring Compliance**: Validate Google-style docstring format and complete coverage
- **Markdown Validation**: Check markdown syntax and formatting consistency
- **Link Validation**: Detect broken internal and external links in generated documentation
- **Build Validation**: Confirm documentation builds without errors or warnings

## Prerequisites

Install all required testing and validation dependencies:

```bash
# Install documentation build and validation tools
uv pip install -r docs/requirements.txt

# Verify installation
mkdocs --version
mypy --version
pytest --version
ruff --version
```

**Required Python Version**: Python >=3.10.0,<4.0.0

**Tested Python Versions**: Python 3.10, 3.13

## Documentation Build Commands

### Local Development Build

Build documentation locally for development and testing:

```bash
# Build documentation site (output to site/ directory)
mkdocs build

# Build with clean output (removes previous build artifacts)
mkdocs build --clean

# Build with strict mode (warnings treated as errors)
mkdocs build --strict
```

**Expected Output**: Documentation built successfully in `site/` directory

### Live Preview with Hot Reload

Preview documentation with automatic rebuild on file changes:

```bash
# Start local development server with live reload
mkdocs serve

# Server starts at: http://127.0.0.1:8000
# Press Ctrl+C to stop server
```

**Usage**: Open http://127.0.0.1:8000 in your browser. Documentation automatically rebuilds when you save changes to markdown files.

### Production Build

Build documentation for production deployment with all optimizations:

```bash
# Production build with strict validation
mkdocs build --clean --strict

# Output directory: site/
# Fail-fast behavior: Build stops on first warning or error
```

**Validation**: This command ensures production-ready documentation with zero warnings.

## Example Code Validation

### Running All Examples

Execute all example files to verify they run without errors:

```bash
# Run all example files with summary output
pytest examples/ --tb=short

# Run with verbose output showing each example
pytest examples/ -v

# Run with detailed failure information
pytest examples/ --tb=long
```

**Expected Result**: All tests pass, confirming examples execute successfully.

### Running Specific Example Categories

Test specific example directories:

```bash
# Test basic chain examples only
pytest examples/basic_chains/ -v

# Test advanced chain examples
pytest examples/advanced_chains/ -v

# Test type pattern examples
pytest examples/type_patterns/ -v

# Test debugging utility examples
pytest examples/debugging/ -v
```

### Testing with API Keys

Examples requiring external API access:

```bash
# Test with actual API key (if available)
OPENAI_API_KEY=your-actual-key-here pytest examples/ --disable-warnings

# Test with mock/graceful skipping (no API key required)
pytest examples/ --tb=short
```

**Note**: Examples should gracefully handle missing API keys by either skipping tests or using mock responses. They should never fail due to missing credentials alone.

### Code Style Validation for Examples

Ensure example code follows project style guidelines:

```bash
# Check example code style with ruff
ruff check examples/

# Auto-fix style issues
ruff check --fix examples/

# Format example code
ruff format examples/
```

**Source**: Style checking pattern from `libs/core/Makefile` line 53 and `libs/langchain/Makefile` line 57

## Type Checking Commands

### Comprehensive Type Checking

Validate 100% type annotation coverage with strict type checking:

```bash
# Type check all LangChain integration source code with strict mode
mypy --strict libs/langchain/langchain_classic libs/core/langchain_core

# This is the exact command from libs/core/Makefile line 55
# Target: Zero type errors for production-ready code
```

**Expected Result**: No type errors reported. Any errors must be resolved before documentation is considered complete.

### Targeted Type Checking

Type check specific modules:

```bash
# Type check chains module only
mypy --strict libs/langchain/langchain_classic/chains/

# Type check runnables module only
mypy --strict libs/core/langchain_core/runnables/

# Type check with specific cache directory
mkdir -p .mypy_cache && mypy --strict libs/ --cache-dir .mypy_cache
```

### Type Coverage Reporting

Generate reports showing type annotation coverage:

```bash
# Generate HTML report for visual inspection
mypy --strict --html-report mypy-report/ libs/langchain/langchain_classic libs/core/langchain_core

# Open report: open mypy-report/index.html (macOS) or xdg-open mypy-report/index.html (Linux)

# Generate any-expressions report to track annotation percentage
mypy --strict --any-exprs-report mypy-coverage/ libs/langchain/langchain_classic libs/core/langchain_core

# This helps identify areas with incomplete type annotations
```

**Coverage Target**: 100% type annotation coverage per Agent Action Plan section 0.1.1

## Docstring Validation Commands

### Google-Style Docstring Format Checking

Validate all docstrings follow Google-style conventions:

```bash
# Check docstring format compliance (Google style required per AGENTS.md line 133)
pydocstyle --convention=google libs/langchain/langchain_classic/

# Check core module docstrings
pydocstyle --convention=google libs/core/langchain_core/

# Check with verbose output showing all issues
pydocstyle --convention=google --verbose libs/
```

**Required Format**: All docstrings must include Args, Returns, Raises, and Example sections per Agent Action Plan template.

### Docstring Coverage Measurement

Measure docstring coverage percentage:

```bash
# Check docstring coverage with verbose output and fail if below 100%
interrogate -vv --fail-under=100 libs/langchain/langchain_classic/

# Check core module coverage
interrogate -vv --fail-under=100 libs/core/langchain_core/

# Generate detailed coverage report
interrogate -vv --generate-badge . libs/
```

**Coverage Target**: 100% complete docstrings per Agent Action Plan requirement

**Failure Handling**: If coverage is below 100%, interrogate lists all functions missing docstrings. Add complete documentation to those functions.

## Markdown Validation

### Markdown Syntax and Style Checking

Validate markdown files follow consistent formatting:

```bash
# Lint all documentation markdown files
markdownlint 'docs/**/*.md'

# Lint example README files
markdownlint 'examples/**/*.md'

# Lint with auto-fix for correctable issues
markdownlint 'docs/**/*.md' --fix

# Check specific file
markdownlint docs/contributing/testing.md
```

**Configuration**: Markdown rules are defined in `.markdownlint.json` (if present) or use default GitHub Flavored Markdown rules.

### Checking for Incomplete Documentation Markers

Detect TODO, FIXME, or other pending work markers:

```bash
# Search for incomplete documentation indicators
grep -r "TODO\|FIXME\|XXX" docs/ examples/ && echo "⚠️  Found pending items" || echo "✓ No pending items"

# Search with line numbers for easier location
grep -rn "TODO\|FIXME\|XXX" docs/ examples/

# Exclude this from testing files that legitimately document these patterns
grep -r "TODO\|FIXME\|XXX" docs/ examples/ --exclude="*testing.md"
```

**Zero Placeholder Policy**: Per Agent Action Plan, all documentation must be complete with no placeholders, stubs, or deferred work indicators.

## Link Validation

### Validating Internal and External Links

Check all links in generated documentation resolve correctly:

```bash
# First, build the documentation
mkdocs build --clean

# Then validate all links in the generated site
linkchecker site/

# Check with verbose output
linkchecker site/ --verbose

# Check only internal links (faster for development)
linkchecker site/ --check-extern=false

# Generate report file
linkchecker site/ --output=html > link-report.html
```

**Expected Result**: Zero broken links. All internal documentation references and external URLs must resolve successfully.

**Common Issues**:
- Broken internal links: Usually due to incorrect relative paths in markdown files
- External link failures: May be temporary network issues; verify manually if persistent
- Anchor link failures: Section headings changed without updating references

## Integrated Validation Pipeline

### Complete Validation Script

Run all validation checks in sequence with fail-fast behavior:

```bash
#!/bin/bash
# Complete documentation validation pipeline
# Save as: scripts/validate_docs.sh
# Usage: bash scripts/validate_docs.sh

set -e  # Exit immediately if any command fails

echo "=================================="
echo "Documentation Validation Pipeline"
echo "=================================="
echo ""

echo "Step 1: Type Checking..."
echo "------------------------"
mypy --strict libs/langchain/langchain_classic libs/core/langchain_core
echo "✓ Type checking passed"
echo ""

echo "Step 2: Example Validation..."
echo "----------------------------"
pytest examples/ --tb=short
echo "✓ Example validation passed"
echo ""

echo "Step 3: Documentation Build..."
echo "-----------------------------"
mkdocs build --clean --strict
echo "✓ Documentation build passed"
echo ""

echo "Step 4: Link Validation..."
echo "-------------------------"
linkchecker site/ --check-extern
echo "✓ Link validation passed"
echo ""

echo "Step 5: Code Style Validation..."
echo "--------------------------------"
ruff check examples/
echo "✓ Example code style passed"
echo ""

echo "Step 6: Markdown Validation..."
echo "------------------------------"
markdownlint 'docs/**/*.md' 'examples/**/*.md'
echo "✓ Markdown validation passed"
echo ""

echo "Step 7: Docstring Format Validation..."
echo "--------------------------------------"
pydocstyle --convention=google libs/langchain/langchain_classic/ libs/core/langchain_core/
echo "✓ Docstring format passed"
echo ""

echo "Step 8: Docstring Coverage Check..."
echo "-----------------------------------"
interrogate -vv --fail-under=100 libs/langchain/langchain_classic/ libs/core/langchain_core/
echo "✓ Docstring coverage passed"
echo ""

echo "=================================="
echo "✓ All validation checks passed!"
echo "=================================="
```

**Usage**:

```bash
# Make script executable
chmod +x scripts/validate_docs.sh

# Run complete validation
bash scripts/validate_docs.sh
```

**Execution Order Rationale**:
1. Type checking first (fastest, catches fundamental issues)
2. Example validation (ensures code examples work)
3. Documentation build (verifies structure and syntax)
4. Link checking (requires built documentation)
5. Style validation (aesthetic and consistency checks)

## Troubleshooting Common Validation Failures

### MyPy Type Errors

**Issue**: `error: Function is missing a type annotation`

**Solution**: Add type hints to function signature:
```python
# Before (missing types)
def process_data(input_data):
    return input_data

# After (with types)
def process_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    return input_data
```

**Issue**: `error: Need type annotation for variable`

**Solution**: Add explicit type annotation:
```python
# Before
results = []

# After
results: List[str] = []
```

**Issue**: `error: Argument has incompatible type`

**Solution**: Verify imported types match actual usage. Check dependency file exports match your imports.

### Example Execution Failures

**Issue**: `TimeoutError` during example execution

**Solution**: Add timeout handling or reduce example complexity:
```python
# Add timeout to prevent hanging
import signal
signal.alarm(30)  # 30 second timeout
```

**Issue**: `ModuleNotFoundError` for example imports

**Solution**: 
1. Verify `examples/*/requirements.txt` includes all dependencies
2. Install example dependencies: `uv pip install -r examples/basic_chains/requirements.txt`
3. Ensure PYTHONPATH includes repository root

**Issue**: `ConnectionError` or API authentication failures

**Solution**: Examples should gracefully handle missing API keys:
```python
import os
if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("API key not available")
```

### Broken Internal Links

**Issue**: `linkchecker` reports broken internal documentation links

**Solution**:
1. Verify target file exists in `docs/` directory
2. Check relative path is correct from source file location
3. Use repository-relative paths: `../guides/lcel-composition.md` not absolute paths
4. Verify anchor links match exact heading text: `#type-checking-commands`

**Issue**: Links work in preview but fail in built site

**Solution**: 
1. Ensure `mkdocs.yml` includes target page in navigation
2. Check for typos in path casing (case-sensitive on Linux/Unix)
3. Rebuild with `mkdocs build --clean` to clear cache

### Markdownlint Rule Violations

**Issue**: `MD001/heading-increment: Heading levels should only increment by one level at a time`

**Solution**: Fix heading hierarchy:
```markdown
<!-- Before (skips from # to ###) -->
# Main Heading
### Subheading

<!-- After (proper hierarchy) -->
# Main Heading
## Subheading
```

**Issue**: `MD012/no-multiple-blanks: Multiple consecutive blank lines`

**Solution**: Remove extra blank lines (max 1 blank line between sections)

**Issue**: `MD041/first-line-heading: First line in file should be a top level heading`

**Solution**: Ensure file starts with `# Heading`

### Docstring Coverage Failures

**Issue**: `interrogate` reports missing docstrings

**Solution**: Add complete Google-style docstring:
```python
def create_chain(llm, prompt):
    """Create a simple chain from LLM and prompt.
    
    Args:
        llm: Language model instance implementing Runnable protocol.
        prompt: Prompt template with input variables.
    
    Returns:
        Runnable chain combining prompt and LLM.
    
    Raises:
        ValueError: If llm or prompt is None.
    
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> llm = ChatOpenAI()
        >>> prompt = ChatPromptTemplate.from_template("Say {input}")
        >>> chain = create_chain(llm, prompt)
    """
    if llm is None or prompt is None:
        raise ValueError("Both llm and prompt are required")
    return prompt | llm
```

**Issue**: Docstring present but coverage check fails

**Solution**: Verify docstring includes ALL required sections: Args, Returns, Raises, Example

## CI/CD Integration

### GitHub Actions Workflow Configuration

Add documentation validation to CI pipeline (`.github/workflows/docs-build.yml`):

```yaml
name: Documentation Build and Validation

on:
  pull_request:
    paths:
      - 'docs/**'
      - 'examples/**'
      - 'libs/langchain/langchain_classic/**/*.py'
      - 'libs/core/langchain_core/**/*.py'
      - 'mkdocs.yml'
      - 'docs/requirements.txt'
  push:
    branches:
      - main

jobs:
  validate-docs:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.13']
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install uv
        run: pip install uv
      
      - name: Install dependencies
        run: uv pip install -r docs/requirements.txt
      
      - name: Run type checking
        run: mypy --strict libs/langchain/langchain_classic libs/core/langchain_core
      
      - name: Validate examples
        run: pytest examples/ --tb=short
      
      - name: Build documentation
        run: mkdocs build --clean --strict
      
      - name: Validate links
        run: linkchecker site/ --check-extern
      
      - name: Check code style
        run: ruff check examples/
      
      - name: Validate markdown
        run: markdownlint 'docs/**/*.md' 'examples/**/*.md'
      
      - name: Check docstring format
        run: pydocstyle --convention=google libs/langchain/langchain_classic/ libs/core/langchain_core/
      
      - name: Check docstring coverage
        run: interrogate -vv --fail-under=100 libs/langchain/langchain_classic/ libs/core/langchain_core/
```

### Pre-Commit Hook Integration

The repository uses pre-commit hooks defined in `.pre-commit-config.yaml`. Documentation changes in `libs/core/` and `libs/langchain/` automatically trigger format and lint checks.

**Verify pre-commit setup**:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files

# Run hooks on staged files only
pre-commit run
```

**Source**: Pre-commit configuration from `.pre-commit-config.yaml` uses `make -C libs/core format lint` and `make -C libs/langchain format lint`

## Quick Reference

### Essential Commands

```bash
# Quick validation (most common checks)
mkdocs build --strict && pytest examples/ --tb=short && mypy --strict libs/

# Full validation (all checks)
bash scripts/validate_docs.sh

# Local preview
mkdocs serve

# Fix auto-correctable style issues
ruff format examples/ && markdownlint 'docs/**/*.md' --fix
```

### Validation Checklist

Before submitting documentation changes, verify:

- [ ] Type checking passes: `mypy --strict libs/`
- [ ] All examples execute: `pytest examples/`
- [ ] Documentation builds: `mkdocs build --strict`
- [ ] No broken links: `linkchecker site/`
- [ ] Code style compliant: `ruff check examples/`
- [ ] Markdown valid: `markdownlint 'docs/**/*.md'`
- [ ] Docstrings complete: `interrogate -vv --fail-under=100 libs/`
- [ ] No placeholders: `grep -r "TODO\|FIXME" docs/ examples/` returns empty

## Additional Resources

- **Agent Action Plan**: Section 0.9.1 for complete documentation build requirements
- **Style Guidelines**: `AGENTS.md` line 133 for Google-style docstring requirement
- **Make Targets**: See `libs/core/Makefile` and `libs/langchain/Makefile` for available commands
- **Type Annotation Standards**: PEP 484 compliant using `typing` module
- **Docstring Template**: See Agent Action Plan section 0.1.2 for required template format

## Summary

This testing guide provides comprehensive validation procedures ensuring documentation quality, accuracy, and maintainability. Following these steps guarantees:

- **100% Type Annotation Coverage**: All public APIs have complete type hints validated with mypy strict mode
- **Executable Examples**: All example files run successfully without modification
- **Complete Documentation**: All public APIs have Google-style docstrings with Args, Returns, Raises, and Example sections
- **Build Integrity**: Documentation builds without errors or warnings
- **Link Integrity**: All internal and external links resolve correctly
- **Style Consistency**: Code and markdown follow project standards

By integrating these validation procedures into your development workflow and CI/CD pipeline, you ensure documentation changes meet the highest quality standards before merging.
