# SQL Query Chain API Reference

## Overview

The `create_sql_query_chain` factory function creates a modern LCEL-based chain for translating natural language questions into SQL queries. This function replaces the deprecated `SQLDatabaseChain` with a more flexible and composable implementation using LangChain Expression Language.

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py`

---

## ⚠️ CRITICAL SECURITY WARNINGS

!!! danger "SQL Injection and Data Leakage Risks"
    
    **This chain generates SQL queries with potential security implications:**
    
    - **SQL Injection Risk**: The LLM generates SQL queries that are executed against your database. Malicious or crafted questions could potentially generate harmful queries.
    - **Sensitive Data Exposure**: The `db.get_table_info()` method exposes column information and sample data from tables, which may leak sensitive information to the LLM.
    - **Unauthorized Data Access**: Without proper scoping, users may access tables they shouldn't have access to.
    
    **Required Mitigation Strategies:**
    
    1. **Read-Only Permissions**: Ensure database connection uses read-only credentials
    2. **Table Scoping**: Use `SQLInputWithTables` to restrict accessible tables
    3. **Access Control**: Implement authentication/authorization before allowing chain submission
    4. **Audit Logging**: Log all generated queries and user requests
    5. **Review Permissions**: Limit database user to only necessary tables and columns
    
    **See:** [LangChain Security Documentation](https://python.langchain.com/docs/security) for comprehensive security guidelines.

---

## Function Signature

```python
def create_sql_query_chain(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    prompt: BasePromptTemplate | None = None,
    k: int = 5,
    *,
    get_col_comments: bool | None = None,
) -> Runnable[SQLInput | SQLInputWithTables | dict[str, Any], str]
```

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:33-40`

---

## Parameters

### `llm: BaseLanguageModel`

The language model used to generate SQL queries from natural language questions.

**Type:** `langchain_core.language_models.BaseLanguageModel`

**Purpose:** Interprets the user's natural language question and database schema to produce syntactically correct SQL queries.

**Supported Models:** Any LangChain-compatible language model:
- OpenAI models (e.g., `gpt-3.5-turbo`, `gpt-4`)
- Anthropic Claude models
- Open-source models (e.g., via Ollama)
- Azure OpenAI models

**Example:**
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

---

### `db: SQLDatabase`

Database wrapper providing connection and schema introspection capabilities.

**Type:** `langchain_community.utilities.sql_database.SQLDatabase`

**Purpose:** Provides database connection, schema information retrieval, and table metadata access.

**Key Methods Used Internally:**
- `db.get_table_info()`: Retrieves table schemas and sample data
- `db.dialect`: Database dialect (e.g., "sqlite", "postgresql", "mysql")

**Example:**
```python
from langchain_community.utilities import SQLDatabase

# SQLite database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# PostgreSQL database
db = SQLDatabase.from_uri(
    "postgresql://user:password@localhost:5432/mydb",
    include_tables=["customers", "orders"]  # Restrict to specific tables
)
```

---

### `prompt: BasePromptTemplate | None = None`

Custom prompt template for SQL generation (optional).

**Type:** `langchain_core.prompts.BasePromptTemplate | None`

**Default:** Dialect-specific prompt from `SQL_PROMPTS` dictionary, or `PROMPT` if dialect not found

**Required Input Variables:**
- `input` (str): User question with suffix `"\nSQLQuery: "` appended
- `top_k` (str): Number of results to limit queries to
- `table_info` (str): Table schemas and sample rows
- `dialect` (str, optional): Database dialect (auto-filled if variable present)

**Prompt Selection Logic:**
1. If `prompt` parameter provided → use custom prompt
2. Else if `db.dialect` in `SQL_PROMPTS` → use dialect-specific prompt
3. Else → use default `PROMPT` template

**Supported Dialects with Custom Prompts:**
- `crate` → CrateDB expert prompt
- `duckdb` → DuckDB expert prompt
- `googlesql` → GoogleSQL expert prompt
- `mssql` → MS SQL expert prompt
- `mysql` → MySQL expert prompt
- `mariadb` → MariaDB expert prompt
- `oracle` → Oracle SQL expert prompt
- `postgresql` → PostgreSQL expert prompt
- `sqlite` → SQLite expert prompt
- `clickhouse` → ClickHouse expert prompt
- `prestodb` → PrestoDB expert prompt

**Source:** `libs/langchain/langchain_classic/chains/sql_database/prompt.py:270-282`

**Example Custom Prompt:**
```python
from langchain_core.prompts import PromptTemplate

template = '''Given an input question, create a syntactically correct {dialect} query.

Use the following format:
Question: "Question here"
SQLQuery: "SQL Query to run"

Only use the following tables:
{table_info}

Question: {input}'''

custom_prompt = PromptTemplate(
    input_variables=["input", "table_info", "top_k", "dialect"],
    template=template
)
```

---

### `k: int = 5`

Maximum number of results to return per SELECT statement.

**Type:** `int`

**Default:** `5`

**Purpose:** Controls the `LIMIT` clause (or equivalent) in generated SQL queries to prevent retrieving excessive data.

**Usage:** Passed to prompt as `top_k` variable, instructing the LLM to limit query results.

**Example:**
```python
# Return up to 10 results
chain = create_sql_query_chain(llm, db, k=10)
```

---

### `get_col_comments: bool | None = None`

Whether to retrieve column comments alongside table information.

**Type:** `bool | None`

**Default:** `None` (do not retrieve comments)

**Supported Dialects:** Only `postgresql`, `mysql`, and `oracle`

**Purpose:** Includes column-level comments in the table schema information passed to the LLM, providing additional context for query generation.

**Validation:** Raises `ValueError` if set to `True` for unsupported dialects.

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:136-144`

**Example:**
```python
# PostgreSQL database with column comments
chain = create_sql_query_chain(llm, db, get_col_comments=True)
```

---

## Returns

### Return Type

```python
Runnable[SQLInput | SQLInputWithTables | dict[str, Any], str]
```

A LangChain `Runnable` that accepts input as `SQLInput`, `SQLInputWithTables`, or a dictionary, and returns a SQL query string.

**Input Types:**
- `SQLInput`: Dictionary with `question` key for unrestricted table access
- `SQLInputWithTables`: Dictionary with `question` and `table_names_to_use` keys for restricted access
- `dict[str, Any]`: Generic dictionary containing required keys

**Output Type:**
- `str`: Generated SQL query string (stripped of whitespace)

---

## Input Schemas

### SQLInput

Basic input schema for unrestricted database queries.

```python
class SQLInput(TypedDict):
    """Input for SQL Chain with unrestricted table access."""
    question: str
```

**Fields:**
- `question` (str): Natural language question to translate into SQL

**Usage:** Use when all database tables should be accessible for query generation.

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:20-24`

**Example:**
```python
input_data = {"question": "How many employees are there?"}
result = chain.invoke(input_data)
```

---

### SQLInputWithTables

Enhanced input schema with table access restrictions (SECURITY RECOMMENDED).

```python
class SQLInputWithTables(TypedDict):
    """Input for SQL Chain with restricted table access."""
    question: str
    table_names_to_use: list[str]
```

**Fields:**
- `question` (str): Natural language question to translate into SQL
- `table_names_to_use` (list[str]): Explicit list of table names to include in schema information

**Security Benefit:** Restricts `db.get_table_info()` to only specified tables, preventing unauthorized table access.

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:26-30`

**Example:**
```python
# Restrict to customer and orders tables only
input_data = {
    "question": "What are the top 5 customers by order count?",
    "table_names_to_use": ["customers", "orders"]
}
result = chain.invoke(input_data)
```

---

## LCEL Type Flow and Composition

The `create_sql_query_chain` function returns a complex LCEL chain with the following composition:

```mermaid
graph LR
    A[Input Dict] --> B[RunnablePassthrough.assign]
    B --> C[Filter Keys]
    C --> D[Prompt.partial]
    D --> E[LLM.bind]
    E --> F[StrOutputParser]
    F --> G[_strip]
    G --> H[SQL Query String]
    
    style A fill:#e1f5ff
    style H fill:#c8e6c9
    style B fill:#fff9c4
    style D fill:#fff9c4
    style E fill:#ffccbc
```

### Detailed Phase Breakdown

#### Phase 1: Input Assignment and Transformation

**Component:** `RunnablePassthrough.assign(**inputs)`

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:146-154`

**Operations:**
```python
inputs = {
    "input": lambda x: x["question"] + "\nSQLQuery: ",
    "table_info": lambda x: db.get_table_info(
        table_names=x.get("table_names_to_use"),
        **table_info_kwargs,
    ),
}
```

**Transformation:**
- **Input:** `{"question": str, "table_names_to_use": list[str] | None}`
- **Output:** `{"question": str, "table_names_to_use": list[str] | None, "input": str, "table_info": str}`

**Purpose:**
- Adds `"input"` key with formatted question (appends `"\nSQLQuery: "`)
- Adds `"table_info"` key with database schema information
- Preserves original `"question"` and `"table_names_to_use"` keys

---

#### Phase 2: Key Filtering

**Component:** Lambda function filtering dictionary keys

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:156-161`

**Operations:**
```python
lambda x: {
    k: v
    for k, v in x.items()
    if k not in ("question", "table_names_to_use")
}
```

**Transformation:**
- **Input:** `{"question": str, "table_names_to_use": list[str] | None, "input": str, "table_info": str}`
- **Output:** `{"input": str, "table_info": str}`

**Purpose:** Removes `"question"` and `"table_names_to_use"` keys, retaining only prompt-relevant variables.

---

#### Phase 3: Prompt Partial Application

**Component:** `prompt_to_use.partial(top_k=str(k))`

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:162`

**Transformation:**
- **Input:** `{"input": str, "table_info": str}`
- **Output:** `List[BaseMessage]` (formatted prompt messages)

**Purpose:**
- Populates prompt template with `input`, `table_info`, and `top_k` variables
- Generates LLM-ready messages with database context

---

#### Phase 4: LLM Generation with Stop Sequence

**Component:** `llm.bind(stop=["\nSQLResult:"])`

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:163`

**Transformation:**
- **Input:** `List[BaseMessage]`
- **Output:** `AIMessage` or `str` (depending on LLM)

**Purpose:**
- Generates SQL query using language model
- Stops generation at `"\nSQLResult:"` marker to prevent extraneous output

---

#### Phase 5: String Extraction and Cleanup

**Components:** `StrOutputParser()` → `_strip`

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:164-165`

**Transformation:**
- **Input:** `AIMessage` or `str`
- **Output:** `str` (cleaned SQL query)

**Operations:**
1. `StrOutputParser()`: Extracts string content from `AIMessage`
2. `_strip`: Applies `str.strip()` to remove leading/trailing whitespace

**Purpose:** Returns clean SQL query string ready for execution.

---

## Complete Example

### Basic Usage

```python
# pip install -U langchain langchain-community langchain-openai
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

# Initialize database connection
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Initialize language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create SQL query chain
chain = create_sql_query_chain(llm, db)

# Generate SQL query
response = chain.invoke({"question": "How many employees are there?"})
print(response)
# Output: SELECT COUNT(*) FROM employees;
```

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:70-81`

---

### Restricted Table Access (Security Recommended)

```python
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("postgresql://user:pass@localhost/db")
llm = ChatOpenAI(model="gpt-4", temperature=0)

chain = create_sql_query_chain(llm, db)

# Restrict to specific tables for security
input_data = {
    "question": "What are the top 5 customers by total purchases?",
    "table_names_to_use": ["customers", "orders", "order_items"]
}

sql_query = chain.invoke(input_data)
print(sql_query)
# Output: SELECT c.customer_id, c.name, SUM(oi.quantity * oi.price) as total
#         FROM customers c
#         JOIN orders o ON c.customer_id = o.customer_id
#         JOIN order_items oi ON o.order_id = oi.order_id
#         GROUP BY c.customer_id, c.name
#         ORDER BY total DESC
#         LIMIT 5;
```

---

### Custom Prompt with Domain-Specific Instructions

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

# Custom prompt with business logic
custom_template = '''You are a SQL expert for an e-commerce database.
Generate syntactically correct {dialect} queries.
ALWAYS use table aliases for readability.
NEVER query for PII fields (email, phone, address) unless explicitly requested.

Limit results to {top_k} rows.

Available tables:
{table_info}

Question: {input}
SQLQuery:'''

custom_prompt = PromptTemplate(
    input_variables=["input", "table_info", "top_k", "dialect"],
    template=custom_template
)

db = SQLDatabase.from_uri("postgresql://localhost/ecommerce")
llm = ChatOpenAI(model="gpt-4", temperature=0)

chain = create_sql_query_chain(llm, db, prompt=custom_prompt, k=10)

result = chain.invoke({"question": "Show me recent high-value orders"})
```

---

### PostgreSQL with Column Comments

```python
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

# PostgreSQL database with column-level comments
db = SQLDatabase.from_uri("postgresql://user:pass@localhost/analytics")

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Include column comments for better context
chain = create_sql_query_chain(llm, db, get_col_comments=True)

sql_query = chain.invoke({
    "question": "What metrics are tracked for user engagement?"
})
```

---

### Integration with Full QA Chain

```python
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Step 1: Create SQL query generation chain
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
sql_chain = create_sql_query_chain(llm, db)

# Step 2: Execute query and generate answer
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, SQL query, and SQL result, 
    provide a natural language answer.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Answer:"""
)

# Step 3: Compose full QA chain
full_chain = (
    RunnablePassthrough.assign(query=sql_chain)
    | RunnablePassthrough.assign(
        result=lambda x: db.run(x["query"])
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

# Execute full chain
answer = full_chain.invoke({
    "question": "How many employees are there?"
})
print(answer)
# Output: There are 8 employees in the database.
```

---

## Prompt Requirements and Validation

### Required Input Variables

All custom prompts MUST include these input variables:

- **`input`** (str): User question with `"\nSQLQuery: "` suffix
- **`top_k`** (str): Maximum number of results (passed as string)
- **`table_info`** (str): Database schema and sample data

**Optional Variable:**
- **`dialect`** (str): Database dialect (auto-populated from `db.dialect`)

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:123-131`

### Validation Logic

```python
if {"input", "top_k", "table_info"}.difference(
    prompt_to_use.input_variables + list(prompt_to_use.partial_variables),
):
    raise ValueError(
        f"Prompt must have input variables: 'input', 'top_k', "
        f"'table_info'. Received prompt with input variables: "
        f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
    )
```

**Error Example:**
```python
# Invalid prompt missing required variables
bad_prompt = PromptTemplate.from_template("Generate SQL for: {question}")
chain = create_sql_query_chain(llm, db, prompt=bad_prompt)
# Raises: ValueError: Prompt must have input variables: 'input', 'top_k', 'table_info'...
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. ValueError: Prompt Missing Required Variables

**Symptom:**
```
ValueError: Prompt must have input variables: 'input', 'top_k', 'table_info'
```

**Cause:** Custom prompt does not include all required input variables.

**Solution:** Ensure prompt template includes `input`, `top_k`, and `table_info`:
```python
template = "Schema: {table_info}\nLimit: {top_k}\nQuestion: {input}"
prompt = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=template
)
```

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:123-131`

---

#### 2. ValueError: get_col_comments Unsupported Dialect

**Symptom:**
```
ValueError: get_col_comments=True is only supported for dialects 
'postgresql', 'mysql', and 'oracle'. Received dialect: sqlite
```

**Cause:** `get_col_comments=True` used with unsupported database dialect.

**Solution:** Only use `get_col_comments=True` with PostgreSQL, MySQL, or Oracle:
```python
# Check dialect before enabling
if db.dialect in ("postgresql", "mysql", "oracle"):
    chain = create_sql_query_chain(llm, db, get_col_comments=True)
else:
    chain = create_sql_query_chain(llm, db)
```

**Source:** `libs/langchain/langchain_classic/chains/sql_database/query.py:136-144`

---

#### 3. LLM Generates Invalid SQL

**Symptom:** Generated SQL query fails when executed against database.

**Cause:** LLM hallucinated column/table names or generated syntactically incorrect SQL.

**Solutions:**
- **Lower Temperature:** Set `temperature=0` for deterministic output
- **Better Prompts:** Use dialect-specific prompts from `SQL_PROMPTS`
- **Schema Clarity:** Ensure `db.get_table_info()` returns comprehensive schema
- **Few-Shot Examples:** Add example queries to custom prompts
- **Post-Validation:** Parse and validate SQL before execution

```python
# Add validation wrapper
def validate_and_execute(query: str, db: SQLDatabase) -> str:
    try:
        # Basic SQL parsing validation
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries allowed")
        return db.run(query)
    except Exception as e:
        return f"Query execution failed: {e}"
```

---

#### 4. Context Window Exceeded with Large Schemas

**Symptom:** LLM request fails due to token limit exceeded.

**Cause:** Database schema from `db.get_table_info()` too large for LLM context window.

**Solutions:**
- **Restrict Tables:** Use `SQLInputWithTables` to limit schema to relevant tables
- **Sample Fewer Rows:** Configure `SQLDatabase` with `sample_rows_in_table_info=1`
- **Exclude Tables:** Initialize `SQLDatabase` with `include_tables` or `ignore_tables`

```python
# Limit schema size
db = SQLDatabase.from_uri(
    "postgresql://localhost/large_db",
    include_tables=["customers", "orders"],  # Only include necessary tables
    sample_rows_in_table_info=1  # Reduce sample data
)

# Or use table restrictions per query
chain.invoke({
    "question": "Customer analysis query",
    "table_names_to_use": ["customers"]  # Limit to specific tables
})
```

---

#### 5. SQL Injection Concerns

**Symptom:** Generated queries could potentially be malicious or access unauthorized data.

**Cause:** LLM is a generative model and may produce unexpected SQL.

**Solutions:**
- ✅ **Read-Only Credentials:** Database user should have SELECT-only permissions
- ✅ **Table Restrictions:** Always use `table_names_to_use` in production
- ✅ **Query Allowlisting:** Validate queries against allowed patterns
- ✅ **Audit Logging:** Log all generated queries with user context
- ✅ **Human-in-the-Loop:** Require approval for query execution in sensitive environments

```python
# Production security pattern
def secure_sql_chain_invoke(question: str, allowed_tables: list[str]):
    """Secure wrapper with table restrictions and logging."""
    import logging
    logger = logging.getLogger(__name__)
    
    input_data = {
        "question": question,
        "table_names_to_use": allowed_tables
    }
    
    sql_query = chain.invoke(input_data)
    
    # Log for audit
    logger.info(f"Generated SQL: {sql_query} for question: {question}")
    
    # Validate query structure
    if not sql_query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries permitted")
    
    return sql_query
```

---

## Dialect-Specific Considerations

### Supported SQL Dialects

| Dialect | Prompt Key | Special Syntax | Current Date Function |
|---------|-----------|----------------|----------------------|
| **CrateDB** | `crate` | Double quotes for identifiers | `CURRENT_DATE` |
| **DuckDB** | `duckdb` | Double quotes for identifiers | `today()` |
| **GoogleSQL** | `googlesql` | Backticks for identifiers | `CURRENT_DATE()` |
| **MS SQL** | `mssql` | Square brackets for identifiers | `CAST(GETDATE() as date)` |
| **MySQL** | `mysql` | Backticks for identifiers | `CURDATE()` |
| **MariaDB** | `mariadb` | Backticks for identifiers | `CURDATE()` |
| **Oracle** | `oracle` | Double quotes for identifiers | `TRUNC(SYSDATE)` |
| **PostgreSQL** | `postgresql` | Double quotes for identifiers | `CURRENT_DATE` |
| **SQLite** | `sqlite` | Double quotes for identifiers | `date('now')` |
| **ClickHouse** | `clickhouse` | Double quotes for identifiers | `today()` |
| **PrestoDB** | `prestodb` | Double quotes for identifiers | `current_date` |

**Source:** `libs/langchain/langchain_classic/chains/sql_database/prompt.py:270-282`

---

## Related Documentation

- **[LCEL Composition Guide](../../guides/lcel-composition.md)**: Learn LCEL pipe operators and composition patterns
- **[Chain Types Guide](../../guides/chain-types.md)**: Compare different chain implementations
- **[Production Deployment Guide](../../guides/production-deployment.md)**: Security and operational best practices
- **[SQLDatabase API](../../utilities/sql-database.md)**: Database wrapper documentation
- **[Runnable Protocol](../../runnables/base.md)**: Understanding LCEL Runnable interface

---

## See Also

### Related Functions
- `SQLDatabase.from_uri()`: Initialize database connection
- `db.get_table_info()`: Retrieve schema information
- `db.run()`: Execute SQL queries

### Migration Notes
- **Deprecated:** `SQLDatabaseChain` (use `create_sql_query_chain` instead)
- **Migration Path:** Replace `SQLDatabaseChain.from_llm()` with `create_sql_query_chain()` + query execution chain

---

## Source Code References

- **Primary Implementation:** `libs/langchain/langchain_classic/chains/sql_database/query.py:33-166`
- **Input Schemas:** `libs/langchain/langchain_classic/chains/sql_database/query.py:20-30`
- **Prompt Templates:** `libs/langchain/langchain_classic/chains/sql_database/prompt.py:1-283`
- **Default Prompt:** `libs/langchain/langchain_classic/chains/sql_database/prompt.py:24-27`
- **Dialect Prompts:** `libs/langchain/langchain_classic/chains/sql_database/prompt.py:270-282`

---

*Last Updated: 2024*  
*LangChain Version: 1.0.0+*
