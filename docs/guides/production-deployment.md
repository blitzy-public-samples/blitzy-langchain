# Production Deployment Guide

This comprehensive guide provides best practices, patterns, and checklists for deploying LangChain applications to production environments with enterprise-grade reliability, observability, and performance.

## Table of Contents

- [Introduction](#introduction)
- [Logging Configuration](#logging-configuration)
- [Monitoring Integration](#monitoring-integration)
- [Rate Limiting Strategies](#rate-limiting-strategies)
- [Error Handling Resilience](#error-handling-resilience)
- [Performance Optimization](#performance-optimization)
- [Security Considerations](#security-considerations)
- [Cost Control](#cost-control)
- [Observability Best Practices](#observability-best-practices)
- [Scalability Patterns](#scalability-patterns)
- [Testing Strategies](#testing-strategies)
- [Deployment Checklist](#deployment-checklist)
- [Example Production Setup](#example-production-setup)
- [Troubleshooting Production Issues](#troubleshooting-production-issues)

---

## Introduction

Deploying LangChain applications to production requires careful consideration of reliability, observability, performance, and cost control. Unlike development environments where failures can be tolerated, production systems must handle:

- **Reliability**: External API failures, network issues, and rate limiting
- **Observability**: Comprehensive logging, monitoring, and tracing for debugging
- **Performance**: Low latency, high throughput, and efficient resource usage
- **Cost Control**: Token usage monitoring and budget enforcement

This guide synthesizes best practices from [Error Handling](error-handling.md), [Async Usage](async-usage.md), and [Callbacks](callbacks.md) guides, providing a complete production deployment strategy.

**Audience**: This guide is designed for developers deploying LangChain applications to production and DevOps engineers managing LangChain services.

**Prerequisites**:
- Understanding of [LCEL composition patterns](lcel-composition.md)
- Familiarity with [callback handlers](callbacks.md)
- Knowledge of [async/await patterns](async-usage.md) for performance

---

## Logging Configuration

Comprehensive logging is essential for debugging production issues, monitoring system health, and auditing chain execution.

### Log Levels

Use appropriate log levels based on your environment:

| Log Level | Development Usage | Production Usage | Example |
|-----------|------------------|------------------|---------|
| **DEBUG** | Verbose chain execution details, intermediate steps | Not recommended (performance impact) | "Chain input variables: {vars}" |
| **INFO** | Key milestones in execution | Standard production logging | "Chain completed in 2.3s" |
| **WARNING** | Recoverable issues, fallbacks triggered | Track degradation | "Falling back to secondary model" |
| **ERROR** | Failures requiring attention | Alert-worthy events | "API call failed after 3 retries" |

### Structured Logging

Use structured JSON logging for machine-parseable logs in production:

```python
import json
import logging
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Include extra context fields
        if hasattr(record, "chain_id"):
            log_data["chain_id"] = record.chain_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        return json.dumps(log_data)


# Configure structured logging for production
def configure_production_logging():
    """Set up structured JSON logging for production."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    
    # Set LangChain-specific loggers
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("langchain_core").setLevel(logging.INFO)
```

### Chain Execution Logging

Use callbacks to log chain execution at key lifecycle points. Source: `libs/core/langchain_core/callbacks/base.py`

```python
import time
from typing import Any, Dict, Optional
from uuid import UUID
from langchain_core.callbacks.base import BaseCallbackHandler


class LoggingCallback(BaseCallbackHandler):
    """Callback handler for logging chain execution.
    
    Logs chain start, end, error events with execution time and I/O data.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize logging callback.
        
        Args:
            logger: Logger instance. If None, uses root logger.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.start_times: Dict[UUID, float] = {}
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log when chain starts execution."""
        self.start_times[run_id] = time.time()
        self.logger.info(
            "Chain started",
            extra={
                "run_id": str(run_id),
                "chain_type": serialized.get("name", "unknown"),
                "input_keys": list(inputs.keys()),
                "tags": tags or [],
            }
        )
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Log when chain completes successfully."""
        duration = time.time() - self.start_times.pop(run_id, time.time())
        self.logger.info(
            "Chain completed",
            extra={
                "run_id": str(run_id),
                "duration_seconds": round(duration, 3),
                "output_keys": list(outputs.keys()),
            }
        )
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Log when chain encounters an error."""
        duration = time.time() - self.start_times.pop(run_id, time.time())
        self.logger.error(
            "Chain failed",
            extra={
                "run_id": str(run_id),
                "duration_seconds": round(duration, 3),
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            exc_info=error,
        )


# Usage with a chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logging_callback = LoggingCallback()

chain = ChatPromptTemplate.from_template("Answer: {question}") | ChatOpenAI()

result = chain.invoke(
    {"question": "What is LangChain?"},
    config={"callbacks": [logging_callback]}
)
```

### LLM Call Logging

Track token usage and costs with LLM-specific callbacks:

```python
class LLMLoggingCallback(BaseCallbackHandler):
    """Callback for logging LLM calls with token usage and cost estimation."""
    
    # Cost per 1K tokens (example prices, update with current rates)
    COST_PER_1K_TOKENS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.start_times: Dict[UUID, float] = {}
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Log when LLM call starts."""
        self.start_times[run_id] = time.time()
        self.logger.info(
            "LLM call started",
            extra={
                "run_id": str(run_id),
                "model": serialized.get("name", "unknown"),
                "prompt_count": len(prompts),
            }
        )
    
    def on_llm_end(
        self,
        response: Any,  # LLMResult type
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Log when LLM call completes with token usage."""
        duration = time.time() - self.start_times.pop(run_id, time.time())
        
        # Extract token usage from response
        token_usage = {}
        estimated_cost = 0.0
        model_name = "unknown"
        
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            model_name = response.llm_output.get("model_name", "unknown")
            
            # Estimate cost
            if model_name in self.COST_PER_1K_TOKENS:
                costs = self.COST_PER_1K_TOKENS[model_name]
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)
                estimated_cost = (
                    (input_tokens / 1000 * costs["input"]) +
                    (output_tokens / 1000 * costs["output"])
                )
        
        self.logger.info(
            "LLM call completed",
            extra={
                "run_id": str(run_id),
                "duration_seconds": round(duration, 3),
                "model": model_name,
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
                "estimated_cost_usd": round(estimated_cost, 4),
            }
        )
```

### Production Logging Setup

Complete production logging configuration with rotating file handlers:

```python
import logging
from logging.handlers import RotatingFileHandler
import os


def setup_production_logging(
    log_dir: str = "/var/log/langchain",
    max_bytes: int = 100_000_000,  # 100MB
    backup_count: int = 10,
):
    """Configure production logging with rotation.
    
    Args:
        log_dir: Directory for log files
        max_bytes: Maximum size per log file before rotation
        backup_count: Number of backup files to keep
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    json_formatter = JSONFormatter()
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "langchain.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(json_formatter)
    
    # Console handler for immediate visibility
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set library-specific levels
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("langchain_core").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
```

**Best Practices**:
- Use structured JSON logging for production
- Include context fields (chain_id, user_id, request_id) for correlation
- Rotate logs to prevent disk space exhaustion
- Set appropriate log levels to balance detail and performance
- Use callbacks for chain-specific logging

---

## Monitoring Integration

Production monitoring enables proactive issue detection and performance optimization.

### Metrics to Collect

Track key metrics across three categories:

#### Chain Execution Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Latency** | Chain execution time (p50, p95, p99) | p95 < 5s |
| **Throughput** | Requests per second | Based on capacity |
| **Error Rate** | Percentage of failed requests | < 1% |
| **Success Rate** | Percentage of successful completions | > 99% |

#### LLM Metrics

| Metric | Description | Monitoring |
|--------|-------------|------------|
| **Token Usage** | Input/output tokens per request | Track against quotas |
| **Cost Per Request** | Estimated API cost | Budget adherence |
| **Model Latency** | Time for LLM API calls | p95 < 3s |
| **Rate Limit Hits** | Frequency of 429 errors | Should be 0 |

#### Business Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Task Completion** | Successful task completion rate | User satisfaction |
| **Feature Usage** | Which chains/features are used | Product insights |
| **User Retention** | Repeat usage patterns | Product health |

### Metrics Collection with Callbacks

Implement custom callback for metrics collection:

```python
from typing import Dict, Optional
from collections import defaultdict
import time


class MetricsCallback(BaseCallbackHandler):
    """Callback handler for collecting execution metrics.
    
    Tracks execution time, token usage, and error counts.
    """
    
    def __init__(self):
        self.metrics: Dict[str, list] = defaultdict(list)
        self.start_times: Dict[UUID, float] = {}
        self.error_counts: Dict[str, int] = defaultdict(int)
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record chain start time."""
        self.start_times[run_id] = time.time()
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record successful chain execution."""
        if run_id in self.start_times:
            duration = time.time() - self.start_times.pop(run_id)
            self.metrics["chain_duration_seconds"].append(duration)
            self.metrics["chain_success_count"].append(1)
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record chain error."""
        error_type = type(error).__name__
        self.error_counts[error_type] += 1
        self.metrics["chain_error_count"].append(1)
        self.start_times.pop(run_id, None)
    
    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record token usage from LLM response."""
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            self.metrics["total_tokens"].append(
                token_usage.get("total_tokens", 0)
            )
            self.metrics["input_tokens"].append(
                token_usage.get("prompt_tokens", 0)
            )
            self.metrics["output_tokens"].append(
                token_usage.get("completion_tokens", 0)
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from collected metrics."""
        import statistics
        
        summary = {}
        
        # Calculate latency percentiles
        if self.metrics["chain_duration_seconds"]:
            durations = self.metrics["chain_duration_seconds"]
            summary["latency_p50"] = statistics.median(durations)
            summary["latency_p95"] = statistics.quantiles(durations, n=20)[18]
            summary["latency_p99"] = statistics.quantiles(durations, n=100)[98]
            summary["latency_avg"] = statistics.mean(durations)
        
        # Calculate success rate
        total = (
            sum(self.metrics.get("chain_success_count", [])) +
            sum(self.metrics.get("chain_error_count", []))
        )
        if total > 0:
            success = sum(self.metrics.get("chain_success_count", []))
            summary["success_rate"] = success / total
            summary["error_rate"] = 1 - summary["success_rate"]
        
        # Token usage
        if self.metrics["total_tokens"]:
            summary["total_tokens"] = sum(self.metrics["total_tokens"])
            summary["avg_tokens_per_request"] = statistics.mean(
                self.metrics["total_tokens"]
            )
        
        # Error breakdown
        summary["errors_by_type"] = dict(self.error_counts)
        
        return summary
```

### Integration with Prometheus

Export metrics to Prometheus for monitoring and alerting:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server


# Define Prometheus metrics
chain_duration = Histogram(
    "langchain_chain_duration_seconds",
    "Time spent in chain execution",
    ["chain_type"],
)
chain_requests_total = Counter(
    "langchain_chain_requests_total",
    "Total number of chain requests",
    ["chain_type", "status"],
)
token_usage_total = Counter(
    "langchain_token_usage_total",
    "Total tokens used",
    ["model", "token_type"],
)
active_chains = Gauge(
    "langchain_active_chains",
    "Number of currently executing chains",
)


class PrometheusMetricsCallback(BaseCallbackHandler):
    """Callback for recording metrics to Prometheus."""
    
    def __init__(self):
        self.start_times: Dict[UUID, float] = {}
        self.chain_types: Dict[UUID, str] = {}
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record chain start."""
        self.start_times[run_id] = time.time()
        self.chain_types[run_id] = serialized.get("name", "unknown")
        active_chains.inc()
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record successful chain completion."""
        if run_id in self.start_times:
            duration = time.time() - self.start_times.pop(run_id)
            chain_type = self.chain_types.pop(run_id, "unknown")
            
            chain_duration.labels(chain_type=chain_type).observe(duration)
            chain_requests_total.labels(
                chain_type=chain_type,
                status="success"
            ).inc()
            active_chains.dec()
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record chain error."""
        chain_type = self.chain_types.pop(run_id, "unknown")
        chain_requests_total.labels(
            chain_type=chain_type,
            status="error"
        ).inc()
        active_chains.dec()
        self.start_times.pop(run_id, None)
    
    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record token usage."""
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            model = response.llm_output.get("model_name", "unknown")
            
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)
            
            token_usage_total.labels(
                model=model,
                token_type="input"
            ).inc(input_tokens)
            token_usage_total.labels(
                model=model,
                token_type="output"
            ).inc(output_tokens)


# Start Prometheus metrics server
def start_metrics_server(port: int = 8000):
    """Start HTTP server for Prometheus metrics scraping."""
    start_http_server(port)
    logging.info(f"Metrics server started on port {port}")
```

### Integration with Datadog

Send custom metrics to Datadog:

```python
from datadog import initialize, statsd


# Initialize Datadog
initialize(
    api_key=os.getenv("DATADOG_API_KEY"),
    app_key=os.getenv("DATADOG_APP_KEY"),
)


class DatadogMetricsCallback(BaseCallbackHandler):
    """Callback for sending metrics to Datadog."""
    
    def __init__(self):
        self.start_times: Dict[UUID, float] = {}
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Send chain execution time to Datadog."""
        if run_id in self.start_times:
            duration = time.time() - self.start_times.pop(run_id)
            statsd.histogram("langchain.chain.duration", duration)
            statsd.increment("langchain.chain.success")
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Send error metric to Datadog."""
        statsd.increment("langchain.chain.error")
        self.start_times.pop(run_id, None)
```

### Alerting Configuration

Set up alerts for critical metric thresholds:

**Recommended Alert Rules**:

1. **High Error Rate**
   - Condition: `error_rate > 5% for 5 minutes`
   - Action: Page on-call engineer
   - Severity: Critical

2. **High Latency**
   - Condition: `p95_latency > 10s for 5 minutes`
   - Action: Send alert to Slack
   - Severity: Warning

3. **Token Budget Exceeded**
   - Condition: `daily_token_usage > budget * 0.9`
   - Action: Send notification
   - Severity: Warning

4. **Rate Limit Hits**
   - Condition: `rate_limit_errors > 0`
   - Action: Send alert
   - Severity: Warning

**See Also**: [LangSmith](https://www.langchain.com/langsmith) provides built-in tracing and monitoring capabilities.

---

## Rate Limiting Strategies

Protect your application from API quota exhaustion and service degradation.

### API Quota Management

Track token usage against provider limits:

```python
from typing import Dict


class TokenBudgetManager:
    """Track and enforce token usage budgets."""
    
    def __init__(
        self,
        daily_budget: int,
        alert_threshold: float = 0.9,
    ):
        """Initialize budget manager.
        
        Args:
            daily_budget: Maximum tokens per day
            alert_threshold: Fraction of budget that triggers alert (0-1)
        """
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold
        self.usage: Dict[str, int] = defaultdict(int)
        self.logger = logging.getLogger(__name__)
    
    def record_usage(self, tokens: int, date: str = None) -> bool:
        """Record token usage and check against budget.
        
        Args:
            tokens: Number of tokens used
            date: Date key (default: today)
            
        Returns:
            True if within budget, False if budget exceeded
        """
        if date is None:
            from datetime import datetime
            date = datetime.utcnow().strftime("%Y-%m-%d")
        
        self.usage[date] += tokens
        current_usage = self.usage[date]
        
        # Check for alert threshold
        if current_usage >= self.daily_budget * self.alert_threshold:
            self.logger.warning(
                f"Token usage at {current_usage}/{self.daily_budget} "
                f"({current_usage/self.daily_budget:.1%}) for {date}"
            )
        
        # Check for budget exceeded
        if current_usage >= self.daily_budget:
            self.logger.error(
                f"Daily token budget exceeded: {current_usage}/{self.daily_budget}"
            )
            return False
        
        return True
```

### Request Throttling

Limit concurrent requests to prevent overwhelming APIs:

```python
import asyncio
from asyncio import Semaphore


class RateLimiter:
    """Rate limiter using semaphore for concurrent request limiting."""
    
    def __init__(self, max_concurrent: int = 10):
        """Initialize rate limiter.
        
        Args:
            max_concurrent: Maximum concurrent requests allowed
        """
        self.semaphore = Semaphore(max_concurrent)
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """Acquire semaphore."""
        await self.semaphore.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release semaphore."""
        self.semaphore.release()


# Usage with async chains
async def process_with_rate_limit(
    chain,
    inputs: list[Dict[str, Any]],
    max_concurrent: int = 10,
):
    """Process inputs with rate limiting.
    
    Args:
        chain: The chain to execute
        inputs: List of input dictionaries
        max_concurrent: Maximum concurrent executions
        
    Returns:
        List of results
    """
    rate_limiter = RateLimiter(max_concurrent)
    
    async def process_one(input_data):
        async with rate_limiter:
            return await chain.ainvoke(input_data)
    
    tasks = [process_one(input_data) for input_data in inputs]
    return await asyncio.gather(*tasks)
```

### Circuit Breaker Pattern

Prevent cascading failures by failing fast when services are degraded:

```python
from enum import Enum
from datetime import datetime, timedelta


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        expected_exception: type = Exception,
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before testing recovery
            expected_exception: Exception type to track
        """
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_seconds)
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.logger = logging.getLogger(__name__)
    
    def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            # Check if timeout expired
            if datetime.now() - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN, failing fast")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.logger.info("Circuit breaker CLOSED after successful test")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.error(
                f"Circuit breaker OPEN after {self.failure_count} failures"
            )


# Example usage with LLM provider
from langchain_openai import ChatOpenAI

openai_circuit = CircuitBreaker(failure_threshold=3, timeout_seconds=30)


def call_llm_with_circuit_breaker(chain, input_data):
    """Call chain with circuit breaker protection."""
    return openai_circuit.call(chain.invoke, input_data)
```

### Provider-Specific Rate Limits

Be aware of provider-specific rate limits:

| Provider | Model | Rate Limit | Tokens/Min | Recommended Strategy |
|----------|-------|------------|------------|---------------------|
| OpenAI | GPT-4 | 10,000 RPM | 300,000 | Use circuit breaker, respect Retry-After |
| OpenAI | GPT-3.5-turbo | 60,000 RPM | 2,000,000 | Higher concurrency allowed |
| Anthropic | Claude 3 Opus | 5,000 RPM | 400,000 | Aggressive rate limiting |
| Anthropic | Claude 3 Sonnet | 10,000 RPM | 800,000 | Moderate rate limiting |

**Best Practices**:
- Respect `Retry-After` headers in 429 responses
- Implement exponential backoff for rate limit errors
- Use circuit breakers to protect against sustained failures
- Monitor rate limit hits as a key metric

---

## Error Handling Resilience

Building resilient production systems requires comprehensive error handling strategies. See [Error Handling Guide](error-handling.md) for detailed patterns.

### Retry Logic with Exponential Backoff

Use `with_retry()` for transient failures. Source: `libs/core/langchain_core/runnables/base.py`

```python
from langchain_core.runnables import Runnable


# Add retry to specific components
model_with_retry = ChatOpenAI().with_retry(
    retry_exception_types=(ConnectionError, TimeoutError),
    max_attempt_number=5,
    wait_exponential_jitter=True,
)

# Use in chain
chain = prompt | model_with_retry | parser
```

### Fallback Chains

Provide alternative execution paths for graceful degradation:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableLambda


# Primary model with fallbacks
primary_model = ChatOpenAI(model="gpt-4")
fallback_model = ChatAnthropic(model="claude-3-sonnet")

def hardcoded_fallback(input_data):
    """Last resort fallback with hardcoded response."""
    return "I'm currently experiencing technical difficulties. Please try again later."

model_with_fallbacks = primary_model.with_fallbacks(
    [fallback_model, RunnableLambda(hardcoded_fallback)],
    exceptions_to_handle=(Exception,),
)

chain = prompt | model_with_fallbacks | parser
```

### Exception Categorization

Handle different error types appropriately:

| Error Type | Category | Strategy | Example |
|------------|----------|----------|---------|
| ConnectionError | Transient | Retry with backoff | Network issues |
| TimeoutError | Transient | Retry with longer timeout | Slow API response |
| RateLimitError (429) | Transient | Retry after delay | API quota |
| AuthenticationError (401) | Permanent | Raise immediately, alert | Invalid API key |
| ValidationError | Permanent | Return user-friendly message | Invalid input |
| InternalError (500) | Transient | Retry limited times | Provider issue |

### Timeout Configuration

Set appropriate timeouts for all external calls:

```python
import asyncio


# LLM API calls: 30-60 seconds
async def call_llm_with_timeout(chain, input_data, timeout: int = 60):
    """Execute chain with timeout."""
    try:
        return await asyncio.wait_for(
            chain.ainvoke(input_data),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logging.error(f"Chain execution timeout after {timeout}s")
        raise


# Database queries: 5-10 seconds
async def query_db_with_timeout(query, timeout: int = 10):
    """Execute database query with timeout."""
    return await asyncio.wait_for(query, timeout=timeout)


# HTTP requests: 10-30 seconds
import aiohttp

async def fetch_with_timeout(url: str, timeout: int = 30):
    """Fetch URL with timeout."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            return await response.text()
```

### Error Aggregation and Alerting

Track error patterns and alert on sustained issues:

```python
from collections import Counter
from datetime import datetime, timedelta


class ErrorAggregator:
    """Aggregate errors and trigger alerts."""
    
    def __init__(
        self,
        alert_threshold: int = 10,
        time_window_minutes: int = 5,
    ):
        """Initialize error aggregator.
        
        Args:
            alert_threshold: Number of errors to trigger alert
            time_window_minutes: Time window for counting errors
        """
        self.alert_threshold = alert_threshold
        self.time_window = timedelta(minutes=time_window_minutes)
        self.errors: list[tuple[datetime, str]] = []
        self.logger = logging.getLogger(__name__)
    
    def record_error(self, error_type: str):
        """Record an error occurrence.
        
        Args:
            error_type: Type of error that occurred
        """
        now = datetime.now()
        self.errors.append((now, error_type))
        
        # Remove old errors outside time window
        cutoff = now - self.time_window
        self.errors = [(ts, err) for ts, err in self.errors if ts > cutoff]
        
        # Check for alert condition
        error_counts = Counter(err for _, err in self.errors)
        for err_type, count in error_counts.items():
            if count >= self.alert_threshold:
                self._trigger_alert(err_type, count)
    
    def _trigger_alert(self, error_type: str, count: int):
        """Trigger alert for sustained error pattern."""
        self.logger.error(
            f"ALERT: {count} {error_type} errors in last "
            f"{self.time_window.total_seconds()/60:.0f} minutes"
        )
        # Send to alerting system (PagerDuty, Slack, etc.)
```

**See Also**: [Error Handling Guide](error-handling.md) for comprehensive error recovery patterns.

---

## Performance Optimization

Optimize chain performance for production workloads.

### Caching Strategies

Cache LLM responses to reduce API calls and costs:

```python
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache, RedisCache


# In-memory cache (single process)
set_llm_cache(InMemoryCache())


# Redis cache (distributed, production-ready)
from redis import Redis

redis_client = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
)
set_llm_cache(RedisCache(redis_client))
```

Implement custom semantic caching:

```python
from langchain_core.embeddings import Embeddings
from langchain_core.caches import BaseCache


class SemanticCache(BaseCache):
    """Cache based on semantic similarity of inputs."""
    
    def __init__(
        self,
        embeddings: Embeddings,
        similarity_threshold: float = 0.95,
    ):
        """Initialize semantic cache.
        
        Args:
            embeddings: Embedding model for similarity computation
            similarity_threshold: Minimum similarity for cache hit (0-1)
        """
        self.embeddings = embeddings
        self.threshold = similarity_threshold
        self.cache: list[tuple[list[float], str]] = []
    
    def lookup(self, prompt: str, llm_string: str) -> Optional[str]:
        """Look up cached response for similar prompt."""
        if not self.cache:
            return None
        
        # Compute embedding for prompt
        prompt_embedding = self.embeddings.embed_query(prompt)
        
        # Find most similar cached prompt
        from numpy import dot
        from numpy.linalg import norm
        
        best_similarity = 0
        best_response = None
        
        for cached_embedding, cached_response in self.cache:
            similarity = dot(prompt_embedding, cached_embedding) / (
                norm(prompt_embedding) * norm(cached_embedding)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_response = cached_response
        
        # Return if above threshold
        if best_similarity >= self.threshold:
            return best_response
        
        return None
    
    def update(self, prompt: str, llm_string: str, return_val: str):
        """Store prompt and response in cache."""
        prompt_embedding = self.embeddings.embed_query(prompt)
        self.cache.append((prompt_embedding, return_val))
```

### Batch Processing

Process multiple inputs concurrently with `batch()`:

```python
# Sequential processing (slow)
results = [chain.invoke(input) for input in inputs]  # O(n) time


# Batch processing (fast)
results = chain.batch(inputs)  # Concurrent execution


# Optimal batch size
OPTIMAL_BATCH_SIZE = 10

def batch_process(chain, inputs: list, batch_size: int = OPTIMAL_BATCH_SIZE):
    """Process inputs in batches.
    
    Args:
        chain: Chain to execute
        inputs: List of inputs to process
        batch_size: Size of each batch
        
    Returns:
        List of results
    """
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        batch_results = chain.batch(batch)
        results.extend(batch_results)
    return results
```

### Async Operations

Use async for I/O-bound operations. See [Async Usage Guide](async-usage.md).

```python
import asyncio


# Concurrent API calls
async def process_documents_concurrently(chain, documents: list):
    """Process documents concurrently with async."""
    tasks = [chain.ainvoke({"doc": doc}) for doc in documents]
    return await asyncio.gather(*tasks)


# FastAPI async endpoint
from fastapi import FastAPI

app = FastAPI()

@app.post("/process")
async def process_request(data: dict):
    """Async endpoint for chain execution."""
    result = await chain.ainvoke(data)
    return {"result": result}
```

### Connection Pooling

Reuse connections for better performance:

```python
# HTTP client with connection pooling
import aiohttp


class HTTPClientManager:
    """Manage HTTP client with connection pooling."""
    
    def __init__(self, max_connections: int = 100):
        """Initialize HTTP client manager."""
        connector = aiohttp.TCPConnector(limit=max_connections)
        self.session = aiohttp.ClientSession(connector=connector)
    
    async def __aenter__(self):
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()


# Database connection pooling
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@localhost/db",
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
)
```

### Performance Profiling

Profile chain execution to identify bottlenecks:

```python
import cProfile
import pstats
from io import StringIO


def profile_chain(chain, input_data):
    """Profile chain execution."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = chain.invoke(input_data)
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())
    
    return result
```

**Best Practices**:
- Cache LLM responses for repeated queries
- Use `batch()` for processing multiple inputs
- Use async operations for I/O-bound tasks
- Profile production workloads to identify bottlenecks
- Implement connection pooling for external services

---

## Security Considerations

Production deployments must implement comprehensive security measures.

### API Key Management

Never hardcode API keys - use environment variables and secrets management:

```python
import os
from typing import Optional


def get_api_key(key_name: str, required: bool = True) -> Optional[str]:
    """Securely retrieve API key from environment.
    
    Args:
        key_name: Name of environment variable
        required: Whether key is required
        
    Returns:
        API key value or None
        
    Raises:
        ValueError: If required key is missing
    """
    key = os.getenv(key_name)
    
    if required and not key:
        raise ValueError(f"Required API key {key_name} not found in environment")
    
    return key


# Example usage
openai_key = get_api_key("OPENAI_API_KEY")
anthropic_key = get_api_key("ANTHROPIC_API_KEY")
```

Use secrets management services in production:

```python
# AWS Secrets Manager
import boto3
import json


def get_secret_from_aws(secret_name: str) -> dict:
    """Retrieve secret from AWS Secrets Manager.
    
    Args:
        secret_name: Name of secret in AWS
        
    Returns:
        Secret value as dictionary
    """
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])


# HashiCorp Vault
import hvac


def get_secret_from_vault(path: str) -> dict:
    """Retrieve secret from HashiCorp Vault.
    
    Args:
        path: Secret path in Vault
        
    Returns:
        Secret data
    """
    client = hvac.Client(url=os.getenv("VAULT_ADDR"))
    client.auth.approle.login(
        role_id=os.getenv("VAULT_ROLE_ID"),
        secret_id=os.getenv("VAULT_SECRET_ID"),
    )
    return client.secrets.kv.v2.read_secret_version(path=path)['data']['data']
```

### Input Validation and Sanitization

Validate all user inputs with Pydantic:

```python
from pydantic import BaseModel, Field, validator


class UserQuery(BaseModel):
    """Validated user query input."""
    
    question: str = Field(..., min_length=1, max_length=1000)
    user_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+$')
    
    @validator('question')
    def sanitize_question(cls, v):
        """Sanitize question input."""
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '{', '}', '|', '\\', '^', '`']
        for char in dangerous_chars:
            v = v.replace(char, '')
        return v.strip()


# Usage
def process_user_query(raw_input: dict):
    """Process user query with validation."""
    try:
        validated = UserQuery(**raw_input)
        return chain.invoke({"question": validated.question})
    except ValidationError as e:
        logging.warning(f"Invalid input: {e}")
        raise ValueError("Invalid input format")
```

Protect against SQL injection:

```python
from sqlalchemy import text


# BAD: SQL injection vulnerable
def bad_query(user_input: str):
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return engine.execute(query)


# GOOD: Parameterized query
def safe_query(user_input: str):
    query = text("SELECT * FROM users WHERE name = :name")
    return engine.execute(query, {"name": user_input})
```

### Output Filtering

Filter sensitive information from LLM outputs:

```python
import re
from typing import Any


class OutputFilter(BaseCallbackHandler):
    """Callback for filtering sensitive information from outputs."""
    
    # Patterns for PII detection
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Filter output for PII."""
        if hasattr(response, 'generations'):
            for generation_list in response.generations:
                for generation in generation_list:
                    text = generation.text
                    
                    # Check for PII
                    if self.SSN_PATTERN.search(text):
                        logging.warning("SSN detected in output")
                    if self.CREDIT_CARD_PATTERN.search(text):
                        logging.warning("Credit card detected in output")
                    if self.EMAIL_PATTERN.search(text):
                        logging.info("Email detected in output")
                    
                    # Optionally redact
                    # generation.text = self._redact_pii(text)
    
    def _redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        text = self.SSN_PATTERN.sub('[SSN REDACTED]', text)
        text = self.CREDIT_CARD_PATTERN.sub('[CARD REDACTED]', text)
        text = self.EMAIL_PATTERN.sub('[EMAIL REDACTED]', text)
        return text
```

### Authentication and Authorization

Implement API authentication:

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


app = FastAPI()
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API token.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User ID if valid
        
    Raises:
        HTTPException: If token is invalid
    """
    token = credentials.credentials
    
    # Verify token (implement your verification logic)
    user_id = validate_token(token)
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    
    return user_id


@app.post("/api/process")
async def process_with_auth(
    data: dict,
    user_id: str = Depends(verify_token)
):
    """Protected endpoint requiring authentication."""
    # Process with user context
    result = await chain.ainvoke(data, config={"metadata": {"user_id": user_id}})
    return {"result": result}
```

### Network Security

Enforce HTTPS and implement security headers:

```python
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware


app = FastAPI()

# Redirect HTTP to HTTPS in production
if os.getenv("ENVIRONMENT") == "production":
    app.add_middleware(HTTPSRedirectMiddleware)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
```

**Best Practices**:
- Store API keys in secrets management systems (AWS Secrets Manager, HashiCorp Vault)
- Rotate keys regularly (every 90 days minimum)
- Validate all user inputs with Pydantic models
- Sanitize inputs to prevent injection attacks
- Filter outputs for PII and sensitive information
- Implement authentication and authorization
- Enforce HTTPS in production
- Use API gateways for rate limiting and DDoS protection

---

## Cost Control

Monitor and control LangChain application costs.

### Token Budget Enforcement

Track usage per user/request and enforce limits:

```python
class TokenBudgetCallback(BaseCallbackHandler):
    """Callback for enforcing token budgets."""
    
    def __init__(
        self,
        budget_manager: TokenBudgetManager,
        request_budget: Optional[int] = None,
    ):
        """Initialize token budget callback.
        
        Args:
            budget_manager: Budget manager instance
            request_budget: Optional per-request token limit
        """
        self.budget_manager = budget_manager
        self.request_budget = request_budget
        self.request_tokens = 0
    
    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Track token usage and enforce budget."""
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            tokens = token_usage.get("total_tokens", 0)
            
            self.request_tokens += tokens
            
            # Check daily budget
            within_budget = self.budget_manager.record_usage(tokens)
            if not within_budget:
                raise Exception("Daily token budget exceeded")
            
            # Check per-request budget
            if self.request_budget and self.request_tokens > self.request_budget:
                raise Exception(
                    f"Request token budget exceeded: "
                    f"{self.request_tokens} > {self.request_budget}"
                )


# Usage
budget_manager = TokenBudgetManager(daily_budget=1_000_000)
budget_callback = TokenBudgetCallback(
    budget_manager=budget_manager,
    request_budget=10_000,
)

result = chain.invoke(input_data, config={"callbacks": [budget_callback]})
```

### Model Selection Strategies

Choose appropriate models based on task complexity:

```python
from enum import Enum


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"          # Classification, extraction
    MODERATE = "moderate"      # Summarization, simple reasoning
    COMPLEX = "complex"        # Multi-step reasoning, creative tasks


def select_model(task_complexity: TaskComplexity) -> str:
    """Select appropriate model based on task complexity.
    
    Args:
        task_complexity: Complexity level of the task
        
    Returns:
        Model name to use
    """
    if task_complexity == TaskComplexity.SIMPLE:
        # Use cheaper model for simple tasks
        return "gpt-3.5-turbo"
    elif task_complexity == TaskComplexity.MODERATE:
        # Use mid-tier model
        return "gpt-4o-mini"
    else:
        # Use most capable model for complex tasks
        return "gpt-4"


# Dynamic model selection
def create_chain_with_model_selection(task_complexity: TaskComplexity):
    """Create chain with appropriate model."""
    model_name = select_model(task_complexity)
    model = ChatOpenAI(model=model_name)
    return prompt | model | parser
```

### Response Caching

Reduce API calls by caching responses:

```python
class CachingChain:
    """Wrapper for chains with cache-aside pattern."""
    
    def __init__(self, chain, cache_ttl_seconds: int = 3600):
        """Initialize caching chain.
        
        Args:
            chain: Underlying chain to execute
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self.chain = chain
        self.cache_ttl = cache_ttl_seconds
        self.cache: Dict[str, tuple[Any, float]] = {}
    
    def invoke(self, input_data: Dict[str, Any]) -> Any:
        """Invoke chain with caching.
        
        Args:
            input_data: Chain input
            
        Returns:
            Chain output (from cache or execution)
        """
        import time
        import hashlib
        import json
        
        # Create cache key from input
        cache_key = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Check cache
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logging.info("Cache hit for request")
                return result
        
        # Cache miss - execute chain
        logging.info("Cache miss, executing chain")
        result = self.chain.invoke(input_data)
        
        # Store in cache
        self.cache[cache_key] = (result, time.time())
        
        return result
```

### Cost Monitoring

Track estimated costs in real-time:

```python
class CostTrackingCallback(BaseCallbackHandler):
    """Callback for tracking estimated API costs."""
    
    # Model costs per 1K tokens (update with current rates)
    COSTS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    }
    
    def __init__(self):
        self.total_cost = 0.0
        self.cost_by_model: Dict[str, float] = defaultdict(float)
    
    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Calculate and track cost."""
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            model = response.llm_output.get("model_name", "unknown")
            
            if model in self.COSTS:
                costs = self.COSTS[model]
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)
                
                cost = (
                    (input_tokens / 1000 * costs["input"]) +
                    (output_tokens / 1000 * costs["output"])
                )
                
                self.total_cost += cost
                self.cost_by_model[model] += cost
                
                logging.info(
                    f"Request cost: ${cost:.4f}, Total: ${self.total_cost:.4f}"
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "cost_by_model": {
                model: round(cost, 4)
                for model, cost in self.cost_by_model.items()
            }
        }
```

**Best Practices**:
- Track token usage against daily/monthly budgets
- Use GPT-3.5 or GPT-4o-mini for simple tasks
- Use GPT-4 only for complex reasoning that requires it
- Cache responses for repeated queries
- Implement semantic caching for similar queries
- Monitor costs in real-time with callbacks
- Set alerts for budget thresholds

---

## Observability Best Practices

Comprehensive observability enables effective debugging and monitoring.

### Distributed Tracing

Propagate trace IDs through chain execution:

```python
import uuid
from contextvars import ContextVar


# Context variable for trace ID
trace_id_var: ContextVar[str] = ContextVar('trace_id', default=None)


class TracingCallback(BaseCallbackHandler):
    """Callback for distributed tracing."""
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Set trace ID at chain start."""
        # Get or create trace ID
        trace_id = trace_id_var.get()
        if not trace_id:
            trace_id = str(uuid.uuid4())
            trace_id_var.set(trace_id)
        
        # Include trace ID in logs
        logging.info(
            "Chain started",
            extra={
                "trace_id": trace_id,
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
            }
        )


# OpenTelemetry integration
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter


def setup_opentelemetry():
    """Configure OpenTelemetry for distributed tracing."""
    provider = TracerProvider()
    processor = BatchSpanProcessor(
        JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


# Use tracer in chain execution
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("chain_execution"):
    result = chain.invoke(input_data)
```

### Debug Mode Configuration

Enable verbose logging for development:

```python
import langchain


# Development: verbose debugging
def configure_debug_mode():
    """Enable debug mode for development."""
    langchain.debug = True
    logging.getLogger("langchain").setLevel(logging.DEBUG)
    logging.getLogger("langchain_core").setLevel(logging.DEBUG)


# Production: structured logging with callbacks
def configure_production_mode():
    """Configure production observability."""
    langchain.debug = False  # Disable verbose output
    logging.getLogger("langchain").setLevel(logging.INFO)
    
    # Use callbacks for observability
    callbacks = [
        LoggingCallback(),
        MetricsCallback(),
        TracingCallback(),
    ]
    return callbacks
```

### Health Checks

Implement health checks for dependencies:

```python
from fastapi import FastAPI, status
from enum import Enum


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


async def check_llm_health() -> tuple[HealthStatus, str]:
    """Check LLM API availability.
    
    Returns:
        Tuple of (status, message)
    """
    try:
        # Simple test call
        model = ChatOpenAI(model="gpt-3.5-turbo")
        await model.ainvoke("test")
        return (HealthStatus.HEALTHY, "LLM API accessible")
    except Exception as e:
        return (HealthStatus.UNHEALTHY, f"LLM API error: {str(e)}")


async def check_database_health() -> tuple[HealthStatus, str]:
    """Check database connectivity."""
    try:
        # Test query
        engine.execute("SELECT 1")
        return (HealthStatus.HEALTHY, "Database accessible")
    except Exception as e:
        return (HealthStatus.UNHEALTHY, f"Database error: {str(e)}")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint.
    
    Returns:
        Health status of application and dependencies
    """
    checks = {
        "llm": await check_llm_health(),
        "database": await check_database_health(),
    }
    
    # Determine overall health
    overall = HealthStatus.HEALTHY
    if any(status == HealthStatus.UNHEALTHY for status, _ in checks.values()):
        overall = HealthStatus.UNHEALTHY
    elif any(status == HealthStatus.DEGRADED for status, _ in checks.values()):
        overall = HealthStatus.DEGRADED
    
    return {
        "status": overall.value,
        "checks": {
            name: {"status": status.value, "message": message}
            for name, (status, message) in checks.items()
        }
    }
```

**See Also**: [Callbacks Guide](callbacks.md) for custom callback implementation patterns.

---

## Scalability Patterns

Design chains for horizontal scaling and high throughput.

### Stateless Chain Design

Avoid in-memory state to enable horizontal scaling:

```python
# BAD: In-memory state doesn't scale
class StatefulChain:
    def __init__(self):
        self.conversation_history = []  # Won't work across instances
    
    def invoke(self, input_data):
        self.conversation_history.append(input_data)
        # Process with history
        ...


# GOOD: Stateless with external storage
class StatelessChain:
    def __init__(self, session_store):
        self.session_store = session_store  # Redis, database, etc.
    
    def invoke(self, input_data, session_id: str):
        # Load session from external storage
        history = self.session_store.get(session_id)
        
        # Process
        result = chain.invoke({"input": input_data, "history": history})
        
        # Save updated session
        self.session_store.set(session_id, updated_history)
        
        return result
```

### Load Balancing

Distribute requests across instances:

```python
# Kubernetes deployment example
"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-service
spec:
  replicas: 3  # Multiple instances
  selector:
    matchLabels:
      app: langchain-service
  template:
    metadata:
      labels:
        app: langchain-service
    spec:
      containers:
      - name: langchain
        image: langchain-service:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: langchain-service
spec:
  selector:
    app: langchain-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
"""
```

### Async and Concurrency

Use async operations for high concurrency. See [Async Usage Guide](async-usage.md).

```python
from fastapi import FastAPI
import asyncio


app = FastAPI()


@app.post("/api/process")
async def process_async(data: dict):
    """Async endpoint for high concurrency."""
    # Use semaphore for rate limiting
    async with rate_limiter:
        result = await chain.ainvoke(data)
    return {"result": result}


# Handle concurrent requests
@app.post("/api/batch")
async def process_batch(items: list[dict]):
    """Process multiple items concurrently."""
    # Limit concurrency
    semaphore = asyncio.Semaphore(10)
    
    async def process_one(item):
        async with semaphore:
            return await chain.ainvoke(item)
    
    tasks = [process_one(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {"results": results}
```

### Resource Limits

Set appropriate resource limits:

```python
# Kubernetes resource limits
"""
resources:
  limits:
    cpu: "2"          # Maximum CPU
    memory: "4Gi"     # Maximum memory
  requests:
    cpu: "1"          # Guaranteed CPU
    memory: "2Gi"     # Guaranteed memory
"""


# Application-level limits
MAX_REQUEST_SIZE = 100_000  # 100KB
MAX_TIMEOUT = 60  # 60 seconds


async def process_with_limits(data: dict):
    """Process with resource limits."""
    # Check request size
    import sys
    if sys.getsizeof(data) > MAX_REQUEST_SIZE:
        raise ValueError("Request too large")
    
    # Apply timeout
    try:
        result = await asyncio.wait_for(
            chain.ainvoke(data),
            timeout=MAX_TIMEOUT
        )
        return result
    except asyncio.TimeoutError:
        raise TimeoutError(f"Request exceeded {MAX_TIMEOUT}s timeout")
```

### Database Connection Pooling

Use connection pools to limit database connections:

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool


# Configure connection pool
engine = create_engine(
    "postgresql://user:pass@localhost/db",
    poolclass=QueuePool,
    pool_size=20,              # Number of permanent connections
    max_overflow=10,           # Additional connections when needed
    pool_timeout=30,           # Wait time for connection
    pool_recycle=3600,         # Recycle connections after 1 hour
    pool_pre_ping=True,        # Verify connections before use
)
```

**Best Practices**:
- Design chains to be stateless
- Store session data in external storage (Redis, database)
- Use load balancers to distribute traffic
- Implement health checks for automatic instance rotation
- Use async operations for I/O-bound tasks
- Set appropriate resource limits (CPU, memory, timeout)
- Use connection pooling for databases and HTTP clients

---

## Testing Strategies

Comprehensive testing ensures production reliability.

### Unit Testing

Test individual components with mocks:

```python
import pytest
from unittest.mock import Mock, patch
from langchain_core.tools import BaseTool


def test_custom_tool():
    """Test custom tool in isolation."""
    tool = CalculatorTool()
    
    # Test addition
    result = tool._run(operation="add", operands=[2, 3])
    assert result == 5
    
    # Test error handling
    with pytest.raises(ValueError):
        tool._run(operation="invalid", operands=[1, 2])


@pytest.fixture
def mock_llm():
    """Fixture providing mock LLM."""
    mock = Mock()
    mock.invoke.return_value = "Test response"
    return mock


def test_chain_with_mock(mock_llm):
    """Test chain with mocked LLM."""
    chain = prompt | mock_llm | parser
    result = chain.invoke({"input": "test"})
    
    assert result is not None
    mock_llm.invoke.assert_called_once()
```

### Integration Testing

Test with real LLM APIs using test keys:

```python
@pytest.mark.integration
async def test_chain_end_to_end():
    """Test complete chain execution."""
    # Use test API key
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_TEST_KEY")
    
    chain = create_qa_chain()
    result = await chain.ainvoke({"question": "What is 2+2?"})
    
    assert "4" in result["answer"]


@pytest.mark.integration
def test_error_handling():
    """Test chain error handling and fallbacks."""
    chain_with_fallbacks = model.with_fallbacks([fallback_model])
    
    # Should succeed even if primary fails
    result = chain_with_fallbacks.invoke({"input": "test"})
    assert result is not None
```

### Load Testing

Simulate production traffic:

```python
# locust load test
from locust import HttpUser, task, between


class ChainUser(HttpUser):
    """Simulated user for load testing."""
    
    wait_time = between(1, 3)  # Wait 1-3s between requests
    
    @task
    def process_query(self):
        """Send query to chain endpoint."""
        self.client.post(
            "/api/process",
            json={"question": "What is LangChain?"}
        )
    
    @task(3)  # 3x more frequent
    def process_simple_query(self):
        """Send simple query."""
        self.client.post(
            "/api/process",
            json={"question": "Hello"}
        )


# Run: locust -f load_test.py --host=http://localhost:8000
```

Example `load_test.py` for k6:

```javascript
// k6 load test
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '30s', target: 20 },  // Ramp up to 20 users
    { duration: '1m', target: 20 },   // Stay at 20 users
    { duration: '30s', target: 0 },   // Ramp down
  ],
};

export default function () {
  let response = http.post(
    'http://localhost:8000/api/process',
    JSON.stringify({ question: 'What is LangChain?' }),
    { headers: { 'Content-Type': 'application/json' } }
  );
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 5s': (r) => r.timings.duration < 5000,
  });
  
  sleep(1);
}
```

### Chaos Engineering

Test resilience by injecting failures:

```python
@pytest.mark.chaos
async def test_retry_on_api_failure():
    """Test retry behavior under failures."""
    failure_count = 0
    
    def failing_llm(*args, **kwargs):
        nonlocal failure_count
        failure_count += 1
        if failure_count < 3:
            raise ConnectionError("Simulated failure")
        return "Success"
    
    with patch('langchain_openai.ChatOpenAI.invoke', side_effect=failing_llm):
        model_with_retry = ChatOpenAI().with_retry(max_attempt_number=5)
        result = model_with_retry.invoke("test")
        
        assert result == "Success"
        assert failure_count == 3  # Failed 2 times, succeeded on 3rd


@pytest.mark.chaos
def test_fallback_on_sustained_failure():
    """Test fallback triggers on sustained failures."""
    def always_fail(*args, **kwargs):
        raise Exception("Simulated sustained failure")
    
    with patch('langchain_openai.ChatOpenAI.invoke', side_effect=always_fail):
        model_with_fallback = ChatOpenAI().with_fallbacks([
            Mock(invoke=Mock(return_value="Fallback response"))
        ])
        
        result = model_with_fallback.invoke("test")
        assert result == "Fallback response"
```

### Test Suite Structure

```
tests/
├── unit/
│   ├── test_tools.py           # Tool unit tests
│   ├── test_chains.py          # Chain logic tests
│   └── test_callbacks.py       # Callback tests
├── integration/
│   ├── test_chain_e2e.py       # End-to-end chain tests
│   ├── test_llm_integration.py # Real LLM API tests
│   └── test_retrieval.py       # Retrieval integration tests
├── load/
│   ├── locustfile.py           # Locust load tests
│   └── k6_test.js              # k6 load tests
└── chaos/
    ├── test_retry_logic.py     # Retry behavior tests
    └── test_fallbacks.py       # Fallback tests
```

**Best Practices**:
- Unit test with mocks for fast feedback
- Integration test with real APIs using test keys
- Load test to identify bottlenecks and capacity limits
- Chaos test to verify retry and fallback logic
- Run tests in CI/CD pipeline
- Set performance targets (e.g., p95 < 5s)

---

## Deployment Checklist

Use this comprehensive checklist before deploying to production.

### Configuration

- [ ] Environment variables set for all services
- [ ] API keys configured in secrets management (not hardcoded)
- [ ] Secrets stored in AWS Secrets Manager/HashiCorp Vault
- [ ] Logging configured for production (INFO level, JSON format)
- [ ] Rotation policy set for API keys (90 days)
- [ ] Database connection strings configured
- [ ] Redis/cache connection configured

### Error Handling

- [ ] Retry logic configured with appropriate `max_attempt_number` (3-5)
- [ ] Exponential backoff enabled with jitter
- [ ] Fallback chains configured for critical paths
- [ ] Timeouts set on all external calls (LLM: 30-60s, DB: 5-10s)
- [ ] Circuit breakers implemented for external dependencies
- [ ] Error aggregation and alerting configured
- [ ] Exception types properly categorized (transient vs permanent)

### Monitoring

- [ ] Metrics collection enabled (latency, throughput, errors)
- [ ] Prometheus/Datadog integration configured
- [ ] Dashboards created for key metrics
- [ ] Alerts configured for critical conditions:
  - [ ] Error rate > 5%
  - [ ] p95 latency > 10s
  - [ ] Token usage > 90% of budget
  - [ ] Rate limit hits > 0
- [ ] Distributed tracing enabled (OpenTelemetry/LangSmith)
- [ ] Log aggregation configured (Elasticsearch/CloudWatch)

### Performance

- [ ] Caching enabled for appropriate use cases
- [ ] Cache backend configured (Redis for production)
- [ ] Async operations used for I/O-bound tasks
- [ ] Connection pooling configured (database, HTTP)
- [ ] Resource limits set (CPU, memory, timeout)
- [ ] Load testing completed with acceptable results:
  - [ ] p95 latency < target
  - [ ] Throughput meets requirements
  - [ ] Error rate < 1%

### Security

- [ ] API keys in secrets manager (not environment variables)
- [ ] Input validation in place with Pydantic models
- [ ] Output filtering enabled for PII
- [ ] HTTPS enforced (HTTP redirect configured)
- [ ] API authentication implemented
- [ ] Rate limiting per user configured
- [ ] SQL injection protection verified (parameterized queries)
- [ ] XSS protection enabled
- [ ] Security headers configured (CORS, CSP)

### Cost Control

- [ ] Token budgets configured (daily/monthly limits)
- [ ] Model selection logic implemented (GPT-3.5 for simple tasks)
- [ ] Cost monitoring enabled with callbacks
- [ ] Response caching active
- [ ] Budget alerts configured (90% threshold)
- [ ] Cost dashboard created
- [ ] Token usage tracked per user/tenant

### Testing

- [ ] Unit tests passing (>80% coverage)
- [ ] Integration tests with real APIs passing
- [ ] Load tests completed showing acceptable performance
- [ ] Error scenarios tested (retry, fallback, timeout)
- [ ] Chaos tests verified resilience
- [ ] Smoke tests for production deployment
- [ ] Rollback procedure tested

### Documentation

- [ ] Runbooks created for common issues
- [ ] On-call procedures documented
- [ ] Architecture diagrams current
- [ ] API documentation updated
- [ ] Deployment procedures documented
- [ ] Incident response plan documented

### Deployment

- [ ] Blue-green or canary deployment strategy defined
- [ ] Health check endpoints implemented (`/health`)
- [ ] Readiness probes configured (Kubernetes)
- [ ] Liveness probes configured
- [ ] Graceful shutdown handling implemented
- [ ] Database migrations tested
- [ ] Rollback procedure defined and tested
- [ ] Post-deployment smoke tests defined

---

## Example Production Setup

Complete example showing production-ready FastAPI application with all best practices.

### FastAPI Application

```python
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional
import time

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from prometheus_client import start_http_server
import redis.asyncio as redis

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# Configure logging
setup_production_logging()

logger = logging.getLogger(__name__)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=1, max_length=1000)
    user_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+$')
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    request_id: str
    execution_time_ms: int
    tokens_used: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup/shutdown)."""
    # Startup
    logger.info("Starting LangChain service")
    
    # Start metrics server
    start_metrics_server(port=9090)
    
    # Initialize Redis
    app.state.redis = redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        encoding="utf-8",
        decode_responses=True,
    )
    
    # Initialize rate limiter
    app.state.rate_limiter = RateLimiter(max_concurrent=50)
    
    # Initialize budget manager
    app.state.budget_manager = TokenBudgetManager(
        daily_budget=int(os.getenv("DAILY_TOKEN_BUDGET", 1_000_000))
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down LangChain service")
    await app.state.redis.close()


# Create FastAPI app
app = FastAPI(
    title="LangChain Production Service",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "").split(","),
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify authentication token."""
    token = credentials.credentials
    # Implement your token verification logic
    user_id = validate_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
        )
    return user_id


# Create chain with monitoring
def create_monitored_chain():
    """Create chain with callbacks for monitoring."""
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question concisely:\n\n{question}"
    )
    
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=0.7,
    ).with_retry(
        retry_exception_types=(ConnectionError, TimeoutError),
        max_attempt_number=3,
    )
    
    parser = StrOutputParser()
    
    return prompt | model | parser


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    checks = {
        "llm": await check_llm_health(),
        "redis": await check_redis_health(app.state.redis),
    }
    
    overall = HealthStatus.HEALTHY
    if any(s == HealthStatus.UNHEALTHY for s, _ in checks.values()):
        overall = HealthStatus.UNHEALTHY
    
    return {
        "status": overall.value,
        "checks": {
            name: {"status": status.value, "message": msg}
            for name, (status, msg) in checks.items()
        }
    }


async def check_redis_health(redis_client) -> tuple[HealthStatus, str]:
    """Check Redis connectivity."""
    try:
        await redis_client.ping()
        return (HealthStatus.HEALTHY, "Redis accessible")
    except Exception as e:
        return (HealthStatus.UNHEALTHY, f"Redis error: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    user_id: str = Depends(verify_token)
):
    """Process user query with full production setup.
    
    Features:
    - Rate limiting
    - Caching
    - Error handling
    - Monitoring
    - Cost tracking
    """
    import uuid
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        "Processing query",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "question_length": len(request.question),
        }
    )
    
    # Rate limiting
    async with app.state.rate_limiter:
        try:
            # Check cache
            cache_key = f"query:{hash(request.question)}"
            cached_result = await app.state.redis.get(cache_key)
            
            if cached_result:
                logger.info("Cache hit", extra={"request_id": request_id})
                execution_time = int((time.time() - start_time) * 1000)
                return QueryResponse(
                    answer=cached_result,
                    request_id=request_id,
                    execution_time_ms=execution_time,
                )
            
            # Create chain with callbacks
            metrics_callback = PrometheusMetricsCallback()
            logging_callback = LoggingCallback()
            llm_logging_callback = LLMLoggingCallback()
            budget_callback = TokenBudgetCallback(
                budget_manager=app.state.budget_manager,
                request_budget=10_000,
            )
            cost_callback = CostTrackingCallback()
            
            callbacks = [
                metrics_callback,
                logging_callback,
                llm_logging_callback,
                budget_callback,
                cost_callback,
            ]
            
            # Execute chain
            chain = create_monitored_chain()
            
            result = await asyncio.wait_for(
                chain.ainvoke(
                    {"question": request.question},
                    config={"callbacks": callbacks}
                ),
                timeout=60.0  # 60 second timeout
            )
            
            # Cache result
            await app.state.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                result
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Get cost info from callback
            cost_summary = cost_callback.get_summary()
            
            logger.info(
                "Query completed",
                extra={
                    "request_id": request_id,
                    "execution_time_ms": execution_time,
                    "estimated_cost": cost_summary.get("total_cost_usd", 0),
                }
            )
            
            return QueryResponse(
                answer=result,
                request_id=request_id,
                execution_time_ms=execution_time,
                estimated_cost_usd=cost_summary.get("total_cost_usd"),
            )
            
        except asyncio.TimeoutError:
            logger.error("Request timeout", extra={"request_id": request_id})
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request processing timeout"
            )
        except Exception as e:
            logger.error(
                "Request failed",
                extra={"request_id": request_id, "error": str(e)},
                exc_info=e
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None,  # Use our custom logging
    )
```

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  langchain-service:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"  # Prometheus metrics
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DAILY_TOKEN_BUDGET=1000000
      - ALLOWED_ORIGINS=https://yourdomain.com
      - OPENAI_MODEL=gpt-3.5-turbo
    depends_on:
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped

volumes:
  redis-data:
  prometheus-data:
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-service
  labels:
    app: langchain-service
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: langchain-service
  template:
    metadata:
      labels:
        app: langchain-service
    spec:
      containers:
      - name: langchain
        image: langchain-service:1.0.0
        ports:
        - name: http
          containerPort: 8000
        - name: metrics
          containerPort: 9090
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: langchain-secrets
              key: openai-api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DAILY_TOKEN_BUDGET
          value: "1000000"
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: langchain-service
spec:
  type: LoadBalancer
  selector:
    app: langchain-service
  ports:
  - name: http
    protocol: TCP
    port: 80
    targetPort: 8000
  - name: metrics
    protocol: TCP
    port: 9090
    targetPort: 9090
---
apiVersion: v1
kind: Secret
metadata:
  name: langchain-secrets
type: Opaque
stringData:
  openai-api-key: "YOUR_API_KEY_HERE"
```

---

## Troubleshooting Production Issues

Common production issues and their solutions.

### High Latency

**Symptoms**: p95 latency > 10s, slow response times

**Diagnosis**:
1. Check metrics dashboard for bottleneck:
   - LLM API latency - most common culprit
   - Database query time - check for slow queries
   - Network latency - check service connectivity
2. Enable profiling to identify slow components
3. Check for serialization in execution (should be async/concurrent)

**Solutions**:
```python
# Profile chain execution
profiler = cProfile.Profile()
profiler.enable()
result = chain.invoke(input_data)
profiler.disable()

stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)


# Optimize with caching
set_llm_cache(RedisCache(redis_client))


# Use batch processing
results = chain.batch(inputs)  # Concurrent execution


# Switch to async
results = await asyncio.gather(*[
    chain.ainvoke(input) for input in inputs
])
```

### Error Rate Spike

**Symptoms**: Error rate suddenly increases, alerts firing

**Diagnosis**:
1. Check error logs for common pattern:
   ```bash
   grep "ERROR" /var/log/langchain/langchain.log | tail -100
   ```
2. Check error types in metrics:
   - 429 errors → rate limiting
   - Timeout errors → slow responses
   - Connection errors → service down
3. Verify external service health (LLM API status page)
4. Check for recent deployments or configuration changes

**Solutions**:
```python
# Rate limit errors - implement backoff
model_with_retry = model.with_retry(
    max_attempt_number=5,
    wait_exponential_jitter=True
)

# Timeout errors - increase timeout or optimize
result = await asyncio.wait_for(
    chain.ainvoke(input_data),
    timeout=90  # Increased from 60
)

# Connection errors - check circuit breaker
if circuit_breaker.state == CircuitState.OPEN:
    # Use fallback
    result = fallback_chain.invoke(input_data)
```

### Memory Leak

**Symptoms**: Memory usage grows over time, eventual OOM crash

**Diagnosis**:
1. Profile memory usage:
   ```python
   from memory_profiler import profile
   
   @profile
   def process_request(input_data):
       return chain.invoke(input_data)
   ```
2. Check for unclosed async resources
3. Verify callback cleanup
4. Check cache size growth

**Solutions**:
```python
# Proper async resource cleanup
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.text()
# Session automatically closed


# Limit cache size
class BoundedCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value


# Clear callback references
callbacks = [LoggingCallback()]
result = chain.invoke(input_data, config={"callbacks": callbacks})
callbacks.clear()  # Release references
```

### Cost Overrun

**Symptoms**: Token usage exceeds budget, unexpected charges

**Diagnosis**:
1. Review token usage metrics:
   ```python
   summary = metrics_callback.get_summary()
   print(f"Total tokens: {summary['total_tokens']}")
   print(f"By model: {summary.get('cost_by_model', {})}")
   ```
2. Check for caching misses
3. Identify high-token requests
4. Review model selection logic

**Solutions**:
```python
# Enforce token budgets
budget_callback = TokenBudgetCallback(
    budget_manager=budget_manager,
    request_budget=5_000  # Limit per request
)

# Use cheaper models
def select_model_by_complexity(complexity: str):
    if complexity == "simple":
        return "gpt-3.5-turbo"  # Cheaper
    return "gpt-4"  # Only for complex tasks

# Improve caching
set_llm_cache(RedisCache(redis_client))

# Add semantic caching for similar queries
semantic_cache = SemanticCache(embeddings, threshold=0.95)
```

### Request Failures

**Symptoms**: Requests failing with 5xx errors

**Diagnosis**:
1. Check API key validity:
   ```python
   model = ChatOpenAI()
   try:
       model.invoke("test")
   except Exception as e:
       print(f"API test failed: {e}")
   ```
2. Verify rate limits not exceeded
3. Test external service connectivity
4. Check application logs for stack traces

**Solutions**:
```python
# Verify API keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set")

# Check rate limits
if rate_limit_error:
    # Respect Retry-After header
    retry_after = int(response.headers.get("Retry-After", 60))
    await asyncio.sleep(retry_after)
    retry()

# Test connectivity
import aiohttp

async def test_connectivity():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.openai.com/v1/models") as resp:
            if resp.status == 200:
                print("OpenAI API accessible")
            else:
                print(f"OpenAI API error: {resp.status}")
```

---

## Cross-References

For detailed information on specific topics covered in this guide:

- **Error Handling**: See [Error Handling Guide](error-handling.md) for comprehensive retry and fallback patterns
- **Async Operations**: See [Async Usage Guide](async-usage.md) for event loop management and performance optimization
- **Callbacks**: See [Callbacks Guide](callbacks.md) for custom callback implementation and event sequences
- **LCEL Composition**: See [LCEL Guide](lcel-composition.md) for type-safe chain building patterns
- **Debugging Utilities**: See `examples/debugging/` for reusable debugging tools
- **Glossary**: See [Glossary](../glossary.md) for term definitions

---

## Summary

Production deployment of LangChain applications requires:

1. **Comprehensive Logging**: Structured JSON logging with appropriate levels and rotation
2. **Monitoring**: Metrics collection, Prometheus/Datadog integration, alerting
3. **Rate Limiting**: Semaphores, circuit breakers, quota management
4. **Error Handling**: Retry with exponential backoff, fallback chains, timeout configuration
5. **Performance**: Caching, batching, async operations, connection pooling
6. **Security**: Secrets management, input validation, output filtering, authentication
7. **Cost Control**: Token budgets, model selection, response caching
8. **Observability**: Distributed tracing, debug mode, health checks
9. **Scalability**: Stateless design, load balancing, resource limits
10. **Testing**: Unit, integration, load, and chaos testing

Use the [Deployment Checklist](#deployment-checklist) to ensure all aspects are covered before going live.

**Remember**: Production readiness is an ongoing process. Continuously monitor metrics, respond to alerts, and iterate on your deployment strategy based on real-world usage patterns.
