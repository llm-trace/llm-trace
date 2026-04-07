# Configuration

llm-trace is configured via environment variables. No config files required.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_TRACE_DB_PATH` | `~/.llm-trace/traces.db` | Path to the SQLite database file |
| `LLM_TRACE_ENVIRONMENT` | `development` | Environment tag attached to all traces |
| `LLM_TRACE_RELEASE` | *(none)* | Release version attached to all traces |

### Examples

```bash
# custom database location
export LLM_TRACE_DB_PATH=./project-traces.db

# production environment
export LLM_TRACE_ENVIRONMENT=production

# track which version of your app generated the traces
export LLM_TRACE_RELEASE=v1.4.2
```

## Configuring the Tracer in Python

You can also configure the tracer programmatically at startup:

```python
from llm_trace import Tracer

tracer = Tracer(
    db_path="./my-traces.db",
    environment="staging",
    release="v1.4.2",
    flush_interval=2.0,   # seconds between background flushes (default: 1.0)
    enabled=True,         # set False to disable tracing entirely
)
```

> The `tracer` imported from `llm_trace` is a global singleton. If you need a custom configuration, create your own `Tracer` instance and use it directly instead of importing `tracer`.

## Disabling tracing

To disable all tracing (e.g. in unit tests):

```python
from llm_trace import Tracer

tracer = Tracer(enabled=False)
```

Or conditionally:

```python
import os
from llm_trace import Tracer

tracer = Tracer(enabled=os.getenv("TRACING_ENABLED", "true") == "true")
```

## Per-project databases

Use different databases for different projects:

```bash
# project A
LLM_TRACE_DB_PATH=./project-a.db python my_app.py

# project B
LLM_TRACE_DB_PATH=./project-b.db llm-trace dashboard
```

## Database location

The default database is created at `~/.llm-trace/traces.db`. The parent directory is created automatically on first use.

WAL (Write-Ahead Logging) mode is enabled by default, which allows concurrent reads from the dashboard while the tracer is writing.

## Flush interval

Observations are written to SQLite in batches by a background thread. The default flush interval is 1 second. Increase it to reduce I/O, decrease it to get near-real-time visibility in the dashboard:

```python
Tracer(flush_interval=0.5)   # flush every 500ms
Tracer(flush_interval=5.0)   # flush every 5 seconds
```

Always call `flush()` or `shutdown()` before the process exits to avoid losing buffered data.
