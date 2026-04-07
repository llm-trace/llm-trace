<div align="center">

<img src="assets/banner.png" alt="llm-trace" width="800"/>

[![CI](https://github.com/llm-trace/llm-trace/actions/workflows/ci.yml/badge.svg)](https://github.com/llm-trace/llm-trace/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/llmtrace.svg)](https://pypi.org/project/llmtrace/)
[![Python](https://img.shields.io/pypi/pyversions/llmtrace)](https://pypi.org/project/llmtrace/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/llmtrace)](https://pypi.org/project/llmtrace/)

[Installation](#installation) • [Quick Start](docs/quickstart.md) • [@observe()](docs/observe.md) • [Wrappers](docs/wrappers.md) • [LangChain](docs/langchain.md) • [Dashboard](docs/dashboard.md) • [Full Docs](docs/index.md)

</div>

**Lightweight LLM observability. Zero infrastructure.**

One SQLite file. One `pip install`. Full tracing for any LLM application.

```bash
pip install llmtrace
```

## Why llm-trace?

| Feature | Langfuse | LangSmith | **llm-trace** |
|---------|----------|-----------|---------------|
| Setup | Docker + PostgreSQL + Redis | Cloud account | `pip install llmtrace` |
| Infrastructure | 4 services | Managed | **Zero** |
| Storage | ClickHouse | Cloud | **SQLite** |
| Dependencies | Many | Many | **Zero** |
| Dashboard | Yes | Yes | **Built-in (:7600)** |
| Cost | Free/Paid | Paid | **Free forever** |

## Quick Start

### 1. Decorate any function

```python
from llm_trace import observe

@observe()
def my_pipeline(query: str) -> str:
    docs = retrieve(query)
    return generate(query, docs)
```

### 2. Wrap OpenAI / Anthropic SDKs

```python
from llm_trace.wrappers import wrap_openai
import openai

client = wrap_openai(openai.OpenAI())
# Every call is traced automatically — tokens, cost, latency
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

### 3. LangChain / LangGraph (recommended)

```python
from llm_trace.langchain import CallbackHandler

handler = CallbackHandler(trace_name="my-agent", session_id="chat-123")
result = graph.invoke(input, config={"callbacks": [handler]})
handler.flush()
# 1 invoke = 1 trace. All nodes captured automatically.
```

### 4. OpenTelemetry (LlamaIndex, Haystack, CrewAI, DSPy...)

```python
from llm_trace.otel import install_otel

install_otel(instrumentors=["llama_index", "haystack"])
# Any OTEL-instrumented framework is now traced
```

### 5. Webhook (any language)

```bash
curl -X POST http://localhost:7601/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"trace":{"name":"from-curl"}, "observations":[{"type":"generation","model":"gpt-4o","usage":{"input_tokens":100,"output_tokens":50}}]}'
```

### 6. Scores

```python
from llm_trace import score
score("quality", 0.92, trace_id=handler.get_trace_id())
```

## Dashboard

```bash
llm-trace dashboard
# or
python -c "from llm_trace import tracer; tracer.dashboard()"

# Custom port
llm-trace dashboard --port 8080
llm-trace dashboard -p 3000 --no-browser
```

Opens `http://localhost:7600` with:
- Overview with tokens/cost/latency charts over time
- Trace list with search and filtering
- Tree view with expandable observations
- Graph view showing execution flow
- Per-trace stats (tokens, cost, latency, status)
- Delete individual traces or clear all

## CLI

```bash
llm-trace stats              # Summary
llm-trace list                # Recent traces
llm-trace show <trace-id>     # Detail view
llm-trace dashboard            # Web UI
llm-trace clear               # Delete all
llm-trace export > traces.json # Export
```

## Architecture

```
┌─────────────┬──────────────┬───────────────┬──────────────┐
│  @observe() │ wrap_openai()│ CallbackHandler│ install_otel()│
│  any func   │ wrap_anthropic│ LangChain/LG │ OTEL spans   │
├─────────────┴──────────────┴───────────────┴──────────────┤
│                    Tracer (singleton)                      │
│              background flush, contextvars                 │
├───────────────────────────────────────────────────────────┤
│                  SQLite WAL (~/.llm-trace/)                │
│            traces → observations → scores                  │
├───────────────────────────────────────────────────────────┤
│           Dashboard (:7600) │ CLI │ Webhook (:7601)       │
└───────────────────────────────────────────────────────────┘
```

## Data Model

Follows the Langfuse data model:

- **Trace** — one execution (1 `graph.invoke()` = 1 trace)
- **Observation** — a step within a trace (generation, tool, retriever, agent, span, guardrail, embedding, event)
- **Score** — evaluation metric attached to a trace

## Configuration

```bash
# Database location (default: ~/.llm-trace/traces.db)
export LLM_TRACE_DB_PATH=./my-traces.db

# Environment tag
export LLM_TRACE_ENVIRONMENT=production

# Release version
export LLM_TRACE_RELEASE=v1.2.3
```

## Optional Dependencies

```bash
pip install llmtrace                    # Zero deps, @observe + wrappers + CLI
pip install llmtrace[langchain]         # + LangChain CallbackHandler
pip install llmtrace[langgraph]         # + LangChain + LangGraph
pip install llmtrace[otel]              # + OpenTelemetry SpanProcessor
pip install llmtrace[webhook]           # + FastAPI webhook router
pip install llmtrace[all]               # Everything
```

## License

MIT

---

<div align="center">
  Made with ❤️ from Mallorca by <a href="https://github.com/marcmayol">marcmayol</a>
</div>
