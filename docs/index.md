# llm-trace Documentation

**Lightweight LLM observability. Zero infrastructure.**

One SQLite file. One `pip install`. Full tracing for any LLM application.

## Table of Contents

| Section | Description |
|---------|-------------|
| [Quick Start](quickstart.md) | Up and running in 2 minutes |
| [@observe() Decorator](observe.md) | Trace any Python function |
| [OpenAI & Anthropic Wrappers](wrappers.md) | Drop-in SDK wrappers with automatic tracing |
| [LangChain / LangGraph](langchain.md) | CallbackHandler for LangChain apps |
| [OpenTelemetry](otel.md) | LlamaIndex, CrewAI, Haystack, DSPy, and 80+ frameworks |
| [Webhook Ingestion](webhook.md) | Ingest traces from any language |
| [Scoring API](scoring.md) | Attach evaluation metrics to traces |
| [Dashboard](dashboard.md) | Built-in web UI at localhost:7600 |
| [CLI Reference](cli.md) | Command-line interface |
| [Data Model](data-model.md) | Trace, Observation, Score structures |
| [Configuration](configuration.md) | Environment variables and options |

## Architecture

```
┌─────────────┬──────────────┬───────────────┬──────────────┐
│  @observe() │ wrap_openai()│ CallbackHandler│ install_otel()│
│  any func   │ wrap_anthropic│ LangChain/LG  │ OTEL spans   │
├─────────────┴──────────────┴───────────────┴──────────────┤
│                    Tracer (singleton)                      │
│              background flush, contextvars                 │
├───────────────────────────────────────────────────────────┤
│                  SQLite WAL (~/.llm-trace/)                │
│             traces → observations → scores                 │
├───────────────────────────────────────────────────────────┤
│           Dashboard (:7600) │ CLI │ Webhook (:7601)        │
└───────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install llmtracing                    # zero deps — @observe, wrappers, CLI
pip install llmtracing[langchain]         # + LangChain CallbackHandler
pip install llmtracing[langgraph]         # + LangChain + LangGraph
pip install llmtracing[otel]              # + OpenTelemetry SpanProcessor
pip install llmtracing[webhook]           # + FastAPI webhook router
pip install llmtracing[all]               # everything
```
