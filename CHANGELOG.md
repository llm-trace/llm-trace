# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-04-07

### Added
- Zero-dependency core with SQLite-based tracing
- `@observe()` decorator for tracing any Python function
- OpenAI and Anthropic SDK wrappers (`wrap_openai`, `wrap_anthropic`)
- LangChain / LangGraph `CallbackHandler`
- OpenTelemetry `SpanProcessor` (`install_otel`)
- FastAPI webhook router for cross-language ingestion
- Built-in web dashboard at `localhost:7600` with charts, trace trees, and execution graphs
- CLI (`llm-trace stats`, `list`, `show`, `dashboard`, `clear`, `export`)
- Scoring / evaluation metrics (`score()`)
- Background flush with `contextvars` for async-safe tracing
- Support for Python 3.11, 3.12, 3.13

[Unreleased]: https://github.com/llm-trace/llm-trace/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/llm-trace/llm-trace/releases/tag/v0.2.0
