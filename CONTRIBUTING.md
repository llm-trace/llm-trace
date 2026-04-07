# Contributing to llm-trace

Thank you for your interest in contributing! Here's everything you need to get started.

## Development Setup

```bash
git clone https://github.com/llm-trace/llm-trace.git
cd llm-trace
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Linting & Type Checking

```bash
ruff check .          # lint
ruff format .         # format
mypy llm_trace/ --ignore-missing-imports  # type check
```

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes and add tests if applicable
3. Ensure the test suite passes and linting is clean
4. Open a PR with a clear description of the change

## Reporting Bugs

Use the [bug report template](https://github.com/llm-trace/llm-trace/issues/new?template=bug_report.yml) to open an issue.

## Proposing Features

Use the [feature request template](https://github.com/llm-trace/llm-trace/issues/new?template=feature_request.yml) to suggest new functionality.

## Code Style

- Python 3.11+
- Type annotations required
- Line length: 88 (ruff default)
- Follow existing patterns in the codebase

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
