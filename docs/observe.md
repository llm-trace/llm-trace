# @observe() Decorator

The `@observe()` decorator is the core primitive of llm-trace. It wraps any Python function (sync or async) and records it as a trace or a nested observation.

## Basic usage

```python
from llm_trace import observe

@observe()
def my_pipeline(query: str) -> str:
    return generate(query)
```

Every call to `my_pipeline()` creates a **Trace** in the database with the function name, input arguments, return value, duration, and status.

## Nested functions

When an `@observe()`-decorated function is called from inside another `@observe()` function, the inner call becomes a child **Observation** instead of a new Trace.

```python
@observe()
def pipeline(query: str) -> str:
    docs = retrieve(query)        # child observation
    return generate(query, docs)  # child observation

@observe(as_type="retriever")
def retrieve(query: str) -> list:
    ...

@observe(as_type="generation")
def generate(query: str, docs: list) -> str:
    ...
```

This produces:
```
Trace: pipeline
  └─ Observation [RETRIEVER]: retrieve
  └─ Observation [GENERATION]: generate
```

## Parameters

```python
@observe(
    name="custom-name",          # override function name (default: __qualname__)
    as_type="generation",        # observation type (see below)
    capture_input=True,          # record function arguments (default: True)
    capture_output=True,         # record return value (default: True)
    session_id="session-abc",    # group traces by session
    user_id="user-123",          # group traces by user
    tags=["rag", "v2"],          # filter tags
    metadata={"version": "v2"},  # arbitrary metadata
)
def my_func():
    ...
```

### Observation types (`as_type`)

| Value | Use case |
|-------|----------|
| `"span"` | Generic step (default) |
| `"generation"` | LLM call |
| `"tool"` | Tool / function call |
| `"agent"` | Agent decision or action |
| `"retriever"` | Vector search / RAG retrieval |
| `"embedding"` | Embedding generation |
| `"guardrail"` | Safety or moderation check |
| `"event"` | Discrete event, no duration |

## Async functions

`@observe()` works transparently with `async def`:

```python
@observe()
async def async_pipeline(query: str) -> str:
    docs = await retrieve(query)
    return await generate(query, docs)
```

## Error handling

Exceptions are caught, recorded on the observation with the full traceback, and re-raised. The observation status is set to `"error"`.

```python
@observe()
def risky():
    raise ValueError("something went wrong")

# The exception propagates normally, but is recorded in the trace.
```

## Accessing the current trace

Inside an `@observe()` function you can access the active trace:

```python
from llm_trace import tracer, score

@observe()
def pipeline(query: str) -> str:
    result = generate(query)
    # attach a score to the current trace
    score("quality", 0.95)
    return result
```

## Input/output capture limits

To avoid storing huge payloads, inputs and outputs are automatically truncated:

- Strings: 500 characters max
- Lists: 20 items max
- Dicts: 50 keys max

Set `capture_input=False` or `capture_output=False` to skip capture entirely.

## Flushing

Writes are batched in a background thread. Call `flush()` before the process exits to ensure all data is saved:

```python
from llm_trace import flush
flush()
```

Or use `shutdown()` for a full graceful stop:

```python
from llm_trace import shutdown
shutdown()
```
