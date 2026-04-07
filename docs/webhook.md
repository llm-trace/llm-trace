# Webhook Ingestion

The webhook integration lets you send traces to llm-trace from **any language or platform** — Node.js, Go, Ruby, bash, no-code tools, or any system that can make HTTP requests.

## Installation

```bash
pip install llmtracing[webhook]
```

## Starting the webhook server

```bash
llm-trace dashboard  # starts dashboard on :7600
```

The standalone webhook server runs on port `7601`:

```python
from llm_trace.webhook import run_webhook_server
run_webhook_server(port=7601)
```

## Sending traces

**POST** `/api/ingest` with a JSON payload:

```bash
curl -X POST http://localhost:7601/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "trace": {
      "name": "my-pipeline",
      "session_id": "chat-123",
      "user_id": "user-456",
      "input": {"query": "Hello"},
      "output": {"response": "Hi there"},
      "tags": ["production"]
    },
    "observations": [
      {
        "type": "generation",
        "name": "gpt-4o-call",
        "model": "gpt-4o",
        "input": {"messages": [{"role": "user", "content": "Hello"}]},
        "output": {"content": "Hi there"},
        "usage": {
          "input_tokens": 10,
          "output_tokens": 5
        },
        "duration_ms": 850
      }
    ]
  }'
```

## Full payload format

```json
{
  "trace": {
    "id": "optional-custom-id",
    "name": "trace-name",
    "session_id": "session-abc",
    "user_id": "user-123",
    "input": {},
    "output": {},
    "metadata": {},
    "tags": ["tag1"],
    "environment": "production",
    "release": "v1.0"
  },
  "observations": [
    {
      "id": "optional-custom-id",
      "type": "generation|span|tool|agent|retriever|embedding|guardrail|event",
      "name": "operation-name",
      "input": {},
      "output": {},
      "model": "gpt-4o",
      "model_parameters": {
        "temperature": 0.7,
        "max_tokens": 1024
      },
      "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_tokens": 0,
        "reasoning_tokens": 0
      },
      "cost": {
        "input_cost": 0.001,
        "output_cost": 0.002,
        "total_cost": 0.003
      },
      "duration_ms": 1500,
      "status": "ok|error",
      "level": "DEFAULT|WARNING|ERROR",
      "error": "error message if failed",
      "metadata": {}
    }
  ],
  "scores": [
    {
      "name": "quality",
      "value": 0.95,
      "source": "api|human|llm_judge|user_feedback",
      "comment": "Optional rationale",
      "observation_id": "optional — attaches to a specific observation"
    }
  ]
}
```

### Notes on the payload

- `trace.id` — optional. If omitted, a random ID is generated.
- `observations[].duration_ms` — used to calculate `start_time` and `end_time` automatically.
- `observations[].cost` — if omitted and `model` + `usage` are present, cost is calculated automatically from the built-in pricing table.
- All fields except `trace` are optional.

## Adding scores to an existing trace

**POST** `/api/score`:

```bash
curl -X POST http://localhost:7601/api/score \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "abc123",
    "name": "user_feedback",
    "value": 1,
    "source": "user_feedback",
    "comment": "thumbs up"
  }'
```

## Embedding in a FastAPI app

```python
from fastapi import FastAPI
from llm_trace.webhook import create_fastapi_router

app = FastAPI()
app.include_router(create_fastapi_router(), prefix="/traces")

# endpoints now available at:
# POST /traces/api/ingest
# POST /traces/api/score
```

## Response

On success, the ingest endpoint returns `201 Created`:

```json
{
  "trace_id": "a1b2c3d4e5f6",
  "observations": 3,
  "scores": 1
}
```

## JavaScript / Node.js example

```javascript
await fetch("http://localhost:7601/api/ingest", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    trace: { name: "node-pipeline", input: { query: "hello" } },
    observations: [{
      type: "generation",
      name: "openai-call",
      model: "gpt-4o",
      usage: { input_tokens: 50, output_tokens: 30 },
      duration_ms: 700,
    }]
  })
});
```
