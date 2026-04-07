# Quick Start

Up and running in under 2 minutes.

## 1. Install

```bash
pip install llmtrace
```

## 2. Trace your first function

```python
from llm_trace import observe, tracer

@observe()
def ask_llm(question: str) -> str:
    # your LLM call here
    return "answer"

ask_llm("What is the capital of France?")

# open dashboard
tracer.dashboard()
```

That's it. A trace is saved to `~/.llm-trace/traces.db` automatically.

## 3. View the dashboard

```bash
llm-trace dashboard
# opens http://localhost:7600
```

---

## Common patterns

### OpenAI — automatic tracing

```python
from llm_trace.wrappers import wrap_openai
import openai

client = wrap_openai(openai.OpenAI())

# every call is now traced — tokens, cost, latency
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Anthropic — automatic tracing

```python
from llm_trace.wrappers import wrap_anthropic
import anthropic

client = wrap_anthropic(anthropic.Anthropic())

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

### LangChain / LangGraph

```python
from llm_trace.langchain import CallbackHandler

handler = CallbackHandler(trace_name="my-agent", session_id="chat-123")
result = graph.invoke(input, config={"callbacks": [handler]})
handler.flush()  # always call flush() at the end
```

### Any other framework (OpenTelemetry)

```python
from llm_trace.otel import install_otel

install_otel(instrumentors=["llama_index"])
# any LlamaIndex call is now traced
```

---

## Next steps

- [`@observe()` decorator](observe.md) — trace any function, including async
- [Scoring API](scoring.md) — attach quality metrics to your traces
- [Configuration](configuration.md) — custom DB path, environments, releases
