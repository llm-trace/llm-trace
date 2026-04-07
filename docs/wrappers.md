# OpenAI & Anthropic Wrappers

Drop-in wrappers that instrument the official SDKs without changing your code.

## OpenAI

```python
from llm_trace.wrappers import wrap_openai
import openai

client = wrap_openai(openai.OpenAI())
```

`wrap_openai()` mutates and returns the same client. Every subsequent call to `client.chat.completions.create()` and `client.embeddings.create()` is automatically traced.

### What gets captured

| Field | Source |
|-------|--------|
| Model | `model` parameter |
| Input | `messages` array |
| Output | Response content + tool calls |
| Tokens | `usage.prompt_tokens`, `completion_tokens` |
| Cached tokens | `usage.prompt_tokens_details.cached_tokens` |
| Reasoning tokens | `usage.completion_tokens_details.reasoning_tokens` (o1 models) |
| Cost | Calculated from built-in pricing table |
| Latency | Wall clock time of the call |
| TTFT | Time to first token (streaming) |
| Model parameters | temperature, max_tokens, top_p, stop, response_format |

### Async client

```python
client = wrap_openai(openai.AsyncOpenAI())

response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Streaming

Streaming is fully supported. TTFT (time to first token) is captured automatically.

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Nesting with @observe()

If you call the wrapped client inside an `@observe()` function, the LLM call becomes a child observation of the outer trace:

```python
@observe()
def rag_pipeline(query: str) -> str:
    docs = retrieve(query)
    response = client.chat.completions.create(...)  # nested GENERATION
    return response.choices[0].message.content
```

---

## Anthropic

```python
from llm_trace.wrappers import wrap_anthropic
import anthropic

client = wrap_anthropic(anthropic.Anthropic())
```

### What gets captured

| Field | Source |
|-------|--------|
| Model | `model` parameter |
| Input | `messages` + `system` |
| Output | Content blocks joined as text |
| Tokens | `usage.input_tokens`, `output_tokens` |
| Cached tokens | `usage.cache_read_input_tokens` |
| Cost | Calculated from built-in pricing table |
| Latency | Wall clock time |
| Model parameters | temperature, max_tokens, top_p, top_k, stop_sequences |

### Async client

```python
client = wrap_anthropic(anthropic.AsyncAnthropic())

response = await client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

---

## Supported models and pricing

Built-in pricing ($/1M tokens) for:

**OpenAI:** gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini, o3, o3-mini, o4-mini

**Anthropic:** claude-sonnet-4, claude-opus-4, claude-3.5-sonnet, claude-3.5-haiku, claude-3-opus, claude-3-haiku

If the model is not in the pricing table, tokens are still recorded but cost will be `$0.000`.

---

## Installation

```bash
pip install llmtracing[openai]      # for OpenAI
pip install llmtracing[anthropic]   # for Anthropic
pip install llmtracing[all]         # both
```
