# OpenTelemetry Integration

`LlmTraceSpanProcessor` captures OTEL spans from 80+ frameworks without requiring any code changes in your application.

## Installation

```bash
pip install llm-trace[otel]
```

## Quick start

```python
from llm_trace.otel import install_otel

# activate instrumentors and register the span processor globally
install_otel(instrumentors=["llama_index"])
```

That's all. Every LlamaIndex call will now be traced automatically.

## Supported instrumentors

Pass any combination to `instrumentors`:

| Name | Framework |
|------|-----------|
| `llama_index` | LlamaIndex (RAG) |
| `anthropic` | Anthropic SDK |
| `openai` | OpenAI SDK (via OpenInference) |
| `haystack` | Haystack (RAG) |
| `crewai` | CrewAI (multi-agent) |
| `dspy` | DSPy (structured prompting) |
| `mistral` | Mistral AI |
| `groq` | Groq |
| `bedrock` | AWS Bedrock |
| `langchain` | LangChain |

```python
# multiple frameworks at once
install_otel(instrumentors=["llama_index", "crewai", "dspy"])
```

## `install_otel()` parameters

```python
install_otel(
    instrumentors=["llama_index"],  # list of instrumentors to activate
    capture_all=False,              # if True, capture ALL OTEL spans (not just LLM-related)
)
```

## Manual setup

For more control, use `LlmTraceSpanProcessor` directly:

```python
from llm_trace.otel import LlmTraceSpanProcessor
from opentelemetry.sdk.trace import TracerProvider

processor = LlmTraceSpanProcessor(
    capture_all=False,
    filter_scopes={"openai", "anthropic"},  # only capture these instrumentors
)

provider = TracerProvider()
provider.add_span_processor(processor)
```

## Span type mapping

The processor maps OpenInference span kinds to llm-trace observation types:

| OTEL span kind | llm-trace type |
|---------------|----------------|
| LLM | GENERATION |
| CHAIN | SPAN |
| TOOL | TOOL |
| AGENT | AGENT |
| RETRIEVER | RETRIEVER |
| EMBEDDING | EMBEDDING |
| GUARDRAIL | GUARDRAIL |

## Attribute conventions

The processor understands two standards:

**GenAI Semantic Conventions (OTEL standard):**
- `gen_ai.request.model` → model
- `gen_ai.usage.input_tokens` / `output_tokens` → token counts
- `gen_ai.request.temperature`, `max_tokens` → model parameters
- `gen_ai.prompt`, `gen_ai.completion` → input/output

**OpenInference Conventions (Arize/Phoenix):**
- `openinference.span.kind` → observation type
- `input.value`, `output.value` → input/output
- `llm.model_name`, `llm.input_messages`, `llm.output_messages` → LLM details
- `llm.token_count.prompt`, `.completion` → usage
- `retrieval.documents` → retriever output
- `tool.name` → tool name

## Example: LlamaIndex

```python
from llm_trace.otel import install_otel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

install_otel(instrumentors=["llama_index"])

# build and query — everything is traced automatically
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
```

## Example: CrewAI

```python
from llm_trace.otel import install_otel
from crewai import Crew

install_otel(instrumentors=["crewai"])

crew = Crew(agents=[...], tasks=[...])
result = crew.kickoff()
```

## Notes

- `install_otel()` must be called **before** any framework code is imported or executed.
- If you already have an OTEL `TracerProvider` configured, use the manual setup to add the processor to your existing provider.
- Use `capture_all=True` with caution — it will also capture HTTP, database, and other non-LLM spans.
