# LangChain / LangGraph

`CallbackHandler` is a LangChain `BaseCallbackHandler` that automatically traces chains, agents, tools, retrievers, and LLM calls.

## Installation

```bash
pip install llmtrace[langchain]   # LangChain only
pip install llmtrace[langgraph]   # LangChain + LangGraph
```

## Basic usage

```python
from llm_trace.langchain import CallbackHandler

handler = CallbackHandler(
    trace_name="my-agent",
    session_id="chat-123",
)

result = chain.invoke(input, config={"callbacks": [handler]})

handler.flush()  # always call flush() when done
```

> **Important:** Always call `handler.flush()` after the last call. This finalizes the trace and commits all pending observations to storage. Without it, data may be lost — especially when streaming.

## Parameters

```python
CallbackHandler(
    trace_name="pipeline",       # name of the root trace (default: inferred from first callback)
    session_id="sess-abc",       # group traces by session
    user_id="user-123",          # group traces by user
    tags=["rag", "production"],  # filter tags
    metadata={"version": "v2"},  # arbitrary metadata
    environment="production",    # override environment
)
```

## What gets traced

Every LangChain primitive is captured as a typed observation:

| Primitive | Observation type |
|-----------|-----------------|
| Chain | SPAN (or inferred — see below) |
| LLM / ChatModel | GENERATION |
| Tool | TOOL |
| Retriever | RETRIEVER |
| Agent action | AGENT |

### Type inference from node names

The handler infers observation types from the node name when the type isn't explicitly set:

| Name contains | Inferred type |
|---------------|--------------|
| agent, planner, reasoner, orchestrat | AGENT |
| retriev, search, vector, rag | RETRIEVER |
| guardrail, moderat, safety, filter | GUARDRAIL |
| tool | TOOL |
| *(anything else)* | SPAN |

## LangGraph

Works the same way — pass the handler in the config:

```python
from llm_trace.langchain import CallbackHandler

handler = CallbackHandler(trace_name="my-graph", session_id="chat-456")

result = graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"callbacks": [handler]},
)

handler.flush()
```

Each graph node becomes a child observation with the node name as the observation name.

## Getting the trace ID

```python
handler.flush()
trace_id = handler.get_trace_id()
trace_url = handler.get_trace_url()  # http://localhost:7600/traces/<id>
print(f"View trace: {trace_url}")
```

## Streaming

For streaming responses, flush in a `finally` block to guarantee data is saved:

```python
try:
    async for chunk in graph.astream(input, config={"callbacks": [handler]}):
        yield chunk
finally:
    handler.flush()
```

## Combining with @observe()

`CallbackHandler` propagates context via `contextvars`, so `@observe()` functions called inside a LangGraph node are automatically attached as child observations:

```python
@observe(as_type="guardrail")
def check_safety(text: str) -> bool:
    ...

def my_node(state):
    safe = check_safety(state["output"])  # becomes a GUARDRAIL child observation
    return state
```

## Attaching scores

```python
from llm_trace import score

handler.flush()
score("quality", 0.92, trace_id=handler.get_trace_id())
score("grounded", True, trace_id=handler.get_trace_id(), source="llm_judge")
```

## Per-request handlers

Create a new `CallbackHandler` for each request to get separate traces:

```python
async def handle_request(user_id: str, message: str):
    handler = CallbackHandler(
        trace_name="chat",
        session_id=f"session-{user_id}",
        user_id=user_id,
    )
    try:
        result = await graph.ainvoke({"message": message}, config={"callbacks": [handler]})
        return result
    finally:
        handler.flush()
```
