# Data Model

llm-trace follows the [Langfuse data model](https://langfuse.com/docs/tracing-data-model). The hierarchy is:

```
Session
  └─ Trace           (one pipeline execution)
      └─ Observation  (one step: generation, tool call, retrieval...)
          └─ Score    (evaluation metric)
```

---

## Trace

A **Trace** represents one top-level execution — one `graph.invoke()`, one `@observe()` call, one webhook payload.

```python
@dataclass
class Trace:
    id: str                    # 16-char hex, auto-generated
    name: str                  # function name or custom name
    session_id: str | None     # groups traces by conversation or session
    user_id: str | None        # groups traces by user
    input: Any                 # function arguments or pipeline input
    output: Any                # return value or pipeline output
    metadata: dict             # arbitrary key-value data
    tags: list[str]            # filter labels
    environment: str           # "development" | "staging" | "production"
    release: str | None        # version / release identifier
    start_time: str            # ISO 8601 UTC
    end_time: str | None       # ISO 8601 UTC (None if still running)
    observations: list[Observation]  # child observations
```

**Computed properties:**
- `duration_ms` — total execution time in milliseconds
- `total_tokens` — sum of all observation tokens
- `total_cost` — sum of all observation costs

---

## Observation

An **Observation** is one step inside a trace. Observations can be nested (parent-child) to represent the call tree.

```python
@dataclass
class Observation:
    id: str                          # 16-char hex
    trace_id: str                    # parent trace
    parent_id: str | None            # parent observation (None = root level)
    type: ObservationType            # GENERATION, SPAN, TOOL, etc.
    name: str                        # step name
    status: str                      # "ok" | "error"
    level: str                       # "DEFAULT" | "WARNING" | "ERROR" | "DEBUG"
    input: Any                       # step input
    output: Any                      # step output
    model: str | None                # LLM model name (for GENERATION)
    model_parameters: dict           # temperature, max_tokens, etc.
    usage: UsageDetails | None       # token counts
    cost: CostDetails | None         # estimated cost
    metadata: dict                   # arbitrary key-value data
    start_time: str                  # ISO 8601 UTC
    end_time: str | None             # ISO 8601 UTC
    completion_start_time: str | None  # time to first token (streaming)
    error_message: str | None        # full traceback if status="error"
```

**Computed properties:**
- `duration_ms` — end_time minus start_time in milliseconds
- `ttft_ms` — time to first token (streaming only)

### ObservationType

| Type | Description |
|------|-------------|
| `GENERATION` | LLM call — tokens and cost are tracked here |
| `SPAN` | Generic step (function, chain, pipeline stage) |
| `TOOL` | Tool or function call |
| `AGENT` | Agent decision or reasoning step |
| `RETRIEVER` | Vector search or document retrieval |
| `EMBEDDING` | Embedding generation |
| `GUARDRAIL` | Safety or moderation check |
| `EVENT` | Point-in-time event (no duration) |

---

## UsageDetails

Token breakdown for a generation observation.

```python
@dataclass
class UsageDetails:
    input_tokens: int        # prompt tokens
    output_tokens: int       # completion tokens
    total_tokens: int        # input + output
    cached_tokens: int       # prompt cache hits (Claude / GPT)
    reasoning_tokens: int    # extended thinking tokens (o1 models)
```

---

## CostDetails

Estimated cost for a generation observation.

```python
@dataclass
class CostDetails:
    input_cost: float    # cost for input tokens
    output_cost: float   # cost for output tokens
    total_cost: float    # input_cost + output_cost
```

Cost is calculated automatically when the model is known. See the [pricing table in wrappers.md](wrappers.md#supported-models-and-pricing).

---

## Score

A **Score** is an evaluation metric attached to a trace or a specific observation.

```python
@dataclass
class Score:
    id: str                     # 16-char hex
    trace_id: str               # associated trace
    observation_id: str | None  # associated observation (optional)
    name: str                   # "quality", "groundedness", etc.
    value: float | str | bool   # numeric, categorical, or boolean
    data_type: ScoreDataType    # NUMERIC | CATEGORICAL | BOOLEAN
    source: ScoreSource         # API | HUMAN | LLM_JUDGE | USER_FEEDBACK
    comment: str | None         # optional rationale
    created_at: str             # ISO 8601 UTC
```

### ScoreDataType

| Type | Python value | Example |
|------|-------------|---------|
| `NUMERIC` | `float` or `int` | `0.92`, `5` |
| `BOOLEAN` | `bool` | `True`, `False` |
| `CATEGORICAL` | `str` | `"helpful"`, `"toxic"` |

### ScoreSource

| Source | Meaning |
|--------|---------|
| `API` | Set programmatically |
| `HUMAN` | Annotated by a human |
| `LLM_JUDGE` | Generated by an LLM |
| `USER_FEEDBACK` | Submitted by the end user |

---

## Database schema

Data is stored in SQLite at `~/.llm-trace/traces.db` (configurable). The schema has three tables:

- `traces` — one row per trace
- `observations` — one row per observation, with `trace_id` and `parent_id` foreign keys
- `scores` — one row per score, with `trace_id` and optional `observation_id` foreign keys

All `input`, `output`, and `metadata` fields are stored as JSON strings for schema flexibility.
