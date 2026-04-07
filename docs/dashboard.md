# Dashboard

llm-trace ships with a built-in web dashboard at `http://localhost:7600`. No external dependencies, no configuration required.

## Starting the dashboard

**From the CLI:**
```bash
llm-trace dashboard
llm-trace dashboard --port 8080
llm-trace dashboard --no-browser    # start without opening browser
```

**From Python:**
```python
from llm_trace import tracer

tracer.dashboard()                         # default port 7600
tracer.dashboard(port=8080)
tracer.dashboard(port=7600, open_browser=False)
```

## Dashboard sections

### Overview

The main view shows:
- Total traces, observations, generations, and scores
- Total tokens consumed and estimated cost
- Average latency and error rate
- Top models by usage
- Time-series charts for tokens, cost, and latency

### Trace list

All traces sorted by time (newest first), with:
- Status indicator (ok / error)
- Name, trace ID
- Duration, token count, cost
- Tags
- Search and filter by name, environment, tags

### Trace detail

Click any trace to see:
- Full input/output
- Metadata, session, user, tags, environment
- **Tree view** — nested observation hierarchy with timing
- **Graph view** — visual execution flow showing connections between nodes
- Tokens and cost per observation
- Scores attached to the trace

### Scores

Dedicated section listing all scores across traces with filtering by name and source.

## REST API

The dashboard exposes a REST API that you can query directly:

| Endpoint | Description |
|----------|-------------|
| `GET /api/stats` | Global statistics |
| `GET /api/traces` | List traces (supports `limit`, `offset`, `name`, `environment`) |
| `GET /api/traces/{id}` | Full trace with observations and scores |
| `GET /api/timeline?hours=24` | Hourly time-series |
| `GET /api/timeseries` | Auto-granular time-series |
| `GET /api/scores?trace_id=…` | Scores for a trace |
| `DELETE /api/traces` | Delete all traces |
| `DELETE /api/traces/{id}` | Delete one trace |

Example:

```bash
# get recent traces
curl http://localhost:7600/api/traces?limit=10

# get trace detail
curl http://localhost:7600/api/traces/abc123def456
```

## Custom port

```bash
llm-trace dashboard --port 3000
```

If the port is already in use, start with a different one or kill the existing process.

## Custom database

The dashboard reads from the default database (`~/.llm-trace/traces.db`). To use a different file:

```bash
llm-trace --db ./my-project.db dashboard
```

Or set the environment variable:

```bash
LLM_TRACE_DB_PATH=./my-project.db llm-trace dashboard
```
