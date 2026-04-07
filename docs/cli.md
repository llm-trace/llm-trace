# CLI Reference

The `llm-trace` CLI lets you inspect traces, view stats, and launch the dashboard from the terminal.

## Global option

```bash
llm-trace --db PATH <command>
# Use a custom database instead of ~/.llm-trace/traces.db
```

---

## `llm-trace stats`

Print a summary of all stored trace data.

```
llm-trace stats
```

Output includes:
- Total traces, observations, generations, scores
- Total tokens and estimated cost
- Average latency and error rate
- Top 10 models by usage

---

## `llm-trace list`

List recent traces.

```
llm-trace list [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--limit N` | Number of traces to show (default: 20) |
| `--name TEXT` | Filter by name substring |
| `--env TEXT` | Filter by environment (`development`, `production`, etc.) |
| `--tag TEXT` | Filter by tag |

Each row shows: status, name, ID, timestamp, duration, observation count, tokens, cost, tags.

---

## `llm-trace show <trace_id>`

Show full details for a single trace.

```
llm-trace show abc123
llm-trace show abc    # partial ID matching works
```

Output includes:
- Trace metadata (session, user, environment, release)
- Input and output
- Full observation tree with timing and nesting
- Error messages (if any observations failed)
- All attached scores

---

## `llm-trace dashboard`

Launch the web UI.

```
llm-trace dashboard [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--port N` / `-p N` | Port to listen on (default: 7600) |
| `--no-browser` | Do not open the browser automatically |

Opens `http://localhost:7600` in your default browser.

---

## `llm-trace export`

Export all traces to JSON.

```
llm-trace export [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--limit N` | Number of traces to export (default: all) |
| `--output FILE` | Write to file instead of stdout |

```bash
llm-trace export > traces.json
llm-trace export --output backup.json
llm-trace export --limit 100 > recent.json
```

---

## `llm-trace clear`

Delete all trace data from the database.

```
llm-trace clear [--yes]
```

Prompts for confirmation unless `--yes` is passed.

```bash
llm-trace clear          # asks "Are you sure? [y/N]"
llm-trace clear --yes    # skip confirmation
```
