"""CLI para llm-trace.

Consulta traces, stats y lanza el dashboard desde terminal.

Usage:
    llm-trace stats                    # Estadísticas globales
    llm-trace list                     # Últimos 20 traces
    llm-trace list --name rag --limit 5
    llm-trace show <trace_id>          # Detalle de un trace
    llm-trace dashboard                # Lanza el dashboard
    llm-trace export                   # Exporta traces a JSON
    llm-trace clear                    # Borra todos los datos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_trace.storage import Storage

# ── ANSI colors ───────────────────────────────────────────


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


TYPE_COLORS = {
    "generation": C.BLUE,
    "span": C.GREEN,
    "tool": C.YELLOW,
    "agent": C.MAGENTA,
    "retriever": C.CYAN,
    "embedding": C.MAGENTA,
    "event": C.DIM,
    "guardrail": C.RED,
}


def _fmt_duration(ms: float | None) -> str:
    if ms is None:
        return "-"
    if ms < 1:
        return "<1ms"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.2f}s"


def _fmt_tokens(n: int) -> str:
    if not n:
        return "-"
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


def _fmt_cost(c: float) -> str:
    if not c:
        return "-"
    if c < 0.001:
        return f"${c:.6f}"
    if c < 1:
        return f"${c:.4f}"
    return f"${c:.2f}"


def cmd_stats(storage: Storage, args: argparse.Namespace) -> None:
    stats = storage.get_stats()
    print(f"\n{C.BOLD}📊 llm-trace stats{C.RESET}")
    print(f"{'─' * 40}")
    print(f"  Traces:       {C.BOLD}{stats['trace_count']}{C.RESET}")
    print(f"  Observations: {stats['observation_count']}")
    print(f"  Generations:  {stats['generation_count']}")
    print(f"  Scores:       {stats['score_count']}")
    print(f"  Total tokens: {_fmt_tokens(stats['total_tokens'])}")
    print(f"  Total cost:   {C.GREEN}{_fmt_cost(stats['total_cost'])}{C.RESET}")
    print(f"  Avg latency:  {_fmt_duration(stats['avg_latency_ms'])}")

    err = stats["error_rate"]
    err_color = C.RED if err > 5 else C.GREEN
    print(f"  Error rate:   {err_color}{err}%{C.RESET}")

    if stats["top_models"]:
        print(f"\n  {C.BOLD}Top Models:{C.RESET}")
        for m in stats["top_models"]:
            print(
                f"    {m['model']:30s} "
                f"{m['count']:>5d} calls  "
                f"{_fmt_tokens(m['tokens']):>8s} tok  "
                f"{_fmt_cost(m['cost']):>10s}"
            )
    print(f"\n  DB: {C.DIM}{storage.db_path}{C.RESET}\n")


def cmd_list(storage: Storage, args: argparse.Namespace) -> None:
    traces = storage.list_traces(
        limit=args.limit,
        name=args.name,
        environment=args.env,
        tag=args.tag,
    )

    if not traces:
        print(f"\n{C.DIM}  No traces found.{C.RESET}\n")
        return

    print(f"\n{C.BOLD}📋 Traces{C.RESET} (showing {len(traces)})")
    print(f"{'─' * 90}")

    for t in traces:
        has_error = any(o.status == "error" for o in t.observations)
        status = f"{C.RED}✗{C.RESET}" if has_error else f"{C.GREEN}✓{C.RESET}"
        dur = _fmt_duration(t.duration_ms)
        tokens = _fmt_tokens(t.total_tokens)
        cost = _fmt_cost(t.total_cost)
        obs_count = len(t.observations)
        time_str = t.start_time.strftime("%H:%M:%S")
        tags_str = " ".join(f"[{tag}]" for tag in t.tags) if t.tags else ""

        print(
            f"  {status} {C.BOLD}{t.name:30s}{C.RESET} "
            f"{C.DIM}{t.id[:12]}{C.RESET}  "
            f"{time_str}  {dur:>8s}  "
            f"{obs_count:>2d} obs  "
            f"{tokens:>6s} tok  {cost:>8s}  "
            f"{C.DIM}{tags_str}{C.RESET}"
        )
    print()


def cmd_show(storage: Storage, args: argparse.Namespace) -> None:
    # Buscar por ID parcial
    trace_id = args.trace_id
    trace = storage.get_trace(trace_id)

    if not trace:
        # Intentar match parcial
        all_traces = storage.list_traces(limit=100)
        matches = [t for t in all_traces if t.id.startswith(trace_id)]
        if len(matches) == 1:
            trace = matches[0]
        elif len(matches) > 1:
            print(f"\n{C.YELLOW}Multiple matches:{C.RESET}")
            for t in matches:
                print(f"  {t.id}  {t.name}")
            return
        else:
            print(f"\n{C.RED}Trace not found: {trace_id}{C.RESET}\n")
            return

    # Header
    print(f"\n{C.BOLD}🔍 Trace: {trace.name}{C.RESET}")
    print(f"{'─' * 60}")
    print(f"  ID:          {trace.id}")
    print(f"  Time:        {trace.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration:    {_fmt_duration(trace.duration_ms)}")
    print(f"  Environment: {trace.environment}")
    if trace.session_id:
        print(f"  Session:     {trace.session_id}")
    if trace.user_id:
        print(f"  User:        {trace.user_id}")
    if trace.tags:
        print(f"  Tags:        {', '.join(trace.tags)}")
    print(f"  Tokens:      {_fmt_tokens(trace.total_tokens)}")
    print(f"  Cost:        {_fmt_cost(trace.total_cost)}")

    # Observation tree
    print(f"\n{C.BOLD}  Observations:{C.RESET}")

    # Build tree
    obs_map = {o.id: o for o in trace.observations}
    roots = [
        o
        for o in trace.observations
        if o.parent_id is None or o.parent_id not in obs_map
    ]

    def print_tree(obs_list: list, depth: int = 0) -> None:
        for obs in obs_list:
            indent = "  │ " * depth + "  ├─ " if depth > 0 else "  "
            color = TYPE_COLORS.get(obs.type.value, C.RESET)
            status_icon = f"{C.RED}✗{C.RESET}" if obs.status == "error" else " "

            dur = _fmt_duration(obs.duration_ms)
            model_str = f" ({obs.model})" if obs.model else ""
            tokens_str = ""
            if obs.usage:
                tokens_str = f" {_fmt_tokens(obs.usage.total_tokens)} tok"

            print(
                f"{indent}{status_icon} {color}{obs.type.value:12s}{C.RESET} "
                f"{obs.name}{model_str}"
                f"{C.DIM}{tokens_str}  {dur}{C.RESET}"
            )

            if obs.error_message:
                err_indent = "  │ " * (depth + 1) + "  "
                print(f"{err_indent}{C.RED}{obs.error_message[:200]}{C.RESET}")

            # Children
            children = [o for o in obs_map.values() if o.parent_id == obs.id]
            if children:
                print_tree(children, depth + 1)

    print_tree(roots)

    # Scores
    scores = storage.get_scores(trace_id=trace.id)
    if scores:
        print(f"\n{C.BOLD}  Scores:{C.RESET}")
        for s in scores:
            print(f"    {s.name}: {C.BOLD}{s.value}{C.RESET} ({s.source.value})")

    # Input/Output
    if trace.input:
        print(f"\n{C.BOLD}  Input:{C.RESET}")
        print(f"    {C.DIM}{str(trace.input)[:500]}{C.RESET}")
    if trace.output:
        print(f"\n{C.BOLD}  Output:{C.RESET}")
        print(f"    {C.DIM}{str(trace.output)[:500]}{C.RESET}")

    print()


def cmd_dashboard(storage: Storage, args: argparse.Namespace) -> None:
    from llm_trace.dashboard import run_dashboard

    run_dashboard(storage, port=args.port, open_browser=not args.no_browser)


def cmd_export(storage: Storage, args: argparse.Namespace) -> None:
    traces = storage.list_traces(limit=args.limit)
    data = []
    for t in traces:
        td = t.to_dict()
        td["scores"] = [s.to_dict() for s in storage.get_scores(trace_id=t.id)]
        data.append(td)

    output = json.dumps(data, indent=2, default=str, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Exported {len(data)} traces to {args.output}")
    else:
        print(output)


def cmd_clear(storage: Storage, args: argparse.Namespace) -> None:
    if not args.yes:
        confirm = input(f"Delete all data in {storage.db_path}? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    conn = storage._get_conn()
    conn.execute("DELETE FROM scores")
    conn.execute("DELETE FROM observations")
    conn.execute("DELETE FROM traces")
    conn.commit()
    print(f"✓ All data cleared from {storage.db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-trace",
        description="🔍 Observabilidad ligera para aplicaciones LLM",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to SQLite database",
    )

    sub = parser.add_subparsers(dest="command")

    # stats
    sub.add_parser("stats", help="Show global statistics")

    # list
    p_list = sub.add_parser("list", help="List recent traces")
    p_list.add_argument("--limit", "-n", type=int, default=20)
    p_list.add_argument("--name", help="Filter by trace name")
    p_list.add_argument("--env", help="Filter by environment")
    p_list.add_argument("--tag", help="Filter by tag")

    # show
    p_show = sub.add_parser("show", help="Show trace details")
    p_show.add_argument("trace_id", help="Trace ID (partial match supported)")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch web dashboard")
    p_dash.add_argument("--port", "-p", type=int, default=7600)
    p_dash.add_argument("--no-browser", action="store_true")

    # export
    p_export = sub.add_parser("export", help="Export traces to JSON")
    p_export.add_argument("--limit", "-n", type=int, default=100)
    p_export.add_argument("--output", "-o", help="Output file path")

    # clear
    p_clear = sub.add_parser("clear", help="Delete all trace data")
    p_clear.add_argument("--yes", "-y", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    storage = Storage(args.db)

    commands = {
        "stats": cmd_stats,
        "list": cmd_list,
        "show": cmd_show,
        "dashboard": cmd_dashboard,
        "export": cmd_export,
        "clear": cmd_clear,
    }
    commands[args.command](storage, args)


if __name__ == "__main__":
    main()
