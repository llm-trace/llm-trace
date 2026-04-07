"""Dashboard web integrado con servidor HTTP y API REST.

Servidor ligero basado en http.server — sin dependencias externas.
Sirve un dashboard SPA y expone una API JSON para datos.

Usage:
    from llm_trace import tracer
    tracer.dashboard(port=7600)
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from llm_trace.storage import Storage

logger = logging.getLogger("llm-trace.dashboard")

_storage: Storage | None = None


class DashboardHandler(BaseHTTPRequestHandler):
    """Handler HTTP que sirve API JSON y dashboard HTML."""

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(format, *args)

    def _json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html: str) -> None:
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/api/stats":
            self._handle_stats()
        elif path == "/api/traces":
            self._handle_list_traces(params)
        elif path.startswith("/api/traces/"):
            trace_id = path.split("/")[-1]
            self._handle_get_trace(trace_id)
        elif path == "/api/timeline":
            hours = int(params.get("hours", ["24"])[0])
            self._handle_timeline(hours)
        elif path == "/api/timeseries":
            self._handle_timeseries()
        elif path.startswith("/api/scores"):
            self._handle_scores(params)
        else:
            self._html_response(_DASHBOARD_HTML)

    def _handle_stats(self) -> None:
        assert _storage is not None
        stats = _storage.get_stats()
        self._json_response(stats)

    def _handle_list_traces(self, params: dict[str, list[str]]) -> None:
        assert _storage is not None
        limit = int(params.get("limit", ["50"])[0])
        offset = int(params.get("offset", ["0"])[0])
        name = params.get("name", [None])[0]
        env = params.get("environment", [None])[0]

        traces = _storage.list_traces(
            limit=limit, offset=offset, name=name, environment=env
        )
        self._json_response([t.to_dict() for t in traces])

    def _handle_get_trace(self, trace_id: str) -> None:
        assert _storage is not None
        trace = _storage.get_trace(trace_id)
        if trace:
            data = trace.to_dict()
            data["scores"] = [
                s.to_dict() for s in _storage.get_scores(trace_id=trace_id)
            ]
            self._json_response(data)
        else:
            self._json_response({"error": "Trace not found"}, 404)

    def _handle_timeline(self, hours: int) -> None:
        assert _storage is not None
        self._json_response(_storage.get_timeline(hours))

    def _handle_timeseries(self) -> None:
        assert _storage is not None
        self._json_response(_storage.get_timeseries())

    def _handle_scores(self, params: dict[str, list[str]]) -> None:
        assert _storage is not None
        trace_id = params.get("trace_id", [None])[0]
        scores = _storage.get_scores(trace_id=trace_id)
        self._json_response([s.to_dict() for s in scores])

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/traces":
            assert _storage is not None
            count = _storage.delete_all()
            self._json_response({"deleted": count})
        elif path.startswith("/api/traces/"):
            trace_id = path.split("/")[-1]
            assert _storage is not None
            ok = _storage.delete_trace(trace_id)
            if ok:
                self._json_response({"deleted": trace_id})
            else:
                self._json_response({"error": "Trace not found"}, 404)
        else:
            self._json_response({"error": "Not found"}, 404)


def run_dashboard(
    storage: Storage,
    port: int = 7600,
    open_browser: bool = True,
) -> None:
    """Inicia el servidor del dashboard."""
    global _storage
    _storage = storage

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    url = f"http://localhost:{port}"

    print(f"\n  🔍 llm-trace dashboard → {url}\n")

    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=[url]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.shutdown()


# ── Embedded Dashboard HTML ───────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llm-trace</title>
<style>
  :root {
    --bg: #0a0a0f; --surface: #12121a; --border: #1e1e2e;
    --text: #e2e2e8; --muted: #6e6e82; --accent: #6366f1;
    --green: #22c55e; --red: #ef4444; --amber: #f59e0b;
    --font: 'SF Mono', 'Cascadia Code', 'JetBrains Mono', monospace;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 13px; display: flex; flex-direction: column; height: 100vh; margin: 0; }

  .header {
    border-bottom: 1px solid var(--border); padding: 16px 24px;
    display: flex; align-items: center; gap: 12px; flex-shrink: 0;
  }
  .header h1 { font-size: 15px; font-weight: 600; letter-spacing: -0.5px; }
  .header .badge {
    background: var(--accent); color: white; padding: 2px 8px;
    border-radius: 4px; font-size: 10px; font-weight: 600;
  }
  .header .refresh {
    margin-left: auto; background: var(--surface); border: 1px solid var(--border);
    color: var(--muted); padding: 6px 12px; border-radius: 6px; cursor: pointer;
    font-family: var(--font); font-size: 11px;
  }
  .header .refresh:hover { color: var(--text); border-color: var(--accent); }

  .stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1px; background: var(--border); border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  .stat {
    background: var(--surface); padding: 16px 20px;
  }
  .stat .label { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .stat .value { font-size: 22px; font-weight: 700; letter-spacing: -1px; }
  .stat .value.green { color: var(--green); }
  .stat .value.red { color: var(--red); }

  .content { display: flex; flex: 1; min-height: 0; }

  .trace-list {
    width: 420px; border-right: 1px solid var(--border);
    overflow-y: auto; flex-shrink: 0;
  }
  .trace-item {
    padding: 12px 20px; border-bottom: 1px solid var(--border);
    cursor: pointer; transition: background 0.1s;
  }
  .trace-item:hover { background: var(--surface); }
  .trace-item.active { background: var(--surface); border-left: 2px solid var(--accent); }
  .trace-item .name { font-weight: 600; font-size: 12px; margin-bottom: 4px; }
  .trace-item .meta { color: var(--muted); font-size: 10px; display: flex; gap: 12px; }
  .trace-item .tag { background: var(--border); padding: 1px 6px; border-radius: 3px; font-size: 9px; }
  .trace-item .error-badge { color: var(--red); }
  .trace-item .trace-delete {
    float: right; color: var(--border); font-size: 11px; cursor: pointer;
    padding: 0 4px; border-radius: 3px; line-height: 1;
  }
  .trace-item .trace-delete:hover { color: var(--red); background: #ef444420; }

  .detail { flex: 1; overflow-y: auto; padding: 20px; min-height: 0; }
  .detail .empty { color: var(--muted); padding: 40px; text-align: center; }

  .detail h2 { font-size: 14px; margin-bottom: 16px; }

  .obs-tree { margin: 0; }
  .obs-node {
    border-left: 2px solid var(--border); margin-left: 12px;
    padding: 8px 0 8px 16px; position: relative;
  }
  .obs-node::before {
    content: ''; position: absolute; left: -2px; top: 16px;
    width: 12px; height: 2px; background: var(--border);
  }
  .obs-node.generation { border-left-color: var(--accent); }
  .obs-node.generation::before { background: var(--accent); }
  .obs-node.tool { border-left-color: var(--amber); }
  .obs-node.tool::before { background: var(--amber); }
  .obs-node.error { border-left-color: var(--red); }
  .obs-node.error::before { background: var(--red); }

  .obs-header {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 10px; background: var(--surface); border-radius: 6px;
    cursor: pointer;
  }
  .obs-header:hover { background: #1a1a28; }
  .obs-type {
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.5px;
    padding: 2px 6px; border-radius: 3px; font-weight: 600;
  }
  .obs-type.generation { background: #6366f130; color: var(--accent); }
  .obs-type.span { background: #22c55e20; color: var(--green); }
  .obs-type.tool { background: #f59e0b20; color: var(--amber); }
  .obs-type.event { background: #6e6e8230; color: var(--muted); }
  .obs-type.agent { background: #a855f720; color: #a855f7; }
  .obs-type.retriever { background: #06b6d420; color: #06b6d4; }
  .obs-type.embedding { background: #ec489920; color: #ec4899; }

  .obs-name { font-weight: 500; font-size: 12px; }
  .obs-duration { color: var(--muted); font-size: 10px; margin-left: auto; }
  .obs-tokens { color: var(--muted); font-size: 10px; }

  .obs-detail {
    margin-top: 8px; padding: 10px; background: var(--bg);
    border-radius: 6px; font-size: 11px; display: none;
  }
  .obs-detail.open { display: block; }
  .obs-detail pre {
    background: #080810; padding: 10px; border-radius: 4px;
    overflow-x: auto; max-height: 300px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-all;
    margin: 6px 0; color: #b0b0c0;
  }
  .obs-detail .field { margin-bottom: 8px; }
  .obs-detail .field-label { color: var(--muted); font-size: 10px; text-transform: uppercase; margin-bottom: 2px; }

  .scores-section { margin-top: 20px; }
  .score-item {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface); padding: 4px 10px; border-radius: 4px;
    margin: 2px; font-size: 11px;
  }
  .score-name { color: var(--muted); }
  .score-value { font-weight: 600; }

  .graph-container {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px; margin-bottom: 16px;
    overflow-x: auto;
  }
  .graph-container svg { max-width: 100%; height: auto; }
  .graph-container .node rect,
  .graph-container .node polygon,
  .graph-container .node circle { rx: 8; ry: 8; }
  .graph-tabs {
    display: flex; gap: 2px; margin-bottom: 12px;
  }
  .graph-tab {
    padding: 4px 12px; font-size: 11px; font-family: var(--font);
    background: none; border: 1px solid var(--border); color: var(--muted);
    border-radius: 4px; cursor: pointer;
  }
  .graph-tab.active { background: var(--accent); color: white; border-color: var(--accent); }

  .search {
    padding: 12px 20px; border-bottom: 1px solid var(--border);
  }
  .search input {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    color: var(--text); padding: 8px 12px; border-radius: 6px;
    font-family: var(--font); font-size: 12px; outline: none;
  }
  .search input:focus { border-color: var(--accent); }
  .search input::placeholder { color: var(--muted); }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div class="header">
  <h1>🔍 llm-trace</h1>
  <span class="badge">v0.2</span>
  <button class="refresh" onclick="loadAll()">↻ Refresh</button>
  <button class="refresh" onclick="clearAll()" style="color:#ef4444">✕ Clear all</button>
</div>

<div class="stats" id="stats"></div>

<div class="content">
  <div class="trace-list">
    <div class="search">
      <input type="text" id="searchInput" placeholder="Search traces..." oninput="filterTraces()">
    </div>
    <div id="traceList"></div>
  </div>
  <div class="detail" id="detail">
    <div class="empty">← Select a trace to view details</div>
  </div>
</div>

<script>
let allTraces = [];
let selectedTrace = null;
let globalStats = null;
let currentView = 'tree';

async function api(path) {
  const r = await fetch(path);
  return r.json();
}

function fmt(ms) {
  if (!ms) return '-';
  if (ms < 1) return '<1ms';
  if (ms < 1000) return Math.round(ms) + 'ms';
  return (ms / 1000).toFixed(2) + 's';
}

function fmtCost(c) {
  if (!c || c === 0) return '-';
  if (c < 0.001) return '$' + c.toFixed(6);
  if (c < 1) return '$' + c.toFixed(4);
  return '$' + c.toFixed(2);
}

function fmtTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleTimeString() + ' ' + d.toLocaleDateString();
}

function fmtTokens(n) {
  if (!n) return '-';
  if (n < 1000) return n.toString();
  return (n / 1000).toFixed(1) + 'k';
}

let _lastTraceCount = 0;

async function loadAll() {
  const [stats, traces] = await Promise.all([
    api('/api/stats'), api('/api/traces?limit=100')
  ]);
  if (stats.trace_count === _lastTraceCount && _lastTraceCount > 0 && !selectedTrace) return;
  _lastTraceCount = stats.trace_count;
  globalStats = stats;
  allTraces = traces;
  renderTraces(traces);

  if (!selectedTrace) {
    renderStats(stats);
    renderOverview();
  }
  // When a trace is selected, don't touch the detail pane — preserves open nodes
}

function renderStats(s, traceView) {
  const back = traceView
    ? '<div class="stat" style="cursor:pointer;text-align:center" onclick="clearSelection()"><div class="label">viewing trace</div><div class="value" style="font-size:13px;color:var(--accent)">← all traces</div></div>'
    : '<div class="stat"><div class="label">Traces</div><div class="value">' + s.trace_count + '</div></div>';
  document.getElementById('stats').innerHTML = `
    ${back}
    <div class="stat"><div class="label">${traceView ? 'Observations' : 'Generations'}</div><div class="value">${traceView ? s.obs_count : s.generation_count}</div></div>
    <div class="stat"><div class="label">Tokens</div><div class="value">${fmtTokens(s.total_tokens)}</div></div>
    <div class="stat"><div class="label">Cost</div><div class="value green">${fmtCost(s.total_cost)}</div></div>
    <div class="stat"><div class="label">Latency</div><div class="value">${fmt(s.latency_ms || s.avg_latency_ms)}</div></div>
    <div class="stat"><div class="label">${traceView ? 'Status' : 'Error Rate'}</div><div class="value ${traceView ? (s.has_error ? 'red' : 'green') : (s.error_rate > 5 ? 'red' : '')}">${traceView ? (s.has_error ? 'error' : 'ok') : s.error_rate + '%'}</div></div>
  `;
}

function clearSelection() {
  selectedTrace = null;
  if (globalStats) renderStats(globalStats);
  renderTraces(allTraces);
  renderOverview();
}

function renderOverview() {
  const el = document.getElementById('detail');
  if (!allTraces.length) { el.innerHTML = '<div class="empty">No traces yet</div>'; return; }

  const byName = {};
  allTraces.forEach(t => {
    const n = t.name || 'unnamed';
    if (!byName[n]) byName[n] = {count: 0, totalMs: 0, errors: 0, totalTokens: 0, totalCost: 0};
    byName[n].count++;
    byName[n].totalMs += t.duration_ms || 0;
    byName[n].errors += (t.observations || []).some(o => o.status === 'error') ? 1 : 0;
    byName[n].totalTokens += t.total_tokens || 0;
    byName[n].totalCost += t.total_cost || 0;
  });

  let tableRows = Object.entries(byName)
    .sort((a, b) => b[1].count - a[1].count)
    .map(([name, d]) => {
      const avgMs = d.count ? Math.round(d.totalMs / d.count) : 0;
      const errPct = d.count ? Math.round(d.errors / d.count * 100) : 0;
      return '<tr style="border-bottom:1px solid var(--border)">'
        + '<td style="padding:5px 8px;font-weight:600;color:var(--text)">' + esc(name) + '</td>'
        + '<td style="padding:5px 8px">' + d.count + '</td>'
        + '<td style="padding:5px 8px">' + fmt(avgMs) + '</td>'
        + '<td style="padding:5px 8px">' + fmtTokens(d.totalTokens) + '</td>'
        + '<td style="padding:5px 8px">' + fmtCost(d.totalCost) + '</td>'
        + '<td style="padding:5px 8px;color:' + (errPct > 0 ? 'var(--red)' : 'var(--muted)') + '">' + errPct + '%</td>'
        + '</tr>';
    }).join('');

  const models = (globalStats && globalStats.top_models) || [];
  let modelRows = models.map(m =>
    '<tr style="border-bottom:1px solid var(--border)">'
    + '<td style="padding:5px 8px;font-weight:600;color:var(--text)">' + esc(m.model) + '</td>'
    + '<td style="padding:5px 8px">' + m.count + '</td>'
    + '<td style="padding:5px 8px">' + fmtTokens(m.tokens) + '</td>'
    + '<td style="padding:5px 8px">' + fmtCost(m.cost) + '</td></tr>'
  ).join('');

  el.innerHTML = '<h2>Overview</h2>'
    + '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:20px">'
    + '<div><div style="color:var(--muted);font-size:10px;margin-bottom:4px">TOKENS OVER TIME <span class="chart-gran" style="color:var(--accent)"></span></div><div id="chart-tokens"></div></div>'
    + '<div><div style="color:var(--muted);font-size:10px;margin-bottom:4px">COST OVER TIME <span class="chart-gran" style="color:var(--accent)"></span></div><div id="chart-cost"></div></div>'
    + '<div><div style="color:var(--muted);font-size:10px;margin-bottom:4px">LATENCY OVER TIME <span class="chart-gran" style="color:var(--accent)"></span></div><div id="chart-latency"></div></div>'
    + '</div>'
    + '<table style="width:100%;border-collapse:collapse;font-size:11px;margin-bottom:20px">'
    + '<tr style="color:var(--muted);text-align:left;border-bottom:1px solid var(--border)">'
    + '<th style="padding:6px 8px">Trace</th><th style="padding:6px 8px">Runs</th>'
    + '<th style="padding:6px 8px">Avg latency</th><th style="padding:6px 8px">Tokens</th>'
    + '<th style="padding:6px 8px">Cost</th><th style="padding:6px 8px">Errors</th></tr>'
    + tableRows + '</table>'
    + (modelRows ? '<h2>Models</h2>'
    + '<table style="width:100%;border-collapse:collapse;font-size:11px">'
    + '<tr style="color:var(--muted);text-align:left;border-bottom:1px solid var(--border)">'
    + '<th style="padding:6px 8px">Model</th><th style="padding:6px 8px">Calls</th>'
    + '<th style="padding:6px 8px">Tokens</th><th style="padding:6px 8px">Cost</th></tr>'
    + modelRows + '</table>' : '');

  api('/api/timeseries').then(res => {
    const data = res.data || [];
    const gran = res.granularity || 'hour';
    if (data.length > 0) {
      const suffix = {minute:'min',hour:'hour',day:'day',month:'month',year:'year'}[gran] || gran;
      document.querySelectorAll('.chart-gran').forEach(el => el.textContent = 'by ' + suffix);
      renderChart('chart-tokens', data, 'tokens', '#6366f1', gran);
      renderChart('chart-cost', data, 'cost', '#22c55e', gran);
      renderChart('chart-latency', data, 'avg_latency_ms', '#f59e0b', gran);
    }
  });
}

function renderChart(containerId, data, key, color, gran) {
  const el = document.getElementById(containerId);
  if (!el || data.length === 0) return;

  const W = 300, H = 120, padL = 40, padR = 8, padT = 10, padB = 22;
  const chartW = W - padL - padR, chartH = H - padT - padB;
  const values = data.map(d => d[key] || 0);
  const maxVal = Math.max(...values, 0.001);

  function fmtLabel(bucket, gran, spansDays) {
    if (gran === 'minute') return bucket.split('T')[1] || bucket;
    if (gran === 'hour') {
      const parts = bucket.split('T');
      const time = parts[1] ? parts[1].substring(0,5) : '';
      if (spansDays && parts[0]) {
        const d = parts[0].split('-');
        return d[2] + '/' + d[1] + ' ' + time;
      }
      return time;
    }
    if (gran === 'day') { const p = bucket.split('-'); return p[2] + '/' + p[1]; }
    if (gran === 'month') { const p = bucket.split('-'); return p[1] + '/' + p[0].substring(2); }
    return bucket;
  }

  // Detect if data spans multiple days
  const first = data[0].bucket || '';
  const last = data[data.length - 1].bucket || '';
  const spansDays = first.substring(0, 10) !== last.substring(0, 10);
  const labels = data.map(d => fmtLabel(d.bucket, gran, spansDays));

  const points = values.map((v, i) => {
    const x = padL + (i / Math.max(values.length - 1, 1)) * chartW;
    const y = padT + chartH - (v / maxVal) * chartH;
    return x + ',' + y;
  });

  const areaPoints = padL + ',' + (padT + chartH) + ' ' + points.join(' ') + ' ' + (padL + chartW) + ',' + (padT + chartH);
  const lastVal = values[values.length - 1];
  const fmtVal = key === 'cost' ? fmtCost(lastVal) : key === 'avg_latency_ms' ? fmt(lastVal) : fmtTokens(lastVal);

  let svg = '<svg width="100%" viewBox="0 0 ' + W + ' ' + H + '" style="display:block;background:var(--surface);border-radius:6px;border:1px solid var(--border)">';
  svg += '<polygon points="' + areaPoints + '" fill="' + color + '" opacity="0.1"/>';
  svg += '<polyline points="' + points.join(' ') + '" fill="none" stroke="' + color + '" stroke-width="1.5" stroke-linejoin="round"/>';

  // Y axis labels
  svg += '<text x="' + (padL - 4) + '" y="' + (padT + 4) + '" text-anchor="end" fill="#6e6e82" font-size="8" font-family="monospace">' + (key === 'cost' ? fmtCost(maxVal) : key === 'avg_latency_ms' ? fmt(maxVal) : fmtTokens(maxVal)) + '</text>';
  svg += '<text x="' + (padL - 4) + '" y="' + (padT + chartH) + '" text-anchor="end" fill="#6e6e82" font-size="8" font-family="monospace">0</text>';

  // X axis labels (first and last)
  if (labels.length > 0) {
    svg += '<text x="' + padL + '" y="' + (H - 4) + '" fill="#6e6e82" font-size="8" font-family="monospace">' + labels[0] + '</text>';
    svg += '<text x="' + (padL + chartW) + '" y="' + (H - 4) + '" text-anchor="end" fill="#6e6e82" font-size="8" font-family="monospace">' + labels[labels.length - 1] + '</text>';
  }

  // Current value dot
  if (values.length > 0) {
    const lx = padL + chartW;
    const ly = padT + chartH - (lastVal / maxVal) * chartH;
    svg += '<circle cx="' + lx + '" cy="' + ly + '" r="3" fill="' + color + '"/>';
  }

  svg += '</svg>';
  el.innerHTML = svg;
}

function renderTraces(traces) {
  const el = document.getElementById('traceList');
  if (!traces.length) { el.innerHTML = '<div class="empty">No traces yet</div>'; return; }
  el.innerHTML = traces.map(t => {
    const hasError = t.observations?.some(o => o.status === 'error');
    return `<div class="trace-item ${selectedTrace === t.id ? 'active' : ''}" onclick="showTrace('${t.id}')">
      <div class="name">${t.name || 'unnamed'}${hasError ? ' <span class="error-badge">✗</span>' : ''}<span class="trace-delete" onclick="event.stopPropagation();deleteTrace('${t.id}')" title="Delete trace">✕</span></div>
      <div class="meta">
        <span>${fmtTime(t.start_time)}</span>
        <span>${fmt(t.duration_ms)}</span>
        <span>${fmtTokens(t.total_tokens)} tok</span>
        <span>${fmtCost(t.total_cost)}</span>
      </div>
      ${t.tags?.length ? '<div class="meta">' + t.tags.map(tag => '<span class="tag">' + tag + '</span>').join('') + '</div>' : ''}
    </div>`;
  }).join('');
}

async function showTrace(id) {
  selectedTrace = id;
  const t = await api('/api/traces/' + id);
  renderTraces(allTraces);

  const obs = t.observations || [];
  const totalTokens = obs.reduce((s, o) => s + (o.usage?.total_tokens || 0), 0);
  const totalCost = obs.reduce((s, o) => s + (o.cost?.total_cost || 0), 0);
  const hasError = obs.some(o => o.status === 'error');

  renderStats({
    obs_count: obs.length,
    total_tokens: totalTokens,
    total_cost: totalCost,
    latency_ms: t.duration_ms,
    has_error: hasError,
  }, true);

  const detail = document.getElementById('detail');
  const tree = buildTree(obs);

  detail.innerHTML = `
    <h2>${t.name || 'unnamed'} <span style="color:var(--muted);font-size:11px;font-weight:400">${t.id}</span></h2>
    <div style="color:var(--muted);font-size:11px;margin-bottom:16px;">
      ${fmtTime(t.start_time)} · ${fmt(t.duration_ms)} · ${t.environment}
      ${t.user_id ? ' · user:' + t.user_id : ''}
      ${t.session_id ? ' · session:' + t.session_id : ''}
    </div>
    <div class="graph-tabs">
      <button class="graph-tab${currentView==='tree'?' active':''}" onclick="showView('tree',this)">Tree</button>
      <button class="graph-tab${currentView==='graph'?' active':''}" onclick="showView('graph',this)">Graph</button>
    </div>
    <div id="view-tree" class="obs-tree" style="display:${currentView==='tree'?'block':'none'}">${renderTree(tree)}</div>
    <div id="view-graph" class="graph-container" style="display:${currentView==='graph'?'block':'none'}"></div>
    ${t.scores?.length ? '<div class="scores-section"><h2>Scores</h2>' +
      t.scores.map(s => '<div class="score-item"><span class="score-name">' + s.name +
        '</span><span class="score-value">' + s.value + '</span></div>').join('') + '</div>' : ''}
  `;
  window._currentObs = obs;
  if (currentView === 'graph') renderGraph('view-graph', obs);
}

function buildTree(observations) {
  const map = {}; const roots = [];
  observations.forEach(o => { map[o.id] = {...o, children: []}; });
  observations.forEach(o => {
    if (o.parent_id && map[o.parent_id]) map[o.parent_id].children.push(map[o.id]);
    else roots.push(map[o.id]);
  });
  return roots;
}

function renderTree(nodes) {
  return nodes.map(n => {
    const cls = [n.type, n.status === 'error' ? 'error' : ''].filter(Boolean).join(' ');
    const tokStr = n.usage ? fmtTokens(n.usage.total_tokens) + ' tok' : '';
    const costStr = n.cost ? fmtCost(n.cost.total_cost) : '';
    const uid = 'obs-' + n.id;
    return `
      <div class="obs-node ${cls}">
        <div class="obs-header" onclick="toggle('${uid}')">
          <span class="obs-type ${n.type}">${n.type}</span>
          <span class="obs-name">${n.name}</span>
          ${n.model ? '<span style="color:var(--muted);font-size:10px">' + n.model + '</span>' : ''}
          <span class="obs-tokens">${tokStr} ${costStr}</span>
          <span class="obs-duration">${fmt(n.duration_ms)}${n.ttft_ms ? ' (TTFT ' + fmt(n.ttft_ms) + ')' : ''}</span>
        </div>
        <div class="obs-detail" id="${uid}">
          ${n.input ? '<div class="field"><div class="field-label">Input</div><pre>' + esc(JSON.stringify(n.input, null, 2)) + '</pre></div>' : ''}
          ${n.output ? '<div class="field"><div class="field-label">Output</div><pre>' + esc(JSON.stringify(n.output, null, 2)) + '</pre></div>' : ''}
          ${n.error_message ? '<div class="field"><div class="field-label" style="color:var(--red)">Error</div><pre style="color:var(--red)">' + esc(n.error_message) + '</pre></div>' : ''}
          ${n.metadata && Object.keys(n.metadata).length ? '<div class="field"><div class="field-label">Metadata</div><pre>' + esc(JSON.stringify(n.metadata, null, 2)) + '</pre></div>' : ''}
        </div>
        ${n.children?.length ? renderTree(n.children) : ''}
      </div>`;
  }).join('');
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function toggle(id) { document.getElementById(id)?.classList.toggle('open'); }

function showView(view, btn) {
  currentView = view;
  document.getElementById('view-graph').style.display = view === 'graph' ? 'block' : 'none';
  document.getElementById('view-tree').style.display = view === 'tree' ? 'block' : 'none';
  document.querySelectorAll('.graph-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  if (view === 'graph' && window._currentObs) renderGraph('view-graph', window._currentObs);
}

function filterTraces() {
  const q = document.getElementById('searchInput').value.toLowerCase();
  const filtered = allTraces.filter(t =>
    (t.name || '').toLowerCase().includes(q) ||
    (t.id || '').includes(q) ||
    (t.tags || []).some(tag => tag.toLowerCase().includes(q))
  );
  renderTraces(filtered);
}

function showNodeDetail(idx) {
  const n = window._graphNodes[idx];
  if (!n) return;
  const el = document.getElementById('graph-node-detail');
  if (!el) return;
  const tc = TYPE_COLORS[n.type] || TYPE_COLORS.span;
  const dur = n.duration_ms ? fmt(n.duration_ms) : '-';
  let html = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">';
  html += '<span class="obs-type ' + n.type + '">' + n.type + '</span>';
  html += '<strong style="color:#e2e2e8">' + esc(n.name) + '</strong>';
  if (n.model) html += '<span style="color:var(--muted)">' + n.model + '</span>';
  html += '<span style="color:var(--muted);margin-left:auto">' + dur + '</span>';
  html += '</div>';
  if (n.input) html += '<div class="field"><div class="field-label">Input</div><pre>' + esc(JSON.stringify(n.input, null, 2)) + '</pre></div>';
  if (n.output) html += '<div class="field"><div class="field-label">Output</div><pre>' + esc(JSON.stringify(n.output, null, 2)) + '</pre></div>';
  if (n.error_message) html += '<div class="field"><div class="field-label" style="color:var(--red)">Error</div><pre style="color:var(--red)">' + esc(n.error_message) + '</pre></div>';
  if (n.usage) html += '<div class="field"><div class="field-label">Usage</div><pre>' + esc(JSON.stringify(n.usage, null, 2)) + '</pre></div>';
  if (n.metadata && Object.keys(n.metadata).length) html += '<div class="field"><div class="field-label">Metadata</div><pre>' + esc(JSON.stringify(n.metadata, null, 2)) + '</pre></div>';
  el.style.display = 'block';
  el.innerHTML = html;
}

async function deleteTrace(id) {
  await fetch('/api/traces/' + id, {method: 'DELETE'});
  if (selectedTrace === id) clearSelection();
  loadAll();
}

async function clearAll() {
  if (!confirm('Delete all traces?')) return;
  await fetch('/api/traces', {method: 'DELETE'});
  clearSelection();
  loadAll();
}

loadAll();
setInterval(loadAll, 5000);

const TYPE_COLORS = {
  generation: {fill:'#6366f130', stroke:'#6366f1', label:'GEN'},
  tool: {fill:'#f59e0b20', stroke:'#f59e0b', label:'TOOL'},
  agent: {fill:'#a855f720', stroke:'#a855f7', label:'AGENT'},
  retriever: {fill:'#06b6d420', stroke:'#06b6d4', label:'RET'},
  guardrail: {fill:'#ef444420', stroke:'#ef4444', label:'GUARD'},
  embedding: {fill:'#ec489920', stroke:'#ec4899', label:'EMB'},
  span: {fill:'#22c55e15', stroke:'#22c55e', label:'SPAN'},
  event: {fill:'#6e6e8220', stroke:'#6e6e82', label:'EVT'},
};

function renderGraph(containerId, observations) {
  const el = document.getElementById(containerId);
  if (!el) return;

  const nodes = observations
    .sort((a, b) => (a.start_time || '').localeCompare(b.start_time || ''));

  if (nodes.length === 0) { el.innerHTML = '<span style="color:var(--muted);font-size:11px">no graph data</span>'; return; }

  const pad = 16, nodeH = 30, gapY = 14, nodeW = 220;
  const totalW = nodeW + pad * 2;
  const totalH = nodes.length * (nodeH + gapY) - gapY + pad * 2;

  let svg = '<svg width="' + totalW + '" height="' + totalH + '" viewBox="0 0 ' + totalW + ' ' + totalH + '" xmlns="http://www.w3.org/2000/svg" style="font-family:monospace;font-size:10px;display:block;cursor:default">';
  svg += '<defs><marker id="ah" viewBox="0 0 10 10" refX="5" refY="9" markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1L5 9L9 1" fill="none" stroke="#4a4a60" stroke-width="1.5" stroke-linecap="round"/></marker></defs>';

  // Store node data for click handler
  window._graphNodes = nodes;

  nodes.forEach((n, i) => {
    const x = pad;
    const y = pad + i * (nodeH + gapY);
    const tc = TYPE_COLORS[n.type] || TYPE_COLORS.span;
    const isErr = n.status === 'error';
    const isChain = n.name === 'chain' || n.name === 'RunnableSequence';
    const fill = isErr ? '#ef444420' : isChain ? '#1a1a28' : tc.fill;
    const stroke = isErr ? '#ef4444' : isChain ? '#3a3a50' : tc.stroke;
    const label = n.name.length > 26 ? n.name.substring(0, 25) + '..' : n.name;
    const dur = n.duration_ms ? fmt(n.duration_ms) : '';

    const rx = n.type === 'agent' ? 15 : 4;
    svg += '<g style="cursor:pointer" onclick="showNodeDetail(' + i + ')">';
    svg += '<rect x="' + x + '" y="' + y + '" width="' + nodeW + '" height="' + nodeH + '" rx="' + rx + '" fill="' + fill + '" stroke="' + stroke + '" stroke-width="1"/>';
    svg += '<text x="' + (x + 8) + '" y="' + (y + nodeH/2 + 1) + '" dominant-baseline="central" fill="#c8c8d0" font-size="10">' + esc(label) + '</text>';
    svg += '<text x="' + (x + nodeW - 6) + '" y="' + (y + nodeH/2 + 1) + '" text-anchor="end" dominant-baseline="central" fill="' + stroke + '" font-size="8" font-weight="600">' + (isChain ? '' : tc.label + ' ') + dur + '</text>';
    svg += '</g>';

    if (i < nodes.length - 1) {
      const ax = x + nodeW / 2;
      svg += '<line x1="' + ax + '" y1="' + (y + nodeH) + '" x2="' + ax + '" y2="' + (pad + (i+1) * (nodeH + gapY) - 1) + '" stroke="#3a3a50" stroke-width="0.5" marker-end="url(#ah)"/>';
    }
  });

  svg += '</svg>';
  svg += '<div id="graph-node-detail" style="display:none;margin-top:8px;padding:10px;background:var(--bg);border:1px solid var(--border);border-radius:6px;font-size:11px"></div>';
  el.innerHTML = svg;
}
</script>
</body>
</html>"""
