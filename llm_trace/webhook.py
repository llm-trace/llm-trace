"""HTTP webhook receiver para llm-trace.

Expone un endpoint HTTP que acepta traces y observations via POST,
permitiendo instrumentar aplicaciones en cualquier lenguaje o
plataformas no-code (n8n, Flowise, Langflow, etc.).

Usage (como servidor standalone):
    from llm_trace.webhook import run_webhook_server
    run_webhook_server(port=7601)

Usage (como middleware FastAPI):
    from llm_trace.webhook import create_fastapi_router
    app.include_router(create_fastapi_router(), prefix="/traces")

Enviar datos desde cualquier lenguaje:
    curl -X POST http://localhost:7601/api/ingest \\
      -H "Content-Type: application/json" \\
      -d '{
        "trace": {"name": "my-trace", "session_id": "s1"},
        "observations": [
          {
            "type": "generation",
            "name": "llm-call",
            "model": "gpt-4o",
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "output": {"content": "Hello!"},
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "duration_ms": 450
          }
        ],
        "scores": [
          {"name": "quality", "value": 0.95}
        ]
      }'
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from llm_trace.core import tracer
from llm_trace.models import (
    CostDetails,
    Observation,
    ObservationType,
    Score,
    ScoreDataType,
    ScoreSource,
    Trace,
    UsageDetails,
    _new_id,
    _now,
)

logger = logging.getLogger("llm-trace.webhook")


def _parse_trace(data: dict[str, Any]) -> Trace:
    """Parsea un dict a un Trace."""
    return Trace(
        id=data.get("id", _new_id()),
        name=data.get("name", "webhook-trace"),
        session_id=data.get("session_id"),
        user_id=data.get("user_id"),
        input=data.get("input"),
        output=data.get("output"),
        metadata=data.get("metadata", {}),
        tags=data.get("tags", []),
        environment=data.get("environment", tracer.environment),
        release=data.get("release", tracer.release),
    )


def _parse_observation(data: dict[str, Any], trace_id: str) -> Observation:
    """Parsea un dict a una Observation."""
    obs_type_str = data.get("type", "span").lower()
    try:
        obs_type = ObservationType(obs_type_str)
    except ValueError:
        obs_type = ObservationType.SPAN

    start_time = _now()
    end_time = None
    duration_ms = data.get("duration_ms")
    if duration_ms:
        end_time = start_time + timedelta(milliseconds=0)  # Already finished
        start_time = end_time - timedelta(milliseconds=duration_ms)

    # Parse usage
    usage = None
    usage_data = data.get("usage")
    if usage_data:
        usage = UsageDetails(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
            or usage_data.get("input_tokens", 0)
            + usage_data.get("output_tokens", 0),
            cached_tokens=usage_data.get("cached_tokens", 0),
            reasoning_tokens=usage_data.get("reasoning_tokens", 0),
        )

    # Parse cost
    cost = None
    cost_data = data.get("cost")
    if cost_data:
        cost = CostDetails(
            input_cost=cost_data.get("input_cost", 0),
            output_cost=cost_data.get("output_cost", 0),
            total_cost=cost_data.get("total_cost", 0),
        )
    elif usage and data.get("model"):
        from llm_trace.wrappers import _calculate_cost

        cost = _calculate_cost(
            data["model"], usage.input_tokens, usage.output_tokens
        )
        if cost.total_cost == 0:
            cost = None

    return Observation(
        id=data.get("id", _new_id()),
        trace_id=trace_id,
        parent_id=data.get("parent_id"),
        type=obs_type,
        name=data.get("name", ""),
        start_time=start_time,
        end_time=end_time or _now(),
        completion_start_time=None,
        status=data.get("status", "ok"),
        level=data.get("level", "DEFAULT"),
        input=data.get("input"),
        output=data.get("output"),
        metadata=data.get("metadata", {}),
        model=data.get("model"),
        model_parameters=data.get("model_parameters"),
        usage=usage,
        cost=cost,
        error_message=data.get("error"),
    )


def _parse_score(data: dict[str, Any], trace_id: str) -> Score:
    """Parsea un dict a un Score."""
    value = data.get("value", 0)

    if isinstance(value, bool):
        data_type = ScoreDataType.BOOLEAN
    elif isinstance(value, str):
        data_type = ScoreDataType.CATEGORICAL
    else:
        data_type = ScoreDataType.NUMERIC

    source_str = data.get("source", "api")
    try:
        source = ScoreSource(source_str)
    except ValueError:
        source = ScoreSource.API

    return Score(
        id=data.get("id", _new_id()),
        trace_id=trace_id,
        observation_id=data.get("observation_id"),
        name=data.get("name", "score"),
        value=value,
        data_type=data_type,
        source=source,
        comment=data.get("comment"),
    )


def ingest(payload: dict[str, Any]) -> dict[str, Any]:
    """Procesa un payload de ingesta completo.

    Formato del payload:
    {
        "trace": { ... },           // Opcional: datos del trace
        "observations": [ ... ],    // Opcional: lista de observations
        "scores": [ ... ]           // Opcional: lista de scores
    }

    Returns:
        {"trace_id": "...", "observations": N, "scores": N}
    """
    trace_data = payload.get("trace", {})
    obs_list = payload.get("observations", [])
    score_list = payload.get("scores", [])

    # Crear trace
    trace = _parse_trace(trace_data)

    # Crear observations
    observations = []
    for obs_data in obs_list:
        obs = _parse_observation(obs_data, trace.id)
        observations.append(obs)
        trace.observations.append(obs)

    # Crear scores
    scores = []
    for score_data in score_list:
        s = _parse_score(score_data, trace.id)
        scores.append(s)

    # Finalizar trace
    if observations:
        trace.start_time = min(o.start_time for o in observations)
        ends = [o.end_time for o in observations if o.end_time]
        if ends:
            trace.end_time = max(ends)
    trace.end_time = trace.end_time or _now()

    # Enqueue todo
    tracer._enqueue(trace)
    for obs in observations:
        tracer._enqueue(obs)
    for s in scores:
        tracer._enqueue(s)

    return {
        "trace_id": trace.id,
        "observations": len(observations),
        "scores": len(scores),
    }


# ── HTTP Server ───────────────────────────────────────────


class WebhookHandler(BaseHTTPRequestHandler):
    """Handler HTTP para recibir datos de trazabilidad."""

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(format, *args)

    def _json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._json_response({})

    def do_POST(self) -> None:
        if self.path == "/api/ingest":
            self._handle_ingest()
        elif self.path == "/api/score":
            self._handle_score()
        else:
            self._json_response({"error": f"Unknown endpoint: {self.path}"}, 404)

    def _handle_ingest(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            result = ingest(body)
            self._json_response(result, 201)
        except json.JSONDecodeError:
            self._json_response({"error": "Invalid JSON"}, 400)
        except Exception as e:
            logger.error("Ingest error: %s", e)
            self._json_response({"error": str(e)}, 500)

    def _handle_score(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            trace_id = body.get("trace_id", "")
            s = _parse_score(body, trace_id)
            tracer._enqueue(s)
            self._json_response({"id": s.id}, 201)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)


def run_webhook_server(port: int = 7601) -> None:
    """Inicia el servidor webhook standalone."""
    server = HTTPServer(("0.0.0.0", port), WebhookHandler)
    print(f"\n  📡 llm-trace webhook → http://localhost:{port}/api/ingest\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Webhook server stopped.")
        server.shutdown()


# ── FastAPI Router ────────────────────────────────────────


def create_fastapi_router() -> Any:
    """Crea un router FastAPI para integrar en apps existentes.

    Usage:
        from llm_trace.webhook import create_fastapi_router
        app.include_router(create_fastapi_router(), prefix="/traces")
    """
    try:
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError("FastAPI required: pip install fastapi")

    router = APIRouter()

    @router.post("/ingest")
    async def ingest_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        result = ingest(body)
        return JSONResponse(result, status_code=201)

    @router.post("/score")
    async def score_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        trace_id = body.get("trace_id", "")
        s = _parse_score(body, trace_id)
        tracer._enqueue(s)
        return JSONResponse({"id": s.id}, status_code=201)

    return router
