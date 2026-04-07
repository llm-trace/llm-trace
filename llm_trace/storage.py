"""Storage backend basado en SQLite con WAL mode.

Zero configuración. Un solo archivo .db.
WAL permite lecturas concurrentes mientras se escriben datos.
Los datos se comprimen con JSON para flexibilidad de schema.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from llm_trace.models import (
    CostDetails,
    Observation,
    ObservationType,
    Score,
    ScoreDataType,
    ScoreSource,
    Trace,
    UsageDetails,
)

_DEFAULT_DB_PATH = Path.home() / ".llm-trace" / "traces.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    user_id TEXT,
    name TEXT NOT NULL DEFAULT '',
    start_time TEXT NOT NULL,
    end_time TEXT,
    input TEXT,
    output TEXT,
    metadata TEXT DEFAULT '{}',
    tags TEXT DEFAULT '[]',
    environment TEXT DEFAULT 'development',
    release TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_id TEXT,
    type TEXT NOT NULL DEFAULT 'span',
    name TEXT NOT NULL DEFAULT '',
    start_time TEXT NOT NULL,
    end_time TEXT,
    completion_start_time TEXT,
    status TEXT DEFAULT 'ok',
    level TEXT DEFAULT 'DEFAULT',
    input TEXT,
    output TEXT,
    metadata TEXT DEFAULT '{}',
    model TEXT,
    model_parameters TEXT,
    usage TEXT,
    cost TEXT,
    error_message TEXT,
    FOREIGN KEY (trace_id) REFERENCES traces(id)
);

CREATE TABLE IF NOT EXISTS scores (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    observation_id TEXT,
    name TEXT NOT NULL,
    value TEXT NOT NULL,
    data_type TEXT NOT NULL DEFAULT 'numeric',
    source TEXT NOT NULL DEFAULT 'api',
    comment TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (trace_id) REFERENCES traces(id)
);

CREATE INDEX IF NOT EXISTS idx_obs_trace ON observations(trace_id);
CREATE INDEX IF NOT EXISTS idx_obs_parent ON observations(parent_id);
CREATE INDEX IF NOT EXISTS idx_scores_trace ON scores(trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_start ON traces(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_traces_name ON traces(name);
CREATE INDEX IF NOT EXISTS idx_traces_env ON traces(environment);
"""


def _serialize(obj: Any) -> str | None:
    """Serializa a JSON, manejando None y objetos complejos."""
    if obj is None:
        return None
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def _deserialize(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


class Storage:
    """SQLite storage con WAL mode y connection pooling por thread."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ── Traces ──────────────────────────────────────────────

    def save_trace(self, trace: Trace) -> None:
        with self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO traces
                (id, session_id, user_id, name, start_time, end_time,
                 input, output, metadata, tags, environment, release)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace.id,
                    trace.session_id,
                    trace.user_id,
                    trace.name,
                    trace.start_time.isoformat(),
                    trace.end_time.isoformat() if trace.end_time else None,
                    _serialize(trace.input),
                    _serialize(trace.output),
                    _serialize(trace.metadata),
                    _serialize(trace.tags),
                    trace.environment,
                    trace.release,
                ),
            )

    def get_trace(self, trace_id: str) -> Trace | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM traces WHERE id = ?", (trace_id,)
        ).fetchone()
        if not row:
            return None
        trace = self._row_to_trace(row)
        trace.observations = self.get_observations(trace_id)
        return trace

    def list_traces(
        self,
        limit: int = 50,
        offset: int = 0,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        environment: str | None = None,
        tag: str | None = None,
    ) -> list[Trace]:
        conn = self._get_conn()
        query = "SELECT * FROM traces WHERE 1=1"
        params: list[Any] = []

        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if environment:
            query += " AND environment = ?"
            params.append(environment)
        if tag:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')

        query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        traces = []
        for row in rows:
            trace = self._row_to_trace(row)
            trace.observations = self.get_observations(trace.id)
            traces.append(trace)
        return traces

    def count_traces(self, **filters: Any) -> int:
        conn = self._get_conn()
        query = "SELECT COUNT(*) FROM traces WHERE 1=1"
        params: list[Any] = []
        for key, value in filters.items():
            if value is not None:
                query += f" AND {key} = ?"
                params.append(value)
        row = conn.execute(query, params).fetchone()
        return row[0] if row else 0

    def _row_to_trace(self, row: sqlite3.Row) -> Trace:
        start = datetime.fromisoformat(row["start_time"])
        end = (
            datetime.fromisoformat(row["end_time"])
            if row["end_time"]
            else None
        )
        return Trace(
            id=row["id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            name=row["name"],
            start_time=start,
            end_time=end,
            input=_deserialize(row["input"]),
            output=_deserialize(row["output"]),
            metadata=_deserialize(row["metadata"]) or {},
            tags=_deserialize(row["tags"]) or [],
            environment=row["environment"] or "development",
            release=row["release"],
        )

    # ── Observations ───────────────────────────────────────

    def save_observation(self, obs: Observation) -> None:
        with self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO observations
                (id, trace_id, parent_id, type, name, start_time, end_time,
                 completion_start_time, status, level, input, output,
                 metadata, model, model_parameters, usage, cost, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    obs.id,
                    obs.trace_id,
                    obs.parent_id,
                    obs.type.value,
                    obs.name,
                    obs.start_time.isoformat(),
                    obs.end_time.isoformat() if obs.end_time else None,
                    (
                        obs.completion_start_time.isoformat()
                        if obs.completion_start_time
                        else None
                    ),
                    obs.status,
                    obs.level,
                    _serialize(obs.input),
                    _serialize(obs.output),
                    _serialize(obs.metadata),
                    obs.model,
                    _serialize(obs.model_parameters),
                    _serialize(obs.usage.to_dict() if obs.usage else None),
                    _serialize(obs.cost.to_dict() if obs.cost else None),
                    obs.error_message,
                ),
            )

    def get_observations(self, trace_id: str) -> list[Observation]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM observations WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        ).fetchall()
        return [self._row_to_observation(r) for r in rows]

    def _row_to_observation(self, row: sqlite3.Row) -> Observation:
        usage_raw = _deserialize(row["usage"])
        cost_raw = _deserialize(row["cost"])
        start = datetime.fromisoformat(row["start_time"])
        end = (
            datetime.fromisoformat(row["end_time"])
            if row["end_time"]
            else None
        )
        cst = (
            datetime.fromisoformat(row["completion_start_time"])
            if row["completion_start_time"]
            else None
        )
        return Observation(
            id=row["id"],
            trace_id=row["trace_id"],
            parent_id=row["parent_id"],
            type=ObservationType(row["type"]),
            name=row["name"],
            start_time=start,
            end_time=end,
            completion_start_time=cst,
            status=row["status"],
            level=row["level"],
            input=_deserialize(row["input"]),
            output=_deserialize(row["output"]),
            metadata=_deserialize(row["metadata"]) or {},
            model=row["model"],
            model_parameters=_deserialize(row["model_parameters"]),
            usage=UsageDetails(**usage_raw) if usage_raw else None,
            cost=CostDetails(**cost_raw) if cost_raw else None,
            error_message=row["error_message"],
        )

    # ── Scores ─────────────────────────────────────────────

    def save_score(self, score: Score) -> None:
        with self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO scores
                (id, trace_id, observation_id, name, value, data_type,
                 source, comment, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    score.id,
                    score.trace_id,
                    score.observation_id,
                    score.name,
                    json.dumps(score.value),
                    score.data_type.value,
                    score.source.value,
                    score.comment,
                    score.created_at.isoformat(),
                ),
            )

    def get_scores(
        self, trace_id: str | None = None, observation_id: str | None = None
    ) -> list[Score]:
        conn = self._get_conn()
        query = "SELECT * FROM scores WHERE 1=1"
        params: list[Any] = []
        if trace_id:
            query += " AND trace_id = ?"
            params.append(trace_id)
        if observation_id:
            query += " AND observation_id = ?"
            params.append(observation_id)
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_score(r) for r in rows]

    def _row_to_score(self, row: sqlite3.Row) -> Score:
        return Score(
            id=row["id"],
            trace_id=row["trace_id"],
            observation_id=row["observation_id"],
            name=row["name"],
            value=json.loads(row["value"]),
            data_type=ScoreDataType(row["data_type"]),
            source=ScoreSource(row["source"]),
            comment=row["comment"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ── Analytics ──────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Estadísticas globales para el dashboard."""
        conn = self._get_conn()
        trace_count = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        obs_count = conn.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]
        gen_count = conn.execute(
            "SELECT COUNT(*) FROM observations WHERE type = 'generation'"
        ).fetchone()[0]
        score_count = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]

        # Total tokens y costos
        token_row = conn.execute(
            """SELECT
                COALESCE(SUM(json_extract(usage, '$.total_tokens')), 0) as total_tokens,
                COALESCE(SUM(json_extract(cost, '$.total_cost')), 0) as total_cost
            FROM observations WHERE usage IS NOT NULL"""
        ).fetchone()

        # Latencia promedio de traces
        latency_row = conn.execute(
            """SELECT AVG(
                (julianday(end_time) - julianday(start_time)) * 86400000
            ) as avg_latency_ms
            FROM traces
            WHERE end_time IS NOT NULL"""
        ).fetchone()

        # Tasa de error
        error_row = conn.execute(
            """SELECT
                COUNT(CASE WHEN status = 'error' THEN 1 END) as errors,
                COUNT(*) as total
            FROM observations"""
        ).fetchone()

        # Top modelos
        models = conn.execute(
            """SELECT model, COUNT(*) as count,
                COALESCE(SUM(json_extract(usage, '$.total_tokens')), 0) as tokens,
                COALESCE(SUM(json_extract(cost, '$.total_cost')), 0) as cost
            FROM observations
            WHERE model IS NOT NULL
            GROUP BY model ORDER BY count DESC LIMIT 10"""
        ).fetchall()

        return {
            "trace_count": trace_count,
            "observation_count": obs_count,
            "generation_count": gen_count,
            "score_count": score_count,
            "total_tokens": token_row["total_tokens"] if token_row else 0,
            "total_cost": round(token_row["total_cost"], 6) if token_row else 0,
            "avg_latency_ms": (
                round(latency_row["avg_latency_ms"], 1)
                if latency_row and latency_row["avg_latency_ms"]
                else 0
            ),
            "error_rate": (
                round(error_row["errors"] / max(error_row["total"], 1) * 100, 1)
                if error_row
                else 0
            ),
            "top_models": [
                {
                    "model": m["model"],
                    "count": m["count"],
                    "tokens": m["tokens"],
                    "cost": round(m["cost"], 6),
                }
                for m in models
            ],
        }

    def get_timeline(self, hours: int = 24) -> list[dict[str, Any]]:
        """Datos de timeline para el dashboard."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT
                strftime('%Y-%m-%dT%H:00:00', start_time) as hour,
                COUNT(*) as traces,
                COUNT(DISTINCT session_id) as sessions
            FROM traces
            WHERE start_time >= datetime('now', ?)
            GROUP BY hour ORDER BY hour""",
            (f"-{hours} hours",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_timeseries(self) -> dict[str, Any]:
        """Datos de series temporales con granularidad automática.

        Adapta el agrupamiento según el rango de datos:
        - < 2 horas → por minuto
        - < 3 días → por hora
        - < 90 días → por día
        - < 2 años → por mes
        - >= 2 años → por año
        """
        conn = self._get_conn()

        # Determinar rango de datos
        range_row = conn.execute(
            """SELECT
                MIN(start_time) as first,
                MAX(start_time) as last
            FROM traces"""
        ).fetchone()

        if not range_row or not range_row["first"] or not range_row["last"]:
            return {"granularity": "hour", "data": []}

        first = range_row["first"]
        last = range_row["last"]

        # Calcular diferencia y días distintos
        diff_row = conn.execute(
            """SELECT
                (julianday(?) - julianday(?)) * 86400 as diff_secs,
                julianday(date(?)) - julianday(date(?)) as diff_days""",
            (last, first, last, first),
        ).fetchone()
        diff_secs = float(diff_row["diff_secs"] or 0)
        diff_days = int(diff_row["diff_days"] or 0)

        # Elegir granularidad
        if diff_secs < 7200 and diff_days == 0:  # < 2h, mismo día
            fmt = "%Y-%m-%dT%H:%M:00"
            granularity = "minute"
        elif diff_days == 0:  # mismo día
            fmt = "%Y-%m-%dT%H:00:00"
            granularity = "hour"
        elif diff_days < 90:  # < 90 días
            fmt = "%Y-%m-%d"
            granularity = "day"
        elif diff_days < 730:  # < 2 años
            fmt = "%Y-%m"
            granularity = "month"
        else:
            fmt = "%Y"
            granularity = "year"

        rows = conn.execute(
            f"""SELECT
                strftime('{fmt}', t.start_time) as bucket,
                COUNT(DISTINCT t.id) as traces,
                COALESCE(SUM(json_extract(o.usage, '$.total_tokens')), 0) as tokens,
                COALESCE(SUM(json_extract(o.cost, '$.total_cost')), 0) as cost,
                AVG(CASE WHEN o.end_time IS NOT NULL
                    THEN (julianday(o.end_time) - julianday(o.start_time)) * 86400000
                    ELSE NULL END) as avg_latency_ms
            FROM traces t
            LEFT JOIN observations o ON o.trace_id = t.id
            GROUP BY bucket
            ORDER BY bucket"""
        ).fetchall()

        return {
            "granularity": granularity,
            "data": [
                {
                    "bucket": r["bucket"],
                    "traces": r["traces"],
                    "tokens": int(r["tokens"] or 0),
                    "cost": round(float(r["cost"] or 0), 6),
                    "avg_latency_ms": round(float(r["avg_latency_ms"] or 0), 1),
                }
                for r in rows
            ],
        }

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Delete ─────────────────────────────────────────────

    def delete_trace(self, trace_id: str) -> bool:
        """Borra un trace y sus observations y scores."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM scores WHERE trace_id = ?", (trace_id,))
            conn.execute("DELETE FROM observations WHERE trace_id = ?", (trace_id,))
            r = conn.execute("DELETE FROM traces WHERE id = ?", (trace_id,))
            return r.rowcount > 0

    def delete_all(self) -> int:
        """Borra todas las trazas, observations y scores."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM scores")
            conn.execute("DELETE FROM observations")
            r = conn.execute("DELETE FROM traces")
            return r.rowcount
