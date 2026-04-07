"""Modelo de datos para llm-trace.

Jerarquía: Session → Trace → Observation (árbol anidado).
Cada Observation puede ser Generation, Span, Event, Tool, Agent, etc.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ObservationType(str, Enum):
    """Tipos semánticos de observation, inspirados en Langfuse V4."""

    GENERATION = "generation"
    SPAN = "span"
    EVENT = "event"
    TOOL = "tool"
    AGENT = "agent"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"
    GUARDRAIL = "guardrail"


class ScoreDataType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class ScoreSource(str, Enum):
    API = "api"
    HUMAN = "human"
    LLM_JUDGE = "llm_judge"
    USER_FEEDBACK = "user_feedback"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class Observation:
    """Un nodo en el árbol de ejecución dentro de un trace."""

    id: str = field(default_factory=_new_id)
    trace_id: str = ""
    parent_id: str | None = None
    type: ObservationType = ObservationType.SPAN
    name: str = ""
    start_time: datetime = field(default_factory=_now)
    end_time: datetime | None = None
    completion_start_time: datetime | None = None  # TTFT para generations
    status: str = "ok"  # ok | error
    level: str = "DEFAULT"  # DEFAULT | WARNING | ERROR | DEBUG
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Campos específicos de generation
    model: str | None = None
    model_parameters: dict[str, Any] | None = None
    usage: UsageDetails | None = None
    cost: CostDetails | None = None
    # Error tracking
    error_message: str | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    @property
    def ttft_ms(self) -> float | None:
        """Time to first token (solo para generations con streaming)."""
        if self.completion_start_time and self.start_time:
            return (
                self.completion_start_time - self.start_time
            ).total_seconds() * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["start_time"] = self.start_time.isoformat()
        d["end_time"] = self.end_time.isoformat() if self.end_time else None
        d["completion_start_time"] = (
            self.completion_start_time.isoformat()
            if self.completion_start_time
            else None
        )
        d["duration_ms"] = self.duration_ms
        d["ttft_ms"] = self.ttft_ms
        return d


@dataclass
class UsageDetails:
    """Tracking de tokens con soporte para cached/reasoning tokens."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class CostDetails:
    """Costos calculados por tipo de uso."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class Trace:
    """Una ejecución completa de extremo a extremo."""

    id: str = field(default_factory=_new_id)
    session_id: str | None = None
    user_id: str | None = None
    name: str = ""
    start_time: datetime = field(default_factory=_now)
    end_time: datetime | None = None
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    environment: str = "development"
    release: str | None = None
    observations: list[Observation] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    @property
    def total_tokens(self) -> int:
        return sum(
            o.usage.total_tokens
            for o in self.observations
            if o.usage
        )

    @property
    def total_cost(self) -> float:
        return sum(
            o.cost.total_cost
            for o in self.observations
            if o.cost
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "tags": self.tags,
            "environment": self.environment,
            "release": self.release,
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "observations": [o.to_dict() for o in self.observations],
        }
        return d


@dataclass
class Score:
    """Evaluación adjunta a un trace u observation."""

    id: str = field(default_factory=_new_id)
    trace_id: str = ""
    observation_id: str | None = None
    name: str = ""
    value: float | str | bool = 0.0
    data_type: ScoreDataType = ScoreDataType.NUMERIC
    source: ScoreSource = ScoreSource.API
    comment: str | None = None
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "observation_id": self.observation_id,
            "name": self.name,
            "value": self.value,
            "data_type": self.data_type.value,
            "source": self.source.value,
            "comment": self.comment,
            "created_at": self.created_at.isoformat(),
        }
