"""Core tracer con decorador @observe() y propagación de contexto.

Usa contextvars para propagar trace/observation IDs a través de
funciones anidadas, async/await, y generadores. El flush es
asíncrono via un background thread.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import queue
import threading
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar, ParamSpec, overload

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
from llm_trace.storage import Storage

logger = logging.getLogger("llm-trace")

P = ParamSpec("P")
R = TypeVar("R")

# ── Context propagation ──────────────────────────────────

_current_trace: ContextVar[Trace | None] = ContextVar(
    "llm_trace_current_trace", default=None
)
_current_observation: ContextVar[Observation | None] = ContextVar(
    "llm_trace_current_observation", default=None
)


class Tracer:
    """Singleton principal de llm-trace.

    Gestiona storage, flush asíncrono, y el dashboard.
    """

    _instance: Tracer | None = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> Tracer:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        db_path: str | Path | None = None,
        environment: str | None = None,
        release: str | None = None,
        flush_interval: float = 1.0,
        enabled: bool = True,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.enabled = enabled
        self.environment = environment or os.getenv(
            "LLM_TRACE_ENVIRONMENT", "development"
        )
        self.release = release or os.getenv("LLM_TRACE_RELEASE")
        self.flush_interval = flush_interval

        # Storage
        db = db_path or os.getenv("LLM_TRACE_DB_PATH")
        self.storage = Storage(db)

        # Async flush queue
        self._queue: queue.Queue[Trace | Observation | Score | None] = (
            queue.Queue()
        )
        self._shutdown_event = threading.Event()
        self._flush_complete = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._flush_worker, daemon=True, name="llm-trace-flush"
        )
        self._flush_thread.start()

        logger.info(
            "llm-trace initialized | db=%s | env=%s",
            self.storage.db_path,
            self.environment,
        )

    def _flush_worker(self) -> None:
        """Background thread que escribe a SQLite en batches."""
        while not self._shutdown_event.is_set():
            batch: list[Trace | Observation | Score] = []
            got_flush_signal = False

            try:
                item = self._queue.get(timeout=self.flush_interval)
                if item is None:
                    got_flush_signal = True
                else:
                    batch.append(item)
            except queue.Empty:
                continue

            # Drain everything currently in the queue
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if item is None:
                        got_flush_signal = True
                    else:
                        batch.append(item)
                except queue.Empty:
                    break

            if batch:
                self._write_batch(batch)

            if got_flush_signal:
                self._flush_complete.set()

    def _write_batch(self, batch: list[Trace | Observation | Score]) -> None:
        for item in batch:
            try:
                if isinstance(item, Trace):
                    self.storage.save_trace(item)
                elif isinstance(item, Observation):
                    self.storage.save_observation(item)
                elif isinstance(item, Score):
                    self.storage.save_score(item)
            except Exception as e:
                logger.error("Failed to write %s: %s", type(item).__name__, e)

    def _enqueue(self, item: Trace | Observation | Score) -> None:
        if self.enabled:
            self._queue.put(item)

    # ── Public API ────────────────────────────────────────

    def create_trace(
        self,
        name: str = "",
        session_id: str | None = None,
        user_id: str | None = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Trace:
        """Crea un trace manualmente (alternativa al decorador)."""
        trace = Trace(
            name=name,
            session_id=session_id,
            user_id=user_id,
            input=input,
            metadata=metadata or {},
            tags=tags or [],
            environment=self.environment,
            release=self.release,
        )
        return trace

    def end_trace(
        self, trace: Trace, output: Any = None, error: str | None = None
    ) -> None:
        """Finaliza y encola un trace."""
        trace.end_time = _now()
        if output is not None:
            trace.output = output
        self._enqueue(trace)
        for obs in trace.observations:
            self._enqueue(obs)

    def create_observation(
        self,
        name: str = "",
        obs_type: ObservationType = ObservationType.SPAN,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Observation:
        """Crea una observation manualmente."""
        trace = _current_trace.get()
        parent = _current_observation.get()

        obs = Observation(
            trace_id=trace.id if trace else "",
            parent_id=parent.id if parent else None,
            type=obs_type,
            name=name,
            input=input,
            metadata=metadata or {},
            model=model,
        )

        if trace:
            trace.observations.append(obs)
        _current_observation.set(obs)
        return obs

    def end_observation(
        self,
        obs: Observation,
        output: Any = None,
        usage: UsageDetails | None = None,
        cost: CostDetails | None = None,
        error: str | None = None,
    ) -> None:
        """Finaliza una observation."""
        obs.end_time = _now()
        if output is not None:
            obs.output = output
        if usage:
            obs.usage = usage
        if cost:
            obs.cost = cost
        if error:
            obs.status = "error"
            obs.level = "ERROR"
            obs.error_message = error

    def score(
        self,
        name: str,
        value: float | str | bool,
        trace_id: str | None = None,
        observation_id: str | None = None,
        source: ScoreSource = ScoreSource.API,
        comment: str | None = None,
    ) -> Score:
        """Adjunta un score a un trace u observation."""
        if trace_id is None:
            trace = _current_trace.get()
            trace_id = trace.id if trace else ""

        data_type = ScoreDataType.NUMERIC
        if isinstance(value, bool):
            data_type = ScoreDataType.BOOLEAN
        elif isinstance(value, str):
            data_type = ScoreDataType.CATEGORICAL

        s = Score(
            trace_id=trace_id,
            observation_id=observation_id,
            name=name,
            value=value,
            data_type=data_type,
            source=source,
            comment=comment,
        )
        self._enqueue(s)
        return s

    def flush(self) -> None:
        """Fuerza el flush de todos los eventos encolados."""
        self._flush_complete.clear()
        self._queue.put(None)  # Signal flush
        self._flush_complete.wait(timeout=10.0)

    def shutdown(self) -> None:
        """Shutdown graceful: flush + cierra storage."""
        self.flush()
        self._shutdown_event.set()
        self._flush_thread.join(timeout=5.0)
        self.storage.close()
        logger.info("llm-trace shut down")

    def dashboard(self, port: int = 7600, open_browser: bool = True) -> None:
        """Lanza el dashboard web integrado."""
        from llm_trace.dashboard import run_dashboard

        run_dashboard(self.storage, port=port, open_browser=open_browser)

    @property
    def current_trace(self) -> Trace | None:
        return _current_trace.get()

    @property
    def current_observation(self) -> Observation | None:
        return _current_observation.get()


# ── Singleton global ──────────────────────────────────────

tracer = Tracer()


def flush() -> None:
    tracer.flush()


def shutdown() -> None:
    tracer.shutdown()


# ── @observe() decorator ─────────────────────────────────


def _serialize_arg(arg: Any) -> Any:
    """Serializa argumentos de forma segura para almacenamiento."""
    if arg is None or isinstance(arg, (str, int, float, bool)):
        return arg
    if isinstance(arg, (list, tuple)):
        return [_serialize_arg(a) for a in arg[:20]]  # Limitar listas largas
    if isinstance(arg, dict):
        return {k: _serialize_arg(v) for k, v in list(arg.items())[:50]}
    return str(arg)[:500]


def _extract_input(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Extrae input de los argumentos de la función."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # Excluir self/cls
    input_dict: dict[str, Any] = {}
    offset = 0
    if params and params[0] in ("self", "cls"):
        offset = 1

    for i, val in enumerate(args[offset:], start=offset):
        if i < len(params):
            input_dict[params[i]] = _serialize_arg(val)

    for key, val in kwargs.items():
        if key not in (
            "langfuse_trace_id",
            "langfuse_session_id",
            "langfuse_user_id",
        ):
            input_dict[key] = _serialize_arg(val)

    if len(input_dict) == 1:
        return list(input_dict.values())[0]
    return input_dict or None


@overload
def observe(func: Callable[P, R]) -> Callable[P, R]: ...

@overload
def observe(
    *,
    name: str | None = None,
    as_type: ObservationType = ObservationType.SPAN,
    capture_input: bool = True,
    capture_output: bool = True,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def observe(
    func: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    as_type: ObservationType = ObservationType.SPAN,
    capture_input: bool = True,
    capture_output: bool = True,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorador que crea traces y observations automáticamente.

    La función más externa crea un Trace.
    Las funciones anidadas crean Observations como hijos.

    Usage:
        @observe()
        async def my_agent(query: str) -> str:
            result = await my_tool(query)  # Crea observation hija
            return result

        @observe(as_type=ObservationType.TOOL)
        async def my_tool(query: str) -> str:
            ...

        # Sin paréntesis también funciona:
        @observe
        def simple_fn(x: int) -> int:
            ...
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        obs_name = name or fn.__qualname__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                return await _run_observed(
                    fn, args, kwargs, obs_name, as_type,
                    capture_input, capture_output,
                    session_id, user_id, tags, metadata,
                    is_async=True,
                )

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass

                if loop and loop.is_running():
                    # Ya estamos en un event loop — no podemos hacer await
                    # Ejecutar sincrónicamente
                    return _run_observed_sync(
                        fn, args, kwargs, obs_name, as_type,
                        capture_input, capture_output,
                        session_id, user_id, tags, metadata,
                    )
                return _run_observed_sync(
                    fn, args, kwargs, obs_name, as_type,
                    capture_input, capture_output,
                    session_id, user_id, tags, metadata,
                )

            return sync_wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


async def _run_observed(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    obs_name: str,
    obs_type: ObservationType,
    capture_input: bool,
    capture_output: bool,
    session_id: str | None,
    user_id: str | None,
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
    is_async: bool = True,
) -> Any:
    """Ejecuta la función envuelta con observabilidad (async)."""
    existing_trace = _current_trace.get()
    parent_obs = _current_observation.get()
    is_root = existing_trace is None

    # Crear trace si es root
    if is_root:
        trace = tracer.create_trace(
            name=obs_name,
            session_id=session_id or kwargs.pop("session_id", None),
            user_id=user_id or kwargs.pop("user_id", None),
            input=_extract_input(fn, args, kwargs) if capture_input else None,
            tags=tags,
            metadata=metadata,
        )
    else:
        trace = existing_trace

    # Crear observation
    obs = Observation(
        trace_id=trace.id,
        parent_id=parent_obs.id if parent_obs else None,
        type=obs_type,
        name=obs_name,
        input=_extract_input(fn, args, kwargs) if capture_input else None,
        metadata=metadata or {},
    )
    trace.observations.append(obs)

    # Set context
    token_trace = _current_trace.set(trace)
    token_obs = _current_observation.set(obs)

    try:
        if is_async:
            result = await fn(*args, **kwargs)
        else:
            result = fn(*args, **kwargs)

        obs.end_time = _now()
        obs.output = _serialize_arg(result) if capture_output else None

        if is_root:
            trace.end_time = _now()
            trace.output = obs.output
            tracer._enqueue(trace)
            for o in trace.observations:
                tracer._enqueue(o)

        return result

    except Exception as e:
        obs.end_time = _now()
        obs.status = "error"
        obs.level = "ERROR"
        obs.error_message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        if is_root:
            trace.end_time = _now()
            tracer._enqueue(trace)
            for o in trace.observations:
                tracer._enqueue(o)
        raise

    finally:
        _current_trace.reset(token_trace)
        _current_observation.reset(token_obs)


def _run_observed_sync(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    obs_name: str,
    obs_type: ObservationType,
    capture_input: bool,
    capture_output: bool,
    session_id: str | None,
    user_id: str | None,
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
) -> Any:
    """Versión síncrona de _run_observed."""
    existing_trace = _current_trace.get()
    parent_obs = _current_observation.get()
    is_root = existing_trace is None

    if is_root:
        trace = tracer.create_trace(
            name=obs_name,
            session_id=session_id or kwargs.pop("session_id", None),
            user_id=user_id or kwargs.pop("user_id", None),
            input=_extract_input(fn, args, kwargs) if capture_input else None,
            tags=tags,
            metadata=metadata,
        )
    else:
        trace = existing_trace  # type: ignore

    obs = Observation(
        trace_id=trace.id,
        parent_id=parent_obs.id if parent_obs else None,
        type=obs_type,
        name=obs_name,
        input=_extract_input(fn, args, kwargs) if capture_input else None,
        metadata=metadata or {},
    )
    trace.observations.append(obs)

    token_trace = _current_trace.set(trace)
    token_obs = _current_observation.set(obs)

    try:
        result = fn(*args, **kwargs)
        obs.end_time = _now()
        obs.output = _serialize_arg(result) if capture_output else None

        if is_root:
            trace.end_time = _now()
            trace.output = obs.output
            tracer._enqueue(trace)
            for o in trace.observations:
                tracer._enqueue(o)

        return result

    except Exception as e:
        obs.end_time = _now()
        obs.status = "error"
        obs.level = "ERROR"
        obs.error_message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        if is_root:
            trace.end_time = _now()
            tracer._enqueue(trace)
            for o in trace.observations:
                tracer._enqueue(o)
        raise

    finally:
        _current_trace.reset(token_trace)
        _current_observation.reset(token_obs)


def score(
    name: str,
    value: float | str | bool,
    trace_id: str | None = None,
    observation_id: str | None = None,
    source: ScoreSource = ScoreSource.API,
    comment: str | None = None,
) -> Score:
    """Shortcut global para adjuntar un score."""
    return tracer.score(
        name=name,
        value=value,
        trace_id=trace_id,
        observation_id=observation_id,
        source=source,
        comment=comment,
    )
