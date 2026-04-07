"""Unit tests for llm-trace core functionality. No API keys required."""

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

import llm_trace as llm_trace_module
from llm_trace.core import Tracer
from llm_trace.models import ObservationType, ScoreDataType, Trace


@pytest.fixture
def tracer(tmp_path: Path):
    """Fresh tracer with isolated temp database for each test."""
    Tracer._instance = None
    t = Tracer(db_path=str(tmp_path / "test.db"), flush_interval=0.1)
    # also patch the module-level singleton so @observe() uses this tracer
    with patch.object(llm_trace_module, "tracer", t), \
         patch("llm_trace.core.tracer", t):
        yield t
    t.shutdown()
    Tracer._instance = None


class TestTracer:
    def test_create_trace(self, tracer: Tracer) -> None:
        trace = tracer.create_trace(name="test-trace")
        assert trace.id
        assert trace.name == "test-trace"
        assert trace.start_time

    def test_end_trace(self, tracer: Tracer) -> None:
        trace = tracer.create_trace(name="test-trace")
        tracer.end_trace(trace, output={"result": "ok"})
        assert trace.end_time
        assert trace.output == {"result": "ok"}

    def test_create_observation(self, tracer: Tracer) -> None:
        trace = tracer.create_trace(name="parent")
        obs = tracer.create_observation(name="step", obs_type=ObservationType.SPAN)
        tracer.end_observation(obs)
        tracer.end_trace(trace)
        assert obs.id
        assert obs.name == "step"
        assert obs.type == ObservationType.SPAN

    def test_trace_stored_in_db(self, tracer: Tracer) -> None:
        trace = tracer.create_trace(name="stored")
        tracer.end_trace(trace, output="done")
        tracer.flush()
        time.sleep(0.2)

        result = tracer.storage.get_trace(trace.id)
        assert result is not None
        assert result.name == "stored"

    def test_error_trace(self, tracer: Tracer) -> None:
        trace = tracer.create_trace(name="failing")
        tracer.end_trace(trace, error="Something went wrong")
        assert trace.end_time

    def test_trace_with_metadata(self, tracer: Tracer) -> None:
        trace = tracer.create_trace(
            name="meta-trace",
            session_id="sess-1",
            user_id="user-42",
            tags=["production", "v2"],
            metadata={"model": "gpt-4o"},
        )
        assert trace.session_id == "sess-1"
        assert trace.user_id == "user-42"
        assert "production" in trace.tags
        assert trace.metadata["model"] == "gpt-4o"


class TestObserveDecorator:
    def test_sync_function(self, tracer: Tracer) -> None:
        from llm_trace import observe

        @observe()
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_async_function(self, tracer: Tracer) -> None:
        import asyncio
        from llm_trace import observe

        @observe()
        async def async_add(a: int, b: int) -> int:
            return a + b

        result = asyncio.run(async_add(4, 5))
        assert result == 9

    def test_exception_propagates(self, tracer: Tracer) -> None:
        from llm_trace import observe

        @observe()
        def boom() -> None:
            raise ValueError("expected error")

        with pytest.raises(ValueError, match="expected error"):
            boom()

    def test_nested_creates_observations(self, tracer: Tracer) -> None:
        from llm_trace import observe

        @observe()
        def outer() -> str:
            return inner()

        @observe()
        def inner() -> str:
            return "done"

        outer()
        tracer.flush()
        time.sleep(0.2)

        traces = tracer.storage.list_traces(limit=10)
        assert len(traces) >= 1

    def test_custom_name(self, tracer: Tracer) -> None:
        from llm_trace import observe

        @observe(name="my-custom-name")
        def func() -> None:
            pass

        func()
        tracer.flush()
        time.sleep(0.2)

        traces = tracer.storage.list_traces(limit=10)
        assert any(t.name == "my-custom-name" for t in traces)


class TestModels:
    def test_trace_duration(self) -> None:
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        end = start + timedelta(milliseconds=1500)
        trace = Trace(name="t", start_time=start, end_time=end)
        assert trace.duration_ms == pytest.approx(1500, abs=1)

    def test_score_data_type_inference(self, tracer: Tracer) -> None:
        from llm_trace import score

        trace = tracer.create_trace(name="scored")
        tracer.end_trace(trace)

        s1 = tracer.score("quality", 0.9, trace_id=trace.id)
        s2 = tracer.score("passed", True, trace_id=trace.id)
        s3 = tracer.score("category", "helpful", trace_id=trace.id)

        assert s1.data_type == ScoreDataType.NUMERIC
        assert s2.data_type == ScoreDataType.BOOLEAN
        assert s3.data_type == ScoreDataType.CATEGORICAL

    def test_observation_types_are_strings(self) -> None:
        assert ObservationType.GENERATION == "generation"
        assert ObservationType.SPAN == "span"
        assert ObservationType.TOOL == "tool"


class TestStorage:
    def test_list_traces_empty(self, tracer: Tracer) -> None:
        traces = tracer.storage.list_traces()
        assert traces == []

    def test_save_and_retrieve_trace(self, tracer: Tracer) -> None:
        trace = tracer.create_trace(name="retrieve-me", tags=["test"])
        tracer.end_trace(trace, output="result")
        tracer.flush()
        time.sleep(0.2)

        retrieved = tracer.storage.get_trace(trace.id)
        assert retrieved is not None
        assert retrieved.name == "retrieve-me"
        assert "test" in retrieved.tags

    def test_delete_trace(self, tracer: Tracer) -> None:
        trace = tracer.create_trace(name="to-delete")
        tracer.end_trace(trace)
        tracer.flush()
        time.sleep(0.2)

        deleted = tracer.storage.delete_trace(trace.id)
        assert deleted is True
        assert tracer.storage.get_trace(trace.id) is None

    def test_get_stats(self, tracer: Tracer) -> None:
        stats = tracer.storage.get_stats()
        assert "trace_count" in stats
        assert "total_tokens" in stats
        assert "total_cost" in stats
