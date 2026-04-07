"""Shortcuts para scoring de traces y observations.

Usage:
    from llm_trace import score

    @observe()
    def my_fn():
        result = ...
        score("quality", 0.95)
        score("category", "correct")
        score("passed", True)
"""

from llm_trace.models import Score, ScoreSource


def score(
    name: str,
    value: float | str | bool,
    trace_id: str | None = None,
    observation_id: str | None = None,
    source: ScoreSource = ScoreSource.API,
    comment: str | None = None,
) -> Score:
    """Adjunta un score al trace u observation actual."""
    from llm_trace.core import tracer

    return tracer.score(
        name=name,
        value=value,
        trace_id=trace_id,
        observation_id=observation_id,
        source=source,
        comment=comment,
    )
