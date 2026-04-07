"""llm-trace: Observabilidad ligera para aplicaciones LLM.

Zero infraestructura. SQLite local. Decoradores Pythónicos.
Dashboard integrado. Compatible con OpenAI, Anthropic, LangChain.

Usage:
    from llm_trace import observe, tracer

    @observe()
    async def my_agent(query: str) -> str:
        ...

    # Launch dashboard
    tracer.dashboard(port=7600)
"""

from llm_trace.core import Tracer, flush, observe, shutdown, tracer
from llm_trace.models import Observation, ObservationType, Score, Trace
from llm_trace.scores import score
from llm_trace.wrappers import wrap_anthropic, wrap_openai


# Lazy imports para módulos con dependencias opcionales
def __getattr__(name: str):
    if name == "CallbackHandler":
        from llm_trace.langchain import CallbackHandler
        return CallbackHandler
    if name == "install_otel":
        from llm_trace.otel import install_otel
        return install_otel
    if name == "LlmTraceSpanProcessor":
        from llm_trace.otel import LlmTraceSpanProcessor
        return LlmTraceSpanProcessor
    if name == "webhook_ingest":
        from llm_trace.webhook import ingest
        return ingest
    raise AttributeError(f"module 'llm_trace' has no attribute {name!r}")

__version__ = "0.2.0"
__all__ = [
    "observe",
    "tracer",
    "Tracer",
    "flush",
    "shutdown",
    "Trace",
    "Observation",
    "Score",
    "ObservationType",
    "wrap_openai",
    "wrap_anthropic",
    "score",
    "CallbackHandler",
    "install_otel",
    "LlmTraceSpanProcessor",
    "webhook_ingest",
]
