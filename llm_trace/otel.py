"""OpenTelemetry SpanProcessor para llm-trace.

Captura spans OTEL de cualquier instrumentador y los convierte
a observations de llm-trace. Esta es la misma estrategia que usa
Langfuse v3 para soportar 80+ frameworks.

Con esto, cualquier framework con instrumentación OTEL funciona
automáticamente: LlamaIndex, Haystack, CrewAI, DSPy, Anthropic,
Vercel AI SDK, Spring AI, smolagents, etc.

Usage:
    from llm_trace.otel import install_otel

    # Opción 1: Auto-setup (registra el processor globalmente)
    install_otel()

    # Opción 2: Con instrumentadores específicos
    install_otel(
        instrumentors=[
            "llama_index",      # openinference-instrumentation-llama-index
            "anthropic",        # opentelemetry-instrumentation-anthropic
            "haystack",         # openinference-instrumentation-haystack
            "crewai",           # openinference-instrumentation-crewai
            "dspy",             # openinference-instrumentation-dspy
        ]
    )

    # Ahora usa cualquier framework normalmente — todo se traza
    from llama_index.core import VectorStoreIndex
    index = VectorStoreIndex.from_documents(docs)
    response = index.as_query_engine().query("What is observability?")
    # → Trace automático con retriever + generation observations

Requiere: pip install opentelemetry-api opentelemetry-sdk
Instrumentadores opcionales: pip install openinference-instrumentation-llama-index etc.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from llm_trace.core import tracer
from llm_trace.models import (
    CostDetails,
    Observation,
    ObservationType,
    Trace,
    UsageDetails,
    _now,
)

logger = logging.getLogger("llm-trace.otel")

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import (
        ReadableSpan,
        SpanProcessor,
        TracerProvider,
    )
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: F401
    from opentelemetry.trace import StatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

    class SpanProcessor:  # type: ignore[no-redef]
        pass

    class ReadableSpan:  # type: ignore[no-redef]
        pass


# ── Attribute mapping ─────────────────────────────────────
# Mapea atributos OTEL estándar (GenAI semantic conventions)
# y atributos de OpenInference a campos de llm-trace.

# GenAI semantic conventions (OTEL estándar)
_GENAI_MODEL = "gen_ai.request.model"
_GENAI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_GENAI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_GENAI_TEMPERATURE = "gen_ai.request.temperature"
_GENAI_MAX_TOKENS = "gen_ai.request.max_tokens"
_GENAI_SYSTEM = "gen_ai.system"
_GENAI_PROMPT = "gen_ai.prompt"
_GENAI_COMPLETION = "gen_ai.completion"

# OpenInference conventions (Arize/Phoenix ecosystem)
_OI_SPAN_KIND = "openinference.span.kind"
_OI_INPUT_VALUE = "input.value"
_OI_INPUT_MIME = "input.mime_type"
_OI_OUTPUT_VALUE = "output.value"
_OI_OUTPUT_MIME = "output.mime_type"
_OI_LLM_MODEL = "llm.model_name"
_OI_LLM_INPUT_MESSAGES = "llm.input_messages"
_OI_LLM_OUTPUT_MESSAGES = "llm.output_messages"
_OI_LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
_OI_LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
_OI_LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
_OI_RETRIEVAL_DOCUMENTS = "retrieval.documents"
_OI_EMBEDDING_MODEL = "embedding.model_name"
_OI_TOOL_NAME = "tool.name"

# Span kind mapping: OpenInference span kinds → ObservationType
_SPAN_KIND_MAP: dict[str, ObservationType] = {
    "LLM": ObservationType.GENERATION,
    "CHAIN": ObservationType.SPAN,
    "TOOL": ObservationType.TOOL,
    "AGENT": ObservationType.AGENT,
    "RETRIEVER": ObservationType.RETRIEVER,
    "EMBEDDING": ObservationType.EMBEDDING,
    "RERANKER": ObservationType.SPAN,
    "GUARDRAIL": ObservationType.GUARDRAIL,
    "EVALUATOR": ObservationType.SPAN,
}


def _get_attr(span: Any, key: str, default: Any = None) -> Any:
    """Extrae un atributo de un span OTEL de forma segura."""
    attrs = getattr(span, "attributes", {}) or {}
    return attrs.get(key, default)


def _span_to_observation_type(span: Any) -> ObservationType:
    """Determina el tipo de observation basándose en atributos del span."""
    # 1. OpenInference span.kind
    oi_kind = _get_attr(span, _OI_SPAN_KIND)
    if oi_kind and isinstance(oi_kind, str):
        mapped = _SPAN_KIND_MAP.get(oi_kind.upper())
        if mapped:
            return mapped

    # 2. GenAI conventions (si tiene gen_ai.system → es generation)
    if _get_attr(span, _GENAI_SYSTEM) or _get_attr(span, _GENAI_MODEL):
        return ObservationType.GENERATION

    # 3. Nombre del span como heurístico
    name = getattr(span, "name", "").lower()
    if any(kw in name for kw in ("llm", "chat", "completion", "generate")):
        return ObservationType.GENERATION
    if any(kw in name for kw in ("retriev", "search", "query", "vector")):
        return ObservationType.RETRIEVER
    if any(kw in name for kw in ("tool", "function_call")):
        return ObservationType.TOOL
    if any(kw in name for kw in ("embed",)):
        return ObservationType.EMBEDDING
    if any(kw in name for kw in ("agent", "plan", "reason")):
        return ObservationType.AGENT

    return ObservationType.SPAN


def _extract_usage(span: Any) -> UsageDetails | None:
    """Extrae token usage desde GenAI o OpenInference attributes."""
    input_t = (
        _get_attr(span, _GENAI_INPUT_TOKENS)
        or _get_attr(span, _OI_LLM_TOKEN_COUNT_PROMPT)
        or 0
    )
    output_t = (
        _get_attr(span, _GENAI_OUTPUT_TOKENS)
        or _get_attr(span, _OI_LLM_TOKEN_COUNT_COMPLETION)
        or 0
    )
    total = _get_attr(span, _OI_LLM_TOKEN_COUNT_TOTAL) or (input_t + output_t)

    if input_t or output_t or total:
        return UsageDetails(
            input_tokens=int(input_t),
            output_tokens=int(output_t),
            total_tokens=int(total),
        )
    return None


def _extract_input(span: Any) -> Any:
    """Extrae el input del span."""
    # GenAI prompt
    prompt = _get_attr(span, _GENAI_PROMPT)
    if prompt:
        return prompt

    # OpenInference input
    oi_input = _get_attr(span, _OI_INPUT_VALUE)
    if oi_input:
        return oi_input

    # LLM input messages (OpenInference)
    messages = _get_attr(span, _OI_LLM_INPUT_MESSAGES)
    if messages:
        return messages

    return None


def _extract_output(span: Any) -> Any:
    """Extrae el output del span."""
    completion = _get_attr(span, _GENAI_COMPLETION)
    if completion:
        return completion

    oi_output = _get_attr(span, _OI_OUTPUT_VALUE)
    if oi_output:
        return oi_output

    messages = _get_attr(span, _OI_LLM_OUTPUT_MESSAGES)
    if messages:
        return messages

    return None


def _extract_model(span: Any) -> str | None:
    """Extrae el nombre del modelo."""
    return (
        _get_attr(span, _GENAI_MODEL)
        or _get_attr(span, _OI_LLM_MODEL)
        or _get_attr(span, _OI_EMBEDDING_MODEL)
    )


def _ns_to_datetime(ns: int | None) -> datetime | None:
    """Convierte nanosegundos OTEL a datetime."""
    if ns is None or ns == 0:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=UTC)


class LlmTraceSpanProcessor(SpanProcessor):
    """SpanProcessor que convierte spans OTEL a observations de llm-trace.

    Se registra en el TracerProvider global de OTEL y captura
    automáticamente spans de cualquier instrumentador.

    Filtros aplicados:
    - Solo procesa spans con atributos gen_ai.*, openinference.*,
      o de scopes conocidos (openai, anthropic, llama_index, etc.)
    - Ignora spans internos de OTEL (HTTP client, etc.)
    """

    # Scopes/prefijos de instrumentadores que capturamos
    KNOWN_SCOPES = {
        "openai",
        "anthropic",
        "llama_index",
        "llamaindex",
        "haystack",
        "crewai",
        "dspy",
        "langchain",
        "langgraph",
        "litellm",
        "cohere",
        "mistral",
        "bedrock",
        "vertexai",
        "google_genai",
        "groq",
        "openinference",
        "traceloop",
        "openlit",
    }

    def __init__(
        self,
        capture_all: bool = False,
        filter_scopes: set[str] | None = None,
    ) -> None:
        """
        Args:
            capture_all: Si True, captura TODOS los spans (no solo LLM).
            filter_scopes: Set custom de scopes a capturar.
        """
        self.capture_all = capture_all
        self.filter_scopes = filter_scopes or self.KNOWN_SCOPES
        self._traces: dict[str, Trace] = {}  # trace_id → Trace
        self._observations: dict[str, Observation] = {}  # span_id → Observation

    def _should_capture(self, span: ReadableSpan) -> bool:
        """Determina si este span debe capturarse."""
        if self.capture_all:
            return True

        attrs = getattr(span, "attributes", {}) or {}

        # Tiene atributos GenAI o OpenInference
        for key in attrs:
            if isinstance(key, str) and (
                key.startswith("gen_ai.")
                or key.startswith("openinference.")
                or key.startswith("llm.")
                or key.startswith("retrieval.")
                or key.startswith("embedding.")
                or key.startswith("tool.")
            ):
                return True

        # Scope del instrumentador es conocido
        instrumentation_scope = getattr(span, "instrumentation_scope", None)
        if instrumentation_scope:
            scope_name = getattr(instrumentation_scope, "name", "").lower()
            for known in self.filter_scopes:
                if known in scope_name:
                    return True

        # Nombre del span sugiere LLM
        name = getattr(span, "name", "").lower()
        llm_keywords = {
            "chat", "completion", "generate", "embed", "retrieve",
            "llm", "agent", "tool", "rerank",
        }
        return any(kw in name for kw in llm_keywords)

    def on_start(self, span: ReadableSpan, parent_context: Any = None) -> None:
        """Llamado cuando un span inicia (no usado, procesamos on_end)."""
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """Llamado cuando un span termina — convierte a observation."""
        if not self._should_capture(span):
            return

        try:
            self._process_span(span)
        except Exception as e:
            logger.error("Error processing OTEL span: %s", e)

    def _process_span(self, span: ReadableSpan) -> None:
        """Convierte un span OTEL completado en trace + observation."""
        ctx = getattr(span, "context", None)
        if not ctx:
            return

        trace_id_int = getattr(ctx, "trace_id", 0)
        span_id_int = getattr(ctx, "span_id", 0)
        trace_id = format(trace_id_int, "032x")[:16]
        span_id = format(span_id_int, "016x")

        # Parent
        parent_ctx = getattr(span, "parent", None)
        parent_span_id = None
        if parent_ctx:
            parent_id_int = getattr(parent_ctx, "span_id", 0)
            if parent_id_int:
                parent_span_id = format(parent_id_int, "016x")

        # Asegurar que existe el trace
        if trace_id not in self._traces:
            trace = Trace(
                id=trace_id,
                name=getattr(span, "name", "otel-trace"),
                environment=tracer.environment,
                release=tracer.release,
            )
            self._traces[trace_id] = trace

        trace = self._traces[trace_id]

        # Crear observation
        obs_type = _span_to_observation_type(span)
        start_time = _ns_to_datetime(getattr(span, "start_time", None)) or _now()
        end_time = _ns_to_datetime(getattr(span, "end_time", None)) or _now()

        # Extraer error
        status = getattr(span, "status", None)
        is_error = False
        error_msg = None
        if status and hasattr(status, "status_code"):
            is_error = status.status_code == StatusCode.ERROR
            error_msg = getattr(status, "description", None)

        # Extraer metadata (atributos no-estándar)
        attrs = dict(getattr(span, "attributes", {}) or {})
        metadata = {}
        standard_prefixes = ("gen_ai.", "llm.", "openinference.", "retrieval.", "embedding.", "tool.", "input.", "output.")
        for k, v in attrs.items():
            if not any(k.startswith(p) for p in standard_prefixes):
                metadata[k] = v

        model = _extract_model(span)
        usage = _extract_usage(span)

        obs = Observation(
            id=span_id,
            trace_id=trace_id,
            parent_id=self._observations.get(parent_span_id, Observation()).id
            if parent_span_id and parent_span_id in self._observations
            else None,
            type=obs_type,
            name=getattr(span, "name", "unknown"),
            start_time=start_time,
            end_time=end_time,
            status="error" if is_error else "ok",
            level="ERROR" if is_error else "DEFAULT",
            input=_extract_input(span),
            output=_extract_output(span),
            metadata=metadata,
            model=model,
            model_parameters={
                k: v
                for k, v in {
                    "temperature": _get_attr(span, _GENAI_TEMPERATURE),
                    "max_tokens": _get_attr(span, _GENAI_MAX_TOKENS),
                }.items()
                if v is not None
            } or None,
            usage=usage,
            cost=self._calculate_cost(model, usage) if model and usage else None,
            error_message=error_msg,
        )

        self._observations[span_id] = obs
        trace.observations.append(obs)

        # Si es un root span (sin parent), finalizar y enqueue el trace
        if parent_span_id is None or parent_span_id not in self._observations:
            trace.name = obs.name
            trace.start_time = start_time
            trace.end_time = end_time
            trace.input = obs.input
            trace.output = obs.output

            tracer._enqueue(trace)
            for o in trace.observations:
                tracer._enqueue(o)

            # Cleanup
            del self._traces[trace_id]
            for o in trace.observations:
                self._observations.pop(o.id, None)

    def _calculate_cost(
        self, model: str | None, usage: UsageDetails | None
    ) -> CostDetails | None:
        if not model or not usage:
            return None
        from llm_trace.wrappers import _calculate_cost

        cost = _calculate_cost(model, usage.input_tokens, usage.output_tokens)
        return cost if cost.total_cost > 0 else None

    def shutdown(self) -> None:
        """Flush traces pendientes."""
        for trace in self._traces.values():
            tracer._enqueue(trace)
            for o in trace.observations:
                tracer._enqueue(o)
        self._traces.clear()
        self._observations.clear()
        tracer.flush()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        tracer.flush()
        return True


# ── Auto-instrumentor registry ────────────────────────────

_INSTRUMENTOR_REGISTRY: dict[str, dict[str, str]] = {
    "llama_index": {
        "package": "openinference.instrumentation.llama_index",
        "class": "LlamaIndexInstrumentor",
    },
    "anthropic": {
        "package": "openinference.instrumentation.anthropic",
        "class": "AnthropicInstrumentor",
    },
    "haystack": {
        "package": "openinference.instrumentation.haystack",
        "class": "HaystackInstrumentor",
    },
    "crewai": {
        "package": "openinference.instrumentation.crewai",
        "class": "CrewAIInstrumentor",
    },
    "dspy": {
        "package": "openinference.instrumentation.dspy",
        "class": "DSPyInstrumentor",
    },
    "openai": {
        "package": "openinference.instrumentation.openai",
        "class": "OpenAIInstrumentor",
    },
    "mistral": {
        "package": "openinference.instrumentation.mistralai",
        "class": "MistralAIInstrumentor",
    },
    "groq": {
        "package": "openinference.instrumentation.groq",
        "class": "GroqInstrumentor",
    },
    "bedrock": {
        "package": "openinference.instrumentation.bedrock",
        "class": "BedrockInstrumentor",
    },
    "langchain": {
        "package": "openinference.instrumentation.langchain",
        "class": "LangChainInstrumentor",
    },
}


def install_otel(
    instrumentors: list[str] | None = None,
    capture_all: bool = False,
) -> LlmTraceSpanProcessor:
    """Instala el SpanProcessor de llm-trace en el TracerProvider global.

    Args:
        instrumentors: Lista de instrumentadores a activar.
            Nombres válidos: llama_index, anthropic, haystack, crewai,
            dspy, openai, mistral, groq, bedrock, langchain.
            Si es None, solo instala el processor (captura spans
            de instrumentadores ya activos).
        capture_all: Si True, captura todos los spans OTEL
            (no solo los de frameworks LLM).

    Returns:
        El SpanProcessor instalado (para control manual).

    Raises:
        ImportError: Si opentelemetry-sdk no está instalado.
    """
    if not _HAS_OTEL:
        raise ImportError(
            "opentelemetry-sdk is required for OTEL integration. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk"
        )

    # Crear y registrar el processor
    processor = LlmTraceSpanProcessor(capture_all=capture_all)

    # Obtener o crear TracerProvider
    current_provider = otel_trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        provider = current_provider
    else:
        provider = TracerProvider()
        otel_trace.set_tracer_provider(provider)

    provider.add_span_processor(processor)

    logger.info("llm-trace OTEL SpanProcessor installed")

    # Activar instrumentadores solicitados
    if instrumentors:
        for name in instrumentors:
            _activate_instrumentor(name, provider)

    return processor


def _activate_instrumentor(name: str, provider: Any) -> None:
    """Intenta activar un instrumentador del registry."""
    entry = _INSTRUMENTOR_REGISTRY.get(name)
    if not entry:
        logger.warning(
            "Unknown instrumentor '%s'. Available: %s",
            name,
            ", ".join(_INSTRUMENTOR_REGISTRY.keys()),
        )
        return

    try:
        import importlib

        module = importlib.import_module(entry["package"])
        cls = getattr(module, entry["class"])
        instrumentor = cls()
        instrumentor.instrument(tracer_provider=provider)
        logger.info("Activated instrumentor: %s (%s)", name, entry["class"])
    except ImportError:
        logger.warning(
            "Instrumentor '%s' requires package '%s'. "
            "Install with: pip install %s",
            name,
            entry["package"],
            entry["package"].replace(".", "-"),
        )
    except Exception as e:
        logger.error("Failed to activate instrumentor '%s': %s", name, e)
