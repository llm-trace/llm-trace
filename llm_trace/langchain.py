"""CallbackHandler para LangChain y LangGraph.

Implementa BaseCallbackHandler de LangChain para capturar
automáticamente chains, agents, tools, retrievers y LLM calls
con su jerarquía completa.

Usage con LangChain:
    from llm_trace.langchain import CallbackHandler

    handler = CallbackHandler(session_id="chat-123", user_id="marc")
    chain.invoke({"topic": "AI"}, config={"callbacks": [handler]})

Usage con LangGraph:
    graph = workflow.compile()
    result = graph.invoke(
        {"messages": [HumanMessage(content="hola")]},
        config={"callbacks": [handler]}
    )

Usage con LangGraph streaming + FastAPI:
    async def stream_endpoint(request: Request):
        handler = CallbackHandler(session_id=request.session_id)
        async for event in graph.astream(
            input_data,
            config={"callbacks": [handler]}
        ):
            yield event
        handler.flush()
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from uuid import UUID

from llm_trace.core import _current_observation, _current_trace, tracer
from llm_trace.models import (
    CostDetails,
    Observation,
    ObservationType,
    Trace,
    UsageDetails,
    _now,
)

logger = logging.getLogger("llm-trace.langchain")

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

try:
    from langchain_core.callbacks import BaseCallbackHandler  # type: ignore[no-redef]
    from langchain_core.messages import BaseMessage  # type: ignore[no-redef]
    from langchain_core.outputs import LLMResult  # type: ignore[no-redef]

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


def _serialize_messages(messages: Any) -> Any:
    """Serializa mensajes de LangChain a dict."""
    if isinstance(messages, list):
        result: list[Any] = []
        for msg in messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                result.append(
                    {
                        "role": getattr(msg, "type", "unknown"),
                        "content": str(getattr(msg, "content", ""))[:2000],
                    }
                )
            elif isinstance(msg, dict):
                result.append(msg)
            else:
                result.append(str(msg)[:500])
        return result
    return str(messages)[:2000] if messages else None


def _safe_str(obj: Any, max_len: int = 2000) -> str | None:
    """Convierte a string truncado de forma segura."""
    if obj is None:
        return None
    try:
        s = str(obj)
        return s[:max_len] if len(s) > max_len else s
    except Exception:
        return "<unserializable>"


class CallbackHandler(BaseCallbackHandler):
    """Callback handler de llm-trace para LangChain y LangGraph.

    Captura automáticamente:
    - Chain runs → Observations tipo SPAN
    - LLM calls → Observations tipo GENERATION (con tokens y costos)
    - Tool calls → Observations tipo TOOL
    - Retriever calls → Observations tipo RETRIEVER
    - Agent actions → Observations tipo AGENT
    - Chat model calls → Observations tipo GENERATION

    La jerarquía parent-child se mantiene via run_id → parent_run_id.

    Args:
        session_id: ID de sesión para agrupar traces
        user_id: ID del usuario
        trace_name: Nombre del trace raíz
        tags: Tags para filtrado
        metadata: Metadata adicional
        environment: Environment (default: tracer.environment)
    """

    def __init__(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        trace_name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        environment: str | None = None,
    ) -> None:
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for CallbackHandler. "
                "Install with: pip install llm-trace[langchain]"
            )

        super().__init__()
        self.session_id = session_id
        self.user_id = user_id
        self.trace_name = trace_name
        self.tags = tags or []
        self.metadata = metadata or {}
        self.environment = environment or tracer.environment

        # Estado interno
        self._trace: Trace | None = None
        self._observations: dict[str, Observation] = {}  # run_id → obs
        self._run_parents: dict[str, str | None] = {}  # run_id → parent_run_id

    def _ensure_trace(self) -> Trace:
        """Crea el trace raíz si no existe y lo propaga al contexto."""
        if self._trace is None:
            self._trace = Trace(
                name=self.trace_name or "langchain",
                session_id=self.session_id,
                user_id=self.user_id,
                tags=self.tags,
                metadata=self.metadata,
                environment=self.environment,
            )
            # Propagar al contexto para que @observe() anide bajo este trace
            _current_trace.set(self._trace)
        return self._trace

    def _start_observation(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        obs_type: ObservationType,
        name: str,
        input_data: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Observation:
        """Crea y registra una nueva observation, propagando contexto."""
        trace = self._ensure_trace()
        rid = str(run_id)
        pid = str(parent_run_id) if parent_run_id else None
        self._run_parents[rid] = pid

        # Resolver parent observation
        parent_obs_id = None
        if pid and pid in self._observations:
            parent_obs_id = self._observations[pid].id

        obs = Observation(
            trace_id=trace.id,
            parent_id=parent_obs_id,
            type=obs_type,
            name=name,
            input=input_data,
            metadata=metadata or {},
            model=model,
        )
        self._observations[rid] = obs
        trace.observations.append(obs)

        # Propagar al contexto para que @observe() anide bajo esta observation
        _current_observation.set(obs)

        return obs

    def _end_observation(
        self,
        run_id: UUID,
        output: Any = None,
        error: str | None = None,
        usage: UsageDetails | None = None,
        cost: CostDetails | None = None,
    ) -> None:
        """Finaliza una observation existente."""
        rid = str(run_id)
        obs = self._observations.get(rid)
        if obs is None:
            return  # Normal for LangGraph internal chain events

        obs.end_time = _now()
        if output is not None:
            obs.output = output
        if error:
            obs.status = "error"
            obs.level = "ERROR"
            obs.error_message = error
        if usage:
            obs.usage = usage
        if cost:
            obs.cost = cost

    # ── Chain callbacks ────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        serialized = serialized or {}
        # LangGraph pasa el nombre del nodo en kwargs['name']
        name = (
            kwargs.get("name")
            or serialized.get("name")
            or (serialized.get("id", ["chain"])[-1] if serialized.get("id") else None)
            or "chain"
        )
        # Inferir tipo por nombre del nodo
        name_lower = name.lower()
        if any(k in name_lower for k in ("agent", "planner", "reasoner", "orchestrat")):
            obs_type = ObservationType.AGENT
        elif any(k in name_lower for k in ("retriev", "search", "vector", "rag")):
            obs_type = ObservationType.RETRIEVER
        elif any(k in name_lower for k in ("guardrail", "moderat", "safety", "filter")):
            obs_type = ObservationType.GUARDRAIL
        elif any(k in name_lower for k in ("tool",)):
            obs_type = ObservationType.TOOL
        else:
            obs_type = ObservationType.SPAN
        self._start_observation(
            run_id=run_id,
            parent_run_id=parent_run_id,
            obs_type=obs_type,
            name=name,
            input_data=_safe_str(inputs),
            metadata={**(metadata or {}), "tags": tags or []},
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._end_observation(run_id, output=_safe_str(outputs))

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._end_observation(run_id, error=f"{type(error).__name__}: {error}")

    # ── LLM callbacks ─────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        serialized = serialized or {}
        model = (
            kwargs.get("invocation_params", {}).get("model_name")
            or kwargs.get("invocation_params", {}).get("model")
            or serialized.get("kwargs", {}).get("model_name", "unknown")
        )
        self._start_observation(
            run_id=run_id,
            parent_run_id=parent_run_id,
            obs_type=ObservationType.GENERATION,
            name=f"llm.{model}",
            input_data=prompts,
            model=model,
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        serialized = serialized or {}
        model = (
            kwargs.get("invocation_params", {}).get("model_name")
            or kwargs.get("invocation_params", {}).get("model")
            or serialized.get("kwargs", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model", "unknown")
        )
        # Flatten messages
        flat_msgs = []
        for batch in messages:
            flat_msgs.extend(_serialize_messages(batch))

        self._start_observation(
            run_id=run_id,
            parent_run_id=parent_run_id,
            obs_type=ObservationType.GENERATION,
            name=f"chat.{model}",
            input_data=flat_msgs,
            model=model,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        output = None
        usage = None
        cost = None

        # Extraer output
        if response.generations:
            gen = response.generations[0]
            if gen:
                first = gen[0]
                if hasattr(first, "message"):
                    msg = first.message
                    output = {
                        "role": getattr(msg, "type", "assistant"),
                        "content": _safe_str(getattr(msg, "content", "")),
                    }
                    # Tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        output["tool_calls"] = [
                            {
                                "name": tc.get("name", ""),
                                "args": _safe_str(tc.get("args", {})),
                            }
                            for tc in msg.tool_calls[:10]
                        ]
                else:
                    output = _safe_str(getattr(first, "text", str(first)))

        # Extraer usage
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage", {})
        if token_usage:
            input_t = token_usage.get("prompt_tokens", 0)
            output_t = token_usage.get("completion_tokens", 0)
            usage = UsageDetails(
                input_tokens=input_t,
                output_tokens=output_t,
                total_tokens=token_usage.get("total_tokens", input_t + output_t),
            )

            # Calcular costo
            obs = self._observations.get(str(run_id))
            if obs and obs.model:
                from llm_trace.wrappers import _calculate_cost

                cost = _calculate_cost(obs.model, input_t, output_t)

        self._end_observation(run_id, output=output, usage=usage, cost=cost)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._end_observation(run_id, error=f"{type(error).__name__}: {error}")

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Captura TTFT en el primer token de streaming."""
        obs = self._observations.get(str(run_id))
        if obs and obs.completion_start_time is None:
            obs.completion_start_time = _now()

    # ── Tool callbacks ────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        serialized = serialized or {}
        name = kwargs.get("name") or serialized.get("name", "tool")
        self._start_observation(
            run_id=run_id,
            parent_run_id=parent_run_id,
            obs_type=ObservationType.TOOL,
            name=f"tool.{name}",
            input_data=_safe_str(input_str),
            metadata=metadata,
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._end_observation(run_id, output=_safe_str(output))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._end_observation(run_id, error=f"{type(error).__name__}: {error}")

    # ── Retriever callbacks ───────────────────────────────

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        serialized = serialized or {}
        name = serialized.get("name", "retriever")
        self._start_observation(
            run_id=run_id,
            parent_run_id=parent_run_id,
            obs_type=ObservationType.RETRIEVER,
            name=f"retriever.{name}",
            input_data=query,
        )

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        doc_summaries = []
        for doc in documents[:20]:
            content = getattr(doc, "page_content", str(doc))
            meta = getattr(doc, "metadata", {})
            doc_summaries.append(
                {
                    "content": content[:300],
                    "metadata": meta,
                }
            )
        self._end_observation(
            run_id,
            output={"documents": doc_summaries, "count": len(documents)},
        )

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._end_observation(run_id, error=f"{type(error).__name__}: {error}")

    # ── Agent callbacks ───────────────────────────────────

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tool = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", "")
        self._start_observation(
            run_id=run_id,
            parent_run_id=parent_run_id,
            obs_type=ObservationType.AGENT,
            name=f"agent.action.{tool}",
            input_data=_safe_str(tool_input),
        )

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        output = getattr(finish, "return_values", str(finish))
        self._end_observation(run_id, output=_safe_str(output))

    # ── Flush / finalize ──────────────────────────────────

    def flush(self) -> None:
        """Envía el trace completo al storage.

        Llamar después de que la invocación termine,
        especialmente en entornos serverless o streaming.
        """
        if self._trace is None:
            return

        self._trace.end_time = _now()

        # Cerrar observations sin end_time
        for obs in self._trace.observations:
            if obs.end_time is None:
                obs.end_time = _now()

        # Enqueue todo
        tracer._enqueue(self._trace)
        for obs in self._trace.observations:
            tracer._enqueue(obs)

        tracer.flush()

        # Limpiar contexto
        _current_trace.set(None)
        _current_observation.set(None)

    def get_trace_id(self) -> str | None:
        """Retorna el trace_id para linking externo."""
        return self._trace.id if self._trace else None

    def get_trace_url(self, base_url: str = "http://localhost:7600") -> str | None:
        """Retorna URL del trace en el dashboard."""
        if self._trace:
            return f"{base_url}/#/traces/{self._trace.id}"
        return None
