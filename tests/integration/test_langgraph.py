"""
Test limpio: LangGraph + llm-trace

1 ejecución = 1 trace. Sin decoradores en los nodos.
Solo pasas el CallbackHandler y ya.

Ejecutar:
    cd llm-trace
    python tests/test_langgraph.py
"""

import pytest

pytestmark = pytest.mark.integration

import os
import sys
import time
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reset singleton
from llm_trace.core import Tracer

Tracer._instance = None

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from llm_trace import flush, observe, score, tracer
from llm_trace.langchain import CallbackHandler
from llm_trace.models import ObservationType

# ═══════════════════════════════════════════════════════════
# Herramientas
# ═══════════════════════════════════════════════════════════

@tool
def buscar_clima(ciudad: str) -> str:
    """Busca el clima actual de una ciudad."""
    climas = {
        "valencia": "22°C, soleado, humedad 65%",
        "madrid": "18°C, nublado, humedad 45%",
        "barcelona": "20°C, parcialmente nublado, humedad 70%",
    }
    return climas.get(ciudad.lower(), f"Sin datos para {ciudad}")


@tool
def calcular(expresion: str) -> str:
    """Calcula una expresión matemática."""
    try:
        return f"Resultado: {eval(expresion)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def buscar_documentos(query: str) -> str:
    """Busca documentos en la base de conocimiento."""
    docs = {
        "observabilidad": "La observabilidad en LLMs permite monitorizar trazas, tokens, costos y latencia.",
        "langgraph": "LangGraph construye agentes stateful con grafos dirigidos sobre LangChain.",
    }
    for key, doc in docs.items():
        if key in query.lower():
            return doc
    return f"Sin resultados para: {query}"


tools = [buscar_clima, calcular, buscar_documentos]


# ═══════════════════════════════════════════════════════════
# Estado y nodos — SIN decoradores, funciones puras
# ═══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


_call_count = 0


def agent_node(state: AgentState) -> dict:
    """Nodo agente: simula LLM que decide usar herramientas."""
    global _call_count
    _call_count += 1
    time.sleep(0.05)

    if _call_count % 2 == 1:
        return {"messages": [AIMessage(
            content="Voy a buscar info y calcular.",
            tool_calls=[
                {"id": f"call_{_call_count}a", "name": "buscar_clima",
                 "args": {"ciudad": "Valencia"}},
                {"id": f"call_{_call_count}b", "name": "calcular",
                 "args": {"expresion": "25*4"}},
            ],
        )]}
    else:
        return {"messages": [AIMessage(
            content="🌤️ Valencia: 22°C, soleado\n🔢 25×4 = 100",
        )]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ═══════════════════════════════════════════════════════════
# Construir grafo
# ═══════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


# ═══════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════

def run():
    graph = build_graph()

    print("=" * 60)
    print("🔍 llm-trace + LangGraph — test limpio")
    print("=" * 60)

    # ── Test 1: Solo CallbackHandler, nada más ───────────
    print("\n📋 Test 1: Una ejecución = un trace")
    print("-" * 40)

    handler = CallbackHandler(
        session_id="session-001",
        user_id="marc",
        trace_name="agente-clima",
        tags=["test", "langgraph"],
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content="¿Clima en Valencia y cuánto es 25*4?")]},
        config={"callbacks": [handler]},
    )

    score("quality", 0.92, trace_id=handler.get_trace_id())
    handler.flush()

    print(f"  Respuesta: {result['messages'][-1].content}")
    print(f"  Trace ID:  {handler.get_trace_id()}")

    # ── Test 2: Segunda ejecución = segundo trace ────────
    print("\n📋 Test 2: Segunda ejecución, trace independiente")
    print("-" * 40)

    handler2 = CallbackHandler(
        session_id="session-001",  # Misma sesión
        user_id="marc",
        trace_name="agente-docs",
        tags=["test", "rag"],
    )

    # Nuevo graph para RAG
    _rag_step = [0]
    def rag_node(s):
        _rag_step[0] += 1
        time.sleep(0.03)
        if _rag_step[0] == 1:
            return {"messages": [AIMessage(
                content="Busco info.",
                tool_calls=[{"id": "r1", "name": "buscar_documentos",
                            "args": {"query": "observabilidad"}}])]}
        return {"messages": [AIMessage(
            content="La observabilidad en LLMs permite monitorizar trazas y costos.")]}

    g2 = StateGraph(AgentState)
    g2.add_node("agent", rag_node)
    g2.add_node("tools", ToolNode(tools))
    g2.add_edge(START, "agent")
    g2.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g2.add_edge("tools", "agent")

    result2 = g2.compile().invoke(
        {"messages": [HumanMessage(content="¿Qué es observabilidad?")]},
        config={"callbacks": [handler2]},
    )

    score("relevance", 0.88, trace_id=handler2.get_trace_id())
    handler2.flush()

    print(f"  Respuesta: {result2['messages'][-1].content}")
    print(f"  Trace ID:  {handler2.get_trace_id()}")

    # ── Test 3: @observe() DENTRO de un nodo (opcional) ──
    print("\n📋 Test 3: @observe() opcional dentro de nodo")
    print("-" * 40)
    print("  (demuestra que se anida bajo el trace del handler)")

    @observe(as_type=ObservationType.GUARDRAIL)
    def check_safety(text: str) -> bool:
        """Sub-paso decorado dentro de un nodo."""
        time.sleep(0.02)
        return "hack" not in text.lower()

    def agent_with_guardrail(s):
        time.sleep(0.03)
        # Este @observe se anida bajo el trace del handler
        is_safe = check_safety(s["messages"][-1].content)
        return {"messages": [AIMessage(
            content=f"Respuesta segura (safe={is_safe})")]}

    g3 = StateGraph(AgentState)
    g3.add_node("agent", agent_with_guardrail)
    g3.add_edge(START, "agent")
    g3.add_edge("agent", END)

    handler3 = CallbackHandler(
        trace_name="agent-con-guardrail",
        tags=["test", "guardrail"],
    )

    result3 = g3.compile().invoke(
        {"messages": [HumanMessage(content="Hola, ¿cómo estás?")]},
        config={"callbacks": [handler3]},
    )

    handler3.flush()
    print(f"  Respuesta: {result3['messages'][-1].content}")
    print(f"  Trace ID:  {handler3.get_trace_id()}")

    # ═══════════════════════════════════════════════════════
    # Resultados
    # ═══════════════════════════════════════════════════════

    time.sleep(1)
    flush()

    stats = tracer.storage.get_stats()
    traces = tracer.storage.list_traces()

    print("\n" + "=" * 60)
    print("📊 RESULTADOS")
    print("=" * 60)
    print(f"  Traces:       {stats['trace_count']}")
    print(f"  Observations: {stats['observation_count']}")
    print(f"  Scores:       {stats['score_count']}")
    print(f"  Error rate:   {stats['error_rate']}%")
    print()

    for t in traces:
        has_err = any(o.status == "error" for o in t.observations)
        status = "❌" if has_err else "✅"
        dur = f"{t.duration_ms:.0f}ms" if t.duration_ms else "-"
        sess = f"  session={t.session_id}" if t.session_id else ""
        print(f"  {status} {t.name}  [{dur}]{sess}  tags={t.tags}")

        for o in t.observations:
            d = f"{o.duration_ms:.0f}ms" if o.duration_ms else "-"
            parent = " (child)" if o.parent_id else ""
            print(f"     ├─ {o.type.value:12s} {o.name[:45]:45s} {d:>8s}{parent}")

        t_scores = tracer.storage.get_scores(trace_id=t.id)
        if t_scores:
            s_str = ", ".join(f"{s.name}={s.value}" for s in t_scores)
            print(f"     └─ scores: {s_str}")
        print()

    print(f"  📁 DB: {tracer.storage.db_path}")
    print(f"  📁 Size: {tracer.storage.db_path.stat().st_size / 1024:.1f} KB")
    print()
    print("  🚀 Dashboard: python -c \"from llm_trace import tracer; tracer.dashboard()\"")
    print()

    expected = 3
    actual = stats["trace_count"]
    if actual == expected:
        print(f"  ✅ {expected} ejecuciones = {actual} traces — CORRECTO")
    else:
        print(f"  ⚠️  {expected} ejecuciones pero {actual} traces")

    print()


if __name__ == "__main__":
    run()
