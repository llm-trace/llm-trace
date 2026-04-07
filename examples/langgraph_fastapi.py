"""Ejemplo de integración: LangGraph + FastAPI + llm-trace.

Demuestra el patrón exacto para un multi-LLM orchestration system
similar al CANCOM Assistant:
- FastAPI SSE streaming endpoint
- LangGraph workflow con nodos tipados
- CallbackHandler para trazabilidad automática
- Scoring en tiempo real
- Session tracking para replay conversacional

Requiere: pip install llm-trace[langchain] fastapi uvicorn langchain-openai langgraph
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# llm-trace imports
from llm_trace import flush, observe, score, tracer
from llm_trace.langchain import CallbackHandler
from llm_trace.models import ObservationType

# ═══════════════════════════════════════════════════════════
# Patrón 1: @observe() en nodos del graph
# ═══════════════════════════════════════════════════════════
# Cada nodo LangGraph se convierte en una observation tipada
# dentro del trace creado por el CallbackHandler.


@observe(as_type=ObservationType.AGENT)
async def router_node(state: dict) -> dict:
    """Nodo router que decide qué subgraph ejecutar."""
    query = state.get("query", "")

    # Lógica de routing basada en intent
    if any(word in query.lower() for word in ["precio", "coste", "factura"]):
        return {**state, "route": "billing"}
    elif any(word in query.lower() for word in ["error", "fallo", "problema"]):
        return {**state, "route": "support"}
    else:
        return {**state, "route": "general"}


@observe(as_type=ObservationType.RETRIEVER)
async def retriever_node(state: dict) -> dict:
    """Nodo de retrieval — busca contexto relevante."""
    query = state.get("query", "")
    # Simula búsqueda en vector store
    await asyncio.sleep(0.1)
    docs = [
        {"content": f"Documento relevante para: {query}", "score": 0.92},
        {"content": "Información de contexto adicional", "score": 0.85},
    ]
    return {**state, "context": docs}


@observe(as_type=ObservationType.GENERATION)
async def llm_node(state: dict) -> dict:
    """Nodo de generación — llama al LLM con contexto."""
    # En producción: usa tu provider configurado por YAML
    # client = get_provider(state.get("provider", "azure-openai"))
    await asyncio.sleep(0.2)
    response = f"Respuesta generada para: {state.get('query', '')}"

    # Score automático de la generación
    score("response_quality", 0.88)
    score("grounded", True, comment="Based on retrieved context")

    return {**state, "response": response}


@observe(as_type=ObservationType.GUARDRAIL)
async def guardrail_node(state: dict) -> dict:
    """Nodo guardrail — verifica seguridad de la respuesta."""
    response = state.get("response", "")
    # Simula verificación de contenido
    is_safe = "hack" not in response.lower()
    score("safety_check", is_safe)
    return {**state, "is_safe": is_safe}


# ═══════════════════════════════════════════════════════════
# Patrón 2: CallbackHandler con LangGraph
# ═══════════════════════════════════════════════════════════

app = FastAPI(title="CANCOM-style Assistant")


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Endpoint SSE con trazabilidad completa."""
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id", "default")
    user_id = body.get("user_id", "anonymous")

    # Crear handler con metadata de sesión
    handler = CallbackHandler(
        session_id=session_id,
        user_id=user_id,
        trace_name=f"chat:{query[:50]}",
        tags=["api", "chat"],
        metadata={
            "endpoint": "/api/chat",
            "query_length": len(query),
        },
    )

    async def stream_response():
        try:
            # Ejecutar pipeline
            state = {"query": query}

            # Simular ejecución del graph
            state = await router_node(state)
            yield f"data: {json.dumps({'event': 'route', 'route': state['route']})}\n\n"

            state = await retriever_node(state)
            yield f"data: {json.dumps({'event': 'context', 'docs': len(state.get('context', []))})}\n\n"

            state = await llm_node(state)
            yield f"data: {json.dumps({'event': 'response', 'text': state['response']})}\n\n"

            state = await guardrail_node(state)
            yield f"data: {json.dumps({'event': 'guardrail', 'safe': state['is_safe']})}\n\n"

            # Score final del trace
            score("user_satisfaction", 0.9, trace_id=handler.get_trace_id())

            yield f"data: {json.dumps({'event': 'done', 'trace_id': handler.get_trace_id()})}\n\n"

        finally:
            # CRÍTICO: flush al terminar el stream
            handler.flush()

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "X-Trace-ID": handler.get_trace_id() or "",
            "X-Trace-URL": handler.get_trace_url() or "",
        },
    )


@app.get("/api/traces")
async def list_traces(limit: int = 20, name: str | None = None):
    """Endpoint para consultar traces programáticamente."""
    traces = tracer.storage.list_traces(limit=limit, name=name)
    return [t.to_dict() for t in traces]


@app.get("/api/stats")
async def get_stats():
    """Estadísticas de observabilidad."""
    return tracer.storage.get_stats()


# ═══════════════════════════════════════════════════════════
# Patrón 3: Scoring como middleware
# ═══════════════════════════════════════════════════════════

@app.post("/api/feedback")
async def submit_feedback(request: Request):
    """Endpoint para feedback de usuarios (thumbs up/down)."""
    body = await request.json()
    from llm_trace.models import ScoreSource

    score(
        name="user_feedback",
        value=body.get("positive", True),
        trace_id=body.get("trace_id"),
        source=ScoreSource.USER_FEEDBACK,
        comment=body.get("comment"),
    )
    flush()
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════
# Patrón 4: LangGraph real con CompiledGraph
# ═══════════════════════════════════════════════════════════
# (Descomenta cuando tengas langgraph instalado)
#
# from langgraph.graph import StateGraph, END
# from langchain_openai import ChatOpenAI
#
# def build_graph():
#     graph = StateGraph(dict)
#     graph.add_node("router", router_node)
#     graph.add_node("retriever", retriever_node)
#     graph.add_node("llm", llm_node)
#     graph.add_node("guardrail", guardrail_node)
#
#     graph.set_entry_point("router")
#     graph.add_edge("router", "retriever")
#     graph.add_edge("retriever", "llm")
#     graph.add_edge("llm", "guardrail")
#     graph.add_edge("guardrail", END)
#
#     return graph.compile()
#
# compiled_graph = build_graph()
#
# # Uso con CallbackHandler:
# result = await compiled_graph.ainvoke(
#     {"query": "¿Cuánto cuesta el servicio?"},
#     config={"callbacks": [handler]}
# )
#
# # Uso con streaming:
# async for event in compiled_graph.astream(
#     {"query": "Tengo un error"},
#     config={"callbacks": [handler]},
#     stream_mode="updates",
# ):
#     yield event


# ═══════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("🔍 llm-trace dashboard: http://localhost:7600")
    print("🚀 API server:          http://localhost:8000")
    print()

    # Lanzar dashboard en background
    import threading

    threading.Thread(
        target=lambda: tracer.dashboard(port=7600, open_browser=False),
        daemon=True,
    ).start()

    # Lanzar API
    uvicorn.run(app, host="0.0.0.0", port=8000)
