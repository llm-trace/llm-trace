"""Ejemplos de uso de llm-trace.

Demuestra los 4 patrones principales:
1. @observe() decorator para funciones
2. wrap_openai() para OpenAI SDK
3. wrap_anthropic() para Anthropic SDK
4. score() para evaluaciones
"""

import asyncio
import time

from llm_trace import flush, observe, score, shutdown, tracer
from llm_trace.models import ObservationType
from llm_trace.wrappers import wrap_anthropic, wrap_openai

# ═══════════════════════════════════════════════════════════
# Ejemplo 1: Decoradores básicos con anidamiento
# ═══════════════════════════════════════════════════════════

@observe(as_type=ObservationType.RETRIEVER)
def search_documents(query: str) -> list[str]:
    """Simula búsqueda en vector store."""
    time.sleep(0.1)  # Simular latencia
    return [
        f"Document 1 about {query}",
        f"Document 2 about {query}",
    ]


@observe(as_type=ObservationType.GENERATION)
def generate_response(query: str, context: list[str]) -> str:
    """Simula generación de respuesta."""
    time.sleep(0.2)
    return f"Based on {len(context)} documents: Answer to '{query}'"


@observe(tags=["rag", "production"])
def rag_pipeline(query: str) -> str:
    """Pipeline RAG completo — crea un trace con observations anidadas."""
    docs = search_documents(query)
    response = generate_response(query, docs)

    # Adjuntar score al trace actual
    score("relevance", 0.92)
    score("grounded", True)

    return response


# ═══════════════════════════════════════════════════════════
# Ejemplo 2: Agente con herramientas (async)
# ═══════════════════════════════════════════════════════════

@observe(as_type=ObservationType.TOOL)
async def calculator_tool(expression: str) -> float:
    """Herramienta de cálculo."""
    await asyncio.sleep(0.05)
    return eval(expression)  # ⚠️ Solo para demo


@observe(as_type=ObservationType.TOOL)
async def weather_tool(city: str) -> dict:
    """Herramienta de clima."""
    await asyncio.sleep(0.1)
    return {"city": city, "temp": 22, "condition": "sunny"}


@observe(as_type=ObservationType.AGENT)
async def agent(query: str, session_id: str = "demo-session") -> str:
    """Agente que decide qué herramientas usar."""
    if "calcul" in query.lower():
        result = await calculator_tool("2 + 2")
        return f"El resultado es {result}"
    elif "weather" in query.lower() or "clima" in query.lower():
        data = await weather_tool("Valencia")
        return f"En {data['city']}: {data['temp']}°C, {data['condition']}"
    else:
        return f"No tengo una herramienta para: {query}"


# ═══════════════════════════════════════════════════════════
# Ejemplo 3: Con OpenAI SDK (wrapper)
# ═══════════════════════════════════════════════════════════

def openai_example():
    """Usa wrap_openai() para instrumentar el SDK."""
    try:
        import openai
        client = wrap_openai(openai.OpenAI())

        @observe(tags=["openai-demo"])
        def chat_with_gpt(question: str) -> str:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un asistente útil."},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
            )
            answer = response.choices[0].message.content
            score("quality", 0.9)
            return answer

        result = chat_with_gpt("¿Qué es LangGraph?")
        print(f"OpenAI response: {result[:100]}...")

    except ImportError:
        print("⚠️  openai no instalado — skip openai_example")


# ═══════════════════════════════════════════════════════════
# Ejemplo 4: Con Anthropic SDK (wrapper)
# ═══════════════════════════════════════════════════════════

def anthropic_example():
    """Usa wrap_anthropic() para instrumentar el SDK."""
    try:
        import anthropic
        client = wrap_anthropic(anthropic.Anthropic())

        @observe(tags=["anthropic-demo"])
        def chat_with_claude(question: str) -> str:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": question}],
            )
            return response.content[0].text

        result = chat_with_claude("Explica hexagonal architecture en 2 frases")
        print(f"Anthropic response: {result[:100]}...")

    except ImportError:
        print("⚠️  anthropic no instalado — skip anthropic_example")


# ═══════════════════════════════════════════════════════════
# Ejemplo 5: Integración manual (bajo nivel)
# ═══════════════════════════════════════════════════════════

def manual_example():
    """Usa la API de bajo nivel para control total."""
    from llm_trace.models import CostDetails, UsageDetails

    trace = tracer.create_trace(
        name="manual-trace",
        session_id="session-123",
        user_id="marc",
        tags=["manual", "demo"],
    )

    obs = tracer.create_observation(
        name="custom-llm-call",
        obs_type=ObservationType.GENERATION,
        input={"prompt": "Hello world"},
        model="gpt-4o",
    )

    # Simular trabajo
    time.sleep(0.15)

    tracer.end_observation(
        obs,
        output={"text": "Hello! How can I help?"},
        usage=UsageDetails(input_tokens=10, output_tokens=8, total_tokens=18),
        cost=CostDetails(input_cost=0.000025, output_cost=0.00008, total_cost=0.000105),
    )

    tracer.score("manual-quality", 0.95, trace_id=trace.id)
    tracer.end_trace(trace, output="Done")


# ═══════════════════════════════════════════════════════════
# Run all examples
# ═══════════════════════════════════════════════════════════

async def main():
    print("🔍 llm-trace examples\n")

    # 1. RAG pipeline
    print("1. RAG Pipeline:")
    result = rag_pipeline("¿Qué es observabilidad en LLMs?")
    print(f"   → {result}\n")

    # 2. Agent
    print("2. Agent con herramientas:")
    result = await agent("calcula 2+2")
    print(f"   → {result}")
    result = await agent("clima en Valencia")
    print(f"   → {result}\n")

    # 3. Manual
    print("3. Tracing manual:")
    manual_example()
    print("   → Done\n")

    # 4. OpenAI (si disponible)
    print("4. OpenAI wrapper:")
    openai_example()
    print()

    # 5. Anthropic (si disponible)
    print("5. Anthropic wrapper:")
    anthropic_example()
    print()

    # Flush antes de lanzar dashboard
    flush()
    time.sleep(1)  # Esperar a que el flush complete

    # Mostrar stats
    stats = tracer.storage.get_stats()
    print(f"📊 Stats: {stats['trace_count']} traces, "
          f"{stats['generation_count']} generations, "
          f"{stats['score_count']} scores")
    print(f"   DB: {tracer.storage.db_path}\n")

    # Lanzar dashboard
    print("🚀 Launching dashboard...")
    tracer.dashboard(port=7600, open_browser=False)


if __name__ == "__main__":
    asyncio.run(main())
