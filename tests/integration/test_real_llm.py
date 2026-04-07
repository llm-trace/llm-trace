"""
Test con LLM real: ChatOpenAI + LangGraph + llm-trace

    export OPENAI_API_KEY=sk-...
    pip install langchain-openai
    python tests/test_real_llm.py
"""

import pytest

pytestmark = pytest.mark.integration

import os
import sys
import time
from typing import Annotated, Literal, TypedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from llm_trace import flush, score, tracer
from llm_trace.langchain import CallbackHandler

# ── Tools ─────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {
        "valencia": "22°C, sunny, humidity 65%",
        "madrid": "18°C, cloudy, humidity 45%",
        "barcelona": "20°C, partly cloudy, humidity 70%",
    }
    return data.get(city.lower(), f"No data for {city}")

@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [get_weather, calculate]

# ── Graph ─────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def agent(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

def should_continue(state: State) -> Literal["tools", "__end__"]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

g = StateGraph(State)
g.add_node("agent", agent)
g.add_node("tools", ToolNode(tools))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
g.add_edge("tools", "agent")
graph = g.compile()

# ── Run ───────────────────────────────────────────────────

print("=" * 50)
print("🔍 llm-trace — LLM real (gpt-4o-mini)")
print("=" * 50)

# Llamada 1: con tools
h1 = CallbackHandler(session_id="real", trace_name="weather-calc", tags=["real", "tools"])
r1 = graph.invoke(
    {"messages": [HumanMessage(content="What's the weather in Valencia and how much is 42*17?")]},
    config={"callbacks": [h1]},
)
print(f"\n1. {r1['messages'][-1].content[:150]}")
score("quality", 0.9, trace_id=h1.get_trace_id())
h1.flush()

# Llamada 2: directa
h2 = CallbackHandler(session_id="real", trace_name="explain", tags=["real", "direct"])
r2 = graph.invoke(
    {"messages": [HumanMessage(content="What is LLM observability? Answer in 2 sentences.")]},
    config={"callbacks": [h2]},
)
print(f"2. {r2['messages'][-1].content[:150]}")
score("relevance", 0.85, trace_id=h2.get_trace_id())
h2.flush()

# Resultados
time.sleep(1)
flush()
stats = tracer.storage.get_stats()
print(f"\n📊 {stats['trace_count']} traces, {stats['total_tokens']} tokens, ${stats['total_cost']}, {stats['avg_latency_ms']}ms avg")
print(f"   Models: {[m['model'] for m in stats['top_models']]}")
print('\n🚀 python -c "from llm_trace import tracer; tracer.dashboard(port=7600, open_browser=False)"')
