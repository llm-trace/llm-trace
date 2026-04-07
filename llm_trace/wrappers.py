"""Wrappers drop-in para OpenAI y Anthropic SDKs.

Interceptan llamadas a los SDKs y crean observations de tipo
generation con tracking completo de tokens, costos y latencia.

Usage:
    from llm_trace.wrappers import wrap_openai
    import openai

    client = wrap_openai(openai.OpenAI())
    # Usa client normalmente — todo se traza automáticamente
"""

from __future__ import annotations

import functools
import time
from datetime import datetime, timezone
from typing import Any

from llm_trace.core import (
    _current_observation,
    _current_trace,
    tracer,
)
from llm_trace.models import (
    CostDetails,
    Observation,
    ObservationType,
    UsageDetails,
    _new_id,
    _now,
)

# ── Pricing table ─────────────────────────────────────────
# Precios por 1M tokens (input, output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o1-pro": (150.00, 600.00),
    "o3": (10.00, 40.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Anthropic
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # Aliases cortos
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
}


def _calculate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> CostDetails:
    """Calcula el costo basado en el modelo y tokens."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        # Intentar match parcial
        for key, val in MODEL_PRICING.items():
            if key in model or model in key:
                pricing = val
                break

    if not pricing:
        return CostDetails()

    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return CostDetails(
        input_cost=round(input_cost, 8),
        output_cost=round(output_cost, 8),
        total_cost=round(input_cost + output_cost, 8),
    )


# ── OpenAI wrapper ────────────────────────────────────────


def wrap_openai(client: Any) -> Any:
    """Envuelve un cliente OpenAI para trazado automático.

    Args:
        client: Instancia de openai.OpenAI() o openai.AsyncOpenAI()

    Returns:
        El mismo cliente con métodos instrumentados.
    """
    _wrap_openai_chat(client)
    _wrap_openai_embeddings(client)
    return client


def _wrap_openai_chat(client: Any) -> None:
    """Instrumenta chat.completions.create."""
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        return

    original_create = client.chat.completions.create

    @functools.wraps(original_create)
    def traced_create(*args: Any, **kwargs: Any) -> Any:
        trace = _current_trace.get()
        parent = _current_observation.get()

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)

        obs = Observation(
            trace_id=trace.id if trace else _new_id(),
            parent_id=parent.id if parent else None,
            type=ObservationType.GENERATION,
            name=f"openai.chat.{model}",
            input=messages,
            model=model,
            model_parameters={
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "stop",
                    "response_format",
                )
            },
        )

        if trace:
            trace.observations.append(obs)

        prev_obs = _current_observation.set(obs)

        try:
            response = original_create(*args, **kwargs)

            if stream:
                return _wrap_openai_stream(response, obs)

            # Sync response
            obs.end_time = _now()
            _extract_openai_response(response, obs)
            tracer._enqueue(obs)
            return response

        except Exception as e:
            obs.end_time = _now()
            obs.status = "error"
            obs.level = "ERROR"
            obs.error_message = str(e)
            tracer._enqueue(obs)
            raise
        finally:
            _current_observation.reset(prev_obs)

    client.chat.completions.create = traced_create


def _extract_openai_response(response: Any, obs: Observation) -> None:
    """Extrae datos de un ChatCompletion response."""
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message"):
            obs.output = {
                "role": getattr(choice.message, "role", "assistant"),
                "content": getattr(choice.message, "content", ""),
            }

    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        input_t = getattr(usage, "prompt_tokens", 0)
        output_t = getattr(usage, "completion_tokens", 0)
        cached = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = 0
        if cached and hasattr(cached, "cached_tokens"):
            cached_tokens = cached.cached_tokens or 0

        reasoning = 0
        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details and hasattr(
            completion_details, "reasoning_tokens"
        ):
            reasoning = completion_details.reasoning_tokens or 0

        obs.usage = UsageDetails(
            input_tokens=input_t,
            output_tokens=output_t,
            total_tokens=input_t + output_t,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning,
        )
        obs.cost = _calculate_cost(obs.model or "", input_t, output_t)

    if hasattr(response, "model"):
        obs.model = response.model


def _wrap_openai_stream(stream: Any, obs: Observation) -> Any:
    """Wrapper para streaming responses que captura TTFT y chunks."""
    chunks_content: list[str] = []
    first_chunk = True

    for chunk in stream:
        if first_chunk:
            obs.completion_start_time = _now()
            first_chunk = False

        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                chunks_content.append(delta.content)

        # Capturar usage del último chunk (OpenAI stream_options)
        if hasattr(chunk, "usage") and chunk.usage:
            _extract_openai_response(chunk, obs)

        yield chunk

    obs.end_time = _now()
    if chunks_content:
        obs.output = {"role": "assistant", "content": "".join(chunks_content)}
    tracer._enqueue(obs)


def _wrap_openai_embeddings(client: Any) -> None:
    """Instrumenta embeddings.create."""
    if not hasattr(client, "embeddings"):
        return

    original = client.embeddings.create

    @functools.wraps(original)
    def traced_embeddings(*args: Any, **kwargs: Any) -> Any:
        trace = _current_trace.get()
        parent = _current_observation.get()
        model = kwargs.get("model", "text-embedding-3-small")

        obs = Observation(
            trace_id=trace.id if trace else _new_id(),
            parent_id=parent.id if parent else None,
            type=ObservationType.EMBEDDING,
            name=f"openai.embedding.{model}",
            model=model,
            input=kwargs.get("input", "")[:200],  # Truncar input largo
        )

        if trace:
            trace.observations.append(obs)

        try:
            response = original(*args, **kwargs)
            obs.end_time = _now()

            if hasattr(response, "usage"):
                obs.usage = UsageDetails(
                    input_tokens=getattr(response.usage, "prompt_tokens", 0),
                    total_tokens=getattr(response.usage, "total_tokens", 0),
                )
            obs.output = f"{len(response.data)} embeddings"
            tracer._enqueue(obs)
            return response

        except Exception as e:
            obs.end_time = _now()
            obs.status = "error"
            obs.error_message = str(e)
            tracer._enqueue(obs)
            raise

    client.embeddings.create = traced_embeddings


# ── Anthropic wrapper ─────────────────────────────────────


def wrap_anthropic(client: Any) -> Any:
    """Envuelve un cliente Anthropic para trazado automático.

    Args:
        client: Instancia de anthropic.Anthropic() o anthropic.AsyncAnthropic()

    Returns:
        El mismo cliente con métodos instrumentados.
    """
    if not hasattr(client, "messages"):
        return client

    original_create = client.messages.create

    @functools.wraps(original_create)
    def traced_create(*args: Any, **kwargs: Any) -> Any:
        trace = _current_trace.get()
        parent = _current_observation.get()

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", None)

        obs = Observation(
            trace_id=trace.id if trace else _new_id(),
            parent_id=parent.id if parent else None,
            type=ObservationType.GENERATION,
            name=f"anthropic.messages.{model}",
            input={"system": system, "messages": messages}
            if system
            else messages,
            model=model,
            model_parameters={
                k: v
                for k, v in kwargs.items()
                if k in ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences")
            },
        )

        if trace:
            trace.observations.append(obs)

        prev_obs = _current_observation.set(obs)

        try:
            response = original_create(*args, **kwargs)

            obs.end_time = _now()

            # Extraer output
            if hasattr(response, "content") and response.content:
                content_blocks = []
                for block in response.content:
                    if hasattr(block, "text"):
                        content_blocks.append(block.text)
                    elif hasattr(block, "type"):
                        content_blocks.append(f"[{block.type}]")
                obs.output = (
                    content_blocks[0]
                    if len(content_blocks) == 1
                    else content_blocks
                )

            # Extraer usage
            if hasattr(response, "usage"):
                usage = response.usage
                input_t = getattr(usage, "input_tokens", 0)
                output_t = getattr(usage, "output_tokens", 0)
                cached = getattr(usage, "cache_read_input_tokens", 0)

                obs.usage = UsageDetails(
                    input_tokens=input_t,
                    output_tokens=output_t,
                    total_tokens=input_t + output_t,
                    cached_tokens=cached,
                )
                obs.cost = _calculate_cost(
                    obs.model or "", input_t, output_t
                )

            if hasattr(response, "model"):
                obs.model = response.model

            tracer._enqueue(obs)
            return response

        except Exception as e:
            obs.end_time = _now()
            obs.status = "error"
            obs.level = "ERROR"
            obs.error_message = str(e)
            tracer._enqueue(obs)
            raise
        finally:
            _current_observation.reset(prev_obs)

    client.messages.create = traced_create
    return client
