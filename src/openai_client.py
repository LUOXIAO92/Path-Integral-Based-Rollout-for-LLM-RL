from __future__ import annotations

import asyncio
from collections.abc import Sequence

from openai import AsyncOpenAI

from src.schemas import StudentGeneration


def make_async_client(api_key: str, base_url: str) -> AsyncOpenAI:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


async def generate_student_path(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    messages: Sequence[dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: dict | None,
) -> StudentGeneration:
    kwargs = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "logprobs": True,
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    async with semaphore:
        response = await client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    path_text = choice.message.content or ""
    token_logprobs = _extract_token_logprobs(choice)
    if not token_logprobs:
        raise ValueError("student response did not include token logprobs")
    return StudentGeneration(path_text=path_text, token_logprobs=token_logprobs)


async def evaluate_reward(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    messages: Sequence[dict],
    temperature: float,
    max_tokens: int,
    response_format: dict | None,
    extra_body: dict | None,
) -> str:
    kwargs = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    async with semaphore:
        response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _extract_token_logprobs(choice) -> list[float]:
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        return []
    content = getattr(logprobs, "content", None)
    if not content:
        return []
    values: list[float] = []
    for item in content:
        value = getattr(item, "logprob", None)
        if value is not None:
            values.append(float(value))
    return values
