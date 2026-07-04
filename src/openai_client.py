from __future__ import annotations

import asyncio
from collections.abc import Sequence

from openai import AsyncOpenAI


def make_async_client(api_key: str, base_url: str) -> AsyncOpenAI:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


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
