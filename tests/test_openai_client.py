from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.openai_client import evaluate_reward


def test_evaluate_reward_omits_none_extra_body() -> None:
    calls = []
    client = fake_client(calls, content="{}")

    result = asyncio.run(
        evaluate_reward(
            client=client,
            semaphore=asyncio.Semaphore(1),
            model="reward",
            messages=[{"role": "user", "content": "judge"}],
            temperature=0.0,
            max_tokens=128,
            response_format=None,
            extra_body=None,
        )
    )

    assert result == "{}"
    assert "extra_body" not in calls[0]


def test_evaluate_reward_passes_extra_body_when_set() -> None:
    calls = []
    client = fake_client(calls, content="{}")

    asyncio.run(
        evaluate_reward(
            client=client,
            semaphore=asyncio.Semaphore(1),
            model="reward",
            messages=[{"role": "user", "content": "judge"}],
            temperature=0.0,
            max_tokens=128,
            response_format={"type": "json_object"},
            extra_body={"guided_json": True},
        )
    )

    assert calls[0]["extra_body"] == {"guided_json": True}
    assert calls[0]["response_format"] == {"type": "json_object"}


def fake_client(calls: list[dict], content: str):
    async def create(**kwargs):
        calls.append(kwargs)
        choice = SimpleNamespace(
            message=SimpleNamespace(content=content),
        )
        return SimpleNamespace(choices=[choice])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
