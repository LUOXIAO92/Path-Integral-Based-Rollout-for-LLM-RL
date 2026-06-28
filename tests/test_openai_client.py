from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.openai_client import evaluate_reward, generate_student_path


def test_generate_student_path_passes_extra_body() -> None:
    calls = []
    client = fake_client(calls, content="answer", logprobs=[-0.1, -0.2])

    result = asyncio.run(
        generate_student_path(
            client=client,
            semaphore=asyncio.Semaphore(1),
            model="student",
            messages=[{"role": "user", "content": "problem"}],
            temperature=0.8,
            top_p=0.95,
            max_tokens=128,
            extra_body={"top_k": 20},
        )
    )

    assert result.path_text == "answer"
    assert result.token_logprobs == [-0.1, -0.2]
    assert calls[0]["extra_body"] == {"top_k": 20}
    assert calls[0]["logprobs"] is True


def test_evaluate_reward_omits_none_extra_body() -> None:
    calls = []
    client = fake_client(calls, content="{}", logprobs=None)

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
    client = fake_client(calls, content="{}", logprobs=None)

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


def fake_client(calls: list[dict], content: str, logprobs: list[float] | None):
    async def create(**kwargs):
        calls.append(kwargs)
        choice = SimpleNamespace(
            message=SimpleNamespace(content=content),
            logprobs=make_logprobs(logprobs),
        )
        return SimpleNamespace(choices=[choice])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def make_logprobs(values: list[float] | None):
    if values is None:
        return None
    return SimpleNamespace(content=[SimpleNamespace(logprob=value) for value in values])
