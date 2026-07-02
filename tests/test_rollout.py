from __future__ import annotations

import asyncio

from src.rollout import generate_rollout_record, run_student_rollouts, student_messages
from src.schemas import ProblemInput, StudentGeneration
from tests.test_helpers import ANSWER, PROBLEM


def test_student_messages_use_prompt_template() -> None:
    problem = ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)
    messages = student_messages(problem, "system", "Question: {problem}")
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": f"Question: {PROBLEM}"},
    ]


def test_generate_rollout_record_writes_valid_rollout(monkeypatch) -> None:
    import src.rollout as rollout

    async def fake_generate_student_path(**kwargs):
        return StudentGeneration(path_text=ANSWER, token_logprobs=[-0.1, -0.2])

    monkeypatch.setattr(rollout, "generate_student_path", fake_generate_student_path)
    problem = ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)

    record = asyncio.run(
        generate_rollout_record(
            run_id="run",
            problem=problem,
            rollout_index=2,
            client=object(),
            semaphore=asyncio.Semaphore(1),
            model="student",
            temperature=0.8,
            top_p=0.95,
            max_tokens=128,
            extra_body={"top_k": 20},
            system_prompt="system",
            user_template="{problem}",
        )
    )

    assert record.path_id == "p1-0002"
    assert record.path_text == ANSWER
    assert record.token_logprobs == [-0.1, -0.2]
    assert record.output_token_count == 2
    assert record.is_valid is True
    assert record.error is None


def test_generate_rollout_record_marks_missing_logprobs_invalid(monkeypatch) -> None:
    import src.rollout as rollout

    async def fake_generate_student_path(**kwargs):
        return StudentGeneration(path_text=ANSWER, token_logprobs=[])

    monkeypatch.setattr(rollout, "generate_student_path", fake_generate_student_path)
    problem = ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)

    record = asyncio.run(
        generate_rollout_record(
            run_id="run",
            problem=problem,
            rollout_index=0,
            client=object(),
            semaphore=asyncio.Semaphore(1),
            model="student",
            temperature=0.8,
            top_p=0.95,
            max_tokens=128,
            extra_body={"top_k": 20},
            system_prompt="system",
            user_template="{problem}",
        )
    )

    assert record.is_valid is False
    assert record.path_text == ANSWER
    assert record.token_logprobs == []
    assert "logprobs" in record.error


def test_run_student_rollouts_respects_budget(monkeypatch) -> None:
    import src.rollout as rollout

    async def fake_generate_student_path(**kwargs):
        return StudentGeneration(path_text=ANSWER, token_logprobs=[-0.1])

    monkeypatch.setattr(rollout, "generate_student_path", fake_generate_student_path)
    problems = [
        ProblemInput(problem_id="p1", subject="math", problem=PROBLEM),
        ProblemInput(problem_id="p2", subject="math", problem=PROBLEM),
    ]

    records = asyncio.run(
        run_student_rollouts(
            run_id="run",
            problems=problems,
            rollout_budget=2,
            client=object(),
            semaphore=asyncio.Semaphore(2),
            model="student",
            temperature=0.8,
            top_p=0.95,
            max_tokens=128,
            extra_body={"top_k": 20},
            system_prompt="system",
            user_template="{problem}",
        )
    )

    assert [record.path_id for record in records] == ["p1-0000", "p1-0001", "p2-0000", "p2-0001"]
    assert all(record.is_valid for record in records)
