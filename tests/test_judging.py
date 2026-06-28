from __future__ import annotations

import asyncio
import json

from src.judging import judge_rollout
from src.schemas import ProblemInput, RolloutRecord
from tests.test_helpers import ANSWER, PROBLEM, valid_reward_payload


def valid_rollout() -> RolloutRecord:
    return RolloutRecord(
        run_id="run",
        problem_id="p1",
        path_id="p1-0000",
        rollout_index=0,
        path_text=ANSWER,
        token_logprobs=[-0.1],
        is_valid=True,
    )


def test_judge_rollout_skips_invalid_rollout(monkeypatch) -> None:
    import src.judging as judging

    async def fail_if_called(**kwargs):
        raise AssertionError("reward should not be called")

    monkeypatch.setattr(judging, "evaluate_reward", fail_if_called)
    problem = ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)
    rollout = valid_rollout().model_copy(update={"is_valid": False, "error": "student failed"})

    evaluation, raw_rows, attempts, error = asyncio.run(
        judge_rollout(
            problem=problem,
            rollout=rollout,
            reward_template="{{student_answer}}",
            client=object(),
            semaphore=asyncio.Semaphore(1),
            model="reward",
            temperature=0.0,
            max_tokens=128,
            max_retries=1,
            response_format=None,
            extra_body=None,
        )
    )

    assert evaluation is None
    assert raw_rows == []
    assert attempts == 0
    assert error == "student failed"


def test_judge_rollout_retries_after_validation_failure(monkeypatch) -> None:
    import src.judging as judging

    invalid_payload = valid_reward_payload()
    invalid_payload["student_spans"][0]["raw_text_span"]["start_text"] = "not in answer"
    responses = [json.dumps(invalid_payload), json.dumps(valid_reward_payload())]

    async def fake_evaluate_reward(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(judging, "evaluate_reward", fake_evaluate_reward)
    problem = ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)

    evaluation, raw_rows, attempts, error = asyncio.run(
        judge_rollout(
            problem=problem,
            rollout=valid_rollout(),
            reward_template="{{student_answer}}",
            client=object(),
            semaphore=asyncio.Semaphore(1),
            model="reward",
            temperature=0.0,
            max_tokens=128,
            max_retries=1,
            response_format={"type": "json_object"},
            extra_body={"guided_json": True},
        )
    )

    assert evaluation is not None
    assert len(raw_rows) == 2
    assert raw_rows[0]["parsed"] is True
    assert raw_rows[0]["valid"] is False
    assert raw_rows[1]["valid"] is True
    assert attempts == 2
    assert error is None


def test_judge_rollout_records_parse_failure(monkeypatch) -> None:
    import src.judging as judging

    async def fake_evaluate_reward(**kwargs):
        return "not json"

    monkeypatch.setattr(judging, "evaluate_reward", fake_evaluate_reward)
    problem = ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)

    evaluation, raw_rows, attempts, error = asyncio.run(
        judge_rollout(
            problem=problem,
            rollout=valid_rollout(),
            reward_template="{{student_answer}}",
            client=object(),
            semaphore=asyncio.Semaphore(1),
            model="reward",
            temperature=0.0,
            max_tokens=128,
            max_retries=0,
            response_format=None,
            extra_body=None,
        )
    )

    assert evaluation is None
    assert len(raw_rows) == 1
    assert raw_rows[0]["parsed"] is False
    assert attempts == 1
    assert error.startswith("reward_attempt_failed:")
