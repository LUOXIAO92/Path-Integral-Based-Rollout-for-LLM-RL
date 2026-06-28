from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from src.openai_client import evaluate_reward
from src.rewarding import build_reward_prompt, parse_reward_evaluation
from src.schemas import ProblemInput, RewardEvaluation, RolloutRecord
from src.validation import validate_reward_evaluation


async def judge_rollout(
    problem: ProblemInput,
    rollout: RolloutRecord,
    reward_template: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    response_format: dict | None,
    extra_body: dict | None,
) -> tuple[RewardEvaluation | None, list[dict], int, str | None]:
    if not rollout.is_valid:
        return None, [], 0, rollout.error

    raw_rows: list[dict] = []
    last_error = None
    for attempt in range(1, max_retries + 2):
        raw_text = ""
        try:
            raw_text = await evaluate_reward(
                client=client,
                semaphore=semaphore,
                model=model,
                messages=reward_messages(reward_template, problem, rollout.path_text),
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=extra_body,
            )
            evaluation = parse_reward_evaluation(raw_text)
            validation = validate_reward_evaluation(evaluation, problem.problem, rollout.path_text)
            raw_rows.append(
                {
                    "run_id": rollout.run_id,
                    "problem_id": rollout.problem_id,
                    "path_id": rollout.path_id,
                    "attempt": attempt,
                    "raw_text": raw_text,
                    "parsed": True,
                    "valid": validation.ok,
                    "errors": validation.errors,
                }
            )
            if validation.ok:
                return evaluation, raw_rows, attempt, None
            last_error = "reward_validation_failed: " + "; ".join(validation.errors)
        except Exception as exc:
            last_error = f"reward_attempt_failed: {exc}"
            raw_rows.append(
                {
                    "run_id": rollout.run_id,
                    "problem_id": rollout.problem_id,
                    "path_id": rollout.path_id,
                    "attempt": attempt,
                    "raw_text": raw_text,
                    "parsed": False,
                    "valid": False,
                    "errors": [str(exc)],
                }
            )

    return None, raw_rows, max_retries + 1, last_error or "reward_failed"


def reward_messages(template: str, problem: ProblemInput, student_answer: str) -> list[dict]:
    return [{"role": "user", "content": build_reward_prompt(template, problem, student_answer)}]
