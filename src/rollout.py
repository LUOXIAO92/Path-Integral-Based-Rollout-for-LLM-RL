from __future__ import annotations

import asyncio
from collections.abc import Sequence

from openai import AsyncOpenAI

from src.openai_client import generate_student_path
from src.schemas import ProblemInput, RolloutRecord


def student_messages(problem: ProblemInput, system_prompt: str, user_template: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(problem=problem.problem)},
    ]


async def run_student_rollouts(
    run_id: str,
    problems: Sequence[ProblemInput],
    rollout_budget: int,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: dict | None,
    system_prompt: str,
    user_template: str,
) -> list[RolloutRecord]:
    tasks = []
    for problem in problems:
        for rollout_index in range(rollout_budget):
            tasks.append(
                generate_rollout_record(
                    run_id=run_id,
                    problem=problem,
                    rollout_index=rollout_index,
                    client=client,
                    semaphore=semaphore,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    system_prompt=system_prompt,
                    user_template=user_template,
                )
            )
    return list(await asyncio.gather(*tasks))


async def generate_rollout_record(
    run_id: str,
    problem: ProblemInput,
    rollout_index: int,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: dict | None,
    system_prompt: str,
    user_template: str,
) -> RolloutRecord:
    path_id = f"{problem.problem_id}-{rollout_index:04d}"
    try:
        generation = await generate_student_path(
            client=client,
            semaphore=semaphore,
            model=model,
            messages=student_messages(problem, system_prompt, user_template),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
    except Exception as exc:
        return RolloutRecord(
            run_id=run_id,
            problem_id=problem.problem_id,
            path_id=path_id,
            rollout_index=rollout_index,
            is_valid=False,
            error=f"student_generation_failed: {exc}",
        )

    if not generation.token_logprobs:
        return RolloutRecord(
            run_id=run_id,
            problem_id=problem.problem_id,
            path_id=path_id,
            rollout_index=rollout_index,
            path_text=generation.path_text,
            is_valid=False,
            error="student response did not include token logprobs",
        )

    output_token_count = len(generation.token_logprobs)
    return RolloutRecord(
        run_id=run_id,
        problem_id=problem.problem_id,
        path_id=path_id,
        rollout_index=rollout_index,
        path_text=generation.path_text,
        token_logprobs=generation.token_logprobs,
        output_token_count=output_token_count,
        is_valid=True,
    )
