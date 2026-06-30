from __future__ import annotations

import asyncio

from src.candidates import build_candidate_from_judgement, failed_candidate_from_rollout, rollout_has_required_logprobs
from src.judging import judge_rollout
from src.schemas import PathRecord, ProblemInput, RolloutRecord, ScoreConfig


async def build_candidate_for_rollout(
    problem: ProblemInput,
    rollout: RolloutRecord,
    reward_template: str,
    reward_client,
    reward_semaphore: asyncio.Semaphore,
    reward_model: str,
    reward_temperature: float,
    reward_max_tokens: int,
    reward_max_retries: int,
    response_format: dict | None,
    reward_extra_body: dict | None,
    score_config: ScoreConfig,
    eta: float,
    lambda_g: float,
    lambda_n: float,
    lambda_kl: float,
    length_max: int,
    length_scale: float,
) -> tuple[PathRecord, list[dict]]:
    if not rollout.is_valid:
        return failed_candidate_from_rollout(rollout, rollout.error), []
    if not rollout_has_required_logprobs(rollout):
        return failed_candidate_from_rollout(
            rollout,
            "rollout_missing_raw_or_proposal_logprobs",
        ), []

    evaluation, raw_rows, attempts, error = await judge_rollout(
        problem=problem,
        rollout=rollout,
        reward_template=reward_template,
        client=reward_client,
        semaphore=reward_semaphore,
        model=reward_model,
        temperature=reward_temperature,
        max_tokens=reward_max_tokens,
        max_retries=reward_max_retries,
        response_format=response_format,
        extra_body=reward_extra_body,
    )
    if evaluation is None:
        return failed_candidate_from_rollout(rollout, error, reward_attempts=attempts), raw_rows

    try:
        candidate = build_candidate_from_judgement(
            rollout=rollout,
            evaluation=evaluation,
            score_config=score_config,
            eta=eta,
            lambda_g=lambda_g,
            lambda_n=lambda_n,
            lambda_kl=lambda_kl,
            length_max=length_max,
            length_scale=length_scale,
            reward_attempts=attempts,
        )
    except Exception as exc:
        return failed_candidate_from_rollout(
            rollout,
            f"candidate_scoring_failed: {exc}",
            reward_attempts=attempts,
        ), raw_rows
    return candidate, raw_rows
