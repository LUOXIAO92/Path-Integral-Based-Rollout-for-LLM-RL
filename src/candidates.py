from __future__ import annotations

from src.metrics import compute_path_metrics
from src.schemas import PathRecord, RewardEvaluation, RolloutRecord, ScoreConfig
from src.scoring import score_reward_evaluation


def failed_candidate_from_rollout(
    rollout: RolloutRecord,
    error: str | None,
    reward_attempts: int = 0,
) -> PathRecord:
    return PathRecord(
        run_id=rollout.run_id,
        problem_id=rollout.problem_id,
        method="mcmc_candidate",
        path_id=rollout.path_id,
        path_text=rollout.path_text,
        reward_valid=False,
        reward_attempts=reward_attempts,
        error=error,
    )


def build_candidate_from_judgement(
    rollout: RolloutRecord,
    evaluation: RewardEvaluation,
    score_config: ScoreConfig,
    eta: float,
    lambda_g: float,
    lambda_n: float,
    lambda_kl: float,
    length_max: int,
    length_scale: float,
    reward_attempts: int,
) -> PathRecord:
    scored = score_reward_evaluation(evaluation, score_config)
    metrics = compute_path_metrics(
        token_logprobs=rollout.token_logprobs,
        g=scored.g,
        eta=eta,
        lambda_g=lambda_g,
        lambda_n=lambda_n,
        lambda_kl=lambda_kl,
        length_max=length_max,
        length_scale=length_scale,
    )
    return PathRecord(
        run_id=rollout.run_id,
        problem_id=rollout.problem_id,
        method="mcmc_candidate",
        path_id=rollout.path_id,
        path_text=rollout.path_text,
        g=scored.g,
        n=metrics.n,
        k=metrics.k,
        f=metrics.f,
        s0=metrics.s0,
        s_eta=metrics.s_eta,
        final_correctness=scored.final_correctness,
        reward_valid=True,
        reward_attempts=reward_attempts,
    )
