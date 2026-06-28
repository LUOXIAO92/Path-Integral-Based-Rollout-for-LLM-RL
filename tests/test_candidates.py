from __future__ import annotations

from src.candidates import build_candidate_from_judgement, failed_candidate_from_rollout
from src.schemas import RewardEvaluation, RolloutRecord, ScoreConfig
from tests.test_helpers import ANSWER, valid_reward_payload


def rollout_record() -> RolloutRecord:
    return RolloutRecord(
        run_id="run",
        problem_id="p1",
        path_id="p1-0000",
        rollout_index=0,
        path_text=ANSWER,
        token_logprobs=[-0.1, -0.2],
        is_valid=True,
    )


def test_build_candidate_from_judgement_computes_scores_and_metrics() -> None:
    candidate = build_candidate_from_judgement(
        rollout=rollout_record(),
        evaluation=RewardEvaluation.model_validate(valid_reward_payload()),
        score_config=ScoreConfig(),
        eta=1.0,
        lambda_g=1.0,
        lambda_n=1.0,
        lambda_kl=0.0,
        length_max=10,
        length_scale=2.0,
        reward_attempts=2,
    )

    assert candidate.reward_valid is True
    assert candidate.reward_attempts == 2
    assert candidate.g == 1.0
    assert candidate.s0 == 0.30000000000000004
    assert candidate.s_eta == -0.7
    assert candidate.final_correctness is True


def test_failed_candidate_from_rollout_preserves_rollout_text_and_error() -> None:
    rollout = rollout_record().model_copy(update={"is_valid": False, "error": "student failed"})
    candidate = failed_candidate_from_rollout(rollout, rollout.error)

    assert candidate.reward_valid is False
    assert candidate.path_text == ANSWER
    assert candidate.error == "student failed"
    assert candidate.s_eta is None
