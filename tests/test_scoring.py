from __future__ import annotations

from src.schemas import RewardEvaluation, ScoreConfig
from src.scoring import score_reward_evaluation
from tests.test_helpers import valid_reward_payload


def test_scoring_combines_process_and_final_score() -> None:
    evaluation = RewardEvaluation.model_validate(valid_reward_payload())
    scored = score_reward_evaluation(evaluation, ScoreConfig())
    assert scored.process_score == 1.0
    assert scored.final_score == 1.0
    assert scored.g == 1.0
    assert scored.final_correctness is True


def test_scoring_keeps_process_score_when_final_answer_wrong() -> None:
    payload = valid_reward_payload()
    payload["final_answer_check"]["is_correct"] = False
    evaluation = RewardEvaluation.model_validate(payload)
    scored = score_reward_evaluation(evaluation, ScoreConfig())
    assert scored.process_score == 1.0
    assert scored.final_score == 0.0
    assert scored.g == 0.4


def test_scoring_off_task_returns_off_task_score() -> None:
    payload = valid_reward_payload()
    payload["student_alignment"]["off_task"] = True
    evaluation = RewardEvaluation.model_validate(payload)
    scored = score_reward_evaluation(evaluation, ScoreConfig(off_task_score=0.0))
    assert scored.g == 0.0


def test_scoring_uses_no_scored_span_default_when_only_final_span_counts() -> None:
    payload = valid_reward_payload()
    payload["span_evaluations"][0]["is_relevant"] = False
    payload["span_evaluations"][0]["is_key_reasoning"] = False
    payload["span_evaluations"][0]["step_score"] = 0.0
    evaluation = RewardEvaluation.model_validate(payload)
    scored = score_reward_evaluation(
        evaluation,
        ScoreConfig(no_scored_span_process_score=0.25),
    )
    assert scored.process_score == 0.25
