from __future__ import annotations

from src.schemas import RewardEvaluation
from src.validation import validate_reward_evaluation
from tests.test_helpers import ANSWER, PROBLEM, valid_reward_payload


def test_validator_accepts_valid_reward_output() -> None:
    evaluation = RewardEvaluation.model_validate(valid_reward_payload())
    result = validate_reward_evaluation(evaluation, PROBLEM, ANSWER)
    assert result.ok, result.errors


def test_validator_rejects_span_gap_even_when_schema_is_valid() -> None:
    payload = valid_reward_payload()
    payload["student_spans"][1]["raw_text_span"]["start_text"] = "answer: 42"
    payload["student_spans"][1]["raw_text_span"]["end_text"] = "answer: 42"
    evaluation = RewardEvaluation.model_validate(payload)
    result = validate_reward_evaluation(evaluation, PROBLEM, ANSWER)
    assert not result.ok
    assert any("overlap or gap" in error for error in result.errors)


def test_validator_rejects_unknown_reference() -> None:
    payload = valid_reward_payload()
    payload["span_evaluations"][0]["reference_point_refs"] = ["missing"]
    evaluation = RewardEvaluation.model_validate(payload)
    result = validate_reward_evaluation(evaluation, PROBLEM, ANSWER)
    assert not result.ok
    assert any("unknown id 'missing'" in error for error in result.errors)


def test_validator_rejects_missing_final_answer_span() -> None:
    payload = valid_reward_payload()
    payload["final_answer_check"]["student_final_answer_span_id"] = "s_missing"
    evaluation = RewardEvaluation.model_validate(payload)
    result = validate_reward_evaluation(evaluation, PROBLEM, ANSWER)
    assert not result.ok
    assert any("does not exist" in error for error in result.errors)


def test_validator_rejects_nonzero_final_answer_step_score() -> None:
    payload = valid_reward_payload()
    payload["span_evaluations"][1]["step_score"] = 0.5
    evaluation = RewardEvaluation.model_validate(payload)
    result = validate_reward_evaluation(evaluation, PROBLEM, ANSWER)
    assert not result.ok
    assert any("span_type=final_answer" in error for error in result.errors)
