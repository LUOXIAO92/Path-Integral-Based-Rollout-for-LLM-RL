from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import RewardEvaluation
from tests.test_helpers import valid_reward_payload


def test_reward_evaluation_schema_accepts_valid_payload() -> None:
    evaluation = RewardEvaluation.model_validate(valid_reward_payload())
    assert evaluation.final_answer_check.is_correct is True


def test_reward_evaluation_schema_rejects_missing_required_field() -> None:
    payload = valid_reward_payload()
    del payload["student_alignment"]
    with pytest.raises(ValidationError):
        RewardEvaluation.model_validate(payload)


def test_reward_evaluation_schema_rejects_illegal_enum() -> None:
    payload = valid_reward_payload()
    payload["student_spans"][0]["span_type"] = "reasoning"
    with pytest.raises(ValidationError):
        RewardEvaluation.model_validate(payload)
