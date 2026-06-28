from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from src.schemas import ProblemInput, RewardEvaluation


def load_reward_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_reward_prompt(template: str, problem: ProblemInput, student_answer: str) -> str:
    text = template
    replacements = {
        "{{subject}}": problem.subject,
        "{{problem}}": problem.problem,
        "{{reference_answer}}": problem.reference_answer,
        "{{student_answer}}": student_answer,
        "{{test_result}}": problem.test_result,
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def reward_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "reward_evaluation",
            "strict": True,
            "schema": RewardEvaluation.model_json_schema(),
        },
    }


def parse_reward_evaluation(raw_text: str) -> RewardEvaluation:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"reward output is not valid JSON: {exc}") from exc
    try:
        return RewardEvaluation.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"reward output does not match RewardEvaluation schema: {exc}") from exc
