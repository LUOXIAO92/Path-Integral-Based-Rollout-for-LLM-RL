from __future__ import annotations

from src.schemas import ProblemInput


def student_messages(problem: ProblemInput, system_prompt: str, user_template: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(problem=problem.problem)},
    ]
