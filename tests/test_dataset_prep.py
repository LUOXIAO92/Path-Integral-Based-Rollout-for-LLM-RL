from __future__ import annotations

from src.dataset_prep import (
    is_numeric_only_answer,
    iter_competition_math_rows,
    iter_livecodebench_rows,
    iter_olympiadbench_rows,
)
from src.schemas import ProblemInput


def test_prepare_data_uses_fixed_normalized_output_paths() -> None:
    import scripts.prepare_data as script

    assert script.NORMALIZED_DATA_JSONL.name == "problems.jsonl"
    assert script.NORMALIZED_DATA_JSONL.parent.name == "data"
    assert script.NORMALIZED_DATA_MANIFEST.name == "problems_manifest.json"
    assert script.NORMALIZED_DATA_MANIFEST.parent.name == "data"


def test_numeric_only_answer_filter_excludes_plain_numeric_singletons() -> None:
    assert is_numeric_only_answer(["1.03"])
    assert is_numeric_only_answer(["$1.03$"])
    assert is_numeric_only_answer(["-2"])
    assert is_numeric_only_answer(["3.5e-2"])
    assert is_numeric_only_answer('["1.03"]')
    assert is_numeric_only_answer('["$1.03$"]')


def test_numeric_only_answer_filter_keeps_symbolic_and_non_singletons() -> None:
    assert not is_numeric_only_answer([r"\frac{1}{2}"])
    assert not is_numeric_only_answer([r"\sqrt{2}"])
    assert not is_numeric_only_answer(["1.03 m/s"])
    assert not is_numeric_only_answer(["1.03", "2.04"])
    assert not is_numeric_only_answer("1.03")


def test_competition_math_adapter_outputs_problem_input() -> None:
    rows = [
        {
            "problem": "Find x.",
            "solution": "x=\\boxed{1}.",
            "level": "Level 1",
            "type": "Algebra",
        }
    ]
    problems, manifest = iter_competition_math_rows(rows)

    assert problems == [
        ProblemInput(
            problem_id="competition_math:0",
            subject="math",
            problem="Find x.",
            reference_answer="x=\\boxed{1}.",
        )
    ]
    assert manifest.raw_rows == 1
    assert manifest.kept_rows == 1
    assert manifest.skipped_rows == 0


def test_olympiadbench_adapter_filters_numeric_and_mm_rows() -> None:
    rows = [
        (
            "OE_TO_physics_en_COMP",
            0,
            {
                "question": "Compute a density.",
                "final_answer": ["$1.03$"],
                "solution": ["Use formula."],
            },
        ),
        (
            "OE_TO_physics_en_COMP",
            1,
            {
                "question": "Derive the period.",
                "final_answer": [r"\sqrt{2}"],
                "solution": ["Use conservation of energy."],
            },
        ),
        (
            "OE_MM_physics_en_COMP",
            0,
            {
                "question": "Image task.",
                "final_answer": [r"\sqrt{3}"],
                "solution": ["Look at image."],
            },
        ),
    ]
    problems, manifest = iter_olympiadbench_rows(rows)

    assert len(problems) == 1
    assert problems[0].problem_id == "olympiadbench:OE_TO_physics_en_COMP:1"
    assert problems[0].subject == "physics"
    assert "final_answer" in problems[0].reference_answer
    assert manifest.raw_rows == 3
    assert manifest.kept_rows == 1
    assert manifest.skipped_rows == 2
    assert manifest.numeric_filtered_rows == 1


def test_livecodebench_adapter_outputs_code_problem() -> None:
    rows = [
        {
            "question_title": "A. Short Sort",
            "question_content": "Solve the task.",
            "question_id": "1873_A",
            "starter_code": "def solve(): pass",
            "public_test_cases": '[{"input":"abc","output":"YES"}]',
        }
    ]
    problems, manifest = iter_livecodebench_rows(rows)

    assert len(problems) == 1
    assert problems[0].problem_id == "livecodebench:1873_A"
    assert problems[0].subject == "code"
    assert "A. Short Sort" in problems[0].problem
    assert "Public tests" in problems[0].problem
    assert manifest.raw_rows == 1
    assert manifest.kept_rows == 1
