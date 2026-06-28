from __future__ import annotations

from dataclasses import dataclass

from src.schemas import RewardEvaluation, RawTextSpan


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


def validate_reward_evaluation(
    evaluation: RewardEvaluation,
    problem_text: str,
    student_answer: str,
) -> ValidationResult:
    errors: list[str] = []

    problem_span_ids = {item.problem_span_id for item in evaluation.problem_spans}
    student_span_ids = {item.student_span_id for item in evaluation.student_spans}
    reference_point_ids = {
        item.reference_point_id
        for item in (
            evaluation.reference_analysis.given_solution_key_points
            + evaluation.reference_analysis.inferred_minimal_requirements
        )
    }

    _check_unique("problem_span_id", [item.problem_span_id for item in evaluation.problem_spans], errors)
    _check_unique("student_span_id", [item.student_span_id for item in evaluation.student_spans], errors)
    _check_unique(
        "reference_point_id",
        [
            item.reference_point_id
            for item in (
                evaluation.reference_analysis.given_solution_key_points
                + evaluation.reference_analysis.inferred_minimal_requirements
            )
        ],
        errors,
    )

    for item in evaluation.problem_spans:
        if not _span_exists(problem_text, item.raw_text_span):
            errors.append(f"problem_span {item.problem_span_id} anchors do not match problem text")

    student_ranges = _student_span_ranges(student_answer, evaluation.student_spans, errors)
    if student_ranges:
        if student_ranges[0][0] != 0:
            errors.append("student_spans do not start at beginning of student_answer")
        for index in range(1, len(student_ranges)):
            if student_ranges[index][0] != student_ranges[index - 1][1]:
                errors.append(
                    f"student_spans have overlap or gap before {evaluation.student_spans[index].student_span_id}"
                )
        if student_ranges[-1][1] != len(student_answer):
            errors.append("student_spans do not cover end of student_answer")
    else:
        errors.append("student_spans are empty or unmatchable")

    for span in evaluation.student_spans:
        _check_refs("student_span.problem_span_refs", span.problem_span_refs, problem_span_ids, errors)

    for claim in evaluation.student_answer_analysis.main_claims:
        _check_refs("main_claim.student_span_refs", claim.student_span_refs, student_span_ids, errors)

    for item in evaluation.student_alignment.alignment_items:
        _check_refs("alignment.problem_span_refs", item.problem_span_refs, problem_span_ids, errors)
        _check_refs("alignment.student_span_refs", item.student_span_refs, student_span_ids, errors)

    for item in evaluation.student_alignment.off_task_evidence:
        _check_refs("off_task_evidence.student_span_refs", item.student_span_refs, student_span_ids, errors)

    evaluated_span_ids = [item.student_span_id for item in evaluation.span_evaluations]
    _check_unique("span_evaluation.student_span_id", evaluated_span_ids, errors)
    if set(evaluated_span_ids) != student_span_ids:
        missing = sorted(student_span_ids - set(evaluated_span_ids))
        extra = sorted(set(evaluated_span_ids) - student_span_ids)
        if missing:
            errors.append(f"span_evaluations missing student spans: {missing}")
        if extra:
            errors.append(f"span_evaluations contain unknown student spans: {extra}")

    final_span_id = evaluation.final_answer_check.student_final_answer_span_id
    final_span = next((span for span in evaluation.student_spans if span.student_span_id == final_span_id), None)
    if final_span is None:
        errors.append(f"final answer span {final_span_id!r} does not exist")
    elif final_span.span_type != "final_answer":
        errors.append(f"final answer span {final_span_id!r} is not span_type=final_answer")

    for item in evaluation.span_evaluations:
        _check_refs("span_evaluation.problem_span_refs", item.problem_span_refs, problem_span_ids, errors)
        _check_refs("span_evaluation.reference_point_refs", item.reference_point_refs, reference_point_ids, errors)
        if item.student_span_id not in student_span_ids:
            errors.append(f"span_evaluation references unknown student span {item.student_span_id!r}")
        span = next((span for span in evaluation.student_spans if span.student_span_id == item.student_span_id), None)
        if span and span.span_type in {"irrelevant", "final_answer"} and item.step_score != 0.0:
            errors.append(f"{span.student_span_id} has span_type={span.span_type} but step_score is not 0")

    return ValidationResult(ok=not errors, errors=errors)


def _check_unique(name: str, values: list[str], errors: list[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            errors.append(f"duplicate {name}: {value!r}")
        seen.add(value)


def _check_refs(name: str, refs: list[str], valid_ids: set[str], errors: list[str]) -> None:
    for ref in refs:
        if ref not in valid_ids:
            errors.append(f"{name} references unknown id {ref!r}")


def _span_exists(text: str, span: RawTextSpan) -> bool:
    if not span.start_text or not span.end_text:
        return False
    start = text.find(span.start_text)
    if start < 0:
        return False
    end = text.find(span.end_text, start)
    return end >= start


def _student_span_ranges(student_answer: str, spans: list, errors: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for span in spans:
        raw = span.raw_text_span
        if not raw.start_text or not raw.end_text:
            errors.append(f"student_span {span.student_span_id} has empty anchor")
            return []
        start = student_answer.find(raw.start_text, cursor)
        if start < 0:
            errors.append(f"student_span {span.student_span_id} start_text not found after previous span")
            return []
        end_start = student_answer.find(raw.end_text, start)
        if end_start < 0:
            errors.append(f"student_span {span.student_span_id} end_text not found after start_text")
            return []
        end = end_start + len(raw.end_text)
        ranges.append((start, end))
        cursor = end
    return ranges
