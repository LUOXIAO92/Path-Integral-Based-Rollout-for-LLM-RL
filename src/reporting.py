from __future__ import annotations

from src.schemas import PathRecord, Summary


def build_summary(run_id: str, candidates: list[PathRecord]) -> Summary:
    valid = [item for item in candidates if item.reward_valid and item.s_eta is not None]
    accepted = [item for item in valid if item.is_accepted is True]
    rejected = [item for item in valid if item.is_accepted is False]
    return Summary(
        run_id=run_id,
        total_candidates=len(candidates),
        valid_candidates=len(valid),
        accepted_candidates=len(accepted),
        rejected_candidates=len(rejected),
        mean_all_s_eta=mean([item.s_eta for item in valid]),
        mean_accepted_s_eta=mean([item.s_eta for item in accepted]),
        mean_rejected_s_eta=mean([item.s_eta for item in rejected]),
        all_candidate_final_correctness=rate([item.final_correctness for item in valid]),
        accepted_final_correctness=rate([item.final_correctness for item in accepted]),
        rejected_final_correctness=rate([item.final_correctness for item in rejected]),
    )


def mean(values: list[float | None]) -> float | None:
    real_values = [item for item in values if item is not None]
    if not real_values:
        return None
    return sum(real_values) / len(real_values)


def rate(values: list[bool | None]) -> float | None:
    real_values = [item for item in values if item is not None]
    if not real_values:
        return None
    return sum(1 for item in real_values if item) / len(real_values)
