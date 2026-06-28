from __future__ import annotations

import math
import random
from collections import defaultdict

from src.schemas import PathRecord


def metropolis_acceptance_probability(
    current_s_eta: float,
    candidate_s_eta: float,
    rho_prop: float,
) -> float:
    if not 0 < rho_prop <= 1:
        raise ValueError("rho_prop must be in (0, 1]")
    delta_s_eta = candidate_s_eta - current_s_eta
    try:
        value = rho_prop * math.exp(-delta_s_eta)
    except OverflowError:
        value = float("inf")
    return min(1.0, value)


def metropolis_accept(
    current_s_eta: float,
    candidate_s_eta: float,
    rho_prop: float,
    rng: random.Random,
) -> tuple[bool, float]:
    prob = metropolis_acceptance_probability(current_s_eta, candidate_s_eta, rho_prop)
    return rng.random() <= prob, prob


def run_mcmc_chain(
    candidates: list[PathRecord],
    rho_prop: float,
    rng: random.Random,
) -> tuple[list[PathRecord], list[PathRecord]]:
    grouped: dict[str, list[PathRecord]] = defaultdict(list)
    for record in candidates:
        grouped[record.problem_id].append(record)

    updated_candidates: list[PathRecord] = []
    chain_records: list[PathRecord] = []
    for problem_id in sorted(grouped):
        current: PathRecord | None = None
        chain_step = 0
        for record in grouped[problem_id]:
            if not record.reward_valid or record.s_eta is None:
                updated_candidates.append(record)
                continue

            if current is None:
                candidate = record.model_copy(update={"is_accepted": True, "acceptance_prob": 1.0})
                current = candidate
            else:
                accepted, prob = metropolis_accept(current.s_eta, record.s_eta, rho_prop, rng)
                candidate = record.model_copy(update={"is_accepted": accepted, "acceptance_prob": prob})
                if accepted:
                    current = candidate

            updated_candidates.append(candidate)
            chain_records.append(
                current.model_copy(
                    update={
                        "method": "mcmc_chain_state",
                        "chain_step": chain_step,
                        "source_path_id": current.path_id,
                        "is_accepted": True,
                    }
                )
            )
            chain_step += 1

    return updated_candidates, chain_records


def select_best_of_n(candidates: list[PathRecord]) -> list[PathRecord]:
    grouped: dict[str, list[PathRecord]] = defaultdict(list)
    for record in candidates:
        if record.reward_valid and record.s_eta is not None and record.g is not None:
            grouped[record.problem_id].append(record)

    rows: list[PathRecord] = []
    for problem_id in sorted(grouped):
        records = grouped[problem_id]
        by_s_eta = min(records, key=lambda item: item.s_eta)
        by_g = max(records, key=lambda item: (item.g, bool(item.final_correctness), -item.s_eta))
        rows.append(by_s_eta.model_copy(update={"method": "best_of_n_s_eta", "source_path_id": by_s_eta.path_id}))
        rows.append(by_g.model_copy(update={"method": "best_of_n_g", "source_path_id": by_g.path_id}))
    return rows
