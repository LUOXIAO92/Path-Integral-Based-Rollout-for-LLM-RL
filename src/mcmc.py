from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Literal

from src.schemas import PathRecord, ScoringConfig

ProposalRatioMode = Literal["normalized", "strict"]


def metropolis_acceptance_probability(
    current_s_eta: float,
    candidate_s_eta: float,
    proposal_log_ratio: float,
) -> float:
    delta_s_eta = candidate_s_eta - current_s_eta
    log_accept = -delta_s_eta + proposal_log_ratio
    if log_accept >= 0:
        return 1.0
    if log_accept < -745:
        return 0.0
    return math.exp(log_accept)


def metropolis_accept(
    current_s_eta: float,
    candidate_s_eta: float,
    proposal_log_ratio: float,
    rng: random.Random,
) -> tuple[bool, float]:
    prob = metropolis_acceptance_probability(current_s_eta, candidate_s_eta, proposal_log_ratio)
    return rng.random() <= prob, prob


def run_mcmc_chain(
    candidates: list[PathRecord],
    proposal_ratio_mode: ProposalRatioMode,
    rng: random.Random,
    scoring_config: ScoringConfig | None = None,
) -> tuple[list[PathRecord], list[PathRecord]]:
    if proposal_ratio_mode not in ("normalized", "strict"):
        raise ValueError("proposal_ratio_mode must be 'normalized' or 'strict'")
    if proposal_ratio_mode == "strict" and scoring_config is None:
        raise ValueError("scoring_config is required for strict proposal ratio mode")
    if (
        proposal_ratio_mode == "strict"
        and scoring_config is not None
        and scoring_config.strict_length_alpha is None
    ):
        raise ValueError(
            "strict_length_alpha is required in scoring_config for strict proposal ratio mode"
        )

    grouped: dict[str, list[PathRecord]] = defaultdict(list)
    for record in candidates:
        grouped[record.problem_id].append(record)

    updated_candidates: list[PathRecord] = []
    chain_records: list[PathRecord] = []
    for problem_id in sorted(grouped):
        current: PathRecord | None = None
        chain_step = 0
        for record in grouped[problem_id]:
            transition_update: dict[str, float | str] | None = None
            if not record.reward_valid or record.s_eta is None:
                updated_candidates.append(record)
                continue

            if current is None:
                candidate = record.model_copy(update={"is_accepted": True, "acceptance_prob": 1.0})
                current = candidate
            else:
                proposal = compute_proposal_transition(current, record, proposal_ratio_mode)
                action = compute_transition_action(
                    current,
                    record,
                    proposal_ratio_mode,
                    scoring_config,
                )
                accepted, prob = metropolis_accept(
                    action["selected_s_eta_current"],
                    action["selected_s_eta_candidate"],
                    proposal["proposal_log_ratio"],
                    rng,
                )
                transition_update = {**proposal, **action, "acceptance_prob": prob}
                candidate = record.model_copy(
                    update={
                        **transition_update,
                        "is_accepted": accepted,
                    }
                )
                if accepted:
                    current = candidate

            updated_candidates.append(candidate)
            chain_update = {
                "method": "mcmc_chain_state",
                "chain_step": chain_step,
                "source_path_id": current.path_id,
                "is_accepted": True,
            }
            if transition_update is not None:
                chain_update.update(transition_update)
            chain_records.append(
                current.model_copy(update=chain_update)
            )
            chain_step += 1

    return updated_candidates, chain_records


def compute_transition_action(
    current: PathRecord,
    candidate: PathRecord,
    proposal_ratio_mode: ProposalRatioMode,
    scoring_config: ScoringConfig | None,
) -> dict[str, float]:
    if proposal_ratio_mode == "strict":
        if scoring_config is None:
            raise ValueError("scoring_config is required for strict proposal ratio mode")
        if scoring_config.strict_length_alpha is None:
            raise ValueError(
                "strict_length_alpha is required in scoring_config for strict proposal ratio mode"
            )
        current_action = compute_strict_action(current, scoring_config)
        candidate_action = compute_strict_action(candidate, scoring_config)
        return {
            "strict_length_alpha": scoring_config.strict_length_alpha,
            "strict_length_penalty_scaled": candidate_action["strict_length_penalty_scaled"],
            "strict_f": candidate_action["strict_f"],
            "strict_s_eta": candidate_action["strict_s_eta"],
            "selected_s_eta_current": current_action["strict_s_eta"],
            "selected_s_eta_candidate": candidate_action["strict_s_eta"],
        }
    return {
        "selected_s_eta_current": require_path_value(current.s_eta, current.path_id, "Sη[τ]"),
        "selected_s_eta_candidate": require_path_value(candidate.s_eta, candidate.path_id, "Sη[τ]"),
    }


def compute_strict_action(
    record: PathRecord,
    scoring_config: ScoringConfig,
) -> dict[str, float]:
    if scoring_config.strict_length_alpha is None:
        raise ValueError(
            "strict_length_alpha is required in scoring_config for strict proposal ratio mode"
        )
    length = require_output_token_count(record.output_token_count, record.path_id)
    n_scaled = compute_strict_length_penalty(
        length=length,
        length_max=scoring_config.length_max,
        length_scale=scoring_config.length_scale,
        strict_length_alpha=scoring_config.strict_length_alpha,
    )
    g = require_path_value(record.g, record.path_id, "G[τ]")
    k = require_path_value(record.k, record.path_id, "K[τ]")
    s0 = require_path_value(record.s0, record.path_id, "S0[τ]")
    strict_f = (
        scoring_config.lambda_G * g
        - scoring_config.lambda_N * n_scaled
        - scoring_config.lambda_KL * k
    )
    strict_s_eta = s0 - scoring_config.eta * strict_f
    return {
        "strict_length_penalty_scaled": n_scaled,
        "strict_f": strict_f,
        "strict_s_eta": strict_s_eta,
    }


def compute_strict_length_penalty(
    length: int,
    length_max: int,
    length_scale: float,
    strict_length_alpha: float,
) -> float:
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")
    if not 0.0 <= strict_length_alpha <= 1.0:
        raise ValueError("strict_length_alpha must be in [0, 1]")
    return length * strict_length_alpha * math.tanh(max(0, length - length_max) / length_scale)


def compute_proposal_transition(
    current: PathRecord,
    candidate: PathRecord,
    proposal_ratio_mode: ProposalRatioMode,
) -> dict[str, float | str]:
    strict_forward = require_proposal_logprob(candidate.proposal_logprob_sum, candidate.path_id, "sum")
    strict_reverse = require_proposal_logprob(current.proposal_logprob_sum, current.path_id, "sum")
    normalized_forward = require_proposal_logprob(
        candidate.proposal_logprob_mean,
        candidate.path_id,
        "mean",
    )
    normalized_reverse = require_proposal_logprob(
        current.proposal_logprob_mean,
        current.path_id,
        "mean",
    )
    strict_ratio = strict_reverse - strict_forward
    normalized_ratio = normalized_reverse - normalized_forward
    if proposal_ratio_mode == "strict":
        forward = strict_forward
        reverse = strict_reverse
        selected_ratio = strict_ratio
    else:
        forward = normalized_forward
        reverse = normalized_reverse
        selected_ratio = normalized_ratio
    return {
        "proposal_ratio_mode": proposal_ratio_mode,
        "proposal_log_q_forward": forward,
        "proposal_log_q_reverse": reverse,
        "proposal_log_ratio": selected_ratio,
        "proposal_log_ratio_strict": strict_ratio,
        "proposal_log_ratio_normalized": normalized_ratio,
    }


def require_proposal_logprob(value: float | None, path_id: str, scale: str) -> float:
    if value is None:
        raise ValueError(f"{path_id} is missing proposal_logprob_{scale}")
    return value


def require_path_value(value: float | None, path_id: str, name: str) -> float:
    if value is None:
        raise ValueError(f"{path_id} is missing {name}")
    return value


def require_output_token_count(value: int | None, path_id: str) -> int:
    if value is None:
        raise ValueError(f"{path_id} is missing output_token_count")
    if value <= 0:
        raise ValueError(f"{path_id} output_token_count must be positive")
    return value


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
