from __future__ import annotations

import math
import random

from src.mcmc import (
    compute_strict_length_penalty,
    metropolis_accept,
    metropolis_acceptance_probability,
    run_mcmc_chain,
    select_best_of_n,
)
from src.schemas import PathRecord, ScoreConfig, ScoringConfig


def test_metropolis_acceptance_probability_uses_action_difference() -> None:
    assert metropolis_acceptance_probability(5.0, 4.0, 0.0) == 1.0
    uphill = metropolis_acceptance_probability(4.0, 5.0, 0.0)
    assert 0.0 < uphill < 1.0


def test_metropolis_accept_uses_rng() -> None:
    accepted, prob = metropolis_accept(4.0, 5.0, 0.0, random.Random(0))
    assert 0.0 < prob < 1.0
    assert accepted is False


def test_metropolis_acceptance_probability_handles_extreme_log_ratios() -> None:
    assert metropolis_acceptance_probability(0.0, 0.0, -1000.0) == 0.0
    assert metropolis_acceptance_probability(0.0, 0.0, 1000.0) == 1.0


def test_compute_strict_length_penalty_scales_bounded_penalty() -> None:
    penalty = compute_strict_length_penalty(
        length=12,
        length_max=10,
        length_scale=2.0,
        strict_length_alpha=0.5,
    )

    assert penalty == 12 * 0.5 * math.tanh(1.0)


def test_run_mcmc_chain_keeps_endpoint_when_candidate_rejected() -> None:
    candidates = [
        record("p1", "a", s_eta=1.0, g=0.5),
        record("p1", "b", s_eta=100.0, g=1.0),
    ]
    updated, chain = run_mcmc_chain(candidates, "normalized", random.Random(0))

    assert updated[0].is_accepted is True
    assert updated[1].is_accepted is False
    assert chain[0].source_path_id == "a"
    assert chain[1].source_path_id == "a"


def test_run_mcmc_chain_preserves_invalid_candidates_without_chain_update() -> None:
    candidates = [
        record("p1", "bad", s_eta=None, g=None, reward_valid=False),
        record("p1", "good", s_eta=1.0, g=0.5),
    ]
    updated, chain = run_mcmc_chain(candidates, "normalized", random.Random(0))

    assert updated[0].path_id == "bad"
    assert updated[0].is_accepted is None
    assert updated[1].is_accepted is True
    assert len(chain) == 1
    assert chain[0].source_path_id == "good"


def test_run_mcmc_chain_uses_normalized_proposal_ratio() -> None:
    candidates = [
        record(
            "p1",
            "a",
            s_eta=0.0,
            g=0.5,
            proposal_logprob_sum=-10.0,
            proposal_logprob_mean=-0.5,
        ),
        record(
            "p1",
            "b",
            s_eta=0.0,
            g=0.5,
            proposal_logprob_sum=-12.0,
            proposal_logprob_mean=-0.1,
        ),
    ]

    updated, chain = run_mcmc_chain(candidates, "normalized", random.Random(0))

    assert updated[1].proposal_log_ratio_strict == 2.0
    assert updated[1].proposal_log_ratio_normalized == -0.4
    assert updated[1].proposal_log_ratio == -0.4
    assert updated[1].proposal_log_q_forward == -0.1
    assert updated[1].proposal_log_q_reverse == -0.5
    assert updated[1].acceptance_prob == math.exp(-0.4)
    assert chain[1].proposal_log_ratio == -0.4
    assert chain[1].proposal_log_ratio_strict == 2.0
    assert chain[1].proposal_log_ratio_normalized == -0.4


def test_run_mcmc_chain_uses_strict_proposal_ratio() -> None:
    candidates = [
        record(
            "p1",
            "a",
            s_eta=0.0,
            g=0.5,
            proposal_logprob_sum=-10.0,
            proposal_logprob_mean=-0.5,
        ),
        record(
            "p1",
            "b",
            s_eta=0.0,
            g=0.5,
            proposal_logprob_sum=-12.0,
            proposal_logprob_mean=-0.1,
        ),
    ]

    updated, chain = run_mcmc_chain(candidates, "strict", random.Random(0), scoring_config())

    assert updated[1].proposal_log_ratio == 2.0
    assert updated[1].proposal_log_q_forward == -12.0
    assert updated[1].proposal_log_q_reverse == -10.0
    assert updated[1].acceptance_prob == 1.0
    assert chain[1].proposal_log_ratio == 2.0


def test_strict_mode_requires_scoring_config() -> None:
    candidates = [
        record("p1", "a", s_eta=0.0, g=0.5),
        record("p1", "b", s_eta=0.0, g=0.5),
    ]

    try:
        run_mcmc_chain(candidates, "strict", random.Random(0))
    except ValueError as exc:
        assert "scoring_config" in str(exc)
    else:
        raise AssertionError("strict mode should require scoring config")


def test_strict_mode_requires_strict_length_alpha_in_scoring_config() -> None:
    candidates = [
        record("p1", "a", s_eta=0.0, g=0.5),
        record("p1", "b", s_eta=0.0, g=0.5),
    ]

    try:
        run_mcmc_chain(candidates, "strict", random.Random(0), scoring_config(None))
    except ValueError as exc:
        assert "strict_length_alpha" in str(exc)
    else:
        raise AssertionError("strict mode should require strict_length_alpha")


def test_strict_mode_uses_strict_scaled_length_action() -> None:
    candidates = [
        record(
            "p1",
            "short",
            s_eta=0.0,
            g=0.0,
            output_token_count=10,
            proposal_logprob_sum=-10.0,
            proposal_logprob_mean=-0.1,
        ),
        record(
            "p1",
            "long",
            s_eta=-100.0,
            g=0.0,
            s0=0.0,
            output_token_count=20,
            proposal_logprob_sum=-20.0,
            proposal_logprob_mean=-0.1,
        ),
    ]

    updated, chain = run_mcmc_chain(candidates, "strict", random.Random(0), scoring_config())

    expected_penalty = 20 * math.tanh(1.0)
    expected_candidate_action = expected_penalty
    assert updated[1].strict_length_alpha == 1.0
    assert updated[1].strict_length_penalty_scaled == expected_penalty
    assert updated[1].strict_s_eta == expected_candidate_action
    assert updated[1].selected_s_eta_current == 0.0
    assert updated[1].selected_s_eta_candidate == expected_candidate_action
    assert updated[1].proposal_log_ratio == 10.0
    assert updated[1].acceptance_prob == math.exp(-expected_candidate_action + 10.0)
    assert updated[1].is_accepted is False
    assert chain[1].source_path_id == "short"


def test_normalized_mode_ignores_scoring_config() -> None:
    candidates = [
        record("p1", "a", s_eta=0.0, g=0.5, output_token_count=10),
        record("p1", "b", s_eta=-1.0, g=0.5, output_token_count=100),
    ]

    updated, _ = run_mcmc_chain(candidates, "normalized", random.Random(0), scoring_config())

    assert updated[1].selected_s_eta_current == 0.0
    assert updated[1].selected_s_eta_candidate == -1.0
    assert updated[1].strict_s_eta is None
    assert updated[1].acceptance_prob == 1.0


def test_select_best_of_n_selects_by_s_eta_and_g() -> None:
    candidates = [
        record("p1", "low-action", s_eta=0.5, g=0.2),
        record("p1", "high-reward", s_eta=1.0, g=0.9),
    ]

    rows = select_best_of_n(candidates)

    assert [item.method for item in rows] == ["best_of_n_s_eta", "best_of_n_g"]
    assert rows[0].source_path_id == "low-action"
    assert rows[1].source_path_id == "high-reward"


def record(
    problem_id: str,
    path_id: str,
    s_eta: float | None,
    g: float | None,
    reward_valid: bool = True,
    s0: float | None = None,
    proposal_logprob_sum: float | None = -1.0,
    proposal_logprob_mean: float | None = -0.1,
    output_token_count: int | None = 10,
) -> PathRecord:
    return PathRecord(
        run_id="run",
        problem_id=problem_id,
        method="mcmc_candidate",
        path_id=path_id,
        path_text=path_id,
        output_token_count=output_token_count if reward_valid else None,
        proposal_logprob_sum=proposal_logprob_sum if reward_valid else None,
        proposal_logprob_mean=proposal_logprob_mean if reward_valid else None,
        proposal_distribution="test_distribution" if reward_valid else "",
        reward_valid=reward_valid,
        g=g,
        n=0.0 if reward_valid else None,
        k=0.0 if reward_valid else None,
        f=g if reward_valid else None,
        s0=s0 if s0 is not None else s_eta if reward_valid and s_eta is not None else None,
        s_eta=s_eta,
        final_correctness=bool(g and g > 0.8),
    )


def scoring_config(strict_length_alpha: float | None = 1.0) -> ScoringConfig:
    return ScoringConfig(
        run_id="run",
        dataset="test",
        reward_model="reward",
        reward_base_url="",
        prompt_template_id="prompt",
        eta=1.0,
        lambda_G=1.0,
        lambda_N=1.0,
        lambda_KL=0.0,
        length_max=10,
        length_scale=10.0,
        strict_length_alpha=strict_length_alpha,
        score_config=ScoreConfig(),
    )
