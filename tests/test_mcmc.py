from __future__ import annotations

import random

from src.mcmc import (
    metropolis_accept,
    metropolis_acceptance_probability,
    run_mcmc_chain,
    select_best_of_n,
)
from src.schemas import PathRecord


def test_metropolis_acceptance_probability_uses_action_difference() -> None:
    assert metropolis_acceptance_probability(5.0, 4.0, 1.0) == 1.0
    uphill = metropolis_acceptance_probability(4.0, 5.0, 1.0)
    assert 0.0 < uphill < 1.0


def test_metropolis_accept_uses_rng() -> None:
    accepted, prob = metropolis_accept(4.0, 5.0, 1.0, random.Random(0))
    assert 0.0 < prob < 1.0
    assert accepted is False


def test_run_mcmc_chain_keeps_endpoint_when_candidate_rejected() -> None:
    candidates = [
        record("p1", "a", s_eta=1.0, g=0.5),
        record("p1", "b", s_eta=100.0, g=1.0),
    ]
    updated, chain = run_mcmc_chain(candidates, rho_prop=1.0, rng=random.Random(0))

    assert updated[0].is_accepted is True
    assert updated[1].is_accepted is False
    assert chain[0].source_path_id == "a"
    assert chain[1].source_path_id == "a"


def test_run_mcmc_chain_preserves_invalid_candidates_without_chain_update() -> None:
    candidates = [
        record("p1", "bad", s_eta=None, g=None, reward_valid=False),
        record("p1", "good", s_eta=1.0, g=0.5),
    ]
    updated, chain = run_mcmc_chain(candidates, rho_prop=1.0, rng=random.Random(0))

    assert updated[0].path_id == "bad"
    assert updated[0].is_accepted is None
    assert updated[1].is_accepted is True
    assert len(chain) == 1
    assert chain[0].source_path_id == "good"


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
) -> PathRecord:
    return PathRecord(
        run_id="run",
        problem_id=problem_id,
        method="mcmc_candidate",
        path_id=path_id,
        path_text=path_id,
        reward_valid=reward_valid,
        g=g,
        n=0.0 if reward_valid else None,
        k=0.0 if reward_valid else None,
        f=g if reward_valid else None,
        s0=s_eta if reward_valid and s_eta is not None else None,
        s_eta=s_eta,
        final_correctness=bool(g and g > 0.8),
    )
