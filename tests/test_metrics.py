from __future__ import annotations

import pytest

from src.metrics import compute_path_metrics


def test_metrics_use_per_token_normalized_s0() -> None:
    metrics = compute_path_metrics(
        token_logprobs=[-0.2, -0.3],
        g=0.8,
        eta=2.0,
        lambda_g=1.0,
        lambda_n=0.5,
        lambda_kl=0.0,
        length_max=10,
        length_scale=5.0,
    )
    assert metrics.k == 0.0
    assert metrics.s0 == 0.25
    assert metrics.n == 0.0
    assert metrics.f == 0.8
    assert metrics.s_eta == -1.35


def test_metrics_reject_empty_token_logprobs() -> None:
    with pytest.raises(ValueError, match="token_logprobs"):
        compute_path_metrics(
            token_logprobs=[],
            g=0.8,
            eta=2.0,
            lambda_g=1.0,
            lambda_n=0.5,
            lambda_kl=0.0,
            length_max=10,
            length_scale=5.0,
        )
