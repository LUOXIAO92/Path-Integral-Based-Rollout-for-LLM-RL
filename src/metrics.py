from __future__ import annotations

import math

from src.schemas import PathMetrics


def compute_length_penalty(length: int, length_max: int, length_scale: float) -> float:
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")
    return math.tanh(max(0, length - length_max) / length_scale)


def compute_path_metrics(
    token_logprobs: list[float],
    g: float,
    eta: float,
    lambda_g: float,
    lambda_n: float,
    lambda_kl: float,
    length_max: int,
    length_scale: float,
) -> PathMetrics:
    output_token_count = len(token_logprobs)
    if output_token_count == 0:
        raise ValueError("token_logprobs must not be empty")
    s0 = -sum(token_logprobs) / output_token_count
    n = compute_length_penalty(output_token_count, length_max, length_scale)
    k = 0.0
    f = lambda_g * g - lambda_n * n - lambda_kl * k
    s_eta = s0 - eta * f
    return PathMetrics(n=n, k=k, s0=s0, f=f, s_eta=s_eta)
