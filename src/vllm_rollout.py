from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Any

from src.rollout import student_messages
from src.schemas import ProblemInput, RolloutRecord


class VLLMUnavailableError(RuntimeError):
    pass


def ensure_vllm_available() -> None:
    if importlib.util.find_spec("vllm") is None:
        raise VLLMUnavailableError(
            "vLLM is not installed in this environment. "
            "Run the vLLM backend on a Linux/GPU environment with vLLM installed, "
            "or use BACKEND='mock' for local tests."
        )


def run_local_rollouts(
    backend: str,
    run_id: str,
    problems: Sequence[ProblemInput],
    rollout_budget: int,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    system_prompt: str,
    user_template: str,
) -> list[RolloutRecord]:
    if backend == "mock":
        return run_mock_rollouts(run_id, problems, rollout_budget)
    if backend == "vllm":
        return run_vllm_rollouts(
            run_id=run_id,
            problems=problems,
            rollout_budget=rollout_budget,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_template=user_template,
        )
    raise ValueError("backend must be 'mock' or 'vllm'")


def run_mock_rollouts(
    run_id: str,
    problems: Sequence[ProblemInput],
    rollout_budget: int,
) -> list[RolloutRecord]:
    rows: list[RolloutRecord] = []
    for problem in problems:
        for rollout_index in range(rollout_budget):
            path_text = f"Mock reasoning for {problem.problem_id}. Final answer: mock"
            raw_logprobs = [-0.30, -0.20, -0.10]
            proposal_logprobs = [-0.25, -0.15, -0.05]
            rows.append(
                RolloutRecord(
                    run_id=run_id,
                    problem_id=problem.problem_id,
                    path_id=f"{problem.problem_id}-{rollout_index:04d}",
                    rollout_index=rollout_index,
                    path_text=path_text,
                    token_logprobs=raw_logprobs,
                    raw_token_logprobs=raw_logprobs,
                    proposal_token_logprobs=proposal_logprobs,
                    raw_logprob_sum=sum(raw_logprobs),
                    proposal_logprob_sum=sum(proposal_logprobs),
                    proposal_distribution="mock_processed",
                    raw_logprob_source="mock_prefill",
                    is_valid=True,
                )
            )
    return rows


def run_vllm_rollouts(
    run_id: str,
    problems: Sequence[ProblemInput],
    rollout_budget: int,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    system_prompt: str,
    user_template: str,
) -> list[RolloutRecord]:
    ensure_vllm_available()
    from vllm import LLM, SamplingParams

    llm = LLM(model=model)
    proposal_params = make_sampling_params(
        SamplingParams,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        logprobs=1,
        logprobs_mode="processed_logprobs",
    )
    raw_params = make_sampling_params(
        SamplingParams,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs_mode="raw_logprobs",
    )

    rows: list[RolloutRecord] = []
    for problem in problems:
        prompt = render_prompt(problem, system_prompt, user_template)
        prompts = [prompt] * rollout_budget
        outputs = llm.generate(prompts, proposal_params)
        for rollout_index, output in enumerate(outputs):
            path_id = f"{problem.problem_id}-{rollout_index:04d}"
            try:
                completion = output.outputs[0]
                path_text = completion.text or ""
                token_ids = list(getattr(completion, "token_ids", []) or [])
                proposal_logprobs = extract_selected_logprobs(completion.logprobs, token_ids)
                raw_logprobs = raw_prefill_logprobs(
                    llm=llm,
                    raw_params=raw_params,
                    prompt=prompt,
                    completion_text=path_text,
                    completion_token_ids=token_ids,
                )
                rows.append(
                    valid_vllm_record(
                        run_id,
                        problem.problem_id,
                        path_id,
                        rollout_index,
                        path_text,
                        raw_logprobs,
                        proposal_logprobs,
                    )
                )
            except Exception as exc:
                rows.append(
                    RolloutRecord(
                        run_id=run_id,
                        problem_id=problem.problem_id,
                        path_id=path_id,
                        rollout_index=rollout_index,
                        path_text=getattr(output.outputs[0], "text", "") if getattr(output, "outputs", None) else "",
                        is_valid=False,
                        error=f"vllm_rollout_failed: {exc}",
                    )
                )
    return rows


def render_prompt(problem: ProblemInput, system_prompt: str, user_template: str) -> str:
    messages = student_messages(problem, system_prompt, user_template)
    return "\n\n".join(f"{item['role']}: {item['content']}" for item in messages)


def make_sampling_params(sampling_params_type, **kwargs):
    try:
        return sampling_params_type(**kwargs)
    except TypeError as exc:
        if "logprobs_mode" in kwargs:
            raise RuntimeError("This vLLM version does not accept logprobs_mode; cannot request processed/raw logprobs explicitly.") from exc
        raise


def raw_prefill_logprobs(
    llm,
    raw_params,
    prompt: str,
    completion_text: str,
    completion_token_ids: list[int],
) -> list[float]:
    full_text = prompt + completion_text
    outputs = llm.generate([full_text], raw_params)
    prompt_logprobs = getattr(outputs[0], "prompt_logprobs", None)
    if not prompt_logprobs:
        raise ValueError("vLLM did not return prompt_logprobs for raw prefill pass")

    tokenizer = llm.get_tokenizer()
    prompt_token_count = len(tokenizer.encode(prompt))
    raw_items = prompt_logprobs[prompt_token_count : prompt_token_count + len(completion_token_ids)]
    return extract_selected_logprobs(raw_items, completion_token_ids)


def extract_selected_logprobs(logprob_items: Any, token_ids: list[int]) -> list[float]:
    if not logprob_items:
        raise ValueError("missing selected-token logprobs")
    values: list[float] = []
    for item, token_id in zip(logprob_items, token_ids, strict=True):
        values.append(extract_one_logprob(item, token_id))
    return values


def extract_one_logprob(item: Any, token_id: int) -> float:
    if isinstance(item, dict):
        if token_id in item:
            return float(getattr(item[token_id], "logprob", item[token_id]))
        token_text = str(token_id)
        if token_text in item:
            return float(getattr(item[token_text], "logprob", item[token_text]))
    value = getattr(item, "logprob", None)
    if value is not None:
        return float(value)
    raise ValueError(f"missing logprob for selected token id {token_id}")


def valid_vllm_record(
    run_id: str,
    problem_id: str,
    path_id: str,
    rollout_index: int,
    path_text: str,
    raw_logprobs: list[float],
    proposal_logprobs: list[float],
) -> RolloutRecord:
    if not raw_logprobs or not proposal_logprobs:
        raise ValueError("raw and proposal logprobs are required")
    if len(raw_logprobs) != len(proposal_logprobs):
        raise ValueError("raw and proposal logprobs must have the same length")
    return RolloutRecord(
        run_id=run_id,
        problem_id=problem_id,
        path_id=path_id,
        rollout_index=rollout_index,
        path_text=path_text,
        token_logprobs=raw_logprobs,
        raw_token_logprobs=raw_logprobs,
        proposal_token_logprobs=proposal_logprobs,
        raw_logprob_sum=sum(raw_logprobs),
        proposal_logprob_sum=sum(proposal_logprobs),
        proposal_distribution="vllm_processed",
        raw_logprob_source="vllm_prefill",
        is_valid=True,
    )
