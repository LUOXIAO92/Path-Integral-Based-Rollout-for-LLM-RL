from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from src.prompts import student_messages
from src.schemas import ProblemInput, RolloutRecord

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is installed with vLLM in runtime use.
    tqdm = None


class VLLMUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class RolloutRequest:
    problem: ProblemInput
    rollout_index: int
    prompt: str


@dataclass(frozen=True)
class GeneratedRollout:
    request: RolloutRequest
    output: Any
    path_id: str
    path_text: str
    token_ids: list[int]
    proposal_logprobs: list[float] | None = None
    error: str | None = None


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
    batch_size: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> Iterator[RolloutRecord]:
    if backend == "mock":
        yield from run_mock_rollouts(run_id, problems, rollout_budget)
        return
    if backend == "vllm":
        yield from run_vllm_rollouts(
            run_id=run_id,
            problems=problems,
            rollout_budget=rollout_budget,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            batch_size=batch_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            system_prompt=system_prompt,
            user_template=user_template,
        )
        return
    raise ValueError("backend must be 'mock' or 'vllm'")


def run_mock_rollouts(
    run_id: str,
    problems: Sequence[ProblemInput],
    rollout_budget: int,
) -> Iterator[RolloutRecord]:
    for problem in problems:
        for rollout_index in range(rollout_budget):
            path_text = f"Mock reasoning for {problem.problem_id}. Final answer: mock"
            raw_logprobs = [-0.30, -0.20, -0.10]
            proposal_logprobs = [-0.25, -0.15, -0.05]
            output_token_count = len(raw_logprobs)
            yield RolloutRecord(
                run_id=run_id,
                problem_id=problem.problem_id,
                path_id=f"{problem.problem_id}-{rollout_index:04d}",
                rollout_index=rollout_index,
                path_text=path_text,
                token_logprobs=raw_logprobs,
                raw_token_logprobs=raw_logprobs,
                proposal_token_logprobs=proposal_logprobs,
                output_token_count=output_token_count,
                raw_logprob_sum=sum(raw_logprobs),
                proposal_logprob_sum=sum(proposal_logprobs),
                raw_logprob_mean=sum(raw_logprobs) / output_token_count,
                proposal_logprob_mean=sum(proposal_logprobs) / output_token_count,
                proposal_distribution="mock_processed",
                raw_logprob_source="mock_prefill",
                is_valid=True,
            )


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
    batch_size: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> Iterator[RolloutRecord]:
    ensure_vllm_available()
    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {"model": model}
    if gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs
    llm = make_llm(LLM, **llm_kwargs)
    explicit_logprobs_mode = sampling_params_accepts(SamplingParams, "logprobs_mode")
    proposal_distribution = (
        "vllm_processed" if explicit_logprobs_mode else "vllm_logprobs_default"
    )
    raw_logprob_source = (
        "vllm_prefill_raw" if explicit_logprobs_mode else "vllm_prefill_default"
    )
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
        top_k=0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs_mode="raw_logprobs",
    )

    total_rollouts = len(problems) * rollout_budget
    progress = make_progress(total_rollouts, "vLLM rollout")
    try:
        for request_chunk in iter_request_chunks(
            problems,
            rollout_budget,
            system_prompt,
            user_template,
            batch_size,
        ):
            outputs = llm.generate(
                [request.prompt for request in request_chunk],
                proposal_params,
                use_tqdm=False,
            )
            generated = [
                prepare_generated_rollout(request, output)
                for request, output in zip(request_chunk, outputs, strict=True)
            ]
            raw_results = batch_raw_prefill_logprobs(
                llm=llm,
                raw_params=raw_params,
                generated=[item for item in generated if item.error is None],
            )
            raw_results_by_path_id = {
                item.path_id: raw_result
                for item, raw_result in zip(
                    [item for item in generated if item.error is None],
                    raw_results,
                    strict=True,
                )
            }
            for item in generated:
                yield build_vllm_record(
                    run_id=run_id,
                    generated=item,
                    raw_result=raw_results_by_path_id.get(item.path_id),
                    proposal_distribution=proposal_distribution,
                    raw_logprob_source=raw_logprob_source,
                )
                progress.update(1)
    finally:
        progress.close()


def iter_request_chunks(
    problems: Sequence[ProblemInput],
    rollout_budget: int,
    system_prompt: str,
    user_template: str,
    batch_size: int | None,
) -> Iterator[list[RolloutRequest]]:
    chunk_size = batch_size if batch_size and batch_size > 0 else rollout_budget
    chunk: list[RolloutRequest] = []
    for problem in problems:
        prompt = render_prompt(problem, system_prompt, user_template)
        for rollout_index in range(rollout_budget):
            chunk.append(RolloutRequest(problem, rollout_index, prompt))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


class NullProgress:
    def update(self, count: int) -> None:
        pass

    def close(self) -> None:
        pass


def make_progress(total: int, desc: str):
    if tqdm is None:
        return NullProgress()
    return tqdm(total=total, desc=desc, dynamic_ncols=True)


def prepare_generated_rollout(request: RolloutRequest, output: Any) -> GeneratedRollout:
    path_id = f"{request.problem.problem_id}-{request.rollout_index:04d}"
    try:
        completion = output.outputs[0]
        path_text = completion.text or ""
        token_ids = list(getattr(completion, "token_ids", []) or [])
        proposal_logprobs = extract_selected_logprobs(completion.logprobs, token_ids)
        return GeneratedRollout(
            request=request,
            output=output,
            path_id=path_id,
            path_text=path_text,
            token_ids=token_ids,
            proposal_logprobs=proposal_logprobs,
        )
    except Exception as exc:
        return GeneratedRollout(
            request=request,
            output=output,
            path_id=path_id,
            path_text=extract_output_text(output),
            token_ids=[],
            error=f"vllm_rollout_failed: {exc}",
        )


def extract_output_text(output: Any) -> str:
    return (
        getattr(output.outputs[0], "text", "")
        if getattr(output, "outputs", None)
        else ""
    )


def batch_raw_prefill_logprobs(
    llm,
    raw_params,
    generated: Sequence[GeneratedRollout],
) -> list[list[float] | Exception]:
    if not generated:
        return []
    full_texts = [item.request.prompt + item.path_text for item in generated]
    try:
        outputs = llm.generate(full_texts, raw_params, use_tqdm=False)
    except Exception as exc:
        return [exc for _ in generated]

    tokenizer = llm.get_tokenizer()
    results: list[list[float] | Exception] = []
    for item, output in zip(generated, outputs, strict=True):
        try:
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            if not prompt_logprobs:
                raise ValueError(
                    "vLLM did not return prompt_logprobs for raw prefill pass"
                )
            prompt_token_count = len(tokenizer.encode(item.request.prompt))
            raw_items = prompt_logprobs[
                prompt_token_count : prompt_token_count + len(item.token_ids)
            ]
            results.append(extract_selected_logprobs(raw_items, item.token_ids))
        except Exception as exc:
            results.append(exc)
    return results


def build_vllm_record(
    run_id: str,
    generated: GeneratedRollout,
    raw_result: list[float] | Exception | None,
    proposal_distribution: str,
    raw_logprob_source: str,
) -> RolloutRecord:
    problem = generated.request.problem
    if generated.error:
        return invalid_vllm_record(run_id, generated, generated.error)
    if isinstance(raw_result, Exception):
        return invalid_vllm_record(
            run_id,
            generated,
            f"vllm_rollout_failed: {raw_result}",
        )
    if raw_result is None:
        return invalid_vllm_record(
            run_id,
            generated,
            "vllm_rollout_failed: missing raw prefill result",
        )
    try:
        return valid_vllm_record(
            run_id,
            problem.problem_id,
            generated.path_id,
            generated.request.rollout_index,
            generated.path_text,
            raw_result,
            generated.proposal_logprobs or [],
            proposal_distribution=proposal_distribution,
            raw_logprob_source=raw_logprob_source,
        )
    except Exception as exc:
        return invalid_vllm_record(run_id, generated, f"vllm_rollout_failed: {exc}")


def invalid_vllm_record(
    run_id: str,
    generated: GeneratedRollout,
    error: str,
) -> RolloutRecord:
    return RolloutRecord(
        run_id=run_id,
        problem_id=generated.request.problem.problem_id,
        path_id=generated.path_id,
        rollout_index=generated.request.rollout_index,
        path_text=generated.path_text,
        is_valid=False,
        error=error,
    )


def render_prompt(problem: ProblemInput, system_prompt: str, user_template: str) -> str:
    messages = student_messages(problem, system_prompt, user_template)
    return "\n\n".join(f"{item['role']}: {item['content']}" for item in messages)


def make_llm(llm_type, **kwargs):
    filtered_kwargs = filter_supported_kwargs(llm_type, kwargs)
    return llm_type(**filtered_kwargs)


def sampling_params_accepts(sampling_params_type, parameter_name: str) -> bool:
    try:
        parameters = inspect.signature(sampling_params_type).parameters
    except (TypeError, ValueError):
        return False
    return parameter_name in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def make_sampling_params(sampling_params_type, **kwargs):
    filtered_kwargs = filter_supported_kwargs(sampling_params_type, kwargs)
    return sampling_params_type(**filtered_kwargs)


def filter_supported_kwargs(callable_type, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(callable_type).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


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
    proposal_distribution: str = "vllm_processed",
    raw_logprob_source: str = "vllm_prefill",
) -> RolloutRecord:
    if not raw_logprobs or not proposal_logprobs:
        raise ValueError("raw and proposal logprobs are required")
    if len(raw_logprobs) != len(proposal_logprobs):
        raise ValueError("raw and proposal logprobs must have the same length")
    output_token_count = len(raw_logprobs)
    raw_logprob_sum = sum(raw_logprobs)
    proposal_logprob_sum = sum(proposal_logprobs)
    return RolloutRecord(
        run_id=run_id,
        problem_id=problem_id,
        path_id=path_id,
        rollout_index=rollout_index,
        path_text=path_text,
        token_logprobs=raw_logprobs,
        raw_token_logprobs=raw_logprobs,
        proposal_token_logprobs=proposal_logprobs,
        output_token_count=output_token_count,
        raw_logprob_sum=raw_logprob_sum,
        proposal_logprob_sum=proposal_logprob_sum,
        raw_logprob_mean=raw_logprob_sum / output_token_count,
        proposal_logprob_mean=proposal_logprob_sum / output_token_count,
        proposal_distribution=proposal_distribution,
        raw_logprob_source=raw_logprob_source,
        is_valid=True,
    )
