from __future__ import annotations

import importlib.util
import inspect
import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
    prompt_text: str
    prompt_token_ids: list[int]


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
    debug_event_path: str | Path | None = None,
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
            debug_event_path=debug_event_path,
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
    debug_event_path: str | Path | None = None,
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
    llm = make_llm(
        LLM,
        required_kwargs=set(llm_kwargs) - {"model"},
        **llm_kwargs,
    )
    tokenizer = llm.get_tokenizer()
    if top_k <= 0:
        raise ValueError(
            "vLLM rollout requires top_k > 0 so returned proposal logprobs can "
            "cover the sampled-token candidate set"
        )
    proposal_logprobs_count = max(1, top_k if top_k and top_k > 0 else 1)
    proposal_distribution = "vllm_sample_logprobs"
    raw_logprob_source = "vllm_prompt_logprobs"
    proposal_params = make_sampling_params(
        SamplingParams,
        required_kwargs={
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "logprobs",
        },
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        logprobs=proposal_logprobs_count,
    )
    raw_params = make_sampling_params(
        SamplingParams,
        required_kwargs={
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "prompt_logprobs",
        },
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=1,
        prompt_logprobs=proposal_logprobs_count,
    )

    total_rollouts = len(problems) * rollout_budget
    progress = make_progress(total_rollouts, "vLLM rollout")
    debug_events = DebugEventWriter(debug_event_path)
    vllm_call_id = 0
    try:
        debug_events.open()
        for request_chunk in iter_request_chunks(
            problems,
            rollout_budget,
            tokenizer,
            system_prompt,
            user_template,
            batch_size,
        ):
            vllm_call_id += 1
            proposal_started = time.monotonic()
            debug_events.write(
                "vllm_call_start",
                run_id=run_id,
                vllm_call_id=vllm_call_id,
                operation="sample_completion",
                batch_size=len(request_chunk),
                sampling={
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_tokens": max_tokens,
                    "logprobs": proposal_logprobs_count,
                },
                items=[request_debug_item(request) for request in request_chunk],
            )
            try:
                outputs = llm.generate(
                    [token_prompt(request.prompt_token_ids) for request in request_chunk],
                    proposal_params,
                    use_tqdm=False,
                )
            except Exception as exc:
                debug_events.write(
                    "vllm_call_error",
                    run_id=run_id,
                    vllm_call_id=vllm_call_id,
                    operation="sample_completion",
                    elapsed_s=round(time.monotonic() - proposal_started, 6),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
            debug_events.write(
                "vllm_call_end",
                run_id=run_id,
                vllm_call_id=vllm_call_id,
                operation="sample_completion",
                elapsed_s=round(time.monotonic() - proposal_started, 6),
                items=[
                    proposal_output_debug_item(request, output)
                    for request, output in zip(request_chunk, outputs, strict=True)
                ],
            )
            generated = [
                prepare_generated_rollout(request, output)
                for request, output in zip(request_chunk, outputs, strict=True)
            ]
            proposal_errors = [item.error for item in generated if item.error is not None]
            if proposal_errors and len(proposal_errors) == len(generated):
                raise RuntimeError(
                    "vLLM proposal logprob extraction failed for every item in chunk: "
                    + "; ".join(proposal_errors[:3])
                )
            raw_ready = [item for item in generated if item.error is None]
            if raw_ready:
                vllm_call_id += 1
                raw_started = time.monotonic()
                debug_events.write(
                    "vllm_call_start",
                    run_id=run_id,
                    vllm_call_id=vllm_call_id,
                    operation="score_completion_logprobs",
                    batch_size=len(raw_ready),
                    sampling={
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 0,
                        "max_tokens": 1,
                        "prompt_logprobs": proposal_logprobs_count,
                    },
                    items=[raw_prefill_debug_item(item) for item in raw_ready],
                )
                try:
                    raw_results = batch_raw_prefill_logprobs(
                        llm=llm,
                        raw_params=raw_params,
                        generated=raw_ready,
                    )
                except Exception as exc:
                    debug_events.write(
                        "vllm_call_error",
                        run_id=run_id,
                        vllm_call_id=vllm_call_id,
                        operation="score_completion_logprobs",
                        elapsed_s=round(time.monotonic() - raw_started, 6),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
                debug_events.write(
                    "vllm_call_end",
                    run_id=run_id,
                    vllm_call_id=vllm_call_id,
                    operation="score_completion_logprobs",
                    elapsed_s=round(time.monotonic() - raw_started, 6),
                    items=[
                        raw_result_debug_item(item, raw_result)
                        for item, raw_result in zip(raw_ready, raw_results, strict=True)
                    ],
                )
            else:
                raw_results = []
            raw_errors = [item for item in raw_results if isinstance(item, Exception)]
            if raw_ready and len(raw_errors) == len(raw_results):
                raise RuntimeError(
                    "vLLM raw prefill logprob extraction failed for every valid item in chunk: "
                    + "; ".join(str(error) for error in raw_errors[:3])
                )
            raw_results_by_path_id = {
                item.path_id: raw_result
                for item, raw_result in zip(
                    raw_ready,
                    raw_results,
                    strict=True,
                )
            }
            for item in generated:
                record = build_vllm_record(
                    run_id=run_id,
                    generated=item,
                    raw_result=raw_results_by_path_id.get(item.path_id),
                    proposal_distribution=proposal_distribution,
                    raw_logprob_source=raw_logprob_source,
                )
                debug_events.write(
                    "record_yield",
                    run_id=run_id,
                    path_id=record.path_id,
                    problem_id=record.problem_id,
                    rollout_index=record.rollout_index,
                    is_valid=record.is_valid,
                    output_token_count=record.output_token_count,
                    error=record.error,
                )
                yield record
                progress.update(1)
    finally:
        debug_events.close()
        progress.close()


def iter_request_chunks(
    problems: Sequence[ProblemInput],
    rollout_budget: int,
    tokenizer: Any,
    system_prompt: str,
    user_template: str,
    batch_size: int | None,
) -> Iterator[list[RolloutRequest]]:
    chunk_size = batch_size if batch_size and batch_size > 0 else rollout_budget
    chunk: list[RolloutRequest] = []
    for problem in problems:
        for rollout_index in range(rollout_budget):
            chunk.append(
                build_chat_rollout_request(
                    problem,
                    rollout_index,
                    tokenizer,
                    system_prompt,
                    user_template,
                )
            )
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


class DebugEventWriter:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.handle = None

    def open(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        if self.handle is None:
            return
        self.handle.close()
        self.handle = None

    def write(self, event: str, **payload: Any) -> None:
        if self.handle is None:
            return
        record = {
            "event_time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event": event,
            **payload,
        }
        self.handle.write(json.dumps(record, ensure_ascii=False))
        self.handle.write("\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())


def request_path_id(request: RolloutRequest) -> str:
    return f"{request.problem.problem_id}-{request.rollout_index:04d}"


def request_debug_item(request: RolloutRequest) -> dict[str, Any]:
    return {
        "path_id": request_path_id(request),
        "problem_id": request.problem.problem_id,
        "rollout_index": request.rollout_index,
        "prompt_token_count": len(request.prompt_token_ids),
    }


def proposal_output_debug_item(request: RolloutRequest, output: Any) -> dict[str, Any]:
    item = request_debug_item(request)
    output_items = getattr(output, "outputs", None) or []
    if output_items:
        completion = output_items[0]
        token_ids = list(getattr(completion, "token_ids", []) or [])
        item["completion_token_count"] = len(token_ids)
        item["finish_reason"] = getattr(completion, "finish_reason", None)
    return item


def raw_prefill_debug_item(generated: GeneratedRollout) -> dict[str, Any]:
    prompt_token_count = len(generated.request.prompt_token_ids)
    completion_token_count = len(generated.token_ids)
    return {
        "path_id": generated.path_id,
        "problem_id": generated.request.problem.problem_id,
        "rollout_index": generated.request.rollout_index,
        "prompt_token_count": prompt_token_count,
        "completion_token_count": completion_token_count,
        "full_token_count": prompt_token_count + completion_token_count,
    }


def raw_result_debug_item(
    generated: GeneratedRollout,
    raw_result: list[float] | Exception,
) -> dict[str, Any]:
    item = raw_prefill_debug_item(generated)
    if isinstance(raw_result, Exception):
        item["status"] = "error"
        item["error_type"] = type(raw_result).__name__
        item["error"] = str(raw_result)
    else:
        item["status"] = "ok"
        item["raw_logprob_count"] = len(raw_result)
    return item


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
    full_prompts = [
        token_prompt(item.request.prompt_token_ids + item.token_ids) for item in generated
    ]
    try:
        outputs = llm.generate(full_prompts, raw_params, use_tqdm=False)
    except Exception as exc:
        return [exc for _ in generated]

    results: list[list[float] | Exception] = []
    for item, output in zip(generated, outputs, strict=True):
        try:
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            if not prompt_logprobs:
                raise ValueError(
                    "vLLM did not return prompt_logprobs for raw prefill pass"
                )
            prompt_token_count = len(item.request.prompt_token_ids)
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


def build_chat_rollout_request(
    problem: ProblemInput,
    rollout_index: int,
    tokenizer: Any,
    system_prompt: str,
    user_template: str,
) -> RolloutRequest:
    messages = student_messages(problem, system_prompt, user_template)
    prompt_text = apply_chat_template_required(tokenizer, messages, tokenize=False)
    prompt_token_ids = apply_chat_template_required(tokenizer, messages, tokenize=True)
    return RolloutRequest(
        problem=problem,
        rollout_index=rollout_index,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_token_ids,
    )


def apply_chat_template_required(
    tokenizer: Any,
    messages: list[dict],
    tokenize: bool,
) -> str | list[int]:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        raise ValueError("vLLM tokenizer must provide a chat_template")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        raise ValueError("vLLM tokenizer must support apply_chat_template")
    value = apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=True,
    )
    if tokenize:
        return normalize_token_ids(value)
    if not isinstance(value, str):
        raise ValueError("tokenizer.apply_chat_template(tokenize=False) must return str")
    return value


def normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            keys = ", ".join(str(key) for key in value.keys())
            raise ValueError(
                "tokenizer.apply_chat_template(tokenize=True) returned mapping "
                f"without input_ids: type={type(value).__name__}, keys=[{keys}]"
            )
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("chat template tokenization returned multiple sequences")
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("tokenizer.apply_chat_template(tokenize=True) must return token ids")
    return list(value)


def token_prompt(token_ids: list[int]) -> dict[str, list[int]]:
    return {"prompt_token_ids": list(token_ids)}


def make_llm(llm_type, required_kwargs: set[str] | None = None, **kwargs):
    filtered_kwargs = filter_supported_kwargs(llm_type, kwargs, required_kwargs)
    return llm_type(**filtered_kwargs)


def make_sampling_params(
    sampling_params_type,
    required_kwargs: set[str] | None = None,
    **kwargs,
):
    filtered_kwargs = filter_supported_kwargs(
        sampling_params_type,
        kwargs,
        required_kwargs,
    )
    return sampling_params_type(**filtered_kwargs)


def filter_supported_kwargs(
    callable_type,
    kwargs: dict[str, Any],
    required_kwargs: set[str] | None = None,
) -> dict[str, Any]:
    required_kwargs = required_kwargs or set()
    try:
        parameters = inspect.signature(callable_type).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return kwargs
    unsupported = set(kwargs) - set(parameters)
    missing_required = unsupported & required_kwargs
    if missing_required:
        names = ", ".join(sorted(missing_required))
        raise TypeError(f"{callable_type} does not support required kwargs: {names}")
    return {key: value for key, value in kwargs.items() if key in parameters}


def extract_selected_logprobs(logprob_items: Any, token_ids: list[int]) -> list[float]:
    if not logprob_items:
        raise ValueError("missing selected-token logprobs")
    if len(logprob_items) != len(token_ids):
        raise ValueError(
            "selected-token logprob length mismatch: "
            f"token_count={len(token_ids)}, logprob_count={len(logprob_items)}"
        )
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
    proposal_distribution: str = "vllm_sample_logprobs",
    raw_logprob_source: str = "vllm_prompt_logprobs",
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
