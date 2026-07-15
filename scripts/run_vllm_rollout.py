from __future__ import annotations

import hashlib
import json
import os
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_LOG_STATS_INTERVAL"] = "1"

import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import (
    ROLLOUT_ARTIFACT_FORMAT,
    RolloutResumeState,
    RolloutShardWriter,
    load_rollout_resume_state,
    read_jsonl,
    write_json_atomic,
)
from src.schemas import RolloutConfig
from src.vllm_rollout import (
    DebugEventWriter,
    make_progress,
    run_local_rollout_batches,
)


BACKEND = "vllm"
STUDENT_MODEL = "Qwen/Qwen3-1.7B"
STUDENT_TEMPERATURE = 1 # 0.8
STUDENT_TOP_P = 0.95
STUDENT_TOP_K = 20
STUDENT_MAX_NEW_TOKENS = 8192
STUDENT_MAX_NUM_BATCHED_TOKENS = 16384
STUDENT_MAX_NUM_SEQS = 1
STUDENT_GPU_MEMORY_UTILIZATION = 0.8
STUDENT_BATCH_SIZE = 1
STUDENT_ENFORCE_EAGER = True
STUDENT_DISABLE_LOG_STATS = False
RESUME = True

INPUT_JSONL = REPO_ROOT / "data" / "problems.jsonl"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vllm_rollout"
DEBUG_EVENT_PATH = OUTPUT_DIR / "debug" / "vllm_generate_events.jsonl"
DATASET_NAME = "prepared_problems_jsonl"
ROLLOUT_BUDGET = 10

STUDENT_SYSTEM_PROMPT = "Solve the problem. Show the reasoning needed to support the final answer."
STUDENT_USER_TEMPLATE = "{problem}"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_rollout_config(run_id: str) -> RolloutConfig:
    return RolloutConfig(
        run_id=run_id,
        dataset=DATASET_NAME,
        student_model=STUDENT_MODEL,
        student_base_url="",
        backend=BACKEND,
        temperature=STUDENT_TEMPERATURE,
        top_p=STUDENT_TOP_P,
        top_k=STUDENT_TOP_K,
        max_tokens=STUDENT_MAX_NEW_TOKENS,
        extra_body={
            "vllm_engine_args": {
                "gpu_memory_utilization": STUDENT_GPU_MEMORY_UTILIZATION,
                "max_num_batched_tokens": STUDENT_MAX_NUM_BATCHED_TOKENS,
                "max_num_seqs": STUDENT_MAX_NUM_SEQS,
                "enforce_eager": STUDENT_ENFORCE_EAGER,
                "disable_log_stats": STUDENT_DISABLE_LOG_STATS,
            },
            "debug": {
                "vllm_generate_events": display_path(DEBUG_EVENT_PATH),
            },
            "resume_contract": {
                "artifact_format": ROLLOUT_ARTIFACT_FORMAT,
                "input_jsonl_sha256": sha256_file(INPUT_JSONL),
                "system_prompt_sha256": sha256_text(STUDENT_SYSTEM_PROMPT),
                "user_template_sha256": sha256_text(STUDENT_USER_TEMPLATE),
            },
        },
        rollout_budget=ROLLOUT_BUDGET,
    )


def semantic_config(config: RolloutConfig) -> dict:
    extra_body = config.extra_body or {}
    return {
        "dataset": config.dataset,
        "student_model": config.student_model,
        "student_base_url": config.student_base_url,
        "backend": config.backend,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_tokens": config.max_tokens,
        "rollout_budget": config.rollout_budget,
        "resume_contract": extra_body.get("resume_contract"),
    }


def read_rollout_config(path: Path) -> RolloutConfig:
    try:
        return RolloutConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError(f"{path}: invalid rollout config: {exc}") from exc


def expected_rollout_paths(problems) -> dict[str, tuple[str, int]]:
    expected: dict[str, tuple[str, int]] = {}
    for problem in problems:
        for rollout_index in range(ROLLOUT_BUDGET):
            path_id = f"{problem.problem_id}-{rollout_index:04d}"
            if path_id in expected:
                raise ValueError(f"duplicate expected path_id: {path_id}")
            expected[path_id] = (problem.problem_id, rollout_index)
    return expected


def prepare_resume_state(
    problems,
    attempt_id: str,
    discarded: list[tuple[str | None, str]],
) -> tuple[RolloutConfig, RolloutResumeState]:
    config_path = OUTPUT_DIR / "rollout_config.json"
    rollouts_path = OUTPUT_DIR / "rollouts.jsonl"
    output_exists = OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir())
    desired_config = build_rollout_config(attempt_id)
    if not RESUME:
        if output_exists:
            raise FileExistsError(
                f"{OUTPUT_DIR} is not empty and RESUME=False; refusing to overwrite"
            )
        write_json_atomic(config_path, desired_config)
        return desired_config, RolloutResumeState([], frozenset(), (), False)

    if not output_exists:
        write_json_atomic(config_path, desired_config)
        return desired_config, RolloutResumeState([], frozenset(), (), False)
    if not config_path.exists():
        raise ValueError(
            f"{OUTPUT_DIR} contains artifacts but no rollout_config.json; "
            "resume requires a clean sharded rollout run"
        )
    existing_config = read_rollout_config(config_path)
    desired_with_existing_run_id = desired_config.model_copy(
        update={"run_id": existing_config.run_id}
    )
    if semantic_config(existing_config) != semantic_config(desired_with_existing_run_id):
        raise ValueError(
            "rollout resume contract mismatch; model, sampling, prompts, input, "
            "backend, and rollout budget must match the existing run"
        )
    state = load_rollout_resume_state(
        rollouts_path,
        OUTPUT_DIR,
        expected_rollout_paths(problems),
        existing_config.run_id,
        on_discard=lambda path_id, reason: discarded.append((path_id, reason)),
    )
    return existing_config, state


def main() -> None:
    attempt_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    problems = read_jsonl(INPUT_JSONL)
    discarded: list[tuple[str | None, str]] = []
    config, resume_state = prepare_resume_state(problems, attempt_id, discarded)
    run_id = config.run_id
    debug_events = DebugEventWriter(
        DEBUG_EVENT_PATH,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    debug_events.open()
    for path_id, reason in discarded:
        debug_events.write("resume_discard", path_id=path_id, reason=reason)
    total_rollouts = len(problems) * ROLLOUT_BUDGET
    completed_count = len(resume_state.completed_path_ids)
    debug_events.write(
        "resume_state",
        resume_enabled=RESUME,
        completed_count=completed_count,
        remaining_count=total_rollouts - completed_count,
        total_count=total_rollouts,
        execution={
            "batch_size": STUDENT_BATCH_SIZE,
            "max_num_batched_tokens": STUDENT_MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": STUDENT_MAX_NUM_SEQS,
            "gpu_memory_utilization": STUDENT_GPU_MEMORY_UTILIZATION,
            "enforce_eager": STUDENT_ENFORCE_EAGER,
            "disable_log_stats": STUDENT_DISABLE_LOG_STATS,
        },
    )
    print(
        "Running vLLM rollout: "
        f"run_id={run_id}, attempt_id={attempt_id}, resume={RESUME}, "
        f"completed={completed_count}/{total_rollouts}, "
        f"backend={BACKEND}, model={STUDENT_MODEL}, rollout_budget={ROLLOUT_BUDGET}, "
        f"max_new_tokens={STUDENT_MAX_NEW_TOKENS}, "
        f"gpu_memory_utilization={STUDENT_GPU_MEMORY_UTILIZATION}, "
        f"max_num_batched_tokens={STUDENT_MAX_NUM_BATCHED_TOKENS}, "
        f"max_num_seqs={STUDENT_MAX_NUM_SEQS}, batch_size={STUDENT_BATCH_SIZE}, "
        f"enforce_eager={STUDENT_ENFORCE_EAGER}"
    )
    print(f"Writing vLLM debug events to {DEBUG_EVENT_PATH}")
    if completed_count == total_rollouts:
        debug_events.write("rollout_complete", committed_count=completed_count)
        debug_events.close()
        print("vLLM rollout is already complete; no model initialization required")
        return

    progress = make_progress(
        total_rollouts,
        "vLLM rollout",
        initial=completed_count,
    )
    writer = RolloutShardWriter(
        OUTPUT_DIR / "rollouts.jsonl",
        attempt_id=attempt_id,
    )
    try:
        for batch in run_local_rollout_batches(
            backend=BACKEND,
            run_id=run_id,
            problems=problems,
            rollout_budget=ROLLOUT_BUDGET,
            model=STUDENT_MODEL,
            temperature=STUDENT_TEMPERATURE,
            top_p=STUDENT_TOP_P,
            top_k=STUDENT_TOP_K,
            max_tokens=STUDENT_MAX_NEW_TOKENS,
            batch_size=STUDENT_BATCH_SIZE,
            max_num_batched_tokens=STUDENT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=STUDENT_MAX_NUM_SEQS,
            gpu_memory_utilization=STUDENT_GPU_MEMORY_UTILIZATION,
            enforce_eager=STUDENT_ENFORCE_EAGER,
            disable_log_stats=STUDENT_DISABLE_LOG_STATS,
            system_prompt=STUDENT_SYSTEM_PROMPT,
            user_template=STUDENT_USER_TEMPLATE,
            completed_path_ids=resume_state.completed_path_ids,
            debug_events=debug_events,
        ):
            commit = writer.commit_batch(batch)
            completed_count += len(commit.records)
            debug_events.write(
                "artifact_batch_commit",
                logprob_file=commit.logprob_file,
                committed_count=len(commit.records),
                committed_path_ids=[row.path_id for row in commit.records],
                invalid_path_ids=list(commit.invalid_path_ids),
            )
            progress.update(len(commit.records))
        if completed_count == total_rollouts:
            debug_events.write("rollout_complete", committed_count=completed_count)
    finally:
        progress.close()
        debug_events.close()
    if completed_count != total_rollouts:
        raise RuntimeError(
            f"rollout finished with {completed_count}/{total_rollouts} committed valid "
            "records; rerun with RESUME=True to retry invalid paths"
        )


if __name__ == "__main__":
    main()
