from __future__ import annotations

import os
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import read_jsonl, write_json, write_rollouts_with_logprob_sidecars
from src.schemas import RolloutConfig
from src.vllm_rollout import run_local_rollouts


BACKEND = "vllm"
STUDENT_MODEL = "Qwen/Qwen3-0.6B"
STUDENT_TEMPERATURE = 1 # 0.8
STUDENT_TOP_P = 1 # 0.95
STUDENT_TOP_K = 0 # 20
STUDENT_MAX_NEW_TOKENS = 2048
STUDENT_MAX_NUM_BATCHED_TOKENS = 8192
STUDENT_MAX_NUM_SEQS = 4
STUDENT_GPU_MEMORY_UTILIZATION = 0.85
STUDENT_BATCH_SIZE = 30

INPUT_JSONL = REPO_ROOT / "data" / "problems.jsonl"
OUTPUT_DIR = REPO_ROOT / "outputs" / "openai_rollout"
DATASET_NAME = "prepared_problems_jsonl"
ROLLOUT_BUDGET = 10

STUDENT_SYSTEM_PROMPT = "Solve the problem. Show the reasoning needed to support the final answer."
STUDENT_USER_TEMPLATE = "{problem}"


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    problems = read_jsonl(INPUT_JSONL)
    config = RolloutConfig(
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
            }
        },
        rollout_budget=ROLLOUT_BUDGET,
    )
    print(
        "Running vLLM rollout: "
        f"backend={BACKEND}, model={STUDENT_MODEL}, rollout_budget={ROLLOUT_BUDGET}, "
        f"max_new_tokens={STUDENT_MAX_NEW_TOKENS}, "
        f"gpu_memory_utilization={STUDENT_GPU_MEMORY_UTILIZATION}, "
        f"max_num_batched_tokens={STUDENT_MAX_NUM_BATCHED_TOKENS}, "
        f"max_num_seqs={STUDENT_MAX_NUM_SEQS}, batch_size={STUDENT_BATCH_SIZE}"
    )
    rows = run_local_rollouts(
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
        system_prompt=STUDENT_SYSTEM_PROMPT,
        user_template=STUDENT_USER_TEMPLATE,
    )
    write_json(OUTPUT_DIR / "rollout_config.json", config)
    write_rollouts_with_logprob_sidecars(OUTPUT_DIR / "rollouts.jsonl", rows)


if __name__ == "__main__":
    main()
