from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import read_jsonl, write_json, write_jsonl
from src.schemas import RolloutConfig
from src.vllm_rollout import run_local_rollouts


BACKEND = "mock"
STUDENT_MODEL = "Qwen/Qwen3-0.6B"
STUDENT_TEMPERATURE = 0.8
STUDENT_TOP_P = 0.95
STUDENT_TOP_K = 20
STUDENT_MAX_TOKENS = 2048

INPUT_JSONL = REPO_ROOT / "data" / "problems.jsonl"
OUTPUT_DIR = REPO_ROOT / "outputs" / "openai_rollout"
DATASET_NAME = "prepared_problems_jsonl"
ROLLOUT_BUDGET = 4

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
        max_tokens=STUDENT_MAX_TOKENS,
        rollout_budget=ROLLOUT_BUDGET,
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
        max_tokens=STUDENT_MAX_TOKENS,
        system_prompt=STUDENT_SYSTEM_PROMPT,
        user_template=STUDENT_USER_TEMPLATE,
    )
    write_json(OUTPUT_DIR / "rollout_config.json", config)
    write_jsonl(OUTPUT_DIR / "rollouts.jsonl", rows)


if __name__ == "__main__":
    main()
