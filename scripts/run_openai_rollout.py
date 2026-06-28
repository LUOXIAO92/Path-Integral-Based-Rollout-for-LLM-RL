from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import read_jsonl, write_json, write_jsonl
from src.openai_client import make_async_client
from src.rollout import run_student_rollouts
from src.schemas import RolloutConfig


STUDENT_BASE_URL = os.getenv("STUDENT_BASE_URL", "")
STUDENT_API_KEY = os.getenv("STUDENT_API_KEY", "")
STUDENT_MODEL = os.getenv("STUDENT_MODEL", "qwen3-0.6b")
STUDENT_CONCURRENCY = 8
STUDENT_TEMPERATURE = 0.8
STUDENT_TOP_P = 0.95
STUDENT_MAX_TOKENS = 2048
STUDENT_EXTRA_BODY = {"top_k": 20}

INPUT_JSONL = REPO_ROOT / "data" / "problems.jsonl"
OUTPUT_DIR = REPO_ROOT / "outputs" / "openai_rollout"
DATASET_NAME = "prepared_problems_jsonl"
ROLLOUT_BUDGET = 4

STUDENT_SYSTEM_PROMPT = "Solve the problem. Show the reasoning needed to support the final answer."
STUDENT_USER_TEMPLATE = "{problem}"


async def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    problems = read_jsonl(INPUT_JSONL)
    client = make_async_client(STUDENT_API_KEY, STUDENT_BASE_URL)
    semaphore = asyncio.Semaphore(STUDENT_CONCURRENCY)

    config = RolloutConfig(
        run_id=run_id,
        dataset=DATASET_NAME,
        student_model=STUDENT_MODEL,
        student_base_url=STUDENT_BASE_URL,
        temperature=STUDENT_TEMPERATURE,
        top_p=STUDENT_TOP_P,
        max_tokens=STUDENT_MAX_TOKENS,
        extra_body=STUDENT_EXTRA_BODY,
        rollout_budget=ROLLOUT_BUDGET,
    )
    rows = await run_student_rollouts(
        run_id=run_id,
        problems=problems,
        rollout_budget=ROLLOUT_BUDGET,
        client=client,
        semaphore=semaphore,
        model=STUDENT_MODEL,
        temperature=STUDENT_TEMPERATURE,
        top_p=STUDENT_TOP_P,
        max_tokens=STUDENT_MAX_TOKENS,
        extra_body=STUDENT_EXTRA_BODY,
        system_prompt=STUDENT_SYSTEM_PROMPT,
        user_template=STUDENT_USER_TEMPLATE,
    )

    write_json(OUTPUT_DIR / "rollout_config.json", config)
    write_jsonl(OUTPUT_DIR / "rollouts.jsonl", rows)


if __name__ == "__main__":
    asyncio.run(main())
