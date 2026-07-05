from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import (
    hydrate_rollout_logprobs,
    read_jsonl,
    read_model_jsonl,
    write_json,
    write_jsonl,
)
from src.openai_client import make_async_client
from src.rewarding import load_reward_prompt, reward_response_format
from src.schemas import RolloutRecord, ScoreConfig, ScoringConfig
from src.scoring_pipeline import build_candidate_for_rollout


REWARD_BASE_URL = os.getenv("REWARD_BASE_URL", "")
REWARD_API_KEY = os.getenv("REWARD_API_KEY", "")
REWARD_MODEL = os.getenv("REWARD_MODEL", "DeepSeek-V4-Pro")
REWARD_CONCURRENCY = 8
REWARD_TEMPERATURE = 0.0
REWARD_MAX_TOKENS = 4096
REWARD_MAX_RETRIES = 2
USE_REWARD_JSON_SCHEMA = True
REWARD_EXTRA_BODY = None

INPUT_JSONL = REPO_ROOT / "data" / "problems.jsonl"
ROLLOUTS_JSONL = REPO_ROOT / "outputs" / "vllm_rollout" / "rollouts.jsonl"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vllm_rollout"
REWARD_PROMPT_PATH = REPO_ROOT / "docs" / "Reward_prompt.md"
DATASET_NAME = "prepared_problems_jsonl"

ETA = 1.0
LAMBDA_G = 1.0
LAMBDA_N = 1.0
LAMBDA_KL = 0.0
LENGTH_MAX = 2048
LENGTH_SCALE = 512.0

SCORE_CONFIG = ScoreConfig()


async def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    problems = read_jsonl(INPUT_JSONL)
    problems_by_id = {problem.problem_id: problem for problem in problems}
    rollouts = hydrate_rollout_logprobs(
        read_model_jsonl(ROLLOUTS_JSONL, RolloutRecord),
        ROLLOUTS_JSONL.parent,
    )
    template = load_reward_prompt(REWARD_PROMPT_PATH)
    client = make_async_client(REWARD_API_KEY, REWARD_BASE_URL)
    semaphore = asyncio.Semaphore(REWARD_CONCURRENCY)
    response_format = reward_response_format() if USE_REWARD_JSON_SCHEMA else None

    config = ScoringConfig(
        run_id=run_id,
        dataset=DATASET_NAME,
        reward_model=REWARD_MODEL,
        reward_base_url=REWARD_BASE_URL,
        prompt_template_id=str(REWARD_PROMPT_PATH),
        extra_body=REWARD_EXTRA_BODY,
        eta=ETA,
        lambda_G=LAMBDA_G,
        lambda_N=LAMBDA_N,
        lambda_KL=LAMBDA_KL,
        length_max=LENGTH_MAX,
        length_scale=LENGTH_SCALE,
        strict_length_alpha=None,
        score_config=SCORE_CONFIG,
    )

    results = await asyncio.gather(
        *[
            build_candidate_for_rollout(
                problem=problems_by_id[rollout.problem_id],
                rollout=rollout,
                reward_template=template,
                reward_client=client,
                reward_semaphore=semaphore,
                reward_model=REWARD_MODEL,
                reward_temperature=REWARD_TEMPERATURE,
                reward_max_tokens=REWARD_MAX_TOKENS,
                reward_max_retries=REWARD_MAX_RETRIES,
                response_format=response_format,
                reward_extra_body=REWARD_EXTRA_BODY,
                score_config=SCORE_CONFIG,
                eta=ETA,
                lambda_g=LAMBDA_G,
                lambda_n=LAMBDA_N,
                lambda_kl=LAMBDA_KL,
                length_max=LENGTH_MAX,
                length_scale=LENGTH_SCALE,
            )
            for rollout in rollouts
        ]
    )
    candidates = [candidate for candidate, _ in results]
    raw_rows = [raw for _, rows in results for raw in rows]

    write_json(OUTPUT_DIR / "scoring_config.json", config)
    write_jsonl(OUTPUT_DIR / "candidates.jsonl", candidates)
    write_jsonl(OUTPUT_DIR / "reward_raw.jsonl", raw_rows)


if __name__ == "__main__":
    asyncio.run(main())
