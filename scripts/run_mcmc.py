from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import read_model_jsonl, write_json, write_jsonl
from src.mcmc import run_mcmc_chain, select_best_of_n
from src.reporting import build_summary
from src.schemas import MCMCConfig, PathRecord, ScoringConfig


CANDIDATES_JSONL = REPO_ROOT / "outputs" / "vllm_rollout" / "candidates.jsonl"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vllm_rollout"
PROPOSAL_RATIO_MODE = "normalized"
SCORING_CONFIG_JSON = OUTPUT_DIR / "scoring_config.json"
RANDOM_SEED = 1234


def main() -> None:
    candidates = read_model_jsonl(CANDIDATES_JSONL, PathRecord)
    scoring_config = load_strict_scoring_config()
    updated_candidates, chain = run_mcmc_chain(
        candidates,
        PROPOSAL_RATIO_MODE,
        random.Random(RANDOM_SEED),
        scoring_config=scoring_config,
    )
    best_of_n = select_best_of_n(updated_candidates)
    run_id = updated_candidates[0].run_id if updated_candidates else ""
    summary = build_summary(run_id, updated_candidates)
    config = MCMCConfig(
        proposal_ratio_mode=PROPOSAL_RATIO_MODE,
        scoring_config_json=str(SCORING_CONFIG_JSON),
        random_seed=RANDOM_SEED,
        candidates_jsonl=str(CANDIDATES_JSONL),
        output_dir=str(OUTPUT_DIR),
    )

    write_json(OUTPUT_DIR / "mcmc_config.json", config)
    write_jsonl(OUTPUT_DIR / "chain.jsonl", chain)
    write_jsonl(OUTPUT_DIR / "best_of_n.jsonl", best_of_n)
    write_json(OUTPUT_DIR / "summary.json", summary)


def load_strict_scoring_config() -> ScoringConfig | None:
    if PROPOSAL_RATIO_MODE != "strict":
        return None
    with SCORING_CONFIG_JSON.open("r", encoding="utf-8") as handle:
        return ScoringConfig.model_validate(json.load(handle))


if __name__ == "__main__":
    main()
