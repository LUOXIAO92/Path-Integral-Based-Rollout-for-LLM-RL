from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import read_model_jsonl, write_json, write_jsonl
from src.mcmc import run_mcmc_chain, select_best_of_n
from src.reporting import build_summary
from src.schemas import MCMCConfig, PathRecord


CANDIDATES_JSONL = REPO_ROOT / "outputs" / "openai_rollout" / "candidates.jsonl"
OUTPUT_DIR = REPO_ROOT / "outputs" / "openai_rollout"
RHO_PROP = 1.0
RANDOM_SEED = 1234


def main() -> None:
    candidates = read_model_jsonl(CANDIDATES_JSONL, PathRecord)
    updated_candidates, chain = run_mcmc_chain(candidates, RHO_PROP, random.Random(RANDOM_SEED))
    best_of_n = select_best_of_n(updated_candidates)
    run_id = updated_candidates[0].run_id if updated_candidates else ""
    summary = build_summary(run_id, updated_candidates)
    config = MCMCConfig(
        rho_prop=RHO_PROP,
        random_seed=RANDOM_SEED,
        candidates_jsonl=str(CANDIDATES_JSONL),
        output_dir=str(OUTPUT_DIR),
    )

    write_json(OUTPUT_DIR / "mcmc_config.json", config)
    write_jsonl(OUTPUT_DIR / "chain.jsonl", chain)
    write_jsonl(OUTPUT_DIR / "best_of_n.jsonl", best_of_n)
    write_json(OUTPUT_DIR / "summary.json", summary)


if __name__ == "__main__":
    main()
