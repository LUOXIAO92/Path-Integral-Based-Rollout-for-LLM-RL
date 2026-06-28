from __future__ import annotations

import random
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_prep import (
    LIVE_CODE_BENCH_RELEASE_LATEST_FILES,
    OLYMPIADBENCH_TEXT_CONFIGS,
    iter_competition_math_rows,
    iter_livecodebench_rows,
    iter_olympiadbench_rows,
    load_jsonl_rows,
    load_parquet_rows,
)
from src.io_utils import write_json, write_jsonl
from src.schemas import DatasetPrepManifest, ProblemInput


# Dataset repos.
COMPETITION_MATH_REPO = "qwedsacf/competition_math"
OLYMPIADBENCH_REPO = "Hothan/OlympiadBench"
LIVECODEBENCH_REPO = "livecodebench/code_generation_lite"

NORMALIZED_DATA_JSONL = REPO_ROOT / "data" / "problems.jsonl"
NORMALIZED_DATA_MANIFEST = REPO_ROOT / "data" / "problems_manifest.json"

# Row limits. Use None for all rows.
MAX_COMPETITION_MATH = None
MAX_OLYMPIADBENCH = None
MAX_LIVECODEBENCH = None

OLYMPIADBENCH_CONFIGS = OLYMPIADBENCH_TEXT_CONFIGS
LIVECODEBENCH_FILES = LIVE_CODE_BENCH_RELEASE_LATEST_FILES

SHUFFLE = False
RANDOM_SEED = 1234


def main() -> None:
    problems: list[ProblemInput] = []
    manifests = []

    competition_dir = snapshot_dataset(COMPETITION_MATH_REPO)
    competition_files = sorted((competition_dir / "data").glob("*.parquet"))
    competition_rows = load_parquet_rows(competition_files)
    competition_problems, competition_manifest = iter_competition_math_rows(
        competition_rows,
        max_rows=MAX_COMPETITION_MATH,
    )
    problems.extend(competition_problems)
    manifests.append(competition_manifest)

    olympiad_dir = snapshot_dataset(OLYMPIADBENCH_REPO)
    olympiad_rows = []
    for config in OLYMPIADBENCH_CONFIGS:
        parquet_file = olympiad_dir / "OlympiadBench" / config / f"{config}.parquet"
        rows = load_parquet_rows([parquet_file])
        for index, row in enumerate(rows):
            olympiad_rows.append((config, index, row))
    olympiad_problems, olympiad_manifest = iter_olympiadbench_rows(
        olympiad_rows,
        max_rows=MAX_OLYMPIADBENCH,
    )
    problems.extend(olympiad_problems)
    manifests.append(olympiad_manifest)

    livecodebench_dir = snapshot_dataset(LIVECODEBENCH_REPO)
    livecodebench_files = [livecodebench_dir / file_name for file_name in LIVECODEBENCH_FILES]
    livecodebench_rows = load_jsonl_rows(livecodebench_files)
    livecodebench_problems, livecodebench_manifest = iter_livecodebench_rows(
        livecodebench_rows,
        max_rows=MAX_LIVECODEBENCH,
    )
    problems.extend(livecodebench_problems)
    manifests.append(livecodebench_manifest)

    if SHUFFLE:
        random.Random(RANDOM_SEED).shuffle(problems)

    manifest = DatasetPrepManifest(
        output_jsonl=str(NORMALIZED_DATA_JSONL),
        total_rows=len(problems),
        sources=manifests,
    )
    write_jsonl(NORMALIZED_DATA_JSONL, problems)
    write_json(NORMALIZED_DATA_MANIFEST, manifest)
    print(f"Wrote {len(problems)} rows to {NORMALIZED_DATA_JSONL}")
    print(f"Wrote manifest to {NORMALIZED_DATA_MANIFEST}")


def snapshot_dataset(repo_id: str) -> Path:
    path = snapshot_download(repo_id=repo_id, repo_type="dataset")
    return Path(path)


if __name__ == "__main__":
    main()
