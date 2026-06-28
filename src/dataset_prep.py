from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.schemas import DatasetSourceManifest, ProblemInput


NUMERIC_ANSWER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

OLYMPIADBENCH_TEXT_CONFIGS = [
    "OE_TO_maths_en_COMP",
    "OE_TO_maths_zh_CEE",
    "OE_TO_maths_zh_COMP",
    "OE_TO_physics_en_COMP",
    "OE_TO_physics_zh_CEE",
    "TP_TO_maths_en_COMP",
    "TP_TO_maths_zh_CEE",
    "TP_TO_maths_zh_COMP",
    "TP_TO_physics_en_COMP",
]

LIVE_CODE_BENCH_RELEASE_LATEST_FILES = [
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
]


def is_numeric_only_answer(value: object) -> bool:
    items = _as_single_answer_list(value)
    if items is None:
        return False
    item = str(items[0]).strip()
    if item.startswith("$") and item.endswith("$") and len(item) >= 2:
        item = item[1:-1].strip()
    return bool(NUMERIC_ANSWER_RE.fullmatch(item))


def iter_competition_math_rows(
    rows: Iterable[Mapping[str, Any]],
    max_rows: int | None = None,
) -> tuple[list[ProblemInput], DatasetSourceManifest]:
    manifest = DatasetSourceManifest(source="qwedsacf/competition_math", configs=["train"])
    output: list[ProblemInput] = []
    for index, row in enumerate(rows):
        manifest.raw_rows += 1
        problem = _text(row.get("problem"))
        solution = _text(row.get("solution"))
        if not problem or not solution:
            manifest.skipped_rows += 1
            continue
        output.append(
            ProblemInput(
                problem_id=f"competition_math:{index}",
                subject="math",
                problem=problem,
                reference_answer=solution,
            )
        )
        manifest.kept_rows += 1
        if max_rows is not None and manifest.kept_rows >= max_rows:
            break
    return output, manifest


def iter_olympiadbench_rows(
    config_rows: Iterable[tuple[str, int, Mapping[str, Any]]],
    max_rows: int | None = None,
) -> tuple[list[ProblemInput], DatasetSourceManifest]:
    manifest = DatasetSourceManifest(source="Hothan/OlympiadBench")
    output: list[ProblemInput] = []
    for config, index, row in config_rows:
        if config not in manifest.configs:
            manifest.configs.append(config)
        manifest.raw_rows += 1
        if "_MM_" in config or "_TO_" not in config:
            manifest.skipped_rows += 1
            continue

        answer_value = row.get("final_answer", row.get("answer"))
        if is_numeric_only_answer(answer_value):
            manifest.numeric_filtered_rows += 1
            manifest.skipped_rows += 1
            continue

        problem = _first_text(row, ["question", "problem", "prompt"])
        reference_answer = _combine_reference(row, ["answer", "final_answer", "solution"])
        if not problem or not reference_answer:
            manifest.skipped_rows += 1
            continue

        subject = "physics" if "physics" in config else "math"
        output.append(
            ProblemInput(
                problem_id=f"olympiadbench:{config}:{index}",
                subject=subject,
                problem=problem,
                reference_answer=reference_answer,
            )
        )
        manifest.kept_rows += 1
        if max_rows is not None and manifest.kept_rows >= max_rows:
            break
    return output, manifest


def iter_livecodebench_rows(
    rows: Iterable[Mapping[str, Any]],
    max_rows: int | None = None,
) -> tuple[list[ProblemInput], DatasetSourceManifest]:
    manifest = DatasetSourceManifest(
        source="livecodebench/code_generation_lite",
        configs=["release_latest"],
    )
    output: list[ProblemInput] = []
    for index, row in enumerate(rows):
        manifest.raw_rows += 1
        question_content = _text(row.get("question_content"))
        if not question_content:
            manifest.skipped_rows += 1
            continue
        question_id = _text(row.get("question_id")) or str(index)
        output.append(
            ProblemInput(
                problem_id=f"livecodebench:{question_id}",
                subject="code",
                problem=_livecodebench_problem(row),
                reference_answer="",
                test_result="",
            )
        )
        manifest.kept_rows += 1
        if max_rows is not None and manifest.kept_rows >= max_rows:
            break
    return output, manifest


def load_parquet_rows(parquet_files: list[Path]) -> Iterable[Mapping[str, Any]]:
    dataset = load_dataset(
        "parquet",
        data_files=[str(path) for path in parquet_files],
        split="train",
    )
    return dataset


def load_jsonl_rows(paths: list[Path]) -> Iterable[Mapping[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)


def _as_single_answer_list(value: object) -> list[object] | None:
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
        if not isinstance(parsed, list):
            return None
        items = parsed
    else:
        return None
    if len(items) != 1:
        return None
    return items


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value if _text(item)).strip()
    return str(value).strip()


def _first_text(row: Mapping[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = _text(row.get(field))
        if value:
            return value
    return ""


def _combine_reference(row: Mapping[str, Any], fields: list[str]) -> str:
    chunks: list[str] = []
    for field in fields:
        value = _text(row.get(field))
        if value:
            chunks.append(f"{field}:\n{value}")
    return "\n\n".join(chunks)


def _livecodebench_problem(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    title = _text(row.get("question_title"))
    content = _text(row.get("question_content"))
    starter_code = _text(row.get("starter_code"))
    public_tests = _text(row.get("public_test_cases"))
    if title:
        parts.append(f"Title:\n{title}")
    if content:
        parts.append(f"Problem:\n{content}")
    if starter_code:
        parts.append(f"Starter code:\n{starter_code}")
    if public_tests:
        parts.append(f"Public tests:\n{public_tests}")
    return "\n\n".join(parts)
