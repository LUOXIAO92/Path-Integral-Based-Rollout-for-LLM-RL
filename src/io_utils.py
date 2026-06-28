from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

from src.schemas import ProblemInput

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: Path) -> list[ProblemInput]:
    return read_model_jsonl(path, ProblemInput)


def read_model_jsonl(path: Path, model_type: type[T]) -> list[T]:
    rows: list[T] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            try:
                rows.append(model_type.model_validate(payload))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid {model_type.__name__} row: {exc}") from exc
    return rows


def dump_jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, list):
        return [dump_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: dump_jsonable(item) for key, item in value.items()}
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dump_jsonable(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dump_jsonable(row), ensure_ascii=False))
            handle.write("\n")
