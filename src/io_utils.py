from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np
from pydantic import BaseModel

from src.schemas import ProblemInput, RolloutRecord

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


def write_jsonl_stream(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dump_jsonable(row), ensure_ascii=False))
            handle.write("\n")
            handle.flush()


def write_rollouts_with_logprob_sidecars(
    path: Path,
    rows: Iterable[RolloutRecord],
    logprob_file: str = "logprobs",
    logprob_dtype: str = "float32",
) -> None:
    logprob_root = _output_relative_path(logprob_file)
    output_root = path.parent
    raw_path = output_root / logprob_root / "raw.npz"
    proposal_path = output_root / logprob_root / "proposal.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("w", encoding="utf-8") as jsonl_handle,
        zipfile.ZipFile(raw_path, "w", compression=zipfile.ZIP_DEFLATED) as raw_zip,
        zipfile.ZipFile(proposal_path, "w", compression=zipfile.ZIP_DEFLATED) as proposal_zip,
    ):
        for row in rows:
            payload = dump_jsonable(row)
            raw_logprobs = payload.pop("raw_token_logprobs", [])
            proposal_logprobs = payload.pop("proposal_token_logprobs", [])
            payload.pop("token_logprobs", None)
            if raw_logprobs and proposal_logprobs:
                path_id = payload["path_id"]
                _write_npz_array(raw_zip, path_id, raw_logprobs, logprob_dtype)
                _write_npz_array(proposal_zip, path_id, proposal_logprobs, logprob_dtype)
                payload["logprob_file"] = logprob_root.as_posix()
                payload["logprob_dtype"] = logprob_dtype
            jsonl_handle.write(json.dumps(payload, ensure_ascii=False))
            jsonl_handle.write("\n")
            jsonl_handle.flush()


def hydrate_rollout_logprobs(
    rollouts: Iterable[RolloutRecord],
    output_root: Path,
) -> list[RolloutRecord]:
    rows = list(rollouts)
    hydrated: list[RolloutRecord | None] = [None] * len(rows)
    sidecar_groups: dict[str, list[tuple[int, RolloutRecord]]] = {}
    for index, row in enumerate(rows):
        if row.raw_token_logprobs and row.proposal_token_logprobs:
            hydrated[index] = row
        elif row.logprob_file:
            sidecar_groups.setdefault(row.logprob_file, []).append((index, row))
        else:
            hydrated[index] = row

    for logprob_file, group in sidecar_groups.items():
        logprob_root = _output_relative_path(logprob_file)
        raw_path = output_root / logprob_root / "raw.npz"
        proposal_path = output_root / logprob_root / "proposal.npz"
        with np.load(raw_path) as raw_npz, np.load(proposal_path) as proposal_npz:
            for index, row in group:
                if row.path_id not in raw_npz:
                    raise KeyError(f"{raw_path}: missing raw logprobs for {row.path_id}")
                if row.path_id not in proposal_npz:
                    raise KeyError(
                        f"{proposal_path}: missing proposal logprobs for {row.path_id}"
                    )
                raw_logprobs = raw_npz[row.path_id].astype(float).tolist()
                proposal_logprobs = proposal_npz[row.path_id].astype(float).tolist()
                output_token_count = len(raw_logprobs)
                raw_logprob_sum = sum(raw_logprobs)
                proposal_logprob_sum = sum(proposal_logprobs)
                hydrated[index] = row.model_copy(
                    update={
                        "token_logprobs": raw_logprobs,
                        "raw_token_logprobs": raw_logprobs,
                        "proposal_token_logprobs": proposal_logprobs,
                        "output_token_count": row.output_token_count
                        or output_token_count,
                        "raw_logprob_mean": row.raw_logprob_mean
                        if row.raw_logprob_mean is not None
                        else raw_logprob_sum / output_token_count,
                        "proposal_logprob_mean": row.proposal_logprob_mean
                        if row.proposal_logprob_mean is not None
                        else proposal_logprob_sum / output_token_count,
                    }
                )
    return [row for row in hydrated if row is not None]


def _write_npz_array(
    archive: zipfile.ZipFile,
    key: str,
    values: list[float],
    dtype: str,
) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values, dtype=dtype))
    archive.writestr(f"{key}.npy", buffer.getvalue())


def _output_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"logprob_file must be relative to the output root: {value}")
    return path
