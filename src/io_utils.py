from __future__ import annotations

import io
import json
import os
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
from pydantic import BaseModel

from src.schemas import ProblemInput, RolloutRecord

T = TypeVar("T", bound=BaseModel)
ROLLOUT_ARTIFACT_FORMAT = "sharded_npz_v1"


@dataclass(frozen=True)
class RolloutResumeState:
    rows: list[RolloutRecord]
    completed_path_ids: frozenset[str]
    discarded_path_ids: tuple[str, ...]
    truncated_tail: bool


@dataclass(frozen=True)
class RolloutBatchCommit:
    logprob_file: str | None
    records: list[RolloutRecord]
    invalid_path_ids: tuple[str, ...]


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


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(dump_jsonable(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
    _fsync_directory(path.parent)


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


class RolloutShardWriter:
    def __init__(
        self,
        path: Path,
        attempt_id: str,
        logprob_file: str = "logprobs",
        logprob_dtype: str = "float32",
    ) -> None:
        self.path = path
        self.output_root = path.parent
        self.logprob_root = _output_relative_path(logprob_file)
        self.logprob_dtype = logprob_dtype
        self.attempt_id = attempt_id
        self.next_shard_index = self._find_next_shard_index()

    def commit_batch(self, rows: Iterable[RolloutRecord]) -> RolloutBatchCommit:
        batch = list(rows)
        valid_rows = [row for row in batch if row.is_valid]
        invalid_path_ids = tuple(row.path_id for row in batch if not row.is_valid)
        if not valid_rows:
            return RolloutBatchCommit(None, [], invalid_path_ids)
        for row in valid_rows:
            _validate_valid_rollout_arrays(row)

        shard_name = f"shard-{self.next_shard_index:06d}"
        self.next_shard_index += 1
        shard_relative = self.logprob_root / shard_name
        shard_root = self.output_root / self.logprob_root
        final_directory = self.output_root / shard_relative
        temporary_directory = shard_root / (
            f".{shard_name}.{self.attempt_id}.{os.getpid()}.tmp"
        )
        shard_root.mkdir(parents=True, exist_ok=True)
        temporary_directory.mkdir()
        raw_path = temporary_directory / "raw.npz"
        proposal_path = temporary_directory / "proposal.npz"
        with (
            zipfile.ZipFile(raw_path, "w", compression=zipfile.ZIP_DEFLATED) as raw_zip,
            zipfile.ZipFile(
                proposal_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as proposal_zip,
        ):
            for row in valid_rows:
                _write_npz_array(
                    raw_zip,
                    row.path_id,
                    row.raw_token_logprobs,
                    self.logprob_dtype,
                )
                _write_npz_array(
                    proposal_zip,
                    row.path_id,
                    row.proposal_token_logprobs,
                    self.logprob_dtype,
                )
        _fsync_file(raw_path)
        _fsync_file(proposal_path)
        _fsync_directory(temporary_directory)
        os.replace(temporary_directory, final_directory)
        _fsync_directory(shard_root)

        encoded_rows = b"".join(
            (
                json.dumps(
                    _rollout_metadata_payload(
                        row,
                        shard_relative.as_posix(),
                        self.logprob_dtype,
                    ),
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n"
            )
            for row in valid_rows
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(encoded_rows)
            handle.flush()
            os.fsync(handle.fileno())
        return RolloutBatchCommit(
            shard_relative.as_posix(),
            valid_rows,
            invalid_path_ids,
        )

    def _find_next_shard_index(self) -> int:
        shard_root = self.output_root / self.logprob_root
        if not shard_root.exists():
            return 0
        indices: list[int] = []
        for child in shard_root.iterdir():
            if not child.is_dir() or not child.name.startswith("shard-"):
                continue
            suffix = child.name.removeprefix("shard-")
            if suffix.isdigit():
                indices.append(int(suffix))
        return max(indices, default=-1) + 1


def load_rollout_resume_state(
    path: Path,
    output_root: Path,
    expected_paths: Mapping[str, tuple[str, int]],
    expected_run_id: str,
    on_discard: Callable[[str | None, str], None] | None = None,
) -> RolloutResumeState:
    parsed_rows, truncated_tail, missing_final_newline = _read_resume_rows(path)
    candidates: list[RolloutRecord] = []
    discarded_path_ids: list[str] = []
    for row in parsed_rows:
        expected = expected_paths.get(row.path_id)
        if expected is None:
            raise ValueError(f"{path}: unexpected path_id {row.path_id}")
        if (row.problem_id, row.rollout_index) != expected:
            raise ValueError(
                f"{path}: path metadata mismatch for {row.path_id}: "
                f"expected={expected}, actual={(row.problem_id, row.rollout_index)}"
            )
        if row.run_id != expected_run_id:
            raise ValueError(
                f"{path}: run_id mismatch for {row.path_id}: "
                f"expected={expected_run_id}, actual={row.run_id}"
            )
        if not row.is_valid:
            discarded_path_ids.append(row.path_id)
            _notify_discard(on_discard, row.path_id, "record is not valid")
            continue
        if not _is_sharded_logprob_file(row.logprob_file):
            raise ValueError(
                f"{path}: {row.path_id} does not use resumable sharded sidecars: "
                f"{row.logprob_file!r}"
            )
        candidates.append(row)

    complete_rows: list[RolloutRecord] = []
    groups: dict[str, list[RolloutRecord]] = {}
    for row in candidates:
        groups.setdefault(row.logprob_file, []).append(row)
    for logprob_file, group in groups.items():
        logprob_root = _output_relative_path(logprob_file)
        raw_path = output_root / logprob_root / "raw.npz"
        proposal_path = output_root / logprob_root / "proposal.npz"
        try:
            with np.load(raw_path) as raw_npz, np.load(proposal_path) as proposal_npz:
                for row in group:
                    reason = _sidecar_validation_error(row, raw_npz, proposal_npz)
                    if reason is None:
                        complete_rows.append(row)
                    else:
                        discarded_path_ids.append(row.path_id)
                        _notify_discard(on_discard, row.path_id, reason)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            reason = f"unreadable sidecar {logprob_file}: {exc}"
            for row in group:
                discarded_path_ids.append(row.path_id)
                _notify_discard(on_discard, row.path_id, reason)

    rows_by_path: dict[str, RolloutRecord] = {}
    for row in complete_rows:
        if row.path_id in rows_by_path:
            raise ValueError(f"{path}: duplicate complete path_id {row.path_id}")
        rows_by_path[row.path_id] = row
    canonical_rows = [
        row for row in parsed_rows if rows_by_path.get(row.path_id) is row
    ]
    if truncated_tail:
        _notify_discard(on_discard, None, "truncated final JSONL line")
    if truncated_tail or missing_final_newline or len(canonical_rows) != len(parsed_rows):
        _write_jsonl_atomic(path, canonical_rows)
    return RolloutResumeState(
        rows=canonical_rows,
        completed_path_ids=frozenset(rows_by_path),
        discarded_path_ids=tuple(discarded_path_ids),
        truncated_tail=truncated_tail,
    )


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
                        "raw_logprob_sum": row.raw_logprob_sum
                        if row.raw_logprob_sum is not None
                        else raw_logprob_sum,
                        "proposal_logprob_sum": row.proposal_logprob_sum
                        if row.proposal_logprob_sum is not None
                        else proposal_logprob_sum,
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


def _rollout_metadata_payload(
    row: RolloutRecord,
    logprob_file: str,
    logprob_dtype: str,
) -> dict:
    payload = dump_jsonable(row)
    payload.pop("token_logprobs", None)
    payload.pop("raw_token_logprobs", None)
    payload.pop("proposal_token_logprobs", None)
    payload["logprob_file"] = logprob_file
    payload["logprob_dtype"] = logprob_dtype
    return payload


def _validate_valid_rollout_arrays(row: RolloutRecord) -> None:
    raw_count = len(row.raw_token_logprobs)
    proposal_count = len(row.proposal_token_logprobs)
    if not row.is_valid or raw_count == 0 or proposal_count == 0:
        raise ValueError(f"{row.path_id}: valid rollout requires both logprob arrays")
    if raw_count != proposal_count or raw_count != row.output_token_count:
        raise ValueError(
            f"{row.path_id}: token/logprob counts differ: "
            f"token_count={row.output_token_count}, raw_count={raw_count}, "
            f"proposal_count={proposal_count}"
        )


def _read_resume_rows(path: Path) -> tuple[list[RolloutRecord], bool, bool]:
    if not path.exists():
        return [], False, False
    lines = path.read_bytes().splitlines(keepends=True)
    rows: list[RolloutRecord] = []
    truncated_tail = False
    missing_final_newline = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        is_last_partial = index == len(lines) - 1 and not line.endswith(b"\n")
        try:
            payload = json.loads(line.decode("utf-8"))
            rows.append(RolloutRecord.model_validate(payload))
            missing_final_newline = is_last_partial
        except Exception as exc:
            if is_last_partial:
                truncated_tail = True
                break
            raise ValueError(f"{path}:{index + 1}: invalid rollout row: {exc}") from exc
    return rows, truncated_tail, missing_final_newline


def _is_sharded_logprob_file(value: str) -> bool:
    path = _output_relative_path(value)
    return (
        len(path.parts) == 2
        and path.parts[0] == "logprobs"
        and path.parts[1].startswith("shard-")
    )


def _sidecar_validation_error(row, raw_npz, proposal_npz) -> str | None:
    if row.path_id not in raw_npz:
        return f"missing raw logprobs in {row.logprob_file}"
    if row.path_id not in proposal_npz:
        return f"missing proposal logprobs in {row.logprob_file}"
    raw_count = len(raw_npz[row.path_id])
    proposal_count = len(proposal_npz[row.path_id])
    if raw_count == 0 or proposal_count == 0:
        return "empty logprob sidecar array"
    if raw_count != proposal_count or raw_count != row.output_token_count:
        return (
            "token/logprob sidecar counts differ: "
            f"token_count={row.output_token_count}, raw_count={raw_count}, "
            f"proposal_count={proposal_count}"
        )
    return None


def _notify_discard(
    callback: Callable[[str | None, str], None] | None,
    path_id: str | None,
    reason: str,
) -> None:
    if callback is not None:
        callback(path_id, reason)


def _write_jsonl_atomic(path: Path, rows: Iterable[RolloutRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.resume.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dump_jsonable(row), ensure_ascii=False))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _output_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"logprob_file must be relative to the output root: {value}")
    return path
