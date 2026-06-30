from __future__ import annotations

import json

import pytest

from src.schemas import ProblemInput
from src.vllm_rollout import (
    VLLMUnavailableError,
    ensure_vllm_available,
    run_local_rollouts,
    valid_vllm_record,
)
from tests.test_helpers import PROBLEM


def test_mock_backend_writes_dual_logprobs() -> None:
    rows = run_local_rollouts(
        backend="mock",
        run_id="run",
        problems=[ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)],
        rollout_budget=2,
        model="mock-model",
        temperature=0.8,
        top_p=0.95,
        top_k=20,
        max_tokens=128,
        system_prompt="system",
        user_template="{problem}",
    )

    assert len(rows) == 2
    assert all(row.is_valid for row in rows)
    assert rows[0].raw_token_logprobs
    assert rows[0].proposal_token_logprobs
    assert rows[0].raw_logprob_sum == sum(rows[0].raw_token_logprobs)
    assert rows[0].proposal_logprob_sum == sum(rows[0].proposal_token_logprobs)
    assert rows[0].proposal_distribution == "mock_processed"
    assert rows[0].raw_logprob_source == "mock_prefill"


def test_valid_vllm_record_requires_raw_and_proposal_logprobs() -> None:
    with pytest.raises(ValueError, match="raw and proposal"):
        valid_vllm_record(
            run_id="run",
            problem_id="p1",
            path_id="p1-0000",
            rollout_index=0,
            path_text="answer",
            raw_logprobs=[],
            proposal_logprobs=[-0.1],
        )

    with pytest.raises(ValueError, match="same length"):
        valid_vllm_record(
            run_id="run",
            problem_id="p1",
            path_id="p1-0000",
            rollout_index=0,
            path_text="answer",
            raw_logprobs=[-0.2],
            proposal_logprobs=[-0.1, -0.2],
        )


def test_vllm_backend_reports_missing_vllm(monkeypatch) -> None:
    import src.vllm_rollout as vllm_rollout

    monkeypatch.setattr(vllm_rollout.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(VLLMUnavailableError, match="vLLM is not installed"):
        ensure_vllm_available()


def test_run_vllm_rollout_script_mock_backend(tmp_path, monkeypatch) -> None:
    import scripts.run_vllm_rollout as script

    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "problem_id": "p1",
                "subject": "math",
                "problem": PROBLEM,
                "reference_answer": "42",
                "test_result": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(script, "BACKEND", "mock")
    monkeypatch.setattr(script, "INPUT_JSONL", input_path)
    monkeypatch.setattr(script, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(script, "ROLLOUT_BUDGET", 1)

    script.main()

    assert (output_dir / "rollout_config.json").exists()
    assert (output_dir / "rollouts.jsonl").exists()
    rows = [
        json.loads(line)
        for line in (output_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["raw_token_logprobs"]
    assert rows[0]["proposal_token_logprobs"]
