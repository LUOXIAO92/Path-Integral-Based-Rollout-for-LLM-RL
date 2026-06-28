from __future__ import annotations

import asyncio
import json

from src.schemas import RolloutRecord
from tests.test_helpers import ANSWER, PROBLEM


def test_openai_rollout_dry_run_writes_only_rollout_outputs(tmp_path, monkeypatch) -> None:
    import scripts.run_openai_rollout as script

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

    async def fake_run_student_rollouts(**kwargs):
        run_id = kwargs["run_id"]
        return [
            RolloutRecord(
                run_id=run_id,
                problem_id="p1",
                path_id="p1-0000",
                rollout_index=0,
                path_text=ANSWER,
                token_logprobs=[-0.1, -0.2, -0.3],
                is_valid=True,
            ),
            RolloutRecord(
                run_id=run_id,
                problem_id="p1",
                path_id="p1-0001",
                rollout_index=1,
                path_text=ANSWER,
                token_logprobs=[-0.1, -0.2, -0.3],
                is_valid=True,
            ),
        ]

    monkeypatch.setattr(script, "INPUT_JSONL", input_path)
    monkeypatch.setattr(script, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(script, "ROLLOUT_BUDGET", 2)
    monkeypatch.setattr(script, "STUDENT_EXTRA_BODY", {"top_k": 10})
    monkeypatch.setattr(script, "make_async_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(script, "run_student_rollouts", fake_run_student_rollouts)

    asyncio.run(script.main())

    assert (output_dir / "rollout_config.json").exists()
    assert (output_dir / "rollouts.jsonl").exists()
    assert not (output_dir / "candidates.jsonl").exists()
    assert not (output_dir / "reward_raw.jsonl").exists()
    assert not (output_dir / "chain.jsonl").exists()
    assert not (output_dir / "best_of_n.jsonl").exists()
    assert not (output_dir / "summary.json").exists()

    rows = [
        json.loads(line)
        for line in (output_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert all(row["is_valid"] for row in rows)

    config = json.loads((output_dir / "rollout_config.json").read_text(encoding="utf-8"))
    assert config["extra_body"] == {"top_k": 10}
