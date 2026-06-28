from __future__ import annotations

import asyncio
import json

from src.schemas import RolloutRecord
from tests.test_helpers import ANSWER, PROBLEM, valid_reward_payload


def test_reward_scoring_dry_run_writes_candidates_and_raw_rewards(tmp_path, monkeypatch) -> None:
    import scripts.run_reward_scoring as script
    import src.judging as judging

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
    rollouts_path = tmp_path / "rollouts.jsonl"
    rollout = RolloutRecord(
        run_id="rollout-run",
        problem_id="p1",
        path_id="p1-0000",
        rollout_index=0,
        path_text=ANSWER,
        token_logprobs=[-0.1, -0.2, -0.3],
        is_valid=True,
    )
    rollouts_path.write_text(rollout.model_dump_json() + "\n", encoding="utf-8")
    prompt_path = tmp_path / "Reward_prompt.md"
    prompt_path.write_text(
        "Subject {{subject}}\n<problem>{{problem}}</problem>\n<student_answer>{{student_answer}}</student_answer>",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    async def fake_evaluate_reward(**kwargs):
        return json.dumps(valid_reward_payload())

    monkeypatch.setattr(script, "INPUT_JSONL", input_path)
    monkeypatch.setattr(script, "ROLLOUTS_JSONL", rollouts_path)
    monkeypatch.setattr(script, "REWARD_PROMPT_PATH", prompt_path)
    monkeypatch.setattr(script, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(script, "REWARD_EXTRA_BODY", None)
    monkeypatch.setattr(script, "make_async_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(judging, "evaluate_reward", fake_evaluate_reward)

    asyncio.run(script.main())

    assert (output_dir / "scoring_config.json").exists()
    assert (output_dir / "candidates.jsonl").exists()
    assert (output_dir / "reward_raw.jsonl").exists()
    assert not (output_dir / "chain.jsonl").exists()
    assert not (output_dir / "best_of_n.jsonl").exists()
    assert not (output_dir / "summary.json").exists()

    rows = [
        json.loads(line)
        for line in (output_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["reward_valid"] is True
    assert "Sη[τ]" in rows[0]

    raw_rows = [
        json.loads(line)
        for line in (output_dir / "reward_raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw_rows) == 1
    assert raw_rows[0]["valid"] is True

    config = json.loads((output_dir / "scoring_config.json").read_text(encoding="utf-8"))
    assert config["extra_body"] is None
