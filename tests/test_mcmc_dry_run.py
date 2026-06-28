from __future__ import annotations

import json

from src.schemas import PathRecord


def test_mcmc_dry_run_writes_chain_best_of_n_and_summary(tmp_path, monkeypatch) -> None:
    import scripts.run_mcmc as script

    candidates_path = tmp_path / "candidates.jsonl"
    candidates = [
        candidate("p1", "a", s_eta=1.0, g=0.4),
        candidate("p1", "b", s_eta=0.5, g=0.9),
    ]
    candidates_path.write_text(
        "\n".join(record.model_dump_json(by_alias=True) for record in candidates) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(script, "CANDIDATES_JSONL", candidates_path)
    monkeypatch.setattr(script, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(script, "RHO_PROP", 1.0)
    monkeypatch.setattr(script, "RANDOM_SEED", 1234)

    script.main()

    assert (output_dir / "mcmc_config.json").exists()
    assert (output_dir / "chain.jsonl").exists()
    assert (output_dir / "best_of_n.jsonl").exists()
    assert (output_dir / "summary.json").exists()

    chain_rows = [
        json.loads(line)
        for line in (output_dir / "chain.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(chain_rows) == 2

    best_rows = [
        json.loads(line)
        for line in (output_dir / "best_of_n.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["method"] for row in best_rows] == ["best_of_n_s_eta", "best_of_n_g"]


def candidate(problem_id: str, path_id: str, s_eta: float, g: float) -> PathRecord:
    return PathRecord(
        run_id="run",
        problem_id=problem_id,
        method="mcmc_candidate",
        path_id=path_id,
        path_text=path_id,
        reward_valid=True,
        g=g,
        n=0.0,
        k=0.0,
        f=g,
        s0=s_eta,
        s_eta=s_eta,
        final_correctness=g > 0.8,
    )
