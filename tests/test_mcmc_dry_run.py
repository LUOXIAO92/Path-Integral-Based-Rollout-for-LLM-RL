from __future__ import annotations

import json

from src.schemas import PathRecord, ScoreConfig, ScoringConfig


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
    output_dir.mkdir()
    scoring_config_path = output_dir / "scoring_config.json"
    scoring_config_path.write_text(
        ScoringConfig(
            run_id="run",
            dataset="test",
            reward_model="reward",
            reward_base_url="",
            prompt_template_id="prompt",
            eta=1.0,
            lambda_G=1.0,
            lambda_N=1.0,
            lambda_KL=0.0,
            length_max=10,
            length_scale=10.0,
            score_config=ScoreConfig(),
        ).model_dump_json(),
        encoding="utf-8",
    )

    monkeypatch.setattr(script, "CANDIDATES_JSONL", candidates_path)
    monkeypatch.setattr(script, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(script, "PROPOSAL_RATIO_MODE", "normalized")
    monkeypatch.setattr(script, "STRICT_LENGTH_ALPHA", 1.0)
    monkeypatch.setattr(script, "SCORING_CONFIG_JSON", scoring_config_path)
    monkeypatch.setattr(script, "RANDOM_SEED", 1234)

    script.main()

    assert (output_dir / "mcmc_config.json").exists()
    assert (output_dir / "chain.jsonl").exists()
    assert (output_dir / "best_of_n.jsonl").exists()
    assert (output_dir / "summary.json").exists()
    config = json.loads((output_dir / "mcmc_config.json").read_text(encoding="utf-8"))
    assert config["proposal_ratio_mode"] == "normalized"
    assert config["strict_length_alpha"] == 1.0
    assert config["scoring_config_json"] == str(scoring_config_path)
    assert "rho_prop" not in config

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


def test_strict_mcmc_config_rejects_missing_scoring_coefficients(tmp_path, monkeypatch) -> None:
    import scripts.run_mcmc as script

    scoring_config_path = tmp_path / "bad_scoring_config.json"
    scoring_config_path.write_text(
        json.dumps(
            {
                "run_id": "run",
                "dataset": "test",
                "reward_model": "reward",
                "reward_base_url": "",
                "prompt_template_id": "prompt",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(script, "PROPOSAL_RATIO_MODE", "strict")
    monkeypatch.setattr(script, "SCORING_CONFIG_JSON", scoring_config_path)

    try:
        script.load_strict_action_config()
    except Exception as exc:
        assert "eta" in str(exc)
    else:
        raise AssertionError("strict mode should reject incomplete scoring config")


def candidate(problem_id: str, path_id: str, s_eta: float, g: float) -> PathRecord:
    return PathRecord(
        run_id="run",
        problem_id=problem_id,
        method="mcmc_candidate",
        path_id=path_id,
        path_text=path_id,
        output_token_count=10,
        proposal_logprob_sum=-1.0,
        proposal_logprob_mean=-0.1,
        proposal_distribution="test_distribution",
        reward_valid=True,
        g=g,
        n=0.0,
        k=0.0,
        f=g,
        s0=s_eta,
        s_eta=s_eta,
        final_correctness=g > 0.8,
    )
