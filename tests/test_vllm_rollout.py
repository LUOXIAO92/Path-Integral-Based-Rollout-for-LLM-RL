from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from src.io_utils import hydrate_rollout_logprobs, read_model_jsonl
from src.prompts import student_messages
from src.schemas import ProblemInput, RolloutRecord
from src.vllm_rollout import (
    GeneratedRollout,
    RolloutRequest,
    VLLMUnavailableError,
    batch_raw_prefill_logprobs,
    ensure_vllm_available,
    iter_request_chunks,
    make_llm,
    make_sampling_params,
    run_local_rollouts,
    valid_vllm_record,
)
from tests.test_helpers import PROBLEM


def test_student_messages_use_prompt_template() -> None:
    problem = ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)
    messages = student_messages(problem, "system", "Question: {problem}")

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": f"Question: {PROBLEM}"},
    ]


def test_mock_backend_writes_dual_logprobs() -> None:
    rows = list(
        run_local_rollouts(
            backend="mock",
            run_id="run",
            problems=[ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)],
            rollout_budget=2,
            model="mock-model",
            temperature=0.8,
            top_p=0.95,
            top_k=20,
            max_tokens=128,
            batch_size=None,
            max_num_batched_tokens=None,
            max_num_seqs=None,
            gpu_memory_utilization=None,
            system_prompt="system",
            user_template="{problem}",
        )
    )

    assert len(rows) == 2
    assert all(row.is_valid for row in rows)
    assert rows[0].raw_token_logprobs
    assert rows[0].proposal_token_logprobs
    assert rows[0].output_token_count == 3
    assert rows[0].raw_logprob_sum == sum(rows[0].raw_token_logprobs)
    assert rows[0].proposal_logprob_sum == sum(rows[0].proposal_token_logprobs)
    assert rows[0].raw_logprob_mean == pytest.approx(-0.20)
    assert rows[0].proposal_logprob_mean == pytest.approx(-0.15)
    assert rows[0].proposal_distribution == "mock_processed"
    assert rows[0].raw_logprob_source == "mock_prefill"


def test_make_sampling_params_drops_unsupported_logprobs_mode() -> None:
    class FakeSamplingParams:
        def __init__(
            self,
            max_tokens: int | None = 16,
            logprobs: int | None = None,
        ) -> None:
            self.max_tokens = max_tokens
            self.logprobs = logprobs

    params = make_sampling_params(
        FakeSamplingParams,
        max_tokens=2048,
        logprobs=1,
        logprobs_mode="processed_logprobs",
    )

    assert params.max_tokens == 2048
    assert params.logprobs == 1


def test_make_sampling_params_preserves_supported_logprobs_mode() -> None:
    class FakeSamplingParams:
        def __init__(
            self,
            max_tokens: int | None = 16,
            logprobs: int | None = None,
            logprobs_mode: str | None = None,
        ) -> None:
            self.max_tokens = max_tokens
            self.logprobs = logprobs
            self.logprobs_mode = logprobs_mode

    params = make_sampling_params(
        FakeSamplingParams,
        max_tokens=2048,
        logprobs=1,
        logprobs_mode="processed_logprobs",
    )

    assert params.max_tokens == 2048
    assert params.logprobs == 1
    assert params.logprobs_mode == "processed_logprobs"


def test_make_llm_preserves_supported_scheduler_limits() -> None:
    class FakeLLM:
        def __init__(
            self,
            model: str,
            gpu_memory_utilization: float | None = None,
            max_num_batched_tokens: int | None = None,
            max_num_seqs: int | None = None,
        ) -> None:
            self.model = model
            self.gpu_memory_utilization = gpu_memory_utilization
            self.max_num_batched_tokens = max_num_batched_tokens
            self.max_num_seqs = max_num_seqs

    llm = make_llm(
        FakeLLM,
        model="mock-model",
        gpu_memory_utilization=0.92,
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    )

    assert llm.model == "mock-model"
    assert llm.gpu_memory_utilization == 0.92
    assert llm.max_num_batched_tokens == 8192
    assert llm.max_num_seqs == 4


def test_make_llm_drops_unsupported_scheduler_limits() -> None:
    class FakeLLM:
        def __init__(self, model: str) -> None:
            self.model = model

    llm = make_llm(
        FakeLLM,
        model="mock-model",
        gpu_memory_utilization=0.92,
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    )

    assert llm.model == "mock-model"


def test_batch_raw_prefill_logprobs_handles_different_completion_lengths() -> None:
    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text)))

    class FakeLLM:
        def __init__(self) -> None:
            self.generate_use_tqdm = None

        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompts, params, use_tqdm=True):
            self.generate_use_tqdm = use_tqdm
            return [
                SimpleNamespace(
                    prompt_logprobs=[
                        {},
                        {},
                        {10: -0.10},
                        {11: -0.20},
                    ]
                ),
                SimpleNamespace(
                    prompt_logprobs=[
                        {},
                        {},
                        {},
                        {20: -0.30},
                    ]
                ),
            ]

    problem = ProblemInput(problem_id="p1", subject="math", problem="x")
    generated = [
        GeneratedRollout(
            request=RolloutRequest(problem, 0, "ab"),
            output=SimpleNamespace(),
            path_id="p1-0000",
            path_text="cd",
            token_ids=[10, 11],
        ),
        GeneratedRollout(
            request=RolloutRequest(problem, 1, "xyz"),
            output=SimpleNamespace(),
            path_id="p1-0001",
            path_text="q",
            token_ids=[20],
        ),
    ]

    llm = FakeLLM()
    results = batch_raw_prefill_logprobs(llm, object(), generated)

    assert llm.generate_use_tqdm is False
    assert results == [[-0.10, -0.20], [-0.30]]


def test_iter_request_chunks_flattens_by_default() -> None:
    problems = [
        ProblemInput(problem_id="p1", subject="math", problem="one"),
        ProblemInput(problem_id="p2", subject="math", problem="two"),
    ]

    chunks = list(
        iter_request_chunks(
            problems=problems,
            rollout_budget=3,
            system_prompt="system",
            user_template="{problem}",
            batch_size=None,
        )
    )

    assert [
        (request.problem.problem_id, request.rollout_index)
        for chunk in chunks
        for request in chunk
    ] == [
        ("p1", 0),
        ("p1", 1),
        ("p1", 2),
        ("p2", 0),
        ("p2", 1),
        ("p2", 2),
    ]
    assert [len(chunk) for chunk in chunks] == [3, 3]


def test_iter_request_chunks_flattens_with_explicit_batch_size() -> None:
    problems = [
        ProblemInput(problem_id="p1", subject="math", problem="one"),
        ProblemInput(problem_id="p2", subject="math", problem="two"),
    ]

    chunks = list(
        iter_request_chunks(
            problems=problems,
            rollout_budget=3,
            system_prompt="system",
            user_template="{problem}",
            batch_size=4,
        )
    )

    assert [len(chunk) for chunk in chunks] == [4, 2]
    assert chunks[0][-1].problem.problem_id == "p2"
    assert chunks[0][-1].rollout_index == 0


def test_vllm_rollouts_use_single_progress_bar(monkeypatch) -> None:
    import src.vllm_rollout as vllm_rollout

    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return []

    class FakeLLM:
        instances = []

        def __init__(self, model: str) -> None:
            self.model = model
            self.generate_use_tqdm: list[bool] = []
            FakeLLM.instances.append(self)

        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompts, params, use_tqdm=True):
            self.generate_use_tqdm.append(use_tqdm)
            if getattr(params, "prompt_logprobs", None):
                return [SimpleNamespace(prompt_logprobs=[{7: -0.70}]) for _ in prompts]
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="a",
                            token_ids=[7],
                            logprobs=[{7: -0.50}],
                        )
                    ]
                )
                for _ in prompts
            ]

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeProgress:
        def __init__(self, total: int, desc: str) -> None:
            self.total = total
            self.desc = desc
            self.updates: list[int] = []
            self.closed = False

        def update(self, count: int) -> None:
            self.updates.append(count)

        def close(self) -> None:
            self.closed = True

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    progress = FakeProgress(total=0, desc="")

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(vllm_rollout, "ensure_vllm_available", lambda: None)
    monkeypatch.setattr(
        vllm_rollout,
        "make_progress",
        lambda total, desc: progress.__dict__.update(total=total, desc=desc) or progress,
    )

    rows = list(
        vllm_rollout.run_vllm_rollouts(
            run_id="run",
            problems=[ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)],
            rollout_budget=2,
            model="fake",
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_tokens=8,
            batch_size=2,
            max_num_batched_tokens=None,
            max_num_seqs=None,
            gpu_memory_utilization=None,
            system_prompt="system",
            user_template="{problem}",
        )
    )

    assert [row.is_valid for row in rows] == [True, True]
    assert progress.total == 2
    assert progress.desc == "vLLM rollout"
    assert progress.updates == [1, 1]
    assert progress.closed is True
    assert FakeLLM.instances[0].generate_use_tqdm == [False, False]


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
    monkeypatch.setattr(script, "STUDENT_BATCH_SIZE", None)
    monkeypatch.setattr(script, "STUDENT_MAX_NUM_BATCHED_TOKENS", 8192)
    monkeypatch.setattr(script, "STUDENT_MAX_NUM_SEQS", 4)
    monkeypatch.setattr(script, "STUDENT_GPU_MEMORY_UTILIZATION", 0.92)
    captured: dict[str, object] = {}
    real_run_local_rollouts = script.run_local_rollouts
    real_write_rollouts_with_logprob_sidecars = script.write_rollouts_with_logprob_sidecars

    def capture_run_local_rollouts(**kwargs):
        captured.update(kwargs)
        return real_run_local_rollouts(**kwargs)

    def capture_write_rollouts_with_logprob_sidecars(path, rows):
        captured["rows_is_list"] = isinstance(rows, list)
        return real_write_rollouts_with_logprob_sidecars(path, rows)

    monkeypatch.setattr(script, "run_local_rollouts", capture_run_local_rollouts)
    monkeypatch.setattr(
        script,
        "write_rollouts_with_logprob_sidecars",
        capture_write_rollouts_with_logprob_sidecars,
    )

    script.main()

    assert captured["batch_size"] is None
    assert captured["max_num_batched_tokens"] == 8192
    assert captured["max_num_seqs"] == 4
    assert captured["gpu_memory_utilization"] == 0.92
    assert captured["rows_is_list"] is False
    assert (output_dir / "rollout_config.json").exists()
    assert (output_dir / "rollouts.jsonl").exists()
    assert (output_dir / "logprobs" / "raw.npz").exists()
    assert (output_dir / "logprobs" / "proposal.npz").exists()
    config = json.loads((output_dir / "rollout_config.json").read_text(encoding="utf-8"))
    assert config["extra_body"] == {
        "vllm_engine_args": {
            "gpu_memory_utilization": 0.92,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 4,
        }
    }
    rows = [
        json.loads(line)
        for line in (output_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["logprob_file"] == "logprobs"
    assert rows[0]["logprob_dtype"] == "float32"
    assert rows[0]["output_token_count"] == 3
    assert rows[0]["raw_logprob_sum"] == pytest.approx(-0.60)
    assert rows[0]["proposal_logprob_sum"] == pytest.approx(-0.45)
    assert rows[0]["raw_logprob_mean"] == pytest.approx(-0.20)
    assert rows[0]["proposal_logprob_mean"] == pytest.approx(-0.15)
    assert "token_logprobs" not in rows[0]
    assert "raw_token_logprobs" not in rows[0]
    assert "proposal_token_logprobs" not in rows[0]

    path_id = rows[0]["path_id"]
    with (
        np.load(output_dir / rows[0]["logprob_file"] / "raw.npz") as raw_npz,
        np.load(output_dir / rows[0]["logprob_file"] / "proposal.npz") as proposal_npz,
    ):
        np.testing.assert_allclose(raw_npz[path_id], [-0.30, -0.20, -0.10])
        np.testing.assert_allclose(proposal_npz[path_id], [-0.25, -0.15, -0.05])

    rollout = read_model_jsonl(output_dir / "rollouts.jsonl", RolloutRecord)[0].model_copy(
        update={
            "raw_logprob_sum": None,
            "proposal_logprob_sum": None,
            "raw_logprob_mean": None,
            "proposal_logprob_mean": None,
        }
    )
    assert rollout.raw_token_logprobs == []
    hydrated = hydrate_rollout_logprobs([rollout], output_dir)
    assert hydrated[0].output_token_count == 3
    assert hydrated[0].raw_logprob_sum == pytest.approx(-0.60)
    assert hydrated[0].proposal_logprob_sum == pytest.approx(-0.45)
    assert hydrated[0].raw_logprob_mean == pytest.approx(-0.20)
    assert hydrated[0].proposal_logprob_mean == pytest.approx(-0.15)
    assert hydrated[0].raw_token_logprobs == pytest.approx([-0.30, -0.20, -0.10])
    assert hydrated[0].proposal_token_logprobs == pytest.approx([-0.25, -0.15, -0.05])
