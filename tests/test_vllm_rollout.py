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
    build_chat_rollout_request,
    ensure_vllm_available,
    iter_request_chunks,
    make_llm,
    normalize_token_ids,
    prepare_generated_rollout,
    run_local_rollouts,
    valid_vllm_record,
)
from tests.test_helpers import PROBLEM


class FakeChatTokenizer:
    chat_template = "qwen"

    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        assert add_generation_prompt is True
        if tokenize:
            return [101]
        return "<chat>"


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


def test_make_llm_rejects_unsupported_required_scheduler_limits() -> None:
    class FakeLLM:
        def __init__(self, model: str) -> None:
            self.model = model

    with pytest.raises(TypeError, match="max_num_seqs"):
        make_llm(
            FakeLLM,
            required_kwargs={
                "gpu_memory_utilization",
                "max_num_batched_tokens",
                "max_num_seqs",
            },
            model="mock-model",
            gpu_memory_utilization=0.92,
            max_num_batched_tokens=8192,
            max_num_seqs=4,
        )


def test_build_chat_rollout_request_requires_chat_template() -> None:
    class FakeTokenizer:
        chat_template = "qwen"

        def __init__(self) -> None:
            self.calls: list[tuple[bool, bool]] = []

        def apply_chat_template(
            self,
            messages,
            tokenize: bool,
            add_generation_prompt: bool,
        ):
            self.calls.append((tokenize, add_generation_prompt))
            assert messages == [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"Question: {PROBLEM}"},
            ]
            if tokenize:
                return [101, 102, 103]
            return "<chat>Question</chat>"

    tokenizer = FakeTokenizer()
    problem = ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)

    request = build_chat_rollout_request(
        problem,
        0,
        tokenizer,
        "system",
        "Question: {problem}",
    )

    assert request.prompt_text == "<chat>Question</chat>"
    assert request.prompt_token_ids == [101, 102, 103]
    assert tokenizer.calls == [(False, True), (True, True)]


def test_normalize_token_ids_accepts_mapping_input_ids() -> None:
    assert normalize_token_ids(
        {
            "input_ids": [101, 102],
            "attention_mask": [1, 1],
        }
    ) == [101, 102]


def test_build_chat_rollout_request_accepts_batch_encoding_style_token_ids() -> None:
    class FakeBatchEncoding(dict):
        pass

    class FakeTokenizer:
        chat_template = "qwen"

        def apply_chat_template(
            self,
            messages,
            tokenize: bool,
            add_generation_prompt: bool,
        ):
            assert add_generation_prompt is True
            if tokenize:
                return FakeBatchEncoding(
                    {
                        "input_ids": [101, 102],
                        "attention_mask": [1, 1],
                    }
                )
            return "<chat>"

    request = build_chat_rollout_request(
        ProblemInput(problem_id="p1", subject="math", problem=PROBLEM),
        0,
        FakeTokenizer(),
        "system",
        "{problem}",
    )

    assert request.prompt_text == "<chat>"
    assert request.prompt_token_ids == [101, 102]


def test_normalize_token_ids_rejects_mapping_without_input_ids() -> None:
    with pytest.raises(ValueError, match="without input_ids.*keys=\\[attention_mask\\]"):
        normalize_token_ids({"attention_mask": [1, 1]})


def test_build_chat_rollout_request_rejects_missing_chat_template() -> None:
    class FakeTokenizer:
        chat_template = None

        def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
            raise AssertionError("should fail before applying template")

    with pytest.raises(ValueError, match="chat_template"):
        build_chat_rollout_request(
            ProblemInput(problem_id="p1", subject="math", problem=PROBLEM),
            0,
            FakeTokenizer(),
            "system",
            "{problem}",
        )


def test_batch_raw_prefill_logprobs_handles_different_completion_lengths() -> None:
    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text)))

    class FakeLLM:
        def __init__(self) -> None:
            self.generate_use_tqdm = None
            self.generate_prompts = None

        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompts, params, use_tqdm=True):
            self.generate_use_tqdm = use_tqdm
            self.generate_prompts = prompts
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
            request=RolloutRequest(problem, 0, "ab", [1, 2]),
            output=SimpleNamespace(),
            path_id="p1-0000",
            path_text="cd",
            token_ids=[10, 11],
            proposal_logprobs=[-0.01, -0.02],
        ),
        GeneratedRollout(
            request=RolloutRequest(problem, 1, "xyz", [1, 2, 3]),
            output=SimpleNamespace(),
            path_id="p1-0001",
            path_text="q",
            token_ids=[20],
            proposal_logprobs=[-0.03],
        ),
    ]

    llm = FakeLLM()
    results = batch_raw_prefill_logprobs(llm, object(), generated)

    assert llm.generate_use_tqdm is False
    assert llm.generate_prompts == [
        {"prompt_token_ids": [1, 2, 10, 11]},
        {"prompt_token_ids": [1, 2, 3, 20]},
    ]
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
            tokenizer=FakeChatTokenizer(),
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
            tokenizer=FakeChatTokenizer(),
            system_prompt="system",
            user_template="{problem}",
            batch_size=4,
        )
    )

    assert [len(chunk) for chunk in chunks] == [4, 2]
    assert chunks[0][-1].problem.problem_id == "p2"
    assert chunks[0][-1].rollout_index == 0


def test_vllm_rollouts_use_single_progress_bar(tmp_path, monkeypatch) -> None:
    import src.vllm_rollout as vllm_rollout

    class FakeTokenizer:
        chat_template = "qwen"

        def __init__(self) -> None:
            self.chat_calls = 0

        def apply_chat_template(
            self,
            messages,
            tokenize: bool,
            add_generation_prompt: bool,
        ):
            self.chat_calls += 1
            assert add_generation_prompt is True
            if tokenize:
                return [101, 102]
            return "<chat>"

    class FakeLLM:
        instances = []

        def __init__(self, model: str) -> None:
            self.model = model
            self.generate_use_tqdm: list[bool] = []
            self.generate_prompts: list[list[object]] = []
            self.sampling_params = []
            self.tokenizer = FakeTokenizer()
            FakeLLM.instances.append(self)

        def get_tokenizer(self):
            return self.tokenizer

        def generate(self, prompts, params, use_tqdm=True):
            self.generate_use_tqdm.append(use_tqdm)
            self.generate_prompts.append(prompts)
            self.sampling_params.append(params)
            if getattr(params, "prompt_logprobs", None):
                return [
                    SimpleNamespace(prompt_logprobs=[{}, {}, {7: -0.70}])
                    for _ in prompts
                ]
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
        def __init__(
            self,
            temperature: float,
            top_p: float,
            top_k: int,
            max_tokens: int,
            logprobs: int | None = None,
            prompt_logprobs: int | None = None,
        ) -> None:
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.max_tokens = max_tokens
            self.logprobs = logprobs
            self.prompt_logprobs = prompt_logprobs

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

    debug_path = tmp_path / "debug" / "vllm_generate_events.jsonl"
    rows = list(
        vllm_rollout.run_vllm_rollouts(
            run_id="run",
            problems=[ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)],
            rollout_budget=2,
            model="fake",
            temperature=1.0,
            top_p=1.0,
            top_k=20,
            max_tokens=8,
            batch_size=2,
            max_num_batched_tokens=None,
            max_num_seqs=None,
            gpu_memory_utilization=None,
            system_prompt="system",
            user_template="{problem}",
            debug_event_path=debug_path,
        )
    )

    assert [row.is_valid for row in rows] == [True, True]
    assert progress.total == 2
    assert progress.desc == "vLLM rollout"
    assert progress.updates == [1, 1]
    assert progress.closed is True
    assert FakeLLM.instances[0].generate_use_tqdm == [False, False]
    assert FakeLLM.instances[0].generate_prompts[0] == [
        {"prompt_token_ids": [101, 102]},
        {"prompt_token_ids": [101, 102]},
    ]
    assert FakeLLM.instances[0].generate_prompts[1] == [
        {"prompt_token_ids": [101, 102, 7]},
        {"prompt_token_ids": [101, 102, 7]},
    ]
    assert FakeLLM.instances[0].sampling_params[0].logprobs == 20
    assert not hasattr(FakeLLM.instances[0].sampling_params[0], "logprobs_mode")
    assert FakeLLM.instances[0].sampling_params[1].prompt_logprobs == 20
    assert not hasattr(FakeLLM.instances[0].sampling_params[1], "logprobs_mode")
    assert rows[0].proposal_token_logprobs == [-0.50]
    assert rows[0].raw_token_logprobs == [-0.70]
    assert rows[0].proposal_distribution == "vllm_sample_logprobs"
    assert rows[0].raw_logprob_source == "vllm_prompt_logprobs"

    events = [
        json.loads(line)
        for line in debug_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "vllm_call_start",
        "vllm_call_end",
        "vllm_call_start",
        "vllm_call_end",
        "record_yield",
        "record_yield",
    ]
    assert all(event["event_time"] for event in events)
    assert events[0]["vllm_call_id"] == events[1]["vllm_call_id"] == 1
    assert events[0]["operation"] == "sample_completion"
    assert events[0]["batch_size"] == 2
    assert events[0]["items"][0]["path_id"] == "p1-0000"
    assert events[1]["items"][0]["completion_token_count"] == 1
    assert events[2]["vllm_call_id"] == events[3]["vllm_call_id"] == 2
    assert events[2]["operation"] == "score_completion_logprobs"
    assert events[2]["items"][0]["full_token_count"] == 3
    assert events[3]["items"][0]["raw_logprob_count"] == 1
    assert events[4]["event"] == "record_yield"
    assert events[4]["path_id"] == "p1-0000"


def test_prepare_generated_rollout_requires_sampled_token_logprob() -> None:
    problem = ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)
    request = RolloutRequest(problem, 0, "<chat>", [101])
    output = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                text="x",
                token_ids=[7],
                logprobs=[{8: -0.1}],
            )
        ]
    )

    generated = prepare_generated_rollout(request, output)

    assert generated.error is not None
    assert "missing logprob for selected token id 7" in generated.error


def test_vllm_rollouts_fail_fast_when_chunk_lacks_proposal_logprobs(monkeypatch) -> None:
    import src.vllm_rollout as vllm_rollout

    class FakeTokenizer:
        chat_template = "qwen"

        def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
            return [101] if tokenize else "<chat>"

    class FakeLLM:
        def __init__(self, model: str) -> None:
            self.tokenizer = FakeTokenizer()

        def get_tokenizer(self):
            return self.tokenizer

        def generate(self, prompts, params, use_tqdm=True):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="x",
                            token_ids=[7],
                            logprobs=[{8: -0.1}],
                        )
                    ]
                )
                for _ in prompts
            ]

    class FakeSamplingParams:
        def __init__(
            self,
            temperature: float,
            top_p: float,
            top_k: int,
            max_tokens: int,
            logprobs: int | None = None,
            prompt_logprobs: int | None = None,
        ) -> None:
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.max_tokens = max_tokens
            self.logprobs = logprobs
            self.prompt_logprobs = prompt_logprobs

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(vllm_rollout, "ensure_vllm_available", lambda: None)

    with pytest.raises(RuntimeError, match="proposal logprob extraction failed"):
        list(
            vllm_rollout.run_vllm_rollouts(
                run_id="run",
                problems=[ProblemInput(problem_id="p1", subject="math", problem=PROBLEM)],
                rollout_budget=2,
                model="fake",
                temperature=1.0,
                top_p=1.0,
                top_k=20,
                max_tokens=8,
                batch_size=2,
                max_num_batched_tokens=None,
                max_num_seqs=None,
                gpu_memory_utilization=None,
                system_prompt="system",
                user_template="{problem}",
            )
        )


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
    monkeypatch.setattr(
        script,
        "DEBUG_EVENT_PATH",
        output_dir / "debug" / "vllm_generate_events.jsonl",
    )
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
    assert captured["debug_event_path"] == output_dir / "debug" / "vllm_generate_events.jsonl"
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
        },
        "debug": {
            "vllm_generate_events": script.display_path(
                output_dir / "debug" / "vllm_generate_events.jsonl"
            ),
        },
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
