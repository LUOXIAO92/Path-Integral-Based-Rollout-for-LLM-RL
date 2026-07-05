from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RawTextSpan(StrictModel):
    start_text: str
    end_text: str


class ProblemInput(StrictModel):
    problem_id: str
    subject: str
    problem: str
    reference_answer: str = ""
    test_result: str = ""


class DatasetSourceManifest(StrictModel):
    source: str
    raw_rows: int = 0
    kept_rows: int = 0
    skipped_rows: int = 0
    numeric_filtered_rows: int = 0
    configs: list[str] = []
    notes: list[str] = []


class DatasetPrepManifest(StrictModel):
    output_jsonl: str
    total_rows: int
    sources: list[DatasetSourceManifest]


class ProblemAnalysis(StrictModel):
    task_target: str
    required_output: str
    hard_constraints: list[str]
    possible_off_task_patterns: list[str]


class ProblemSpan(StrictModel):
    problem_span_id: str
    raw_text_span: RawTextSpan
    span_role: Literal[
        "task_target",
        "condition",
        "constraint",
        "input_format",
        "output_format",
        "other_requirement",
    ]
    normalized_requirement: str


class ReferencePoint(StrictModel):
    reference_point_id: str
    content: str
    source: Literal["given_solution", "minimal_requirement"]


class ReferenceAnalysis(StrictModel):
    reference_type: Literal["final_only", "detailed_solution", "test_based", "no_reference"]
    final_answer: str
    given_solution_key_points: list[ReferencePoint]
    inferred_minimal_requirements: list[ReferencePoint]


class StudentSpan(StrictModel):
    student_span_id: str
    raw_text_span: RawTextSpan
    span_type: Literal["key_reasoning", "non_key_reasoning", "irrelevant", "final_answer"]
    problem_span_refs: list[str]


class MainClaim(StrictModel):
    claim_id: str
    student_span_refs: list[str]
    claim_text: str
    claim_role: str


class StudentAnswerAnalysis(StrictModel):
    answer_summary: str
    main_claims: list[MainClaim]


class AlignmentItem(StrictModel):
    problem_span_refs: list[str]
    student_span_refs: list[str]
    alignment_status: Literal["matched", "partially_matched", "missing", "contradicted", "irrelevant"]
    reason: str


class OffTaskEvidence(StrictModel):
    student_span_refs: list[str]
    reason: str


class StudentAlignment(StrictModel):
    responds_to_problem: bool
    off_task: bool
    alignment_items: list[AlignmentItem]
    off_task_evidence: list[OffTaskEvidence]
    decision_reason: str


class SpanEvaluation(StrictModel):
    student_span_id: str
    problem_span_refs: list[str]
    reference_point_refs: list[str]
    is_relevant: bool
    is_key_reasoning: bool
    step_score: float = Field(ge=0.0, le=1.0)
    error_type: Literal[
        "none",
        "minor_error",
        "major_error",
        "fatal_error",
        "unsupported_claim",
        "irrelevant",
        "off_task",
        "unclear",
    ]
    reason: str


class FinalAnswerCheck(StrictModel):
    student_final_answer_span_id: str
    is_correct: bool
    reason: str


class RewardEvaluation(StrictModel):
    problem_analysis: ProblemAnalysis
    problem_spans: list[ProblemSpan]
    reference_analysis: ReferenceAnalysis
    student_spans: list[StudentSpan]
    student_answer_analysis: StudentAnswerAnalysis
    student_alignment: StudentAlignment
    span_evaluations: list[SpanEvaluation]
    final_answer_check: FinalAnswerCheck


class ScoreConfig(StrictModel):
    lambda_R: float = 0.4
    lambda_A: float = 0.6
    w_key_reasoning: float = 1.0
    w_non_key_reasoning: float = 0.25
    w_irrelevant: float = 0.0
    final_correct_score: float = 1.0
    final_wrong_score: float = 0.0
    off_task_score: float = 0.0
    no_scored_span_process_score: float = 0.0


class RolloutRecord(StrictModel):
    run_id: str
    problem_id: str
    path_id: str
    rollout_index: int
    path_text: str = ""
    token_logprobs: list[float] = []
    raw_token_logprobs: list[float] = []
    proposal_token_logprobs: list[float] = []
    output_token_count: int = 0
    raw_logprob_sum: float | None = None
    proposal_logprob_sum: float | None = None
    raw_logprob_mean: float | None = None
    proposal_logprob_mean: float | None = None
    logprob_file: str = ""
    logprob_dtype: str = ""
    proposal_distribution: str = ""
    raw_logprob_source: str = ""
    is_valid: bool = False
    error: str | None = None


class RolloutConfig(StrictModel):
    run_id: str
    dataset: str
    student_model: str
    student_base_url: str
    backend: str = "openai"
    temperature: float
    top_p: float
    top_k: int | None = None
    max_tokens: int
    extra_body: dict | None = None
    rollout_budget: int


class ScoredReward(StrictModel):
    process_score: float
    final_score: float
    g: float
    final_correctness: bool


class PathMetrics(StrictModel):
    n: float = Field(validation_alias="N[τ]", serialization_alias="N[τ]")
    k: float = Field(validation_alias="K[τ]", serialization_alias="K[τ]")
    s0: float = Field(validation_alias="S0[τ]", serialization_alias="S0[τ]")
    f: float = Field(validation_alias="F[τ]", serialization_alias="F[τ]")
    s_eta: float = Field(validation_alias="Sη[τ]", serialization_alias="Sη[τ]")


class PathRecord(StrictModel):
    run_id: str
    problem_id: str
    method: str
    path_id: str
    chain_step: int | None = None
    source_path_id: str | None = None
    path_text: str = ""
    output_token_count: int | None = None
    proposal_logprob_sum: float | None = None
    proposal_logprob_mean: float | None = None
    proposal_distribution: str = ""
    proposal_ratio_mode: Literal["normalized", "strict"] | None = None
    proposal_log_q_forward: float | None = None
    proposal_log_q_reverse: float | None = None
    proposal_log_ratio: float | None = None
    proposal_log_ratio_strict: float | None = None
    proposal_log_ratio_normalized: float | None = None
    strict_length_alpha: float | None = None
    strict_length_penalty_scaled: float | None = None
    strict_f: float | None = None
    strict_s_eta: float | None = None
    selected_s_eta_current: float | None = None
    selected_s_eta_candidate: float | None = None
    is_accepted: bool | None = None
    g: float | None = Field(default=None, validation_alias="G[τ]", serialization_alias="G[τ]")
    n: float | None = Field(default=None, validation_alias="N[τ]", serialization_alias="N[τ]")
    k: float | None = Field(default=None, validation_alias="K[τ]", serialization_alias="K[τ]")
    f: float | None = Field(default=None, validation_alias="F[τ]", serialization_alias="F[τ]")
    s0: float | None = Field(default=None, validation_alias="S0[τ]", serialization_alias="S0[τ]")
    s_eta: float | None = Field(default=None, validation_alias="Sη[τ]", serialization_alias="Sη[τ]")
    acceptance_prob: float | None = Field(
        default=None,
        validation_alias="A_k(τ→τ')",
        serialization_alias="A_k(τ→τ')",
    )
    final_correctness: bool | None = None
    reward_valid: bool = False
    reward_attempts: int = 0
    error: str | None = None


class ScoringConfig(StrictModel):
    run_id: str
    dataset: str
    reward_model: str
    reward_base_url: str
    prompt_template_id: str
    extra_body: dict | None = None
    eta: float
    lambda_G: float
    lambda_N: float
    lambda_KL: float
    length_max: int
    length_scale: float
    strict_length_alpha: float | None = None
    score_config: ScoreConfig


class MCMCConfig(StrictModel):
    proposal_ratio_mode: Literal["normalized", "strict"]
    scoring_config_json: str | None = None
    random_seed: int
    candidates_jsonl: str
    output_dir: str


class Summary(StrictModel):
    run_id: str
    total_candidates: int
    valid_candidates: int
    accepted_candidates: int
    rejected_candidates: int
    mean_all_s_eta: float | None = None
    mean_accepted_s_eta: float | None = None
    mean_rejected_s_eta: float | None = None
    all_candidate_final_correctness: float | None = None
    accepted_final_correctness: float | None = None
    rejected_final_correctness: float | None = None
