from __future__ import annotations

from src.schemas import RewardEvaluation, ScoreConfig, ScoredReward


def score_reward_evaluation(evaluation: RewardEvaluation, score_config: ScoreConfig) -> ScoredReward:
    if evaluation.student_alignment.off_task:
        final_score = (
            score_config.final_correct_score
            if evaluation.final_answer_check.is_correct
            else score_config.final_wrong_score
        )
        return ScoredReward(
            process_score=score_config.no_scored_span_process_score,
            final_score=final_score,
            g=score_config.off_task_score,
            final_correctness=evaluation.final_answer_check.is_correct,
        )

    span_by_id = {span.student_span_id: span for span in evaluation.student_spans}
    weighted_sum = 0.0
    weight_sum = 0.0
    for item in evaluation.span_evaluations:
        span = span_by_id[item.student_span_id]
        if span.span_type == "final_answer" or not item.is_relevant:
            weight = score_config.w_irrelevant
        elif item.is_key_reasoning:
            weight = score_config.w_key_reasoning
        else:
            weight = score_config.w_non_key_reasoning
        weighted_sum += weight * item.step_score
        weight_sum += weight

    if weight_sum == 0:
        process_score = score_config.no_scored_span_process_score
    else:
        process_score = weighted_sum / weight_sum

    final_score = (
        score_config.final_correct_score
        if evaluation.final_answer_check.is_correct
        else score_config.final_wrong_score
    )
    g = score_config.lambda_R * process_score + score_config.lambda_A * final_score
    return ScoredReward(
        process_score=process_score,
        final_score=final_score,
        g=g,
        final_correctness=evaluation.final_answer_check.is_correct,
    )
