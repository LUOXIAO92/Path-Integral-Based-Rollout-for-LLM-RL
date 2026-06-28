from __future__ import annotations

from copy import deepcopy


PROBLEM = "What is 40 + 2?"
ANSWER = "Step one. Final answer: 42"


def valid_reward_payload() -> dict:
    return deepcopy(
        {
            "problem_analysis": {
                "task_target": "Compute 40 + 2.",
                "required_output": "A final numeric answer.",
                "hard_constraints": [],
                "possible_off_task_patterns": ["Answering a different arithmetic problem."],
            },
            "problem_spans": [
                {
                    "problem_span_id": "p1",
                    "raw_text_span": {
                        "start_text": PROBLEM,
                        "end_text": PROBLEM,
                    },
                    "span_role": "task_target",
                    "normalized_requirement": "Compute 40 + 2.",
                }
            ],
            "reference_analysis": {
                "reference_type": "final_only",
                "final_answer": "42",
                "given_solution_key_points": [],
                "inferred_minimal_requirements": [
                    {
                        "reference_point_id": "r_min_1",
                        "content": "The final answer should be 42.",
                        "source": "minimal_requirement",
                    }
                ],
            },
            "student_spans": [
                {
                    "student_span_id": "s1",
                    "raw_text_span": {
                        "start_text": "Step one. ",
                        "end_text": "Step one. ",
                    },
                    "span_type": "key_reasoning",
                    "problem_span_refs": ["p1"],
                },
                {
                    "student_span_id": "s_final",
                    "raw_text_span": {
                        "start_text": "Final answer: 42",
                        "end_text": "Final answer: 42",
                    },
                    "span_type": "final_answer",
                    "problem_span_refs": ["p1"],
                },
            ],
            "student_answer_analysis": {
                "answer_summary": "The student gives a short computation and final answer.",
                "main_claims": [
                    {
                        "claim_id": "c1",
                        "student_span_refs": ["s1", "s_final"],
                        "claim_text": "The answer is 42.",
                        "claim_role": "attempted_solution",
                    }
                ],
            },
            "student_alignment": {
                "responds_to_problem": True,
                "off_task": False,
                "alignment_items": [
                    {
                        "problem_span_refs": ["p1"],
                        "student_span_refs": ["s1", "s_final"],
                        "alignment_status": "matched",
                        "reason": "The student answers the arithmetic question.",
                    }
                ],
                "off_task_evidence": [],
                "decision_reason": "The answer responds to the task.",
            },
            "span_evaluations": [
                {
                    "student_span_id": "s1",
                    "problem_span_refs": ["p1"],
                    "reference_point_refs": ["r_min_1"],
                    "is_relevant": True,
                    "is_key_reasoning": True,
                    "step_score": 1.0,
                    "error_type": "none",
                    "reason": "The reasoning is acceptable for the simple task.",
                },
                {
                    "student_span_id": "s_final",
                    "problem_span_refs": ["p1"],
                    "reference_point_refs": ["r_min_1"],
                    "is_relevant": True,
                    "is_key_reasoning": False,
                    "step_score": 0.0,
                    "error_type": "none",
                    "reason": "Final answer span is scored by final_answer_check.",
                },
            ],
            "final_answer_check": {
                "student_final_answer_span_id": "s_final",
                "is_correct": True,
                "reason": "The final answer is 42.",
            },
        }
    )
