# Path Integral Based Rollout for LLM RL

Status: this project is under active construction. APIs, scripts, output schemas, and experiment defaults may change. Treat the current repository as a research prototype, not a stable release.

This project studies path-level rollout sampling for LLM reasoning. A full model answer is treated as a path, then the project computes path-level reward, length penalty, base action, effective action, and Metropolis acceptance records.

The current stage does not train a model. It separates the workflow into student rollout, reward scoring, and path-level MCMC filtering so each cost boundary can be inspected on its own.

## Workflow

Prepare normalized problem rows:

```bash
conda run -n rl-rollout python scripts/prepare_data.py
```

This writes normalized problems to `data/problems.jsonl` and a preparation manifest to `data/problems_manifest.json`.

Generate local vLLM-compatible student rollouts with raw/proposal logprobs:

```bash
conda run -n rl-rollout python scripts/run_vllm_rollout.py
```

This writes `outputs/vllm_rollout/rollouts.jsonl`. Stop here when you only want to test the student model. The default backend is `mock` for local development. Use the vLLM backend on a Linux/GPU machine with vLLM installed when true proposal logprobs are required.

Score rollouts with the reward or teacher model:

```bash
conda run -n rl-rollout python scripts/run_reward_scoring.py
```

This reads `outputs/vllm_rollout/rollouts.jsonl` and writes scored candidates to `outputs/vllm_rollout/candidates.jsonl`. It also writes raw reward-model responses to `outputs/vllm_rollout/reward_raw.jsonl` for audit.

Run path-level MCMC filtering:

```bash
conda run -n rl-rollout python scripts/run_mcmc.py
```

This consumes scored candidates and writes chain, best-of-n, and summary outputs under `outputs/vllm_rollout/`. MCMC uses proposal logprobs recorded on each scored path. The default proposal-ratio mode is length-normalized for stable first-stage experiments; switch `PROPOSAL_RATIO_MODE` in `scripts/run_mcmc.py` to `strict` for exact path-level MH proposal ratios. Strict mode uses full proposal logprob sums and a strict-only scaled length penalty from `scoring_config.json`.

## Model Endpoints

Set the reward or teacher model endpoint before scoring:

```bash
export REWARD_BASE_URL="..."
export REWARD_API_KEY="..."
export REWARD_MODEL="DeepSeek-V4-Pro"
```

`scripts/run_vllm_rollout.py` produces student paths. `scripts/run_reward_scoring.py` calls the reward or teacher model.

## Path-Level Quantities

For each generated path `tau`, the project records:

- `G[tau]`: path-level reward from the validated reward schema and score config.
- `N[tau]`: soft length penalty.
- `K[tau]`: sampled KL term. It can be disabled with `lambda_KL = 0`.
- `S0[tau]`: base action from the student model log probabilities.
- `S_eta[tau]`: effective action used for path selection.

MCMC filtering compares the current path and the proposed path through `S_eta[tau]` plus a proposal correction computed from recorded proposal logprobs. In strict proposal mode, acceptance uses a strict-only action with `N_scaled = L_tau * alpha_L * tanh(...)` while keeping scoring-time `S_eta[tau]` unchanged. It records accepted paths, rejected paths, chain endpoints, proposal-ratio diagnostics, strict action diagnostics, and best-of-n selections.

## External Theory Notes

The project follows the path-integral and resampling picture described in these external notes:

- Chinese: https://github.com/LUOXIAO92/ai_learning/blob/main/From_history-based_RL_to_resampling_and_physical_picture_of_RL_ZH.md
- English: https://github.com/LUOXIAO92/ai_learning/blob/main/From_history-based_RL_to_resampling_and_physical_picture_of_RL_EN.md
- Japanese: https://github.com/LUOXIAO92/ai_learning/blob/main/From_history-based_RL_to_resampling_and_physical_picture_of_RL_JA.md
