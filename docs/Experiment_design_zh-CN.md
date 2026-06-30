# 路径级 MCMC 推理采样实验设计

## 0. 目标

实验分两步:
- 第一阶段: 只用问题作为前缀，多次生成完整路径，验证路径级 MCMC 是否能比同预算 best-of-n 得到更低 $S_\eta[\tau]$、更高 $G[\tau]$ 和更高最终正确率。
- 第二阶段: 在第一阶段基础上，构造可续写的步骤级 span，从已有路径的步骤前缀继续 rollout，验证 accepted 路径是否比 best-of-n 路径更稳定。

---

## 1. 基础公式以及符号定义

带源项路径分布:
```math
q_\eta(\tau) \propto \exp(-S_0[\tau]+\eta F[\tau])
```

有效作用量: 
```math
S_\eta[\tau] = S_0[\tau]-\eta F[\tau]
```

源项
```math
F[\tau;\lambda_G,\lambda_N,\lambda_{KL}] = \lambda_GG[\tau]-\lambda_NN[\tau]-\lambda_{KL}K[\tau]
```

符号定义:
- $G[\tau]$: 路径级回报
- $N[\tau]$: 思维链长度软惩罚
- $S_0[\tau]$: 策略模型诱导的基础路径测度
- $K[\tau]$: 采样 KL 项
- $\lambda_G,\lambda_N,\lambda_{KL}$: 各项系数。某一项不使用时，直接把对应系数设为 0
- $q_{\mathrm{mcmc\_prop}}(\tau\rightarrow\tau')$: 路径级 MCMC 提案分布，表示从完整路径 $\tau$ 提出完整候选路径 $\tau'$ 的概率

MCMC 接受率:
```math
A(\tau\rightarrow\tau')=\min\left(1,\exp[-(S_\eta[\tau']-S_\eta[\tau])]\frac{q_{\mathrm{mcmc\_prop}}(\tau'\rightarrow\tau)}{q_{\mathrm{mcmc\_prop}}(\tau\rightarrow\tau')}\right)
```


### 1.1 公式定义的具体数学建模

单步长链推理取 $T=0$ ，问题为初始历史 $h_0=[o_0]$ ，完整回答为阶段级动作 $a_0=[a_{L_0},\ldots,a_{L_1-1}]$ ，路径为 $\tau=[a_0,o_1,r_0]$ ，其中 $a_0$ 是完整回答路径， $o_1$ 是回答完成后的观测， $r_0$ 是裁判给出的软回报。

#### 1.1.1 路径长度

记路径 $\tau$ 的生成长度为 $L_\tau=L_1-L_0$ 。如果可以分离思维链和最终答案，则 $L_\tau$ 只统计思维链长度；如果暂时不能稳定分离，则 $L_\tau$ 统计完整回答长度。

#### 1.1.2 路径级回报 $G[\tau]$

路径级回报由第 7 节的奖励评价与计算规则得到。若只裁判一次，则 $G[\tau]=r_0$ 。若同一条路径裁判 $M_G$ 次，则有 
```math
G[\tau]=\frac{1}{M_G}\sum_{j=1}^{M_G}r_0^{(j)},\quad r_0^{(j)}\in[0,1]
```

最终正确性单独记录为 `final_correctness`。

#### 1.1.3 思维链长度软惩罚 $N[\tau]$

长度惩罚使用 tanh 软惩罚。令 $L_{max}$ 为软长度上限， $s>0$ 为变化尺度
```math
N[\tau] = \tanh\left(\frac{\max(0,L_\tau-L_{max})}{s}\right)
```

因此 $0\leq N[\tau] \lt 1$ 。当 $L_\tau\le L_{max}$ 时， $N[\tau]=0$ 。

#### 1.1.4 基础路径测度 $S_0[\tau]$

令 $\pi_{\theta_t}$ 表示当前策略模型。对路径 $\tau=[a_0,o_1,r_0]$ ，基础路径测度取完整动作 $a_0$ 的负对数概率
```math
S_0[\tau]=-\sum_{i=L_0}^{L_1-1}\log \pi_{\theta_t}(a_i | h_{i,0})
```

其中 $h_{i,0}=[h_0,a_{L_0},\ldots,a_{i-1}]$ 。 $\pi_{\theta_t}$ 使用 raw logprob，即原始 logits 在完整词表上的 softmax 分布
```math 
\log \pi_{\theta_t}(a_i | h_{i,0}) = z_{i,a_i}-\log\sum_{v\in V}\exp(z_{i,v})
```

#### 1.1.5 采样 KL 项 $K[\tau]$

采样 KL 使用同一条已经生成的路径计算。路径上的 token $a_i$ 由 rollout 采样得到，但 KL 中的概率使用策略模型和参考模型的原始分布
```math
K[\tau]=\sum_{i=L_0}^{L_1-1}\log\frac{\pi_{\theta_t}(a_i | h_{i,0})}{\pi_{\mathrm{ref}}(a_i | h_{i,0})}
```

其中 $\pi_{\theta_t}$ 和 $\pi_{\mathrm{ref}}$ 都使用 raw logprob。rollout proposal 只决定采到哪些 token，不进入 KL 比值。参考模型不生成新回答，只读取策略模型已经生成的同一条回答，并逐个位置计算参考模型对原 token 的 logprob。

当前实验使用原始模型测试，因此 $t=0$ 且 $\pi_{\theta_0}=\pi_{\mathrm{ref}}$ ，所以 $K[\tau]=0$ 。后续只有在策略更新后， $\pi_{\theta_t}$ 偏离 $\pi_{\mathrm{ref}}$ 时， $K[\tau]$ 才产生实际作用。

#### 1.1.6 源项 $F[\tau]$ 与有效作用量 $S_\eta[\tau]$

代入各项定义
```math
\begin{aligned}
F[\tau] &= \lambda_G G[\tau]-\lambda_N N[\tau]-\lambda_{KL}K[\tau]\\
S_\eta[\tau] &= S_0[\tau]-\eta\lambda_GG[\tau]+\eta\lambda_NN[\tau]+\eta\lambda_{KL}K[\tau]
\end{aligned}
```

当前实验如果不使用 KL 惩罚，设 $\lambda_{KL}=0$ 即可。

#### 1.1.7 路径级 MCMC 提案分布

第一阶段只用问题前缀 $h_0=[o_0]$ 生成完整候选路径，因此 $q_{\mathrm{mcmc\_prop}}(\tau\rightarrow\tau')=q_{\mathrm{mcmc\_prop}}(\tau' | h_0),~q_{\mathrm{mcmc\_prop}}(\tau'\rightarrow\tau)=q_{\mathrm{mcmc\_prop}}(\tau | h_0)$ 。

记本次 rollout 实际使用的解码分布为 $p_{\mathrm{dec}}(\cdot | h_i)$ 。若候选路径为 $\tau'=[x,y'_1,\ldots,y'_{n'}]$ ，当前路径为 $\tau=[x,y_1,\ldots,y_n]$ ，则:
```math
\begin{aligned}
\log q_{\mathrm{mcmc\_prop}}(\tau\rightarrow\tau') &= \sum_{i=1}^{n'}\log p_{\mathrm{dec}}(y'_i | x,y'_{<i}) \\
\log q_{\mathrm{mcmc\_prop}}(\tau'\rightarrow\tau) &= \sum_{i=1}^{n}\log p_{\mathrm{dec}}(y_i | x,y_{<i})
\end{aligned}
```

$p_{\mathrm{dec}}$ 由本次 rollout 的实际解码规则决定。若只使用原始分布，则 $p_{\mathrm{dec}}=\pi_{\theta_t}$ 。若使用 temperature、top-k、top-p、logit bias 或其他解码偏置，则 $p_{\mathrm{dec}}$ 为这些规则作用后并重新归一化的真实解码概率。

第二阶段若指定步骤级前缀 $c'_a=[x,y_{1:m}]$ 续写，则前缀选择概率为1。当前路径与候选路径写成 $\tau=[x,y_{1:m},y_{m+1:n}]$ ， $\tau'=[x,y_{1:m},y'_{m+1:n'}]$ 。此时:

```math
\begin{aligned}
\log q_{\mathrm{mcmc\_prop}}(\tau\rightarrow\tau') &= \sum_{j=m+1}^{n'}\log p_{\mathrm{dec}}(y'_j | x,y_{1:m},y'_{m+1:j-1}) \\
\log q_{\mathrm{mcmc\_prop}}(\tau'\rightarrow\tau) &= \sum_{j=m+1}^{n}\log p_{\mathrm{dec}}(y_j | x,y_{1:m},y_{m+1:j-1})
\end{aligned}
```

如果后续设计为随机选择步骤级前缀，则需要把前缀选择概率加入 $q_{\mathrm{mcmc\_prop}}$ 。第一轮第二阶段使用指定步骤级前缀，前缀选择概率固定为1。

#### 1.1.8 接受 / 拒绝

令 $\Delta S_\eta=S_\eta[\tau']-S_\eta[\tau]$ ，接受率为
```math
A(\tau\rightarrow\tau')=\min\left(1,\exp[-\Delta S_\eta]\frac{q_{\mathrm{mcmc\_prop}}(\tau'\rightarrow\tau)}{q_{\mathrm{mcmc\_prop}}(\tau\rightarrow\tau')}\right)
```

采样 $u\sim \mathrm{Uniform}(0,1)$ 。若 $u\le A(\tau\rightarrow\tau')$ ，则接受候选路径 $\tau'$；否则拒绝，链状态保持为原路径 $\tau$ 。

#### 1.1.9 best-of-n 的两个选择目标

对同一问题前缀 $h_0=[o_0]$ 采样得到 $\tau_1,\ldots,\tau_n$ 。按最低作用量选择 $\tau^{(S)} = \arg\min_{\tau_b,\ b=1,\ldots,n} S_\eta[\tau_b]$ 。按最高回报选择 $\tau^{(G)} = \arg\max_{\tau_b,\ b=1,\ldots,n} G[\tau_b]$ 。 $\tau^{(S)}$ 用于比较作用量目标， $\tau^{(G)}$ 用于比较回报目标，二者分开报告。

---

## 2. 第一阶段：问题前缀下的完整路径采样

### 2.1 路径定义

单步长链推理取 $T=0$ ，问题作为初始历史 $h_0=[o_0]$ ，完整回答作为一个阶段级动作 $a_0=[a_{L_0},\ldots,a_{L_1-1}]$ ，路径为 $\tau=[a_0,o_1,r_0]$ 。第一阶段只用问题作为候选动作前缀 $c'_a=h_0$ ，不扰动观测，即 $o_{\mathrm{allow}}'=o_{\mathrm{allow}}$ 。候选路径
```math
\begin{aligned}
\tau' &= \mathrm{Rollout}(c'_a,o_{\mathrm{allow}},r'_{\mathrm{soft}})\\
&= \mathrm{Rollout}(h_0,o_{\mathrm{allow}},r'_{\mathrm{soft}})
\end{aligned}
```

---

### 2.2 采样与记录

每次生成候选路径 $\tau'$ ，都按第 7 节计算 $G[\tau']$ ，并计算 $N[\tau'], K[\tau'], F[\tau'], S_0[\tau'], S_\eta[\tau']$ ，再用 $A(\tau\rightarrow\tau')$ 决定接受或拒绝。实验中必须保存:
- 全部候选路径
- accepted 候选路径
- rejected 候选路径
- MCMC 链状态路径

每条路径记录:
```text
- problem_id
- path_id
- path_text
- is_accepted
- G[τ]
- N[τ]
- K[τ]
- F[τ]
- S0[τ]
- Sη[τ]
- log_q_mcmc_prop_forward
- log_q_mcmc_prop_reverse
- A(τ→τ')
- final_correctness
```

---

### 2.3 best-of-n 对比

对同一问题 $o_0$ ，从同一问题前缀 $h_0=[o_0]$ 采样 $\tau_1,\ldots,\tau_n$ ，并且比较:
- MCMC accepted 样本
- MCMC 链 endpoint
- best-of-n 样本

best-of-n 需要分别报告两种选择方式:
1. 按最低 $S_\eta[\tau]$ 选择
2. 按最高 $G[\tau]$ 或最终正确性选择

两者分开报告。

---

### 2.4 第一阶段指标

作用量筛选效果为 $\mathrm{mean}_{all} S_\eta [\tau] - \mathrm{mean}_{accepted} S_\eta [\tau]$ 。同时统计 $\mathrm{mean}_{accepted} S_\eta [\tau]$、$\mathrm{mean}_{all} S_\eta [\tau]$、$\mathrm{mean}_{rejected} S_\eta [\tau]$ 。

最终正确率:
```text
- accepted_final_correctness
- all_candidate_final_correctness
- rejected_final_correctness
- best_of_n_final_correctness
- chain_endpoint_final_correctness
```

分解项: $G[\tau], N[\tau], S_0[\tau], S_\eta[\tau]$ 。

第一阶段关注:
1. 接受样本是否低于全样本的平均作用量 $S_\eta[\tau]$
2. 接受样本最终正确率是否高于全样本
3. MCMC 是否在同预算下接近或优于 best-of-n
4. 作用量下降是否伴随 $G[\tau]$ 提升，以及是否主要来自 $N[\tau]$ 下降
5. 不同 $T_{\mathrm{dec}}$ 下，高温 proposal 是否更容易产生错误路径
6. 真实 $q_{\mathrm{mcmc\_prop}}$ 修正前后的接受率和路径质量差异

---

## 3. 第二阶段: 步骤级 span 与前缀续写稳定性

### 3.1 目标

第二阶段验证: 不训练模型，只通过续写某些步骤级 prefix，能否得到比 best-of-n 更稳定的回答。“稳定”指多次续写时的正确率和回报能稳定下来，并且保持高正确率。

---

### 3.2 步骤级 span

第二阶段需要构造可续写的步骤级 span。步骤级 span 用于从已有路径 $\tau$ 中构造动作前缀 $c'_a$ 。步骤级 span 的抽取和评价使用第 7 节的奖励评价 schema。

---

### 3.3 前缀续写

从已有路径 $\tau=(a_0,o_1,r_0)$ 中选择可续写的步骤级 span，构造 $c'_a$ ，然后继续续写 $\tau' = \mathrm{Rollout}(c'_a,o_{\mathrm{allow}},r'_{\mathrm{soft}})$ 。对每个 $c'_a$ 多次采样 $\tau'_1,\ldots,\tau'_m$ ，每条路径重新计算 $G[\tau'], N[\tau'], F[\tau'], S_0[\tau'], S_\eta[\tau']$ ，并判断最终正确性。

---

### 3.4 第二阶段对比对象

第二阶段只比较:
1. best-of-n 选中的路径
2. MCMC 接受路径

拒绝路径不进入第二阶段前缀续写实验，原因是未来用于蒸馏或更新的路径是接受路径，拒绝路径没有必要作为续写对象。

---

### 3.5 第二阶段指标

对每类路径来源，统计前缀续写后的 $\mathrm{mean} S_\eta[\tau'],~ \mathrm{Var} S_\eta[\tau'],~ \mathrm{mean}\ G[\tau'],~\mathrm{mean} N[\tau']$ 。

以及:
```text
- prefix_continuation_final_correctness
- prefix_continuation_error_rate
- accepted_ratio
```

其中 `prefix_continuation_error_rate` 为续写错误率。回答错误、推理错误、最终答案错误、回答偏离问题，都计为错误。

---

### 3.6 第二阶段成功标准

第二阶段关注:
1. MCMC 接受路径的前缀续写最终正确率是否高于 best-of-n 路径
2. MCMC 接受路径的前缀续写错误率是否低于 best-of-n 路径
3. MCMC 接受路径的前缀续写平均 $S_\eta[\tau']$ 是否更低
4. MCMC 接受路径的前缀续写 $S_\eta[\tau']$ 方差是否更小
5. 前缀续写是否比从问题前缀重新 best-of-n 更高效

若 best-of-n 路径单点 $S_\eta[\tau]$ 很低，但从步骤级 $c'_a$ 续写后错误率高、$S_\eta[\tau']$ 方差大，则该路径更像孤立尖峰。若 MCMC 接受路径从步骤级 $c'_a$ 续写后仍能稳定产生正确路径，则该路径更可能处在低有效作用量路径盆地。

---

## 4. 两阶段关系

第一阶段: $h_0 \rightarrow \mathrm{Rollout} \rightarrow \tau' \rightarrow A(\tau\rightarrow\tau')$

- 目标是验证问题前缀下的路径级 MCMC 是否有效。

第二阶段: $\tau \rightarrow c'_a \rightarrow \mathrm{Rollout} \rightarrow \tau' \rightarrow A(\tau\rightarrow\tau')$

- 目标是验证接受路径是否比 best-of-n 路径具有更稳定的步骤级前缀续写能力。

---

## 5. 输出报告结构

最终报告至少包含:
1. 第一阶段 accepted / all / rejected 的 $S_\eta[\tau], G[\tau], N[\tau]$、最终正确率
2. 第一阶段 MCMC 与 best-of-n 的同预算对比
3. 不同 $T_{\mathrm{dec}}$ 下的最终正确率、接受率和平均 $S_\eta[\tau]$
4. 第一阶段 $q_{\mathrm{mcmc\_prop}}$ 修正前后的接受率和路径质量对比
5. 第二阶段 best-of-n 路径与 MCMC 接受路径的前缀续写正确率
6. 第二阶段 best-of-n 路径与 MCMC 接受路径的前缀续写错误率
7. 第二阶段前缀续写后的 $S_\eta[\tau']$ 均值与方差
8. 对孤立尖峰与稳定路径盆地的区分

---

## 6. 实验具体实施

### 6.1 学生模型

第一阶段只做 rollout 和路径筛选，不做训练。学生模型只负责生成路径 $\tau$ 。第一轮先使用 Qwen3 小尺寸模型:
```text
- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
```

暂时不使用 Qwen3.6 / Gemma4 这类更新、更强的模型，避免 baseline 过强导致 MCMC 和 best-of-n 的差异不明显。同时加入数理推理和代码特化的小模型:
```text
- Qwen2.5-Math-1.5B-Instruct
- Qwen2.5-Coder-1.5B-Instruct
```

这两个模型用于比较普通小模型和领域特化小模型在路径级 MCMC 下的差异。后续可以再追加:
```text
- Qwen2.5-Coder-3B-Instruct
- DeepSeek-R1-Distill-Qwen-1.5B
```

---

### 6.2 教师 / 裁判模型

教师 / 裁判模型固定使用:
```text
- DeepSeek-V4-Pro
```

第一阶段使用第 7 节的奖励评价 schema。裁判模型只输出受限分析，程序根据 `score_config` 计算路径级回报 $G[\tau]$ 和最终正确性 `final_correctness`。

所有学生模型、MCMC 样本、best-of-n 样本使用同一个裁判模型和同一组裁判参数。

第一阶段默认每条路径只裁判一次。若需要估计裁判波动，只对关键样本重复裁判:
```text
- MCMC accepted-best
- MCMC chain endpoint
- best-of-n by lowest Sη[τ]
- best-of-n by highest G[τ]
```

---

### 6.3 数据集选择

数学物理任务使用:
```text
- MATH 筛选子集
- OlympiadBench 数学 / 物理文本筛选子集
```

筛选规则:
```text
- 优先选择符号推导、概念推理、物理一致性、边界条件、极限情况相关题目
- 排除主要依赖数值计算的题目
- 排除必须依赖图片、图表、几何图形解析的题目
- 排除答案过短、无法形成长链推理路径的题目
```

代码任务使用:
```text
- HumanEval
- MBPP
- livecodebench/code_generation easy / medium 子集
```

HumanEval 和 MBPP 用于基础检查，`livecodebench/code_generation` 用于正式代码对比。若题目自带测试用例，测试结果单独记录，不直接混进 $G[\tau]$ 。第一轮每类任务先选 50～200 题。

---

### 6.4 采样 backend

第一阶段主采样 backend 使用 vLLM。vLLM 负责批量 rollout、高温采样、返回生成 token、raw logprob 和 rollout 实际分布下的 logprob。当前流程是:
```text
vLLM 生成路径
→ DeepSeek-V4-Pro 裁判
→ 计算 G[τ], N[τ], K[τ], F[τ], S0[τ], Sη[τ]
→ 计算 log_q_mcmc_prop_forward 和 log_q_mcmc_prop_reverse
→ 计算 A(τ→τ')
→ 保存 accepted / rejected / chain endpoint
```

Transformers 作为辅助 backend，主要用于:
```text
- 小规模 debug
- 校验 raw logprob
- 读取同一条回答并计算参考模型对原 token 的 logprob
- 第二阶段的步骤级 prefix 续写
```

第一阶段不使用 TRL、PPO、GRPO、GSPO。后续如果进入训练阶段，再单独接更新框架。

---

### 6.5 采样参数

同一个问题下，MCMC 和 best-of-n 使用同一组采样参数:
```text
- student_model
- prompt_template
- T_dec
- top_p
- max_tokens
- rollout_budget
- judge_model
- judge_params
```

第一轮设置 2～3 个温度:
```text
- low T_dec
- middle T_dec
- high T_dec
```

MCMC 参数包括:
```text
- η
- λ_G
- λ_N
- λ_KL
- mcmc_prop_distribution
```

`mcmc_prop_distribution` 记录本次 rollout 实际使用的 proposal 分布。若 rollout 使用原始分布，则按原始 token logprob 计算 $q_{\mathrm{mcmc\_prop}}$。若 rollout 使用完整词表温度分布，则按温度采样后的 token logprob 计算 $q_{\mathrm{mcmc\_prop}}$。若 rollout 使用 top-p，则按 top-p 截断并重新归一化后的 token logprob 计算 $q_{\mathrm{mcmc\_prop}}$。

第一轮真实 $q_{\mathrm{mcmc\_prop}}$ 只使用 raw 或完整词表温度分布。若启用 top-k / top-p，必须记录截断后重新归一化的真实 token 概率，否则不能作为真 MCMC 的提案概率。

当前实验使用原始模型测试，因此 $t=0$ 且 $\pi_{\theta_0}=\pi_{\mathrm{ref}}$ 。如果不使用 KL 项，直接设 $\lambda_{KL}=0$ 。

---

### 6.6 第一轮先跑哪些模型和数据

数学物理任务先跑:
```text
学生模型:
- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen2.5-Math-1.5B-Instruct

数据:
- MATH 筛选子集
- OlympiadBench 数学 / 物理文本筛选子集

裁判:
- DeepSeek-V4-Pro
```

代码任务先跑:
```text
学生模型:
- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen2.5-Coder-1.5B-Instruct

数据:
- HumanEval
- MBPP
- livecodebench/code_generation easy / medium 子集

裁判:
- DeepSeek-V4-Pro
```

每个任务都比较:
```text
- MCMC accepted 样本
- MCMC chain endpoint
- best-of-n by lowest Sη[τ]
- best-of-n by highest G[τ]
- all candidates
- rejected candidates
```

---

### 6.7 日志字段

实验日志分成 run 配置和路径记录。不会变化的内容放在 run 配置里，不在每条路径里重复保存。

run 配置保存:
```text
- run_id
- dataset
- student_model
- judge_model
- backend
- prompt_template_id
- T_dec
- top_p
- max_tokens
- rollout_budget
- η
- λ_G
- λ_N
- λ_KL
- mcmc_prop_distribution
- score_config
```

路径记录保存为 JSONL:
```text
- run_id
- problem_id
- method
- path_id
- chain_step
- path_text
- is_accepted
- G[τ]
- N[τ]
- K[τ]
- S0[τ]
- F[τ]
- Sη[τ]
- log_q_mcmc_prop_forward
- log_q_mcmc_prop_reverse
- A(τ→τ')
- final_correctness
```

第二阶段再追加:
```text
- source_path_type
- prefix_span_id
- prefix_text
- continuation_id
- prefix_continuation_final_correctness
- prefix_continuation_error
```

---

### 6.8 第一阶段运行顺序

第一轮按这个顺序执行:
```text
1. 准备数据子集
2. 固定 prompt 模板
3. 固定学生模型
4. 固定 vLLM 采样参数
5. 对每个问题生成 MCMC 候选路径
6. 调用 DeepSeek-V4-Pro 生成奖励评价 JSON
7. 程序侧 validator 校验评价 JSON
8. 根据 score_config 计算 G[τ] 和 final_correctness
9. 计算 N[τ], K[τ], F[τ], S0[τ], Sη[τ]
10. 计算 log_q_mcmc_prop_forward 和 log_q_mcmc_prop_reverse
11. 计算 A(τ→τ')
12. 保存 accepted / rejected / chain endpoint
13. 用同样 rollout 预算生成 best-of-n
14. 生成第一阶段统计报告
```

第一阶段结束后再进入第二阶段。第二阶段只使用 best-of-n 选中路径和 MCMC accepted 路径，不使用 rejected 路径做前缀续写实验。

---

## 7. 奖励评价与计算设计

### 7.1 目标

奖励评价不能直接让裁判模型给总分。裁判模型只负责受限分析，具体分数由固定规则计算。评价顺序如下:
1. 分析问题要求
2. 抽取问题局部 span
3. 分析参考答案
4. 抽取学生回答 span
5. 分析学生回答实际回答了什么
6. 判断学生回答是否回应问题
7. 对每个学生 span 做局部评价
8. 判断最终答案是否正确
9. 用 `score_config` 计算 $G[\tau]$

---

### 7.2 统一评价 schema

```json
{
  "problem_analysis": {
    "task_target": "",
    "required_output": "",
    "hard_constraints": [],
    "possible_off_task_patterns": []
  },

  "problem_spans": [
    {
      "problem_span_id": "p1",
      "raw_text_span": {
        "start_text": "",
        "end_text": ""
      },
      "span_role": "task_target",
      "normalized_requirement": ""
    }
  ],

  "reference_analysis": {
    "reference_type": "final_only",
    "final_answer": "",
    "given_solution_key_points": [],
    "inferred_minimal_requirements": []
  },

  "student_spans": [
    {
      "student_span_id": "s1",
      "raw_text_span": {
        "start_text": "",
        "end_text": ""
      },
      "span_type": "key_reasoning",
      "problem_span_refs": ["p1"]
    },
    {
      "student_span_id": "s_final",
      "raw_text_span": {
        "start_text": "",
        "end_text": ""
      },
      "span_type": "final_answer",
      "problem_span_refs": ["p1"]
    }
  ],

  "student_answer_analysis": {
    "answer_summary": "",
    "main_claims": [
      {
        "claim_id": "c1",
        "student_span_refs": ["s1"],
        "claim_text": "",
        "claim_role": "attempted_solution"
      }
    ]
  },

  "student_alignment": {
    "responds_to_problem": true,
    "off_task": false,
    "alignment_items": [
      {
        "problem_span_refs": ["p1"],
        "student_span_refs": ["s1"],
        "alignment_status": "matched",
        "reason": ""
      }
    ],
    "off_task_evidence": [
      {
        "student_span_refs": [],
        "reason": ""
      }
    ],
    "decision_reason": ""
  },

  "span_evaluations": [
    {
      "student_span_id": "s1",
      "problem_span_refs": ["p1"],
      "reference_point_refs": [],
      "is_relevant": true,
      "is_key_reasoning": true,
      "step_score": 1.0,
      "error_type": "none",
      "reason": ""
    },
    {
      "student_span_id": "s_final",
      "problem_span_refs": ["p1"],
      "reference_point_refs": [],
      "is_relevant": true,
      "is_key_reasoning": false,
      "step_score": 0.0,
      "error_type": "none",
      "reason": ""
    }
  ],

  "final_answer_check": {
    "student_final_answer_span_id": "s_final",
    "is_correct": false,
    "reason": ""
  }
}
```

---

### 7.3 字段规则

`problem_analysis` 用于分析题目要求。`task_target` 写题目真正要解决什么，`required_output` 写最终答案应该输出什么，`hard_constraints` 写题目里的硬约束，`possible_off_task_patterns` 只写少量可预见的偏题情况，无需穷举。

`problem_spans` 用于抽取题目中的局部要求。它不需要覆盖完整题目，只需要覆盖后续评价会引用的题目要求。每个 `problem_span` 用 `raw_text_span.start_text` 和 `raw_text_span.end_text` 定位，首尾各使用约 10 个词。`normalized_requirement` 写这个题目片段对应的局部要求。

`reference_analysis` 用于分析参考答案。`reference_type` 只能取 `final_only`、`detailed_solution`、`test_based`、`no_reference`。如果有详细解法，`given_solution_key_points` 提取参考解法里的关键点；如果只有最终答案，`given_solution_key_points` 可以为空。`inferred_minimal_requirements` 只写保守的最低要求，不生成完整标准解法。

`student_spans` 必须 100% 覆盖完整学生回答。每个 span 用 `raw_text_span.start_text` 和 `raw_text_span.end_text` 定位，首尾各使用约 10 个词。`span_type` 只能取 `key_reasoning`、`non_key_reasoning`、`irrelevant`、`final_answer`。`problem_span_refs` 引用该学生片段正在回应的题目局部要求；无关内容可以为空。

`student_answer_analysis` 用于分析学生回答实际回答了什么。`answer_summary` 概括学生回答的主线。`main_claims` 必须引用 `student_span_refs`，`claim_text` 只能概括学生原文内容，不能补充新推理。

`student_alignment` 用于判断学生回答是否回应题目目标。`alignment_items` 必须同时引用 `problem_span_refs` 和 `student_span_refs`，说明学生回答的哪些片段对应题目的哪些局部要求。`alignment_status` 只能取 `matched`、`partially_matched`、`missing`、`contradicted`、`irrelevant`。`off_task_evidence` 必须引用学生回答 span。若 `off_task=true`，最终 $G[\tau]=0$ 。

`span_evaluations` 用于评价每个学生 span。每个 `student_span_id` 至少有一条评价。`is_relevant=false` 时不进入推理过程分。`is_key_reasoning` 由裁判判断。`step_score` 取 0 到 1。`irrelevant` 和 `final_answer` 的 `step_score` 固定为 0。`reference_point_refs` 在有参考解法关键点时使用，没有时可以为空。评价必须基于该 span 引用的 `problem_span_refs` 和 `reference_point_refs`。

`final_answer_check` 用于判断最终答案。`student_final_answer_span_id` 必须引用 `span_type=final_answer` 的学生 span。数学物理由裁判判断最终答案是否正确；代码任务优先使用测试结果判断最终答案是否正确。最终答案错误时 $A_{\mathrm{final}}[\tau]=0$ ，但仍保留推理过程分。

validator 不属于 reward LLM 输出 schema。程序侧 validator 只检查硬规则: JSON 是否可解析、必填字段是否存在、字段类型是否合法、枚举值是否合法、引用 id 是否存在、`step_score` 是否在 0 到 1、raw text span 是否能匹配原文、`student_spans` 是否完整覆盖学生回答、span 是否顺序正确且无重叠遗漏。validator 失败时不计算 $G[\tau]$ ，把错误信息作为 retry 输入重新请求 reward LLM 输出完整 JSON。

---

### 7.4 计分超参数

计分规则不写死，作为 run 配置保存。

```json
{
  "score_config": {
    "lambda_R": 0.4,
    "lambda_A": 0.6,
    "w_key_reasoning": 1.0,
    "w_non_key_reasoning": 0.25,
    "w_irrelevant": 0.0,
    "final_correct_score": 1.0,
    "final_wrong_score": 0.0,
    "off_task_score": 0.0,
    "no_scored_span_process_score": 0.0
  }
}
```

---

### 7.5 推理过程分

令第 $j$ 个学生 span 的步骤分为 $s_j$ ，来自 `span_evaluations.step_score`。span 权重由 `score_config` 决定:
```math
w_j=\begin{cases}
w_{\mathrm{key}}, & \text{is\_key\_reasoning}=1 \\ 
w_{\mathrm{nonkey}}, & \text{is\_key\_reasoning}=0 \text{ and is\_relevant}=1 \\ 
w_{\mathrm{irrelevant}}, & \text{is\_relevant}=0
\end{cases}
```

推理过程分为 $R_{\mathrm{process}}[\tau]=\frac{\sum_j w_js_j}{\sum_j w_j}$ 。若 $\sum_j w_j=0$ ，则 $R_{\mathrm{process}}[\tau]=\mathrm{no\_scored\_span\_process\_score}$ 。当前默认权重会降低无关步骤和非关键步骤的取巧收益。

---

### 7.6 最终答案分

最终答案分为:
```math
A_{\mathrm{final}}[\tau]=\begin{cases} 
\mathrm{final\_correct\_score}, & \text{final\_answer\_check.is\_correct}=true \\ 
\mathrm{final\_wrong\_score}, & \text{final\_answer\_check.is\_correct}=false 
\end{cases}
```

代码任务如果存在测试结果，则测试结果优先决定 `final_answer_check.is_correct`。

---

### 7.7 路径级回报 $G[\tau]$

路径级回报为:
```math
G[\tau]=\begin{cases} 
\mathrm{off\_task\_score}, & \text{student\_alignment.off\_task}=true \\ 
\lambda_R R_{\mathrm{process}}[\tau]+\lambda_A A_{\mathrm{final}}[\tau], & \text{student\_alignment.off\_task}=false 
\end{cases}
```

默认设置下，推理过程占 0.4，最终答案占 0.6。若需要调整比例，只改 `score_config.lambda_R` 和 `score_config.lambda_A`。

---

### 7.8 程序侧 validator 校验

每次裁判输出后先做程序侧校验，再计算 $G[\tau]$ 。

校验顺序:
1. 检查 JSON 是否能解析
2. 检查必填字段是否存在
3. 检查字段类型是否合法
4. 检查枚举值是否合法
5. 检查 `student_spans` 是否能用 raw text span 匹配回原回答
6. 检查 `student_spans` 是否 100% 覆盖完整学生回答
7. 检查 span 是否顺序正确、无重叠、无遗漏
8. 检查 `student_span_id`、`problem_span_id`、`reference_point_id` 引用是否存在
9. 检查 `span_evaluations` 是否覆盖所有学生 span
10. 检查 `step_score` 是否在 0 到 1
11. 检查 `final_answer_check.student_final_answer_span_id` 是否存在

validator 失败时不计算 $G[\tau]$ ，把错误信息作为 retry 输入重新请求 reward LLM 输出完整 JSON。
