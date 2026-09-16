# 迁移案例分析（Task C）

`analysis.py` 的 §7 输出的是**候选**：evo 注入了 skill、且与 baseline 在同一道题上结果相反的题目（10 个正向、6 个负向）。候选不是结论 —— 一个迁移案例是"注入之后模型**做得有什么不同**"的论证，必须把轨迹读出来。本文是对其中三个的逐轨迹分析。

所有数字可从 `outputs/harness/{adapt-baseline,adapt-evo}/trajectories/problem_XXXX.json` 与 `outputs/harness/adapt-evo/store/{selection_log,harness_log}.jsonl` 复核。

> **注入文本需要重建。** 轨迹文件里**没有**存 system message —— skill 是在 `arms.py:system_prompt_for()` 里逐题写进 system 通道的，而 `full_config.policy_model_cfg.system_prompt` 存的是配置里的空串。要知道某道题当时看到的 skill 原文，得拿 `selection_log.jsonl` 的 `skill_ids`，再把 `harness_log.jsonl` 里该 batch 之前所有 accepted 的编辑重放一遍。本文每个案例引用的 skill 文本都是这样重建的，**不是**最终导出的 `harness.md` 里那一版（两者往往已经不同 —— 见 §10.3 的语义漂移）。

---

## 案例 1（正向）：p_84 —— 未漂移的具体 skill，但因果链只能说"可能"

**题目**（number_theory）：求所有正整数 $n$ 之和，使 $1^3+2^3+\cdots+n^3$ 除以 $n+5$ 余 17。GT = 239。

**结果**：baseline 答 37（错），evo 答 239（对），两臂都在 round 0 结束。

**当时注入的 3 条 skill（188 token）**：

| id | 层 | trigger | lesson 摘要 |
|---|---|---|---|
| `sk_0001` | topic / `factorization_conditions` | when solving problems involving polynomial factorization with integer coefficients | 识别可分解的必要条件；把约束翻译成关于根的方程；数出同时满足和与积条件的根对 |
| `sk_0006` | general | When multiple constraints or conditions are present in a problem. | 系统枚举所有可能配置；逐一对照全部约束校验；**确保 case 分析没有重叠或遗漏** |
| `sk_0008` | topic / `logical_reasoning` | When solving multi-step problems with potential errors | 不要假设变量间独立；校验所有约束同时成立 |

`sk_0001` 是全场**唯一一条从 ADD 之后再没被编辑过**的 skill（`harness_log.jsonl` 里它只有 b0 的一条 accepted 记录），因此它的 trigger 与 lesson 仍然互相匹配，而且和这道题的结构真的对得上（把余数条件翻译成整除条件、数出满足条件的解）。

**两臂实际做了什么（这是本案例的关键，不是分数）**：

| | baseline | evo |
|---|---|---|
| 搜索上界 | `for n in range(1, 100)` | `for n in range(1, 1000)` |
| 工具返回 | `"78\n"`（它的末表达式是 `sum(valid_n)`，所以这个 78 是**和**，不是解集） | `"[78, 161]\n"` |
| 下一步 | **无视工具返回**，自称"用计算方法找到 n = 6, 8, 9, 14"，答 37 | 读取返回值，再手算验证 $3081^2 \bmod 83 = 17$、$13113^2 \bmod 166 = 17$，答 239 |

决定对错的是**搜索上界 100 还是 1000**：真解是 78 和 161，上界 100 必然漏掉 161。baseline 还叠加了第二个独立故障 —— 它凭空编造了工具输出（既不是 78 也不是它自己代码算出的任何东西）。

**能说到什么程度**：`sk_0006` 的 "Ensure no overlaps or omissions in case analysis" 与"把上界放宽到 1000"之间有合理的联系，`sk_0001` 的 failure_mode（"providing answers without showing the logical steps"）与"evo 逐个手算验证、baseline 直接编结论"之间也有。**但这是 temperature=0.7 下的单次采样**，搜索上界的差异完全可能只是采样噪声。要把"可能"变成"是"，需要在不注入的条件下把 p_84 重跑 N 次、看 `range` 上界的分布 —— 本项目没有做这件事，所以这个案例的正确写法是"行为差异是具体且可复核的，因果归因是未证实的"。

---

## 案例 2（负向）：p_105 —— verifier 推翻了正确答案，policy 照单全收

**题目**（number_theory）：求所有三位回文数的算术平均。GT = 550。

**结果**：baseline round 0 答 550、round 1 仍答 550，最终对。evo **round 0 答 550（对）**，round 1 改成 **549.5（错）**，`pass_final = 0`。

**注入的 3 条 skill（178 token）**：`sk_0014`、`sk_0007`、`sk_0013` —— 三条的 lesson 分别是"系统追踪所有约束/逐步校验"、"系统枚举、避免重复或遗漏"、"逐个候选校验全部约束、检查边界情况"。**三条说的是同一件事**，且这道题根本没有多重约束可校验。

**事件链**（`problem_0105.json`，逐 step）：

1. round 0，policy 答 **550**，推理正确（按 $a,b$ 拆贡献，总和 49500，除以 90）。但这条最终消息只有 **756 字符** —— 远短于 evo 全局均值 2353。
2. round 0，verifier 判 **错**，并主动给出自己的答案：*"The correct mean is 549.5, not 550."*
3. round 1，policy **接受了 verifier 的数字**，并为它倒推出一套错误的算术（把总和算成 49455），答 **549.5**。
4. round 1，verifier **自己翻供**：*"The correct total sum is 49500, not 49455, which leads to an arithmetic mean of 550, not 549.5."*

翻供来得太晚 —— `pass_final` 取最后一轮，记录下来的是 549.5。**一道本来做对的题，被题内演化轮 + 一个错误的 verifier 判决毁掉了。**

注意这里**不是** skill 内容把模型带向了错误的数学方法。skill 没有提出任何具体做法；它们施加的是一种**"再验证一遍"的压力**，作用在一道不需要验证的题上，产物是一次过度修正。这是 context interference 的一种形态，但机制是**风格层面的**，不是知识层面的。

---

## 案例 3（系统性）：上面那条链不是孤例，而是三臂结果差异的主要来源

案例 2 暴露的模式可以直接在 144 题上统计。定义：**loss** = round 0 做对、最终做错；**gain** = round 0 做错、最终做对。

| arm | loss | gain | 净 | round-0 Pass@1 | 最终 |
|---|---|---|---|---|---|
| baseline | **8** | 7 | −1 | 22.2% | 21.5% |
| raw | **1** | 8 | **+7** | 16.7% | 21.5% |
| evo | **6** | 7 | +1 | 23.6% | 24.3% |

以及 verifier 的误判率（`summary.verifier_correctness_stats`，144 题累计，分母是 policy **确实答对**的轮次）：

| arm | policy 对 / verifier 对 | policy 对 / verifier 错 | **误拒率** |
|---|---|---|---|
| baseline | 42 | 21 | **33%** |
| raw | 41 | 14 | **25%** |
| evo | 47 | 22 | **32%** |

**policy 答对时，verifier 大约每三次否定一次**，而且常常附带一个具体的错误数字。

**三条结论**：

1. **题内演化轮在 baseline 上是净负的**（−1）。它毁掉的正确答案（8）多于它救回的（7）。这是上游 AlphaApollo 自己的行为，与本项目无关，但它是所有三臂共同的噪声源。
2. **Raw 臂在 adaptation 上追平 baseline，靠的不是多做对题，而是少毁题。** 它的 round-0 Pass@1 比 baseline 低 5.5 个百分点（16.7% vs 22.2%），最终却打平 —— 全部差距来自 loss 从 8 降到 1。任何把 raw 的成绩解读成"原始经验也有用"的写法都是错的：它的收益发生在**防止答案被改坏**这一环，不在解题能力上。
3. **被毁掉的题，round-0 那条最终消息明显更短**：

   | arm | 被毁题的 round-0 消息均长 | 全部题均长 |
   |---|---|---|
   | baseline | 1827 | 2575 |
   | evo | 1302 | 2353 |
   | raw | 766 | 2491 |

   三臂一致。verifier 是按**看得见的推理**打分的，一条简短、断言式的正确答案会被判成"没有给出推理过程"，verifier 随即给出自己的（经常是错的）数字，policy 在下一轮采纳它。

这解释了案例 2 的 756 字符，也解释了为什么 evo 的 loss（6）比 raw（1）高得多：注入 skill 之后 evo 把验证工作**前置到 step 0**，最终那条消息因此更像结论而非推导（evo 最终消息长度中位数 2280，baseline 2596）。**这是本项目最有代表性的失败模式：harness 影响的不是模型的数学能力，而是它的输出风格，而风格恰好是 verifier 的评分对象。**

---

## 对实验解读的影响

- 三臂之间 2–3 个百分点的差距，**至少有一半可以由 loss 数的差异解释**（8 / 1 / 6），而 loss 数由"最终消息有多长"这样一个与 skill 质量无关的变量驱动。在这个噪声水平下，Task C 的主结果不足以支持任何关于 skill 质量的因果结论。
- 一个**不改变任何 skill 机制**就能拿到的改进：让 `pass_final` 取"所有轮里被 verifier 判对过的答案"而不是"最后一轮的答案"，或者在 verifier 报告不含具体反例时不让 policy 改答案。按上表，这一条会让 baseline +8、evo +6、raw +1，且对三臂都公平。这条没有实现 —— 它动的是上游的题内逻辑，任务书明确说了不需要动。
- 若要重做本实验，最值得加的一项测量是**对同一道题在不注入条件下重复采样**，用来把案例 1 那种"行为差异"与采样噪声分开。
