# Task C 实验报告与验收对照

本文是 Task C 三组对照实验的结果报告，以及对六项验收重点的逐条自查。系统设计与使用说明在 [`README_HARNESS.md`](README_HARNESS.md)，迁移案例的逐轨迹分析在 [`docs/findings/transfer-cases.md`](docs/findings/transfer-cases.md)，本文与它们互不重复。

**每个数字都可以从仓库内的产物重算。** 汇总表来自 `outputs/harness/report/results.{md,json}`（由 `analysis.py` 生成），最终 harness 与完整编辑日志来自 `outputs/harness/export-evo/`，逐题结果来自各 run 的 `metrics.jsonl`。

---

## 摘要

| | |
|---|---|
| **跑完了什么** | 6 个 run = 3 臂 × {adaptation 144 题, held-out 30 题}，**丢题 0** |
| **主结果** | adaptation 最终正确率 21.5% / 21.5% / **24.3%**（baseline / raw / evo）；held-out 10.0% / **23.3%** / 16.7% |
| **统计结论** | **六个配对 McNemar 精确检验全部不显著**（最小 p = 0.125）。这是一个 null result |
| **机制结论** | 闭环成立且可审计：skill 全部来自真实轨迹、有界增长（5 general + 11 topic / 1003 token）、136/144 题被实际注入、209 条编辑决策全程留痕 |
| **最有价值的发现** | 三臂之间的差距**小于**"题内演化轮把已答对的题改错"的数量（8 / 1 / 6 题），而后者由一个与 skill 质量无关的变量驱动 —— 最终消息的长度 |
| **诚实的定位** | 本实验证明了**机制是闭环的**，但**没有证明编译出的 skill 有用**；也没有证明它没用 —— 在这个信噪比下两者都无法分辨 |

---

# Part I · 实验结果

## 1. 实验设置

三条臂，除下表所列外**一切相同**（同一份 `examples/configs/harness_base.yaml`，6 个 overlay 只覆盖使其成为该臂的那几项）：

| arm | 跨问题机制 | 注入内容 |
|---|---|---|
| **Baseline** | 无 | 中性 system prompt（`NEUTRAL_SYSTEM_PROMPT`） |
| **Raw Experience** | 过去轨迹的直接摘要池 | 检索到的原始经验摘要，**同一 token 预算** |
| **Evo-Harness** | Task A/B 的 skill 编译闭环 | 选中的 general + topic skills |

| 共享项 | 值 |
|---|---|
| policy / verifier 模型 | `qwen3-8b`（DashScope），全程冻结，temperature 0.7 / 0.4 |
| selector 模型（仅 evo） | `qwen3-32b`, temperature 0.0 |
| 题内演化轮 | `evolving_round: 2` |
| 工具 | Python 计算工具开启，`max_steps: 4` |
| 题序 | 严格年份序，`seed: 1234`，`batch_size: 8` |
| 注入预算 | `b=6, general_max=3, topic_max=4, tokens=800`（Raw 与 Evo **逐字共用**） |
| 增长上界（仅 evo） | `caps: general=5, per_topic=5`（论文 Appendix F 默认值） |

**数据切分（按年份，绝不 shuffle-then-split）**：adaptation = AIME **2018–2022**（149 题，实跑 144 —— 尾部 5 题因 `loader.batches(drop_last=True)` 被**三臂同等**丢弃）；held-out = **AIME 2025 全年 30 题**，从未被任何 skill 更新触及，仅用 `frozen: true` 评估一次。

## 2. 主结果

| arm | adaptation Pass@1 (round 0) | adaptation 最终 | held-out Pass@1 | **held-out 最终** |
|---|---|---|---|---|
| Baseline | 22.2% (32/144) | 21.5% (31/144) | 13.3% (4/30) | 10.0% (3/30) |
| Raw Experience | 16.7% (24/144) | 21.5% (31/144) | 23.3% (7/30) | **23.3% (7/30)** |
| **Evo-Harness** | 23.6% (34/144) | **24.3% (35/144)** | 13.3% (4/30) | 16.7% (5/30) |

**配对 McNemar 精确检验**（同一道题上比较，括号内是两个方向的不一致样本数）：

| 比较 | adaptation (n=144) | held-out (n=30) |
|---|---|---|
| baseline vs raw | p = 1.000 (9 / 9) | p = 0.125 (0 / 4) |
| baseline vs evo | p = 0.481 (7 / 11) | p = 0.625 (1 / 3) |
| raw vs evo | p = 0.503 (8 / 12) | p = 0.625 (3 / 1) |

**一项都不显著。** Evo 在 adaptation 上高出 baseline 的 2.8 个百分点等于 4 道题，而本系统实测的噪声底是 ±2 题 / 20 题 ≈ ±3.6 个百分点（`docs/findings/diagnostics/`，diagA vs diagA2 同配置重跑）。这个优势完全落在噪声里。

**held-out 上 Raw 反而赢了 Evo**（23.3% vs 16.7%），只有 4 对不一致样本。这同样是噪声，但它是必须主动解释的数字 —— 见 §9 第 2 条，Raw 的收益有一个具体且与"原始经验有用"无关的来源。

## 3. adaptation 上随时间的变化（窗口 = 25）

| 窗口 | 0-24 | 25-49 | 50-74 | 75-99 | 100-124 | 125-143 |
|---|---|---|---|---|---|---|
| baseline | 24.0% | 32.0% | 16.0% | 28.0% | 20.0% | 5.3% |
| raw | 20.0% | 28.0% | 24.0% | 32.0% | 8.0% | 15.8% |
| **evo** | 28.0% | 32.0% | 16.0% | 36.0% | 16.0% | 15.8% |

**没有学习曲线。** 如果 skill 编译在累积，evo 的后段窗口应当高于前段 —— 它没有。而且三条曲线同升同降，说明波动由**题目难度顺序**驱动而非由 arm 驱动。这是本实验对自身假设最直接的负面证据，比任何一个总分都重要，因此放在主结果之后立刻给出。

## 4. per-topic（最终正确率）

| topic | n | adaptation base / raw / **evo** | n | held-out base / raw / **evo** |
|---|---|---|---|---|
| algebra | 30 | 36.7% / 36.7% / **53.3%** | 9 | 0.0% / 22.2% / 11.1% |
| number_theory | 32 | 28.1% / 28.1% / **21.9%** | 5 | 40.0% / 40.0% / 20.0% |
| combinatorics | 33 | 12.1% / 9.1% / **15.2%** | 9 | 11.1% / 11.1% / 11.1% |
| geometry | 49 | 14.3% / 16.3% / **14.3%** | 7 | 0.0% / 28.6% / 28.6% |

唯一超出噪声量级的移动是 **adaptation / algebra：36.7% → 53.3%（+16.7pp, n=30）**。它恰好是 `sk_0001` 所在的领域，而 `sk_0001` 是全场效用最高（50%）、且**唯一从未被编辑过**的 skill。这构成一个连贯的假设 ——「未漂移的具体 skill 才产生迁移」—— 但它是**事后**在 n=30 上观察到的相关，只能作为下一步的研究方向写进 slides，不能作为结论。

held-out 上 geometry 的 0% → 28.6% 只有 7 题、2 道题之差，不承载信息。此外 topic 标注器本身有偏差（adaptation 的 geometry 占 35%，人工标注的 held-out 只有 23%），per-topic 表必须带这个 caveat。

## 5. harness 规模与增长

| 跑完第 N 题 | 7 | 31 | 55 | 79 | 103 | 127 | 143 |
|---|---|---|---|---|---|---|---|
| general | 1 | 3 | **5** | 5 | 5 | 5 | 5 |
| topic | 5 | 9 | 9 | 9 | 10 | 11 | 11 |
| 总 token | 398 | 768 | 913 | 891 | 912 | 977 | **1003** |
| 每条均长 | 66 | 64 | 65 | 64 | 61 | 61 | 63 |

**增长控制按设计工作**：general 层第 55 题起顶到 `caps.general = 5` 就不再增长；144 题只累积到 1003 token；每条 skill 的长度稳定在 60 余 token 而没有膨胀。

**curator 决策：209 次，接受 89、拒绝 120。**

| op | 提出 | 接受 | | 拒绝原因 | 次数 |
|---|---|---|---|---|---|
| ADD | 17 | 16 | | `skipped` | 70 |
| MERGE | 105 | 56 | | **`wrong_layer`** | **49** |
| REVISE | 17 | 17 | | `capacity_full` | 1 |
| SKIP | 70 | — | | | |

`wrong_layer` 占全部决策的 **23%** —— curator 试图改另一层的 skill，被越层闸门（commit `e62ea41`）挡下。闸门本身是对的，但被拒的决策消耗了完整的一次模型调用。**这是 prompt 契约的结构性失败，不是质量判决**，近四分之一的管理预算烧在一类可以在 prompt 层面消除的错误上。

## 6. 开销（solver 与管理**分开报**）

> `metrics.jsonl` 里的 `calls/*`、`tokens/*` 是**累积值**，跨 batch 行求和会得到约 9.5 倍的虚高数字 —— 取每个 run 的最后一行，或直接用 `results.json`。

**adaptation（每臂 144 题）**

| arm | solver 调用 | 管理调用 | 合计 | 调用/题 | 管理占比 | solver token | 管理 token | **总 token** |
|---|---|---|---|---|---|---|---|---|
| baseline | 1468 | 0 | 1468 | 10.2 | 0% | 3.00M | 0 | 3.00M |
| raw | 1448 | 144 | 1592 | 11.1 | 9.0% | 3.14M | 0.12M | 3.25M |
| **evo** | 1228 | **321** | 1549 | 10.8 | **20.7%** | 2.33M | 0.29M | **2.62M** |

evo 的 321 次管理调用：`selector` 136、`reflect` 109、`topic_curator` 58、`general_curator` 18。

**held-out（每臂 30 题，frozen）**

| arm | solver | 管理 | 合计 | 调用/题 |
|---|---|---|---|---|
| baseline | 309 | 0 | 309 | 10.3 |
| raw | 298 | **0** | 298 | 9.9 |
| evo | 263 | 30（**全部是 `selector`**） | 293 | 9.8 |

**注入 context 的 token**

| arm / phase | 总量 | 每题均值 | 最大 | 每题条数 | 零注入题 |
|---|---|---|---|---|---|
| evo / adaptation | 26,648 | **185.1** | 367 | 3.04 | 8 |
| raw / adaptation | 56,128 | **389.8** | 540 | — | 8 |
| evo / held-out | 6,264 | 208.8 | 369 | 3.57 | 0 |
| raw / held-out | 12,244 | 408.1 | 485 | — | 0 |

**六个 run 总计 5,509 次调用 / 10,622,564 token。**

三点值得强调：

1. **evo 多花了 20.7% 的管理调用，总 token 反而比 baseline 少 12.8%** —— solver 输出从 1.66M 降到 1.06M（−36%），注入 skill 让 rollout 变短了。"Evo 更贵"这个直觉在本数据上只对**调用次数**成立（+5.5%），对 token 不成立。
2. **预算上限 800 token，evo 实际只用到 185（23%）；Raw 注入量是它的 2.1 倍，held-out 上还赢了。** 因此 evo 的劣势**不能**归因于"注入得不够多"，token 开销这一项可以从 null result 的嫌疑名单里排除。
3. **两臂在 adaptation 上各有 8 题零注入，正好是 batch 0 的 p_0–p_7**（harness 当时为空）。这是"一道题不可能被自己产生的 skill 影响"在数据上的直接证据。

## 7. skill 使用频次

| skill | 注入次数 | 注入且做对 | 占题数 | 被接受编辑次数 |
|---|---|---|---|---|
| `sk_0006` (general) | **118** | 27 (22.9%) | **81.9%** | 10 |
| `sk_0008` | 74 | 20 (27.0%) | 51.4% | 12 |
| `sk_0007` | 47 | 7 (14.9%) | 32.6% | 3 |
| `sk_0012` | 31 | 6 (19.4%) | 21.5% | 14 |
| `sk_0013` | 31 | 5 (16.1%) | 21.5% | 4 |
| `sk_0003` | 28 | 6 (21.4%) | 19.4% | 4 |
| `sk_0014` | 26 | 6 (23.1%) | 18.1% | 7 |
| **`sk_0001`** | 18 | **9 (50.0%)** | 12.5% | **1** |
| `sk_0011` / `sk_0015` / `sk_0010` / `sk_0009` / `sk_0002` | 16 / 15 / 12 / 11 / 9 | 2 / 2 / 2 / 3 / 2 | — | 11 / 7 / 5 / 5 / 2 |
| `sk_0005` / `sk_0016` | 1 / 1 | 0 / 0 | 0.7% | 2 / 1 |
| `sk_0004` | **0** | 0 | 0% | 1 |

"注入且做对"是**相关而非归因** —— 一条专挑简单题的 skill 不用帮忙也会好看。两个结构性问题：

- **`sk_0006` 被注入到 82% 的题上，命中率 22.9% —— 等于 base rate。** 它是一条占满注入槽的 no-op。原因是语义漂移（§8）：它的 trigger 在 10 次编辑中退化成"When multiple constraints or conditions are present in a problem"这种对几乎任何 AIME 题都成立的空话，选择器因此无从区分。
- **`sk_0004` 一次都没被选中**，却始终占着 `isosceles_triangle_counting` 的 topic 槽位直到最后。cap 限制了总量，但**没有任何淘汰压力**把无用的 skill 挤出去。

## 8. 语义漂移：harness 质量的头号缺陷

MERGE 与 REVISE 整条改写 skill 的 `trigger` / `lesson` / `failure_mode`，**没有任何机制保证改写后 trigger 仍然描述 lesson**。`sk_0006` 的 10 次接受编辑（`export-evo/evolution.jsonl`）：

| batch | op | trigger |
|---|---|---|
| b0 | ADD | When solving a problem, ensure all possible cases are considered… |
| b3 | REVISE | When multiple conditions or constraints are present in a problem. |
| b6 | REVISE | When multiple constraints or conditions are present in a problem. |
| b14 | REVISE | When multiple constraints are present in a problem. |
| b15 | REVISE | When multiple constraints or relationships exist in a problem. |
| **b16** | **MERGE** | **When a circle is tangent to multiple sides of a figure and intersects a diagonal.** |

两个方向的退化同时发生：前半程 trigger 被反复"泛化"成空话，末尾一次 MERGE 又把一条**圆的切线**的具体几何 lesson 并了进来。最终导出的 `harness.md` 里这条 skill 标题是泛化的 `when-solving-a-problem-ensure-all-possible-cases`，body 讲的却是坐标几何切线 —— **trigger 与 lesson 已经不描述同一件事**。`sk_0013`、`sk_0010` 同样。

跨层重复也未解决：p_68 一次注入的 5 条 skill 里有 4 条（`sk_0006`/`sk_0012`/`sk_0013`/`sk_0014`）说的是同一件事（"系统枚举、逐条校验、不要假设"），311 token 大部分是重复内容。

**一个诱人但证据不足的解释，我们主动推翻了它**："编辑次数越多质量越差"整体上只有很弱的支持（≤4 次编辑的 skill 平均效用 24.9%，≥5 次的 20.3%），而且这个差距几乎完全由 `sk_0001` 一个点撑着，`sk_0008` 被编辑 12 次仍有 27.0%。**漂移的结论应当靠上表的文本证据来立，不能靠这个相关性。**

## 9. 迁移案例

完整逐轨迹分析见 [`docs/findings/transfer-cases.md`](docs/findings/transfer-cases.md)。

**① 正向 · p_84**（number_theory，注入 `sk_0001`/`sk_0006`/`sk_0008`，188 token）。baseline 答 37、evo 答 239（对）。可复核的行为差异是**搜索上界 `range(1,100)` vs `range(1,1000)`**（真解 78 和 161，上界 100 必漏 161），外加 baseline **凭空编造了工具返回值**（它的代码实际返回 "78"，它却自称找到 "6, 8, 9, 14"），而 evo 读取了返回值并逐个手算验证。`sk_0006` 的 "no omissions in case analysis" 与放宽上界之间有合理联系。**但这是 temperature=0.7 的单次采样，因果归因未经证实** —— 要坐实需要在不注入条件下重复采样 p_84，本项目没有做。

**② 负向 · p_105**（"所有三位回文数的算术平均"，GT 550）。evo **round 0 答对了 550**，round 1 改成 549.5 而错。事件链：evo 那条正确答案只有 756 字符 → verifier 判"没有给出推理过程"并主动断言 *"The correct mean is 549.5"* → policy 在 round 1 采纳该数字并为它倒推出一套错误算术 → verifier 在 round 1 **自己翻供**回 550，但 `pass_final` 取最后一轮，已经来不及。

注意注入的三条 skill **没有**把模型带向错误的**数学方法**；它们施加的是一种"再验证一遍"的**风格压力**，作用在一道不需要验证的题上。这是 context interference 的一种形态，但机制是风格层面的，不是知识层面的。

**③ 系统性 · 上面那条链不是孤例。** 定义 loss = round 0 对、最终错：

| arm | loss | gain | 净 | 被毁题的 round-0 消息均长 | 全部题均长 |
|---|---|---|---|---|---|
| baseline | **8** | 7 | −1 | 1827 | 2575 |
| raw | **1** | 8 | **+7** | 766 | 2491 |
| evo | **6** | 7 | +1 | 1302 | 2353 |

以及 verifier 的误拒率（分母是 policy **确实答对**的轮次）：baseline **21/63 = 33%**、raw **14/55 = 25%**、evo **22/69 = 32%**。**policy 答对时，verifier 大约每三次否定一次**，而且常常附带一个具体的错误数字。

三条结论：

1. **题内演化轮在 baseline 上是净负的**（毁掉 8 条、救回 7 条）。这是上游 AlphaApollo 自身的行为，与本项目无关，但它是三臂共同的噪声源。
2. **Raw 臂在 adaptation 上追平 baseline，靠的不是多做对题，而是少毁题。** 它 round-0 比 baseline 低 5.5 个百分点，最终却打平 —— 全部差距来自 loss 从 8 降到 1。**任何把 Raw 的成绩读成"原始经验也有用"的写法都是错的**：它的收益发生在"防止答案被改坏"这一环，不在解题能力上。
3. 三臂一致地，**被毁掉的题其 round-0 最终消息明显更短**。verifier 按**看得见的推理**打分，一条简短、断言式的正确答案会被判成"没有给出推理"，verifier 随即给出自己（常常是错的）数字，policy 在下一轮采纳它。

**这个 loss 数（8 / 1 / 6）大于三臂之间的全部差距，而它由"最终消息有多长"这样一个与 skill 质量无关的变量驱动。** 解读任何 arm 间差异之前必须先看这张表。

## 10. 对 null result 的分析

任务书明确写了"Evo-Harness 赢过 baseline 不是及格线，但 null result 必须被分析"。按它列出的六个方向逐条：

1. **任务相似度。** AIME 跨年之间共享的是**领域**而非**可复用的解题程序**。harness 里真正具体的 skill（如 `sk_0001`）只在 18/144 题上被选中，82% 的题拿到的是通用套话。这可能是最根本的限制：AIME 的设计初衷就是每题需要一个新想法，而 Evo-Harness 论文的场景有可复用的操作流程。
2. **skill 质量。** §8 的语义漂移。这是**可修的工程缺陷**，不是方法本身的问题。
3. **选择错误。** `sk_0006` 注入率 82%、效用等于 base rate；`sk_0004` 注入率 0%。选择器不是选错了，而是**在漂移后的 trigger 上无法区分** —— 修好 (2) 才谈得上评价 (3)。
4. **context 干扰。** 确有其事，但机制出乎意料：不是知识层面的误导，而是 §9 的**风格层面**干扰。注入使 evo 把验证前置、最终消息变短（中位数 2280 vs baseline 2596），而短消息更容易被 verifier 误判，进而在下一轮被改坏（evo loss 6 vs raw 1）。
5. **verifier 反馈质量 —— 比预想的严重得多。** policy 答对时 verifier 否定它的比例是 33% / 25% / 32%。编译 skill 所依赖的成败标签本身就带着这个量级的噪声；更糟的是这噪声不只污染 skill，它还经由题内演化轮**直接破坏最终答案**。
6. **token 开销。** **不是**瓶颈，可以排除：evo 只用掉 800 预算中的 185 token，总 token 还比 baseline 少 12.8%。

**结论**：本实验证明了**机制是闭环的**，但**没有证明编译出的 skill 有用**。最大的单一嫌疑并不是"跨题迁移在 AIME 上不成立"这个方法论结论 —— 而是在能检验它之前，两个工程缺陷（语义漂移、23% 的 `wrong_layer` 浪费）与一个测量缺陷（verifier 噪声经由题内演化轮破坏最终答案，且与消息长度耦合）已经把信噪比压到了差异无法分辨的水平。**下一次实验该修的是这三项，而不是换一个更大的模型。**

---

# Part II · 对照验收重点

## A. 概念正确性：题内 solution memory 与跨问题 skill harness 的区分

这不是靠文档约定，而是**四个相互独立的层面**都做了分离：

| 层面 | 题内 solution memory | 跨问题 skill harness |
|---|---|---|
| **代码位置** | `alphaapollo/core/environments/memory/`（上游，未改一行） | `alphaapollo/core/harness/`（本项目新增的独立包） |
| **生命周期** | 每道新题 `env.reset()` 全部清空 | 跨题持久化到 `store_root`，进程重启后仍在 |
| **注入通道** | user prompt 模板的 `{previous_solutions}` / `{memory_context}` 槽位 | **system message**（`arms.py:system_prompt_for()`） |
| **内容** | 本题的历史解答与 verifier 反馈原文 | 自然语言的可复用操作知识（trigger / lesson / failure_mode），**从不含原题、答案或完整解法** |

**结构性保证（而非约定）**：`tests/harness/test_smoke.py::test_harness_does_not_import_inproblem_memory` 用 **AST 扫描** harness 包下的每一个 `*.py`（**包括 `__init__.py`** —— `pkgutil.iter_modules()` 从不产出它，这是早期版本的漏报点），检测 8 种 import 写法（含相对导入、函数体内导入）。同时用 7 个反例锁死误报：docstring 里提到这个包名、注释里提到、前缀相同的另一个包名（`memoryX`），都不得触发。

**为什么用 system message 作为注入通道**：它是题内 memory 完全不使用的通道，两种机制因此在 prompt 里也不会互相污染或被混淆。代价是必须保证三臂**都**发非空 system prompt（见 D）。

**上游源码零修改。** 对比 fork 基线 `712a04d`，被修改（`M`）的上游文件只有 `.gitignore` 与 `pyproject.toml`（后者仅加了一段 pytest 配置），其余全部是新增。需要改上游行为的两处（开销统计）通过**类级 monkey-patch** 在运行期完成，`uninstall()` 后复原。这是刻意的约束：Task C 要求三臂共享同一个冻结 solver，改上游文件就无法论证"baseline 还是原来那个 baseline"。

## B. Online protocol：无未来信息、无当前题反馈、无测试集泄漏

### B1. 无未来信息 —— 结构性保证，不是调用顺序约定

每个有状态的 arm 在 `begin_batch()` 时**冻结**跨问题状态的一份副本；该 batch 内所有 `system_prompt_for()` 调用**只读这份冻结副本**，从不读 live store。只有 `end_batch()` 被允许写 live 状态（`arms.py` 模块 docstring 不变式 2）。

这样即使一个调用方绕过本类、在 batch 中途直接改 live store，也**无法**改变在飞 batch 的任何一道题看到的内容 —— 因为它们从一开始就没在读 live 状态。

测试：`test_evo_arm_prompt_changes_only_after_a_batch_boundary`、`test_mutating_the_live_store_after_begin_batch_does_not_change_this_batchs_prompt`、`test_end_batch_tags_harness_edits_with_the_batch_index_not_a_problem_index`。

**运行时实证**：adaptation 的前 8 题（batch 0）注入 token 恰为 **0**，且 `selection_log.jsonl` 里每条 skill 的 `evidence` 引用的 `p_xx` 全部小于使用它的题号。

### B2. 不使用当前题的 ground truth

- **Reflect 根本拿不到它**：`build_reflect_context` 的源码里**没有出现过** `ground_truth` 或 `gt_traj` 标识符（可 grep 验证）。它的输入只有轨迹文本、工具输出、成败标签与 verifier 反馈。
- **verifier 的答案比对通道被剥离**：`sanitize_feedback()` 在把 verifier 报告交给 Reflect 之前，按 `_GT_CHANNEL` 正则整行删除 AlphaApollo 自带的 `Matches ground truth: True/False` 行（先剥 `<think>` 块，因为 reasoning 模型的回包会带这层包装）。
- **policy 若试图自己去问答案，整个 run 立刻终止**：`LeakageError`（`evolving_harness_main.py:124`）在检测到 action 文本调用 `<informalmath_verify>`（其工具回包会把 "Matches ground truth" 回显给 policy）时抛出，且是**唯一一个不被 per-problem try/except 吞掉**的异常 —— 泄漏会让这次 rollout 的数据整体失效，继续跑只是浪费预算。测试 `test_driver.py:206`。
- **`guard.py` 是整个 harness 包里唯一允许接触 `ground_truth` 的模块**，且只用于**拒绝**候选：ground-truth 字符串被比对之后立即丢弃，从不流入返回值、日志行或异常消息。这个例外在模块 docstring 里写明，其余模块由 CI 检查该标识符不出现。
- Reflect **只从失败学**：`test_evo_arm_reflects_only_on_failures`、`test_mixed_batch_evo_only_reflects_on_the_failure`。

### B3. 不写入原题 / 答案 / 完整解法

`guard.validate_skill()` 在候选入库前逐条检查，拒绝原因是固定字符串、原样写进 `harness_log.jsonl` 的 `reject_reason`：`empty_section` / `non_english` / `lesson_too_long` / `skill_too_long` / **`question_overlap`**（与原题的 8-gram 重叠）/ **`answer_leak`**。

`answer_leak` 的判定经过一次实测修正：早期版本用"裸数字等于 ground truth"，对真实 adaptation pool 实测**误拒 21.7–27.9%** 的枚举类 skill（commit `1ff380f`），因为 AIME 的答案有 4% 恰好等于常见的界数、模数或步数。现在要求数字**同时**出现在断言性措辞附近（`_ASSERTION_CUE`），仅仅巧合相等则**保留但标记** `numeric_coincidence` 到 `guard_note`，供审计而非拒绝。`test_guard.py` 有 18 个用例锁死这套判定，其中 8 个专门覆盖数字正则的边界情况（句末句点、真小数、标识符内的数字、点分版本号等）。

**实测验证**：本次 run 的 `store/skills/*.md` 与 `pool/pool.jsonl` 中，`ground_truth` 出现 0 次，`\boxed` 与 "answer is" 出现 0 次。

### B4. 测试集不泄漏

- **按年份切分，绝不 shuffle-then-split**：adaptation = AIME 2018–2022，held-out = AIME 2025，题面**零重叠**（已实测）。
- held-out 用 `frozen: true` 跑：Reflect 与两个 curator **完全不运行**，skill 只被选择和注入，不被新增或修改。
- **实证而非声明**：`heldout-evo` 的管理调用是 **30 次，全部是 `selector`**，`reflect` / `topic_curator` / `general_curator` 均为 **0**（§6 表）。这是"最终 harness 确实被冻结"的运行时证据。
- `frozen` 在每个能学习的 arm 上都可用：`test_the_raw_arm_can_be_frozen_for_held_out_evaluation`、`test_adversarial_frozen_is_available_on_every_arm_that_can_learn`；且冻结的 arm **仍然注入它已加载的内容**（"停止学习"而非"停止使用已学到的"）：`test_a_frozen_raw_arm_still_injects_the_pool_it_loaded`。

## C. 闭环完整性：skills 从真实轨迹产生、被组织、并实际影响后续问题

| 环节 | 证据 |
|---|---|
| **来自真实轨迹** | 209 条编辑记录每条都带 `source_problems` 与 `candidate.evidence: [p_xx]`，指向真实跑过的题。`test_evo_arm_does_not_compile_a_skill_from_a_problem_that_never_ran` 与 `test_adversarial_a_round_that_ran_but_produced_no_text_is_still_a_failure` 锁死"不从空轨迹编 skill"（这是 commit `70e77b6` 修的真实缺陷） |
| **被组织** | 两层结构（5 general + 11 topic），四种算子 ADD / MERGE / REVISE / SKIP，接受 89 / 拒绝 120，全部落 `harness_log.jsonl` |
| **实际影响后续问题** | 144 题中 **136 题**被实际注入（其余 8 题是 batch 0 的冷启动），平均 3.04 条 / 185 token；`sk_0006` 单条被注入 118 次。held-out 30 题全部被注入 |
| **只影响后面的题** | 见 B1：batch 级冻结 + evidence 题号严格小于使用题号 |
| **增长有界** | general 层第 55 题顶到 cap 后不再增长；总量 144 题累积 1003 token（§5） |
| **可导出** | `python -m alphaapollo.core.harness.export` → `harness.md`（人读）+ `summary.json`（机器读）+ `evolution.jsonl`（完整演化日志），已随仓库提交 |
| **失败不破坏 baseline** | `test_store_apply_failure_mid_end_batch_leaves_the_harness_unchanged`：`end_batch` 中途失败时 harness 原样不动。运行时实测**丢题 0**，且 `test_evo_harness_arm_introduces_no_bare_unaccounted_model_call` 保证没有未计入开销的模型调用 |

**闭环的薄弱处也要说清楚**：闭环在**机制**上完整，但在**效果**上未被证明 —— §7/§8 显示 82% 的注入是一条效用等于 base rate 的 no-op。"skills 实际影响了后续问题"是可验证的事实（注入发生了、内容进了 context），"skills 有益地影响了后续问题"不是。

## D. 实验公平性：相同问题顺序与 solver 设置，完整报告额外开销

### D1. 三臂共享的东西被放在一处，因此可检查而非仅可声明

`examples/configs/harness_base.yaml` 是唯一来源，6 个 arm/phase overlay 用 `base_config:` 继承它，**只覆盖使自己成为该臂的那几项**。因此 `diff harness_adapt_baseline.yaml harness_adapt_evo.yaml` 显示的就是实验操纵本身，没有别的。模型、题序、题内演化轮数、工具、生成参数全部在 base 里。

题序与配置的一致性由 `progress.json` 的 fingerprint 记录（`stream_path` / `arm` / `batch_size` / `seed` / `frozen`），三个 adaptation run 除 `arm` 外完全一致。

### D2. 三个刻意设置的公平性保障

- **三臂都发非空 system prompt。** 上游 `utils/agent.py:37` 是 `if self.system_prompt:` —— 空串会导致**根本不发 system message**。若 baseline 无 system message 而 evo 有，比较的就是"有没有 system 角色"而不是"注入内容是什么"。因此 `CrossProblemArm` 的默认实现返回 `NEUTRAL_SYSTEM_PROMPT` 而非 `""`，且冷启动时三臂发出**完全相同**的中性 prompt。测试：`test_every_arm_emits_a_non_empty_system_prompt`、`test_cold_start_arms_all_emit_the_same_neutral_prompt`。
- **Raw 是一个认真的对手，不是稻草人。** 它与 Evo **逐字共用**同一份注入预算（`b=6, tokens=800`），因为实验问的是"编译出的 skill 是否胜过原始经验"，两臂必须在**注入什么**上不同，绝不能在**注入多少**上不同。测试：`test_raw_experience_respects_the_same_token_budget_as_evo`、`test_raw_experience_selection_count_never_exceeds_b`。Raw 还**同时存成功与失败**（Evo 只从失败 Reflect），这对它有利：`test_raw_experience_stores_successes_too`。
- **`max_workers` 只影响吞吐，不影响协议。** batch 边界（哪些题能看到哪些 skill）与并发度完全无关，且 `max_workers` 被**刻意排除**在 resume fingerprint 之外，以便在重试之间调整而不使已完成的批次失效。

### D3. 开销口径主动选择了对自己不利的算法

`accounting.py` 按 role 分桶：**solver 侧** = `solver` / `summarizer` / `aggregator`；**管理侧** = `reflect` / `topic_curator` / `general_curator` / `selector` / `raw_summarizer`。

- **`raw_summarizer` 单独成 role，不复用 `summarizer`。** 后者已经命名了上游**题内**的求解侧助手（baseline 本来就有）。把 Raw 臂的每题摘要调用折进去，会让 Raw Experience 在成本报告里显得几乎不花钱，而 Evo 的 reflect/curate 却被正确计为管理开销 —— 那是一个**会错误地削弱本项目自身论点**的数字（commit `7b4336d`）。这里选择了诚实的口径。
- **`calls/unscoped` 是一个公开的自检列，六个 run 全为 0。** 它非零就意味着 solver/管理的拆分是错的。这个列的存在源于一个真实缺陷：`role_scope` 用 `ContextVar`，而 `ThreadPoolExecutor` **不传播** context，上游扇出到线程池的 verifier 调用曾整批落进 `unscoped`（第一次干净运行里 38 次 solver 侧调用有 9 次落在那里）。修复是在父线程 `.start()` 时捕获 role、子线程 `.run()` 时应用；且它是**默认值不是锁** —— 没有 scoped 祖先的线程仍然保持 `unscoped`，这样一个忘记打 scope 的管理调用点仍然**可见**，而不是被悄悄折进 solver 桶（commit `62b024b`）。
- **丢题数被当作一项指标记录并报告**（本次 0 / 0 / 0）。这来自烟囱测试的教训：一次运行因本机 SOCKS 代理断连丢了 25% 的题而 `exit=0`，"某条臂成绩低"完全可能只是它那轮网络更差。

### D4. 公平性上仍存在的一个缺口

**只跑了单 seed 一轮。** README §12.1 与本文 §2 都写明：三臂差距（2–3 个百分点）正好是单 seed 分辨不了的量级，论文自己是 3 次取平均。这是本项目结论强度上最大的单一缺口，已列入未完成项。

## E. 研究判断：transfer、负迁移、context growth 与失败原因

不重复 Part I，只列出体现判断而非记账的几处：

1. **主动做了显著性检验而不是只报百分比。** 六个配对 McNemar 全部不显著，并据此把所有 arm 间比较的措辞定为"未能分辨"而非"A 优于 B"（§2）。
2. **主动指出最有力的负面证据。** §3 的"没有学习曲线"紧跟在主结果之后，而不是埋在末尾 —— 因为它比总分更能说明问题。
3. **负迁移分析追到了真实机制，而不是停在"技能不好"。** p_105 的链条是 verifier 误判 + 风格耦合（§9②），并把它量化成 loss 表（§9③），最终得出一个反直觉但有数据支撑的结论：**三臂差距小于题内演化轮自身造成的损失**。
4. **推翻了对自己有利的解释。** "编辑次数越多质量越差"本可以直接用来支持漂移论点，但整体相关性很弱且由单点撑着，因此明确写明"不要用这个相关性去论证"（§8）。
5. **纠正了一个自己先前的口径错误。** verifier 误拒率最初按 21/144 算成"15% 量级"，分母错了 —— 正确分母是 policy 答对的轮次数，实际是 33% / 25% / 32%（§9③）。
6. **正向案例只主张能主张的部分。** p_84 给出了可复核的行为差异（搜索上界、是否读工具输出），但明确写明因果归因**未经证实**，并指出坐实它需要什么实验（§9①）。
7. **对 Raw 的意外胜利给出了具体解释而非回避。** 它的收益来自"少毁题"而非"多做对题"，并据此警告"任何把 Raw 读成原始经验有用的写法都是错的"（§9③第 2 条）。
8. **context growth 的两面都报。** cap 确实生效（§5），但**没有淘汰机制** —— `sk_0004` 零注入仍占槽位到最后（§7）。
9. **null result 按六个方向逐条分析，并明确排除其中一项**（token 开销不是瓶颈，§10）。
10. **给出了可执行的下一步**：修语义漂移、修 `wrong_layer`、解耦 verifier 噪声与 `pass_final`，而不是"换更大的模型"。

**一个不涉及 skill 机制就能做的改进（已识别、未实现）**：让 `pass_final` 取"所有轮里被 verifier 判对过的答案"而非"最后一轮的答案"。按 §9③ 的表，这会让 baseline +8、evo +6、raw +1，且对三臂都公平。没有实现是因为它要动上游的题内逻辑，而任务书明确说不需要动 —— 这是一个**有意识的范围决定**，写在这里以便 reviewer 判断是否同意。

## F. 工程表达：配置、日志、README、commit history、复现命令

| | |
|---|---|
| **配置** | 7 份 YAML：1 份 `harness_base.yaml` + 6 个 arm/phase overlay，`base_config:` 继承，每份都有解释"为什么是这个值"的注释（含实测依据） |
| **日志** | 每题：`metrics.jsonl`（逐题成败 + 累积开销）、`selection_log.jsonl`（注入了哪些 skill、多少 token、成败）；每次编辑：`harness_log.jsonl` / `evolution.jsonl`（op / actor / 接受与否 / 拒绝原因 / reason / 证据题号）；每题轨迹落盘 |
| **可恢复** | `progress.json` + fingerprint 的断点续跑；续跑继续同一个 wandb run 而非新开一条 |
| **测试** | **571 个，`571 passed in 5.22s`**，含 AST 边界守卫、防泄漏、预算、公平性、resume、清理 |
| **README** | `README_HARNESS.md` 1112 行，12 节，含一张**"本文档中每个数字的来源"**附录表 |
| **commit history** | 97 个 purpose-scoped commit，消息写"为什么"而非"改了什么"，例如 `docs: ten in-problem evolution rounds buy nothing at 3.9x the cost`、`fix(harness): release each problem's model clients, which killed the first run`、`fix(harness): a curator may only edit skills in its own layer` |
| **产物入库** | 63 个小产物（约 600KB）随仓库提交 —— 汇总 metrics、导出的 harness 与完整编辑日志、选择日志、以及本文逐行分析的 6 份轨迹。43MB 轨迹全集与 wandb 目录排除，规则与理由写在 `.gitignore` 末尾 |
| **无密钥/无权重/无大数据** | 每个模型配置块都是 `api_key: ""`，key 只从 `OPENAI_API_KEY` 环境变量读；已扫描全部入库产物确认无 key、无 ground truth |

**复现命令**（完整版见 README §2 与 §7.4）：

```bash
# 环境
conda create -n alphaapollo python==3.12 -y && conda activate alphaapollo
cd AlphaApollo && bash installation.sh
export OPENAI_API_KEY=<your key>
export NO_PROXY="$NO_PROXY,dashscope.aliyuncs.com"   # 见 README §2.2b

# 测试
python -m pytest tests/ -q            # 571 passed

# 数据（按年份切分，不 shuffle）
python -m alphaapollo.data_preprocess.prepare_harness_stream \
    --out_dir ./data/harness --label_model qwen3-32b \
    --base_url https://dashscope.aliyuncs.com/compatible-mode/v1

# 三臂 × 两阶段，六个 run（默认三臂并行，约 15 小时）
bash scripts/run_experiments.sh
bash scripts/status.sh                # 读产物看进度，不看终端

# 出结果（本文 Part I 的每一张表）
python -m alphaapollo.core.harness.analysis --root ./outputs/harness --out_dir ./outputs/harness/report
python -m alphaapollo.core.harness.export --store_root ./outputs/harness/adapt-evo/store --out_dir ./outputs/harness/export-evo
```

---

## 已知局限（与未完成项）

1. **单 seed 一轮。** 结论强度上最大的缺口；三臂差距正好落在单 seed 分辨不了的量级。
2. **地板效应。** `qwen3-8b` 在 AIME 上约 20% 的正确率 + held-out 仅 30 题（单题 = 3.3pp），在结构上就难以分辨 arm 差异。
3. **p_84 的因果归因未经检验** —— 缺少不注入条件下的重复采样对照。
4. **语义漂移与 `wrong_layer` 浪费均未修复**，两者都在本次 run 中实际发生并可能压低了 evo 的表现。
5. **harness 无淘汰机制** —— 零注入的 skill 会永久占据槽位。
6. **topic 标注器有偏差**（adaptation geometry 35% vs held-out 23%），per-topic 表须带此 caveat。
7. **Slides 尚未制作。**
8. **轨迹文件不记录注入的 system prompt**，复核某题当时看到的 skill 原文需要从 `selection_log.jsonl` + `harness_log.jsonl` 重放编辑（方法写在 `docs/findings/transfer-cases.md` 开头）。这是一个应当修掉的可审计性缺口。
