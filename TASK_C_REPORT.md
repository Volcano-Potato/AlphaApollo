# Task C 实验报告

在 AlphaApollo 之上新增**跨问题 skill 演化层**（Evo-Harness），并以三臂对照实验检验它。本文是自足的实验记录：环境与数据、两种 memory 的边界、skill 机制设计、三组实验的命令 / 结果 / 日志、真实演化记录、失败尝试与限制、上游归属，最后是对六项验收重点的自查。

系统的完整设计论证在 [`README_HARNESS.md`](README_HARNESS.md)（1112 行），迁移案例的逐轨迹分析在 [`docs/findings/transfer-cases.md`](docs/findings/transfer-cases.md)。本文引用它们，不复述。

**每个数字都可以从仓库内的产物重算**：`outputs/harness/report/results.{md,json}`（`analysis.py` 生成）、`outputs/harness/export-evo/`（最终 harness + 209 条编辑日志）、各 run 的 `metrics.jsonl` 与 `selection_log.jsonl`。

---

## 摘要

| | |
|---|---|
| **跑完了什么** | 6 个 run = 3 臂 × {adaptation 144 题, held-out 30 题}，**丢题 0**，harness 层异常 0 |
| **主结果** | adaptation 最终 21.5% / 21.5% / **24.3%**（baseline / raw / evo）；held-out 10.0% / **23.3%** / 16.7% |
| **统计结论** | **六个配对 McNemar 精确检验全部不显著**（最小 p = 0.125）。这是一个 null result |
| **机制结论** | 闭环成立且可审计：skill 全部来自真实轨迹、有界增长（5 general + 11 topic / 1003 token）、136/144 题被实际注入、209 条决策全程留痕 |
| **最有价值的发现** | 三臂之间的差距**小于**"题内演化轮把已答对的题改错"的数量（8 / 1 / 6 题），而后者由一个与 skill 质量无关的变量驱动 —— 最终消息的长度 |
| **诚实的定位** | 证明了**机制是闭环的**，但**没有证明编译出的 skill 有用**；也没有证明它没用 —— 在这个信噪比下两者都无法分辨 |
| **成本** | 六个 run 合计 5,509 次模型调用 / 10,622,564 token，墙钟约 15 小时 |

---

# 1. 环境、硬件、模型服务与数据

## 1.1 硬件与模型服务（本项目实际使用的配置）

| 项目 | 实测值 |
|---|---|
| 机器 | MacBook Air M4 / 24 GB，**无本地 GPU** |
| 模型服务 | 全部走托管 API：阿里云 DashScope 的 OpenAI 兼容端点 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Solver / verifier（全程冻结） | `qwen3-8b`，temperature 0.7 / 0.4，`max_tokens: 8192` |
| Selector / topic 标注器 | `qwen3-32b`，temperature 0.0 |
| 并发上限（实测） | **12 并发安全；16 开始被限流；24 丢失约 40% 请求** |
| 单次调用延迟（实测） | 中位数 16.5 s，约 1200 output tokens |

任务书建议的 `Qwen3-4B-Instruct-2507` 在 DashScope 上**不存在**，solver 因此改为 `qwen3-8b`。这是一个会压低所有三臂绝对分数的偏离，已在 §7 的地板效应里计入。

三个 solver 参数偏离了 AlphaApollo 出厂的 `vllm_informal_math.yaml`，每一处都有 20 题实测依据（配置与逐题结果在 `docs/findings/diagnostics/`）：

| 参数 | 本项目 | 上游 | 依据 |
|---|---|---|---|
| `evolving_round` | **2** | 10 | 10 轮得分 4/20，与 2 轮**完全相同**，代价是 3.9 倍调用 |
| `verifier_env_num` | **1** | 5 | 5 个 verifier 得分 4/20，与 1 个相同，代价 2.0 倍；且 100 次 problem-run 里**从未**出现正确答案被多数判决改错 |
| `max_tokens` | 8192 | 8192 | 唯一真正有影响的旋钮（+2 题），且比 2048 **更便宜** —— 推理被截断会驱动额外的重试轮 |

**DashScope 的一个硬性要求**：qwen3 系列 hybrid-reasoning 模型在**非流式**调用下，请求体必须带 `enable_thinking: false`，否则直接报错。上游 `Agent` 无法表达该参数，由 `accounting.install_accounting(..., extra_body=...)` 注入，配置项 `harness.extra_body`。默认不传，因此指向 vLLM 端点时请求体与上游逐字节一致。

## 1.2 安装与前置

```bash
conda create -n alphaapollo python==3.12 -y && conda activate alphaapollo
cd AlphaApollo && bash installation.sh
export OPENAI_API_KEY=<your key>          # 任何配置文件里都没有 key，只从环境读
export NO_PROXY="$NO_PROXY,dashscope.aliyuncs.com"

python -m pytest tests/ -q               # 571 passed in 5.22s
```

**`NO_PROXY` 不是可选项。** 开发机的 `ALL_PROXY` / `HTTP_PROXY` / `HTTPS_PROXY` 全部指向本地 SOCKS 代理，所有流量默认走它。一次 12 题烟囱测试因此丢了 3 道题（25%），全部是代理侧 `ConnectError: [Errno 61] Connection refused`，而非 DashScope 拒绝服务（实测直连 0.38 s / 走代理 1.00 s）。`run_experiments.sh` 现在自己会加这条，但单独跑某个 run 时必须手动设置。

## 1.3 数据准备

```bash
python -m alphaapollo.data_preprocess.prepare_harness_stream \
    --out_dir ./data/harness --label_model qwen3-32b \
    --base_url https://dashscope.aliyuncs.com/compatible-mode/v1
```

**按年份切分，绝不 shuffle-then-split。**

| | adaptation | held-out |
|---|---|---|
| 来源 | `gneubig/aime-1983-2024`，**2018–2022** | `MathArena/aime_2025` |
| 题数 | **149**（实跑 144） | **30**（全部） |
| 按年 | 2018:30 / 2019:30 / 2020:30 / 2021:30 / 2022:29 | 2025:30 |
| topic 来源 | 模型标注（`qwen3-32b`） | **MathArena 人工 `problem_type` 标签** |
| topic 分布 | geometry 52 / combinatorics 34 / number_theory 33 / algebra 30 | combinatorics 9 / algebra 9 / geometry 7 / number_theory 5 |

已验证：`problem_idx` 连续、按 (year, contest, number) 排序、**两个 split 题面零重叠**、每个答案都是 0–999 的纯整数、`gt_traj` 全部为空串。

- **`gt_traj` 留空是刻意的** —— 这两个数据源本来就不提供完整解法，把答案填进去等于给 Reflect 开一条 ground-truth 通道。
- **`2022-II-8` 被剔除**（150 → 149，仍在任务书 80–150 区间内）：它公布的答案是 `"080 or 081 (both were accepted)"`，而打分器是字符串比较，留着它会让**每一条臂的每一次尝试**都被判错。它不是难题，是一道**不可打分**的题。
- **实跑 144 而非 149**：`loader.batches(drop_last=True)` 丢弃填不满一个 batch 的尾部 5 题，**三臂同等**丢弃，比较集完全一致。

**topic 标注是什么、不是什么。** 方法本身**不读任何一道题的 topic**：Reflect 自己命名 topic，selector 检索时也没有 topic 过滤。topic 列的唯一用途是让本报告能做 per-topic 拆解 —— 而那是三条臂都要报的结果，只有在三条臂看到**逐字节相同**的标注时才可比，所以它被一次性冻进 stream 文件，而不是每条臂在线各标一遍。标注调用记在 `offline_labeling` 这个管理 role 下，作为跨题开销出现在成本报告里。`classify_topic` 的签名收一个**裸 question 字符串**而非 problem dict，带着 `ground_truth` 的字典不可能被误传。

标注质量（对 30 道 AIME 2025 题与人工标签比对）：`qwen3-8b` 21/30 = 70%，`qwen3-32b` 25/30 = **83%**。`qwen3-8b` 的错误是**系统性**的（它输出 geometry 14 次而人工只有 7 次），这就是标注模型与冻结 solver 分开配置的原因。

> **n=30 的区间约 ±13pp，且 adaptation 的 geometry 占比（35%）明显高于人工标注的 held-out（23%），方向与该模型的已知偏差一致。§4.5 的 per-topic 表必须带这个 caveat。**

---

# 2. 题内 solution memory 与跨问题 skill harness 的区别

这是本项目最核心的概念区分。它不靠文档约定，而是**四个相互独立的层面**都做了分离：

| 层面 | 题内 solution memory（上游） | 跨问题 skill harness（本项目） |
|---|---|---|
| **代码位置** | `alphaapollo/core/environments/memory/`（`SimpleMemory` / `EvolvingMemory` / `NDimensionalMemory`），**未改一行** | `alphaapollo/core/harness/`（新增的独立包，19 个模块） |
| **生命周期** | 每道新题 `env.reset()` 全部清空，**题与题之间不留下任何东西** | 持久化到 `store_root`，跨题累积，进程重启后仍在 |
| **注入通道** | user prompt 模板的 `{previous_solutions}` / `{memory_context}` 槽位 | **system message**（`arms.py:system_prompt_for()`） |
| **内容** | 本题此前的解答原文与 verifier 反馈 | 自然语言的可复用操作知识（trigger / lesson / failure_mode）；**从不含原题、答案或完整解法** |
| **能否看到本题答案** | 能（就是本题） | **结构性地不能**（见 §3.5） |

**为什么用 system message 作注入通道。** 它是题内 memory 完全不使用的通道，两种机制因此在 prompt 里也不会互相污染或被混淆；且 skill 是"求解前就成立的先验知识"，语义上属于系统指令而非对话历史。代价是必须保证三臂**都**发非空 system prompt（见 §4.2）。

**边界由 AST import guard 强制，不是靠约定。** `tests/harness/test_smoke.py::test_harness_does_not_import_inproblem_memory` 扫描 harness 包下每一个 `*.py`（**包括 `__init__.py`** —— `pkgutil.iter_modules()` 从不产出它，这是早期版本的漏报点），用 AST 而非文本匹配检测 8 种 import 写法（含相对导入、函数体内导入）。同时用 7 个反例锁死误报：docstring 里提到这个包名、注释里提到、前缀相同的另一个包名（`memoryX`），都不得触发。

> 早期版本用 `pkgutil` + 子串搜索，**同时漏报 `__init__.py` 与误报 docstring** —— 一个既会放过真违规、又会把解释这条边界的文字本身判成违规的守卫（commit `35a811b`）。

---

# 3. Skill 机制设计

## 3.1 表示

一个 skill 就是一个磁盘文件 `<store_root>/skills/<slug>--sk_NNNN.md`，markdown + JSON frontmatter，双向可逆：

```markdown
---
id: "sk_0001"
name: "when-solving-problems-involving-polynomial-factorization-wit"
level: "topic"
topic: "factorization_conditions"
evidence: ["p_0"]
created_at: 0
revised_at: []          # <- 144 题之后仍是空的
n_selected: 18
n_selected_success: 9
n_tokens: 63
---
## When to use
when solving problems involving polynomial factorization with integer coefficients

## Strategy
- Identify necessary conditions for factorization.
- Translate constraints into equations involving roots.
- Count valid root pairs that satisfy both sum and product conditions.

## Avoid
providing answers without showing the logical steps or reasoning process.
```

三段与论文的 trigger / rule / scope 模式同构：

- **`## When to use`（trigger）** —— 适用条件，**同时就是检索键**：selector 看到的目录只列 `[id] (scope) trigger`。
- **`## Strategy`（lesson）** —— 可复用的操作策略。
- **`## Avoid`（failure_mode）** —— 要避免的失败模式。

`render_harness` 注入给 policy 的**只有这三段**，所有簿记字段（id / evidence / 计数器）永不进入模型上下文。

两层：`general`（跨 topic，`Caps.general = 5`）与 `topic`（topic 内局部流程，`Caps.per_topic = 5`），均为论文默认值。

文件名由 **id** 派生并做路径清洗，而非由 LLM 产出的 `name` —— 后者曾导致两个不同 id 的 skill 静默覆盖同一个文件（内存里还留着两个，reload 后不一致且无任何报错），且 `../../pwned` 这样的名字能写到 `skills_dir` 之外（commit `f475ff5`）。

## 3.2 选择（Select）

**模型驱动**，对应论文 Algorithm 1 line 5 与 Appendix F。`selector.select_skills(agent, question, skills, budget)` 把题面 + 整个 harness 的 trigger 目录交给 selector 模型，让它按有用性排序回若干 id，或回 `NONE`。

**预算在代码里强制，不交给模型。** 模型只决定"选哪些、什么顺序"，`apply_budget()` 之后按确定性规则裁剪：

| 预算项 | 默认 | 含义 |
|---|---|---|
| `budget.b` | 6 | 最多注入几条 |
| `budget.general_max` | 3 | general 层配额 |
| `budget.topic_max` | 4 | topic 层配额 |
| `budget.tokens` | 800 | 累计 token 上限 |

超预算的 skill 被**跳过而非终止遍历**，排名靠后的小 skill 仍可用掉剩余额度。理由写在 `selector.py` 的 docstring 里：**模型可以忽略的指令不是约束**。

**失败降级为空选择**：selector 调用抛异常 → 该题用中性 system prompt 跑完，结构上与 Baseline 完全一致，绝不把整条 run 带崩（Task B 硬约束）。

## 3.3 更新（Reflect → Curate → Apply）

**① Reflect**（每道**失败**的题一次调用，成功的题不 reflect，对应论文 eq. 6）。输入是白名单化的上下文，输出 `CandidateMemory(trigger, lesson, failure_mode, scope_hint, topic, action_hint, target_id)`。

`topic` 是模型在**输出侧**命名的，不是数据集标签 —— Appendix E.1 的输入清单里没有 topic。早期版本把它当输入，凭空制造了对逐题 topic 标注的依赖（commit `4015c8b`）。传入 `existing_topics` 只是为了让模型**复用**已有名字而不是造同义词。

解析器保守：四个字段任一缺失，或显式 `ACTION: NONE`，一律返回 `None` 而非半成品候选。

**② Curate**（每 batch 结束时两个 curator 各一次调用）。`TopicCurator` 只看某一个 topic 的既有 skill 与本 batch 属于该 topic 的候选；`GeneralCurator` 看 general 层既有 skill 与本 batch 的**全部**候选（跨 topic 的模式，单 topic 视角看不出来）。输出 `ADD | MERGE | REVISE | DELETE | SKIP`。

payload 上的 layer 字段由调用方 curator **无条件覆写** —— 模型对"这条属于哪一层"的意见从不采信，只采信它写的内容。

**③ Apply**（`store.apply()`）两阶段：先应用 `DELETE / MERGE / REVISE / SKIP`，再把 `ADD` 对**阶段一之后**的占用量做容量检查。这不是优化而是必需：curator 常给"删掉那条弱的、加一条更好的"，若按列表顺序单遍执行，满了的 harness 只会一直增长、永远不会换血。单条 edit 抛任何异常都被**逐条捕获**。

## 3.4 增长上界

| 机制 | 位置 |
|---|---|
| `Caps.general = 5` / `Caps.per_topic = 5` | `store.capacity_full`，**代码强制**，不只是写在 prompt 里 |
| 单条 skill 长度上限 | `guard`：lesson ≤ 60 词、整条 ≤ 200 词 |
| 注入预算 | `selector.apply_budget`（§3.2） |
| **general 必须跨 ≥2 道题** | `store.py:405`，reject reason `general_needs_two_problems` |
| **curator 只能改自己那一层** | `store.py:379`，reject reason `wrong_layer` |

后两条都是**真实运行中观察到模型违反 prompt 指令之后**补的，见 §6.2。

## 3.5 防泄漏设计（四道结构性关卡）

**① 类型层面的白名单。** `build_reflect_context(*, existing_topics, final_answer_given, outcome, verifier_feedback, tool_errors, reasoning_excerpt, round_count, related_skills)` —— 全部是逐个具名的关键字参数，**没有 question 参数、没有答案参数、没有 `**kwargs`、没有 dict 逃生口**。想把题面或答案塞进 skill，在这个函数的签名上就做不到。

**② GT 通道清洗。** AlphaApollo 自己的验证工具会把 `Matches ground truth: True` 直接写进工具返回文本再喂回 policy。`sanitize_feedback()` 按**整行删除**这个通道，且在 `build_reflect_context` **内部**执行而不是留给下游记得调用。另外 `extract_result` 只从 `tool_payload` 的 `stderr` / `run_status` 取 tool error，**绝不取 `stdout` 或 `raw_observation`** —— 因为那行字正是打印在 stdout 里的。结构性排除优于事后正则清洗（commit `9e1b554`）。

**③ Guard 拒绝**（`guard.py` 是本包中**唯一**允许接触 `ground_truth` 的模块，且只用于拒绝，比对完即丢弃，绝不写回任何返回值 / 日志 / 异常消息）。六个固定 reject reason：`empty_section` / `non_english` / `lesson_too_long` / `skill_too_long` / `question_overlap`（与题面 8-gram 重合）/ `answer_leak`。

`answer_leak` 的判定经过一次**实测修正**：原规则是"skill 里出现任何等于某个 GT 的数字就算泄漏"。对真实 adaptation pool 量过，AIME 答案有 **4%** 恰好就是 skill 自然会写的整数界（4 / 10 / 20 / 50…），该规则会**误拒 21.7%–27.9%** 的枚举类 skill —— 正好是"先枚举一个小范围"这类最想学到的 skill。现改为要求数字**同时**出现在断言线索词附近；纯数字巧合**保留但打** `numeric_coincidence` 标记进 `guard_note`，使这个取舍可审计（commit `1ff380f`）。

**④ 硬失败：GT 工具调用。** `assert_no_gt_tool_call()` 每轮检查一次 action 文本，一旦出现 `<informalmath_verify>` 就抛 `LeakageError` —— 这是 `run_stream` 中**唯一不被吞掉**的异常。普通单题异常降级成全零结果继续跑，而泄漏一旦发生，这条 rollout 下游的一切都不可信，正确反应只有停机。

**实测验证**：本次 run 的 `store/skills/*.md` 与 `pool/pool.jsonl` 中，`ground_truth` 出现 0 次，`\boxed` 与 "answer is" 出现 0 次。

---

# 4. 三组核心实验

## 4.1 三条臂与共享设置

| arm | 跨问题机制 | 注入内容 |
|---|---|---|
| **Baseline** | 无 | 中性 system prompt（`NEUTRAL_SYSTEM_PROMPT`） |
| **Raw Experience** | 过去轨迹的直接摘要池 | 检索到的原始经验摘要，**同一 token 预算** |
| **Evo-Harness** | Task A/B 的 skill 编译闭环 | 选中的 general + topic skills |

除上表外**一切相同**：同一份 `examples/configs/harness_base.yaml`，6 个 overlay 只覆盖使其成为该臂的那几项，因此 `diff harness_adapt_baseline.yaml harness_adapt_evo.yaml` 显示的就是实验操纵本身。共享项：模型、题序（`seed: 1234`，`batch_size: 8`）、题内演化轮（2）、工具（Python 开启，`max_steps: 4`）、生成参数、注入预算。

**三个刻意设置的公平性保障：**

- **三臂都发非空 system prompt。** 上游 `utils/agent.py:37` 是 `if self.system_prompt:` —— 空串会导致**根本不发 system message**。若 baseline 无 system message 而 evo 有，比较的就是"有没有 system 角色"而不是"注入内容是什么"。冷启动时三臂发出**完全相同**的中性 prompt。
- **Raw 是认真的对手，不是稻草人。** 与 Evo **逐字共用**同一份注入预算，因为实验问的是"编译出的 skill 是否胜过原始经验"，两臂必须在**注入什么**上不同、绝不能在**注入多少**上不同。Raw 还**同时存成功与失败**（Evo 只从失败 Reflect），这对它有利。
- **`max_workers` 只影响吞吐，不影响协议。** batch 边界与并发度完全无关，且它被**刻意排除**在 resume fingerprint 之外，以便在重试之间调整而不使已完成的批次失效。

## 4.2 运行命令

```bash
export OPENAI_API_KEY=<your key>

nohup ./scripts/run_experiments.sh > run.log 2>&1 &
disown
```

`nohup` + `disown` **不是装饰**：两次诊断跑曾在半途被杀，不是机器问题（进程占 491 MB，机器 24 GB），而是启动它的 agent 会话在拆除后台任务。多小时的跑必须归属系统，而不是归属启动它的那个 shell。

脚本按 adaptation（三臂并行）→ 冻结状态 → held-out 的顺序跑完六个 run，**中断后重跑同一条命令即可续**。它在花掉第一次调用之前就拒绝三种错误启动：没有 `OPENAI_API_KEY`、held-out 跑在 adaptation 产物不存在时、未知的 phase。

```bash
./scripts/run_experiments.sh adapt          # 只跑 adaptation
./scripts/run_experiments.sh heldout        # 只跑 held-out（需 adaptation 已完成）
PARALLEL=0 ./scripts/run_experiments.sh     # 三臂串行
ARMS="baseline evo" ./scripts/run_experiments.sh

# 出结果
python -m alphaapollo.core.harness.analysis --root ./outputs/harness --out_dir ./outputs/harness/report
python -m alphaapollo.core.harness.export --store_root ./outputs/harness/adapt-evo/store --out_dir ./outputs/harness/export-evo
```

单独跑一个 run：

```bash
python -m alphaapollo.workflows.evo --config examples/configs/harness_adapt_evo.yaml
```

> ⚠️ flag 是 `--config`，**不是 `--config_path`**。`parse_known_args` 不会拒绝未知参数，它会把 `--config_path` 变成一条没人读的 override，于是 `--config` 取默认值、跑的是上游的 `evolving_main` —— 25 秒跑完 30 道题、零次模型调用、最后一行还写着 "Finished run"。这个坑吞掉过一整轮诊断（commit `18fc851`）。

## 4.3 主结果

| arm | adaptation Pass@1 (round 0) | adaptation 最终 | held-out Pass@1 | **held-out 最终** |
|---|---|---|---|---|
| Baseline | 22.2% (32/144) | 21.5% (31/144) | 13.3% (4/30) | 10.0% (3/30) |
| Raw Experience | 16.7% (24/144) | 21.5% (31/144) | 23.3% (7/30) | **23.3% (7/30)** |
| **Evo-Harness** | 23.6% (34/144) | **24.3% (35/144)** | 13.3% (4/30) | 16.7% (5/30) |

**配对 McNemar 精确检验**（同题比较，括号内是两个方向的不一致样本数）：

| 比较 | adaptation (n=144) | held-out (n=30) |
|---|---|---|
| baseline vs raw | p = 1.000 (9 / 9) | p = 0.125 (0 / 4) |
| baseline vs evo | p = 0.481 (7 / 11) | p = 0.625 (1 / 3) |
| raw vs evo | p = 0.503 (8 / 12) | p = 0.625 (3 / 1) |

**一项都不显著。** Evo 高出 baseline 的 2.8 个百分点等于 4 道题，而本系统实测噪声底是 ±2 题 / 20 题 ≈ ±3.6 个百分点（同配置重跑两次，总分相同但**有 2 道题翻转**）。这个优势完全落在噪声里。

**held-out 上 Raw 反而赢了 Evo**（23.3% vs 16.7%），只有 4 对不一致样本。这同样是噪声，但它是必须主动解释的数字 —— §6.4 给出了一个与"原始经验有用"无关的具体来源。

## 4.4 随时间的变化（窗口 = 25）

| 窗口 | 0-24 | 25-49 | 50-74 | 75-99 | 100-124 | 125-143 |
|---|---|---|---|---|---|---|
| baseline | 24.0% | 32.0% | 16.0% | 28.0% | 20.0% | 5.3% |
| raw | 20.0% | 28.0% | 24.0% | 32.0% | 8.0% | 15.8% |
| **evo** | 28.0% | 32.0% | 16.0% | 36.0% | 16.0% | 15.8% |

**没有学习曲线。** 如果 skill 编译在累积，evo 的后段窗口应当高于前段 —— 它没有。而且三条曲线同升同降，说明波动由**题目难度顺序**驱动而非由 arm 驱动。这是本实验对自身假设最直接的负面证据，因此紧跟主结果给出而不是埋在末尾。

## 4.5 per-topic（最终正确率）

| topic | n | adaptation base / raw / **evo** | n | held-out base / raw / **evo** |
|---|---|---|---|---|
| algebra | 30 | 36.7% / 36.7% / **53.3%** | 9 | 0.0% / 22.2% / 11.1% |
| number_theory | 32 | 28.1% / 28.1% / **21.9%** | 5 | 40.0% / 40.0% / 20.0% |
| combinatorics | 33 | 12.1% / 9.1% / **15.2%** | 9 | 11.1% / 11.1% / 11.1% |
| geometry | 49 | 14.3% / 16.3% / **14.3%** | 7 | 0.0% / 28.6% / 28.6% |

唯一超出噪声量级的移动是 **adaptation / algebra：36.7% → 53.3%（+16.7pp, n=30）**。它恰好是 `sk_0001` 所在的领域，而 `sk_0001` 是全场效用最高（50%）、且**唯一从未被编辑过**的 skill。这构成一个连贯的假设 ——「未漂移的具体 skill 才产生迁移」—— 但它是**事后**在 n=30 上观察到的相关，只能作为下一步的研究方向，不能作为结论。

held-out 上 geometry 的 0% → 28.6% 只有 7 题、2 道题之差，不承载信息。**本表须带 §1.3 的标注器偏差 caveat。**

## 4.6 harness 规模与增长

| 跑完第 N 题 | 7 | 31 | 55 | 79 | 103 | 127 | 143 |
|---|---|---|---|---|---|---|---|
| general | 1 | 3 | **5** | 5 | 5 | 5 | 5 |
| topic | 5 | 9 | 9 | 9 | 10 | 11 | 11 |
| 总 token | 398 | 768 | 913 | 891 | 912 | 977 | **1003** |
| 每条均长 | 66 | 64 | 65 | 64 | 61 | 61 | 63 |

**增长控制按设计工作**：general 层第 55 题起顶到 cap 就不再增长；144 题只累积到 1003 token；每条长度稳定在 60 余 token 而没有膨胀。

**curator 决策：209 次，接受 89、拒绝 120。**

| op | 提出 | 接受 | | 拒绝原因 | 次数 |
|---|---|---|---|---|---|
| ADD | 17 | 16 | | `skipped` | 70 |
| MERGE | 105 | 56 | | **`wrong_layer`** | **49** |
| REVISE | 17 | 17 | | `capacity_full` | 1 |
| SKIP | 70 | — | | | |

`wrong_layer` 占全部决策的 **23%**。闸门本身是对的，但被拒的决策消耗了完整的一次模型调用 —— **这是 prompt 契约的结构性失败，不是质量判决**。

## 4.7 开销（solver 与管理分开报）

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

三点：

1. **evo 多花 20.7% 的管理调用，总 token 反而比 baseline 少 12.8%** —— solver 输出从 1.66M 降到 1.06M（−36%），注入 skill 让 rollout 变短了。"Evo 更贵"在本数据上只对**调用次数**成立（+5.5%），对 token 不成立。
2. **预算上限 800，evo 实际只用到 185（23%）；Raw 注入量是它的 2.1 倍，held-out 上还赢了。** evo 的劣势**不能**归因于"注入得不够多"。
3. **两臂各有 8 题零注入，正好是 batch 0 的 p_0–p_7**（harness 当时为空）。这是"一道题不可能被自己产生的 skill 影响"的直接数据证据。

**开销口径主动选了对自己不利的算法。** `raw_summarizer` 单独成 role、不复用 `summarizer`（后者已命名上游**题内**的求解侧助手，baseline 本来就有）。把 Raw 的每题摘要折进去，会让 Raw 在成本报告里显得几乎不花钱，而 Evo 的 reflect/curate 却被正确计为管理开销 —— 那是一个**会错误地削弱本项目自身论点**的数字（commit `7b4336d`）。

`calls/unscoped` 是一个公开的自检列，**六个 run 全为 0**。它源于一个真实缺陷：`role_scope` 用 `ContextVar`，而 `ThreadPoolExecutor` **不传播** context，上游扇出到线程池的 verifier 调用曾整批落进 `unscoped`（第一次干净运行里 38 次 solver 侧调用有 9 次落在那里），要求上报的 solver-vs-management 拆分直接是错的（commit `62b024b`）。

## 4.8 skill 使用频次

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

- **`sk_0006` 被注入到 82% 的题上，命中率 22.9% —— 等于 base rate。** 一条占满注入槽的 no-op。原因见 §5.2。
- **`sk_0004` 一次都没被选中**，却始终占着 `isosceles_triangle_counting` 的槽位到最后。cap 限制了总量，但**没有任何淘汰压力**把无用的 skill 挤出去。

## 4.9 关键日志

四类产物，各取一条**本次 Task C 正式运行**的真实记录。

**① `<run_dir>/metrics.jsonl` —— 逐题一行**

```json
{"step": 84, "adapt/pass1_round0": 1, "adapt/pass_final": 1, "adapt/error": 0, "topic": "number_theory"}
```

`step` 是题号；`pass1_round0` 与 `pass_final` 分开记，两者不等就意味着题内演化轮改变了结局（§6.4 的整张表都建在这个差上）；`error` 区分"跑挂"与"答错"，两者的 `pass_final` 都是 0。每 batch 另有一行累积开销：`calls/<role>`、`tokens/<role>_{in,out}`。

**② `store/selection_log.jsonl`（evo）/ `pool/selection_log.jsonl`（raw）—— 逐题注入了什么**

```json
{"ts": "2026-09-16T05:37:31Z", "problem_idx": 84, "topic": "number_theory",
 "skill_ids": ["sk_0001", "sk_0006", "sk_0008"], "n_tokens": 188, "success": true}
```

这是"哪些 skill 被用在哪道题上、花了多少 context"的唯一来源，§4.7 的注入 token 表与 §4.8 的使用频次表都从它算出。

> **一个产物口径上的坑**：held-out 的这个文件有 **174 行不是 30 行** —— `run_experiments.sh` 把 adaptation 的 store 整个拷过去，那 144 行选择记录跟着带了过来，held-out 只是往后追加。取 `[-30:]`，直接求和会把 adaptation 的开销算进 held-out。

**③ `store/harness_log.jsonl` —— 每一条 curator 决策（接受与拒绝都记）**

```json
{"ts": "2026-09-16T06:19:28Z", "skill_id": "sk_0012", "accepted": true,
 "reject_reason": null, "guard_note": null, "problem_idx": 16, "batch": 16,
 "op": "MERGE", "actor": "topic_curator",
 "reason": "Candidate 3 reinforces the handling of divisibility conditions on consecutive numbers...",
 "source_problems": [132],
 "candidate": {"trigger": "When applying multiple divisibility conditions to consecutive numbers.",
               "lesson": "- Explicitly verify all intermediate steps and edge cases. ...",
               "failure_mode": "Assuming the correctness of unverified numerical outcomes.",
               "scope_hint": "topic", "topic": "combinatorial_constraints",
               "evidence": ["p_132"]}}
```

被拒的决策同样留痕，reject reason 是固定字符串：

```json
{"problem_idx": 2, "batch": 2, "op": "MERGE", "actor": "general_curator",
 "skill_id": "sk_0005", "accepted": false, "reject_reason": "wrong_layer"}
```

> **字段命名的一处硬伤**：harness edit 行里的 `problem_idx` 存的其实是 **batch 索引**（上表两条都是 `problem_idx == batch`）。这是刻意的 —— 一条 edit 由整个 batch 的候选共同产生，绑到某一道题上是假精确 —— 但字段名没跟着改。真正的来源题号在 `source_problems` 与 `candidate.evidence` 里。读日志时不要把它当题号。

**④ `<run_dir>/*.log` —— run 级健康状况**

```
RUN                  DONE  PASS@1   FINAL ERRORS  BATCHES UPDATED
adapt-baseline        144   22.2%   21.5%      0       18 392m ago
adapt-raw             144   16.7%   21.5%      0       18 421m ago
adapt-evo             144   23.6%   24.3%      0       18 483m ago
heldout-baseline       30   13.3%   10.0%      0        4 368m ago
heldout-raw            30   23.3%   23.3%      0        4 368m ago
heldout-evo            30   13.3%   16.7%      0        4 367m ago
```

（`./scripts/status.sh`，它**只读磁盘产物**、不碰运行中的终端，因此跑中查、跑完查、写入过程中查都安全 —— 半行 JSON 会被跳过而不是报错。）

**六个 run 的日志级健康检查**：`WARNING` **0 条**、harness 层异常 **0 次**、`ERRORS` 列全 0。日志里确实出现 Traceback（12 / 14 / 6 / 11 / 4 / 10 条），但全部是 **Python 工具沙箱把执行错误回传给模型、模型读了报错自己改对**——工具按设计工作的证据，不是故障：

```
<tool_response>{"result": "Traceback ... ValueError: Both input arrays must be
(arrays of) 3-dimensional vectors, but they are 2 and 2 dimensional instead.",
 "status": "failed", "returncode": 1}</tool_response>
→ 模型下一步：「np.cross 需要 3D 向量……改用 2×2 行列式」
```

`heldout-raw.log` 里另有 4 条 `AttributeError`，路径写着 `C:\Users\user\AppData\...\Python310` —— 在一台 MacBook 上。那是**模型自己编造**的 traceback（复述训练数据里见过的报错），不是真实执行。这类幻觉不会污染结果（工具输出与模型自述在轨迹里是分开的字段），但值得记一笔。

对照 §7 的烟囱测试：那次运行 25% 的题死于 `APIConnectionError` 而整条 run 仍然 `exit=0`。正式跑之所以能做到丢题 0，靠的是 `NO_PROXY` 修复 + 首 batch 断路器 + 每题显式释放 client/env（两次真实崩溃后补的，见 §7.2）。

---

# 5. 代表性 skill 演化记录（来自本次 Task C 正式运行）

## 5.1 一条**没有**漂移的 skill：`sk_0001`

```
b0  ADD  topic_curator  evidence=[p_0]
    trigger: when solving problems involving polynomial factorization with integer coefficients
    lesson : - Identify necessary conditions for factorization.
             - Translate constraints into equations involving roots.
             - Count valid root pairs that satisfy both sum and product conditions.
    avoid  : providing answers without showing the logical steps or reasoning process.
```

**此后 144 题里再没有被编辑过一次。** 它是全场唯一如此的 skill，也是效用最高的（注入 18 次、9 次做对 = 50%，其余 skill 全部落在 12.5%–27.3% 这个与 base rate 无法区分的带里）。它的 trigger 具体到可以判断"这道题是不是它管的"，lesson 是三个可照做的动作。

## 5.2 一条漂移到面目全非的 skill：`sk_0006`（10 次接受编辑）

| batch | op | trigger |
|---|---|---|
| b0 | ADD | When solving a problem, ensure all possible cases are considered… |
| b3 | REVISE | When multiple conditions or constraints are present in a problem. |
| b6 | REVISE | When multiple constraints or conditions are present in a problem. |
| b14 | REVISE | When multiple constraints are present in a problem. |
| b15 | REVISE | When multiple constraints or relationships exist in a problem. |
| **b16** | **MERGE** | **When a circle is tangent to multiple sides of a figure and intersects a diagonal.** |

两个方向的退化同时发生：前半程 trigger 被反复"泛化"成对几乎任何 AIME 题都成立的空话；末尾一次 MERGE 又把一条**圆的切线**的具体几何 lesson 并了进来。最终导出的 `harness.md` 里，这条 skill 标题是泛化的 `when-solving-a-problem-ensure-all-possible-cases`，body 讲的却是坐标几何切线 —— **trigger 与 lesson 已经不描述同一件事**。`sk_0013`、`sk_0010` 同样。

后果可测：它被注入 **118/144 题（82%）**，命中率 22.9% = base rate。**选择器并非选错，而是在漂移后的 trigger 上无从区分。**

**根因**：MERGE 与 REVISE 整条改写三个字段，**没有任何机制校验改写后 trigger 仍然描述 lesson**。两个现成的修法（均未实现）：(a) 改写后做 trigger–lesson 一致性校验，不一致则拒绝；(b) 记录编辑次数，超阈值就冻结或强制分裂。

## 5.3 跨层重复仍未解决

p_68 一次注入的 5 条 skill 里有 4 条（`sk_0006` / `sk_0012` / `sk_0013` / `sk_0014`）说的是同一件事 ——"系统枚举、逐条校验、不要假设"—— 311 token 里大部分是重复内容。`064e799` 的"general 必须跨 ≥2 题"拦住了最明显的一类，但**它拦的是来源，不是内容**。

## 5.4 一个诱人但被推翻的解释

"编辑次数越多质量越差"本可以直接用来支持漂移论点，但整体只有很弱的支持（≤4 次编辑的 skill 平均效用 24.9%，≥5 次的 20.3%），而且差距几乎完全由 `sk_0001` 一个点撑着，`sk_0008` 被编辑 12 次仍有 27.0%。**漂移的结论应当靠 §5.2 的文本证据来立，不能靠这个相关性。**

---

# 6. 迁移案例

逐轨迹分析见 [`docs/findings/transfer-cases.md`](docs/findings/transfer-cases.md)。

> **注入文本需要重建**：轨迹文件里**没有**存 system message（skill 是逐题写进 system 通道的，而 `full_config` 存的是配置里的空串）。要知道某题当时看到的 skill 原文，得拿 `selection_log.jsonl` 的 `skill_ids`，再把 `harness_log.jsonl` 里该 batch 之前所有 accepted 的编辑重放一遍。**这是一个应当修掉的可审计性缺口。**

## 6.1 正向 · p_84

number_theory，注入 `sk_0001` / `sk_0006` / `sk_0008`（188 token）。baseline 答 37、evo 答 239（对）。

| | baseline | evo |
|---|---|---|
| 搜索上界 | `range(1, 100)` | `range(1, 1000)` |
| 工具返回 | `"78\n"` | `"[78, 161]\n"` |
| 下一步 | **无视工具返回**，自称"找到 n = 6, 8, 9, 14"，答 37 | 读取返回值，手算验证 $3081^2 \bmod 83 = 17$、$13113^2 \bmod 166 = 17$，答 239 |

决定对错的是**搜索上界**（真解 78 和 161，上界 100 必漏 161）；baseline 还叠加了第二个独立故障 —— 凭空编造工具输出。`sk_0006` 的 "no omissions in case analysis" 与放宽上界之间有合理联系，`sk_0001` 的 failure_mode 与"evo 逐个手算验证"之间也有。

**但这是 temperature = 0.7 的单次采样，因果归因未经证实。** 要坐实需要在不注入条件下重复采样 p_84，本项目没有做。本案例正确的写法是"行为差异具体且可复核，因果归因是未证实的假设"。

## 6.2 负向 · p_105

"求所有三位回文数的算术平均"，GT = 550。baseline 两轮都答 550（对）；evo **round 0 答对 550**，round 1 改成 549.5（错）。

1. round 0，evo 答 **550**，推理正确，但这条最终消息只有 **756 字符**（evo 全局均值 2353）。
2. round 0，verifier 判**错**，并主动给出自己的答案：*"The correct mean is 549.5, not 550."*
3. round 1，policy **接受了 verifier 的数字**，并为它倒推出一套错误算术（总和算成 49455），答 **549.5**。
4. round 1，verifier **自己翻供**：*"The correct total sum is 49500, not 49455, which leads to 550, not 549.5."* —— `pass_final` 取最后一轮，来不及了。

注入的三条 skill **没有**把模型带向错误的**数学方法**；它们施加的是"再验证一遍"的**风格压力**，作用在一道不需要验证的题上。这是 context interference，但机制是**风格层面**的，不是知识层面的。

## 6.3 系统性 · 这条链不是孤例

定义 **loss** = round 0 对、最终错：

| arm | loss | gain | 净 | 被毁题的 round-0 消息均长 | 全部题均长 |
|---|---|---|---|---|---|
| baseline | **8** | 7 | −1 | 1827 | 2575 |
| raw | **1** | 8 | **+7** | 766 | 2491 |
| evo | **6** | 7 | +1 | 1302 | 2353 |

verifier 误拒率（分母是 policy **确实答对**的轮次）：

| arm | policy 对 / verifier 对 | policy 对 / verifier 错 | **误拒率** |
|---|---|---|---|
| baseline | 42 | 21 | **33%** |
| raw | 41 | 14 | **25%** |
| evo | 47 | 22 | **32%** |

## 6.4 三条结论

1. **题内演化轮在 baseline 上是净负的**（毁掉 8 条、救回 7 条）。这是上游 AlphaApollo 自身的行为，与本项目无关，但它是三臂共同的噪声源。
2. **Raw 臂在 adaptation 上追平 baseline，靠的不是多做对题，而是少毁题。** 它 round-0 比 baseline 低 5.5 个百分点，最终却打平 —— 全部差距来自 loss 从 8 降到 1。**任何把 Raw 的成绩读成"原始经验也有用"的写法都是错的**：它的收益发生在"防止答案被改坏"这一环，不在解题能力上。
3. 三臂一致地，**被毁掉的题其 round-0 最终消息明显更短**。verifier 按**看得见的推理**打分，一条简短、断言式的正确答案会被判成"没有给出推理"，verifier 随即给出自己（常常是错的）数字，policy 在下一轮采纳它。

**这个 loss 数（8 / 1 / 6）大于三臂之间的全部差距，而它由"最终消息有多长"这样一个与 skill 质量无关的变量驱动。解读任何 arm 间差异之前必须先看这张表。**

一个**不涉及 skill 机制**就能拿到的改进（已识别、未实现）：让 `pass_final` 取"所有轮里被 verifier 判对过的答案"而非"最后一轮的答案"。按上表这会让 baseline +8、evo +6、raw +1，对三臂都公平。没有实现是因为它要动上游的题内逻辑，而任务书明确说不需要动 —— 这是一个**有意识的范围决定**，写在这里以便 reviewer 判断是否同意。

---

# 7. 对 null result 的分析

任务书明确写了"Evo-Harness 赢过 baseline 不是及格线，但 null result 必须被分析"。按它列出的六个方向：

1. **任务相似度。** AIME 跨年之间共享的是**领域**而非**可复用的解题程序**。harness 里真正具体的 skill（如 `sk_0001`）只在 18/144 题上被选中，82% 的题拿到的是通用套话。这可能是最根本的限制：AIME 的设计初衷就是每题需要一个新想法，而 Evo-Harness 论文的场景有可复用的操作流程。
2. **skill 质量。** §5.2 的语义漂移。这是**可修的工程缺陷**，不是方法本身的问题。
3. **选择错误。** `sk_0006` 注入率 82% 而效用等于 base rate；`sk_0004` 注入率 0%。选择器不是选错了，而是**在漂移后的 trigger 上无法区分** —— 修好 (2) 才谈得上评价 (3)。
4. **context 干扰。** 确有其事，但机制出乎意料：不是知识层面的误导，而是 §6.2 的**风格层面**干扰。注入使 evo 把验证前置、最终消息变短（中位数 2280 vs baseline 2596），而短消息更容易被 verifier 误判，进而在下一轮被改坏。
5. **verifier 反馈质量 —— 比预想的严重得多。** policy 答对时 verifier 否定它的比例是 33% / 25% / 32%。编译 skill 所依赖的成败标签本身就带着这个量级的噪声；更糟的是这噪声不只污染 skill，它还经由题内演化轮**直接破坏最终答案**。
6. **token 开销。** **不是**瓶颈，可以排除：evo 只用掉 800 预算中的 185 token，总 token 还比 baseline 少 12.8%。

**结论**：本实验证明了**机制是闭环的**，但**没有证明编译出的 skill 有用**。最大的单一嫌疑并不是"跨题迁移在 AIME 上不成立"这个方法论结论 —— 而是在能检验它之前，两个工程缺陷（语义漂移、23% 的 `wrong_layer` 浪费）与一个测量缺陷（verifier 噪声经由题内演化轮破坏最终答案，且与消息长度耦合）已经把信噪比压到了差异无法分辨的水平。**下一次实验该修的是这三项，而不是换一个更大的模型。**

---

# 8. 失败尝试与已知限制

## 8.1 失败的尝试（都已回滚或取代）

| 尝试 | 为什么放弃 |
|---|---|
| 确定性词重叠 + 硬 topic 过滤做 Select | 偏离论文，且隐式要求逐题 topic 标注（`f591906`） |
| 把 topic 作为 Reflect 的**输入** | 与 Appendix E.1 相反，凭空制造 topic 标注依赖（`4015c8b`） |
| topic 交错题流，缩短 mint-to-reuse 延迟 | AIME 一年 30 题内部本就四个 topic 混排；且交错需要**运行前**的 topic 标签，而那正是方法后来被证明不需要的依赖（`e756835`） |
| `answer_leak` 用裸数字相等 | 实测误拒 21.7–27.9% 的枚举类 skill（`1ff380f`） |
| batch size 16（论文默认） | 149 题只给约 9 个更新点，看不出增长趋势 |
| `pkgutil` + 子串搜索做边界守卫 | 同时漏报 `__init__.py` 与误报 docstring（`35a811b`） |
| 把 Raw 臂的摘要调用记在 `summarizer` role 下 | 会让 Raw 看起来零管理开销（`7b4336d`） |

## 8.2 只有真机运行才暴露的缺陷

`2aedcb1`（`data_source` KeyError）、`9e1b554`（虚构的 `step_outputs` schema）、`70e77b6`（从空轨迹编译 skill）、`62b024b`（线程池吞 role）、`064e799` / `e62ea41`（模型忽略 prompt 约束）、`5564906` / `5776191`（每题不释放 client/env，第一次 adaptation 跑在第 ~104/144 题死于 `OSError: Too many open files`）—— 这些都是在测试套件已有几百个通过用例时，**第一次接上真实 API** 才暴露的。

两种共同模式：

1. **手写 fixture 与手写实现共享盲点**。补救是 `tests/harness/fixtures/real_problem_payload.json` —— 本包第一个取自真实执行的 fixture。
2. **韧性机制掩盖配置错误**。每一层降级都工作正常，结果是一个彻底坏掉的 run 安静地"完成"。补救是首 batch 断路器与 frozen-空状态前置检查 —— 它们都是**故意不降级**的地方。

**"降级"和"可观测"必须成对设计**，是本项目最有价值的工程结论之一。

## 8.3 已知限制

1. **只跑了单 seed 一轮。** 结论强度上最大的单一缺口 —— 三臂差距（2–3 个百分点）正好是单 seed 分辨不了的量级，论文自己是 3 次取平均。
2. **地板效应。** `qwen3-8b` 在 AIME 上约 20% 正确率 + held-out 仅 30 题（单题 = 3.3pp），在结构上就难以分辨 arm 差异。事前就预测到了，事后被证实：三个 held-out 配对检验的不一致样本数分别只有 4、4、4。
3. **seed 不给可复现性。** seed 确实注入了每一次请求，但 DashScope 的 seed 是 best-effort：两个生成参数完全相同的 run，round-0 结果仍会在个别题上不一致。**不要把"三臂 round-0 应逐题一致"写成断言** —— 那会是一个必然误报的 gate。
4. **语义漂移与 `wrong_layer` 浪费均未修复**，两者都在本次 run 中实际发生并可能压低了 evo 的表现。
5. **harness 无淘汰机制** —— 零注入的 skill 会永久占据槽位。
6. **topic 标注器有偏差**，per-topic 表须带 caveat（§1.3）。
7. **p_84 的因果归因未经检验** —— 缺少不注入条件下的重复采样对照。
8. **轨迹文件不记录注入的 system prompt**，复核需要重放编辑日志（§6 开头）。
9. **`run()` 本身没有被测试覆盖**：`run_stream` / `extract_result` / `assert_no_gt_tool_call` 都有覆盖，但 `run()` 这段配置装配代码没有（它已通过真实 API 跑通）。
10. **设计文档 `docs/design/cross-problem-skill-harness-design.md` 部分已过时**，早于"模型驱动 selector"与"模型命名 topic"两次改动。**代码与 git log 是权威。**
11. **Slides 尚未制作。**

---

# 9. 上游代码与许可证

| | |
|---|---|
| 上游代码 | [`tmlr-group/AlphaApollo`](https://github.com/tmlr-group/AlphaApollo)（内含 vendored `verl` / `verl-agent`） |
| 许可证 | **Apache License 2.0**，见仓库根目录 `LICENSE` 与 `Notice.txt`（Bytedance / NTU verl-agent / HKBU + TMLR Group） |
| 上游 README | `README.md` **原样保留，未作任何修改** |
| 方法参考 | Evo-Harness 论文（arXiv:2608.15071）与 [`A-EVO-Lab/a-evolve`](https://github.com/A-EVO-Lab/a-evolve) 的 `release/evo-harness` 分支。本项目**不 import、不依赖** a-evolve，仅作设计参考 |
| 新增代码 | 统一带 Apache-2.0 头，版权署 `Copyright 2026 TMLR Group` |

**改动边界：上游源码零修改。** 对比 fork 基线 `712a04d`，被修改（`M`）的上游文件**只有两个**：`.gitignore`（末尾追加反向排除，把小体积的 Task C 产物放进库）与 `pyproject.toml`（仅加一段 pytest 配置）。其余全部是新增（`A`）。

**没有任何一个上游的求解 / 环境 / verifier / 工具文件被触碰** —— 这是刻意的约束：Task C 要求三组实验共享**同一个冻结 solver**，改上游文件就无法论证"baseline 还是原来那个 baseline"。需要改上游行为的两处（开销统计、`extra_body` 注入）通过**类级 monkey-patch** 在运行期完成，`uninstall()` 后复原。

---

# 附录 · 对六项验收重点的自查

## A. 概念正确性

见 §2：四个层面的分离（代码包 / 生命周期 / 注入通道 / 内容），以及 AST import guard 的结构性强制（8 种违规写法 + 7 个反例）。上游 memory 包**未改一行**。

## B. Online protocol

**无未来信息**（§3 + `arms.py` 不变式 2）：每个有状态的 arm 在 `begin_batch()` **冻结**跨问题状态的一份副本，该 batch 内所有 `system_prompt_for()` **只读这份冻结副本**，只有 `end_batch()` 能写 live 状态。即使调用方绕过本类、在 batch 中途直接改 live store，也**无法**改变在飞 batch 的任何一道题看到的内容。测试：`test_evo_arm_prompt_changes_only_after_a_batch_boundary`、`test_mutating_the_live_store_after_begin_batch_does_not_change_this_batchs_prompt`。**运行时实证**：batch 0 的 8 题注入 token 恰为 0，且每条 skill 的 evidence 题号严格小于使用它的题号。

**无当前题答案**（§3.5）：四道关卡 —— 类型白名单（`build_reflect_context` 源码里**没有出现过** `ground_truth` 标识符）、GT 通道整行清洗、guard 拒绝、以及**唯一不被吞掉**的 `LeakageError`。

**测试集不泄漏**（§1.3 + §4.7）：年份切分、题面零重叠、held-out 用 `frozen: true`，**实证是 heldout-evo 的 reflect / curator 调用数为 0**。

## C. 闭环完整性

| 环节 | 证据 |
|---|---|
| 来自真实轨迹 | 209 条编辑各带 `source_problems` 与 `evidence: [p_xx]`；`test_evo_arm_does_not_compile_a_skill_from_a_problem_that_never_ran` 锁死"不从空轨迹编 skill" |
| 被组织 | 两层（5 general + 11 topic），四种算子，接受 89 / 拒绝 120，全部落盘 |
| 实际影响后续问题 | 144 题中 **136 题**被实际注入，平均 3.04 条 / 185 token；held-out 30 题全部被注入 |
| 只影响后面的题 | batch 级冻结 + evidence 题号严格小于使用题号 |
| 增长有界 | general 第 55 题顶到 cap 后不再增长（§4.6） |
| 可导出 | `harness.md` + `summary.json` + `evolution.jsonl`，已随仓库提交 |
| 失败不破坏 baseline | `test_store_apply_failure_mid_end_batch_leaves_the_harness_unchanged`；运行时**丢题 0** |

**薄弱处**：闭环在**机制**上完整，但 82% 的注入是一条效用等于 base rate 的 no-op。"skills 影响了后续问题"是可验证的事实；"skills **有益地**影响了后续问题"不是。

## D. 实验公平性

见 §4.1（单一 base config + 三个刻意的公平性保障）与 §4.7（按 role 分桶、`raw_summarizer` 单列、`calls/unscoped` 全 0、丢题数作为指标上报）。**缺口**：单 seed（§8.3.1）。

## E. 研究判断

主动做显著性检验并据此定措辞（§4.3）；把最有力的负面证据放在主结果之后而非末尾（§4.4）；负迁移追到真实机制并量化成 loss 表（§6）；推翻一个对自己有利但证据不足的解释（§5.4）；纠正自己先前的口径错误（verifier 误拒率分母）；对 Raw 的意外胜利给出具体解释而非回避（§6.4.2）；context growth 两面都报（cap 生效 / 无淘汰机制）；null result 按六方向逐条分析并明确排除其中一项（§7）；给出可执行的下一步而非"换更大的模型"；以及一个**有意识的范围决定**写出来供 reviewer 反对（§6.4 末）。

## F. 工程表达

| | |
|---|---|
| 配置 | 7 份 YAML（1 base + 6 overlay），每份都注明"为什么是这个值"及实测依据 |
| 日志 | 四类产物，样例与读法见 §4.9；含字段命名硬伤与口径坑的显式警告 |
| 可恢复 | `progress.json` + fingerprint 断点续跑；续跑继续同一个 wandb run |
| 测试 | **571 个**，`571 passed in 5.22s` |
| README | `README_HARNESS.md` 1112 行，12 节，含"每个数字的来源"附录表 |
| commit history | 101 个 purpose-scoped commit，消息写"为什么"（如 `docs: ten in-problem evolution rounds buy nothing at 3.9x the cost`） |
| 产物入库 | 63 个小产物（约 600KB）随仓库提交；43MB 轨迹全集与 wandb 排除，规则与理由写在 `.gitignore` 末尾 |
| 无密钥 / 权重 / 大数据 | 每个模型配置块都是 `api_key: ""`；已扫描全部入库产物确认无 key、无 ground truth |
