# AlphaApollo 跨问题 Skill Evolution 设计文档 (v2)

> 对应作业：`TMLR_MiniProject_EvoHarness.md`（Task A / B / C）
> 参照论文：Evo-Harness, *Context-to-Harness Skill Compilation for Self-Evolving Agents* (arXiv:2608.15071)
> 参照实现：`a-evolve` 的 `release/evo-harness` 分支（仅阅读参照，不 import）
> 日期：2026-09-11 ｜ 状态：v2，已吸收独立审阅意见 ｜ v1 存于 `.v1.md.bak`

**v2 相对 v1 的实质改动**：修正了"evo 路径不依赖 torch"的错误断言（§3.1）；修正了题内 memory 的层级与清空位置（§2）；发现并处置了 `informalmath_verify` 的 GT 泄漏通道（§5.4）；把 wall-clock 从"顺序执行"改为"batch 内并行、batch 间串行"（§3.3、§8）；补上 seed / 重复次数 / 指标定义 / system-message 对齐等实验公平性漏洞（§6）；把 feedback-grounding 从可选扩展提为主论述（§11）。

**v2.1（读完论文附录后补）**：Reflect 的输入补上 `related_skills`，输出补上 `action_hint`/`target_id`（附录 E.1）；Reflect prompt 补上 "Filter aggressively" 四条过滤规则（附录 E.1）；两个 curator 补上 generalizability test 与 general 层的 context-free 约束（附录 E.2/E.3）；案例展示改用附录 B 的五段式与其配对方法（§7.2）。

**论文引用的核实状态**：正文 §1–5 与 Table 1–4、附录 A–F、I 均已逐页读过。附录 F 三条默认值经原文确认 —— *"For harness selection, we use Claude Sonnet 4.5"*、*"we use a batch size of 16"*、*"maximum number of general skills and ... under each task-type topic to 5"*。附录 F 还确认了 skill 的四要素与本设计的 schema 完全同构：*"a **trigger** describing when it should be retrieved, a short actionable **rule or procedure**, optional **evidence** linking it to prior executions, and a **scope**"*。附录 I.2/I.3 确认论文用固定 seed 42、结果 *averaged over three runs*（与 §6.2③ 的 `test_times ≥ 3` 一致）。

---

## 1. 目标与范围

给 AlphaApollo 的 `informal_math_evolving` 环境加一个**跨问题 skill 演化闭环**：agent 做完一道题后，从本次轨迹与 verifier 反馈中提炼可复用的自然语言经验，存入外部 harness；处理后续题目时从 harness 中选出有限条注入 policy context。模型参数全程冻结。

**明确不做**：不改动 `run_problem` 内部的题内推理/验证逻辑；不做任何训练；不引入 embedding 模型或向量库；不追求超过 baseline（作业明示这不是验收条件）。

---

## 2. 核心概念边界：题内 memory vs 跨题 harness

验收重点第一条。必须在代码结构上成立，而不是靠文档解释。

**⚠️ 先修正一个常见误解**：AlphaApollo 的题内记忆是**两级**的，且都不在 `env.py` 里清空。

| 级别 | 对象 | 创建/清空位置 | 生命周期 | 注入槽位 |
| --- | --- | --- | --- | --- |
| 轮内 | `InformalMathEvolvingEnvironmentManager.memory`（`SimpleMemory` 等） | `env_manager.py:59-66` 创建，`:85` `reset(batch_size=...)` | **一个 evolving round** | `{memory_context}` |
| 题内 | `solution_memory`（`NDimensionalMemory`） | `evolving_main.py:488` 创建，`:489` reset | **一道题**（跨该题的所有 round） | `{previous_solutions}` |

`env.py:37` 的 `reset()` 只重置 `chat_history / done / turns / _score_list` 并调 `set_ground_truth`（`:52`）—— 它不清空任何 memory 对象。

在此之上，本项目新增第三级：

| 维度 | 轮内 memory | 题内 solution memory | **跨题 skill harness（新增）** |
| --- | --- | --- | --- |
| 代码位置 | `core/environments/memory/` | 同左 | `core/harness/`（新包，不 import 前者） |
| 存储内容 | 本轮的 obs/action | 本题历次解答全文 + 打分 | 抽象操作知识：`trigger`/`lesson`/`failure_mode` |
| 生命周期 | 一个 round | 一道题 | **整个 task stream，落盘持久化** |
| 载体 | 进程内 Python 对象 | 进程内 Python 对象 | `harness/skills/*.md` + jsonl 日志 |
| prompt 位置 | user prompt 槽位 | user prompt 槽位 | **system message**（见 §3.1） |
| 可影响的题 | 当前轮 | 当前题 | **只有后续 batch 的题** |
| 是否见 GT | 见 | 见 | **绝不见**（§5.4） |

三级在 prompt 中占据物理上不同的位置（system vs user），在代码中位于互不引用的两个包，在磁盘上是两套独立产物。"不会混淆"因此是结构性事实。README 需要把这张三级表作为核心插图。

---

## 3. 架构总览

### 3.1 已核实的事实（架构建立在这些之上）

✅ **成立**：

1. `utils/agent.py` 的 `Agent.system_prompt` 是实例属性，在 `get_action_from_gpt` 中每次调用时才读（`if self.system_prompt: messages.append({"role": "system", ...})`），且 `policy_model_cfg.system_prompt` 已在 config 中、默认空串。`run_problem` 内部**不重建 policy agent**（`:476` 从 runtime 取）。
   → **harness 注入 = 每个 batch 设置一次 `policy_agent.system_prompt`**。无需改 prompt 模板、`run_problem` 或 `env.py`。
2. `run_problem(problem_idx, current_problem, runtime)` 的 `runtime` 是普通 dict（`create_runtime_for_problem()` 构造，`:864`），调用方可自行组装。
3. `prompts/informal_math_evolving.py:383` 有 `SUMMARIZER_TEMPLATE`，baseline 自身在 `evolving_main.py:607` 就会创建 summarizer agent。
4. evo 路径**没有任何 wandb 集成**（`wandb` 只出现在 `rl_*/sft_*` 的 config、示例脚本和 `setup.py` 的依赖里）。

❌ **v1 的错误断言，已推翻**：

5. ~~"evo 路径不依赖 torch/vllm，可在 macOS 裸跑"~~ —— 错。`evolving_main.py:49` → `from verl.utils.reward_score.math import ...` 触发 `verl/protocol.py:29-32` 的 `import ray / tensordict / torch / torch.distributed`；`env_manager.py:26` 和 `environments/base.py:19` 也直接 `import torch`。
   **正确说法**：evo 路径依赖 `torch / ray / tensordict / pandas / transformers`，但**不依赖 vllm / flash-attn / CUDA**。torch-cpu、ray、tensordict 在 macOS arm64 上都有 wheel，所以仍然可以在 Mac 上跑，但**不能直接 `pip install -e .`** —— `setup.py:28-55` 的 `install_requires` 含 `qwen-vl-utils[decord]`（decord 无 macOS-arm64 wheel）、`rdkit`、`vllm` 等。
   **Day 1 正确做法**：`pip install -e . --no-deps`，再手工装 `torch(cpu) ray tensordict pandas transformers omegaconf fire openai datasets pyarrow wandb`；或完全不 install，只设 `PYTHONPATH=$PROJECT_ROOT/alphaapollo/core/generation`（`examples/evo/run_evo_informal_math.sh:9` 就是这么做的）。

### 3.2 目录结构

```
alphaapollo/core/harness/           # 新增，与 core/environments/memory/ 零耦合
  schema.py        # Skill / CandidateMemory / SkillEdit + 序列化
  store.py         # SkillStore：持久化、分层选择、预算、两阶段 apply
  render.py        # skills -> system message 文本；token 计数
  reflect.py       # Reflect(context) -> CandidateMemory | None
  evolver.py       # TopicCurator / GeneralCurator -> [SkillEdit]
  guard.py         # validate_skill()：防泄漏 + 语言 + 长度 + 格式
  arms.py          # CrossProblemArm 抽象 + 三个实现
  accounting.py    # Agent.get_action_from_gpt 的记账 patch
  tracker.py       # wandb + jsonl 双写
  loader.py        # 带 topic/year 的问题加载（不复用 load_informal_math_data）

alphaapollo/core/generation/evolving/
  evolving_harness_main.py          # 新 driver：batch 间串行 / batch 内并行

alphaapollo/data_preprocess/prepare_harness_stream.py
examples/configs/harness_{baseline,raw_experience,evo}.yaml
tests/harness/                      # 纯单测
scripts/run_{baseline,raw,evo}.sh, scripts/run_tests.sh
```

**对上游已有文件的改动：0 个。** 记账通过 monkey-patch 实现（§6.2），数据加载自写（见下）。

⚠️ **不能复用 `load_informal_math_data`**：`utils/dataset_loader.py:116-118` 只保留 `question / ground_truth / gt_traj`，会把 topic 和 year 字段丢掉。自写 20 行 loader；同时注意 `run_problem` 在 `:563` 和 `:715` **无条件**读 `current_problem["gt_traj"]`，新 parquet 必须带该字段（可为空串）。

### 3.3 执行协议：batch 间串行，batch 内并行

```
for i, batch in enumerate(batches(problems, size=B)):      # batch 间严格串行
    H_i = store.snapshot()                                 # 本 batch 全程冻结
    ctx = render(H_i)                                      # 同一 batch 共用同一段注入文本
    policy_agent.system_prompt = ctx                       # 设一次，batch 内不变
    results = parallel_map(run_problem, batch, workers=B)  # ← batch 内并行，协议合法
    R = [reflect(c) for c in results if failed(c)]         # 每失败题 1 次管理调用
    edits  = [topic_curator(H_i, R, tp) for tp in topics(R)]   # ≤4 次
    edits += general_curator(H_i, R)                            # 1 次
    store.apply(edits)                                     # 0 次 → H_{i+1}
```

**为什么 batch 内并行不违反在线协议**：同一 batch 的所有题都用同一个冻结的 `H_i`，彼此之间没有任何信息流动，第 t 题的结果不影响第 t+1 题的 context。这正是论文式 (2)(3)(4) 的语义（`S_{i,j} = Select(x_{i,j}, H_i, b)`，下标 `i` 是 batch），参考实现 `evolve_batch(..., batch_workers=4)` 也是这么做的。题目顺序完全固定，batch 只是更新时机的粒度。

**附带好处**：同一 batch 内注入文本相同，所以共享一个 `policy_agent` 是线程安全的 —— 不需要每题 new 一个 `Agent`（那会反复新建 `OpenAI` client，`utils/agent.py:27`）。

**代价与约束**：并发 = B × arm 数 = 8 × 3 = 24 个在途请求，Day 1 必须实测服务商能否承受。

**管理调用开销**：`Select` 与 `Apply` 都是 0 次模型调用（在线部分；topic 标注是离线一次性的，见 §6.3）。管理开销完全集中在 Reflect 与两个 curator，因此"solver 调用 vs 管理调用分开报告"是精确数字而非估算。

---

## 4. Task A：Skill 表示、存储与选择

### 4.1 Skill schema

磁盘上是 markdown（人可读，便于放进 README 做 case study），内存中是 dataclass。

```python
@dataclass
class Skill:
    id: str; name: str           # sk_0007 / kebab-case
    level: str                   # "general" | "topic"
    topic: str | None
    trigger: str                 # 适用条件；同时是 selection 检索键
    lesson: str                  # 操作知识，bullet points，≤60 词
    failure_mode: str            # 要避免的失败模式，≤40 词
    evidence: list[str]          # 仅 problem_id + 失败类别，绝无题面/答案
    created_at: int; revised_at: list[int]
    n_selected: int; n_selected_success: int
    n_tokens: int
```

**⚠️ 全部 skill 必须用英文。** AIME 题面是英文，§4.3 的词面检索若拿中文 trigger 去匹配英文题面，重合恒为 0，selection 会静默退化成 utility-only（早期全是 Laplace 先验 0.5 ≈ 随机）。Reflect/curator 的 prompt 里写死 `Write in English`，`guard.py` 加非 ASCII 比例检查。

```markdown
---
id: sk_0007
level: topic
topic: number_theory
created_at: 23
revised_at: [41]
evidence: ["p_0023:symbolic_slip", "p_0041:unverified_generalization"]
n_selected: 12
n_selected_success: 7
---
## When to use
Counting integers subject to divisibility or congruence conditions.

## Strategy
- Brute-force small range (n <= 50) in python first to get the initial terms.
- Then conjecture a closed form and **check it back** against the enumeration.
- If they disagree, suspect boundary conditions before arithmetic.

## Avoid
Do not extrapolate a small-range pattern to the problem's full range without numeric verification.
```

`## When to use` ↔ `trigger`，`## Strategy` ↔ `lesson`，`## Avoid` ↔ `failure_mode`。与参考实现的 `## Key points`/`## Gotchas`、`## Pattern`/`## Strategy` 同构。

### 4.2 增长边界（硬约束）

| 层 | 限制 | 默认 | 依据 |
| --- | --- | --- | --- |
| 容量 | general 上限 | 5 | 论文**附录 F**（参考实现代码里默认是 10，我们跟论文） |
| 容量 | 每 topic 上限 | 5 | 论文附录 F；参考实现 `max_skills_per_context=5` |
| 注入 | 每题条数 `b` | 6（general ≤ 3，topic ≤ 4） | 论文式 (2) `\|S\| ≤ b`，具体值论文未给 |
| 注入 | 每题 token `T` | 800 | 自定 |
| 单条 | `lesson` ≤ 60 词，整条 ≤ 200 词 | — | 参考实现 `CONTENT under 200 words` |

**`store.apply()` 必须是两阶段**：先应用 `DELETE / MERGE / REVISE`，再用**更新后**的占用数校验 `ADD`。否则 GeneralCurator 输出的 `DELETE x + NEW y` 组合里 `NEW` 永远进不去，harness 一旦打满就再也无法换血。单测 `test_delete_then_add_within_one_batch` 锁住这个行为。

统一算子集合（全文一致）：`{ADD, MERGE, REVISE, DELETE, SKIP}`。curator 的输出格式映射到这五个。容量满时 prompt 声明只能 `MERGE/REVISE/DELETE/SKIP`，且代码层在阶段二过滤掉超额 `ADD` 并记日志 —— 即使 LLM 无视指令也接得住。

### 4.3 Selection：确定性分层检索

**⚠️ 这是相对论文的有意偏离**：论文附录 F 明确 *"For harness selection, we use Claude Sonnet 4.5 across all experiments"* —— 论文用的是 LLM selector。我们用确定性检索，理由是它让 selection 的在线模型调用**恒为 0**，使作业要求的"solver vs 管理调用分开报告"变成精确数字。README 必须写明这是偏离，并把 LLM selector 作为可选 A/B（作业 §3 第 2 条）。

```
1. 候选 = topic(x_t) 对应的 topic skills ∪ 全部 general skills
2. score = α·utility + β·lexical_overlap(trigger, question),  α=0.3, β=0.7
     utility = (n_success + 1) / (n_selected + 2)                # Laplace
     lexical_overlap = BM25-lite，词形归一后加权重合，归一到 [0,1]
3. 分层配额：general ≤ 3，topic ≤ 4，合计 ≤ b=6
4. 累加 n_tokens，超 T 即停
```

分层配额是写死的，不是"general 保底 2 条 + 全局竞争"——后者在 b=6 时可能退化成 general 占 5 席、topic 只剩 1 席。`test_selection_budget` 固定断言这个配额。

### 4.4 日志与导出

**`harness_log.jsonl`**（harness 侧，append-only）：

```json
{"problem_idx": 41, "batch": 5, "op": "REVISE", "skill_id": "sk_0007",
 "actor": "topic_curator", "before_hash": "a3f", "after_hash": "9c1",
 "reason": "...", "accepted": true, "reject_reason": null}
```

被 guard 或预算拒掉的也照写一行（`accepted:false` + `reject_reason`），因为作业要求记录"产生了哪些候选更新，以及最终是否接受"。

**`selection_log.jsonl`**（选择侧，每题一行）—— 作业 Task B 硬性要求"保存每个问题使用了哪些 skills"，而 wandb 不是仓库产物，必须落盘：

```json
{"problem_idx": 41, "batch": 5, "topic": "number_theory",
 "selected": [{"skill_id": "sk_0007", "score": 0.72, "tokens": 94}],
 "total_inject_tokens": 611, "pass1_round0": 0, "pass_final": 1}
```

这一个文件同时喂 §7.2 的 `usage/*` 和 §7.3 的"注入 token / 使用频率"两项报告要求。

导出：`harness_final/` + 上述两个 jsonl + `harness_timeline.csv`。

### 4.5 Task A 测试

纯单测，stub LLM，零 API 费用：

| 测试 | 断言 |
| --- | --- |
| `test_cross_problem_persistence` | 重新实例化 `SkillStore` 后 id/内容/计数一致 |
| `test_selection_budget` | 20 条下返回 ≤6 条、≤800 tokens，且 general ≤3 / topic ≤4 |
| `test_capacity_is_hard_bound` | 满槽时 ADD 被过滤，harness 不增长，日志有 `accepted:false` |
| `test_delete_then_add_within_one_batch` | 两阶段 apply：同 batch 内 DELETE 腾出的槽位可被 ADD 占用 |
| `test_no_interference_with_inproblem_memory` | **针对正确的对象**：`env_manager.memory.reset()`（`:85`）与 `solution_memory.reset()`（`:489`）后 SkillStore 不变；harness 文本只出现在 system message，从不进入任一 memory 对象 |
| `test_anti_leakage` | 含题面 8-gram / GT 数字 / 非英文的候选被 `validate_skill()` 拒绝并记日志 |
| `test_log_completeness` | 每个候选在 `harness_log.jsonl` 中恰好一条终局记录 |

---

## 5. Task B：Context-to-Harness Compilation

### 5.1 Batch 协议

论文 §3.4 与 Algorithm 1：harness 按 batch 更新；`|B_i| = 1` 时退化成只有 `CompileTaskType`（topic skills），**cross-task pattern 提炼不出来**，因为没有多条候选可横向比较。参考实现的 general curator 进一步要求模式跨多个任务出现（`cl_bench.py:180` 是 3+，swe/tau/terminal/webarena 是 2+）。

**B = 8**。论文附录 F 默认是 16 —— 这是有意偏离，为了在 80 题的适应流里拿到 10 个更新点（16 的话只有 5 个），让"增长趋势"这条报告要求有足够的时间分辨率。README 需注明。

顺序协议的三条保证：batch 内所有题在更新**之前**完成求解，用 `H_i`；第 t 题的候选只进 `H_{i+1}`；题目顺序完全固定。

### 5.2 Reflect

**只在失败或收到负反馈时触发**。

⚠️ **这里有三方分歧，README 必须点明**：论文正文 §3.2 式 (6) 明确写 *"produces a candidate memory only when the execution fails or receives negative feedback"*，但论文 **Algorithm 1 第 9 行是无条件的 `r_ij ← Reflect(c_ij)`**（论文自身不自洽）；参考实现两条路径都有（对成功轨迹也 propose，见 `ANALYZE_AND_PROPOSE_PASS_PROMPT`）。我们跟正文（failure-only），理由是省调用且避免成功轨迹的题目特异细节污染 harness；"成功轨迹能产生什么类型的 skill"正好是作业 §3 可选扩展第 6 条。

**输入构造 = 防泄漏第一道闸**。用**白名单**取字段，不是黑名单删字段：

```python
ReflectContext = {
  "topic": str,                    # 离线标注的 topic 标签
  "problem_shape": str,            # 离线标注的题型标签（如 "counting-with-constraints"）
  "final_answer_given": str,       # 自己给的答案，不是 GT
  "outcome": Literal["failed"],    # 只有 0/1 标签
  "verifier_feedback": str,        # 已剥离 GT 痕迹，见 §5.4
  "tool_errors": str,              # python_code 报错/异常摘要
  "reasoning_excerpt": str,        # 轨迹片段，不含题面
  "round_count": int,
  "related_skills": list[Skill],   # ← 当前 harness 中与本题相关的 skill（见下）
}
```

⚠️ **v1 里的 `question_summary: 题面前 200 字` 已删除**。那是"先制造污染再靠 guard 清洗"的反模式。改为只送离线标注器产出的结构化特征（`topic` + `problem_shape`），题面原文完全不进入 Reflect 的 prompt。

⚠️ **`related_skills` 是 v2 补入的**。论文**附录 E.1** 明确把 *"related existing skills"* 列为 proposal prompt 的输入之一，参考实现的 `PROPOSE_SKILL_PROMPT` 也有 `{existing_skills_section}`。不给的话 solver 会反复提炼 harness 里已有的东西，白白消耗 curator 预算并抬高重复率。这里直接复用 `store.select()` 的结果（本题已经选中的那几条），零额外成本。传 skill 是安全的：skill 本身已经过 `guard.validate_skill()`，不含题面与答案。

**输出**（对齐附录 E.1 的 *"decide NEW, ENHANCE, or NONE"*）：

```python
CandidateMemory(trigger, lesson, failure_mode, scope_hint, topic, evidence,
                action_hint,   # "NEW" | "ENHANCE"，NONE 时整体返回 None
                target_id)     # ENHANCE 时指向要增强的现有 skill id
```

`scope_hint` 与 `action_hint` 都只是**建议**，最终落层与落盘由 curator 决定；但 `action_hint="ENHANCE"` + `target_id` 给了 curator 一个强信号去 MERGE/REVISE 而不是 ADD，这正是控制重复增长的机制。

**Reflect prompt 必须包含附录 E.1 的过滤规则**（原文：*"Filter aggressively. Skip generic advice, basic tool usage, exact task replay, and skills that would not help unseen tasks."*）。其中 **"exact task replay"** 这一条同时承担防泄漏职责 —— 它在 prompt 层就压制了"把本题解法抄进 skill"的倾向，是 §5.4 四道闸之外的一道软闸。

### 5.3 Evolver（双 curator）

**TopicCurator** —— 输入 `(该 topic 现有 skills, 本 batch 该 topic 的候选, 槽位预算)`，输出 `ADD / MERGE / REVISE / SKIP`。槽位 5。
**GeneralCurator** —— 输入 `(现有 general skills, 本 batch 全部失败摘要)`，跨 topic 分析失败模式，输出 `ADD / REVISE / DELETE / SKIP`（无模式时 NO_PATTERNS）。要求模式跨 ≥2 题出现（对齐参考实现多数分支）。槽位 5。

两个 curator 的 prompt 必须包含论文**附录 E.2/E.3** 的两条判据：

- **Generalizability test**（E.2 原文：*"apply a generalizability test such as usefulness for multiple unseen tasks"*）—— 每条决策都要问"这条 skill 对多个**未见过**的题有用吗"，而不是"它是否准确描述了刚才那次失败"。这是区分 skill 与流水账的核心判据。
- **General 层的额外约束**（E.3 原文：*"Create or update a general skill only when a pattern appears across multiple contexts. General skills must avoid context-specific references"*）—— GeneralCurator 必须拒绝任何带特定题型引用的候选。

每 batch 管理调用 = `#failed`（≤8）+ `#distinct_topics`（≤4）+ 1 ≈ 4–13 次。

### 5.4 防泄漏：闸门设计

**已确认的真实泄漏通道**：`core/tools/informalmath_verify.py:107` 与 `:176` 会把 `Matches ground truth: {True|False}` / `Matches GT: ...` 写进返回文本，经 `env.py:106` 包成 `<tool_response>` 回传给 agent。它目前不触发，仅仅因为 policy/verifier 的 prompt 模板没有宣传 `<informalmath_verify>` 标签 —— 但 `env.py:125-128` 的 `_parse_action` 是**无条件**解析该标签的。靠"模板没写"当防线不合格。

| 闸 | 位置 | 机制 |
| --- | --- | --- |
| 1 | `ReflectContext` builder | **白名单**取字段，题面与 GT 在结构上不可达 |
| 2 | driver 运行时断言 | 全 run 的 action 文本若出现 `<informalmath_verify>` → fail-fast |
| 3 | feedback 净化 | verifier report 进 Reflect 前正则剥掉 `Matches GT:` / `Matches ground truth:` / `score=` |
| 4 | `guard.validate_skill()` | 拒绝：与题面 8-gram 重合、含与 GT 相同的孤立数字、非英文、超长 |
| 5 | CI 静态检查 | grep 断言 `reflect.py` / `evolver.py` / `render.py` 不出现 `ground_truth` 标识符 |

**关于 GT 的唯一合法用途（README 必须显式论证）**：作业允许"使用成功/失败标签和 verifier feedback"，禁止"skill compilation 接收 ground-truth answer"。本设计中 GT 只出现在两处：
- **(a) 成功/失败标签**：`compute_answer_correctness` 用 GT 产出 0/1，作业明确允许。
- **(b) `guard.py` 的负向过滤**：需要 GT 才能判断 skill 里是否混进了答案数字。这是**信息单向删除** —— guard 只 reject 不 generate，GT 绝不流入 harness。

因此闸 5 的 grep 白名单里 `guard.py` 是例外，README 中如实说明理由。

### 5.5 失败处理

作业要求"skill 更新失败时有清晰、可记录的处理方式，不能破坏原有 baseline"。

所有 harness 操作包在 `try/except`，**任何异常降级为 no-op 并记日志**：LLM 超时/限流 → 重试 3 次后跳过（`Agent` 自带 `max_retries=5`）；解析失败 → 记 `parse_error`；guard 拒绝 → 记 `rejected`。**harness 只在完整校验通过的编辑集上原子替换**（写临时目录 + rename），中途崩溃不留半更新状态。求解路径与 harness 路径完全解耦，后者全挂不影响前者产出结果。

---

## 6. Task C：三个 arm 与公平性

### 6.1 统一接口

```python
class CrossProblemArm(Protocol):
    def begin_batch(self, batch_idx) -> str:              # -> 本 batch 的注入文本
    def record_selection(self, problem_idx, ...) -> None
    def observe(self, problem_idx, result) -> None
    def end_batch(self, batch_idx) -> list[dict]          # 更新，返回变更记录
```

| Arm | 注入 | 更新 | 管理调用/batch |
| --- | --- | --- | --- |
| **Baseline** | 中性占位（见下） | no-op | 0 |
| **Raw Experience** | 用同样的 BM25-lite + 同样的 `b`/`T` 选历史摘要 | 每题存一条摘要 | 8（每题 1 次 summarize） |
| **Evo-Harness** | `store.select()` | Reflect + 双 curator | 4–13 |

### 6.2 公平性：逐条落实

**① system message 必须结构对齐（v1 的漏洞）**
`utils/agent.py:37` 是 `if self.system_prompt:` —— 空串是 falsy，**整个 system role 消息不存在**。这意味着 v1 设计里 Baseline 根本不发 system message 而另两个 arm 发，多了一个结构变量；更糟的是 Evo arm 的第一个 batch（harness 为空）也不发，同一 arm 内部在第 9 题发生结构性跳变。

**修正**：三个 arm 一律发送同构 system message。Baseline 与空 harness 时用固定中性占位（`"You are a competition mathematics solver."`），该占位的 token 也计入成本表。一行改动，但这是"实验公平性"上最容易被挑的一刀。

**② 记账必须 monkey-patch，不能包装 Agent 对象**
`utils/agent.py:68` 的 `get_action_from_gpt` 只 `return reasoning_text + action`，**`response.usage` 被丢弃**，不 patch 就报不出 token 数。而且 `evolving_main.py:607` 的 summarizer 和 `:180` 的 aggregator 都是在函数内部**现场 new** 的 `Agent`，从外部包装对象完全抓不到。

**方案**：在新 driver 启动时 `functools.wraps` patch `Agent.get_action_from_gpt`，内部读 `response.usage`，按 `contextvars` 中的当前 role 标签记账。仍是 0 文件改动。role taxonomy：`solver / summarizer / aggregator / reflect / topic_curator / general_curator / offline_labeling`。⚠️ `aggregator` 是 v1 漏掉的（`aggregate_verifier_judgments` 在 `verifier_env_num>1` 时每轮调一次）；`summarizer` 与 `aggregator` 属 **solver 侧**（baseline 自带），不能计进管理开销。

**③ 必须有 seed 和重复**
`chat.completions.create`（`utils/agent.py:47-54`）**没有传 `seed`**，policy `temperature=0.7`。同一个 patch 里补 `seed` 参数（仍是 0 文件改动）。held-out 必须 `test_times ≥ 3`，报 mean ± std。

**④ 指标必须先定义**
`run_problem` 返回的 `success_rate` 是**跨 evolving round 平均**（`:546-548` 每轮 `total += 1`，`:822-824` 做 `success/total`），`evolving_round=3` 时取值只能是 {0, ⅓, ⅔, 1}，**不是 Pass@1**。可用的干净信号是 `policy_answer_correct`（`:566-567`）。

定义两个全 arm 统一的指标：
- **`pass1_round0`** —— 第 0 轮（题内进化之前）的正确率。**这是 harness 注入效果最干净的度量**，因为它不掺杂题内多轮修正。
- **`pass_final`** —— 最后一轮的正确率，题内 + 跨题的联合效果。

两个分开报，§7.3 的信息量显著提升。

**⑤ 其余对齐项**：同一问题序列文件、同一 topic fixture、同一模型 ID / temperature / max_tokens、同样的题内配置（`evolving_round` / `max_steps` / `verifier_env_num` / 工具开关）、同样的 batch 划分（Baseline 也走 batch 结构，只是 `end_batch` 为 no-op）、三个 arm 在同一时间窗内跑完。**Raw Experience 复用已有的 `SUMMARIZER_TEMPLATE`** 并使用与 Evo 相同的 `b`/`T` 注入预算 —— 两者差别纯粹是"提炼过的经验 vs 原始摘要"，不是"注入多少"。

**⑥ 统计功效必须事先算清楚**
held-out 30 题时 1 题 = 3.3pp。`test_times=3` 后有效样本 90，但题目仍只有 30 道，arm 间差异需超过约 10pp 才脱离噪声 —— 对 4B 模型 + 80 题适应流不现实。**README 必须事先写明"本实验规模下可分辨的最小差异约 X pp"**。这是对"Evo-Harness 不如 baseline 不是失败"的正确回应方式：作业要的是分析，而"实验规模不具备区分功效"本身就是合格分析 —— 但必须事先算出来，不能事后找补。

### 6.3 数据

严格按年份，**绝不先混合再随机切分**。以下数据源已于 2026-09-11 实测确认（行数与列名均为实际 `load_dataset` 结果，非文档推断）：

| 用途 | 数据集 | 实测题数 | topic 来源 |
| --- | --- | --- | --- |
| Adaptation stream | `gneubig/aime-1983-2024`，`Year ∈ 2018–2022`，**主题交错排列见 §15.3** | 150 可用 → 取 80（10 batch × 8） | 自建离线 LLM 标注 |
| **主 held-out** | `MathArena/aime_2025` | 30 × 3 次重复 | **内置 `problem_type`** |
| 次 held-out（污染对照） | `math-ai/aime24` | 30 × 3 次重复 | 自建离线 LLM 标注 |

`gneubig/aime-1983-2024` 共 933 行，列为 `ID / Year / Problem Number / Question / Answer / Part`。实测 2018–2022 每年恰好 30 题（Part I 15 + Part II 15），完整无缺口；`(Year, Part, Problem Number)` 三元组给出确定性排序键。

⚠️ **该数据集的 2024 年只有 14 题（仅 Part II），2023 年 29 题** —— 都不完整，因此 2024 的污染对照必须改用 `math-ai/aime24`（实测 30 题完整，且它正是 AlphaApollo `prepare_evolving_data.py` 的默认源，`solution` 字段为 `\boxed{204}` 形式，现成的 `extract_solution` 能解析）。

### 意外收获：`MathArena/aime_2025` 自带人工 topic 标注

实测 `problem_type` 分布：**Algebra 9 / Combinatorics 9 / Geometry 8 / Number Theory 6**，且 **30 题里只有 2 题是多标签**。

三个直接后果：

1. **§13 待确认第 2 条（"4 类 topic 够不够，AIME 常跨主题"）得到了经验答案** —— 93% 的题是单标签，4 类分法站得住。此前的担心基本不成立。
2. **主 held-out 不需要自己标注**，直接用它的标签，少一处噪声源。
3. **它可以当作标注器的验证集** —— 用我们自建的离线 LLM 标注器去标这 30 题，与人工标签算一致率，把"标注质量"从口头声明变成一个可报告的数字。这几乎零成本（30 次调用），却能显著加强 README 里"分主题分析"那部分的可信度。**列为 Day 2 必做。**

关于报告口径：即便 4 类分法成立，held-out 每类仍只有 6–9 题，画曲线仍不可读。§6.3 原定的"主曲线 2 类粗分、4 类细分降格为定性观察"的决定**维持不变** —— 那是样本量问题，不是分类学问题。topic skills 的槽位仍按 4 类细分。

**为什么主 held-out 用 2025 而不是 2024**：Qwen3-4B-Instruct-2507 公开自报 AIME25 成绩，说明训练覆盖到 2025 年初，2024 几乎肯定被污染。用 2025 为主、2024 为次，反而得到一组"污染 vs 相对未污染"的对照 —— 这是个加分的分析点，不是负担。

held-out **从不参与任何 harness 更新**：用冻结的 `H_final` 跑，代码里 `frozen=True` 强制关闭 `end_batch`。

**⚠️ Day 2 之前必须确定数据源**：`math-ai/aime24` 只有 2024 年。需要找一个带年份元数据的 HF AIME 数据集（如 `di-zhang-fdu/AIME_1983_2024` 一类），否则会卡住。这是 Day 1 的阻塞项。

**Topic 标注**：离线一次性 LLM 标注，产出 `topics.json` fixture 并 commit，三个 arm 共用。同一次标注还要产出**细粒度技法标签**（20–30 个，如 `divisibility-counting` / `recursion` / `similar-triangles`），供 §15.4 的复用机会分析使用 —— 技法标签只用于**离线诊断与任务流构造**，不进入 selection，也不进入任何 prompt。⚠️ 但**不能说"不计入调用预算"** —— 作业明写"若各方法引入的额外模型调用不同，应分别报告"。标注只有 Evo/RawExp 实际消费，必须在 §7.3 成本表里单列 `calls/offline_labeling`，标注为"一次性、三 arm 共享"。

**⚠️ topic 粒度**：4 类 × 80 题 ≈ 每类 20 题，分主题 Pass@1 每格样本量 <20，曲线不可读，且 AIME 常有跨主题题。**决定：主曲线用 2 类粗分（`algebra_nt` / `combinatorics_geometry`），4 类细分降格为"定性观察 + 具体案例"而非曲线。** topic skills 的槽位仍按细分的 4 类（保持局部性），只是报告口径变粗。

---

## 7. 指标与 wandb

### 7.1 为什么要 wandb

作业 md 的提交物没写 wandb，但对方额外要了。原因：run 自带时间戳与 config snapshot（可信度）、三个 arm 叠在同一坐标系（公平性最强证据）、要求的指标本来就是时间序列。evo 路径零 wandb 集成，需自建 `tracker.py`，同时双写 jsonl 以便离线复现图表。

### 7.2 log schema（`step = problem_idx`）

```
# 性能（两个定义，见 §6.2④）
adapt/pass1_round0, adapt/pass_final
adapt/running_pass1_w20                     滑动窗口 20 题
adapt/pass1_by_topic2/<algebra_nt|comb_geo> 2 类粗分曲线

# harness 规模
harness/n_general, harness/n_topic_total, harness/n_topic/<topic>
harness/total_tokens, harness/mean_skill_tokens        ← 饱和后的主曲线

# 注入与使用
inject/n_skills_selected, inject/n_tokens
usage/skill_hit/<skill_id>, usage/skill_utility/<skill_id>

# 成本（solver 侧 vs 管理侧严格分开）
calls/{solver,summarizer,aggregator}                  ← solver 侧
calls/{reflect,topic_curator,general_curator}         ← 管理侧
calls/offline_labeling                                ← 一次性，单列
tokens/solver_{in,out}, tokens/mgmt_{in,out}

# 编辑决策
edits/{add,merge,revise,delete,skip,rejected_by_guard}
edits/accept_rate                                     ← Day 4 smoke 的健康指标
```

Tables/Artifacts：`harness_final`、两个 jsonl、`representative_traces`（正/负迁移各 2–3 条）。

**案例展示格式直接采用论文附录 B 的五段式**（作业要求"至少 2–3 个正迁移或负迁移案例"）：

```
Task:          问题标识（年份 + 题号，不贴题面）
Baseline:      无 harness 时的失败表现 + verifier 判据
Evolved:       有 harness 时的表现变化
Learned skill: 起作用的那条 skill 的 trigger/lesson 原文
Core point:    一句话说明 harness 到底学到了什么区分
```

论文用这个格式展示了 6 个跨 benchmark 的案例（附录 B）。照搬它的好处是 grader 一眼就能看出你读过附录、且案例是按同一标准挑的，而不是事后翻日志硬凑。**负迁移案例用同一格式，只是 `Evolved` 一栏记录变差**——作业明确要求正/负都报。

配对方法（附录 B 原文）：*"we align the same task identifiers between the no-evolve and evolved runs. We only call a case a direct improvement when the baseline run fails and the evolved run succeeds."* 即严格按 problem_idx 对齐 baseline 与 evo 两次运行，只有 baseline 失败且 evo 成功才算正迁移。论文同时诚实声明这**不是单条 skill 的因果消融**（一次注入多条 skill），我们的 README 也要照此声明。`selection_log.jsonl`（§4.4）正好提供了每题注入了哪几条，使这个配对可自动化。

### 7.3 最终结果表

| | Baseline | Raw Exp | Evo-Harness |
| --- | --- | --- | --- |
| Adaptation `pass1_round0`（整体 / 后 1/3） | | | |
| Adaptation `pass_final`（整体 / 后 1/3） | | | |
| **Held-out AIME 2025**（mean ± std, n=3） | | | |
| 次 held-out AIME 2024（污染对照） | | | |
| 2 类粗分 Pass@1 | | | |
| 最终 harness 条数 / 总 tokens | 0 | | |
| 平均注入 tokens / 题 | （占位文本 tokens） | | |
| Solver 调用 / token（含 summarizer+aggregator） | | | |
| **管理调用 / token** | 0 | | |
| 一次性离线标注调用 | — | 共享 | 共享 |
| 端到端 wall-clock | | | |

**"后 1/3" 是主比较区间**，不是附栏。理由：初始 harness 为空（参考实现是带 `seed_skills/` 起步的），前 2–3 个 batch 的 harness 质量最低且样本最少，Evo arm 的前 8 题与 Baseline 完全等价。这个理由要写进 README。

---

## 8. 成本与 wall-clock

**降配（三个 arm 完全一致）**：

```yaml
evolving_round: 3            # 10 -> 3
verifier_env_num: 1          # 5 -> 1，关掉多数投票
policy_model_cfg.max_tokens: 4096
problem_max_workers: 8       # = batch size B，batch 内并行
```

`verifier_env_num: 1` 同时省掉 aggregator 那次调用（`aggregate_verifier_judgments` 仅在 >1 时触发）并把 verifier 串行链条缩短 2/3 —— 这是最大的单点延迟优化。代价是失去多数投票的稳健性，需在 README 说明。

**Token**：每题约 20–25 次 solver 调用 ×（~3k in + ~1.5k out）≈ 100k token/题。每 arm = 80（adapt）+ 90（held-out ×3）+ 90（次 held-out ×3）= 260 题 ≈ 2600 万 token；三 arm ≈ **7800 万 token**。按 Qwen3-4B-Instruct-2507 约 ¥0.35/M ≈ **¥27**，预留 ¥100 足够。**成本不是约束。**

**Wall-clock（这才是瓶颈）**：单题串行 API 往返约 20–24 次 × 15–30s ≈ 6–12 分钟。
- 顺序执行：260 题 × 9 分钟 ≈ **39 小时/arm** —— 不可行（v1 估的 5–8 小时低估了 4–5 倍）。
- **batch 内并行 8**：≈ **5 小时/arm**；三 arm 并行进程 ≈ **5–7 小时总计**。可行。

⚠️ 并发 24 个在途请求。**Day 1 必须实测**：20 次调用的延迟分布 + 24 并发下是否限流。若限流，降到 B=4（并发 12），wall-clock 翻倍到 ~10 小时，仍在排期内。§8/§9 的数字 Day 1 实测后回填。

---

## 9. 七天排期

| Day | 目标 | 产出 / commit |
| --- | --- | --- |
| **1** | **环境 + 数据源 + 实测 + 泄漏审计**（全是阻塞项） | ① `pip install -e . --no-deps` + 手工依赖，跑通 evo 路径 2 题；② 确认服务商 Qwen3-4B-Instruct-2507 可用，**实测 20 次调用延迟分布 + 24 并发限流**，回填 §8；③ **确定带年份元数据的 AIME 数据源**；④ 审计 `informalmath_verify` 泄漏路径。`commit: chore: slim env + evo smoke test` |
| **2** | Task A + 数据准备（并行，互不依赖）+ **复用机会分析（门禁）** | `schema/store/render/guard/loader.py` + 7 个单测全绿；`prepare_harness_stream.py` + topic/技法 fixture + 主题交错排列。**跑 §15.4 的三个统计量，不达标就地调整任务流再往下走。** `commit: feat(harness): skill store, layered selection, two-phase apply` / `test(harness): ...` / `feat(data): topic-interleaved AIME stream + reuse-opportunity analysis` |
| **3** | Task B 机制 | `reflect/evolver/accounting.py`，用 Day 1 录制的真实轨迹做 fixture 离线测。`commit: feat(harness): reflect + dual curator compilation` |
| **4** | Driver + 三 arm + wandb + **当晚开跑** | `evolving_harness_main.py` / `arms.py` / `tracker.py` + 三份 config。8 题 smoke run 检查三个健康指标：`inject/n_skills_selected > 0`、`edits/accept_rate > 20%`、三 arm 的 system message 结构一致。**当晚启动 adaptation。** `commit: feat(evo): batched harness driver with three arms` |
| **5** | adaptation 跑完 + 边跑边写分析脚本 | 三 arm 的 80 题 adaptation 结果；`scripts/analyze.py` 出全部表格与曲线。当晚启动 held-out。 |
| **6** | held-out + 分析 + 案例 | AIME 2025 / 2024 各 ×3；正/负迁移案例 2–3 个；feedback-grounding 消融（§11）。`commit: exp: adaptation + held-out results` |
| **7** | README + slides | 三级 memory 边界图、设计论证、命令、结果、案例、失败与限制、上游归属。`commit: docs: README ...` |

**关键变更**：数据准备从 Day 5 提到 Day 2（它不依赖 Task A/B 任何代码，而 Day 4 的 smoke run 需要带 topic 的真实题流）；adaptation 从 Day 5 提到 **Day 4 当晚**启动。

**降级预案**：若 Day 5 晚 adaptation 未跑完，砍次 held-out（AIME 2024 污染对照）而**不砍 adaptation 题数** —— 80 是作业下限，跌破需在 README 中解释，代价比砍一个可选对照大。

---

## 10. 风险与预案

| 风险 | 触发信号 | 预案 |
| --- | --- | --- |
| Mac 依赖装不上 | Day 1 import 失败 | 已知 decord/rdkit/vllm 是坑，用 `--no-deps`；仍不行则租 GPU 机 |
| 找不到带年份的 AIME 数据集 | Day 1 | 从 AoPS 或多个单年数据集拼接，自己打 year 字段 |
| 24 并发被限流 | Day 1 实测 | B 降到 4，wall-clock 翻倍仍可行 |
| **harness 全程没被选中** | `inject/n_skills_selected == 0` | 闭环没闭上，最严重；Day 4 smoke 必须发现 |
| **Reflect 输出全被 guard 拒** | `edits/accept_rate < 20%` | 与上一条是不同的失败模式；调 Reflect prompt（参考实现："Be SPECIFIC — not generic advice like 'read carefully'"） |
| skill 全是空话 | Day 4 人工看产出 | 同上 |
| **数据导致的 null**（任务流无复用机会） | Day 2 的 §15.4 统计：技法标签 80% 只出现 1 次 | **Day 2 门禁**，不达标就调整任务流再开跑。绝不能跑完才发现 |
| **三个 arm 在 held-out 上完全打平** | Day 6 | 已预期（§6.2⑥）。事先算好可分辨最小差异，把它作为结论之一而非事故 |
| harness 3–4 个 batch 就打满 | `harness/n_*` 变平线 | 已预期（§4.2 容量 25 槽）。主曲线切到 `harness/total_tokens` + `edits/{merge,revise}` 时间序列，论述"饱和后进入 refinement 阶段" |
| Evo 不如 baseline | Day 6 | **不是失败**，作业明示。用论文 Table 4（Self-Generated 反而掉点）+ Table 2（EDS 类掉点）做有依据的分析 |

---

## 11. Feedback grounding：本项目最大的方法论风险（升格为主论述）

论文 Table 4：`Self-Generated` feedback（LLM 自判）在 CL-Bench 上 **27.96，低于 No-Evolve 的 29.54** —— 即 LLM 自评的反馈会让 harness 越进化越差。只有 grounded 的 `Minimal`（环境级 0/1）和 `Standard`（错误信息/trace/verifier 输出）有用。

**而 AlphaApollo 的 verifier 本身就是 LLM judge**（`informalmath_verify.py:161` 走 `_agent_generate`，`aggregate_verifier_judgments` 也是 LLM）。也就是说，我们系统里看起来像 "Standard" 的反馈，在论文的分类里其实落在 **`Self-Generated`** 一档。这是整个项目最大的方法论风险，不是可选扩展。

唯一真正 grounded 的信号是 `compute_answer_correctness` 基于 GT 产出的 0/1 标签（作业明确允许使用）。

**因此把 feedback-grounding 对比纳入 Day 6 必做项**（而非"有余量才做"）：

| 配置 | Reflect 收到的反馈 | 对应论文 |
| --- | --- | --- |
| `minimal` | 仅 0/1 正误标签 + tool_errors | Minimal |
| `standard`（默认） | 完整 verifier report + tool_errors | 论文的 Self-Generated |

**成本控制**：只在 held-out 上跑这一组对照（30 题 × 3 次），不重跑整个 adaptation —— 即用两套不同 feedback 配置各跑一遍 adaptation 产出两个 `H_final`，代价是多一个 adaptation arm。若 Day 5 时间紧，退化为"只在最后 3 个 batch 上切换配置"的轻量版，并在 README 说明这是简化。

**其余可选扩展**（Day 6 有余量才做，按性价比排序）：
1. General-only vs Topic-only vs 全量 —— 论文 Table 3 消融，改两个开关重跑 held-out，零新代码。
2. Selection 方法 A/B —— 确定性检索 vs LLM selector（同时补上与论文附录 F 的对齐）。
3. Skill utility 驱动的淘汰 —— `n_selected_success` 已在记，加一条 DELETE 规则即可。
4. 成功轨迹 vs 失败轨迹分别产生什么 skill —— 正好检验 §5.2 的三方分歧。

---

## 12. 上游代码复用与许可证

- **AlphaApollo**：Apache 2.0（源文件头部 `Copyright 2026 TMLR Group, Apache License 2.0`）。本项目为其 fork，新增文件保留同一许可证头。
- **a-evolve / Evo-Harness**：仅作**设计参照**阅读（skill 的 markdown 三段式、双 curator 结构、槽位预算语义、batch 并行），**不 import、不复制代码**。Day 1 核对其 `LICENSE` 并在 README 如实标注借鉴了哪些设计点。
- **Evo-Harness 论文**：README 与 slides 中引用 arXiv:2608.15071。
- **为什么该设计适合 AlphaApollo 的数学推理环境**（作业 Task A 明确要求的论证，README 需专节）：三段式 markdown 对应数学解题的"识别题型 → 选择方法 → 规避陷阱"；词面检索适用是因为竞赛数学的题型关键词（`divisibility` / `recursion` / `inscribed circle`）高度标准化，无需语义嵌入；general/topic 分层对应"通用解题纪律（先数值验证再推广）"与"局部方法（特定题型的套路）"的天然分野。

---

## 13. 待确认事项

已决策（v2 中拍板，如有异议请指出）：
- ✅ 主 held-out 用 **AIME 2025**，2024 作污染对照（§6.3）
- ✅ Batch size **B=8**（论文默认 16，有意偏离，§5.1）
- ✅ topic 报告口径 **2 类粗分**，槽位仍按 4 类细分（§6.3）
- ✅ Skill 全部**英文**（§4.1）
- ✅ `verifier_env_num: 1`（§8）
- ✅ Feedback-grounding 对比升格为**必做**（§11）

- ✅ 预算 `b=6` / `T=800` / 容量 `5+5` **接受**。核算：每题多 ~16k 输入 token，260 题 × 3 arm ≈ ¥4，对 §8 的 ~5h/arm 无影响；管理调用每 arm 40–130 次。⚠️ 监控点：T=800 vs AIME 题面 ~200 token，harness 在 prompt 里占 4:1，对 4B 模型是实打实的 context interference 风险。Day 4 smoke 若 `pass1_round0` 低于 baseline，**T 是第一个该调的旋钮**（降到 500 ≈ 4 条）。
- ✅ 任务流用**主题交错**，见 §15。
- ✅ 仓库：在 `AlphaApollo/` 开 feature 分支，设计文档与实现一并进该分支。

仍待确认：无。

---

## 15. 任务流构造：主题交错（防"数据导致的 null"）

### 15.1 两种 null 必须能区分

- **方法导致的 null**：机制跑通、复用机会存在，但没效果（skill 太空泛 / 选错 / context 干扰 / 反馈不可靠）。**有信息量**，可顺着分析，正是作业 §5"研究判断"想看的。
- **数据导致的 null**：机制可能完全正确，但任务流里**根本没有复用机会**。结果必然平，且**无法区分**"机制没用"和"没机会用"，作业列的分析角度一个都验证不了。

两者在最终数字上长得一模一样（三条曲线交缠、held-out 打平），所以必须靠**独立于结果的指标**事先区分。

### 15.2 AIME 的粒度错配风险

粗粒度 topic（number theory）在 80 题里出现 ~20 次，看似机会充足。但有用的 skill 是**细粒度技法**级别的（"带整除约束的计数 → 先小范围枚举再回验"），20 道数论题里真正适用的可能只有 3–5 道。更关键的是它们在流中的**位置**：若分布在第 7/31/58/74 题，skill 在第 7 题产生后要等 24 题才有第一次复用机会，期间只是占着注入预算的噪声。

真正该量化的不是"有多少道数论题"，而是**"一条 skill 从产生到第一次真正适用，平均隔多少题"**。

### 15.3 决策：主题交错

作业允许三种任务流。取舍：

| 方案 | 复用机会 | GeneralCurator 有无材料 | 真实性 |
| --- | --- | --- | --- |
| 纯年份顺序 | 低 | 好（batch 内主题天然多样） | 最高 |
| 主题分段 | 最高 | **失效**（每 batch 单一 topic，提不出跨主题模式） | 最低 |
| **主题交错（采用）** | 中高 | 好 | 中 |

**构造规则**：每个 batch 的 8 题配成 `2 algebra + 2 number_theory + 2 combinatorics + 2 geometry`，各主题内部保持年份/题号顺序。效果：batch 内主题多样 → GeneralCurator 有跨主题材料；每个 topic 在**每个** batch 都出现 2 次 → topic skill 从产生到复用的间隔压到 1 个 batch。

主题分段被排除，因为它会废掉 general 层 —— 而 general/topic 双层是论文核心贡献之一。

⚠️ **README 必须主动回答的质疑**："构造任务流时用了全部题目的 topic 标注，算不算使用未来信息？" 答：不算。任务顺序是 benchmark 的固定属性，实验前一次性确定且对三个 arm 完全相同；agent 运行时仍只能看到当前题和 `H_i`。作业原文明确允许"主题交错"，只要求"实验前固定并用于全部对照"。别等 grader 问。

### 15.4 Day 2 必做：复用机会分析（零模型调用）

数据准备完成后、**跑任何实验之前**，算三个纯统计量：

1. **同 topic 相邻间隔分布**（中位数、p90）—— 交错设计下应稳定在 ~4 题。
2. **细粒度技法标签重复次数直方图** —— 用 20–30 个技法标签（`divisibility-counting` / `recursion` / `similar-triangles` / `generating-function` …）离线标注。**若 80% 的标签只出现 1 次，就地调整任务流，不要开跑。**
3. **理论复用上界** —— 假设 skill 完美，有多少道题"本该"被某条已有 skill 覆盖。

这三个数字直接进 README：它们把"可能出现的 null"提前转化为**已量化的实验条件**。即使最终打平，也能写"本任务流理论复用上界 X%，实测 skill 命中率 Y%" —— 这是分析，不是找补。

**兜底**：若上述统计显示复用机会仍太低，adaptation 收窄到 2 个 topic（algebra + number_theory），80 题集中在两类，细粒度重复率翻倍。代价是 held-out 出现 topic mismatch，需单独讨论。备选，非首选。

## 14. Day 1 阻塞项清单（必须在写任何 Task A 代码前完成）

1. `pip install -e . --no-deps` + 手工依赖，跑通 evo 路径 2 题并录制轨迹（供 Day 3 做 fixture）。
2. 实测 20 次 API 调用的延迟分布 + 24 并发是否限流 → 回填 §8 wall-clock、§9 排期。
3. 确定带年份元数据的 AIME 数据源。
4. 审计 `informalmath_verify` 的 GT 泄漏路径，确认闸 2/3 的正则能覆盖实际输出格式。
5. 核对 a-evolve 的 LICENSE。
