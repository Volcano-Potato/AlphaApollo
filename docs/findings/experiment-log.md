# 实验发现日志

跑真机时发现的问题、诊断过程和结论。按时间倒序，最新在上。

与其他文档的分工：
- `README_HARNESS.md` —— 交付物，写**结论**
- 本文件 —— 工作记录，写**怎么得出结论的**，包括走错的路
- git commit —— 写**每一处改动为什么**

写进来的每一条都必须有可复现的证据（日志路径、commit、实测数字）。**不写猜测。**

---

## 2026-09-13 · general 层的三轮演变：一个错误、两个症状、三次修复

### 起因

12 题冒烟跑了三轮，每轮之间改了代码。三轮的 general 层行为完全不同，而根因是同一个。

| | 第一轮 `s12` | 第二轮 `s12b` | 第三轮 `s12c` |
|---|---|---|---|
| 代码 | 修复前 | `064e799` + `e62ea41` | `8180fc8` + `ff3856e` |
| 墙钟 | 1354.7 s | 941.6 s | **663.7 s** |
| 丢题 | **3/12（25%）** | 0 | 0 |
| 解对 | 2/12 | 3/12 | 2/12 |
| 调用 | 93 solver + 20 mgmt | 122 + 27 | 109 + 26 |
| **general 层** | 3 条，**全是 topic 的逐字副本** | **0 条**（9 次 ADD 全被拒） | 2 条，跨 4–6 题，**但内容空洞** |

### 第一轮观察到的两个现象

**(a) general 技能是 topic 技能的逐字副本。** 同一条候选被两个 curator 各收一次：

| topic 层 | general 层 |
|---|---|
| `When solving problems requiring enumeration under constraints.` (`sk_0001`) | `when counting pairs or tuples under specific conditions` (`sk_0003`) |
| `When setting up coordinates for geometric problems with variable side lengths.` (`sk_0002`) | `when setting up coordinates for geometric problems` (`sk_0005`) |

三条 general 技能的证据分别是 `p_0`、`p_2`、`p_3` —— **三条单题教训被归档成了跨任务模式**。

**(b) 跨层改写。** `general_curator` 对 `sk_0001` 发了 MERGE 和 REVISE 并被接受，而 `sk_0001` 的文件里写着 `level=topic`。`_apply_one` 按 id 找到目标就改写，从不检查目标属于哪一层。两个 curator 于是把内容和证据累积进了同一条技能。

### 我当时的诊断 —— 错的

`GENERAL_CURATOR_PROMPT` 里早就写着 *"Each general skill must address a pattern seen in at least 2 (2+) different problems"*。我看到单题证据，判断是**模型忽略了指令**，于是在 `store.apply` 的 ADD 路径加了代码强制（`064e799`）。同时修了跨层改写（`e62ea41`）。

### 第二轮：规则生效了，但层死了

```
b0 ADD None general_curator ok=False reject=general_needs_two_problems
b1 ADD None general_curator ok=False reject=general_needs_two_problems   ×3
b2 ADD None general_curator ok=False reject=general_needs_two_problems   ×4
```

**9 次全拒，general 层空。** 查下去发现机制上不可能满足：

```
ADD: <candidate number>      ← 单个序号，且逐字复制那条候选的文本
每次 Reflect → 一个候选 → evidence 恒为一道题
```

general ADD 的证据**永远是 1 道题**。我的规则让 general 层结构上无法到达 —— 把一个退化的层变成了一个死掉的层。

### 真正的根因

我让两个 curator **共用同一套输出格式**（`_COMMON_FORMAT`）。

论文附录 E.2 / E.3 是**两套不同的契约**：

| | 论文原文 | 动作 | 粒度 |
|---|---|---|---|
| E.2 topic | *"For each proposal, choose ACCEPT, MERGE, or SKIP"* | 挑选候选 | 逐条 |
| E.3 general | *"**Create or update** a general skill only when a pattern appears **across multiple contexts**"* | **自己写一条** | 跨多个 context |

共用格式的结果是：general curator 只拿到了"从候选里挑一条"的动作。它**既凑不出跨题证据，也没有"写"这个动作** —— 于是复制成了它唯一能做的事。

**一个错误，两个症状。** 我之前把 (a) 和第二轮的全拒当成两件独立的事在修。

### 三次修复

| commit | 做了什么 | 够不够 |
|---|---|---|
| `064e799` | 加"general 必须跨 2 题"规则 | 规则忠于论文，**但当时不可满足** |
| `8180fc8` | 解析器支持 `ADD: 1,3` 多序号 + 可带自撰文本 | **提示词还在说用单数**，等于没生效 |
| `ff3856e` | 两个 curator 拆成各自的格式块 | 打通 |

`8180fc8` 的失败值得单独记：我教会了解析器，却把提示词留在 *"Use EXACTLY these forms: ADD: `<candidate number>`"* —— 那句 "EXACTLY" 才是模型会听的。**新能力只有在模型违背自己的格式说明时才可能触发。**

受控验证（qwen3-8b，三条确实共享同一失败模式的候选）：

```
ADD  evidence=['p_1', 'p_2', 'p_3']
     trigger: When a problem involves complex constraints or unbounded parameters, start with...
```

一条 ADD 引用全部三条候选、证据跨 3 题、trigger 是自撰而非复制。

### 第三轮：机制通了，质量塌了

```
b0  ADD    sk_0002  general_curator  src=[0, 1, 2, 3]
b1  ADD    sk_0005  general_curator  src=[4, 5, 6, 7]
b2  REVISE sk_0002  general_curator  src=[8, 9]
b2  SKIP   ×2
```

跨层重复消失了，ADD/REVISE/SKIP 都在用。但技能内容：

```
sk_0002 [general] "When solving any problem, ensure all steps are logically justified and verified."
sk_0004 [topic  ] "When solving a problem, ensure all steps are logically justified and verified."
sk_0005 [general] "When solving complex problems, ensure all steps are logically justified and verified."
```

**三条说的是同一句废话**，只差 "any problem" / "a problem" / "complex problems"。而这正是 `REFLECT_PROMPT` 明令禁止的：*"Filter aggressively. Output ACTION: NONE rather than proposing: generic advice such as 'read carefully' or 'double-check the work'"*。

`sk_0004` 的 topic 是 `mathematical_reasoning` —— 那不是主题，那是全部。topic 命名一并退化。

**从"具体而冗余"变成了"通用而无用"。** 逼 curator 找跨题共性，而 4 条来自不同主题的失败，最大公约数必然空洞。问题被从一端推到了另一端。

### 官方实现怎么做的（`a-evolve` `release/evo-harness` 分支，`evo_harness/cl_bench.py`）

**质量约束全在提示词，脚本只管预算和格式。**

`_execute_general_curation` 的全部代码层检查：

```python
if name and content and current_count < max_general:
    _write_general_skill(name, desc, content)
```

name 非空、content 非空、没超预算。**没有任何质量闸门。**

这些全是纯提示词，一条都没有代码强制：

> *- Each skill must address a pattern seen in **3+ different contexts**.*
> *- Be SPECIFIC and ACTIONABLE — **not generic advice like "read carefully"**.*

两处和本实现不同：

1. **官方的 general curator 不引用候选编号** —— 输出是 `NEW_GENERAL: <kebab-name>` + `DESCRIPTION` + `CONTENT`，直接看着失败摘要写。代价是 **general 技能没有 provenance**，追溯不到来源任务。本实现保留 evidence 链（作业要求"记录每个问题产生了哪些候选更新"），这点是刻意的偏离。
2. **官方要求 3+ contexts，本实现是 2+。** 论文正文只说 "multiple contexts"。

### 尚未回答的问题

官方那条"不要泛泛之词"同样是纯提示词，**大概率也拦不住** —— 但他们用 Claude Sonnet 4.5 做 curation，本项目用 qwen3-8b。

所以空洞技能到底是**方法的退化倾向**还是**模型能力不足**，目前**未知**。

可判定的实验：拿同一批候选用 qwen3-32b 跑一次 general curator。

- 32b 写得具体 → 是模型强度问题，不应加代码闸门，README 记录"curator 模型强度影响技能质量"
- 32b 同样空洞 → 是方法的退化倾向，值得加闸门并作为研究发现报告

**在这个实验做完之前不加任何质量闸门。**

> **已做，结论见下一条**：空洞不是 general curator 产生的，是被喂进去的。第一次测试因把空洞技能放进候选池而得出了相反的错误结论。

### 产物位置

三轮的 store 与日志都在 scratchpad（不入库）：
`s12_store` / `s12b_store` / `s12c_store`，日志 `s12.log` / `s12b.log` / `s12c.log`。
验收脚本 `verify_smoke.py <run_dir> <batch_size>`，7 条可证伪检查。

---

## 2026-09-13 · 两篇原始工作的参数设定，以及我偏离了哪些

Task C 要设参数，先把两边的原始值查清楚。**都取自实际文件，不是论文转述。**

### A. AlphaApollo 侧（solver）—— `examples/configs/vllm_informal_math.yaml`

| 参数 | 官方值 | 我的冒烟 | 性质 |
|---|---|---|---|
| `evolving_round` | **10** | 2 | ⚠️ 题内自进化被砍到 1/5 |
| `verifier_env_num` | **5**（奇数）| 1 | ⚠️ **多数判决被关掉** |
| `max_tokens` | **8192** | 2048 | ⚠️ AIME 推理链可能被截断 |
| `verifier_max_workers` | 5 | 1 | 仅速度 |
| `problem_max_workers` | 30 | 由 harness 的 `max_workers` 接管 | — |
| `policy_env_num` | 1 | 1 | ✓ |
| `max_steps` / `history_length` | 4 / 4 | 4 / 4 | ✓ |
| `memory_type` | simple | simple | ✓ |
| `python_code_timeout` | 300 | 60 | 轻微 |
| temperature policy / verifier | 0.7 / 0.4 | 0.7 / 0.4 | ✓ |
| 模型 | `qwen3_4b_inst` | `qwen3-8b` | 平台无 4B，已记录 |

配置注释原文：*"make sure the number of verifier environment to be **odd** to ensure the **majority judgment**"*。用 1 个 verifier 等于关掉多数判决，而 **verifier 的判词正是 Reflect 的主要输入**。

### B. Evo-Harness 侧 —— 论文 Appendix F vs 参考实现 `evo_harness/cl_bench.py`

| 参数 | 论文 App.F | 参考实现默认值 | 我的 |
|---|---|---|---|
| batch size | 16 | **10** (`--batch-size`) | 8 |
| batch workers | — | 4 (`--batch-workers`) | 4 |
| max general skills | **5** | **10** (`--max-general-skills`) | 5 |
| max skills per topic | **5** | 5 (`--max-skills-per-context`) | 5 |
| selector 模型 | Claude Sonnet 4.5 | Sonnet 4.5 (`--selector-model`) | qwen3-32b |
| curator 模型 | — | Sonnet 4.5 (`--curator-model`) | qwen3-8b |
| propose(Reflect) 温度 / max_tokens | — | **0.3** / 1024 | **0.7** / 2048 |
| topic curate 温度 / max_tokens | — | **0.0** / 2048 | **0.7** / 2048 |
| general curate 温度 / max_tokens | — | **0.0** / 4096 | **0.7** / 2048 |
| select 温度 / max_tokens | — | **0.0** / 1024 | 0.0 / 512 |
| feedback level | — | 2（rubric count）| standard |
| seed / 重复次数 | 42 / **结果取 3 次平均** | — | 1234 / 1 次 |
| general curator 触发门槛 | — | **本批失败数 ≥ 3** | 无 |
| 注入预算 | — | **"Select FEWER skills (0-5)"，纯提示词，无 token 上限** | b=6 / general≤3 / topic≤4 / **800 token，代码强制** |

论文 App.F 说 general 和 per-topic 都是 5，而参考实现的 `cl_bench` 默认 general=10 —— 两者不一致，可能是 benchmark 相关。

### C. 我的偏离，分成两类

**是 bug，该改：**

1. **管理调用全跑在 temperature 0.7** —— `mgmt_agent = Agent(cfg_bundle["policy_model_cfg"])` 直接继承了 policy 的温度。参考实现 curate 用 **0.0**、propose 用 **0.3**。0.7 下做 curation 意味着同样的候选每次给出不同决策，直接损害 Task C 的可比性。
2. `verifier_env_num=1` 关掉了多数判决 —— 这不是"简化"，是改变了 baseline 的行为。
3. `max_tokens=2048` vs 官方 8192 —— AIME 推理链很长，**这可能才是解题率卡在 2–3/12 的原因**，而我一直归因于模型能力。

**是刻意偏离，要在 README 声明：**

| 偏离 | 理由 |
|---|---|
| batch 8（论文 16 / 参考 10）| 149 题在 16 下只有约 9 个更新点，看不出增长趋势 |
| selector/annotator 用 qwen3-32b（论文 Sonnet 4.5）| 手头没有 Sonnet；32b 是可得的最接近选择 |
| 注入预算**代码强制** + token 上限（参考实现纯提示词、无 token 上限）| 作业 Task A 明确要求"注入内容受可配置的数量或 token budget 限制" |
| 保留 general 技能的 evidence 链（参考实现没有）| 作业要求"记录每个问题产生了哪些候选更新" |
| general 要求跨 2 题（参考实现是 3，且纯提示词）| 详见上文 general 层三轮演变那条 |

### D. 尚未决定

`evolving_round` 10 vs 2、`max_tokens` 8192 vs 2048、`verifier_env_num` 5 vs 1 三项叠加，单题成本可能是现在的 5–10 倍，149 题 × 3 arm 会很贵。

**在跑一轮官方参数的诊断之前不定。** 现在的 2–3/12 解题率究竟是模型上限还是我配出来的地板，没有这个数就只能猜。

---

## 2026-09-13 · 空洞技能的源头不是 general curator（一次被自己污染的实验）

承接上一条的开放问题：空洞技能是方法的退化倾向，还是 qwen3-8b 能力不足？

### 第一次测试 —— 结论错的

用第三轮真实产出的 3 条 topic 技能重建候选池，8b 与 32b 各跑一次 general curator。两个模型输出本质相同的空洞内容（"ensure each step is logically sound and verified" / "logically justified and verified"），看上去支持"方法退化，与模型无关"。

**这个实验是污染的。** 候选 3 本身就是第三轮那条空洞技能（`sk_0004`，`topic="mathematical_reasoning"`，trigger *"When solving a problem, ensure all steps are logically justified and verified"*）—— 我把退化喂回去测退化。

顺带：我写的空洞检测正则只标出了 8b、漏掉了 32b，纯粹因为用词差一个字（"justified" vs "sound"）。**这类检测器不可靠，不要用它判定。**

### 第二次测试 —— 干净候选池

只保留两条**内容具体**的真实候选（几何对称、对数定义域），两个模型各跑两次：

| 模型 | trigger |
|---|---|
| 8b · 1 | Recognize inherent structure or constraints before diving into algebraic manipulation |
| 8b · 2 | Before solving a complex problem, analyze its structure and constraints |
| 32b · 1 | pause before acting on the first representation |
| 32b · 2 | Look for inherent structure (symmetry, domain rules) before diving into algebraic manipulation |

**四次全部是站得住的跨题策略**，不是泛泛之词。"先识别结构再动手代数操作"是从"几何对称"和"对数定义域"两条候选里真正抽象出来的数学启发式。

32b 略具体（"pause before acting on the first representation"），但 8b 同样可用。

### 结论

**空洞不是 general curator 产生的，是它被喂进去的。**

```
Reflect 产出空洞候选              ← 真正的源头
      ↓
topic curator 接受了它            ← 这里本该拦住（提示词已写"不要泛泛之词"）
      ↓
general curator 放大并复制成两份   ← 我之前以为的病灶
```

这也解释了 topic 名字为什么会退化成 `mathematical_reasoning`：一条内容空洞的技能，本来就没有具体主题可归。

### 对上一条开放问题的回答

- 不是模型能力问题 —— 8b 在干净输入上表现可用
- 不是 general curator 的退化倾向 —— 它在干净输入上产出合理
- **是空洞候选在链条里自我强化**：一旦一条进了 harness，后续 curation 会把它放大

### 下一步（未做）

闸门应加在 Reflect 输出或 topic curator 接受处，而不是 general 层。但在加之前需要先确认：

- 空洞候选出现的频率（三轮里只观察到 1 条，样本太小）
- 是否值得为它写代码 —— 官方实现同样只有提示词约束，加闸门是**偏离参考实现**，必须在 README 里标注为主动改动

### 本次实验的样本量

候选池 n=1，每模型 2 次。**很薄**。定性差异（污染池→空洞、干净池→具体，4/4 一致）明显，但不足以给出频率估计。

### 顺带发现的可审计性缺口

**原始候选没有被持久化。** Reflect 产出后被 curator 消费即丢弃，`harness_log` 只记 `skill_id` / `op` / `accepted` / `reject_reason` / `source_problems`，**不记候选内容**。所以一条被拒候选说了什么，事后无法追溯 —— 本次实验只能用"被接受的 topic 技能"反推候选，这本身就是个 workaround。

作业要求"保存每个问题产生了哪些候选更新"，目前只做到了记录**决策**，没记录**内容**。

---

## 2026-09-13 · 丢题 25% 的根因是本机代理，不是 DashScope

第一轮冒烟丢了 3/12 道题，全部是 `openai.APIConnectionError` ← `ConnectError: [Errno 61] Connection refused`，并且 p8 的 Reflect 调用也失败，导致 **batch 2 产出 0 条 harness edit，而整个 run 仍然 `exit=0`**。

开发机的 `ALL_PROXY` / `HTTP_PROXY` / `HTTPS_PROXY` 全部指向本地 SOCKS 代理，所有流量默认走它。同一端点实测：

| | 延迟 |
|---|---|
| 经代理 | 1.00 s |
| 直连 | **0.38 s** |

把 `dashscope.aliyuncs.com` 加入 `NO_PROXY` 后，第二、三轮**丢题均为 0**，墙钟从 1354.7 s 降到 941.6 s 再到 663.7 s。

两条结论：

1. 正式跑 149 题**必须**先排除代理（README §2.2b 已记录命令，大小写两个变量都要设）。
2. **丢题率必须作为一项指标记录**。否则"某条臂成绩低"可能只是它那轮网络更差，而退出码完全看不出来。

---

## 2026-09-13 · `prepare_harness_stream` 的 CLI 坏了，且坏得像成功

`main()` 直接构造 `Agent`，没有任何东西注入 `extra_body`。DashScope 对 qwen3 系列的非流式调用强制要求 `enable_thinking: false`，于是 **149 次标注调用全部 400 失败，脚本照常 `exit 0`，topic 列静默全空**。

之前能跑出结果，是因为我在外面手动包了一层 `install_accounting`。也就是说：**任何人照 README 的命令跑一遍，都会得到一份 topic 全空的题流，并且以为成功了。**

修复（`ae625c6`）：标注跑在 `install_accounting` 里（既注入 `extra_body`，也把调用计入 `offline_labeling`），且**一次标注全军覆没直接抛异常**而不是交付一份不可用的数据。

---

## 共同模式：静默失败

到目前为止，真机暴露的缺陷几乎全是同一类 —— **出错了，但退出码是 0**：

| 缺陷 | 表现 | commit |
|---|---|---|
| `data_source` KeyError | 每题都挂，harness 却从空轨迹编出技能 | `2aedcb1` / `70e77b6` |
| 虚构的 `step_outputs` schema | 对真实 payload 返回全零 | `9e1b554` |
| 线程池吞掉 role | 9/38 次调用归入 `unscoped`，成本分离报错 | `62b024b` |
| wandb 无凭据 | 抛 `KeyboardInterrupt`（`BaseException`）杀进程，连 jsonl 兜底都没写 | `9f51c21` |
| 冻结 arm 加载空 store | 退化成 Baseline，三条曲线重合，**与真 null result 无法区分** | `c6cbd9e` |
| 标注全军覆没 | topic 列全空 | `ae625c6` |
| 代理断连 | 丢 25% 题，batch 2 零产出 | 环境 |

**这些都躲过了 400 多个通过的单元测试。** 两个成因：

1. **手写 fixture 和手写实现共享盲点** —— 我按想象中的数据形状写测试，也按想象写实现，两边一致地错。补救是 `tests/harness/fixtures/real_problem_payload.json`，第一份取自真实执行的 fixture。
2. **韧性机制掩盖配置错误** —— 每一层降级都正常工作，结果是一个彻底坏掉的 run 安静地"完成"。补救是首 batch 断路器和 frozen-空状态前置检查，它们是**故意不降级**的地方。

处理原则现在是固定的：**要么大声失败，要么在代码里强制，不依赖调用者记得做对。**
