# 跨问题 Skill Harness（Evo-Harness on AlphaApollo）

本文档描述在 AlphaApollo 之上新增的**跨问题 skill 演化层**（TMLR mini-project：Task A 机制 / Task B 编译闭环 / Task C 对照实验）。

> **只想看实验？** 读 [`TASK_C_REPORT.md`](TASK_C_REPORT.md) —— 那是一份自足的实验报告：环境与数据、两种 memory 的边界、skill 机制、三组实验的命令 / 结果 / **日志**、真实演化记录、失败尝试与限制、上游归属，外加六项验收重点的逐条自查。本文档是系统的完整设计论证，§8 给出同一批结果的表格。

> **当前状态（务必先读）**
>
> - Task A（skill 机制）与 Task B（context→harness 编译闭环）的代码与测试**已完成**：`tests/harness/` 共 **619 个测试全部通过**（本机实测 `619 passed in 4.64s`）。
> - **Task C 的三组对照实验已全部跑完**：6 个 run（3 臂 × adaptation/held-out），144 + 30 题，**丢题 0**。结果见第 8 节，产物在 `outputs/harness/`。
> - **主结果是一个 null result，且三臂之间没有一项差异是统计显著的**（McNemar 精确检验最小 p = 0.125）。第 8.9 节按任务书要求对它做了分析，而不是把它当成失败藏起来。
> - 尚未完成的内容集中列在第 12 节，请以该节为准。

---

## 1. 这是什么 / 与上游 AlphaApollo 的关系

AlphaApollo 原生的 `informal_math_evolving` 环境，是在**单道题目内部**做多轮自我演化：同一道题反复求解 → verifier 反馈 → 把自己此前的解法写回 prompt → 再解一次。环境 `reset()` 在每道新题时触发，**题与题之间不留下任何东西**。

本项目新增的是一个**跨问题**层：把一道题跑完之后的轨迹、工具输出、成败标签与 verifier 反馈，压缩成自然语言的可复用 **skill**，存进一个持久化的 harness，并在**后续**题目求解前注入 policy 的 system message。策略模型、verifier、工具、生成参数在整个过程中**完全冻结**，唯一变化的是 harness。

### 1.1 代码归属与许可证

| | |
|---|---|
| 上游代码 | [`tmlr-group/AlphaApollo`](https://github.com/tmlr-group/AlphaApollo)（内含 vendored `verl` / `verl-agent`） |
| 许可证 | Apache License 2.0，见仓库根目录 `LICENSE` 与 `Notice.txt`（Bytedance / NTU verl-agent / HKBU + TMLR Group） |
| 上游 README | `README.md` **原样保留，未作任何修改** |
| 方法参考 | Evo-Harness 论文（arXiv:2608.15071）与 [`A-EVO-Lab/a-evolve`](https://github.com/A-EVO-Lab/a-evolve) 的 `release/evo-harness` 分支。本项目**不 import、不依赖** a-evolve，仅作设计参考。 |

新增代码统一带 Apache-2.0 头，版权署 `Copyright 2026 TMLR Group`。

### 1.2 改动边界：上游**源码**零修改

对比 fork 基线 commit `712a04d`（上游最后一个 commit）到 `HEAD`：

```
$ git diff --stat 712a04d..HEAD
140 files changed, 26980 insertions(+), 2 deletions(-)
```

（其中 63 个文件、约 8.0k 行是 §8 引用的 Task C 产物 —— `outputs/harness/` 下的汇总指标、导出的 harness 与完整编辑日志、选择日志，以及 `docs/findings/transfer-cases.md` 逐行分析的那 6 份轨迹。轨迹全集与 wandb 目录仍然不入库，见 `.gitignore` 末尾的说明。）

其中 **被修改（`M`）的上游文件只有两个**。一个是 `.gitignore`（末尾追加若干反向排除，把上面那批 Task C 产物放进库，其余 run 输出仍然排除）；另一个是 `pyproject.toml`，改动内容只是加了一段 pytest 配置：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = [".", "alphaapollo/core/generation"]
```

其余 138 个文件全部是新增（`A`）。没有任何一个上游的求解/环境/verifier/工具文件被触碰 —— 这是刻意的约束：Task C 要求三组实验共享**同一个冻结 solver**，改上游文件就无法论证"baseline 还是原来那个 baseline"。

需要改上游行为的两处，都通过**类级 monkey-patch**（`accounting.install_accounting` 补 `Agent.get_action_from_gpt`）在运行期完成，`uninstall()` 后复原（`alphaapollo/core/harness/accounting.py`）。

### 1.3 新增文件一览

```
alphaapollo/core/harness/                     # Task A/B 全部机制（不含驱动）
├── schema.py          Skill / CandidateMemory / SkillEdit + markdown 双向序列化
├── store.py           SkillStore：磁盘持久化、两份 JSONL 日志、edit 应用与容量上界
├── guard.py           防泄漏 / 格式 guard（本包中唯一允许接触 ground_truth 的模块）
├── render.py          把选中的 skill 渲染成 system message；count_tokens 预算启发式
├── selector.py        模型驱动的 Select（论文 Algorithm 1 line 5），预算在代码里强制
├── reflect.py         Reflect：轨迹 → 候选 skill（论文 Appendix E.1）
├── evolver.py         TopicCurator / GeneralCurator（Appendix E.2 / E.3）
├── topic.py           题目 topic 的模型标注（仅供分析口径，方法本身不读）
├── arms.py            Task C 三条臂：Baseline / RawExperience / EvoHarness
├── accounting.py      按 role 分桶的调用与 token 计数 + seed 注入
├── tracker.py         wandb + jsonl 双写（wandb 永远不能让 run 失败）
├── wandb_backfill.py  跑完之后把 analysis.py 的结果回填到原来那个 wandb run 上
├── export.py          导出最终 harness、演化日志与汇总数字
└── loader.py          保留 topic/year 的 stream loader（上游 loader 会丢掉这些字段）

alphaapollo/core/generation/evolving/evolving_harness_main.py   # Task B 驱动器 + CLI
alphaapollo/data_preprocess/prepare_harness_stream.py           # AIME 数据流构建
tests/harness/                                                  # 619 个测试
docs/design/cross-problem-skill-harness-design.md               # 设计文档（部分已过时，见 §11）
requirements-harness.txt                                        # 两个额外依赖
scripts/run_tests.sh
```

---

## 2. 环境安装与硬件

### 2.1 硬件与模型（本项目实际使用的配置）

| 项目 | 实测值 |
|---|---|
| 机器 | MacBook Air M4 / 24 GB，**无本地 GPU** |
| 模型服务 | 全部走托管 API：阿里云 DashScope 的 OpenAI 兼容端点 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Solver / verifier（冻结） | `qwen3-8b` |
| Selector / topic annotator | `qwen3-32b`（与 solver 分开配置，见 §4.2、§6.2） |
| 并发上限（实测） | **12 并发安全；16 开始被限流；24 丢失约 40% 请求** |
| 单次调用延迟（实测） | 中位数 16.5 s，约 1200 output tokens |

任务书建议的 `Qwen3-4B-Instruct-2507` 在 DashScope 上**不存在**，因此 solver 改为 `qwen3-8b`。

**DashScope 的一个硬性要求：** qwen3 系列的 hybrid-reasoning 模型在**非流式**调用下，请求体里必须带 `enable_thinking: false`，否则直接报错。上游 `Agent` 无法表达这个参数，所以由 `accounting.install_accounting(..., extra_body=...)` 注入；配置项是 `harness.extra_body`。默认不传，因此指向 vLLM 端点时请求体与上游逐字节一致。

### 2.2 安装

```bash
conda create -n alphaapollo python==3.12 -y && conda activate alphaapollo
cd AlphaApollo && bash installation.sh          # 上游依赖（torch / vllm / ray / pip install -e .）
pip install -r requirements-harness.txt          # 本项目额外的两个依赖
```

`requirements-harness.txt` 只有两项，每一项都是"没有它真实运行会挂"才加进来的：

- **`scipy`** —— AlphaApollo 的 Python 计算工具用 `sys.executable` 执行模型写的代码，注入的前置代码只有标准库，模型自己写 `import scipy`。一次真实 rollout 观察到模型调用 `scipy.optimize.root_scalar` 后抛 `ModuleNotFoundError`；工具把它当作普通 tool error 上报，于是 run 不会失败，只是那道题被判错 —— **静默地拉低每一条臂的准确率**。
- **`httpx[socks]`** —— `openai.OpenAI(...)` 在**构造时**就建立 httpx transport 并解析代理环境变量。机器上有 `ALL_PROXY=socks5://...` 时，构造一个 `Agent` 就会在发出任何请求之前抛 `ImportError`。

（本项目开发机使用的是 conda env `alphaapollo-dev`。）

### 2.2b 代理：跑之前务必把 DashScope 排除掉

开发机上 `ALL_PROXY` / `HTTP_PROXY` / `HTTPS_PROXY` 全部指向本地 SOCKS 代理，**所有流量默认都会走它**。一次 12 题烟囱测试因此丢了 3 道题（25%），全部是代理侧的 `ConnectError: [Errno 61] Connection refused`，而非 DashScope 拒绝服务。

实测同一个端点：

| | 延迟 |
|---|---|
| 经代理 | 1.00 s |
| 直连 | **0.38 s** |

DashScope 是国内端点，经代理既慢又不稳。跑实验前把它排除：

```bash
export NO_PROXY="$NO_PROXY,dashscope.aliyuncs.com"
export no_proxy="$no_proxy,dashscope.aliyuncs.com"
```

注意大小写两个变量都要设 —— 不同库读的不是同一个。这不影响 `httpx[socks]` 那条依赖：`openai.OpenAI(...)` 在构造时仍然会解析代理环境，缺 socks 扩展照样在发请求之前就抛 `ImportError`。

### 2.3 API key

**任何配置文件里都不得出现 API key。** 所有配置写 `api_key: ""`，`Agent.__init__` 会回落到环境变量：

```python
api_key = vllm_config.get("api_key") or os.environ.get("OPENAI_API_KEY", "EMPTY")
```

运行前：

```bash
export OPENAI_API_KEY=<your key>
```

### 2.4 跑测试

```bash
# scripts/run_tests.sh 的内容就是：
#   PYTHONPATH="$PWD:$PWD/alphaapollo/core/generation" python -m pytest tests/harness "$@"
bash scripts/run_tests.sh
# 或（pyproject.toml 的 [tool.pytest.ini_options] 已经配好 pythonpath）
python -m pytest tests/harness/ -q
```

实测输出：

```
619 passed in 4.64s
```

各文件测试数：`test_reflect` 59、`test_configs` 56、`test_wandb_backfill` 48、`test_driver` 45、`test_evolver` 43、`test_store_apply` 38、`test_arms` 38、`test_loader` 31、`test_resume` 29、`test_tracker` 27、`test_analysis` 25、`test_prepare_stream` 23、`test_accounting` 23、`test_selector` 20、`test_topic` 19、`test_runtime_cleanup` 19、`test_guard` 18、`test_store_select` 16、`test_store_persistence` 12、`test_export` 11、`test_schema` 10、`test_render` 5、`test_smoke` 4。

单元测试**不触碰网络栈**：`Agent.__init__` 会真的构造 `openai.OpenAI(...)`，测试里用 monkeypatch 换掉 agent 模块内的 `OpenAI` 名字（见 commit `5580627`）。这是因为曾经出现过"同一套测试在 A 机 97 passed、在 B 机 84 passed + 13 errors"，差别纯粹来自 shell 里有没有 SOCKS 代理变量。

---

## 3. 三层记忆的边界（本项目最核心的概念区分）

AlphaApollo + 本项目一共存在**三层**记忆。它们的生命周期、注入通道和实现位置**互不重叠**：

| | 层 1：轮内 step history | 层 2：题内 solution memory | 层 3：跨题 skill harness（本项目） |
|---|---|---|---|
| 作用范围 | 一个 evolving round 内的若干 step | 一道题的若干 evolving rounds | 整条题流，跨越所有题目 |
| 存了什么 | 该 episode 的 `(text_obs, action)` 序列 | 本题此前的**完整解法**与 verifier 反馈 | 自然语言的可复用**操作性知识**（适用条件 / 策略 / 要避免的失败模式） |
| 实现 | `core/environments/memory/memory.py`（`SimpleMemory` 等），由 env manager 持有 | `evolving_main.run_problem` 内部 `solution_memory = NDimensionalMemory(...)`（`evolving_main.py:488`），**在 run_problem 里构造**，因而每题重建 | `alphaapollo/core/harness/store.py` 的 `SkillStore`，磁盘持久化 |
| 注入通道 | prompt 模板的 `{memory_context}` 槽（**user message**） | prompt 模板的 `{previous_solutions}` 槽（**user message**） | **system message**（`policy_agent.system_prompt`） |
| 生命周期 | env `reset()` 清空 | 每道新题重建 | **跨题累积，跨进程持久**（held-out 评测在另一个进程里 reload） |
| 能否看到原题/答案 | 能（就是本题） | 能（就是本题） | **结构性地不能**，见 §5 |

### 3.1 为什么 system message 是干净的通道

上游的 policy prompt 模板（`core/environments/prompts/informal_math_evolving.py`，如 `INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_NO_HIS`）只插值 `{question}` / `{previous_solutions}` / `{memory_context}`，**全部进 user message**；上游配置里 `policy_model_cfg.system_prompt` 是空串，而 `utils/agent.py:37` 是 `if self.system_prompt:` —— 空串意味着 system message **整条消失**。

也就是说：system 通道在上游是完全空置的。本项目把 harness 放进这个通道，于是层 3 的注入与层 1/层 2 在消息结构上天然不重叠，既不会覆盖也不会被上游模板意外读到。

同时这也带来一个公平性陷阱：如果 Baseline 臂返回空串（system message 消失）而另两条臂返回真实文本（system message 存在），三条臂差的就不只是内容、而是**消息结构**了。所以 `render.render_harness([])` 返回常量 `NEUTRAL_SYSTEM_PROMPT = "You are a competition mathematics solver."` 而不是 `""`，三条臂在冷启动时的消息形状完全一致（`arms.py` 模块 docstring 第 1 条不变式）。

### 3.2 边界由 AST import guard 强制，不是靠约定

`tests/harness/test_smoke.py` 用 `ast.parse` / `ast.walk` 扫描 `alphaapollo/core/harness/` 下**每一个** `.py` 文件（包括 `__init__.py`），只要出现对 `alphaapollo.core.environments.memory` 的 import 就让测试失败。

它检测：普通 import、带别名的 import、from-import 名字或子模块、from-import 父包的 `memory` 属性，以及**相对 import**（`from ..environments.memory import SimpleMemory`）—— 相对 import 的 `node.level` 会先解析成绝对路径再比较。它**不**误报注释、docstring 或字符串字面量里出现的同名文本（`harness/__init__.py` 的 docstring 里就写着这个完整模块名）。

为什么要这么较真：最早的版本用 `pkgutil.iter_modules()` + 子串搜索，同时存在**漏报**（`iter_modules` 从不返回包自己的 `__init__.py`，而那正是唯一提到边界的文件）和**误报**（docstring 里的说明文字会触发）。commit `35a811b` / `81a90cb` 记录了完整推理。

驱动器 `evolving_harness_main.py` 同样从不 import 该包。

---

## 4. Skill 的表示 / 选择 / 更新 / 预算 / 防泄漏

### 4.1 表示

一个 skill 就是一个磁盘文件 `<store_root>/skills/<slug>--sk_NNNN.md`，markdown + JSON frontmatter（`schema.skill_to_markdown` / `skill_from_markdown` 双向可逆）：

```markdown
---
id: "sk_0001"
name: "when-counting-pairs-or-tuples-under-specific-conditions"
level: "topic"
topic: "counting_with_constraints"
evidence: ["p_0", "p_5", "p_6", "p_7"]
created_at: 0
revised_at: [1]
n_selected: 6
n_selected_success: 1
n_tokens: 46
---
## When to use
When solving problems requiring enumeration under constraints.

## Strategy
- Clearly define constraints and valid cases.
- Use systematic counting methods.
- Verify with alternative approaches.

## Avoid
Omitting logical steps and intermediate results.
```

三个自然语言段落与论文的 trigger / rule / evidence / scope 模式同构：

- **`## When to use`（trigger）** —— 适用条件。它同时**就是检索键**：selector 看到的 harness 目录只列 `[id] (scope) trigger`。
- **`## Strategy`（lesson）** —— 可复用的操作策略。
- **`## Avoid`（failure_mode）** —— 要避免的失败模式。

`render.render_harness` 注入给 policy 的**只有这三段**，所有簿记字段（`id` / `evidence` / 计数器）永不进入模型上下文。

两层结构：

- `level: "general"` —— 跨 topic 的通用策略，容量 `Caps.general = 5`（论文默认）。
- `level: "topic"` —— 某个 topic 内的局部流程，每个 topic 容量 `Caps.per_topic = 5`（论文默认）。

**为什么这个表示适合 AlphaApollo 的数学推理环境：**AIME 题目之间共享的不是答案、也不是完整解法，而是"什么时候该先枚举一个小范围"、"等角多边形不要默认边长相等"这一类**操作条件 + 策略 + 陷阱**。三段式正好把这三件事分开写，且都是 policy 可以直接照做的自然语言；而 trigger 单独成段，使得"这条 skill 该不该被这道题读到"可以在**不读完整条 skill** 的情况下判断，这正是预算控制所需要的。另外，文件即 skill（a-evolve 的 workspace 契约思想）让最终 harness 本身就是一份可读的交付物，人可以直接翻 `skills/` 目录做 case study。

`name` 由 trigger 文本 slugify 而来，不是 `f"{level}-{id}"` —— 后者会把 `topic-sk_0007` 这种内部 id 当噪声渲染进模型上下文，且生成的文件名不可读。文件名本身由 **id** 保证唯一并做路径清洗（`_safe_filename`），因为 `name` 是 LLM 产出的、不可信：曾经直接用 `name` 当文件名，导致两个不同 id 的 skill 静默覆盖同一个文件（内存 dict 还留着两个，reload 后不一致且无任何报错），以及 `../../pwned` 这样的名字能写到 `skills_dir` 之外（commit `f475ff5`）。

### 4.2 选择（Select）

**模型驱动**，对应论文 Algorithm 1 line 5 与 Appendix F："For harness selection, we use Claude Sonnet 4.5 across all experiments to retrieve relevant skills from the current harness before task execution."

`selector.select_skills(agent, question, skills, budget)`：把题面 + 整个 harness 的 `[id] (scope) trigger` 目录交给 selector 模型，让它按有用性排序回若干 id，或回 `NONE`。

**预算在代码里强制，不交给模型。** 模型只决定"选哪些、什么顺序"；`apply_budget()` 之后按确定性规则裁剪：

| 预算项 | 默认值 | 含义 |
|---|---|---|
| `budget.b` | 6 | 最多注入几条 |
| `budget.general_max` | 3 | general 层配额 |
| `budget.topic_max` | 4 | topic 层配额 |
| `budget.tokens` | 800 | 累计 token 上限 |

超预算的 skill 被**跳过而非终止遍历**，所以排名靠后的小 skill 仍可用掉剩余额度；单条就超过总预算的 skill 永不入选。理由写在 `selector.py` 的 docstring 里：Task A 要求注入内容遵守可配置的 count/token 预算，而"告诉模型有预算"的模型有时会超 —— **模型可以忽略的指令不是约束**。

**失败降级为空选择。** selector 调用抛异常 → 该题用中性 system prompt 跑完，结构上与 Baseline 完全一致，绝不把整条 run 带崩（Task B 硬约束）。

selector 模型由 `harness.selector_model_cfg` 单独配置（本项目用 `qwen3-32b`），默认回落到管理侧 agent。

> 早期版本用的是确定性的词重叠打分 + 硬 topic 过滤（`store.select()`）。它被放弃了，因为它**隐式要求每道题都带 topic 标注** —— 而这是方法本身并不具备的依赖（commit `f591906`）。`store.select()` 仍保留在代码里作为无依赖的参考实现，多个 store 层测试用它在无模型的情况下验证排序；**但实验走的是 `selector.select_skills`**，`store.py` 的 docstring 里对此有显式说明。

### 4.3 更新（Reflect → Curate → Apply）

三段式，对应论文 Appendix E.1 / E.2 / E.3：

**① Reflect（`reflect.py`）** —— 每道**失败**的题一次调用（成功的题不 reflect，对应论文 eq. 6）。输入是白名单化的上下文：`existing_topics` / `final_answer_given` / `outcome` / `verifier_feedback` / `tool_errors` / `reasoning_excerpt` / `round_count` / `related_skills`。输出 `CandidateMemory(trigger, lesson, failure_mode, scope_hint, topic, action_hint∈{NEW,ENHANCE}, target_id)`。

- `related_skills` 传的就是 selector 为这道题选中的那几条（Appendix E.1 的 "related existing skills"）。不额外花钱，而且能避免模型反复提出 harness 里已有的东西、把 curator 的固定预算耗在近重复上。
- **`topic` 是模型在输出侧命名的，不是数据集标签**。Appendix E.1 的输入清单里没有 topic，指令是 "Propose a reusable skill. Choose a broad topic"。早期版本把它当输入，凭空制造了对逐题 topic 标注的依赖（commit `4015c8b`）。传入 `existing_topics` 只是为了让模型**复用**已有名字而不是造同义词 —— 否则 bucket 会碎成"每个 topic 一条 skill"，topic 层就不成其为层。
- 解析器保守：`SCOPE` / `TRIGGER` / `LESSON` / `AVOID` 任一缺失，或显式 `ACTION: NONE` / `SCOPE: none`，一律返回 `None` 而不是半成品候选。

**② Curate（`evolver.py`）** —— 每个 batch 结束时，两个 curator 各自发一次模型调用：

- `TopicCurator`：只看**某一个 topic** 的既有 skill 和本 batch 属于该 topic 的候选。
- `GeneralCurator`：看 general 层的既有 skill 和本 batch 的**全部**候选（跨 topic 的模式，单 topic 视角根本看不出来）。

输出 `ADD | MERGE | REVISE | DELETE | SKIP`。payload 上的 layer 字段（`scope_hint` / `topic`）由调用方 curator **无条件覆写** —— 模型对"这条属于哪一层"的意见从不采信，只采信它写的内容。

任何模型调用失败或解析失败 → 返回 `[]`（no-op）。候选为空时**不发调用**。

**③ Apply（`store.apply()`）** —— 两阶段：先应用 `DELETE / MERGE / REVISE / SKIP`，再把 `ADD` 对**阶段一之后**的占用量做容量检查。这不是优化而是必需：curator 常给出"删掉那条弱的、加一条更好的"，若按列表顺序单遍执行，满了的 harness 只会一直增长、永远不会换血。

单条 edit 抛任何异常都被**逐条捕获**，不会中断同 batch 的其余 edit、更不会波及底层 baseline。

### 4.4 增长上界

| 机制 | 位置 |
|---|---|
| `Caps.general = 5` / `Caps.per_topic = 5`（论文默认） | `store.capacity_full`，代码强制，不只是写在 prompt 里 |
| 单条 skill 长度上限 | `guard.validate_skill`：`lesson` ≤ 60 词（`lesson_too_long`），整条 ≤ 200 词（`skill_too_long`） |
| 注入预算 | `selector.apply_budget`，见 §4.2 |
| **general 层必须跨 ≥2 道题** | `store.py:405`，reject reason `general_needs_two_problems` |
| **curator 只能改自己那一层** | `store.py:379`，reject reason `wrong_layer` |

后两条都是**真实运行中观察到模型违反 prompt 指令**之后补的，见 §10-D。

### 4.5 防泄漏设计

一条 skill 里**永远不能**出现原题文本、ground-truth 答案或完整解法。这不靠自觉，而是靠四道结构性关卡：

**① 类型层面的白名单。** `build_reflect_context(*, existing_topics, final_answer_given, outcome, verifier_feedback, tool_errors, reasoning_excerpt, round_count, related_skills)` —— 全部是逐个具名的关键字参数，**没有 question 参数、没有答案参数、没有 `**kwargs`、没有 dict 逃生口**。想把题面或答案塞进 skill，在这个函数的签名上就做不到。`reflect.py` 模块本身也被测试用 `inspect.getsource` 扫描，禁止出现 ground-truth 标识符。

**② GT 通道清洗。** AlphaApollo 自己的验证工具（`core/tools/informalmath_verify.py:107,176`）会把 `Matches ground truth: True` 这样一行直接写进工具返回文本，`env.py:106` 再把它包进 `<tool_response>` 喂回 policy。`sanitize_feedback()` 按**整行删除**这个通道（该行是工具独立输出的，行级删除精确且不伤及上下文），并且在 `build_reflect_context` 内部就执行，而不是留给下游记得调用。

另外 `extract_result` 只从 `tool_payload` 的 `stderr` / `run_status` 取 tool error，**绝不取 `stdout` 或 `raw_observation`** —— 因为 `Matches ground truth: ...` 正是打印在 stdout 里的。结构性排除优于事后正则清洗（commit `9e1b554`）。

**③ Guard 拒绝（`guard.py`，本包中唯一允许接触 `ground_truth` 的模块，且只用于拒绝、绝不写回任何返回值/日志/异常消息）。** 六个固定 reject reason：

| reject reason | 规则 |
|---|---|
| `empty_section` | 三段中有空段 |
| `non_english` | 非 ASCII 字符占比 > 5% |
| `lesson_too_long` | lesson > 60 词 |
| `skill_too_long` | 整条 > 200 词 |
| `question_overlap` | 与本 batch 任一题面有 8-gram 重合 |
| `answer_leak` | 某一行既含等于某个 ground truth 的数字，**又**含断言线索词（`answer` / `solution` / `equals` / `is exactly` / `turns out to be` …） |

`answer_leak` 的断言线索词要求是**实测之后放宽**的：原规则是"skill 文本里出现任何等于某个 GT 的数字就算泄漏"。对真实 adaptation pool（gneubig/aime-1983-2024, 2018–2022）量过：AIME 答案里有 **4%** 恰好就是 skill 自然会写的那些整数界（4 / 10 / 20 / 50…）。按每 batch 8 题、模型错 6–8 题计，裸相等规则会**误拒 21.7%–27.9%** 含常见界数的 skill —— 正好是"先枚举一个小范围"这类本项目最想学到的 skill。而 GT 泄漏通道在结构上已经关死（②），为一个没有活体泄漏路径的通道付 1/4 误拒率不划算。纯数字巧合改为保留 + 打 `numeric_coincidence` 咨询标记写进 `harness_log.jsonl` 的 `guard_note`，使这个取舍**可审计**（commit `1ff380f`）。

配套的 `_NUMBER` 正则也修过一次：原来的负向前瞻 `(?![\w.])` 本意是排除小数 `738.5`，却顺带让**任何以句号结尾的数字串**（`The answer is 50.`—— 最自然的泄漏写法）对 `answer_leak` 和 `numeric_coincidence` 双双不可见。现为 `(?<![\w.])\d+(?!\.?\d)(?!\w)`（commit `acbc69a`）。

**④ 硬失败：GT 工具调用。** `assert_no_gt_tool_call()` 每轮检查一次 action 文本，一旦出现 `<informalmath_verify>` 就抛 `LeakageError`。这是 `run_stream` 中**唯一不被吞掉**的异常 —— 普通的单题异常降级成全零结果继续跑，而泄漏一旦发生，这条 rollout 下游的一切（本轮结果、由它编译出的 skill）都不可信，正确反应只有停机。

**⑤ 唯一一处合法读取 ground_truth 的地方**：`EvoHarnessArm.end_batch` 把失败题的 `question` / `ground_truth` 直接透传给 `store.apply()`，仅供 guard 用于**拒绝**。代码注释和测试（sentinel 测试：在七个 ground-truth 字段位置各植入唯一字符串，断言它不出现在返回的任何字段里）都钉死了这一点。

---

## 5. 在线协议与保证

Task B 的四条硬约束，以及它们各自由什么机制保证：

### 5.1 没有未来信息

题流按固定顺序处理，`loader.batches` 顺序切 batch。驱动器每个 batch 严格五步（`evolving_harness_main.py` 模块 docstring）：

```
1. arm.begin_batch(i)        冻结本 batch 允许看到的跨题知识（harness 快照 / 经验池副本）
2. 串行：为 batch 内每道题算出注入文本 system_prompt_for(problem)   ← 在任何并行工作开始前
3. 并行：run_problem_fn（唯一的并行步骤）
4. 串行、按 batch 原序：record_selection / observe / 打点
5. arm.end_batch(i)          只有此刻才允许写 harness
```

### 5.2 一道题自己产生的 skill 只能影响后面的题，永远影响不到它自己

这是**结构性**保证而非调用顺序约定：每条有状态的臂在 `begin_batch()` 时把跨题状态**深拷贝冻结**，此后整个 batch 的 `system_prompt_for()` 只读那份冻结副本、从不读活的 store。即使有人绕过这个类直接在 batch 中途改活 store，也无法改变在飞题目看到的东西 —— 它们本来就没在读活状态。

### 5.3 只用成败标签与 verifier 反馈，绝不用 ground-truth 答案

见 §4.5。此外 `reflect` 只对 `pass_final == 0` 的题触发。

### 5.4 没有死题目污染 harness

`has_trajectory(result)` 按 `round_count` 判断而非 `pass_final`：驱动器给执行失败的题返回的全零结果 `pass_final` 也是 0，与"真的答错了"无法区分。曾经有一次端到端运行里**每一道题都死于 KeyError**，harness 却回来了四条流畅自信的 skill —— 从四条空轨迹编译出来的，而且在该次运行报告的每一个指标里都不可见（commit `70e77b6`）。现在这个判断放在 `observe` 边界上，所以死题连 Reflect 都到不了，也不会贡献 evidence tag 或花掉一次管理调用。

### 5.5 测试集不泄漏

- 数据按**年份**切分，绝不 shuffle-then-split（§6）。held-out 是完整的 AIME 2025，adaptation 是 2018–2022，两者题面**零重叠**（已实测）。
- held-out 评测时三条臂**全部 frozen**：`EvoHarnessArm.frozen=True` 让 `end_batch` 成为彻底的 no-op（不 reflect、不 curate、不写日志、不改 store）；`RawExperienceArm.frozen=True` 同理（不 summarise、不写池）。frozen 仍然**注入**已加载的内容 —— "停止学习"，不是"停止使用已学到的"，否则 held-out 等于把 Baseline 量了三遍。
- `assert_frozen_arm_has_state()` 在**一次调用都还没发出**之前就拒绝"frozen 但跨题状态为空"的 run。理由写在 docstring 里：指错路径的 frozen 臂什么都不注入、行为与 Baseline 完全一样，得到三条一模一样的曲线，而这与**真实的 null result 无法区分** —— 对一个全部意义在于解释 null result 的实验来说，这是最糟糕的失败方式。（Baseline 臂 `cross_problem_state_size()` 返回 `None`，豁免；未 frozen 的臂从空开始是正常冷启动，不拦。）
- topic 标注的 prompt **刻意没有**针对 AIME 2025 的一致率数字调优过 —— 2025 是 held-out，哪怕只用它的 metadata 来挑措辞也是测试集泄漏。它只被用来**测量**候选标注模型，没有别的用途。

### 5.6 skill 机制失败绝不破坏底层 baseline

| 失败点 | 降级行为 |
|---|---|
| selector 调用失败 | 空选择 → 该题用中性 prompt 跑，等同 Baseline |
| reflect 调用失败 | 跳过该候选，记 warning |
| curator 调用/解析失败 | 返回 `[]`，harness 本 batch 不变 |
| 单条 edit 异常 | 逐条捕获，不影响同 batch 其余 edit |
| `store.apply` 整体异常 | 捕获 + `logger.exception`，harness 不变 |
| `arm.end_batch` 异常 | 驱动器捕获，继续跑下一个 batch |
| 单题 rollout 异常 | 计入 `n_errors`，降级为全零结果，继续跑 |
| wandb 不可用 | 只写 jsonl（含未登录时 `wandb.init()` 抛 `KeyboardInterrupt` 这一 `BaseException` 情形） |
| **首个 batch 全军覆没** | **不降级 —— 抛 `StreamAbort`**。上面这一整套韧性机制恰恰会让一个彻底配置错误的 run 安静地"跑完"。断路器只作用于第一个 batch：在一个 batch 成功之前，没有任何东西证明过配置、stream schema、凭证和上游调用路径能拼在一起；此后整批失败就是 provider 故障，那正是这套韧性该管的事。 |

---

## 6. 数据准备

```bash
export OPENAI_API_KEY=<your key>
python -m alphaapollo.data_preprocess.prepare_harness_stream \
    --out_dir ./data/harness \
    --label_model qwen3-32b \
    --base_url https://dashscope.aliyuncs.com/compatible-mode/v1
```

产出两个 parquet（`--label_model` 可省略，此时 `topic` 列留空，其余字段依然正确）。

### 6.1 实测结果（重新读取 parquet 校验过）

| | adaptation | held-out |
|---|---|---|
| 来源 | `gneubig/aime-1983-2024`，2018–2022 | `MathArena/aime_2025` |
| 题数 | **149** | **30** |
| 按年 | 2018:30 / 2019:30 / 2020:30 / 2021:30 / **2022:29** | 2025:30 |
| topic | 模型标注（`qwen3-32b`） | **MathArena 人工 `problem_type` 标签** |
| topic 分布 | geometry 52 / combinatorics 34 / number_theory 33 / algebra 30 | combinatorics 9 / algebra 9 / geometry 7 / number_theory 5 |

另已验证：`problem_idx` 连续、按 (year, contest, number) 排序、两个 split 题面**零重叠**、每个答案都是 0–999 的纯整数、`gt_traj` 全部为空串。

`gt_traj` 留空是刻意的 —— 这两个数据源本来就不提供完整解法，而把答案填进去等于给 Reflect 开一条 ground-truth 通道。

**`2022-II-8` 被剔除**（150 → 149，仍在任务书 80–150 区间内）：它公布的答案是 `"080 or 081 (both were accepted)"`，而打分器是**字符串比较**，留着它会让**每一条臂的每一次尝试**都被判错 —— 它不是一道难题，是一道**不可打分**的题。

### 6.2 topic 标注：它是什么、不是什么

**方法本身不读任何一道题的 topic。** reflect 自己命名 topic（§4.3 ①），selector 检索时也没有 topic 过滤（§4.2）。topic 列的唯一用途是让 write-up 能做 per-topic 拆解 —— 而这是**三条臂都要报的结果**，只有在三条臂看到**逐字节相同**的标注时才可比，所以它被一次性冻进 stream 文件，而不是每条臂在线各标一遍。标注调用记在 `offline_labeling` 这个管理 role 下，作为跨题开销出现在成本报告里，不会藏进 solver 桶。

`classify_topic` 的签名收一个**裸 question 字符串**，不收 problem dict，所以带着 `ground_truth` 的字典不可能被误传进去。topic 是题面的函数，而 solver 本来就完整读到了题面，这里没有它不知道的信息。

**标注质量（实测，对 30 道 AIME 2025 题与人工 `problem_type` 比对）：**

| 标注模型 | 一致率 (n=30) | 未标注 | 耗时 |
|---|---|---|---|
| `qwen3-8b` | 21/30 = **70%** | 1 | 64 s |
| `qwen3-32b` | 25/30 = **83%** | 0 | 21 s |
| `qwen3.7-flash` | 25/30 = **83%** | 0 | 21 s |

`qwen3-8b` 的错误是**系统性**的：它输出 `geometry` 14 次而人工只有 7 次，9 个错误里有 6 个是"模型说 geometry、人工说 algebra 或 combinatorics"。这就是标注模型与冻结 solver 分开配置的原因 —— 这是**数据集标注**，与 MathArena 花钱雇人标自己那份 split 属于同一类行为，不是被测方法的一部分。

**n=30 的区间很宽（约 ±13pp），必须当作实测局限而非保证来引用。** 并且 adaptation split 的 geometry 占比（52/149 = **35%**）明显高于人工标注的 held-out split（7/30 = **23%**），方向与该标注模型的已知偏差一致 —— **任何 per-topic 表格都必须带上这个 caveat**。

---

## 7. 实验设计与运行命令

> **本节是设计、配置与运行命令。三组实验已全部跑完 —— 结果数字在 §8，尚未完成的部分在 §12。**

### 7.1 三条臂

| 臂 | 跨题机制 | 回答的问题 |
|---|---|---|
| `baseline` | 无。只有 AlphaApollo 原生的题内 memory | 性能地板 |
| `raw` | 存每道**已完成**题（成功与失败都存）的一段自由文本摘要，按相关性**原样**注入 | "直接把历史贴进 context 是不是就够了？" |
| `evo` | Task A/B 机制：持久化 `SkillStore`，**只从失败学**（论文 eq. 6），Reflect + 双 curator | 本项目的方法 |

`raw` 是**认真的**对照，不是稻草人：与 `evo` 共用同一套 `b` / `tokens` 预算、同样的 batch 边界纪律、同样的非空 prompt 保证、同样支持 `frozen` 与磁盘持久化、同样上报自己的注入 token 成本。它与 `evo` 唯一允许不同的是**存什么**（原始摘要 vs 编译后的 skill），绝不是**注入多少**。

`raw` 刻意**保留无模型的检索器**（词重叠）：它存在的全部意义就是"把相关历史贴进去"这个朴素做法；给它也配一个 LLM selector，它就变成第二套 skill 编译系统了。

### 7.2 相同的东西

三条臂共享：同一个 `qwen3-8b` solver + verifier、同一份 stream 文件（同一题序、同一份 topic 标注）、同样的 `evolving_round` / `max_steps` / `history_length` / 温度 / `seed`、同样的工具集、同样的 batch 划分。

### 7.3 与论文的偏离（明确声明）

| 项 | 论文 | 本项目 | 理由 |
|---|---|---|---|
| batch size | 16（Appendix F） | **8** | adaptation 只有 149 题；batch 16 只给出约 9 个 harness 更新点，不足以显示增长趋势 |
| selector 模型 | Claude Sonnet 4.5，所有实验 | **`qwen3-32b`**，与冻结的 `qwen3-8b` solver 分开配置 | 可用性；论文本身也是"用比 solver 更强的模型做选择" |
| 题流 topic 标注 | 论文没有这一步 | 有 | **纯分析口径**，方法不读它（§6.2） |
| 题序 | — | 年份序，非 topic 交错 | 任务书两者皆可；"题目按序到达"最自然的读法就是时间序 |

`batch_size`（协议旋钮：harness 多久能变一次）与 `max_workers`（吞吐旋钮：同时跑几道题）是**两个独立参数**，由 `tests/harness/test_driver.py` 钉死。混淆二者曾导致一个不存在的设计取舍（"batch 8 × 3 臂并行 = 24 并发，超过 provider 上限 12，所以只能降 batch"）—— 实际上 `batch_size=8` + `max_workers=4` 就能在一半并发下保持论文的更新粒度。这一点重要，因为 `batch_size=1` **不是**安全回退：general curator 的全部意义是在多道题之间找模式，单题 batch 只能产出 topic skill，Task A 要求的双层设计会直接塌掉（commit `385104d`）。

### 7.4 运行

```bash
export OPENAI_API_KEY=<your key>          # 任何配置里都没有 key，只从环境读

nohup ./scripts/run_experiments.sh > run.log 2>&1 &
disown
```

`nohup` + `disown` **不是装饰**。两次诊断跑曾在半途被杀 —— 不是这台机器的问题（进程占 491MB，机器有 24GB），而是启动它的 agent 会话在拆除后台任务。多小时的跑必须归属系统，而不是归属启动它的那个 shell。

脚本按 adaptation（三臂并行）→ 冻结状态 → held-out 的顺序跑完六个 run。**中断后重跑同一条命令即可续**（§7.5）。

```bash
./scripts/run_experiments.sh adapt        # 只跑 adaptation
./scripts/run_experiments.sh heldout      # 只跑 held-out（需 adaptation 已完成）
PARALLEL=0 ./scripts/run_experiments.sh   # 三臂串行
ARMS="baseline evo" ./scripts/run_experiments.sh   # 只跑指定的臂
PY=python3.12 ./scripts/run_experiments.sh         # 指定解释器
```

脚本在花掉第一次调用之前就拒绝三种错误启动：没有 `OPENAI_API_KEY`、held-out 跑在 adaptation 产物不存在时、未知的 phase。它还会把 `dashscope.aliyuncs.com` 加进 `NO_PROXY` —— 本机 SOCKS 代理曾拒绝约 25% 的连接，表现为**丢题而不是报错**（直连 0.38s，走代理 1.00s）。

单独跑一个 run：

```bash
python -m alphaapollo.workflows.evo --config examples/configs/harness_adapt_evo.yaml
```

⚠️ flag 是 `--config`，**不是 `--config_path`**。`parse_known_args` 不会拒绝未知参数，它会把 `--config_path` 变成一条没人读的 override，于是 `--config` 取默认值、跑的是上游的 `evolving_main`，25 秒跑完 30 道 aime24 题、零次模型调用、最后一行还写着 "Finished run"。

#### 查看进度

```bash
./scripts/status.sh          # 看一次
./scripts/status.sh -w       # 每 30 秒刷新
```

```
════════════════════════════════  [2026-09-16 01:57:45]
processes : 3 alive

RUN                  DONE  PASS@1   FINAL ERRORS  BATCHES UPDATED
adapt-baseline         72   18.1%   20.8%      1        9 12s ago
adapt-raw              64   17.2%   18.8%      0        8 41s ago
adapt-evo              64   15.6%   21.9%      2        8 8s ago

adapt-evo          ===> Problem 64 overall success: 0.0000, elapsed: 61.2s
```

它**只读磁盘产物**（`metrics.jsonl` / `progress.json` / 各 run 的日志），不碰运行中的终端。因此在 tmux 里跑、在另一个窗口查是安全的，跑完之后查也一样有效，写入过程中查也安全（半行 JSON 会被跳过而不是报错）。

怎么读这张表：

| 列 | 含义 |
|---|---|
| `DONE` | 已记录的题数（含跑挂的）|
| `PASS@1` / `FINAL` | **只统计成功跑完的题**，第 0 轮 / 最后一轮 |
| `ERRORS` | 跑挂的题数。**持续上涨说明在被限流** —— 降 `harness.max_workers` 后续跑 |
| `BATCHES` | 已提交的批数，也是续跑时会从哪里接上 |
| `UPDATED` | 该 run 的 `metrics.jsonl` 多久没变过 |

**`UPDATED` 是盯运行时最要紧的一列。** 卡住的 run 和跑得慢的 run 在准确率那几列上长得一模一样，只有"多久没更新"能区分。所以表格下面还会印每个 run 的最后一行日志 —— 一个刚跑完的 run 会显示 "15m ago" 但那不是卡住，要结合 `processes` 那行一起看。

三种状态的读法：

```
processes: 3 alive  +  UPDATED 都在几十秒内      → 正常
processes: 3 alive  +  某个 run 十几分钟没动     → 那个 run 卡住了
processes: none     +  所有 run 都 DONE=满       → 跑完了
```

其他产物随时可以直接看：

```bash
tail -f run.log                                  # 脚本自己的阶段日志
tail -f outputs/harness/adapt-evo.log            # 单个 run 的详细输出
cat outputs/harness/adapt-evo/progress.json      # 跑到第几批
ls outputs/harness/adapt-evo/store/skills/       # 目前编译出了哪些技能
tail -3 outputs/harness/adapt-evo/store/harness_log.jsonl   # 最近几个 curator 决策
```

wandb 上同时有实时曲线（`alphaapollo-evo-harness` project，按 `adapt` / `heldout` 分组），但跑的过程中**只有原始序列** —— 七项报告结果要跑完之后用 `wandb_backfill.py` 回填上去，见 §7.6 末尾那张对照表。

### 7.5 配置文件

七份，一份基座 + 六份 overlay，用的是上游自带的 `base_config:` 继承（`utils.load_run_configuration` → `_apply_base_config`），不是本项目发明的机制：

```
examples/configs/
├── harness_base.yaml                 # 三臂共享的 solver 环境
├── harness_adapt_{baseline,raw,evo}.yaml
└── harness_heldout_{baseline,raw,evo}.yaml
```

每份 overlay 只写让这条臂成为这条臂的东西，所以 `diff harness_adapt_baseline.yaml harness_adapt_evo.yaml` 显示的就是实验操作本身。`tests/harness/test_configs.py` 钉死了六份配置**共享同一套 solver 环境**、`raw` 与 `evo` 拿到**逐字相同的注入预算**、以及没有任何一份含有 key —— 公平性从声明变成了可验证事实。

> `entrypoint_module` 在六份 overlay 里各写一遍而**不是**继承自基座：`api.evo` 用一个**不做 `base_config` 合并**的 `OmegaConf.load` 来挑驱动器（`workflows/api.py:239`），只写在基座里的话，每一次运行都会静默地跑成上游的 `evolving_main`。

#### 可以调的参数

**`harness:` 段**（由 `evolving_harness_main.run()` 读取）

| 键 | 默认 | 说明 |
|---|---|---|
| `arm` | — | `baseline` \| `raw` \| `evo` |
| `stream_path` | — | 题流 parquet |
| `run_dir` | — | 本次 run 的全部产物：`metrics.jsonl` / `progress.json` / `trajectories/` |
| `store_root` | — | skill store（`evo`）或经验池（`raw`）；`baseline` 不设 |
| `batch_size` | 8 | **协议旋钮**：harness 多久能变一次 |
| `max_workers` | 4 | **吞吐旋钮**：同时跑几道题。脚本默认三臂并行，provider 看到的是 **3×** 这个数；实测 12 安全、16 开始限流 |
| `frozen` | false | held-out 阶段为 true：只选择注入，不再增删改 |
| `seed` | 1234 | 注入每一次请求；DashScope 的 seed 是 best-effort，**不保证复现** |
| `feedback_level` | standard | `standard` \| `minimal`（verifier 反馈质量消融） |
| `save_trajectories` | true | 每题约 62KB |
| `caps` | `{general: 5, per_topic: 5}` | 增长上限。**仅 `evo`** |
| `budget` | `{b: 6, general_max: 3, topic_max: 4, tokens: 800}` | 注入预算。`raw` 与 `evo` 必须相同 |
| `mgmt_model_overrides` | `reflect: {temperature: 0.3}`<br>`curator: {temperature: 0.0}` | 管理侧采样。在 solver 的 0.7 上做 curation 意味着同一候选每次裁决不同 |
| `selector_model_cfg` | `qwen3-32b` @ 0.0 | 选择用的模型。论文用比 solver 更强的模型 |
| `extra_body` | `{enable_thinking: false}` | DashScope qwen3 必需；指向 vLLM 时删掉 |
| `wandb` | `{enabled: true, ...}` | 可选。每张图都能仅凭 `metrics.jsonl` 离线重画 |

**solver 环境**（`env:` / `policy_model_cfg` / `verifier_cfg` / `run:`）沿用上游形状，全部在 `harness_base.yaml` 里，**三臂必须逐字相同**。其中三项刻意偏离上游 `vllm_informal_math.yaml`，每项都有 20 题实测支撑（产物在 `docs/findings/diagnostics/`）：

| | 上游 | 本项目 | 实测依据 |
|---|---|---|---|
| `evolving_round` | 10 | **2** | 10 得 4/20，与 2 相同，3.9× 调用 |
| `verifier_env_num` | 5 | **1** | 5 得 4/20，与 1 相同，2.0× 调用。100 个题次里**零**次"对改错"，多数判决没有作用面 |
| `max_tokens` | 8192 | 8192 | 唯一有效的旋钮（+2 题），且比 2048 **更便宜**（不截断 → 不重试） |

#### 断点续跑

每个 phase 从自己的 `progress.json` 按**批**恢复。按批而不按题，是因为 `end_batch` 是唯一的提交点：第 5 批跑到一半崩了，store 里是第 4 批的状态，协议正确的恢复点是第 5 批开头。崩溃前写下的部分行会先被清掉再重跑，否则那几道题会被双计。

指纹（stream / arm / batch_size / seed / frozen）不匹配时**拒绝启动**而不是从头开始 —— 前者会覆盖一次真实的 run，后者会把两个实验拼在一起。`max_workers` 不在指纹里：限流之后换更小的并发接着跑，正是它该被允许的用法。

### 7.6 出结果

```bash
python -m alphaapollo.core.harness.analysis --root ./outputs/harness --out_dir ./outputs/harness/report
```

写出 `results.md` 与 `results.json`，覆盖任务书要求的全部七项：

| 要求 | 出处 |
|---|---|
| adaptation 上按区间的 success rate / Pass@1 | §1，窗口默认 25（`--window`）|
| held-out 上最终 frozen harness 的准确率 | §2 |
| 不同数学主题上的表现变化 | §3，逐臂逐 phase |
| harness 的数量、长度与增长趋势 | §4，每批一行 |
| 注入 context 的平均 token 与总调用量 | §5，solver / 管理**分开报** |
| skills 被选择或使用的频率 | §6，按频次排序 |
| 2-3 个正/负迁移案例 | §7，给出**候选**与要读的轨迹文件 |

两个口径值得说明：

**三臂在「全部成功跑完的题」的交集上比较。** 一道跑挂的题和一道答错的题都是 `pass_final == 0`，靠 `adapt/error` 区分。三臂并行打同一个 provider，会在同一场限流里丢**不同**的题；比较各自的原始比率就是在比较不同的题集。报告同时给出"各自题集上的比率"，因为两者不一致本身就是值得看见的发现。

**迁移案例给的是候选，不是结论。** 一个迁移案例是"注入了技能之后模型做得有什么不同"的论证，需要人读轨迹。能自动化的是**找出哪几道题值得读** —— 即 evo 注入了技能、且结果与 baseline 在同一道题上相反的那些。正负两个方向都给，负向的甚至更重要。

#### wandb 看得到什么，看不到什么

**这条界限要说清楚，否则很容易把 dashboard 当成结果。**

| | wandb 实时（`tracker.py`）| wandb 回填（`wandb_backfill.py`）| `analysis.py` |
|---|---|---|---|
| 逐题 `pass1_round0` / `pass_final` / `error` | ✅ 原始序列 | — | ✅ |
| 每批 `harness/n_general`、`total_tokens`、`mean_skill_tokens` | ✅ 增长曲线 | — | ✅ |
| 每批 `calls/*`、`tokens/*` 分角色 | ✅ | — | ✅ |
| **累积 / 滚动 success rate 曲线** | ❌ | ✅ history | ✅ |
| **按区间的 success rate** | ❌ | ✅ table | ✅ |
| **per-topic 通过率** | ❌ | ✅ summary + table | ✅ |
| **三臂在交集上的比较** | ❌ | ✅ summary（`analysis/common/*`）| ✅ |
| **skill 使用频次** | ❌ | ✅ table | ✅ |
| **迁移案例候选** | ❌ | ✅ table | ✅ |

**实时那一列拿到的只是原始序列** —— 七项报告结果一项都不在里面。它们要么是跨 run 的（交集比较），要么是需要分组聚合的（窗口曲线、per-topic），wandb 的逐 step 模型在**跑的过程中**表达不了。

跑完之后可以：

```bash
python -m alphaapollo.core.harness.wandb_backfill --root ./outputs/harness --dry_run  # 先看要推什么
python -m alphaapollo.core.harness.wandb_backfill --root ./outputs/harness           # 推
```

它按 `<run_dir>/wandb_run_id.txt` 里的 id `resume` 原来那个 run，把 `analysis.py` 算出的量挂上去，**不重跑任何东西**（只读 `metrics.jsonl` 与 `selection_log.jsonl`）。每个数字都直接调 `analysis.py` 的函数得到，不另算一遍 —— dashboard 与 `results.md` 对不上会比 dashboard 是空的更糟。

三条 wandb 的硬约束决定了它的形状：

- **派生曲线不能复用原来的 step。** wandb 的 history step 单调递增，续上的 run 从上一个 step 往后走，写到更早的 step 会被静默丢弃。所以逐题曲线用自定义 x 轴（`analysis/problem_idx`，经 `define_metric` 声明），全局 `_step` 随它去 —— 对这些曲线来说 `_step` 轴本来就没有意义。
- **因此 history 推一次就不能再推**（只能追加，再推一次会在同样的 x 上叠第二组点，而 wandb 没有删除第一组的办法）。挡住它的是**两个**文件，因为它们回答两个不同的问题，合成一个就会让 `--force` 同时绕过两者：
  - `<run_dir>/wandb_backfill.json` 记录**已经发过什么**，其中 `series_sent` 是粘性的 —— 在第一行 history 发出去**之前**翻转，之后（包括只推 summary 的修复）永不回退。已经发过 history 的 run **无条件**拒绝再发，`--force` 也不行：再多的"我确定"也不能让重复的曲线变成对的。
  - 并发由 harness 自己的 pid 锁（`resume.acquire_run_lock`）管，每次推送都取，**永远不可绕过**。顺带保证了回填不会和一个还在往 `metrics.jsonl` 追加的 run 撞上。

  `--force` 只管前一个文件，意思是"是的，我要重做我做过的事"。两个文件都放在 run 目录里，语义和 run id 一致：删掉目录就是"这次实验没了"。
- **它永远不是事实来源。** `metrics.jsonl` + `analysis.py` 才是可复现的产物，这里只是单向复制到 dashboard 上。`--dry_run` 把完整 payload 写到 `<run_dir>/wandb_backfill_preview.json`，一个网络请求都不发。

已经推过的 run 要改数字，唯一的路径是只推按 key 覆盖的 summary 和 Table：

```bash
python -m alphaapollo.core.harness.wandb_backfill --root ./outputs/harness \
  --phases heldout --only heldout-evo,heldout-raw --skip_series --force
```

`--only` 只限制**推哪几个 run**，跨臂的量仍按整个 phase 计算 —— 否则修一个 run 会顺手改掉它 `analysis/common/*` 的含义。

**上传前先做 preflight**：请求的每个 phase 都要凑齐 baseline/raw/evo，每个目录都要有非空 metrics、有 wandb run id、project 可解析。任何一条不满足就在**发第一个请求之前**整体拒绝并以非零码退出 —— 这里要防的不是崩溃，而是"推了六个里的五个然后愉快地报成功"：单看一个 run，缺目录/空 metrics/没有 run id 都长得像"没事可做"，五个成功旁边的一次静默跳过是看不见的。中途实验或 jsonl-only 的 run 用 `--allow_missing` 显式放行；`--dry_run` 只警告不拒绝（什么都没上传，也就没什么可搞坏的）。

退出码：只要有 run 尝试上传并失败，或 preflight 不通过，就非零。"已经推过"这种主动跳过不算失败 —— 把两者混为一谈，会让六个 run 全部认证失败的一次运行在 shell 看来是成功的。

> **一个已经踩到的坑：held-out 的 selection log 混着 adaptation 的行。** `run_experiments.sh` 给 frozen run 的状态是**整目录复制**adaptation 臂的，`selection_log.jsonl` 也在里面；held-out 跑起来之后往同一个文件追加自己的 30 行，而两个 phase 的题号都从 0 开始，所以这 30 行是**压在** adaptation 的 0–29 上、而不是接在 144 行后面。整份读进来就等于拿 adaptation 的数据回答关于 held-out 的问题（实测：174 道"题"而不是 30，平均注入 189 token 而不是 209，held-out evo 的 skill 数 15 而不是 13）。`analysis.load_selections` 用两条规则修正：**后写覆盖先写**（撞号时解析到最后追加的那个 phase，也就是正在报告的这个），加上 **`only`** 限定到本 run 真正有 metrics 的题号集合。`only` 传的是**全部**题目而非 `completed` —— 一道跑挂的题照样选过技能、照样花了 token，按 `completed` 过滤会把这部分开销悄悄抹掉。`results.md` 不受影响（§5/§6 只覆盖 adapt 臂），逐字节比对已确认。

分工因此是：**实时 wandb 用来盯 run 还活着**（曲线在动、`calls/unscoped` 没有变成非零、harness 在长），**`analysis.py` 用来出结果**，**回填让 dashboard 事后也能读**。每一张 wandb 图都能仅凭 `metrics.jsonl` 离线重画。

两个实现细节：

- **只有数值转发给 wandb。** 逐题的 `topic` 是字符串，画不了图，转发过去只会给每个 run 加一列不可图表化的东西，还会让人误以为 per-topic 拆解在 dashboard 上是实时的。jsonl 保留全部字段。
- **续跑会继续同一个 wandb run**，不会新开一个。run id 存在 `<run_dir>/wandb_run_id.txt`，语义与 `metrics.jsonl` 对齐：续跑接着写，删掉目录重跑则是全新的 run。没有这一条，一次被中断的实验会在 dashboard 上显示成两条半截曲线，而 resume 存在的理由恰恰就是"15 小时的跑会被中断"。

导出最终 harness 本身：

```bash
python -m alphaapollo.core.harness.export --store_root ./outputs/harness/adapt-evo/store --out_dir ./outputs/harness/export-evo
# -> harness.md / summary.json / evolution.jsonl
```

`run_experiments.sh` 在 held-out 跑完后会自动执行这一步。

wandb 是**可选**的（任务书从未要求）：六个 run（3 臂 × adaptation/held-out）共用一个 project，run name 默认 `<phase>-<arm>`、group 默认 phase，这样 dashboard 上三条臂才分得清、一个 phase 的三条臂才画在同一组轴上。每个图表都能仅凭 `<run_dir>/metrics.jsonl` 离线重画 —— jsonl 才是可复现的产物，wandb 只拿到一份副本。

### 7.7 开销口径

`accounting.py` 按 role 分桶统计调用数与 token：

- **solver 侧**：`solver`、`summarizer`（上游题内的压缩 agent）、`aggregator`（上游 verifier 报告聚合）
- **管理侧**：`reflect`、`topic_curator`、`general_curator`、`raw_summarizer`、`offline_labeling`、`selector`

`raw_summarizer` 单独成一个 role、不复用 `summarizer`，是因为后者已经命名了上游**题内**的求解侧助手（baseline 本来就有）。把 `raw` 臂的每题摘要调用折进去，会让 Raw Experience 在成本报告里显得几乎不花钱，而 EvoHarness 的 reflect/curate 却被正确计为管理开销 —— 那是一种**会错误地削弱本项目自身论点**的数字（commit `7b4336d`）。

`role_scope` 用的是 `ContextVar`，而 `ThreadPoolExecutor` **不会**把 context 传进 worker 线程。上游把 verifier 调用扇出到自己的线程池，于是那些调用曾落进 `unscoped` 桶 —— 第一次干净运行里 38 次 solver 侧调用有 **9 次**落在那里，要求上报的 solver-vs-management 拆分直接是错的。现在 role 在父线程 `.start()` 时捕获、在子线程 `.run()` 时应用，复现 contextvars 本该给出的继承语义。它是**默认值不是锁**：子线程里显式的 `role_scope` 仍然优先，没有 scoped 祖先的线程仍然保持 `unscoped`，这样一个忘记打 scope 的管理调用点仍然**可见**，而不是被悄悄折进 solver 桶（commit `62b024b`）。

---

## 8. Task C 实验结果

本节的每一个数字都可以从 `outputs/harness/` 重新算出来。汇总表由 `python -m alphaapollo.core.harness.analysis --root ./outputs/harness --out_dir ./outputs/harness/report` 生成（`report/results.md` 与 `results.json`），最终 harness 由 `export-evo/` 给出。

### 8.1 运行状态

| | |
|---|---|
| run | 6 个：3 臂 × {adaptation, held-out}，全部跑完 |
| 题量 | adaptation 144（stream 149 题，尾部 5 题因 `loader.batches(drop_last=True)` 被**三臂同等**丢弃）；held-out 30（AIME 2025 全年） |
| **丢题** | **0 / 0 / 0**。对照 §9 烟囱测试的 25% 丢题率 —— `NO_PROXY` 那条修复是有效的 |
| `calls/unscoped` | 6 个 run 全为 **0**，solver/管理开销拆分可信 |
| 测试 | `619 passed in 4.64s` |
| resume | 三个 adaptation run 各 18 批、held-out 各 4 批，`progress.json` 的 fingerprint 一致 |

held-out 的 evo 臂管理调用**只有 `selector`、没有 `reflect`/`curator`**（§8.6 表），这是"最终 harness 确实被冻结"的直接证据，而不是靠配置声明。

### 8.2 主结果

| arm | adaptation Pass@1 (round 0) | adaptation 最终 | held-out Pass@1 | **held-out 最终** |
|---|---|---|---|---|
| Baseline | 22.2% (32/144) | 21.5% (31/144) | 13.3% (4/30) | 10.0% (3/30) |
| Raw Experience | 16.7% (24/144) | 21.5% (31/144) | 23.3% (7/30) | **23.3% (7/30)** |
| **Evo-Harness** | 23.6% (34/144) | **24.3% (35/144)** | 13.3% (4/30) | 16.7% (5/30) |

配对 McNemar 精确检验（同题比较）：

| 比较 | adaptation | held-out |
|---|---|---|
| baseline vs raw | p = 1.000（9 / 9 不一致） | p = 0.125（0 / 4） |
| baseline vs evo | p = 0.481（7 / 11） | p = 0.625（1 / 3） |
| raw vs evo | p = 0.503（8 / 12） | p = 0.625（3 / 1） |

**一项都不显著。** Evo 在 adaptation 上的 +2.8 个百分点等于 4 道题，而 §12.1 记录的本系统噪声底是 ±2 题 / 20 题 ≈ ±3.6 个百分点 —— 这个优势完全落在噪声里。held-out 上 **Raw 反而赢了 Evo**（23.3% vs 16.7%），只有 4 对不一致样本，同样是噪声，但它是必须主动解释的数字而不是可以略过的数字。

### 8.3 adaptation 上随时间的变化（窗口 = 25）

| 窗口 | 0-24 | 25-49 | 50-74 | 75-99 | 100-124 | 125-143 |
|---|---|---|---|---|---|---|
| baseline | 24.0% | 32.0% | 16.0% | 28.0% | 20.0% | 5.3% |
| raw | 20.0% | 28.0% | 24.0% | 32.0% | 8.0% | 15.8% |
| **evo** | 28.0% | 32.0% | 16.0% | 36.0% | 16.0% | 15.8% |

**没有学习曲线。** 如果 skill 编译在累积，evo 的后段窗口应当高于前段 —— 它没有，而且三条曲线的形状几乎一致（同升同降），说明波动由**题目顺序**驱动而非由 arm 驱动。这是本实验最直接的负面证据，比任何一个总分都重要。

### 8.4 per-topic（最终正确率）

| topic | n | adaptation base / raw / **evo** | n | held-out base / raw / **evo** |
|---|---|---|---|---|
| algebra | 30 | 36.7% / 36.7% / **53.3%** | 9 | 0.0% / 22.2% / 11.1% |
| number_theory | 32 | 28.1% / 28.1% / **21.9%** | 5 | 40.0% / 40.0% / 20.0% |
| combinatorics | 33 | 12.1% / 9.1% / **15.2%** | 9 | 11.1% / 11.1% / 11.1% |
| geometry | 49 | 14.3% / 16.3% / **14.3%** | 7 | 0.0% / 28.6% / 28.6% |

唯一超出噪声量级的移动是 **adaptation / algebra：36.7% → 53.3%（+16.7pp, n=30）**。它恰好是 `sk_0001` 所在的领域，而 `sk_0001` 是全场效用最高、且**唯一从未被编辑过**的 skill（§8.7）。这是一个**事后**观察到的、样本量 30 的相关，应当作为假设写进 slides，不能作为结论。held-out 上 geometry 的 0% → 28.6% 只有 7 题、2 道题的差别，不承载信息。

### 8.5 harness 规模与增长

| 跑完第 N 题 | 7 | 31 | 55 | 79 | 103 | 127 | 143 |
|---|---|---|---|---|---|---|---|
| general | 1 | 3 | **5** | 5 | 5 | 5 | 5 |
| topic | 5 | 9 | 9 | 9 | 10 | 11 | 11 |
| 总 token | 398 | 768 | 913 | 891 | 912 | 977 | **1003** |
| 每条均长 | 66 | 64 | 65 | 64 | 61 | 61 | 63 |

**增长控制按设计工作**：general 层第 55 题起顶到 `caps.general = 5` 就不再增长，总量 144 题只累积到 1003 token，每条 skill 的长度稳定在 60 余 token 而没有膨胀。Task A 要求的"显式有界增长"是达成的。

curator 决策（`export-evo/summary.json`）：**209 次决策，接受 89、拒绝 120**。

| op | 提出 | 接受 |
|---|---|---|
| ADD | 17 | 16 |
| MERGE | 105 | 56 |
| REVISE | 17 | 17 |
| SKIP | 70 | —（SKIP 即不改） |

拒绝原因只有三种：`skipped` 70、**`wrong_layer` 49**、`capacity_full` 1。`wrong_layer` 占全部决策的 **23%** —— curator 试图改另一层的 skill、被 `e62ea41` 那道闸门挡下。这**不是质量判决，是 prompt 契约的结构性失败**：近四分之一的管理预算烧在一类本可以在 prompt 层面消除的错误上。

### 8.6 开销（solver 与管理**分开报**）

`metrics.jsonl` 里的 `calls/*`、`tokens/*` 是**累积值**，跨 batch 行求和会得到约 9.5 倍的虚高数字 —— 取每个 run 的最后一行，或直接用 `results.json`。

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
| evo | 263 | 30（全部是 `selector`） | 293 | 9.8 |

**注入 context 的 token**

| arm / phase | 总量 | 每题均值 | 非零均值 | 最大 | 每题 skill 数 | 零注入题 |
|---|---|---|---|---|---|---|
| evo / adaptation | 26,648 | **185.1** | 195.9 | 367 | 3.04 | 8 |
| raw / adaptation | 56,128 | **389.8** | 412.7 | 540 | — | 8 |
| evo / held-out | 6,264 | 208.8 | 208.8 | 369 | 3.57 | 0 |
| raw / held-out | 12,244 | 408.1 | 408.1 | 485 | — | 0 |

**六个 run 总计 5,509 次调用 / 10,622,564 token。**

三点值得写进 slides：

1. **evo 加了 20.7% 的管理调用，总 token 反而比 baseline 少 12.8%** —— solver 输出从 1.66M 降到 1.06M（−36%），注入 skill 让 rollout 变短了。"Evo 更贵"这个直觉在本数据上只对**调用次数**成立（+5.5%），对 token 不成立。
2. **预算上限是 800 token，evo 实际只用到 185（23%）；Raw 注入量是它的 2.1 倍，held-out 上还赢了。** 所以 evo 的劣势不能归因于"注入得不够多"。
3. **两臂在 adaptation 上各有 8 题零注入，正好是 batch 0 的 p_0–p_7**（harness 当时为空）。这是"一道题不可能被自己产生的 skill 影响"（§5.2）在数据上的直接证据。

> **一个产物口径上的坑**：held-out 的 `selection_log.jsonl` 有 **174 行不是 30 行** —— `run_experiments.sh` 把 adaptation 的 store 整个拷过去，那 144 行选择记录跟着带了过来，held-out 只是往后追加。直接对该文件求和会把 adaptation 的开销算进 held-out。读取一律走 `analysis.load_selections(store_root, only)`：**按 `problem_idx` 后写覆盖 + 限定到本 run 的题号集合**。上表最初是用 `[-30:]` 取的，在这份数据上结果相同（held-out 的 30 行确实是最后追加的），但那个口径依赖"追加顺序恰好如此"，换个中断/续跑的 run 就不成立了 —— 已统一到前者，§7.6 有完整说明。

### 8.7 skill 使用频次

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
| `sk_0011` | 16 | 2 (12.5%) | 11.1% | 11 |
| `sk_0015` | 15 | 2 (13.3%) | 10.4% | 7 |
| `sk_0010` / `sk_0009` / `sk_0002` | 12 / 11 / 9 | 2 / 3 / 2 | — | 5 / 5 / 2 |
| `sk_0005` / `sk_0016` | 1 / 1 | 0 / 0 | 0.7% | 2 / 1 |
| `sk_0004` | **0** | 0 | 0% | 1 |

"注入且做对"是**相关而非归因** —— 一条专挑简单题的 skill 不用帮忙也会好看。两个结构性问题：

- **`sk_0006` 被注入到 82% 的题上，命中率 22.9% —— 和 base rate 一模一样。** 它是一条占预算的 no-op，却占了 4/5 题的注入槽。原因见 §11.3：它的 trigger 在 10 次编辑中漂移成了"When multiple constraints or conditions are present in a problem"这种对几乎任何 AIME 题都成立的空话，选择器因此无法区分。
- **`sk_0004` 一次都没被选中**，却始终占着 `isosceles_triangle_counting` 的 topic 槽位。harness 没有淘汰机制。

至于"编辑次数越多质量越差"这个诱人的解释：整体上**只有很弱的证据**（≤4 次编辑的 skill 平均效用 24.9%，≥5 次的 20.3%），而且这个差距几乎完全由 `sk_0001` 一个点撑着，`sk_0008` 被编辑 12 次仍有 27.0%。**漂移这个结论应当靠 §11.3 的文本证据来立，不要用这个相关性去论证。**

### 8.8 迁移案例

完整的逐轨迹分析在 **[`docs/findings/transfer-cases.md`](docs/findings/transfer-cases.md)**。三个案例的摘要：

**正向 · p_84**（number_theory，注入 `sk_0001`/`sk_0006`/`sk_0008`，188 token）。baseline 答 37、evo 答 239（对）。可复核的行为差异是**搜索上界 `range(1,100)` vs `range(1,1000)`**（真解 78 和 161，上界 100 必漏 161），外加 baseline **凭空编造了工具返回值**而 evo 读取了返回值并逐个手算验证。`sk_0006` 的 "no omissions in case analysis" 与放宽上界之间有合理联系，但**这是 temperature=0.7 的单次采样，因果归因未经证实** —— 要坐实需要在不注入条件下重复采样 p_84。

**负向 · p_105**（"所有三位回文数的算术平均"，GT 550）。evo **round 0 答对了 550**，round 1 改成 549.5 而错。事件链：evo 那条正确答案只有 756 字符 → verifier 判"没有给出推理过程"并主动断言 *"The correct mean is 549.5"* → policy 在 round 1 采纳该数字并为它倒推出一套错误算术 → verifier 在 round 1 **自己翻供**回 550，但 `pass_final` 取最后一轮，已经来不及。注入的三条 skill 没有把模型带向错误的**数学方法**，它们施加的是一种"再验证一遍"的**风格压力**，作用在一道不需要验证的题上。

**系统性 · 上面那条链不是孤例。** 定义 loss = round 0 对、最终错：

| arm | loss | gain | 净 | 被毁题的 round-0 消息均长 | 全部题均长 |
|---|---|---|---|---|---|
| baseline | **8** | 7 | −1 | 1827 | 2575 |
| raw | **1** | 8 | **+7** | 766 | 2491 |
| evo | **6** | 7 | +1 | 1302 | 2353 |

三条结论：(a) **题内演化轮在 baseline 上净负**（毁掉 8 条、救回 7 条），这是上游行为、三臂共同的噪声源；(b) **Raw 臂在 adaptation 上追平 baseline 靠的不是多做对题，而是少毁题** —— 它 round-0 比 baseline 低 5.5pp，最终却打平，全部差距来自 loss 从 8 降到 1；(c) 三臂一致地，**被毁掉的题其 round-0 最终消息明显更短**，因为 verifier 按"看得见的推理"打分，简短的正确答案会被判成"没有推理"，verifier 随即给出自己（常常是错）的数字。

### 8.9 对 null result 的分析

任务书明确写了"Evo-Harness 赢过 baseline 不是及格线，但 null result 必须被分析"。按它列出的六个方向：

1. **任务相似度。** AIME 跨年之间的题目共享的是**领域**而非**可复用的解题程序**。harness 里真正具体的 skill（如 `sk_0001` 的多项式分解条件）只在 18/144 题上被选中，而 82% 的题拿到的是通用套话。这可能是最根本的限制：AIME 的设计初衷就是每题需要一个新想法，而 Evo-Harness 的论文场景（SkillBench 等）有可复用的操作流程。
2. **skill 质量。** 见 §11.3：MERGE/REVISE 导致 trigger 与 lesson 语义脱钩，`sk_0006` 最终的 trigger 是泛化的"ensure all possible cases"而 body 讲的是圆的切线。这是**可修的工程缺陷**，不是方法本身的问题。
3. **选择错误。** `sk_0006` 注入率 82%、效用等于 base rate；`sk_0004` 注入率 0%。选择器不是选错了，而是**在漂移后的 trigger 上无法区分** —— 修好 (2) 才谈得上评价 (3)。
4. **context 干扰。** 确有其事，但机制出乎意料：不是知识层面的误导，而是 §8.8 的**风格层面**干扰 —— 注入使 evo 把验证前置、最终消息变短（中位数 2280 vs baseline 2596），而短消息更容易被 verifier 误判，进而在下一轮被改坏（evo loss 6 vs raw 1）。
5. **verifier 反馈质量 —— 这一项比预想的严重得多。** 在 policy 确实答对的那些轮次里，verifier 判它错的比例是：baseline **21/63 = 33%**、raw **14/55 = 25%**、evo **22/69 = 32%**（`summary.verifier_correctness_stats` 三臂累计）。**policy 答对时，verifier 大约每三次就否定一次**，而且常常附带一个具体的错误数字（p_105 里它断言 549.5，下一轮又自己翻供回 550）。编译 skill 所依赖的成败标签本身就带着这个量级的噪声；更糟的是这噪声不只污染 skill，它还经由题内演化轮**直接破坏最终答案**（§8.8 的 loss 列）。
6. **token 开销。** **不是**瓶颈：evo 只用掉 800 预算中的 185 token，总 token 还比 baseline 少 12.8%。可以排除这一项。

**最诚实的总结**：本实验证明了**机制是闭环的**（skill 来自真实轨迹、被组织、有界增长、确实影响后续题目、全过程可审计），但**没有证明编译出的 skill 有用**。最大的单一嫌疑不是"跨题迁移在 AIME 上不成立"这个方法论结论 —— 而是在能检验它之前，两个工程缺陷（语义漂移、23% 的 `wrong_layer` 浪费）与一个测量缺陷（verifier 噪声经由题内演化轮破坏最终答案，且与消息长度耦合）已经把信噪比压到了差异无法分辨的水平。**下一次实验该修的是这三项，而不是换一个更大的模型。**

---

## 9. 代表性 skill 演化记录（真实 trace，**烟囱测试**，非 Task C 实验）

> **想看 Task C 正式运行的演化记录，读 [`TASK_C_REPORT.md`](TASK_C_REPORT.md) §5** —— 那里有 `sk_0001`（144 题里唯一从未被编辑、效用最高）与 `sk_0006`（10 次编辑后 trigger 与 lesson 彻底脱钩）两条真实轨迹。同一份报告的 §4.9 给出 Task C 四类日志的真实样例与读法。
>
> 本节保留的是**更早的一次 12 题烟囱测试**，它的价值不在演化本身，而在于它暴露的两个缺陷（§9.3）与一次网络中断的处理（§9.4）—— 那些恰恰是正式运行**没有**再发生的事情。

以下全部来自一次**真实的 12 题 evo 臂运行**，跑在真实 adaptation stream 的前 12 道题上，`batch_size=4`、3 个 batch，对接 DashScope 的 `qwen3-8b`。

> **两条必须一并声明的 caveat：**
> 1. 这是**烟囱测试**，不是 Task C 实验，12 题不构成任何结论。
> 2. 该次运行执行的是**进程启动时**的代码，即 commit `064e799` 与 `e62ea41` **之前**的版本。因此它的 harness 演化 trace **仍然展示着这两个 commit 所修复的行为**（见下）。

### 9.1 运行层面的真实数字

| | |
|---|---|
| 墙钟 | **1354.7 s** |
| 题目 | 12/12 被驱动器执行，其中 **9 道产出真实轨迹** |
| 解对 | **2/12**（p1、p4） |
| 调用 | **93 solver + 20 management**（management ≈ 18%）<br>明细：`reflect` 6、`selector` 8、`topic_curator` 4、`general_curator` 2 |
| `calls/unscoped` | **0**（role 归属完整） |
| token | solver in 75386 / out 91312；management in 12178 / out 1657 |
| 跨题复用 | batch 0 的 4 道题看到空 harness（注入 0 token）；其后 8 道题中 **7 道被选中注入了 skill**（60–120 token，预算上限 800）。其中 **5 道真正跑完并带着 skill 产出轨迹**（p4–p8），另外 2 道（p9、p11）选中后死于连接错误 —— "被注入"与"带着 skill 跑完"是两个数，不能混用 |
| 丢题 | **3/12（25%）** 死于 `APIConnectionError`（p9/p10/p11），且 p8 的 Reflect 调用也失败，于是 **batch 2 产出 0 条 harness edit**，而整个 run 仍然 `exit=0`。根因是本机的 SOCKS 代理断连而非 DashScope：实测直连 0.38s / 走代理 1.0s，把 `dashscope.aliyuncs.com` 加入 `NO_PROXY` 即可绕开。**正式跑必须把丢题率作为一项指标记录** —— 否则"某条臂成绩低"可能只是它那轮网络更差 |
| 最终 harness | 3 general + 2 topic（`counting_with_constraints`、`coordinate_geometry_setup`），共 292 token |
| harness edit | 13 条 log 行：5 ADD、6 MERGE、1 REVISE、1 SKIP |

### 9.2 一条 skill 的真实演化路径（`sk_0001`）

```
batch 0 end: ADD    sk_0001  actor=topic_curator  source_problems=[0]   accepted
batch 1 end: MERGE  sk_0001  actor=topic_curator  source_problems=[5]   accepted
batch 1 end: MERGE  sk_0001  actor=topic_curator  source_problems=[6]   accepted
```

最终文件 `skills/when-counting-pairs-or-tuples-under-specific-conditions--sk_0001.md` 的 frontmatter：
`evidence: ["p_0", "p_5", "p_6", "p_7"]`、`revised_at: [1]`、`n_selected: 6`、`n_selected_success: 1`。

即：它由 p0 的失败产生 → 被注入给 p4–p7 → p5/p6/p7 的失败又反过来强化了它。这是一条**完整闭合**的跨题回路：skill 来自真实轨迹、被组织、并实际影响了后续题目。

### 9.3 同一份 trace 暴露的两个缺陷（已修，修复不在该次运行的代码里）

**(a) general 层退化成 topic 层的副本。** 该次 batch 0 产出的三条 "general" skill，evidence 分别是 `p_0`、`p_2`、`p_3` —— **三条单题教训被归档成了跨任务模式**。更糟的是其中两条与同 batch、同候选产出的 topic skill **逐字重复**：

| | topic 层 | general 层 |
|---|---|---|
| trigger | `When solving problems requiring enumeration under constraints.` (`sk_0001`) | `when counting pairs or tuples under specific conditions` (`sk_0003`) |
| trigger | `When setting up coordinates for geometric problems with variable side lengths.` (`sk_0002`) | `when setting up coordinates for geometric problems` (`sk_0005`) |

`GENERAL_CURATOR_PROMPT` 里早就写着 "Each general skill must address a pattern seen in at least 2 (2+) different problems"，但**没有任何东西强制它，模型就忽略了它**。两层于是不再是两层，而是同一层的两份拷贝，还要一起去抢同一份注入预算。修复（commit `064e799`）：在 `store.apply` 的 ADD 路径上按 evidence 的**不同题目数 < 2 即拒绝**，reject reason `general_needs_two_problems`。只查 ADD —— MERGE 是在强化一条已经过关的 skill，只看传入 payload 会把每一次合法强化都拒掉。

**(b) curator 越层改写。** `harness_log.jsonl` 里能看到：

```
{"problem_idx": 1, "op": "REVISE", "actor": "general_curator", "skill_id": "sk_0001", "accepted": true, ...}
{"problem_idx": 1, "op": "MERGE",  "actor": "general_curator", "skill_id": "sk_0001", "accepted": true, ...}
```

而 `sk_0001` 自己的文件写的是 `level: "topic"`。`GeneralCurator` 只被展示了 general 层的 skill，却**幻觉出了一个 topic skill 的 id**，而 `_apply_one` 按 id 查找目标、从不检查目标在哪一层，于是照改不误。后果不是美观问题：两个 curator 把内容和 evidence 往同一批 skill 里累加，两层不再独立 —— 论文报告的 General-Only / Topic-Only 消融会在"看起来在量两层"的同时**实际上在量一团纠缠**。修复（commit `e62ea41`）：`MERGE` / `REVISE` 校验目标层级（topic 层还要校验 bucket），不符则拒，reject reason `wrong_layer`。

### 9.4 另一处真实故障：网络中断如何被吸收

该次运行的最后一个 batch 遭遇本机代理中断（`openai.APIConnectionError` / `ConnectError: [Errno 61] Connection refused`）：

```
WARNING: run_problem failed for problem 9;  recording an all-zero result
WARNING: run_problem failed for problem 10; recording an all-zero result
WARNING: run_problem failed for problem 11; recording an all-zero result
WARNING: problem 9  produced no trajectory; not reflecting on it
WARNING: problem 10 produced no trajectory; not reflecting on it
WARNING: problem 11 produced no trajectory; not reflecting on it
WARNING: reflect failed for problem 8; skipping candidate
```

结果：三道题降级为全零结果、`has_trajectory` 把它们挡在 Reflect 之外、p8 的 Reflect 调用本身也失败并降级为"无候选"，于是 batch 2 **一条 harness edit 都没产生，harness 原样保留**，run 以 `exit=0` 正常结束。

这正是 §5.4 与 §5.6 想要的行为：**skill 更新失败不会破坏底层 baseline，也不会用空轨迹污染 harness**。

### 9.5 早期的 4 题端到端运行

在此之前还跑过若干次 4 题端到端运行，其中一次跑在**把所有 topic 标签剥光**的 stream 上，0 errors、39 solver + 6 management 调用、编译出 3 条 skill、topic 全部由模型命名 —— 这正是"方法不再需要逐题 topic 标注"的实证。同一次运行里 selector 对 batch 1 的两道题都回了 `NONE`，而且回得对：当时 harness 里只有 inradius 和周期函数两条 skill，那两道题是概率和代数。

---

## 10. 修改内容与溯源

按**主题**分组（而非时间顺序），每条给出可 `git show` 的 commit。commit message 本身写得很长，记录的是当时的推理过程，是本节的一手来源。

### A. 防泄漏与在线协议闸门（reviewer 最该查的部分）

| commit | 改了什么 / 为什么 |
|---|---|
| `d112cdc` | `build_reflect_context()` 改成逐个具名关键字参数，**没有 question/答案参数、没有 `**kwargs`**。泄漏在类型层面不可能，而非靠自觉。同时加 `sanitize_feedback()` 按整行剥掉 `Matches ground truth` 通道。 |
| `024c28e` → `1ff380f` → `acbc69a` → `06314f0` | 候选 skill 的 anti-leakage guard；随后把 `answer_leak` 从"裸数字相等"放宽到"需同行出现断言线索词"（实测裸规则会误拒 21.7–27.9% 的枚举类 skill，而 GT 通道已在结构上关死），巧合改为 `numeric_coincidence` 咨询标记；再修 `_NUMBER` 正则对句末数字的盲区（`The answer is 50.` 原本完全不可见）。 |
| `9e1b554` | `extract_result` 的 tool error **只从 `stderr` / `run_status` 取，绝不取 `stdout` / `raw_observation`** —— `informalmath_verify` 正是把 `Matches ground truth:` 打印到自己的 stdout。结构性排除优于事后清洗。附 sentinel 测试：在七个 ground-truth 位置各植入唯一字符串，断言它不出现在任何返回字段里。 |
| `36bc29f` | `reasoning_excerpt` 原本也走 `sanitize_feedback()`，而后者会剥 `<think>` 块 —— 但 AlphaApollo 自己的 policy prompt 强制模型把整段推导写进 `<think>`，于是每条真实推理 trace 都被压成了一个裸 `<answer>` 标签。新增 `_sanitize_reasoning_excerpt()` 只刷 GT 通道那一行、保留 `<think>` 内容。 |
| `bffc63b` | 解析 Reflect / curator / selector 回复前**先剥 `<think>` 块**。推理模型的 think-aloud 文本会例行地提出又否决候选答案，一行 `SCOPE: general` 或 `ACTION: NONE` 出现在 think 块里就会被 `re.search`（按位置取首个匹配）抢先命中 —— 对 curator 而言，被劫持的一行会变成对 harness 的一次真实 DELETE/MERGE。 |
| `d9d570f` | `assert_no_gt_tool_call` / `LeakageError`：`run_stream` 里唯一不被吞掉的异常。 |
| `35a811b` → `81a90cb` | 题内 memory 边界的守卫从"子串搜索"改成 **AST 扫描**（修掉一个漏报：`pkgutil.iter_modules()` 从不返回 `__init__.py`，而那是唯一提到边界的文件；和一个误报：docstring 里的说明文字会触发）；随后补上**相对 import** 的解析，`from ..environments.memory import X` 原本可以整个绕过守卫。 |
| `c6cbd9e` | frozen 且跨题状态为空的 run **在发出任何一次调用之前就被拒绝** —— 它产生的三条相同曲线与真实 null result 无法区分。 |

### B. 与论文的两处对齐修正

| commit | 改了什么 / 为什么 |
|---|---|
| `e1da737` | 一手读完 arXiv:2608.15071 的 Appendix A–F、I 之后，把 Reflect 与两个 curator 的 prompt 对齐 Appendix E：Reflect 接受 `related_skills` 输入、输出 `ACTION: NEW/ENHANCE/NONE (+TARGET)`、带上 E.1 的 "filter aggressively" 规则（其 "exact task replay" 条款同时充当 prompt 级的泄漏刹车）；两个 curator 应用 E.2 的 generalizability test，general curator 执行 E.3 对 context-specific 引用的禁令。 |
| `5706ff8` + `4015c8b` | **topic 由模型命名，不是数据集标签。** Appendix E.1 的输入清单里没有 topic，指令是模型在输出侧 "Choose a broad topic"。本包原先反了过来，凭空制造了对逐题 topic 标注的依赖 —— 而 2018–2022 的 AIME 数据源不带 topic 标签，这使得"手工整理一套分类体系"看起来像是必需的，其实从来不是。读取侧的 topic 标注（`topic.py`）仍然保留，但只作为**分析口径**。 |
| `f591906` | **Select 改为模型调用**（Algorithm 1 line 5 / Appendix F）。原先的确定性词重叠 + 硬 topic 过滤是为了可复现，但它**隐式要求每道题都带 topic 标签**。预算**仍然在代码里强制**，不下放给模型。`_select_from_snapshot` 被删除而非留作第二条互相矛盾的选择路径。 |

### C. 三臂公平性

| commit | 改了什么 / 为什么 |
|---|---|
| `9939d70` | 三条臂共用 `CrossProblemArm` 协议，**结构性 prompt 对等**：冷启动时都返回非空且结构相同的 `NEUTRAL_SYSTEM_PROMPT`（`utils/agent.py:37` 会把空串对应的 system message 整条删掉）；`raw` 与 `evo` 用**完全相同**的 `b`/`tokens` 预算；两条有状态的臂都在 `begin_batch()` 冻结跨题状态。 |
| `37ced30` | Raw Experience 臂补齐三处与 Evo 臂的不对称：① 没有 `frozen` → held-out 会继续摘要、继续长池，**等于在测试集上学习**；② 池子只在内存里 → held-out 的独立进程会从空开始，这条臂**静默退化成 Baseline**，对照直接作废；③ 没有 `record_selection` → 它的注入 token 成本在报告里**整个缺席**。 |
| `7b4336d` | Raw 臂的摘要调用原本没有 `role_scope`，落进 `calls/unscoped`，solver/management 两个总数都不包含它 —— 会让 Raw 看起来几乎零管理开销。新增独立 role `raw_summarizer`。附带以真实 `Agent`（而非 ScriptedAgent stub）做集成测试，因为 `install_accounting` 打的是**类级** patch，非 Agent 的 stub 根本不经过它 —— 这正是该缺口能穿过 18 个已通过测试的原因。 |
| `8f0848e` + `62b024b` | 按 role 的调用/token 计数与 seed 注入（类级 patch，因为上游的 summarizer 与 aggregator 都在函数内部自建 `Agent`，外部拿不到引用）；随后修 role 在上游线程池中的继承（第一次干净运行里 38 次 solver 侧调用有 9 次落进 `unscoped`），并顺带注入 `extra_body`。 |
| `385104d` | 用测试钉死 `batch_size` 与 `max_workers` 是**两个独立旋钮**（见 §7.3）。 |
| `e756835` | 构建三条臂共享的固定 AIME 题流：严格按年份切分、剔除不可打分的 `2022-II-8`、topic 标注**一次性冻进文件**使三条臂逐字节一致、行布局同时满足 `harness.load_stream` 和上游 `run_problem` 两个读者、`gt_traj` 刻意留空。 |

### D. 只有真机运行才暴露的缺陷（300+ 通过的测试都没抓到）

| commit | 缺陷 |
|---|---|
| `2aedcb1` | 首次端到端运行的**四道题全部死于 `KeyError: 'data_source'`**。loader 的字段白名单已为此带上了 `gt_traj` / `ground_truth`，独独漏了 `data_source`。它的可怕之处在于：`KeyError` 在 worker 线程里抛出、被 `run_stream` 捕获降级、run 照常"跑完" —— 不是崩溃，是**静默**。现以一个测试钉死：枚举上游用 `[...]`（而非 `.get(...)`）读 `current_problem` 的每一个 key。 |
| `9e1b554` + `36bc29f` | `extract_result` 是对着一个**并不存在的 `step_outputs` schema** 写的。真实结构把 policy 和 verifier 条目交错在同一个列表里、用 `role` 区分、`policy_actions` 是 list 不是 str、轮次叫 `evolving_round`、根本没有 per-round `verifier_report` 字段。接到真 `run_problem` 上，每道题都会返回 0 分和空反馈，Reflect 永远收不到可用输入，harness 整条 run 保持为空，**而且不会有任何东西报错**。`36bc29f` 另外发现 `role` 有第三个取值 `verifier_aggregation`，并首次引入一个**取自真实执行**的 fixture（`tests/harness/fixtures/real_problem_payload.json`）。 |
| `70e77b6` | 两条跨题臂都用 `pass_final == 0` 判断"有东西可学"，而驱动器给死题的全零结果 `pass_final` 也是 0。改为按 `round_count` 判断（`has_trajectory`）。 |
| `5cd0d26` | 首 batch 全失败的断路器（见 §5.6 末行）。 |
| `62b024b` | DashScope 对 qwen3 hybrid-reasoning 模型的非流式调用**直接拒绝**，除非请求体带 `enable_thinking: false`；以及上游线程池吞掉 role。 |
| `93b6609` | 钉住 `scipy` 与 `httpx[socks]`（见 §2.2）。 |
| `9f51c21` | wandb 0.30 在"已安装但从未登录"（新 checkout 的默认状态）下，`wandb.init()` 会阻塞约 4 s 等交互式 API-key 输入，然后抛 **`KeyboardInterrupt`** —— 那是 `BaseException`，`except Exception` 接不住，于是 tracker 构造函数把整个进程带走，而为这种情况准备的 jsonl 回退**一行都没写出来**。现在改为 init 之前先查凭证（光是走到 init 就意味着阻塞 + 往 run log 里倒注册横幅），并显式 catch `KeyboardInterrupt`。 |
| `064e799` / `e62ea41` | general 层跨题数强制、curator 越层改写 —— 完整叙述见 §9.3。 |

### E. 可审计性与导出

| commit | 改了什么 / 为什么 |
|---|---|
| `d56d35a` + `1fd5cf1` + `3ec612e` | 持久化 store + 两份 JSONL 日志；两阶段 edit 应用（DELETE/MERGE/REVISE/SKIP 先行，ADD 再对**阶段一之后**的占用量做检查，否则满了的 harness 只会增长不会换血）；容量上界在代码里强制。 |
| `f475ff5` + `e6d4512` | 文件名由 **id** 派生而非不可信的 `name`（同名不同 id 的 skill 曾静默覆盖同一文件；`../../pwned` 能写出 `skills_dir`）；frontmatter 分隔符按**整行**匹配而非子串（LLM 写的 `name` 里出现字面 `---` 会把 JSON 劈成两半）。另修 `_next_id()`：原先纯按当前持有的 skill 推导下一个 id，同 batch 内 DELETE-then-ADD 会**重发刚释放的 id**，在 `harness_log.jsonl` 的历史里把两条无关 skill 别名成同一个。 |
| `87dab18` | 让每一条候选都能追溯到提出它的题：① `harness_log` 的 `problem_idx` 是 **batch** 索引（一次 `apply()` 的 edit 来自 batch 里多道失败的题），新增 `source_problems` 字段从 `p_<idx>` evidence 标签推导；② `REVISE` 原本绑 `evidence=[]`，无从追溯，改为继承候选池 evidence 的并集（这确实是 curator 看到的东西；指名单一候选会是更精确的谎言）；③ **curator 从未提及**的候选原本不产生任何 log 行 —— 与 `store.py` docstring 自己许诺的"保留每一条被提出但被拒绝的候选"矛盾 —— 现在以合成 SKIP（reason `not referenced by the curator's reply`）走既有的拒绝路径。 |
| `2aa8e9f` | `export_harness` / `build_report`：导出最终 harness（`harness.md`）、机读汇总（`summary.json`）、演化日志副本（`evolution.jsonl`）。`build_report` 纯函数、不碰磁盘，数字可直接断言。usage 从 **selection log** 关联而非读各 Skill 的计数器：被后续 curator 删掉的 skill 已经没有计数器可读了，但它此前的注入**确实发生过、确实花过 token**，丢掉它会低估 harness 的真实注入成本（这类 skill 以 `present_in_final_harness: False` 出现）。 |
| `c80dbab` + `b9d8a8a` | wandb + jsonl 双写 tracker（evo 路径此前完全没有 wandb 集成）；六个 run 的命名与分组。 |

### F. 计划与设计文档中被推翻的东西

| commit | 内容 |
|---|---|
| `0fd3c59` + `a2663b4` | 前六个任务修掉的八个缺陷里，**几乎全部来自计划文档自己提供的参考代码**，而不是实现者写错。最清楚的一例是 `_NUMBER` 正则对句末数字的盲区：计划的实现漏了它，计划的测试恰好写成 `'...738 here.'`（数字后面跟着单词），于是测试从洞上方跨了过去。**实现和与之同源的测试共享同一个盲点。** 因此计划文档里九个未开始任务的实现代码被改写成"方法说明"，只保留真正具契约性的东西（签名、reject reason 字符串、预算数值，以及 Reflect/curator 的 prompt 模板 —— 它们本身就是交付物）；并要求每个实现者**另外自行设计三个对抗性用例**。 |
| `af394cc` | 数据源靠**实际加载**确认而非读 dataset card：`gneubig/aime-1983-2024` 的 2024 split 只有 14 题（仅 Part II）、2023 只有 29 题，都不能充当 held-out 年份；`MathArena/aime_2025` 带人工 `problem_type` 标签，这既回答了"四个 topic 对 AIME 是否太粗"，也提供了衡量离线标注器一致率的 30 题验证集。 |
| `20a6efa` | 计划里"每个新 `.py` 都要 Apache 头"与计划自己所有测试文件示例都不带头相矛盾；仓库本身也只有 94 个非 verl Python 文件中的 29 个带头。约束收窄到 `alphaapollo/` 源文件。 |

---

## 11. 已知限制与失败尝试

### 11.1 地板效应 —— 事前的担心，事后**基本被证实**

跑 Task C 之前本节预测：`qwen3-8b` 的正确率太低、held-out 只有 30 题，"即使方法有效，这个配置也大概率给出 null result"。实测结果（§8.2）与该预测一致，且可以把它写得更精确：

- **held-out 三臂最终正确率是 10.0% / 23.3% / 16.7%，n = 30 —— 单题 = 3.3 个百分点。** 三个配对 McNemar 检验的不一致样本数分别只有 4、4、4，p ≥ 0.125。这个尺寸的 held-out **在结构上就无法**分辨 arm 之间的差异，与方法是否有效无关。
- **成功信号稀疏这一条完全应验。** `utility() = (n_selected_success + 1) / (n_selected + 2)` 在 144 题之后给出的效用仍然贴着先验：15 条被选中过的 skill 里，13 条的命中率落在 12.5%–27.3% 这个和 base rate（约 22%）无法区分的带里（§8.7）。唯一跳出来的 `sk_0001`（50%，n=18）也可能只是它恰好被选到了简单题上。
- 但**地板效应不再是排在第一位的解释**。§8.9 给出了两个更具体、更可修的嫌疑：skill 文本的语义漂移（§11.3）与 verifier 噪声经由题内演化轮破坏最终答案（§8.8）。地板效应决定了"分辨不出差异"，而这两项决定了"即使样本量足够也未必分辨得出" —— 写 null result 分析时两者要分开讲。

### 11.2 标注器的 geometry 偏差

见 §6.2。`qwen3-32b` 一致率 83%，但 n=30 意味着约 ±13pp 的区间。更具体的问题是：adaptation split 的 geometry 占比 35%，人工标注的 held-out split 只有 23% —— **per-topic 拆解表必须带这个 caveat**，否则会把标注偏差读成分布偏移。

### 11.3 语义漂移、跨层重复与 topic 粒度

**语义漂移（Task C 暴露的头号缺陷）。** MERGE 与 REVISE 会整条改写 skill 的 `trigger` / `lesson` / `failure_mode`，没有任何机制保证改写后 trigger 仍然描述 lesson。144 题之后这成了系统性的：

`sk_0006` 的完整轨迹（`export-evo/evolution.jsonl`，10 次 accepted 编辑）：

| batch | op | trigger |
|---|---|---|
| b0 | ADD | When solving a problem, ensure all possible cases are considered… |
| b3 | REVISE | When multiple conditions or constraints are present in a problem. |
| b6 | REVISE | When multiple constraints or conditions are present in a problem. |
| b14 | REVISE | When multiple constraints are present in a problem. |
| b15 | REVISE | When multiple constraints or relationships exist in a problem. |
| **b16** | **MERGE** | **When a circle is tangent to multiple sides of a figure and intersects a diagonal.** |

两个方向的退化同时发生：前半程 trigger 被反复"泛化"成对几乎任何 AIME 题都成立的空话，末尾一次 MERGE 又把一条**圆的切线**的具体几何 lesson 并了进来。最终导出的 `harness.md` 里这条 skill 的标题是泛化的 `when-solving-a-problem-ensure-all-possible-cases`，body 讲的却是坐标几何切线 —— **trigger 与 lesson 已经不描述同一件事**。`sk_0013`、`sk_0010` 是同样的模式。

后果是可测的：`sk_0006` 被注入 **118/144 题（82%）**，命中率 22.9% —— 等于 base rate，是一条占满注入槽的 no-op（§8.7）。选择器并非选错，而是**在漂移后的 trigger 上无从区分**。

两个现成的修法（都未实现）：(a) MERGE/REVISE 之后校验 trigger 与 lesson 的一致性，不一致则拒绝；(b) 给每条 skill 记录编辑次数，超过阈值就冻结或强制分裂。

**`wrong_layer` 浪费。** 209 次 curator 决策里 **49 次（23%）**因 curator 试图改另一层的 skill 而被 `e62ea41` 的闸门拒掉。闸门本身是对的（它防的是越层改写），但被拒的决策消耗了完整的一次模型调用。这是 prompt 契约的问题：curator 没有被有效地约束到只看自己那一层的候选。

- **跨层重复**：§9.3(a) 里 topic 层与 general 层出现逐字重复的 skill，两份拷贝还要抢同一份注入预算。`064e799` 用"general 必须跨 ≥2 题"拦住了最明显的一类，但**它拦的是来源，不是内容** —— 两条来自不同题目的 general skill 仍然可能语义重复，目前只能靠 curator 自己 MERGE。Task C 证实这没被解决：p_68 一次注入的 5 条 skill 里有 4 条（`sk_0006`/`sk_0012`/`sk_0013`/`sk_0014`）说的是同一件事（"系统枚举、逐条校验、不要假设"），311 token 里大部分是重复内容。
- **没有淘汰机制**：`sk_0004` 在 144 题里**一次都没有被选中**，却始终占着 `isosceles_triangle_counting` 的 topic 槽位直到最后。cap 限制了总量，但没有任何压力把无用的 skill 挤出去。
- **topic 名字偏窄**：一次 4 题运行里模型给出的 topic 是 `inradius_calculation`。这种粒度的 bucket 几乎不可能有第二道题命中，topic 层会碎成"每个 bucket 一条 skill"。`existing_topics` 传入就是为了缓解这一点（让模型复用已有名字），但它只是一个**提示**，没有强制。烟囱测试里产出的 `counting_with_constraints` / `coordinate_geometry_setup` 粒度明显更合理，说明这在很大程度上取决于模型当次的发挥。

### 11.4 若干缺陷在 300+ 通过的测试下完全不可见

`2aedcb1`（`data_source` KeyError）、`9e1b554`（虚构的 `step_outputs` schema）、`70e77b6`（从空轨迹编译 skill）、`62b024b`（线程池吞 role）、`064e799` / `e62ea41`（模型忽略 prompt 约束）—— 这些都是在测试套件已有几百个通过用例时，**第一次接上真实 API** 才暴露的。共同模式有两种：

1. **手写 fixture 与手写实现共享盲点**（见 §10-F）。补救是 `tests/harness/fixtures/real_problem_payload.json` —— 本包第一个取自真实执行的 fixture。
2. **韧性机制掩盖配置错误**。每一层降级都工作正常，结果是一个彻底坏掉的 run 安静地"完成"。补救是首 batch 断路器与 frozen-空状态前置检查 —— 它们都是**故意不降级**的地方。

这一点应当写进 slides：本项目最有价值的工程结论之一，是"降级"和"可观测"必须成对设计。

### 11.5 其他已知问题

- **~~`problem_shape` 恒为空串~~ —— 已删除（`474de1b`）。** 它曾是 Reflect 白名单里的一个字段，但 `prepare_harness_stream.py` 写死 `""`，149 道题全部为空，于是每次真实 Reflect 调用都发出一行**有标签、无内容**的 `Problem shape: `。提示词里的空字段不是中性的，它读起来像"模型本该知道却缺失的信息"。选择删除而非填充：论文 Appendix E.1 的 proposal 输入清单里没有这一项，而模型自己命名的 `TOPIC` 已经回答了"这是什么类型的题"；填充它要多花 149 次调用去换方法并不要求的字段。
- **`run()` 本身没有被测试覆盖。** `run_stream` / `extract_result` / `assert_no_gt_tool_call` 被 `tests/harness/test_driver.py` 用最小和完全真实的两种 payload 覆盖，但 `run()` 这段配置装配代码没有。它已经通过 scratch 配置对真实 API 跑通过（§9），`run()` 的 docstring 里那句 "has never been invoked against a real, running `run_problem`" 曾经因此过时，已在 `474de1b` 更正为实际的运行记录。
- **`loader.interleave()` 是保留但未被使用的代码。** topic 交错的题流（设计文档 S15.3）在 `e756835` 里被年份序取代，函数和它的 16 个测试都还在，但驱动器不调用它。`474de1b` 在它的 docstring 里写明了这个状态以及保留（而非删除）的理由：若实验显示 skill 的 mint-to-reuse 间隔过长，它是现成的备选排序，且任务书明确允许主题交错。
- **设计文档 `docs/design/cross-problem-skill-harness-design.md` 部分已过时**：它早于"模型驱动 selector"（`f591906`）与"模型命名 topic"（`4015c8b`）两次改动。**代码与 git log 是权威**，设计文档与之冲突处以前者为准。
- **`store.select()`（确定性检索）与 `selector.select_skills()`（模型检索）并存。** 前者仅供 store 层测试与参考，实验路径走后者；`store.py` 的 docstring 已明确说明，但读代码的人仍可能走错路。

### 11.6 失败的尝试（都已回滚或取代）

| 尝试 | 为什么放弃 |
|---|---|
| 确定性词重叠 + 硬 topic 过滤做 Select | 偏离论文，且隐式要求逐题 topic 标注（`f591906`） |
| 把 topic 作为 Reflect 的**输入** | 与 Appendix E.1 相反，凭空制造 topic 标注依赖（`4015c8b`） |
| topic 交错题流，缩短 mint-to-reuse 延迟 | AIME 一年 30 题内部本就四个 topic 混排；且交错需要**运行前**的 topic 标签，而那正是方法后来被证明不需要的依赖（`e756835`） |
| `answer_leak` 用裸数字相等 | 实测误拒 21.7–27.9% 的枚举类 skill（`1ff380f`） |
| batch size 16（论文默认） | 149 题只给约 9 个更新点，看不出增长趋势 |
| `pkgutil` + 子串搜索做边界守卫 | 同时漏报 `__init__.py` 与误报 docstring（`35a811b`） |
| 把 Raw 臂的摘要调用记在 `summarizer` role 下 | 会让 Raw 看起来零管理开销（`7b4336d`） |

---

## 12. 尚未完成

以下内容**尚不存在**，请勿在任何地方当作已完成引用：

1. **Slides。**
2. **多 seed 重复。** Task C 只跑了**单 seed 一轮**（§12.1）。这是本项目结论强度上最大的单一缺口 —— 三臂之间的差距（2–3 个百分点）正好是单 seed 分辨不了的量级。
3. **p_84 的重复采样对照。** §8.8 的正向迁移案例给出了可复核的**行为差异**（搜索上界 100 vs 1000），但没有在不注入的条件下把该题重跑 N 次，因此"skill 导致了这个差异"是未证实的假设而非结论。

一项已知的文档欠债（§11.5）：设计文档 `docs/design/cross-problem-skill-harness-design.md` 部分过时。（`run()` docstring 与 `problem_shape` 两项已在 `474de1b` 处理。）

### 12.1 会影响结论解读的三件事

**噪声底是 ±2 题 / 20 题。** 同一份配置重跑两次，final 总数一样但**有 2 道题翻转**（`docs/findings/diagnostics/`，diagA vs diagA2）。折到 144 题约 ±3.6 个百分点。**arm 之间若只差 2–3 个百分点，单 seed 分不出来** —— 论文自己也是 3 次取平均（Appendix F）。Task C 实测的差距恰好落在这个区间（evo 比 baseline 高 2.8pp），三个配对 McNemar 检验也全部不显著（§8.2）。**本实验的所有 arm 间比较都应当以"未能分辨"而非"A 优于 B"陈述。**

**seed 不给可复现性。** seed 确实注入了每一次请求（日志里没有 provider 拒绝的警告），但 DashScope 的 seed 是 best-effort：两个 policy 生成参数完全相同的 run，round-0 结果仍会在个别题上不一致。**不要把"三臂 round-0 应逐题一致"写成断言** —— 那会是一个必然误报的 gate。实测三臂 round-0 分别是 22.2% / 16.7% / 23.6%，本来就不一致。

**`pass_final` 取最后一轮，而最后一轮可能比第一轮更差。** §8.8 测到：round 0 做对、最终做错的题，baseline 有 **8** 道、evo **6** 道、raw **1** 道。这个量级**大于三臂之间的全部差距**，而它由"最终消息有多长"这样一个与 skill 质量无关的变量驱动（verifier 按可见推理打分，简短的正确答案会被误判、随后在下一轮被改坏）。解读任何 arm 间差异之前必须先看这张表。

---

## 附：本文档中每个数字的来源

| 数字 | 来源 |
|---|---|
| 619 tests / 4.64s | 本机 `python -m pytest tests/ -q` |
| Task C 全部结果表（§8） | `outputs/harness/report/results.{md,json}`（由 `analysis.py` 生成）、`outputs/harness/export-evo/{harness.md,summary.json,evolution.jsonl}`、各 run 的 `metrics.jsonl` 与 `selection_log.jsonl` |
| McNemar 精确检验 p 值 | 对各 run `metrics.jsonl` 的 `adapt/pass_final` 做逐题配对计算 |
| 迁移案例的轨迹级细节 | `outputs/harness/{adapt-baseline,adapt-evo}/trajectories/problem_{0084,0105,0068}.json`，分析见 `docs/findings/transfer-cases.md` |
| loss/gain 与消息长度（8 / 1 / 6，1827 / 766 / 1302 字符） | 遍历三臂 144 份轨迹的 `step_outputs` 计算 |
| 140 files / 26980 insertions / 2 deletions | `git diff --stat 712a04d..HEAD`（含已入库的 Task C 产物）|
| adaptation 149、held-out 30、按年与 topic 分布 | 重新读取 `data/harness/*.parquet` 校验 |
| 标注器一致率 70% / 83% / 83% | `alphaapollo/core/harness/topic.py` 模块 docstring 记录的实测 |
| 4% 的 AIME 答案恰为常见界数、21.7–27.9% 误拒率 | commit `1ff380f` 记录的对真实 adaptation pool 的测量 |
| 12 并发 / 16 限流 / 24 丢 40%、16.5 s 中位延迟 | commit `d32a9f6` 记录的实测 |
| 烟囱测试的全部数字（1354.7 s、2/12、93+20 调用、注入 token 等） | 该次运行的 `metrics.jsonl`、`harness_log.jsonl`、`selection_log.jsonl`、`skills/*.md` 与 stdout |
| `Caps` / `Budget` 默认值 | `alphaapollo/core/harness/store.py:149-159` |
| reject reason 全集 | `guard.py` 与 `store.py` 源码 grep |
