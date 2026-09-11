# Cross-Problem Skill Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 AlphaApollo 的 `informal_math_evolving` 之上加一层跨问题 skill 演化闭环（Task A + Task B + 三个对照 arm 的运行基建），与已有的题内 solution memory 完全隔离，且对上游文件零改动。

**Architecture:** 新增独立包 `alphaapollo/core/harness/`（不 import `core/environments/memory/`）+ 新 driver `evolving_harness_main.py`。harness 通过每个 batch 设置一次 `policy_agent.system_prompt` 注入（`utils/agent.py:37` 每次调用时才读该属性），因此不需要改任何 prompt 模板、`run_problem` 或 `env.py`。执行协议是 **batch 间严格串行、batch 内并行**：同 batch 的题共用冻结的 `H_i`，彼此无信息流动。

**Tech Stack:** Python 3.12 / dataclasses / OmegaConf / pytest / openai SDK / wandb（可 disabled）。**不引入** embedding 模型、向量库、tiktoken 或任何新的重量级依赖。

## Global Constraints

这些约束对**每一个** task 都隐含生效：

- **对上游已有文件的改动必须为 0。** 记账通过 monkey-patch 类方法实现，数据加载自写。唯一允许新增的目录是 `alphaapollo/core/harness/`、`tests/`、`scripts/`，以及 `alphaapollo/core/generation/evolving/evolving_harness_main.py`、`alphaapollo/data_preprocess/prepare_harness_stream.py`、`examples/configs/harness_*.yaml`。`pyproject.toml` 只允许追加 `[tool.pytest.ini_options]` 一节。
- **`alphaapollo/core/harness/` 中任何模块都不得 import `alphaapollo.core.environments.memory`。** 这是概念边界的结构性保证。
- **`reflect.py` / `evolver.py` / `render.py` 中不得出现标识符 `ground_truth`。** 两个例外，README 需如实说明：`guard.py`（负向过滤需要 GT 才能检测答案泄漏，只 reject 不 generate，属信息单向删除）；`arms.py`（仅把 GT 透传给 `store.apply` → `guard.validate_skill`，自身不读取、不拼进任何 prompt）。GT 到达 harness 的路径有且只有这一条，且终点是 reject。
- **所有 skill 文本必须是英文。** AIME 题面是英文，中文 trigger 会让词面检索恒等于 0。
- **预算硬编码默认值**：`Caps(general=5, per_topic=5)`；`Budget(b=6, general_max=3, topic_max=4, tokens=800)`；batch size `B=8`。
- **许可证头**：所有新增 `.py` 文件顶部加 `# Copyright 2026 TMLR Group` + Apache 2.0 声明块，与 `evolving_main.py:1-13` 格式一致。
- **测试不得发起真实网络请求。** 所有涉及 LLM 的 task 用 stub agent。
- 代码风格跟随仓库现有配置：`ruff`，`line-length = 300`。

**环境**（已于 2026-09-11 验证可用）：conda 环境 `alphaapollo-dev`（Python 3.12.14, macOS arm64），`pip install -e . --no-deps` + 手工最小依赖集。已验证可 import：`Agent`、`run_problem`、`create_runtime_for_problem`、`load_informal_math_data`、`pandas/openai/omegaconf/fire`、`wandb 0.30.0`、`pytest 9.1.1`、`ruff 0.16.7`。

⚠️ **两个必须知道的环境事实**：

1. **`alphaapollo` 包并未被安装**。`pyproject.toml` 的 `packages.find where = ["alphaapollo/core/generation"]` 只暴露了 `verl`；实测 `cd /tmp && python -c "import alphaapollo"` 报 `ModuleNotFoundError`。**所有命令必须从仓库根目录执行**，或让 PYTHONPATH 包含它。`tests/conftest.py`（Task 0）与 `[tool.pytest.ini_options].pythonpath` 负责在测试里兜住这一点。
2. **`alphaapollo/core/__init__.py` 的最后一行是裸调用 `ensure_verl_alias()`**，无条件加载整个 verl 栈（ray / tensordict / torch / transformers）。实测 `import alphaapollo.core` 耗时约 2.9s。这是硬编码的包级耦合，绕不开；但该开销在一次 pytest session 内只付一次，不影响 TDD 循环，因此 harness 包保持在 `alphaapollo/core/harness/` 不动。

**运行测试的标准命令**（每个 task 的验证步骤都用它）：

```bash
conda activate alphaapollo-dev
cd AlphaApollo && ./scripts/run_tests.sh -v
```

**分支**：`feat/cross-problem-skill-harness`（已创建）。

---

## File Structure

| 文件 | 职责 |
| --- | --- |
| `alphaapollo/core/harness/schema.py` | `Skill` / `CandidateMemory` / `SkillEdit` 数据类；markdown ↔ dataclass 序列化 |
| `alphaapollo/core/harness/guard.py` | `validate_skill()` —— 防泄漏、语言、长度、格式校验。**唯一接触 GT 的模块** |
| `alphaapollo/core/harness/store.py` | `SkillStore` —— 持久化、变更日志、两阶段 apply、分层选择 |
| `alphaapollo/core/harness/render.py` | skills → system message 文本；`count_tokens()` |
| `alphaapollo/core/harness/accounting.py` | monkey-patch `Agent.get_action_from_gpt`：按 role 记账 + 注入 seed |
| `alphaapollo/core/harness/reflect.py` | `ReflectContext` 白名单构造 + feedback 净化 + `reflect()` |
| `alphaapollo/core/harness/evolver.py` | `TopicCurator` / `GeneralCurator` + 输出解析 |
| `alphaapollo/core/harness/arms.py` | `CrossProblemArm` 协议 + Baseline / RawExperience / EvoHarness |
| `alphaapollo/core/harness/loader.py` | 带 topic/year/technique 的问题加载（不复用 `load_informal_math_data`） |
| `alphaapollo/core/harness/tracker.py` | wandb + jsonl 双写 |
| `alphaapollo/core/generation/evolving/evolving_harness_main.py` | driver：batch 间串行 / batch 内并行 |
| `alphaapollo/data_preprocess/prepare_harness_stream.py` | AIME 按年取数 + topic/technique 标注 + 主题交错 |
| `scripts/reuse_opportunity.py` | Day 2 门禁：复用机会分析 |

拆分原则：`store.py` 只管状态与预算，不碰 LLM；`reflect.py` / `evolver.py` 只管 LLM 调用与解析，不碰持久化；`guard.py` 独立出来是因为它是唯一的 GT 白名单例外，隔离便于审计。

---

## Task 0: 测试基建与包骨架

**Files:**
- Create: `tests/__init__.py`, `tests/harness/__init__.py`, `tests/conftest.py`
- Create: `alphaapollo/core/harness/__init__.py`
- Create: `scripts/run_tests.sh`
- Create: `tests/harness/test_smoke.py`
- Modify: `pyproject.toml`（仅追加 `[tool.pytest.ini_options]`）

**Interfaces:**
- Consumes: 无
- Produces: 可运行的 pytest 环境；`alphaapollo.core.harness` 可 import

仓库目前没有 `tests/` 目录也没有 pytest 配置，且 `pyproject.toml` 的 `[tool.setuptools.packages.find] where = ["alphaapollo/core/generation"]` 只打包 `verl` —— `alphaapollo` 包依赖 PYTHONPATH 导入（见 `alphaapollo/workflows/api.py:38-47`）。`conftest.py` 负责把两条路径插进 `sys.path`。

- [ ] **Step 1: 写失败的冒烟测试**

`tests/harness/test_smoke.py`:

```python
def test_harness_package_importable():
    import alphaapollo.core.harness as h
    assert h.__name__ == "alphaapollo.core.harness"


def test_harness_does_not_import_inproblem_memory():
    """概念边界的结构性保证：harness 包不得依赖题内 memory。"""
    import importlib
    import pkgutil

    import alphaapollo.core.harness as h

    offenders = []
    for mod in pkgutil.iter_modules(h.__path__):
        m = importlib.import_module(f"alphaapollo.core.harness.{mod.name}")
        src = open(m.__file__, encoding="utf-8").read()
        if "core.environments.memory" in src:
            offenders.append(mod.name)
    assert offenders == [], f"harness modules import in-problem memory: {offenders}"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd AlphaApollo && PYTHONPATH="$PWD:$PWD/alphaapollo/core/generation" python -m pytest tests/harness -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'alphaapollo.core.harness'`（或 pytest 找不到 tests 目录）

- [ ] **Step 3: 建包骨架与 conftest**

`tests/conftest.py`:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION = REPO_ROOT / "alphaapollo" / "core" / "generation"

for p in (REPO_ROOT, GENERATION):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
```

`tests/__init__.py` 和 `tests/harness/__init__.py`：空文件。

`alphaapollo/core/harness/__init__.py`:

```python
# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cross-problem skill harness.

This package is deliberately independent of
``alphaapollo.core.environments.memory`` (the in-problem solution memory).
Nothing here may import that package.
"""
```

追加到 `pyproject.toml` 末尾：

```toml
# -------------------------------
# tool.pytest
# -------------------------------
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = [".", "alphaapollo/core/generation"]
```

`scripts/run_tests.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH="$PWD:$PWD/alphaapollo/core/generation" python -m pytest tests/harness "$@"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `chmod +x scripts/run_tests.sh && ./scripts/run_tests.sh -v`
Expected: PASS，2 passed

- [ ] **Step 5: 提交**

```bash
git add tests/ scripts/run_tests.sh alphaapollo/core/harness/__init__.py pyproject.toml
git commit -m "test: add pytest harness scaffolding and package boundary guard"
```

---

## Task 1: schema.py —— 数据类与 markdown 序列化

**Files:**
- Create: `alphaapollo/core/harness/schema.py`
- Test: `tests/harness/test_schema.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `@dataclass Skill`：字段 `id: str, name: str, level: str, topic: str | None, trigger: str, lesson: str, failure_mode: str, evidence: list[str], created_at: int, revised_at: list[int], n_selected: int, n_selected_success: int, n_tokens: int`
  - `@dataclass CandidateMemory`：`trigger: str, lesson: str, failure_mode: str, scope_hint: str, topic: str | None, evidence: list[str], action_hint: str = "NEW", target_id: str | None = None`
    - `action_hint ∈ {"NEW", "ENHANCE"}`，对齐论文附录 E.1 的 *"decide NEW, ENHANCE, or NONE"*（NONE 时 Reflect 整体返回 `None`，不构造 CandidateMemory）。`target_id` 只在 ENHANCE 时有值，给 curator 一个偏向 MERGE/REVISE 的强信号。
  - `@dataclass SkillEdit`：`op: str, actor: str, reason: str, skill_id: str | None = None, payload: CandidateMemory | None = None`
  - `OPS: frozenset = {"ADD", "MERGE", "REVISE", "DELETE", "SKIP"}`
  - `skill_to_markdown(skill: Skill) -> str`
  - `skill_from_markdown(text: str) -> Skill`
  - `Skill.utility() -> float`（Laplace 平滑：`(n_selected_success + 1) / (n_selected + 2)`）

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_schema.py`:

```python
import pytest

from alphaapollo.core.harness.schema import (OPS, CandidateMemory, Skill,
                                             SkillEdit, skill_from_markdown,
                                             skill_to_markdown)


def make_skill(**kw) -> Skill:
    base = dict(
        id="sk_0007", name="enumerate-before-generalize", level="topic",
        topic="number_theory",
        trigger="Counting integers subject to divisibility or congruence conditions.",
        lesson="- Brute-force small range in python first.\n- Check the closed form back against the enumeration.",
        failure_mode="Extrapolating a small-range pattern without numeric verification.",
        evidence=["p_0023:symbolic_slip"], created_at=23, revised_at=[41],
        n_selected=12, n_selected_success=7, n_tokens=94,
    )
    base.update(kw)
    return Skill(**base)


def test_markdown_round_trip_preserves_every_field():
    original = make_skill()
    restored = skill_from_markdown(skill_to_markdown(original))
    assert restored == original


def test_markdown_uses_the_three_section_layout():
    text = skill_to_markdown(make_skill())
    assert "## When to use" in text
    assert "## Strategy" in text
    assert "## Avoid" in text


def test_utility_is_laplace_smoothed():
    assert make_skill(n_selected=0, n_selected_success=0).utility() == pytest.approx(0.5)
    assert make_skill(n_selected=12, n_selected_success=7).utility() == pytest.approx(8 / 14)


def test_general_skill_has_no_topic():
    s = make_skill(level="general", topic=None)
    assert skill_from_markdown(skill_to_markdown(s)).topic is None


def test_skill_edit_rejects_unknown_op():
    with pytest.raises(ValueError):
        SkillEdit(op="FROBNICATE", actor="topic_curator", reason="x")


def test_candidate_memory_rejects_unknown_scope_hint():
    with pytest.raises(ValueError):
        CandidateMemory(trigger="t", lesson="l", failure_mode="f",
                        scope_hint="sideways", topic=None, evidence=[])


def test_candidate_memory_defaults_to_a_new_skill_proposal():
    c = CandidateMemory(trigger="t", lesson="l", failure_mode="f",
                        scope_hint="topic", topic="number_theory", evidence=[])
    assert c.action_hint == "NEW" and c.target_id is None


def test_candidate_memory_rejects_unknown_action_hint():
    with pytest.raises(ValueError):
        CandidateMemory(trigger="t", lesson="l", failure_mode="f", scope_hint="topic",
                        topic="number_theory", evidence=[], action_hint="OBLITERATE")


def test_ops_frozen_set_is_exactly_the_five_operators():
    assert OPS == frozenset({"ADD", "MERGE", "REVISE", "DELETE", "SKIP"})
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_schema.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'alphaapollo.core.harness.schema'`

- [ ] **Step 3: 实现**

`alphaapollo/core/harness/schema.py`（顶部加 Apache 头，下略）:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

OPS = frozenset({"ADD", "MERGE", "REVISE", "DELETE", "SKIP"})
LEVELS = frozenset({"general", "topic"})
SCOPE_HINTS = frozenset({"general", "topic"})
ACTION_HINTS = frozenset({"NEW", "ENHANCE"})


@dataclass
class Skill:
    id: str
    name: str
    level: str
    topic: str | None
    trigger: str
    lesson: str
    failure_mode: str
    evidence: list[str] = field(default_factory=list)
    created_at: int = 0
    revised_at: list[int] = field(default_factory=list)
    n_selected: int = 0
    n_selected_success: int = 0
    n_tokens: int = 0

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {sorted(LEVELS)}, got {self.level!r}")

    def utility(self) -> float:
        return (self.n_selected_success + 1) / (self.n_selected + 2)


@dataclass
class CandidateMemory:
    trigger: str
    lesson: str
    failure_mode: str
    scope_hint: str
    topic: str | None
    evidence: list[str] = field(default_factory=list)
    # Paper Appendix E.1: the proposal step decides NEW / ENHANCE / NONE.
    # NONE is represented by returning no CandidateMemory at all.
    action_hint: str = "NEW"
    target_id: str | None = None

    def __post_init__(self) -> None:
        if self.scope_hint not in SCOPE_HINTS:
            raise ValueError(f"scope_hint must be one of {sorted(SCOPE_HINTS)}, got {self.scope_hint!r}")
        if self.action_hint not in ACTION_HINTS:
            raise ValueError(f"action_hint must be one of {sorted(ACTION_HINTS)}, got {self.action_hint!r}")


@dataclass
class SkillEdit:
    op: str
    actor: str
    reason: str
    skill_id: str | None = None
    payload: CandidateMemory | None = None

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"op must be one of {sorted(OPS)}, got {self.op!r}")


_FRONTMATTER_KEYS = ("id", "name", "level", "topic", "evidence", "created_at",
                     "revised_at", "n_selected", "n_selected_success", "n_tokens")


def skill_to_markdown(skill: Skill) -> str:
    meta: dict[str, Any] = {k: getattr(skill, k) for k in _FRONTMATTER_KEYS}
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    lines += [
        "---",
        "## When to use",
        skill.trigger.strip(),
        "",
        "## Strategy",
        skill.lesson.strip(),
        "",
        "## Avoid",
        skill.failure_mode.strip(),
        "",
    ]
    return "\n".join(lines)


def skill_from_markdown(text: str) -> Skill:
    _, frontmatter, body = text.split("---", 2)
    meta = {}
    for line in frontmatter.strip().splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = json.loads(v.strip())

    sections: dict[str, list[str]] = {}
    current = None
    for line in body.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    def section(name: str) -> str:
        return "\n".join(sections.get(name, [])).strip()

    return Skill(
        trigger=section("When to use"),
        lesson=section("Strategy"),
        failure_mode=section("Avoid"),
        **meta,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_schema.py -v`
Expected: PASS，9 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/schema.py tests/harness/test_schema.py
git commit -m "feat(harness): skill schema with markdown round-trip serialization"
```

---

## Task 2: guard.py —— 防泄漏与格式校验

**Files:**
- Create: `alphaapollo/core/harness/guard.py`
- Test: `tests/harness/test_guard.py`

**Interfaces:**
- Consumes: `Skill`, `CandidateMemory`（Task 1）
- Produces: `validate_skill(candidate: CandidateMemory, *, question_texts: list[str], ground_truths: list[str], max_words: int = 200, max_lesson_words: int = 60) -> tuple[bool, str | None]` —— 返回 `(accepted, reject_reason)`

**这是唯一允许接触 `ground_truth` 的模块。** 它只 reject 不 generate，GT 绝不流入 harness。Global Constraints 里的 grep 检查把它列为白名单例外。

拒绝条件：
1. 非英文（非 ASCII 字符占比 > 5%）
2. `lesson` 词数 > 60，或三段合计词数 > 200
3. 与任一题面存在 8-gram 重合
4. 文本中出现与任一 GT 相同的孤立数字
5. 任一必填段为空

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_guard.py`:

```python
from alphaapollo.core.harness.guard import validate_skill
from alphaapollo.core.harness.schema import CandidateMemory

QUESTION = ("Find the number of ordered pairs of positive integers (a, b) such that "
            "a + b = 1000 and neither a nor b has a zero digit.")
GT = "738"


def candidate(**kw) -> CandidateMemory:
    base = dict(
        trigger="Counting integer pairs under digit constraints.",
        lesson="- Enumerate a small analogue in python before generalizing.",
        failure_mode="Extrapolating without numeric verification.",
        scope_hint="topic", topic="number_theory", evidence=["p_0001:symbolic_slip"],
    )
    base.update(kw)
    return CandidateMemory(**base)


def check(c):
    return validate_skill(c, question_texts=[QUESTION], ground_truths=[GT])


def test_clean_candidate_is_accepted():
    assert check(candidate()) == (True, None)


def test_non_english_is_rejected():
    ok, reason = check(candidate(lesson="- 先用 python 暴力枚举小范围再推广到一般情况。"))
    assert ok is False and reason == "non_english"


def test_verbatim_question_span_is_rejected():
    leaked = "- Find the number of ordered pairs of positive integers such that a + b"
    ok, reason = check(candidate(lesson=leaked))
    assert ok is False and reason == "question_overlap"


def test_ground_truth_digits_are_rejected():
    ok, reason = check(candidate(failure_mode="Forgetting that the answer is 738 here."))
    assert ok is False and reason == "answer_leak"


def test_numbers_unrelated_to_ground_truth_are_allowed():
    ok, reason = check(candidate(lesson="- Brute-force the range n <= 50 first."))
    assert ok is True and reason is None


def test_overlong_lesson_is_rejected():
    ok, reason = check(candidate(lesson="- " + " ".join(["word"] * 61)))
    assert ok is False and reason == "lesson_too_long"


def test_empty_section_is_rejected():
    ok, reason = check(candidate(trigger="   "))
    assert ok is False and reason == "empty_section"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_guard.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'alphaapollo.core.harness.guard'`

- [ ] **Step 3: 实现**

`alphaapollo/core/harness/guard.py`:

```python
from __future__ import annotations

import re

from alphaapollo.core.harness.schema import CandidateMemory

_WORD = re.compile(r"[A-Za-z0-9']+")
_NUMBER = re.compile(r"(?<![\w.])\d+(?![\w.])")
_NGRAM = 8
_NON_ASCII_RATIO = 0.05


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def validate_skill(
    candidate: CandidateMemory,
    *,
    question_texts: list[str],
    ground_truths: list[str],
    max_words: int = 200,
    max_lesson_words: int = 60,
) -> tuple[bool, str | None]:
    sections = (candidate.trigger, candidate.lesson, candidate.failure_mode)
    if any(not s.strip() for s in sections):
        return False, "empty_section"

    blob = "\n".join(sections)

    non_ascii = sum(1 for ch in blob if ord(ch) > 127)
    if blob and non_ascii / len(blob) > _NON_ASCII_RATIO:
        return False, "non_english"

    if len(_words(candidate.lesson)) > max_lesson_words:
        return False, "lesson_too_long"
    if len(_words(blob)) > max_words:
        return False, "skill_too_long"

    skill_ngrams = _ngrams(_words(blob), _NGRAM)
    for q in question_texts:
        if skill_ngrams & _ngrams(_words(q), _NGRAM):
            return False, "question_overlap"

    gt_numbers = set()
    for gt in ground_truths:
        gt_numbers.update(_NUMBER.findall(gt))
    if gt_numbers & set(_NUMBER.findall(blob)):
        return False, "answer_leak"

    return True, None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_guard.py -v`
Expected: PASS，7 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/guard.py tests/harness/test_guard.py
git commit -m "feat(harness): anti-leakage guard for candidate skills"
```

---

## Task 3: render.py —— 注入文本渲染与 token 计数

**Files:**
- Create: `alphaapollo/core/harness/render.py`
- Test: `tests/harness/test_render.py`

**Interfaces:**
- Consumes: `Skill`（Task 1）
- Produces:
  - `count_tokens(text: str) -> int` —— 启发式：`ceil(len(tokens) * 1.3)`，`tokens = re.findall(r"\w+|[^\w\s]", text)`
  - `NEUTRAL_SYSTEM_PROMPT: str = "You are a competition mathematics solver."`
  - `render_harness(skills: list[Skill]) -> str` —— 空列表时返回 `NEUTRAL_SYSTEM_PROMPT`

**为什么不用真 tokenizer**：引入 tiktoken/transformers tokenizer 会给预算计算带来模型依赖和测试不确定性。启发式只用于**预算控制**；成本表里报告的 token 数来自 `response.usage`（真实值，见 Task 4）。README 需说明这个区分。

**为什么空 harness 也要返回非空文本**：`utils/agent.py:37` 是 `if self.system_prompt:`，空串会导致整个 system message 不存在。Baseline 与 Evo 的第一个 batch 都处于这个状态，若返回空串，三个 arm 之间（以及 Evo arm 内部第 8 题前后）会多出一个结构变量。

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_render.py`:

```python
from alphaapollo.core.harness.render import (NEUTRAL_SYSTEM_PROMPT, count_tokens,
                                             render_harness)
from alphaapollo.core.harness.schema import Skill


def skill(i: int, level: str = "topic") -> Skill:
    return Skill(id=f"sk_{i:04d}", name=f"skill-{i}", level=level,
                 topic=None if level == "general" else "number_theory",
                 trigger=f"Trigger {i}.", lesson=f"- Lesson {i}.",
                 failure_mode=f"Avoid {i}.")


def test_count_tokens_is_monotonic_and_positive():
    assert count_tokens("") == 0
    assert 0 < count_tokens("hello world") < count_tokens("hello world again and again")


def test_empty_harness_renders_the_neutral_prompt_not_an_empty_string():
    out = render_harness([])
    assert out == NEUTRAL_SYSTEM_PROMPT
    assert out, "empty system_prompt would suppress the system message entirely"


def test_render_groups_general_before_topic():
    out = render_harness([skill(1, "topic"), skill(2, "general")])
    assert out.index("skill-2") < out.index("skill-1")


def test_render_includes_all_three_sections_of_each_skill():
    out = render_harness([skill(1)])
    assert "Trigger 1." in out and "- Lesson 1." in out and "Avoid 1." in out


def test_render_never_includes_frontmatter_metadata():
    out = render_harness([skill(1)])
    for leaked in ("n_selected", "evidence", "created_at", "sk_0001"):
        assert leaked not in out
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_render.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'alphaapollo.core.harness.render'`

- [ ] **Step 3: 实现**

`alphaapollo/core/harness/render.py`:

```python
from __future__ import annotations

import math
import re

from alphaapollo.core.harness.schema import Skill

NEUTRAL_SYSTEM_PROMPT = "You are a competition mathematics solver."

_TOKEN = re.compile(r"\w+|[^\w\s]")

_HEADER = ("You are a competition mathematics solver. The following reusable skills were "
           "distilled from your own past attempts on other problems. Apply the ones that fit; "
           "ignore the ones that do not.")


def count_tokens(text: str) -> int:
    return math.ceil(len(_TOKEN.findall(text)) * 1.3)


def _render_one(skill: Skill) -> str:
    return (f"### {skill.name}\n"
            f"When to use: {skill.trigger}\n"
            f"Strategy:\n{skill.lesson}\n"
            f"Avoid: {skill.failure_mode}")


def render_harness(skills: list[Skill]) -> str:
    if not skills:
        return NEUTRAL_SYSTEM_PROMPT

    general = [s for s in skills if s.level == "general"]
    topic = [s for s in skills if s.level != "general"]

    parts = [_HEADER]
    if general:
        parts.append("## General strategies\n" + "\n\n".join(_render_one(s) for s in general))
    if topic:
        parts.append("## Topic-specific procedures\n" + "\n\n".join(_render_one(s) for s in topic))
    return "\n\n".join(parts)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_render.py -v`
Expected: PASS，5 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/render.py tests/harness/test_render.py
git commit -m "feat(harness): system-message rendering with structural parity for empty harness"
```

---

## Task 4: store.py 第一部分 —— 持久化与变更日志

**Files:**
- Create: `alphaapollo/core/harness/store.py`
- Test: `tests/harness/test_store_persistence.py`

**Interfaces:**
- Consumes: `Skill`, `CandidateMemory`, `SkillEdit`（Task 1）；`skill_to_markdown` / `skill_from_markdown`（Task 1）；`count_tokens`（Task 3）
- Produces:
  - `@dataclass Caps`：`general: int = 5, per_topic: int = 5`
  - `@dataclass Budget`：`b: int = 6, general_max: int = 3, topic_max: int = 4, tokens: int = 800`
  - `class SkillStore`：
    - `__init__(self, root: str | Path, caps: Caps = Caps(), budget: Budget = Budget())`
    - `.all() -> list[Skill]`
    - `.snapshot() -> list[Skill]`（深拷贝，供 batch 内冻结使用）
    - `.log_event(**fields) -> None`（append 一行到 `harness_log.jsonl`）
    - `.log_selection(**fields) -> None`（append 一行到 `selection_log.jsonl`）
    - `.reload() -> None`（从磁盘重建，用于持久化测试）
    - 内部：`_next_id() -> str`，`_write_skill(skill)`，`_delete_skill(skill_id)`

落盘布局：

```
<root>/
  skills/<name>.md          # 一条 skill 一个文件
  harness_log.jsonl         # 每次 add/merge/revise/delete/skip，含被拒记录
  selection_log.jsonl       # 每题一行：用了哪些 skill
```

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_store_persistence.py`:

```python
import json

from alphaapollo.core.harness.schema import Skill
from alphaapollo.core.harness.store import SkillStore


def skill(i: int, level: str = "topic", topic: str = "number_theory") -> Skill:
    return Skill(id=f"sk_{i:04d}", name=f"skill-{i}", level=level,
                 topic=None if level == "general" else topic,
                 trigger=f"Trigger {i}.", lesson=f"- Lesson {i}.",
                 failure_mode=f"Avoid {i}.", created_at=i)


def test_skills_survive_a_fresh_store_instance(tmp_path):
    store = SkillStore(tmp_path)
    store._write_skill(skill(1))
    store._write_skill(skill(2, level="general"))

    reopened = SkillStore(tmp_path)
    got = {s.id: s for s in reopened.all()}
    assert set(got) == {"sk_0001", "sk_0002"}
    assert got["sk_0002"].level == "general" and got["sk_0002"].topic is None
    assert got["sk_0001"].lesson == "- Lesson 1."


def test_counters_survive_reload(tmp_path):
    store = SkillStore(tmp_path)
    s = skill(1)
    s.n_selected, s.n_selected_success = 12, 7
    store._write_skill(s)

    restored = SkillStore(tmp_path).all()[0]
    assert (restored.n_selected, restored.n_selected_success) == (12, 7)


def test_log_event_appends_one_json_line_each_call(tmp_path):
    store = SkillStore(tmp_path)
    store.log_event(problem_idx=1, batch=0, op="ADD", skill_id="sk_0001",
                    actor="topic_curator", reason="new pattern", accepted=True,
                    reject_reason=None)
    store.log_event(problem_idx=2, batch=0, op="ADD", skill_id=None,
                    actor="topic_curator", reason="dup", accepted=False,
                    reject_reason="question_overlap")

    lines = (tmp_path / "harness_log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    second = json.loads(lines[1])
    assert second["accepted"] is False and second["reject_reason"] == "question_overlap"


def test_selection_log_is_a_separate_file(tmp_path):
    store = SkillStore(tmp_path)
    store.log_selection(problem_idx=41, batch=5, topic="number_theory",
                        selected=[{"skill_id": "sk_0007", "score": 0.72, "tokens": 94}],
                        total_inject_tokens=611, pass1_round0=0, pass_final=1)
    payload = json.loads((tmp_path / "selection_log.jsonl").read_text().strip())
    assert payload["selected"][0]["skill_id"] == "sk_0007"
    assert not (tmp_path / "harness_log.jsonl").exists()


def test_snapshot_is_isolated_from_later_mutations(tmp_path):
    store = SkillStore(tmp_path)
    store._write_skill(skill(1))
    frozen = store.snapshot()
    store._write_skill(skill(2))
    assert len(frozen) == 1 and len(store.all()) == 2


def test_next_id_is_monotonic_across_reloads(tmp_path):
    store = SkillStore(tmp_path)
    store._write_skill(skill(1))
    store._write_skill(skill(9))
    assert SkillStore(tmp_path)._next_id() == "sk_0010"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_store_persistence.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'alphaapollo.core.harness.store'`

- [ ] **Step 3: 实现**

`alphaapollo/core/harness/store.py`（本 task 只实现持久化与日志；选择与 apply 在 Task 5/6 追加）:

```python
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alphaapollo.core.harness.render import count_tokens
from alphaapollo.core.harness.schema import Skill, skill_from_markdown, skill_to_markdown


@dataclass
class Caps:
    general: int = 5
    per_topic: int = 5


@dataclass
class Budget:
    b: int = 6
    general_max: int = 3
    topic_max: int = 4
    tokens: int = 800


class SkillStore:
    def __init__(self, root: str | Path, caps: Caps | None = None, budget: Budget | None = None):
        self.root = Path(root)
        self.caps = caps or Caps()
        self.budget = budget or Budget()
        self.skills_dir = self.root / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.harness_log = self.root / "harness_log.jsonl"
        self.selection_log = self.root / "selection_log.jsonl"
        self._skills: dict[str, Skill] = {}
        self.reload()

    # ---- persistence -------------------------------------------------
    def reload(self) -> None:
        self._skills = {}
        for path in sorted(self.skills_dir.glob("*.md")):
            skill = skill_from_markdown(path.read_text(encoding="utf-8"))
            self._skills[skill.id] = skill

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def snapshot(self) -> list[Skill]:
        return copy.deepcopy(self.all())

    def _write_skill(self, skill: Skill) -> None:
        skill.n_tokens = count_tokens(f"{skill.trigger}\n{skill.lesson}\n{skill.failure_mode}")
        (self.skills_dir / f"{skill.name}.md").write_text(skill_to_markdown(skill), encoding="utf-8")
        self._skills[skill.id] = skill

    def _delete_skill(self, skill_id: str) -> None:
        skill = self._skills.pop(skill_id, None)
        if skill is not None:
            (self.skills_dir / f"{skill.name}.md").unlink(missing_ok=True)

    def _next_id(self) -> str:
        used = [int(sid.split("_")[1]) for sid in self._skills if sid.startswith("sk_")]
        return f"sk_{(max(used) + 1 if used else 1):04d}"

    # ---- logging -----------------------------------------------------
    @staticmethod
    def _append(path: Path, payload: dict) -> None:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_event(self, **fields) -> None:
        self._append(self.harness_log, fields)

    def log_selection(self, **fields) -> None:
        self._append(self.selection_log, fields)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_store_persistence.py -v`
Expected: PASS，6 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/store.py tests/harness/test_store_persistence.py
git commit -m "feat(harness): persistent skill store with harness and selection logs"
```

---

## Task 5: store.apply() —— 两阶段编辑与容量硬约束

**Files:**
- Modify: `alphaapollo/core/harness/store.py`（追加方法）
- Test: `tests/harness/test_store_apply.py`

**Interfaces:**
- Consumes: Task 4 的 `SkillStore`；`SkillEdit` / `CandidateMemory`（Task 1）；`validate_skill`（Task 2）
- Produces: `SkillStore.apply(self, edits: list[SkillEdit], *, problem_idx: int, batch: int, question_texts: list[str], ground_truths: list[str]) -> list[dict]` —— 返回每条编辑的结果记录，同时写 `harness_log.jsonl`

**两阶段是必须的**：先应用 `DELETE / MERGE / REVISE`，再用**更新后**的占用数校验 `ADD`。否则 GeneralCurator 输出的 `DELETE x + ADD y` 组合里 `ADD` 永远进不去，harness 一旦打满就无法换血。

容量满时 `ADD` 被过滤并记 `accepted: false, reject_reason: "capacity_full"` —— 即使 LLM 无视 prompt 里的槽位声明也接得住。

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_store_apply.py`:

```python
import json

from alphaapollo.core.harness.schema import CandidateMemory, Skill, SkillEdit
from alphaapollo.core.harness.store import Caps, SkillStore

QUESTIONS = ["Find the number of ordered pairs of positive integers."]
GTS = ["738"]


def cand(n: int, scope: str = "topic", topic: str = "number_theory") -> CandidateMemory:
    return CandidateMemory(trigger=f"Trigger {n}.", lesson=f"- Lesson {n}.",
                           failure_mode=f"Avoid {n}.", scope_hint=scope,
                           topic=None if scope == "general" else topic,
                           evidence=[f"p_{n:04d}:symbolic_slip"])


def add(n: int, scope: str = "topic") -> SkillEdit:
    return SkillEdit(op="ADD", actor="topic_curator", reason=f"new {n}", payload=cand(n, scope))


def apply(store, edits):
    return store.apply(edits, problem_idx=1, batch=0, question_texts=QUESTIONS, ground_truths=GTS)


def test_add_creates_a_skill_and_logs_acceptance(tmp_path):
    store = SkillStore(tmp_path)
    results = apply(store, [add(1)])
    assert len(store.all()) == 1 and results[0]["accepted"] is True
    logged = json.loads((tmp_path / "harness_log.jsonl").read_text().strip())
    assert logged["op"] == "ADD" and logged["accepted"] is True


def test_add_beyond_topic_capacity_is_filtered_and_logged(tmp_path):
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=2))
    apply(store, [add(1), add(2)])
    results = apply(store, [add(3)])

    assert len(store.all()) == 2, "capacity must be a hard bound, not a prompt suggestion"
    assert results[0]["accepted"] is False
    assert results[0]["reject_reason"] == "capacity_full"


def test_general_and_topic_capacities_are_independent(tmp_path):
    store = SkillStore(tmp_path, caps=Caps(general=1, per_topic=1))
    apply(store, [add(1, "topic"), add(2, "general")])
    assert len(store.all()) == 2


def test_delete_then_add_within_one_batch_frees_a_slot(tmp_path):
    """两阶段 apply 的核心：同 batch 内 DELETE 腾出的槽位必须能被 ADD 占用。"""
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=1))
    apply(store, [add(1)])
    victim = store.all()[0].id

    apply(store, [add(2), SkillEdit(op="DELETE", actor="general_curator",
                                    reason="stale", skill_id=victim)])

    remaining = store.all()
    assert len(remaining) == 1
    assert remaining[0].id != victim, "the ADD should have taken the freed slot"


def test_revise_updates_content_and_records_the_problem_index(tmp_path):
    store = SkillStore(tmp_path)
    apply(store, [add(1)])
    sid = store.all()[0].id

    store.apply([SkillEdit(op="REVISE", actor="topic_curator", reason="sharpen",
                           skill_id=sid, payload=cand(99))],
                problem_idx=41, batch=5, question_texts=QUESTIONS, ground_truths=GTS)

    revised = store.all()[0]
    assert revised.id == sid and revised.lesson == "- Lesson 99." and 41 in revised.revised_at


def test_guard_rejection_does_not_mutate_the_store_but_is_logged(tmp_path):
    store = SkillStore(tmp_path)
    leaky = SkillEdit(op="ADD", actor="topic_curator", reason="leak",
                      payload=cand(1).__class__(trigger="T.", lesson="- The answer is 738.",
                                                failure_mode="A.", scope_hint="topic",
                                                topic="number_theory", evidence=[]))
    results = apply(store, [leaky])
    assert store.all() == [] and results[0]["reject_reason"] == "answer_leak"
    assert json.loads((tmp_path / "harness_log.jsonl").read_text().strip())["accepted"] is False


def test_skip_is_a_noop_but_still_logged(tmp_path):
    store = SkillStore(tmp_path)
    apply(store, [SkillEdit(op="SKIP", actor="topic_curator", reason="low confidence")])
    assert store.all() == []
    assert len((tmp_path / "harness_log.jsonl").read_text().strip().splitlines()) == 1


def test_every_edit_produces_exactly_one_log_line(tmp_path):
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=1))
    apply(store, [add(1), add(2), SkillEdit(op="SKIP", actor="x", reason="y")])
    assert len((tmp_path / "harness_log.jsonl").read_text().strip().splitlines()) == 3
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_store_apply.py -v`
Expected: FAIL —— `AttributeError: 'SkillStore' object has no attribute 'apply'`

- [ ] **Step 3: 实现**

追加到 `alphaapollo/core/harness/store.py`（并在文件顶部补 `from alphaapollo.core.harness.guard import validate_skill` 与 `from alphaapollo.core.harness.schema import CandidateMemory, SkillEdit`）:

```python
    # ---- editing -----------------------------------------------------
    def _occupancy(self, level: str, topic: str | None) -> tuple[int, int]:
        if level == "general":
            used = sum(1 for s in self._skills.values() if s.level == "general")
            return used, self.caps.general
        used = sum(1 for s in self._skills.values() if s.level != "general" and s.topic == topic)
        return used, self.caps.per_topic

    def _materialize(self, payload: CandidateMemory, problem_idx: int) -> Skill:
        level = payload.scope_hint
        sid = self._next_id()
        return Skill(
            id=sid, name=f"{level}-{sid}", level=level,
            topic=None if level == "general" else payload.topic,
            trigger=payload.trigger, lesson=payload.lesson,
            failure_mode=payload.failure_mode, evidence=list(payload.evidence),
            created_at=problem_idx,
        )

    def apply(
        self,
        edits: list[SkillEdit],
        *,
        problem_idx: int,
        batch: int,
        question_texts: list[str],
        ground_truths: list[str],
    ) -> list[dict]:
        # Phase 1: everything that frees or rewrites an existing slot.
        # Phase 2: additions, checked against post-phase-1 occupancy.
        phase1 = [e for e in edits if e.op in ("DELETE", "MERGE", "REVISE", "SKIP")]
        phase2 = [e for e in edits if e.op == "ADD"]

        results: list[dict] = []
        for edit in phase1 + phase2:
            record = self._apply_one(edit, problem_idx, question_texts, ground_truths)
            record.update(problem_idx=problem_idx, batch=batch, op=edit.op,
                          actor=edit.actor, reason=edit.reason)
            self.log_event(**record)
            results.append(record)
        return results

    def _apply_one(self, edit, problem_idx, question_texts, ground_truths) -> dict:
        if edit.op == "SKIP":
            return {"skill_id": edit.skill_id, "accepted": False, "reject_reason": "skipped"}

        if edit.op == "DELETE":
            existed = edit.skill_id in self._skills
            self._delete_skill(edit.skill_id)
            return {"skill_id": edit.skill_id, "accepted": existed,
                    "reject_reason": None if existed else "unknown_skill_id"}

        if edit.payload is None:
            return {"skill_id": edit.skill_id, "accepted": False, "reject_reason": "missing_payload"}

        ok, reason = validate_skill(edit.payload, question_texts=question_texts,
                                    ground_truths=ground_truths)
        if not ok:
            return {"skill_id": edit.skill_id, "accepted": False, "reject_reason": reason}

        if edit.op in ("REVISE", "MERGE"):
            target = self._skills.get(edit.skill_id)
            if target is None:
                return {"skill_id": edit.skill_id, "accepted": False,
                        "reject_reason": "unknown_skill_id"}
            target.trigger = edit.payload.trigger
            target.lesson = edit.payload.lesson
            target.failure_mode = edit.payload.failure_mode
            target.evidence = sorted(set(target.evidence) | set(edit.payload.evidence))
            target.revised_at = sorted(set(target.revised_at) | {problem_idx})
            self._write_skill(target)
            return {"skill_id": target.id, "accepted": True, "reject_reason": None}

        # ADD
        level = edit.payload.scope_hint
        used, cap = self._occupancy(level, edit.payload.topic)
        if used >= cap:
            return {"skill_id": None, "accepted": False, "reject_reason": "capacity_full"}
        skill = self._materialize(edit.payload, problem_idx)
        self._write_skill(skill)
        return {"skill_id": skill.id, "accepted": True, "reject_reason": None}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_store_apply.py -v`
Expected: PASS，8 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/store.py tests/harness/test_store_apply.py
git commit -m "feat(harness): two-phase edit application with hard capacity bounds"
```

---

## Task 6: store.select() —— 确定性分层检索与预算

**Files:**
- Modify: `alphaapollo/core/harness/store.py`（追加方法）
- Test: `tests/harness/test_store_select.py`

**Interfaces:**
- Consumes: Task 4/5 的 `SkillStore`；`Budget`；`count_tokens`（Task 3）
- Produces:
  - `SkillStore.select(self, question: str, topic: str | None) -> list[Skill]`
  - `SkillStore.record_usage(self, selected: list[Skill], success: bool) -> None`（更新 `n_selected` / `n_selected_success` 并落盘）

评分：`score = 0.3 * utility + 0.7 * lexical_overlap(trigger, question)`。`lexical_overlap` 是归一到 `[0,1]` 的词面重合（IDF 加权的 BM25-lite 简化版：`|trigger_words ∩ question_words| / |trigger_words|`，停用词剔除）。

配额写死：`general ≤ budget.general_max`、`topic ≤ budget.topic_max`、合计 `≤ budget.b`、累计 token `≤ budget.tokens`。不用"general 保底 N + 全局竞争"——后者在 b=6 时可能退化成 general 占满。

**这是相对论文的有意偏离**：论文附录 F 用 Claude Sonnet 4.5 做 selection。确定性检索让在线管理调用恒为 0，使"solver vs 管理调用分开报告"成为精确数字。README 必须写明。

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_store_select.py`:

```python
from alphaapollo.core.harness.schema import Skill
from alphaapollo.core.harness.store import Budget, Caps, SkillStore

QUESTION = "Count the ordered pairs of integers whose sum is divisible by seven."


def make(store, n, level, topic, trigger, n_sel=0, n_ok=0):
    s = Skill(id=f"sk_{n:04d}", name=f"{level}-{n}", level=level,
              topic=None if level == "general" else topic,
              trigger=trigger, lesson=f"- Lesson {n}.", failure_mode=f"Avoid {n}.",
              n_selected=n_sel, n_selected_success=n_ok)
    store._write_skill(s)
    return s


def populate(tmp_path, caps=None, budget=None):
    store = SkillStore(tmp_path, caps=caps or Caps(general=50, per_topic=50), budget=budget)
    for i in range(10):
        make(store, i, "general", None, "Verify intermediate results numerically.")
    for i in range(10, 20):
        make(store, i, "topic", "number_theory", "Counting integers divisible by a modulus.")
    for i in range(20, 30):
        make(store, i, "topic", "geometry", "Inscribed circles and tangent lengths.")
    return store


def test_selection_respects_count_and_quota(tmp_path):
    store = populate(tmp_path)
    picked = store.select(QUESTION, topic="number_theory")
    assert len(picked) <= 6
    assert sum(1 for s in picked if s.level == "general") <= 3
    assert sum(1 for s in picked if s.level != "general") <= 4


def test_selection_respects_the_token_budget(tmp_path):
    store = populate(tmp_path, budget=Budget(b=6, general_max=3, topic_max=4, tokens=40))
    picked = store.select(QUESTION, topic="number_theory")
    assert sum(s.n_tokens for s in picked) <= 40


def test_selection_never_returns_a_foreign_topic(tmp_path):
    store = populate(tmp_path)
    picked = store.select(QUESTION, topic="number_theory")
    assert all(s.topic in (None, "number_theory") for s in picked)


def test_utility_breaks_ties_among_equal_triggers(tmp_path):
    store = SkillStore(tmp_path)
    make(store, 1, "general", None, "Verify numerically.", n_sel=20, n_ok=1)
    make(store, 2, "general", None, "Verify numerically.", n_sel=20, n_ok=19)
    picked = store.select(QUESTION, topic="number_theory")
    assert picked[0].id == "sk_0002"


def test_lexical_overlap_outranks_utility(tmp_path):
    store = SkillStore(tmp_path)
    make(store, 1, "general", None, "Unrelated advice about calendars.", n_sel=20, n_ok=20)
    make(store, 2, "general", None, "Count ordered pairs divisible by a modulus.", n_sel=0, n_ok=0)
    assert store.select(QUESTION, topic="number_theory")[0].id == "sk_0002"


def test_selection_is_deterministic(tmp_path):
    store = populate(tmp_path)
    a = [s.id for s in store.select(QUESTION, topic="number_theory")]
    b = [s.id for s in store.select(QUESTION, topic="number_theory")]
    assert a == b


def test_empty_store_selects_nothing(tmp_path):
    assert SkillStore(tmp_path).select(QUESTION, topic="number_theory") == []


def test_record_usage_updates_counters_and_persists(tmp_path):
    store = populate(tmp_path)
    picked = store.select(QUESTION, topic="number_theory")
    store.record_usage(picked, success=True)

    reloaded = {s.id: s for s in SkillStore(tmp_path).all()}
    for s in picked:
        assert reloaded[s.id].n_selected == 1 and reloaded[s.id].n_selected_success == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_store_select.py -v`
Expected: FAIL —— `AttributeError: 'SkillStore' object has no attribute 'select'`

- [ ] **Step 3: 实现**

追加到 `alphaapollo/core/harness/store.py`（顶部补 `import re`）:

```python
_SELECT_WORD = re.compile(r"[A-Za-z']+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that the "
    "this to was were when where which with you your".split()
)


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _SELECT_WORD.findall(text) if w.lower() not in _STOPWORDS}
```

以及 `SkillStore` 的方法：

```python
    # ---- selection ---------------------------------------------------
    @staticmethod
    def _lexical_overlap(trigger: str, question: str) -> float:
        t = _content_words(trigger)
        if not t:
            return 0.0
        return len(t & _content_words(question)) / len(t)

    def _score(self, skill: Skill, question: str) -> float:
        return 0.3 * skill.utility() + 0.7 * self._lexical_overlap(skill.trigger, question)

    def select(self, question: str, topic: str | None) -> list[Skill]:
        def ranked(pool: list[Skill]) -> list[Skill]:
            # id as the final tiebreaker keeps selection fully deterministic
            return sorted(pool, key=lambda s: (-self._score(s, question), s.id))

        general = ranked([s for s in self._skills.values() if s.level == "general"])
        topical = ranked([s for s in self._skills.values()
                          if s.level != "general" and s.topic == topic])

        quota = [(general[:self.budget.general_max], "general"),
                 (topical[:self.budget.topic_max], "topic")]
        pool = ranked([s for group, _ in quota for s in group])

        picked: list[Skill] = []
        used_tokens = 0
        n_general = n_topic = 0
        for skill in pool:
            if len(picked) >= self.budget.b:
                break
            if skill.level == "general" and n_general >= self.budget.general_max:
                continue
            if skill.level != "general" and n_topic >= self.budget.topic_max:
                continue
            if used_tokens + skill.n_tokens > self.budget.tokens:
                continue
            picked.append(skill)
            used_tokens += skill.n_tokens
            n_general += skill.level == "general"
            n_topic += skill.level != "general"
        return picked

    def record_usage(self, selected: list[Skill], success: bool) -> None:
        for picked in selected:
            skill = self._skills.get(picked.id)
            if skill is None:
                continue
            skill.n_selected += 1
            skill.n_selected_success += int(success)
            self._write_skill(skill)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_store_select.py -v`
Expected: PASS，8 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/store.py tests/harness/test_store_select.py
git commit -m "feat(harness): deterministic layered selection under count and token budgets"
```

---

## Task 7: accounting.py —— 调用记账与 seed 注入

**Files:**
- Create: `alphaapollo/core/harness/accounting.py`
- Test: `tests/harness/test_accounting.py`

**Interfaces:**
- Consumes: `alphaapollo.core.generation.evolving.utils.agent.Agent`
- Produces:
  - `class CallAccountant`：`.calls: dict[str, int]`、`.tokens_in: dict[str, int]`、`.tokens_out: dict[str, int]`、`.snapshot() -> dict`
  - `role_scope(role: str)` —— contextmanager，设置当前 role
  - `install_accounting(accountant: CallAccountant, *, seed: int | None = None) -> Callable[[], None]` —— monkey-patch `Agent.get_action_from_gpt`，返回卸载函数
  - `SOLVER_ROLES = ("solver", "summarizer", "aggregator")`
  - `MGMT_ROLES = ("reflect", "topic_curator", "general_curator", "offline_labeling")`

**为什么必须 monkey-patch 而不是包装对象**：`utils/agent.py:68` 的 `get_action_from_gpt` 只 `return reasoning_text + action`，`response.usage` 被丢弃；而 `evolving_main.py:607` 的 summarizer 和 `:180` 的 aggregator 都是在函数内部**现场 new** 的 `Agent`，外层包装抓不到它们。patch 类方法是唯一能覆盖全部调用点、又不改上游文件的办法。

`seed` 同理注入 —— `utils/agent.py:47-54` 的 `chat.completions.create` 没有传 `seed`，不注入就没有可复现性。

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_accounting.py`:

```python
import pytest

from alphaapollo.core.generation.evolving.utils.agent import Agent
from alphaapollo.core.harness.accounting import (CallAccountant, install_accounting,
                                                 role_scope)


class FakeUsage:
    prompt_tokens, completion_tokens = 11, 5


class FakeMessage:
    content = "  an answer  "


class FakeResponse:
    usage = FakeUsage()
    choices = [type("C", (), {"message": FakeMessage()})()]


class FakeCompletions:
    def __init__(self):
        self.kwargs_seen = []

    def create(self, **kwargs):
        self.kwargs_seen.append(kwargs)
        return FakeResponse()


@pytest.fixture
def agent(monkeypatch):
    a = Agent({"model_name": "m", "base_url": "http://x/v1", "api_key": "EMPTY"})
    completions = FakeCompletions()
    a.client = type("C", (), {"chat": type("Ch", (), {"completions": completions})()})()
    a._completions = completions
    return a


def test_calls_are_attributed_to_the_active_role(agent):
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        with role_scope("solver"):
            agent.get_action_from_gpt("q")
        with role_scope("reflect"):
            agent.get_action_from_gpt("q")
            agent.get_action_from_gpt("q")
    finally:
        uninstall()

    assert acc.calls == {"solver": 1, "reflect": 2}
    assert acc.tokens_in["reflect"] == 22 and acc.tokens_out["solver"] == 5


def test_seed_is_injected_into_every_request(agent):
    acc = CallAccountant()
    uninstall = install_accounting(acc, seed=1234)
    try:
        with role_scope("solver"):
            agent.get_action_from_gpt("q")
    finally:
        uninstall()
    assert agent._completions.kwargs_seen[0]["seed"] == 1234


def test_a_provider_that_rejects_seed_falls_back_instead_of_failing(agent):
    """Not every OpenAI-compatible provider accepts `seed`; a 400 on an unknown
    parameter must not take down the whole run."""
    calls = {"n": 0}
    original_create = agent._completions.create

    def picky_create(**kwargs):
        calls["n"] += 1
        if "seed" in kwargs:
            raise TypeError("Unrecognized request argument supplied: seed")
        return original_create(**kwargs)

    agent._completions.create = picky_create

    acc = CallAccountant()
    uninstall = install_accounting(acc, seed=1234)
    try:
        with role_scope("solver"):
            assert agent.get_action_from_gpt("q") == "an answer"
        with role_scope("solver"):
            agent.get_action_from_gpt("q")
    finally:
        uninstall()

    # first call retries without seed; the second must not retry again
    assert calls["n"] == 3
    assert acc.calls["solver"] == 2


def test_patch_preserves_the_original_return_value(agent):
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        with role_scope("solver"):
            assert agent.get_action_from_gpt("q") == "an answer"
    finally:
        uninstall()


def test_uninstall_restores_the_original_method(agent):
    original = Agent.get_action_from_gpt
    install_accounting(CallAccountant())()
    assert Agent.get_action_from_gpt is original


def test_snapshot_separates_solver_side_from_management_side(agent):
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        for role in ("solver", "summarizer", "aggregator", "reflect", "topic_curator"):
            with role_scope(role):
                agent.get_action_from_gpt("q")
    finally:
        uninstall()

    snap = acc.snapshot()
    assert snap["calls/solver_side_total"] == 3
    assert snap["calls/mgmt_side_total"] == 2
    assert snap["calls/aggregator"] == 1


def test_calls_outside_any_role_scope_are_attributed_to_unscoped(agent):
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        agent.get_action_from_gpt("q")
    finally:
        uninstall()
    assert acc.calls["unscoped"] == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_accounting.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'alphaapollo.core.harness.accounting'`

- [ ] **Step 3: 实现**

`alphaapollo/core/harness/accounting.py`:

```python
from __future__ import annotations

import contextlib
import contextvars
import functools
import logging
from collections import defaultdict
from typing import Callable

from alphaapollo.core.generation.evolving.utils.agent import Agent

log = logging.getLogger(__name__)

SOLVER_ROLES = ("solver", "summarizer", "aggregator")
MGMT_ROLES = ("reflect", "topic_curator", "general_curator", "offline_labeling")

_current_role: contextvars.ContextVar[str] = contextvars.ContextVar("harness_role", default="unscoped")


@contextlib.contextmanager
def role_scope(role: str):
    token = _current_role.set(role)
    try:
        yield
    finally:
        _current_role.reset(token)


class CallAccountant:
    def __init__(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.tokens_in: dict[str, int] = defaultdict(int)
        self.tokens_out: dict[str, int] = defaultdict(int)

    def record(self, role: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.calls[role] += 1
        self.tokens_in[role] += prompt_tokens
        self.tokens_out[role] += completion_tokens

    def snapshot(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for role, n in self.calls.items():
            out[f"calls/{role}"] = n
            out[f"tokens/{role}_in"] = self.tokens_in[role]
            out[f"tokens/{role}_out"] = self.tokens_out[role]
        out["calls/solver_side_total"] = sum(self.calls[r] for r in SOLVER_ROLES)
        out["calls/mgmt_side_total"] = sum(self.calls[r] for r in MGMT_ROLES)
        out["tokens/solver_side_in"] = sum(self.tokens_in[r] for r in SOLVER_ROLES)
        out["tokens/mgmt_side_in"] = sum(self.tokens_in[r] for r in MGMT_ROLES)
        return out


def install_accounting(accountant: CallAccountant, *, seed: int | None = None) -> Callable[[], None]:
    """Patch Agent.get_action_from_gpt to record usage and inject a seed.

    Wrapping Agent *instances* is not enough: evolving_main.py:607 and :180
    construct their own Agent objects inside functions, so only a class-level
    patch covers every call site without editing upstream files.
    """
    original = Agent.get_action_from_gpt
    seed_supported = {"value": seed is not None}

    @functools.wraps(original)
    def patched(self, obs):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": obs})

        kwargs = dict(model=self.model_name, messages=messages,
                      temperature=self.temperature, max_tokens=self.max_tokens,
                      n=1, stop=None)
        if seed_supported["value"]:
            kwargs["seed"] = seed

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Not every OpenAI-compatible provider accepts `seed`. Degrade once,
            # loudly, rather than failing every call for the rest of the run.
            if not (seed_supported["value"] and "seed" in str(exc).lower()):
                raise
            log.warning("provider rejected `seed`; continuing without it. "
                        "Reproducibility is reduced — record this in the README.")
            seed_supported["value"] = False
            kwargs.pop("seed", None)
            response = self.client.chat.completions.create(**kwargs)

        usage = getattr(response, "usage", None)
        accountant.record(_current_role.get(),
                          getattr(usage, "prompt_tokens", 0) or 0,
                          getattr(usage, "completion_tokens", 0) or 0)

        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        prefix = f"<think>\n{reasoning.strip()}\n</think>\n" if isinstance(reasoning, str) and reasoning.strip() else ""
        return prefix + message.content.strip()

    Agent.get_action_from_gpt = patched

    def uninstall() -> None:
        Agent.get_action_from_gpt = original

    return uninstall
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_accounting.py -v`
Expected: PASS，7 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/accounting.py tests/harness/test_accounting.py
git commit -m "feat(harness): per-role call accounting and seed injection via class patch"
```

---

## Task 8: reflect.py —— 白名单上下文与候选提炼

**Files:**
- Create: `alphaapollo/core/harness/reflect.py`
- Test: `tests/harness/test_reflect.py`

**Interfaces:**
- Consumes: `CandidateMemory`（Task 1）；`role_scope`（Task 7）
- Produces:
  - `sanitize_feedback(text: str) -> str` —— 剥离 GT 痕迹
  - `build_reflect_context(*, topic: str, problem_shape: str, final_answer_given: str, outcome: str, verifier_feedback: str, tool_errors: str, reasoning_excerpt: str, round_count: int, related_skills: list) -> dict` —— **纯白名单关键字参数**，题面与 GT 在签名层面不可达
    - `related_skills` 对齐论文附录 E.1 的 *"Inputs: ... and related existing skills"*。直接传本题 `store.select()` 已选中的那几条，零额外成本。传 skill 是安全的：每条都已过 `guard.validate_skill()`，不含题面与答案。不给这一项，solver 会反复提炼 harness 里已有的东西，浪费 curator 预算并抬高重复率。
  - `REFLECT_PROMPT: str`
  - `reflect(agent, context: dict, *, feedback_level: str = "standard") -> CandidateMemory | None`
  - `parse_reflection(text: str) -> CandidateMemory | None`

**已确认的泄漏通道**：`core/tools/informalmath_verify.py:107` 与 `:176` 会把 `Matches ground truth: {bool}` / `Matches GT: {bool}` 写进返回文本，经 `env.py:106` 包成 `<tool_response>` 回传。`sanitize_feedback` 负责剥掉。

`feedback_level` 支持设计文档 §11 的必做对照：`"minimal"` 只给 0/1 标签 + tool_errors；`"standard"` 给完整 verifier report。

**Prompt 必须逐条落实论文附录 E.1 的 "Filter aggressively"**：跳过 generic advice、basic tool usage、**exact task replay**、以及 would-not-help-unseen-tasks 的候选。其中 "exact task replay" 同时是防泄漏的软闸 —— 它在 prompt 层压制"把本题解法抄进 skill"的倾向。

**本模块不得出现标识符 `ground_truth`。**

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_reflect.py`:

```python
import inspect

from alphaapollo.core.harness import reflect as reflect_mod
from alphaapollo.core.harness.reflect import (build_reflect_context, parse_reflection,
                                              reflect, sanitize_feedback)
from alphaapollo.core.harness.schema import Skill


class StubAgent:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def get_action_from_gpt(self, obs):
        self.prompts.append(obs)
        return self.reply


GOOD = """ACTION: NEW
SCOPE: topic
TRIGGER: Counting integers under congruence constraints.
LESSON:
- Enumerate a small range in python before generalizing.
AVOID: Extrapolating without numeric verification."""

ENHANCE = """ACTION: ENHANCE
TARGET: sk_0003
SCOPE: topic
TRIGGER: Counting integers under congruence constraints.
LESSON:
- Also check the modulus boundary case.
AVOID: Assuming residues are uniform."""


def a_skill(sid="sk_0003"):
    return Skill(id=sid, name=f"topic-{sid}", level="topic", topic="number_theory",
                 trigger="Counting integers divisible by a modulus.",
                 lesson="- Enumerate first.", failure_mode="Skipping verification.")


def context(**kw):
    base = dict(topic="number_theory", problem_shape="counting-with-constraints",
                final_answer_given="412", outcome="failed",
                verifier_feedback="The modular step is wrong.", tool_errors="",
                reasoning_excerpt="I assumed the residues were uniform.", round_count=3,
                related_skills=[a_skill()])
    base.update(kw)
    return build_reflect_context(**base)


def test_module_never_references_ground_truth():
    src = inspect.getsource(reflect_mod)
    assert "ground_truth" not in src, "reflect.py must not be able to touch the ground truth"


def test_context_builder_takes_no_question_or_answer_key():
    params = set(inspect.signature(build_reflect_context).parameters)
    assert "question" not in params and "ground_truth" not in params
    assert params == {"topic", "problem_shape", "final_answer_given", "outcome",
                      "verifier_feedback", "tool_errors", "reasoning_excerpt",
                      "round_count", "related_skills"}


def test_prompt_shows_the_related_existing_skills():
    """Paper Appendix E.1 lists "related existing skills" as a proposal input;
    without them the solver keeps re-proposing what the harness already has."""
    agent = StubAgent(GOOD)
    reflect(agent, context())
    assert "sk_0003" in agent.prompts[0]
    assert "Counting integers divisible by a modulus." in agent.prompts[0]


def test_prompt_handles_an_empty_related_skill_list():
    agent = StubAgent(GOOD)
    reflect(agent, context(related_skills=[]))
    assert agent.prompts[0]


def test_prompt_carries_the_aggressive_filter_rules():
    agent = StubAgent(GOOD)
    reflect(agent, context())
    prompt = agent.prompts[0]
    for rule in ("generic advice", "task replay", "unseen"):
        assert rule in prompt.lower()


def test_parse_reflection_reads_the_action_hint():
    assert parse_reflection(GOOD).action_hint == "NEW"


def test_parse_reflection_reads_enhance_with_its_target():
    cand = parse_reflection(ENHANCE)
    assert cand.action_hint == "ENHANCE" and cand.target_id == "sk_0003"


def test_parse_reflection_returns_none_on_action_none():
    assert parse_reflection("ACTION: NONE\nnothing reusable here") is None


def test_missing_action_line_defaults_to_new():
    legacy = GOOD.split("\n", 1)[1]
    assert parse_reflection(legacy).action_hint == "NEW"


def test_sanitize_strips_the_matches_gt_channel():
    raw = "Result: 7\nMatches ground truth: True\nMatches GT: False\nLooks fine."
    cleaned = sanitize_feedback(raw)
    assert "Matches ground truth" not in cleaned and "Matches GT" not in cleaned
    assert "Looks fine." in cleaned


def test_minimal_feedback_level_drops_the_verifier_report():
    agent = StubAgent(GOOD)
    reflect(agent, context(), feedback_level="minimal")
    assert "The modular step is wrong." not in agent.prompts[0]


def test_standard_feedback_level_includes_the_verifier_report():
    agent = StubAgent(GOOD)
    reflect(agent, context(), feedback_level="standard")
    assert "The modular step is wrong." in agent.prompts[0]


def test_parse_reflection_extracts_all_four_fields():
    cand = parse_reflection(GOOD)
    assert cand.scope_hint == "topic"
    assert cand.trigger.startswith("Counting integers")
    assert "Enumerate a small range" in cand.lesson
    assert cand.failure_mode.startswith("Extrapolating")


def test_parse_reflection_returns_none_on_explicit_none():
    assert parse_reflection("SCOPE: none\nnothing useful here") is None


def test_parse_reflection_returns_none_on_malformed_output():
    assert parse_reflection("I am not going to follow the format.") is None


def test_reflect_returns_none_when_the_model_output_is_unparseable():
    assert reflect(StubAgent("garbage"), context()) is None


def test_reflect_prompt_demands_english():
    agent = StubAgent(GOOD)
    reflect(agent, context())
    assert "English" in agent.prompts[0]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_reflect.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'alphaapollo.core.harness.reflect'`

- [ ] **Step 3: 实现**

`alphaapollo/core/harness/reflect.py`:

```python
from __future__ import annotations

import re

from alphaapollo.core.harness.accounting import role_scope
from alphaapollo.core.harness.schema import CandidateMemory

# core/tools/informalmath_verify.py:107 and :176 emit these lines into the
# tool response; they must never reach the compilation stage.
_GT_CHANNEL = re.compile(r"^.*(matches ground truth|matches gt)\s*:.*$",
                         re.IGNORECASE | re.MULTILINE)

REFLECT_PROMPT = """You just failed a competition mathematics problem. Distill ONE reusable \
lesson that would help you on FUTURE, DIFFERENT problems.

Topic: {topic}
Problem shape: {problem_shape}
Answer you produced: {final_answer_given}
Outcome: {outcome}
Rounds used: {round_count}
{feedback_block}
Your reasoning (excerpt):
{reasoning_excerpt}

## Related skills already in your harness
{related_skills}

Write in English. Output EXACTLY this format and nothing else:

ACTION: NEW | ENHANCE | NONE
TARGET: <existing skill id>          (only when ACTION is ENHANCE)
SCOPE: general | topic
TRIGGER: <one sentence naming the situation where this applies>
LESSON:
- <bullet, imperative, at most 3 bullets, 60 words total>
AVOID: <one sentence naming the failure mode>

Filter aggressively. Output ACTION: NONE rather than proposing:
- generic advice such as "read carefully" or "double-check the work";
- basic tool usage that any solver already knows;
- an exact replay of this task, its statement, or its numeric answer;
- anything that would not help on a problem you have never seen.

Other rules:
- If a listed existing skill already covers this lesson, use ACTION: ENHANCE with its id.
- Describe the METHOD, never the problem statement or its numeric answer.
- SCOPE: general only if it would help on a DIFFERENT mathematical topic."""


def sanitize_feedback(text: str) -> str:
    return _GT_CHANNEL.sub("", text or "").strip()


def build_reflect_context(
    *,
    topic: str,
    problem_shape: str,
    final_answer_given: str,
    outcome: str,
    verifier_feedback: str,
    tool_errors: str,
    reasoning_excerpt: str,
    round_count: int,
    related_skills: list,
) -> dict:
    """Whitelist builder. The problem statement and the reference answer are
    not parameters of this function, so they cannot reach compilation.

    related_skills mirrors Appendix E.1's "related existing skills" input. The
    skills are safe to pass on: each one already passed guard.validate_skill(),
    so none of them carries a problem statement or an answer.
    """
    return {
        "topic": topic,
        "problem_shape": problem_shape,
        "final_answer_given": final_answer_given,
        "outcome": outcome,
        "verifier_feedback": sanitize_feedback(verifier_feedback),
        "tool_errors": sanitize_feedback(tool_errors),
        "reasoning_excerpt": reasoning_excerpt,
        "round_count": round_count,
        "related_skills": list(related_skills),
    }


def _render_related(skills: list) -> str:
    if not skills:
        return "(none yet)"
    return "\n".join(f"- {s.id}: {s.trigger} | {s.lesson}" for s in skills)


def _feedback_block(context: dict, feedback_level: str) -> str:
    errors = context["tool_errors"]
    if feedback_level == "minimal":
        return f"Tool errors: {errors}\n" if errors else ""
    parts = [f"Verifier feedback: {context['verifier_feedback']}"]
    if errors:
        parts.append(f"Tool errors: {errors}")
    return "\n".join(parts) + "\n"


def parse_reflection(text: str) -> CandidateMemory | None:
    action = re.search(r"^ACTION:\s*(NEW|ENHANCE|NONE)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if action is not None and action.group(1).upper() == "NONE":
        return None

    scope = re.search(r"^SCOPE:\s*(general|topic|none)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if scope is None or scope.group(1).lower() == "none":
        return None

    trigger = re.search(r"^TRIGGER:\s*(.+)$", text, re.MULTILINE)
    lesson = re.search(r"^LESSON:\s*\n(.*?)(?=^AVOID:)", text, re.MULTILINE | re.DOTALL)
    avoid = re.search(r"^AVOID:\s*(.+)$", text, re.MULTILINE)
    if not (trigger and lesson and avoid):
        return None

    hint = action.group(1).upper() if action is not None else "NEW"
    target = re.search(r"^TARGET:\s*(\S+)\s*$", text, re.MULTILINE)
    return CandidateMemory(
        trigger=trigger.group(1).strip(),
        lesson=lesson.group(1).strip(),
        failure_mode=avoid.group(1).strip(),
        scope_hint=scope.group(1).lower(),
        topic=None,
        evidence=[],
        action_hint=hint,
        target_id=target.group(1) if (hint == "ENHANCE" and target) else None,
    )


def reflect(agent, context: dict, *, feedback_level: str = "standard") -> CandidateMemory | None:
    fields = {k: v for k, v in context.items() if k != "related_skills"}
    prompt = REFLECT_PROMPT.format(
        feedback_block=_feedback_block(context, feedback_level),
        related_skills=_render_related(context["related_skills"]),
        **fields,
    )
    with role_scope("reflect"):
        raw = agent.get_action_from_gpt(prompt)
    candidate = parse_reflection(raw)
    if candidate is not None and candidate.scope_hint == "topic":
        candidate.topic = context["topic"]
    return candidate
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_reflect.py -v`
Expected: PASS，18 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/reflect.py tests/harness/test_reflect.py
git commit -m "feat(harness): whitelist reflect context with GT-channel sanitization"
```

---

## Task 9: evolver.py —— 双 curator 与编辑解析

**Files:**
- Create: `alphaapollo/core/harness/evolver.py`
- Test: `tests/harness/test_evolver.py`

**Interfaces:**
- Consumes: `Skill` / `CandidateMemory` / `SkillEdit`（Task 1）；`Caps`（Task 4）；`role_scope`（Task 7）
- Produces:
  - `TOPIC_CURATOR_PROMPT: str`、`GENERAL_CURATOR_PROMPT: str`
  - `parse_curator_output(text: str, actor: str) -> list[SkillEdit]`
  - `class TopicCurator`：`.curate(agent, *, existing: list[Skill], candidates: list[CandidateMemory], topic: str, caps: Caps) -> list[SkillEdit]`
  - `class GeneralCurator`：`.curate(agent, *, existing: list[Skill], candidates: list[CandidateMemory], caps: Caps) -> list[SkillEdit]`

两个 curator 都必须在 LLM 调用异常或输出不可解析时**返回空列表而非抛异常** —— 设计文档 §5.5 要求 skill 更新失败降级为 no-op，绝不能破坏求解路径。

GeneralCurator 要求模式跨 ≥2 题出现（对齐参考实现 swe/tau/terminal/webarena 分支的 `2+`）。

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_evolver.py`:

```python
from alphaapollo.core.harness.evolver import (GeneralCurator, TopicCurator,
                                              parse_curator_output)
from alphaapollo.core.harness.schema import CandidateMemory, Skill
from alphaapollo.core.harness.store import Caps


class StubAgent:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def get_action_from_gpt(self, obs):
        self.prompts.append(obs)
        return self.reply


class ExplodingAgent:
    def get_action_from_gpt(self, obs):
        raise RuntimeError("API is down")


def cand(n: int, scope="topic") -> CandidateMemory:
    return CandidateMemory(trigger=f"Trigger {n}.", lesson=f"- Lesson {n}.",
                           failure_mode=f"Avoid {n}.", scope_hint=scope,
                           topic="number_theory" if scope == "topic" else None,
                           evidence=[f"p_{n:04d}:slip"])


def skill(n: int, level="topic") -> Skill:
    return Skill(id=f"sk_{n:04d}", name=f"{level}-{n}", level=level,
                 topic=None if level == "general" else "number_theory",
                 trigger=f"T{n}.", lesson=f"- L{n}.", failure_mode=f"A{n}.")


ACCEPT = """ADD: 1
REASON: distinct and actionable"""

MERGE = """MERGE: 1 INTO sk_0003
REASON: overlaps existing guidance
TRIGGER: Counting integers under congruence constraints.
LESSON:
- Enumerate before generalizing.
AVOID: Skipping numeric verification."""


def test_parse_add_maps_to_the_candidate_index():
    edits = parse_curator_output(ACCEPT, actor="topic_curator")
    assert len(edits) == 1 and edits[0].op == "ADD" and edits[0].reason


def test_parse_merge_carries_the_target_id_and_new_content():
    edits = parse_curator_output(MERGE, actor="topic_curator")
    assert edits[0].op == "MERGE" and edits[0].skill_id == "sk_0003"
    assert "Enumerate before generalizing." in edits[0].payload.lesson


def test_parse_delete_needs_no_payload():
    edits = parse_curator_output("DELETE: sk_0004\nREASON: stale", actor="general_curator")
    assert edits[0].op == "DELETE" and edits[0].skill_id == "sk_0004" and edits[0].payload is None


def test_parse_skip_is_recorded_not_dropped():
    edits = parse_curator_output("SKIP: 1\nREASON: too vague", actor="topic_curator")
    assert edits[0].op == "SKIP"


def test_parse_no_proposals_yields_no_edits():
    assert parse_curator_output("NO_PROPOSALS", actor="topic_curator") == []
    assert parse_curator_output("NO_PATTERNS", actor="general_curator") == []


def test_parse_ignores_unparseable_noise():
    assert parse_curator_output("I think maybe we should keep things as they are.",
                                actor="topic_curator") == []


def test_topic_curator_binds_add_payload_to_the_right_candidate():
    agent = StubAgent(ACCEPT)
    edits = TopicCurator().curate(agent, existing=[], candidates=[cand(1), cand(2)],
                                  topic="number_theory", caps=Caps())
    assert edits[0].payload.lesson == "- Lesson 1."
    assert edits[0].payload.topic == "number_theory"


def test_topic_curator_declares_the_remaining_slots_in_the_prompt():
    agent = StubAgent("NO_PROPOSALS")
    TopicCurator().curate(agent, existing=[skill(1), skill(2)], candidates=[cand(1)],
                          topic="number_theory", caps=Caps(general=5, per_topic=5))
    assert "2/5" in agent.prompts[0]


def test_topic_curator_with_no_candidates_makes_no_model_call():
    agent = StubAgent(ACCEPT)
    assert TopicCurator().curate(agent, existing=[], candidates=[],
                                 topic="number_theory", caps=Caps()) == []
    assert agent.prompts == []


def test_general_curator_forces_general_scope_on_its_payloads():
    agent = StubAgent(ACCEPT)
    edits = GeneralCurator().curate(agent, existing=[], candidates=[cand(1, "topic")],
                                    caps=Caps())
    assert edits[0].payload.scope_hint == "general" and edits[0].payload.topic is None


def test_general_curator_requires_a_pattern_across_at_least_two_problems():
    agent = StubAgent("NO_PATTERNS")
    GeneralCurator().curate(agent, existing=[], candidates=[cand(1), cand(2)], caps=Caps())
    assert "2+" in agent.prompts[0] or "at least 2" in agent.prompts[0]


def test_topic_curator_prompt_carries_the_generalizability_test():
    """Paper Appendix E.2: apply a generalizability test such as usefulness
    for multiple unseen tasks."""
    agent = StubAgent("NO_PROPOSALS")
    TopicCurator().curate(agent, existing=[], candidates=[cand(1)],
                          topic="number_theory", caps=Caps())
    assert "unseen" in agent.prompts[0].lower()


def test_general_curator_forbids_context_specific_references():
    """Paper Appendix E.3: general skills must avoid context-specific references."""
    agent = StubAgent("NO_PATTERNS")
    GeneralCurator().curate(agent, existing=[], candidates=[cand(1)], caps=Caps())
    assert "context-specific" in agent.prompts[0].lower()


def test_enhance_candidates_surface_their_target_in_the_prompt():
    agent = StubAgent("NO_PROPOSALS")
    enhancing = cand(1)
    enhancing.action_hint, enhancing.target_id = "ENHANCE", "sk_0042"
    TopicCurator().curate(agent, existing=[skill(42)], candidates=[enhancing],
                          topic="number_theory", caps=Caps())
    assert "ENHANCE" in agent.prompts[0] and "sk_0042" in agent.prompts[0]


def test_curator_degrades_to_noop_when_the_model_call_raises():
    """设计文档 §5.5：skill 更新失败必须降级为 no-op，不能破坏求解路径。"""
    assert TopicCurator().curate(ExplodingAgent(), existing=[], candidates=[cand(1)],
                                 topic="number_theory", caps=Caps()) == []
    assert GeneralCurator().curate(ExplodingAgent(), existing=[], candidates=[cand(1)],
                                   caps=Caps()) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_evolver.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'alphaapollo.core.harness.evolver'`

- [ ] **Step 3: 实现**

`alphaapollo/core/harness/evolver.py`:

```python
from __future__ import annotations

import logging
import re

from alphaapollo.core.harness.accounting import role_scope
from alphaapollo.core.harness.schema import CandidateMemory, Skill, SkillEdit
from alphaapollo.core.harness.store import Caps

log = logging.getLogger(__name__)

_COMMON_FORMAT = """For each decision output ONE block. Use EXACTLY these forms:

ADD: <candidate number>
REASON: <brief>

MERGE: <candidate number> INTO <existing skill id>
REASON: <brief>
TRIGGER: <one sentence>
LESSON:
- <bullets, 60 words total>
AVOID: <one sentence>

REVISE: <existing skill id>
REASON: <brief>
TRIGGER: <one sentence>
LESSON:
- <bullets, 60 words total>
AVOID: <one sentence>

DELETE: <existing skill id>
REASON: <brief>

SKIP: <candidate number>
REASON: <brief>

Write in English. Be SPECIFIC and ACTIONABLE, never generic advice like "read carefully"."""

TOPIC_CURATOR_PROMPT = """You curate topic-specific skills for a competition mathematics solver.

## Current skills for topic "{topic}" ({used}/{cap} slots used)
{existing}

## Candidate lessons from this batch
{candidates}

Decision criteria:
- GENERALIZABILITY TEST: keep a candidate only if it would be useful on MULTIPLE
  problems you have never seen. Judging that it accurately describes what just
  went wrong is NOT sufficient.
- Overlaps an existing skill -> MERGE (preferred over ADD).
- A candidate marked ENHANCE already names a target; prefer MERGE into that target.
- Budget full ({used}/{cap}) -> only MERGE, REVISE, DELETE or SKIP are allowed.
- One skill = one specific procedure. Few broad skills beat many narrow ones.
- Require a clear trigger description; keep content short and actionable.
- Low confidence -> SKIP.

{fmt}

If no candidate is worth keeping, output: NO_PROPOSALS"""

GENERAL_CURATOR_PROMPT = """You are a meta-learning curator. Analyse failure patterns ACROSS \
different topics to distil general skills that help on ANY mathematics problem.

## Current general skills ({used}/{cap} slots used)
{existing}

## Candidate lessons from this batch ({n} failed problems)
{candidates}

Your job:
- Find failure patterns that repeat across DIFFERENT problems and DIFFERENT topics.
- Each general skill must address a pattern seen in at least 2 (2+) different problems.
- GENERALIZABILITY TEST: a general skill must be useful on MULTIPLE unseen problems,
  and must encode procedures for planning, verification, recovery or tool use.
- General skills MUST avoid context-specific references. Reject any candidate that
  names a particular problem type, formula, or mathematical object.
- Do NOT create general skills for topic-specific procedures.
- Prefer REVISE over ADD when an existing general skill already covers the pattern.

{fmt}

If no cross-topic pattern is present, output: NO_PATTERNS"""


def _render_existing(skills: list[Skill]) -> str:
    if not skills:
        return "(none)"
    return "\n".join(f"- {s.id}: {s.trigger} | {s.lesson}" for s in skills)


def _render_candidates(candidates: list[CandidateMemory]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        hint = c.action_hint + (f" -> {c.target_id}" if c.target_id else "")
        lines.append(f"{i + 1}. [{hint}] TRIGGER: {c.trigger}\n"
                     f"   LESSON: {c.lesson}\n   AVOID: {c.failure_mode}")
    return "\n".join(lines)


def _payload(block: str) -> CandidateMemory | None:
    trigger = re.search(r"^TRIGGER:\s*(.+)$", block, re.MULTILINE)
    lesson = re.search(r"^LESSON:\s*\n(.*?)(?=^AVOID:)", block, re.MULTILINE | re.DOTALL)
    avoid = re.search(r"^AVOID:\s*(.+)$", block, re.MULTILINE)
    if not (trigger and lesson and avoid):
        return None
    return CandidateMemory(trigger=trigger.group(1).strip(), lesson=lesson.group(1).strip(),
                           failure_mode=avoid.group(1).strip(), scope_hint="topic",
                           topic=None, evidence=[])


_HEAD = re.compile(r"^(ADD|MERGE|REVISE|DELETE|SKIP):\s*(\S+)(?:\s+INTO\s+(\S+))?\s*$",
                   re.MULTILINE)


def parse_curator_output(text: str, actor: str) -> list[SkillEdit]:
    heads = list(_HEAD.finditer(text or ""))
    edits: list[SkillEdit] = []
    for i, head in enumerate(heads):
        op, first, target = head.group(1), head.group(2), head.group(3)
        block = text[head.end():heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        reason_match = re.search(r"^REASON:\s*(.+)$", block, re.MULTILINE)
        reason = reason_match.group(1).strip() if reason_match else ""

        edit = SkillEdit(op=op, actor=actor, reason=reason)
        if op in ("ADD", "SKIP"):
            edit.skill_id = None
            edit._candidate_index = int(first) - 1 if first.isdigit() else None  # type: ignore[attr-defined]
        elif op == "MERGE":
            edit.skill_id = target
            edit._candidate_index = int(first) - 1 if first.isdigit() else None  # type: ignore[attr-defined]
            edit.payload = _payload(block)
        elif op == "REVISE":
            edit.skill_id = first
            edit._candidate_index = None  # type: ignore[attr-defined]
            edit.payload = _payload(block)
        else:  # DELETE
            edit.skill_id = first
            edit._candidate_index = None  # type: ignore[attr-defined]
        edits.append(edit)
    return edits


def _bind_payloads(edits: list[SkillEdit], candidates: list[CandidateMemory],
                   *, force_general: bool, topic: str | None) -> list[SkillEdit]:
    bound: list[SkillEdit] = []
    for edit in edits:
        idx = getattr(edit, "_candidate_index", None)
        if edit.op == "ADD":
            if idx is None or not 0 <= idx < len(candidates):
                continue
            edit.payload = CandidateMemory(**vars(candidates[idx]))
        if edit.payload is not None:
            edit.payload.scope_hint = "general" if force_general else "topic"
            edit.payload.topic = None if force_general else topic
        bound.append(edit)
    return bound


class _BaseCurator:
    actor = "curator"

    def _call(self, agent, prompt: str) -> str:
        with role_scope(self.actor):
            return agent.get_action_from_gpt(prompt)


class TopicCurator(_BaseCurator):
    actor = "topic_curator"

    def curate(self, agent, *, existing: list[Skill], candidates: list[CandidateMemory],
               topic: str, caps: Caps) -> list[SkillEdit]:
        if not candidates:
            return []
        prompt = TOPIC_CURATOR_PROMPT.format(
            topic=topic, used=len(existing), cap=caps.per_topic,
            existing=_render_existing(existing), candidates=_render_candidates(candidates),
            fmt=_COMMON_FORMAT,
        )
        try:
            raw = self._call(agent, prompt)
        except Exception as exc:  # noqa: BLE001 - harness failures must never break solving
            log.warning("topic curator call failed, degrading to no-op: %s", exc)
            return []
        return _bind_payloads(parse_curator_output(raw, self.actor), candidates,
                              force_general=False, topic=topic)


class GeneralCurator(_BaseCurator):
    actor = "general_curator"

    def curate(self, agent, *, existing: list[Skill], candidates: list[CandidateMemory],
               caps: Caps) -> list[SkillEdit]:
        if not candidates:
            return []
        prompt = GENERAL_CURATOR_PROMPT.format(
            used=len(existing), cap=caps.general, n=len(candidates),
            existing=_render_existing(existing), candidates=_render_candidates(candidates),
            fmt=_COMMON_FORMAT,
        )
        try:
            raw = self._call(agent, prompt)
        except Exception as exc:  # noqa: BLE001
            log.warning("general curator call failed, degrading to no-op: %s", exc)
            return []
        return _bind_payloads(parse_curator_output(raw, self.actor), candidates,
                              force_general=True, topic=None)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_evolver.py -v`
Expected: PASS，15 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/evolver.py tests/harness/test_evolver.py
git commit -m "feat(harness): dual curator with no-op degradation on failure"
```

---

## Task 10: arms.py —— 三个对照 arm

**Files:**
- Create: `alphaapollo/core/harness/arms.py`
- Test: `tests/harness/test_arms.py`

**Interfaces:**
- Consumes: `SkillStore` / `Caps` / `Budget`（Task 4–6）；`render_harness` / `NEUTRAL_SYSTEM_PROMPT` / `count_tokens`（Task 3）；`reflect` / `build_reflect_context`（Task 8）；`TopicCurator` / `GeneralCurator`（Task 9）
- Produces:
  - `class CrossProblemArm`（协议基类）：
    - `.begin_batch(self, batch_idx: int) -> None`
    - `.system_prompt_for(self, problem: dict) -> str`
    - `.record_selection(self, problem: dict, result: dict) -> None`
    - `.observe(self, problem: dict, result: dict) -> None`
    - `.end_batch(self, batch_idx: int) -> list[dict]`
  - `class BaselineArm`、`class RawExperienceArm`、`class EvoHarnessArm`
  - `build_arm(name: str, **kwargs) -> CrossProblemArm`

**结构对齐是硬要求**：三个 arm 的 `system_prompt_for` 都必须返回非空字符串。`utils/agent.py:37` 是 `if self.system_prompt:` —— 返回空串会让整个 system message 消失，Baseline 与另两个 arm 之间就多出一个结构变量，而 Evo arm 在第一个 batch 结束时还会发生内部跳变。

`frozen=True` 时 `end_batch` 必须是 no-op —— held-out 评测用。

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_arms.py`:

```python
import pytest

from alphaapollo.core.harness.arms import (BaselineArm, EvoHarnessArm, RawExperienceArm,
                                           build_arm)
from alphaapollo.core.harness.render import NEUTRAL_SYSTEM_PROMPT

PROBLEM = {"problem_idx": 0, "question": "Count integers divisible by seven.",
           "topic": "number_theory", "problem_shape": "counting-with-constraints"}

FAILED = {"pass_final": 0, "pass1_round0": 0, "final_answer_given": "412",
          "verifier_feedback": "The modular step is wrong.\nMatches GT: False",
          "tool_errors": "", "reasoning_excerpt": "I assumed uniform residues.",
          "round_count": 3}

PASSED = {**FAILED, "pass_final": 1, "pass1_round0": 1}

REFLECTION = """SCOPE: topic
TRIGGER: Counting integers under congruence constraints.
LESSON:
- Enumerate a small range in python before generalizing.
AVOID: Extrapolating without numeric verification."""


class ScriptedAgent:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def get_action_from_gpt(self, obs):
        self.prompts.append(obs)
        return self.replies.pop(0) if self.replies else "NO_PROPOSALS"


@pytest.fixture
def arms(tmp_path):
    return {
        "baseline": BaselineArm(),
        "raw": RawExperienceArm(agent=ScriptedAgent(["A prior attempt failed on modular arithmetic."] * 20)),
        "evo": EvoHarnessArm(store_root=tmp_path / "evo",
                             agent=ScriptedAgent([REFLECTION, "ADD: 1\nREASON: useful", "NO_PATTERNS"])),
    }


def test_every_arm_emits_a_non_empty_system_prompt(arms):
    """agent.py:37 is `if self.system_prompt:` — an empty string deletes the
    system message entirely and makes the arms structurally different."""
    for name, arm in arms.items():
        arm.begin_batch(0)
        prompt = arm.system_prompt_for(PROBLEM)
        assert prompt, f"{name} produced an empty system prompt"


def test_cold_start_arms_all_emit_the_same_neutral_prompt(arms):
    for arm in arms.values():
        arm.begin_batch(0)
        assert arm.system_prompt_for(PROBLEM) == NEUTRAL_SYSTEM_PROMPT


def test_baseline_never_changes_its_prompt(arms):
    arm = arms["baseline"]
    arm.begin_batch(0)
    before = arm.system_prompt_for(PROBLEM)
    arm.observe(PROBLEM, FAILED)
    assert arm.end_batch(0) == []
    arm.begin_batch(1)
    assert arm.system_prompt_for(PROBLEM) == before


def test_evo_arm_prompt_changes_only_after_a_batch_boundary(arms):
    arm = arms["evo"]
    arm.begin_batch(0)
    cold = arm.system_prompt_for(PROBLEM)

    arm.observe(PROBLEM, FAILED)
    assert arm.system_prompt_for(PROBLEM) == cold, "a problem must not affect itself"

    arm.end_batch(0)
    arm.begin_batch(1)
    assert arm.system_prompt_for(PROBLEM) != cold


def test_evo_arm_reflects_only_on_failures(arms):
    arm = arms["evo"]
    arm.begin_batch(0)
    arm.observe(PROBLEM, PASSED)
    assert arm.end_batch(0) == []


def test_evo_arm_strips_the_gt_channel_before_reflecting(arms):
    arm = arms["evo"]
    arm.begin_batch(0)
    arm.observe(PROBLEM, FAILED)
    arm.end_batch(0)
    assert all("Matches GT" not in p for p in arm.agent.prompts)


def test_frozen_arm_does_not_update(tmp_path):
    arm = EvoHarnessArm(store_root=tmp_path / "f",
                        agent=ScriptedAgent([REFLECTION, "ADD: 1\nREASON: x", "NO_PATTERNS"]),
                        frozen=True)
    arm.begin_batch(0)
    arm.observe(PROBLEM, FAILED)
    assert arm.end_batch(0) == []
    assert arm.store.all() == []


def test_raw_experience_stores_successes_too(arms):
    arm = arms["raw"]
    arm.begin_batch(0)
    arm.observe(PROBLEM, PASSED)
    arm.end_batch(0)
    arm.begin_batch(1)
    assert arm.system_prompt_for(PROBLEM) != NEUTRAL_SYSTEM_PROMPT


def test_raw_experience_respects_the_same_token_budget_as_evo(arms):
    from alphaapollo.core.harness.render import count_tokens
    arm = arms["raw"]
    for batch in range(3):
        arm.begin_batch(batch)
        for _ in range(8):
            arm.observe(PROBLEM, FAILED)
        arm.end_batch(batch)
    arm.begin_batch(3)
    assert count_tokens(arm.system_prompt_for(PROBLEM)) <= 800 + count_tokens(NEUTRAL_SYSTEM_PROMPT)


def test_build_arm_dispatches_by_name(tmp_path):
    assert isinstance(build_arm("baseline"), BaselineArm)
    assert isinstance(build_arm("evo", store_root=tmp_path, agent=ScriptedAgent([])), EvoHarnessArm)
    with pytest.raises(ValueError):
        build_arm("nonexistent")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_arms.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'alphaapollo.core.harness.arms'`

- [ ] **Step 3: 实现**

`alphaapollo/core/harness/arms.py`:

```python
from __future__ import annotations

import logging
from pathlib import Path

from alphaapollo.core.harness.accounting import role_scope
from alphaapollo.core.harness.evolver import GeneralCurator, TopicCurator
from alphaapollo.core.harness.reflect import (build_reflect_context, reflect,
                                              sanitize_feedback)
from alphaapollo.core.harness.render import (NEUTRAL_SYSTEM_PROMPT, count_tokens,
                                             render_harness)
from alphaapollo.core.harness.store import Budget, Caps, SkillStore

log = logging.getLogger(__name__)

_RAW_HEADER = ("You are a competition mathematics solver. Summaries of your own previous "
               "attempts on other problems follow. Use them if relevant.")

_SUMMARY_PROMPT = """Summarise, in at most 40 English words, what went right or wrong in this \
attempt in a way that could help on a DIFFERENT problem. Do not restate the problem or its answer.

Topic: {topic}
Outcome: {outcome}
Verifier feedback: {verifier_feedback}
Reasoning excerpt: {reasoning_excerpt}"""


class CrossProblemArm:
    """Every arm must emit a non-empty system prompt so that the three runs
    differ in content only, never in message structure."""

    name = "base"

    def begin_batch(self, batch_idx: int) -> None: ...
    def system_prompt_for(self, problem: dict) -> str: return NEUTRAL_SYSTEM_PROMPT
    def record_selection(self, problem: dict, result: dict) -> None: ...
    def observe(self, problem: dict, result: dict) -> None: ...
    def end_batch(self, batch_idx: int) -> list[dict]: return []


class BaselineArm(CrossProblemArm):
    name = "baseline"


class RawExperienceArm(CrossProblemArm):
    name = "raw_experience"

    def __init__(self, agent, budget: Budget | None = None):
        self.agent = agent
        self.budget = budget or Budget()
        self.pool: list[dict] = []
        self._pending: list[dict] = []
        self._frozen_pool: list[dict] = []

    def begin_batch(self, batch_idx: int) -> None:
        self._frozen_pool = list(self.pool)

    def system_prompt_for(self, problem: dict) -> str:
        picked, used = [], 0
        question_words = set(problem["question"].lower().split())
        ranked = sorted(self._frozen_pool,
                        key=lambda e: (-len(question_words & set(e["text"].lower().split())), e["idx"]))
        for entry in ranked:
            if len(picked) >= self.budget.b or used + entry["tokens"] > self.budget.tokens:
                continue
            picked.append(entry)
            used += entry["tokens"]
        if not picked:
            return NEUTRAL_SYSTEM_PROMPT
        body = "\n".join(f"- {e['text']}" for e in picked)
        return f"{_RAW_HEADER}\n\n## Previous attempts\n{body}"

    def observe(self, problem: dict, result: dict) -> None:
        self._pending.append((problem, result))

    def end_batch(self, batch_idx: int) -> list[dict]:
        records = []
        for problem, result in self._pending:
            prompt = _SUMMARY_PROMPT.format(
                topic=problem["topic"],
                outcome="solved" if result["pass_final"] else "failed",
                verifier_feedback=sanitize_feedback(result["verifier_feedback"]),
                reasoning_excerpt=result["reasoning_excerpt"],
            )
            try:
                with role_scope("summarizer"):
                    text = self.agent.get_action_from_gpt(prompt).strip()
            except Exception as exc:  # noqa: BLE001
                log.warning("raw-experience summarisation failed, skipping: %s", exc)
                continue
            entry = {"idx": problem["problem_idx"], "text": text, "tokens": count_tokens(text)}
            self.pool.append(entry)
            records.append({"op": "ADD", "actor": "summarizer", "accepted": True,
                            "problem_idx": problem["problem_idx"]})
        self._pending = []
        return records


class EvoHarnessArm(CrossProblemArm):
    name = "evo_harness"

    def __init__(self, store_root: str | Path, agent, caps: Caps | None = None,
                 budget: Budget | None = None, frozen: bool = False,
                 feedback_level: str = "standard"):
        self.store = SkillStore(store_root, caps=caps or Caps(), budget=budget or Budget())
        self.agent = agent
        self.frozen = frozen
        self.feedback_level = feedback_level
        self.topic_curator, self.general_curator = TopicCurator(), GeneralCurator()
        self._frozen_skills = []
        self._pending: list[tuple[dict, dict]] = []
        self._last_selection: dict[int, list] = {}

    def begin_batch(self, batch_idx: int) -> None:
        self._frozen_skills = self.store.snapshot()

    def system_prompt_for(self, problem: dict) -> str:
        picked = self.store.select(problem["question"], problem["topic"])
        self._last_selection[problem["problem_idx"]] = picked
        return render_harness(picked)

    def record_selection(self, problem: dict, result: dict) -> None:
        picked = self._last_selection.get(problem["problem_idx"], [])
        self.store.record_usage(picked, success=bool(result["pass_final"]))
        self.store.log_selection(
            problem_idx=problem["problem_idx"], topic=problem["topic"],
            selected=[{"skill_id": s.id, "tokens": s.n_tokens} for s in picked],
            total_inject_tokens=sum(s.n_tokens for s in picked),
            pass1_round0=result["pass1_round0"], pass_final=result["pass_final"],
        )

    def observe(self, problem: dict, result: dict) -> None:
        self._pending.append((problem, result))

    def end_batch(self, batch_idx: int) -> list[dict]:
        pending, self._pending = self._pending, []
        if self.frozen:
            return []

        candidates, questions, truths = [], [], []
        for problem, result in pending:
            if result["pass_final"]:
                continue  # paper Eq. (6): reflect on failures only
            context = build_reflect_context(
                topic=problem["topic"], problem_shape=problem["problem_shape"],
                final_answer_given=result["final_answer_given"], outcome="failed",
                verifier_feedback=result["verifier_feedback"],
                tool_errors=result["tool_errors"],
                reasoning_excerpt=result["reasoning_excerpt"],
                round_count=result["round_count"],
                # Appendix E.1 feeds the related existing skills back into the
                # proposal step; reuse what was already selected for this problem
                # so this costs nothing extra.
                related_skills=self._last_selection.get(problem["problem_idx"], []),
            )
            try:
                candidate = reflect(self.agent, context, feedback_level=self.feedback_level)
            except Exception as exc:  # noqa: BLE001
                log.warning("reflection failed for problem %s: %s", problem["problem_idx"], exc)
                continue
            if candidate is None:
                continue
            candidate.evidence = [f"p_{problem['problem_idx']:04d}"]
            candidates.append((problem["topic"], candidate))
            questions.append(problem["question"])
            truths.append(str(problem.get("ground_truth", "")))

        if not candidates:
            return []

        edits = []
        for topic in sorted({t for t, _ in candidates}):
            subset = [c for t, c in candidates if t == topic]
            existing = [s for s in self.store.all() if s.level != "general" and s.topic == topic]
            edits += self.topic_curator.curate(self.agent, existing=existing, candidates=subset,
                                               topic=topic, caps=self.store.caps)
        general_existing = [s for s in self.store.all() if s.level == "general"]
        edits += self.general_curator.curate(self.agent, existing=general_existing,
                                             candidates=[c for _, c in candidates],
                                             caps=self.store.caps)

        try:
            return self.store.apply(edits, problem_idx=pending[-1][0]["problem_idx"],
                                    batch=batch_idx, question_texts=questions,
                                    ground_truths=truths)
        except Exception as exc:  # noqa: BLE001
            log.error("harness update failed, harness left unchanged: %s", exc)
            return []


def build_arm(name: str, **kwargs) -> CrossProblemArm:
    if name == "baseline":
        return BaselineArm()
    if name in ("raw", "raw_experience"):
        return RawExperienceArm(**kwargs)
    if name in ("evo", "evo_harness"):
        return EvoHarnessArm(**kwargs)
    raise ValueError(f"unknown arm: {name!r}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_arms.py -v`
Expected: PASS，10 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/arms.py tests/harness/test_arms.py
git commit -m "feat(harness): three cross-problem arms with structural prompt parity"
```

---

## Task 11: loader.py + 主题交错

**Files:**
- Create: `alphaapollo/core/harness/loader.py`
- Test: `tests/harness/test_loader.py`

**Interfaces:**
- Consumes: 无（只读 parquet）
- Produces:
  - `load_stream(path: str | Path) -> list[dict]` —— 保留 `question / ground_truth / gt_traj / topic / problem_shape / technique / year / contest / number`
  - `interleave(problems: list[dict], batch_size: int = 8, topics: tuple[str, ...] = ("algebra", "number_theory", "combinatorics", "geometry")) -> list[dict]`
  - `batches(problems: list[dict], batch_size: int = 8) -> list[list[dict]]`

**不能复用 `load_informal_math_data`**：`utils/dataset_loader.py:116-118` 只保留 `question / ground_truth / gt_traj`，topic 与 year 会被丢掉。同时 `run_problem` 在 `evolving_main.py:563` 和 `:715` 无条件读 `current_problem["gt_traj"]`，所以本 loader 必须保证该键存在（缺失时填空串）。

`interleave` 实现设计文档 §15.3：每个 batch 配成 `2 algebra + 2 number_theory + 2 combinatorics + 2 geometry`，各主题内部保持原有年份/题号顺序。

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_loader.py`:

```python
import pandas as pd
import pytest

from alphaapollo.core.harness.loader import batches, interleave, load_stream

TOPICS = ("algebra", "number_theory", "combinatorics", "geometry")


def make_parquet(tmp_path):
    rows = []
    for i in range(8):
        rows.append({
            "extra_info": {"question": f"Q{i}", "ground_truth": str(i),
                           "topic": TOPICS[i % 4], "problem_shape": "shape",
                           "technique": "divisibility-counting", "year": 2018 + i // 4,
                           "contest": "I", "number": i}
        })
    path = tmp_path / "stream.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_loader_preserves_topic_and_year(tmp_path):
    problems = load_stream(make_parquet(tmp_path))
    assert problems[0]["topic"] == "algebra"
    assert problems[0]["year"] == 2018
    assert problems[0]["technique"] == "divisibility-counting"


def test_loader_always_supplies_gt_traj(tmp_path):
    """run_problem reads current_problem["gt_traj"] unconditionally."""
    for problem in load_stream(make_parquet(tmp_path)):
        assert "gt_traj" in problem


def test_loader_assigns_sequential_problem_idx(tmp_path):
    problems = load_stream(make_parquet(tmp_path))
    assert [p["problem_idx"] for p in problems] == list(range(8))


def make_pool(per_topic=10):
    pool = []
    for t_i, topic in enumerate(TOPICS):
        for i in range(per_topic):
            pool.append({"problem_idx": t_i * per_topic + i, "topic": topic,
                         "year": 2018 + i // 5, "number": i, "question": f"{topic}-{i}"})
    return pool


def test_interleave_gives_every_batch_two_of_each_topic():
    stream = interleave(make_pool(), batch_size=8)
    for batch in batches(stream, 8):
        counts = {t: sum(1 for p in batch if p["topic"] == t) for t in TOPICS}
        assert counts == {t: 2 for t in TOPICS}


def test_interleave_preserves_within_topic_chronology():
    stream = interleave(make_pool(), batch_size=8)
    for topic in TOPICS:
        seq = [p["number"] for p in stream if p["topic"] == topic]
        assert seq == sorted(seq)


def test_interleave_renumbers_problem_idx_to_stream_position():
    stream = interleave(make_pool(), batch_size=8)
    assert [p["problem_idx"] for p in stream] == list(range(len(stream)))


def test_interleave_is_deterministic():
    assert [p["question"] for p in interleave(make_pool(), 8)] == \
           [p["question"] for p in interleave(make_pool(), 8)]


def test_interleave_stops_when_a_topic_runs_out():
    pool = [p for p in make_pool() if not (p["topic"] == "geometry" and p["number"] >= 2)]
    stream = interleave(pool, batch_size=8)
    assert len(stream) == 8, "only one full balanced batch is possible"


def test_batches_drops_the_incomplete_tail():
    assert len(batches(list(range(20)), 8)) == 2


def test_interleave_rejects_a_batch_size_not_divisible_by_topic_count():
    with pytest.raises(ValueError):
        interleave(make_pool(), batch_size=6)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_loader.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'alphaapollo.core.harness.loader'`

- [ ] **Step 3: 实现**

`alphaapollo/core/harness/loader.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_TOPICS = ("algebra", "number_theory", "combinatorics", "geometry")

_FIELDS = ("question", "ground_truth", "gt_traj", "topic", "problem_shape",
           "technique", "year", "contest", "number")


def load_stream(path: str | Path) -> list[dict]:
    """Keep topic/year/technique, which load_informal_math_data drops
    (utils/dataset_loader.py:116-118)."""
    frame = pd.read_parquet(path)
    problems: list[dict] = []
    for idx, row in enumerate(frame.to_dict("records")):
        info = row.get("extra_info") or {}
        problem = {field: info.get(field, "") for field in _FIELDS}
        problem["problem_idx"] = idx
        problems.append(problem)
    return problems


def batches(problems: list, batch_size: int = 8) -> list[list]:
    full = len(problems) // batch_size
    return [problems[i * batch_size:(i + 1) * batch_size] for i in range(full)]


def interleave(problems: list[dict], batch_size: int = 8,
               topics: tuple[str, ...] = DEFAULT_TOPICS) -> list[dict]:
    if batch_size % len(topics) != 0:
        raise ValueError(f"batch_size {batch_size} must be divisible by {len(topics)} topics")
    per_topic = batch_size // len(topics)

    queues = {
        t: sorted((p for p in problems if p["topic"] == t),
                  key=lambda p: (p.get("year", 0), str(p.get("contest", "")), p.get("number", 0)))
        for t in topics
    }

    stream: list[dict] = []
    while all(len(queues[t]) >= per_topic for t in topics):
        for topic in topics:
            stream.extend(queues[topic][:per_topic])
            queues[topic] = queues[topic][per_topic:]

    for position, problem in enumerate(stream):
        problem["problem_idx"] = position
    return stream
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_loader.py -v`
Expected: PASS，11 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/loader.py tests/harness/test_loader.py
git commit -m "feat(harness): topic-preserving loader and interleaved task stream"
```

---

## Task 12: 复用机会分析（Day 2 门禁）

**Files:**
- Create: `scripts/reuse_opportunity.py`
- Test: `tests/harness/test_reuse_opportunity.py`

**Interfaces:**
- Consumes: `load_stream`（Task 11）
- Produces:
  - `topic_gap_stats(stream: list[dict]) -> dict` —— `{"median": float, "p90": float}`
  - `technique_repeat_histogram(stream: list[dict]) -> dict[int, int]` —— 出现次数 → 有多少个技法标签
  - `reuse_upper_bound(stream: list[dict]) -> float` —— 技法在流中**非首次**出现的题目占比
  - `analyze(stream: list[dict]) -> dict`
  - `main()` —— CLI，打印报告并在 `reuse_upper_bound < 0.2` 时退出码 1

这是设计文档 §15.4 的门禁：数据准备完成后、跑任何实验之前执行。若 80% 的技法标签只出现一次，就地调整任务流再往下走。**零模型调用。**

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_reuse_opportunity.py`:

```python
from scripts.reuse_opportunity import (analyze, reuse_upper_bound,
                                       technique_repeat_histogram, topic_gap_stats)

TOPICS = ("algebra", "number_theory", "combinatorics", "geometry")


def stream(techniques):
    return [{"problem_idx": i, "topic": TOPICS[i % 4], "technique": t}
            for i, t in enumerate(techniques)]


def test_topic_gap_is_four_for_a_perfectly_interleaved_stream():
    s = [{"problem_idx": i, "topic": TOPICS[i % 4], "technique": "t"} for i in range(16)]
    assert topic_gap_stats(s)["median"] == 4.0


def test_histogram_counts_labels_by_their_frequency():
    hist = technique_repeat_histogram(stream(["a", "a", "a", "b", "c"]))
    assert hist == {3: 1, 1: 2}


def test_upper_bound_is_zero_when_every_technique_is_unique():
    assert reuse_upper_bound(stream(["a", "b", "c", "d"])) == 0.0


def test_upper_bound_counts_non_first_occurrences():
    # a,a,a,b -> three of four problems are repeats? no: positions 1 and 2 only
    assert reuse_upper_bound(stream(["a", "a", "a", "b"])) == 0.5


def test_analyze_flags_a_stream_with_no_reuse_opportunity():
    report = analyze(stream(["a", "b", "c", "d", "e", "f", "g", "h"]))
    assert report["passes_gate"] is False
    assert report["reuse_upper_bound"] == 0.0


def test_analyze_passes_a_stream_with_repeated_techniques():
    report = analyze(stream(["a", "a", "b", "b", "c", "c", "a", "b"]))
    assert report["passes_gate"] is True


def test_analyze_reports_the_share_of_single_occurrence_techniques():
    report = analyze(stream(["a", "a", "b", "c"]))
    assert report["share_of_singleton_techniques"] == 2 / 3
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_reuse_opportunity.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'scripts'`（需在 `scripts/` 下加 `__init__.py`）

- [ ] **Step 3: 实现**

`scripts/__init__.py`：空文件。

`scripts/reuse_opportunity.py`:

```python
"""Day-2 gate: does the adaptation stream contain any reuse opportunity at all?

Run this after building the stream and BEFORE spending any API budget. A stream
where almost every technique appears once cannot demonstrate cross-problem
transfer, and a null result on it would be attributable to the data rather than
to the method. Zero model calls.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict

GATE_MIN_REUSE = 0.2


def topic_gap_stats(stream: list[dict]) -> dict:
    positions: dict[str, list[int]] = defaultdict(list)
    for problem in stream:
        positions[problem["topic"]].append(problem["problem_idx"])

    gaps = [b - a for seq in positions.values() for a, b in zip(seq, seq[1:])]
    if not gaps:
        return {"median": 0.0, "p90": 0.0}
    gaps.sort()
    return {"median": float(statistics.median(gaps)),
            "p90": float(gaps[min(len(gaps) - 1, int(0.9 * len(gaps)))])}


def technique_repeat_histogram(stream: list[dict]) -> dict[int, int]:
    counts = Counter(p["technique"] for p in stream if p.get("technique"))
    return dict(Counter(counts.values()))


def reuse_upper_bound(stream: list[dict]) -> float:
    if not stream:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for problem in stream:
        technique = problem.get("technique")
        if technique in seen:
            repeats += 1
        seen.add(technique)
    return repeats / len(stream)


def analyze(stream: list[dict]) -> dict:
    histogram = technique_repeat_histogram(stream)
    total_labels = sum(histogram.values())
    singletons = histogram.get(1, 0)
    bound = reuse_upper_bound(stream)
    return {
        "n_problems": len(stream),
        "topic_gap": topic_gap_stats(stream),
        "technique_histogram": histogram,
        "n_distinct_techniques": total_labels,
        "share_of_singleton_techniques": singletons / total_labels if total_labels else 0.0,
        "reuse_upper_bound": bound,
        "passes_gate": bound >= GATE_MIN_REUSE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, help="path to stream.parquet")
    args = parser.parse_args()

    from alphaapollo.core.harness.loader import load_stream

    report = analyze(load_stream(args.stream))
    print(json.dumps(report, indent=2))
    if not report["passes_gate"]:
        print(f"\nGATE FAILED: reuse upper bound {report['reuse_upper_bound']:.1%} "
              f"< {GATE_MIN_REUSE:.0%}. Adjust the task stream before spending API budget.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_reuse_opportunity.py -v`
Expected: PASS，7 passed

- [ ] **Step 5: 提交**

```bash
git add scripts/__init__.py scripts/reuse_opportunity.py tests/harness/test_reuse_opportunity.py
git commit -m "feat(analysis): day-2 reuse-opportunity gate for the adaptation stream"
```

---

## Task 13: tracker.py —— wandb 与 jsonl 双写

**Files:**
- Create: `alphaapollo/core/harness/tracker.py`
- Test: `tests/harness/test_tracker.py`

**Interfaces:**
- Consumes: `CallAccountant`（Task 7）；`SkillStore`（Task 4–6）
- Produces:
  - `class HarnessTracker`：`__init__(self, run_dir, *, project=None, run_name=None, config=None, enabled=True)`
  - `.log(self, step: int, metrics: dict) -> None`
  - `.log_harness_state(self, step: int, store) -> None`
  - `.log_accounting(self, step: int, accountant) -> None`
  - `.finish(self) -> None`

evo 路径完全没有 wandb 集成（`wandb` 只出现在 `rl_*/sft_*` 的 config 与 `setup.py` 依赖里），所以这层要自建。**wandb 不可用或 `enabled=False` 时必须静默降级为只写 jsonl** —— 单测不能依赖网络。

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_tracker.py`:

```python
import json

from alphaapollo.core.harness.accounting import CallAccountant
from alphaapollo.core.harness.schema import Skill
from alphaapollo.core.harness.store import SkillStore
from alphaapollo.core.harness.tracker import HarnessTracker


def test_metrics_are_written_to_jsonl_without_wandb(tmp_path):
    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log(step=3, metrics={"adapt/pass_final": 1})
    tracker.finish()

    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["step"] == 3 and row["adapt/pass_final"] == 1


def test_harness_state_reports_counts_and_tokens(tmp_path):
    store = SkillStore(tmp_path / "store")
    store._write_skill(Skill(id="sk_0001", name="g-1", level="general", topic=None,
                             trigger="T.", lesson="- L.", failure_mode="A."))
    store._write_skill(Skill(id="sk_0002", name="t-2", level="topic", topic="number_theory",
                             trigger="T.", lesson="- L.", failure_mode="A."))

    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log_harness_state(step=1, store=store)
    tracker.finish()

    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["harness/n_general"] == 1
    assert row["harness/n_topic_total"] == 1
    assert row["harness/total_tokens"] > 0


def test_accounting_separates_solver_side_from_management_side(tmp_path):
    acc = CallAccountant()
    acc.record("solver", 10, 5)
    acc.record("reflect", 7, 3)

    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log_accounting(step=1, accountant=acc)
    tracker.finish()

    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["calls/solver_side_total"] == 1 and row["calls/mgmt_side_total"] == 1


def test_tracker_degrades_silently_when_wandb_is_unavailable(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no wandb here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_wandb)

    tracker = HarnessTracker(tmp_path, enabled=True, project="p", run_name="r")
    tracker.log(step=1, metrics={"x": 1})
    tracker.finish()
    assert tracker.wandb_run is None
    assert (tmp_path / "metrics.jsonl").exists()


def test_every_log_call_appends_one_line(tmp_path):
    tracker = HarnessTracker(tmp_path, enabled=False)
    for step in range(4):
        tracker.log(step=step, metrics={"x": step})
    tracker.finish()
    assert len((tmp_path / "metrics.jsonl").read_text().strip().splitlines()) == 4
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_tracker.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'alphaapollo.core.harness.tracker'`

- [ ] **Step 3: 实现**

`alphaapollo/core/harness/tracker.py`:

```python
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)


class HarnessTracker:
    """Dual-writes metrics to wandb and to a local jsonl.

    The evo path has no wandb integration at all (wandb only appears in the
    rl_*/sft_* configs, which go through verl's trainer), so this layer is new.
    The jsonl copy keeps every figure reproducible offline.
    """

    def __init__(self, run_dir: str | Path, *, project: str | None = None,
                 run_name: str | None = None, config: dict | None = None,
                 enabled: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.wandb_run = None

        if not enabled:
            return
        try:
            import wandb

            self.wandb_run = wandb.init(project=project, name=run_name,
                                        config=config or {}, dir=str(self.run_dir))
        except Exception as exc:  # noqa: BLE001 - tracking must never break a run
            log.warning("wandb unavailable, falling back to jsonl only: %s", exc)
            self.wandb_run = None

    def log(self, step: int, metrics: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, **metrics}, ensure_ascii=False) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def log_harness_state(self, step: int, store) -> None:
        skills = store.all()
        general = [s for s in skills if s.level == "general"]
        topical = [s for s in skills if s.level != "general"]
        per_topic = Counter(s.topic for s in topical)

        metrics = {
            "harness/n_general": len(general),
            "harness/n_topic_total": len(topical),
            "harness/total_tokens": sum(s.n_tokens for s in skills),
            "harness/mean_skill_tokens": (sum(s.n_tokens for s in skills) / len(skills)) if skills else 0,
        }
        metrics.update({f"harness/n_topic/{t}": n for t, n in per_topic.items()})
        metrics.update({f"usage/skill_hit/{s.id}": s.n_selected for s in skills})
        metrics.update({f"usage/skill_utility/{s.id}": s.utility() for s in skills})
        self.log(step, metrics)

    def log_accounting(self, step: int, accountant) -> None:
        self.log(step, accountant.snapshot())

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_tracker.py -v`
Expected: PASS，5 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/harness/tracker.py tests/harness/test_tracker.py
git commit -m "feat(obs): wandb + jsonl dual-write tracker for the evo path"
```

---

## Task 14: driver —— batch 间串行、batch 内并行

**Files:**
- Create: `alphaapollo/core/generation/evolving/evolving_harness_main.py`
- Test: `tests/harness/test_driver.py`

**Interfaces:**
- Consumes: 全部前序 task；`run_problem` / `create_runtime_for_problem`（`evolving_main.py:461` / `:864`）
- Produces:
  - `extract_result(problem_payload: dict) -> dict` —— 从 `run_problem` 的返回值抽出 `pass1_round0 / pass_final / final_answer_given / verifier_feedback / tool_errors / reasoning_excerpt / round_count`
  - `assert_no_gt_tool_call(action_text: str) -> None` —— 命中 `<informalmath_verify>` 即抛 `LeakageError`
  - `class LeakageError(RuntimeError)`
  - `run_stream(*, problems, arm, runtime_factory, run_problem_fn, tracker, accountant, batch_size=8, max_workers=8) -> dict`
  - `run(config: str | None = None) -> None` —— fire 入口

**核心协议**：`run_stream` 必须让同一 batch 的所有题共用 `begin_batch` 时冻结的 harness，并且 `end_batch` 只在 batch 内所有题都完成后才调用。

**指标定义**：`run_problem` 返回的 `success_rate` 是跨 evolving round 的平均（`evolving_main.py:546-548` 每轮 `total += 1`，`:822-824` 做 `success/total`），`evolving_round=3` 时取值只能是 `{0, ⅓, ⅔, 1}`，**不是 Pass@1**。干净信号是每个 step_output 的 `policy_answer_correct`（`:566-567`）。

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_driver.py`:

```python
import pytest

from alphaapollo.core.generation.evolving.evolving_harness_main import (
    LeakageError, assert_no_gt_tool_call, extract_result, run_stream)
from alphaapollo.core.harness.accounting import CallAccountant
from alphaapollo.core.harness.arms import BaselineArm, CrossProblemArm
from alphaapollo.core.harness.tracker import HarnessTracker


def problems(n=16):
    return [{"problem_idx": i, "question": f"Q{i}", "ground_truth": str(i), "gt_traj": "",
             "topic": "number_theory", "problem_shape": "shape"} for i in range(n)]


PAYLOAD = {
    "step_outputs": [
        {"round": 0, "policy_answer_correct": 0, "policy_action": "<answer>1</answer>",
         "verifier_report": "Wrong modulus.\nMatches GT: False", "tool_errors": "NameError: x"},
        {"round": 1, "policy_answer_correct": 1, "policy_action": "<answer>7</answer>",
         "verifier_report": "Looks right.", "tool_errors": ""},
    ]
}


class RecordingArm(CrossProblemArm):
    def __init__(self):
        self.events = []
        self.batch = -1

    def begin_batch(self, batch_idx):
        self.batch = batch_idx
        self.events.append(("begin", batch_idx))

    def system_prompt_for(self, problem):
        self.events.append(("prompt", problem["problem_idx"], self.batch))
        return f"harness-v{self.batch}"

    def observe(self, problem, result):
        self.events.append(("observe", problem["problem_idx"]))

    def end_batch(self, batch_idx):
        self.events.append(("end", batch_idx))
        return []


def fake_run_problem(problem_idx, problem, runtime):
    return {"problem_idx": problem_idx, "problem_payload": PAYLOAD,
            "system_prompt_seen": runtime["policy_agent"].system_prompt}


def runtime_factory(system_prompt):
    agent = type("A", (), {"system_prompt": system_prompt})()
    return {"policy_agent": agent}


def test_extract_result_uses_round0_and_final_not_success_rate():
    result = extract_result(PAYLOAD)
    assert result["pass1_round0"] == 0
    assert result["pass_final"] == 1
    assert result["round_count"] == 2


def test_extract_result_sanitizes_the_gt_channel():
    assert "Matches GT" not in extract_result(PAYLOAD)["verifier_feedback"]


def test_leakage_detector_rejects_the_verify_tool_call():
    with pytest.raises(LeakageError):
        assert_no_gt_tool_call("let me check <informalmath_verify>x</informalmath_verify>")


def test_leakage_detector_allows_normal_actions():
    assert_no_gt_tool_call("<python_code>print(1)</python_code><answer>7</answer>")


def test_end_batch_runs_only_after_every_problem_in_the_batch(tmp_path):
    arm = RecordingArm()
    run_stream(problems=problems(16), arm=arm, runtime_factory=runtime_factory,
               run_problem_fn=fake_run_problem,
               tracker=HarnessTracker(tmp_path, enabled=False),
               accountant=CallAccountant(), batch_size=8, max_workers=4)

    order = [e for e in arm.events if e[0] in ("observe", "end")]
    first_end = next(i for i, e in enumerate(order) if e == ("end", 0))
    observed_before = {e[1] for e in order[:first_end] if e[0] == "observe"}
    assert observed_before == set(range(8))


def test_all_problems_in_a_batch_see_the_same_frozen_harness(tmp_path):
    arm = RecordingArm()
    run_stream(problems=problems(16), arm=arm, runtime_factory=runtime_factory,
               run_problem_fn=fake_run_problem,
               tracker=HarnessTracker(tmp_path, enabled=False),
               accountant=CallAccountant(), batch_size=8, max_workers=4)

    prompts = [e for e in arm.events if e[0] == "prompt"]
    assert {e[2] for e in prompts[:8]} == {0}
    assert {e[2] for e in prompts[8:]} == {1}


def test_incomplete_trailing_batch_is_dropped(tmp_path):
    arm = RecordingArm()
    summary = run_stream(problems=problems(20), arm=arm, runtime_factory=runtime_factory,
                         run_problem_fn=fake_run_problem,
                         tracker=HarnessTracker(tmp_path, enabled=False),
                         accountant=CallAccountant(), batch_size=8, max_workers=4)
    assert summary["n_problems"] == 16


def test_a_failing_problem_does_not_abort_the_stream(tmp_path):
    def flaky(problem_idx, problem, runtime):
        if problem_idx == 3:
            raise RuntimeError("transient API error")
        return fake_run_problem(problem_idx, problem, runtime)

    summary = run_stream(problems=problems(8), arm=BaselineArm(),
                         runtime_factory=runtime_factory, run_problem_fn=flaky,
                         tracker=HarnessTracker(tmp_path, enabled=False),
                         accountant=CallAccountant(), batch_size=8, max_workers=4)
    assert summary["n_problems"] == 8 and summary["n_errors"] == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_driver.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named '...evolving_harness_main'`

- [ ] **Step 3: 实现**

`alphaapollo/core/generation/evolving/evolving_harness_main.py`:

```python
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from alphaapollo.core.harness.accounting import role_scope
from alphaapollo.core.harness.loader import batches
from alphaapollo.core.harness.reflect import sanitize_feedback

log = logging.getLogger(__name__)

# env.py:125-128 parses <informalmath_verify> unconditionally, and
# core/tools/informalmath_verify.py:107,176 echo "Matches ground truth" back to
# the agent. No prompt template advertises the tag today, but that is not a
# defence we are willing to rely on.
_VERIFY_TAG = re.compile(r"<informalmath_verify>")


class LeakageError(RuntimeError):
    """Raised when a run touches a channel that would expose the ground truth."""


def assert_no_gt_tool_call(action_text: str) -> None:
    if _VERIFY_TAG.search(action_text or ""):
        raise LeakageError("policy invoked <informalmath_verify>, which echoes the ground truth")


def extract_result(problem_payload: dict) -> dict:
    """run_problem's own success_rate averages over evolving rounds
    (evolving_main.py:546-548, :822-824) and is therefore not Pass@1.
    policy_answer_correct per round is the clean signal."""
    steps = (problem_payload or {}).get("step_outputs") or []
    if not steps:
        return {"pass1_round0": 0, "pass_final": 0, "final_answer_given": "",
                "verifier_feedback": "", "tool_errors": "", "reasoning_excerpt": "",
                "round_count": 0}

    last = steps[-1]
    for step in steps:
        assert_no_gt_tool_call(step.get("policy_action", ""))

    return {
        "pass1_round0": int(steps[0].get("policy_answer_correct", 0)),
        "pass_final": int(last.get("policy_answer_correct", 0)),
        "final_answer_given": str(last.get("policy_action", ""))[-200:],
        "verifier_feedback": sanitize_feedback(last.get("verifier_report", "")),
        "tool_errors": sanitize_feedback(last.get("tool_errors", "")),
        "reasoning_excerpt": str(last.get("policy_action", ""))[:1200],
        "round_count": len(steps),
    }


def run_stream(*, problems, arm, runtime_factory, run_problem_fn, tracker, accountant,
               batch_size: int = 8, max_workers: int = 8) -> dict:
    """Batch-serial, within-batch-parallel.

    Every problem in a batch uses the harness frozen at begin_batch, and no
    information flows between them, so parallelising inside a batch is exactly
    the semantics of the paper's Eq. (2)-(4). Updates happen only at batch
    boundaries, so a problem can never affect itself.
    """
    n_done = n_errors = 0

    for batch_idx, batch in enumerate(batches(problems, batch_size)):
        arm.begin_batch(batch_idx)
        prompts = {p["problem_idx"]: arm.system_prompt_for(p) for p in batch}

        results: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_one, p, prompts[p["problem_idx"]], runtime_factory,
                            run_problem_fn): p
                for p in batch
            }
            for future in as_completed(futures):
                problem = futures[future]
                try:
                    results[problem["problem_idx"]] = future.result()
                except LeakageError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.error("problem %s failed: %s", problem["problem_idx"], exc)
                    n_errors += 1
                    results[problem["problem_idx"]] = extract_result({})

        for problem in batch:  # deterministic order for logging
            result = results[problem["problem_idx"]]
            arm.record_selection(problem, result)
            arm.observe(problem, result)
            tracker.log(problem["problem_idx"], {
                "adapt/pass1_round0": result["pass1_round0"],
                "adapt/pass_final": result["pass_final"],
                "inject/n_tokens": len(prompts[problem["problem_idx"]].split()),
            })
            n_done += 1

        arm.end_batch(batch_idx)
        if getattr(arm, "store", None) is not None:
            tracker.log_harness_state(n_done, arm.store)
        tracker.log_accounting(n_done, accountant)

    return {"n_problems": n_done, "n_errors": n_errors}


def _run_one(problem, system_prompt, runtime_factory, run_problem_fn) -> dict:
    runtime = runtime_factory(system_prompt)
    with role_scope("solver"):
        payload = run_problem_fn(problem["problem_idx"], problem, runtime)
    return extract_result(payload.get("problem_payload"))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./scripts/run_tests.sh tests/harness/test_driver.py -v`
Expected: PASS，8 passed

- [ ] **Step 5: 提交**

```bash
git add alphaapollo/core/generation/evolving/evolving_harness_main.py tests/harness/test_driver.py
git commit -m "feat(evo): batch-serial within-batch-parallel harness driver"
```

---

## Task 15: 配置、运行脚本与全量回归

**Files:**
- Create: `examples/configs/harness_baseline.yaml`, `harness_raw_experience.yaml`, `harness_evo.yaml`
- Create: `scripts/run_baseline.sh`, `scripts/run_raw.sh`, `scripts/run_evo.sh`
- Test: `tests/harness/test_configs.py`

**Interfaces:**
- Consumes: 全部前序 task
- Produces: 三份 config，它们之间**只有 `harness.arm` 及其附属字段不同**

`harness_evo.yaml` 的关键段（三份 config 的其余部分必须逐字节相同）：

```yaml
harness:
  arm: evo                      # baseline | raw | evo
  batch_size: 8
  max_workers: 8
  seed: 20260911
  frozen: false                 # held-out 评测时置 true
  feedback_level: standard      # standard | minimal
  store_root: ./outputs/evo/harness
  caps: {general: 5, per_topic: 5}
  budget: {b: 6, general_max: 3, topic_max: 4, tokens: 800}
  wandb: {enabled: true, project: alphaapollo-evo-harness}
```

共享的降配段（三份完全一致）：

```yaml
env:
  config:
    informal_math_evolving:
      evolving_round: 3          # 默认 10 -> 3
      concurrency:
        verifier_max_workers: 1
        problem_max_workers: 0   # 并行由 harness driver 在 batch 内控制
run:
  verifier_env_num: 1            # 默认 5 -> 1，同时省掉 aggregator 调用
policy_model_cfg:
  max_tokens: 4096               # 默认 8192 -> 4096
  temperature: 0.7
```

- [ ] **Step 1: 写失败的测试**

`tests/harness/test_configs.py`:

```python
from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "configs"
NAMES = ("harness_baseline", "harness_raw_experience", "harness_evo")
ARM_ONLY_KEYS = {"arm", "store_root", "frozen", "feedback_level"}


@pytest.fixture
def configs():
    return {n: OmegaConf.load(CONFIG_DIR / f"{n}.yaml") for n in NAMES}


def test_all_three_configs_exist(configs):
    assert set(configs) == set(NAMES)


def test_solver_settings_are_identical_across_arms(configs):
    """实验公平性：三个 arm 只能差一个变量。"""
    stripped = []
    for cfg in configs.values():
        copy = OmegaConf.to_container(cfg, resolve=True)
        copy.pop("harness")
        stripped.append(copy)
    assert stripped[0] == stripped[1] == stripped[2]


def test_harness_sections_differ_only_in_arm_scoped_keys(configs):
    baseline = OmegaConf.to_container(configs["harness_baseline"].harness)
    evo = OmegaConf.to_container(configs["harness_evo"].harness)
    differing = {k for k in set(baseline) | set(evo) if baseline.get(k) != evo.get(k)}
    assert differing <= ARM_ONLY_KEYS, f"unexpected differences: {differing - ARM_ONLY_KEYS}"


def test_problem_level_parallelism_is_delegated_to_the_driver(configs):
    for cfg in configs.values():
        assert cfg.env.config.informal_math_evolving.concurrency.problem_max_workers == 0


def test_scaled_down_settings_match_the_design(configs):
    for cfg in configs.values():
        assert cfg.env.config.informal_math_evolving.evolving_round == 3
        assert cfg.run.verifier_env_num == 1
        assert cfg.policy_model_cfg.max_tokens == 4096


def test_every_arm_uses_the_same_seed_and_batch_size(configs):
    seeds = {cfg.harness.seed for cfg in configs.values()}
    sizes = {cfg.harness.batch_size for cfg in configs.values()}
    assert len(seeds) == 1 and len(sizes) == 1


def test_budget_matches_the_design_document(configs):
    budget = configs["harness_evo"].harness.budget
    assert (budget.b, budget.general_max, budget.topic_max, budget.tokens) == (6, 3, 4, 800)
    caps = configs["harness_evo"].harness.caps
    assert (caps.general, caps.per_topic) == (5, 5)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./scripts/run_tests.sh tests/harness/test_configs.py -v`
Expected: FAIL —— config 文件不存在

- [ ] **Step 3: 写三份 config 与三个运行脚本**

以 `examples/configs/harness_evo.yaml` 为模板，另两份只改 `harness.arm` / `harness.store_root` / `harness.frozen` / `harness.feedback_level`。三份的其余字段必须逐字节一致（上面的测试会强制这一点）。

`scripts/run_evo.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/alphaapollo/core/generation"
python -m alphaapollo.core.generation.evolving.evolving_harness_main \
  --config examples/configs/harness_evo.yaml "$@"
```

`run_baseline.sh` / `run_raw.sh` 同构，只换 config 路径。

- [ ] **Step 4: 运行全量回归**

Run: `./scripts/run_tests.sh -v`
Expected: PASS，全部测试通过（131 项）

- [ ] **Step 5: 提交**

```bash
chmod +x scripts/run_baseline.sh scripts/run_raw.sh scripts/run_evo.sh
git add examples/configs/harness_*.yaml scripts/run_*.sh tests/harness/test_configs.py
git commit -m "feat(config): three order-matched arm configs with enforced parity"
```

---

## 自检结果

**Spec 覆盖检查**（对照设计文档 v2）：

| 设计文档章节 | 对应 task |
| --- | --- |
| §2 三级 memory 边界 | Task 0（包边界静态检查）、Task 10（结构对齐） |
| §4.1 Skill schema | Task 1 |
| §4.2 增长边界 + 两阶段 apply | Task 5 |
| §4.3 确定性分层检索 | Task 6 |
| §4.4 harness_log / selection_log | Task 4、Task 10 |
| §4.5 Task A 七个测试 | Task 1/2/4/5/6（逐条覆盖） |
| §5.1 batch 协议 | Task 14 |
| §5.2 Reflect 白名单 + failure-only | Task 8、Task 10 |
| §5.3 双 curator | Task 9 |
| §5.4 五道闸 | 闸1 Task 8、闸2 Task 14、闸3 Task 8、闸4 Task 2、闸5 Task 0+Task 8 |
| §5.5 失败降级 | Task 9、Task 10、Task 14 |
| §6.1/6.2 三 arm 与公平性 | Task 10、Task 15 |
| §6.2② 记账 monkey-patch | Task 7 |
| §6.2③ seed | Task 7、Task 15 |
| §6.2④ pass1_round0 / pass_final | Task 14 |
| §7.2 log schema | Task 13 |
| §8 降配 | Task 15 |
| §15.3 主题交错 | Task 11 |
| §15.4 复用机会门禁 | Task 12 |

**未被本计划覆盖、需第二份计划**：§6.3 数据获取（`prepare_harness_stream.py` 依赖 Day 1 才能确定的 AIME 数据源）、§7.3 结果表、§9 Day 5–7 的实验执行与分析、§11 feedback-grounding 对照的实际运行、README 与 slides。这些是运行与写作任务，不适合 TDD 形态，将在 `2026-09-12-experiments-and-writeup.md` 中单独规划。

**已知前置依赖**：Task 15 的 config 需要 `prepare_harness_stream.py` 产出的 `stream.parquet` 才能真正跑起来；Task 12 的门禁同理。两者都不阻塞本计划的实现与单测。

---

## Day 1 阻塞项

- [x] **1. 环境** —— 完成并验证（2026-09-11）。conda `alphaapollo-dev`，`pip install -e . --no-deps` + 最小依赖集；跳过 flash-attn / vllm / sglang / liger-kernel / torch-memory-saver（CUDA-only，evo 路径不需要）。代价：`workflows.rl` / `workflows.sft` 在本机跑不了，但本计划全部 15 个 task 都不依赖它们。
- [ ] **2. API 实测** —— 20 次调用的延迟分布 + 24 并发（8 × 3 arm）是否限流 → 回填设计文档 §8 的 wall-clock 与 §9 的排期。**这是唯一可能推翻排期的未知数。**
- [ ] **3. AIME 数据源** —— 找到带年份元数据的数据集（`math-ai/aime24` 只有 2024 一年）。**Task 11 / 12 / 15 都依赖它**，且存在"找不到"的真实风险。
- [ ] **4. 泄漏审计** —— 跑通 evo 路径 2 题并**录制真实轨迹**（同时作为 Task 8/9/14 的 fixture 来源），核对 `informalmath_verify` 的实际输出格式，确认 Task 8 的 `_GT_CHANNEL` 正则与 Task 14 的 `_VERIFY_TAG` 能覆盖。
- [ ] **5. LICENSE** —— 核对 `a-evolve` 的 LICENSE，供 README 归属声明使用。

阻塞关系：Task 0–10、13 不依赖任何未完成项，**可以立即开工**。Task 11/12/15 需要第 3 项。Task 8/9/14 的 fixture 质量取决于第 4 项（但用合成 fixture 也能先写完测试）。
