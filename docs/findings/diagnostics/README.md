# Solver 参数诊断阶梯（2026-09-13/14）

这里是**原始产物**，不是结论的复述 —— 结论在 `../experiment-log.md`。每个 run 存两个文件：

- `<run>.yaml` —— 跑这一次用的完整配置，逐字保留，只把 scratchpad 绝对路径换成 `<RUN_DIR>`
- `<run>.metrics.jsonl` —— driver 写出的逐题记录和调用计账

## 为什么存在

正式实验的 solver 参数必须**三臂逐字共享**（作业的公平性要求），所以定错一次，三个 arm 一起错。开跑前先用 20 题把参数定下来。

AlphaApollo 官方配置（`examples/configs/vllm_informal_math.yaml`）是 `evolving_round: 10` / `verifier_env_num: 5` / `max_tokens: 8192`；全量照搬的成本在 149 题 × 3 arm × 2 阶段的规模上不可接受，所以需要知道**每个旋钮各自买到了什么**。

## 设置

固定的 20 道题：`data/harness/adaptation.parquet` 的前 20 道（AIME 2018-I，四个 topic 齐全）。**baseline arm** —— 不注入任何跨题知识，把 harness 从测量里摘出去，量的是 solver 本身。模型 qwen3-8b @ DashScope，`max_workers=5`，`seed=1234`。

| run | `evolving_round` | `verifier_env_num` | `max_tokens` | wandb |
|---|---|---|---|---|
| `diagA` | 2 | 1 | 2048 | `moegrvnn` |
| `diagA2` | 2 | 1 | 2048 | `4z1wlm78` |
| `diagB` | 2 | **5** | 2048 | `ks84hdar` |
| `diagC` | 2 | 5 | **8192** | `zmvvfpfr` |
| `diagE` | 2 | 1 | **8192** | `l0a7w001` |
| `diagD` | **10** | 1 | 2048 | `o91ntebx` |

`diagA2` 和 `diagA` 的配置**逐字相同**，唯一作用是量噪声底 —— 没有它，上面任何两行之间的差值都无法解读。

阶梯是刻意设计成可归因的：`diagB` 把 `verifier_env_num` 单独隔离出来，所以 `diagC` 相对 `diagA` 的差值能被拆开；`diagE` 再把 `max_tokens` 单独隔离，用来交叉验证 `diagC`（结果证明这一步是必要的，见日志）。

## 怎么读

```bash
# 逐题 round0 / final
grep adapt/pass1_round0 docs/findings/diagnostics/diagA.metrics.jsonl

# 调用与 token 计账（每个 batch 末尾一行）
grep calls/solver docs/findings/diagnostics/diagA.metrics.jsonl | tail -1
```

`pass1_round0` 是第 0 轮的正确性（题内 evolution 之前），`pass_final` 是最后一轮的。两者都由上游自己对 ground truth 打分，不经过本项目的任何代码。

## 补充

全部六个点已跑完，结论见 `../experiment-log.md`。
