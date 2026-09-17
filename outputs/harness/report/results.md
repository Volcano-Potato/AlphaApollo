# Task C results

## 1. Adaptation stream

Compared on the 144 of 144 problems every arm completed. Problems each arm lost: baseline (0), raw (0), evo (0).

| arm | n | Pass@1 (round 0) | final | on own set (final) | lost |
|---|---|---|---|---|---|
| baseline | 144 | 22.2% | 21.5% | 21.5% | 0 |
| raw | 144 | 16.7% | 21.5% | 21.5% | 0 |
| evo | 144 | 23.6% | 24.3% | 24.3% | 0 |

### Over time (windows of 25)

**baseline**

| window | n | Pass@1 | final |
|---|---|---|---|
| 0-24 | 25 | 24.0% | 24.0% |
| 25-49 | 25 | 24.0% | 32.0% |
| 50-74 | 25 | 20.0% | 16.0% |
| 75-99 | 25 | 36.0% | 28.0% |
| 100-124 | 25 | 24.0% | 20.0% |
| 125-149 | 19 | 0.0% | 5.3% |

**raw**

| window | n | Pass@1 | final |
|---|---|---|---|
| 0-24 | 25 | 20.0% | 20.0% |
| 25-49 | 25 | 16.0% | 28.0% |
| 50-74 | 25 | 16.0% | 24.0% |
| 75-99 | 25 | 24.0% | 32.0% |
| 100-124 | 25 | 8.0% | 8.0% |
| 125-149 | 19 | 15.8% | 15.8% |

**evo**

| window | n | Pass@1 | final |
|---|---|---|---|
| 0-24 | 25 | 28.0% | 28.0% |
| 25-49 | 25 | 20.0% | 32.0% |
| 50-74 | 25 | 20.0% | 16.0% |
| 75-99 | 25 | 32.0% | 36.0% |
| 100-124 | 25 | 24.0% | 16.0% |
| 125-149 | 19 | 15.8% | 15.8% |

## 2. Held-out set (final frozen harness)

| arm | n | Pass@1 (round 0) | final | lost |
|---|---|---|---|---|
| baseline | 30 | 13.3% | 10.0% | 0 |
| raw | 30 | 23.3% | 23.3% | 0 |
| evo | 30 | 13.3% | 16.7% | 0 |

## 3. By topic

**adaptation / baseline**

| topic | n | Pass@1 | final |
|---|---|---|---|
| algebra | 30 | 33.3% | 36.7% |
| combinatorics | 33 | 18.2% | 12.1% |
| geometry | 49 | 14.3% | 14.3% |
| number_theory | 32 | 28.1% | 28.1% |

**adaptation / raw**

| topic | n | Pass@1 | final |
|---|---|---|---|
| algebra | 30 | 16.7% | 36.7% |
| combinatorics | 33 | 9.1% | 9.1% |
| geometry | 49 | 18.4% | 16.3% |
| number_theory | 32 | 21.9% | 28.1% |

**adaptation / evo**

| topic | n | Pass@1 | final |
|---|---|---|---|
| algebra | 30 | 50.0% | 53.3% |
| combinatorics | 33 | 9.1% | 15.2% |
| geometry | 49 | 12.2% | 14.3% |
| number_theory | 32 | 31.2% | 21.9% |

**held-out / baseline**

| topic | n | Pass@1 | final |
|---|---|---|---|
| algebra | 9 | 11.1% | 0.0% |
| combinatorics | 9 | 11.1% | 11.1% |
| geometry | 7 | 0.0% | 0.0% |
| number_theory | 5 | 40.0% | 40.0% |

**held-out / raw**

| topic | n | Pass@1 | final |
|---|---|---|---|
| algebra | 9 | 22.2% | 22.2% |
| combinatorics | 9 | 11.1% | 11.1% |
| geometry | 7 | 28.6% | 28.6% |
| number_theory | 5 | 40.0% | 40.0% |

**held-out / evo**

| topic | n | Pass@1 | final |
|---|---|---|---|
| algebra | 9 | 0.0% | 11.1% |
| combinatorics | 9 | 11.1% | 11.1% |
| geometry | 7 | 28.6% | 28.6% |
| number_theory | 5 | 20.0% | 20.0% |

## 4. Harness size, length and growth

**evo**

| after problem | general | topic | total tokens | mean/skill |
|---|---|---|---|---|
| 7 | 1 | 5 | 398 | 66 |
| 15 | 2 | 7 | 600 | 67 |
| 23 | 3 | 7 | 647 | 65 |
| 31 | 3 | 9 | 768 | 64 |
| 39 | 3 | 9 | 787 | 66 |
| 47 | 3 | 9 | 781 | 65 |
| 55 | 5 | 9 | 913 | 65 |
| 63 | 5 | 9 | 912 | 65 |
| 71 | 5 | 9 | 887 | 63 |
| 79 | 5 | 9 | 891 | 64 |
| 87 | 5 | 9 | 876 | 63 |
| 95 | 5 | 9 | 871 | 62 |
| 103 | 5 | 10 | 912 | 61 |
| 111 | 5 | 10 | 908 | 61 |
| 119 | 5 | 11 | 981 | 61 |
| 127 | 5 | 11 | 977 | 61 |
| 135 | 5 | 11 | 978 | 61 |
| 143 | 5 | 11 | 1003 | 63 |

## 5. Injected context and model calls

| arm | solver calls | mgmt calls | calls/problem | tokens in | tokens out | mean injected tok | unscoped |
|---|---|---|---|---|---|---|---|
| baseline | 1468 | 0 | 10.2 | 1.34M | 1.66M | 0 | 0 |
| raw | 1448 | 144 | 11.1 | 1.64M | 1.61M | 390 | 0 |
| evo | 1228 | 321 | 10.8 | 1.53M | 1.09M | 185 | 0 |

Management calls are reported separately from solver calls: an arm that wins on accuracy while spending materially more calls has not obviously won. A non-zero `unscoped` column is an instrumentation bug, not a cost category.

### Held-out (frozen, no updates)

| arm | solver calls | mgmt calls | calls/problem | tokens in | tokens out | total tokens |
|---|---|---|---|---|---|---|
| baseline | 309 | 0 | 10.30 | 289,078 | 286,503 | 575,581 |
| raw | 298 | 0 | 9.93 | 323,470 | 266,737 | 590,207 |
| evo | 263 | 30 | 9.77 | 359,659 | 227,676 | 587,335 |

Raw token counts here rather than the M-rounded adaptation table above: the held-out spread across arms is a few percent of the total, and rounding to millions would hide exactly that. Across the 30 problems/arm, total tokens span 2.5% of the smallest. On adaptation, evo's total token count was -12.8% relative to baseline; on held-out it is +2.0% -- if that saving was meant to generalize, it does not repeat here. With only 30 problems per arm and a single seed, this is as consistent with sampling noise as with a real reversal, not evidence either way.

## 6. Skill usage frequency

**evo** — 15 skill(s) were injected at least once

| skill | times injected | solved when injected | share of problems |
|---|---|---|---|
| sk_0006 | 118 | 27 | 81.9% |
| sk_0008 | 74 | 20 | 51.4% |
| sk_0007 | 47 | 7 | 32.6% |
| sk_0012 | 31 | 6 | 21.5% |
| sk_0013 | 31 | 5 | 21.5% |
| sk_0003 | 28 | 6 | 19.4% |
| sk_0014 | 26 | 6 | 18.1% |
| sk_0001 | 18 | 9 | 12.5% |
| sk_0011 | 16 | 2 | 11.1% |
| sk_0015 | 15 | 2 | 10.4% |
| sk_0010 | 12 | 2 | 8.3% |
| sk_0009 | 11 | 3 | 7.6% |
| sk_0002 | 9 | 2 | 6.2% |
| sk_0005 | 1 | 0 | 0.7% |
| sk_0016 | 1 | 0 | 0.7% |

"Solved when injected" is an association, not an attribution -- a skill picked for easy problems scores well without helping. Section 7 pairs the same problems against Baseline, which is where a causal claim can start.

## 7. Transfer cases

**Positive (Evo solved, Baseline did not)** — 10 candidate(s)

| problem | topic | skills injected | tokens | trajectory |
|---|---|---|---|---|
| 11 | combinatorics | sk_0006 | 72 | trajectories/problem_0011.json |
| 19 | algebra | sk_0006,sk_0008 | 135 | trajectories/problem_0019.json |
| 36 | algebra | sk_0006,sk_0008,sk_0010 | 168 | trajectories/problem_0036.json |
| 61 | algebra | sk_0006,sk_0008,sk_0013 | 187 | trajectories/problem_0061.json |
| 64 | combinatorics | sk_0007,sk_0014,sk_0006 | 182 | trajectories/problem_0064.json |
| 78 | geometry | sk_0003,sk_0008,sk_0014 | 196 | trajectories/problem_0078.json |
| 84 | number_theory | sk_0001,sk_0006,sk_0008 | 188 | trajectories/problem_0084.json |
| 108 | algebra | sk_0001,sk_0006,sk_0008,sk_0013 | 233 | trajectories/problem_0108.json |
| 109 | geometry | sk_0015,sk_0006,sk_0008 | 186 | trajectories/problem_0109.json |
| 137 | geometry | sk_0003,sk_0015,sk_0009 | 174 | trajectories/problem_0137.json |

**Negative (Baseline solved, Evo did not)** — 6 candidate(s)

| problem | topic | skills injected | tokens | trajectory |
|---|---|---|---|---|
| 45 | geometry | sk_0003,sk_0006,sk_0008 | 184 | trajectories/problem_0045.json |
| 58 | number_theory | sk_0006,sk_0013,sk_0008 | 187 | trajectories/problem_0058.json |
| 68 | number_theory | sk_0012,sk_0007,sk_0006,sk_0013,sk_0014 | 311 | trajectories/problem_0068.json |
| 105 | number_theory | sk_0014,sk_0007,sk_0013 | 178 | trajectories/problem_0105.json |
| 110 | combinatorics | sk_0012,sk_0007,sk_0006 | 163 | trajectories/problem_0110.json |
| 116 | geometry | sk_0015,sk_0006,sk_0003 | 187 | trajectories/problem_0116.json |

These are candidates, not conclusions: a transfer case is an argument about what the model did differently, which needs the trajectory read alongside the injected skill text. The tables above say which files to open.
