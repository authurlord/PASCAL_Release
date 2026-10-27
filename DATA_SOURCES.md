# Data and attribution

This static adaptation of **PASCAL_demo_portable_20260923.zip** preserves the original demo interface and exports its replay-facing responses as per-task JSON. Replay timing is simulated by the browser. It does not execute SQL, call an LLM, or reproduce original runtime latency.

- Paper: [When the Bottleneck Shifts: Diagnosing and Closing Information Gaps in Interactive Text-to-SQL](https://aclanthology.org/2026.emnlp-main.465/), EMNLP 2026.
- Method code: [PASCAL_Release](https://github.com/authurlord/PASCAL_Release).
- Benchmark: [BIRD-Interact](https://bird-interact.github.io/), [upstream code](https://github.com/bird-bench/BIRD-Interact), [lite data](https://huggingface.co/datasets/birdsql/bird-interact-lite), [full data](https://huggingface.co/datasets/birdsql/bird-interact-full).
- The portable bundle's upstream data cards declare **CC BY-SA 4.0**; the exported benchmark-derived data retain that license. See [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Software is MIT; the upstream notice is preserved in [LICENSE](LICENSE).

## Contents

| Stored run | Lite | Full |
|---|---:|---:|
| PASCAL frozen | 300 | 600 |
| PASCAL submitted NVD | 300 | 593 |
| ReAct NVD | 290 | 0 |

There are 900 distinct tasks and 2,083 stored runs, not 2,083 distinct tasks. Historical missing records are not fabricated. The frozen runs differ from the submitted NVD runs; the selector identifies which set is being displayed. Labels and outcomes are taken from each selected archive, not recomputed scores.

Schema, knowledge-base entries, archived samples, reference SQL, and cached Chinese explanations are presentation data from the supplied portable demo. Cached explanations were generated previously and may contain errors. Neither these explanations nor replay success labels substitute for a new benchmark evaluation. No benchmark grader, database dumps, API keys, Python environment, or model weights are deployed with this website.
