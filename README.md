# PASCAL — anonymous code release

PASCAL is a **training-free protocol for interactive text-to-SQL** on the
BIRD-Interact (`lite_300`, `full_600`), mini-interact, and PRACTIQ
benchmarks. The release contains the main-method code, the model
cards for the two Qwen3.6 checkpoints used in the paper, preprocessed
PostgreSQL / SQLite dumps for offline reproduction, and one command
that runs either the PASCAL anchor or the official ReACT baseline.

## Repository layout

```
PASCAL_release/
├── README.md                ← you are here
├── LICENSE                  ← MIT
├── requirements.txt
├── docs/
│   └── MODEL_CARDS.md       ← Qwen3.6-35B / 27B + Gemini setup
├── data/
│   ├── README.md            ← BIRD-Interact + PRACTIQ download / GT merge
│   ├── combine_public_with_gt.py
│   └── hard_60.jsonl        ← 60-task hard subset (smoke test)
├── dumps/
│   ├── README.md            ← restore recipe (no docker / sudo required)
│   ├── lite.sql.gz          ← BIRD-Interact lite_300 (22 MB)
│   ├── full.sql.gz          ← BIRD-Interact full_600 (17 MB)
│   └── mini_interact.tar.xz ← mini-interact 30 SQLite dbs (13 MB)
├── src/
│   ├── orchestrator/        ← parallel runner + a-interact pipeline
│   ├── system_agent/        ← ADK agent (PASCAL prompt + ReACT prompt)
│   ├── user_simulator/      ← two-stage Gemini-driven user sim
│   ├── db_environment/      ← PG / SQLite isolation + evaluation + KB
│   └── shared/              ← config, LLM, KB pre-loader, etc.
├── scripts/
│   ├── start_vllm_qwen36_35b.sh
│   ├── start_vllm_qwen36_27b.sh
│   ├── start_services.sh    ← boots the three microservices
│   └── run_eval.sh          ← anchor | react entry point
└── examples/
    └── smoke_hard12.sh      ← 12-task smoke test on hard_60
```

## Quick start

```bash
# 1. Install deps + CUDA build of vLLM (>=0.19.1)
pip install -r requirements.txt
pip install "vllm>=0.19.1"

# 2. Bring up PostgreSQL + restore the lite dump (~5 min, one-time).
#    See dumps/README.md for the full recipe (no docker / sudo required).
SCOPE=lite PORT=5432 bash -c '
  conda install -n base -c conda-forge -y postgresql &&
  initdb -D $HOME/pgdata_$SCOPE --auth-local=trust --auth-host=md5 &&
  echo "port = $PORT"                       >> $HOME/pgdata_$SCOPE/postgresql.conf &&
  echo "listen_addresses = \"127.0.0.1\""   >> $HOME/pgdata_$SCOPE/postgresql.conf &&
  pg_ctl -D $HOME/pgdata_$SCOPE -l $HOME/pglog_$SCOPE.log start &&
  gunzip -c dumps/$SCOPE.sql.gz | psql -h 127.0.0.1 -p $PORT -U $USER -d postgres'

# 3. Download BIRD-Interact lite split + merge ground truth.
#    The GT JSONL is obtained by emailing the upstream maintainers; see
#    data/README.md.
mkdir -p data && cd data && \
  git clone https://huggingface.co/datasets/birdsql/bird-interact-lite bird-interact-lite-hf-meta && \
  cd .. && \
  python data/combine_public_with_gt.py \
    data/bird-interact-lite-hf-meta/bird_interact_data.jsonl \
    /path/to/bird_interact_gt_kg_testcases.jsonl \
    data/bird-interact-lite-hf-meta/bird_interact_data.jsonl

# 4. Boot vLLM, then run the smoke test
export GOOGLE_API_KEY=<your_gemini_key>                    # see docs/MODEL_CARDS.md
PASCAL_GPUS=0,1 bash scripts/start_vllm_qwen36_35b.sh &    # ~5 min warmup
bash examples/smoke_hard12.sh                              # ~10-20 min
```

Inspect `results/smoke_hard12_anchor.json` — `metrics.phase1_rate` lands
in the 35–45 % range on the 12-task hard subset under the 35B-A3B
anchor.

## Reproduce on the full lite split

```bash
bash scripts/run_eval.sh anchor          # PASCAL anchor  (~3-4 h on lite_300)
bash scripts/run_eval.sh react           # official ReACT (~2 h on lite_300)
```

`run_eval.sh anchor` activates the PASCAL anchor (PASCAL prompt +
streamlined tools + schema pre-injection; `PASCAL_NO_VALUE_DIFF=1`
disables value-diff oracle feedback). `run_eval.sh react` activates the
official ReACT baseline (`PASCAL_NO_PROTOCOL=1` — minimal prompt and
the upstream 9-tool surface minus KB tools).

## Running on the other benchmarks

The same code path runs all four splits — the only thing that changes
is the dataset directory and (for the SQLite-backed splits) the
`SPIDER_DB_ROOT` env var. The agent routes tasks to PostgreSQL or
SQLite using each task's own metadata (`_practiq_meta` → SQLite,
`follow_up` → PG, otherwise the file-system probe rooted at
`SPIDER_DB_ROOT`).

| Split | Data file | Backend | Env overrides |
|---|---|---|---|
| BIRD-Interact lite | `data/bird-interact-lite-hf-meta/bird_interact_data.jsonl` | PostgreSQL | `DB_METADATA_ROOT=<lite root>` |
| BIRD-Interact full | `data/bird-interact-full-hf-meta/bird_interact_data.jsonl` | PostgreSQL | `DB_METADATA_ROOT=<full root>`, `PG_PORT=<full port>` |
| Mini-interact | `data/mini-interact-hf-meta/mini_interact.jsonl` | SQLite | `DB_METADATA_ROOT=<mini root>`, `SPIDER_DB_ROOT=<mini root>` |
| PRACTIQ medium | upstream `bird_format.jsonl` | SQLite (Spider) | `SPIDER_DB_ROOT=<Spider data root>` |

PRACTIQ medium requires the upstream Spider database collection —
see the PRACTIQ repo.

## Hardware

| Component | Recipe |
|---|---|
| vLLM server | 2× GPUs, total VRAM ≥ 40 GB (35B-A3B-FP8) or ≥ 48 GB (27B-FP8) |
| Eval runner | CPU + ~10 GB RAM per concurrency unit |
| PostgreSQL | userland 14.x (conda); ≤ 1 GB data |

The included vLLM launcher enables MTP=3 speculative decoding by
default (set `PASCAL_NUM_SPEC_TOKENS=0` to disable).

## Datasets

BIRD-Interact (`lite_300`, `full_600`), mini-interact, and PRACTIQ come
from upstream maintainers and are **not redistributed in raw form**:

* BIRD-Interact: public splits via HuggingFace; ground truth obtained
  by email to the upstream team (see `data/README.md`).
* PRACTIQ: download from
  https://github.com/amazon-science/conversational-ambiguous-unanswerable-text2sql .

For convenience this release ships preprocessed PostgreSQL / SQLite
dumps in `dumps/` so reviewers can reproduce without docker. The dumps
are derived solely from upstream public artifacts and are for
academic use.

## License

MIT — see `LICENSE`.

## Citation

Anonymous submission. The canonical citation will be added upon
de-anonymization. Please cite BIRD-Interact and PRACTIQ as appropriate
for their respective benchmarks.
