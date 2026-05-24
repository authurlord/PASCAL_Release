# Datasets

PASCAL is evaluated on the following benchmarks:

| Benchmark | Tasks | Source |
|---|---:|---|
| BIRD-Interact **lite** | 300 | https://bird-interact.github.io/ |
| BIRD-Interact **full** | 600 | https://bird-interact.github.io/ |
| **PRACTIQ medium** | 1069 | https://github.com/amazon-science/conversational-ambiguous-unanswerable-text2sql |

This release reuses both datasets unchanged for academic comparison. We
**do not redistribute ground truth** (`sol_sql`, `test_cases`) for
BIRD-Interact — the upstream maintainers require requesters to email
them directly (see step 2 below).

This directory contains the helper scripts and the smoke-test subset
shipped with the release:

- `combine_public_with_gt.py` — merge the upstream public JSONL with the
  GT JSONL you receive by email, to produce the runnable
  `bird_interact_data.jsonl` that the orchestrator expects.
- `hard_60.jsonl` — the 60-task hard subset of BIRD-Interact-lite.
  The release's smoke test (`examples/smoke_hard12.sh`) runs the first
  12 tasks of this file.

## Recommended layout

```
data/
├── bird-interact-lite-hf-meta/    # from HuggingFace, see step 1
│   ├── bird_interact_data.jsonl   # merged with GT — step 3
│   └── <db_name>/                 # per-DB schema / column meanings / KB
└── bird-interact-full-hf-meta/    # (optional, same layout)
```

`db_environment/server.py` probes `data/bird-interact-lite-hf-meta` and
`data/bird-interact-full-hf-meta` automatically; override the search
root with `DB_METADATA_ROOT=/abs/path/to/your/layout` if needed.

---

## BIRD-Interact (lite / full)

### 1. Public split (HuggingFace)

```bash
cd data
git lfs install
git clone https://huggingface.co/datasets/birdsql/bird-interact-lite bird-interact-lite-hf-meta
# optional full set
git clone https://huggingface.co/datasets/birdsql/bird-interact-full bird-interact-full-hf-meta
```

### 2. Ground truth + test cases (email)

The public release does **not** include `sol_sql` or `test_cases`. To
obtain them, email **bird.bench25@gmail.com** with one of:

* subject `[bird-interact-lite GT&Test Cases]` for the lite split
* subject `[bird-interact-full GT&Test Cases]` for the full split

You will receive a GT JSONL by reply.

### 3. Merge

```bash
python data/combine_public_with_gt.py \
  data/bird-interact-lite-hf-meta/bird_interact_data.jsonl \
  /path/to/bird_interact_gt_kg_testcases.jsonl \
  data/bird-interact-lite-hf-meta/bird_interact_data.jsonl
```

### 4. Databases — two options

**Option A (recommended): upstream Docker image.** The BIRD-Interact team
publishes pre-built PostgreSQL images:

```bash
docker pull shawnxxh/bird-interact-postgresql:latest      # lite
docker pull shawnxxh/bird-interact-postgresql-full:latest # full
docker run -d --name pg_lite -p 5432:5432 \
  -e POSTGRES_USER=root -e POSTGRES_PASSWORD=123123 \
  shawnxxh/bird-interact-postgresql:latest
```

**Option B (academic-use): preprocessed PG dumps.** We ship our own
`pg_dumpall` outputs in `dumps/lite.sql.gz` and `dumps/full.sql.gz` so
you can run without docker. See `dumps/README.md` for the restore
recipe. The schema/roles/indexes/data are byte-identical to the upstream
docker image; we verified the test cases pass under both backends.

---

## PRACTIQ (medium, 1069 tasks)

The release includes the **task JSONL** for PRACTIQ medium at
`data/practiq_medium.jsonl.gz` (1069 tasks, ~640 KB, ground truth
stripped).  Each task carries `_practiq_meta`, `external_knowledge`,
`amb_user_query`, etc. — enough to inspect task structure and dispatch
the agent.

The release does **not** redistribute the underlying Spider databases
(~5 GB).  To run PRACTIQ medium end-to-end:

1. Download the Spider database collection per the PRACTIQ repo
   instructions:
   https://github.com/amazon-science/conversational-ambiguous-unanswerable-text2sql
2. Email the PRACTIQ team for ground truth (or regenerate it from the
   PRACTIQ generation pipeline in the repo above).
3. Merge the GT JSONL into `data/practiq_medium.jsonl.gz` (extract,
   run `combine_public_with_gt.py`, gzip back).
4. Point the runtime at the Spider data root:
   ```bash
   export SPIDER_DB_ROOT=<spider-data-root>
   bash scripts/run_eval.sh anchor \
     --data data/practiq_medium.jsonl
   ```

Upstream PRACTIQ paper: *PRACTIQ: A Practical Conversational
Text-to-SQL dataset with Ambiguous and Unanswerable Queries* (Dong
et al., NAACL 2025) — https://aclanthology.org/2025.naacl-long.13/ .
