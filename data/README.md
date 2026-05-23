# Datasets

PASCAL is evaluated on three benchmarks:

| Benchmark | Tasks | Source |
|---|---:|---|
| BIRD-Interact **lite** | 300 | https://bird-interact.github.io/ |
| BIRD-Interact **full** | 600 | https://bird-interact.github.io/ |
| PRACTIQ **mini-interact** | 300 | https://github.com/amazon-science/conversational-ambiguous-unanswerable-text2sql |

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
├── bird-interact-full-hf-meta/    # (optional, same layout)
└── mini-interact-hf-meta/         # PRACTIQ tasks (SQLite-backed)
    └── <db>/<db>.sqlite
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

## PRACTIQ mini-interact

We use the PRACTIQ split as reorganised under `mini-interact-hf-meta/`,
which keeps each task's SQLite database alongside its metadata. The
release ships this set in `dumps/mini_interact.tar.xz` (per the PRACTIQ
licence — academic use).

```bash
tar -xJf dumps/mini_interact.tar.xz -C data/
ls data/mini-interact-hf-meta/   # 30 db dirs
```

`shared/sqlite_utils.py:_detect_backend(db_name)` auto-routes queries
through SQLite when the task's `selected_database` matches a PRACTIQ
DB; no extra configuration needed.

The upstream PRACTIQ paper is "PRACTIQ: A Practical Conversational
Text-to-SQL dataset with Ambiguous and Unanswerable Queries"; see the
GitHub repo for the canonical citation.
