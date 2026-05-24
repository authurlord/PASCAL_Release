# Preprocessed dataset dumps (academic use)

Self-contained PostgreSQL / SQLite database snapshots so reproduction
works **without docker or sudo** — userland Postgres via conda
suffices.  See the *Source* and *Provenance* sections below for the
exact upstream artifacts and the recipe used to produce each file.

## Inventory

| File | Size | Backend | Restored DBs | Upstream source |
|---|---:|---|---:|---|
| `lite.sql.gz` | 22 MB | PostgreSQL 14 (`pg_dumpall`) | 62 (incl. templates / sub-DBs) | [`shawnxxh/bird-interact-postgresql:latest`](https://hub.docker.com/r/shawnxxh/bird-interact-postgresql) (BIRD-Interact `lite_300` + `hard_60`) |
| `full.sql.gz` | 17 MB | PostgreSQL 14 (`pg_dumpall`) | 57 (incl. templates / sub-DBs) | [`shawnxxh/bird-interact-postgresql-full:latest`](https://hub.docker.com/r/shawnxxh/bird-interact-postgresql-full) (BIRD-Interact `full_600`) |
| `mini_interact.tar.xz` | 13 MB | SQLite | 30 dbs | [`birdsql/bird-interact-lite`](https://huggingface.co/datasets/birdsql/bird-interact-lite) — `mini-interact-hf-meta/` reorganisation of the BIRD-Interact tasks under a SQLite layout |

> **PRACTIQ medium (1069 tasks)** is **not** shipped in this release —
> the underlying Spider databases (~5 GB) are distributed separately
> by the upstream PRACTIQ team.  See `data/README.md` for the download
> recipe (PRACTIQ repo + Spider data root).

## Provenance / what we merged

**Nothing.**  Every byte in these dumps is a direct export of the
canonical upstream image:

| File | Recipe |
|---|---|
| `lite.sql.gz`, `full.sql.gz` | `pg_dumpall` against the upstream Docker image at the port it exposes, then `gzip`. No schema, role, or row modifications. |
| `mini_interact.tar.xz` | `tar -cJf` over the in-tree `mini-interact-hf-meta/` directory, packed with `--numeric-owner --owner=0 --group=0` so the archive contains no UID/GID metadata. |

The dumps contain **only databases** — no ground truth (`sol_sql`,
`test_cases`) is embedded anywhere.  The upstream BIRD-Interact team
distributes ground truth separately by email request; see
`data/README.md` for the merge step.

---

## 0. Prerequisites

* conda or any Postgres 14 install (no docker / sudo required)
* ~1 GB free disk for PG data + ~150 MB for SQLite files
* TCP ports 5432 / 5433 free (or pick your own)

## 1. PostgreSQL (`lite.sql.gz`, `full.sql.gz`)

### 1.1 Install Postgres in user space

```bash
conda install -n base -c conda-forge -y postgresql      # 14.x
which postgres pg_ctl initdb psql                       # all under ~/miniconda3/bin/
```

### 1.2 Restore — pick `lite` or `full`, or both

The dumps create a role `root` with password `123123` (this matches
the upstream Docker image's default).  Match that role so the eval
runtime's `PG_HOST=127.0.0.1 PG_USER=root PG_PASSWORD=123123
PG_PORT=5432` works unchanged.

**Single dataset:**

```bash
SCOPE=lite                     # or: full
PORT=5432
PGDATA=$HOME/pgdata_$SCOPE
PGLOG=$HOME/pglog_$SCOPE.log

initdb -D "$PGDATA" --auth-local=trust --auth-host=md5
echo "port = $PORT" >> "$PGDATA/postgresql.conf"
echo "listen_addresses = '127.0.0.1'" >> "$PGDATA/postgresql.conf"

pg_ctl -D "$PGDATA" -l "$PGLOG" start
gunzip -c $SCOPE.sql.gz | psql -h 127.0.0.1 -p $PORT -U $USER -d postgres

# Sanity check (expect ~62 lite / ~57 full):
psql -h 127.0.0.1 -p $PORT -U root -d postgres -c \
    "SELECT count(*) FROM pg_database WHERE datistemplate = false AND datname NOT IN ('postgres', 'root');"
```

**Both lite + full on the same host:** use two `PGDATA` dirs and two
ports (e.g. lite on 5432, full on 5433).  Set `PG_PORT` in your shell
to whichever target each run should use.

### 1.3 Stop / restart

```bash
pg_ctl -D ~/pgdata_lite stop          # or: -m fast for immediate
pg_ctl -D ~/pgdata_lite -l ~/pglog_lite.log start
```

### 1.4 Faithfulness vs upstream docker

Postgres 14 in both; no docker-only extensions are required.
Performance is within ~1 % of the docker container.

---

## 2. Mini-interact (`mini_interact.tar.xz`)

Mini-interact is a SQLite-backed reorganisation of a 300-task subset of
BIRD-Interact.  Each task DB is a single `.sqlite` file under
`mini-interact-hf-meta/<db>/<db>.sqlite`.  The eval runtime auto-routes
queries through SQLite when the task's metadata indicates SQLite (see
`src/db_environment/server.py:_is_sqlite_backend`).

### 2.1 Restore into the data/ tree

```bash
cd <release root>
tar -xJf dumps/mini_interact.tar.xz -C data/
ls data/mini-interact-hf-meta/   # 30 db dirs: alien/, archeology/, …, virtual/
```

Each db dir contains `<db>.sqlite` and `<db>_template.sqlite`.  The
db-environment service copies from the template at the start of every
task to guarantee a clean state; runtime writes only `<db>.sqlite`.

### 2.2 Smoke check

```bash
sqlite3 data/mini-interact-hf-meta/exchange_traded_funds/exchange_traded_funds.sqlite \
    ".tables" | head
```

---

## 3. Point the runtime at this setup

Set the following in your shell before launching the services
(`scripts/start_services.sh` reads them):

```bash
export PG_HOST=127.0.0.1
export PG_PORT=5432             # or 5433 if you ran full on its own port
export PG_USER=root
export PG_PASSWORD=123123
export PG_DATABASE=postgres
export DATASET=lite             # or: full
```

For mini-interact tasks the runtime auto-detects SQLite — no env
change needed beyond extracting the tarball above and pointing
`SPIDER_DB_ROOT=<path to mini-interact-hf-meta>` so the dispatcher
recognises the SQLite layout.

---

## 4. PRACTIQ medium (not included; download separately)

PRACTIQ medium tasks reference the **Spider** database collection,
which the upstream PRACTIQ team distributes separately.  To run
PRACTIQ medium with this release:

1. Clone the PRACTIQ repo and follow its setup to obtain the Spider
   data root:
   https://github.com/amazon-science/conversational-ambiguous-unanswerable-text2sql
2. Get the task JSONL (BIRD-format conversion):
   https://github.com/amazon-science/conversational-ambiguous-unanswerable-text2sql
3. Place the Spider DB collection at any path and point the eval
   runtime at it: `export SPIDER_DB_ROOT=<spider-data-root>`.
4. Run with `bash scripts/run_eval.sh anchor-mini --data <practiq jsonl>`.

The release does not redistribute Spider DBs or PRACTIQ ground truth.
