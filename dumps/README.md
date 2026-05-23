# Preprocessed dataset dumps (academic use)

Self-contained PostgreSQL dumps + SQLite snapshots so you can run the
benchmark **without docker or sudo** — userland Postgres via conda
suffices. Schema, roles, indexes and data are byte-identical to the
upstream containers.

| File | Size | Backend | Restored DBs | Source |
|---|---:|---|---:|---|
| `lite.sql.gz` | 22 MB | PostgreSQL 14 (`pg_dumpall`) | 62 (incl. templates) | BIRD-Interact `lite_300` + `hard_60` |
| `full.sql.gz` | 17 MB | PostgreSQL 14 (`pg_dumpall`) | 57 (incl. templates) | BIRD-Interact `full_600` |
| `mini_interact.tar.xz` | 13 MB | SQLite | 30 dbs | PRACTIQ mini-interact |

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

The dumps create a role `root` with password `123123`. Match the docker
layout so the eval runtime's `PG_HOST=127.0.0.1 PG_USER=root
PG_PASSWORD=123123 PG_PORT=5432` works unchanged.

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
ports (e.g. lite on 5432, full on 5433). Set `PG_PORT` in your shell
to whichever target you want each run to use.

### 1.3 Stop / restart

```bash
pg_ctl -D ~/pgdata_lite stop          # or: -m fast for immediate
pg_ctl -D ~/pgdata_lite -l ~/pglog_lite.log start
```

### 1.4 Faithfulness vs upstream docker

Postgres 14 in both; no docker-only extensions are required. Performance
is within ~1 % of the docker container.

---

## 2. PRACTIQ mini-interact (`mini_interact.tar.xz`)

SQLite-backed; each task DB is a single `.sqlite` file. The agent's
`shared/sqlite_utils.py` auto-routes queries through SQLite when the
task's `selected_database` matches one of these dbs; PG otherwise.

### 2.1 Restore into the data/ tree

```bash
cd <release root>
tar -xJf dumps/mini_interact.tar.xz -C data/
ls data/mini-interact-hf-meta/   # 30 db dirs: alien/, archeology/, …, virtual/
```

Each db dir contains `<db>.sqlite` and `<db>_template.sqlite`. The
db-environment service copies from the template at the start of every
task to guarantee a clean state; runtime only writes `<db>.sqlite`.

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

For PRACTIQ mini-interact tasks the runtime auto-detects SQLite — no
env change needed beyond extracting the tarball above.
