"""SQLite backend for PRACTIQ tasks (Spider dataset).

Minimal mirror of the PostgreSQL paths in `shared/db_utils.py` for SQLite,
used only when the task's selected_database matches a Spider DB name.

Dispatch: `db_environment/server.py` checks `_detect_backend(db_name)` and
routes `create_task_db / execute / reset / drop` here when appropriate.
"""
from __future__ import annotations
import logging
import os
import pathlib
import shutil
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Set SPIDER_DB_ROOT to the directory holding `<db>/<db>.sqlite` template
# files (PRACTIQ uses the upstream Spider data root). See data/README.md.
def _default_spider_db_root() -> str:
    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    return str(repo_root / "data" / "spider-db-root")

_SPIDER_DB_ROOT = pathlib.Path(os.environ.get("SPIDER_DB_ROOT", _default_spider_db_root()))
# Task-specific copies live here (each task gets an isolated .sqlite file)
_TASK_DB_ROOT = pathlib.Path(
    os.environ.get("SQLITE_TASK_ROOT", "/tmp/practiq_task_dbs")
)
_TASK_DB_ROOT.mkdir(parents=True, exist_ok=True)

_conn_lock = threading.Lock()


def spider_template_path(db_name: str) -> pathlib.Path:
    """Return path to the read-only Spider template SQLite for this DB."""
    return _SPIDER_DB_ROOT / db_name / f"{db_name}.sqlite"


def has_spider_db(db_name: str) -> bool:
    return spider_template_path(db_name).is_file()


def task_db_path(task_db: str) -> pathlib.Path:
    return _TASK_DB_ROOT / f"{task_db}.sqlite"


def create_task_db(base_db: str, task_id: str) -> str:
    """Copy Spider template to a per-task SQLite file.

    Returns the task_db name (without extension).
    """
    safe_id = task_id.replace("-", "_").replace(".", "_")
    task_db = f"{base_db}__{safe_id}"
    src = spider_template_path(base_db)
    if not src.is_file():
        raise FileNotFoundError(f"Spider template not found: {src}")
    dst = task_db_path(task_db)
    shutil.copyfile(src, dst)
    logger.info("sqlite create_task_db: %s -> %s", src, dst)
    return task_db


def drop_task_db(task_db: str) -> None:
    p = task_db_path(task_db)
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def apply_schema_modification(task_db: str, schema_mod: Dict) -> Dict:
    """Apply PRACTIQ schemaModification to a task SQLite DB.

    Supports:
      - removeColumn: [{"table": T, "column": C}, ...]
      - addColumn:   [{"table": T, "column": C, "value": [...], "type": "TEXT"?}, ...]
      - removeColumnSemanticallyRelated / removeColumnLexicallyRelated: same as removeColumn

    Returns a summary dict with applied counts and any errors.
    """
    import sqlite3
    path = task_db_path(task_db)
    if not path.is_file():
        return {"error": f"task_db not found: {path}"}

    summary = {"removed": [], "added": [], "errors": []}
    rm_specs = []
    for key in ("removeColumn", "removeColumnSemanticallyRelated", "removeColumnLexicallyRelated"):
        for entry in schema_mod.get(key, []) or []:
            if isinstance(entry, dict) and "table" in entry and "column" in entry:
                rm_specs.append((entry["table"], entry["column"]))

    add_specs = schema_mod.get("addColumn", []) or []

    conn = sqlite3.connect(str(path), timeout=30.0)
    try:
        # 1) DROP columns (SQLite 3.35+ supports ALTER TABLE DROP COLUMN)
        for table, column in rm_specs:
            try:
                conn.execute(f'ALTER TABLE "{table}" DROP COLUMN "{column}"')
                summary["removed"].append({"table": table, "column": column})
            except sqlite3.OperationalError as e:
                summary["errors"].append(
                    {"op": "drop", "table": table, "column": column, "err": str(e)}
                )

        # 2) ADD columns + populate values if provided
        for spec in add_specs:
            if not isinstance(spec, dict): continue
            table = spec.get("table")
            column = spec.get("column")
            values = spec.get("value", []) or []
            col_type = spec.get("type", "TEXT")
            if not (table and column):
                continue
            try:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type}')
                if values:
                    # Align values to rowid in sequential order
                    conn.executemany(
                        f'UPDATE "{table}" SET "{column}" = ? WHERE rowid = ?',
                        [(v, i) for i, v in enumerate(values, start=1)],
                    )
                summary["added"].append(
                    {"table": table, "column": column, "n_values": len(values)}
                )
            except sqlite3.OperationalError as e:
                summary["errors"].append(
                    {"op": "add", "table": table, "column": column, "err": str(e)}
                )

        conn.commit()
    finally:
        conn.close()

    logger.info(
        "schema_mod %s: removed=%d added=%d errs=%d",
        task_db, len(summary["removed"]), len(summary["added"]),
        len(summary["errors"]),
    )
    return summary


def reset_task_db(task_db: str, template_source: str) -> None:
    """Re-copy the template over the task DB (Phase-2 reset)."""
    # template_source should be the base_db name (e.g. "concert_singer")
    # or a snapshot path (unused for now).
    src = spider_template_path(template_source)
    if not src.is_file():
        raise FileNotFoundError(f"Spider template not found: {src}")
    dst = task_db_path(task_db)
    shutil.copyfile(src, dst)


def execute_queries(
    queries, task_db: str, conn: Optional[sqlite3.Connection] = None
) -> Tuple[Optional[List], Optional[str], bool, Optional[List]]:
    """Execute queries on a task SQLite DB.

    Returns (result, error, timed_out, desc). Mirrors `db_utils.execute_queries`.
    """
    if isinstance(queries, str):
        queries = [queries]
    if not queries:
        return None, None, False, None

    MAX_ROWS = 10000
    STATEMENT_TIMEOUT_S = 60.0

    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(str(task_db_path(task_db)), timeout=STATEMENT_TIMEOUT_S)
        # Return rows as tuples (matching psycopg2 default)
        conn.row_factory = None

    result = None
    desc = None
    err: Optional[str] = None
    try:
        cur = conn.cursor()
        for q in queries:
            if not q or not q.strip():
                continue
            try:
                cur.execute(q)
                lower_q = q.strip().lower()
                if lower_q.startswith("select") or lower_q.startswith("with"):
                    rows = cur.fetchmany(MAX_ROWS + 1)
                    result = rows[:MAX_ROWS]
                else:
                    try:
                        result = cur.fetchall()
                    except sqlite3.Error:
                        result = None
                desc = _to_psycopg2_desc(cur.description)
            except sqlite3.OperationalError as e:
                msg = str(e)
                if "timeout" in msg.lower():
                    return None, None, True, None
                err = msg
                break
            except sqlite3.Error as e:
                err = str(e)
                break
        conn.commit()
    finally:
        if own_conn:
            try:
                conn.close()
            except Exception:
                pass
    return result, err, False, desc


def _to_psycopg2_desc(cursor_desc) -> Optional[List]:
    """sqlite3 cursor.description is a list of 7-tuples (name, ...); downstream
    callers use both desc[0] (index) and desc.name (attribute). Use a
    subscriptable named-tuple-ish class."""
    if cursor_desc is None:
        return None

    class _Col(tuple):
        @property
        def name(self):
            return self[0]

    return [_Col((d[0],) + (None,) * 6) for d in cursor_desc]
