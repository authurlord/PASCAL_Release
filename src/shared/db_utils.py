"""Database utilities ported from BIRD-Interact evaluation code."""

import json
import logging
import os
import re
import threading
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2 import OperationalError, sql
from psycopg2.pool import ThreadedConnectionPool

from shared.config import settings

logger = logging.getLogger(__name__)

_postgresql_pools: Dict[str, ThreadedConnectionPool] = {}
_pool_lock = threading.Lock()


_PG_FULL_PORT = 5433  # Full dataset DBs on separate PG instance

def _resolve_pg_port(db_name: str) -> int:
    """Auto-detect which PG port has this database (lite=5432, full=5433)."""
    # Try primary port first
    try:
        conn = psycopg2.connect(
            dbname=db_name, user=settings.pg_user,
            password=settings.pg_password, host=settings.pg_host,
            port=settings.pg_port, connect_timeout=3,
        )
        conn.close()
        return settings.pg_port
    except Exception:
        pass
    # Try full port
    try:
        conn = psycopg2.connect(
            dbname=db_name, user=settings.pg_user,
            password=settings.pg_password, host=settings.pg_host,
            port=_PG_FULL_PORT, connect_timeout=3,
        )
        conn.close()
        return _PG_FULL_PORT
    except Exception:
        pass
    return settings.pg_port  # fallback to default


# Cache resolved ports to avoid repeated probing
_db_port_cache: Dict[str, int] = {}

def _get_or_init_pool(db_name: str) -> ThreadedConnectionPool:
    with _pool_lock:
        if db_name not in _postgresql_pools:
            if db_name not in _db_port_cache:
                _db_port_cache[db_name] = _resolve_pg_port(db_name)
            port = _db_port_cache[db_name]
            _postgresql_pools[db_name] = ThreadedConnectionPool(
                settings.pg_minconn, settings.pg_maxconn,
                dbname=db_name, user=settings.pg_user,
                password=settings.pg_password, host=settings.pg_host,
                port=port,
            )
        return _postgresql_pools[db_name]


def close_pool(db_name: str):
    with _pool_lock:
        if db_name in _postgresql_pools:
            pool = _postgresql_pools.pop(db_name)
            pool.closeall()


def perform_query(query: str, db_name: str, conn=None):
    MAX_ROWS = 10000
    pool = _get_or_init_pool(db_name)
    if conn is None:
        conn = pool.getconn()
    cursor = conn.cursor()
    cursor.execute("SET statement_timeout = '60s';")
    try:
        cursor.execute(query)
        conn.commit()
        lower_q = query.strip().lower()
        if lower_q.startswith("select") or lower_q.startswith("with"):
            rows = cursor.fetchmany(MAX_ROWS + 1)
            result = rows[:MAX_ROWS]
        else:
            try:
                result = cursor.fetchall()
            except psycopg2.ProgrammingError:
                result = None
        desc = cursor.description
        return result, conn, desc
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()


def execute_queries(queries, db_name: str, conn=None):
    """Execute queries and return (result, error, timeout, cursor_description)."""
    if isinstance(queries, str):
        queries = [queries]
    if not queries:
        return None, None, False, None
    result = None
    desc = None
    for query in queries:
        if not query or not query.strip():
            continue
        try:
            result, conn, desc = perform_query(query, db_name, conn=conn)
        except psycopg2.errors.QueryCanceled:
            return None, None, True, None
        except (OperationalError, psycopg2.Error) as e:
            return None, str(e), False, None
        except Exception as e:
            return None, str(e), False, None
    return result, None, False, desc


def _admin_connection(db_name: str = "postgres"):
    """Open an autocommit admin connection for DB-level DDL."""
    if db_name not in _db_port_cache:
        _db_port_cache[db_name] = _resolve_pg_port(db_name)
    port = _db_port_cache[db_name]
    conn = psycopg2.connect(
        dbname=db_name,
        user=settings.pg_user,
        password=settings.pg_password,
        host=settings.pg_host,
        port=port,
    )
    conn.autocommit = True
    return conn


def _terminate_db_connections(cursor, db_name: str):
    cursor.execute(
        """
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = %s AND pid <> pg_backend_pid();
        """,
        (db_name,),
    )


def _drop_and_create_db(db_name: str, template_db: str):
    """Drop db_name (if exists) and recreate from template_db."""
    close_pool(db_name)
    # Resolve port from template_db (strip _template suffix to get base DB name)
    base_for_port = template_db.replace("_template", "").split("__")[0]
    if base_for_port not in _db_port_cache:
        _db_port_cache[base_for_port] = _resolve_pg_port(base_for_port)
    port = _db_port_cache[base_for_port]
    conn = psycopg2.connect(
        dbname="postgres", user=settings.pg_user,
        password=settings.pg_password, host=settings.pg_host, port=port,
    )
    conn.autocommit = True
    try:
        cursor = conn.cursor()
        try:
            _terminate_db_connections(cursor, db_name)
            cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))
            cursor.execute(
                sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                    sql.Identifier(db_name),
                    sql.Identifier(template_db),
                )
            )
        finally:
            cursor.close()
    finally:
        conn.close()


def reset_and_restore_database(db_name: str):
    template_db = f"{db_name}_template"
    _drop_and_create_db(db_name, template_db)


def create_task_db(base_db: str, task_id: str, template: str = None) -> str:
    """Create a per-task DB copy. Returns task DB name.

    Args:
        base_db: Base database name (used for naming the task DB).
        task_id: Task identifier (sanitized for DB naming).
        template: Template DB to copy from. Defaults to {base_db}_template.
    """
    import hashlib
    safe_id = task_id.replace("-", "_").replace(".", "_")
    task_db = f"{base_db}__{safe_id}"
    if len(task_db) > 63:
        h = hashlib.md5(safe_id.encode()).hexdigest()[:10]
        max_base = 63 - 2 - len(h)
        task_db = f"{base_db[:max_base]}__{h}"
        logger.info("create_task_db: hashed long name -> %s (base=%s id=%s)",
                    task_db, base_db, task_id)
    template_db = template or f"{base_db}_template"
    _drop_and_create_db(task_db, template_db)
    return task_db


def reset_task_db(task_db: str, template_source: str):
    """Reset a per-task DB from a template/snapshot DB."""
    _drop_and_create_db(task_db, template_source)


def drop_task_db(task_db: str):
    """Drop a per-task DB and close its connection pool."""
    close_pool(task_db)
    # Resolve port from task_db name (format: base_db__task_id)
    base = task_db.split("__")[0]
    conn = _admin_connection(base)
    try:
        cursor = conn.cursor()
        try:
            _terminate_db_connections(cursor, task_db)
            cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(task_db)))
        finally:
            cursor.close()
    finally:
        conn.close()


def get_connection_for_phase(db_name: str):
    pool = _get_or_init_pool(db_name)
    return pool.getconn()


def process_decimals_recursive(item, decimal_places: int):
    quantizer = Decimal(1).scaleb(-decimal_places)
    if isinstance(item, Decimal):
        return item.quantize(quantizer, rounding=ROUND_HALF_UP)
    elif isinstance(item, float):
        return round(item, decimal_places)
    elif isinstance(item, (list, tuple)):
        return type(item)(process_decimals_recursive(x, decimal_places) for x in item)
    elif isinstance(item, dict):
        return {k: process_decimals_recursive(v, decimal_places) for k, v in item.items()}
    return item


def preprocess_results(results, decimal_places: int = 2):
    if results is None:
        return []
    processed = []
    for row in results:
        processed_row = []
        for item in row:
            if isinstance(item, (date, datetime)):
                processed_row.append(item.strftime("%Y-%m-%d"))
            else:
                pi = process_decimals_recursive(item, decimal_places)
                if isinstance(pi, (dict, list)):
                    processed_row.append(json.dumps(pi, sort_keys=True))
                else:
                    processed_row.append(pi)
        processed.append(tuple(processed_row))
    return processed


def remove_comments(sql_list: List[str]) -> List[str]:
    cleaned = []
    for sql in sql_list:
        no_block = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        no_line = re.sub(r"--.*?(\r\n|\r|\n)", r"\1", no_block)
        no_blank = re.sub(r"\n\s*\n+", "\n", no_line)
        cleaned.append(no_blank.strip())
    return cleaned


def remove_distinct(sql_list: List[str]) -> List[str]:
    return [" ".join(t for t in q.split(" ") if t.lower() != "distinct") for q in sql_list]


def _remove_round_functions(sql_string: str) -> str:
    def find_matching_paren(text, start):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(": depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0: return i
        return -1

    def find_first_arg_end(text, start):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(": depth += 1
            elif text[i] == ")":
                if depth == 0: return i
                depth -= 1
            elif text[i] == "," and depth == 0: return i
        return len(text)

    result = sql_string
    while True:
        match = re.search(r"ROUND\s*\(", result, re.IGNORECASE)
        if not match: break
        start = match.start()
        open_p = match.end() - 1
        first_end = find_first_arg_end(result, open_p + 1)
        close_p = find_matching_paren(result, open_p)
        if close_p == -1: break
        first_arg = result[open_p + 1: first_end].strip()
        result = result[:start] + first_arg + result[close_p + 1:]
    return result


def remove_round(sql_list: List[str]) -> List[str]:
    return [_remove_round_functions(sql) for sql in sql_list]


def ex_base(pred_sqls, sol_sqls, db_name, conn, conditions=None) -> int:
    if not pred_sqls or not sol_sqls:
        return 0
    pred_res, pred_err, pred_to, _ = execute_queries(pred_sqls, db_name, conn)
    gt_res, gt_err, gt_to, _ = execute_queries(sol_sqls, db_name, conn)
    if any([pred_err, pred_to, gt_err, gt_to]):
        return 0
    pred_res = preprocess_results(pred_res)
    gt_res = preprocess_results(gt_res)
    if not pred_res or not gt_res:
        return 0
    if conditions and conditions.get("order", False):
        return 1 if pred_res == gt_res else 0
    return 1 if set(pred_res) == set(gt_res) else 0


def test_case_default(pred_sqls, sol_sqls, db_name, conn, conditions=None):
    pred_sqls = remove_round(remove_distinct(remove_comments(pred_sqls)))
    sol_sqls = remove_round(remove_distinct(remove_comments(sol_sqls)))
    result = ex_base(pred_sqls, sol_sqls, db_name, conn, conditions)
    assert result == 1, f"ex_base returned {result} but expected 1."
    return result
