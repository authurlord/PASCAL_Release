"""DB Environment Service (Port 6002). SQL execution, submission, schema/knowledge."""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import settings
from shared.db_utils import (
    _get_or_init_pool, close_pool, execute_queries,
    reset_and_restore_database, test_case_default,
    ex_base, remove_distinct, remove_comments, remove_round,
    create_task_db, reset_task_db, drop_task_db,
    preprocess_results,
)
from shared import sqlite_utils


def _is_sqlite_backend(db_name: str, task_data: dict | None = None) -> bool:
    """Return True if this DB should use SQLite, else PG.

    Task-level disambiguation (prevents BIRD-Interact / PRACTIQ name
    collisions):

      * `_practiq_meta` truthy → SQLite (PRACTIQ medium / Spider).
      * `follow_up` truthy → PG (BIRD-Interact lite / full).
      * Otherwise → file-system probe via `has_spider_db()`. The probe
        looks for `<SPIDER_DB_ROOT>/<db_name>/<db_name>.sqlite`; set
        `SPIDER_DB_ROOT` to the Spider data root for PRACTIQ medium.
    """
    if task_data is not None:
        if task_data.get("_practiq_meta"):
            return True
        if task_data.get("follow_up"):
            return False
    return sqlite_utils.has_spider_db(db_name)
from shared.feedback import compute_value_diff
from shared.models import (
    ExecuteSQLRequest, ExecuteSQLResponse, InitTaskRequest,
    SchemaRequest, ColumnMeaningRequest, KnowledgeRequest,
    SubmitSQLRequest, SubmitSQLResponse,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="BIRD-Interact DB Environment", version="1.0.0")

MAX_RESULT_LENGTH = 500
KNOWLEDGE_VISIBLE_FIELDS = ["id", "knowledge", "description", "definition"]
PASCAL_NO_VALUE_DIFF = os.environ.get(
    "PASCAL_NO_VALUE_DIFF", os.environ.get("NO_VALUE_DIFF", "0")
) == "1"
PASCAL_NO_SHAPE_HINT = os.environ.get(
    "PASCAL_NO_SHAPE_HINT", os.environ.get("NO_SHAPE_HINT", "0")
) == "1"
# Submit discipline + SQL-repetition breaker (PASCAL anchor extras).
PASCAL_SUBMIT_CAP = int(os.environ.get("PASCAL_SUBMIT_CAP", "0") or "0")
PASCAL_SQL_REPEAT_CAP = int(os.environ.get("PASCAL_SQL_REPEAT_CAP", "0") or "0")
# Evidence ledger — accumulates KB / column facts the agent has acquired
# and surfaces them back once the submit budget is mostly spent. Only
# positive facts are recorded (NOT-FOUND lookups are skipped to avoid
# planting authoritative-looking empty facts), and injection is delayed
# until submit_count >= PASCAL_SUBMIT_CAP + PASCAL_LEDGER_DELAY_AFTER_CAP
# so the soft warning has a chance to nudge the agent first.
PASCAL_EVIDENCE_LEDGER = os.environ.get("PASCAL_EVIDENCE_LEDGER", "0") == "1"
PASCAL_LEDGER_DELAY_AFTER_CAP = int(os.environ.get("PASCAL_LEDGER_DELAY_AFTER_CAP", "3") or "3")

_task_data: Dict[str, Dict[str, Any]] = {}
_schema_cache: Dict[str, str] = {}
_column_meanings_cache: Dict[str, Dict] = {}
_external_knowledge_cache: Dict[str, Dict] = {}
_submit_attempts: Dict[str, Dict[int, int]] = {}
_successful_phase1_sql: Dict[str, str] = {}
# Track normalized SQL signatures per (task_id, phase) for repetition detection.
_submit_sql_history: Dict[str, Dict[int, list]] = {}
# Per-task evidence ledger — chronological list of concise fact strings
# extracted from KB / column-meaning responses as the agent acquires them.
_evidence_ledger: Dict[str, list] = {}


def _record_fact(task_id: str, fact: str, max_len: int = 220) -> None:
    """Append a single fact to the evidence ledger. No-op when toggle off.

    Dedupe: if `fact` is identical to the most recent ledger entry for this
    task, skip — repeated identical lookups (e.g. agent calls
    get_column_meaning twice for the same column) shouldn't push older
    facts out of the latest-N window.
    """
    if not PASCAL_EVIDENCE_LEDGER:
        return
    if not task_id or not fact:
        return
    if len(fact) > max_len:
        fact = fact[:max_len].rstrip() + "..."
    led = _evidence_ledger.setdefault(task_id, [])
    if led and led[-1] == fact:
        return  # exact-repeat dedupe
    led.append(fact)


def _format_ledger(task_id: str, max_entries: int = 12) -> str:
    """Render the per-task ledger as a compact bullet list. Empty string when toggle off or no facts."""
    if not PASCAL_EVIDENCE_LEDGER:
        return ""
    ledger = _evidence_ledger.get(task_id) or []
    if not ledger:
        return ""
    shown = ledger[-max_entries:]
    elided = len(ledger) - len(shown)
    header = f"[EVIDENCE LEDGER — {len(ledger)} acquired facts in this task]"
    if elided > 0:
        header += f" (showing latest {len(shown)}; {elided} earlier omitted)"
    body = "\n".join(f"  - {f}" for f in shown)
    return f"{header}\n{body}"


def _sql_signature(sql: str) -> str:
    """Whitespace-collapsed, lowercased, first-200-char signature for SQL equivalence."""
    import re as _re
    return _re.sub(r"\s+", " ", (sql or "").strip().lower())[:200]


def _apply_submit_discipline(task_id: str, phase: int, sql: str, original_msg: str) -> str:
    """Protocol-v2 message annotator (only active when env caps are set).

    Two behaviors, env-toggled (defaults OFF — backwards compatible). Both
    APPEND advisory text to the original message; they NEVER replace the
    shape/value-diff feedback. Removing the evaluator feedback (as the
    round-1 hard override did) destabilized agents that would otherwise
    recover from shape mismatch — see smoke40 mental_2 / polar_M_3 traces.

    - PASCAL_SUBMIT_CAP=N: after N failed submits on the same phase,
      append a [SUBMIT WARNING] that suggests changing strategy or asking
      the user. The evaluator's feedback (shape, value-diff) is preserved.

    - PASCAL_SQL_REPEAT_CAP=N: when the same normalized SQL signature
      appears N+ times in this phase's submit history, prepend a
      [REPETITION DETECTED] warning to the original message.
    """
    if PASCAL_SUBMIT_CAP <= 0 and PASCAL_SQL_REPEAT_CAP <= 0:
        return original_msg

    history = _submit_sql_history.setdefault(task_id, {}).setdefault(phase, [])
    sig = _sql_signature(sql)
    history.append(sig)
    submit_count = len(history)
    same_sig_count = history.count(sig)

    parts = []
    if PASCAL_SQL_REPEAT_CAP > 0 and same_sig_count >= PASCAL_SQL_REPEAT_CAP:
        parts.append(
            f"[REPETITION DETECTED] You have submitted the SAME SQL "
            f"{same_sig_count} times. The test case will return the same "
            f"result. Re-explore the schema/KB for a missing column or "
            f"term, or call ask_user to clarify the user's intent."
        )
    parts.append(original_msg)
    if PASCAL_SUBMIT_CAP > 0 and submit_count >= PASCAL_SUBMIT_CAP:
        # Ledger injection is delayed: only attach acquired facts when
        # the agent has burned through
        # SUBMIT_CAP + LEDGER_DELAY_AFTER_CAP failures AND has at least one
        # positive fact recorded. This gives the lighter [SUBMIT WARNING]
        # alone a chance to nudge the agent first.
        warning = (
            f"[SUBMIT WARNING] You have submitted {submit_count} times in "
            f"this phase. Preserve the shape/error feedback above; before "
            f"another submit, change a join/filter/formula or call "
            f"ask_user for the missing rule. Calling abstain is also "
            f"valid if the task is genuinely impossible."
        )
        ledger_threshold = PASCAL_SUBMIT_CAP + PASCAL_LEDGER_DELAY_AFTER_CAP
        if submit_count >= ledger_threshold:
            ledger_block = _format_ledger(task_id)
            if ledger_block:
                parts.append(ledger_block)
        parts.append(warning)
    return "\n\n".join(parts)


# Per-DB schema / column-meaning / external-KB files are looked up under
# `<root>/<db_name>/`. We probe a small set of roots in order so the
# same code runs whether the dataset directory is the ADK default
# (`bird-interact-{lite,full}/`) or the upstream HuggingFace layout
# (`bird-interact-{lite,full}-hf-meta/`). Set DB_METADATA_ROOT to override.
import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parent.parent.parent
_DB_METADATA_ROOTS = [
    r for r in [
        os.environ.get("DB_METADATA_ROOT"),
        settings.db_data_path,
        str(_REPO_ROOT / "data" / "bird-interact-lite-hf-meta"),
        str(_REPO_ROOT / "data" / "bird-interact-full-hf-meta"),
    ] if r
]


def _find_db_folder(db_name: str) -> str:
    """Return first roots/{db_name} folder that exists, else first root."""
    for root in _DB_METADATA_ROOTS:
        candidate = os.path.join(root, db_name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(_DB_METADATA_ROOTS[0], db_name)


def _load_sqlite_schema(db_name: str) -> str:
    """Pull DDL from sqlite_master for a Spider DB."""
    import sqlite3
    src = sqlite_utils.spider_template_path(db_name)
    try:
        conn = sqlite3.connect(str(src))
        try:
            rows = conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
                "ORDER BY type DESC, name"
            ).fetchall()
        finally:
            conn.close()
        return ";\n\n".join(r[0] for r in rows) + ";\n"
    except Exception as e:
        logger.error(f"sqlite schema load failed for {db_name}: {e}")
        return "Schema not available"


def _load_db_data(db_name: str, use_sqlite: bool = False):
    if db_name in _schema_cache:
        return
    # SQLite (PRACTIQ): schema from sqlite_master, no KB
    if use_sqlite:
        _schema_cache[db_name] = _load_sqlite_schema(db_name)
        _column_meanings_cache[db_name] = {}
        _external_knowledge_cache[db_name] = {}
        return
    db_folder = _find_db_folder(db_name)
    # Schema
    try:
        with open(os.path.join(db_folder, f"{db_name}_schema.txt")) as f:
            _schema_cache[db_name] = f.read()
    except Exception as e:
        logger.error(f"Schema load failed for {db_name} (folder={db_folder}): {e}")
        _schema_cache[db_name] = "Schema not available"
    # Column meanings
    try:
        with open(os.path.join(db_folder, f"{db_name}_column_meaning_base.json")) as f:
            raw = json.load(f)
        _column_meanings_cache[db_name] = {k.lower(): v for k, v in raw.items()}
    except Exception as e:
        logger.error(f"Column meanings load failed for {db_name} (folder={db_folder}): {e}")
        _column_meanings_cache[db_name] = {}
    # Knowledge
    try:
        kb = {}
        with open(os.path.join(db_folder, f"{db_name}_kb.jsonl")) as f:
            for line in f:
                if not line.strip(): continue
                entry = json.loads(line.strip())
                kb[entry["knowledge"]] = entry
        _external_knowledge_cache[db_name] = kb
    except Exception as e:
        logger.error(f"Knowledge load failed for {db_name} (folder={db_folder}): {e}")
        _external_knowledge_cache[db_name] = {}


def _filter_knowledge(db_name: str, record: Dict) -> Dict:
    full_kb = _external_knowledge_cache.get(db_name, {})
    if not full_kb: return {}
    agent_kb = full_kb.copy()
    deleted_ids = set()
    for amb in record.get("knowledge_ambiguity", []):
        dk = amb.get("deleted_knowledge")
        if dk is not None: deleted_ids.add(dk)
    if deleted_ids:
        to_remove = [k for k, v in agent_kb.items() if v.get("id") in deleted_ids]
        for k in to_remove: del agent_kb[k]
    return agent_kb


def _format_result(result, cursor_desc=None) -> str:
    if result is None: return "Query executed successfully."
    if not isinstance(result, list): return str(result)
    if not result: return "Query executed, empty result set."
    lines = []
    if cursor_desc:
        cols = [desc[0] for desc in cursor_desc]
        lines.append(" | ".join(cols))
        lines.append("-" * min(len(lines[0]), 200))
    for row in result[:100]:
        cells = [str(c)[:100] for c in row]
        lines.append(" | ".join(cells))
    text = "\n".join(lines)
    words = text.split()
    if len(words) > MAX_RESULT_LENGTH:
        text = " ".join(words[:MAX_RESULT_LENGTH]) + "..."
    return text


@app.post("/init_task")
async def init_task(req: InitTaskRequest):
    _task_data[req.task_id] = req.task_data
    _submit_attempts[req.task_id] = {1: 0, 2: 0}
    _submit_sql_history[req.task_id] = {1: [], 2: []}
    _evidence_ledger[req.task_id] = []
    db_name = req.task_data["selected_database"]
    # PRACTIQ tasks carry `_practiq_meta` → SQLite backend.
    # BIRD-Interact tasks carry `follow_up` / `external_knowledge` → PG.
    use_sqlite = _is_sqlite_backend(db_name, req.task_data)
    req.task_data["_backend"] = "sqlite" if use_sqlite else "pg"
    if not use_sqlite:
        _load_db_data(db_name, use_sqlite=False)
        task_db = await asyncio.to_thread(create_task_db, db_name, req.task_id)
    else:
        task_db = await asyncio.to_thread(
            sqlite_utils.create_task_db, db_name, req.task_id
        )
        # Apply PRACTIQ schema modification if present (ambig/unans requires
        # the original column to be removed and/or replacements added so the
        # agent sees the task-relevant schema).
        sm = (req.task_data.get("_practiq_meta", {}) or {}).get("schema_modification")
        if sm:
            summary = await asyncio.to_thread(
                sqlite_utils.apply_schema_modification, task_db, sm
            )
            req.task_data["_schema_mod_summary"] = summary
            # Invalidate cached schema so next /schema call reloads from
            # the modified sqlite_master of the TASK DB (not the template).
            _schema_cache.pop(db_name, None)
    req.task_data["_task_db"] = task_db
    return {"status": "ok", "task_id": req.task_id}


def _execute_sql_sync(task_db: str, sql: str, backend: str = "pg") -> ExecuteSQLResponse:
    """Blocking SQL execution — runs in thread pool."""
    if backend == "sqlite":
        try:
            result, err, timeout, desc = sqlite_utils.execute_queries(sql, task_db)
            if err:
                return ExecuteSQLResponse(result="", success=False, error=f"SQL error: {err}")
            if timeout:
                return ExecuteSQLResponse(result="", success=False, error="SQL execution timed out")
            formatted = _format_result(result, desc)
            return ExecuteSQLResponse(result=formatted, success=True)
        except Exception as e:
            logger.error(f"execute_sql (sqlite) error for {task_db}: {type(e).__name__}: {e}")
            return ExecuteSQLResponse(result="", success=False, error=str(e))
    try:
        pool = _get_or_init_pool(task_db)
        conn = pool.getconn()
        try:
            # Reset connection if it's in a bad state
            if conn.closed:
                pool.putconn(conn, close=True)
                conn = pool.getconn()
            try:
                conn.reset()
            except Exception as reset_err:
                logger.warning(f"conn.reset() failed for {task_db}: {reset_err}, getting fresh conn")
                try:
                    pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = pool.getconn()
            result, err, timeout, desc = execute_queries(sql, task_db, conn)
            if err:
                return ExecuteSQLResponse(result="", success=False, error=f"SQL error: {err}")
            if timeout:
                return ExecuteSQLResponse(result="", success=False, error="SQL execution timed out")
            formatted = _format_result(result, desc)
            return ExecuteSQLResponse(result=formatted, success=True)
        finally:
            try:
                pool.putconn(conn)
            except Exception:
                pass
    except Exception as e:
        logger.error(f"execute_sql error for {task_db}: {type(e).__name__}: {e}")
        error_msg = str(e) or f"{type(e).__name__}: {repr(e)}"
        return ExecuteSQLResponse(result="", success=False, error=error_msg)


@app.post("/execute", response_model=ExecuteSQLResponse)
async def execute_sql_endpoint(req: ExecuteSQLRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    task_db = td.get("_task_db", td["selected_database"])
    backend = td.get("_backend", "pg")
    # SELECT-only: prevent agent from modifying DB state (strip comments first)
    sql_cleaned = re.sub(r'--.*$', '', req.sql, flags=re.MULTILINE)
    sql_cleaned = re.sub(r'/\*.*?\*/', '', sql_cleaned, flags=re.DOTALL)
    sql_upper = sql_cleaned.strip().upper()
    if not sql_upper.startswith(("SELECT", "WITH", "EXPLAIN")):
        return ExecuteSQLResponse(result="", success=False, error="Only SELECT queries allowed in execute_sql")
    return await asyncio.to_thread(_execute_sql_sync, task_db, req.sql, backend)


def _practiq_submit_sync(req_task_id, req_sql, td, _submit_attempts) -> SubmitSQLResponse:
    """PRACTIQ scoring: match-any-valid for ambig, abstain for unans."""
    meta = td.get("_practiq_meta", {})
    category = meta.get("category", "")
    expected = meta.get("expected_behaviour", "")
    task_db = td.get("_task_db", td["selected_database"])

    if req_task_id not in _submit_attempts:
        _submit_attempts[req_task_id] = {1: 0, 2: 0}
    _submit_attempts[req_task_id][1] += 1

    # UNANSWERABLE: agent must use the abstain() tool (not submit_sql).
    # If agent reaches here via submit_sql, this task SHOULD NOT have been
    # submitted — return a neutral failure message that does NOT leak how
    # to abstain (avoids scoring-hint gaming). Correct abstain comes through
    # the dedicated /abstain endpoint below.
    if expected == "abstain":
        try:
            _, pred_err, _, _ = sqlite_utils.execute_queries(req_sql, task_db)
        except Exception as e:
            pred_err = str(e)
        err_desc = f" (sql error: {pred_err})" if pred_err else ""
        return SubmitSQLResponse(
            passed=False, reward=0.0, phase_completed=None, has_follow_up=False,
            message=(
                f"Your submission did not satisfy the task requirements.{err_desc} "
                "If you have determined the question cannot be answered from "
                "the available schema, use the abstain tool instead of submit_sql."
            ),
        )

    # AMBIGUOUS: run pred + each valid candidate, pass if any result-match
    sol_sqls = td.get("sol_sql", []) or []
    if isinstance(sol_sqls, str):
        sol_sqls = [sol_sqls]
    if not sol_sqls:
        return SubmitSQLResponse(
            passed=False, reward=0.0, phase_completed=None, has_follow_up=False,

            message="PRACTIQ: no reference SQLs available (corrupt task)",
        )

    try:
        pred_res, pred_err, pred_to, _ = sqlite_utils.execute_queries(req_sql, task_db)
    except Exception as e:
        return SubmitSQLResponse(
            passed=False, reward=0.0, phase_completed=None, has_follow_up=False,

            message=f"SQL failed to execute: {e}",
        )
    if pred_err:
        return SubmitSQLResponse(
            passed=False, reward=0.0, phase_completed=None, has_follow_up=False,

            message=f"Submitted SQL error: {pred_err}",
        )
    if pred_to:
        return SubmitSQLResponse(
            passed=False, reward=0.0, phase_completed=None, has_follow_up=False,

            message="Submitted SQL timed out",
        )

    def _norm(rows):
        if not rows: return []
        try: return sorted([tuple(r) for r in rows])
        except Exception: return rows

    pred_n = _norm(pred_res)
    for sql in sol_sqls:
        try:
            gold_res, gold_err, _, _ = sqlite_utils.execute_queries(sql, task_db)
            if gold_err or gold_res is None: continue
            if _norm(gold_res) == pred_n:
                td["_current_phase"] = 2
                td["phase1_completed"] = True
                td["task_done"] = True
                return SubmitSQLResponse(
                    passed=True, reward=0.7, phase_completed=1, has_follow_up=False,
                    message=f"Phase 1 correct! (PRACTIQ {category}: matched valid interpretation) Reward: 0.7. Task finished.",
                )
        except Exception:
            continue

    # No match
    return SubmitSQLResponse(
        passed=False, reward=0.0, phase_completed=None, has_follow_up=False,

        message=(
            f"PRACTIQ {category}: submitted SQL did not match any of "
            f"{len(sol_sqls)} valid interpretations. Ask the user for "
            "clarification before submitting."
        ),
    )


def _bird_sqlite_submit_sync(req_task_id, req_sql, td, _submit_attempts, _successful_phase1_sql) -> SubmitSQLResponse:
    """BIRD-Interact-style submit on a SQLite backend.

    Mirrors `_submit_sql_sync` PG logic but uses sqlite_utils.* + sqlite3.
    BIRD-Interact-on-SQLite tasks have no follow_up → Phase 1 only.
    Reset / submit / test_cases scoring all go through sqlite_utils.
    """
    import shutil
    import sqlite3 as _sqlite3

    base_db = td["selected_database"]
    task_db = td.get("_task_db", base_db)
    current_phase = td.get("_current_phase", 1)

    if req_task_id not in _submit_attempts:
        _submit_attempts[req_task_id] = {1: 0, 2: 0}
    _submit_attempts[req_task_id][current_phase] = _submit_attempts[req_task_id].get(current_phase, 0) + 1
    is_first_try = _submit_attempts[req_task_id][current_phase] == 1

    interact_mode = td.get("_interact_mode", "a-interact")
    phase_rewards_first = {1: 0.7, 2: 0.3}
    phase_rewards_debug = {1: 0.5, 2: 0.2}

    try:
        # Reset task DB for clean evaluation. For phase 2, restore from snapshot.
        if current_phase == 1:
            sqlite_utils.reset_task_db(task_db, base_db)
        else:
            snapshot = td.get("_snapshot_db")
            if snapshot:
                # Snapshot is a sibling sqlite file — copy it back over task_db
                src_p = sqlite_utils.task_db_path(snapshot)
                dst_p = sqlite_utils.task_db_path(task_db)
                if src_p.is_file():
                    shutil.copyfile(str(src_p), str(dst_p))
                else:
                    sqlite_utils.reset_task_db(task_db, base_db)

        if current_phase == 2 and td.get("follow_up"):
            fu = td["follow_up"]
            sol_sqls = fu.get("sol_sql", [])
            test_cases = fu.get("test_cases", [])
            conditions = fu.get("conditions", {})
            category = fu.get("category", "Query")
        else:
            sol_sqls = td.get("sol_sql", [])
            test_cases = td.get("test_cases", [])
            conditions = td.get("conditions", {})
            category = td.get("category", "Query")

        if isinstance(sol_sqls, str):
            sol_sqls = [sol_sqls]
        pred_sqls = [req_sql] if isinstance(req_sql, str) else req_sql

        passed = False
        message = "Test case execution failed."

        if sol_sqls:
            db_path = sqlite_utils.task_db_path(task_db)
            conn = _sqlite3.connect(str(db_path), timeout=60.0)
            try:
                pred_query_result, pred_err, pred_to, _ = sqlite_utils.execute_queries(
                    pred_sqls, task_db, conn
                )
                if pred_err:
                    message = f"[exec_err_flg] Error executing submitted SQL: {pred_err}"
                elif pred_to:
                    message = "[exec_err_flg] Submitted SQL execution timed out"
                elif category == "Query" or not test_cases:
                    # Default: result-set equality (mirrors ex_base)
                    sol_result, sol_err, sol_to, sol_desc = sqlite_utils.execute_queries(
                        sol_sqls, task_db, conn
                    )
                    if sol_err or sol_to or sol_result is None:
                        message = f"Reference SQL execution failed: {sol_err or 'timeout'}"
                    else:
                        pp = preprocess_results(pred_query_result) if pred_query_result else []
                        gp = preprocess_results(sol_result)
                        ordered = bool(conditions and conditions.get("order", False))
                        match = (pp == gp) if ordered else (set(pp) == set(gp) and len(pp) == len(gp))
                        if match:
                            passed = True
                            message = "SQL passed test case."
                        else:
                            sol_rows = len(sol_result)
                            sol_cols = len(sol_result[0]) if sol_result else 0
                            pred_rows = len(pred_query_result) if pred_query_result else 0
                            pred_cols = len(pred_query_result[0]) if pred_query_result and pred_query_result[0] else 0
                            if not PASCAL_NO_SHAPE_HINT:
                                hint = (f" Expected result shape: {sol_rows} rows × {sol_cols} columns."
                                        f" Your result shape: {pred_rows} rows × {pred_cols} columns.")
                                if (
                                    not PASCAL_NO_VALUE_DIFF
                                    and sol_rows == pred_rows
                                    and sol_cols == pred_cols
                                    and pred_query_result
                                ):
                                    col_names = [d[0] for d in sol_desc] if sol_desc else None
                                    vdiff = compute_value_diff(
                                        pp, gp, column_names=col_names,
                                        max_sample=3, ordered=ordered,
                                    )
                                    if vdiff:
                                        hint += f"\nValue diff:\n{vdiff}"
                            else:
                                hint = ""
                            message = _apply_submit_discipline(req_task_id, current_phase, req_sql, f"Your SQL is not correct.{hint}")
                else:
                    # Custom test_cases (Management category) — exec each tc_code
                    def _execute_queries_compat(queries, db_name, conn=None):
                        result, error, timeout, _ = sqlite_utils.execute_queries(queries, db_name, conn)
                        return result, error, timeout

                    def _ex_base_sqlite(pred, sol, db, c=None, cond=None):
                        if not pred or not sol:
                            return 0
                        pr, pe, pt, _ = sqlite_utils.execute_queries(pred, db, c)
                        gr, ge, gt, _ = sqlite_utils.execute_queries(sol, db, c)
                        if pe or pt or ge or gt:
                            return 0
                        pr_n = preprocess_results(pr) if pr else []
                        gr_n = preprocess_results(gr) if gr else []
                        if not pr_n or not gr_n:
                            return 0
                        if cond and cond.get("order", False):
                            return 1 if pr_n == gr_n else 0
                        return 1 if (set(pr_n) == set(gr_n) and len(pr_n) == len(gr_n)) else 0

                    exec_globals = {
                        "execute_queries": _execute_queries_compat,
                        "ex_base": _ex_base_sqlite,
                        "remove_distinct": remove_distinct,
                        "remove_comments": remove_comments,
                        "remove_round": remove_round,
                        "pred_query_result": pred_query_result,
                    }
                    all_passed = True
                    for tc_code in test_cases:
                        if not isinstance(tc_code, str):
                            continue
                        try:
                            exec_locals = {}
                            exec(tc_code, exec_globals, exec_locals)
                            tc_func = exec_locals.get("test_case")
                            if tc_func and callable(tc_func):
                                tc_func(pred_sqls, sol_sqls, task_db, conn)
                        except (AssertionError, Exception):
                            all_passed = False
                            hint = ""
                            try:
                                sol_result, sol_err, sol_to, sol_desc = sqlite_utils.execute_queries(
                                    sol_sqls, task_db, conn
                                )
                                if sol_result and not sol_err:
                                    sol_rows = len(sol_result)
                                    sol_cols = len(sol_result[0]) if sol_result else 0
                                    pred_rows = len(pred_query_result) if pred_query_result else 0
                                    pred_cols = len(pred_query_result[0]) if pred_query_result and pred_query_result[0] else 0
                                    if not PASCAL_NO_SHAPE_HINT:
                                        hint = (f" Expected result shape: {sol_rows} rows × {sol_cols} columns."
                                                f" Your result shape: {pred_rows} rows × {pred_cols} columns.")
                                        if (
                                            not PASCAL_NO_VALUE_DIFF
                                            and sol_rows == pred_rows
                                            and sol_cols == pred_cols
                                            and pred_query_result
                                        ):
                                            col_names = [dd[0] for dd in sol_desc] if sol_desc else None
                                            pp = preprocess_results(pred_query_result)
                                            gp = preprocess_results(sol_result)
                                            ordered = bool(conditions and conditions.get("order", False))
                                            vdiff = compute_value_diff(
                                                pp, gp, column_names=col_names,
                                                max_sample=3, ordered=ordered,
                                            )
                                            if vdiff:
                                                hint += f"\nValue diff:\n{vdiff}"
                            except Exception:
                                pass
                            message = _apply_submit_discipline(req_task_id, current_phase, req_sql, f"Your SQL is not correct.{hint}")
                            break
                    if all_passed:
                        passed = True
                        message = "SQL passed all test cases."
            finally:
                try: conn.close()
                except Exception: pass

        if passed:
            if interact_mode == "c-interact":
                reward = phase_rewards_first.get(current_phase, 0) if is_first_try else phase_rewards_debug.get(current_phase, 0)
            else:
                reward = phase_rewards_first.get(current_phase, 0)
            if current_phase == 1:
                _successful_phase1_sql[req_task_id] = req_sql
                # Apply Phase 1 SQL on a fresh task DB and snapshot for Phase 2
                sqlite_utils.reset_task_db(task_db, base_db)
                conn2 = _sqlite3.connect(str(sqlite_utils.task_db_path(task_db)), timeout=60.0)
                try:
                    sqlite_utils.execute_queries(pred_sqls, task_db, conn2)
                finally:
                    try: conn2.close()
                    except Exception: pass

                has_follow_up = bool(td.get("follow_up") and td["follow_up"].get("sol_sql"))
                if has_follow_up:
                    snapshot_db = f"{task_db}__p1snap"
                    shutil.copyfile(
                        str(sqlite_utils.task_db_path(task_db)),
                        str(sqlite_utils.task_db_path(snapshot_db)),
                    )
                    td["_snapshot_db"] = snapshot_db
                    _submit_attempts[req_task_id][2] = 0
                    td["_current_phase"] = 2
                    follow_up_query = td["follow_up"].get("query", "Please complete the follow-up task.")
                    return SubmitSQLResponse(
                        passed=True, message=f"Phase 1 correct! (Reward: {reward}). Moving to Phase 2.",
                        reward=reward, phase_completed=1, has_follow_up=True,
                        follow_up_query=follow_up_query)
                else:
                    return SubmitSQLResponse(
                        passed=True, message=f"Phase 1 correct! (Reward: {reward}). Task finished.",
                        reward=reward, phase_completed=1, has_follow_up=False)
            else:
                return SubmitSQLResponse(
                    passed=True, message=f"Phase 2 correct! (Reward: {reward}). Task finished.",
                    reward=reward, phase_completed=2, has_follow_up=False)
        else:
            # Failed: reset task DB for retry exploration
            if current_phase == 1:
                sqlite_utils.reset_task_db(task_db, base_db)
            else:
                snapshot = td.get("_snapshot_db")
                if snapshot:
                    src_p = sqlite_utils.task_db_path(snapshot)
                    if src_p.is_file():
                        shutil.copyfile(str(src_p), str(sqlite_utils.task_db_path(task_db)))
            return SubmitSQLResponse(
                passed=False, message=f"SQL failed Phase {current_phase}. {message}",
                reward=0.0)
    except Exception as e:
        logger.error(f"Bird-sqlite submit error for {req_task_id}: {e}", exc_info=True)
        return SubmitSQLResponse(passed=False, message=f"Error: {e}", reward=0.0)


def _submit_sql_sync(req_task_id, req_sql, td, _submit_attempts, _successful_phase1_sql) -> SubmitSQLResponse:
    """Blocking submit logic — runs in thread pool."""
    # PRACTIQ path (SQLite backend + match-any-valid / abstain scoring)
    if td.get("_practiq_meta"):
        return _practiq_submit_sync(req_task_id, req_sql, td, _submit_attempts)

    # BIRD-Interact-on-SQLite path (same task structure as BIRD-Interact
    # PG but stored as Spider-style sqlite per-DB files)
    if td.get("_backend") == "sqlite":
        return _bird_sqlite_submit_sync(
            req_task_id, req_sql, td, _submit_attempts, _successful_phase1_sql
        )

    base_db = td["selected_database"]
    task_db = td.get("_task_db", base_db)
    current_phase = td.get("_current_phase", 1)

    if req_task_id not in _submit_attempts:
        _submit_attempts[req_task_id] = {1: 0, 2: 0}
    _submit_attempts[req_task_id][current_phase] = _submit_attempts[req_task_id].get(current_phase, 0) + 1
    is_first_try = _submit_attempts[req_task_id][current_phase] == 1

    interact_mode = td.get("_interact_mode", "a-interact")
    phase_rewards_first = {1: 0.7, 2: 0.3}
    phase_rewards_debug = {1: 0.5, 2: 0.2}

    try:
        # Reset task DB for clean evaluation
        if current_phase == 1:
            template = f"{base_db}_template"
        else:
            # Phase 2: reset from Phase 1 snapshot (has Phase 1 state applied)
            template = td.get("_snapshot_db", f"{base_db}_template")
        reset_task_db(task_db, template)

        pool = _get_or_init_pool(task_db)
        conn = pool.getconn()
        try:
            if current_phase == 2 and td.get("follow_up"):
                fu = td["follow_up"]
                sol_sqls = fu.get("sol_sql", [])
                test_cases = fu.get("test_cases", [])
                conditions = fu.get("conditions", {})
                category = fu.get("category", "Query")
            else:
                sol_sqls = td.get("sol_sql", [])
                test_cases = td.get("test_cases", [])
                conditions = td.get("conditions", {})
                category = td.get("category", "Query")

            if isinstance(sol_sqls, str): sol_sqls = [sol_sqls]
            pred_sqls = [req_sql] if isinstance(req_sql, str) else req_sql

            passed = False
            message = "Test case execution failed."

            if sol_sqls:
                # Execute pred SQL (also serves as executability check)
                pred_query_result, pred_err, pred_to, _ = execute_queries(pred_sqls, task_db, conn)
                if pred_err:
                    message = f"[exec_err_flg] Error executing submitted SQL: {pred_err}"
                elif pred_to:
                    message = "[exec_err_flg] Submitted SQL execution timed out"
                elif category == "Query" or not test_cases:
                    try:
                        test_case_default(pred_sqls, sol_sqls, task_db, conn, conditions)
                        passed = True
                        message = "SQL passed test case."
                    except (AssertionError, Exception):
                        # Provide shape + value diff hints for diagnosis
                        hint = ""
                        try:
                            sol_result, sol_err, sol_to, sol_desc = execute_queries(sol_sqls, task_db, conn)
                            if sol_result and not sol_err:
                                sol_rows = len(sol_result)
                                sol_cols = len(sol_result[0]) if sol_result else 0
                                pred_rows = len(pred_query_result) if pred_query_result else 0
                                pred_cols = len(pred_query_result[0]) if pred_query_result and pred_query_result[0] else 0
                                if not PASCAL_NO_SHAPE_HINT:
                                    hint = (f" Expected result shape: {sol_rows} rows × {sol_cols} columns."
                                            f" Your result shape: {pred_rows} rows × {pred_cols} columns.")
                                    # Value diff when shapes match
                                    if (
                                        not PASCAL_NO_VALUE_DIFF
                                        and sol_rows == pred_rows
                                        and sol_cols == pred_cols
                                        and pred_query_result
                                    ):
                                        col_names = [d[0] for d in sol_desc] if sol_desc else None
                                        pp = preprocess_results(pred_query_result)
                                        gp = preprocess_results(sol_result)
                                        ordered = bool(conditions and conditions.get("order", False))
                                        vdiff = compute_value_diff(pp, gp, column_names=col_names, max_sample=3, ordered=ordered)
                                        if vdiff:
                                            hint += f"\nValue diff:\n{vdiff}"
                        except Exception:
                            pass
                        message = _apply_submit_discipline(req_task_id, current_phase, req_sql, f"Your SQL is not correct.{hint}")
                else:
                    # Compat wrapper: custom test cases expect 3-value return (result, error, timeout)
                    def _execute_queries_compat(queries, db_name, conn=None):
                        result, error, timeout, _ = execute_queries(queries, db_name, conn)
                        return result, error, timeout

                    exec_globals = {
                        "execute_queries": _execute_queries_compat, "ex_base": ex_base,
                        "remove_distinct": remove_distinct, "remove_comments": remove_comments,
                        "remove_round": remove_round,
                        "pred_query_result": pred_query_result,
                    }
                    all_passed = True
                    for i, tc_code in enumerate(test_cases):
                        if not isinstance(tc_code, str): continue
                        try:
                            exec_locals = {}
                            exec(tc_code, exec_globals, exec_locals)
                            tc_func = exec_locals.get("test_case")
                            if tc_func and callable(tc_func):
                                tc_func(pred_sqls, sol_sqls, task_db, conn)
                        except (AssertionError, Exception):
                            all_passed = False
                            # Add shape + value diff hints for Management tasks too
                            hint = ""
                            try:
                                sol_result, sol_err, sol_to, sol_desc = execute_queries(sol_sqls, task_db, conn)
                                if sol_result and not sol_err:
                                    sol_rows = len(sol_result)
                                    sol_cols = len(sol_result[0]) if sol_result else 0
                                    pred_rows = len(pred_query_result) if pred_query_result else 0
                                    pred_cols = len(pred_query_result[0]) if pred_query_result and pred_query_result[0] else 0
                                    if not PASCAL_NO_SHAPE_HINT:
                                        hint = (f" Expected result shape: {sol_rows} rows × {sol_cols} columns."
                                                f" Your result shape: {pred_rows} rows × {pred_cols} columns.")
                                        if (
                                            not PASCAL_NO_VALUE_DIFF
                                            and sol_rows == pred_rows
                                            and sol_cols == pred_cols
                                            and pred_query_result
                                        ):
                                            col_names = [dd[0] for dd in sol_desc] if sol_desc else None
                                            pp = preprocess_results(pred_query_result)
                                            gp = preprocess_results(sol_result)
                                            ordered = bool(conditions and conditions.get("order", False))
                                            vdiff = compute_value_diff(pp, gp, column_names=col_names, max_sample=3, ordered=ordered)
                                            if vdiff:
                                                hint += f"\nValue diff:\n{vdiff}"
                            except Exception:
                                pass
                            message = _apply_submit_discipline(req_task_id, current_phase, req_sql, f"Your SQL is not correct.{hint}")
                            break
                    if all_passed:
                        passed = True
                        message = "SQL passed all test cases."

            pool.putconn(conn)

            if passed:
                if interact_mode == "c-interact":
                    reward = phase_rewards_first.get(current_phase, 0) if is_first_try else phase_rewards_debug.get(current_phase, 0)
                else:
                    reward = phase_rewards_first.get(current_phase, 0)
                if current_phase == 1:
                    _successful_phase1_sql[req_task_id] = req_sql
                    # Apply Phase 1 SQL and snapshot for Phase 2
                    reset_task_db(task_db, f"{base_db}_template")
                    p_pool = _get_or_init_pool(task_db)
                    p_conn = p_pool.getconn()
                    try:
                        execute_queries(pred_sqls, task_db, p_conn)
                    finally:
                        p_pool.putconn(p_conn)
                    close_pool(task_db)  # must close connections before using as template
                    snapshot_db = create_task_db(task_db, "p1snap", template=task_db)
                    td["_snapshot_db"] = snapshot_db

                    has_follow_up = bool(td.get("follow_up") and td["follow_up"].get("sol_sql"))
                    if has_follow_up:
                        _submit_attempts[req_task_id][2] = 0
                        td["_current_phase"] = 2
                        follow_up_query = td["follow_up"].get("query", "Please complete the follow-up task.")
                        return SubmitSQLResponse(
                            passed=True, message=f"Phase 1 correct! (Reward: {reward}). Moving to Phase 2.",
                            reward=reward, phase_completed=1, has_follow_up=True,
                            follow_up_query=follow_up_query)
                    else:
                        return SubmitSQLResponse(
                            passed=True, message=f"Phase 1 correct! (Reward: {reward}). Task finished.",
                            reward=reward, phase_completed=1, has_follow_up=False)
                else:
                    return SubmitSQLResponse(
                        passed=True, message=f"Phase 2 correct! (Reward: {reward}). Task finished.",
                        reward=reward, phase_completed=2, has_follow_up=False)
            else:
                # Failed: restore task DB to pre-submit state for agent exploration
                if current_phase == 1:
                    reset_task_db(task_db, f"{base_db}_template")
                else:
                    snapshot = td.get("_snapshot_db")
                    if snapshot:
                        reset_task_db(task_db, snapshot)

                return SubmitSQLResponse(
                    passed=False, message=f"SQL failed Phase {current_phase}. {message}",
                    reward=0.0)
        except Exception as inner_e:
            try: pool.putconn(conn)
            except: pass
            raise inner_e
    except Exception as e:
        logger.error(f"Submit error for {req_task_id}: {e}", exc_info=True)
        return SubmitSQLResponse(passed=False, message=f"Error: {e}", reward=0.0)


@app.post("/submit", response_model=SubmitSQLResponse)
async def submit_sql_endpoint(req: SubmitSQLRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    return await asyncio.to_thread(
        _submit_sql_sync, req.task_id, req.sql, td, _submit_attempts, _successful_phase1_sql
    )


@app.post("/schema")
async def get_schema(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    # PRACTIQ + sqlite: pull DDL from the *task* DB (post-schema-mod) so the
    # agent sees the modified schema (removed/added cols).
    if td.get("_backend") == "sqlite" and td.get("_schema_mod_summary"):
        task_db = td.get("_task_db")
        import sqlite3
        try:
            conn = sqlite3.connect(str(sqlite_utils.task_db_path(task_db)))
            rows = conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
                "ORDER BY type DESC, name"
            ).fetchall()
            conn.close()
            return {"schema": ";\n\n".join(r[0] for r in rows) + ";\n"}
        except Exception as e:
            logger.error(f"post-mod schema load failed: {e}")
    _load_db_data(db_name, use_sqlite=(td.get("_backend") == "sqlite"))
    return {"schema": _schema_cache.get(db_name, "Schema not available")}


@app.post("/explore_tables")
async def explore_tables(req: SchemaRequest):
    """Backend-aware tables listing.

    On PG: row counts + columns from information_schema.
    On SQLite: tables + columns from sqlite_master and PRAGMA table_info.
    """
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    if td.get("_backend") == "sqlite":
        import sqlite3
        task_db = td.get("_task_db")
        try:
            conn = sqlite3.connect(str(sqlite_utils.task_db_path(task_db)))
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()]
            lines = ["table_name | row_count | columns",
                     "--" * 20]
            for t in tables:
                try:
                    n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                except Exception:
                    n = "?"
                cols = []
                for row in conn.execute(f'PRAGMA table_info("{t}")').fetchall():
                    # row: (cid, name, type, notnull, dflt_value, pk)
                    cols.append(f"{row[1]} ({row[2]})")
                lines.append(f"{t} | {n} | {', '.join(cols)}")
            conn.close()
            return {"result": "\n".join(lines)}
        except Exception as e:
            return {"result": f"Error listing tables: {e}"}
    sql = """
    SELECT t.table_name,
           (xpath('/row/cnt/text()',
                  query_to_xml(format('SELECT COUNT(*) AS cnt FROM %I', t.table_name), false, true, ''))
           )[1]::text::int AS row_count,
           string_agg(c.column_name || ' (' || c.data_type || ')', ', ' ORDER BY c.ordinal_position) AS columns
    FROM information_schema.tables t
    JOIN information_schema.columns c ON t.table_name = c.table_name AND c.table_schema = 'public'
    WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
    GROUP BY t.table_name ORDER BY t.table_name
    """
    resp = await execute_sql_endpoint(ExecuteSQLRequest(task_id=req.task_id, sql=sql))
    return {"result": resp.result if hasattr(resp, "result") else str(resp)}


@app.post("/list_foreign_keys")
async def list_foreign_keys(req: SchemaRequest):
    """Backend-aware foreign-key listing.

    On PG: information_schema.table_constraints / key_column_usage.
    On SQLite: PRAGMA foreign_key_list(table) for each table.
    """
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    if td.get("_backend") == "sqlite":
        import sqlite3
        task_db = td.get("_task_db")
        try:
            conn = sqlite3.connect(str(sqlite_utils.task_db_path(task_db)))
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()]
            lines = ["from_table | from_column | to_table | to_column",
                     "--" * 20]
            for t in tables:
                for fk in conn.execute(f'PRAGMA foreign_key_list("{t}")').fetchall():
                    # fk: (id, seq, table, from, to, on_update, on_delete, match)
                    lines.append(f"{t} | {fk[3]} | {fk[2]} | {fk[4]}")
            conn.close()
            return {"result": "\n".join(lines) if len(lines) > 2 else "No foreign keys found."}
        except Exception as e:
            return {"result": f"Error listing foreign keys: {e}"}
    sql = """
    SELECT
        tc.table_name AS from_table,
        kcu.column_name AS from_column,
        ccu.table_name AS to_table,
        ccu.column_name AS to_column
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage ccu
        ON tc.constraint_name = ccu.constraint_name AND tc.table_schema = ccu.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
    ORDER BY tc.table_name, kcu.column_name
    """
    resp = await execute_sql_endpoint(ExecuteSQLRequest(task_id=req.task_id, sql=sql))
    return {"result": resp.result if hasattr(resp, "result") else str(resp)}


@app.post("/all_column_meanings")
async def get_all_column_meanings(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name, use_sqlite=(td.get("_backend") == "sqlite"))
    return {"column_meanings": json.dumps(_column_meanings_cache.get(db_name, {}), indent=2)}


@app.post("/column_meaning")
async def get_column_meaning(req: ColumnMeaningRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name, use_sqlite=(td.get("_backend") == "sqlite"))
    key = f"{db_name}|{req.table_name.lower()}|{req.column_name.lower()}"
    meaning = _column_meanings_cache.get(db_name, {}).get(key, "Column meaning not found")
    out = meaning if isinstance(meaning, str) else json.dumps(meaning)
    if meaning != "Column meaning not found":
        _record_fact(req.task_id, f"col[{req.table_name}.{req.column_name}]: {out}")
    return {"meaning": out}


@app.post("/knowledge_names")
async def get_knowledge_names(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name, use_sqlite=(td.get("_backend") == "sqlite"))
    agent_kb = _filter_knowledge(db_name, td)
    return {"names": list(agent_kb.keys())}


def _fuzzy_kb_lookup(query: str, kb_dict: dict) -> Optional[dict]:
    """3-tier fuzzy KB lookup: exact → abbreviation → token match.

    Uses rapidfuzz for token-level fuzzy matching (similar to
    ElasticSearch's match query). No GPU required.
    """
    # Tier 1: exact match
    if query in kb_dict:
        return kb_dict[query]

    kb_names = list(kb_dict.keys())

    # Tier 2: case-insensitive + abbreviation in parentheses
    q_lower = query.strip().lower()
    for name in kb_names:
        if q_lower == name.lower():
            return kb_dict[name]
        # Match abbreviation: "ESI" → "Environmental Suitability Index (ESI)"
        if "(" in name and ")" in name:
            abbrev = name[name.rfind("(") + 1 : name.rfind(")")]
            if q_lower == abbrev.lower():
                return kb_dict[name]

    # Tier 3 (rapidfuzz) removed — it produced harmful false positives.
    # All remaining NOT FOUND cases in easy_30 are entries deliberately
    # deleted by the benchmark (deleted_knowledge in knowledge_ambiguity).
    # The agent should use ask_user to obtain these definitions instead.

    return None


@app.post("/knowledge")
async def get_knowledge(req: KnowledgeRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name, use_sqlite=(td.get("_backend") == "sqlite"))
    agent_kb = _filter_knowledge(db_name, td)
    if req.knowledge_name:
        entry = _fuzzy_kb_lookup(req.knowledge_name, agent_kb)
        if entry:
            visible = {k: entry[k] for k in KNOWLEDGE_VISIBLE_FIELDS if k in entry}
            out = json.dumps(visible, indent=2)
            # Compact one-line fact for the ledger: prefer definition over description.
            _fact = entry.get("definition") or entry.get("description") or ""
            if _fact:
                _record_fact(req.task_id, f"KB[{req.knowledge_name}]: {_fact}")
            return {"knowledge": out}
        # NOT-FOUND lookups are intentionally not recorded in the ledger:
        # an authoritative-looking empty entry would displace useful
        # answer-binding facts in the latest-N window.
        return {"knowledge": "Knowledge not found. This definition may be intentionally hidden — use ask_user to request the formula or definition from the user."}
    else:
        visible_kbs = []
        for e in agent_kb.values():
            visible_kbs.append({k: e[k] for k in KNOWLEDGE_VISIBLE_FIELDS if k in e})
        return {"knowledge": json.dumps(visible_kbs, indent=2)}


def _cleanup_task_sync(task_db, snapshot_db, backend: str = "pg"):
    """Blocking cleanup — runs in thread pool."""
    if backend == "sqlite":
        if snapshot_db:
            sqlite_utils.drop_task_db(snapshot_db)
        if task_db:
            sqlite_utils.drop_task_db(task_db)
        return
    if snapshot_db:
        drop_task_db(snapshot_db)
    if task_db:
        drop_task_db(task_db)


class AbstainRequest(BaseModel):
    task_id: str
    reason: str = ""


try:
    from shared.models import AbstainRequest as _AR  # if someone added to models later
    AbstainRequest = _AR  # type: ignore
except Exception:
    pass


@app.post("/abstain", response_model=SubmitSQLResponse)
async def abstain_endpoint(req: "AbstainRequest"):
    td = _task_data.get(req.task_id)
    if not td:
        raise HTTPException(404, f"Task {req.task_id} not initialized")
    meta = td.get("_practiq_meta", {}) or {}
    expected = meta.get("expected_behaviour", "")
    category = meta.get("category", "")
    # Only PRACTIQ-unanswerable tasks allow this path
    if expected == "abstain":
        td["_current_phase"] = 2
        td["phase1_completed"] = True
        td["task_done"] = True
        return SubmitSQLResponse(
            passed=True, reward=0.7, phase_completed=1, has_follow_up=False,
            message=f"Phase 1 correct! (PRACTIQ {category}: abstained) Reward: 0.7. Task finished.",
        )
    return SubmitSQLResponse(
        passed=False, reward=0.0, phase_completed=None, has_follow_up=False,
        message=(
            "Abstention declined — this task expects a SQL answer. "
            "Use submit_sql with a SELECT / WITH statement instead."
        ),
    )


@app.post("/cleanup_task")
async def cleanup_task(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td:
        return {"status": "ok", "task_id": req.task_id}
    task_db = td.get("_task_db")
    snapshot_db = td.get("_snapshot_db")
    backend = td.get("_backend", "pg")
    try:
        await asyncio.to_thread(_cleanup_task_sync, task_db, snapshot_db, backend)
    except Exception as e:
        logger.warning(f"Cleanup failed for {req.task_id}: {e}")
    _task_data.pop(req.task_id, None)
    _submit_attempts.pop(req.task_id, None)
    _successful_phase1_sql.pop(req.task_id, None)
    _submit_sql_history.pop(req.task_id, None)
    _evidence_ledger.pop(req.task_id, None)
    return {"status": "ok", "task_id": req.task_id}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "db_environment"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.db_env_port)
