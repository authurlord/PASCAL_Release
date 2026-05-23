"""BIRD-Interact ADK Orchestrator - a-interact agent pipeline.

This version delegates the full tool-use loop to the ADK-backed system-agent
service on port 6000. The orchestrator only:
1. Initializes the DB environment and user simulator services
2. Initializes an agent session on the system-agent service
3. Sends the initial user request once
4. Reads the final session state for metrics
"""

import argparse
import asyncio
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import httpx
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import settings
from shared.kb_preloader import load_knowledge_base, preload_task_knowledge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load KB once at module level (lazy init)
_kb_cache = {}

# Mode toggles (read at module import — set via env before launching runner).
#
#   PASCAL_NO_PROTOCOL=1   official ReACT baseline (schema + KB pre-injection
#                          stripped; agent uses minimal prompt + 9-tool surface
#                          minus KB tools).
#   PASCAL_KB_INJECTION=1  pre-inject the full per-DB KB into the initial
#                          user message.  Off by default; the PASCAL anchor
#                          relies on the agent's on-demand KB retrieval tools.
PASCAL_NO_PROTOCOL = os.environ.get("PASCAL_NO_PROTOCOL", "0") == "1"
PASCAL_KB_INJECTION = os.environ.get("PASCAL_KB_INJECTION", "0") == "1"
# Derived: ReACT baseline strips schema + KB pre-injection.
PASCAL_ABLATE_SCHEMA = PASCAL_NO_PROTOCOL
PASCAL_ABLATE_KB = PASCAL_NO_PROTOCOL
if PASCAL_NO_PROTOCOL:
    logger.warning(
        "AGENT MODE: official ReACT baseline (PASCAL_NO_PROTOCOL=1, "
        "schema + KB pre-injection disabled)"
    )

SYSTEM_AGENT_URL = f"http://localhost:{settings.system_agent_port}"
USER_SIM_URL = f"http://localhost:{settings.user_sim_port}"
DB_ENV_URL = f"http://localhost:{settings.db_env_port}"


async def _post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def _meta_roots(suffix: str) -> list:
    """Return ordered list of candidate metadata directories for `suffix`.

    Honors `DB_METADATA_ROOT` env override, then falls back to the
    in-tree hardcoded paths the anchor probes (so original behavior is
    preserved when the env var is unset).
    """
    roots = []
    env_root = os.environ.get("DB_METADATA_ROOT")
    if env_root:
        roots.append(Path(env_root) / suffix)
    roots.extend([
        Path(__file__).parent.parent.parent / "bird-interact-full-hf-meta" / suffix,
        Path(__file__).parent.parent.parent / "bird-interact-lite-hf-meta" / suffix,
        Path(__file__).parent.parent / "bird-interact-full-hf-meta" / suffix,
        Path(__file__).parent.parent / "bird-interact-lite-hf-meta" / suffix,
    ])
    return roots


def calculate_initial_budget(task_data: Dict[str, Any]) -> float:
    """a-interact budget in bird-coins (per task, paper Section 3.2).

    Stress Mode (default): B = 6 + 2 * m_amb + 2 * patience
      - 6 = ENV_INTERACT(3) + SUBMIT(3) base budget
      - 2 * m_amb = one ask_user (cost=2) per ambiguity point
      - 2 * patience = extra exploration tolerance
      - patience=3 in config = patience_budget=6 in reference

    Free Mode (PASCAL_FREE_MODE=1): B = 999. Effectively unbounded — agent
    self-regulates via 100-turn cap and natural termination. Used to measure
    leaderboard-comparable Efficiency (BIRD COIN naturally consumed).
    """
    if os.environ.get("PASCAL_FREE_MODE") == "1":
        return 999.0
    critical = len(task_data.get("user_query_ambiguity", {}).get("critical_ambiguity", []))
    knowledge = len(task_data.get("knowledge_ambiguity", []))
    m_amb = critical + knowledge
    return 6.0 + 2.0 * m_amb + 2.0 * settings.patience


async def init_task_on_services(task_id: str, task_data: dict):
    payload = {
        "task_id": task_id,
        "task_data": {**task_data, "_interact_mode": "a-interact"},
    }
    await _post(f"{DB_ENV_URL}/init_task", payload)
    await _post(f"{USER_SIM_URL}/init_task", payload)
    logger.info("  [%s] Services initialized", task_id)


async def init_agent_session(task_id: str, task_data: dict, budget: float):
    state = {
        "task_id": task_id,
        "db_name": task_data["selected_database"],
        "user_query": task_data.get("amb_user_query", ""),
        "current_phase": 1,
        "budget_remaining": budget,
        "initial_budget": budget,
        "total_reward": 0.0,
        "dialogue_history": [],
        "tool_trajectory": [],
        "adk_events": [],
        "phase1_completed": False,
        "phase2_completed": False,
        "task_done": False,
    }
    # Expose PRACTIQ metadata to callbacks so the turn-count nudge can gate on
    # it without leaking into BIRD runs. Only set when the task carries the
    # marker; otherwise state is identical to the BIRD path.
    practiq_meta = task_data.get("_practiq_meta")
    if practiq_meta:
        state["_practiq_meta"] = practiq_meta
    # Raised from 30s to 120s: user_simulator's init path calls Gemini,
    # which can 503/retry up to ~60s under bursty launches.
    last_exc = None
    for attempt in range(2):
        try:
            return await _post(
                f"{SYSTEM_AGENT_URL}/init_session",
                {"task_id": task_id, "mode": "a-interact", "state": state, "reset": True},
                timeout=120.0,
            )
        except Exception as e:
            last_exc = e
            if attempt == 0:
                logger.warning("init_session timed out for %s, retrying once: %s", task_id, e)
                await asyncio.sleep(2.0)
            else:
                raise
    raise last_exc  # unreachable


async def run_agent_session(task_id: str, message: str):
    # 120-minute wall-clock cap per session.  Long enough that slow runs
    # under KV-cache pressure can finish instead of being marked as
    # infrastructure errors.
    return await _post(
        f"{SYSTEM_AGENT_URL}/run_session",
        {"task_id": task_id, "mode": "a-interact", "message": message},
        timeout=7200.0,
    )


async def cleanup_task_service(task_id: str):
    try:
        await _post(f"{DB_ENV_URL}/cleanup_task", {"task_id": task_id}, timeout=30.0)
    except Exception as e:
        logger.warning("Cleanup failed for %s: %s", task_id, e)


async def run_single_task(task_data: dict) -> Dict[str, Any]:
    instance_id = task_data["instance_id"]
    db_name = task_data["selected_database"]
    logger.info("Starting task: %s (db: %s)", instance_id, db_name)
    start_time = time.time()

    await init_task_on_services(instance_id, task_data)

    try:
        initial_budget = calculate_initial_budget(task_data)
        await init_agent_session(instance_id, task_data, initial_budget)

        # Pre-load domain knowledge if available
        global _kb_cache
        if not _kb_cache:
            # Try to load KB from the task data's knowledge base
            kb_path = task_data.get("_kb_path", "")
            if not kb_path:
                # Try standard locations
                for p in [
                    Path(__file__).parent.parent / "bird-interact-lite-hf-meta" / "bird_interact_data.jsonl",
                    Path(__file__).parent.parent.parent / "bird-interact-lite-hf-meta" / "bird_interact_data.jsonl",
                ]:
                    if p.exists():
                        # KB is embedded in task data, not a separate file
                        break

        kb_section = ""
        ek = task_data.get("external_knowledge", [])
        if isinstance(ek, str):
            try:
                import json as _json
                ek = _json.loads(ek)
            except Exception:
                ek = []
        # ReACT baseline strips KB pre-injection entirely.
        if PASCAL_ABLATE_KB:
            ek = []
        # PASCAL_KB_INJECTION pre-loads the full per-DB KB into the initial
        # user message.  Default OFF — the PASCAL anchor relies on the
        # agent's on-demand KB-retrieval tools.
        if PASCAL_KB_INJECTION:
            db = task_data.get("selected_database", "")
            full_kb_entries = []
            for meta_root in _meta_roots(db):
                kb_file = meta_root / f"{db}_kb.jsonl"
                if kb_file.exists():
                    try:
                        import json as _json
                        with open(kb_file) as _f:
                            for _line in _f:
                                if not _line.strip():
                                    continue
                                full_kb_entries.append(_json.loads(_line))
                        logger.info("  [%s] PASCAL kb-injection: loaded %d entries from %s",
                                    instance_id, len(full_kb_entries), kb_file.name)
                    except Exception as _e:
                        logger.warning("  [%s] kb-injection read failed: %s",
                                       instance_id, _e)
                    break

            if full_kb_entries:
                kb_parts = [
                    "# Pre-loaded Domain Knowledge",
                    "(These entries are pre-loaded for fast reference. You "
                    "may still call get_knowledge_definition / "
                    "get_all_external_knowledge_names to cross-check or look "
                    "up additional terms.)",
                    f"\n## Per-database knowledge ({db})",
                ]
                for entry in full_kb_entries:
                    name = entry.get("knowledge", f"Entry {entry.get('id', '?')}")
                    desc = entry.get("description", "")
                    definition = entry.get("definition", "")
                    body = desc
                    if definition and definition not in desc:
                        body = f"{desc}\n\nDefinition: {definition}".strip()
                    kb_parts.append(f"\n### {name}\n{body}")
                kb_section = "\n".join(kb_parts) + "\n\n"
            ek = []  # skip the per-task path below
        if isinstance(ek, list) and ek:
            # Knowledge IDs exist — resolve from agent_kb in task data;
            # fall back to <db>_kb.jsonl under bird-interact-{lite,full}-hf-meta/<db>/
            # when the shipment does not populate agent_kb (full_600 case).
            agent_kb = task_data.get("agent_kb", {}) or {}
            if not agent_kb:
                # Build agent_kb from the per-db KB file.
                db = task_data.get("selected_database", "")
                for meta_root in _meta_roots(db):
                    kb_file = meta_root / f"{db}_kb.jsonl"
                    if kb_file.exists():
                        try:
                            import json as _json
                            with open(kb_file) as _f:
                                for _line in _f:
                                    if not _line.strip():
                                        continue
                                    _e = _json.loads(_line)
                                    if "id" in _e:
                                        agent_kb[str(_e["id"])] = _e
                            logger.info("  [%s] kb fallback loaded %d entries from %s",
                                        instance_id, len(agent_kb), kb_file.name)
                        except Exception as _e:
                            logger.warning("  [%s] kb fallback read failed: %s",
                                           instance_id, _e)
                        break
            if agent_kb:
                kb_parts = ["# Pre-loaded Domain Knowledge"]
                for kid in ek:
                    entry = agent_kb.get(str(kid))
                    if entry:
                        name = entry.get("knowledge", f"Entry {kid}")
                        desc = entry.get("description", "")
                        definition = entry.get("definition", "")
                        body = desc
                        if definition and definition not in desc:
                            body = f"{desc}\n\nDefinition: {definition}"
                        kb_parts.append(f"\n## {name}\n{body}")
                if len(kb_parts) > 1:
                    kb_section = "\n".join(kb_parts) + "\n\n"

        # Pre-inject schema: call explore_tables + list_foreign_keys via DB env
        schema_section = ""
        if not PASCAL_ABLATE_SCHEMA:
            try:
                schema_resp = await _post(f"{DB_ENV_URL}/execute", {"task_id": instance_id, "sql": """
                    SELECT t.table_name,
                           (xpath('/row/cnt/text()',
                                  query_to_xml(format('SELECT COUNT(*) AS cnt FROM %I', t.table_name), false, true, ''))
                           )[1]::text::int AS row_count,
                           string_agg(c.column_name || ' (' || c.data_type || ')', ', ' ORDER BY c.ordinal_position) AS columns
                    FROM information_schema.tables t
                    JOIN information_schema.columns c ON t.table_name = c.table_name AND c.table_schema = 'public'
                    WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
                    GROUP BY t.table_name ORDER BY t.table_name
                """}, timeout=30.0)
                fk_resp = await _post(f"{DB_ENV_URL}/execute", {"task_id": instance_id, "sql": """
                    SELECT tc.table_name || '.' || kcu.column_name || ' → ' ||
                           ccu.table_name || '.' || ccu.column_name AS fk
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                """}, timeout=30.0)
                schema_text = schema_resp.get("result", "")
                fk_text = fk_resp.get("result", "")
                if schema_text:
                    schema_section = f"# Database Schema (pre-loaded, no need to call explore_tables)\n{schema_text}\n"
                if fk_text:
                    schema_section += f"\n# Foreign Keys\n{fk_text}\n"
            except Exception as e:
                logger.warning("Schema pre-injection failed for %s: %s", instance_id, e)

        # PRACTIQ tasks also include an "abstain()" tool; note this in the
        # prompt so the agent considers un-realizability alongside the
        # normal clarify/submit path. Also state the hard turn cap so the
        # agent commits instead of burning the full budget on execute_sql
        # (v2 diagnosis: 11/120 failures were exactly 100-turn no-commit).
        is_practiq = bool(task_data.get("_practiq_meta"))
        abstain_hint = (
            "\n[IMPORTANT — PRACTIQ-style task]\n"
            "This benchmark mixes two task types, and you must decide which this is:\n"
            "  1. AMBIGUOUS — the question is answerable but under-specified "
            "(e.g. a filter term maps to multiple columns, or a value appears in "
            "multiple tables). Resolve via ask_user(...), then submit_sql(sql).\n"
            "  2. UNREALIZABLE — the question references a column / value / "
            "join path that does NOT exist in the schema. Verify absence via "
            "explore_tables + sample_values + execute_sql, then call "
            "abstain(reason=\"<what is missing>\"). Do NOT abstain merely "
            "because your SQL attempts keep failing.\n\n"
            "TURN BUDGET: you have a hard cap of 100 agent turns. Every tool "
            "call consumes one turn. The session is force-stopped without "
            "credit if you reach 100 turns without calling submit_sql OR "
            "abstain. Plan your exploration budget accordingly — do NOT spend "
            "all turns on execute_sql and leave yourself no turn to commit.\n"
            if is_practiq else ""
        )

        # Don't claim "schema pre-loaded" when ABLATE_SCHEMA hides it.
        schema_hint = (
            "Schema and foreign keys are pre-loaded above. "
            if schema_section else
            "No schema is pre-loaded — call get_schema()/explore_tables() to discover it. "
        )
        initial_message = (
            f"Database: {db_name}\n"
            f"Task ID: {instance_id}\n\n"
            f"User Query:\n{task_data.get('amb_user_query', '')}\n\n"
            f"{schema_section}\n"
            f"{kb_section}"
            f"You have a budget of {initial_budget:.1f} bird-coins. "
            f"{schema_hint}"
            f"Use your tools to clarify ambiguities with the user, look up knowledge definitions, "
            f"and submit your final SQL efficiently."
            f"{abstain_hint}"
        )

        run_result = await run_agent_session(instance_id, initial_message)
        state = run_result.get("state", {})
        elapsed = time.time() - start_time

        result = {
            "task_id": instance_id,
            "instance_id": instance_id,
            "database": db_name,
            "phase1_passed": state.get("phase1_completed", False),
            "phase2_passed": state.get("phase2_completed", False),
            "has_follow_up": bool(task_data.get("follow_up") and task_data["follow_up"].get("sol_sql")),
            "total_reward": state.get("total_reward", 0.0),
            "elapsed_seconds": elapsed,
            "budget_used": initial_budget - max(0, state.get("budget_remaining", initial_budget)),
            "budget_remaining": max(0, state.get("budget_remaining", initial_budget)),
            "dialogue_history": state.get("dialogue_history", []),
            "tool_trajectory": state.get("tool_trajectory", []),
            "adk_events": state.get("adk_events", []),
            "final_response": run_result.get("response", ""),
            "ctx_overflow": run_result.get("ctx_overflow"),
            "invalid_tool_call": run_result.get("invalid_tool_call"),
        }
        logger.info(
            "Task %s done. Reward: %.2f, Budget used: %.1f, Time: %.1fs",
            instance_id,
            result["total_reward"],
            result["budget_used"],
            elapsed,
        )
        return result
    finally:
        await cleanup_task_service(instance_id)


async def run_evaluation(data_path: str, output_path: str, limit: int = None):
    tasks = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    if limit:
        tasks = tasks[:limit]
    logger.info("A-Interact: Evaluating %d tasks", len(tasks))

    results = []
    total_reward = 0.0
    p1_count = 0
    p2_count = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for i, td in enumerate(tasks):
        logger.info("=== Task %d/%d: %s ===", i + 1, len(tasks), td["instance_id"])
        try:
            r = await run_single_task(td)
            results.append(r)
            total_reward += r["total_reward"]
            if r["phase1_passed"]:
                p1_count += 1
            if r["phase2_passed"]:
                p2_count += 1
        except Exception as e:
            logger.error("Error: %s: %s", td["instance_id"], e)
            traceback.print_exc()
            results.append({"task_id": td["instance_id"], "error": str(e), "total_reward": 0})

        if (i + 1) % 5 == 0 or i == len(tasks) - 1:
            n = len(results)
            output = {
                "mode": "a-interact",
                "metrics": {
                    "total_tasks": n,
                    "total_reward": total_reward,
                    "average_reward": total_reward / n if n else 0,
                    "phase1_rate": p1_count / n if n else 0,
                    "phase2_rate": p2_count / n if n else 0,
                    "phase1_count": p1_count,
                    "phase2_count": p2_count,
                },
                "results": results,
            }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2, default=str)

    n = len(tasks)
    if n:
        logger.info(
            "\nDone! Tasks: %d, Avg Reward: %.4f, P1: %d/%d (%.1f%%), P2: %d/%d (%.1f%%)",
            n,
            total_reward / n,
            p1_count,
            n,
            p1_count / n * 100,
            p2_count,
            n,
            p2_count / n * 100,
        )


def main():
    parser = argparse.ArgumentParser(description="BIRD-Interact a-interact evaluation")
    parser.add_argument("--data", default=settings.data_path)
    parser.add_argument("--output", default="results/eval_ainteract.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.data, args.output, args.limit))


if __name__ == "__main__":
    main()
