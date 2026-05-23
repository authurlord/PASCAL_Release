"""ADK tools for a-interact mode.

These tools allow the system agent to interact with:
1. DB Environment (port 6002): execute SQL, get schema, get column meanings, get knowledge
2. User Simulator (port 6001): ask clarification questions
3. Submission (port 6002): submit final SQL for evaluation

All tools route through the FastAPI services via HTTP.
Budget deduction and trajectory logging are handled by callbacks (callbacks.py).
"""

import json
import logging
import httpx
from typing import Optional

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from shared.config import settings

logger = logging.getLogger(__name__)

# ── Global task context (fallback, set by orchestrator before each task) ──
_current_task_id: str = ""


def set_current_task_id(task_id: str):
    global _current_task_id
    _current_task_id = task_id


def _get_task_id(tool_context: Optional[ToolContext] = None) -> str:
    if tool_context:
        tid = tool_context.state.get("task_id", "")
        if tid:
            return tid
    return _current_task_id


def _db_url(path: str) -> str:
    return f"http://localhost:{settings.db_env_port}{path}"


def _user_url(path: str) -> str:
    return f"http://localhost:{settings.user_sim_port}{path}"


# ── DB Environment Tools ──

def execute_sql(sql: str, tool_context: ToolContext) -> str:
    """Execute a SQL query against the PostgreSQL database and return the results.
    Use this to explore the database, test queries, or verify your SQL before submitting.
    Cost: 1 bird-coin.

    Args:
        sql: The PostgreSQL SQL query to execute.

    Returns:
        The query results formatted as a table, or an error message.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            resp = client.post(_db_url("/execute"),
                               json={"task_id": task_id, "sql": sql})
            if resp.status_code != 200:
                return f"SQL Error: Server returned status {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            if data.get("success"):
                return data.get("result", "Query executed successfully.")
            else:
                return f"SQL Error: {data.get('error') or 'Execution failed (no details)'}"
    except Exception as e:
        return f"Error calling DB environment: {type(e).__name__}: {e}"


def get_schema(tool_context: ToolContext) -> str:
    """Get the full database schema (CREATE TABLE statements) for the current task's database.
    Cost: 1 bird-coin.

    Returns:
        The database schema as text.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/schema"),
                               json={"task_id": task_id})
            return resp.json().get("schema", "Schema not available")
    except Exception as e:
        return f"Error: {e}"


def get_all_column_meanings(tool_context: ToolContext) -> str:
    """Get the meanings/descriptions of all columns in the database.
    Cost: 1 bird-coin.

    Returns:
        JSON string with column meanings for all tables.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/all_column_meanings"),
                               json={"task_id": task_id})
            return resp.json().get("column_meanings", "{}")
    except Exception as e:
        return f"Error: {e}"


def get_column_meaning(table_name: str, column_name: str, tool_context: ToolContext) -> str:
    """Get the meaning/description of a specific column in a table.
    Cost: 0.5 bird-coins.

    Args:
        table_name: Name of the table.
        column_name: Name of the column.

    Returns:
        The column meaning/description.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/column_meaning"),
                               json={"task_id": task_id,
                                     "table_name": table_name,
                                     "column_name": column_name})
            return resp.json().get("meaning", "Column meaning not found")
    except Exception as e:
        return f"Error: {e}"


def get_all_external_knowledge_names(tool_context: ToolContext) -> str:
    """Get the names of all available external knowledge entries for this database.
    Use this to discover what domain knowledge is available.
    Cost: 0.5 bird-coins.

    Returns:
        JSON list of knowledge entry names.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/knowledge_names"),
                               json={"task_id": task_id})
            return json.dumps(resp.json().get("names", []))
    except Exception as e:
        return f"Error: {e}"


def get_knowledge_definition(knowledge_name: str, tool_context: ToolContext) -> str:
    """Get the definition/details of a specific external knowledge entry.
    Cost: 0.5 bird-coins.

    Args:
        knowledge_name: The name of the knowledge entry to look up.

    Returns:
        JSON string with the knowledge definition.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/knowledge"),
                               json={"task_id": task_id,
                                     "knowledge_name": knowledge_name})
            return resp.json().get("knowledge", "Knowledge not found")
    except Exception as e:
        return f"Error: {e}"


def get_all_knowledge_definitions(tool_context: ToolContext) -> str:
    """Get all external knowledge definitions for this database.
    Cost: 1 bird-coin.

    Returns:
        JSON string with all knowledge definitions.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/knowledge"),
                               json={"task_id": task_id})
            return resp.json().get("knowledge", "[]")
    except Exception as e:
        return f"Error: {e}"


# ── User Simulator Tool ──

def ask_user(question: str, tool_context: ToolContext) -> str:
    """Ask the user a clarification question about their query.
    Use this when the user's request is ambiguous and you need more information.
    Cost: 2 bird-coins.

    Args:
        question: The clarification question to ask the user.

    Returns:
        The user's response to your question.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            resp = client.post(_user_url("/ask"),
                               json={"task_id": task_id,
                                     "question": question})
            answer = resp.json().get("answer", "No response from user.")
            history = tool_context.state.get("dialogue_history", [])
            history.append({"role": "agent", "content": question})
            history.append({"role": "user", "content": answer})
            tool_context.state["dialogue_history"] = history
            return answer
    except Exception as e:
        return f"Error: {e}"


# ── Submit Tool ──

def submit_sql(sql: str, tool_context: ToolContext) -> str:
    """Submit your final SQL query for evaluation.
    This tests your SQL against the ground truth. Only submit when confident.
    Cost: 3 bird-coins.

    Args:
        sql: The final PostgreSQL SQL query to submit.

    Returns:
        Evaluation result including pass/fail, reward, and any follow-up instructions.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            resp = client.post(_db_url("/submit"),
                               json={"task_id": task_id, "sql": sql})
            data = resp.json()

            # Update session state based on result
            if data.get("passed"):
                reward = data.get("reward", 0.0)
                tool_context.state["total_reward"] = tool_context.state.get("total_reward", 0.0) + reward
                phase = data.get("phase_completed")
                if phase == 1:
                    tool_context.state["phase1_completed"] = True
                    tool_context.state["current_phase"] = 2
                    if data.get("has_follow_up"):
                        try:
                            client.post(_user_url("/phase_transition"), json={"task_id": task_id})
                        except Exception as exc:
                            logger.warning("Phase transition failed for %s: %s", task_id, exc)
                    else:
                        tool_context.state["task_done"] = True
                elif phase == 2:
                    tool_context.state["phase2_completed"] = True
                    tool_context.state["task_done"] = True

            # Build response message
            raw_msg = data.get("message", "")
            # Store raw message for orchestrator (has [exec_err_flg] for debug routing)
            tool_context.state["_last_submit_raw"] = raw_msg
            # Clean message for agent
            agent_msg = raw_msg.replace("[exec_err_flg] ", "")
            parts = [agent_msg]
            if data.get("reward", 0) > 0:
                parts.append(f"Reward: {data['reward']}")
            if data.get("has_follow_up"):
                parts.append(f"Follow-up question: {data['follow_up_query']}")
            budget = tool_context.state.get("budget_remaining", 0)
            parts.append(f"Budget remaining: {budget} bird-coins")
            return "\n".join(parts)
    except Exception as e:
        return f"Error: {e}"


def abstain(reason: str, tool_context: ToolContext) -> str:
    """Declare that this question cannot be answered given the available schema.

    Use this ONLY when you have good evidence the task is un-realizable:
      - A column referenced by the question does not exist after schema inspection
      - A value referenced by the question is confirmed absent from the data
      - No foreign-key or join path exists to satisfy the relationship
    Do NOT use this just because your SQL attempts keep failing; investigate
    the schema and use ask_user for ambiguity first.

    Cost: 2 bird-coins (cheaper than submit_sql; discourages guess-then-bail).

    Args:
        reason: one-sentence rationale for abstaining (for the trajectory log).

    Returns:
        Evaluation result — on a truly unanswerable task the orchestrator
        awards Phase-1 credit; on an answerable task this returns a rejection.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            resp = client.post(
                _db_url("/abstain"),
                json={"task_id": task_id, "reason": reason or ""},
            )
            data = resp.json()
            if data.get("passed"):
                reward = data.get("reward", 0.0)
                tool_context.state["total_reward"] = (
                    tool_context.state.get("total_reward", 0.0) + reward
                )
                if data.get("phase_completed") == 1:
                    tool_context.state["phase1_completed"] = True
                    tool_context.state["current_phase"] = 2
                    tool_context.state["task_done"] = True
            raw_msg = data.get("message", "")
            tool_context.state["_last_submit_raw"] = raw_msg
            parts = [raw_msg]
            budget = tool_context.state.get("budget_remaining", 0)
            parts.append(f"Budget remaining: {budget} bird-coins")
            return "\n".join(parts)
    except Exception as e:
        return f"Error: {e}"


# ── PASCAL Phase 0: Free Exploration Tools ──

def explore_tables(tool_context: ToolContext) -> str:
    """List all tables with row counts and column names. Cost: 0 bird-coins (free).
    Use this FIRST to understand the database structure before writing any SQL.

    Works on both PostgreSQL and SQLite backends — routing is handled
    server-side in db_environment based on the task's `_backend` field.

    Returns:
        Table with table_name, row_count, columns for each table.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/explore_tables"),
                               json={"task_id": task_id})
            data = resp.json()
            return data.get("result", "No tables found.")
    except Exception as e:
        return f"Error: {e}"


def list_foreign_keys(tool_context: ToolContext) -> str:
    """List all foreign key relationships in the database. Cost: 0 bird-coins (free).
    Shows join paths: from_table.column -> to_table.column.

    Works on both PostgreSQL and SQLite backends — routing is handled
    server-side in db_environment based on the task's `_backend` field.

    Returns:
        Table of FK relationships showing join paths.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/list_foreign_keys"),
                               json={"task_id": task_id})
            data = resp.json()
            return data.get("result", "No foreign keys found.")
    except Exception as e:
        return f"Error: {e}"


def sample_values(table_name: str, column_name: str, tool_context: ToolContext) -> str:
    """Get sample distinct values from a column. Cost: 0.5 bird-coins.
    Use this to understand what values a column contains before writing WHERE clauses.

    Args:
        table_name: Name of the table.
        column_name: Name of the column to sample.

    Returns:
        Up to 10 distinct values from the column.
    """
    task_id = _get_task_id(tool_context)
    sql = f'SELECT DISTINCT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL LIMIT 10'
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/execute"),
                               json={"task_id": task_id, "sql": sql})
            data = resp.json()
            if data.get("success"):
                return data.get("result", "No values found.")
            return f"Error: {data.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error: {e}"


def check_value_exists(table_name: str, column_name: str, value: str, tool_context: ToolContext) -> str:
    """Check if a value exists in a column, with fuzzy matching. Cost: 0.25 bird-coins.
    Use this to verify filter values before building WHERE clauses.

    Args:
        table_name: Name of the table.
        column_name: Name of the column.
        value: The value to search for.

    Returns:
        Whether the value exists, and similar values if not found.
    """
    task_id = _get_task_id(tool_context)
    # Exact check + fuzzy fallback
    sql = f"""
    SELECT CASE WHEN EXISTS(SELECT 1 FROM "{table_name}" WHERE "{column_name}"::text = '{value}')
           THEN 'FOUND' ELSE 'NOT_FOUND' END AS status,
           (SELECT string_agg(DISTINCT "{column_name}"::text, ', ')
            FROM (SELECT "{column_name}" FROM "{table_name}"
                  WHERE "{column_name}"::text ILIKE '%{value}%' LIMIT 5) sub
           ) AS similar_values
    """
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/execute"),
                               json={"task_id": task_id, "sql": sql})
            data = resp.json()
            if data.get("success"):
                return data.get("result", "Check failed.")
            return f"Error: {data.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error: {e}"


# ── PASCAL Phase 1: Coarse Query Skills ──

def select_aggregate(table: str, columns: str, agg_function: str,
                     agg_column: str, group_by: str, order_by: str,
                     limit: str, tool_context: ToolContext) -> str:
    """Build and execute a single-table aggregate query with validated columns. Cost: 1 bird-coin.
    The SQL is generated from your parameters, executed, and both the SQL and results are returned.

    Args:
        table: Table name (must exist in the database).
        columns: Comma-separated column names for SELECT (e.g. "col1, col2").
        agg_function: Aggregation function: COUNT, SUM, AVG, MIN, MAX, or empty string for none.
        agg_column: Column to aggregate (use * for COUNT), or empty string.
        group_by: Comma-separated GROUP BY columns, or empty string.
        order_by: Column to sort by, or empty string.
        limit: Number of rows to return (e.g. "10"), or empty string for all.

    Returns:
        Generated SQL + query results.
    """
    task_id = _get_task_id(tool_context)
    # Build SELECT clause
    select_parts = [c.strip() for c in columns.split(",") if c.strip()] if columns.strip() else []
    if agg_function and agg_column:
        select_parts.append(f"{agg_function}({agg_column}) AS {agg_function.lower()}_{agg_column.replace('*', 'all')}")
    if not select_parts:
        return "Error: Must specify at least one column or aggregation."

    sql = f'SELECT {", ".join(select_parts)} FROM "{table}"'
    if group_by.strip():
        sql += f" GROUP BY {group_by}"
    if order_by.strip():
        sql += f" ORDER BY {order_by}"
    if limit.strip():
        sql += f" LIMIT {limit}"

    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/execute"),
                               json={"task_id": task_id, "sql": sql})
            data = resp.json()
            result_str = f"Generated SQL:\n{sql}\n\n"
            if data.get("success"):
                result_str += f"Results:\n{data.get('result', 'No results.')}"
            else:
                result_str += f"Execution Error: {data.get('error', 'Unknown error')}"
            return result_str
    except Exception as e:
        return f"Error: {e}"


def join_query(tables: str, join_conditions: str, select_columns: str,
               join_type: str, tool_context: ToolContext) -> str:
    """Build and execute a multi-table join query. Cost: 1 bird-coin.
    The SQL is generated from your parameters, executed, and both the SQL and results are returned.

    Args:
        tables: Comma-separated table names (first is the base table, e.g. "signals, telescopes, observatories").
        join_conditions: Semicolon-separated join conditions (e.g. "signals.telescref=telescopes.telescregistry; telescopes.observstation=observatories.observstation").
        select_columns: Comma-separated columns with table prefixes (e.g. "observatories.observstation, COUNT(signals.signalregistry)").
        join_type: Join type: INNER, LEFT, RIGHT (default INNER).

    Returns:
        Generated SQL + query results (first 20 rows).
    """
    task_id = _get_task_id(tool_context)
    table_list = [t.strip() for t in tables.split(",") if t.strip()]
    if len(table_list) < 2:
        return "Error: Need at least 2 tables for a join."
    conditions = [c.strip() for c in join_conditions.split(";") if c.strip()]
    if len(conditions) < len(table_list) - 1:
        return f"Error: Need {len(table_list)-1} join conditions for {len(table_list)} tables, got {len(conditions)}."

    jt = join_type.strip().upper() if join_type.strip() else "INNER"
    sql = f'SELECT {select_columns} FROM "{table_list[0]}"'
    for i, tbl in enumerate(table_list[1:]):
        sql += f' {jt} JOIN "{tbl}" ON {conditions[i]}'
    sql += " LIMIT 20"

    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/execute"),
                               json={"task_id": task_id, "sql": sql})
            data = resp.json()
            result_str = f"Generated SQL:\n{sql}\n\n"
            if data.get("success"):
                result_str += f"Results:\n{data.get('result', 'No results.')}"
            else:
                result_str += f"Execution Error: {data.get('error', 'Unknown error')}"
            return result_str
    except Exception as e:
        return f"Error: {e}"


# ── PASCAL Phase 2-3: Refinement + Repair Skills ──

def explain_error(sql: str, error_message: str, tool_context: ToolContext) -> str:
    """Analyze a SQL error and suggest fixes. Cost: 0.5 bird-coins.
    Checks column names, table names, and common PG pitfalls.

    Args:
        sql: The failing SQL query.
        error_message: The error message from execution.

    Returns:
        Diagnosis with specific fix suggestions.
    """
    task_id = _get_task_id(tool_context)
    diagnosis = [f"Error: {error_message}", ""]

    # Check common PG issues
    if "does not exist" in error_message.lower():
        # Extract the problematic name
        import re
        match = re.search(r'(?:column|relation|function)\s+"?(\w+)"?\s+does not exist', error_message, re.I)
        if match:
            name = match.group(1)
            diagnosis.append(f"'{name}' was not found. Checking database for similar names...")
            # Search for similar column/table names
            check_sql = f"""
            SELECT 'column' AS type, table_name, column_name
            FROM information_schema.columns
            WHERE column_name ILIKE '%{name}%' AND table_schema='public'
            UNION ALL
            SELECT 'table', table_name, '' FROM information_schema.tables
            WHERE table_name ILIKE '%{name}%' AND table_schema='public'
            LIMIT 10
            """
            try:
                with httpx.Client(timeout=15.0, trust_env=False) as client:
                    resp = client.post(_db_url("/execute"),
                                       json={"task_id": task_id, "sql": check_sql})
                    data = resp.json()
                    if data.get("success"):
                        diagnosis.append(f"Similar names found:\n{data.get('result', 'None')}")
            except Exception:
                pass

    if "round" in error_message.lower() and "double precision" in error_message.lower():
        diagnosis.append("FIX: PostgreSQL ROUND requires numeric type. Use ROUND(CAST(expr AS numeric), n)")

    if "group by" in error_message.lower():
        diagnosis.append("FIX: Every non-aggregated column in SELECT must appear in GROUP BY.")

    return "\n".join(diagnosis)


# ── Build tool list for ADK Agent ──

def get_ainteract_tools():
    """Return full PASCAL + original tool set for a-interact mode.

    16 tools total. Larger models (Qwen3.6+, Gemini 2.5) can handle the
    full set; for smaller models (Qwen3.5-35B-A3B), use the streamlined
    10-tool set via get_ainteract_tools_streamlined().
    """
    return [
        # Original tools
        FunctionTool(execute_sql),
        FunctionTool(get_schema),
        FunctionTool(get_all_column_meanings),
        FunctionTool(get_column_meaning),
        FunctionTool(get_all_external_knowledge_names),
        FunctionTool(get_knowledge_definition),
        FunctionTool(get_all_knowledge_definitions),
        FunctionTool(ask_user),
        FunctionTool(submit_sql),
        FunctionTool(abstain),
        # PASCAL Phase 0: Free exploration
        FunctionTool(explore_tables),
        FunctionTool(list_foreign_keys),
        FunctionTool(sample_values),
        FunctionTool(check_value_exists),
        # PASCAL Phase 1: Coarse query
        FunctionTool(select_aggregate),
        FunctionTool(join_query),
        # PASCAL Phase 3: Repair
        FunctionTool(explain_error),
    ]


def get_ainteract_tools_streamlined():
    """Streamlined tool set for the PASCAL anchor."""
    return [
        FunctionTool(explore_tables),
        FunctionTool(list_foreign_keys),
        FunctionTool(get_schema),
        FunctionTool(sample_values),
        FunctionTool(get_all_external_knowledge_names),
        FunctionTool(get_knowledge_definition),
        FunctionTool(select_aggregate),
        FunctionTool(join_query),
        FunctionTool(execute_sql),
        FunctionTool(ask_user),
        FunctionTool(submit_sql),
        FunctionTool(abstain),
    ]


def get_ainteract_tools_original():
    """Original 9-tool set for baseline comparison."""
    return [
        FunctionTool(execute_sql),
        FunctionTool(get_schema),
        FunctionTool(get_all_column_meanings),
        FunctionTool(get_column_meaning),
        FunctionTool(get_all_external_knowledge_names),
        FunctionTool(get_knowledge_definition),
        FunctionTool(get_all_knowledge_definitions),
        FunctionTool(ask_user),
        FunctionTool(submit_sql),
        FunctionTool(abstain),
    ]
