"""System agent — PASCAL anchor and official ReACT baseline.

Two configurations:

* default — PASCAL anchor.  Streamlined toolset + phased instruction.
* PASCAL_NO_PROTOCOL=1 — official ReACT baseline.  Strips KB tools,
  uses the minimal upstream agentic prompt.
"""

import logging
import os
from typing import Any

from shared.config import settings

PASCAL_NO_PROTOCOL = os.environ.get("PASCAL_NO_PROTOCOL", "0") == "1"
# Internal: PASCAL_NO_PROTOCOL implies KB tools are stripped.
PASCAL_ABLATE_KB = PASCAL_NO_PROTOCOL

try:
    from google.adk import Agent
    from google.adk.tools import FunctionTool
    from google.genai import types
    ADK_AVAILABLE = True
    ADK_IMPORT_ERROR = ""
except ImportError as exc:
    Agent = Any
    FunctionTool = None
    types = None
    ADK_AVAILABLE = False
    ADK_IMPORT_ERROR = str(exc)

logger = logging.getLogger(__name__)


from shared.llm import build_adk_model as _build_model


# ── PASCAL anchor instruction (byte-exact paper anchor) ─────────────
AINTERACT_INSTRUCTION_PASCAL = """You are a PostgreSQL expert agent. Follow this phased strategy to solve the user's query efficiently.

# Phase 0: Explore the database
1. explore_tables() — lists all tables, columns, row counts
2. list_foreign_keys() — shows join paths (table.col → table.col)
3. get_all_external_knowledge_names — discover what domain knowledge is available

# Phase 1: Understand the query + look up knowledge
Read the user query carefully. Identify which knowledge terms are referenced.
- get_knowledge_definition(name) — look up the specific terms/formulas mentioned in the query
  If the query mentions a metric/score by name, query KB for that name first.
  Abbreviations work: "ESI" finds "Environmental Suitability Index (ESI)".
- ask_user(question) — ask a focused clarification question
  IMPORTANT: If get_knowledge_definition returns "not found" for a key metric,
  DO NOT guess the formula. Use ask_user to ask the user for the exact definition.
  Example: if "Battery Efficiency Ratio" is not in KB, ask:
  "How should I calculate the Battery Efficiency Ratio? What columns and formula should I use?"

# Phase 2: Build + verify SQL
Write SQL using the EXACT column names from explore_tables output and formulas from knowledge definitions.
- execute_sql — test your SQL, check the results make sense
- sample_values(table, column) — check actual data values if needed for WHERE clauses
- Iterate: run SQL → check results → fix → run again

# Phase 3: Submit
- submit_sql(sql) — submit your final PostgreSQL query
- If it fails, read the error carefully, fix the issue, and resubmit.

# CRITICAL RULES
- ALWAYS reserve 3 coins for submit_sql. Never spend your last 3 coins on anything else.
- Start with explore_tables and list_foreign_keys (both FREE) before any paid tool.
- Read external knowledge definitions BEFORE writing SQL — they contain exact formulas, thresholds, and column mappings you need.
- Use EXACT column names from explore_tables output. Do NOT guess column names.
- PostgreSQL: ROUND(x, n) requires ROUND(CAST(x AS numeric), n). Every non-aggregated SELECT column must be in GROUP BY.
- After a successful Phase 1, you may receive a follow-up question for Phase 2.

# For DDL/Management tasks (CREATE, ALTER, UPDATE, DELETE, INSERT):
- These do NOT need complex JOINs. Write the DDL/DML directly after exploring the schema.
- Budget allocation: 0 coins explore → 1 coin verify → 3 coins submit.
"""


# ── Official ReACT baseline instruction ─────────────────────────────
# Matches the upstream BIRD-Interact agentic scaffold: minimal prompt
# + original 9-tool surface (minus KB tools).
AINTERACT_INSTRUCTION_REACT = (
    "You are a PostgreSQL agent. Solve the user's query using the "
    "available tools. Use ask_user(question) for ambiguity, "
    "execute_sql(sql) to test queries, and submit_sql(sql) when done. "
    "You may also call abstain(reason) if the question cannot be answered."
)


def build_agent(mode: str = "a-interact") -> Agent:
    """Build the system agent.  Only a-interact is supported in this release."""
    if not ADK_AVAILABLE:
        raise RuntimeError(f"google-adk runtime unavailable: {ADK_IMPORT_ERROR}")
    if mode != "a-interact":
        raise ValueError(
            f"build_agent: unsupported mode={mode!r}; this release ships "
            "only a-interact (PASCAL anchor + official ReACT)."
        )

    from system_agent.tools import (
        get_ainteract_tools_streamlined,
        get_ainteract_tools_original,
    )
    from system_agent.callbacks import (
        before_model_callback, after_model_callback,
        before_tool_callback, after_tool_callback,
    )

    model = _build_model(settings.system_agent_model)

    if PASCAL_NO_PROTOCOL:
        base_tools = get_ainteract_tools_original()
        kb_tool_names = {
            "get_all_external_knowledge_names",
            "get_knowledge_definition",
            "get_all_knowledge_definitions",
        }
        base_tools = [
            t for t in base_tools
            if not (hasattr(t, "func") and t.func.__name__ in kb_tool_names)
        ]
        instruction = AINTERACT_INSTRUCTION_REACT
        logger.warning(
            "AGENT MODE: official ReACT baseline (PASCAL_NO_PROTOCOL=1, %d tools)",
            len(base_tools),
        )
    else:
        base_tools = get_ainteract_tools_streamlined()
        instruction = AINTERACT_INSTRUCTION_PASCAL
        logger.info(
            "AGENT MODE: PASCAL anchor (streamlined %d tools)",
            len(base_tools),
        )

    return Agent(
        model=model,
        name="bird_interact_agent",
        description="Text-to-SQL agent for BIRD-Interact a-interact benchmark.",
        instruction=instruction,
        tools=base_tools,
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
        before_tool_callback=before_tool_callback,
        after_tool_callback=after_tool_callback,
        generate_content_config=types.GenerateContentConfig(temperature=0.7),
    )
