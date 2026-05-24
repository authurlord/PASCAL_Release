"""ADK callbacks for a-interact mode: budget management and turn limiting."""

import json
import logging
import os
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types
from shared.config import settings
from shared.retry_breaker import RetryBreaker
from shared.context_manager import count_tokens, compress_tool_history

logger = logging.getLogger(__name__)

# Global retry breaker instance (shared across all tasks in this process)
_retry_breaker = RetryBreaker(max_same_error=4, max_total_submits=30, max_same_vdiff=3)
# A/B toggle — set RETRY_BREAKER_DISABLE=1 to short-circuit the breaker
# (for baseline runs that match the pre-fix behavior).
_RETRY_BREAKER_DISABLED = os.getenv("RETRY_BREAKER_DISABLE", "0") == "1"

# Context window config (matches vLLM MAX_MODEL_LEN)
_MAX_CONTEXT = 80000
_OUTPUT_RESERVE = 4096

MAX_MODEL_TURNS = 100

# PRACTIQ turn-pressure nudge thresholds.  PRACTIQ failures are often
# pure execute_sql exploration that runs out the 100-turn cap without
# ever calling submit_sql or abstain.  A soft nudge at turn 60 plus a
# force-commit nudge at turn 80 catches those cases without false-
# positiving on legitimate exploration.
_PRACTIQ_NUDGE_SOFT_TURN = 60
_PRACTIQ_NUDGE_STRONG_TURN = 80

TOOL_COSTS = {
    # Original tools
    "execute_sql": 1.0,
    "get_schema": 1.0,
    "get_all_column_meanings": 1.0,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1.0,
    "ask_user": 2.0,
    "submit_sql": 3.0,
    "abstain": 2.0,
    # PASCAL Phase 0: Free/cheap exploration
    "explore_tables": 0.0,
    "list_foreign_keys": 0.0,
    "sample_values": 0.5,
    "check_value_exists": 0.25,
    # PASCAL Phase 1: Coarse query skills
    "select_aggregate": 1.0,
    "join_query": 1.0,
    # PASCAL Phase 3: Repair
    "explain_error": 0.5,
}


def _preview(value: Any, limit: int = 2000) -> Any:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return text[:limit] + "...<truncated>" if len(text) > limit else text


def _inject_practiq_nudge(callback_context, llm_request, turns: int) -> None:
    """PRACTIQ-only turn-pressure nudge.

    Appends a synthesized user-role message to llm_request.contents at the
    soft and strong thresholds, one-shot per threshold (guarded by state
    flags). The agent reads it as the latest environment signal before
    picking its next action, and is told explicitly to call submit_sql or
    abstain. Safe to call on every model turn; the one-shot guards short-
    circuit after the first fire.
    """
    if llm_request is None or not hasattr(llm_request, "contents"):
        return
    contents = llm_request.contents
    if contents is None:
        return

    remaining = MAX_MODEL_TURNS - turns
    state = callback_context.state
    nudge_text: str | None = None

    if (turns >= _PRACTIQ_NUDGE_STRONG_TURN
            and not state.get("_practiq_strong_nudged", False)):
        state["_practiq_strong_nudged"] = True
        nudge_text = (
            f"[SYSTEM NUDGE — turn {turns}/{MAX_MODEL_TURNS}]\n"
            f"You have ~{remaining} turns left before the session is "
            f"force-stopped with zero credit. You MUST now commit: either "
            f"call submit_sql(sql) with your best-effort SQL, or call "
            f"abstain(reason=...) if you have verified the task is "
            f"unrealizable (missing column / value / join path). Do NOT "
            f"run further execute_sql exploration — commit first, refine "
            f"only if turns remain."
        )
    elif (turns >= _PRACTIQ_NUDGE_SOFT_TURN
          and not state.get("_practiq_soft_nudged", False)):
        state["_practiq_soft_nudged"] = True
        nudge_text = (
            f"[SYSTEM NUDGE — turn {turns}/{MAX_MODEL_TURNS}]\n"
            f"You have burned {turns} of {MAX_MODEL_TURNS} turns. Start "
            f"converging: within ~{MAX_MODEL_TURNS - _PRACTIQ_NUDGE_SOFT_TURN} "
            f"turns you must call submit_sql(sql) OR abstain(reason=...). "
            f"If the question is ambiguous, use ask_user now. If a column / "
            f"value looks missing, verify and then call abstain. Do not "
            f"loop on execute_sql without a commit plan."
        )

    if nudge_text is None:
        return

    try:
        new_contents = list(contents)
        new_contents.append(genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=nudge_text)],
        ))
        llm_request.contents = new_contents
        logger.warning(
            "PRACTIQ nudge injected at turn %d (task=%s, level=%s)",
            turns,
            state.get("task_id", "?"),
            "strong" if turns >= _PRACTIQ_NUDGE_STRONG_TURN else "soft",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("PRACTIQ nudge injection failed: %s", exc)


async def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Cap LLM invocations at MAX_MODEL_TURNS."""
    turns = callback_context.state.get("model_turns", 0) + 1
    callback_context.state["model_turns"] = turns
    if turns > MAX_MODEL_TURNS:
        logger.warning("Max model turns (%d) reached, forcing stop.", MAX_MODEL_TURNS)
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(
                    text="Maximum interaction turns reached. Task ended."
                )],
            ),
        )

    if callback_context.state.get("task_done", False):
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text="Task completed.")],
            ),
        )

    # PRACTIQ-only: inject turn-pressure nudge when the agent is burning
    # budget on exploration without committing. Gated on _practiq_meta so
    # the BIRD flow stays unchanged. One-shot per threshold via state flags.
    if callback_context.state.get("_practiq_meta"):
        _inject_practiq_nudge(callback_context, llm_request, turns)

    # Front-end context compression: check token count before sending to LLM
    # (context compression front-end: see shared/context_manager.py)
    # Note: ADK llm_request.contents only has text parts, not full serialized
    # tool calls. Actual token count is ~10x higher than what we see here.
    # Real overflow protection comes from 80K context + vLLM rejection handling.
    if llm_request and hasattr(llm_request, 'contents') and llm_request.contents:
        try:
            # Convert ADK contents to flat text for token estimation
            ctx_text = ""
            for content in llm_request.contents:
                if hasattr(content, 'parts'):
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            ctx_text += part.text + "\n"
            token_count = count_tokens(ctx_text)
            budget = _MAX_CONTEXT - _OUTPUT_RESERVE
            if token_count > budget:
                # Compress: convert contents to message list, compress, convert back
                msgs = []
                for content in llm_request.contents:
                    role = getattr(content, 'role', 'user')
                    text_parts = []
                    for part in (content.parts or []):
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    msgs.append({"role": role, "content": "\n".join(text_parts)})

                compressed, level = compress_tool_history(msgs, _MAX_CONTEXT, _OUTPUT_RESERVE)
                if level > 0:
                    # Rebuild contents from compressed messages
                    new_contents = []
                    for msg in compressed:
                        new_contents.append(genai_types.Content(
                            role=msg["role"] if msg["role"] in ("user", "model") else "user",
                            parts=[genai_types.Part.from_text(text=msg["content"])],
                        ))
                    llm_request.contents = new_contents
                    new_count = count_tokens("".join(m["content"] for m in compressed))
                    logger.info("Context compressed L%d: %d → %d tokens", level, token_count, new_count)
        except Exception as e:
            logger.debug("Context compression skipped: %s", e)

    # Research mode: don't stop on negative budget.
    # NOTE: Uncomment for official benchmark evaluation:
    # budget = callback_context.state.get("budget_remaining", None)
    # if budget is not None and budget < 0:
    #     return LlmResponse(
    #         content=genai_types.Content(
    #             role="model",
    #             parts=[genai_types.Part.from_text(text="Budget exhausted. Task ended.")],
    #         ),
    #     )

    return None


# Match the legacy `<tool_call><function=NAME(...)></tool_call>` and
# also a bare `<function=NAME(arg1=...)>` form that Qwen sometimes
# emits when the vllm `qwen3_coder` tool-call parser fails to fire.
_LEGACY_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<args>.*?)\)\s*>?\s*</tool_call>",
    re.DOTALL,
)
_LEGACY_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<args>.*?)\)>",
    re.DOTALL,
)


def _parse_legacy_args(arg_str: str) -> dict:
    """Best-effort kwarg parser for `<function=NAME(k=v, k=v)>` strings.

    Falls back to {} when parsing fails; the agent's tool layer
    handles missing args gracefully.
    """
    arg_str = (arg_str or "").strip()
    if not arg_str:
        return {}
    # Try JSON first
    try:
        if arg_str.startswith("{") and arg_str.endswith("}"):
            return json.loads(arg_str)
    except Exception:
        pass
    out = {}
    for m in re.finditer(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*("
        r'"(?:\\.|[^"\\])*"|'        # double-quoted
        r"'(?:\\.|[^'\\])*'|"        # single-quoted
        r"[^,]+"                      # bare token until next comma
        r")",
        arg_str,
    ):
        k, v = m.group(1), m.group(2).strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            try:
                v = json.loads(v) if v.startswith('"') else v[1:-1]
            except Exception:
                v = v[1:-1]
        out[k] = v
    return out


async def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Rescue malformed tool calls produced by the model.

    The vllm ``qwen3_coder`` tool-call parser occasionally fails to fire
    on responses that use the legacy ``<tool_call><function=NAME(args)>``
    XML shape.  When that happens the response arrives as pure text and
    the agent never invokes a tool.  This callback detects the legacy
    shape in any text part and replaces the response with a proper
    function_call so the agent can continue.
    """
    if not llm_response or not llm_response.content:
        return None
    parts = list(llm_response.content.parts or [])
    if not parts:
        return None

    new_parts: list = []
    rescued = False
    for p in parts:
        # Already a structured function_call — leave alone.
        if getattr(p, "function_call", None):
            new_parts.append(p)
            continue
        text = getattr(p, "text", None)
        if not text:
            new_parts.append(p)
            continue
        # Try the strict <tool_call><function=...></tool_call> shape first.
        match = _LEGACY_TOOL_CALL_RE.search(text) or _LEGACY_FUNCTION_RE.search(text)
        if not match:
            new_parts.append(p)
            continue
        name = match.group("name")
        args = _parse_legacy_args(match.group("args"))
        prefix = text[: match.start()].rstrip()
        if prefix:
            new_parts.append(genai_types.Part.from_text(text=prefix))
        new_parts.append(
            genai_types.Part(
                function_call=genai_types.FunctionCall(name=name, args=args)
            )
        )
        rescued = True
        logger.warning(
            "after_model_callback: rescued legacy tool call -> %s(%s)",
            name, list(args.keys()),
        )

    if not rescued:
        return None
    return LlmResponse(
        content=genai_types.Content(role="model", parts=new_parts),
    )


async def before_tool_callback(
    tool, args: dict, tool_context: ToolContext
) -> dict | None:
    """Deduct budget. Free submit exit when exhausted."""
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    cost = TOOL_COSTS.get(tool_name)
    if cost is None:
        return None

    budget = tool_context.state.get("budget_remaining", 0)

    # Research mode: track costs but don't enforce budget limits.
    # This allows the agent to explore freely. For official benchmark
    # evaluation, re-enable the budget enforcement below.
    tool_context.state["_budget_before"] = budget
    remaining = budget - cost
    tool_context.state["budget_remaining"] = remaining

    # NOTE: Uncomment the block below for official benchmark evaluation:
    # if budget < cost:
    #     if tool_name == "submit_sql":
    #         tool_context.state["budget_remaining"] = -1
    #         return None
    #     return {"error": f"Budget exhausted ({budget:.1f} remaining). "
    #             "You MUST call submit_sql now with your best SQL."}
    # if tool_name == "submit_sql" and remaining <= 0:
    #     remaining = -1
    #     tool_context.state["budget_remaining"] = remaining

    return None


async def after_tool_callback(
    tool, args: dict, tool_context: ToolContext, tool_response
) -> dict | None:
    """Record tool event in trajectory and append budget note to response."""
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    cost = TOOL_COSTS.get(tool_name, 0)
    budget_before = tool_context.state.get("_budget_before")
    budget_after = tool_context.state.get("budget_remaining")
    initial = tool_context.state.get("initial_budget", 0)

    trajectory = tool_context.state.get("tool_trajectory", [])
    trajectory.append({
        "type": "tool",
        "tool": tool_name,
        "args": args,
        "result": _preview(tool_response),
        "cost": cost,
        "budget_before": budget_before,
        "budget_after": budget_after,
    })
    tool_context.state["tool_trajectory"] = trajectory

    # Retry breaker for submit_sql failures
    retry_hint = ""
    if tool_name == "submit_sql" and not _RETRY_BREAKER_DISABLED:
        resp_str = str(tool_response)
        task_id = tool_context.state.get("task_id", "unknown")
        # Server message formats (db_environment/server.py):
        #   Success: "Phase 1 correct! (Reward: ..." / "Phase 2 correct! (Reward: ..."
        #   Failure: "SQL failed Phase <N>. ..."
        is_success = ("phase 1 correct!" in resp_str.lower()
                      or "phase 2 correct!" in resp_str.lower())
        is_failure = "sql failed phase" in resp_str.lower()

        if is_failure:
            # Parse shape hint ("Expected result shape: X rows × Y columns. Your result shape: A rows × B columns")
            shape_match = re.search(
                r'Expected result shape: (\d+) rows × (\d+) columns\. '
                r'Your result shape: (\d+) rows × (\d+) columns',
                resp_str)
            if shape_match:
                er, ec, ar, ac = [int(x) for x in shape_match.groups()]
                gt_shape = (er, ec)
                pred_shape = (ar, ac)
                err_cat = "wrong_values" if (er == ar and ec == ac) else "shape_mismatch"
            elif "[exec_err_flg]" in resp_str:
                gt_shape = pred_shape = (0, 0)
                err_cat = "sql_error"
            else:
                gt_shape = pred_shape = (0, 0)
                err_cat = "unknown"

            # Value-diff hash: include diff body in signature so "same diff 3x" counts
            # as strong evidence the agent is stuck (not changing SQL / not changing
            # result) vs merely repeating the shape category.
            vdiff_match = re.search(r'Value diff:\n(.+?)(?:\n\[SYSTEM NOTE|\Z)',
                                    resp_str, re.DOTALL)
            vdiff_sig = hash(vdiff_match.group(1).strip()) if vdiff_match else 0

            # Track both (category, shape) and (category, vdiff_sig). Fire on
            # either pattern — whichever repeats first signals a real stall.
            hint = _retry_breaker.record_failure(
                task_id, pred_shape, gt_shape, err_cat, vdiff_sig=vdiff_sig)
            if hint:
                retry_hint = f"\n\n[STRATEGY HINT: {hint}]"
                # Persist the hint text into the trajectory entry so
                # post-hoc analysis (scripts/trajectory_diff) can count
                # and attribute breaker firings. Without this the hint
                # is stripped from result JSON because _preview stores
                # the pre-suffix response.
                if trajectory:
                    trajectory[-1]["retry_hint"] = hint
                    trajectory[-1]["retry_hint_category"] = err_cat
        elif is_success:
            # Phase 1 success (moving to Phase 2) should also reset — it starts
            # a new SQL task, old patterns don't apply.
            _retry_breaker.record_success(task_id)

    # Append budget note to agent-visible response (matches reference implementation)
    suffix = retry_hint
    if budget_after is not None and budget_after >= 0:
        suffix += f"\n\n[SYSTEM NOTE: Remaining budget: {budget_after:.1f}/{initial:.1f}]"
    if suffix:
        return str(tool_response) + suffix
    return None
