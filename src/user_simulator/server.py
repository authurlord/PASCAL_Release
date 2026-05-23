"""User Simulator Service (Port 6001). Two-stage function-driven pipeline."""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import diskcache
from fastapi import FastAPI, HTTPException

from shared.config import settings
from shared.models import AskUserRequest, AskUserResponse, InitTaskRequest, PhaseTransitionRequest
from user_simulator.prompts import USER_SIMULATOR_ACTION_PARSER, USER_SIMULATOR_RESPONSE_GENERATOR
from user_simulator.sql_parser import segment_sql

logger = logging.getLogger(__name__)
app = FastAPI(title="BIRD-Interact User Simulator", version="1.0.0")

PROMPT_VERSION = settings.prompt_version  # v1=legacy, v2=recommended

# Persistent ask_user cache — shared across SA/MA paths, across runs.
# Disable with USIM_CACHE_DISABLE=1.
_CACHE_DISABLED = os.environ.get("USIM_CACHE_DISABLE", "0") == "1"
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "ask_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_ask_cache = diskcache.Cache(str(_CACHE_DIR)) if not _CACHE_DISABLED else None


def _cache_key(task_id: str, question: str, phase: int) -> str:
    """Key the cache on (prompt_version, usim_model, task_id, phase, normalized_question).

    Including usim_model prevents a Qwen-authored answer from serving a Gemini call.
    Including phase prevents phase-1 reasoning leaking into phase-2 follow-ups.
    """
    material = "||".join([
        PROMPT_VERSION,
        settings.user_sim_model,
        task_id,
        str(phase),
        question.strip(),
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

# Retry configuration for transient LLM failures (rate limits, network errors, 5xx).
# User sim calls are user-facing — we should not fail a whole task on a single flaky call.
LLM_MAX_RETRIES = int(__import__("os").environ.get("USIM_LLM_MAX_RETRIES", "5"))
LLM_RETRY_BACKOFF = float(__import__("os").environ.get("USIM_LLM_RETRY_BACKOFF", "2.0"))


class TaskSimState:
    def __init__(self, task_data: Dict[str, Any]):
        self.task_data = task_data
        self.db_name = task_data["selected_database"]
        self.amb_user_query = task_data["amb_user_query"]
        self.clear_query = task_data.get("query", task_data["amb_user_query"])
        self.reference_sql = task_data["sol_sql"]
        self.user_query_ambiguity = task_data.get("user_query_ambiguity", {})
        self.knowledge_ambiguity = task_data.get("knowledge_ambiguity", [])
        self.current_phase = 1
        self.db_schema = ""
        fu = task_data.get("follow_up") or {}
        self.follow_up_sol_sql = fu.get("sol_sql") if fu else None

    def get_sql_segments(self, sql: str) -> str:
        segs = segment_sql(sql)
        return "\n\n".join(f"{clause}:\n{text}" for clause, text in segs)

    def get_all_sql_segments(self) -> str:
        sql_list = self.reference_sql if isinstance(self.reference_sql, list) else [self.reference_sql]
        return "\n===\n".join(self.get_sql_segments(sql) for sql in sql_list)

    def get_ambiguity_json(self) -> str:
        if self.current_phase == 1:
            return ("user_query_ambiguity: \n"
                    + json.dumps(self.user_query_ambiguity, indent=4)
                    + "\n\nknowledge_ambiguity: \n"
                    + json.dumps(self.knowledge_ambiguity, indent=4))
        return json.dumps({}, indent=4)

    def get_gt_sql_str(self) -> str:
        if isinstance(self.reference_sql, list):
            return "\n".join(self.reference_sql)
        return self.reference_sql

    def transition_to_phase2(self):
        self.current_phase = 2
        if self.follow_up_sol_sql:
            self.reference_sql = self.follow_up_sol_sql


_task_states: Dict[str, TaskSimState] = {}


def _call_llm(prompt: str, max_tokens: int = 200) -> str:
    """Call the user-sim LLM with exponential-backoff retries on transient failures.

    Fails after LLM_MAX_RETRIES attempts rather than on the first error so
    that a single rate-limit or network glitch does not poison a whole task.
    Also retries on empty-string responses (common when providers return 429
    or truncate mid-stream) up to half the retry budget.
    """
    from shared.llm import call_llm
    last_err = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            result = call_llm(
                [{"role": "user", "content": prompt}],
                model_name=settings.user_sim_model,
                temperature=0,
                max_tokens=max_tokens,
            )
            # Treat empty responses as retriable (soft failure)
            if result and result.strip():
                if attempt > 0:
                    logger.info(f"LLM call succeeded on retry {attempt + 1}")
                return result
            last_err = ValueError("empty response")
            if attempt >= LLM_MAX_RETRIES // 2:
                # Give up on empty responses past halfway — likely a
                # content/parse issue rather than transient.
                logger.warning(f"LLM returned empty after {attempt + 1} retries, giving up")
                return ""
        except Exception as e:
            last_err = e
            err_msg = str(e)[:150]
            # Retriable errors: rate limits, timeouts, 5xx, connection resets
            retriable_markers = ("429", "rate limit", "RateLimit", "timeout",
                                "Timeout", "500", "502", "503", "504",
                                "Connection", "connection", "reset", "EOF",
                                "temporarily", "overloaded", "Internal")
            is_retriable = any(m in err_msg for m in retriable_markers)
            if not is_retriable and attempt >= 1:
                # Non-retriable error (auth, bad request, etc.) — fail fast after 1 attempt
                logger.error(f"LLM non-retriable error: {err_msg}")
                return ""

        if attempt < LLM_MAX_RETRIES - 1:
            wait = LLM_RETRY_BACKOFF * (2 ** attempt)
            logger.warning(
                f"LLM call attempt {attempt + 1}/{LLM_MAX_RETRIES} failed: "
                f"{str(last_err)[:100]}; retrying in {wait:.1f}s"
            )
            time.sleep(wait)

    logger.error(f"LLM call failed after {LLM_MAX_RETRIES} retries: {last_err}")
    return ""


def _parse_action(state: TaskSimState, question: str) -> str:
    """Stage 1: Action Parser — maps clarification question to action (AMB/LOC/UNA)."""
    template = USER_SIMULATOR_ACTION_PARSER[PROMPT_VERSION]
    prompt = template.replace("[[clarification_Q]]", question)
    prompt = prompt.replace("[[amb_json]]", state.get_ambiguity_json())
    prompt = prompt.replace("[[SQL_Glot]]", state.get_all_sql_segments())
    prompt = prompt.replace("[[DB_schema]]", state.db_schema)
    # v2 includes <think> reasoning, needs more tokens
    max_tok = 500 if PROMPT_VERSION == "v2" else 200
    content = _call_llm(prompt, max_tokens=max_tok)
    # Extract action from <s>...</s> (skip <think>...</think> if present)
    if "</s>" in content:
        action = content.split("</s>")[0].strip()
    else:
        action = content.split("\n")[0].strip()
    if "<s>" in action:
        action = action[action.find("<s>"):].replace("<s>", "").strip()
    logger.info(f"Parsed action: {action}")
    return action


def _generate_response(state: TaskSimState, question: str, action: str) -> str:
    """Stage 2: Response Generator — produces user response from action + context."""
    template = USER_SIMULATOR_RESPONSE_GENERATOR[PROMPT_VERSION]
    prompt = template.replace("[[clarification_Q]]", question)
    prompt = prompt.replace("[[Action]]", action)
    prompt = prompt.replace("[[clear_query]]", state.clear_query)
    prompt = prompt.replace("[[amb_json]]", state.get_ambiguity_json())
    prompt = prompt.replace("[[GT_SQL]]", state.get_gt_sql_str())
    prompt = prompt.replace("[[SQL_Glot]]", state.get_all_sql_segments())
    prompt = prompt.replace("[[DB_schema]]", state.db_schema)
    content = _call_llm(prompt, max_tokens=1024)
    # Extract response: handle <s>...</s> framed replies (Gemini-style prompt)
    if "</s>" in content:
        extracted = content.split("</s>")[0].strip()
        if "<s>" in extracted:
            return extracted[extracted.find("<s>"):].replace("<s>", "").strip()
        return extracted
    elif "<s>" in content:
        return content.split("<s>")[1].strip()
    # Tag-less fallback: Qwen3.5-35B and similar backends don't emit <s>/</s>.
    # Surface the LLM's actual reply rather than the misleading "I'm not sure"
    # string. Strip common chat-template scaffolding before returning.
    cleaned = (content or "").strip()
    for marker in ("<|im_end|>", "<|im_start|>assistant", "<|endoftext|>"):
        cleaned = cleaned.replace(marker, "")
    cleaned = cleaned.strip()
    if cleaned:
        return cleaned
    return "I'm not sure I understand your question."


@app.post("/init_task")
async def init_task(req: InitTaskRequest):
    state = TaskSimState(req.task_data)
    _task_states[req.task_id] = state
    # Load schema from DB env
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            resp = await client.post(
                f"http://localhost:{settings.db_env_port}/schema",
                json={"task_id": req.task_id})
            state.db_schema = resp.json().get("schema", "")
    except Exception as e:
        logger.warning(f"Could not load schema: {e}")
    return {"status": "ok", "task_id": req.task_id}


def _ask_sync(state: "TaskSimState", question: str) -> str:
    """Two-stage pipeline: parse action, then generate response. Runs in thread pool."""
    # PRACTIQ short-circuit: if task carries pre-generated valid_replies for
    # its ambiguity class, return the first one deterministically (no LLM call).
    try:
        td = state.task_data
        crits = (td.get("user_query_ambiguity", {}) or {}).get("critical_ambiguity", []) or []
        if crits:
            c0 = crits[0]
            if isinstance(c0, dict) and str(c0.get("type", "")).startswith("practiq_"):
                replies = c0.get("valid_replies", []) or []
                if replies:
                    logger.info(f"practiq: return valid_reply[0] for {state.db_name}")
                    return replies[0]
    except Exception as e:
        logger.warning(f"practiq reply shortcut failed, falling through: {e}")

    action = _parse_action(state, question)
    return _generate_response(state, question, action)


@app.post("/ask", response_model=AskUserResponse)
async def ask_user(req: AskUserRequest):
    state = _task_states.get(req.task_id)
    if not state:
        raise HTTPException(404, f"Task {req.task_id} not initialized")

    if _ask_cache is not None:
        ck = _cache_key(req.task_id, req.question, state.current_phase)
        cached = _ask_cache.get(ck)
        if cached is not None:
            logger.info(f"User response for {req.task_id} [CACHE HIT]: {cached[:80]}...")
            return AskUserResponse(answer=cached)

    response = await asyncio.to_thread(_ask_sync, state, req.question)
    logger.info(f"User response for {req.task_id}: {response[:100]}...")

    if _ask_cache is not None and response and response.strip():
        # Only store non-empty responses so a transient failure doesn't poison the cache.
        _ask_cache.set(ck, response)
    return AskUserResponse(answer=response)


@app.post("/phase_transition")
async def phase_transition(req: PhaseTransitionRequest):
    state = _task_states.get(req.task_id)
    if not state:
        raise HTTPException(404, f"Task {req.task_id} not initialized")
    state.transition_to_phase2()
    return {"status": "ok", "phase": 2}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "user_simulator"}


@app.get("/debug_state/{task_id}")
async def debug_state(task_id: str):
    state = _task_states.get(task_id)
    if not state:
        return {"error": "not found"}
    return {
        "db_name": state.db_name,
        "schema_len": len(state.db_schema),
        "schema_first_100": state.db_schema[:100],
        "current_phase": state.current_phase,
        "amb_query": state.amb_user_query[:100],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.user_sim_port)
