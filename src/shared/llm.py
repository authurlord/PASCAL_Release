"""Unified LLM call interface.

Release: uses LiteLlm (supports any provider).
Local override: place _local_provider.py in this directory (gitignored)
to use a custom backend.

Gemini models (gemini-*) are called via google-genai directly, bypassing
LiteLlm, for better reliability and native tool-calling support.
"""

import json
import logging
import os
import time
from pathlib import Path

from shared.config import settings

logger = logging.getLogger(__name__)

# MAX_RETRIES=5 caused cascade-retry amplification when vLLM stalled
# under KV-cache pressure (>=95%). LiteLlm would fire 5 in-flight
# retries per gen call, pushing KV to 98% and deadlocking the engine.
# Dropped to 2. LITELLM_CALL_TIMEOUT_S tolerates slow gens (5 tok/s at
# KV 97%) up to 30 min without tripping LiteLlm's default 600s cutoff.
MAX_RETRIES = 2
LITELLM_CALL_TIMEOUT_S = 1800.0
_API_USAGE_PATH = Path(__file__).resolve().parent.parent / "api_usage.jsonl"


# Gemini rotation + RPM/TPM/RPD throttle + cache lives in shared.gemini_pool.
# See that module for the full implementation.


# ── Helpers ──────────────────────────────────────────────────────────

def _is_local_qwen_openai_model(model_name: str) -> bool:
    return bool(model_name) and model_name.startswith("openai/qwen3")


def _is_gemini_model(model_name: str) -> bool:
    # Accept both bare ("gemini-2.5-flash-lite") and litellm-prefixed
    # ("gemini/gemini-2.5-flash-lite") forms.  Without this check the
    # litellm-prefixed form would fall through to litellm-via-vLLM and
    # the user simulator would return its fallback reply.
    if not model_name:
        return False
    return model_name.startswith("gemini-") or model_name.startswith("gemini/")


def _strip_gemini_prefix(model_name: str) -> str:
    """Normalize ``gemini/<name>`` -> ``<name>`` for native google-genai routing."""
    if model_name and model_name.startswith("gemini/"):
        return model_name.split("/", 1)[1]
    return model_name


def _is_openrouter_model(model_name: str) -> bool:
    return bool(model_name) and model_name.startswith("openrouter/")


def _is_dashscope_model(model_name: str) -> bool:
    """Alibaba Cloud Bailian (DashScope) models — no proxy needed."""
    return bool(model_name) and model_name.startswith("dashscope/")


def _openrouter_kwargs(model_name: str) -> dict:
    """Kwargs for OpenRouter models: disable reasoning to avoid wasted tokens."""
    if not _is_openrouter_model(model_name):
        return {}
    return {
        "extra_body": {
            "reasoning": {"enabled": False},
        }
    }


def _local_qwen_kwargs(model_name: str, caller_temperature: float = 0.7) -> dict:
    """Match Qwen3.5 official recommended sampling parameters.

    Qwen3.5 instruction (non-thinking) mode recommended params:
      temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
    Source: https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B-FP8

    When the caller explicitly asks for greedy (caller_temperature < 0.01),
    switch to deterministic sampling so user-sim reproducibility holds.
    Important for ask_user caching and for aligning with Gemini usim which
    runs at temperature=0.

    Always disables thinking mode via chat_template_kwargs to avoid
    message.content=None with reasoning-only payloads.
    """
    if not _is_local_qwen_openai_model(model_name):
        return {}
    if caller_temperature is not None and caller_temperature < 0.01:
        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "extra_body": {
                "top_k": -1,
                "min_p": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        }
    # Match Qwen3.6 Instruct-Mode (non-thinking) recipe verbatim:
    # https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
    return {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    }


def _adk_max_tokens(model_name: str) -> int:
    """ADK generation budget.

    Local Qwen via vLLM can handle 4096 output tokens fine — the earlier
    128-token cap was overly conservative and caused tool-call JSON
    truncation (SQL queries + function-call wrappers easily exceed 128).
    """
    return 4096


def _extract_message_text(resp) -> str:
    msg = resp.choices[0].message
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        if text_parts:
            return "".join(text_parts).strip()

    # Fallback: some local reasoning models may still return reasoning-only data.
    reasoning = getattr(msg, "reasoning", None)
    if isinstance(reasoning, str):
        return reasoning.strip()
    return ""


def _log_api_usage(model: str, prompt_tokens: int, completion_tokens: int,
                   caller: str = ""):
    """Append one line to the local API usage ledger (gitignored)."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "caller": caller,
    }
    try:
        with open(_API_USAGE_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Could not write API usage log: %s", e)


# ── Gemini helpers ───────────────────────────────────────────────────

def _ensure_gemini_proxy():
    """Optional HTTP proxy for Google API (location restriction) and SOCKS
    proxy cleanup. No-op unless GEMINI_HTTP_PROXY is set."""
    proxy_url = os.environ.get("GEMINI_HTTP_PROXY")
    if proxy_url:
        os.environ.setdefault("http_proxy", proxy_url)
        os.environ.setdefault("https_proxy", proxy_url)
    os.environ.pop("all_proxy", None)
    os.environ.pop("ALL_PROXY", None)


# ── Gemini (google-genai, direct) ───────────────────────────────────

def _call_gemini(messages: list, model_name: str, temperature: float = 0,
                 max_tokens: int = 1024) -> str:
    """Call a Gemini model via the shared GeminiPool.

    Delegates rotation, RPM/TPM/RPD throttling, caching, and retry to
    ``shared.gemini_pool``. Kept here for backward compatibility with
    callers that pass OpenAI-style messages lists.
    """
    from shared.gemini_pool import get_pool

    # Strip the litellm-style "gemini/" prefix if present — google-genai
    # expects bare model names.
    normalized = _strip_gemini_prefix(model_name)

    # Cache namespace default matches caller identity so different call
    # sites don't share entries unless they should.
    return get_pool().call(
        prompt=messages,
        model=normalized,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_namespace="llm._call_gemini",
        caller="llm._call_gemini",
    )


# ── Main entry points ───────────────────────────────────────────────

# Try local override first (gitignored, not in release)
try:
    from shared._local_provider import call_llm, build_adk_model
except ImportError:
    # Default: LiteLlm for local/OpenAI models, google-genai for Gemini
    def call_llm(messages: list, model_name: str = None, temperature: float = 0,
                 max_tokens: int = 1024) -> str:
        """Call LLM.  Routes Gemini models to google-genai, others to LiteLlm."""
        model_name = model_name or settings.system_agent_model

        if _is_gemini_model(model_name):
            return _call_gemini(messages, model_name, temperature, max_tokens)

        import litellm
        kwargs = dict(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=MAX_RETRIES,
        )

        if _is_dashscope_model(model_name):
            # Alibaba Cloud Bailian — no proxy, strip prefix
            actual_model = model_name.replace("dashscope/", "", 1)
            kwargs["model"] = f"openai/{actual_model}"
            kwargs["api_base"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            kwargs["api_key"] = os.environ.get("DASHSCOPE_API_KEY", "")
            kwargs["extra_body"] = {"enable_thinking": False}
            # Disable proxy for DashScope
            for var in ("http_proxy", "https_proxy", "all_proxy", "ALL_PROXY"):
                os.environ.pop(var, None)
        elif _is_openrouter_model(model_name):
            # OpenRouter: use OPENROUTER_API_KEY, skip local vLLM base.
            # If your network requires an HTTP proxy to reach OpenRouter,
            # set HTTPS_PROXY / HTTP_PROXY in your shell before launch.
            for var in ("all_proxy", "ALL_PROXY", "SOCKS_PROXY"):
                os.environ.pop(var, None)
            kwargs["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
            kwargs.update(_openrouter_kwargs(model_name))
        else:
            # Local vLLM backend (qwen3*, gemma*, others). litellm's OpenAI
            # client chokes on SOCKS proxy for localhost, so always clear it
            # for any model served via our local vLLM endpoint.
            is_local_vllm = bool(model_name) and model_name.startswith("openai/")
            if is_local_vllm:
                for var in ("all_proxy", "ALL_PROXY", "SOCKS_PROXY"):
                    os.environ.pop(var, None)
            kwargs.update(_local_qwen_kwargs(model_name, caller_temperature=temperature))
            if settings.litellm_api_base:
                kwargs["api_base"] = settings.litellm_api_base
            if settings.litellm_api_key:
                kwargs["api_key"] = settings.litellm_api_key

        resp = litellm.completion(**kwargs)
        return _extract_message_text(resp)

    def build_adk_model(model_name: str = None):
        """Build ADK-compatible model.

        Routes gemini-* models to the native ADK Gemini backend (best
        tool-calling support). All other models go through LiteLlm.
        """
        model_name = model_name or settings.system_agent_model

        if _is_gemini_model(model_name):
            # Native ADK Gemini — needs GOOGLE_API_KEY + HTTP proxy
            _ensure_gemini_proxy()
            from google.adk.models.google_llm import Gemini
            return Gemini(model=_strip_gemini_prefix(model_name))

        from google.adk.models.lite_llm import LiteLlm

        if _is_dashscope_model(model_name):
            # Alibaba Cloud Bailian — no proxy
            actual_model = model_name.replace("dashscope/", "", 1)
            ds_key = os.environ.get("DASHSCOPE_API_KEY", "")
            # Disable proxy for DashScope
            for var in ("http_proxy", "https_proxy", "all_proxy", "ALL_PROXY"):
                os.environ.pop(var, None)
            return LiteLlm(
                model=f"openai/{actual_model}",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=ds_key,
                max_tokens=_adk_max_tokens(model_name),
                num_retries=MAX_RETRIES,
                extra_body={"enable_thinking": False},
            )

        if _is_openrouter_model(model_name):
            # OpenRouter: use OPENROUTER_API_KEY, disable reasoning
            or_key = os.environ.get("OPENROUTER_API_KEY", "")
            or_kwargs = _openrouter_kwargs(model_name)
            return LiteLlm(
                model=model_name,
                max_tokens=_adk_max_tokens(model_name),
                num_retries=MAX_RETRIES,
                api_key=or_key,
                **or_kwargs,
            )

        # Local vLLM backend — drop SOCKS so litellm's OpenAI client can
        # reach 127.0.0.1 without the socksio dependency.
        if model_name and model_name.startswith("openai/"):
            for var in ("all_proxy", "ALL_PROXY", "SOCKS_PROXY"):
                os.environ.pop(var, None)
        kwargs = dict(
            model=model_name,
            max_tokens=_adk_max_tokens(model_name),
            num_retries=MAX_RETRIES,
            timeout=LITELLM_CALL_TIMEOUT_S,
        )
        kwargs.update(_local_qwen_kwargs(model_name))
        if settings.litellm_api_base:
            kwargs["api_base"] = settings.litellm_api_base
        if settings.litellm_api_key:
            kwargs["api_key"] = settings.litellm_api_key
        return LiteLlm(**kwargs)
