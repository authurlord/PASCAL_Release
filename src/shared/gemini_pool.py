"""Reusable Gemini call pool: multi-key rotation + per-key RPM/TPM/RPD
throttle + persistent cache + transient-failure retry.

Use this instead of hand-rolling Gemini calls in each caller. Limits per
model come from ``gemini_limits.json`` — edit that file when Google
changes the policy (the raw numbers for preview models are not published
and must be refreshed from https://aistudio.google.com/rate-limit).

Typical use:

    from shared.gemini_pool import get_pool
    text = get_pool().call(prompt="Summarize X", model="gemini-3.1-flash-lite-preview")

Environment variables read at import time:
  GOOGLE_API_KEY, GOOGLE_API_KEY_2 .. GOOGLE_API_KEY_9  (first N present are used)
  GEMINI_RPM_LIMIT     (override config rpm for all keys; default: per-model config)
  GEMINI_TPM_LIMIT     (override config tpm)
  GEMINI_RPD_LIMIT     (override config rpd)
  GEMINI_LIMITS_CONFIG (path to gemini_limits.json, default: sibling file)
  GEMINI_CACHE_DIR     (default <repo>/.cache/gemini_pool)
  GEMINI_CACHE_DISABLE (default 0)
  GEMINI_MAX_RETRIES   (default 5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import diskcache

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "gemini_pool"
_USAGE_LEDGER = _REPO_ROOT / "api_usage.jsonl"
_DEFAULT_LIMITS_CONFIG = Path(__file__).resolve().parent / "gemini_limits.json"


def _load_limits_config(path: Path | str | None = None) -> dict:
    p = Path(
        path
        or os.environ.get("GEMINI_LIMITS_CONFIG", str(_DEFAULT_LIMITS_CONFIG))
    )
    if not p.exists():
        return {"default": {"rpm": 15, "tpm": 250000, "rpd": 1000}, "models": {}}
    with open(p) as f:
        return json.load(f)


def _limits_for_model(cfg: dict, model: str) -> dict:
    """Return {rpm, tpm, rpd} with env-var overrides."""
    m = (cfg.get("models", {}) or {}).get(model)
    base = cfg.get("default", {}) or {}
    merged = {**base, **(m or {})}
    # Env overrides apply to the active pool (not per-model). Use with care.
    for key, env in [("rpm", "GEMINI_RPM_LIMIT"), ("tpm", "GEMINI_TPM_LIMIT"), ("rpd", "GEMINI_RPD_LIMIT")]:
        raw = os.environ.get(env)
        if raw:
            try:
                merged[key] = int(raw)
            except ValueError:
                pass
    return {
        "rpm": int(merged.get("rpm", 15)),
        "tpm": int(merged.get("tpm", 250_000)),
        "rpd": int(merged.get("rpd", 1000)),
    }


_RETRIABLE_MARKERS = (
    "429", "RESOURCE_EXHAUSTED", "rate limit", "RateLimit",
    "timeout", "Timeout", "500", "502", "503", "504",
    "UNAVAILABLE", "DEADLINE_EXCEEDED", "Connection",
    "connection", "reset", "EOF", "temporarily",
    "overloaded", "Internal", "INTERNAL",
)


def _load_keys() -> list[str]:
    keys: list[str] = []
    primary = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    if primary:
        keys.append(primary)
    for n in range(2, 10):
        extra = (os.environ.get(f"GOOGLE_API_KEY_{n}") or "").strip()
        if extra and extra not in keys:
            keys.append(extra)
    return keys


class GeminiPool:
    """Thread-safe Gemini-API pool.

    * Rotation — round-robin over keys, skipping any whose sliding-window
      RPM bucket is full. Blocks in Python until some key has capacity.
    * Cache — `diskcache.Cache` keyed on (namespace, model, params, content
      hash). Cache is keyed by content only, not by key, so a 3.6-usim and
      a Gemini-usim writing the same prompt do NOT share entries
      (callers pass a distinct `cache_namespace`).
    * Retry — transient errors (rate limits, 5xx, timeouts) retried with
      exponential backoff. Non-retriable errors fail fast after one retry.

    Reserved for Google Gemini endpoints. For local-vLLM calls use
    `shared.llm.call_llm` which keeps the existing routing.
    """

    def __init__(
        self,
        keys: list[str] | None = None,
        limits_config: Path | str | None = None,
        cache_dir: Path | str | None = None,
        cache_enabled: bool | None = None,
        max_retries: int | None = None,
    ):
        self.keys = keys if keys is not None else _load_keys()
        if not self.keys:
            raise RuntimeError(
                "GeminiPool: no keys; set GOOGLE_API_KEY (and optionally "
                "GOOGLE_API_KEY_2..9)"
            )
        self.limits_config = _load_limits_config(limits_config)
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.environ.get("GEMINI_MAX_RETRIES", "5"))
        )
        cdir = (
            cache_dir
            if cache_dir is not None
            else os.environ.get("GEMINI_CACHE_DIR", str(_DEFAULT_CACHE_DIR))
        )
        self.cache_dir = Path(cdir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        enabled_env = os.environ.get("GEMINI_CACHE_DISABLE", "0") != "1"
        self.cache_enabled = cache_enabled if cache_enabled is not None else enabled_env
        self._cache = diskcache.Cache(str(self.cache_dir)) if self.cache_enabled else None
        # Per-(key, model) buckets: RPM window, TPM window, RPD counter.
        self._rpm_buckets: dict[tuple[str, str], deque] = {}
        self._tpm_windows: dict[tuple[str, str], deque] = {}  # deque of (ts, tokens)
        self._rpd_counts: dict[tuple[str, str], tuple[str, int]] = {}  # (utc_date, count)
        self._429_until: dict[str, float] = {}  # explicit cooldowns from server
        self._lock = threading.Lock()
        self._proxy_prepared = False
        logger.info(
            "GeminiPool ready: keys=%d, cache=%s (%s), limits_config=%s",
            len(self.keys),
            "on" if self.cache_enabled else "off",
            self.cache_dir,
            (limits_config or os.environ.get("GEMINI_LIMITS_CONFIG") or _DEFAULT_LIMITS_CONFIG),
        )

    def _get_limits(self, model: str) -> dict:
        return _limits_for_model(self.limits_config, model)

    def _bucket_key(self, api_key: str, model: str) -> tuple[str, str]:
        return (api_key, model)

    # ── proxy + env helpers ────────────────────────────────────────
    def _ensure_proxy(self) -> None:
        """Optional HTTP proxy passthrough for the Google API. Set
        GEMINI_HTTP_PROXY in your shell if you need a proxy (geo-restriction
        bypass). Always drops SOCKS proxies because the ``socksio``
        dependency is not bundled."""
        if self._proxy_prepared:
            return
        proxy_url = os.environ.get("GEMINI_HTTP_PROXY")
        if proxy_url:
            os.environ.setdefault("http_proxy", proxy_url)
            os.environ.setdefault("https_proxy", proxy_url)
        for var in ("all_proxy", "ALL_PROXY", "SOCKS_PROXY"):
            os.environ.pop(var, None)
        self._proxy_prepared = True

    # ── cache ──────────────────────────────────────────────────────
    @staticmethod
    def _content_hash(payload: Any) -> str:
        """Stable hash of a prompt/messages payload."""
        if isinstance(payload, str):
            raw = payload.encode("utf-8")
        else:
            raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_key(
        self,
        *,
        namespace: str,
        model: str,
        prompt: Any,
        temperature: float,
        max_tokens: int,
    ) -> str:
        material = "||".join(
            [
                namespace,
                model,
                f"T={temperature:.3f}",
                f"mx={max_tokens}",
                self._content_hash(prompt),
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    # ── bucket / key selection ─────────────────────────────────────
    @staticmethod
    def _utc_date() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _acquire_key_for_model(self, model: str, est_tokens: int) -> str:
        """Block until some key has (RPM free) AND (TPM headroom for est_tokens)
        AND (RPD not yet exhausted) for the given model. Returns that key."""
        limits = self._get_limits(model)
        rpm_cap, tpm_cap, rpd_cap = limits["rpm"], limits["tpm"], limits["rpd"]
        today = self._utc_date()
        while True:
            with self._lock:
                now = time.time()
                # Evict old entries from every bucket/window.
                for k in self.keys:
                    bk = self._bucket_key(k, model)
                    rpm_q = self._rpm_buckets.setdefault(bk, deque())
                    while rpm_q and now - rpm_q[0] >= 60:
                        rpm_q.popleft()
                    tpm_q = self._tpm_windows.setdefault(bk, deque())
                    while tpm_q and now - tpm_q[0][0] >= 60:
                        tpm_q.popleft()
                    d, c = self._rpd_counts.get(bk, (today, 0))
                    if d != today:
                        self._rpd_counts[bk] = (today, 0)

                # Pick the key with the earliest RPM-free slot that still
                # satisfies TPM + RPD constraints.
                eligible: list[str] = []
                for k in self.keys:
                    if now < self._429_until.get(k, 0.0):
                        continue
                    bk = self._bucket_key(k, model)
                    if len(self._rpm_buckets[bk]) >= rpm_cap:
                        continue
                    used_tpm = sum(t for _, t in self._tpm_windows[bk])
                    if used_tpm + est_tokens > tpm_cap:
                        continue
                    _, rpd_count = self._rpd_counts.setdefault(bk, (today, 0))
                    if rpd_count >= rpd_cap:
                        continue
                    eligible.append(k)
                if eligible:
                    chosen = min(
                        eligible,
                        key=lambda k: len(self._rpm_buckets[self._bucket_key(k, model)]),
                    )
                    bk = self._bucket_key(chosen, model)
                    self._rpm_buckets[bk].append(now)
                    self._tpm_windows[bk].append((now, est_tokens))
                    date_str, cnt = self._rpd_counts.setdefault(bk, (today, 0))
                    self._rpd_counts[bk] = (date_str, cnt + 1)
                    return chosen

                # Nothing available: compute shortest wait.
                waits: list[float] = []
                for k in self.keys:
                    bk = self._bucket_key(k, model)
                    cd = max(0.0, self._429_until.get(k, 0.0) - now)
                    if cd > 0:
                        waits.append(cd)
                    rpm_q = self._rpm_buckets[bk]
                    if len(rpm_q) >= rpm_cap and rpm_q:
                        waits.append(rpm_q[0] + 60 - now)
                    tpm_q = self._tpm_windows[bk]
                    used_tpm = sum(t for _, t in tpm_q)
                    if used_tpm + est_tokens > tpm_cap and tpm_q:
                        waits.append(tpm_q[0][0] + 60 - now)
                    _, rpd_count = self._rpd_counts.setdefault(bk, (today, 0))
                    if rpd_count >= rpd_cap:
                        # Roll at UTC midnight.
                        secs_utc = time.gmtime()
                        day_left = 86400 - (
                            secs_utc.tm_hour * 3600 + secs_utc.tm_min * 60 + secs_utc.tm_sec
                        )
                        waits.append(day_left)
                wait = max(0.1, min(waits)) if waits else 0.5
            logger.debug("GeminiPool: all keys capped for model=%s, sleeping %.2fs", model, wait)
            time.sleep(wait)

    def _mark_429(self, key: str, retry_after: float = 30.0) -> None:
        with self._lock:
            until = time.time() + max(1.0, retry_after)
            cur = self._429_until.get(key, 0.0)
            self._429_until[key] = max(cur, until)

    def _reconcile_tpm(self, api_key: str, model: str, estimated: int, actual: int) -> None:
        """Replace the most-recent estimated-token entry with the server-reported
        actual token count so the TPM bucket stays accurate."""
        if actual <= 0 or actual == estimated:
            return
        with self._lock:
            tpm_q = self._tpm_windows.get(self._bucket_key(api_key, model))
            if not tpm_q:
                return
            # Replace the newest matching estimate with actual.
            for i in range(len(tpm_q) - 1, -1, -1):
                ts, tok = tpm_q[i]
                if tok == estimated:
                    tpm_q[i] = (ts, actual)
                    return

    @staticmethod
    def _parse_retry_after(err_text: str) -> float:
        """Pull 'Please retry in Xs' out of a 429 message; fall back to 30s."""
        import re

        m = re.search(r"retry in ([0-9.]+)s", err_text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return 30.0

    # ── usage ledger ───────────────────────────────────────────────
    def _log_usage(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        caller: str,
        cache_hit: bool = False,
    ) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "caller": caller,
            "cache_hit": cache_hit,
        }
        try:
            with open(_USAGE_LEDGER, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("usage-ledger write failed: %s", e)

    # ── core call ──────────────────────────────────────────────────
    def call(
        self,
        prompt: Any,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cache_namespace: str = "default",
        use_cache: bool = True,
        caller: str = "gemini_pool.call",
    ) -> str:
        """Send a Gemini generateContent request. Returns response text.

        ``prompt`` can be:
          - a string (sent as-is as single user turn), or
          - a list of OpenAI-style messages ``[{role, content}, ...]``
            (converted into a single concatenated prompt to match the
            existing ``_call_gemini`` behavior)
        """
        self._ensure_proxy()

        # Normalize prompt to string for countTokens / cache / downstream.
        if isinstance(prompt, list):
            parts = []
            for m in prompt:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    parts.append(f"[System]\n{content}")
                elif role == "assistant":
                    parts.append(f"[Assistant]\n{content}")
                else:
                    parts.append(str(content))
            prompt_str = "\n\n".join(parts)
        else:
            prompt_str = str(prompt)

        ck = None
        if use_cache and self.cache_enabled:
            ck = self._cache_key(
                namespace=cache_namespace,
                model=model,
                prompt=prompt_str,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            cached = self._cache.get(ck) if self._cache is not None else None
            if cached is not None:
                self._log_usage(
                    model=model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    caller=caller,
                    cache_hit=True,
                )
                logger.debug("GeminiPool CACHE HIT [%s] %s...", cache_namespace, prompt_str[:60])
                return cached

        # Import lazily so callers who never touch Gemini don't pay for it.
        from google.genai import Client, types as genai_types

        # Estimate tokens for TPM accounting: rough 4 chars/token heuristic
        # for the prompt + max_tokens for the output.
        est_prompt_tokens = max(8, len(prompt_str) // 4)
        est_total_tokens = est_prompt_tokens + max_tokens

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            api_key = self._acquire_key_for_model(model, est_total_tokens)
            try:
                client = Client(api_key=api_key)
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt_str,
                    config=genai_types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                text = (resp.text or "").strip()
                usage = getattr(resp, "usage_metadata", None)
                prompt_tok = getattr(usage, "prompt_token_count", 0) or 0
                out_tok = getattr(usage, "candidates_token_count", 0) or 0
                actual_total = prompt_tok + out_tok
                # Reconcile our TPM estimate with the actual usage.
                if actual_total > 0:
                    self._reconcile_tpm(api_key, model, est_total_tokens, actual_total)
                self._log_usage(
                    model=model,
                    prompt_tokens=prompt_tok,
                    completion_tokens=out_tok,
                    caller=caller,
                    cache_hit=False,
                )
                if text and use_cache and self.cache_enabled and ck is not None and self._cache is not None:
                    self._cache.set(ck, text)
                if not text and attempt < self.max_retries - 1:
                    last_err = ValueError("empty response")
                    logger.warning(
                        "GeminiPool: empty response on attempt %d/%d; retrying",
                        attempt + 1, self.max_retries,
                    )
                    continue
                return text
            except Exception as e:
                last_err = e
                err_msg = str(e)[:400]
                retriable = any(m in err_msg for m in _RETRIABLE_MARKERS)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg.upper():
                    retry_after = self._parse_retry_after(err_msg)
                    self._mark_429(api_key, retry_after)
                    logger.warning(
                        "GeminiPool 429 on %s..., cooling %ss",
                        api_key[:10], retry_after,
                    )
                if not retriable and attempt >= 1:
                    logger.error("GeminiPool non-retriable: %s", err_msg[:200])
                    raise
                if attempt < self.max_retries - 1:
                    wait = min(60.0, 2.0 * (2 ** attempt))
                    logger.warning(
                        "GeminiPool attempt %d/%d failed: %s; retry in %.1fs",
                        attempt + 1, self.max_retries, err_msg[:120], wait,
                    )
                    time.sleep(wait)
        logger.error("GeminiPool exhausted retries: %s", last_err)
        if last_err is not None:
            raise last_err
        raise RuntimeError("GeminiPool: no response and no exception (unreachable)")

    def count_tokens(self, prompt: str, model: str) -> int:
        """Use countTokens action (does not consume generate quota)."""
        self._ensure_proxy()
        from google.genai import Client

        # count_tokens has its own (looser) quota; bypass our accounting.
        key = self.keys[0]
        client = Client(api_key=key)
        r = client.models.count_tokens(model=model, contents=prompt)
        return int(getattr(r, "total_tokens", 0) or 0)

    def stats(self, model: str = "gemini-3.1-flash-lite-preview") -> dict[str, dict[str, Any]]:
        """Current per-key bucket state + any active 429 cooldowns for the model."""
        now = time.time()
        limits = self._get_limits(model)
        today = self._utc_date()
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            for k in self.keys:
                bk = self._bucket_key(k, model)
                rpm_q = self._rpm_buckets.setdefault(bk, deque())
                while rpm_q and now - rpm_q[0] >= 60:
                    rpm_q.popleft()
                tpm_q = self._tpm_windows.setdefault(bk, deque())
                while tpm_q and now - tpm_q[0][0] >= 60:
                    tpm_q.popleft()
                used_tpm = sum(t for _, t in tpm_q)
                d, cnt = self._rpd_counts.get(bk, (today, 0))
                if d != today:
                    cnt = 0
                out[k[:12] + "..."] = {
                    "rpm_used": len(rpm_q),
                    "rpm_cap": limits["rpm"],
                    "tpm_used": used_tpm,
                    "tpm_cap": limits["tpm"],
                    "rpd_used": cnt,
                    "rpd_cap": limits["rpd"],
                    "cooldown_s": round(max(0.0, self._429_until.get(k, 0.0) - now), 2),
                }
        return out


# Module-level singleton. Lazily constructed so importing this module in a
# Gemini-less subprocess does not fail on missing keys.
_singleton: GeminiPool | None = None
_singleton_lock = threading.Lock()


def get_pool() -> GeminiPool:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = GeminiPool()
    return _singleton
