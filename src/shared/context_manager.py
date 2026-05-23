"""Context window management for long-running agent sessions.

Two-layer protection:
1. FRONT-END (before_model_callback): tokenizer pre-check, proactive compression
2. BACK-END (after 500): catch context overflow, compress, retry

Uses Qwen tokenizer for accurate token counting (~5ms per call).
"""

import logging
import os
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded tokenizer
_tokenizer = None
_tokenizer_load_failed = False

def _default_model_path() -> str:
    # Env override (CONTEXT_MODEL_PATH or ADK_MODEL_PATH). If neither is
    # set, fall back to the HF identifier — transformers will download
    # it on first call. Set CONTEXT_TOKENIZER_DISABLE=1 to skip entirely.
    for v in ("CONTEXT_MODEL_PATH", "ADK_MODEL_PATH"):
        p = os.environ.get(v)
        if p:
            return p
    return "Qwen/Qwen3.6-35B-A3B-FP8"


_tokenizer_lock = None  # lazy-init to avoid import-time cost


def _get_tokenizer():
    """Thread-safe lazy tokenizer load.

    Disable entirely via CONTEXT_TOKENIZER_DISABLE=1 (falls back to char estimate).
    Useful on NFS-mounted storage where concurrent first-call loads
    stall the agent service for minutes under bursty concurrency.
    """
    global _tokenizer, _tokenizer_load_failed, _tokenizer_lock
    if os.environ.get("CONTEXT_TOKENIZER_DISABLE", "0") == "1":
        if not _tokenizer_load_failed:
            _tokenizer_load_failed = True
            logger.warning("Context manager: tokenizer disabled via CONTEXT_TOKENIZER_DISABLE=1, using char estimate")
        return None
    if _tokenizer is None and not _tokenizer_load_failed:
        # lazy-init lock on first call
        if _tokenizer_lock is None:
            import threading
            _tokenizer_lock = threading.Lock()
        with _tokenizer_lock:
            if _tokenizer is None and not _tokenizer_load_failed:
                try:
                    from transformers import AutoTokenizer
                    path = _default_model_path()
                    _tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
                    logger.info("Context manager: tokenizer loaded from %s", path)
                except Exception as e:
                    _tokenizer_load_failed = True
                    logger.warning("Context manager: tokenizer load failed (%s), using char estimate", e)
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens using Qwen tokenizer, fallback to char estimate."""
    tok = _get_tokenizer()
    if tok is not None:
        return len(tok.encode(text))
    # Fallback: ~3.3 chars per token for mixed SQL/English
    return len(text) // 3


def estimate_context_tokens(messages: list) -> int:
    """Estimate total tokens in a message list (OpenAI format)."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += count_tokens(str(part.get("text", "")))
        # Role/name overhead
        total += 4
    return total


def compress_tool_history(
    messages: list,
    max_tokens: int,
    output_reserve: int = 4096,
    keep_recent: int = 15,
) -> Tuple[list, int]:
    """Compress tool call history to fit within token budget.

    Strategy (progressive):
      Level 0: No compression (under budget)
      Level 1: Summarize tool results older than keep_recent turns
      Level 2: Reduce keep_recent to 8
      Level 3: Reduce keep_recent to 4, truncate long results

    Args:
        messages: OpenAI-format message list.
        max_tokens: Model's max context length.
        output_reserve: Tokens reserved for output generation.
        keep_recent: Number of recent tool results to keep full.

    Returns:
        (compressed_messages, compression_level)
    """
    budget = max_tokens - output_reserve
    current = estimate_context_tokens(messages)

    if current <= budget:
        return messages, 0

    # Level 1: Summarize old tool results
    compressed = _compress_at_level(messages, keep_recent=keep_recent, max_result_chars=2000)
    current = estimate_context_tokens(compressed)
    if current <= budget:
        logger.info("Context compressed at Level 1 (keep %d recent): %d tokens", keep_recent, current)
        return compressed, 1

    # Level 2: Keep fewer recent
    compressed = _compress_at_level(messages, keep_recent=8, max_result_chars=1000)
    current = estimate_context_tokens(compressed)
    if current <= budget:
        logger.info("Context compressed at Level 2 (keep 8 recent): %d tokens", current)
        return compressed, 2

    # Level 3: Aggressive
    compressed = _compress_at_level(messages, keep_recent=4, max_result_chars=500)
    current = estimate_context_tokens(compressed)
    logger.info("Context compressed at Level 3 (keep 4 recent): %d tokens", current)
    return compressed, 3


def _compress_at_level(
    messages: list,
    keep_recent: int,
    max_result_chars: int,
) -> list:
    """Compress messages by summarizing old tool interactions."""
    result = []
    # Count tool-call messages from the end
    tool_msg_indices = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        if role in ("tool", "function") or "tool_call" in content.lower():
            tool_msg_indices.append(i)

    # Messages to keep full (most recent keep_recent tool messages)
    keep_full = set(tool_msg_indices[-keep_recent:]) if tool_msg_indices else set()

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if i in keep_full or role in ("system",):
            # Keep system messages and recent tool calls intact
            result.append(msg)
        elif role in ("tool", "function"):
            # Summarize old tool result
            if isinstance(content, str) and len(content) > max_result_chars:
                # Keep first line (tool status) + truncate
                lines = content.split("\n")
                summary = lines[0][:200]
                result.append({**msg, "content": f"{summary}\n[...truncated, {len(content)} chars original]"})
            else:
                result.append(msg)
        elif role == "assistant" and i not in keep_full:
            # Truncate long assistant messages (old reasoning)
            if isinstance(content, str) and len(content) > max_result_chars:
                result.append({**msg, "content": content[:max_result_chars] + "\n[...truncated]"})
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


def is_context_overflow_error(error) -> bool:
    """Check if an error is a context overflow from vLLM."""
    err_str = str(error)
    return ("input tokens" in err_str and "context length" in err_str) or \
           ("max_model_len" in err_str) or \
           ("VLLMValidationError" in err_str and "input_tokens" in err_str)


def extract_token_counts_from_error(error) -> Optional[Tuple[int, int, int]]:
    """Extract (input_tokens, requested_output, max_context) from overflow error."""
    err_str = str(error)
    match = re.search(
        r"You passed (\d+) input tokens and requested (\d+) output tokens.*"
        r"context length is only (\d+) tokens",
        err_str,
    )
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None
