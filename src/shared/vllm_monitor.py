"""vLLM engine metrics monitor.

Reads Prometheus metrics from a vLLM OpenAI-compatible server and exposes
the running/waiting request counts that drive the AdaptiveTaskScheduler.

The metrics endpoint is unauthenticated on most vLLM deployments, but we
pass the API key in case the server is launched with --api-key.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class VLLMSnapshot:
    """One sample from the vLLM metrics endpoint."""
    ts: float
    running: int = 0           # requests currently in the execution batch
    waiting: int = 0           # requests queued waiting for the batch
    success_total: int = 0     # cumulative successful completions
    prompt_tokens: int = 0
    generation_tokens: int = 0
    reachable: bool = True

    @property
    def in_flight(self) -> int:
        """Total requests the engine is aware of (running + queued)."""
        return self.running + self.waiting


class VLLMMetricsMonitor:
    """Background poller that caches the latest vLLM snapshot.

    Thread-safe single-reader model: the scheduler thread periodically
    asks for ``latest()``; the monitor thread refreshes the snapshot in
    the background every ``interval_s`` seconds.
    """

    _RUNNING_RE = re.compile(r'^vllm:num_requests_running\{[^}]*\}\s+([\d.]+)', re.M)
    _WAITING_RE = re.compile(r'^vllm:num_requests_waiting\{[^}]*\}\s+([\d.]+)', re.M)
    _SUCCESS_RE = re.compile(
        r'^vllm:request_success_total\{[^}]*finished_reason="stop"[^}]*\}\s+([\d.]+)',
        re.M,
    )
    _PROMPT_RE = re.compile(r'^vllm:prompt_tokens_total\{[^}]*\}\s+([\d.]+)', re.M)
    _GEN_RE = re.compile(r'^vllm:generation_tokens_total\{[^}]*\}\s+([\d.]+)', re.M)

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        interval_s: float = 1.0,
        timeout_s: float = 2.0,
    ):
        self.endpoint = endpoint.rstrip("/") + "/metrics"
        self.api_key = api_key
        self.interval_s = interval_s
        self.timeout_s = timeout_s

        self._lock = threading.Lock()
        self._latest = VLLMSnapshot(ts=0.0, reachable=False)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> "VLLMMetricsMonitor":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="vllm-metrics-monitor", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "VLLMMetricsMonitor":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # -- public API -------------------------------------------------------

    def latest(self) -> VLLMSnapshot:
        with self._lock:
            return self._latest

    # -- internals --------------------------------------------------------

    def _run(self) -> None:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # Local vLLM — never use proxies.
        client = httpx.Client(
            timeout=self.timeout_s, trust_env=False, headers=headers
        )
        try:
            while not self._stop.is_set():
                snap = self._fetch(client)
                with self._lock:
                    self._latest = snap
                # Use wait so stop() interrupts promptly.
                self._stop.wait(self.interval_s)
        finally:
            client.close()

    def _fetch(self, client: httpx.Client) -> VLLMSnapshot:
        now = time.time()
        try:
            resp = client.get(self.endpoint)
            if resp.status_code != 200:
                return VLLMSnapshot(ts=now, reachable=False)
            text = resp.text
        except Exception as e:
            logger.debug(f"vLLM metrics fetch error: {e}")
            return VLLMSnapshot(ts=now, reachable=False)

        def _match(pat: re.Pattern) -> int:
            m = pat.search(text)
            if not m:
                return 0
            return int(float(m.group(1)))

        return VLLMSnapshot(
            ts=now,
            running=_match(self._RUNNING_RE),
            waiting=_match(self._WAITING_RE),
            success_total=_match(self._SUCCESS_RE),
            prompt_tokens=_match(self._PROMPT_RE),
            generation_tokens=_match(self._GEN_RE),
            reachable=True,
        )


def default_monitor() -> VLLMMetricsMonitor:
    """Helper that constructs a monitor from standard env vars."""
    base = os.environ.get("LITELLM_API_BASE", "http://127.0.0.1:8000/v1")
    # Strip trailing /v1 for metrics root
    if base.endswith("/v1"):
        base = base[:-3]
    api_key = os.environ.get("LITELLM_API_KEY", "")
    return VLLMMetricsMonitor(endpoint=base, api_key=api_key)
