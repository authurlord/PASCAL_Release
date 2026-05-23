# PASCAL Model Cards

PASCAL uses two model families. The system agent runs the local Qwen
weights via vLLM; the user simulator runs Gemini hosted by Google.

## 1. System agent — Qwen3.6 (FP8)

Set `PASCAL_MODEL_PATH` to a local checkpoint dir or to the HF id. The
vLLM launcher will load whichever path resolves.

### 1.1 Qwen3.6-35B-A3B-FP8 — **paper anchor**

Active-3B mixture-of-experts, FP8 weights for low-VRAM serving.

| Source | URL |
|---|---|
| HuggingFace | https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8 |
| ModelScope | https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B-FP8 |

**Hardware.** Two GPUs with ≥40 GB total VRAM (tested on 2× A100-80G,
2× H100-80G). Tensor-parallel size 2 with `--gpu-memory-utilization
0.92` fits 128 K context + KV cache in FP8 inside ~75 GiB total.

**vLLM launch.** `bash scripts/start_vllm_qwen36_35b.sh`. Set
`PASCAL_MODEL_PATH=/path/to/Qwen3.6-35B-A3B-FP8` if not downloaded
to the HF cache.

**Reasoning parser.** `--reasoning-parser qwen3` is required so the
agent's tool-call traces are parsed correctly.

### 1.2 Qwen3.6-27B-FP8 — dense alternative

Slightly smaller and dense (no MoE). Useful if you only have 48 GB
VRAM total or want a faster smoke test.

| Source | URL |
|---|---|
| HuggingFace | https://huggingface.co/Qwen/Qwen3.6-27B-FP8 |
| ModelScope | https://modelscope.cn/models/Qwen/Qwen3.6-27B-FP8 |

**vLLM launch.** `bash scripts/start_vllm_qwen36_27b.sh`. Two GPUs with
≥48 GB total VRAM; `--gpu-memory-utilization 0.92`.

---

## 2. User simulator — Gemini 2.5 Flash Lite

The user simulator is **fixed to** `gemini/gemini-2.5-flash-lite`. We
do not ship a key. Provide your own via env var(s):

```bash
# At minimum:
export GOOGLE_API_KEY=<your_gemini_key>

# Optional: up to 9 keys for round-robin (raises throughput on long
# parallel runs). shared/gemini_pool.py reads GOOGLE_API_KEY,
# GOOGLE_API_KEY_2, ..., GOOGLE_API_KEY_9.
export GOOGLE_API_KEY_2=<key2>
```

Get a key at https://aistudio.google.com/apikey . The free tier (Flash
Lite) is sufficient for `lite_300` runs at concurrency 8; bursty
parallel runs at concurrency ≥ 16 benefit from multiple keys.

Per-key rate limits are configured in `src/shared/gemini_limits.json`
(60-second sliding-window RPM bucket per key). The pool blocks in
Python when a key's bucket is full, rather than triggering 429 retries.
Override via the `GEMINI_LIMITS_CONFIG` env var if your tier differs.

**Outbound network.** Google's API is geo-restricted in some regions;
set `GEMINI_HTTP_PROXY=http://host:port` to route Gemini calls through
an HTTP proxy. SOCKS proxies are disabled (the `socksio` dependency is
not bundled).
