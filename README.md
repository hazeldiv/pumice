# Pumice

A Vulkan-based LLM inference engine, written from scratch in C and GLSL compute shaders — no ML framework involved: no PyTorch, no CUDA, no llama.cpp. It runs Qwen3.5-2B, Qwen3.5-9B, and the Qwen3.6-35B-A3B MoE locally, tested on an **AMD RX 580 8 GB** and an **NVIDIA RTX 4060**.

The engine implements the models' full text stack: hybrid attention (gated delta-net and full attention), per-layer FP16/INT8/INT4 quantization, MoE expert offloading (experts split between VRAM and host RAM), chunked prefill, look-ahead decode, and an on-GPU sampler (temperature, top-k, top-p, min-p, repetition penalty). To fit the vocabulary in VRAM, the 248,320-token head is pruned to 102,400 rows.

The web UI ships as an npm package with a prebuilt engine binary: model loader with validation, per-layer quantization editor, sampling controls, and a streaming chat playground.

## Installation

Requires **Windows x64**, **Node.js 18+**, and a Vulkan-capable GPU with a current driver.

```bash
npm install -g @h4zel/pumice
```

Models are not scanned automatically. In the **Model** tab, type either a folder or a model path into the single field and press **Scan**:

- a folder is scanned one level deep for safetensors directories and `.gguf`/`.hqm` files, which then appear in a dropdown;
- a model path (a safetensors/GGUF/HQM directory or file) is opened directly.

The last path is remembered in the browser. `pruned-vocab/` is read from `./pruned-vocab` relative to where you run `pumice` (override with `PUMICE_PRUNED_VOCAB_DIR`).

## Usage

```bash
pumice
```

The UI opens automatically at `http://127.0.0.1:8787`.

| Option               | Description                                 |
| -------------------- | ------------------------------------------- |
| `-p, --port <port>`  | port to listen on (default 8787)            |
| `-m, --models <dir>` | scan this models folder at startup (or set `PUMICE_MODELS`) |
| `--host <addr>`      | bind address (default `127.0.0.1`)          |
| `--api-key <key>`    | require `Authorization: Bearer <key>` on `/v1` (or set `PUMICE_API_KEY`) |
| `--no-open`          | do not open the browser                     |

### Agent API

The same server exposes an [OI]-compatible API on `/v1`, so an agent harness (opencode and friends) can use it directly. Point the harness at `http://127.0.0.1:8787/v1`, set the model to any id from `GET /v1/models`, and the model auto-loads on the first request.

- `GET /v1/models`
- `POST /v1/chat/completions` — streaming and non-streaming, `tools` / tool calls, `reasoning_content`, `usage`
- `POST /v1/completions` — legacy text completion

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-2B-1.6gb.hqm","messages":[{"role":"user","content":"Say hi in three words."}],"max_tokens":32}'
```

A growing conversation resumes from the KV cache instead of re-prefilling, so multi-turn agents stay fast. Overlapping requests are queued (up to 8) rather than rejected, so parallel agent tasks (titles, summaries) work; beyond that the server returns 429.

#### opencode

Add a custom provider to `opencode.json` (the model id must match `GET /v1/models`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "pumice": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Pumice (local)",
      "options": { "baseURL": "http://127.0.0.1:8787/v1" },
      "models": {
        "Qwen3.5-2B-1.6gb.hqm": {
          "name": "Qwen3.5 2B (local)",
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```

Set `limit.context` to the `maxCtx` you load with. If the server was started with `--api-key`, add `"apiKey"` to `options`. Tool calling depends on the model — a coder-tuned checkpoint calls tools far more reliably than the pruned 2B.

### Model tab

- Type a models folder or a model path into the single field and press **Scan**; a folder lists its models in a dropdown, a model path is opened directly — the engine validates safetensors shard/config consistency, and reads GGUF/HQM metadata.
- The **Quantization** accordion sets a per-layer `attn`/`ffn` FP16/INT8/INT4 mix (with "set all" presets), plus embed/LM-head quant and, for MoE models, how many experts stay in VRAM. Quantization is baked in for `.hqm` files, so the editor locks.
- **Max context** (bounded by the model's `max_position_embeddings`) and **prefill chunk** size.
- **Prune vocab** bootstraps a fresh model dir (gathers the pruned vocab from the shards); **Export HQM** writes a single-file quantized copy for fast reloads.
- If a load does not fit, the engine reports `out of GPU memory` / `out of host memory` in the status line instead of crashing; lower **Max context**, **Prefill chunk**, or the experts-in-VRAM count and try again.

### Sampling tab

A `Sampling` checkbox reveals temperature, top-k, top-p, min-p, repetition penalty, penalty length, presence penalty, and seed. A `Max new tokens` checkbox reveals a slider capped at the context size. Thinking mode and the system prompt are configured here.

### Chat tab

Streaming chat with thinking support, tokens/second readout, and a Clear button.
