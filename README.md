# VK Compute

A Vulkan-based LLM inference engine, written from scratch in C and GLSL compute shaders. It runs **Qwen3.5-9B** locally on an **AMD RX 580 8 GB**, a GPU with no AI acceleration, at ~25 tokens per second, with no ML framework involved: no PyTorch, no CUDA, no llama.cpp.

The engine implements the model's full text stack: 32 transformer layers with hybrid attention (gated delta-net and full attention), per-layer FP16/INT8/INT4 quantization to fit 8 GB of VRAM, chunked prefill, look-ahead decode, and an on-GPU sampler (temperature, top-k, top-p, min-p, repetition penalty). The architecture is the exact same as HuggingFace's Qwen3.5-9B; greedy decode produces identical output, token for token.

To fit the vocabulary in VRAM, the model's 248,320-token head is pruned to 86,016 rows.

The engine ships with a local web UI (`npm install -g @h4zel/vk-compute`, then run `vk-compute` — see [Web UI](#web-ui)) and a Python CLI frontend.

## Project Layout

| Path | Contents |
|---|---|
| `src/`, `include/` | C engine: weight loading, op dispatch, token server, Node addon |
| `shader/` | 121 GLSL compute shaders (GEMV/GEMM, attention, sampler) |
| `vk_llm.py` | Python frontend: chat template, tokenization, streaming, sampling config |
| `webui/` | React + Vite frontend and Express server on top of the Node addon |
| `cli.js`, `pack.mjs`, `release.mjs`, `platform/win32-x64/` | npm packaging (`@h4zel/vk-compute`) |
| `docs/VK-COMPUTE-SUMMARY.md` | Full technical documentation |
| `model/` | Weights (not in git) |
| `pruned-vocab/` | Pruned-vocab artifacts: mapping.npy, pruned tokenizer (not in git) |

## Web UI

The easiest way to run the engine is the npm package — a local web server with a model loader, per-layer quantization editor, sampling controls, and a streaming chat playground. No build step, no Python.

```bash
npm install -g @h4zel/vk-compute
```

Run it from a folder that contains your models and the pruned-vocab artifacts:

```
my-models/
   model/          # safetensors dirs, .gguf, or .hqm files
   pruned-vocab/   # mapping.npy + pruned tokenizer
```

```bash
cd my-models
vk-compute
```

The UI opens automatically at `http://127.0.0.1:8787`. Options:

| Option | Description |
|---|---|
| `-p, --port <port>` | port to listen on (default 8787) |
| `-m, --models <dir>` | model directory to scan (default `./model`) |
| `--no-open` | do not open the browser |

Web UI requirements:

- Windows x64
- Node.js 18 or newer
- A Vulkan-capable GPU with a current driver (built and tuned for the RX 580 8 GB)
- A models folder as shown above

To run the web UI from a source checkout instead: `make`, then in `webui/`: `npm install`, `npm run build`, `npm start`. End-to-end test: `node test_webui.mjs` from the repo root.

## Requirements (building from source)

- Windows with MinGW-w64 / MSYS2 UCRT64 (`gcc`, `make`)
- Vulkan SDK (tested with 1.4.350)
- Node.js 18+ (for the web UI / addon build)
- Python 3.9+ with [uv](https://docs.astral.sh/uv/), plus `tokenizers` (for the Python frontend)
- A GPU with 8 GB VRAM (built and tuned for the RX 580)
- Qwen3.5 safetensors under `model/Qwen3.5-9B/` (or `model/Qwen3.5-2B/`), with the pruned-vocab artifacts (mapping, pruned tokenizer) under `pruned-vocab/`

## Build & Run (Python frontend)

```bash
# Python deps (once)
uv venv .venv
uv pip install --python .venv/Scripts/python.exe tokenizers

# Build from the repo root
make clean
make

# Validate every shader against its CPU reference
cd bin && main.exe val

# Generate (pruned 86,016-token vocab — fits both models)
.venv/Scripts/python.exe vk_llm.py model/Qwen3.5-9B 4096 "your prompt here" --think

# Original 248,320-token vocab, straight from the shards (2B only: ~1 GB embed;
# the 9B's two untied heads would OOM on 8 GB)
.venv/Scripts/python.exe vk_llm.py model/Qwen3.5-2B 4096 "your prompt here"

# First run of a fresh model dir with --prune: gathers the pruned vocab
# from the shards (skipped automatically when the weight cache or vocab/ exists)
.venv/Scripts/python.exe vk_llm.py model/Qwen3.5-2B 8192 "why the sky is blue?" --think --prune
```

Sampling behavior is set by the constants at the top of `vk_llm.py` (`is_sampling`, `temperature`, `top_k`, `top_p`, `min_p`, `rep_penalty`, `penalty_len`, `seed`).

## Documentation

`docs/VK-COMPUTE-SUMMARY.md` covers the architecture in depth: every shader's algorithm and memory layout, the server wire protocol, the VRAM budget, the validation harness, and the engineering notes behind the quantization and decode decisions.
