# VK Compute

A Vulkan-based LLM inference engine, written from scratch in C and GLSL compute shaders — no ML framework involved: no PyTorch, no CUDA, no llama.cpp. It runs Qwen3.5-2B, Qwen3.5-9B, and the Qwen3.6-35B-A3B MoE locally, tested on an **AMD RX 580 8 GB** and an **NVIDIA RTX 4060**.

The engine implements the models' full text stack: hybrid attention (gated delta-net and full attention), per-layer FP16/INT8/INT4 quantization, MoE expert offloading (experts split between VRAM and host RAM), chunked prefill, look-ahead decode, and an on-GPU sampler (temperature, top-k, top-p, min-p, repetition penalty). To fit the vocabulary in VRAM, the 248,320-token head is pruned to 102,400 rows.

The web UI ships as an npm package with a prebuilt engine binary: model loader with validation, per-layer quantization editor, sampling controls, and a streaming chat playground.

## Installation

Requires **Windows x64**, **Node.js 18+**, and a Vulkan-capable GPU with a current driver.

```bash
npm install -g @h4zel/vk-compute
```

Models live in a folder you own — the command scans `./model` and `./pruned-vocab` relative to where you run it:

```
my-models/
   model/          # safetensors dirs, .gguf, or .hqm files
   pruned-vocab/   # mapping.npy + pruned tokenizer
```

## Usage

```bash
cd my-models
vk-compute
```

The UI opens automatically at `http://127.0.0.1:8787`.

| Option               | Description                                 |
| -------------------- | ------------------------------------------- |
| `-p, --port <port>`  | port to listen on (default 8787)            |
| `-m, --models <dir>` | model directory to scan (default `./model`) |
| `--no-open`          | do not open the browser                     |

### Model tab

- Pick a model from the dropdown (scanned from your models folder) or type a path, then probe it — the engine validates safetensors shard/config consistency, and reads GGUF/HQM metadata.
- The **Quantization** accordion sets a per-layer `attn`/`ffn` FP16/INT8/INT4 mix (with "set all" presets), plus embed/LM-head quant and, for MoE models, how many experts stay in VRAM. Quantization is baked in for `.hqm` files, so the editor locks.
- **Max context** (bounded by the model's `max_position_embeddings`) and **prefill chunk** size.
- **Prune vocab** bootstraps a fresh model dir (gathers the pruned vocab from the shards); **Export HQM** writes a single-file quantized copy for fast reloads.

### Sampling tab

A `Sampling` checkbox reveals temperature, top-k, top-p, min-p, repetition penalty, penalty length, presence penalty, and seed. A `Max new tokens` checkbox reveals a slider capped at the context size. Thinking mode and the system prompt are configured here.

### Chat tab

Streaming chat with thinking support, tokens/second readout, and a Clear button.
