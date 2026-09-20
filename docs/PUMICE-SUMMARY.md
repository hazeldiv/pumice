# Pumice — Complete Technical Summary

A Vulkan-based GPU compute engine for running LLM inference — **multi-model** (any Qwen3.5/3.6-family checkpoint, currently **Qwen3.6-35B-A3B** (MoE), **Qwen3.5 9B**, and **Qwen3.5 2B**), all model dimensions **read at runtime from config files** (no per-model recompile) — with per-layer **hybrid quantization** (Q4_1 with 32/64/128/256-element blocks, INT8, FP16) and **MoE expert offloading** (per-layer expert pools split between VRAM and host-visible RAM) on AMD RDNA1-class GPUs (RX 580: 36 CUs, wave64). Four entry points — three gated by `main.exe` plus a **Node.js native addon** (`bin/pumice.node`) that backs the React web UI:

- **Server mode** (`main.exe`, default): a **persistent** inference daemon. It loads the real safetensors weights once (§4.7), then serves repeated "tokenize — generate" requests over a length-prefixed binary protocol on stdin/stdout (`[uint32 n][n×uint32 ids]` in — a stream of `[uint32 id]` tokens terminated by a `0xFFFFFFFF` sentinel; `n == 0` shuts down). Tokens are emitted **one at a time** as they are generated. The Python frontend `pumice_llm.py` (repo root, run under `.venv` via **uv**) tokenizes text, drives the daemon, and detokenizes output.
- **Validation mode** (`main.exe val`): the original shader harness — randomized test data, weight transpose/quantize/upload, GPU dispatch, comparison against single-threaded CPU references.
- **Memory-info mode** (`main.exe meminfo`): dumps the device memory heaps/types (§3.2) and exits — used to diagnose the VRAM budget (§12).
- **Node addon mode** (`bin/pumice.node`, §14): the same engine core compiled into an N-API shared library. It owns the model, tokenizer (via the vendored `tokenizers-c` static lib, §14.2), and generation loop in-process, streaming tokens to JavaScript through a threadsafe function. The `webui/` React + Vite frontend talks to a small Node server that exposes model probe/load/unload, SSE chat, and **perplexity evaluation** (§15) endpoints. The same server also exposes an **[OI]-compatible `/v1` API** (§17) so an agent harness (opencode and friends) can drive it unchanged. The whole thing is published to npm as **`@h4zel/pumice`** (§14.9): `npm i -g @h4zel/pumice` then run `pumice` from a folder containing `model/` and `pruned-vocab/`.

Both the server and the harness share the same `operation` dispatch core (§3.3); the Node addon reuses the same `createGenerator`/`generateTokens` core through `src/engine.c`. The addon engine also carries a **tiered KV block cache** (VRAM live session, host RAM pool, disk mirror; §16) so conversations resume instead of re-prefilling the whole history each request.

> **Model status.** The engine runs the **text stack only** (the `mtp.*` and `model.visual.*` tensors are ignored) — and it runs it **perfectly** against the real Qwen3.5 weights, both 9B and 2B, and the **Qwen3.6-35B-A3B MoE** (40 layers: 30 gated delta-net + 10 full-attention, every FFN a 256-expert top-8 MoE with a shared expert — §7.8). The **gated delta-net** is HF-identical to the Qwen3.5 block (§7.3), and the full-attention / FFN layers match the reference, including the **runtime-generalized GQA head mapping** (`kvh = head / gqa`, `gqa = heads/kv_heads` — any ratio, not just 16/4), the attention `1/sqrt(head_dim)` scaling, partial RoPE (64/256), and the `Qwen3_5RMSNorm` `1+weight` convention. Prefill layers track the HF reference at 0.96–0.9999 cosine correlation, and the decode trajectory matches the pruned-vocab-constrained HF greedy exactly, token-for-token (§13). Both **thinking** and **non-thinking** modes produce coherent, correct output (see §13). The 2B additionally validated greedy token-for-token against HF through long-context decode (past the split-K threshold at ctx 256). The 35B-A3B runs with 32 experts/layer in VRAM + 224 in host RAM at ~10.7 tok/s decode (pruned vocab, gqa 8 verified layer-by-layer against a CPU safetensors reference: delta L0 corr 0.9999, attention L3 corr 0.998, all 9 MoE slots >= 0.986).

---

## 1. Project Overview

- **Language/stack:** C (harness), GLSL 450 (compute shaders), Vulkan 1.1, glslangValidator, GNU Make (MinGW-w64 / MSYS2 UCRT64), Windows.
- **Model-agnostic core:** all model dimensions (hidden size, layer count, head counts, FFN size, linear-attn geometry, rope theta, tie flag, vocab size, per-layer quantization) are parsed at runtime from the model folder's `config.json` (HF format) + `quant_config.json` (engine-specific) by `loadModelConfig` (src/model.c). No engine rebuild or shader recompile is needed to switch models — the same `bin/` runs 9B and 2B.
- **Two inference phases:**
  - **Decode (token generation):** `GEMV` / split-K shaders — one token (M=1) Ã— weight matrix, pre-compiled into op groups of `DECODE_GROUP = 4` tokens and double-buffered on the queue.
  - **Prefill (first token / prompt processing):** `GEMM2` shaders — prompt tokens Ã— weight matrix, processed in **chunks** of `prefill_chunk` (512) tokens (chunk size = the state's `maxM`).
- **Precision:** every layer family exists in FP16, INT8 (per-group 256 asymmetric quantization), and Q4_1 versions (`q4_1_32` / `q4_1_64` / `q4_1_128` / `q4_1_256` in `quant_config.json` — 4-bit asymmetric with fp16 scale/zero per block; `int4` is still accepted as an alias for `q4_1_256`). Some shaders (GatedDeltaNet) are precision-agnostic — they operate on already-dequantized float projections.
- **Validation:** every shader has a CPU reference (`*_ref` functions) and a `validate*` function that runs the shader and compares via max absolute error (`main.exe val`).

### Model dimensions (runtime — per model, from `config.json` + `quant_config.json`)

Every value below is a field of `model_dims` (include/model.h), filled by `loadModelConfig` from the model dir. Example values shown for the two supported models:

| Field | 35B-A3B | 9B | 2B | Meaning / derivation |
|---|---|---|---|---|
| `K` | 2048 | 4096 | 2048 | `hidden_size` — hidden size / reduction dim |
| `layerCount` | 40 | 32 | 24 | `num_hidden_layers` (capacity `MODEL_MAX_LAYERS = 64`) |
| `ffnN` | 512 | 12288 | 6144 | `intermediate_size` — FFN gate+up columns |
| `heads` / `kvHeads` / `headDim` | 16 / 2 / 256 | 16 / 4 / 256 | 8 / 2 / 256 | `num_attention_heads` / `num_key_value_heads` / `head_dim` |
| `gqa` (derived) | 8 | 4 | 4 | `heads / kvHeads` — GQA group size, pushed to shaders |
| `qkvN` (derived) | 9216 | 10240 | 5120 | q+g+k+v fused width = `(2Â·heads + 2Â·kvHeads)Â·headDim` |
| `nQk` / `nV` / `dim` | 16 / 32 / 128 | 16 / 32 / 128 | 16 / 16 / 128 | `linear_num_key_heads` / `linear_num_value_heads` / `linear_value_head_dim` |
| `projN` (derived) | 12352 | 12352 | 8224 | delta in_proj width = `2Â·nQkÂ·dim + nVÂ·dim + 2Â·nV` (Q  K  V  Z  A  B) |
| `zqkvN` (derived) | 8192 | 8192 | 6144 | `2Â·nQkÂ·dim + nVÂ·dim` — conv+SiLU channel count |
| `kvRows` (derived) | 512 | 1024 | 512 | `kvHeadsÂ·headDim` — KV cache row count |
| `rotaryDim` | 64 | 64 | 64 | `headDim Â· partial_rotary_factor` (0.25) |
| `ropeTheta` | 1e7 | 1e7 | 1e7 | `rope_parameters.rope_theta` |
| `tied` | 0 | 0 | 1 | `tie_word_embeddings` (2B shares one embed matrix for embed+lm-head) |
| `vocab` | 102400 / 248320 | 102400 / 248320 | 102400 / 248320 | mode-dependent: `--prune` -> hardcoded `MODEL_VOCAB` 102400 (include/generated_vocab.h); no `--prune` -> `config.json` `text_config.vocab_size` (the original HF vocab). The pruned vocab was grown 86016 -> 102400 with coding-domain tokens recovered from full-vocab top-p collection on the 2B (see §13 note 19 / tools/vocab_collect). The 248320 mode fits the 8 GB heap only on the 2B (~1 GB embed); the 9B/35B run pruned. |
| `maxCtx` | 8192 | 32768 | 32768 | `quant_config.json` `max_ctx`; `--max-ctx` is a true override in **both directions** (growing it grows the KV-cache/`attScores` allocations) and is clamped to config.json `max_position_embeddings` (2B: 262144). Decode attention supports up to `ATT_MAX_CHUNKS` × 256 = 262144 tokens (gotcha 49); validated end-to-end at 131072 on the 2B |
| `prefillChunk` | 512 | 512 | 512 | `quant_config.json` `prefill_chunk` |
| `experts` / `expertsPerTok` / `moeI` | 256 / 8 / 512 | 0 / 0 / 0 | 0 / 0 / 0 | `num_experts` / `num_experts_per_tok` / `moe_intermediate_size` — nonzero experts switch every FFN to `FFN_MOE` (§7.8); `ffnN` falls back to `moeI` when `intermediate_size` is absent |
| `expertsVram` (config) | 32 (default) | n/a | n/a | `quant_config.json` `experts_vram`, overridable `--experts-vram N` — routed experts 0..N-1 + the shared expert live in a VRAM pool, the rest in a host-visible RAM pool (§4.7) |
| `max_ops` | 2048 | 2048 | 2048 | max `operation`s per op array (`MODEL_MAX_OPS`; raised 1280 -> 2048 for the 40-layer 35B decode group: ~370 ops/token x 4 + lm head) |

The per-layer spec (attention type from `layer_types[]`, quant from `quant_config.json` `layers[]`) also comes from the configs — the old hardcoded 32-layer `model_config` in compute.c is gone.

---

## 2. Repository Structure

```
pumice/
       package.json              # npm package @h4zel/pumice (bin: pumice -> cli.js)
       cli.js                    # CLI entry: args, platform-binary resolution, browser open
       pack.mjs                  # prepack: vite build + esbuild server bundle + runtime staging
       release.mjs               # version bump + publish (platform package, then main)
       pumice_llm.py                 # Python frontend: start_llm/tokenize/generate (uv venv, drives main.exe server)
       webui.py                  # (legacy) Gradio web UI on top of pumice_llm.py
       test_webui.mjs            # Node end-to-end test for the addon + webui server
       test_kvcache.mjs          # KV cache resume/eviction/restart tests (§16.7)
       test_35b.mjs              # 35B-A3B KV restore + memory footprint (§16.7/16.9)
       test_v1.mjs               # [OI] API end-to-end tests (§17.6)
       tools/tokenize_cli.py  tools/detokenize.py   # standalone tokenize/detokenize helpers
       tools/json_reader.py  tools/cmp_layers.py    # vocab / HF layer-comparison helpers
       tools/ppl_ref.py  tools/ppl_ref_windows.py   # HF perplexity references (§15.5)
       tools/run_dump.py             # drives --dump layer-differential runs
       gen_node_lib.ps1               # builds build/libnode.a from the running node.exe (N-API imports)
       tools/setup_2b.py            # (legacy) 2B prep: checks tokenizer parity, runs the pruner
        tools/pruner/                # vocab pruner (Wikipedia corpus, chat-token protection)
             pruner.py  vocab_select.py  emit.py  gather_weights.py  check.py
             engine_harness.py  compare_hf.py  cmp_hidden.py  (.venv with torch/transformers)
        tools/vocab_collect/         # top-p token collection + vocab growth (2026-09)
             prompts.py  collect.py  build_vocab.py  (uses .venv: tokenizers + numpy)
       tokenizers/                  # vendored tokenizers-cpp (Rust) static libs + C API header
            include/tokenizers_c.h   # C API (new_from_str / encode / decode / id_to_token ...)
            include/tokenizers_cpp.h # C++ wrapper (unused; the addon uses the C API)
            lib/libtokenizers_c.a    # MinGW (x86_64-pc-windows-gnu) Rust static lib, 29 MB
            lib/libtokenizers_cpp.a  # C++ wrapper static lib (unused)
       xxhash/                      # vendored XXH3 (BSD-2) single header, instantiated by src/xxhash.c
       platform/win32-x64/          # npm platform package @h4zel/pumice-win32-x64 (§14.9)
            package.json             # os/cpu restricted; ships pumice.node + shader/ (staged by pack.mjs)
       webui/                       # React + Vite frontend and Node/Express server (§14)
            package.json  vite.config.ts  tsconfig.json  index.html  .npmignore
            server/index.ts          # Express API: probe/load/unload/chat(SSE)/score(SSE) + static dist
            server/v1.ts             # [OI]-compatible /v1 router (§17)
            server/chatTemplate.ts   # Qwen chat template (roles, tools, thinking) + ReasoningSplitter
            server/toolParser.ts     # <tool_call> / JSON tool-call parser + ToolTagHoldBack
            server/load.ts           # shared resolveInput/buildLoadOptions for /api/load + /v1
            server/models.ts         # scan-folder registry shared by /api/models and /v1
            server/paths.ts          # runtime/dist/models/export/pruned-vocab path resolution + chdir
            server/engine.ts         # EngineManager: addon wrapper, request queue, streaming, scoring
            server/probe.ts          # model probe: safetensors shard/config checks, gguf+hqm meta
            server/types.ts          # shared API types
            src/App.tsx  src/api.ts  src/styles.css
                 components/ModelTab.tsx  SamplingTab.tsx  ChatTab.tsx  QuantEditor.tsx
                 components/ScoreTab.tsx  # perplexity eval tab (§15)
       Makefile                    # recursive shader build -> bin/shader/*.spv; builds main.exe + pumice.node
       include/                    # C headers
            buffer.h  data.h  descriptor.h  device.h  dispatch.h
            fence.h  pipeline.h  session.h  validation.h
            json.h                   # generic JSON DOM parser (objects/arrays/strings/numbers)
            safetensors.h           # safetensors parser (BF16/F32, 64-bit offsets)
            model.h                 # model_dims struct + loadModelConfig (runtime, config-driven)
            prune.h                 # vocab pruner entry (pruneVocab)
            weights.h               # weight tensors (block-transposed, quantized)
            state.h                 # activation / KV-cache / scratch buffers
            generate.h              # generator struct + prefill/generateTokens/reset
            engine.h                # engine lifecycle shared by the addon (engineOpen/Close/Generate)
            kvcache.h               # tiered KV block cache: chain hashes, index, RAM pool, snapshots
       src/
            main.c                  # arg dispatch: default=server, `val`=harness, `meminfo`
            compute.c               # serverMain (server loop) + memInfo
            json.c                  # JSON parser used by model.c (configs) and vocab EOS lookup
            model.c                 # loadModelConfig: config.json + quant_config.json -> model_config/dims
            prune.c                 # pruneVocab: gathers pruned embed from mapping.npy, writes vocab/
            safetensors.c           # safetensors header parse + BF16—F32 load
            validation.c            # CPU reference impls + all validate* functions
            generate.c              # op compiler: chunked prefill, decode groups, lm head
            kvcache.c               # KV block cache core (see 16)
            xxhash.c                # XXH3 single-header implementation unit (xxhash/xxhash.h)
            weights.c  state.c      # cache-first weight upload, state buffers (all dims runtime)
            engine.c                # engineOpen/Tokenize/Decode/Generate/Close + tokenizer loading
            addon.c                 # N-API bindings (createEngine/tokenize/decode/generate/destroy)
            session.c device.c buffer.c command.c fence.c   # Vulkan setup
            descriptor.c pipeline.c dispatch.c              # descriptors, pipelines (spec constants), dispatch
            data.c                  # pseudo-random data, fp16/bf16 conversion, transpose_block16, quantize
        model/
            Qwen3.5-9B/             # 9B: shards + config.json + quant_config.json + vocab/
            Qwen3.5-2B/             # 2B: shard + configs (+ vocab/ generated by --prune)
        pruned-vocab/               # pruned-vocab source dir (repo root): mapping.npy + pruned
                                   # tokenizer files — the hardcoded source for --prune
       shader/
           Utility/                # shared matmul primitives
                Q16/ Q8/ Q4/        # quant suffix: Q16 = fp16, Q8 = int8, Q4 = q4_1_* (block in push constants)
                                    # GEMV-SplitK-*, GEMM-ADD2-*, LMHead-GEMV-ArgMax-Q16, LMHead-GEMV-Q16
                RmsNorm-Prologue.comp        # invRms prologue (workgroup tree reduction, used by all GEMM2 kernels)
                Reduce-GEMV-ADD.comp         # split-K reduce + residual add
                ArgMax-Reduce.comp           # token selection: greedy argmax / sampler / perplexity scoring
                Gate-Sigmoid.comp            # sigmoid gate for the delta-net output
                KV-Restripe.comp  KV-Unstripe.comp   # live KV <-> block-contiguous pool (section 16.5)
           Full-Attention/         # QKV projection + RoPE + full attention
                Q16/ Q8/ Q4/        # RmsNorm-QKV-SplitK-* + Reduce-Rope-* (decode)
                Q16/ Q8/ Q4/        # RmsNorm-QKV-GEMM2-* + Rope-GEMM-* (prefill, split passes)
                Q16/ Q8/ Q4/        # Att-full-* (decode), Att-SplitK2-* (decode, split-K)
                Q16/ Q8/ Q4/        # Att-QK2-*, Att-PV2-* (prefill, unfused)
                Att-Softmax.comp    # prefill softmax pass (fp16 score buffer + smSum reciprocals)
                Reduce-Att2.comp    # decode split-K attention reduce
           Linear-Attention/       # gated delta-net
                Q16/ Q8/ Q4/        # RmsNorm-LinearProj-SplitK-* (decode),
                                    # RmsNorm-LinearProj-GEMM2-* (prefill)
                                    # Embed-RmsNorm-LinearProj-Q16 (fused fp16 embed+GEMV)
                Embed-Gather.comp   # prefill embedding pre-fetch (embed/lm-head column gather)
                GatedDeltaNet.comp  GatedDeltaNet-GEMM.comp   (precision-agnostic)
                Conv-SiLU.comp      # depthwise causal conv (kernel 4) + SiLU (realM-aware)
                Reduce-LinearProj.comp       # LinearProj split-K reduce (5-way routing)
           FFN/                    # swiglu feed-forward
                Q16/ Q8/ Q4/        # RmsNorm-up-ffn-SplitK-* (decode, flattened gate|up),
                                    # FFN-Down-SplitK-* (decode down projection)
                                    # RmsNorm-swiglu-ffn-GEMM2-* (prefill, dual-matrix),
                                    # RmsNorm-swiglu-flat-GEMM2-* (prefill, flattened)
                Swiglu-combine.comp          # silu(gAct) * uAct elementwise pass
           MoE/                    # Qwen3.6-35B-A3B expert FFN
                Q4/                 # Expert-Swiglu-Q4, Expert-Down-Q4 (Q4 expert pools)
                Router-TopK.comp    Moe-Combine.comp
           Prototype/              # NOT built (excluded by the Makefile): validation-harness shaders
                                   # (GEMM-*/GEMV-*/GEMV-ADD-*/GEMM-ADD-*/RmsNorm-GEMV-*/Att-SplitK-*/
                                   #  Att-full-GEMM-*/RmsNorm-QKV-*/RmsNorm-LinearProj-*/Reduce-Att/
                                   #  RmsNorm-swiglu-ffn*) plus legacy experiments (gemv1..7, gemm,
                                   #  RmsNorm, online-softmax, RmsNorm-GEMV-Rope-*, test)
```

### Per-model folder layout

```
model/<name>/
       model.safetensors-*.safetensors   # HF shards (any count — globbed)
       config.json                       # HF architecture config (dims, layer_types, rope, tie flag)
       quant_config.json                 # engine config: name, max_ctx, prefill_chunk,
                                       #   embed/lm_head quant, per-layer {"attn","ffn"} quant
       vocab/                            # generated by --prune (or pre-gathered)
           embed_tokens.<V>.safetensors  # pruned embedding rows (VÃ—K BF16)
           lm_head.<V>.safetensors       # pruned lm-head rows (untied models only)
           tokenizer.json  tokenizer_config.json  vocab.json
           mapping.npy                   # new-id — original-id int32[V]

pruned-vocab/                            # repo root by default (PRUNED_VOCAB_DIR); PUMICE_PRUNED_VOCAB_DIR overrides
       mapping.npy                       # new-id — original-id int32[102400]
       tokenizer.json  tokenizer_config.json  vocab.json   # pruned tokenizer (renumbered ids)
       topp_tokens.json                  # union of top-p nuclei from the coding-token collection run
       counts.npy  chat_protect.npy  selection.json  prune_report.txt   # original 86016-prune artifacts
```

---

## 3. Vulkan Runtime (C side)

### 3.1 Session & device

`createSession()` (src/session.c) creates the Vulkan instance, physical device, logical device, compute queue, command pool, command buffer, fence, and a query pool with `TIMESTAMP_QUERY_COUNT` timestamps for GPU timing.

### 3.2 Buffers — `buffer` (src/buffer.c)

```c
typedef enum { MEMORY_RAM, MEMORY_VRAM } memory_type;  // staging vs device-local
buffer createBuffer(VkDevice, VkPhysicalDevice, const void* data, size_t size, memory_type);
buffer createBufferNamed(VkDevice, VkPhysicalDevice, const void* data, size_t size, memory_type, const char* name);
```

- `MEMORY_RAM` = host-visible staging; `MEMORY_VRAM` = device-local (plus a persistent host-visible staging buffer of equal size for the initial copy).
- `createBufferNamed` stores the 64-byte `buffer.name` label and, via `allocateBufferMemory`, meters per-pool bytes onto two file-scope counters (`device_local` / `host_visible`). On allocation failure it no longer calls `exit()`: it records an error (`bufferAllocFail`) naming the buffer and the pool (`out of GPU memory` for DEVICE_LOCAL, `out of host memory` for HOST_VISIBLE/staging, or `out of host memory: could not stage` when the caller's staging array is NULL), returns a null buffer, and fast-fails all further allocations. `createState`/`createWeights`/`createGenerator` detect the flag, tear down what they built, and `engineOpen` returns NULL with the message — so a too-large load surfaces as a `500` from `/api/load` (and a rejected promise from the addon) instead of killing the process (see §12).
- `createTransferAndCopy(device, queue, bufs, n)` stages host data into all buffers and copies RAM—VRAM where needed.
- `readBuffer(...)` copies results back to host.

### 3.3 Operation & dispatch — `operation` (include/dispatch.h)

```c
typedef struct operation {
    char shader[128];                 // e.g. "GEMM-FP16.spv"
    buffer buffers[MAX_OP_BUFFERS];   // bound to set=0 bindings 0..n-1
    int bufferCount;
    int pushConstants[MAX_PUSH_CONSTANTS];  // ints (16 max), copied verbatim to shader
    int pushConstantCount;
    int dispatchX, dispatchY, dispatchZ;    // vkCmdDispatch dims
} operation;

void execute(session s, operation ops[], int opCount);
```

`execute()` (src/dispatch.c) per op: creates a descriptor set, compiles the shader into a pipeline, binds it, pushes constants, inserts a `VK_ACCESS_SHADER_WRITE_BIT — VK_ACCESS_SHADER_READ_BIT` memory barrier **between** ops (so chained ops like LinearProj — GatedDeltaNet — GEMM are correctly ordered), dispatches, and fences. GPU time comes from query-pool timestamps around the whole op chain.

Pipeline creation (src/pipeline.c) supports two model-agnostic mechanisms, set once per process in `createGenerator`:

- **Shader root dir** — `setShaderRootDir(spec->shaderDir)` prefixes every op's shader path (`<root>/<name>.spv`; the default root `""` resolves to `shader/`). Pipelines are cached in `getPipeEntry` by (shader name, push size, buffer count).
- **Specialization constants** — `pipelineSetSpecInt(0, dims.K)` feeds `layout(constant_id = 0) const int d_model` in the RMSNorm-fused GEMV family (`RmsNorm-GEMV-*`, `RmsNorm-QKV-SplitK-*`, `RmsNorm-LinearProj-*`, `Embed-RmsNorm-LinearProj-*`, `LMHead-GEMV-*`, `RmsNorm-swiglu-*`): it sizes the shared staging array `cache[d_model/vec]` and is the RMS divisor. Because the value is fixed per process (one model per run), the pipeline cache stays valid.

`MAX_PUSH_CONSTANTS` is 16 ints (64 bytes) — the attention push blocks now carry up to 9 values (Â§4.5).

---

## 4. Data & Weight Conventions

### 4.1 Test data (src/data.c)

- `getData(seed, M, N)` — float `[M][N]`, pseudo-random in [-1, 1] from a seed-dependent hash of (i, j).
- `getDataFP16(seed, M, N)` — same values converted to IEEE fp16 via `float_to_fp16`.
- `getDataINT8(seed, M, N)` / `getDataQ4(seed, M, N)` — per-group asymmetric quantization: `scale = (max-min)/255` (or `/15` for Q4), `zero = -min`, layout `scale[bj*K + row]`, `bj = col/block`. INT8 uses group 256; Q4 uses Q4_1_256 (the block-size variants only come from `quantizeDataQ4`, below). Q4 packs 2 columns per byte, high nibble first.

### 4.2 Block-transposed weight layout — `transpose_block16`

Weights are stored `[K][N]` on host but uploaded **block-transposed** so each shader thread reads contiguous `uvec4`s. Per output column `n`, 16 consecutive bytes hold a block of k-rows; as `uvec4` the index is:

| Precision | uvec4 index | What one uvec4 contains |
|---|---|---|
| FP16 | `w[(k/8)*N + n]` | 8 halves = 2 vec4 k-rows (`unpackHalf2x16` of `.x,.y` and `.z,.w`) |
| INT8 | `w[(k/16)*N + n]` | 16 bytes = 4 vec4 k-rows (`unpackUint8x4` per component) |
| Q4 | `w[(k/32)*N + n]` | 32 nibbles = 8 vec4 k-rows; a 16-k tile takes low half (`.x,.y`, t even) or high half (`.z,.w`, t odd) |

```c
void transpose_block16(const uint8_t* input, uint8_t* output, int M, int N, QuantType q)
```

The nibble packing is **independent of the Q4 block size** — only the scale/zero arrays change.

### 4.3 Quant scale/zero layout

Q4_1 (`quantizeDataQ4`, block size B = 32 / 64 / 128 / 256 from the quant config): `scale[bj*K + k]` and `zero[bj*K + k]` with `bj = n/B` (asymmetric `scale = (max-min)/15`, `zero = -min`), one value per (output block, input). They are stored as **fp16** (`uvec2[]`, 4 halves per element) at vec index `bj*(K/4) + kvec` (`kvec = k/4`); a 16-k tile at k-tile `t` needs `scale[bj*(K/4) + t*4 + kv]` for `kv = 0..3` (`unpackFp16x4`). INT8 keeps the fp32 `vec4[]` layout with group 256.

**Q4_1 block sizes** (`quant_config.json` per layer: `"q4_1_32"` / `"q4_1_64"` / `"q4_1_128"` / `"q4_1_256"`, `int4` = `q4_1_256`). The block size is carried in the HQM layer config, so a re-export happens automatically when it changes; the shaders get it as an extra push constant (`block`). Because one workgroup spans 256 output columns, the dequant differs by size:

- **B >= 256** (only `q4_1_256` today): every output in the workgroup shares one scale vector, so the split-K GEMV shaders pre-scale the activations in shared memory once per tile (plus a subgroup zero-point reduction) — the original fast path.
- **B < 256**: each output column has its own block, so the scale/zero are applied **per column** against the weights in the inner loop (the GEMM2 prefill shaders already did this). That costs decode throughput: on the 2B pruned model, decode is ~45 tok/s at `q4_1_256` and ~29-30 tok/s at `q4_1_64`/`q4_1_128`, ~23 tok/s at `q4_1_32` (wikitext-2 test PPL 19.59 / 17.65 / 17.47 / 17.47 — smaller blocks are more accurate, fp16 scales limit the gain past 128).

### 4.4 Activations & outputs

- Input x: row-major `[M][K]` float, vec4 index `m*(K/4) + kvec`.
- Outputs: row-major `[M][N]` float.
- KV cache, layout split by tensor: **K is stored transposed** (dim-major) as `kCache[row Â· MAXCTX + token]` (`row = kvhÂ·headDim + dim`, `MAXCTX` = runtime `maxCtx`, pushed to every shader) so the QK dot in decode/prefill reads are coalesced across the token axis, while **V stays token-major** `vCache[token Â· KV_TOTAL_ROWS + row]`. Precision: fp16 (`uint16_t` + `packHalf2x16`) for FP16 layers, `uint8` + per-(kv-head, token) scale/zero for INT8/Q4 layers. Quantized scale/zero live at the **fixed** stride `kvh * maxCtx + token` in every writer and reader (see gotcha 15).
- RoPE theta: `theta[i] = ropeTheta^(-i/(rotaryDim/2))`, length `rotaryDim/2` (32 pairs at rotaryDim 64), built per model from config (`rope_theta`, `partial_rotary_factor`).

### 4.5 Push constants

All layouts below are model-agnostic — every dimension is a runtime value:

- Generic GEMM/GEMV/FFN/LinearProj: `{M, N, K}`; the Q4 variants append `block` (32/64/128/256).
- QKV GEMV (legacy fused `RmsNorm-QKV-*`): `{M, N, K, gOffset, kOffset, vOffset, maxCtx, headDim, rotaryDim}` — 9 values, all runtime. Decode `RmsNorm-QKV-SplitK-*`: `{M, N, K}` (the reduce passes carry the offsets).
- Prefill `RmsNorm-QKV-GEMM2-*`: `{M, N, K}` (raw projection only). `Rope-GEMM-*`: `{N, gOffset, kOffset, vOffset, tokBase, maxCtx, headDim, rotaryDim, heads, kvHeads}` (tokBase = chunk-absolute first token; the position buffer is **not** touched by prefill shaders — `runPrefill` sets it host-side once via `stateSetPosition` before `finalOps`).
- Decode `Reduce-Rope-*`: `{N, gOffset, kOffset, vOffset, maxCtx, headDim, rotaryDim, heads, kvHeads}`.
- Attention decode (`Att-full-*` / `Att-SplitK2-*` / `Reduce-Att2`): `{maxCtx, kvHeads, kvRows, gqa, headDim, heads}` — 6 values. `gqa = heads/kvHeads` is the GQA group size; shaders compute `kvh = head / param.gqa` (the 4th member). **`Reduce-Att2` needs the real `heads` (6th member) for its output-grid guard `h >= HEADS` — reusing the gqa slot for it left heads gqa..heads-1 stale (gotcha 34).** Context length comes from the shared position buffer (`position[0] + 1`).
- Attention prefill (`Att-QK2-*` / `Att-Softmax` / `Att-PV2-*`): `{ctxLen, qOff, mRows, headBase, maxCtx, kvHeads, qDim, kvRows, gqa}` — 9 values (ctxLen = chunk-absolute context, qOff = chunk base for the causal limit `qOff + mGlobal`, qDim = `headsÂ·headDim` = query row stride, headBase = `hbÂ·kvHeads` per GQA group iteration).
- GEMM2 prefill kernels: `RmsNorm-swiglu-flat-GEMM2-*` `{M, N, K, off}` (off = FFN gate/up boundary for `upHalf` routing); `RmsNorm-swiglu-ffn-GEMM2-*` / `GEMM-ADD2-*` `{M, N, K}`; `RmsNorm-QKV-GEMM2-*` `{M, N, K}`; `RmsNorm-LinearProj-GEMM2-*` `{M, N, K, kOff, vOff, zOff, aOff, bOff}`; `RmsNorm-Prologue` `{K}`.
- GatedDeltaNet: `{M, nV, nQk, realM}` (prefill GEMM — realM = true token count for pad masking, Â§8.2) or `{N_V, N_QK, DIM}` (decode).
- Conv-SiLU: `{M, kOff, vOff, zqkv, realM}` — `steps = min(M, realM)` limits the conv+history to real tokens (pad tokens would otherwise poison the shift register carried into decode).
- LM head: `{M, N, K}` (M=1, N=vocab) for the `LMHead-GEMV-*` pass (`GEMV-ArgMax-FP16` when greedy, `GEMV-FP16` writing raw `logits` when sampling); `{vocabSize, doIncrement, passIdx, mode}` for `ArgMax-Reduce` (`mode` = 0 greedy / 1 sampler; doIncrement=1 bumps the position buffer by 1; passIdx tags the write slot for the double-buffered decode groups).
- Embed-Gather: `{V, K}`. Embed-RmsNorm-LinearProj GEMV: `{M, N, K, kOff, vOff, zOff, aOff, bOff, V}` — 9 values.
- Split-K decode reduce passes: `Reduce-GEMV-ADD` `{N, chunks}` (N = K, chunks = 4); `Reduce-LinearProj` `{N, kOff, vOff, zOff, aOff, bOff}`; `Reduce-Att2` as above; `Reduce-Rope` as above.

### 4.6 Dispatch geometry

All grids are computed from runtime dims:

- GEMV shaders: `dispatchX = N/256` (one workgroup of 256 threads per 256 output columns), M=1.
- GEMM shaders: `dispatchX = N/16`, `dispatchY = M/16` (each workgroup computes a 16Ã—16 output tile).
- GEMM2 shaders: `dispatchX = N/TN` (TN=32, or 64 for Q4 GEMM-ADD2 / Q4 QKV-GEMM2), `dispatchY = M/16`; each workgroup computes a 16Ã—TN output tile.
- Flattened swiglu (gate|up in one grid): `dispatchX = 2*N/TN` (N=ffnN, `upHalf = nBase >= off`), `dispatchY = M/16`.
- Rope-GEMM: `dispatchX = 2*heads + 2*kvHeads` (q, g, k, v head workgroups), `dispatchY = M` (one row per workgroup, 256 threads).
- Embed-Gather: `dispatchX = M`, 256 threads (each row gathers its K floats from the embed/lm-head column).
- Attention prefill (QK2): `dispatchX = (ctxLen+63)/64` (64-token KV tiles), `dispatchY = (ceil(M/16)) * kvHeads` (m-tiles Ã— kv-head bands); Softmax: `dispatchX = M`, `dispatchY = kvHeads`; PV2: `dispatchX = headDim/64`, `dispatchY = (ceil(M/16)) * kvHeads`. The QK2/Softmax/PV2 trio is emitted **`heads/kvHeads` times** (one per GQA group, `headBase = hb*kvHeads`), so all query heads are computed while `attScores` stays `kvHeads`-sized (gotcha 25). All prefill GEMM/attention grids use `ceil(M/16)` so short prompts (M < 16) don't dispatch zero workgroups (gotcha 23).
- Attention decode: `Att-full` `dispatchX = heads`; `Att-SplitK2` `dispatchX = heads`, `dispatchY = ceil(maxCtx/256)` capped at `ATT_MAX_CHUNKS` (K-chunks of 4 tiles); `Reduce-Att2` `dispatchX = heads*headDim/256`.
- Split-K decode GEMVs: `dispatchX = N/256`, `dispatchY = 4` (K split over 4 workgroups).
- GatedDeltaNet GEMM: `dispatchX = nV`, one workgroup per v-head.
- Conv-SiLU: `dispatchX = zqkvN/256`, one thread per qkv channel (each thread shifts its own 3-slot history over the `min(M, realM)` real tokens, no barriers).
- LMHead-GEMV-{ArgMax,FP16}: `dispatchX = (vocab+255)/256`; ArgMax-Reduce: single workgroup (both greedy and sampling modes).

### 4.7 Real weight loading (src/safetensors.c + src/weights.c)

Loading is **cache-first and lazy at every level, for every weight class** (matrices, vectors, conv, embeddings): the on-disk cache is checked first, and the model-dir safetensors are opened **only on the first actual cache miss** via the idempotent `shardSource()` — a fully-cached run never touches the model dir at all (it prints `weights: resolved fully from cache`). Shards are therefore optional: a cache-only `bin/weights/<model>/` runs standalone. The only upfront check is a fail-fast directory glob: no shards and `cacheComplete()` false → fatal. `cacheComplete` probes the V-suffixed `embed_<V>_FP16.bin` / `lmHead_<V>_FP16.bin` (plus every matrix/vector cache entry) with header validation — an earlier version probed the un-suffixed `embed_FP16.bin` name, which never existed, so a complete cache was reported incomplete whenever the shards were absent (gotcha 41).

**Config & pruning pipeline** (runs before weights, in `serverMain`):

1. `loadModelConfig` (src/model.c) parses `<modelDir>/config.json` (HF: hidden_size, layer count, layer_types, head counts, head_dim, intermediate_size, linear_* geometry, rope_parameters, tie_word_embeddings, vocab_size) with the generic JSON DOM parser (src/json.c) and derives `qkvN`, `projN`, `zqkvN`, `kvRows`, offsets, `rotaryDim`. It then parses `quant_config.json` (name, max_ctx, prefill_chunk, per-layer attn/ffn quant, embed/lm_head quant). **`--quant-config <path>`** overrides the per-directory file for both safetensors and GGUF models (the web UI uses it to pass a UI-generated quant config without touching the model dir; HQM ignores it — quant is baked in). **`vocab` is mode-dependent**: `--prune` → the hardcoded pruned size `MODEL_VOCAB` = 102400 (include/generated_vocab.h); no `--prune` → `config.json`'s original `vocab_size` (248320). `vocab_size` must be read **before** `json_free(hf)` — after the free the pointer dangles (gotcha 35's use-after-free trap, hit again here).
2. `pruneVocab(modelDir, spec)` (src/prune.c, gated by the `--prune` flag) resolves vocab weights in cache-first order:
   - **weight cache**: if `bin/weights/<name>/embed_<V>_FP16.bin` (and `lmHead_<V>_FP16.bin` for untied models) exists with a matching `{magic, K, V, FP16}` header, skip the gather entirely;
   - **vocab folder**: if `<modelDir>/vocab/embed_tokens.<V>.safetensors` (and `lm_head.<V>` for untied) exists, skip the gather;
   - **auto-prune**: open all shards (`model.safetensors*.safetensors` glob), read `pruned-vocab/mapping.npy` (int32[V], new-id — original-id; the npy header's version bytes must be skipped — gotcha 36; its length must equal V), and gather `row v` = source row `mapping[v]` from `embed_tokens` (and `lm_head` for untied models — the lm-head tensor can live in a different shard, so the multi-shard `safetensors_open` is required) into `vocab/embed_tokens.<V>.safetensors` / `vocab/lm_head.<V>.safetensors` (BF16, proper safetensors header).
   - In every outcome the tokenizer files (`tokenizer.json`/`tokenizer_config.json`/`vocab.json`/`mapping.npy`) are ensured in `vocab/` from the root `pruned-vocab/` dir (`prunedVocabDir()` in src/prune.c — `PUMICE_PRUNED_VOCAB_DIR` if set, else `PRUNED_VOCAB_DIR` `"../pruned-vocab"` from include/prune.h) — both `parseEos` and the Python frontend need them regardless of the gather decision.
3. `parseEos(&spec.dims, modelDir, pruned)` (src/model.c, public) derives **EOS from the tokenizer files** so it can never drift (gotcha 35: a hardcoded id from a stale header once made generation never stop). Pruned mode reads `vocab/tokenizer_config.json`'s `eos_token` name and resolves it in `vocab/vocab.json` (85992). Original mode reads the model-root `tokenizer_config.json` and resolves the name in the root `vocab.json` — but the original `vocab.json` contains **no special tokens**, so on miss it falls back to `tokenizer.json`'s `added_tokens` array (id 248046 for `<|im_end|>`). It runs *after* pruning (pruning is what materializes the vocab files on a fresh model dir) and before `createGenerator`.
4. Weight tensor discovery globs `model.safetensors*.safetensors` (any shard count) instead of hardcoding 4 shards. Vocab-file discovery is V-exact: `findVocabFile` matches `embed_tokens.<V>.safetensors` by name, so a 248320 run ignores the 86016 files in `vocab/` and loads the full-size tensors straight from the shards (the pruned files are the wrong shape and must not be candidates).

**Tensor assembly** mirrors the synthetic path:

1. `safetensors_open` parses the 8-byte-length JSON header (names/dtype/shape/data_offsets) with a minimal in-C JSON walker; data offsets are 64-bit (`_fseeki64`) — several tensors live past the 2 GB mark in a 5.3 GB shard. **`data_offsets` are relative to the data section, so each tensor offset is adjusted by `8 + header_length` to an absolute file offset** (gotcha 19). Both BF16 and F32 tensors are read; BF16 is widened to float (`bf16_to_float`).
2. Per "logical" tensor, the HF `[out][in]` matrices are concatenated along the output axis and transposed to the engine `[in][out]` layout:
   - full-attn `proj` = `q_proj` **de-interleaved** (HF stores q  g per-head interleaved as `[heads][2Â·headDim][K]`; `buildQkvMatrix` takes `headDim`/`heads` from config — an early bug derived them from tensor shapes and scrambled every full-attn layer, gotcha 33) — q    g, then `k_proj`    `v_proj` — `qkvN` columns (Q  G  K  V);
   - delta `proj` = `in_proj_qkv`    `in_proj_z`    `in_proj_a`    `in_proj_b` — `projN` columns (Q  K  V  Z  A  B) — `z` is the output-gate projection (Â§7.3);
   - `gate`/`up`/`down` = `mlp.gate_proj`/`up_proj`/`down_proj`, `out` = `o_proj`/`out_proj`.
3. Quantize per the `quant_config.json` `QuantType` (`quantizeDataINT8` for int8, `quantizeDataQ4(..., q)` for the `q4_1_*` block variants, `float_to_fp16` for fp16), then `transpose_block16`, then upload (`createBufferNamed`). Embeddings and the lm-head (`[K][vocab]`) are built fp16 directly from the BF16 `[vocab][K]` source without a float intermediate. Every weight buffer is registered into `g_wbufs` and copied staging—VRAM by a single `createTransferAndCopy` at the end of `createWeights` (gotcha 20).
4. The `Qwen3_5RMSNorm` vectors (`input_layernorm`, `post_attention_layernorm`, final `norm`, `q_norm`/`k_norm` [headDim]) are loaded **with `+1.0` added** — `Qwen3_5RMSNorm` stores `weight` zeros-init and applies `scale = 1 + weight`, and the checkpoint stores the raw (near-zero) delta. The delta `norm.weight` [dim] (`Qwen3_5RMSNormGated`, ones-init, applied directly — **no** `+1`) is loaded unchanged, as are `A_log`[nV] and `dt_bias`[nV]. `conv1d.weight` (`[zqkvNÃ—1Ã—4]` — zqkvN channels Ã— 4 taps, channel-major `w0..w3` with `w0` = t'3    `w3` = current) is loaded **FP32** and consumed by `Conv-SiLU.spv` (Â§7.3). See gotcha 22.
5. Small vectors (norms, `A_log`, `dt_bias`, `conv1d`) have their own on-disk cache (`vec_<label>_<layer>.bin`, `conv_<layer>.bin`) so a cache-only run needs no safetensors at all. All of them resolve cache → `shardSource()` (the conv loader once checked shards *first* and rewrote its cache file every run — now cache-first like the rest).

**Disk cache** (`bin/weights/<model-name>/` — one directory per model, so two models never collide): quantized matrices as `<name>_<QUANT>.bin` (header `{magic, rows, cols, quant}` — a dim or quant change auto-invalidates), embeddings as `embed_<V>_FP16.bin` / `lmHead_<V>_FP16.bin`. The embed cache filename carries V, so vocab modes **coexist** in one cache dir without invalidation (e.g. `embed_102400_FP16.bin`, `embed_86016_FP16.bin`, `embed_248320_FP16.bin`). **Gotcha: the cache keys on dims, not weight-file content — after re-gathering a vocab or editing quant_config.json, delete the model's cache directory or generation runs on stale tensors** (gotcha 37).

**Tied vs untied embeddings**: `tie_word_embeddings: true` (2B) loads the single embedding matrix once and aliases `lmHead = embed`; `false` (9B) loads two separate matrices from the vocab folder (`embed_tokens.<V>.safetensors` / `lm_head.<V>.safetensors`, both required — they are genuinely different matrices, verified at the source). Vocab weights resolve cache-first — vocab-folder safetensors (opened lazily, V-exact match) — shard tensors. In original-vocab mode the shard tensors are the only source (the vocab folder holds pruned-size files only). `rope_theta`, `partial_rotary_factor`, and `rms_norm_eps = 1e-6` all come from config.

**MoE expert pools (Qwen3.6-35B-A3B)** — when `config.json` has `num_experts > 0` (so every layer's
FFN is `FFN_MOE`), the loader builds, per layer, two **expert pools** covering all 257 experts
(256 routed + the shared expert at index 256):

- `guPool_<L>_INT4.bin` — per expert the fused `[K][2*moeI]` gate|up matrix, Q4_1-quantized
  (group 256) from the BF16 `mlp.experts.gate_up_proj[L][e]` (routed, HF chunks gate|up along dim 0)
  and from `mlp.shared_expert.gate_proj` + `up_proj` (fused the same way);
- `dnPool_<L>_INT4.bin` — per expert the `[moeI][K]` down matrix from
  `mlp.experts.down_proj[L][e]` / `mlp.shared_expert.down_proj`;
- `router_<L>.bin` / `sgGate_<L>.bin` — the FP16 `[K][experts]` router weight (`mlp.gate.weight`,
  transposed to engine layout and block-transposed) and the `[K][1]` shared-expert gate.

HF expert tensors are `[out][in]` and are transposed to engine `[in][out]` before quantization (the
routed read is a direct `fseek` into the 3-D tensor — no full-tensor load; gotcha 43). The pool
cache files carry a 5-int header `{magic, rows, cols, quant, experts}` and are **independent of
`experts_vram`** — the split is applied at buffer-creation time (`expertPoolSplit`), not at
quantization time, so changing `--experts-vram` never requantizes:

- **VRAM pool** = routed experts `0..N-1` **plus the shared expert** (copied into slot N; the shader
  remaps id `experts` -> slot `N`), as device-local buffers;
- **RAM pool** = routed experts `N..255` as one host-visible (PCIe-read) buffer — the Expert-*
  shaders take both pools as bindings and select per (token, slot) by comparing the id against
  `vramExperts` (gotcha 44). This is the substrate for the planned LFRU expert swap: the pool is
  expert-contiguous, so a future swap just re-copies one expert's block into the VRAM pool.

The 35B also breaks the `nV*dim == K` coincidence of every earlier model: the delta `out_proj` is
`[2048][4096]` (engine `[4096][2048]`), so `out` tensors, the `yGated` state buffer, and the decode
out-projection GEMV/SplitK reduction dim are all `nV*dim`/`tensor.rows`-driven now, not hardcoded
`K` (gotcha 45). Weight buffers are uploaded in batches of 12 with their staging memory freed
immediately after each batch (`registerWeightBuffer` + `releaseStaging`) — the one-shot bulk copy
kept every staging allocation alive until the end and blew the 16 GB host-visible heap when the RAM
pools were also resident (gotcha 46).


---

## 5. Build System

`make` recursively finds every `shader/**/*.comp` via a `rwildcard` function and compiles each into a **flat** `bin/shader/<basename>.spv`:

```makefile
rwildcard    = $(foreach d,$(wildcard $(1)*),$(call rwildcard,$(d)/,$(2)) $(filter $(subst *,%,$(2)),$(d)))
SHADERS      := $(filter-out $(SHADER_DIR)/Prototype/%,$(call rwildcard,$(SHADER_DIR)/,*.comp))
SHADER_OUT   := $(BIN_DIR)/shader
SHADERS_OBJS := $(addprefix $(SHADER_OUT)/,$(notdir $(SHADERS:.comp=.spv)))

define COMPILE_SHADER
$(SHADER_OUT)/$(notdir $(basename $(1))).spv: $(1)
	@if not exist $(SHADER_OUT_W) mkdir $(SHADER_OUT_W)
	"$(VULKAN_SDK)/Bin/glslangValidator" -V --target-env vulkan1.1 "$(1)" -o $$@
endef
$(foreach f,$(SHADERS),$(eval $(call COMPILE_SHADER,$(f))))
```

Run: `make` builds shaders + `bin/main.exe` + `bin/pumice.node`; the executable is normally launched by the server/Python path (§11), the addon by the Node server (§14). `make clean` removes `bin/` and `build/` recursively. The quant suffix in shader names is `Q16`/`Q8`/`Q4` (fp16/int8/q4_1_*) and `shader/Prototype/` is **excluded from the build** — it holds the validation-harness and legacy shaders, so `main.exe val` no longer resolves its `.spv` files.

The C build adds `-Ixxhash`; `src/xxhash.c` is the single translation unit that instantiates the vendored XXH3 implementation (`xxhash/xxhash.h`). It is part of `SRCS` and therefore linked into both `main.exe` and the addon.

**Node addon target** (`bin/pumice.node`): the Makefile compiles `src/*.c` except `main.c`/`addon.c`/`engine.c` into `CORE_OBJS`, then links `build/addon.o build/engine.o $(CORE_OBJS)` against the tokenizers static lib and an N-API import library:

```makefile
CORE_SRCS    := $(filter-out $(SRC_DIR)/main.c $(SRC_DIR)/addon.c $(SRC_DIR)/engine.c,$(SRCS))
NODE_EXE     := $(shell node -p "process.execPath")
NODE_INC     ?= $(firstword $(wildcard $(LOCALAPPDATA)/node-gyp/Cache/*/include/node))
NODE_LIB     := $(BUILD_DIR)/libnode.a
TOKENIZERS_LIB := $(BUILD_DIR)/libtokenizers_c.a

$(BUILD_DIR)/libnode.a:
	powershell -NoProfile -ExecutionPolicy Bypass -File gen_node_lib.ps1 -NodeExe "$(NODE_EXE)" -OutLib "$(NODE_LIB)"

$(TOKENIZERS_LIB): $(TOKENIZERS)/lib/libtokenizers_c.a
	objcopy -R .drectve $< $@

$(BIN_DIR)/$(NODE_MODULE): $(ADDON_OBJS) $(NODE_LIB) $(TOKENIZERS_LIB)
	$(CC) $(CFLAGS) $^ -o $@ $(NODE_LDFLAGS)
```

- `gen_node_lib.ps1` (repo root, so it is not swallowed by the gitignored `tools/`) lists the `napi_*` exports of the **running** `node.exe` with `objdump -p`, writes a `.def`, and runs `dlltool` to produce a GNU import library — this avoids depending on an MSVC `node.lib` (MinGW's `ld` cannot consume the MSVC import records) and always matches the active Node version.
- N-API headers come from the node-gyp cache (`%LOCALAPPDATA%/node-gyp/Cache/<ver>/include/node`); N-API is ABI-stable, so headers from an older version link/run against a newer Node.
- `NODE_LDFLAGS` adds `-shared` plus the Rust runtime deps (`-lws2_32 -luserenv -lbcrypt -lntdll -ladvapi32 -lole32 -loleaut32 -lpsapi -lshell32 -lshlwapi -lcrypt32`). The tokenizers lib is a MinGW (`x86_64-pc-windows-gnu`) Rust staticlib (identified by its `___chkstk_ms` references), so it links with the same gcc toolchain as the rest of the engine; `libtokenizers_cpp.a` is not needed (the C API is used directly).
- Rust staticlibs carry `.drectve` `-exclude-symbols` records (thousands of them in `libtokenizers_c.a`). MinGW `ld` does not understand them and prints one `unrecognized` / `corrupt .drectve` warning block per member — ~20,000 stderr lines per addon link. The Makefile strips the section once with `objcopy -R .drectve` into the cached `build/libtokenizers_c.a` (rebuilt only when the vendored lib changes), so the link is silent. The directives only matter to MSVC linkers; GNU `ld` ignores them.

> **Windows quirks** (important): the effective recipe shell is `cmd.exe`, so `mkdir`/`if exist` must use **backslashes** (`bin\shader`) — cmd treats `/` as a switch prefix — and folder names must not contain spaces (GNU make word-splits `$(wildcard)`/`$(foreach)` output on spaces, hence `Full-Attention/` and `Linear-Attention/` rather than `Full Attention/`).

---

## 6. Validation Harness

Each `validate*` function in src/validation.c:

1. Generates reference output on CPU (exact float math).
2. Transposes weights with `transpose_block16`, allocates `MEMORY_RAM` input/weight and `MEMORY_VRAM` output buffers, uploads them.
3. Builds an `operation` and runs it.
4. Reads back the GPU output and prints via `report()`:

```c
static void report(const char* name, int idx, const float* out, const float* ref, int count, double ms) {
    float err = 0.0f;
    for (int i = 0; i < count; i++) {
        float e = fabsf(out[i] - ref[i]);
        if (e > err) err = e;
    }
    printf("%s: shader[%d]= %f ref[%d]= %f max_err= %f\n", name, idx, out[idx], idx, ref[idx], err);
    printf("%s time: %.3f ms\n", name, ms);
}
```

Key CPU references:

- `rms_norm_apply` — RMSNorm: `xn = x * gamma / sqrt(mean(xÂ²) + 1e-6)`.
- `gemv_ref_fp32/fp16/int8/int4` and `gemm_ref_fp16/int8/int4` — naive MÃ—KÃ—N loops (int4/8 apply dequant `q*scale - zero`).
- `swiglu_ref_*` — rms-norm — gate & up GEMM — `o = silu(gate) * up`.
- `qkv_rope_ref` — per-head RMSNorm — partial RoPE over the first 64 dims (`angle = token * theta[col%32]`), q additionally scaled by `1/16` for the attention QK scaling.
- `validate_attention` / `validate_attention_multi` — online-softmax attention, causal masked (`t <= m`) in the multi-token version.
- `deltanet_ref` — the sequential gated delta-net recurrence (see Â§7.3).
- `lmhead_argmax_ref_fp16` — streams one logit column at a time (no full logits array), tracks the running max with smallest-index tie-break.

`validation()` (src/validation.c, reached via `main.exe val`) wires everything: it keeps the original M=1 GEMV validations and adds M=64 (`Mg = 64`) GEMM validations with an M-token input `inputM = getData(4321, Mg, K)`, plus the FP16 lm-head argmax validation over `MODEL_VOCAB` (weights `lmHeadFP16 = getDataFP16(15001, K, vocab_size)`, ~640 MB). `src/compute.c` no longer holds the harness — it now hosts `serverMain` (the persistent inference loop) and `memInfo` (device heap dump).

---

## 7. Shader Documentation

Common patterns: `TS = 16` (16Ã—16 tiling), `vec = 4` (vec4 dot products), workgroup = 256 threads (`local_size_x = local_size_y = 16`), shared tiles `Asub[4][16]` / `Bsub[4][16]` (vec4 over k), **k-tile loop bound = `K / TS`** (each iteration advances k by 16 floats; ×tile vec4 index `m*(K/4) + t*4 + kv`).

### 7.1 Utility — matmul primitives

#### `GEMV-*` (decode path, M=1)

Each of 256 threads computes one output column `n`; input row staged in shared as vec4; weight read as `uvec4` per column pair of k-vec4s. FP16 inner loop:

```glsl
#pragma unroll
for (uint k=0;k<dims.K/vec; k+=2) {
    uvec4 rawInput = B[baseRow + globalCol];
    vec4 outVectorA = vec4(unpackHalf2x16(rawInput.x), unpackHalf2x16(rawInput.y));
    vec4 outVectorB = vec4(unpackHalf2x16(rawInput.z), unpackHalf2x16(rawInput.w));
    acc += dot(cache[k], outVectorA);
    acc += dot(cache[k+1], outVectorB);
    baseRow += dims.N;
}
```

INT8 does `k += 4` with `unpackUint8x4` per component Ã— fp32 scale ' zero; Q4 does `k += 8` with `unpackInt4x4` low/high shifts (fp16 scale/zero read as `uvec2` through `unpackFp16x4`). The Q4 decode GEMVs take the block size in the push constants and pick one of two dequant modes: `block >= 256` pre-scales the shared activation tile once (the subgroup zero-point reduction), while `block < 256` applies that output column's own scale/zero to the unpacked weights in the inner loop (see Â§4.3).

#### `GEMV-ADD-*`

Same as GEMV plus a residual buffer: `C[globalCol] = acc + R[globalCol]` (residual-add for the delta-net output layer).

#### `GEMM-*` (prefill path) — the tiling template

Full source of the foundation shader — every fused GEMM variant derives from this pattern:

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define TS 16
#define vec 4

layout(local_size_x = TS, local_size_y = TS, local_size_z = 1) in;

layout(push_constant) uniform Dimensions { uint M; uint N; uint K; } dims;

layout(set = 0, binding = 0) readonly restrict buffer MatrixA { vec4 A[]; };
layout(set = 0, binding = 1) readonly restrict buffer MatrixB { uvec4 B[]; };
layout(set = 0, binding = 2) writeonly restrict buffer MatrixC { float C[]; };

shared vec4 Asub[4][TS];
shared vec4 Bsub[4][TS];

void main() {
    uint row = gl_LocalInvocationID.y;
    uint col = gl_LocalInvocationID.x;
    uint tid = row * TS + col;
    uint mGlobal = gl_WorkGroupID.y * TS + row;
    uint nGlobal = gl_WorkGroupID.x * TS + col;
    float acc = 0.0;

    for (uint t = 0; t < dims.K / TS; t++) {
        if (tid < 64) {
            uint m = tid / 4;
            uint kv = tid % 4;
            Asub[kv][m] = A[(gl_WorkGroupID.y * TS + m) * (dims.K / vec) + t * 4 + kv];
        }
        if (tid < 32) {
            uint block = tid / TS;
            uint n = tid % TS;
            uvec4 raw = B[(t * 2 + block) * dims.N + (gl_WorkGroupID.x * TS + n)];
            Bsub[block * 2][n]     = vec4(unpackHalf2x16(raw.x), unpackHalf2x16(raw.y));
            Bsub[block * 2 + 1][n] = vec4(unpackHalf2x16(raw.z), unpackHalf2x16(raw.w));
        }
        barrier();
        #pragma unroll
        for (uint kv = 0; kv < 4; kv++) acc += dot(Asub[kv][row], Bsub[kv][col]);
        barrier();
    }
    C[mGlobal * dims.N + nGlobal] = acc;
}
```

**How it works:** workgroup = 16 m-rows Ã— 16 n-cols. Per k-tile of 16 floats: 64 threads stage the A tile (16 rows Ã— 4 vec4s) and 32 threads stage the B tile (FP16: 2 `uvec4` per column = 8 halves each, unpacked into 2 vec4 k-rows); every thread then does 4 `dot()` FMAs (16 k Ã— 1 output element). Loop over `K/16` k-tiles.

INT8 B-load (one `uvec4` per column covers the whole 16-k tile + dequant):

```glsl
if (tid < 16) {
    uint n = tid;
    uint nGlobal = gl_WorkGroupID.x * TS + n;
    uint g = nGlobal / 256;
    uvec4 raw = B[t * dims.N + nGlobal];
    #pragma unroll
    for (uint kv = 0; kv < 4; kv++) {
        vec4 s = scale[g * (dims.K / vec) + t * 4 + kv];
        vec4 z = zeroPoint[g * (dims.K / vec) + t * 4 + kv];
        vec4 q = unpackUint8x4(raw[kv]);
        Bsub[kv][n] = q * s - z;
    }
}
```

INT4 B-load (half of the 32-row `uvec4` selected by k-tile parity):

```glsl
if (tid < 16) {
    uint n = tid;
    uint nGlobal = gl_WorkGroupID.x * TS + n;
    uint g = nGlobal / 256;
    uvec4 raw = B[(t / 2) * dims.N + nGlobal];
    bool hi = ((t & 1) == 1);
    uint c0 = hi ? raw.z : raw.x;
    uint c1 = hi ? raw.w : raw.y;
    Bsub[0][n] = unpackInt4x4(c0, lowShifts) * scale[g*(dims.K/vec)+t*4+0] - zeroPoint[g*(dims.K/vec)+t*4+0];
    Bsub[1][n] = unpackInt4x4(c0, highShifts)* scale[g*(dims.K/vec)+t*4+1] - zeroPoint[g*(dims.K/vec)+t*4+1];
    Bsub[2][n] = unpackInt4x4(c1, lowShifts) * scale[g*(dims.K/vec)+t*4+2] - zeroPoint[g*(dims.K/vec)+t*4+2];
    Bsub[3][n] = unpackInt4x4(c1, highShifts)* scale[g*(dims.K/vec)+t*4+3] - zeroPoint[g*(dims.K/vec)+t*4+3];
}
```

`unpackUint8x4` / `unpackInt4x4`:

```glsl
vec4 unpackUint8x4(uint packedData) {
    uvec4 bytes = (uvec4(packedData) >> uvec4(0, 8, 16, 24)) & uvec4(0xFF);
    return vec4(bytes);
}
const uvec4 lowShifts  = uvec4(0, 4, 8, 12);
const uvec4 highShifts = uvec4(16, 20, 24, 28);
vec4 unpackInt4x4(uint packedComponent, uvec4 shiftOffsets) {
    return vec4(uvec4(packedComponent) >> shiftOffsets & uvec4(0xF));
}
```

#### `GEMM-ADD-*`

Generic GEMM + residual: `C[mGlobal*dims.N + nGlobal] = acc + R[mGlobal*dims.N + nGlobal]`.

#### `RmsNorm-GEMV-*`

Fused RMSNorm + GEMV for decode: stage `x * gamma` into shared while computing `sum(xÂ²)` via subgroup add — `inv_rms`, then the GEMV loop with `cache[k] * inv_rms`.

#### `LMHead-GEMV-ArgMax-FP16` (decode, output token selection)

FP16 GEMV over the lm-head weight `[K][vocab]` fused with a workgroup-wide argmax reduction — one dispatch pass, no full logits buffer (the largest intermediate is 320 Ã— 4 B per output buffer). Workgroup = 256 threads, one output column each, `dispatchX = (vocab+255)/256`. Bindings: `0 = x (vec4[])`, `1 = w (uvec4[])`, `2 = maxValue (float[], fp32)`, `3 = maxIndex (uint[])`; push constants `{M, N, K}`. Full source:

```glsl
#version 450
#define TS 256
#define vec 4
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = TS, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform Dimensions { uint M; uint N; uint K; } dims;

layout(set = 0, binding = 0) readonly restrict buffer MatrixA { vec4 A[]; };
layout(set = 0, binding = 1) readonly restrict buffer MatrixB { uvec4 B[]; };
layout(set = 0, binding = 2) writeonly restrict buffer MaxValueBuffer { float maxValue[]; };
layout(set = 0, binding = 3) writeonly restrict buffer MaxIndexBuffer { uint maxIndex[]; };

shared vec4 Asub[2][TS];
shared float redValue[TS];
shared uint redIndex[TS];

void main() {
    uint col = gl_LocalInvocationID.x;
    uint globalCol = gl_WorkGroupID.x*TS + col;
    uint numTiles = (dims.K + TS - 1) / TS / vec;

    float acc = 0.0;
    if (globalCol < dims.N) {
        Asub[0][col] = A[col];
    }
    barrier();
    uint baseRow = 0;
    for (uint t = 0; t < numTiles; t++) {
        uint cur = t & 1, nxt = (t+1) & 1;
        if (globalCol < dims.N) {
            if (t+1 < numTiles) {
                Asub[nxt][col] = A[(t+1)*TS + col];
            }

            #pragma unroll
            for (uint k=0;k<TS; k+=2) {
                uvec4 rawInput = B[baseRow + globalCol];
                vec4 outVectorA = vec4(unpackHalf2x16(rawInput.x), unpackHalf2x16(rawInput.y));
                vec4 outVectorB = vec4(unpackHalf2x16(rawInput.z), unpackHalf2x16(rawInput.w));

                acc += dot(Asub[cur][k], outVectorA);
                acc += dot(Asub[cur][k+1], outVectorB);
                baseRow += dims.N;
            }
        }
        barrier();
    }
    if (globalCol >= dims.N) {
        acc = uintBitsToFloat(0xFF800000u);
    }

    redValue[col] = acc;
    redIndex[col] = globalCol;
    barrier();
    for (uint stride = TS/2; stride > 0; stride >>= 1) {
        if (col < stride) {
            float otherValue = redValue[col + stride];
            uint otherIndex = redIndex[col + stride];
            if (otherValue > redValue[col] || (otherValue == redValue[col] && otherIndex < redIndex[col])) {
                redValue[col] = otherValue;
                redIndex[col] = otherIndex;
            }
        }
        barrier();
    }
    if (col == 0) {
        maxValue[gl_WorkGroupID.x] = redValue[0];
        maxIndex[gl_WorkGroupID.x] = redIndex[0];
    }
}
```

**How it works:** the GEMV loop is identical to `GEMV-FP16` (ping-pong `Asub`, 2 vec4 k-rows per `uvec4` weight load). Out-of-range threads (`globalCol >= N`, partial last workgroup) skip the compute but still execute every barrier, and carry `-inf` (`uintBitsToFloat(0xFF800000u)`) so they can never win the reduction. Then a shared tree reduction over 256 threads finds the max logit and its column; ties resolve to the **smaller index** for determinism. Thread 0 writes the workgroup's winner to `maxValue[wgID]` / `maxIndex[wgID]` (fp32 value + global column index).

A sampling sibling `LMHead-GEMV-FP16.comp` (Â§7.7) is the same RMSNorm—GEMV but skips the reduction and writes the raw logit per column to a `logits[1..vocab]` buffer; `buildLmHead` picks one or the other from `g->sampling`.

#### `ArgMax-Reduce.comp` (second stage — greedy argmax or full-logits sampler)

One workgroup of `TS = 1024` threads, selected by a `mode` push constant over 9 storage-buffer bindings. Push constants are now `{vocabSize, doIncrement, passIdx, mode}`; `dispatchX = 1`. Bindings:

- `0 = maxValue (float[])`, `1 = maxIndex (uint[])` — per-group winners from `LMHead-GEMV-ArgMax` (used only by `mode == 0`).
- `2 = logits (vec4[], read+write)` — full logit vector from `LMHead-GEMV-FP16`, viewed as `vocab/4` vec4s (used only by `mode == 1`).
- `3 = sampleParams` — std430 block `{temperature, repPenalty, penaltyLength, topK, topP, minP, presencePenalty}` (host-written per request).
- `4 = history (uint[], read+write)` — repetition-penalty ring of the last `penaltyLength` sampled ids, sentinel-filled per request.
- `5 = rng (uint[], read+write)` — 1-element xorshift32 state, seeded host-side and advanced on-GPU per sample.
- `6 = result (uint[])`, `7 = position (uint[])`, `8 = tokenIds (uint[])` (in/out as in the greedy path).

**`mode == 0` (greedy).** `numGroups = ceil(vocabSize/256)` — fixed to the lm-head GEMV's 256-thread workgroup size, **not** this shader's `TS` — so each thread strided-loads `maxValue[i]`/`maxIndex[i]` (`i += TS`, `-inf` padding beyond `numGroups`), keeping the running best with a smallest-index tie-break, then one shared tree reduction; thread 0 writes `result[passIdx]` and `tokenIds[0]`, and bumps `position` when `doIncrement == 1` — the single per-token position update, kept out of any per-layer shader so N layers can never bump it NÃ—.

**`mode == 1` (sampling).** A single-dispatch sampler over the full `logits` vector, where each thread owns `ceil(vocab/4/TS)` contiguous vec4s and phases are separated by `barrier()`; all cross-thread reductions use subgroup ops (`subgroupAdd/Max/Min` + a tiny `TS/64` shared combine, 2 barriers instead of an 8-deep tree):

1. Load `sampleParams`; if the penalty is active, build a membership bitmap in shared memory: clear `penBits[8192]` (262144 ids), then threads `tid < penaltyLength` set one bit each via `atomicOr(penBits[id>>5], 1u<<(id&31))` — sentinel slots (`0xFFFFFFFF`) and out-of-range ids are guarded, duplicates are idempotent (a token is penalized **once** even if it occurs several times in the window), and the cost is `O(vocab/32 + penaltyLength)` — independent of `penaltyLength` (an earlier per-vec4 history scan made penalty_len=256 cost ~5 ms/token and drop sampling from ~30 to ~26 tok/s). The bitmap is shared by both penalties — active if `repPenalty != 1` or `presencePenalty != 0`.
2. Transform in place: `a = logit / temperature`; for penalized ids apply CTRL logit scaling `a = (a > 0) ? a/repPenalty : a*repPenalty` (when `repPenalty != 1`) then subtract the presence term `a -= presencePenalty/temperature` (when `presencePenalty != 0`) — a flat penalty for any id present ≥1× in the window, expressed in the raw-logit frame so it matches Qwen/vLLM exactly; the check is a single shared load per vec4 (`penBits[j>>5]`, 4 bits extracted with one shift: `j` is 4-aligned so the 4 ids never straddle a 32-bit word); subgroup-reduce min/max to `gMax`/`gMin`.
3. Convert to probability space in place: `p = exp(a ' gMax)`; each thread caches its chunk's `mySum`/`myMax`/`myMin`, and `Zall = sumAll(mySum)` — all later comparisons are plain `p >= thr` with the bisection threshold converted once per iteration by thread 0 (`gThrP = exp(mid ' gMax)`), so no `exp` inside any scan loop.
4. Top-k (if `topK > 0`): 28-iteration binary search over `[gMin, gMax]` for the k-th-largest threshold; threads with `myMax < thr` contribute 0 without scanning.
5. Top-p (if `topP < 1`): `Zk = sumAll(massAbove(topK threshold))`, then a 28-iteration binary search for the nucleus threshold over the same bracket (the nucleus set is always a subset of the top-k set, so `[gMin, gMax]` converges to the same threshold); `massAbove(thr)` short-circuits per thread: `myMax < thr — 0`, `myMin >= thr — mySum`, otherwise scan.
6. Min-p (if `minP > 0`): `gThrP = max(gThrP, minP)` — one shared-max, since the kept set is a threshold cut and `p_max = 1` in prob space.
7. Final mass + per-thread block masses — exclusive prefix (`cumBase`); draw `u = xorshift32(rng[0]) ∈ [0,1)`; pick the token whose cumulative mass first exceeds `u`.
8. Thread 0 writes `result[passIdx]` + `tokenIds[0]`, appends the id to `history[position % penaltyLength]`, advances `rng[0] = xorshift32(seed)`, and bumps `position` when `doIncrement`.

This is the optimized form (2026-08): the original 48+48-iteration bisections rescanned all `vocab` floats (scalar loads, `exp` per element, ~1000 barrier-serialized tree reductions) and cost ~15 ms/token — the reason sampling ran at ~20 tok/s vs 30+ greedy. The rewrite (in-place probability conversion, per-thread min/max/sum early-outs, vec4 access, 28 bisections, subgroup reductions, TS 256—1024) brought `ArgMax-Reduce(mode 1)` to **0.64 ms** and the sampling/greedy gap to under 1 tok/s (24.4 vs 24.6 tok/s measured at ctx  130); greedy output is unchanged byte-for-byte. The repetition-penalty bitmap (same round) then made the cost flat in `penaltyLength`: 24.7 tok/s at `penalty_len=256` vs 24.2 at 16 (previously ~26.1 vs ~30). Mode-0 numerics are unchanged; only the group-count formula was fixed to stay tied to the 256-thread GEMV groups (a TS=1024 build with the old `ceil(V/TS)` truncated the argmax to the first 81 groups — caught as a greedy-text divergence and fixed).

`buildLmHead` chains `LMHead-GEMV-ArgMax-FP16` + `ArgMax-Reduce(mode 0)` for greedy, or `LMHead-GEMV-FP16` + `ArgMax-Reduce(mode 1)` for sampling (Â§7.7); both write the same `result`/`tokenIds`/`position` outputs, so `generateTokens` is unchanged between the two.

---

### 7.2 Full Attention

> **GQA mapping is runtime-generalized.** All attention shaders compute `kvh = head / param.gqa` where `gqa = heads/kvHeads` is pushed from config. The old `kvh = head/KV_HEADS` only worked when `heads == kvHeadsÂ²` — coincidentally true for 9B (16 = 4Â²) but wrong for 2B (8 heads, 2 kv-heads). Every dims constant (HEADS, KV_TOTAL_ROWS, MAXCTX, Q_DIM) comes from the push constant block (Â§4.5); only `HEAD_DIM 256` remains a compile-time constant (family-invariant across Qwen3.5).

#### `RmsNorm-QKV-*` (decode, one token)

Workgroup = one head (256 columns). RMSNorm over K (divisor = spec-constant `d_model`) — project `N = qkvN` columns (q, g, k, v sections at runtime offsets); `g` is the sigmoid output gate — for q/k heads: per-head RMSNorm over headDim cols, then **partial RoPE** over the first `rotaryDim` dims (`angle = pos * theta[col & (rotaryDim/2 - 1)]`, the rest pass through); q written to `qOut` (scaled by `inversesqrt(headDim)` — the attention QK `1/šhead_dim` folded into q), k/v stored to the KV cache at slot `pos * (vOffset ' kOffset)`. INT8/Q4 additionally quantize k/v per (head, token) with min/max over the headDim cols and write scale/zero at stride `kvh*maxCtx + pos`. **Position source:** the write slot and RoPE index come from a shared `uint[1]` position buffer (last binding) instead of push constants — the value is the current context length (how many tokens are already cached).

#### `RmsNorm-QKV-GEMM-*` (prefill, head-wide tile — **legacy, unwired**)

Superseded by the `RmsNorm-QKV-GEMM2-*` + `Rope-GEMM-*` pair (Â§7.6); kept compiled for the validation harness (`validateQkvRopeGEMM*`). Workgroup = (head, 16-token block); output tile = 16 tokens Ã— 256 columns; the 16Ã—16 k-tiling from Â§7.1 inside. The k/v cache slot is `(tokenIdx + mGlobal) * cacheRows` and RoPE uses `tokenIdx + mGlobal` (`tokenIdx` = chunk base, kept as a push constant so layers in the same chunk stay consistent). After the projections, the first workgroup's thread 0 writes the new context length to the shared position buffer: `position[0] = tokenIdx + M` — an **absolute** write, so every layer writing the same value is idempotent (last-write-wins is harmless; `execute()` barriers serialize it). Full FP16 source (snapshot predates the partial-RoPE/`1e-6`/q-scale changes — the live `.comp` now uses `ROTARY_DIM 64`, `EPSILON 1e-6`, and writes q scaled by `1/16`):

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_atomic_float : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define TS 16
#define vec 4
#define HEAD_DIM 256
#define HALF 128

layout(local_size_x = TS, local_size_y = TS, local_size_z = 1) in;

layout(push_constant) uniform Dimensions {
    uint M; uint N; uint K; uint tokenIdx; uint kOffset; uint vOffset; uint context_length;
} param;

const float EPSILON = 1e-5;

layout(set = 0, binding = 0) buffer DataBuffer { vec4 x[]; };
layout(set = 0, binding = 1) readonly buffer GammaBuffer { vec4 gamma[]; };
layout(set = 0, binding = 2) readonly buffer weightBuffer { uvec4 w[]; };
layout(set = 0, binding = 3) readonly buffer thetaBuffer { float theta[]; };
layout(set = 0, binding = 4) buffer qOutBuffer { float qOut[]; };
layout(set = 0, binding = 5) buffer kCacheBuffer { uint16_t kCache[]; };
layout(set = 0, binding = 6) buffer vCacheBuffer { uint16_t vCache[]; };

shared vec4 Asub[4][TS];
shared vec4 Bsub[4][HEAD_DIM];
shared float row_sums[TS][TS];
shared float inv_rms_sh[TS];
shared float acc_sh[TS][HEAD_DIM];
shared float norm_sh[TS][HEAD_DIM];
shared float inv_rms2_sh[TS];

void main() {
    uint row = gl_LocalInvocationID.y;
    uint col = gl_LocalInvocationID.x;
    uint tid = row * TS + col;
    uint head = gl_WorkGroupID.x;
    uint mGlobal = gl_WorkGroupID.y * TS + row;
    uint nBase = head * HEAD_DIM;
    uint cacheRows = param.vOffset - param.kOffset;

    // per-token RMSNorm over K: 16 threads per row, tree reduction
    float sum_sq = 0.0;
    for (uint i = 0; i < param.K / (TS * vec); i++) {
        vec4 v = x[mGlobal * (param.K / vec) + col * (param.K / (TS * vec)) + i];
        sum_sq += dot(v, v);
    }
    row_sums[row][col] = sum_sq;
    barrier();
    for (uint stride = TS / 2; stride > 0; stride >>= 1) {
        if (col < stride) row_sums[row][col] += row_sums[row][col + stride];
        barrier();
    }
    if (col == 0) inv_rms_sh[row] = inversesqrt(row_sums[row][0] / float(param.K) + EPSILON);
    barrier();

    // GEMM: 16 tokens x 256 cols, k-tiles of 16
    float acc[16];
    #pragma unroll
    for (uint j = 0; j < 16; j++) acc[j] = 0.0;

    for (uint t = 0; t < param.K / TS; t++) {
        if (tid < 64) {
            uint m = tid / 4;
            uint kv = tid % 4;
            uint mg = gl_WorkGroupID.y * TS + m;
            Asub[kv][m] = x[mg * (param.K / vec) + t * 4 + kv] * gamma[t * 4 + kv] * inv_rms_sh[m];
        }
        for (uint i = tid; i < HEAD_DIM * 2; i += TS * TS) {
            uint block = i / HEAD_DIM;
            uint n = i % HEAD_DIM;
            uvec4 raw = w[(t * 2 + block) * param.N + (nBase + n)];
            Bsub[block * 2][n]     = vec4(unpackHalf2x16(raw.x), unpackHalf2x16(raw.y));
            Bsub[block * 2 + 1][n] = vec4(unpackHalf2x16(raw.z), unpackHalf2x16(raw.w));
        }
        barrier();
        #pragma unroll
        for (uint kv = 0; kv < 4; kv++) {
            vec4 a = Asub[kv][row];
            #pragma unroll
            for (uint j = 0; j < 16; j++) acc[j] += dot(a, Bsub[kv][col * 16 + j]);
        }
        barrier();
    }

    // stage accumulators into shared (head-wide access for QK-norm / RoPE)
    #pragma unroll
    for (uint j = 0; j < 16; j++) acc_sh[row][col * 16 + j] = acc[j];
    barrier();

    // v heads: store raw to cache (no norm, no rope)
    if (nBase >= param.vOffset) {
        uint kvHead = head - param.vOffset / HEAD_DIM;
        #pragma unroll
        for (uint j = 0; j < 16; j++) {
            uint d = col * 16 + j;
            vCache[mGlobal * cacheRows + kvHead * HEAD_DIM + d] = uint16_t(packHalf2x16(vec2(acc_sh[row][d])));
        }
        return;
    }

    // q/k heads: per-head RMSNorm over 256 cols
    float ss = 0.0;
    #pragma unroll
    for (uint j = 0; j < 16; j++) {
        float v2 = acc_sh[row][col * 16 + j];
        ss += v2 * v2;
    }
    row_sums[row][col] = ss;
    barrier();
    for (uint stride = TS / 2; stride > 0; stride >>= 1) {
        if (col < stride) row_sums[row][col] += row_sums[row][col + stride];
        barrier();
    }
    if (col == 0) inv_rms2_sh[row] = inversesqrt(row_sums[row][0] / float(HEAD_DIM) + EPSILON);
    barrier();

    // RoPE (angle uses the token's absolute index)
    float token = float(param.tokenIdx + mGlobal);
    #pragma unroll
    for (uint j = 0; j < 16; j++) {
        uint d = col * 16 + j;
        float angle = token * theta[d & (HALF - 1u)];
        float cs = cos(angle);
        float sn = sin(angle);
        float v;
        if (d < HALF)
            v = acc_sh[row][d] * cs - acc_sh[row][d + HALF] * sn;
        else
            v = acc_sh[row][d - HALF] * sn + acc_sh[row][d] * cs;
        norm_sh[row][d] = v * inv_rms2_sh[row];
    }
    barrier();

    if (nBase < param.kOffset) {
        #pragma unroll
        for (uint j = 0; j < 16; j++) {
            uint d = col * 16 + j;
            qOut[mGlobal * param.kOffset + nBase + d] = norm_sh[row][d];
        }
    } else {
        uint kvHead = head - param.kOffset / HEAD_DIM;
        #pragma unroll
        for (uint j = 0; j < 16; j++) {
            uint d = col * 16 + j;
            kCache[mGlobal * cacheRows + kvHead * HEAD_DIM + d] = uint16_t(packHalf2x16(vec2(norm_sh[row][d])));
        }
    }
}
```

INT8 variant: B-load is 256 `uvec4`s (one per column, full 16-k tile) + scale/zero vec4s; cache stays fp16. INT4 variant: parity-selected nibble half (like Â§7.1), `uint8` k/v cache plus per-(kv-head, token) min/max quantization (`quantizeParams`/`quantize` helpers with per-row `qscale_sh[TS]`/`qzero_sh[TS]`).

#### `Att-full-*` (decode, one query)

Workgroup = one query head (`dispatchX = heads`). Push `{maxCtx, kvHeads, kvRows, gqa, headDim, heads}`; `kvh = q_head_id / param.gqa`. Query staged in shared (headDim floats). Online softmax over KV-token tiles of 256: per tile compute scores `qÂ·k` (fp16 or dequantized int8/int4 keys), tile max via tree reduction, exp, tile sum, then rescale running accumulator with `alpha = exp(m_prev ' m_next)` / `beta = exp(tile_max ' m_next)` and accumulate `PÂ·V` per output dim. Output `o = acc / l`. Context length comes from the shared position buffer (`context = position[0] + 1`) — so the decode chain is fully GPU-driven; the decode QKV shader of the same layer writes the new token's k/v at slot `position[0]` first, and the +1 makes the total context correct for attention. K is read from the transposed cache (`key[(kv_row_base + d) Â· MAXCTX + s]`); V is read token-major (`value[t Â· KV_TOTAL_ROWS + row]`).

#### `Att-full-GEMM-*` (prefill, flash attention, causal)

Workgroup = (head, q-tile); 16 queries Ã— all KV tokens; 16Ã—16 score tiles; causal mask `kvTok <= mTok`. Full FP16 source:

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define TS 16
#define HEAD_DIM 256
#define KV_HEADS 4
#define KV_TOTAL_ROWS (KV_HEADS * HEAD_DIM)
#define Q_DIM (16 * HEAD_DIM)
#define NEG_INF -1e30

layout(local_size_x = TS, local_size_y = TS, local_size_z = 1) in;

layout(push_constant) uniform Dimensions { uint context_length; } param;

layout(set = 0, binding = 0) readonly buffer keyBuffer   { uint16_t key[]; };
layout(set = 0, binding = 1) readonly buffer valueBuffer { uint16_t value[]; };
layout(set = 0, binding = 2) readonly buffer queryBuffer { vec4 query[]; };
layout(set = 0, binding = 3) writeonly buffer outBuffer  { float y[]; };

shared vec4    Qs[HEAD_DIM / 4][TS];   // 16 KB
shared f16vec4 KVs[HEAD_DIM / 4][TS]; // 8 KB, reused for K then V
shared float s_max[TS][TS];
shared float s_exp[TS][TS];
shared float s_p[TS][TS];

vec4 loadHalf4(uint b0, uint b1, uint b2, uint b3) {
    return vec4(unpackHalf2x16(b0 | (b1 << 16)),
                unpackHalf2x16(b2 | (b3 << 16)));
}

void main() {
    uint r = gl_LocalInvocationID.y;
    uint c = gl_LocalInvocationID.x;
    uint tid = r * TS + c;
    uint h = gl_WorkGroupID.x;
    uint kvh = h / KV_HEADS;
    uint mBase = gl_WorkGroupID.y * TS;
    uint mTok = mBase + r;

    // stage Q tile (16 tokens x 256 dims)
    for (uint i = tid; i < (HEAD_DIM / 4) * TS; i += TS * TS) {
        uint dv = i / TS;
        uint tok = i % TS;
        Qs[dv][tok] = query[(mBase + tok) * 1024 + h * (HEAD_DIM / 4) + dv];
    }
    barrier();

    float m_prev = NEG_INF;
    float l_prev = 0.0;
    vec4 oacc[4];
    #pragma unroll
    for (uint j = 0; j < 4; j++) oacc[j] = vec4(0.0);

    uint numTiles = (param.context_length + TS - 1) / TS;
    for (uint t = 0; t < numTiles; t++) {
        uint kvTok = t * TS + c;

        // stage K tile from KV cache ([token][row] layout)
        for (uint i = tid; i < (HEAD_DIM / 4) * TS; i += TS * TS) {
            uint dv = i / TS;
            uint tok = i % TS;
            uint base = (t * TS + tok) * KV_TOTAL_ROWS + kvh * HEAD_DIM + dv * 4;
            KVs[dv][tok] = f16vec4(loadHalf4(uint(key[base]), uint(key[base + 1]), uint(key[base + 2]), uint(key[base + 3])));
        }
        barrier();

        // scores with causal mask (64 vec4 dots over the 256-dim head)
        float score = NEG_INF;
        if (kvTok < param.context_length && kvTok <= mTok) {
            score = 0.0;
            for (uint dv = 0; dv < HEAD_DIM / 4; dv++)
                score += dot(Qs[dv][r], vec4(KVs[dv][c]));
        }
        s_max[r][c] = score;
        barrier();
        for (uint stride = TS / 2; stride > 0; stride >>= 1) {
            if (c < stride) s_max[r][c] = max(s_max[r][c], s_max[r][c + stride]);
            barrier();
        }
        float tile_max = s_max[r][0];
        float m_next = max(m_prev, tile_max);
        float alpha = (m_prev == NEG_INF) ? 0.0 : exp(m_prev - m_next);
        float beta = exp(tile_max - m_next);
        l_prev = alpha * l_prev;

        // exp(score - tile_max) kept in s_p; sum reduced in s_exp
        float e = (score == NEG_INF) ? 0.0 : exp(score - tile_max);
        s_p[r][c] = e;
        s_exp[r][c] = e;
        barrier();
        for (uint stride = TS / 2; stride > 0; stride >>= 1) {
            if (c < stride) s_exp[r][c] += s_exp[r][c + stride];
            barrier();
        }
        l_prev += beta * s_exp[r][0];
        #pragma unroll
        for (uint j = 0; j < 4; j++) oacc[j] *= alpha;

        // stage V tile (reuses the same shared array as K)
        for (uint i = tid; i < (HEAD_DIM / 4) * TS; i += TS * TS) {
            uint dv = i / TS;
            uint tok = i % TS;
            uint base = (t * TS + tok) * KV_TOTAL_ROWS + kvh * HEAD_DIM + dv * 4;
            KVs[dv][tok] = f16vec4(loadHalf4(uint(value[base]), uint(value[base + 1]), uint(value[base + 2]), uint(value[base + 3])));
        }
        barrier();

        // P * V: thread owns 16 output dims (4 vec4 accumulators)
        for (uint kk = 0; kk < TS; kk++) {
            float p = s_p[r][kk] * beta;
            #pragma unroll
            for (uint j = 0; j < 4; j++) oacc[j] += p * vec4(KVs[c * 4 + j][kk]);
        }
        m_prev = m_next;
        barrier();
    }

    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        uint base = mTok * Q_DIM + h * HEAD_DIM + c * 16 + j * 4;
        vec4 o = (l_prev > 0.0) ? (oacc[j] / l_prev) : vec4(0.0);
        y[base + 0] = o.x;
        y[base + 1] = o.y;
        y[base + 2] = o.z;
        y[base + 3] = o.w;
    }
}
```

**How it works:** Q staged once per workgroup. Per 16-KV-token tile: K staged (fp16), scores = 64 vec4 dots/thread with causal mask, row-wise tile max via tree reduction, online-softmax rescale (`alpha`/`beta` per flash-attention), `exp(score ' tile_max)` kept in `s_p` for the PÂ·V step (the sum reduction overwrites `s_exp` in place). V then staged into the same shared buffer and multiplied with the per-row probabilities. Output `O = acc / l`. INT8 is identical (cache is fp16); INT4 dequantizes `uint8` cache with per-(kv-head, token) scale/zero during K/V staging.

---

### 7.3 Linear Attention (Gated DeltaNet)

#### `RmsNorm-LinearProj-*`

RMSNorm + single GEMM over `N = projN` (= `in_proj_qkv`  `in_proj_z`  `in_proj_a`  `in_proj_b`), output routed by column with **runtime** per-model offsets (9B: q 2048, k 4096, v 8192, z 12288, a 12320; 2B: q 2048, k 4096, v 6144, z 8192, a 8208 — pushed as `{kOff, vOff, zOff, aOff, bOff}`): `q[kOff]` (cols < kOff), `k[kOff]` (< vOff), `v[vOff'kOff]` (< zOff), `z[zOff'vOff]` (< aOff), `a[nV]` (< bOff), `b[nV]` (else). Only q/k/v (cols < zqkvN) pass through `Conv-SiLU`; z/a/b are consumed directly.

#### `Embed-RmsNorm-LinearProj-*` (token-embedding fused variant, FP16 only)

Replaces the `x` input with a token-id lookup: the token id(s) (a `uint` buffer, GEMV: 1 element, GEMM: `M` elements) index into the **tied lm-head buffer** — the same block-transposed FP16 `uvec4[]` layout as `LMHead-GEMV-ArgMax-FP16.comp` binding 1 — to fetch the 4096-dim embedding row, then RMSNorm + LinearProj proceed exactly as `RmsNorm-LinearProj-*`. This shader is meant to run right after the LM-head argmax (token selection) so the next decode step can start from the selected token's embedding without CPU readback.

- **Bindings (GEMV and GEMM):** `0 = tokenIds (uint[])`, `1 = lm (uvec4[], transposed lm head [K][vocab])`, `2 = gamma (vec4[])`, `3 = w (uvec4[], transposed w_in [K][12352])`, `4..9 = q/k/v/z/a/b out (float[])`, `10 = embOut`. Push constants `{M, N, K, V}`.
- **Embedding fetch:** vec4 index `i` (0..1023) reads `lm[(i/2)*V + tok]`; even `i` takes `unpackHalf2x16(raw.x), unpackHalf2x16(raw.y)`, odd `i` takes `.z,.w` (each `uvec4` = 2 vec4 k-rows of a column).
- **GEMV (`Embed-RmsNorm-LinearProj-FP16.comp`):** 256 threads/col, subgroup RMSNorm sum identical to `RmsNorm-LinearProj-FP16`; `dispatchX = (12352+255)/256 = 50`.
- **GEMM (`Embed-RmsNorm-LinearProj-GEMM-FP16.comp`):** 16Ã—16 tiled clone of `RmsNorm-LinearProj-GEMM-FP16`; per-token embedding fetched by `tokenIds[mGlobal]` in both the sum-sq pass and the ×tile staging; `dispatchX = 12352/16 = 772`, `dispatchY = M/16`.

Note the lm-head column fetch is strided by `V` uvec4s (column-major access into a 640 MB buffer), so the GEMM variant is slower than the plain `x`-input GEMM (~150 ms vs ~45 ms at M=64) — a layout artifact of tied embeddings, not a correctness issue.

#### `GatedDeltaNet.comp` (decode, one token)

Workgroup = one v-head (`h`); state `S` is a 128Ã—128 matrix per head in VRAM. Bindings: `0..2 = q/k/v`, `3..4 = aRaw/bRaw (32)`, `5 = S`, `6 = yGated`, `7 = z (4096)`, `8 = norm.weight (128)`, `9 = A_log (32)`, `10 = dt_bias (32)`. Push `{N_V, N_QK, DIM}`. Per token:

- q/k are L2-normalized per qk-head (`qk = h>>1`) over the 128 dims: `kÂ·rsqrt(Î£kÂ² + 1e-6)`, `qÂ·rsqrt(Î£qÂ² + 1e-6)Â·(1/š128)` — thread 0 serially sums, then each of the 32 `k4/q4` vec4s is scaled in shared memory.
- `alpha = exp('exp(A_log[h])Â·softplus(aRaw[h] + dt_bias[h]))`, `beta = 1/(1+exp('bRaw[h]))` — **per value-head** (`aRaw[h]`/`bRaw[h]`, 32 values).
- `vhat = SÂ·K`, `delta = V ' alphaÂ·vhat` (the decayed-state correction), `y = alphaÂ·(SÂ·Q) + betaÂ·deltaÂ·(KÂ·Q)`.
- state update: `S    alphaÂ·S + betaÂ·deltaÂ·Káµ€`.
- output gate (fused epilogue): per-head RMSNorm of `y` over 128 dims — `yGated = yÂ·rsqrt(Î£yÂ²/128 + 1e-6)Â·norm.weightÂ·silu(z)`.

The conv+SiLU of qkv runs upstream in `Conv-SiLU.spv` (below); z/a/b are consumed unchanged.

#### `Conv-SiLU.comp` (depthwise causal conv + SiLU on the qkv projection)

Runs between the input projection and the delta rule for every delta-net layer. It applies the short causal `conv1d` (kernel 4) depthwise over the **zqkvN** qkv channels (q, k, v sections at runtime offsets) followed by SiLU, writing the result in place so `GatedDeltaNet*` consumes the transformed q/k/v (`a`/`b` are untouched).

- **Precision-agnostic, FP32.** Bindings: `0 = qProj`, `1 = kProj`, `2 = vProj`, `3 = conv (zqkvNÃ—4 FP32, channel-major)`, `4 = convHist (3Ã—zqkvN FP32 shift register)`. Push `{M, kOff, vOff, zqkv, realM}`. `local_size = 256`, `dispatchX = zqkvN/256`, one thread per channel.
- **Shift-register state (`convHist`):** per delta layer, slot 0/1/2 hold `Zqkv[t'1]` / `Zqkv[t'2]` / `Zqkv[t'3]` (zero-initialized = causal zero-padding). Each thread keeps a 3-slot rolling window in registers and loops `m = 0..steps'1` where `steps = min(M, realM)`:
  - `cur = zqkv[m][c]`, `z = w0Â·h2 + w1Â·h1 + w2Â·h0 + w3Â·cur` (`w0` = t'3    `w3` = current), `zqkv[m][c] = z / (1 + exp('z))`, then `h2=h1; h1=h0; h0=cur`.
- **`realM` matters:** prefill pads the token chunk to a 16-multiple; without the clamp, the conv history advances over the pad tokens (whose projections are garbage), and the poisoned shift register carried into decode made every subsequent delta layer explode (gotcha 38).
- Because the conv is **per-channel** (no cross-channel coupling), each thread owns its channel across the whole chunk — **no barriers needed**, and it is self-consistent across prefill-chunk boundaries and decode steps (no absolute position required). VRAM cost: `layers Ã— 3 Ã— zqkvN Ã— 4 B` (  2.25 MB at 24Ã—8192).

#### `GatedDeltaNet-GEMM.comp` (prefill, chunked linear attention)

Workgroup = one v-head; processes all M tokens in chunks of 16 inside the shader (state persists across chunks in VRAM, synchronized by `barrier()`). Bindings match the decode shader (`0..2 = q/k/v`, `3..4 = aRaw/bRaw`, `5 = S`, `6 = yGated`, `7 = z`, `8 = norm.weight`, `9 = A_log`, `10 = dt_bias`); push `{M, nV, nQk, realM}`. **Pad masking:** rows `mBase+m >= realM` get `g_sh = 0, beta_sh = 0` — without it the recurrence decayed and updated the state over the 16-padding tokens, corrupting the S-state carried into decode (gotcha 38). It implements the same HF block as the decode shader. The source embedded below shows the older prefix-product formulation (`alpha_P`/`w_sh`); the current revision is **log-domain** to avoid fp32 underflow at real decay rates (gotcha 24):

- `Qs[TS][DIM]` is staged alongside `Ks`, both L2-normalized per token (`kÂ·rsqrt(Î£kÂ²+1e-6)`, `qÂ·rsqrt(Î£qÂ²+1e-6)Â·(1/š128)`) by a `tid < TS` serial reduction before the dots; `Dq` reads `Qs` instead of global `Q`.
- Per-token `g_sh[m] = 'exp(A_log)Â·log(1+exp(a+dt_bias))`, `beta_sh[m] = sigmoid(b)`; `G[m+1] = G[m] + g_sh[m]` (cumulative **log** decay, `G[0]=0`).
- `dv = V_m ' exp(G[m+1])Â·(skv[i] + p)` with `p = Î£_{j<m} exp(G[m+1]'G[j+1])Â·beta_jÂ·delta_jÂ·Dk[j][m]` (all exponents    0, so no overflow/underflow-to-inf), stored as `delta[m][i]`.
- `ym = exp(G[m+1])Â·sqv[i] + Î£_{j<m} exp(G[m+1]'G[j+1])Â·beta_jÂ·delta_jÂ·Dq[j][m] + beta_mÂ·dvÂ·Dq[m][m]`.
- fused output gate writes `yGated = yÂ·rsqrt(Î£yÂ²/128+1e-6)Â·norm.weightÂ·silu(z)`.
- state update `S = exp(G[TS])Â·S   + Î£_m exp(G[TS]'G[m+1])Â·beta_mÂ·delta[m]·Ks[m]`.

```glsl
#version 450
#define TS 16
#define DIM 128

layout(local_size_x = TS, local_size_y = TS, local_size_z = 1) in;

layout(push_constant) uniform Dimensions { uint M; } param;

layout(set = 0, binding = 0) readonly buffer QBuffer { float Q[]; };
layout(set = 0, binding = 1) readonly buffer KBuffer { float K[]; };
layout(set = 0, binding = 2) readonly buffer VBuffer { float V[]; };
layout(set = 0, binding = 3) readonly buffer ARawBuffer { float aRaw[]; };
layout(set = 0, binding = 4) readonly buffer BRawBuffer { float bRaw[]; };
layout(set = 0, binding = 5) buffer StateBuffer { float S[]; };
layout(set = 0, binding = 6) buffer YGatedBuffer { float yGated[]; };

shared float Ks[TS][DIM];
shared float Dk[TS][TS];
shared float Dq[TS][TS];
shared float delta[TS][DIM];
shared float alpha_sh[TS];
shared float beta_sh[TS];
shared float alpha_P[TS + 1];
shared float w_sh[TS];
shared float skv[DIM];
shared float sqv[DIM];

void main() {
    uint r = gl_LocalInvocationID.y;
    uint c = gl_LocalInvocationID.x;
    uint tid = r * TS + c;
    uint h = gl_WorkGroupID.x;
    uint qk = h >> 1;

    uint numChunks = param.M / TS;

    for (uint ch = 0; ch < numChunks; ch++) {
        uint mBase = ch * TS;

        // per-token decay/gate and prefix products:
        //   alpha_m = sigmoid(-a_m), beta_m = sigmoid(b_m)
        //   P_m = prod_{p<m} alpha_p, w_m = beta_m / P_{m+1}
        if (tid == 0) {
            for (uint m = 0; m < TS; m++) {
                float a = aRaw[(mBase + m) * 32 + h];
                float b = bRaw[(mBase + m) * 32 + h];
                alpha_sh[m] = 1.0 / (1.0 + exp(a));
                beta_sh[m] = 1.0 / (1.0 + exp(-b));
            }
            alpha_P[0] = 1.0;
            for (uint m = 0; m < TS; m++) alpha_P[m + 1] = alpha_P[m] * alpha_sh[m];
            for (uint m = 0; m < TS; m++) w_sh[m] = beta_sh[m] / alpha_P[m + 1];
        }
        barrier();

        // stage K tile (Q/V/G read from global later)
        for (uint i = tid; i < TS * DIM; i += TS * TS) {
            uint m = i / DIM;
            uint d = i % DIM;
            Ks[m][d] = K[(mBase + m) * (16 * DIM) + qk * DIM + d];
        }
        barrier();

        // 16x16 GEMM over DIM: Dk[j][m] = K_j.K_m, Dq[j][m] = K_j.Q_m
        {
            float dk = 0.0;
            float dq = 0.0;
            uint qoff = (mBase + c) * (16 * DIM) + qk * DIM;
            for (uint d = 0; d < DIM; d += 4) {
                vec4 kj = vec4(Ks[r][d], Ks[r][d + 1], Ks[r][d + 2], Ks[r][d + 3]);
                vec4 km = vec4(Ks[c][d], Ks[c][d + 1], Ks[c][d + 2], Ks[c][d + 3]);
                dk += dot(kj, km);
                vec4 qm = vec4(Q[qoff + d], Q[qoff + d + 1], Q[qoff + d + 2], Q[qoff + d + 3]);
                dq += dot(kj, qm);
            }
            Dk[r][c] = dk;
            Dq[r][c] = dq;
        }
        barrier();

        // sequential scan over the 16 tokens (unavoidable: delta_m depends on delta_{j<m})
        for (uint m = 0; m < TS; m++) {
            if (tid < DIM) {
                uint i = tid;
                float sk = 0.0;
                float sq = 0.0;
                uint qoff = (mBase + m) * (16 * DIM) + qk * DIM;
                for (uint d = 0; d < DIM; d += 4) {
                    vec4 sv = vec4(S[h * (DIM * DIM) + i * DIM + d],
                                   S[h * (DIM * DIM) + i * DIM + d + 1],
                                   S[h * (DIM * DIM) + i * DIM + d + 2],
                                   S[h * (DIM * DIM) + i * DIM + d + 3]);
                    vec4 km = vec4(Ks[m][d], Ks[m][d + 1], Ks[m][d + 2], Ks[m][d + 3]);
                    sk += dot(sv, km);
                    vec4 qm = vec4(Q[qoff + d], Q[qoff + d + 1], Q[qoff + d + 2], Q[qoff + d + 3]);
                    sq += dot(sv, qm);
                }
                skv[i] = sk;   // (S0 . K_m)[i]
                sqv[i] = sq;   // (S0 . Q_m)[i]
            }
            barrier();

            if (tid < DIM) {
                uint i = tid;
                float p = 0.0;
                float qc = 0.0;
                for (uint j = 0; j < m; j++) {
                    float wj = w_sh[j];
                    float dj = delta[j][i];
                    p  += wj * Dk[j][m] * dj;
                    qc += wj * Dq[j][m] * dj;
                }
                float vm = V[(mBase + m) * (32 * DIM) + h * DIM + i];
                float dv = vm - alpha_P[m] * skv[i] - alpha_P[m] * p;
                delta[m][i] = dv;

                float ym = alpha_sh[m] * (alpha_P[m] * sqv[i] + alpha_P[m] * qc) + beta_sh[m] * dv * Dq[m][m];
                yGated[(mBase + m) * (32 * DIM) + h * DIM + i] = ym;
            }
            barrier();
        }

        // state update: S = P_16 * (S0 + sum_m w_m * delta_m * K_m^T)
        for (uint idx = tid; idx < DIM * DIM; idx += TS * TS) {
            uint i = idx / DIM;
            uint d = idx % DIM;
            float acc = 0.0;
            for (uint m = 0; m < TS; m++) {
                acc += w_sh[m] * delta[m][i] * Ks[m][d];
            }
            float s_old = S[h * (DIM * DIM) + i * DIM + d];
            S[h * (DIM * DIM) + i * DIM + d] = alpha_P[TS] * (s_old + acc);
        }
        barrier();
    }
}
```

**Chunked recurrence derivation** (matches the sequential `deltanet_ref` exactly, up to float summation order). Sequential (HF):

```
alpha_m = exp('exp(A_log)Â·softplus(a_m + dt_bias)),   beta_m = sigmoid(b_m)
delta_m = V_m ' alpha_mÂ·(S_mÂ·K_m)
y_m     = alpha_mÂ·(S_mÂ·Q_m) + beta_mÂ·delta_mÂ·(K_mÂ·Q_m)
S_{m+1} = alpha_mÂ·S_m + beta_mÂ·delta_mÂ·K_m^T
```

Unrolling the state over a chunk starting at S   (log-domain form used by the shader):

```
G_m    = Î£_{p<m} g_p            (g_p = log alpha_p, decreasing negative)
e_j    = V_j ' exp(G_{j+1})Â·(S  Â·K_j) ' Î£_{p<j} exp(G_{j+1}'G_{p+1})Â·beta_pÂ·e_pÂ·(K_pÂ·K_j)
y_j    = exp(G_{j+1})Â·(S  Â·Q_j) + Î£_{p  j} exp(G_{j+1}'G_{p+1})Â·beta_pÂ·e_pÂ·(K_pÂ·Q_j)
S_next = exp(G_T)Â·S   + Î£_m exp(G_T ' G_{m+1})Â·beta_mÂ·e_mÂ·K_m^T
```

Every `exp(Â·)` argument is    0, so the terms are bounded even though `exp(G_m)` itself underflows to zero for deeply-decayed heads (where the state has legitimately converged) — unlike the earlier `P_m = Î  alpha` / `w_m = beta/P_{m+1}` form, which divided by an underflowed product and produced NaN (gotcha 24). The two 16Ã—16 dot matrices (`Dk`, `Dq`) are classic 16Ã—16-tiled GEMMs over dim 128; the triangular scan over the 16 tokens is sequential (state dependency) but parallel across the 128 dims.

---

### 7.4 FFN (SwiGLU)

#### `RmsNorm-swiglu-ffn-*` / `-GEMM-*`

RMSNorm — two parallel GEMMs (gate, up) sharing the same A tile — `y = silu(gate) Â· up`:

```glsl
gateResult /= (1.0 + exp2(-gateResult * 1.44269504));
y[globalCol] = gateResult * upResult;
```

GEMV version: 256 threads, one output column each, ping-pong shared buffers for the scale/zero tiles (INT8/INT4). GEMM version: 16Ã—16 tiled (per-row RMS reduction over 16 token rows, dual B tiles `BgSub`/`BuSub`, direct per-tile scale/zero indexing — no ping-pong needed).

---

### 7.5 Prototypes (not used by validation)

`gemv1..7.comp` / `gemv.comp` (experimental GEMV variants), `gemm.comp` (naive scalar 16Ã—16 tiled GEMM), `online-softmax.comp` (standalone online softmax + V accumulation, still validated via `validateOnlineSoftmax`), `RmsNorm.comp` (standalone RMSNorm), `RmsNorm-GEMV.comp` (fp32, unused), `RmsNorm-GEMV-Rope-*.comp` (early rope prototypes, superseded by RmsNorm-QKV), `test.comp`.

### 7.6 Inference shaders (engine mode)

The engine mode introduced a second generation of GEMM kernels ("GEMM2") plus dedicated decode split-K passes. All GEMM2 kernels consume `st->invRms` computed once per chunk by `RmsNorm-Prologue`, and use register-blocked workgroups (each thread accumulates 2  4 output columns via `TN`-wide B tiles) with direct per-tile scale/zero indexing.

#### `RmsNorm-Prologue.comp` (prefill, all GEMM2 kernels)

One 64-thread workgroup per token row. Each lane strided-sums `dot(v,v)` over the row, `subgroupAdd` reduces the 64 lanes, `subgroupElect` writes `invRms[m] = inversesqrt(sum/K + 1e-6)`. Bindings: `0 = x`, `1 = invRms` (a **write-only** scratch buffer, not gamma). Push `{K}`. Cost at M=512: ~0.4 ms — it replaces the per-layer RMSNorm reductions that every fused GEMM2 kernel previously re-ran. **Important:** the prologue input must be the *same* buffer the GEMM2 consumes — the layer-0 prefill passes `st->embStaged` (`addLinearProj(..., st->embStaged)`), and the prologue binding is `input`, not a hardcoded `st->h` (gotcha 21).

#### `GEMM-ADD2-*` (prefill, down/out projections with residual)

`C = AÂ·B + R`. INT4 uses `TN = 64` with 4 accumulators/thread; INT8/FP16 use `TN = 32` with 2. **All three variants are ping-pong double-buffered**: k-tiles for `t+1` are loaded into the alternate `Asub[2][4][TS]` / `Bsub[2][4][TN]` LDS set while tile `t` computes, halving the barriers (one per tile instead of two). B-load for INT4: `tid < 64` threads load 64 columns Ã— one `uvec4` per k-tile (parity-selected nibble half), dequantized with `scale/zero` indexed `g*(K/4) + t*4 + kv`. The residual `R` is read once at the end, not accumulated per tile. Used by `addGemmAdd` for the FFN down projection (`k = FFN_N` when the input is `act`) and the attention `out` projection (`k = K`). When `m == 1` (decode) `addGemmAdd` routes to the split-K GEMV path instead. Ping-pong measured '2.4% (INT4), '1.4% (INT8), '0.2% (FP16).

#### `Gate-Sigmoid` (full-attention output gate)

Precision-agnostic elementwise pass: `attnOut *= sigmoid(gAttn)` per token over the 4096 attention output columns, 256 threads, push `{count}`. Inserted once per full-attention layer after the attention output (`Att-PV2` prefill, `Reduce-Att2` / `Att-full` decode) and before the `o_proj` residual GEMM.
#### `RmsNorm-QKV-GEMM2-*` + `Rope-GEMM-*` (prefill, split QKV projection)

The fused prefill QKV-GEMM (head-wide 256-wide B tiles, RoPE inside) was split into two passes sharing a raw-projection scratch buffer `st->qkvRaw` (`maxM Ã— MODEL_QKV_N` floats):

- **`RmsNorm-QKV-GEMM2-{FP16,INT8,INT4}`** — prologue-style register-blocked GEMM (`TN = 64`/4 accs for INT4, `TN = 32`/2 accs otherwise) computing the *raw* q/k/v projections into `qkvRaw`; no norms, no RoPE, no cache writes. Bindings: `x, gammaIn, w, [scale, zero], qkvRaw, invRms`; push `{M, N, K}`.
- **`Rope-GEMM-{FP16,INT8,INT4}`** — grid `(2*heads + 2*kv_heads, M)`, 256 threads; each workgroup reads one token row of one head from `qkvRaw[row*N + head*256 + col]`. Routing by workgroup id: q heads (0..15) apply per-head RMSNorm scaled by the learned `q_norm[256]`, then **partial RoPE** over the first 64 dims (`angle = (tokBase + row) * theta[col & 31]`, dims 64..255 pass through) and write to `qOut` (scaled by `1/16` = the attention `1/šhead_dim`); g heads (16..31) store the raw gate projection to `gAttn` (no norm/rope); k heads (32..35) apply RMSNorm scaled by `k_norm[256]`, RoPE, then store to the **transposed** K cache at `kCache[(globalCol ' kOffset) Â· MAXCTX + token]` (fp16 pack, or `uint8` + scale/zero at the fixed slot `kvh*MODEL_MAX_CTX + token`); v heads (36..39) store raw to the token-major V cache at `vCache[token Â· (vOffset ' kOffset) + (globalCol ' vOffset)]`. The learned norm gammas are applied to each column **before** the RoPE rotation. Push `{N, gOffset, kOffset, vOffset, tokBase}`.

This cut the QKV chain from ~24.6 s to ~8.7 s per 16k prompt (fused avg   96 ms/call — GEMM2   25  42 ms + Rope   0.3  0.8 ms). It also removed the prefill's dependence on the position buffer entirely (see Â§8.2).

#### `RmsNorm-LinearProj-GEMM2-*` (prefill, delta-net input projection)

Prologue + GEMM2 with `N = 12352`, `TN = 32`. Writes directly into the six routed output buffers (q/k/v/z/a/b) via the per-column stride layout `[m][qk*N_QK*DIM + d]`, `[m][qk]`, etc. A ping-pong probe on this family **regressed** (+1.5% INT4) — small-N grids don't have enough parallelism to hide the prefetch (gotcha 16).

#### `Embed-Gather.comp` (prefill, layer-0 embedding pre-fetch)

Replaces `Embed-RmsNorm-LinearProj-GEMM-FP16` in the engine prefill: the tied-lm-head column fetch is hoisted into its own precision-agnostic gather pass. Grid `(M, 1)`, 256 threads; each thread gathers vec4 chunks `i = col, col+256,   ` of the embedding row `tokenIds[row]`: vec4 index `i` reads `lm[(i/2)*V + tok]`, even `i` unpacks `.x,.y`, odd `i` unpacks `.z,.w`. Writes the fp32 row to **both** `st->embStaged` and `st->embOut` (the residual copy). Bindings: `tokenIds, lm, embStaged, embOut`; push `{V}`; ~0.42 ms/call. The projection then runs as plain `RmsNorm-Prologue` + `RmsNorm-LinearProj-GEMM2-FP16` over the contiguous `embStaged` (via `addLinearProj(..., st->embStaged)`), avoiding the column-major strided fetch inside the GEMM: embed path 8.5 s — ~2.1 s per 16k prompt.

#### `RmsNorm-swiglu-ffn-GEMM2-FP16` (prefill, FP16 swiglu — kept dual-matrix)

Prologue + dual B tiles (`gate` and `up`, `TN = 32`), silu applied inside the kernel: `y = silu(gateResult) * upResult`. This is the **FP16** variant; INT4/INT8 were replaced by the flat kernel below. Also ping-pong double-buffered (`Asub[2][4][TS]`, `BgSub[2][4][TN]`, `BuSub[2][4][TN]`) — 91.5 — 89.7 ms/call.

#### `RmsNorm-swiglu-flat-GEMM2-INT4/INT8` + `Swiglu-combine.comp` (prefill, flattened swiglu)

The INT4/INT8 swiglu kernels. The grid is flattened over `2Ã—FFN_N` columns: workgroup x-tile `nBase < off` computes gate columns, `nBase >= off` computes up columns (`off = FFN_N`, routed by `upHalf`). Each workgroup loads from **one** weight matrix (gate or up) with a single-matrix B-loader — no dual-matrix staging, no silu inside the inner loop. Output goes to `st->gAct` / `st->uAct` (fp32, `maxMÃ—FFN_N`). Both flat kernels are ping-pong double-buffered over k-tiles (~'1..2%; INT4 110.6 ms, INT8 107.4 ms at M=512). `Swiglu-combine.comp` (256 threads, `dispatchX = m*N/256`) then computes `act = silu(gAct) * uAct` elementwise (~0.5  0.7 ms per call; total < 1 s per 16k prompt). FP16 was measured *slower* flat (103.8 vs 91.4 ms) and keeps the dual kernel; TN=64 and TS=32 probes on the flat kernels regressed +50  70% and were reverted (gotcha 16).

#### `Att-QK2-*` / `Att-Softmax.comp` / `Att-PV2-*` (prefill, unfused attention)

Attention was split into three passes over a persistent fp16 score buffer `st->attScores` (`maxM Ã— MAX_CTX Ã— 4` float16 = 4 head-quads Ã— M Ã— ctx, 134 MB at M=512, 16 KB per 4-head band). A "head-quad" is a **GQA group of 4 query heads** sharing one kv-head (`kvh = head/4`); `buildAttention` emits the QK2—Softmax—PV2 trio **4 times** (`headBase = qb*4`, `qb = 0..3`) so all 16 query heads are computed while `attScores` stays 4-head-sized and is reused per batch (gotcha 25):

- **QK2** (`Att-QK2-*`): workgroup = (head-quad, m-tile); 16 q-tokens Ã— 64 kv-tokens; TN=64 B tiles over the head dim, K staged per head-quad lane (`kvh = head/KV_HEADS`) from the **transposed** cache (`key[(kvhÂ·256 + cÂ·16 + j)Â·MAXCTX + tok]`, coalesced across `tok`); causal limit `limit = qOff + mGlobal` is **chunk-absolute** (correct across chunked prefill; the legacy `Att-full-GEMM-*` masked with the chunk-local row and under-attended history). Writes `NEG_INF` outside the causal range into the fp16 score buffer.
- **Softmax**: 256 threads per (m-tile, head-quad) row; strided max — `subgroupMax` — cross-subgroup max via 4 shared slots, then `exp(s - gmax)` in place. The final normalize sweep was **folded into PV2**: Softmax writes `smSum[hL*param.mRows + row] = 1/total` (binding 1) and PV2's epilogue scales each accumulator (`oacc *= smSum[hL*mRows + mGlobal]`; binding 3 for FP16, binding 5 for INT8/INT4). This removed a third full pass over the score buffer: softmax pool 3.2 s — ~0.5 s.
- **PV2**: workgroup = (head-quad, m-tile, v-tile); p-tile staged as `p_sh[16][17]` (padded LDS to avoid bank conflicts), V staged per kv-token (`kvh = head/KV_HEADS`, output written to query head `head`); 16 threads each accumulate 16 output dims per token (`oacc += p_sh[row][kk] * Vsub[col][kk]`), 64 threads per tile — padded `Vsub[TS][TS+1]` staging. Quantized INT4/INT8 V dequantized during staging; epilogue multiplies by the softmax reciprocal from `smSum`.

Whole-attention cost dropped 48.7 s — ~10.4 s for a 16k prompt (QK2    8.2 s, SM    0.5 s, PV2    1.7 s).

#### Decode split-K family (M=1)

- **`GEMV-SplitK-*` + `Reduce-GEMV-ADD.comp`**: the K dimension is split across `dispatchY = 4` workgroups (256 threads each, strided k-tiles), partial sums written to `st->gemvPartial[chunk*N + col]`; `Reduce-GEMV-ADD` sums the 4 partials and adds the residual (`C[g] = Î£_z P[z*N+g] + R[g]`). Used by `addGemvSplit` for attention-out and FFN-down when `m == 1`.
- **`RmsNorm-up-ffn-SplitK-*` + `FFN-Down-SplitK-*`**: decode FFN flattened like the prefill flat kernel — one shader computes `silu(gate)Â·up` over `2Ã—FFN_N` columns (routing via `nBase >= off`, push `{M,N,K,off}`), writing `st->ffnPartial`; `FFN-Down-SplitK` then GEMVs the down matrix over the partial (which `Reduce-GEMV-ADD` reduces into `h`).
- **`RmsNorm-QKV-SplitK-*` + `Reduce-Rope-*`**: split-K QKV projection to `st->qkvPartial`, then a per-head reduce that routes by column ranges — g (raw to `gAttn`), q/k (RMSNorm scaled by learned `q_norm`/`k_norm` + RoPE), v (raw) — and stores q/k/v to cache (replaces the old fused `RmsNorm-QKV-*`). K is written to the transposed cache at `kCache[row Â· MAXCTX + pos]` (`row = globalCol ' kOffset`), V to the token-major cache at `vCache[pos Â· (vOffset ' kOffset) + row]`. Quantized scale/zero are written at the fixed slot `kvHead * MODEL_MAX_CTX + pos`.
- **`RmsNorm-LinearProj-SplitK-*` + `Reduce-LinearProj.comp`**: split-K proj to `st->linprojPartial`; reduce routes the 12352 columns into the six q/k/v/z/a/b output buffers.
- **`Att-SplitK2-*` + `Reduce-Att2.comp`** (decode attention, ctx    256): `Att-SplitK2` splits the KV sequence into up to `MAXC = ATT_MAX_CHUNKS = 1024` chunks of 4 KV-tiles (LOOPS=4, 64-token tiles — 262144-token ceiling; gotcha 49); each workgroup runs an online-softmax over its chunk and writes `{max, sum, acc[headDim]}` per (chunk, head) into `st->attPartial` (sized `ATT_MAX_CHUNKS Â· heads Â· (2 + headDim)` — runtime heads); `Reduce-Att2` re-normalizes across chunks (`w = exp(P[ml*2] - m)`) and writes all `headsÂ·headDim` outputs — its push block carries **both** `gqa` (4th member, cache routing not needed here) and the real `heads` (6th member, the `h >= HEADS` output guard — reusing the gqa slot for the guard once silently dropped heads gqa..heads-1 and collapsed all generation past ctx 256 into garbage; gotcha 34). The QK dot reads the **transposed** K cache (`key[(kv_row_base + d)Â·MAXCTX + s]`, coalesced across `s` = token), while the PÂ·V pass reads V token-major. Below 256 tokens the engine uses `Att-full-*` (single workgroup, online softmax, Â§7.2).

### 7.7 Token selection — greedy vs sampling

Token selection is a two-op chain built by `buildLmHead` (`src/generate.c`), threaded through both the decode groups and the prefill `finalOps`:

- **Stage 1** — `LMHead-GEMV-ArgMax-FP16.spv` (greedy) or `LMHead-GEMV-FP16.spv` (sampling): the same RMSNorm — `xÂ·lm_head` GEMV over `vocab` columns. The argmax variant reduces to per-workgroup `{maxValue, maxIndex}`; the sampling variant instead writes raw `logits[col]` into a new `st->logits` device buffer (`vocab Ã— float`, ~320 KB).
- **Stage 2** — `ArgMax-Reduce.spv`, with `mode = g->sampling` (Â§7.1). Both modes write `result[passIdx]` + `tokenIds[0]` and own the single per-token `position` bump, so the rest of `generateTokens` is identical.

`generatorSetSampling(g, params, seed)` (called per request from `serverMain`) sets `g->sampling = (temperature > 0)` and, whenever that flips, recompiles `groupOps`/`groupOpsShort`/`finalOps` with the matching lm-head chain. Sampling parameters:

| Param | Destination | Meaning |
|---|---|---|
| `temperature` | `sampleParams.temperature` | `<= 0` — greedy; `> 0` — softmax scaling factor |
| `rep_penalty` | `sampleParams.repPenalty` | CTRL logit scaling for ids in the recent window (`1.0` = off) |
| `penalty_len` | `sampleParams.penaltyLength` | history ring length (`0` = off) |
| `top_k` | `sampleParams.topK` | keep only the k largest logits (`0` = off) |
| `top_p` | `sampleParams.topP` | nucleus mass (`1.0` = off) |
| `min_p` | `sampleParams.minP` | drop tokens with `p < min_p` (`0.0` = off); in the prob-space sampler `p_max = 1`, so the filter is a plain threshold — one shared-max after the top-k/top-p phases, no bisection |
| `presence_penalty` | `sampleParams.presencePenalty` | flat `logit -= presence_penalty` (raw-logit frame) for any id present ≥1× in the recent window — Qwen/vLLM semantics, applied once regardless of frequency (`0.0` = off) |
| `seed` | `sampleRng[0]` | xorshift32 seed, reseeded per request |

The repetition-penalty ring `sampleHistory` is a device buffer of `MAX_PENALTY_LEN = 1024` `uint32`, sentinel-filled (`0xFFFFFFFF`) in `resetGenerator`; empty slots never match a vocab id. Both penalties share the same membership bitmap — an id in the window gets the rep CTRL scaling (if `repPenalty != 1`) and the presence subtract (if `presencePenalty != 0`); a token is penalized once even if it occurs several times in the window. The Python frontend (`pumice_llm.py`) drives all this via module-level constants — `is_sampling`, `temperature`, `rep_penalty`, `penalty_len`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `seed` — and `_stream_ids` writes them as a fixed 32-byte header (`struct.pack("<ffIIfffI",  )`) between the length prefix and the token ids; `is_sampling == False` sends `temperature = 0.0` (greedy).

**Cost.** Measured at ctx  130 with `--timing`: `LMHead-GEMV-FP16` 2.69 ms (same as the greedy `LMHead-GEMV-ArgMax-FP16` 2.81 ms — both stream the identical 671 MB lm_head matrix), `ArgMax-Reduce(mode 1)` 0.64 ms vs `(mode 0)` 0.03 ms — the sampling overhead is ~0.6 ms/token, under 1 tok/s end-to-end (24.4 vs 24.6 tok/s greedy). The original sampler rescanned the logits with scalar loads + `exp` per element across 48+48 bisection iterations and cost ~15 ms/token (sampling at ~20 tok/s vs 30+ greedy); see Â§7.1 for the optimized algorithm.

---

### 7.8 MoE — router + expert pools + combine (Qwen3.6-35B-A3B)

Every layer's FFN is a 256-expert top-8 MoE with one shared expert, executed as **four ops**
(`buildMoe` in generate.c) that replace the dense FFN chain; the same op sequence serves decode
(m=1) and prefill (m=padded chunk) because the expert shaders key the slot id off
`globalSlot = wgID.y` (dispatchY = m*(topk+1)):

1. **`Router-TopK.spv`** — one workgroup per token (256 threads). Stages the raw residual `h`,
   computes the RMSNorm **and writes the normed `xn` to a buffer** (the expert GEMVs read it —
   norming once per token instead of once per expert), then each thread < E computes one router
   logit (FP16 router weight, engine blocked layout), a cooperative dot reduces the shared-expert
   gate logit, and thread 0 does softmax -> top-8 (lowest-index tie-break) -> **renormalized
   weights** (`w = p_topk / sum(p_topk)` — the full softmax denominator cancels), writing
   `ids[m*slots+s]`, `weights[m*slots+s]`, and `sharedW[m] = sigmoid(sharedGateLogit)`.
   Push `{M, E, K, slots}`; E <= 256, K <= 2048 (shared-array sized).
2. **`Expert-Swiglu-INT4.spv`** — one workgroup per (slot, 256-col tile): `dispatchX = moeI/256`,
   `dispatchY = m*slots`. Looks up `ids[globalSlot]`, remaps it: `id == experts` (the shared
   expert) -> VRAM slot `vramExperts-1`; `id >= vramExperts-1` -> RAM pool index
   `id-(vramExperts-1)`; else VRAM `id`. Computes `silu(xn @ gate_e) * (xn @ up_e)` from the
   INT4 pool (chunked gate|up layout: gate cols `j`, up cols `j+moeI`), writing
   `h[globalSlot*moeI + j]`. Push `{K, moeI, slots, vramExperts, experts}`.
3. **`Expert-Down-INT4.spv`** — same shape, `h -> partials[m*slots*K + j]` through the down pool.
4. **`Moe-Combine.spv`** — elementwise: `y = h + sharedW*P[sharedSlot] + sum_{s<8} w_s*P[s]`
   (residual add fused; the last slot index `slots-1` is the shared expert).

Buffer layout notes: the pools bind as **six** buffers (VRAM data/scale/zero + RAM data/scale/zero);
the id remap lives entirely in the shaders, so swapping an expert between pools requires no host
bookkeeping beyond the pool contents. `moeIds` is initialized with `experts` in every token's slot
`slots-1` at state creation (the router only writes slots 0..7 — gotcha 47).

## 8. Inference Engine (src/generate.c)

The engine compiles the model into `operation` arrays at startup — all dimensions from `spec->dims` (runtime config) — and executes them with the split record/submit/wait API from Â§3.3 (`executeRecord` — `executeSubmitNow` — `executeWaitLast` — `logLastFrame`), which keeps per-op GPU timestamps and avoids the per-op fence stall of `execute()`.

### 8.1 Op compilation

`addOp` appends a fully-formed `operation` (shader name, buffers, push constants, dispatch). The layer builders (`buildFfn`, `buildMoe`, `buildAttention`, `buildDelta`, `buildLinearProj`, `buildLmHead`) emit the op sequences from Â§7.6 — `buildDelta` inserts a `Conv-SiLU.spv` op between the input projection and `GatedDeltaNet*` for every delta layer. `createGenerator` sets the shader root + spec constant (`pipelineSetSpecInt(0, dims.K)`), loads weights (Â§4.7), allocates state, and pre-compiles four arrays:

- **`groupOps` / `groupOpsShort`** — decode group: `DECODE_GROUP = 4` tokens Ã— (Embed-LinearProj — all layers — lm-head). The two variants differ only in attention: split-K (`Att-SplitK2`+`Reduce-Att2`) vs `Att-full`, selected by `ATT_SPLIT_THRESHOLD = 256` on the current context length.
- **`prefillOps`** — one chunk of `maxM` tokens (`prefillChunk`): Embed-Gather — RmsNorm-Prologue — LinearProj-GEMM2-FP16 (over `st->embStaged`) — all layers — (no lm head).
- **`finalOps`** — lm head over `st->lastRow` (the last prefill row copied back to host and re-uploaded), `doIncrement = 0`.

Weight/state buffers are per-layer **heap arrays** (`layerBufs`/`tensorBufs`), sized `layerCount` at runtime — no fixed 32-layer cap.

### 8.2 Chunked prefill (`runPrefill`, `compilePrefill`, `executeChunked`)

The prompt is processed in chunks of `prefillChunk` (512) tokens:

1. Copy the chunk's token ids into `st->tokenIds` (mapped RAM). `cur` is **padded to a multiple of 16** (`(cur+15)&~15`) and the pad slots filled with token 0, so every GEMM/GEMM2/attention grid gets a full `M/16` m-tile — the shaders divide by `mRows/16` and would divide-by-zero / dispatch nothing for a shard of size < 16 (gotcha 23). The true row count (`cur`) is threaded through `buildLayer`/`buildDelta` as **`realM`** and pushed to `Conv-SiLU` (clamps the conv/history loop) and `GatedDeltaNet-GEMM` (zeroes g/Î² of pad rows) — so the S-state and conv history carried into decode reflect only real tokens (gotcha 38).
2. `compilePrefill(g, padded, offset, realM)` re-emits the whole prefill op list for the padded row count (layers get `m = padded`, so dispatchY shrinks for the tail chunk). Chunk offsets are threaded **at build time** — `buildLayer` / `buildAttention` take the chunk's absolute token base and emit it directly into push constants (`Rope-GEMM tokBase`, attention `{ctxLen = offset + m, qOff = offset}`); there is no post-compile patch loop. Prefill shaders never touch the position buffer; after the last chunk's ops are recorded, `runPrefill` sets it host-side via `stateSetPosition(g->s, &g->st, nextPos + nTokens)` before `finalOps` executes.
3. `executeChunked` submits the op list in slices of    8 ops, waiting between slices — this keeps GPU work below the Windows TDR threshold (`TdrDelay = 8 s`; an over-long slice yields `VkResult -4` = `VK_ERROR_DEVICE_LOST`).
4. After the last chunk, the last processed row is copied from `st->h` to `st->lastRow` (host round-trip, only once per prefill), and `finalOps` selects the first generated token.

### 8.3 Decode loop (`generateTokens`) and the server

- `generateTokens(g, prompt, nPrompt, maxNewTokens, emit, ctx)` = `runPrefill` + the decode loop, streaming each generated token to the caller through the `emit(token, ctx)` callback and **stopping at EOS** (`token == g->eos`). `maxNewTokens` is clamped so `nPrompt + maxNewTokens <= g->maxCtx`.
- The op lists are **double-buffered**: while group `u` executes, group `u+1` is recorded (the current token is written into `tokenIds[0]` after each group's result readback). Two separate op arrays (`groupOps` vs a shadow copy compiled with `passIdx`-tagged lm-head write slots) are alternated each iteration.
- **Token pacing.** The GPU produces tokens in groups of `DECODE_GROUP = 4`, but `generateTokens` emits them **one at a time** through `emit`, sleeping between consecutive tokens of a group. The inter-token delay is the measured per-token time of the *previous* group (`(t1't0)/cur`, QPC-timed around the group's wait + readback); the very first group uses a constant `FIRST_TOKEN_DELAY_MS = 25` ms. This makes the server's output stream look like per-token generation rather than 4-token bursts.
- `resetGenerator(g)` re-zeroes the gated-delta `stateS` recurrence buffers **and** the `convHist` shift registers (`createTransferAndCopy` re-copies their still-zero staging buffers) and resets `nextPos = 0`, so each server request is an independent full prompt.
- `serverMain` (compute.c) sets binary stdio (`_setmode`), parses `--weights/--prune/--max-ctx/--max-new/--dump/--dump-layers/--verbose-weights/--timing/--debug-sampling`, loads the model config, then — if `--prune` is given — runs `pruneVocab` (cache-first: weight cache → vocab folder → auto-gather from the shards via `pruned-vocab/mapping.npy`, §4.7) followed by `parseEos`, creates the generator once, then loops per request: read `n` prompt-id count — read a fixed 32-byte sampling header (`temperature, repPenalty, penaltyLength, topK, topP, minP, presencePenalty, seed`, `<ffIIfffI`) — read the `n` ids — `generatorSetSampling(g, &sp, seed)` — `resetGenerator` — `generateTokens(..., emitToken, ...)`. `emitToken` writes each `uint32` id + `fflush(stdout)`; when generation ends it writes a `0xFFFFFFFF` terminator. The loop exits on `n == 0`.
- The group's `ArgMax-Reduce` increments the shared position buffer once per token (`doIncrement = 1`); prefill never touches it on the GPU (`runPrefill` sets it host-side after the last chunk), and the lm-head final pass runs with `doIncrement = 0` — together these keep the position counter correct across the prefill—decode transition (see gotcha 8).

### 8.4 Timing log

The dispatch layer retains per-op timestamp logging (`logFrame`/`logLastFrame` and the `timing_agg`/`timing_log.txt` machinery in `dispatch.c`), gated by `setTimingEnabled`. It is wired to the **`--timing` CLI flag**: `serverMain` calls `setTimingEnabled(1)` at startup and `closeTimingLog()` at shutdown, which writes `timing_log.txt` (into the server's cwd, `bin/`). The decode loop additionally calls `logLastFrame(..., "decode", ...)` after each group's `executeWaitLast`, so per-op decode timings are logged too, and prints a per-group `[decode][tok=  ] group=4  X.XXX ms/tok` line to stderr whenever timing is enabled (`isTimingEnabled()`). Without `--timing` the decode loop runs with no per-op logging overhead.

A separate `--debug-sampling` flag (`generatorDumpSamplingDebug`) reads back the `sampleHistory` ring, `stateReadPosition`, and the emitted token list after each request and prints them to stderr, for diagnosing the repetition-penalty / sampler state (Â§7.7).

---

## 9. Key Gotchas Discovered

1. **k-tile loop bound is `K / TS`, not `K/(TSÂ·vec)`.** Each iteration advances k by 16 floats (4 vec4s), so the loop runs K/16 times.
2. **GatedDeltaNet `alpha`/`beta` are per-token** (`aRaw[m*16+qk]`, not `aRaw[qk]`). The recurrence is non-stationary; the chunked form needs prefix products `P_m` and weights `w_m = beta_m/P_{m+1}`.
3. **Flash-attention online rescale:** per-element probability must be `exp(score ' tile_max)` (not `' m_next`) or `beta` gets applied twice; the sum reduction clobbers the shared array, so per-element probabilities live in a separate `s_p` buffer.
4. **GatedDeltaNet stability.** The old simplified rule (`alpha = sigmoid('a)`) was numerically unstable with the harness's random weights (state overflowed within ~16 tokens, so `validateGatedDeltaNetGEMM*` scaled the projection weights by 1/64). The HF recurrence (`alpha = exp('exp(A_log)Â·softplus(a+dt_bias))` < 1) is a geometric decay, so the state stays bounded (post-fix state magnitudes ~2 vs ~686000 before); the 1/64 scale remains in the GEMM fixtures but is no longer required for stability.
5. **Windows make quirks:** recipes run through `cmd.exe` — `mkdir`/`if exist` paths must use backslashes (`/` is treated as a switch prefix), and shader folder names must not contain spaces (make word-splits `$(wildcard)` output).
6. **KV cache layout is split: K is `[row][token]`, V is `[token][row]`.** K is stored transposed (`kCache[rowÂ·32768 + token]`, `row = kvhÂ·256 + dim`) so the QK dot reads are coalesced across the token axis, while V stays token-major (`vCache[tokenÂ·1024 + row]`). All writers (`Rope-GEMM`, `Reduce-Rope`) and readers (`Att-QK2`, `Att-full`, `Att-SplitK2`) use the matching index; the validation harness transposes K into `[row][32768]` for the `Att-full`/`Att-SplitK2`/`Reduce-Rope` checkers.
7. **Workgroup-wide barriers need every thread present.** For GEMV shaders with a partial last workgroup (N not divisible by 256), out-of-range threads must not early-return — they still run the loop/barriers and carry a `-inf` (`uintBitsToFloat(0xFF800000u)`) accumulator so they can never win the argmax reduction.
8. **Position buffer write rules (multi-layer safety).** The shared `uint[1]` position buffer must be updated exactly once per phase transition: decode's per-token `+1` lives **only** in `ArgMax-Reduce` (single dispatch, gated by `doIncrement`), and prefill sets it host-side (`stateSetPosition`) after the last chunk — no prefill shader writes it at all now. Historically the fused QKV GEMM wrote it absolutely (`tokenIdx + M`, idempotent across layers); a read-modify-write `+= M` in any per-layer shader would inflate the counter NÃ— per prefill chunk, and an unconditional `+1` in the lm-head step would corrupt the prefill—first-decode transition.
9. **`float atomicAdd` on SSBO is a no-op** on this driver stack. Split-K float accumulation must use `atomicCompSwap` CAS (or the two-pass partial+reduce pattern used here).
10. **The `RmsNorm-Prologue` output binding is `invRms`, not gamma.** A bug bound `{h, gammaF}`/`{h, gammaIn}` and wrote invRms into the weight gamma while `st->invRms` stayed 0 — every GEMM2 projection computed zeros end-to-end (silently, because the demo collapses to token 0 and the validators use their own wiring). Timings were unaffected (same FLOPs). Fix: `proBufs[] = {st->h, st->invRms}`.
11. **Causal masks must use chunk-absolute indices in chunked prefill.** `Att-full-GEMM-*` masked with the chunk-local row (`kvTok <= row`), under-attending history; the QK2 kernels use `limit = qOff + mGlobal`.
12. **Build from the repo root; headers aren't dep-tracked.** Running `make` from `bin/` silently no-ops (no Makefile there). The Makefile compiles `.c`—`.o` with no `.d` dependency generation, so **any** header edit leaves stale `.o` files compiled against the old layout. This bit twice: after `model.h` dim edits, and much worse when a field was added to the `buffer` struct (`buffer.h`) — the not-recompiled `generate.c`/`dispatch.c` kept the old sizeof, corrupted the ABI, and produced a deterministic `0xC0000005` crash mid-weight-load that looked like the transient gotcha 14. **Rule: after editing any header, `make clean && make`.** If shader files were restored with `Copy-Item`, delete the stale `bin/shader/<name>.spv` — timestamps are preserved and the rebuild is skipped.
13. **Chunked execution avoids TDR.** Submitting one giant prefill command buffer (16k tokens) exceeds the Windows 8 s TDR budget (`VK_ERROR_DEVICE_LOST = -4`). Slice submissions to    8 ops with a wait between slices.
14. **Long-running jobs can transiently crash in `0xC0000005` during weight load** (weight file ~2 GB); retrying the run succeeds — not a code bug, worth re-running before debugging. Validation runs can also transiently produce NaN/garbage in one shader; two consecutive green vals = real pass.
15. **Quantized KV-cache scale/zero need a context-independent stride.** Writers originally used `kvh*(pos+1)+pos` / `kvh*(tokenIdx+M)+absTok` — a stride that grows with the current context — while readers assumed whatever the *current* chunk's stride was, so every token written by an earlier chunk had its scale read from the wrong slot once the cache grew (silently, and masked by the zero-token demo signature). Fix: fixed stride `kvh * MODEL_MAX_CTX + token` in all writers (`Rope-GEMM`, `Reduce-Rope`) and readers (`Att-QK2/PV2/SplitK2/full` INT8+INT4), with validator fixtures updated to match.
16. **GCN4 tile geometry for the GEMM2 family: smaller workgroups win.** TN=64 or TS=32 (1024-thread) variants regress +50  70% despite halving barrier count — occupancy/latency-hiding dominates. Ping-pong k-tile prefetch (one barrier per tile, prefetch into the alternate LDS set) gives a reliable but small '1..2% **only when grid.x is wide** (FFN/ADD2 shapes); on narrow-N kernels (LinearProj N=12352, QKV) it regressed or washed out.
17. **Appended shader bindings must match validator buffer order exactly.** When the gated-attention change added `q_norm`/`k_norm`/`gAttn` bindings, the three *legacy prefill-fused* `RmsNorm-QKV-GEMM-*` shaders declared them as `gAttn, qGamma, kGamma` while the validators supplied `qGamma, kGamma, gOut` — q got the wrong gamma and g/k wrote nowhere (q err 6.4, g/k zero). The decode-fused and `Rope-GEMM`/`Reduce-Rope` shaders matched, which is why only `validateQkvRopeGEMM*` failed. Also: the learned per-head norm gamma must be applied to each column **before** the RoPE rotation mixes the two halves (`acc*inv_rms*gamma` then rope), not to the rotated result — applying it after yields a real numeric mismatch against the reference, not just noise.
18. **VRAM is ~7936 MB on the 8 GB RX 580, not 8192, and it's fragmented.** `main.exe meminfo` dumps this: `heap[0]` device-local is 7936 MB (WDDM reserves ~256 MB) and host-visible/staging memory lives in a separate 16/32 GB system heap — staging is *not* the VRAM problem. The 9 B model + state needs ~7480 MB of device-local memory, and because ~450 discrete `vkAllocateMemory` calls fragment the heap, the allocation that tips it over is reproducible: `out of GPU memory: failed to allocate 'attScores' (128.00 MB) | device_local=7352.66 MB ...`. `createBufferNamed(..., name)` + per-pool byte counters in `buffer.c` name the buffer; the failure is now reported as an error instead of aborting (see §12). Freeing device memory = lower `MODEL_MAX_CTX` / smaller `MODEL_PREFILL_CHUNK`, INT8 embed/lm-head, or consolidating the ~450 tiny allocations into arenas.
19. **safetensors `data_offsets` are relative to the data section, not the file start.** `safetensors_load_f32` (and `loadEmbedLike`) sought to `t->offset` directly from file start, missing the `8 + header_length` prefix — so every tensor read from the multi-tensor HF shards returned garbage (e.g. `input_layernorm.weight` looked like `4e30`), while the single-tensor pruned embed lured by "looking plausible" at offset 0. Fix: `safetensors_open` adds `8 + hlen` to each tensor offset.
20. **`createBufferNamed(MEMORY_VRAM)` only *stages*; it never copies to VRAM.** The actual staging—device copy lives in a separate `createTransferAndCopy(device, queue, bufs, n)`. `createWeights` never called it, so every weight stayed zero on the device — the whole residual stream was zero and the lm-head argmax collapsed to token 0 (the "all token 0" signature). Fix: register all weight buffers into `g_wbufs` and call `createTransferAndCopy` once at the end of `createWeights`.
21. **`RmsNorm-Prologue` must RMS-normalize the exact buffer its GEMM2 consumes.** For prefill layer 0 the GEMM2 input is `st->embStaged`, but the prologue was hardcoded to `st->h` (still zero at that point), so `invRms` blew the embedding up by ~1/šeps. Fix: `proBufs[0] = input` (not `st->h`).
22. **`Qwen3_5RMSNorm` stores `weight` zeros-init and applies `scale = 1 + weight`.** The checkpoint vectors (`input_layernorm`, `post_attention_layernorm`, final `norm`, `q_norm`/`k_norm`) are the near-zero deltas, and the effective norm scale is `1 + weight`. `Qwen3_5RMSNormGated` (the delta `norm.weight`) is ones-init and applied directly — **no** `+1`. Got this wrong and the residual stream was ~7Ã— too small.
23. **Prefill GEMM/attention grids must `ceil(M/16)`, and the prompt must be padded to a multiple of 16.** `dispatchY = M/16` is `0` for M < 16 (no workgroups, silent all-zero), and the QK2/Softmax/PV2 shaders divide by `mRows/16` internally (divide-by-zero for M < 16). Fix: `(M+15)/16` in `buildAttention`/`addLinearProj`/`buildFfn`/`addGemmAdd`, and `runPrefill` pads the token chunk to a 16-multiple.
24. **GatedDeltaNet-GEMM prefix products underflow at real decay rates.** `alpha_P = Î  alpha` (with `alpha = exp('exp(A_log)Â·softplus(a+dt_bias))` as small as ~1e-9) underflows to 0, and `w_m = beta/P_{m+1}` divides by it — NaN. Rewrote the chunked recurrence in log-domain (cumulative `G`, relative `exp(G_i ' G_j)`), where every exponent    0 and deeply-decayed heads decay-to-zero correctly. (The sequential decode shader never had this: it multiplies the state by `alpha` inline.)
25. **Prefill full attention must compute all 16 query heads over the 4 kv-heads (GQA).** `Att-QK2`/`Att-PV2` used `head = kv_head` (0..3) for the query, dropping query heads 4..15. Fix: `kvh = head/KV_HEADS` for the K/V cache access (query and output remain per-16-heads), and `buildAttention` emits the QK2/Softmax/PV2 trio 4 times with `headBase = qb*4`. Keeping `attScores` at 4 heads (reused per batch) avoids a 4Ã— VRAM blow-up.
26. **`executeChunked` slices need a cross-slice memory barrier.** Slices of    8 ops are submitted as separate command buffers; without a `SHADER_WRITE—SHADER_READ` barrier spanning the slice boundary, later slices read stale (zero) buffers. Fix: emit the barrier unconditionally before every dispatch in `recordFrame` (not only between ops within one frame).
27. **The full-attention output gate is `sigmoid(gate)`, not `silu(gate)`.** `Gate-Sigmoid.comp` computed `o *= x / (1 + exp(-x))` = `xÂ·sigmoid(x)` (silu), but Qwen3.5 applies `attn_output * sigmoid(gate)` (no extra `gate` factor). With gate    '3.5 this produced a ~'3.5Ã— anti-correlated attention output — the single biggest correctness bug (every full-attention layer). Fix: `o *= 1 / (1 + exp(-x))`; the `validateGateSigmoid` CPU reference had the same wrong formula and was corrected too.
28. **`RmsNorm-swiglu-flat-GEMM2-INT8` unpacked INT8 wrong.** Its `unpackUint8x4` was `vec4(uvec4(v) & 0xFF)` — replicating the low byte 4Ã— instead of `(v >> (0,8,16,24)) & 0xFF`. Every INT8/INT4 FFN gate/up projection was garbage. The dual `RmsNorm-swiglu-ffn-GEMM2` (FP16 path) had the correct unpack, which is why only the flat kernels failed; the harness only validated the dual kernel, so the flat kernels were never covered. Added `validateRmsNormSwigluFlatGEMM2INT8/INT4` to the harness.
29. **`Att-QK2-FP16` read K at half rate, and its dispatch was `ctx/64`.** The FP16 QK2 K-staging used `base = ... + c*8` (INT8/INT4 use `c*16`), misaligning every QÂ·K dim pair and dropping the upper 128 K-dims; and `dispatchX = ctx/64` was `0` for ctx < 64 (no workgroups, all-zero scores — uniform attention). Fix: `c*16` and `(ctx+63)/64`.
30. **`Att-PV2-{FP16,INT8,INT4}` staged only 4 of 16 V dim-groups.** The V staging wrote `Vsub[kv][tok]` for `kv = tid/16` (0..3) but the accumulate reads `Vsub[col][kk]` for `col` 0..15, leaving cols 4..15 as uninitialized shared memory (huge/NaN outputs, e.g. 1e29). Fix: stage all 16 dim-groups with `for (i = tid; i < TS*TS; i += TS*TS)` mapping `cd = i/TS` — V dims `nBase + cd*4`.
31. **`Att-full-{INT8,INT4}` dequantized V with K's scale/zero.** The decode attention (`Att-full`) loaded `scale_s`/`zero_s` from `kscale`/`kzero` and used them for **both** the K dot-product and the V accumulation — but V has its own `vscale`/`vzero`. Every INT8/INT4 full-attention decode step dequantized V with the wrong parameters, producing garbage attention that collapsed generation after the first (prefill-driven) token. Fix: load separate `vscale_s`/`vzero_s` from `vscale`/`vzero` for the V accumulation. (FP16 `Att-full` reads fp16 directly and was unaffected.) This was the dominant **decode-path** bug: prefill was already correct (gotchas 27  30), but decode collapsed because the INT8/INT4 attention layers are where generation diverged.
32. **The chat template must use the `<think>` control token, not literal ` thinking` text.** `apply_chat_template` emitted ` thinking\n` (plain word, token 6017), but Qwen3.5 thinking mode is `<|im_start|>assistant\n<think>\n` (control token `<think>` = 248068, pruned id 81918). The literal text never put the model into thinking mode — one word + EOS. Fix: `<think>\n` (thinking) / `<think>\n\n</think>\n\n` (non-thinking). The pruned vocab already contains all 26 control tokens (ids 81894  81919).
33. **`buildQkvMatrix` must de-interleave q_proj with config dims, not tensor-shape guesses.** An early multi-model refactor derived `hd = qRows/2; qPart = qRows/2` from the tensor shape; the correct mapping is `hd = headDim; qPart = headsÂ·headDim` with the head-major q  g interleave `srcRow = headÂ·2Â·headDim + dim` (q) / `+ headDim + dim` (g). The shape-derived version scrambled every full-attention layer's Q/G/K/V — and the bug was **masked by the on-disk weight cache** (built by the pre-refactor loader) until a `make clean` wiped it. Lesson: a loader refactor is only validated by a **cold-cache** run.
34. **The decode attention push block must carry `gqa` and the real `heads` as separate members.** The GQA generalization replaced `push0[3]` (was `heads`) with `heads/kvHeads` so `Att-full`/`Att-SplitK2` could compute `kvh = head/param.gqa` — but `Reduce-Att2` (which only runs at ctx    256) still used that slot as the head count in `h >= HEADS; return;`, so heads gqa..heads-1 never got their output written (stale `attnOut`). Symptom was model-dependent collapse **exactly at absolute position ~260** regardless of prompt length — greedy fell into `!!!` loops, sampling emitted multilingual junk. The 9B only "passed" earlier because short test prompts never crossed 256 tokens. Fix: 6-member push `{maxCtx, kvHeads, kvRows, gqa, headDim, heads}`. Related lesson: when replacing a push member's *meaning*, audit **every** shader that consumes the block, including ones whose dispatch path you haven't recently exercised.
35. **EOS must be derived from the vocab files, never hardcoded.** A hardcoded `eos: 81896` (stale from an old header) made generation never stop and the model re-imitate the chat template as plain text. The pruned vocab renumbers ids, so any hardcoded id drifts silently. Fix: `loadModelConfig` reads `eos_token` from `vocab/tokenizer_config.json` and resolves the id from `vocab/vocab.json` (9B pruned: 85992). Note also a use-after-free trap when implementing this: the token name string points into the JSON tree — copy it to a local buffer before freeing the tree.
36. **npy headers have a 2-byte version field between the magic and the header length.** The pruner's `mapping.npy` is `\x93NUMPY` + `\x01\x00` (version) + 2-byte little-endian header length + JSON. Reading the version bytes as the length yields a 1-byte "header" and fails the `'<'i4'` descr check. Skip 6+2 bytes before the length.
37. **The disk weight cache is keyed on dims, not source content.** After re-gathering a vocab, re-quantizing, or switching quant configs, a same-shaped stale cache is silently reused — every generation after a vocab re-prune is gibberish until `bin/weights/<model>/` is deleted (an earlier 9B vocab re-prune hit exactly this). The cache directory is per-model (`weights/<name>/`) so two models never collide, but within one model the operator must invalidate manually.
38. **Prefill pad tokens poison the delta-net state carried into decode.** The 16-padding rows contain token-0 embeddings whose projections are nonzero; without masking, (a) `Conv-SiLU`'s shift register advanced over the pad rows, and (b) `GatedDeltaNet-GEMM`'s recurrence decayed and updated the S-state over them — decode then exploded from the first step (layer-0 output magnitude 0.97 vs the reference 0.06). Fix: thread the real token count (`realM`) through `compilePrefill`—`buildLayer`—`buildDelta`; `Conv-SiLU` clamps its loop with `steps = min(M, realM)`, and `GatedDeltaNet-GEMM` zeroes `g_sh`/`beta_sh` for rows    realM (identity update: state unchanged, no output contribution). Full-attention prefill needed no equivalent fix (padded rows only write KV slots beyond `ctxLen`, which the causal mask never reads).
39. **Subgroup-portable reductions: size shared arrays for the worst case and combine with `gl_NumSubgroups`/`active_subgroups`.** The original shaders sized cross-subgroup scratch as `TS/64` (wave64) and hardcoded 4-element combines — fine on the RX 580, garbage or races on wave32 GPUs (RTX 4060: 8 subgroups per 256 threads — out-of-bounds shared writes). Fix: arrays sized `(TS+31)/32` with loops bounded by `gl_NumSubgroups` or `(TS + gl_SubgroupSize - 1)/gl_SubgroupSize`. Also: 64-thread `RmsNorm-Prologue` relied on the whole workgroup being one subgroup (wave64) — `subgroupElect`-written `invRms[m]` raced on wave32; rewritten with a shared-memory tree reduction. And an expression-form trap: `head * KV_HEADS / param.heads` (mult-then-div on push constants) triggered nondeterministic miscompiles on the AMD driver stack, while the mathematically identical `head / (param.heads / KV_HEADS)` — and later a precomputed `gqa` push member — is stable. Prefer pushing the precomputed divisor from the host.
40. **Config JSON values must be read before `json_free`, and vocab-file candidates must match V exactly.** Two bugs from the original-vocab (no `--prune`) mode: (a) `vocab_size` was read from the `text_config` subtree *after* `json_free(hf)` — the dangling pointer returned garbage, the `json_get_int` lookup missed, and the engine silently fell back to 86016 while the Python side tokenized with original ids → every prompt token indexed out-of-range embed rows → pure `?` output; (b) `findVocabFile` globbed `embed_tokens.*.safetensors`, so a 248320 run picked up the 86016-row pruned file as a candidate and `loadEmbedLike` fataled on the shape mismatch instead of falling through to the full-size shard tensor. Fix: copy `vocab_size` into a local before freeing the tree (the same use-after-free trap gotcha 35 documents for the EOS name), and match vocab files by exact `<V>` in the filename (`embed_tokens.<V>.safetensors`) so wrong-size files are never candidates. Also note the original `vocab.json` contains no special tokens at all — `<|im_end|>` lives only in `tokenizer.json`'s `added_tokens` (id 248046), which is why `parseEos` has the `added_tokens` fallback for non-pruned mode.
41. **A completeness probe must check the filenames the writer actually produces.** `cacheComplete` probed `embed_FP16.bin` — but `loadEmbedLike` has written the V-suffixed `embed_<V>_FP16.bin` since the dual-vocab change. The probe never found the embed cache, so `cacheComplete` was always false, and moving the safetensors away (a deliberate cache-only test) fataled with "no safetensors found and weight cache is incomplete" despite a perfect 255-file cache. Fix: probe the V-suffixed names (`embed_<V>_FP16.bin`, `lmHead_<V>_FP16.bin` when untied) with header validation — which also makes the probe vocab-mode-aware for free. The same session reworked the shard open to be genuinely lazy (`shardSource()` opens the shards only on the first cache miss; a fully-cached run never opens the model dir and reports `weights: resolved fully from cache`), and fixed `loadConv`, which checked shards *before* its own cache and rewrote `conv_<L>.bin` on every run.

42. **Shard globs and file caps must match real HF names.** The old glob
    `model.safetensors*.safetensors` matched nothing for Qwen3.6-35B-A3B, whose shards are
    `model-00001-of-00026.safetensors` (and 26 > the 16-file caps in `safetensors.files[]`,
    `findShards`, and prune's `findShardPaths`). Fix: glob `model*.safetensors` and raise the caps
    to 32 (`SA_MAX_FILES`).
43. **HF 3-D expert tensors are `[out][in]` per expert — transpose before quantizing.** The first
    pool build quantized the raw `[cols][rows]` rows as if they were engine `[rows][cols]`
    (the per-expert `fseek` read is fine; the orientation wasn't). Symptom: engine output was
    fluent-looking garbage. The shared expert's gate/up are separate HF tensors and must be fused
    gate-then-up to match the routed `gate_up_proj` chunk order.
44. **`float*` pointer arithmetic with a byte stride.** `expertPoolSplit` computed
    `vramScale + scaleStride*n` — the float pointer multiplied the byte stride by 4 and ran off the
    allocation (0xC0000005). All byte-offset arithmetic on `float*` must cast through
    `uint8_t*` first.
45. **Never let a refactored loader drop its metadata writes.** The `createTensor` ->
    `loadTensorInto` refactor (return-by-value -> out-parameter, needed so the weight-flush can
    hold stable pointers) silently lost the `t->q/rows/cols` assignments; `wt->rows` then read 0
    into push constants. Regression appeared on **all** models (the 2B answered whitespace) even
    though the values "should have been identical" — found by dumping `wt->rows` at op-compile
    time. Corollary: `addGemvSplit`/`addGemmAdd` must take the reduction dim from `wt->rows`, not
    `d->K` — the 35B's delta `out_proj` reduces over `nV*dim = 4096 != K = 2048`, and the
    hardcode produced a correct first (prefill) token followed by instant decode collapse
    ("2!!!" repetition).
46. **Vulkan host-visible memory is a hard ~16 GB heap and staging counts against it.** 40 layers
    of RAM expert pools (~13.5 GB at experts_vram=32) plus every weight buffer's *retained*
    staging allocation (~4 GB) exceeded `heap[1]` (16091 MB per `meminfo`). Fix: batch the weight
    upload (12 buffers) and `releaseStaging` after each batch. Also note the host heap backs RAM
    pools AND system commit charge — at experts_vram=32 the run needs ~16 GB free system RAM.
47. **State buffers created with `createBufferNamed(VRAM)` are not uploaded until a
    `createTransferAndCopy`.** The MoE state buffers (`moeIds`...) were created but never included
    in the state transfer list, so the GPU-side `moeIds` stayed uninitialized — the shared-expert
    slot ran expert 0 instead (ids 0 in slot 8). Symptom: first token correct ("2"), then instant
    drift. Fix: transfer all state buffers once and release their staging. Related trap: writing
    the init through `buffer.mappedMemory` crashes — VRAM buffers are not mapped; pass the init
    array at creation.
48. **A struct-returning loader must fill every metadata field, and header edits require a clean
    rebuild.** Two distinct silent-corruption traps from this integration: (a) the dropped
    `t->rows` writes (gotcha 45); (b) `model_weights` gained fields (`guPool`, `router`, ...) but
    `weights.o` was compiled against the old header — the struct-copy into `generator.w` then read
    garbage `vocab` (a 16-exabyte `maxValue` allocation). The Makefile has no header dependency
    tracking, so **any** header change needs `make clean` (gotcha 12, now enforced by habit).
49. **The context length was pinned by a shrink-only `--max-ctx` and a silent 32768 decode-attention
    cap.** Two independent limiters: (a) `loadModelConfig` only applied `maxCtxOverride` when it was
    *smaller* than quant_config's `max_ctx`, so a larger `--max-ctx` was silently ignored — Python's
    `max_ctx` argument merely rode on top of the config ceiling; (b) decode split-K attention
    (`Att-SplitK2-*` TS=64 × LOOPS=4 × **MAXC=128** = 32768 tokens) hard-dropped every token beyond
    32768 from attention — no crash, just quality corruption past the cap — with `MAXC` copied into
    four shaders (`Att-SplitK2-{FP16,INT8,INT4}`, `Reduce-Att2`), the hardcoded
    `dispatchY = 128` in `buildAttention`, the `attPartial` state sizing, and the validation
    fixtures, all of which must stay in sync (they now share `ATT_MAX_CHUNKS 1024` /
    `ATT_CHUNK_TOKENS 256` from model.h — 1024 × 256 = 262144). Excess chunks are harmless: the
    SplitK2 shaders write NEG_INF partials for `start_tile >= num_tiles` and `Reduce-Att2` weights
    them to zero. Related fixes in the same pass: `max_position_embeddings` is parsed from
    config.json (before `json_free` — gotcha 35's use-after-free trap) and clamps `maxCtx`;
    `generateTokens` rejects `nPrompt >= maxCtx` (previously an out-of-bounds KV write) and
    `pumice_llm.py` raises before sending; state creation now releases the KV-cache / `attScores`
    staging allocations after the initial zero-copy (they are never re-uploaded) — **but**
    `stateS`/`convHist`/`sampleHistory` staging must stay alive: `resetGenerator` re-copies those
    still-intact staging buffers to re-zero state every request. Verified on the 2B at
    `max_ctx=131072` with a 41698-token prompt whose needle sat past position 32768 (retrieved
    correctly; prefill+decode 212 s), with short-prompt runs and `main.exe val` green.
50. **`resetGenerator` was a no-op — the recurrent state staging had been released.** The paragraph
    above documents the rule (`stateS`/`convHist`/`sampleHistory` staging must stay alive because
    `resetGenerator` re-copies it to re-zero state), but the in-process-reload double-free fix
    (gotcha §14.8) ended up calling `releaseStaging` on the persistent buffers too, nulling
    `stagingBuffer`. `createTransferAndCopy` skips VRAM buffers with a null staging handle, so
    **every `resetGenerator` silently did nothing**: requests/windows inherited the previous
    request's delta-net/conv state. The first forward after load was correct, everything after it
    was polluted — which is exactly how it surfaced, as a perplexity that changed with the order
    of prior scoring calls (2B FP16, prefill/decode 4096: 12.698 clean vs 11.755 when a short run
    preceded it; three identical runs now give bit-identical NLL). Fix: drop the
    `releaseStaging(persist)` loop; the staging is freed exactly once by `destroyBuffer` at teardown
    (host cost ~20 MB on the 2B). **`main.exe` was equally affected across requests.**
51. **`readBuffer` copies the whole buffer — size the destination to match.** `readBuffer` always
    copies `buf.size` bytes (`mappedMemory` memcpy or a staging round-trip). The scoring path read
    the 16-byte `result` buffer into a single `uint32_t`, a 12-byte stack smash (UB) on the first
    window of every scoring run; it now reads into `uint32_t[4]` like every other call site.

52. **Staging buffers are not necessarily zeros - `resetGenerator` must clear them.** The GDN snapshot
    restore writes real `stateS`/`convHist` data into the retained staging memory (`writeStaging`) and
    copies it to VRAM. `resetGenerator` originally relied on the staging still holding the zeros from
    `createState`; after a restore it would have re-copied the snapshot instead of clearing. It now calls
    `clearStaging` on every state buffer before `createTransferAndCopy`. (Same class as gotcha 50: the
    staging buffer is shared state between "initialize" and "reset".)

53. **A block and a snapshot share the same chain hash.** Block `b` is keyed by `h[b]`, and a snapshot at
    position `16(b+1)` is keyed by `h[b]` - the same 64-bit value. Snapshots therefore live in their own
    4-deep ring, not in the block table; the block table only ever maps block hashes to slots.

54. **Allocate and commit cache blocks incrementally, not all-then-commit.** The first flush version
    allocated every slot for the turn before inserting any index entry, so when the pool was smaller than
    the turn's block count, LFRU had no victims (the entries did not exist yet) and the flush gave up with
    `pool full`. `engineFlush` now processes 32 blocks at a time and commits them before allocating the
    next chunk, so earlier chunks are valid eviction candidates.

55. **Admitting a cold block must clear its cold offset.** An entry can otherwise be both RAM-resident and
    cold; a later disk-budget eviction would then delete the entry (and its RAM slot mapping) while it is
    still in RAM. `kvAdmit` marks the old cold record dead and clears `coldOffset` before taking the slot.

56. **A snapshot lookup must match the key, not just the position.** The ring stores several snapshots that
    can share a position (every prompt snapshots at its own last 16-boundary, so many prompts have a
    snapshot at 16). Selecting `snapData` by `ringPos == best` alone picked the *last* entry at that
    position, i.e. another conversation's GDN state, and restored it into the wrong conversation (a
    different prompt produced a plausible but unrelated continuation). `kvPlan` now requires
    `ringKey == hashes[pos/16 - 1]` as well. This only surfaced once two prompts shared a boundary; the
    single-conversation tests could not catch it.

57. **Never cache or restore decode-written KV.** Decode (`GEMV`/`Att-full`) and prefill (`GEMM2`/
   `Att-QK2`) are different kernels, so the same token's K/V can differ in the last bits. Restoring a
   block whose tokens were written by decode into a prefill path therefore does **not** reproduce a fresh
   prefill: the first generated token already differed (a turn-end snapshot made the continuation start
   ~3 tokens into the cold output). `engineFlush` now admits only blocks `[0, promptLen/16)` and snapshots
   only during prefill, so the reply is always re-prefilled. The prefill-end snapshot still lets turn 2
   resume at the prompt end (e.g. 592 of a 600-token prompt), which is the win that matters for agents.

58. **A fixed-slot snapshot ring silently starves interleaved conversations.** The first snapshot store was
    a `ringCount % KV_RING` ring of 4. Storing `A,A,B,B,C,C` (three 2000-token conversations) overwrote
    A's two snapshots with C's, so A's next turn found no matching snapshot and fell back to a full
    prefill (`cachedTokens == 0`) -- with no error, just lost work. It is now an LRU table keyed by
    `(pos, key)` with `KV_SNAP_MAX` slots (`snapFind` / `snapAlloc`), so the same workload resumes all
    three. The regression test deliberately uses three conversations because two would still fit in 4.

59. **The host/device byte counters were cumulative, not live.** `allocateBufferMemory` incremented
    `g_deviceLocalBytes` / `g_hostVisibleBytes` but `releaseStaging` and `destroyBuffer` never decremented
    them. Every weight that passed through a VRAM staging buffer was therefore counted as host memory
    *forever*, so a diagnostic or OOM message after loading the 35B reported ~20 GB of host-visible memory
    even though the VRAM staging had long been freed -- exactly the "weights that went to VRAM are still in
    RAM" illusion. Both frees now subtract the allocation (via `vkGetBufferMemoryRequirements`), so the
    counters track live bytes (`engineInfo` exposes them as `hostVisibleBytes` / `deviceLocalBytes`).

60. **Small weight vectors kept their staging because the registry stored copies.** `registerWeightBufferSmall`
    took a `buffer` by value; `weightFlush` released staging only for the pointer-based registry, so the
    originals (norms, `conv1d`, router, shared gate) held their host staging for the process lifetime
    (~45 MB on the 35B, more on wider models). It is now pointer-based and released alongside the large
    weights, and the three loaders register at the storage site (`&w.router[L]`, etc.).

61. **A mirror slot that is not CRC-checked silently restores torn data.** `mirror.bin` is written
    slot-by-slot at turn end; a crash mid-flush leaves the last slot half-written, and a lazy loader would
    hand the stale bytes to the unstripe shader as if they were valid KV. `kvCommitBlock` / `kvAdmit` now
    compute a CRC32 of the slot (`slotCrc`) into `kv_entry.crc`, `persistIndex` stores it, and `kvPlan`
    validates each matched RAM slot (`slotValid`): on mismatch it frees the slot, drops to the cold copy if
    one exists, and otherwise truncates the match so the turn falls back to a full prefill. The index
    version was bumped to 3 so pre-CRC stores are discarded rather than mis-trusted.

62. **A snapshot at `pos == promptLen` is unreachable for the same prompt.** `kvPlan` caps the resume point
    at `promptLen - 1` (there must be at least one token left to prefill and produce logits), so when the
    prompt length is an exact multiple of 16 the only prompt-end snapshot (`pos == promptLen`) is filtered
    out and a same-prompt replay restores nothing. `runPrefill` now also snapshots at
    `promptLen - 16` in that case, so a block-aligned prompt resumes one block short and re-prefills the
    last block. The `pos == promptLen` snapshot is still kept for the usual case where the next turn is
    longer.

63. **A cold-only entry was never mirrored into RAM.** The inclusive-RAM rule says every VRAM-resident
    block also has a RAM copy, but a block streamed back from `cold.bin` (not admitted, `freq < 2`) has
    `slot == -1`, and `engineFlush` skipped it because it was already in the index. The next turn then
    re-read it from disk forever. `engineFlush` now skips only blocks that are *RAM-resident*
    (`kvBlockResident`), so any cold-only block used by the session is restriped into a RAM slot at turn
    end. Cost is bounded: only the first turn after a cold restore mirrors anything.

64. **Concurrent first requests raced the auto-load.** `/v1` loads a model on demand, and two requests
    arriving before the load finished both saw `loaded == false` and both called `engine.load` (which
    unloads, then creates), so they could destroy each other's engine. `ensureModel` now memoizes the
    in-flight load (`loadInFlight`) and re-checks status after awaiting it, and `EngineManager.load`
    itself runs through the request queue so a load can never overlap a generation.

65. **An unterminated thinking block was misclassified as content.** With `enable_thinking: true` the
    prompt ends inside `<think>`, so the model's output is reasoning until it emits `</think>` — but if it
    hits `max_tokens` first, the closing tag never arrives. The old splitter only produced
    `reasoning_content` when `</think>` was present, so a truncated thought streamed as answer text. The
    splitter now takes the `enable_thinking` expectation into account: if reasoning was expected and no
    `</think>` appears, the whole output is reasoning (and `content` is empty).

66. **The web UI wrote its quant config in camelCase.** `/api/load` serialized `opts.quant` verbatim
    (`prefillChunk`, `lmHead`, `maxCtx`), but the engine's JSON reader is exact-match and expects
    `prefill_chunk` / `lm_head` / `max_ctx` — so the **prefill chunk and LM-head quant controls were
    silently ignored** (they fell back to 512 / fp16; `maxCtx` and `expertsVram` still worked because
    they also travel as separate `engine_options`). `EngineManager.doLoad` now writes the snake_case
    shape the engine (and `probe.ts`'s `applyExistingQuant`) reads.

67. **A load-time OOM killed the whole process.** `allocateBufferMemory` called `exit(EXIT_FAILURE)` on
    `vkAllocateMemory` failure, so a model that did not fit took down the web UI server (and the CLI)
    with no error to the client. It now records the failure, returns a null buffer, and the load is torn
    down and returned as an error (`out of GPU memory` / `out of host memory`) through `engineOpen` →
    the addon promise → `/api/load`'s `500`. See §12.


---

## 10. Verification Results (M=64, max absolute error vs CPU reference)

| Shader | max_err | GPU time |
|---|---|---|
| GEMM-FP16 / INT8 / INT4 | 0.000080 / 0.000076 / 0.000080 | ~41 ms |
| GEMM-ADD FP16 / INT8 / INT4 | 0.000069 / 0.000069 / 0.000071 | ~14 ms |
| RmsNorm-swiglu-ffn-GEMM FP16 / INT8 / INT4 | 0.0129 / 0.0176 / 0.0142 | 60  77 ms |
| RmsNorm-LinearProj-GEMM FP16 / INT8 / INT4 | 0.00018 / 0.00020 / 0.00021 | 43  51 ms |
| QKV-Rope-GEMM FP16 (q / k-cache / v-cache) | 0.000014 / 0.0021 / 0.062 | ~56 ms |
| QKV-Rope-GEMM INT8 (q / g / k-cache / v-cache) | 0.000010 / 0.0003 / 0.011 / 0.306 | ~130 ms |
| QKV-Rope-GEMM INT4 | 0.000014 / 0.014 / 0.32 | ~51 ms |
| QKV-Rope-GEMM-pos FP16 / INT8 / INT4 (position buffer = M) | 64 / 64 / 64 exact | included above |
| QKV-Rope-pos FP16 / INT8 / INT4 (decode read, unchanged) | 33 / 33 / 33 exact | included above |
| Attention-GEMM FP16 / INT8 / INT4 | 0.000003 / 0.000003 / 0.0017 | ~0.7 ms |
| GatedDeltaNet-GEMM FP16 (out / state S) | 0.000056 / 0.000001 | ~105 ms |
| GatedDeltaNet (decode) INT8 (out / state S) | 0.023544 / 0.000061 | ~30 ms |
| GatedDeltaNet (decode) INT4 (out / state S) | 0.003357 / 0.000093 | ~13 ms |
| Gate-Sigmoid | 0.000000 | ~0.02 ms |
| Conv-SiLU (M=1 / M=64, out + history) | 0.000000 | ~0.02 / 0.19 ms |
| Conv-SiLU-chunk (2Ã—M=64, out + history) | 0.000000 | ~0.19 ms |
| LMHead-ArgMax FP16 (token index) | exact match | ~117 ms (4096 Ã— 81920 GEMV + reductions) |
| LMHead-ArgMax-pos FP16 (position buffer, +1) | 41 — 42 exact | included above |
| ArgMax-Sampler-A (mode 1: temp 0.1, rep 1.2, penalty_len 256 with duplicate + sentinel ids, top-k 40, top-p 0.9; 3 chained draws) | exact token match vs fp64 CPU reference | ~7.6 ms (3 draws) |
| ArgMax-Sampler-B (mode 1: pure softmax, no penalty/top-k/top-p; 3 chained draws) | exact, or within 5e-4 CDF-mass tolerance (fp32 flat-tail draws) | ~1.0 ms (3 draws) |
| ArgMax-Sampler-C (mode 1: min_p 0.001 only; 3 chained draws) | exact token match vs fp64 CPU reference | ~0.9 ms (3 draws) |
| ArgMax-Sampler-pos / -history (position +1 per draw; ring writes at pos % penaltyLength; rng advance) | 7 — 13 exact / 0 mismatches over 1024 slots | included above |
| Embed-RmsNorm-LinearProj FP16 (q / k / v / z / a / b) | 0.000164 / 0.000145 / 0.000168 / 0.000175 / 0.000088 / 0.000084 | ~2.6 ms |
| Embed-RmsNorm-LinearProj-GEMM FP16 (q / k / v / z / a / b) | 0.000381 / 0.000313 / 0.000290 / 0.000381 / 0.000198 / 0.000252 | ~155 ms |

The larger k/v-cache errors are fp16/int4 cache quantization, matching the existing GEMV baselines. Note the CPU references are single-threaded and dominate total runtime (`main.exe val` takes several minutes); the GPU shader times above are per-shader query-pool timestamps. The sampler validator (`validateArgMaxSampler`) drives `ArgMax-Reduce(mode 1)` standalone: it seeds the logits/history/rng buffers, chains 3 draws per case through the **in-place** buffer (the CPU fp64 reference mirrors the shader's full pipeline — transform, penalty-once via membership set, prob conversion overwrite — so both sides stay in lockstep across draws), replicates the xorshift32 draw `u = (rng>>8)/2^24` and the ascending-id CDF pick exactly, and accepts a draw either as an exact token match or, when the GPU's fp32 accumulation rounds across a near-flat CDF boundary, when `uÂ·Z` falls within 5e-4Â·Z of the GPU token's interval.

The table above is the **validation harness** (M=64). The engine additionally self-checks end-to-end with its synthetic (seeded-random) weights: a 16k-token run must produce the expected deterministic signature, and `main.exe val` must stay green. A clean 16k logged run is the acceptance test for any prefill shader change.

---

## 11. Build & Run

```bash
make clean          # removes bin/ and build/  (required after ANY header edit — see gotcha 12)
make                # shaders -> bin/shader/*.spv, builds bin/main.exe + bin/pumice.node

cd bin && main.exe val       # validation harness (every validate* + max_err + timing)
cd bin && main.exe meminfo   # dump memory heaps/types (device-local vs host-visible) and exit
```

Real weights are loaded by the **server** (`main.exe`, default). It reads a length-prefixed prompt from stdin and writes generated ids to stdout, so it is normally driven by the Python frontend:

```bash
uv venv .venv                                  # once
uv pip install --python .venv/Scripts/python.exe tokenizers   # once

.venv/Scripts/python.exe pumice_llm.py <model_dir> <max_ctx> "prompt text" [--think] [--prune] [--experts-vram N]
# e.g.
.venv/Scripts/python.exe pumice_llm.py model/Qwen3.5-2B 8192 "why the sky is blue?" --think
.venv/Scripts/python.exe pumice_llm.py model/Qwen3.5-2B 131072 "long document question" --prune
.venv/Scripts/python.exe pumice_llm.py model/Qwen3.6-35B-A3B 8192 "why the sky is blue?" --think --prune --experts-vram 32
```

No `--think` means non-thinking mode. `--prune` selects the **pruned 102,400-token vocab** and bootstraps a fresh model dir: the engine's `pruneVocab` resolves the vocab weights cache-first (weight cache → `<model>/vocab/` → auto-gather from the shards via `pruned-vocab/mapping.npy`, §4.7) and always ensures the tokenizer files in `<model>/vocab/` (the Python frontend copies them from the root `pruned-vocab/` before spawning the engine, since it needs the tokenizer first). Once the weight cache or `vocab/` exists, `--prune` is a no-op and can be dropped.

**Without `--prune` the engine runs the original 248,320-token vocab** straight from the shards: `vocab` comes from `config.json`'s `vocab_size`, the tokenizer is the model-root `tokenizer.json` (original ids), and EOS resolves via the `added_tokens` fallback (§4.7). The two modes keep separate cache files (`embed_102400_FP16.bin` vs `embed_248320_FP16.bin`) and can be interleaved freely. **VRAM caveat**: the full vocab fits the 8 GB heap only on the 2B (~1 GB tied embed); the 9B's two untied FP16 heads (~4 GB) OOM — run the 9B with `--prune`.

`pumice_llm.py` exposes `start_llm(weight_dir, max_ctx, max_new_tokens, dump_dir=None, dump_layers=0, debug_sampling=False, prune_vocab=False, experts_vram=0)`, `apply_chat_template(messages, enable_thinking=True)`, `tokenize`, `generate`, `generate_stream`, `detokenize`, `close`. `tokenize(text)` wraps the text in the **Qwen3.5 chat template** with the `<think>` control token (gotcha 32); the tokenizer comes from `<weight_dir>/vocab/tokenizer.json` (pruned ids, 85992 EOS) with `prune_vocab=True`, else the model-root `tokenizer.json` (original ids, 248046 EOS) — the choice must match the engine's `--prune` flag or ids desynchronize.

**Sampling** is opt-in and configured by module-level constants near the top of `pumice_llm.py` — `is_sampling`, `temperature`, `rep_penalty`, `penalty_len`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `seed`. With `is_sampling = False` (default/greedy) the request sends `temperature = 0.0`; with `is_sampling = True` it sends the constants and the backend runs the sampler (§7.7). These are consumed by `_stream_ids`, so `generate`/`generate_stream` take only `(llm, token_ids)`. Defaults follow Qwen3's thinking-mode recommendation: temperature 0.6 / top_k 20 / top_p 0.95 / rep_penalty 1.05 / penalty_len 64 (`min_p = 0.0` optional tail filter, `presence_penalty = 0.0` off — raise it to flat-penalize any token already present in the penalty window).

Generation streams: `generate_stream(llm, ids)` yields **incremental decoded text** (reads token ids from the stream until the `0xFFFFFFFF` sentinel, accumulates them, and yields the new tail of `tokenizer.decode(...)` at each step), while `generate(llm, ids)` returns the full token-id list (used by tests / `detokenize`). The `_main()` CLI driver prints text incrementally (`print(..., end="", flush=True)`) and, after generation, reports **`N tokens, X.X tokens/s`** (timing measured host-side per token after the first 4).

**Layer-differential harness** (`--dump <dir> --dump-layers <N>`): the server dumps per-layer fp32 tensors (`layer_00_embed`, `layer_XX_h`, `layer_XX_hattn`, plus attention internals `qkvRaw`/`qOut`/`kCache`/`vCache`/`scores`/`smSum` and FFN `act`/`gAct`/`uAct`) after each prefill layer. `tools/run_dump.py` drives it; a layer-comparison script (run under `tools/pruner/.venv`, which can load the model in transformers) compares each dump against the HF `output_hidden_states` and prints per-layer cosine correlation / max|diff| / rms — `tools/cmp_layers.py` for the 9B; for the 2B, map pruned ids through `vocab/mapping.npy` before the HF forward. This is the tool that isolated gotchas 27–30, 33, 38. **Note:** `tools/` scripts must not be named after stdlib modules (`tools/tokenize.py` shadowed stdlib `tokenize` and crashed torch imports — renamed `tokenize_cli.py`).
## 12. VRAM budget (8 GB RX 580)

Device-local memory (`heap[0]`) is **7936 MB**, not 8192. With the 9B hybrid spec the totals are (metered by `createBufferNamed` in `buffer.c`):

| Component (9B) | MB |
|---|---|
| weights | 6545.74 |
| KV k/v cache (8 full-attn layers) | 640 |
| KV scale/zero (6 INT8/INT4 layers) | 12 |
| `attScores` (`maxM × maxCtx × kvHeads × 2`) | 128 |
| `act` + `gAct` + `uAct` | 72 |
| `stateS` delta recurrence (24 × 2) | 48 |
| h/emb/attn/q-proj group, `qkvRaw`, partials | ~155 |
| **total device-local** | **~7550** |

This fits arithmetically but overflows at runtime — fragmentation from ~450 discrete allocations means a single further 128 MB `attScores` block can't be satisfied. The failure is deterministic and reported as `out of GPU memory: failed to allocate 'attScores' (128.00 MB) | device_local=7352.66 MB ...`. Levers: lower `max_ctx` in `quant_config.json` (KV + attScores scale with it → −576 MB at 8192; the `--max-ctx` flag also *resizes the allocations* (grow or shrink), not just the generation limit), lower `prefill_chunk` (−160 MB at 256), INT8 embed/lm-head (−640 MB), or consolidating the ~450 tiny allocations into arenas. Staging/host buffers are **not** the issue — they live in the 16/32 GB system heap.

**A failed allocation no longer kills the process.** `allocateBufferMemory` records the error (`bufferAllocFail`) and returns a null buffer; `createBufferNamed` fast-fails afterwards; `createState`/`createWeights`/`createGenerator` check the flag, destroy whatever they allocated, and `engineOpen` returns NULL. The CLI prints the message; the addon rejects the `createEngine` promise, so the web UI shows it in the status line and the server keeps running (verified by `test_webui`'s `oom load rejected` check, which then loads a smaller config successfully). This makes the failure recoverable in-process instead of an `exit()`.

The 2B (all-FP16, 102400 vocab, tied embeddings) totals ~1.7 GB of weights + ~0.4 GB state at
`max_ctx = 32768` — comfortable. Context is now runtime-resizable (gotcha 49): at
`max_ctx = 131072` the ctx-dependent state grows to ~1.3 GB (KV caches ~1.0 GB + `attScores`
256 MB, ~5 GB device total) and is validated end-to-end; the ceiling is `max_position_embeddings`
262144 (~2.5 GB ctx-dependent state, still fits arithmetically). State-buffer staging is released
after the initial zero-copy, so the retained host-visible footprint no longer mirrors the VRAM
state total. In original-vocab mode (no `--prune`) the 2B's tied embed grows to ~1.0 GB (~3.6 GB total, still comfortable); the 9B's two untied heads would need ~4.7 GB at 102400 and does not fit the heap — run the 9B pruned (lower `--max-ctx` if the extra 16384 rows tip it over). Note the 9B's measured budget can also be squeezed by *other applications* holding VRAM (browser/OBS reserve ~1.3 GB some sessions) — an OOM at `max_ctx 8192` that previously worked is usually external pressure, not a regression; retry after freeing VRAM or run at a lower `--max-ctx`.

**Qwen3.6-35B-A3B (MoE, experts_vram=32)**: device-local ~3.6 GB weights (embed+lmHead 704 MB
pruned, INT8-edge/INT4-mid attention ~1.0 GB, 33-expert VRAM pools ~2.0 GB, routers 40 MB) +
~0.5 GB state — comfortable. The **host-visible** side is the constraint: 224 RAM experts/layer
across 40 layers ~= 13.5 GB of host-visible pool + transient staging (batched to ~12 buffers).
The heap is 16091 MB; with ~16 GB free system RAM the 32/224 split just fits. Raising
`--experts-vram` trades host for device memory (each extra VRAM expert costs ~1.55 MB device,
saves ~1.55 MB host per layer): the device ceiling (~7.2 GB usable) caps experts_vram around ~95
on this GPU. Decode runs at ~10.7 tok/s with 224 RAM experts streaming over PCIe (~1.5 MB read
per RAM expert hit) — the planned LFRU expert swap (keep hot experts in VRAM, prefetch on routing
probability) is the path to the all-VRAM ~25 tok/s.

## 13. Known issues / not yet implemented

1. **(Resolved)** Gated-deltaNet `A_log`/`dt_bias` exponential decay, the q/k L2-norm (eps 1e-6, q Ã— 1/š128), the decayed-`vhat` delta rule, and the `in_proj_z` output gate (`rmsnorm(y)Â·norm.weightÂ·silu(z)`) are all implemented (Â§7.3). Linear-attention layers are **HF-identical** to the Qwen3.5 9B block.
2. **(Resolved)** `rms_norm_eps = 1e-6` everywhere (shaders + `rms_norm_apply`), matching the model config.
3. **(Resolved)** Partial RoPE (`0.25`): the engine now rotates only 64 of 256 dims (`MODEL_ROTARY_DIM`, 32 freqs, Î¸ base 1e7); dims 64..255 pass through.
4. **(Resolved — no-op)** Qwen3.5 applies no final logit scaling: `logits = lm_head(rmsnorm(x))`, verified against the reference.
5. **(Resolved)** Full-attention fidelity: `q_proj` q/g interleave (`buildQkvMatrix`), the GQA 16-head mapping (gotcha 25), and the `1/šhead_dim` QK scaling are all applied. `pumice_llm.py` now applies the Qwen3.5 **chat template** (thinking mode) via `apply_chat_template`.
6. **VRAM OOM** (Â§12) blocks a full end-to-end generation on the 8 GB card at `max_ctx = 32768`.
7. **(Resolved) End-to-end correctness, prefill and decode.** The engine now produces coherent, correct output against the real 9B weights in both thinking and non-thinking modes ("why the sky is blue?" — Rayleigh-scattering explanation, "2+2" — "4", "name a color" — "Blue"). A layer-by-layer differential (`--dump` + `cmp_layers.py`, run under `tools/pruner/.venv`) shows all 32 prefill layers tracking the HF reference at 0.96  0.9999 cosine correlation, and the decode trajectory matches the pruned-vocab-constrained HF greedy exactly, token-for-token. Eight bugs were fixed across prefill and decode (gotchas 27  32): the headline prefill bug was the full-attention output gate (`sigmoid` vs `silu`, gotcha 27), and the headline decode bug was `Att-full-{INT8,INT4}` dequantizing V with K's scale/zero (gotcha 31). The chat-template ` thinking` token fix (gotcha 32) was what finally enabled thinking-mode output. Greedy argmax remains the default and still matches the HF greedy reference; an optional sampling path (temperature / repetition penalty / top-k / top-p) is available (Â§7.7).
8. **(Resolved) Streaming output + per-token pacing.** The server now emits token ids one at a time (with an `0xFFFFFFFF` terminator) instead of an `[m][ids]` burst, and `generateTokens` — whose GPU side still computes `DECODE_GROUP = 4` tokens per pass — simulates per-token cadence by sleeping between the 4 emitted tokens of a group (delay = the previous group's measured per-token time; first group constant 25 ms). `pumice_llm.py`'s `generate_stream` yields decoded text incrementally and the CLI prints `N tokens, X.X tokens/s`. Weight loading prints a single stderr progress bar / spinner by default; the old per-tensor (and KV-cache allocation) lines are gated behind `--verbose-weights`.
9. **(Resolved) Decode attention K-cache transpose.** The K cache was stored token-major (`[token][row]`), so the decode QK dot (`Att-SplitK2`/`Att-full`) read K in an uncoalesced, 1024-strided gather. Transposing K to dim-major (`kCache[rowÂ·MAXCTX + token]`) across the writers (`Rope-GEMM`, `Reduce-Rope`) and readers (`Att-QK2`, `Att-full`, `Att-SplitK2`), while leaving V token-major, coalesces those reads. Measured ~3Ã— faster decode attention (`Att-SplitK2` per-call avg 0.901—0.337 ms FP16 and 0.689—0.219 ms INT8/INT4 at 4k ctx); the prefill `Att-QK2` B-load became coalesced too. The `--timing` CLI flag (plus `isTimingEnabled()`) drives the Â§8.4 per-op log, added to measure this.
10. **MTP and vision tensors are ignored** (text-only baseline).
11. **(Resolved) Sampling — temperature / repetition penalty / presence penalty / top-k / top-p.** `ArgMax-Reduce` was extended into a `mode`-selected sampler over a new full-logits buffer (Â§7.7), with a per-request `sampleParams` block, a sentinel-filled repetition-penalty ring (`sampleHistory`), and an on-GPU xorshift32 RNG (`sampleRng`). `g->sampling = (temperature > 0)`; greedy (`mode 0`) is byte-identical to the prior behavior. Sampling is opt-in from Python via module constants (`is_sampling`, `temperature`, `rep_penalty`, `penalty_len`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `seed`), and a `--debug-sampling` flag dumps the history ring / position / generated tokens for diagnostics. The initial sampler cost ~15 ms/token (sampling at ~20 tok/s vs 30+ greedy); the Â§7.1 rewrite (probability-space conversion, per-thread early-outs, vec4 access, 28 bisections, subgroup reductions, TS 256—1024) cut it to 0.64 ms/token — the sampling/greedy gap is now under 1 tok/s. A second-round fix replaced the per-logit repetition-penalty history scan with a shared-memory membership bitmap, making the penalty cost independent of `penalty_len` (penalty_len=256 previously dropped sampling ~30 — ~26 tok/s; now parity with 16) and restoring once-only CTRL scaling for duplicated ids; `validateArgMaxSampler` (Â§6) plus the re-enabled `validateLmHeadArgMaxFP16` cover both modes against a fp64 CPU reference. A `min_p` filter (`p < min_p` dropped; trivial in prob space, Â§7.1) was added as a fourth knob with its own validator case. A fifth knob, **`presence_penalty`** (2026-09), implements Qwen/vLLM semantics: flat `logit -= presence_penalty` in the raw-logit frame for any id present ≥1× in the window, applied in the temperature-scaled frame as `pres × 1/T` so it commutes with temperature exactly like the rep penalty; it reuses the shared membership bitmap (active when `repPenalty != 1` or `presencePenalty != 0`), and the request header grew 28 → 32 bytes (`<ffIIfffI`) with the wire format updated in `pumice_llm.py` and `tools/vocab_collect/collect.py` in lockstep; validator case D covers presence-only (`rep 1.0, presence 0.3, penLen 256`) against the fp64 reference.
12. **(Resolved) Thinking-mode degeneration — pruned vocab was missing the model's reasoning-starter tokens.** Symptom: sampling at temp 0.6+ produced spurious `</thinking>` close-tags assembled from plain tokens, repeated answers, or early EOS; greedy degenerated into `1. 1. 1.` loops. Diagnostics: same-seed runs were reproducible, `top_k=1` matched greedy exactly, and the engine matched the **pruned-constrained** HF greedy 159/160 tokens — the engine was numerically correct. The real cause: HF's top-1 token after `<think>\n` was `'Thinking'` (logit margin +4.75 over rank 1) and `'Okay'`/`'Looking'`/`'Trying'` were also top-10 — all pruned away by the Wikipedia-corpus frequency trim. Every thinking generation started off-distribution. Fix: data-driven protection — teacher-forced HF over diverse engine-generated sequences to collect top-10 missing tokens (772 ids, `chat_protect.npy`), a `--protect-ids` option added to the pruner (`tools/pruner`), re-selected/re-applied the 81920 vocab, and re-gathered the embed/lm-head rows. **Gotcha: `bin/weights/` caches quantized tensors on disk — after any vocab change delete `embed_81920_FP16.bin`/`lmHead_81920_FP16.bin` or every generation is gibberish.** After the re-prune the engine matches unconstrained HF greedy from the very first token (`'Thinking'` — id 40378), thinking closes exactly once, and 6/6 sampled runs at temp 0.6 are structurally clean (previously ~50% broken).

13. **(Resolved) Multi-model support — Qwen3.5 2B.** All model dimensions, the layer spec, and the per-layer quantization now come from the model folder's config.json + quant_config.json (§1, §4.7) — no engine rebuild or shader recompile per model. Three model-specific bugs were found and fixed while enabling the 2B (heads 8 / kvHeads 2 / K 2048 / tied embeddings): the GQA mapping assumption heads == kvHeads² (gotcha 34 — the 9B's 16/4 only worked by coincidence; the stable fix pushes gqa and the real heads as separate push members), the q_proj de-interleave dims (gotcha 33), and prefill pad-token state poisoning (gotcha 38). The 2B runs at ~46 tok/s (all-FP16, pruned vocab, max_ctx 32768, ~3.5 GB VRAM), greedy-matches HF through long-context decode, and its sampling path is clean past ctx 256 (the exact failure the Reduce-Att2 head-count bug caused). A fresh model dir is bootstrapped by the `--prune` flag, which gathers the pruned-row vocab from the shards via the root `pruned-vocab/mapping.npy` (§4.7, §11).
14. **(Resolved) Subgroup portability (wave32 GPUs).** Cross-subgroup scratch arrays were sized TS/64 (wave64) with hardcoded 4-slot combines — out-of-bounds on wave32 devices (RTX 4060: 8 subgroups per 256 threads). All such arrays are now sized (TS+31)/32 with gl_NumSubgroups-bounded combines; RmsNorm-Prologue (whose whole-workgroup-one-subgroup assumption raced on wave32) uses a shared-memory tree reduction; bare subgroupMax/subgroupAdd calls on 64-thread workgroups (Att-SplitK*) were replaced with workgroup-wide shared reductions (gotcha 39). Untested on the actual wave32 hardware end-to-end, but structurally correct.
15. **(Resolved) Cache-first weight loading.** Every load path checks the on-disk cache (in/weights/<model>/) before opening safetensors: shards are optional (a copied cache runs standalone), vocab weights resolve cache → ocab/ folder → shard, and small vectors (norms / A_log / dt_bias / conv1d) have their own ec_*/conv_* cache entries. One model per process keeps the specialization constant (d_model) consistent with the cached pipelines.
16. **Vocab re-prune cache invalidation is manual** (gotcha 37). The weight cache keys on (tensor name, dims, quant) — after re-gathering a vocab or editing quant_config.json, delete in/weights/<model>/ or generation silently runs on stale tensors.

18. **(Resolved) Qwen3.6-35B-A3B MoE support.** `num_experts > 0` in config.json switches every FFN
    to `FFN_MOE`: per-layer 257-expert INT4 pools (cache-complete + cold-build from the 26 shards,
    ~30-60 min one-time), FP16 router + shared-expert gate, four-op MoE chain (§7.8), expert
    pools split VRAM/RAM by `--experts-vram` (default from quant_config, 32), and the
    `nV*dim != K` out-projection generalization (gotchas 42-48). Verified layer-by-layer against a
    CPU safetensors reference (delta L0 corr 0.9999 including the stateful recurrence, full-attn L3
    corr 0.998 with gqa 8, all 9 MoE slots >= 0.986) and end-to-end ("2+2" -> "4", a correct
    Rayleigh-scattering paragraph, ~10.7 tok/s decode, EOS-terminated). The 2B and 9B remain green.
    MoE model constraint: layer 0 attention must be `fp16` (the fused decode
    Embed-RmsNorm-LinearProj op is FP16-only and reads the L0 proj weights as FP16 - an INT8/INT4
    L0 emits NaNs from the first decode step; the 35B quant_config sets layer 0 attn fp16).
    (Lifted 2026-09: the decode L0 input projection is now quantization-aware — for INT8/INT4 L0
    the fused op is replaced by Embed-Gather + RmsNorm-LinearProj-SplitK-{INT8,INT4} +
    Reduce-LinearProj, the same quantized pair prefill already uses; the FP16 path keeps the
    fused kernel and its trajectory is byte-identical. L0 may now use any precision.)

19. **(Resolved, 2026-09) Coding-domain vocab recovery — pruned vocab grown 86016 → 102400.**
    The Wikipedia-corpus pruned vocab struggled at coding tasks: code tokens (BPE identifier
    fragments, operators, keywords, punctuation runs) were largely trimmed away. The fix collects
    the missing tokens **from the engine itself**, using the 2B at the full 248320-vocab (the only
    model whose tied embed fits the 8 GB heap unpruned):

    - **`--dump-topp <file>`** (server flag, src/generate.c): when active the lm-head chain is
      split — `LMHead-GEMV-FP16` runs alone, the full 248320-vocab `logits` buffer is read back,
      the top-512 `(id, logit)` pairs are selected host-side with a fixed min-heap, and one record
      `[step u32][count u32][{id u32, logit f32}×count][expSum f64]` is appended and flushed per
      step (prefill `finalOps` split and a dedicated sequential one-token-per-dispatch decode loop;
      `generatorSetSampling` forces sampling mode at temperature 1e-9 when the flag is set so the
      logits buffer is always written). Verified: greedy picks equal dump top-1 token-for-token,
      trajectory byte-identical to a no-dump run, record count == max_new.
    - **`tools/vocab_collect/collect.py`**: drives the engine over `prompts.py` (111 coding
      prompts: python, web, react, php, sql/sql-server/postgres, mongodb, c, c++, go, shell,
      json/yaml/docker, regex — thinking and non-thinking variants) × 3 seeds × 96 tokens at the
      Qwen sampling defaults; incrementally parses the dump, computes the exact nucleus
      (cumulative exp / expSum ≥ top_p 0.95) per step, and unions distinct ids →
      `pruned-vocab/topp_tokens.json`.
    - **`tools/vocab_collect/build_vocab.py`**: filters the collected ids to those **missing** from
      `mapping.npy`, appends them (append-only — every existing pruned id keeps its value), pads
      with unused original ids up to the next 4096-multiple, and rewrites `mapping.npy` +
      `tokenizer.json` + `vocab.json` (renumbered; `--backup` writes `.bak` copies, `--dry`
      previews).
    - **Result**: 333 runs → 29711 distinct nucleus ids, 13111 missing from the 86016 mapping →
      `MODEL_VOCAB` 102400 (99127 used, 3273 pad; 90112/94208 were too small per the 4096-multiple
      rule). Recovered tokens include `=>`, `.get`, `</`, `!=`, `&&`, `<div`, `(function`,
      `Integer`, `IMPORTANT`, `False`, `Args`.
    - **Cache/mapping notes**: `include/generated_vocab.h` bumped to 102400 (header edit →
      `make clean`, gotcha 12); model weight caches deleted so `--prune` re-gathers
      `embed_102400_FP16.bin` (2B: 419 MB). Old 86016 caches would still load if the header were
      reverted — the V-suffixed filenames keep all vocab modes coexisting (§4.7). The 9B/35B pick
      up the same files unchanged; the extra 16384 rows cost ~670 MB per untied FP16 head (§12).
    - **Validation (2B, --prune, 102400)**: thinking-mode sampled runs produce clean
      think-open/close + Python code with docstrings; Go / SQL / C++ sampled runs coherent; greedy
      fluent; no `?` collapse.

20. **(Resolved, 2026-09) Runtime-resizable context length.** `--max-ctx` is now a true override in
    both directions (grows or shrinks the KV-cache / `attScores` / `attPartial` allocations at
    startup), clamped to config.json `max_position_embeddings`; the decode split-K attention cap
    was lifted from 32768 to 262144 tokens (`ATT_MAX_CHUNKS 1024` shared by the four `Att-SplitK2`/
    `Reduce-Att2` shaders, the dispatch geometry, and the state sizing — gotcha 49); over-long
    prompts are rejected (`generateTokens` guard + `pumice_llm.py` ValueError) instead of writing
    out-of-bounds KV; and state staging is released after the initial zero-copy (except the
    `resetGenerator`-relied-upon `stateS`/`convHist`/`sampleHistory` staging). Validated on the 2B
    at `max_ctx = 131072`: a 41698-token prompt with a needle past position 32768 (unreachable
    under the old decode cap) is retrieved correctly, short-prompt runs are unchanged at 32768 and
    8192, and `main.exe val` is fully green.

---

## 14. Node addon & Web UI

The React web UI runs on top of the same C engine, compiled into a Node native addon instead of the
stdin/stdout daemon. The Python frontend (`pumice_llm.py`, `webui.py`) is kept and still works — both
frontends share `createGenerator`/`generateTokens`.

### 14.1 Layout

```
package.json               # npm package @h4zel/pumice (bin: pumice)
cli.js                     # CLI: args, platform-package resolution, browser open
pack.mjs                   # prepack: build webui + stage runtime into platform/win32-x64
release.mjs                # version bump + publish both packages
platform/win32-x64/        # @h4zel/pumice-win32-x64: pumice.node + shader/ (staged)
bin/pumice.node        # N-API addon (make target)
include/engine.h           # engine_options, engine_* API
src/engine.c               # engineOpen/Close/Tokenize/Decode/Generate + tokenizer loading
src/addon.c                # N-API bindings (C only, no C++ wrapper)
gen_node_lib.ps1           # node.exe -> build/libnode.a (napi_* import lib)
tokenizers/                # vendored tokenizers-cpp (Rust staticlib + C API header)
webui/                     # React + Vite frontend + Express/TypeScript server
test_webui.mjs             # Node end-to-end test (spawns the server, probes/loads/chats/unloads)
```

### 14.2 Tokenizer integration (`tokenizers/`, `src/engine.c`)

Tokenization moved into C via the vendored **tokenizers-cpp** C API
(`tokenizers/include/tokenizers_c.h`): `tokenizers_new_from_str` (HF `tokenizer.json`),
`byte_level_bpe_tokenizers_new_from_str` (BPE from vocab/merges), `tokenizers_encode`,
`tokenizers_decode` + `tokenizers_get_decode_str`, `tokenizers_free`. The static lib
`tokenizers/lib/libtokenizers_c.a` is a MinGW (`x86_64-pc-windows-gnu`) Rust build, linkable with
the project's gcc (see §5).

`loadTokenizer` in `src/engine.c` resolves the tokenizer exactly like the Python frontend, in this
order:

1. **HQM** — an embedded `tokenizer.json` tensor (`tokenizers_new_from_str`); otherwise the
   `tokenizer.tokens`/`tokenizer.merges`/`tokenizer.token_type` tensors, turned into byte-level BPE
   blobs.
2. **GGUF** — `tokenizer.ggml.tokens`/`.merges`/`.token_type` from the GGUF KV metadata.
3. **Safetensors** — `<dir>/vocab/tokenizer.json` when `--prune`, else `<dir>/tokenizer.json`.

The byte-level BPE blobs follow tokenizers-cpp's exact expectations (verified against the Rust
source): **vocab = JSON object `{token: id}`**, **merges = newline-separated `"a b"` lines** (not a
JSON array), and **added_tokens = JSON object `{token: id}`** for token types 2/3/4 (control/user/
unused specials). A wrong format makes the Rust side panic (`Invalid added_tokens.json file.`), which
aborts the process — hence the format is exact.

`engineGenerate` accumulates generated ids and, per token, calls `tokenizers_decode` on the full
list, computes the suffix delta against the previous decode, and hands `(token, delta)` to the emit
callback. This mirrors `pumice_llm.generate_stream_tokens`.

### 14.3 Engine layer (`src/engine.c`)

```c
typedef struct { const char* weights; const char* quantConfig; int maxCtx; int prune;
                 int expertsVram; int exportModel; const char* exportDir; } engine_options;

engine* engineOpen(const engine_options* opts, char* err, size_t errCap);
int  engineTokenize(engine* e, const char* text, int addSpecial, uint32_t** out, size_t* n);
char* engineDecode(engine* e, const uint32_t* ids, size_t n);
void engineGenerate(engine* e, const uint32_t* prompt, size_t n, const sample_params* p,
                    uint32_t seed, int maxNew, engine_emit emit, void* ctx);
void engineClose(engine* e);
```

`engineOpen` runs the same startup sequence as `serverMain` — `loadModelConfig` (with the optional
`--quant-config` override, §4.7) → `hqm_resolve` → `pruneVocab` → `parseEos` → `createSession` →
`createGenerator` → `loadTokenizer`. `engineGenerate` wraps `generatorSetSampling` + `resetGenerator`
+ `generateTokens` with the per-token decode callback.

### 14.4 N-API addon (`src/addon.c`)

Exports an object with `createEngine`, `destroyEngine`, `engineInfo`, `tokenize`, `decode`,
`generate` (pure N-API C, no `node-addon-api`). `createEngine` and `generate` run on
`napi_async_work` worker threads so the JS event loop stays free; `generate` streams tokens through a
`napi_threadsafe_function` whose JS callback receives `{ token, delta }`. A single-generation guard
rejects concurrent calls. `createEngine` returns a Promise; the engine handle is a
`napi_create_external` with a finalizer that calls `engineClose`.

### 14.5 Web server (`webui/server/`)

A small Express + TypeScript server (default `127.0.0.1:8787`) that loads the addon and serves the
built React app. `paths.ts` centralizes path resolution so the server works both from the repo and
from an npm install: the runtime dir (addon + `shader/`, `PUMICE_RUNTIME`), the served `dist`,
and the model root are environment-driven, and `process.chdir(runtimeDir)` is done before any engine
use (the engine's shader lookup is cwd-relative). The models root is not defaulted — it is whatever
`PUMICE_MODELS` points at or the folder the user scans from the UI; `pruned-vocab/` is read from the
launch cwd (passed to C through `PUMICE_PRUNED_VOCAB_DIR`, §4.7), the quant temp file from the OS temp
dir, and exports from the launch cwd's `exported/`.

| Route | Method | Purpose |
|---|---|---|
| `/api/status` | GET | `{ loaded, info, engine }` |
| `/api/models` | GET | `{ models, dir }` — the last scan (empty until a folder is scanned) |
| `/api/models` | POST | `{ dir }` → scan that folder one level for safetensors dirs + `.gguf`/`.hqm` files |
| `/api/open` | POST | `{ path }` → `{kind:"folder", models, dir}` if it is a plain folder, else `{kind:"model", info}` (probe); 400 if the path does not exist |
| `/api/probe` | POST | validate + summarize a model (see below) |
| `/api/load` | POST | write the UI quant config to a temp file, `createEngine`, load |
| `/api/unload` | POST | `destroyEngine` |
| `/api/chat` | POST | build the Qwen chat template, tokenize, generate; **SSE** stream of `{delta}` then `{done,tokens,elapsedMs}` |
| `/api/score` | POST | tokenize the posted text, run the overlap-window perplexity pass (§15); **SSE** stream of `{progress}` then `{done,ppl,loss,count,chunks,tokens,elapsedMs}` |
| `/api/score/stop` | POST | `requestStop` — ends the scoring loop and returns the partial result |

`probe.ts` mirrors the Gradio validation: for **safetensors** it reads `config.json` (required
`text_config` fields, `layer_types` count, `max_position_embeddings`, `num_experts`,
`tie_word_embeddings`), resolves the shard list from `model.safetensors.index.json` or the
`model*.safetensors` glob, and verifies every shard (including the `%05d-of-%05d` sequence) exists;
for **GGUF** it walks the KV metadata (tolerant `Reader` that streams from the file descriptor, so
multi-GB files never load into memory) for `block_count`/`full_attention_interval`/`expert_count`/
`context_length`; for **HQM** it parses the header (KV pairs + tensor directory) and reads the
`config.layer_type`/`layer_attn_quant`/`layer_ffn_quant` int32 tensors. An existing
`quant_config.json` next to the model pre-fills the per-layer quant table.

The quant config written by `/api/load` has the same shape as `quant_config.json`
(`name`, `max_ctx`, `prefill_chunk`, `embed`, `lm_head`, optional `experts_vram`, `layers[]`) and is
passed to the engine through `--quant-config`; the model directory is never modified. HQM models
ignore the quant config (quant is baked into the file) and the UI locks the editor.

### 14.6 React UI (`webui/src/`)

Three tabs, matching the Gradio feature set:

- **Model** — a `model/` dropdown (A1111-style) plus a free-text path field; probe/validation
  status; a collapsible **Quantization** accordion with a per-layer `attn`/`ffn` `FP16`/`INT8`/
  `Q4_1_32`/`Q4_1_64`/`Q4_1_128`/`Q4_1_256` selector and "set all" presets (disabled + dimmed for HQM); max context (bounded by
  `max_position_embeddings`), prefill chunk; `Embed quant`/`LM head quant` for untied models or a
  single `Embed/LM head quant` for tied models; `Experts in VRAM` only for MoE models; prune-vocab
  and export-HQM toggles with an export dir. **Load / Export HQM** writes the `.hqm`.
- **Sampling** — a `Sampling` checkbox reveals temperature / top-k / top-p / min-p / repetition
  penalty / penalty length / presence penalty / seed; a `Max new tokens` checkbox reveals a slider
  capped at the context size; thinking + hide-thinking toggles; system prompt.
- **Chat** — single-session streaming chat (SSE deltas), token/s status, and a Clear button.
- **Eval** — perplexity scoring (§15): pick a text file, set prefill/decode window sizes and the
  test size (% of file), watch per-window progress (running ppl + scored-token count), and read the
  final ppl / mean NLL / ms-per-token. Stop returns the partial result.

### 14.7 Build, run, test

From the repo:

```bash
make                       # also builds bin/pumice.node

cd webui
npm install                # once
npm run build              # vite -> webui/dist
npm start                  # Express server on 127.0.0.1:8787 (serves dist + /api)
# development: npm run dev  (vite on :5173 proxying /api -> :8787)

node test_webui.mjs        # from the repo root: spawns the server and runs the end-to-end test
```

`test_webui.mjs` starts `webui/server/index.ts` via the local `tsx`, waits for `/api/status`, then
checks the model scan, probes for HQM/GGUF/safetensors (and a missing-shard rejection), loads the
pruned 2B HQM, streams a chat reply, serves the built UI, and unloads.

Installed as a global command (see §14.9):

```bash
npm install -g @h4zel/pumice
pumice                 # from a folder containing model/ and pruned-vocab/
pumice --port 9000 --models D:\models --no-open
```

### 14.8 In-process reload fixes (2026-09)

The addon is the first consumer to load, destroy, and reload models **in one process** (`main.exe`
never did), and it exposed two long-standing bugs that the old exit-immediately lifecycle had masked.

1. **Teardown double-free.** `createState` copied its buffer list into a local array and called
   `releaseStaging` on the **copies**, so the staging Vulkan buffer/memory was destroyed early while
   the real state buffers still held the handles — `destroyState` then freed them a second time
   (`STATUS_HEAP_CORRUPTION`, exit `0xC0000374`). The tied-embedding path had the same shape:
   `w.lmHead = w.embed` was assigned *before* `weightFlush()` released the embed staging, leaving
   `w.lmHead` with a stale staging handle that `destroyWeights` freed again. Fixes: `createState`
   tracks `buffer*` pointers and releases staging through them (including the persistent
   `stateS`/`convHist`/`sampleHistory` staging); `createWeights` re-aliases `w.lmHead = w.embed`
   **after** `weightFlush()`.
2. **Stale device-scoped caches.** `pipeCache`, `descPool`/`descCache` (src/dispatch.c) and the
   spec-constant values (src/pipeline.c) are process-global and keyed only by shader/descriptor
   shape, but `createSession` builds a **new `VkDevice`** per engine load. The second engine reused
   pipelines and a descriptor pool created on the destroyed device, hanging or crashing during the
   next load. Fix: `dispatchReset(VkDevice)` destroys every cached pipeline/layout/set-layout and the
   descriptor pool, clears the caches, and calls `pipelineClearSpec()`; `destroySession` calls it
   (via `dispatch.h`) before tearing down the device.

`main.exe` and the addon now tear down cleanly (exit 0) for HQM, GGUF, and safetensors, and the web
UI can load/unload/reload models repeatedly — `test_webui.mjs` covers HQM → chat → unload →
safetensors(+UI quant+prune) → chat → unload.

### 14.9 npm packaging (`@h4zel/pumice`)

The web UI ships as a global npm command. The native binary cannot be built by `node-gyp` (the
tokenizers staticlib is MinGW-only), so the package follows the esbuild/better-sqlite3 pattern:
a **pure-JS main package** plus a **platform companion package** pulled in through
`optionalDependencies`.

```
@h4zel/pumice              # main (JS only): cli.js, webui/dist, webui/dist-server
  optionalDependencies:
    @h4zel/pumice-win32-x64  # prebuilt: pumice.node + shader/*.spv (~14.7 MB unpacked)
```

- **`cli.js`** parses `--port` / `--models` / `--no-open`, resolves the platform package with
  `require.resolve("@h4zel/pumice-win32-x64/package.json")` (clear error if the platform is
  unsupported or the optional dep failed to install), exports `PUMICE_RUNTIME`,
  `PUMICE_CWD`, `PUMICE_MODELS`, `PUMICE_PRUNED_VOCAB_DIR`, `PUMICE_EXPORT_DIR`,
  `PUMICE_QUANT_TMP`, `PUMICE_OPEN` and `PORT`, then imports the bundled server. `PUMICE_MODELS`
  is only set when `--models` is passed (there is no default models folder); pruned-vocab is read from
  `./pruned-vocab` **relative to the directory where the command was run** (not the install dir); the
  browser is opened on listen unless `--no-open`.
- **Server bundling.** `webui/package.json`'s `build:server` runs esbuild
  (`--bundle --platform=node --format=esm --packages=external`) to produce
  `webui/dist-server/index.mjs`; the only runtime dependency is `express`, so the installed CLI needs
  no TypeScript toolchain.
- **`pack.mjs`** (the `prepack` hook) verifies the main and platform versions match and the engine
  artifacts exist, runs `build:all`, then stages `bin/pumice.node` + `bin/shader/` into
  `platform/win32-x64/`. `files` whitelists in both `package.json`s (plus `webui/.npmignore`, which
  overrides the nested `dist/` gitignore rule) control the tarball contents.
- **`release.mjs <major|minor|patch|x.y.z>`** bumps both versions in lockstep (and the
  `optionalDependencies` pin), runs `pack.mjs`, then publishes the platform package **before** the
  main package so the pinned dependency already exists.

```bash
node release.mjs patch               # bump + build + publish both packages
node release.mjs patch --otp 123456  # when the npm account has 2FA enabled
```

Publishing uses the `@h4zel` scope (the npm username). If the account enforces 2FA, pass a fresh
one-time password with `--otp` (or set `NPM_OTP`), or configure an npm **Automation** access token
in `.npmrc` to bypass the OTP prompt.

Install and run:

```bash
npm install -g @h4zel/pumice
cd D:\models\my-project       # contains model/ and pruned-vocab/
pumice                    # http://127.0.0.1:8787 opens automatically
```

Requirements: Windows x64, a Vulkan-capable GPU with a current driver, and Node ≥ 18.

---

## 15. Perplexity evaluation

The addon can score a text file with the model it has loaded and report **perplexity** — the
exponential of the mean negative log-likelihood per token. It runs entirely through the same decode
path as generation (no separate forward implementation), so a validated model gives a validated ppl.

### 15.1 Overlap sliding window

A ppl run is a sequence of **windows** of `prefill + decode` tokens each, with `prefill` tokens of
overlap. Window `k` (0-based, `P = prefill`, `D = decode`) does:

1. reset all state and **prefill** `ids[kP .. kP+P)`;
2. **decode** the next `D` true tokens one at a time (teacher forcing), reading the log-probability
   the model assigns to each one;
3. advance to `s = (k+1)P` for the next window.

So the context grows to `P + D` inside a window, then resets and slides forward by `P`. The scored
tokens are contiguous and counted exactly once:

| Window | Prefill (context) | Scored |
|---|---|---|
| `k = 0` | `ids[0 .. P)` | `ids[P .. P+D]` (`D+1` tokens: the final op scores `ids[P]`) |
| `k ≥ 1` | `ids[kP .. (k+1)P)` | `ids[kP+P+1 .. kP+P+D+1]` (`D` tokens) |

The overlap means the tokens a window decodes become the *next* window's prefill context, so no
token is ever scored without `P` tokens of history behind it, and the first `P` tokens of the file
are pure context (never scored). `generateScore` (`src/generate.c`) is the driver:

```c
for (int k = 0; k < chunks; k++) {
    resetGenerator(g);
    runPrefill(g, ids + k * prefillN, prefillN);
    tokenIds[1] = ids[k * prefillN + prefillN];
    if (k == 0) score final op;          // ids[P] from the prefill's last hidden state
    tokenIds[0] = ids[k * prefillN + prefillN];
    while (j < limit) { feed DECODE_GROUP true tokens; read DECODE_GROUP losses; }
}
```

`resetGenerator` is what makes windows independent — and it is the mechanism gotcha 50 broke.

### 15.2 Scoring mode on the GPU

Scoring reuses the decode group (4 tokens per submission) and adds a third `ArgMax-Reduce` mode:

- `generatorSetScoring(g, 1)` sets `g->sampling = 2` and recompiles the decode/final ops; in this
  mode `buildLmHead` uses `LMHead-GEMV-FP16.spv` (raw logits into `st->logits`) instead of the greedy
  argmax kernel.
- `g->skipFinal = 1` makes `runPrefill` skip its built-in final op (it still updates `lastRow` and
  the position). `generateScore` runs the final op itself, **only for `k == 0`** — for later windows
  that token was already scored as the previous window's last decode token.
- `ArgMax-Reduce.comp` **mode 2** computes, per decode pass, the exact cross-entropy of the target
  token in nats:

  ```glsl
  result[passIdx] = floatBitsToUint(log(Zall) + gMax - ltarget);   // logsumexp - logit[target]
  tokenIds[0] = target;                                            // teacher forcing
  if (doIncrement == 1u) position[0] = position[0] + 1u;
  ```

  `gMax` is the workgroup max of the logits, `Zall = Σ exp(l - gMax)` (so `log(Zall) + gMax` is the
  log-sum-exp), and `ltarget` is the target's logit, found by whichever thread's vec4 block contains
  the id. The write slot is `passIdx`, which is how 4 tokens per submission land in `result[0..3]`.
  In this mode the sampler's temperature/penalty path is not entered at all.

`ppl = exp(lossSum / count)` is computed both in the addon's progress callback and in its final
result; the server streams `{progress}` per window and a final `{done}`.

### 15.3 API chain

`engineScore` (`include/engine.h`) → `score` (`src/addon.c`, async work + threadsafe progress
callback, same single-generation guard as `generate`) → `EngineManager.score`
(`webui/server/engine.ts`) → `POST /api/score` (SSE) → `ScoreTab.tsx`.

The server tokenizes the whole posted file with the loaded tokenizer, computes
`chunks = min(floor(pct·len/P), maxChunks)` from the test-size percentage, and validates
`P + D ≤ maxCtx`. The addon rejects ids outside the model vocab (`token N at index I is outside the
vocab (V)`) — a tokenizer/vocab mismatch would otherwise silently score `logit[target] = 0` and
report a near-vocabulary-size ppl instead of failing.

### 15.4 Reproducibility

A scoring run must be **order-independent and repeatable**: the same window scored twice, or after
any other scoring call, must give the identical NLL. `test` for this is simply running the same
config several times and comparing the reported mean NLL (three identical `4096/9` runs give
`1.839663`). This is what caught gotcha 50 — the first forward after a load was correct and every
subsequent one inherited stale recurrent state, so the reported ppl depended on what had been scored
before.

### 15.5 Validation against HuggingFace

`tools/ppl_ref.py` (single span) and `tools/ppl_ref_windows.py` (overlap windows) compute the same
quantity with HF transformers in `tools/pruner/.venv`, softmax over the full vocab; they are the
ground truth for the engine. `tools/ppl_ref.py` also takes a dtype (`bf16`/`fp32`) to bound the
reference's own precision. The dataset is `data/wikitext-2_test.txt` (1.1 MB, 270,657 tokens with the
full Qwen3.5 tokenizer); "test size 2%" means `chunks = 1`, i.e. the first window.

2B safetensors, **unpruned** (full 248,320 vocab), `maxCtx 8192`, prefill/decode 4096:

| Config | Engine | HF bf16 |
|---|---|---|
| all-FP16, 2% (`chunks 1`) | ppl **12.698**, mean NLL 2.5415 | ppl **12.705**, mean NLL 2.5420 |
| all-FP16, first 10 tokens | mean NLL 1.8397 | 1.851 |
| all-INT4, 2% | ppl **14.937** | — (quantization cost, not error) |

Overlap windows, all-FP16, `P=512 D=512 chunks=3` (1,537 scored tokens):

| | count | mean NLL | ppl |
|---|---|---|---|
| Engine | 1537 | 2.301545 | **9.9896** |
| HF bf16 | 1537 | 2.302019 | **9.9943** |

That is a 0.02% match on a multi-window run, i.e. the overlap design (including the per-window state
reset and the `k == 0` final-op token) reproduces full-context HF scoring. The same holds along the
decode axis at `P=4096`:

| `P=4096` | Engine | HF |
|---|---|---|
| `D=64` | 6.0438 | 6.0241 (bf16) |
| `D=512` | 6.6374 | 6.6489 (bf16) / 6.6370 (fp32) |
| `D=1024` | 7.0668 | — |

The residual spread is ≤0.3% and consistent with fp16 attention/KV accumulation rather than the
scoring formula. `tools/ppl_ref.py` on the same span is the check to run when a number looks off.

Other models measured with the same 2% config (`P=4096 D=4096`): mixed-quant 2B safetensors 13.499,
BF16 GGUF 13.500, pruned HQM (102,400 vocab) 14.004, unpruned HQM 14.937.

### 15.6 Using it

In the UI: **Eval** tab → choose a `.txt`, set prefill/decode (default 4096/4096), set test size
(default 10%; 2% ≈ one window for wikitext-2), **Run**. From code:

```js
const ids = vk.tokenize(engine, text, false);
const r = await vk.score(engine, ids, { prefill: 4096, decode: 4096, chunks: 1 }, (p) => {});
// r = { loss, count, ppl }
```

Requirements: `prefill + decode ≤ maxCtx`, and at least `prefill + decode + 1` tokens of text.
---

## 16. KV Cache Infrastructure (VRAM / RAM / Disk)

A block-granular prefix cache lets a conversation resume instead of re-prefilling the whole history on
every request (`engineGenerate` used to call `resetGenerator` unconditionally). Design notes live in
`this-is-custom-from-inherited-giraffe.md`. The implemented tiers, chain hashing and snapshots are below.

### 16.1 Block format and chain hashing

- **Block = 16 tokens** of KV across all full-attention layers. Full-attention KV is append-only, so a
  complete block never changes (only the tail partial block mutates during decode).
- `include/kvcache.h` + `src/kvcache.c` build a per-slot layout from the runtime dims: for every
  full-attention layer, `K` (`kvRows x 16`), `V` (16 x `kvRows`), plus fp32 `kScale/kZero/vScale/vZero`
  (`kvHeads x 16`) when the layer is INT8/Q4. `slotBytes` is the sum (2B: 133,120 bytes).
- **Chain hash** (XXH3-64, `xxhash/xxhash.h` vendored via `src/xxhash.c`):
  `h[b] = XXH3_64(t[16b..16b+16], seed = h[b-1])`, `h[-1] = XXH3_64(modelFingerprint)`. A stored block can
  only match if every ancestor matched, so the chain is the prefix-validity proof.
- **Model fingerprint** = name + vocab + layer count + K + kvHeads + headDim + per-layer attn quant,
  hashed into the store directory name (`kvstore/<name>-<16hex>/`). A to B to A restores A's pool.

### 16.2 Tiers and LFRU

- **VRAM** -- the live contiguous KV cache (unchanged; sized at load from `maxCtx`). Not an eviction tier.
- **RAM** -- one host-visible `MEMORY_RAM` pool buffer (`kvPool`, `kvRamBudget` bytes), slot-indexed.
  This is the authority tier: every resident block has a copy here.
- **Disk** -- `mirror.bin` is a raw image of the RAM pool (slot `i` at `i x slotBytes`); `cold.bin` is an
  append-only log of blocks demoted from RAM (each record `{key, bytes, crc32, flags}` + slot bytes);
  `index.bin` persists every entry (`key`, `child`, `coldOffset`, `slot`, `freq`, `lastUse`, `blockIndex`,
  `crc`); `snapshots.bin` holds the fp16 GDN anchors. On open the pool is bulk-loaded from `mirror.bin` and
  the index/cold tiers rebuilt, so a restart is warm. Each RAM slot carries a **CRC32** (gotcha 61) that is
  validated lazily when the block is matched, so a torn `mirror.bin` slot falls back to cold or prefill
  instead of restoring corrupt KV.
- **In-memory index**: one open-addressing table `key -> entry` plus an entry array. Each entry tracks its
  RAM slot and/or cold offset, its chain `child` hash, `freq`, `lastUse` and a `pin` count.
- **LFRU** (RAM and disk): `freq` is incremented on every match/insert and halved every 64 inserts
  (aging); the victim is the lowest `freq`, tie-broken by oldest `lastUse`. **Chain-leaf preference**: a
  block whose `child` is absent or non-resident is a leaf and is evicted first, so dead conversations
  collapse tail-ward instead of orphaning live suffixes. Pinned blocks (part of an in-flight restore) are
  never victims.
- **Demotion** (RAM over budget): the victim's slot is appended to `cold.bin` (CRC'd) and its slot freed.
  **Deletion** (cold over `kvDiskBudget`): the coldest cold leaf is tombstoned and removed from the index
  -- the only destructive eviction. `cold.bin` is compacted (live records rewritten, offsets updated)
  once dead bytes reach live bytes.
- **Admission**: a cold block hit during restore is always streamed through the stage buffer; if
  `freq >= 2` (proven reusable) it is also copied into a RAM slot permanently. One-hit blocks are not
  admitted, so a long one-off prefix cannot flush the hot shared prefix out of RAM.
- A pool smaller than the prefix is fine: cold blocks are loaded `KV_STAGE_SLOTS` (8) at a time into the
  host-visible `kvStage` ring and unstripe'd, while RAM-resident blocks are unstripe'd directly from the
  pool. If even the stage path cannot satisfy a restore, `engineGenerate` falls back to a full prefill.

### 16.3 GDN snapshots

Every model in the family has gated delta-net layers, whose recurrent state cannot be reconstructed
from KV alone -- resuming at position R needs the state at R. Snapshots are therefore required for any
restore, not just for the 35B.

- Taken **during prefill**, at every 1024-token boundary **and at the last 16-token boundary of the
  prompt**. `runPrefill` splits its chunk at whichever of those comes next when a boundary hook is
  installed, so the state is captured at an exact token position. The state after P tokens is keyed by
  `h[P/16 - 1]`. The final boundary matters for short prompts: a 600-token conversation snapshots at 592
  instead of never, so turn 2 resumes at 592 and re-prefills only the remainder. Splitting changes the
  final chunk's size only; per-token GEMM results are chunk-size independent, so the KV is unchanged
  (verified: warm == cold in the tests). When the prompt length is an exact multiple of 16, a second
  snapshot is taken one block earlier (`promptLen - 16`), because `pos == promptLen` cannot be used by a
  same-length replay (gotcha 62).
- Stored **fp16** for `stateS` and fp32 for `convHist` in a fixed-capacity **LRU table** (`KV_SNAP_MAX` =
  16 entries, lazily allocated so only used entries cost RAM). A store is keyed by `(pos, key)`; if the
  table is full the least-recently-used entry is replaced, and a hit touches `lastUse`. All entries are
  persisted to `snapshots.bin` (`{magic, count, gdnBytes}` + per-record `{pos, lastUse, key}` + data) and
  reloaded on open, so a restart keeps every conversation's anchor, not just the latest. Copy-out uses
  `generatorReadGdn` / `generatorWriteGdn` (`readBuffer` + `writeStaging` + `createTransferAndCopy`); the
  state buffers keep their staging alive so `resetGenerator` can re-zero them (gotcha 50).
- **Resume point** `R` = largest snapshot position `<= R_kv` (longest matching block run) whose key
  matches, capped below the new prompt length. The suffix `[R, count)` is then re-prefilled: attention
  rewrites those KV slots deterministically and the window rebuilds GDN state, so the fallback is an
  idempotent overwrite.
- **Only prefill-written state is cached** (gotcha 57): `engineFlush` admits blocks `[0, promptLen/16)`
  and snapshots only during prefill. Generated tokens are decode-written, and the decode and prefill
  kernels differ in accumulation order, so a decode-written block restored into a prefill path is *not*
  bit-identical to the same prefix computed by prefill. Caching the reply therefore broke warm == cold
  (a shifted continuation); the reply is now always re-prefilled and only the prompt is cached.
- **Multiple conversations coexist.** The LRU table replaced a fixed 4-slot ring that let interleaved
  conversations evict each other: with three 2000-token conversations (6 snapshots) a 4-slot store left
  none of them resumable, while the 16-entry table resumes all three (and across a restart). Snapshots
  live in their own table rather than the block table because a snapshot at position `16(b+1)` shares
  block `b`'s hash. The capacity is compile-time (`KV_SNAP_MAX`); worst-case RAM is `KV_SNAP_MAX x
  gdnBytes` (2B: 160 MB, 35B: 480 MB) but only for that many distinct live snapshots.

### 16.4 Restore and flush pipeline

`engineGenerate` (src/engine.c):

1. copy the prompt into `sessionIds`, chain-hash it, and plan `R` + the matched block slots + snapshot
   (matching entries are LFRU-touched and pinned);
2. `resetGenerator` (clears GDN + penalty history, matching the cold path exactly); unstripe the
   RAM-resident blocks from the pool and the cold blocks from the stage ring (`generatorKvRestoreCold`),
   admitting `freq >= 2` cold blocks into RAM; `generatorWriteGdn` (snapshot -> state buffers),
   `stateSetPosition(R)`, `g->nextPos = R`; unpin. On failure fall back to `R = 0`;
3. install the boundary hook and call `generateTokens(prompt + R, count - R, ...)`; the hook copies GDN
   state into the ring at each 1024 boundary and at the prompt's final 16-boundary;
4. after generation (EOS, stop, or limit) `engineFlush(e, promptLen)` admits the complete prompt blocks
   that are not already RAM-resident (`kvBlockResident`, so cold-only blocks get a RAM mirror too --
   gotcha 63) via
   `generatorKvRestripe` (live cache -> pool) in chunks of 32 (so committed entries are available as
   eviction victims while the same flush is still allocating), then `kvPersist` writes the dirty mirror
   slots, `index.bin`, `snapshots.bin`, enforces the disk budget and compacts if needed. The partial tail
   block is discarded.

`runPrefill` now starts at `g->nextPos` instead of 0 (token index `done - start`), so the suffix prefill
writes KV at absolute positions; the existing shaders already take a chunk-absolute offset.

**Scoring is isolated by design**: `engineScore` clears the hooks, and `generateScore` uses the VRAM
cache directly without reading or writing the store. The next generate request simply restores the live
session from the RAM pool (verified: a scored-then-regenerated turn is byte-identical).

### 16.5 Shaders

`shader/Utility/KV-Restripe.comp` and `KV-Unstripe.comp` are byte-generic (uint8 storage) gather/scatter
kernels over one region and a run of contiguous slots. Push constants:
`{srcStride, tokBase, rows, blocks, dstBase, dstStride, elemBytes, transpose}` (Restripe) and the inverse
(Unstripe). `transpose=0` covers K and scale/zero (`row x 16`), `transpose=1` covers V (live token-major
to pool row-major). One dispatch per layer per region per contiguous slot run; 2B flushes in ~28
dispatches per run.

### 16.6 Config and stats

- `kvRamBudget` (bytes, default 512 MB), `kvDiskBudget` (bytes, default 1 GB) and `kvStoreDir` (default
  `kvstore` under the runtime dir) are engine options; the addon exposes them on `createEngine` and the
  web UI passes them through `LoadOptions`. `PUMICE_PRUNED_VOCAB_DIR` still governs the pruner source.
- `engineInfo` returns `kvEnabled`, `kvBlocks` (commits this session), `kvEntries` (resident entries),
  `kvSnapshots` (live GDN snapshots), `kvHits`, `kvColdHits`, `kvRestores`, `kvEvictions`,
  `kvColdDeletes`, `kvUsedBytes`, `kvRamBudget`, `kvDiskBudget`, `kvColdBytes`, `kvRestoreMs`.
- The RAM pool and the stage ring are metered by `createBufferNamed` alongside the MoE expert pool, so the
  existing host-visible OOM report covers all of them. `engineInfo` also exposes the live totals as
  `hostVisibleBytes` / `deviceLocalBytes` (see gotcha 59: the counters are decremented on free). `kvOpen`
  additionally warns before allocating when `kvRamBudget` plus the host memory already in use would exceed
  the host heap.
- `PUMICE_LOG_CACHE=1` makes each restore print `kvcache: restored N tokens (M blocks, T ms)`; the
  `/v1` non-stream responses carry the same value in the `X-Vk-Cache-Cached-Tokens` header and in
  `usage.prompt_tokens_details.cached_tokens`.

### 16.7 Verification

`node test_kvcache.mjs` (2B safetensors, 3000-token wikitext prompt, greedy). A large budget exercises
the M1 path; a 10 MB pool exercises eviction, cold restore and disk budgets.

| Check | Result |
|---|---|
| turn 1 caches 188 blocks | 25.0 MB pool |
| turn 2 restore (188 blocks, R=2048) | warm output == cold, token-for-token |
| turn 3 restore | warm output == cold, token-for-token |
| restart (reload from `mirror.bin` + `snapshots.bin`) | 188 entries resident, restores correctly |
| score mid-session then regenerate | byte-identical to pre-score output |
| 10 MB pool, 3000-token prompt | 110 evictions, 14.6 MB cold, output matches |
| cold restore (109 cold blocks) | output matches the warm/cold reference |
| restart with a populated cold tier | 189 entries, 14.8 MB cold, output matches |
| 512 KB disk budget | 282 cold deletions, generation still correct |
| 600-token prompt + 96-token reply, turn 2 | resumes at 592 (prompt end), warm == cold |
| 3 interleaved 2000-token conversations | 9 snapshots, all resume; survives restart |
| model switch A (safetensors) -> B (HQM) -> A | 2 fingerprint dirs, A resumes at 784 |
| reload with a larger `maxCtx` (2048 -> 8192) | cache preserved, resumes at 1488 |
| corrupt one `mirror.bin` slot | CRC rejects it, full prefill, output == cold |

**35B-A3B** (`node test_35b.mjs`, `expertsVram=64`, `model/Qwen3.6-35B-A3B-5.8gb.hqm`, 600-token prompt):
load 68 s, turn 1 generates, turn 2 restores at 592, warm == cold, snapshots survive restart.

### 16.9 35B memory footprint (and the "VRAM weights live in RAM" illusion)

At `expertsVram=64` the 35B needs **7.25 GB device** and **12.56 GB host-visible**, which is exactly the
design: the 192 routed experts that are not in VRAM must live in the host RAM pool.

- Device: expert VRAM pool (65 experts x 1.59 MB x 40 layers) 4.1 GB, embed + lm-head fp16 2.0 GB,
  attention weights ~0.6 GB, KV cache ~0.4 GB.
- Host: **RAM expert pool 12.24 GB** (192 experts x 1.59 MB x 40), KV pool 256 MB (configurable), stage
  ring ~1.6 MB, tokenizer/misc. VRAM staging is freed at upload, so **no VRAM weight is retained in RAM**
  (the pool size scales exactly with `expertsVram`: at `expertsVram=1` the host pool alone reached 15.5 GB
  and hit the 16 GB host-visible heap).

The only levers on the host pool are the VRAM/CPU split (VRAM is 7.9 GB here, so ~64 experts is the
practical maximum), reducing the non-expert device footprint to admit more experts, or backing the pool
with the on-disk HQM (a future LFRU/mmap swap). Note that at 64 experts prefill is PCIe-bound: the
prompt is re-read against ~192 host experts per token per layer (~9 s/token at 1400 tokens, far slower
than the documented decode figure, which used a short prompt).

### 16.8 Deferred

`cache_restore` SSE phase reporting (the non-stream header and `cached_tokens` cover the value today;
streaming only exposes it in the final usage chunk); a disk/mmap-backed expert pool to shrink the host
footprint. Caching the generated reply (a decode-time snapshot) is deliberately not done -- see gotcha 57.
BPE prefix instability (a re-split can break the chain ~1 block per turn) is accepted for now; per-message
tokenization is the fix if measured.

**35B long-prompt bit-exactness caveat:** the 35B restore is functionally correct (coherent output, warm
outputs stable across runs and restarts) and matches cold bit-for-bit up to ~1100-token prompts, but at
longer prompts (e.g. 1400) warm and cold diverge after a few tokens. The cause is not the fp16 GDN
snapshot (forcing fp32 gave the identical warm output) but the GDN prefill state at a chunk boundary:
`runPrefill` splits the final chunk at the prompt's last 16-boundary, so the snapshot position is a chunk
boundary in the warm path but falls mid-chunk in the cold path. The 2B is bit-exact under the same
scheme, so the 35B's larger delta state appears to make that chunk-size rounding observable. Functional
validation of the 35B is therefore done at <=1100-token prompts; making the 35B bit-exact would need
position-independent GDN prefill (a separate investigation).

---

## 17. [OI]-Compatible API (agent harness)

The webui server also exposes an [OI]-compatible HTTP API on `/v1` so an agent harness (opencode and
friends) can drive the engine unchanged. It sits directly on `EngineManager`; the KV block cache makes a
growing conversation resume instead of re-prefilling. The base URL is printed at startup
(`pumice: api http://127.0.0.1:<port>/v1`) and by `pumice`.

### 17.1 Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/v1/models` | `{object:"list", data:[{id, object:"model", ...}]}` — models from the last scanned folder + the loaded one |
| POST | `/v1/chat/completions` | streaming and non-streaming; tools; usage |
| POST | `/v1/completions` | legacy text completion (streaming and non-streaming) |

Auth: if `PUMICE_API_KEY` is set (or `--api-key` is passed to the CLI), every `/v1` request must send
`Authorization: Bearer <key>`; otherwise it returns 401 `invalid_api_key`. Errors use the [OI] envelope
`{error:{message, type, param, code}}`.

### 17.2 Model resolution and auto-load

`model` is matched (case-insensitively) against the scanned label, the probe name, the basename, or the
path, and a trailing `provider/` prefix is stripped. If nothing is loaded (or a different model is
requested) the server probes and loads it with default options — `buildLoadOptions` in
`webui/server/load.ts` uses the probe's quant/context/prefill values, `prune` defaults on for
safetensors/gguf (override with `PUMICE_AUTOLOAD_PRUNE=0`) and KV budgets default to 512 MB RAM /
1 GB disk. This means a harness only needs a base URL; the first request warms the model. An unknown id
returns 404 `model_not_found`.

### 17.3 Chat template and tools

`webui/server/chatTemplate.ts` ports the model's own Jinja template (`tokenizer_config.json`) to
TypeScript for the text path: system/user/assistant/tool roles, `content` as a string or a `[{type,
text}]` array, assistant `reasoning_content` / `<think>` splitting, tool-call rendering
(`<function=...><parameter=...>`), consecutive `tool` messages grouped under one `<|im_start|>user` with
`<tool_response>`, and the `enable_thinking` generation prompt. Thinking is **off** by default
(`<think>\n\n</think>\n\n`); enable it with `enable_thinking: true` or
`chat_template_kwargs.enable_thinking`. `tool_choice:"none"` drops the tools block; `"required"` or a
specific `{type:"function",function:{name}}` appends a short instruction to the tools system block.

Responses are parsed by `webui/server/toolParser.ts`, which handles both the XML form
(`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`) and the JSON form. Values
are coerced back to JSON types; the result is emitted as [OI] `tool_calls` with
`finish_reason:"tool_calls"`.

Reasoning is split out always: `ReasoningSplitter` (`chatTemplate.ts`) routes a leading `<think>…</think>`
block (or, when `enable_thinking` is set, everything before `</think>`) into `reasoning_content` and the
rest into `content`, in both streaming (`delta.reasoning_content`) and non-stream
(`message.reasoning_content`) responses.

### 17.4 Request/response mapping

- `max_tokens` / `max_completion_tokens` → generator `maxNew`, clamped to `maxCtx - promptTokens`.
- `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`, `seed` →
  `sample_params`; `penalty_length` (non-standard) defaults to 64 when a repetition/presence penalty is
  set. Sampling `seed` defaults to the engine's deterministic seed.
- `stop` (string or array) is enforced server-side by a hold-back filter so a stop sequence split across
  tokens is never emitted; the engine is stopped early and `finish_reason` is `stop`.
- `finish_reason` comes from `engineFinishReason` (`src/generate.c` tracks `finishReason`: 0 = EOS or user
  stop, 1 = length/context limit); the addon resolves `generate` with `{tokens, finishReason,
  cachedTokens}` (`cachedTokens` = `engineLastResume`, the restored prefix length).
- `usage` reports `prompt_tokens` (the full prompt, as sent), `completion_tokens`, `total_tokens`, and
  `prompt_tokens_details.cached_tokens` (the KV prefix reused, clamped to `prompt_tokens`); non-stream
  responses also carry it in the `X-Vk-Cache-Cached-Tokens` header, and `stream_options.include_usage`
  adds the final usage-only chunk.
- A prompt at or over `maxCtx` is rejected before any GPU work with 400 `context_length_exceeded`.
- Content streams token-by-token even when tools are attached: `ToolTagHoldBack` (`toolParser.ts`) holds
  back only from a possible `<tool_call` opening tag, so text before the call is delivered live and the
  call itself is emitted as one `tool_calls` delta. Non-streaming returns the full `message`.
- Image/video content parts are rejected with 400 (the text stack has no vision tower).
- Overlapping requests are serialized by a bounded FIFO queue in `EngineManager` (capacity 8) instead of
  failing; the 9th concurrent request returns 429 `rate_limit_exceeded`. Auto-load is de-duplicated so
  concurrent first requests share one model load, and loads run through the same queue as generation.
  Each request carries a `RunHandle`, so a client that disconnects only stops its own run.
- `--host` / `PUMICE_HOST` sets the bind address (default `127.0.0.1`).

### 17.5 Files

`webui/server/v1.ts` (router, stop filter, [OI] serialization), `chatTemplate.ts` (template +
`ReasoningSplitter`), `toolParser.ts` (parser + `ToolTagHoldBack`), `engine.ts` (request queue +
`RunHandle`), `load.ts` (shared `resolveInput` / `buildLoadOptions`, also used by `/api/load`),
`models.ts` (last-scanned folder + entries, used by `/v1/models` and `ensureModel`), mounted in
`server/index.ts` as `app.use("/v1", createV1Router(engine))`. `test_v1.mjs` drives it end-to-end.

### 17.6 Verification

`node test_v1.mjs` (2B HQM auto-loaded, greedy):

| Check | Result |
|---|---|
| `/v1/models` | lists the scanned models (or `PUMICE_MODELS`); a `model` id that is a filesystem path is also accepted and auto-loaded |
| chat non-stream + usage | `"Hello there!"`, prompt 19 / completion 3 |
| chat stream == non-stream | byte-identical |
| `max_tokens:1` | `finish_reason:"length"`, completion_tokens 1 |
| `stop` sequence | content truncated to the prefix before the stop |
| tools | `finish_reason:"tool_calls"`, `get_weather{"city":"Paris"}` |
| tools stream | content deltas arrive before the `tool_calls` delta; shape has `index`/`id`/`function` |
| `tool_choice:"required"` | accepted, `tool_calls` returned |
| `enable_thinking: true` | `reasoning_content` (233 chars), `content` null; streams as `delta.reasoning_content` |
| thinking disabled | no `reasoning_content` |
| image part | 400 "image inputs are not supported" |
| 2 concurrent requests | both 200 (queued) |
| 20 concurrent requests | 8 ok, 12 × 429 `rate_limit_exceeded` |
| `/v1/completions` | `text_completion` object with text |
| `context_length_exceeded` | 400 with the matching code |
| unknown model | 404 `model_not_found` |
| multi-turn (long shared system) | KV restore, `cached_tokens` 800, coherent reply |
| API key | 401 without, 200 with |

`cached_tokens` depends on the shared prefix being block-aligned: a 19-token prompt whose 16th token is
the changing `<think>`/reply token correctly caches nothing (the chain hash invalidates it), which is why
the test uses a long shared system prompt.

### 17.7 opencode

opencode uses the AI SDK's `@ai-sdk/openai-compatible` provider, so it needs nothing beyond the `/v1`
surface above. Add a custom provider (README has the copy-paste block):

```json
{
  "provider": {
    "pumice": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Pumice (local)",
      "options": { "baseURL": "http://127.0.0.1:8787/v1" },
      "models": { "Qwen3.5-2B-1.6gb.hqm": { "name": "Qwen3.5 2B (local)", "limit": { "context": 32768, "output": 8192 } } }
    }
  }
}
```

The `models` keys must match `GET /v1/models` ids; `limit.context` should match the `maxCtx` the server
loads with. With `--api-key`, add `apiKey` to `options`. Tool-calling quality is the model's, not the
API's — the pruned 2B is a weak tool caller, so a coder-tuned checkpoint is recommended.
