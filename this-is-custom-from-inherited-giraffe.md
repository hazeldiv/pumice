# KV Cache Infrastructure — Tiered Block Cache (VRAM/RAM/Disk) with LFRU + GDN Snapshots

## Context

The engine resets the generator on every `engineGenerate` call (src/engine.c:380 — `resetGenerator(e->g)`), so every API request re-prefills the entire conversation. For an agent harness with a long system prompt and growing tool history this is O(n²) prefill per conversation. This plan adds a **block-granular prefix cache** with three tiers plus a recurrent-state snapshot system, so conversations resume from cache:

- **VRAM** — the fixed live-session buffer (today's contiguous KV cache, sized at load from the webui maxCtx). Not an eviction tier.
- **RAM** — host-visible block pool (same mechanism as the existing MoE expert RAM pool), the *authority* tier: every VRAM-resident block is mirrored here. Budget from webui/config.
- **Disk** — persistence in two regions: a **mirror image of the RAM pool** (survives engine restarts; size = RAM budget) plus a **cold archive** for blocks demoted from RAM (own budget). Directories named per model fingerprint, so switching models and back preserves everything.

Eviction (RAM + disk only) is **LFRU** (freq + aging, chain-leaf preference). Over-context requests are rejected with an [OI]-style `context_length_exceeded` before any GPU work. The /v1 API + session layer (separate plan) sits on top.

## Engine ground truth (facts the design relies on)

- **KV layout** (docs §4.4): K transposed `kCache[row * maxCtx + token]` (row = kvh·headDim + dim); V token-major `vCache[token * KV_TOTAL_ROWS + row]`. fp16 for FP16 layers; uint8 + per-(kv-head, token) scale/zero at stride `kvh * maxCtx + token` for INT8/Q4.
- **Position is baked in**: RoPE applied at write time with absolute position → a block's KV is only valid as a contiguous run from position 0. The chain hash enforces this.
- **Cache is append-only**: a complete block's KV never changes; only the tail partial block (during decode) mutates → no write-back protocol, eviction never loses valid data, prefill over restored blocks is an idempotent overwrite (deterministic given the prefix).
- **Attention reads the whole context every decode step** → the live session must be entirely in VRAM; RAM/disk are storage tiers, never decode-time spill.
- **Per-model numbers (fp16 KV):**
  | model | kvRows | full-attn layers | KV bytes/token | KV block (16 tok) |
  |---|---|---|---|---|
  | Qwen3.5-9B | 1024 | 32 | 128 KB | **2 MB** |
  | Qwen3.5-2B | 512 | 24 | 48 KB | **768 KB** |
  | Qwen3.6-35B-A3B | 512 | 10 of 40 | 20 KB | **320 KB** |
- **GDN recurrent state** (Qwen3.6-35B-A3B): 30 of 40 layers are gated delta-net, state `[nV][dim][dim]` = 32×128×128 fp32 ≈ 2 MB/layer → **60 MB fp32 / 30 MB fp16 per full snapshot**.
- **RAM heap contention**: 35B MoE already holds 224 experts in host-visible RAM (docs §4.7). KV RAM budget must be sized with the expert pool; engine warns at load if `kvRamBudget + expertPool > heap`.
- **maxCtx is a load-time allocation** (`--max-ctx` sizes KV/attScores; no runtime resize). Raising the context limit = reload.

## Block format & chain hashing

**Block = 16 tokens** of KV across all full-attention layers (K, V, quantized scale/zero).

Formula — prompt token ids `t[0..P)`, block `b` covers `t[16b .. 16b+16)`:

```
seed0 = XXH3_64(modelId)            // model name + quant fingerprint + vocab + dims
h[-1] = seed0
h[b]  = XXH3_64(t[16b .. 16b+16], seed = h[b-1])
```

A stored block matches only if its hash equals the prompt's `h[b]` computed in this chain — impossible unless every ancestor matched (barring 64-bit collision). The chain is the prefix-validity proof; no separate check needed.

**Search**: compute `h[0..P/16)`, walk `b = 0,1,2,...` against the index, stop at first miss → `R_kv = 16 × consecutive hits`, capped at `min(R_kv, maxCtx)`.

**Known caveat**: BPE tokenization is not prefix-stable — appending a message can re-split the last tokens of the previous text, breaking the chain a block or two early (~lose one block per turn worst case). Fix if measured in practice: per-message tokenization (concatenate ids per chat message; Qwen template boundaries at `<|im_end|>` permit it). Deferred, not built upfront.

**Restriped storage layout (RAM/disk)**: blocks stored block-contiguous — `K[block][row][16]`, V as-is, scale/zero contiguous. Two small shaders convert to/from the live strided layout:
- `KV-Unstripe.comp` — staging → live cache (restore path; RAM→VRAM DMA is linear).
- `KV-Restripe.comp` — live cache → staging (flush path).
One per quant family for the scale/zero variants. Without these, every block move is thousands of tiny strided copy regions.

## GDN snapshot system (1024-token blocks, fp16)

Per-16-token snapshots would be GB-scale and per-prompt-only snapshots render the cache invalid on a 90% match — so snapshots use **1024-token granularity** (= 64 KV blocks), stored **fp16** (the state is gate-decayed every step, so snapshot quantization error decays; one validation run against the HF reference to confirm).

- **Keying**: a snapshot at position `P = 1024k` is keyed by `h[64k − 1]` — the KV chain hash after the 64th block. Stored in the **same index table**, entry type `snapshot`, so fingerprint scoping and chain-validity carry over free.
- **Search** (combined):
  ```
  R_kv  = longest contiguous 16-block run from h[0], capped at maxCtx
  R_gdn = largest snapshot position ≤ R_kv that exists (1024-granular)
  R     = min(R_kv, R_gdn)          // effective resume point
  ```
  Example: KV hits through 8192 but no snapshot at 8192 → `R_gdn = 7168` → resume at 7168.
- **Fallback re-prefill is an idempotent overwrite**: resuming at 7168 re-prefills tokens 7168..8192 (≤ one prefill chunk). Attention rewrites those KV slots with bit-identical values (deterministic given the restored prefix) — no invalidation. The window simultaneously rebuilds the GDN state via `GatedDeltaNet-GEMM` + `Conv-SiLU`, landing on a fresh state at 8192; decode proceeds. Mismatch cost bounded at ≤ 1024 tokens of prefill.
- **Snapshot ring (RAM, per session)**: latest snapshot mandatory (decode continues from it) + previous 1–2 positions for the fallback case; older positions dropped. If no snapshot matches, `R_gdn = 0` → prefill from scratch (today's behavior).
- **Persistence**: only the **end-of-turn snapshot** is written to disk (fp16, in the model's kvstore dir next to mirror/cold) so a restart resumes the last turn exactly; intermediate positions are RAM-only and die with the process. End-of-turn snapshot is always admitted to RAM on a disk hit (it's the resume anchor); intermediate ones skip admission.
- **Cap**: same `min(..., maxCtx)` rule — snapshots beyond maxCtx can't restore into a smaller VRAM buffer.
- **Decode side: zero changes** — decoded tokens already update GDN state incrementally; snapshots are pure read-side copies taken at flush time (one extra memcpy set per turn end).

## Tier model & residency rules

One **in-memory index** in C (per model fingerprint) is the single source of truth across all tiers:

```
key:    u64 chain-hash
value:  { entryType: block|snapshot, tierBitmap, ramSlot | coldOffset, freq, lastUseTick,
          blockIndex, parent, crc, dirtyInMirror }
```

- **Inclusive RAM**: a block in VRAM always also has a RAM copy. VRAM drop is therefore free (no write-back, no cascade) and the live session is always recoverable from RAM alone.
- **Disk = mirror + cold**:
  - `mirror.bin` — direct image of the RAM pool: slot *i* at offset `i × slotSize`. Every RAM-resident block has a mirror copy (possibly stale by one turn).
  - `cold.bin` — append-only log of blocks demoted from RAM (LFRU), each record CRC'd with its chain hash.
- Multi-tier residency is normal (tierBitmap), not a special case. Index + tier state are scoped per model fingerprint; the disk directory is named by it.

## Turn-end pipeline (normal EOS, stop request, OR context-limit error — identical)

1. **VRAM → RAM**: turn's new complete blocks memcpy'd (after `KV-Restripe` into staging) into free RAM slots. Pinned while session live. Partial tail block discarded.
2. **GDN snapshot**: copy live GDN state buffers out (fp16) → session ring + end-of-turn snapshot queued for disk write.
3. **RAM → disk mirror** (background thread, never touches Vulkan): dirty RAM slots since last flush written to `mirror.bin` at slot offsets; `dirtyInMirror` cleared per slot.
4. **RAM budget check**: over `kvRamBudget` → LFRU-demote coldest leaves to `cold.bin`, free slots, mark mirror slots free.

A context-limit stop flushes exactly the same way — complete blocks before the stop are valid prefix material if the model is later reloaded with a larger maxCtx (the only way VRAM grows).

## Restore orchestration (prompt arrives)

1. Tokenize → chain hashes → compute `R_kv`, `R_gdn`, `R` (above).
2. Partition matched blocks by tier; build copy plan; LFRU-touch everything matched (freq++, lastUse=tick).
3. **All prefix blocks must be VRAM-resident before suffix prefill starts** (suffix attention attends to the whole prefix). Hard dependency.
   - In VRAM: nothing to do (inclusive mirror guarantees RAM has it too).
   - In RAM: batched `vkCmdCopyBuffer` host-visible → device-local + `KV-Unstripe`, one submit + fence (~0.2 ms per 2 MB block; 4k-token 9B prefix ≈ 50–100 ms).
   - In cold/disk: `ReadFile` into mapped host-visible memory (transient staging), then the RAM path; double-buffered (read block i+1 while copying block i).
   - **Disk-hit admission**: `freq ≥ 2` (proven reusable, e.g. shared system prefix) → admit permanently into RAM; one-hit blocks stream through a 64–128 MB double-buffered staging ring without admission. GDN end-of-turn snapshot: always admit.
   - Match < 2 blocks: skip the machinery, plain prefill.
4. Set `nextPos = R` (restoring GDN state from the snapshot if `R = R_gdn`); prefill the suffix (idempotent rewrite of any skipped window); new blocks born dirty-in-VRAM and pinned.
5. Report restore as a `cache_restore` phase (SSE progress events from the /v1 layer later).

## Context-limit error behavior

- **Request-time check** before any GPU work: `restoreTokens + suffixTokens + maxNew > maxCtx` → HTTP 400 `context_length_exceeded` ([OI]-compatible; harnesses already handle it). No silent truncation, no sliding window — the harness decides.
- **Unbounded maxNew**: overflow can happen mid-generation. Generator stops at `nextPos == maxCtx`, the normal turn-end flush runs (complete blocks → RAM → mirror; GDN end-of-turn snapshot saved; partial tail discarded), and the same error is returned.
- Blocks/snapshots accumulated before the stop stay indexed — a reload with larger maxCtx restores them.

## LFRU policy (RAM + disk only)

One global ranking:
- `freq` incremented on every use (match counts even without restore); `lastUse` tick on every touch.
- **Aging**: `freq >>= 1` on every N-th insert/restore — old-hot blocks can't pin forever.
- Victim: min `freq`, tie → oldest `lastUse`.
- **Chain-leaf preference**: evicting a mid-chain block orphans all later blocks (they can never match again). Prefer blocks with no resident children — dead conversations collapse tail-ward contiguously.
- **RAM budget** (`kvRamBudget`, e.g. 4 GB): overflow → demote coldest leaves to `cold.bin`. Never destructive.
- **Disk cold budget** (`kvDiskBudget`, e.g. 10 GB): overflow → delete coldest cold-region blocks (tombstone + periodic compaction). The only destructive eviction.
- **VRAM not governed**: fixed live buffer; pin flag only guards model-switch races.
- Persist `freq`/`lastUse` in `index.json` so a restart doesn't cold-reset the ranking.

## Disk store layout & naming

```
kvstore/<modelId>/
    index.json      # chain-hash → {entryType, tiers, ramSlot, coldOffset, freq, lastUse, blockIndex, crc}
    mirror.bin      # size = kvRamBudget; direct image of the RAM pool (slot i at offset i×slotSize)
    cold.bin        # append-only cold archive, CRC'd records
    snapshot.bin    # fp16 end-of-turn GDN snapshot (+ position header)
```

- `modelId` = model name + quant fingerprint + vocab + dims (e.g. `Qwen3.5-9B_q4_1_256_v102400`) — stable across restarts and switches. A→B→A restores A's pool from `A/mirror.bin`.
- **Restart sequence** (`engineOpen` with matching modelId): read `index.json` → allocate RAM pool → bulk sequential read `mirror.bin` into it → validate per-block CRC (eager or lazy) → mark entries RAM-resident; `cold.bin` entries return as disk-tier; `snapshot.bin` restores the GDN resume anchor. Cache is warm immediately.
- **Disk footprint = kvRamBudget + kvDiskBudget** exactly (mirror is a full RAM image). Webui shows both.

## Config knobs (webui-controlled)

- `kvRamBudget` — RAM pool bytes (default: heap minus expert pool; load-time warning if combined over heap).
- `kvDiskBudget` — cold archive bytes.
- `maxCtx` — exists; sizes VRAM at load, bounds the request-time check and match caps.
- `kvStoreDir` — root for `kvstore/` (default under runtime dir).

## Implementation sketch

New: `include/kvcache.h`, `src/kvcache.c` (index, chain hashing, tier state machine, LFRU, mirror/cold/snapshot IO thread), `shader/Utility/KV-Unstripe*.comp` + `KV-Restripe*.comp` (+ Q4/INT8 scale/zero variants), xxhash (vendored single-header or ~100-line implementation).

Touched:
- `src/engine.c` — replace unconditional `resetGenerator` in `engineGenerate` with: restore orchestration, suffix prefill (+ optional GDN state restore), turn-end flush hook; context check before GPU work.
- `src/generate.c` — expose `nextPos` / block-boundary wrapping of `stateSetPosition`; stop-at-maxCtx mid-decode; snapshot copy-out of GDN state buffers at flush.
- `src/weights.c` / `include/state.h` — RAM pool allocation alongside expert pool; `model_state` gains block/snapshot tracking metadata.
- `src/addon.c` — cache stats + config on `createEngine`/`engineInfo` (block/snapshot hits, restore ms, tier occupancy).
- `webui/server/*` — budgets through `LoadOptions`; /v1 layer maps SSE restore phases later.

Reuse: host-visible pool pattern from the MoE expert pool (`expertPoolSplit`, docs §4.7), single-submit + fence copy pattern (`createTransferAndCopy`), shared position buffer convention, `createBufferNamed` OOM metering.

## Verification

1. **Unit**: chain-hash properties (suffix-collision impossibility, fingerprint isolation) vs CPU reference; LFRU ranking property tests (aging, leaf preference) on a simulated workload.
2. **Correctness end-to-end**: greedy generation with cache restore must match the token-for-token HF-validated baseline (docs §13) — turn 2 with restore must be byte-identical to turn 2 with full re-prefill. Test restore points in VRAM, RAM, cold tiers, and GDN-fallback (restore at 7168 of an 8192 match) — including the 35B, where the GDN state restore path is the risk.
3. **fp16 GDN snapshot validation**: snapshot/restore trajectory vs fp32-state reference on the 35B — confirm quantization error stays inside tolerance.
4. **Restart**: conversation → kill engine → restart same model → next turn restores from mirror + snapshot (log line + identical output).
5. **Error path**: exceed maxCtx with unbounded maxNew → error returned, complete blocks + snapshot flushed; reload with larger maxCtx → restore succeeds, output matches.
6. **Model switch**: A→B→A preserves A's cache; fingerprints don't cross-match.
7. **Perf**: decode tok/s unchanged (restore only touches turn boundaries); restore latency logged per tier; sampling/penalty paths unaffected.
