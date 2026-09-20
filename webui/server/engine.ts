import fs from "node:fs";
import { createRequire } from "node:module";
import { buildQwenPrompt, type ChatMessageInput } from "./chatTemplate";
import type { ChatMessage, ProbeInfo, QuantConfig, Sampling } from "./types";

const require = createRequire(import.meta.url);

export interface LoadOptions {
  path: string;
  quant: QuantConfig | null;
  prune: boolean;
  exportModel: boolean;
  exportDir: string;
  maxCtx: number;
  expertsVram: number;
  kvRamBudget: number;
  kvDiskBudget: number;
  kvStoreDir: string;
}

export interface ChatResult {
  tokens: number;
  elapsedMs: number;
  finishReason: "stop" | "length";
  cachedTokens: number;
}

export interface ScoreProgress {
  done: number;
  total: number;
  chunkTokens: number;
  chunkTotal: number;
  loss: number;
  count: number;
  ppl: number;
}

export interface ScoreResult {
  loss: number;
  count: number;
  ppl: number;
}

export function buildChatTemplate(messages: ChatMessage[], system: string, thinking: boolean): string {
  const msgs: ChatMessageInput[] = messages.map((m) => ({ role: m.role, content: m.content }));
  if (system) msgs.unshift({ role: "system", content: system });
  return buildQwenPrompt(msgs, [], thinking);
}

export class EngineManager {
  private vk: any;
  private engine: any = null;
  private info: ProbeInfo | null = null;
  private busy = false;

  constructor(addonPath: string, private quantTmp: string) {
    this.vk = require(addonPath);
  }

  status(): { loaded: boolean; info: ProbeInfo | null } {
    return { loaded: this.engine !== null, info: this.info };
  }

  async load(opts: LoadOptions, info: ProbeInfo): Promise<void> {
    if (this.engine) this.unload();
    let quantConfig: string | null = null;
    if (opts.quant) {
      fs.writeFileSync(this.quantTmp, JSON.stringify(opts.quant, null, 2));
      quantConfig = this.quantTmp;
    }
    const engine = await this.vk.createEngine({
      weights: opts.path,
      quantConfig,
      maxCtx: opts.maxCtx,
      prune: opts.prune,
      expertsVram: opts.expertsVram,
      exportModel: opts.exportModel,
      exportDir: opts.exportDir,
      kvRamBudget: opts.kvRamBudget,
      kvDiskBudget: opts.kvDiskBudget,
      kvStoreDir: opts.kvStoreDir,
    });
    this.engine = engine;
    this.info = info;
  }

  unload(): void {
    if (this.engine) {
      this.vk.destroyEngine(this.engine);
      this.engine = null;
    }
    this.info = null;
  }

  engineInfo(): {
    vocab: number;
    eos: number;
    maxCtx: number;
    kvEnabled: boolean;
    kvBlocks: number;
    kvEntries: number;
    kvSnapshots: number;
    kvHits: number;
    kvColdHits: number;
    kvRestores: number;
    kvEvictions: number;
    kvColdDeletes: number;
    kvUsedBytes: number;
    kvRamBudget: number;
    kvDiskBudget: number;
    kvColdBytes: number;
    kvRestoreMs: number;
  } | null {
    return this.engine ? this.vk.engineInfo(this.engine) : null;
  }

  stopGeneration(): void {
    if (this.engine) this.vk.requestStop(this.engine);
  }

  tokenize(text: string): Uint32Array {
    if (!this.engine) throw new Error("no model loaded");
    return this.vk.tokenize(this.engine, text, false);
  }

  async score(
    ids: Uint32Array,
    prefill: number,
    decode: number,
    chunks: number,
    onProgress: (p: ScoreProgress) => void,
  ): Promise<ScoreResult> {
    if (!this.engine) throw new Error("no model loaded");
    if (this.busy) throw new Error("generation already in progress");
    this.busy = true;
    try {
      return await this.vk.score(this.engine, ids, { prefill, decode, chunks }, onProgress);
    } finally {
      this.busy = false;
    }
  }

  async chat(
    messages: ChatMessage[],
    system: string,
    thinking: boolean,
    sampling: Sampling,
    maxNew: number,
    onDelta: (delta: string) => void,
  ): Promise<ChatResult> {
    const prompt = buildChatTemplate(messages, system, thinking);
    const ids = this.tokenize(prompt);
    return this.complete(ids, sampling, maxNew, onDelta);
  }

  async complete(
    ids: Uint32Array,
    sampling: Sampling,
    maxNew: number,
    onDelta: (delta: string) => void,
  ): Promise<ChatResult> {
    if (!this.engine) throw new Error("no model loaded");
    if (this.busy) throw new Error("generation already in progress");
    this.busy = true;
    const started = Date.now();
    try {
      const result = await this.vk.generate(this.engine, ids, { ...sampling, maxNew }, (ev: any) => {
        if (ev.delta) onDelta(ev.delta);
      });
      return {
        tokens: Number(result?.tokens ?? 0),
        elapsedMs: Date.now() - started,
        finishReason: result?.finishReason === "length" ? "length" : "stop",
        cachedTokens: Number(result?.cachedTokens ?? 0),
      };
    } finally {
      this.busy = false;
    }
  }
}
