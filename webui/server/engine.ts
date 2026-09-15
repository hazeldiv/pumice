import fs from "node:fs";
import { createRequire } from "node:module";
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
}

export interface ChatResult {
  tokens: number;
  elapsedMs: number;
}

export interface ScoreProgress {
  done: number;
  total: number;
  loss: number;
  count: number;
  ppl: number;
}

export interface ScoreResult {
  loss: number;
  count: number;
  ppl: number;
}

const CHAT_BOS = "<|im_start|>";
const CHAT_EOS = "<|im_end|>";

export function buildChatTemplate(messages: ChatMessage[], system: string, thinking: boolean): string {
  const parts: string[] = [];
  if (system) parts.push(`${CHAT_BOS}system\n${system}${CHAT_EOS}\n`);
  for (const msg of messages) parts.push(`${CHAT_BOS}${msg.role}\n${msg.content}${CHAT_EOS}\n`);
  parts.push(`${CHAT_BOS}assistant\n`);
  parts.push(thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
  return parts.join("");
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

  engineInfo(): { vocab: number; eos: number; maxCtx: number } | null {
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
    if (!this.engine) throw new Error("no model loaded");
    if (this.busy) throw new Error("generation already in progress");
    this.busy = true;
    const started = Date.now();
    try {
      const prompt = buildChatTemplate(messages, system, thinking);
      const ids = this.vk.tokenize(this.engine, prompt, false);
      const tokens = await this.vk.generate(this.engine, ids, { ...sampling, maxNew }, (ev: any) => {
        if (ev.delta) onDelta(ev.delta);
      });
      return { tokens, elapsedMs: Date.now() - started };
    } finally {
      this.busy = false;
    }
  }
}
