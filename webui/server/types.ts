export type Quant = "fp16" | "int8" | "q4_1_32" | "q4_1_64" | "q4_1_128" | "q4_1_256";

export interface LayerRow {
  type: string;
  attn: Quant;
  ffn: Quant;
}

export interface ProbeInfo {
  kind: "safetensors" | "gguf" | "hqm";
  path: string;
  ok: boolean;
  errors: string[];
  warnings: string[];
  name: string;
  layers: number;
  layerTypes: string[];
  attnQuants: Quant[];
  ffnQuants: Quant[];
  tied: boolean;
  experts: number;
  maxCtx: number;
  maxPos: number;
  prefillChunk: number;
  embed: Quant;
  lmHead: Quant;
  firstShard: string | null;
}

export interface Sampling {
  temperature: number;
  repPenalty: number;
  penaltyLength: number;
  topK: number;
  topP: number;
  minP: number;
  presencePenalty: number;
  seed: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface QuantConfig {
  name: string;
  maxCtx: number;
  prefillChunk: number;
  embed: Quant;
  lmHead: Quant;
  expertsVram?: number;
  layers: { attn: Quant; ffn: Quant }[];
}
