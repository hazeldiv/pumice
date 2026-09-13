import type { ChatMessage, ProbeInfo, Quant, QuantConfig, Sampling } from "../server/types";

export interface ModelEntry {
  path: string;
  label: string;
  kind: string;
}

export interface LoadPayload {
  path: string;
  maxCtx: number;
  prefillChunk: number;
  embed: Quant;
  lmHead: Quant;
  expertsVram: number;
  prune: boolean;
  exportModel: boolean;
  exportDir: string;
  layers: { attn: Quant; ffn: Quant }[];
}

async function json<T>(res: Response): Promise<T> {
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error ?? `request failed (${res.status})`);
  return data as T;
}

export async function listModels(): Promise<ModelEntry[]> {
  const res = await fetch("/api/models");
  const data = await json<{ models: ModelEntry[] }>(res);
  return data.models;
}

export async function probeModel(path: string): Promise<ProbeInfo> {
  const res = await fetch("/api/probe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  return json<ProbeInfo>(res);
}

export async function loadModel(payload: LoadPayload): Promise<{ info: ProbeInfo; engine: any }> {
  const res = await fetch("/api/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return json<{ info: ProbeInfo; engine: any }>(res);
}

export async function unloadModel(): Promise<void> {
  await fetch("/api/unload", { method: "POST" });
}

export async function getStatus(): Promise<{ loaded: boolean; info: ProbeInfo | null; engine: any }> {
  const res = await fetch("/api/status");
  return json(res);
}

export interface ChatPayload {
  messages: ChatMessage[];
  system: string;
  thinking: boolean;
  sampling: Sampling;
  maxNew: number;
}

export interface ChatDone {
  tokens: number;
  elapsedMs: number;
}

export async function chatStream(
  payload: ChatPayload,
  onDelta: (delta: string) => void,
): Promise<ChatDone> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) throw new Error(`chat failed (${res.status})`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let done: ChatDone | null = null;
  let error: string | null = null;

  for (;;) {
    const { value, done: streamDone } = await reader.read();
    if (streamDone) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const event = JSON.parse(line.slice(6));
      if (event.delta) onDelta(event.delta);
      if (event.done) done = { tokens: event.tokens, elapsedMs: event.elapsedMs };
      if (event.error) error = event.error;
    }
  }

  if (error) throw new Error(error);
  return done ?? { tokens: 0, elapsedMs: 0 };
}

export type { ChatMessage, ProbeInfo, Quant, QuantConfig, Sampling };
