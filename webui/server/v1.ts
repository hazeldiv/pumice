import { Router, type Request, type Response } from "express";
import crypto from "node:crypto";
import path from "node:path";
import { EngineManager } from "./engine";
import { buildLoadOptions } from "./load";
import { modelRoot } from "./paths";
import { probeModel, scanModels } from "./probe";
import { buildQwenPrompt, type ChatMessageInput, type ToolDefinition } from "./chatTemplate";
import { parseToolCalls, type ParsedToolCall } from "./toolParser";
import type { ProbeInfo, Sampling } from "./types";

const DEFAULT_MAX_TOKENS = 4096;

interface ResolvedModel {
  id: string;
  info: ProbeInfo;
}

function apiError(res: Response, status: number, message: string, code: string | null = null): void {
  res.status(status).json({ error: { message, type: "invalid_request_error", param: null, code } });
}

function num(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function newId(prefix: string): string {
  return prefix + crypto.randomBytes(12).toString("hex");
}

function normalizeStops(value: unknown): string[] {
  if (typeof value === "string") return value ? [value] : [];
  if (Array.isArray(value)) return value.filter((s) => typeof s === "string" && s.length > 0);
  return [];
}

class StopFilter {
  private buf = "";
  private stopped = false;
  private maxLen: number;

  constructor(private stops: string[]) {
    this.maxLen = stops.reduce((n, s) => Math.max(n, s.length), 0);
  }

  get isStopped(): boolean {
    return this.stopped;
  }

  push(delta: string): { text: string; stop: boolean } {
    if (this.stopped) return { text: "", stop: true };
    this.buf += delta;
    if (this.stops.length === 0) {
      const out = this.buf;
      this.buf = "";
      return { text: out, stop: false };
    }
    let hit = -1;
    for (const stop of this.stops) {
      const idx = this.buf.indexOf(stop);
      if (idx >= 0 && (hit < 0 || idx < hit)) hit = idx;
    }
    if (hit >= 0) {
      this.stopped = true;
      const out = this.buf.slice(0, hit);
      this.buf = "";
      return { text: out, stop: true };
    }
    const keep = this.maxLen - 1;
    if (this.buf.length > keep) {
      const out = this.buf.slice(0, this.buf.length - keep);
      this.buf = this.buf.slice(this.buf.length - keep);
      return { text: out, stop: false };
    }
    return { text: "", stop: false };
  }

  flush(): string {
    const out = this.buf;
    this.buf = "";
    return out;
  }
}

function matchesLoaded(info: ProbeInfo, requested: string): boolean {
  const req = requested.toLowerCase();
  const short = req.includes("/") ? req.slice(req.lastIndexOf("/") + 1) : req;
  const candidates = [
    info.name,
    path.basename(info.path),
    path.basename(path.dirname(info.path)),
  ].map((s) => s.toLowerCase());
  return candidates.includes(req) || candidates.includes(short);
}

async function ensureModel(engine: EngineManager, requested?: string): Promise<ResolvedModel | null> {
  const status = engine.status();
  if (status.loaded && status.info) {
    if (!requested || matchesLoaded(status.info, requested)) {
      return { id: status.info.name, info: status.info };
    }
  }

  const models = scanModels(modelRoot);
  const req = requested?.toLowerCase();
  const short = req?.includes("/") ? req.slice(req.lastIndexOf("/") + 1) : req;
  const target = requested
    ? models.find((m) =>
        m.label.toLowerCase() === req ||
        m.label.toLowerCase() === short ||
        m.path.toLowerCase() === req ||
        path.basename(m.path).toLowerCase() === req ||
        path.basename(m.path).toLowerCase() === short)
    : models[0];
  if (!target) return null;

  if (status.loaded && status.info && path.resolve(status.info.path) === path.resolve(target.path)) {
    return { id: status.info.name, info: status.info };
  }

  const info = probeModel(target.path);
  if (!info.ok) {
    throw new Error(`model '${target.label}' failed validation: ${info.errors.join("; ")}`);
  }
  await engine.load(buildLoadOptions(target.path, info, {}), info);
  return { id: info.name, info };
}

function buildSampling(body: any): Sampling {
  const repPenalty = num(body.repetition_penalty, 1);
  const presencePenalty = num(body.presence_penalty, 0);
  const defaultPenaltyLength = repPenalty !== 1 || presencePenalty !== 0 ? 64 : 0;
  return {
    temperature: num(body.temperature, 0),
    repPenalty,
    penaltyLength: Math.max(0, Math.floor(num(body.penalty_length, defaultPenaltyLength))),
    topK: Math.max(0, Math.floor(num(body.top_k, 0))),
    topP: num(body.top_p, 1),
    minP: num(body.min_p, 0),
    presencePenalty,
    seed: Math.max(0, Math.floor(num(body.seed, 0))),
  };
}

function buildUsage(promptTokens: number, completionTokens: number, cachedTokens: number) {
  const cached = Math.max(0, Math.min(cachedTokens, promptTokens));
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
    prompt_tokens_details: { cached_tokens: cached },
  };
}

function auth(req: Request, res: Response, next: () => void): void {
  const key = process.env.VK_COMPUTE_API_KEY;
  if (!key) {
    next();
    return;
  }
  const header = req.headers.authorization ?? "";
  if (header === `Bearer ${key}`) {
    next();
    return;
  }
  res.status(401).json({ error: { message: "invalid api key", type: "invalid_request_error", param: null, code: "invalid_api_key" } });
}

export function createV1Router(engine: EngineManager): Router {
  const router = Router();
  router.use(auth);

  router.get("/models", (_req, res) => {
    const models = scanModels(modelRoot);
    const data = models.map((m) => ({ id: m.label, object: "model", created: 0, owned_by: "vk-compute" }));
    const loaded = engine.status();
    if (loaded.loaded && loaded.info) {
      const id = loaded.info.name;
      if (!data.some((d) => d.id.toLowerCase() === id.toLowerCase())) {
        data.unshift({ id, object: "model", created: 0, owned_by: "vk-compute" });
      }
    }
    res.json({ object: "list", data });
  });

  router.post("/chat/completions", async (req, res) => {
    const body = req.body ?? {};
    const messages: ChatMessageInput[] = Array.isArray(body.messages) ? body.messages : [];
    if (messages.length === 0) {
      apiError(res, 400, "messages is required");
      return;
    }

    let resolved: ResolvedModel | null;
    try {
      resolved = await ensureModel(engine, typeof body.model === "string" ? body.model : undefined);
    } catch (e) {
      apiError(res, 400, String((e as Error)?.message ?? e));
      return;
    }
    if (!resolved) {
      apiError(res, 404, `model '${body.model ?? ""}' not found`, "model_not_found");
      return;
    }

    const tools: ToolDefinition[] = Array.isArray(body.tools) ? body.tools : [];
    const useTools = tools.length > 0 && body.tool_choice !== "none";
    const kwargs = body.chat_template_kwargs ?? {};
    const enableThinking = Boolean(body.enable_thinking ?? kwargs.enable_thinking ?? false);

    let ids: Uint32Array;
    try {
      ids = engine.tokenize(buildQwenPrompt(messages, useTools ? tools : [], enableThinking));
    } catch (e) {
      apiError(res, 400, String((e as Error)?.message ?? e));
      return;
    }

    const maxCtx = resolved.info.maxCtx ?? 32768;
    if (ids.length >= maxCtx) {
      apiError(res, 400, `prompt is ${ids.length} tokens, max context is ${maxCtx}`, "context_length_exceeded");
      return;
    }
    const requested = Math.floor(num(body.max_tokens ?? body.max_completion_tokens, DEFAULT_MAX_TOKENS));
    const maxNew = Math.max(1, Math.min(requested > 0 ? requested : DEFAULT_MAX_TOKENS, maxCtx - ids.length));

    const sampling = buildSampling(body);
    const stops = normalizeStops(body.stop);
    const streaming = Boolean(body.stream);
    const includeUsage = Boolean(body.stream_options?.include_usage);

    const id = newId("chatcmpl-");
    const created = Math.floor(Date.now() / 1000);
    const modelId = resolved.id;

    let send: (chunk: unknown) => void = () => {};
    let sentRole = false;
    let clientClosed = false;

    if (streaming) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "close");
      res.flushHeaders?.();
      res.on("close", () => {
        clientClosed = true;
        engine.stopGeneration();
      });
      send = (chunk: unknown) => {
        if (clientClosed || res.writableEnded) return;
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      };
    }

    const base = () => ({ id, object: "chat.completion.chunk", created, model: modelId });
    const ensureRole = () => {
      if (sentRole) return;
      sentRole = true;
      send({ ...base(), choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] });
    };
    const sendContent = (text: string) => {
      if (!text || !streaming) return;
      ensureRole();
      send({ ...base(), choices: [{ index: 0, delta: { content: text }, finish_reason: null }] });
    };

    const filter = new StopFilter(stops);
    let buffered = "";
    const bufferedMode = !streaming || useTools;
    let stopped = false;

    try {
      const result = await engine.complete(ids, sampling, maxNew, (delta) => {
        if (stopped || clientClosed) return;
        const step = filter.push(delta);
        if (step.stop) {
          stopped = true;
          engine.stopGeneration();
        }
        if (bufferedMode) buffered += step.text;
        else sendContent(step.text);
      });

      if (!stopped) {
        const tail = filter.flush();
        if (bufferedMode) buffered += tail;
        else sendContent(tail);
      }

      let content = buffered;
      let toolCalls: ParsedToolCall[] = [];
      if (useTools) {
        const parsed = parseToolCalls(buffered);
        content = parsed.content;
        toolCalls = parsed.toolCalls;
      }
      const finishReason = toolCalls.length > 0 ? "tool_calls" : result.finishReason;
      const usage = buildUsage(ids.length, result.tokens, result.cachedTokens);

      if (!streaming) {
        const message: Record<string, unknown> = { role: "assistant", content: content.length ? content : null };
        if (toolCalls.length) message.tool_calls = toolCalls;
        res.setHeader("X-Vk-Cache-Cached-Tokens", String(usage.prompt_tokens_details.cached_tokens));
        res.json({
          id,
          object: "chat.completion",
          created,
          model: modelId,
          choices: [{ index: 0, message, finish_reason: finishReason }],
          usage,
        });
        return;
      }

      ensureRole();
      if (content) sendContent(content);
      if (toolCalls.length) {
        send({
          ...base(),
          choices: [{
            index: 0,
            delta: {
              tool_calls: toolCalls.map((call, index) => ({
                index,
                id: call.id,
                type: "function",
                function: call.function,
              })),
            },
            finish_reason: null,
          }],
        });
      }
      send({ ...base(), choices: [{ index: 0, delta: {}, finish_reason: finishReason }] });
      if (includeUsage) send({ ...base(), choices: [], usage });
      send("[DONE]");
      res.end();
    } catch (e) {
      const message = String((e as Error)?.message ?? e);
      if (streaming) {
        send({ error: { message, type: "server_error", code: null } });
        send("[DONE]");
        if (!res.writableEnded) res.end();
      } else {
        const status = message.includes("in progress") ? 429 : 500;
        apiError(res, status, message, message.includes("in progress") ? "rate_limit_exceeded" : null);
      }
    }
  });

  router.post("/completions", async (req, res) => {
    const body = req.body ?? {};
    const promptText = Array.isArray(body.prompt) ? body.prompt[0] : body.prompt;
    if (typeof promptText !== "string") {
      apiError(res, 400, "prompt is required");
      return;
    }

    let resolved: ResolvedModel | null;
    try {
      resolved = await ensureModel(engine, typeof body.model === "string" ? body.model : undefined);
    } catch (e) {
      apiError(res, 400, String((e as Error)?.message ?? e));
      return;
    }
    if (!resolved) {
      apiError(res, 404, `model '${body.model ?? ""}' not found`, "model_not_found");
      return;
    }

    let ids: Uint32Array;
    try {
      ids = engine.tokenize(promptText);
    } catch (e) {
      apiError(res, 400, String((e as Error)?.message ?? e));
      return;
    }

    const maxCtx = resolved.info.maxCtx ?? 32768;
    if (ids.length >= maxCtx) {
      apiError(res, 400, `prompt is ${ids.length} tokens, max context is ${maxCtx}`, "context_length_exceeded");
      return;
    }
    const requested = Math.floor(num(body.max_tokens, 16));
    const maxNew = Math.max(1, Math.min(requested > 0 ? requested : 16, maxCtx - ids.length));

    const sampling = buildSampling(body);
    const stops = normalizeStops(body.stop);
    const streaming = Boolean(body.stream);
    const includeUsage = Boolean(body.stream_options?.include_usage);

    const id = newId("cmpl-");
    const created = Math.floor(Date.now() / 1000);
    const modelId = resolved.id;

    let send: (chunk: unknown) => void = () => {};
    let clientClosed = false;
    if (streaming) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "close");
      res.flushHeaders?.();
      res.on("close", () => {
        clientClosed = true;
        engine.stopGeneration();
      });
      send = (chunk: unknown) => {
        if (clientClosed || res.writableEnded) return;
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      };
    }

    const filter = new StopFilter(stops);
    let text = "";
    let stopped = false;

    try {
      const result = await engine.complete(ids, sampling, maxNew, (delta) => {
        if (stopped || clientClosed) return;
        const step = filter.push(delta);
        if (step.stop) {
          stopped = true;
          engine.stopGeneration();
        }
        if (streaming) {
          if (step.text) send({ id, object: "text_completion", created, model: modelId, choices: [{ text: step.text, index: 0, logprobs: null, finish_reason: null }] });
        } else {
          text += step.text;
        }
      });

      if (!stopped) {
        const tail = filter.flush();
        if (streaming) {
          if (tail) send({ id, object: "text_completion", created, model: modelId, choices: [{ text: tail, index: 0, logprobs: null, finish_reason: null }] });
        } else {
          text += tail;
        }
      }

      const usage = buildUsage(ids.length, result.tokens, result.cachedTokens);
      if (!streaming) {
        res.setHeader("X-Vk-Cache-Cached-Tokens", String(usage.prompt_tokens_details.cached_tokens));
        res.json({
          id,
          object: "text_completion",
          created,
          model: modelId,
          choices: [{ text, index: 0, logprobs: null, finish_reason: result.finishReason }],
          usage,
        });
        return;
      }
      send({ id, object: "text_completion", created, model: modelId, choices: [{ text: "", index: 0, logprobs: null, finish_reason: result.finishReason }] });
      if (includeUsage) send({ id, object: "text_completion", created, model: modelId, choices: [], usage });
      send("[DONE]");
      res.end();
    } catch (e) {
      const message = String((e as Error)?.message ?? e);
      if (streaming) {
        send({ error: { message, type: "server_error", code: null } });
        send("[DONE]");
        if (!res.writableEnded) res.end();
      } else {
        const status = message.includes("in progress") ? 429 : 500;
        apiError(res, status, message, message.includes("in progress") ? "rate_limit_exceeded" : null);
      }
    }
  });

  return router;
}

