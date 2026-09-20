import express from "express";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { EngineManager, createRunHandle } from "./engine";
import { buildLoadOptions, resolveInput } from "./load";
import { addonPath, distDir, modelRoot, quantTmp } from "./paths";
import { probeModel, scanModels } from "./probe";
import { createV1Router } from "./v1";
import type { ChatMessage, Sampling } from "./types";

const engine = new EngineManager(addonPath, quantTmp);
const app = express();
app.use(express.json({ limit: "64mb" }));
app.use("/v1", createV1Router(engine));

app.get("/api/status", (_req, res) => {
  res.json({ ...engine.status(), engine: engine.engineInfo() });
});

app.get("/api/models", (_req, res) => {
  res.json({ models: scanModels(modelRoot) });
});

app.post("/api/probe", (req, res) => {
  const target = resolveInput(String(req.body?.path ?? "").trim());
  if (!target) {
    res.status(400).json({ error: "path is required" });
    return;
  }
  res.json(probeModel(target));
});

app.post("/api/load", async (req, res) => {
  const target = resolveInput(String(req.body?.path ?? "").trim());
  if (!target) {
    res.status(400).json({ error: "path is required" });
    return;
  }
  const info = probeModel(target);
  if (!info.ok) {
    res.status(400).json({ error: "validation failed", info });
    return;
  }
  const body = req.body ?? {};
  const opts = buildLoadOptions(target, info, body);

  try {
    await engine.load(opts, info);
    res.json({ ok: true, info, engine: engine.engineInfo() });
  } catch (e) {
    res.status(500).json({ error: String((e as Error)?.message ?? e) });
  }
});

app.post("/api/unload", (_req, res) => {
  engine.unload();
  res.json({ ok: true });
});

app.post("/api/chat/stop", (_req, res) => {
  engine.stopGeneration();
  res.json({ ok: true });
});

app.post("/api/score", async (req, res) => {
  const body = req.body ?? {};
  if (!engine.status().loaded) {
    res.status(400).json({ error: "no model loaded" });
    return;
  }
  const text = String(body.text ?? "");
  if (!text.trim()) {
    res.status(400).json({ error: "text is required" });
    return;
  }
  const prefill = Math.max(1, Number(body.prefill ?? 4096));
  const decode = Math.max(1, Number(body.decode ?? 4096));
  const maxCtx = engine.engineInfo()?.maxCtx ?? prefill + decode;
  if (prefill + decode > maxCtx) {
    res.status(400).json({ error: `prefill + decode must be <= max ctx (${maxCtx})` });
    return;
  }

  let ids: Uint32Array;
  try {
    ids = engine.tokenize(text);
  } catch (e) {
    res.status(500).json({ error: String((e as Error)?.message ?? e) });
    return;
  }

  const maxChunks = Math.floor((ids.length - prefill - decode - 1) / prefill) + 1;
  if (maxChunks < 1) {
    res.status(400).json({ error: `text too short: need at least ${prefill + decode + 1} tokens, got ${ids.length}` });
    return;
  }
  const pct = Math.min(1, Math.max(0.001, Number(body.sizePct ?? 0.1)));
  const wanted = Math.max(1, Math.floor((pct * ids.length) / prefill));
  const chunks = Math.min(wanted, maxChunks);

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "close");
  res.flushHeaders?.();

  let closed = false;
  const scoreHandle = createRunHandle();
  res.on("close", () => {
    closed = true;
    engine.stopGeneration(scoreHandle);
  });
  res.on("error", () => { closed = true; });
  const send = (event: unknown) => {
    if (closed || res.writableEnded) return;
    try {
      res.write(`data: ${JSON.stringify(event)}\n\n`);
    } catch {
      closed = true;
    }
  };

  const scoredTokens = 1 + chunks * decode;
  send({
    start: {
      model: engine.status().info?.name ?? "model",
      fileTokens: ids.length,
      scoredTokens,
      chunks,
      prefill,
      decode,
    },
  });

  const started = Date.now();
  try {
    const result = await engine.score(ids, prefill, decode, chunks, (p) => send({ progress: p }), scoreHandle);
    send({ done: true, ...result, chunks, tokens: ids.length, elapsedMs: Date.now() - started });
  } catch (e) {
    send({ error: String((e as Error)?.message ?? e) });
  } finally {
    if (!res.writableEnded) res.end();
  }
});

app.post("/api/score/stop", (_req, res) => {
  engine.stopGeneration();
  res.json({ ok: true });
});

app.post("/api/chat", async (req, res) => {
  const body = req.body ?? {};
  const messages: ChatMessage[] = Array.isArray(body.messages)
    ? body.messages.filter((m: any) => m && (m.role === "user" || m.role === "assistant"))
    : [];
  const sampling: Sampling = {
    temperature: Number(body.temperature ?? 0),
    repPenalty: Number(body.repPenalty ?? 1),
    penaltyLength: Number(body.penaltyLength ?? 0),
    topK: Number(body.topK ?? 0),
    topP: Number(body.topP ?? 1),
    minP: Number(body.minP ?? 0),
    presencePenalty: Number(body.presencePenalty ?? 0),
    seed: Number(body.seed ?? 0),
  };
  const maxNew = Number(body.maxNew ?? 1024);

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "close");
  res.flushHeaders?.();

  let closed = false;
  const chatHandle = createRunHandle();
  res.on("close", () => {
    closed = true;
    engine.stopGeneration(chatHandle);
  });
  res.on("error", () => { closed = true; });
  const send = (event: unknown) => {
    if (closed || res.writableEnded) return;
    try {
      res.write(`data: ${JSON.stringify(event)}\n\n`);
    } catch {
      closed = true;
    }
  };

  try {
    const result = await engine.chat(
      messages,
      String(body.system ?? ""),
      Boolean(body.thinking),
      sampling,
      maxNew,
      (delta) => send({ delta }),
      chatHandle,
    );
    send({ done: true, ...result });
  } catch (e) {
    send({ error: String((e as Error)?.message ?? e) });
  } finally {
    if (!res.writableEnded) res.end();
  }
});

if (fs.existsSync(distDir)) {
  app.use(express.static(distDir));
  app.get("*", (_req, res) => res.sendFile(path.join(distDir, "index.html")));
} else {
  app.get("*", (_req, res) => res.status(503).send("webui/dist not built. Run: npm run build"));
}

const port = Number(process.env.PORT ?? 8787);
const host = process.env.PUMICE_HOST ?? "127.0.0.1";
process.on("uncaughtException", (e) => console.error("uncaught:", e));
process.on("unhandledRejection", (e) => console.error("unhandled:", e));

function openBrowser(url: string): void {
  const [cmd, args] = process.platform === "win32"
    ? ["cmd", ["/c", "start", "", url]]
    : process.platform === "darwin"
      ? ["open", [url]]
      : ["xdg-open", [url]];
  try {
    spawn(cmd, args, { detached: true, stdio: "ignore" }).unref();
  } catch {}
}

app.listen(port, host, () => {
  const url = `http://${host}:${port}`;
  console.log(`pumice webui server on ${url}`);
  console.log(`pumice [OI] api on ${url}/v1`);
  if (!fs.existsSync(addonPath)) {
    console.warn(`missing addon: ${addonPath} (run make)`);
  }
  if (process.env.PUMICE_OPEN === "1") openBrowser(url);
});
