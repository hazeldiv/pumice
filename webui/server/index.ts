import express from "express";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { EngineManager, type LoadOptions } from "./engine";
import { addonPath, distDir, exportRoot, launchCwd, modelRoot, quantTmp } from "./paths";
import { probeModel, scanModels } from "./probe";
import type { ChatMessage, Quant, QuantConfig, Sampling } from "./types";

const engine = new EngineManager(addonPath, quantTmp);
const app = express();
app.use(express.json({ limit: "4mb" }));

function resolveInput(target: string): string {
  if (!target) return target;
  return path.isAbsolute(target) ? target : path.resolve(launchCwd, target);
}

const QUANT_VALUES = new Set(["fp16", "int8", "int4"]);

function toQuant(value: unknown, fallback: Quant = "fp16"): Quant {
  const s = String(value ?? "");
  return (QUANT_VALUES.has(s) ? s : fallback) as Quant;
}

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
  const maxCtx = Number(body.maxCtx ?? info.maxCtx);
  const prefillChunk = Number(body.prefillChunk ?? info.prefillChunk);
  const expertsVram = Number(body.expertsVram ?? 0);
  const layers: { attn: Quant; ffn: Quant }[] = Array.isArray(body.layers)
    ? body.layers.map((l: any) => ({ attn: toQuant(l.attn), ffn: toQuant(l.ffn) }))
    : Array.from({ length: info.layers }, () => ({ attn: "fp16" as Quant, ffn: "fp16" as Quant }));

  const quant: QuantConfig | null = info.kind === "hqm" ? null : {
    name: info.name,
    maxCtx,
    prefillChunk,
    embed: toQuant(body.embed ?? info.embed),
    lmHead: toQuant(body.lmHead ?? info.lmHead),
    layers,
  };
  if (quant && info.experts > 0) quant.expertsVram = expertsVram;

  const opts: LoadOptions = {
    path: target,
    quant,
    prune: Boolean(body.prune) && info.kind !== "hqm",
    exportModel: Boolean(body.exportModel),
    exportDir: resolveInput(String(body.exportDir ?? exportRoot)),
    maxCtx,
    expertsVram: info.experts > 0 ? expertsVram : 0,
  };

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
  res.on("close", () => {
    closed = true;
    engine.stopGeneration();
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

app.listen(port, "127.0.0.1", () => {
  const url = `http://127.0.0.1:${port}`;
  console.log(`vk-compute webui server on ${url}`);
  if (!fs.existsSync(addonPath)) {
    console.warn(`missing addon: ${addonPath} (run make)`);
  }
  if (process.env.VK_COMPUTE_OPEN === "1") openBrowser(url);
});
