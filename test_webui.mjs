import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
const webui = path.join(root, "webui");
const tsxCli = path.join(webui, "node_modules", "tsx", "dist", "cli.mjs");
const port = Number(process.env.PORT ?? 8787);
const base = `http://127.0.0.1:${port}`;

let server = null;
let failures = 0;

function check(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
}

async function waitForServer(timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${base}/api/status`);
      if (res.ok) return true;
    } catch {
      await new Promise((r) => setTimeout(r, 250));
    }
  }
  return false;
}

async function post(pathname, body) {
  const res = await fetch(base + pathname, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: res.status, data: await res.json().catch(() => null) };
}

async function chat(body) {
  const res = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) return { status: res.status, text: "", done: null };
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  let done = null;
  for (;;) {
    const { value, done: streamDone } = await reader.read();
    if (streamDone) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const ev = JSON.parse(line.slice(6));
      if (ev.delta) text += ev.delta;
      if (ev.done) done = ev;
    }
  }
  return { status: res.status, text, done };
}

async function main() {
  const alreadyUp = await waitForServer(1000).catch(() => false);
  if (!alreadyUp) {
    server = spawn(process.execPath, [tsxCli, "server/index.ts"], {
      cwd: webui,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const log = fs.createWriteStream(path.join(os.tmpdir(), "vk-webui-test-server.log"));
    server.stdout.pipe(log);
    server.stderr.pipe(log);
    server.on("exit", (code, signal) => {
      console.log(`[server exited code=${code} signal=${signal}]`);
    });
    if (!(await waitForServer())) {
      check("server startup", false);
      process.exit(1);
    }
  }
  check("server reachable", true);

  const models = await fetch(`${base}/api/models`).then((r) => r.json());
  check("model scan", Array.isArray(models.models) && models.models.length > 0,
    `${models.models.length} entries`);

  for (const [name, target, kind] of [
    ["probe hqm", "model/qwen3.5-2b-pruned-1.7gb.hqm", "hqm"],
    ["probe gguf", "model/Qwen3.5-2B-BF16.gguf", "gguf"],
    ["probe safetensors", "model/Qwen3.6-35B-A3B", "safetensors"],
  ]) {
    const { data } = await post("/api/probe", { path: target });
    check(name, data?.ok === true && data?.kind === kind,
      `kind=${data?.kind} layers=${data?.layers} experts=${data?.experts} tied=${data?.tied}`);
  }

  const bad = await post("/api/probe", { path: "model/Qwen3.5-9B" });
  check("probe missing shards rejected", bad.data?.ok === false && bad.data.errors.length > 0);

  const load = await post("/api/load", {
    path: "model/qwen3.5-2b-pruned-1.7gb.hqm",
    maxCtx: 8192,
    exportModel: false,
  });
  check("load hqm", load.data?.ok === true,
    `engine=${JSON.stringify(load.data?.engine)}`);

  const status = await fetch(`${base}/api/status`).then((r) => r.json());
  check("status loaded", status.loaded === true);

  const result = await chat({
    messages: [{ role: "user", content: "Say hi in one word." }],
    system: "You are a helpful assistant.",
    thinking: false,
    maxNew: 16,
    temperature: 0,
    repPenalty: 1,
    penaltyLength: 0,
    topK: 0,
    topP: 1,
    minP: 0,
    presencePenalty: 0,
    seed: 1,
  });
  check("chat stream", result.done !== null && result.text.length > 0,
    `reply=${JSON.stringify(result.text)} tokens=${result.done?.tokens}`);

  const html = await fetch(`${base}/`).then((r) => r.text());
  check("serves webui", html.includes('id="root"'));

  const unload = await post("/api/unload", {});
  check("unload", unload.data?.ok === true);

  const stProbe = await post("/api/probe", { path: "model/Qwen3.5-2B" });
  check("probe 2B safetensors", stProbe.data?.ok === true, `layers=${stProbe.data?.layers}`);

  const stLoad = await post("/api/load", {
    path: "model/Qwen3.5-2B",
    maxCtx: 8192,
    prune: true,
    exportModel: false,
    embed: stProbe.data.embed,
    lmHead: stProbe.data.lmHead,
    layers: stProbe.data.attnQuants.map((attn, i) => ({ attn, ffn: stProbe.data.ffnQuants[i] })),
  });
  check("load 2B with UI quant + prune", stLoad.data?.ok === true,
    `engine=${JSON.stringify(stLoad.data?.engine)}`);

  const stChat = await chat({
    messages: [{ role: "user", content: "Say hi in one word." }],
    system: "You are a helpful assistant.",
    thinking: false,
    maxNew: 16,
    temperature: 0,
    repPenalty: 1,
    penaltyLength: 0,
    topK: 0,
    topP: 1,
    minP: 0,
    presencePenalty: 0,
    seed: 1,
  });
  check("chat on safetensors model", stChat.done !== null && stChat.text.length > 0,
    `reply=${JSON.stringify(stChat.text)}`);

  const stUnload = await post("/api/unload", {});
  check("unload safetensors", stUnload.data?.ok === true);

  console.log(failures === 0 ? "\nALL TESTS PASSED" : `\n${failures} TEST(S) FAILED`);
  if (server) server.kill();
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("TEST RUN FAILED:", e);
  if (server) server.kill();
  process.exit(1);
});
