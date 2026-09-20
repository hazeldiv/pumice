import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
const webui = path.join(root, "webui");
const tsxCli = path.join(webui, "node_modules", "tsx", "dist", "cli.mjs");
const port = Number(process.env.V1_PORT ?? 8788);
const base = `http://127.0.0.1:${port}`;
const MODEL = "Qwen3.5-2B-1.6gb.hqm";

let server = null;
let failures = 0;

function check(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
}

function startServer(env = {}) {
  const proc = spawn(process.execPath, [tsxCli, "server/index.ts"], {
    cwd: webui,
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, ...env },
  });
  const logPort = env.PORT ?? port;
  const log = fs.createWriteStream(path.join(os.tmpdir(), `vk-v1-test-${logPort}.log`));
  proc.stdout.pipe(log);
  proc.stderr.pipe(log);
  return proc;
}

async function waitForServer(url, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${url}/api/status`);
      if (res.ok) return true;
    } catch {
      await new Promise((r) => setTimeout(r, 250));
    }
  }
  return false;
}

async function postJson(pathname, body, headers = {}) {
  const res = await fetch(base + pathname, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => null);
  return { status: res.status, data };
}

async function chatStream(body) {
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: true }),
  });
  if (!res.ok || !res.body) return { status: res.status, text: "", finish: null, chunks: [] };
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  let finish = null;
  const chunks = [];
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6);
      if (payload === "[DONE]") continue;
      let ev;
      try {
        ev = JSON.parse(payload);
      } catch {
        continue;
      }
      chunks.push(ev);
      const choice = ev.choices?.[0];
      if (choice?.delta?.content) text += choice.delta.content;
      if (choice?.finish_reason) finish = choice.finish_reason;
    }
  }
  return { status: res.status, text, finish, chunks };
}

const chatBody = (content, extra = {}) => ({
  model: MODEL,
  messages: [{ role: "user", content }],
  temperature: 0,
  seed: 1,
  max_tokens: 64,
  ...extra,
});

async function main() {
  if (!(await waitForServer(base, 1000).catch(() => false))) {
    server = startServer({ PORT: String(port) });
    if (!(await waitForServer(base))) {
      check("server startup", false);
      process.exit(1);
    }
  }
  check("server reachable", true);

  const modelsRes = await fetch(`${base}/v1/models`).then((r) => r.json());
  const ids = (modelsRes.data ?? []).map((m) => m.id);
  check("/v1/models", modelsRes.object === "list" && ids.length > 0, `ids=${ids.join(",")}`);

  const baseline = await postJson("/v1/chat/completions", chatBody("Say hello in exactly three words."));
  const baseText = baseline.data?.choices?.[0]?.message?.content ?? "";
  check("chat non-stream", baseline.status === 200 && baseText.length > 0,
    `finish=${baseline.data?.choices?.[0]?.finish_reason} content=${JSON.stringify(baseText)}`);
  check("usage reported", baseline.data?.usage?.prompt_tokens > 0 && baseline.data?.usage?.completion_tokens > 0,
    `usage=${JSON.stringify(baseline.data?.usage)}`);

  const streamed = await chatStream(chatBody("Say hello in exactly three words."));
  check("chat stream", streamed.status === 200 && streamed.finish !== null && streamed.text.length > 0,
    `finish=${streamed.finish} content=${JSON.stringify(streamed.text)}`);
  check("stream == non-stream", streamed.text === baseText,
    `stream=${JSON.stringify(streamed.text)} baseline=${JSON.stringify(baseText)}`);

  const lengthBody = await postJson("/v1/chat/completions", chatBody("Write a long story.", { max_tokens: 1 }));
  check("max_tokens -> length", lengthBody.data?.choices?.[0]?.finish_reason === "length" &&
    lengthBody.data?.usage?.completion_tokens === 1,
    `finish=${lengthBody.data?.choices?.[0]?.finish_reason} completion=${lengthBody.data?.usage?.completion_tokens}`);

  if (baseText.length > 10) {
    const stopText = baseText.slice(Math.floor(baseText.length / 2), Math.floor(baseText.length / 2) + 4);
    const stopped = await postJson("/v1/chat/completions", chatBody("Say hello in exactly three words.", { stop: [stopText] }));
    const content = stopped.data?.choices?.[0]?.message?.content ?? "";
    const expected = baseText.slice(0, baseText.indexOf(stopText));
    check("stop sequence truncates", content === expected && stopped.data?.choices?.[0]?.finish_reason === "stop",
      `content=${JSON.stringify(content)} expected=${JSON.stringify(expected)}`);
  } else {
    check("stop sequence truncates", false, "baseline too short");
  }

  const tools = [{
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the current weather for a city",
      parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    },
  }];
  const toolRes = await postJson("/v1/chat/completions", {
    model: MODEL,
    messages: [{ role: "user", content: "What is the weather in Paris? You must call get_weather." }],
    tools,
    temperature: 0,
    seed: 1,
    max_tokens: 256,
  });
  const choice = toolRes.data?.choices?.[0];
  const toolCalls = choice?.message?.tool_calls ?? [];
  const shapeOk = toolRes.status === 200 && choice && ["stop", "tool_calls", "length"].includes(choice.finish_reason) &&
    (toolCalls.length === 0 || (choice.finish_reason === "tool_calls" &&
      toolCalls.every((c) => c.type === "function" && c.function?.name && typeof c.function.arguments === "string" &&
        (() => { try { JSON.parse(c.function.arguments); return true; } catch { return false; } })())));
  check("tools response shape", shapeOk,
    `finish=${choice?.finish_reason} tool_calls=${toolCalls.length} content=${JSON.stringify(choice?.message?.content ?? "").slice(0, 80)}`);
  if (toolCalls.length) {
    check("tool call name", toolCalls[0].function.name === "get_weather",
      `name=${toolCalls[0].function.name} args=${toolCalls[0].function.arguments}`);
  }

  const toolStream = await chatStream({
    model: MODEL,
    messages: [{ role: "user", content: "What is the weather in Paris? You must call get_weather." }],
    tools,
    temperature: 0,
    seed: 1,
    max_tokens: 256,
    stream: true,
  });
  check("tools stream terminates", toolStream.status === 200 && toolStream.finish !== null,
    `finish=${toolStream.finish}`);

  const legacy = await postJson("/v1/completions", {
    model: MODEL,
    prompt: "The capital of France is",
    temperature: 0,
    seed: 1,
    max_tokens: 16,
  });
  check("/v1/completions", legacy.status === 200 &&
    (legacy.data?.choices?.[0]?.text?.length ?? 0) > 0 &&
    legacy.data?.object === "text_completion",
    `text=${JSON.stringify(legacy.data?.choices?.[0]?.text ?? "")}`);

  const statusRes = await fetch(`${base}/api/status`).then((r) => r.json());
  const maxCtx = statusRes?.engine?.maxCtx ?? 8192;
  const long = await postJson("/v1/chat/completions", chatBody("word ".repeat(maxCtx + 512), { max_tokens: 8 }));
  check("context_length_exceeded", long.status === 400 && long.data?.error?.code === "context_length_exceeded",
    `status=${long.status} code=${long.data?.error?.code} maxCtx=${maxCtx}`);

  const badModel = await postJson("/v1/chat/completions", {
    model: "does-not-exist",
    messages: [{ role: "user", content: "hi" }],
    max_tokens: 4,
  });
  check("unknown model rejected", badModel.status === 404 && badModel.data?.error?.code === "model_not_found",
    `status=${badModel.status}`);

  const system = "You are a careful assistant. " + "Always answer concisely and follow the user's instructions exactly. ".repeat(60);
  const systemMsg = { role: "system", content: system };
  const turn1 = await postJson("/v1/chat/completions", {
    model: MODEL,
    messages: [systemMsg, { role: "user", content: "Say hello in exactly three words." }],
    temperature: 0,
    seed: 1,
    max_tokens: 64,
  });
  const reply = turn1.data?.choices?.[0]?.message?.content ?? "";
  const turn2 = await postJson("/v1/chat/completions", {
    model: MODEL,
    messages: [
      systemMsg,
      { role: "user", content: "Say hello in exactly three words." },
      { role: "assistant", content: reply },
      { role: "user", content: "Now say goodbye in exactly three words." },
    ],
    temperature: 0,
    seed: 1,
    max_tokens: 64,
  });
  const kv = (await fetch(`${base}/api/status`).then((r) => r.json()))?.engine;
  const cached = turn2.data?.usage?.prompt_tokens_details?.cached_tokens ?? 0;
  check("multi-turn kv restore", turn2.status === 200 && (kv?.kvRestores ?? 0) > 0 && (kv?.kvHits ?? 0) > 0 &&
    cached > 0,
    `restores=${kv?.kvRestores} hits=${kv?.kvHits} cached=${cached} reply=${JSON.stringify(turn2.data?.choices?.[0]?.message?.content ?? "")}`);

  const hdrRes = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(chatBody("Say hello in exactly three words.")),
  });
  await hdrRes.json();
  const cacheHeader = hdrRes.headers.get("x-vk-cache-cached-tokens");
  check("cache header present", cacheHeader !== null, `x-vk-cache-cached-tokens=${cacheHeader}`);

  const authPort = port + 1;
  const authBase = `http://127.0.0.1:${authPort}`;
  const authServer = startServer({ PORT: String(authPort), VK_COMPUTE_API_KEY: "secret" });
  try {
    if (await waitForServer(authBase)) {
      const noKey = await fetch(`${authBase}/v1/models`);
      const withKey = await fetch(`${authBase}/v1/models`, { headers: { Authorization: "Bearer secret" } });
      check("api key enforced", noKey.status === 401 && withKey.status === 200,
        `noKey=${noKey.status} withKey=${withKey.status}`);
    } else {
      check("api key enforced", false, "auth server did not start");
    }
  } finally {
    authServer.kill();
  }

  console.log(failures === 0 ? "\nALL TESTS PASSED" : `\n${failures} TEST(S) FAILED`);
  if (server) server.kill();
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("TEST RUN FAILED:", e);
  if (server) server.kill();
  process.exit(1);
});
