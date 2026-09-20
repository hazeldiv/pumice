import { createRequire } from "node:module";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const addon = require("./bin/vk_compute.node");
const root = path.dirname(fileURLToPath(import.meta.url));
process.env.VK_PRUNED_VOCAB_DIR = path.join(root, "pruned-vocab");
process.chdir(path.join(root, "bin"));

const MB = 1024 * 1024;
const EXPERTS_VRAM = Number(process.env.EXPERT_VRAM ?? 64);
const store = fs.mkdtempSync(path.join(os.tmpdir(), "vk35b-"));
let failures = 0;

function check(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
}

function openEngine(storeDir) {
  return addon.createEngine({
    weights: path.join(root, "model", "Qwen3.6-35B-A3B-5.8gb.hqm"),
    maxCtx: 8192,
    expertsVram: EXPERTS_VRAM,
    kvStoreDir: storeDir,
    kvRamBudget: 256 * MB,
    kvDiskBudget: 256 * MB,
    exportModel: false,
  });
}

function gen(e, ids, maxNew) {
  const out = [];
  const params = {
    temperature: 0,
    maxNew,
    seed: 1,
    repPenalty: 1,
    penaltyLength: 0,
    topK: 0,
    topP: 1,
    minP: 0,
    presencePenalty: 0,
  };
  const t0 = Date.now();
  return addon
    .generate(e, Uint32Array.from(ids), params, (ev) => out.push(ev.token))
    .then((r) => ({ out, r, ms: Date.now() - t0 }));
}

function eq(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

async function main() {
  console.log(`expertsVram=${EXPERTS_VRAM}`);

  const t0 = Date.now();
  let e = await openEngine(path.join(store, "warm"));
  const info = addon.engineInfo(e);
  console.log(
    `loaded in ${((Date.now() - t0) / 1000).toFixed(0)}s  hostVisible=${(info.hostVisibleBytes / MB).toFixed(0)}MB deviceLocal=${(info.deviceLocalBytes / MB).toFixed(0)}MB`,
  );

  const text = fs.readFileSync(path.join(root, "data", "wikitext-2_test.txt"), "utf8").slice(0, 24000);
  const PROMPT_TOKENS = Number(process.env.PROMPT_TOKENS ?? 600);
  const prompt = Array.from(addon.tokenize(e, text, false)).slice(0, PROMPT_TOKENS);
  const tail = Array.from(addon.tokenize(e, "In one sentence, summarize the passage above.", false));
  console.log(`prompt=${prompt.length} tail=${tail.length}`);

  const t1 = await gen(e, prompt, 8);
  const reply = addon.decode(e, Uint32Array.from(t1.out));
  check("turn1 generated", t1.out.length > 0, `${t1.out.length} tokens in ${(t1.ms / 1000).toFixed(1)}s`);
  check("turn1 output coherent", reply.trim().length > 0, JSON.stringify(reply.slice(0, 80)));

  const prompt2 = prompt.concat(t1.out, tail);
  const t2 = await gen(e, prompt2, 8);
  const info2 = addon.engineInfo(e);
  check("turn2 restored from cache", info2.kvRestores > 0 && t2.r.cachedTokens > 0,
    `restores=${info2.kvRestores} cached=${t2.r.cachedTokens} snaps=${info2.kvSnapshots}`);
  console.log(`  warm: cached=${t2.r.cachedTokens} out=[${t2.out}] "${addon.decode(e, Uint32Array.from(t2.out)).slice(0, 60)}"`);
  addon.destroyEngine(e);

  e = await openEngine(path.join(store, "cold"));
  const t2cold = await gen(e, prompt2, 8);
  console.log(`  cold: out=[${t2cold.out}] "${addon.decode(e, Uint32Array.from(t2cold.out)).slice(0, 60)}"`);
  check("turn2 warm == cold", eq(t2.out, t2cold.out),
    `warm=[${t2.out.slice(0, 6)}] cold=[${t2cold.out.slice(0, 6)}]`);
  addon.destroyEngine(e);

  e = await openEngine(path.join(store, "warm"));
  const t2restart = await gen(e, prompt2, 8);
  console.log(`  restart: cached=${t2restart.r.cachedTokens} out=[${t2restart.out}]`);
  check("snapshots survive restart", eq(t2restart.out, t2.out),
    `cached=${t2restart.r.cachedTokens} out=[${t2restart.out.slice(0, 6)}]`);
  addon.destroyEngine(e);

  console.log(failures === 0 ? "\nALL TESTS PASSED" : `\n${failures} TEST(S) FAILED`);
  fs.rmSync(store, { recursive: true, force: true });
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error("TEST RUN FAILED:", err);
  process.exit(1);
});
