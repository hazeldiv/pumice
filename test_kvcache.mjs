import { createRequire } from "node:module";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const addon = require("./bin/pumice.node");
const root = path.dirname(fileURLToPath(import.meta.url));
process.env.PUMICE_PRUNED_VOCAB_DIR = path.join(root, "pruned-vocab");
process.chdir(path.join(root, "bin"));

const store = fs.mkdtempSync(path.join(os.tmpdir(), "kvtest-"));
let failures = 0;

function check(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
}

function openWith(storeDir, overrides = {}) {
  return addon.createEngine({
    weights: path.join(root, "model", "Qwen3.5-2B"),
    maxCtx: 8192,
    prune: true,
    kvStoreDir: storeDir,
    kvRamBudget: 512 * 1024 * 1024,
    kvDiskBudget: 1024 * 1024 * 1024,
    exportModel: false,
    ...overrides,
  });
}

function openEngine(storeDir, kvRamBudget, kvDiskBudget) {
  return openWith(storeDir, {
    kvRamBudget: kvRamBudget ?? 512 * 1024 * 1024,
    kvDiskBudget: kvDiskBudget ?? 1024 * 1024 * 1024,
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
  return addon.generate(e, Uint32Array.from(ids), params, (ev) => out.push(ev.token)).then(() => out);
}

function genFull(e, ids, maxNew) {
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
  return addon
    .generate(e, Uint32Array.from(ids), params, (ev) => out.push(ev.token))
    .then((result) => ({ out, result }));
}

function score(e, ids, prefill, decode, chunks) {
  return addon.score(e, Uint32Array.from(ids), { prefill, decode, chunks }, () => {});
}

function eq(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

async function main() {
  const text = fs.readFileSync(path.join(root, "data", "wikitext-2_test.txt"), "utf8").slice(0, 24000);
  const text2 = fs.readFileSync(path.join(root, "data", "wikitext-2_test.txt"), "utf8").slice(24000, 48000);

  let e = await openEngine(path.join(store, "warm"));
  const ids = Array.from(addon.tokenize(e, text, false)).slice(0, 3000);
  console.log(`prompt tokens=${ids.length}`);

  const out1 = await gen(e, ids, 16);
  const info1 = addon.engineInfo(e);
  check("turn1 generated", out1.length > 0, `tokens=${out1.length}`);
  check("turn1 cached blocks", info1.kvBlocks > 0, `blocks=${info1.kvBlocks} used=${info1.kvUsedBytes}`);
  addon.destroyEngine(e);

  const prompt2 = ids.concat(out1);

  e = await openEngine(path.join(store, "warm"));
  const outWarm2 = await gen(e, prompt2, 16);
  const infoWarm2 = addon.engineInfo(e);
  check("turn2 restored", infoWarm2.kvRestores > 0 && infoWarm2.kvHits > 0,
    `restores=${infoWarm2.kvRestores} hits=${infoWarm2.kvHits}`);

  const prompt3 = prompt2.concat(outWarm2);
  const outWarm3 = await gen(e, prompt3, 16);
  const infoWarm3 = addon.engineInfo(e);
  check("turn3 restored", infoWarm3.kvRestores > infoWarm2.kvRestores,
    `restores=${infoWarm3.kvRestores} hits=${infoWarm3.kvHits}`);
  addon.destroyEngine(e);

  e = await openEngine(path.join(store, "cold"));
  const outCold2 = await gen(e, prompt2, 16);
  const outCold3 = await gen(e, prompt3, 16);
  addon.destroyEngine(e);

  check("turn2 warm == cold", eq(outWarm2, outCold2),
    `warm=[${outWarm2.slice(0, 8)}] cold=[${outCold2.slice(0, 8)}]`);
  check("turn3 warm == cold", eq(outWarm3, outCold3),
    `warm=[${outWarm3.slice(0, 8)}] cold=[${outCold3.slice(0, 8)}]`);

  e = await openEngine(path.join(store, "warm"));
  await gen(e, prompt2, 16);
  await score(e, prompt2.slice(0, 2048), 512, 512, 2);
  const outAfterScore = await gen(e, prompt2, 16);
  check("scoring does not corrupt cache", eq(outAfterScore, outWarm2),
    `after=[${outAfterScore.slice(0, 8)}]`);
  addon.destroyEngine(e);

  const tinyStore = path.join(store, "tiny");
  const tinyRam = 10 * 1024 * 1024;
  e = await openEngine(tinyStore, tinyRam, 256 * 1024 * 1024);
  const outTiny1 = await gen(e, ids, 16);
  const tinyInfo1 = addon.engineInfo(e);
  check("tiny pool turn1 output matches", eq(outTiny1, out1), `tokens=${outTiny1.length}`);
  check("tiny pool evicts to cold", tinyInfo1.kvEvictions > 0 && tinyInfo1.kvColdBytes > 0,
    `evictions=${tinyInfo1.kvEvictions} cold=${tinyInfo1.kvColdBytes} ramUsed=${tinyInfo1.kvUsedBytes}`);

  const outTiny2 = await gen(e, prompt2, 16);
  const tinyInfo2 = addon.engineInfo(e);
  check("cold restore exercised", tinyInfo2.kvColdHits > 0 && tinyInfo2.kvRestores > 0,
    `coldHits=${tinyInfo2.kvColdHits} restores=${tinyInfo2.kvRestores}`);
  check("cold restore output matches", eq(outTiny2, outWarm2),
    `tiny=[${outTiny2.slice(0, 8)}] warm=[${outWarm2.slice(0, 8)}]`);
  addon.destroyEngine(e);

  e = await openEngine(tinyStore, tinyRam, 256 * 1024 * 1024);
  const restartInfo = addon.engineInfo(e);
  check("restart keeps cold tier", restartInfo.kvColdBytes > 0 && restartInfo.kvEntries > 0,
    `cold=${restartInfo.kvColdBytes} entries=${restartInfo.kvEntries}`);
  const outRestart = await gen(e, prompt2, 16);
  check("restart cold restore output matches", eq(outRestart, outWarm2),
    `restart=[${outRestart.slice(0, 8)}]`);
  addon.destroyEngine(e);

  const evStore = path.join(store, "evict");
  e = await openEngine(evStore, 6 * 1024 * 1024, 512 * 1024);
  const ids2 = Array.from(addon.tokenize(e, text2, false)).slice(0, 2500);
  await gen(e, ids2, 8);
  await gen(e, ids2.slice(0, 1200), 8);
  await gen(e, ids2.slice(0, 1800), 8);
  const evInfo = addon.engineInfo(e);
  check("disk budget deletes cold blocks", evInfo.kvColdDeletes > 0,
    `deletes=${evInfo.kvColdDeletes} cold=${evInfo.kvColdBytes} disk=${evInfo.kvDiskBudget}`);
  const outEv = await gen(e, ids2, 8);
  check("generation survives disk eviction", outEv.length > 0, `tokens=${outEv.length}`);
  addon.destroyEngine(e);

  const agentStore = path.join(store, "agent");
  e = await openEngine(agentStore);
  const agentIds = Array.from(addon.tokenize(e, text, false)).slice(0, 600);
  const extra = Array.from(addon.tokenize(e, "Now summarize the above.", false));
  const t1 = await genFull(e, agentIds, 96);
  const replyLen = t1.out.length;
  const agentTurn2 = agentIds.concat(t1.out, extra);
  const t2 = await genFull(e, agentTurn2, 16);
  const promptEnd = agentIds.length - (agentIds.length % 16);
  check("prompt-end snapshot resumes near the prompt end",
    t2.result.cachedTokens === promptEnd && t2.result.cachedTokens > 0,
    `cached=${t2.result.cachedTokens} prompt=${agentIds.length} reply=${replyLen} promptEnd=${promptEnd}`);
  addon.destroyEngine(e);

  e = await openEngine(path.join(store, "agent-cold"));
  const t2cold = await genFull(e, agentTurn2, 16);
  check("agent turn2 warm == cold", eq(t2.out, t2cold.out),
    `warm=[${t2.out.slice(0, 8)}] cold=[${t2cold.out.slice(0, 8)}]`);
  check("agent cold has no cache hit", t2cold.result.cachedTokens === 0,
    `cached=${t2cold.result.cachedTokens}`);
  addon.destroyEngine(e);

  const interStore = path.join(store, "interleave");
  e = await openEngine(interStore);
  const sliceA = Array.from(addon.tokenize(e, text, false)).slice(0, 2000);
  const sliceB = Array.from(addon.tokenize(e, text2, false)).slice(0, 2000);
  const sliceC = Array.from(addon.tokenize(e, text2.slice(12000), false)).slice(0, 2000);
  const tail = Array.from(addon.tokenize(e, "In one sentence, what happened above?", false));
  const convA = await genFull(e, sliceA, 24);
  const convB = await genFull(e, sliceB, 24);
  const convC = await genFull(e, sliceC, 24);
  const turnA = sliceA.concat(convA.out, tail);
  const turnB = sliceB.concat(convB.out, tail);
  const turnC = sliceC.concat(convC.out, tail);
  const snaps = addon.engineInfo(e).kvSnapshots;
  const a2 = await genFull(e, turnA, 8);
  const b2 = await genFull(e, turnB, 8);
  const c2 = await genFull(e, turnC, 8);
  check("interleaved conversations keep snapshots",
    a2.result.cachedTokens > 0 && b2.result.cachedTokens > 0 && c2.result.cachedTokens > 0,
    `snaps=${snaps} A=${a2.result.cachedTokens} B=${b2.result.cachedTokens} C=${c2.result.cachedTokens}`);
  addon.destroyEngine(e);

  e = await openEngine(interStore);
  const a2b = await genFull(e, turnA, 8);
  check("snapshots survive restart",
    a2b.result.cachedTokens >= a2.result.cachedTokens && a2b.result.cachedTokens > 0 && eq(a2b.out, a2.out),
    `before=${a2.result.cachedTokens} after=${a2b.result.cachedTokens}`);
  addon.destroyEngine(e);

  const switchStore = path.join(store, "switch");
  e = await openWith(switchStore);
  const switchIds = Array.from(addon.tokenize(e, text, false)).slice(0, 800);
  await genFull(e, switchIds, 8);
  const aEntries = addon.engineInfo(e).kvEntries;
  addon.destroyEngine(e);

  e = await openWith(switchStore, {
    weights: path.join(root, "model", "Qwen3.5-2B-1.6gb.hqm"),
    prune: false,
  });
  const bIds = Array.from(addon.tokenize(e, text, false)).slice(0, 800);
  await genFull(e, bIds, 8);
  addon.destroyEngine(e);

  const stores = fs.readdirSync(switchStore).filter((d) => fs.statSync(path.join(switchStore, d)).isDirectory());
  check("model fingerprints isolated", stores.length === 2, `stores=${stores.length}`);

  e = await openWith(switchStore);
  const a2c = await genFull(e, switchIds, 8);
  check("A cache survives B", a2c.result.cachedTokens > 0 && addon.engineInfo(e).kvEntries >= aEntries,
    `cached=${a2c.result.cachedTokens} entries=${addon.engineInfo(e).kvEntries} before=${aEntries}`);
  addon.destroyEngine(e);

  const ctxStore = path.join(store, "ctx");
  e = await openWith(ctxStore, { maxCtx: 2048 });
  const ctxIds = Array.from(addon.tokenize(e, text, false)).slice(0, 1500);
  await genFull(e, ctxIds, 8);
  addon.destroyEngine(e);

  e = await openWith(ctxStore, { maxCtx: 8192 });
  const ctx2 = await genFull(e, ctxIds, 8);
  check("larger maxCtx reload restores", ctx2.result.cachedTokens > 0,
    `cached=${ctx2.result.cachedTokens}`);
  addon.destroyEngine(e);

  const crcStore = path.join(store, "crc");
  e = await openWith(crcStore);
  const crcIds = Array.from(addon.tokenize(e, text, false)).slice(0, 800);
  await genFull(e, crcIds, 8);
  addon.destroyEngine(e);

  const crcDir = path.join(crcStore, fs.readdirSync(crcStore)[0]);
  const mirrorPath = path.join(crcDir, "mirror.bin");
  const fd = fs.openSync(mirrorPath, "r+");
  const byte = Buffer.alloc(1);
  fs.readSync(fd, byte, 0, 1, 0);
  byte[0] ^= 0xff;
  fs.writeSync(fd, byte, 0, 1, 0);
  fs.closeSync(fd);

  e = await openWith(crcStore);
  const crcAfter = await genFull(e, crcIds, 8);
  addon.destroyEngine(e);

  e = await openWith(path.join(store, "crc-cold"));
  const crcCold = await genFull(e, crcIds, 8);
  addon.destroyEngine(e);

  check("corrupt mirror slot falls back to prefill",
    eq(crcAfter.out, crcCold.out) && crcAfter.result.cachedTokens === 0,
    `cached=${crcAfter.result.cachedTokens} out=[${crcAfter.out.slice(0, 6)}]`);

  console.log(failures === 0 ? "\nALL TESTS PASSED" : `\n${failures} TEST(S) FAILED`);
  fs.rmSync(store, { recursive: true, force: true });
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error("TEST RUN FAILED:", err);
  process.exit(1);
});
