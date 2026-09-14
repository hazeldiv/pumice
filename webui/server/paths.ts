import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

export const home = path.resolve(here, "..", "..");
export const runtimeDir = process.env.VK_COMPUTE_RUNTIME ?? path.join(home, "bin");
export const distDir = path.join(home, "webui", "dist");
export const launchCwd = process.env.VK_COMPUTE_CWD ?? home;
export const modelRoot = process.env.VK_COMPUTE_MODELS ?? path.join(home, "model");
export const exportRoot = process.env.VK_COMPUTE_EXPORT_DIR ?? path.join(home, "exported");
export const quantTmp = process.env.VK_COMPUTE_QUANT_TMP ?? path.join(os.tmpdir(), "vk-compute-quant.json");
export const addonPath = path.join(runtimeDir, "vk_compute.node");

export const prunedVocabDir = process.env.VK_PRUNED_VOCAB_DIR ?? path.join(home, "pruned-vocab");
process.env.VK_PRUNED_VOCAB_DIR = prunedVocabDir;

process.chdir(runtimeDir);
