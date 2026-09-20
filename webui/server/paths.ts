import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

export const home = path.resolve(here, "..", "..");
export const runtimeDir = process.env.PUMICE_RUNTIME ?? path.join(home, "bin");
export const distDir = path.join(home, "webui", "dist");
export const launchCwd = process.env.PUMICE_CWD ?? home;
export const modelRoot = process.env.PUMICE_MODELS ?? path.join(home, "model");
export const exportRoot = process.env.PUMICE_EXPORT_DIR ?? path.join(home, "exported");
export const quantTmp = process.env.PUMICE_QUANT_TMP ?? path.join(os.tmpdir(), "pumice-quant.json");
export const addonPath = path.join(runtimeDir, "pumice.node");

export const prunedVocabDir = process.env.PUMICE_PRUNED_VOCAB_DIR ?? path.join(home, "pruned-vocab");
process.env.PUMICE_PRUNED_VOCAB_DIR = prunedVocabDir;

process.chdir(runtimeDir);
