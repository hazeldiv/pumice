import fs from "node:fs";
import { resolveInput } from "./load";
import { modelRoot } from "./paths";
import { scanModels } from "./probe";

export interface ModelEntry {
  path: string;
  label: string;
  kind: string;
}

let scannedDir: string | null = null;
let scanned: ModelEntry[] = [];

export function scanDirectory(dir: string): ModelEntry[] {
  const target = resolveInput(dir.trim());
  let stat: fs.Stats;
  try {
    stat = fs.statSync(target);
  } catch {
    throw new Error(`directory not found: ${dir}`);
  }
  if (!stat.isDirectory()) throw new Error(`not a directory: ${dir}`);
  scanned = scanModels(target);
  scannedDir = target;
  return scanned;
}

export function scannedModels(): ModelEntry[] {
  return scanned;
}

export function lastScanDir(): string | null {
  return scannedDir;
}

if (modelRoot) {
  try {
    scanDirectory(modelRoot);
  } catch (e) {
    console.warn(`models: ${String((e as Error)?.message ?? e)}`);
  }
}
