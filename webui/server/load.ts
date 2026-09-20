import path from "node:path";
import type { LoadOptions } from "./engine";
import { exportRoot, launchCwd } from "./paths";
import type { ProbeInfo, Quant, QuantConfig } from "./types";

const QUANT_VALUES = new Set(["fp16", "int8", "q4_1_32", "q4_1_64", "q4_1_128", "q4_1_256"]);

export function toQuant(value: unknown, fallback: Quant = "fp16"): Quant {
  const s = String(value ?? "");
  return (QUANT_VALUES.has(s) ? s : fallback) as Quant;
}

export function resolveInput(target: string): string {
  if (!target) return target;
  return path.isAbsolute(target) ? target : path.resolve(launchCwd, target);
}

export function buildLoadOptions(target: string, info: ProbeInfo, body: any): LoadOptions {
  const maxCtx = Number(body?.maxCtx ?? info.maxCtx);
  const prefillChunk = Number(body?.prefillChunk ?? info.prefillChunk);
  const expertsVram = Number(body?.expertsVram ?? 0);
  const layers: { attn: Quant; ffn: Quant }[] = Array.isArray(body?.layers)
    ? body.layers.map((l: any) => ({ attn: toQuant(l.attn), ffn: toQuant(l.ffn) }))
    : Array.from({ length: info.layers }, (_, i) => ({
        attn: toQuant(info.attnQuants[i] ?? "fp16"),
        ffn: toQuant(info.ffnQuants[i] ?? "fp16"),
      }));

  const quant: QuantConfig | null = info.kind === "hqm" ? null : {
    name: info.name,
    maxCtx,
    prefillChunk,
    embed: toQuant(body?.embed ?? info.embed),
    lmHead: toQuant(body?.lmHead ?? info.lmHead),
    layers,
  };
  if (quant && info.experts > 0) quant.expertsVram = expertsVram;

  const autoPrune = process.env.VK_COMPUTE_AUTOLOAD_PRUNE;
  const prune = body?.prune !== undefined
    ? Boolean(body.prune)
    : (autoPrune !== undefined ? autoPrune !== "0" : info.kind !== "hqm");

  return {
    path: target,
    quant,
    prune: prune && info.kind !== "hqm",
    exportModel: Boolean(body?.exportModel),
    exportDir: resolveInput(String(body?.exportDir ?? exportRoot)),
    maxCtx,
    expertsVram: info.experts > 0 ? expertsVram : 0,
    kvRamBudget: Number(body?.kvRamBudget ?? 0),
    kvDiskBudget: Number(body?.kvDiskBudget ?? 0),
    kvStoreDir: String(body?.kvStoreDir ?? "kvstore"),
  };
}
