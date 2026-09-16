import fs from "node:fs";
import path from "node:path";
import type { ProbeInfo, Quant } from "./types";

const QUANT_BY_VALUE: Record<number, Quant> = { 4: "q4_1_32", 5: "q4_1_64", 6: "q4_1_128", 7: "q4_1_256", 8: "int8", 16: "fp16" };
const TYPE_BY_VALUE: Record<number, string> = { 1: "full_attention", 2: "linear_attention" };
const QUANT_VALUES = new Set<string>(["fp16", "int8", "q4_1_32", "q4_1_64", "q4_1_128", "q4_1_256"]);
const DEFAULT_MAX_CTX = 32768;
const DEFAULT_PREFILL = 512;

const GGUF_SCALAR_SIZE: Record<number, number> = {
  0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
};

class Reader {
  private buf = Buffer.alloc(0);
  private pos = 0;
  private filePos = 0;

  constructor(private fd: number) {}

  private ensure(n: number): void {
    if (this.pos + n <= this.buf.length) return;
    const remain = this.buf.subarray(this.pos);
    const chunk = Buffer.alloc(Math.max(n, 1 << 16));
    const read = fs.readSync(this.fd, chunk, 0, chunk.length, this.filePos);
    this.buf = Buffer.concat([remain, chunk.subarray(0, read)]);
    this.pos = 0;
    this.filePos += read;
  }

  read(n: number): Buffer {
    this.ensure(n);
    const out = this.buf.subarray(this.pos, this.pos + n);
    this.pos += n;
    return out;
  }

  skip(n: number): void {
    const available = this.buf.length - this.pos;
    if (n <= available) {
      this.pos += n;
      return;
    }
    this.filePos += n - available;
    this.pos = this.buf.length;
    this.buf = Buffer.alloc(0);
  }

  u8(): number {
    return this.read(1).readUInt8(0);
  }

  u16(): number {
    return this.read(2).readUInt16LE(0);
  }

  i16(): number {
    return this.read(2).readInt16LE(0);
  }

  u32(): number {
    return this.read(4).readUInt32LE(0);
  }

  i32(): number {
    return this.read(4).readInt32LE(0);
  }

  f32(): number {
    return this.read(4).readFloatLE(0);
  }

  f64(): number {
    return this.read(8).readDoubleLE(0);
  }

  u64(): number {
    return Number(this.read(8).readBigUInt64LE(0));
  }

  i64(): number {
    return Number(this.read(8).readBigInt64LE(0));
  }

  str(): string {
    return this.read(this.u64()).toString("utf8");
  }

  skipString(): void {
    this.skip(this.u64());
  }

  readValue(type: number): unknown {
    switch (type) {
      case 0: return this.u8();
      case 1: return this.read(1).readInt8(0);
      case 2: return this.u16();
      case 3: return this.i16();
      case 4: return this.u32();
      case 5: return this.i32();
      case 6: return this.f32();
      case 7: return this.u8() !== 0;
      case 8: return this.str();
      case 10: return this.u64();
      case 11: return this.i64();
      case 12: return this.f64();
      default: return undefined;
    }
  }

  skipValue(type: number): void {
    if (type === 8) {
      this.skipString();
    } else if (type === 9) {
      const elem = this.u32();
      const count = this.u64();
      if (elem === 8) {
        for (let i = 0; i < count; i++) this.skipString();
      } else {
        this.skip(count * (GGUF_SCALAR_SIZE[elem] ?? 0));
      }
    } else {
      this.skip(GGUF_SCALAR_SIZE[type] ?? 0);
    }
  }
}

function emptyInfo(kind: ProbeInfo["kind"], target: string, name: string): ProbeInfo {
  return {
    kind,
    path: target,
    ok: false,
    errors: [],
    warnings: [],
    name,
    layers: 0,
    layerTypes: [],
    attnQuants: [],
    ffnQuants: [],
    tied: false,
    experts: 0,
    maxCtx: DEFAULT_MAX_CTX,
    maxPos: DEFAULT_MAX_CTX,
    prefillChunk: DEFAULT_PREFILL,
    embed: "fp16",
    lmHead: "fp16",
    firstShard: null,
  };
}

export function modelKind(target: string): "safetensors" | "gguf" | "hqm" | "unknown" {
  let stat: fs.Stats;
  try {
    stat = fs.statSync(target);
  } catch {
    return "unknown";
  }
  if (stat.isDirectory()) return "safetensors";
  const ext = path.extname(target).toLowerCase();
  if (ext === ".gguf") return "gguf";
  if (ext === ".hqm") return "hqm";
  if (ext === ".safetensors") return "safetensors";
  return "unknown";
}

export function resolveModelDir(target: string): string {
  try {
    return fs.statSync(target).isDirectory() ? target : path.dirname(target);
  } catch {
    return target;
  }
}

function readJson(file: string): any {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function applyExistingQuant(info: ProbeInfo, dir: string): void {
  const quantPath = path.join(dir, "quant_config.json");
  if (!fs.existsSync(quantPath)) return;
  let data: any;
  try {
    data = readJson(quantPath);
  } catch {
    return;
  }
  const layers = Array.isArray(data.layers) ? data.layers : [];
  if (layers.length === info.layers) {
    info.attnQuants = layers.map((l: any) => String(l.attn ?? "fp16"));
    info.ffnQuants = layers.map((l: any) => String(l.ffn ?? "fp16"));
  }
  const embed = String(data.embed ?? "fp16");
  const lmHead = String(data.lm_head ?? "fp16");
  info.embed = (QUANT_VALUES.has(embed) ? embed : "fp16") as Quant;
  info.lmHead = (QUANT_VALUES.has(lmHead) ? lmHead : "fp16") as Quant;
  info.prefillChunk = Number(data.prefill_chunk ?? DEFAULT_PREFILL) || DEFAULT_PREFILL;
  if (data.max_ctx) info.maxCtx = Number(data.max_ctx);
}

export function validateSafetensors(dir: string): ProbeInfo {
  const info = emptyInfo("safetensors", dir, path.basename(dir));

  const configPath = path.join(dir, "config.json");
  if (!fs.existsSync(configPath)) {
    info.errors.push("config.json not found");
  } else {
    try {
      const cfg = readJson(configPath);
      const text = cfg.text_config ?? cfg;
      const required = ["hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "layer_types"];
      const missing = required.filter((k) => !(k in text));
      if (missing.length) info.errors.push("config.json missing: " + missing.join(", "));
      const layerTypes: string[] = text.layer_types ?? [];
      const count = Number(text.num_hidden_layers ?? 0) || 0;
      if (count && layerTypes.length !== count) {
        info.errors.push(`layer_types count ${layerTypes.length} != num_hidden_layers ${count}`);
      }
      info.layers = count;
      info.layerTypes = layerTypes;
      const maxPos = Number(text.max_position_embeddings ?? DEFAULT_MAX_CTX) || DEFAULT_MAX_CTX;
      info.maxCtx = maxPos;
      info.maxPos = maxPos;
      info.experts = Number(text.num_experts ?? 0) || 0;
      info.tied = Boolean(text.tie_word_embeddings);
    } catch (e) {
      info.errors.push("config.json invalid: " + String(e));
    }
  }

  let names: string[] = [];
  const index = path.join(dir, "model.safetensors.index.json");
  if (fs.existsSync(index)) {
    try {
      const data = readJson(index);
      names = [...new Set<string>(Object.values(data.weight_map ?? {}))].sort();
    } catch {
      names = [];
    }
  }
  if (!names.length) {
    names = fs.readdirSync(dir).filter((f) => /^model.*\.safetensors$/.test(f)).sort();
  }
  info.firstShard = names[0] ?? null;

  if (!names.length) {
    info.errors.push("no safetensors shards found");
  } else {
    const missing = new Set(names.filter((n) => !fs.existsSync(path.join(dir, n))));
    const totals = new Map<string, number>();
    for (const n of names) {
      const m = /^(.*?)(\d+)-of-(\d+)\.safetensors$/.exec(n);
      if (m) totals.set(m[1], Math.max(totals.get(m[1]) ?? 0, Number(m[3])));
    }
    for (const [prefix, total] of totals) {
      for (let i = 1; i <= total; i++) {
        const candidate = `${prefix}${String(i).padStart(5, "0")}-of-${String(total).padStart(5, "0")}.safetensors`;
        if (!fs.existsSync(path.join(dir, candidate))) missing.add(candidate);
      }
    }
    if (missing.size) info.errors.push("missing shards: " + [...missing].sort().join(", "));
  }

  for (const name of ["tokenizer.json", "tokenizer_config.json", "vocab.json"]) {
    if (!fs.existsSync(path.join(dir, name))) {
      info.warnings.push(`${name} not found (prune mode can supply it)`);
    }
  }

  info.ok = info.errors.length === 0;
  if (info.ok) applyExistingQuant(info, dir);
  return info;
}

function withReader<T>(target: string, fn: (r: Reader) => T): T {
  const fd = fs.openSync(target, "r");
  try {
    return fn(new Reader(fd));
  } finally {
    fs.closeSync(fd);
  }
}

function readGgufMeta(target: string): { meta: Map<string, unknown>; tensors: Set<string> } {
  return withReader(target, (r) => {
    if (r.read(4).toString("ascii") !== "GGUF") throw new Error("not a gguf file");
    r.u32();
    const nTensors = r.u64();
    const nKv = r.u64();
    const meta = new Map<string, unknown>();
    for (let i = 0; i < nKv; i++) {
      const key = r.str();
      const type = r.u32();
      if (type === 9) {
        r.skipValue(type);
        continue;
      }
      meta.set(key, r.readValue(type));
    }
    const tensors = new Set<string>();
    for (let i = 0; i < nTensors; i++) {
      const name = r.str();
      const ndim = r.u32();
      r.skip(ndim * 8);
      r.u32();
      r.u64();
      tensors.add(name);
    }
    return { meta, tensors };
  });
}

function archGet(meta: Map<string, unknown>, suffix: string): unknown {
  for (const [key, value] of meta) {
    if (key === suffix || key.endsWith("." + suffix)) return value;
  }
  return undefined;
}

export function readGgufInfo(target: string): ProbeInfo {
  const { meta, tensors } = readGgufMeta(target);
  const info = emptyInfo("gguf", target, path.basename(target, path.extname(target)));
  const layers = Number(archGet(meta, "block_count") ?? 0) || 0;
  const interval = Number(archGet(meta, "full_attention_interval") ?? 0) || 0;
  info.layers = layers;
  info.layerTypes = Array.from({ length: layers }, (_, i) =>
    interval && (i + 1) % interval === 0 ? "full_attention" : "linear_attention");
  const maxPos = Number(archGet(meta, "context_length") ?? DEFAULT_MAX_CTX) || DEFAULT_MAX_CTX;
  info.maxCtx = maxPos;
  info.maxPos = maxPos;
  info.experts = Number(archGet(meta, "expert_count") ?? 0) || 0;
  info.tied = !tensors.has("output.weight");
  info.ok = true;
  applyExistingQuant(info, path.dirname(target));
  return info;
}

interface HqmTensor {
  type: number;
  dims: number[];
  offset: number;
}

function readHqmMeta(target: string): {
  kvs: Map<string, unknown>;
  tensors: Map<string, HqmTensor>;
  dataOffset: number;
} {
  return withReader(target, (r) => {
    if (r.read(3).toString("ascii") !== "HQM") throw new Error("not an hqm file");
    r.u8();
    r.u32();
    const nTensors = r.u64();
    const nKv = r.u64();
    const kvs = new Map<string, unknown>();
    for (let i = 0; i < nKv; i++) {
      const key = r.str();
      const type = r.u32();
      if (type === 8) {
        kvs.set(key, r.str());
      } else if (type === 12) {
        kvs.set(key, r.f64());
      } else if (type === 4 || type === 7) {
        kvs.set(key, r.u32());
      } else if (type === 10) {
        kvs.set(key, r.u64());
      } else {
        throw new Error(`unknown hqm kv type ${type}`);
      }
    }
    const tensors = new Map<string, HqmTensor>();
    for (let i = 0; i < nTensors; i++) {
      const name = r.str();
      const ndim = r.u32();
      const dims: number[] = [];
      for (let d = 0; d < ndim; d++) dims.push(r.u64());
      const type = r.u32();
      const offset = r.u64();
      tensors.set(name, { type, dims, offset });
    }
    return { kvs, tensors, dataOffset: Number(kvs.get("hqm.data_offset") ?? 0) };
  });
}

function readHqmI32(target: string, dataOffset: number, tensor: HqmTensor | undefined): number[] {
  if (!tensor) return [];
  const count = tensor.dims.reduce((a, b) => a * b, 1);
  const buf = Buffer.alloc(count * 4);
  const fd = fs.openSync(target, "r");
  try {
    fs.readSync(fd, buf, 0, buf.length, dataOffset + tensor.offset);
  } finally {
    fs.closeSync(fd);
  }
  const out: number[] = [];
  for (let i = 0; i < count; i++) out.push(buf.readInt32LE(i * 4));
  return out;
}

export function readHqmInfo(target: string): ProbeInfo {
  const { kvs, tensors, dataOffset } = readHqmMeta(target);
  const info = emptyInfo("hqm", target, String(kvs.get("general.name") ?? path.basename(target, ".hqm")));
  const types = readHqmI32(target, dataOffset, tensors.get("config.layer_type"));
  const aq = readHqmI32(target, dataOffset, tensors.get("config.layer_attn_quant"));
  const fq = readHqmI32(target, dataOffset, tensors.get("config.layer_ffn_quant"));
  const maxCtx = Number(kvs.get("hqm.max_ctx") ?? DEFAULT_MAX_CTX) || DEFAULT_MAX_CTX;
  info.layers = Number(kvs.get("hqm.layer_count") ?? 0) || 0;
  info.layerTypes = types.map((v) => TYPE_BY_VALUE[v] ?? "");
  info.attnQuants = aq.map((v) => QUANT_BY_VALUE[v] ?? "fp16");
  info.ffnQuants = fq.map((v) => QUANT_BY_VALUE[v] ?? "fp16");
  info.embed = QUANT_BY_VALUE[Number(kvs.get("hqm.embed_quant") ?? 16)] ?? "fp16";
  info.lmHead = QUANT_BY_VALUE[Number(kvs.get("hqm.lm_head_quant") ?? 16)] ?? "fp16";
  info.maxCtx = maxCtx;
  info.maxPos = maxCtx;
  info.prefillChunk = Number(kvs.get("hqm.prefill_chunk") ?? DEFAULT_PREFILL) || DEFAULT_PREFILL;
  info.experts = Number(kvs.get("hqm.experts") ?? 0) || 0;
  info.tied = Number(kvs.get("hqm.tied") ?? 0) !== 0;
  info.ok = true;
  return info;
}

export function probeModel(target: string): ProbeInfo {
  const kind = modelKind(target);
  if (kind === "unknown") {
    return { ...emptyInfo("safetensors", target, path.basename(target)), errors: ["unsupported model path"] };
  }
  try {
    if (kind === "safetensors") return validateSafetensors(resolveModelDir(target));
    if (kind === "gguf") return readGgufInfo(target);
    return readHqmInfo(target);
  } catch (e) {
    const info = emptyInfo(kind, target, path.basename(target));
    info.errors.push(String(e));
    return info;
  }
}

export function scanModels(modelRoot: string): { path: string; label: string; kind: string }[] {
  const results: { path: string; label: string; kind: string }[] = [];
  if (!fs.existsSync(modelRoot)) return results;
  for (const entry of fs.readdirSync(modelRoot, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    const full = path.join(modelRoot, entry.name);
    if (entry.isDirectory()) {
      if (fs.existsSync(path.join(full, "config.json"))) {
        results.push({ path: full, label: entry.name, kind: "safetensors" });
      }
    } else {
      const ext = path.extname(entry.name).toLowerCase();
      if (ext === ".gguf" || ext === ".hqm") {
        results.push({ path: full, label: entry.name, kind: ext.slice(1) });
      }
    }
  }
  return results;
}
