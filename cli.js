#!/usr/bin/env node
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const home = path.dirname(fileURLToPath(import.meta.url));
const launchCwd = process.cwd();
const platformPackage = "@h4zel/pumice-win32-x64";

function usage() {
  console.log(`Usage: pumice [options]

Options:
  -p, --port <port>    port to listen on (default 8787)
  -m, --models <dir>   model directory to scan (default ./model)
      --host <addr>    bind address (default 127.0.0.1, env PUMICE_HOST)
      --api-key <key>  require Authorization: Bearer <key> on /v1 (env PUMICE_API_KEY)
      --no-open        do not open the browser
  -h, --help           show this help`);
}

function parseArgs(argv) {
  const opts = { port: 8787, models: null, open: true, apiKey: null, host: null };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--port" || arg === "-p") opts.port = Number(argv[++i]);
    else if (arg === "--models" || arg === "-m") opts.models = argv[++i];
    else if (arg === "--host") opts.host = argv[++i];
    else if (arg === "--api-key") opts.apiKey = argv[++i];
    else if (arg === "--no-open") opts.open = false;
    else if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else {
      console.error(`pumice: unknown option ${arg}`);
      usage();
      process.exit(2);
    }
  }
  if (!Number.isInteger(opts.port) || opts.port < 1 || opts.port > 65535) {
    console.error(`pumice: invalid port ${opts.port}`);
    process.exit(2);
  }
  return opts;
}

function resolveRuntime() {
  try {
    return path.dirname(require.resolve(`${platformPackage}/package.json`));
  } catch {}
  const local = path.join(home, "platform", "win32-x64");
  if (fs.existsSync(path.join(local, "pumice.node"))) return local;
  console.error(`pumice: no engine binary for ${process.platform}-${process.arch}`);
  console.error(`pumice: expected optional dependency ${platformPackage}`);
  process.exit(1);
}

const opts = parseArgs(process.argv.slice(2));
const runtimeDir = resolveRuntime();
const modelsDir = opts.models ? path.resolve(launchCwd, opts.models) : path.join(launchCwd, "model");

process.env.PUMICE_RUNTIME = runtimeDir;
process.env.PUMICE_CWD = launchCwd;
process.env.PUMICE_MODELS = modelsDir;
process.env.PUMICE_PRUNED_VOCAB_DIR = path.join(launchCwd, "pruned-vocab");
process.env.PUMICE_EXPORT_DIR = path.join(launchCwd, "exported");
process.env.PUMICE_QUANT_TMP = path.join(os.tmpdir(), `pumice-quant-${process.pid}.json`);
process.env.PUMICE_OPEN = opts.open ? "1" : "0";
if (opts.apiKey) process.env.PUMICE_API_KEY = opts.apiKey;
if (opts.host) process.env.PUMICE_HOST = opts.host;
process.env.PORT = String(opts.port);

const server = path.join(home, "webui", "dist-server", "index.mjs");
if (!fs.existsSync(server)) {
  console.error("pumice: server bundle missing, reinstall the package");
  process.exit(1);
}
if (!fs.existsSync(path.join(runtimeDir, "pumice.node"))) {
  console.error(`pumice: engine binary missing in ${runtimeDir}`);
  process.exit(1);
}

console.log(`pumice: models  ${modelsDir}`);
console.log(`pumice: runtime ${runtimeDir}`);
console.log(`pumice: api     http://${opts.host ?? "127.0.0.1"}:${opts.port}/v1`);

await import(pathToFileURL(server).href);
