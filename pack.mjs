import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
const webui = path.join(root, "webui");
const bin = path.join(root, "bin");
const platform = path.join(root, "platform", "win32-x64");

function version(file) {
  return JSON.parse(fs.readFileSync(file, "utf8")).version;
}

function fail(message) {
  console.error(`pack: ${message}`);
  process.exit(1);
}

const mainVersion = version(path.join(root, "package.json"));
const platformVersion = version(path.join(platform, "package.json"));
if (mainVersion !== platformVersion) {
  fail(`version mismatch: main ${mainVersion} != platform ${platformVersion}`);
}
if (!fs.existsSync(path.join(webui, "node_modules"))) {
  fail("webui dependencies missing, run: npm --prefix webui install");
}
if (!fs.existsSync(path.join(bin, "pumice.node")) || !fs.existsSync(path.join(bin, "shader"))) {
  fail("engine artifacts missing, run: make");
}

console.log("pack: building webui");
execSync("npm run build:all", { cwd: webui, stdio: "inherit" });

console.log("pack: staging runtime");
fs.rmSync(path.join(platform, "shader"), { recursive: true, force: true });
fs.mkdirSync(platform, { recursive: true });
fs.copyFileSync(path.join(bin, "pumice.node"), path.join(platform, "pumice.node"));
fs.cpSync(path.join(bin, "shader"), path.join(platform, "shader"), { recursive: true });

console.log(`pack: ready (${mainVersion})`);
