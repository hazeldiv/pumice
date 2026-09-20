import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
const platform = path.join(root, "platform", "win32-x64");
const mainPkgPath = path.join(root, "package.json");
const platformPkgPath = path.join(platform, "package.json");

function nextVersion(current, spec) {
  if (/^\d+\.\d+\.\d+$/.test(spec)) return spec;
  const [major, minor, patch] = current.split(".").map(Number);
  if (spec === "major") return `${major + 1}.0.0`;
  if (spec === "minor") return `${major}.${minor + 1}.0`;
  if (spec === "patch") return `${major}.${minor}.${patch + 1}`;
  console.error(`release: invalid version ${spec}`);
  process.exit(2);
}

function writeVersion(file, version) {
  const data = JSON.parse(fs.readFileSync(file, "utf8"));
  data.version = version;
  if (data.optionalDependencies?.["@h4zel/pumice-win32-x64"]) {
    data.optionalDependencies["@h4zel/pumice-win32-x64"] = version;
  }
  fs.writeFileSync(file, JSON.stringify(data, null, 2) + "\n");
}

const mainPkg = JSON.parse(fs.readFileSync(mainPkgPath, "utf8"));

function argValue(name) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

const spec = process.argv[2] && !process.argv[2].startsWith("--") ? process.argv[2] : "patch";
const version = nextVersion(mainPkg.version, spec);
const otp = argValue("--otp") ?? process.env.NPM_OTP;
const publishArgs = `npm publish --access public${otp ? ` --otp=${otp}` : ""}`;

console.log(`release: ${mainPkg.version} -> ${version}`);
writeVersion(mainPkgPath, version);
writeVersion(platformPkgPath, version);

execSync("node pack.mjs", { cwd: root, stdio: "inherit" });

console.log("release: publishing platform package");
execSync(publishArgs, { cwd: platform, stdio: "inherit" });

console.log("release: publishing main package");
execSync(publishArgs, { cwd: root, stdio: "inherit" });

console.log(`release: published @h4zel/pumice@${version}`);
