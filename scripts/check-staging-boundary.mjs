import { execFileSync } from "node:child_process";
import console from "node:console";
import { readdir, stat } from "node:fs/promises";
import { extname, relative, resolve, sep } from "node:path";
import process from "node:process";

const root = resolve(import.meta.dirname, "..");
const forbiddenPaths = Object.freeze([
  "apps/simulator-cli",
  "data",
  "packages/chess-core",
  "packages/chess-evaluator",
  "packages/drawback-engine",
  "packages/drawback-search",
  "packages/probe-search",
  "packages/simulation",
  "apps/web/src/InteractiveBoard.tsx",
  "apps/web/src/board-logic.ts",
  "apps/web/src/diagnostic-search.ts",
  "apps/web/src/live-diagnostic.worker.ts",
  "apps/web/src/live-diagnostic-worker-protocol.ts",
  "apps/web/src/live-diagnostic-controller.ts",
]);
const forbiddenExtensions = new Set([
  ".ndjson",
  ".onnx",
  ".parquet",
  ".pt",
  ".sqlite",
  ".sqlite3",
]);
const forbiddenGeneratedDirectories = new Set([
  "__pycache__",
  ".mypy_cache",
  ".pytest_cache",
  ".ruff_cache",
  "coverage",
  "dist",
  "htmlcov",
  "node_modules",
]);
const failures = [];

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {
      return false;
    }
    throw error;
  }
}

for (const path of forbiddenPaths) {
  if (await exists(resolve(root, path))) {
    failures.push(`forbidden staging path exists: ${path}`);
  }
}

function repositoryCandidatePaths() {
  const output = execFileSync(
    "git",
    ["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
    { cwd: root, encoding: "utf8" },
  );
  return output.split("\0").filter((path) => path.length > 0);
}

for (const projectPath of repositoryCandidatePaths()) {
  if (projectPath === "engine" || projectPath.startsWith("engine/")) {
    continue;
  }
  const segments = projectPath.split("/");
  const filename = segments.at(-1) ?? "";
  if (
    segments.some((segment) => forbiddenGeneratedDirectories.has(segment))
    || filename.endsWith(".pyc")
    || forbiddenExtensions.has(extname(filename).toLowerCase())
  ) {
    failures.push(`generated/private artifact would be tracked: ${projectPath}`);
  }
}

async function checkNestedGitMetadata(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    const projectPath = relative(root, path).split(sep).join("/");
    if (projectPath === ".git") {
      continue;
    }
    if (projectPath === "engine" || projectPath.startsWith("engine/")) {
      continue;
    }
    if (entry.isDirectory()) {
      if (entry.name === ".git") {
        failures.push(`staging tree unexpectedly contains Git metadata: ${projectPath}`);
        continue;
      }
      if (forbiddenGeneratedDirectories.has(entry.name)) {
        continue;
      }
      await checkNestedGitMetadata(path);
    }
  }
}

await checkNestedGitMetadata(root);

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exitCode = 1;
} else {
  console.log(
    "Staging boundary is clean: no copied legality/search/simulation packages or generated artifacts.",
  );
}
