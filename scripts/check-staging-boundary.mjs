import { execFileSync } from "node:child_process";
import console from "node:console";
import { lstat, readdir, realpath, stat } from "node:fs/promises";
import {
  dirname,
  extname,
  isAbsolute,
  join,
  parse,
  relative,
  resolve,
  sep,
} from "node:path";
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
const WINDOWS_SYSTEM_ROOT_ALIAS = String.raw`\\?\GLOBALROOT\SystemRoot`;

function isNodeError(error, code) {
  return error instanceof Error && "code" in error && error.code === code;
}

async function authenticatedSystemGit() {
  let path;
  let systemRoot;
  let system32;
  if (process.platform === "win32") {
    systemRoot = await realpath(WINDOWS_SYSTEM_ROOT_ALIAS);
    system32 = await realpath(join(WINDOWS_SYSTEM_ROOT_ALIAS, "System32"));
    if (
      !isAbsolute(systemRoot)
      || relative(systemRoot, system32).toLocaleLowerCase("en-US") !== "system32"
    ) {
      throw new TypeError("The OS Windows-directory alias resolved unexpectedly.");
    }
    const programFiles = await realpath(join(parse(systemRoot).root, "Program Files"));
    path = await realpath(join(programFiles, "Git", "cmd", "git.exe"));
    const child = relative(programFiles, path);
    if (
      child === ""
      || child === ".."
      || child.startsWith(`..${sep}`)
      || isAbsolute(child)
    ) {
      throw new TypeError("System Git escaped the fixed Program Files root.");
    }
  } else {
    const candidates = new Set();
    for (const candidate of ["/usr/bin/git", "/bin/git"]) {
      try {
        const canonical = await realpath(candidate);
        if ((await lstat(canonical)).isFile()) {
          candidates.add(canonical);
        }
      } catch (error) {
        if (!isNodeError(error, "ENOENT")) {
          throw error;
        }
      }
    }
    if (candidates.size !== 1) {
      throw new TypeError("Exactly one fixed system Git executable is required.");
    }
    path = [...candidates][0];
  }
  const metadata = await lstat(path, { bigint: true });
  if (!metadata.isFile()) {
    throw new TypeError("Authenticated system Git is not a regular file.");
  }
  const nullDevice = process.platform === "win32" ? "NUL" : "/dev/null";
  const environment = Object.fromEntries(
    Object.entries(process.env).filter(([name]) => {
      const normalized = name.toUpperCase();
      return normalized === "TEMP" || normalized === "TMP";
    }),
  );
  Object.assign(environment, {
    PATH: process.platform === "win32"
      ? `${dirname(path)};${system32}`
      : "/usr/bin:/bin",
    GIT_ATTR_NOSYSTEM: "1",
    GIT_CONFIG_COUNT: "0",
    GIT_CONFIG_GLOBAL: nullDevice,
    GIT_CONFIG_NOSYSTEM: "1",
    GIT_OPTIONAL_LOCKS: "0",
    GIT_PAGER: "",
    GIT_TERMINAL_PROMPT: "0",
    LC_ALL: "C",
  });
  if (process.platform === "win32") {
    Object.assign(environment, {
      SystemRoot: systemRoot,
      WINDIR: systemRoot,
      ComSpec: join(system32, "cmd.exe"),
      PATHEXT: ".COM;.EXE;.BAT;.CMD",
    });
  }
  return Object.freeze({ path, metadata, environment });
}

async function assertSystemGitUnchanged(expected) {
  const actual = await lstat(expected.path, { bigint: true });
  if (
    !actual.isFile()
    || actual.dev !== expected.metadata.dev
    || actual.ino !== expected.metadata.ino
    || actual.size !== expected.metadata.size
    || actual.mtimeNs !== expected.metadata.mtimeNs
    || actual.ctimeNs !== expected.metadata.ctimeNs
  ) {
    throw new Error("Authenticated system Git changed during use.");
  }
}

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

async function repositoryCandidatePaths() {
  const git = await authenticatedSystemGit();
  const nullDevice = process.platform === "win32" ? "NUL" : "/dev/null";
  try {
    const output = execFileSync(
      git.path,
      [
        "--no-replace-objects",
        "--no-pager",
        "-c",
        "credential.helper=",
        "-c",
        "credential.interactive=false",
        "-c",
        `core.hooksPath=${nullDevice}`,
        "-c",
        "core.fsmonitor=false",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
      ],
      {
        cwd: root,
        encoding: "utf8",
        env: git.environment,
        timeout: 30_000,
        windowsHide: true,
      },
    );
    return output.split("\0").filter((path) => path.length > 0);
  } finally {
    await assertSystemGitUnchanged(git);
  }
}

for (const projectPath of await repositoryCandidatePaths()) {
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
