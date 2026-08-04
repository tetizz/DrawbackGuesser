import { execFile } from "node:child_process";
import console from "node:console";
import { access, lstat, readFile, realpath } from "node:fs/promises";
import {
  dirname,
  isAbsolute,
  join,
  parse,
  relative,
  resolve,
  sep,
} from "node:path";
import process from "node:process";

const expected = Object.freeze([
  ["engine/packages/shared/package.json", "@drawbackengine/shared"],
  [
    "engine/packages/drawback-engine/package.json",
    "@drawbackengine/drawback-engine",
  ],
  ["engine/packages/chess-core/package.json", "@drawbackengine/chess-core"],
  [
    "engine/packages/chess-evaluator/package.json",
    "@drawbackengine/chess-evaluator",
  ],
  [
    "engine/packages/simulation-trace/package.json",
    "@drawbackengine/simulation-trace",
  ],
]);
const WINDOWS_SYSTEM_ROOT_ALIAS = String.raw`\\?\GLOBALROOT\SystemRoot`;
const root = resolve(import.meta.dirname, "..");
const engine = join(root, "engine");
const failures = [];

function isNodeError(error, code) {
  return error instanceof Error && "code" in error && error.code === code;
}

async function windowsDirectories() {
  const systemRoot = await realpath(WINDOWS_SYSTEM_ROOT_ALIAS);
  const system32 = await realpath(join(WINDOWS_SYSTEM_ROOT_ALIAS, "System32"));
  if (
    !isAbsolute(systemRoot)
    || relative(systemRoot, system32).toLocaleLowerCase("en-US") !== "system32"
  ) {
    throw new TypeError("The OS Windows-directory alias resolved unexpectedly.");
  }
  return Object.freeze({ systemRoot, system32 });
}

async function authenticatedGit() {
  let path;
  let systemRoot;
  let system32;
  if (process.platform === "win32") {
    ({ systemRoot, system32 } = await windowsDirectories());
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

async function assertGitUnchanged(git) {
  const actual = await lstat(git.path, { bigint: true });
  if (
    !actual.isFile()
    || actual.dev !== git.metadata.dev
    || actual.ino !== git.metadata.ino
    || actual.size !== git.metadata.size
    || actual.mtimeNs !== git.metadata.mtimeNs
    || actual.ctimeNs !== git.metadata.ctimeNs
  ) {
    throw new Error("Authenticated system Git changed during use.");
  }
}

async function runGit(git, repository, arguments_, acceptedStatuses = [0]) {
  try {
    const result = await new Promise((accept, reject) => {
      execFile(
        git.path,
        [
          "--no-replace-objects",
          "-c",
          `core.hooksPath=${process.platform === "win32" ? "NUL" : "/dev/null"}`,
          "-c",
          "core.fsmonitor=false",
          "-C",
          repository,
          ...arguments_,
        ],
        {
          cwd: repository,
          encoding: "utf8",
          env: git.environment,
          maxBuffer: 1024 * 1024,
          timeout: 30_000,
          windowsHide: true,
        },
        (error, stdout, stderr) => {
          const status = error === null ? 0 : Number(error.code);
          if (!acceptedStatuses.includes(status)) {
            reject(new Error(`Authenticated Git command failed: ${stderr.trim()}`, {
              cause: error,
            }));
            return;
          }
          accept(Object.freeze({ stdout, status }));
        },
      );
    });
    return result;
  } finally {
    await assertGitUnchanged(git);
  }
}

async function rejectCommandFilters(git, repository, label) {
  const scopes = ["--local"];
  const worktreeConfig = await runGit(git, repository, [
    "config",
    "--local",
    "--get",
    "--bool",
    "extensions.worktreeConfig",
  ], [0, 1]);
  if (worktreeConfig.status === 0) {
    if (worktreeConfig.stdout.trim() !== "true") {
      throw new TypeError(`${label} has invalid worktree-config policy.`);
    }
    scopes.push("--worktree");
  }
  for (const scope of scopes) {
    const result = await runGit(git, repository, [
      "config",
      scope,
      "--null",
      "--name-only",
      "--get-regexp",
      "^filter\\..*\\.(clean|smudge|process)$",
    ], [0, 1]);
    if (result.stdout.length > 0) {
      throw new TypeError(`${label} contains a command-bearing Git filter.`);
    }
  }
}

try {
  const git = await authenticatedGit();
  const gitlink = await runGit(git, root, [
    "ls-files",
    "--stage",
    "--",
    "engine",
  ]);
  const match = /^160000 ([0-9a-f]{40,64}) 0\tengine\r?\n?$/u.exec(
    gitlink.stdout,
  );
  if (match === null) {
    throw new TypeError("The repository index has no initialized engine gitlink.");
  }
  const expectedEngineCommit = match[1];
  const engineHead = (await runGit(git, engine, ["rev-parse", "HEAD"]))
    .stdout.trim();
  if (engineHead !== expectedEngineCommit) {
    throw new TypeError(
      "Engine HEAD does not match the commit recorded by the index gitlink.",
    );
  }
  await rejectCommandFilters(git, root, "Guesser repository");
  await rejectCommandFilters(git, engine, "Engine repository");
  const engineStatus = await runGit(git, engine, [
    "status",
    "--porcelain=v1",
    "--untracked-files=all",
    "--ignore-submodules=none",
  ]);
  if (engineStatus.stdout.length > 0) {
    throw new TypeError("Engine submodule checkout is dirty or inconsistent.");
  }
} catch (error) {
  failures.push(error instanceof Error ? error.message : String(error));
}

for (const [relativePath, packageName] of expected) {
  const path = resolve(root, relativePath);
  try {
    await access(path);
    const manifest = JSON.parse(await readFile(path, "utf8"));
    if (manifest.name !== packageName) {
      failures.push(
        `${relativePath} has package name ${String(manifest.name)}; expected ${packageName}`,
      );
    }
  } catch (error) {
    failures.push(
      `${relativePath}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

if (failures.length > 0) {
  console.error(
    [
      "DrawbackEngine workspace dependency is not ready.",
      ...failures.map((failure) => `- ${failure}`),
      "Populate engine/ with the pinned DrawbackEngine submodule; do not copy rule logic here.",
    ].join("\n"),
  );
  process.exitCode = 2;
} else {
  console.log("DrawbackEngine gitlink and workspace package boundary are ready.");
}
