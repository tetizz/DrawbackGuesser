import { spawnSync } from "node:child_process";
import { lstat, realpath } from "node:fs/promises";
import { basename, isAbsolute } from "node:path";
import process from "node:process";

const WINDOWS_PYTHON_LAUNCHER = String.raw`\\?\GLOBALROOT\SystemRoot\py.exe`;

function isNodeError(error, code) {
  return error instanceof Error && "code" in error && error.code === code;
}

async function readPythonIdentity(candidate) {
  const path = await realpath(candidate);
  if (!isAbsolute(path)) {
    throw new TypeError("Authenticated Python entrypoint is not absolute.");
  }
  const metadata = await lstat(path, { bigint: true });
  if (!metadata.isFile()) {
    throw new TypeError("Authenticated Python entrypoint is not a file.");
  }
  return Object.freeze({ path, metadata });
}

async function closedPythonEnvironment() {
  if (process.platform === "win32") {
    const systemRoot = await realpath(String.raw`\\?\GLOBALROOT\SystemRoot`);
    return Object.freeze({
      SystemRoot: systemRoot,
      WINDIR: systemRoot,
      PATH: "",
      PATHEXT: ".COM;.EXE;.BAT;.CMD",
      PYTHONHASHSEED: "0",
    });
  }
  return Object.freeze({
    LANG: "C.UTF-8",
    LC_ALL: "C.UTF-8",
    PATH: "/usr/bin:/bin",
    PYTHONHASHSEED: "0",
  });
}

async function authenticatedPython(
  environment,
  cwd,
) {
  const configured = process.env.SCHEMA9_PYTHON_EXECUTABLE;
  if (configured !== undefined && !isAbsolute(configured)) {
    throw new TypeError("Configured Schema-9 Python must be absolute.");
  }
  const candidates = configured === undefined
    ? process.platform === "win32"
      ? [WINDOWS_PYTHON_LAUNCHER]
      : ["/usr/bin/python3", "/usr/bin/python"]
    : [configured];
  const paths = new Set();
  for (const candidate of candidates) {
    try {
      paths.add(await realpath(candidate));
    } catch (error) {
      if (!isNodeError(error, "ENOENT")) {
        throw error;
      }
    }
  }
  if (paths.size !== 1) {
    throw new TypeError("Exactly one fixed system Python entrypoint is required.");
  }
  const entrypoint = await readPythonIdentity([...paths][0]);
  if (
    process.platform !== "win32"
    || basename(entrypoint.path).toLocaleLowerCase("en-US") !== "py.exe"
  ) {
    return entrypoint;
  }
  const result = spawnSync(entrypoint.path, [
    "-3",
    "-B",
    "-s",
    "-S",
    "-P",
    "-c",
    "import os, sys; print(os.path.realpath(sys.executable), end='')",
  ], {
    cwd,
    encoding: "utf8",
    env: environment,
    maxBuffer: 64 * 1024,
    timeout: 30_000,
    windowsHide: true,
  });
  await assertPythonUnchanged(entrypoint);
  if (result.error !== undefined) {
    throw new Error("Python interpreter discovery failed.", {
      cause: result.error,
    });
  }
  const selected = result.stdout.trim();
  if (
    result.status !== 0
    || selected.length === 0
    || selected.includes("\n")
    || selected.includes("\r")
    || !isAbsolute(selected)
  ) {
    throw new Error("Python launcher returned an invalid interpreter identity.");
  }
  return readPythonIdentity(selected);
}

async function assertPythonUnchanged(expected) {
  const actual = await lstat(expected.path, { bigint: true });
  if (
    !actual.isFile()
    || actual.dev !== expected.metadata.dev
    || actual.ino !== expected.metadata.ino
    || actual.size !== expected.metadata.size
    || actual.mtimeNs !== expected.metadata.mtimeNs
    || actual.ctimeNs !== expected.metadata.ctimeNs
  ) {
    throw new Error("Authenticated Python entrypoint changed during use.");
  }
}

export async function runIsolatedPython(
  command,
  importPaths,
  cwd,
  timeoutMilliseconds = 30_000,
) {
  if (
    !Number.isSafeInteger(timeoutMilliseconds)
    || timeoutMilliseconds <= 0
  ) {
    throw new RangeError("Python inspection timeout must be positive.");
  }
  if (!isAbsolute(cwd) || importPaths.some((path) => !isAbsolute(path))) {
    throw new TypeError("Python inspection paths must be absolute.");
  }
  const environment = await closedPythonEnvironment();
  const python = await authenticatedPython(
    environment,
    cwd,
  );
  const bootstrap = [
    "import os",
    "import sys",
    `_schema9_expected_python = ${JSON.stringify(python.path)}`,
    "if os.path.normcase(os.path.realpath(sys.executable)) != os.path.normcase(os.path.realpath(_schema9_expected_python)):",
    "  raise RuntimeError('authenticated Python interpreter changed')",
    "_schema9_python_controls = {name: value for name, value in os.environ.items() if name.upper().startswith('PYTHON')}",
    "if _schema9_python_controls != {'PYTHONHASHSEED': '0'}:",
    "  raise RuntimeError('closed Python controls are invalid')",
    "if not (sys.flags.isolated == 0 and sys.flags.ignore_environment == 0 and sys.flags.no_site == 1 and sys.flags.no_user_site == 1 and sys.flags.safe_path == 1 and sys.flags.dont_write_bytecode == 1 and sys.flags.hash_randomization == 0):",
    "  raise RuntimeError('closed Python flags are invalid')",
    "if 'sitecustomize' in sys.modules or 'usercustomize' in sys.modules:",
    "  raise RuntimeError('Python startup customization was loaded')",
    "sys.dont_write_bytecode = True",
    ...[...importPaths].reverse().map((path) =>
      `sys.path.insert(0, ${JSON.stringify(path)})`
    ),
    command,
  ].join("\n");
  const arguments_ = [
    "-B",
    "-s",
    "-S",
    "-P",
    "-c",
    bootstrap,
  ];
  const result = spawnSync(python.path, arguments_, {
    cwd,
    encoding: "utf8",
    env: environment,
    maxBuffer: 1024 * 1024,
    timeout: timeoutMilliseconds,
    windowsHide: true,
  });
  await assertPythonUnchanged(python);
  if (result.error !== undefined) {
    if (isNodeError(result.error, "ETIMEDOUT")) {
      throw new Error("Python inspection exceeded its time limit.", {
        cause: result.error,
      });
    }
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`Python inspection failed: ${result.stderr.trim()}`);
  }
  return result.stdout;
}
