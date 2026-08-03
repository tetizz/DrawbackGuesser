import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { realpathSync } from "node:fs";
import {
  mkdtemp,
  readFile,
  realpath,
  rm,
  stat,
} from "node:fs/promises";
import { registerHooks } from "node:module";
import { tmpdir } from "node:os";
import {
  basename,
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
} from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  canonicalJsonBytes,
  createSchema9LedgerVerificationReceipt,
  loadAndReauthenticateSchema9CorpusLedger,
  schema9CorpusLedgerFileSha256,
  schema9AssignmentScheduler,
  SCHEMA9_CORPUS_LEDGER_FORMAT,
  SCHEMA9_CORPUS_LEDGER_VERSION,
  SCHEMA9_EXECUTION_MANIFEST_ALGORITHM,
  SCHEMA9_LEDGER_SPLITS,
  SCHEMA9_PRODUCER_CONVERTER_POLICIES,
  writeSchema9CorpusLedgerAtomic,
  writeSchema9LedgerVerificationReceiptAtomic,
  type Schema9CorpusLedgerOptions,
  type Schema9ExecutionIdentity,
  type Schema9LedgerSplit,
  type Schema9ProducerConverterPolicy,
  type Schema9RepositoryVerifier,
  type Schema9SplitFiles,
} from "@drawbackguesser/trace-to-dataset";

export interface Schema9LedgerCliOptions {
  readonly operation: "create" | "verify";
  readonly ledgerPath: string;
  readonly guesserRepository: string;
  readonly engineRepository: string;
  readonly corpus: Omit<
    Schema9CorpusLedgerOptions,
    "repositoryVerifier" | "assignmentScheduler"
  >;
}

export interface Schema9LedgerCliIo {
  readonly stdout: { write(chunk: string): unknown };
}

const GLOBAL_FLAGS = Object.freeze([
  "--operation",
  "--ledger",
  "--guesser-repository",
  "--engine-repository",
  "--guesser-commit",
  "--converter-engine-commit",
  "--producer-converter-policy",
] as const);
const SPLIT_FIELDS = Object.freeze([
  "trace",
  "converted",
  "launch-receipt",
  "completion-receipt",
  "schedule-id",
  "label-seed-root",
  "gameplay-seed-root",
  "parameters-seed-root",
  "producer-engine-commit",
] as const);

function splitFlag(split: Schema9LedgerSplit, field: string): string {
  return `--${split}-${field}`;
}

function valueAfter(
  values: ReadonlyMap<string, string>,
  flag: string,
): string {
  const value = values.get(flag);
  if (value === undefined) {
    throw new TypeError(`${flag} is required.`);
  }
  return value;
}

function splitFiles(
  values: ReadonlyMap<string, string>,
  split: Schema9LedgerSplit,
  invocationDirectory: string,
): Schema9SplitFiles {
  const unsignedRoot = (field: string): number => {
    const raw = valueAfter(values, splitFlag(split, field));
    if (!/^(?:0|[1-9][0-9]{0,9})$/u.test(raw)) {
      throw new TypeError(
        `${splitFlag(split, field)} must be a canonical unsigned integer.`,
      );
    }
    const parsed = Number(raw);
    if (!Number.isSafeInteger(parsed) || parsed > 0xffff_ffff) {
      throw new RangeError(
        `${splitFlag(split, field)} must fit an unsigned 32-bit integer.`,
      );
    }
    return parsed;
  };
  return Object.freeze({
    tracePath: resolve(
      invocationDirectory,
      valueAfter(values, splitFlag(split, "trace")),
    ),
    convertedPath: resolve(
      invocationDirectory,
      valueAfter(values, splitFlag(split, "converted")),
    ),
    launchReceiptPath: resolve(
      invocationDirectory,
      valueAfter(values, splitFlag(split, "launch-receipt")),
    ),
    completionReceiptPath: resolve(
      invocationDirectory,
      valueAfter(values, splitFlag(split, "completion-receipt")),
    ),
    scheduleId: valueAfter(values, splitFlag(split, "schedule-id")),
    seedRoots: Object.freeze([
      unsignedRoot("label-seed-root"),
      unsignedRoot("gameplay-seed-root"),
      unsignedRoot("parameters-seed-root"),
    ] as const),
    producerEngineCommit: valueAfter(
      values,
      splitFlag(split, "producer-engine-commit"),
    ),
  });
}

export function parseSchema9LedgerCliArguments(
  arguments_: readonly string[],
  invocationDirectory = process.cwd(),
): Schema9LedgerCliOptions {
  if (arguments_.length % 2 !== 0) {
    throw new TypeError("Every schema-9 ledger flag requires one value.");
  }
  const supported = new Set<string>(GLOBAL_FLAGS);
  for (const split of SCHEMA9_LEDGER_SPLITS) {
    for (const field of SPLIT_FIELDS) {
      supported.add(splitFlag(split, field));
    }
  }
  const values = new Map<string, string>();
  for (let index = 0; index < arguments_.length; index += 2) {
    const flag = arguments_[index];
    const value = arguments_[index + 1];
    if (
      flag === undefined
      || !supported.has(flag)
      || value === undefined
      || value.startsWith("--")
    ) {
      throw new TypeError(
        `Unsupported or incomplete schema-9 ledger flag: ${flag ?? ""}.`,
      );
    }
    if (values.has(flag)) {
      throw new TypeError(`${flag} may appear only once.`);
    }
    values.set(flag, value);
  }
  if (values.size !== supported.size) {
    const missing = [...supported].filter((flag) => !values.has(flag));
    throw new TypeError(
      `Missing schema-9 ledger flags: ${missing.join(", ")}.`,
    );
  }
  const operation = valueAfter(values, "--operation");
  if (operation !== "create" && operation !== "verify") {
    throw new TypeError("--operation must be create or verify.");
  }
  const rawPolicy = valueAfter(values, "--producer-converter-policy");
  if (
    !SCHEMA9_PRODUCER_CONVERTER_POLICIES.includes(
      rawPolicy as Schema9ProducerConverterPolicy,
    )
  ) {
    throw new TypeError(
      "--producer-converter-policy must be exact/v1 "
      + "or converter-ancestor/v1.",
    );
  }
  const producerConverterPolicy =
    rawPolicy as Schema9ProducerConverterPolicy;
  const splits = Object.fromEntries(
    SCHEMA9_LEDGER_SPLITS.map((split) => [
      split,
      splitFiles(values, split, invocationDirectory),
    ]),
  ) as Readonly<Record<Schema9LedgerSplit, Schema9SplitFiles>>;
  return Object.freeze({
    operation,
    ledgerPath: resolve(
      invocationDirectory,
      valueAfter(values, "--ledger"),
    ),
    guesserRepository: resolve(
      invocationDirectory,
      valueAfter(values, "--guesser-repository"),
    ),
    engineRepository: resolve(
      invocationDirectory,
      valueAfter(values, "--engine-repository"),
    ),
    corpus: Object.freeze({
      guesserCommit: valueAfter(values, "--guesser-commit"),
      converterEngineCommit: valueAfter(
        values,
        "--converter-engine-commit",
      ),
      producerConverterPolicy,
      splits,
    }),
  });
}

interface GitResult {
  readonly stdout: string;
}

function git(
  repository: string,
  arguments_: readonly string[],
): Promise<GitResult> {
  const excluded = new Set([
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CEILING_DIRECTORIES",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_SYSTEM",
    "GIT_NAMESPACE",
    "GIT_REPLACE_REF_BASE",
  ]);
  const environment: NodeJS.ProcessEnv = Object.fromEntries(
    Object.entries(process.env).filter(([name]) =>
      !excluded.has(name)
      && !name.startsWith("GIT_CONFIG_KEY_")
      && !name.startsWith("GIT_CONFIG_VALUE_")
    ),
  );
  environment["GIT_CONFIG_NOSYSTEM"] = "1";
  environment["GIT_CONFIG_GLOBAL"] = process.platform === "win32"
    ? "NUL"
    : "/dev/null";
  return new Promise((accept, reject) => {
    execFile(
      "git",
      ["--no-replace-objects", "-C", repository, ...arguments_],
      {
        encoding: "utf8",
        env: environment,
        windowsHide: true,
        maxBuffer: 1024 * 1024,
      },
      (error, stdout) => {
        if (error !== null) {
          reject(
            error instanceof Error
              ? error
              : new Error("Git process failed.", { cause: error }),
          );
          return;
        }
        accept({ stdout });
      },
    );
  });
}

async function optionalGitMatch(
  repository: string,
  arguments_: readonly string[],
): Promise<string> {
  try {
    return (await git(repository, arguments_)).stdout;
  } catch (error: unknown) {
    if (exitCode(error) === 1) {
      return "";
    }
    throw error;
  }
}

async function requireNoIndexIgnoreFlags(
  repository: string,
  label: string,
): Promise<void> {
  const tagged = (await git(repository, ["ls-files", "-v", "-z"])).stdout;
  for (const entry of tagged.split("\0")) {
    if (entry.length === 0) {
      continue;
    }
    const tag = entry[0];
    if (
      tag === "S"
      || (tag !== undefined && /^[a-z]$/u.test(tag))
    ) {
      throw new TypeError(
        `${label} contains forbidden skip-worktree or assume-unchanged flags.`,
      );
    }
  }
}

function hasWeakSubmoduleIgnoreConfiguration(output: string): boolean {
  return output.split("\0").some((record) => {
    const normalized = record.trim().toLocaleLowerCase("en-US");
    return normalized.includes(".ignore\n")
      || normalized.startsWith("diff.ignoresubmodules\n");
  });
}

export async function requireNoSubmoduleIgnoreConfiguration(
  repository: string,
  expectedHead: string,
  label: string,
): Promise<void> {
  const local = await optionalGitMatch(repository, [
    "config",
    "--local",
    "--null",
    "--get-regexp",
    "^(submodule\\..*\\.ignore|diff\\.ignoreSubmodules)$",
  ]);
  const worktreeConfigEnabled = (await optionalGitMatch(repository, [
    "config",
    "--local",
    "--type=bool",
    "--get",
    "extensions.worktreeConfig",
  ])).trim() === "true";
  const worktree = worktreeConfigEnabled
    ? await optionalGitMatch(repository, [
      "config",
      "--worktree",
      "--null",
      "--get-regexp",
      "^(submodule\\..*\\.ignore|diff\\.ignoreSubmodules)$",
    ])
    : "";
  const gitmodulesEntry = (await git(repository, [
    "ls-tree",
    "--name-only",
    expectedHead,
    "--",
    ".gitmodules",
  ])).stdout.trim();
  const committed = gitmodulesEntry === ".gitmodules"
    ? await optionalGitMatch(repository, [
      "config",
      "--blob",
      `${expectedHead}:.gitmodules`,
      "--null",
      "--get-regexp",
      "^submodule\\..*\\.ignore$",
    ])
    : "";
  if (
    hasWeakSubmoduleIgnoreConfiguration(local)
    || hasWeakSubmoduleIgnoreConfiguration(worktree)
    || hasWeakSubmoduleIgnoreConfiguration(committed)
  ) {
    throw new TypeError(
      `${label} contains forbidden submodule ignore configuration.`,
    );
  }
}

async function requireCleanCheckout(
  repository: string,
  expectedHead: string,
  label: string,
): Promise<void> {
  const head = (await git(repository, ["rev-parse", "HEAD"])).stdout.trim();
  if (head !== expectedHead) {
    throw new TypeError(`${label} HEAD does not match the declared commit.`);
  }
  const status = await git(repository, [
    "status",
    "--porcelain=v1",
    "--untracked-files=no",
    "--ignore-submodules=none",
  ]);
  if (status.stdout.length !== 0) {
    throw new TypeError(`${label} tracked worktree or index is dirty.`);
  }
  const replaceRefs = await git(repository, [
    "for-each-ref",
    "--format=%(refname)",
    "refs/replace",
  ]);
  if (replaceRefs.stdout.trim().length !== 0) {
    throw new TypeError(`${label} contains forbidden Git replace refs.`);
  }
  await requireNoIndexIgnoreFlags(repository, label);
  await requireNoSubmoduleIgnoreConfiguration(
    repository,
    expectedHead,
    label,
  );
}

interface ManifestFile {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

interface ModuleGraphManifest {
  readonly identity: {
    readonly entrypoint: string;
    readonly files: number;
    readonly bytes: number;
    readonly sha256: string;
  };
  readonly paths: readonly string[];
}

interface Schema9ExecutionEntrypoints {
  readonly parser: string;
  readonly converter: string;
  readonly scheduler: string;
  readonly verifier: string;
}

function pathEscapesRoot(root: string, candidate: string): boolean {
  const normalized = relative(root, candidate);
  return normalized === ".."
    || normalized.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`)
    || isAbsolute(normalized);
}

export async function assertEngineRuntimeCheckout(
  guesserRepository: string,
  engineRepository: string,
  runtimeEngineEntrypoint: string,
): Promise<void> {
  const guesserRoot = await realpath(guesserRepository);
  const expectedEngineRoot = await realpath(join(guesserRoot, "engine"));
  const suppliedEngineRoot = await realpath(engineRepository);
  if (relative(expectedEngineRoot, suppliedEngineRoot) !== "") {
    throw new TypeError(
      "Explicit Engine repository is not the Guesser engine submodule checkout.",
    );
  }
  if (!runtimeEngineEntrypoint.startsWith("file:")) {
    throw new TypeError("Engine runtime entrypoint must be a file URL.");
  }
  const runtimePath = await realpath(fileURLToPath(runtimeEngineEntrypoint));
  if (pathEscapesRoot(expectedEngineRoot, runtimePath)) {
    throw new TypeError(
      "Loaded Engine runtime does not come from the Guesser engine submodule.",
    );
  }
}

async function assertExplicitEngineSubmoduleCheckout(
  guesserRepository: string,
  engineRepository: string,
): Promise<void> {
  const guesserRoot = await realpath(guesserRepository);
  const expectedEngineRoot = await realpath(join(guesserRoot, "engine"));
  const suppliedEngineRoot = await realpath(engineRepository);
  if (relative(expectedEngineRoot, suppliedEngineRoot) !== "") {
    throw new TypeError(
      "Explicit Engine repository is not the Guesser engine submodule checkout.",
    );
  }
}

const REPRODUCIBLE_BUILD_PREFIX = "schema9-reproducible-build-";

export async function withSchema9TemporaryCheckout<T>(
  operation: (checkoutRoot: string) => Promise<T>,
  temporaryParent = tmpdir(),
): Promise<T> {
  const parent = await realpath(temporaryParent);
  const created = await mkdtemp(join(parent, REPRODUCIBLE_BUILD_PREFIX));
  let outcome:
    | { readonly ok: true; readonly value: T }
    | { readonly ok: false; readonly error: unknown };
  try {
    const checkoutRoot = await realpath(created);
    if (
      relative(parent, dirname(checkoutRoot)) !== ""
      || !basename(checkoutRoot).startsWith(REPRODUCIBLE_BUILD_PREFIX)
    ) {
      throw new TypeError("Temporary schema-9 checkout escaped its parent.");
    }
    outcome = Object.freeze({ ok: true, value: await operation(checkoutRoot) });
  } catch (error: unknown) {
    outcome = Object.freeze({ ok: false, error });
  }
  try {
    if (
      relative(parent, dirname(created)) !== ""
      || !basename(created).startsWith(REPRODUCIBLE_BUILD_PREFIX)
    ) {
      throw new TypeError("Temporary schema-9 cleanup target is invalid.");
    }
    await rm(created, {
      recursive: true,
      force: true,
      maxRetries: 3,
      retryDelay: 50,
    });
  } catch (cleanupError: unknown) {
    if (!outcome.ok) {
      throw new AggregateError(
        [outcome.error, cleanupError],
        "Schema-9 reproducible checkout and cleanup both failed.",
      );
    }
    throw cleanupError;
  }
  if (!outcome.ok) {
    throw outcome.error;
  }
  return outcome.value;
}

function sameFileSignature(
  left: Awaited<ReturnType<typeof stat>>,
  right: Awaited<ReturnType<typeof stat>>,
): boolean {
  return left.dev === right.dev
    && left.ino === right.ino
    && left.size === right.size
    && left.mtimeMs === right.mtimeMs
    && left.ctimeMs === right.ctimeMs;
}

const MODULE_GRAPH_PROBE = String.raw`
import { registerHooks } from "node:module";
import { fileURLToPath } from "node:url";
const entryUrl = process.env.SCHEMA9_MODULE_GRAPH_ENTRY;
if (entryUrl === undefined || !entryUrl.startsWith("file:")) {
  throw new TypeError("Module graph entry URL is invalid.");
}
const visited = new Set();
function record(url) {
  if (url.startsWith("file:")) {
    visited.add(fileURLToPath(url));
    return;
  }
  if (!url.startsWith("node:")) {
    throw new TypeError("Module graph tracing rejected a non-file module.");
  }
}
const hooks = registerHooks({
  resolve(specifier, context, nextResolve) {
    const result = nextResolve(specifier, context);
    record(result.url);
    return result;
  },
  load(url, context, nextLoad) {
    record(url);
    return nextLoad(url, context);
  },
});
try {
  await import(entryUrl);
} finally {
  hooks.deregister();
}
if (visited.size === 0) {
  throw new TypeError("Module graph tracing did not load the entry module.");
}
process.stdout.write(JSON.stringify([...visited]) + "\n");
`;

const SCHEMA9_RUNTIME_ENVIRONMENT_KEYS = Object.freeze([
  "NODE_OPTIONS",
  "NODE_PATH",
  "NODE_PRESERVE_SYMLINKS",
] as const);

export function schema9RuntimeIdentity(
  environment: NodeJS.ProcessEnv = process.env,
  execArgv: readonly string[] = process.execArgv,
) {
  const activeEnvironment = SCHEMA9_RUNTIME_ENVIRONMENT_KEYS.filter(
    (name) => {
      const value = environment[name];
      return value !== undefined && value.trim().length > 0;
    },
  );
  if (activeEnvironment.length > 0 || execArgv.length > 0) {
    throw new TypeError(
      "Schema-9 ledger verification requires a direct Node invocation "
      + "without runtime hooks, search-path overrides, or execution flags.",
    );
  }
  return Object.freeze({
    nodeVersion: process.version,
    platform: process.platform,
    architecture: process.arch,
    execArgv: Object.freeze([] as string[]),
  });
}

async function discoverRuntimeModulePaths(
  entryUrl: string,
): Promise<readonly string[]> {
  const environment: NodeJS.ProcessEnv = { ...process.env };
  delete environment["NODE_OPTIONS"];
  delete environment["NODE_PATH"];
  delete environment["NODE_PRESERVE_SYMLINKS"];
  environment["SCHEMA9_MODULE_GRAPH_ENTRY"] = entryUrl;
  const output = await new Promise<string>((accept, reject) => {
    execFile(
      process.execPath,
      [
        "--input-type=module",
        "--eval",
        MODULE_GRAPH_PROBE,
      ],
      {
        encoding: "utf8",
        env: environment,
        windowsHide: true,
        maxBuffer: 16 * 1024 * 1024,
      },
      (error, stdout) => {
        if (error !== null) {
          reject(
            error instanceof Error
              ? error
              : new Error("Module graph probe failed.", { cause: error }),
          );
          return;
        }
        accept(stdout);
      },
    );
  });
  const parsed: unknown = JSON.parse(output);
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new TypeError("Module graph probe returned an invalid file set.");
  }
  const paths: string[] = [];
  for (const value of parsed) {
    if (typeof value !== "string") {
      throw new TypeError("Module graph probe returned an invalid file set.");
    }
    paths.push(value);
  }
  if (new Set(paths).size !== paths.length) {
    throw new TypeError("Module graph probe returned an invalid file set.");
  }
  return paths;
}

function moduleIdentityBytes(filePath: string, bytes: Buffer): Buffer {
  if (basename(filePath) !== "package.json" || !bytes.includes(13)) {
    return bytes;
  }
  const normalized = Buffer.allocUnsafe(bytes.byteLength);
  let written = 0;
  for (let index = 0; index < bytes.byteLength; index += 1) {
    if (bytes[index] === 13 && bytes[index + 1] === 10) {
      continue;
    }
    normalized[written] = bytes[index] as number;
    written += 1;
  }
  return normalized.subarray(0, written);
}

async function moduleGraphManifest(
  entrypoint: string,
  entryUrl: string,
  repositoryRoot: string,
): Promise<ModuleGraphManifest> {
  const root = await realpath(repositoryRoot);
  const discovered = await discoverRuntimeModulePaths(entryUrl);
  const graphPaths = new Set<string>();
  for (const discoveredPath of discovered) {
    const resolved = await realpath(discoveredPath);
    graphPaths.add(resolved);
    let directory = dirname(resolved);
    for (;;) {
      const relativeDirectory = relative(root, directory);
      if (
        relativeDirectory.startsWith("..")
        || isAbsolute(relativeDirectory)
      ) {
        break;
      }
      const packageManifest = join(directory, "package.json");
      try {
        const manifestInfo = await stat(packageManifest);
        if (manifestInfo.isFile()) {
          graphPaths.add(await realpath(packageManifest));
        }
      } catch (error: unknown) {
        if (
          typeof error !== "object"
          || error === null
          || !("code" in error)
          || error.code !== "ENOENT"
        ) {
          throw error;
        }
      }
      if (directory === root) {
        break;
      }
      directory = dirname(directory);
    }
  }
  const files: ManifestFile[] = [];
  for (const discoveredPath of graphPaths) {
    const current = await realpath(discoveredPath);
    const before = await stat(current);
    const bytes = await readFile(current);
    const after = await stat(current);
    if (!sameFileSignature(before, after)) {
      throw new TypeError("Execution module changed while it was hashed.");
    }
    const identityBytes = moduleIdentityBytes(current, bytes);
    const normalized = relative(root, current).replaceAll("\\", "/");
    if (
      normalized.startsWith("../")
      || normalized === ".."
      || isAbsolute(normalized)
    ) {
      throw new TypeError("Execution module graph escaped the Guesser checkout.");
    }
    files.push(Object.freeze({
      path: normalized,
      bytes: identityBytes.byteLength,
      sha256: createHash("sha256").update(identityBytes).digest("hex"),
    }));
  }
  files.sort((left, right) => left.path.localeCompare(right.path, "en"));
  const payload = canonicalJsonBytes(files);
  return Object.freeze({
    identity: Object.freeze({
      entrypoint,
      files: files.length,
      bytes: files.reduce((total, file) => total + file.bytes, 0),
      sha256: createHash("sha256").update(payload).digest("hex"),
    }),
    paths: Object.freeze([...graphPaths]),
  });
}

export async function moduleGraphIdentity(
  entrypoint: string,
  entryUrl: string,
  repositoryRoot: string,
): Promise<ModuleGraphManifest["identity"]> {
  return (await moduleGraphManifest(
    entrypoint,
    entryUrl,
    repositoryRoot,
  )).identity;
}

export function createSchema9ModuleLoadGuard(
  allowedPaths: Iterable<string>,
) {
  const allowed = new Set(
    [...allowedPaths].map((filePath) => realpathSync(filePath)),
  );
  if (allowed.size === 0) {
    throw new TypeError("Schema-9 module load guard requires known modules.");
  }
  return registerHooks({
    resolve(specifier, context, nextResolve) {
      const result = nextResolve(specifier, context);
      if (result.url.startsWith("node:")) {
        return result;
      }
      if (!result.url.startsWith("file:")) {
        throw new TypeError(
          "Schema-9 verification rejected a non-file runtime module.",
        );
      }
      const resolved = realpathSync(fileURLToPath(result.url));
      if (!allowed.has(resolved)) {
        throw new TypeError(
          "Schema-9 verification rejected a deferred runtime module load.",
        );
      }
      return result;
    },
  });
}

function currentExecutionEntrypoints(): Schema9ExecutionEntrypoints {
  return Object.freeze({
    parser: import.meta.resolve("@drawbackengine/simulation-trace"),
    converter: import.meta.resolve("@drawbackguesser/trace-to-dataset"),
    scheduler: import.meta.resolve("@drawbackguesser/trace-to-dataset"),
    verifier: import.meta.url,
  });
}

function builtExecutionEntrypoints(
  repositoryRoot: string,
): Schema9ExecutionEntrypoints {
  const asUrl = (path: string): string => pathToFileURL(path).href;
  return Object.freeze({
    parser: asUrl(join(
      repositoryRoot,
      "engine",
      "packages",
      "simulation-trace",
      "dist",
      "index.js",
    )),
    converter: asUrl(join(
      repositoryRoot,
      "packages",
      "trace-to-dataset",
      "dist",
      "index.js",
    )),
    scheduler: asUrl(join(
      repositoryRoot,
      "packages",
      "trace-to-dataset",
      "dist",
      "index.js",
    )),
    verifier: asUrl(join(
      repositoryRoot,
      "apps",
      "dataset-cli",
      "dist",
      "schema9-ledger-cli.js",
    )),
  });
}

async function executingCodeManifest(
  repositoryRoot: string,
  entrypoints: Schema9ExecutionEntrypoints = currentExecutionEntrypoints(),
) {
  const runtime = schema9RuntimeIdentity();
  const manifests = await Promise.all([
    moduleGraphManifest(
      "simulation-trace",
      entrypoints.parser,
      repositoryRoot,
    ),
    moduleGraphManifest(
      "trace-to-dataset",
      entrypoints.converter,
      repositoryRoot,
    ),
    moduleGraphManifest(
      "schema9-schedule-replay",
      entrypoints.scheduler,
      repositoryRoot,
    ),
    moduleGraphManifest(
      "schema9-ledger-cli",
      entrypoints.verifier,
      repositoryRoot,
    ),
  ]);
  const [parser, converter, scheduler, verifier] = manifests.map(
    (manifest) => manifest.identity,
  ) as [
    ModuleGraphManifest["identity"],
    ModuleGraphManifest["identity"],
    ModuleGraphManifest["identity"],
    ModuleGraphManifest["identity"],
  ];
  const payload = Object.freeze({
    algorithm: SCHEMA9_EXECUTION_MANIFEST_ALGORITHM,
    runtime,
    parser,
    converter,
    scheduler,
    verifier,
  });
  return Object.freeze({
    identity: Object.freeze({
      ...payload,
      aggregateSha256: createHash("sha256")
        .update(canonicalJsonBytes(payload))
        .digest("hex"),
    }),
    paths: Object.freeze(manifests.flatMap((manifest) => manifest.paths)),
  });
}

export function assertExactReproducibleExecutionIdentity(
  executing: Schema9ExecutionIdentity,
  reproduced: Schema9ExecutionIdentity,
): void {
  if (!canonicalJsonBytes(executing).equals(canonicalJsonBytes(reproduced))) {
    throw new TypeError(
      "Executing schema-9 module graph does not match the reproducible build.",
    );
  }
}

interface CommandResult {
  readonly stdout: string;
}

function runBoundedCommand(
  command: string,
  arguments_: readonly string[],
  cwd: string,
  environment: NodeJS.ProcessEnv,
): Promise<CommandResult> {
  return new Promise((accept, reject) => {
    execFile(
      command,
      [...arguments_],
      {
        cwd,
        encoding: "utf8",
        env: environment,
        windowsHide: true,
        timeout: 10 * 60 * 1000,
        maxBuffer: 16 * 1024 * 1024,
      },
      (error, stdout) => {
        if (error !== null) {
          reject(new Error("Schema-9 reproducible build command failed.", {
            cause: error,
          }));
          return;
        }
        accept(Object.freeze({ stdout }));
      },
    );
  });
}

function reproducibleBuildEnvironment(): NodeJS.ProcessEnv {
  const excluded = new Set([
    "NODE_OPTIONS",
    "NODE_PATH",
    "NODE_PRESERVE_SYMLINKS",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CEILING_DIRECTORIES",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_SYSTEM",
    "GIT_NAMESPACE",
    "GIT_REPLACE_REF_BASE",
  ]);
  const environment: NodeJS.ProcessEnv = Object.fromEntries(
    Object.entries(process.env).filter(([name]) =>
      !excluded.has(name)
      && !name.startsWith("GIT_CONFIG_KEY_")
      && !name.startsWith("GIT_CONFIG_VALUE_")
    ),
  );
  const nullDevice = process.platform === "win32" ? "NUL" : "/dev/null";
  environment["CI"] = "true";
  environment["COREPACK_ENABLE_DOWNLOAD_PROMPT"] = "0";
  environment["GIT_CONFIG_NOSYSTEM"] = "1";
  environment["GIT_CONFIG_GLOBAL"] = nullDevice;
  environment["NPM_CONFIG_GLOBALCONFIG"] = nullDevice;
  environment["NPM_CONFIG_USERCONFIG"] = nullDevice;
  environment["npm_config_offline"] = "true";
  environment["npm_config_ignore_scripts"] = "true";
  return environment;
}

async function pnpmEntrypoint(): Promise<string> {
  const configured = process.env["npm_execpath"];
  if (configured === undefined || configured.length === 0) {
    throw new TypeError(
      "Schema-9 reproducible verification must be launched through pnpm.",
    );
  }
  const resolved = await realpath(configured);
  if (!(await stat(resolved)).isFile()) {
    throw new TypeError("The active pnpm entrypoint is not a regular file.");
  }
  return resolved;
}

async function requiredPnpmVersion(repositoryRoot: string): Promise<string> {
  const parsed: unknown = JSON.parse(
    await readFile(join(repositoryRoot, "package.json"), "utf8"),
  );
  if (
    typeof parsed !== "object"
    || parsed === null
    || !("packageManager" in parsed)
    || typeof parsed.packageManager !== "string"
  ) {
    throw new TypeError("Guesser packageManager declaration is invalid.");
  }
  const match = /^pnpm@([0-9]+\.[0-9]+\.[0-9]+)$/u.exec(
    parsed.packageManager,
  );
  if (match?.[1] === undefined) {
    throw new TypeError("Guesser must pin one exact pnpm version.");
  }
  return match[1];
}

async function installAndBuildReproducibleCheckout(
  repositoryRoot: string,
): Promise<void> {
  const entrypoint = await pnpmEntrypoint();
  const before = await stat(entrypoint);
  const beforeBytes = await readFile(entrypoint);
  const environment = reproducibleBuildEnvironment();
  const invoke = (arguments_: readonly string[]) =>
    runBoundedCommand(
      process.execPath,
      [entrypoint, ...arguments_],
      repositoryRoot,
      environment,
    );
  const requiredVersion = await requiredPnpmVersion(repositoryRoot);
  const actualVersion = (await invoke(["--version"])).stdout.trim();
  if (actualVersion !== requiredVersion) {
    throw new TypeError("Active pnpm version does not match packageManager.");
  }
  await invoke([
    "install",
    "--offline",
    "--frozen-lockfile",
    "--ignore-scripts",
  ]);
  await invoke(["run", "engine:build"]);
  await invoke([
    "--filter",
    "@drawbackguesser/dataset-cli",
    "run",
    "build",
  ]);
  const after = await stat(entrypoint);
  const afterBytes = await readFile(entrypoint);
  if (
    !sameFileSignature(before, after)
    || !beforeBytes.equals(afterBytes)
  ) {
    throw new TypeError("The pnpm entrypoint changed during reproduction.");
  }
}

export async function cloneRepositoryAtCommit(
  sourceRepository: string,
  targetRepository: string,
  commit: string,
): Promise<void> {
  await git(sourceRepository, [
    "clone",
    "--no-checkout",
    "--no-hardlinks",
    "--no-tags",
    "--local",
    "--",
    sourceRepository,
    targetRepository,
  ]);
  await git(targetRepository, [
    "checkout",
    "--detach",
    "--force",
    commit,
  ]);
}

export interface Schema9ReproducibleBuildRequest {
  readonly guesserRepository: string;
  readonly engineRepository: string;
  readonly guesserCommit: string;
  readonly engineCommit: string;
  readonly temporaryParent?: string;
}

export async function reproduceSchema9ExecutionIdentity(
  request: Schema9ReproducibleBuildRequest,
): Promise<Schema9ExecutionIdentity> {
  const guesserRepository = await realpath(request.guesserRepository);
  const engineRepository = await realpath(request.engineRepository);
  await requireCommit(guesserRepository, request.guesserCommit);
  await requireCommit(engineRepository, request.engineCommit);
  return withSchema9TemporaryCheckout(async (checkoutRoot) => {
    const guesserCheckout = join(checkoutRoot, "guesser");
    await cloneRepositoryAtCommit(
      guesserRepository,
      guesserCheckout,
      request.guesserCommit,
    );
    const engineCheckout = join(guesserCheckout, "engine");
    if (pathEscapesRoot(guesserCheckout, engineCheckout)) {
      throw new TypeError("Temporary Engine checkout path is invalid.");
    }
    await rm(engineCheckout, { recursive: true, force: true });
    await cloneRepositoryAtCommit(
      engineRepository,
      engineCheckout,
      request.engineCommit,
    );
    await requireCleanCheckout(
      guesserCheckout,
      request.guesserCommit,
      "Reproduced Guesser repository",
    );
    await requireCleanCheckout(
      engineCheckout,
      request.engineCommit,
      "Reproduced Engine repository",
    );
    await installAndBuildReproducibleCheckout(guesserCheckout);
    await requireCleanCheckout(
      guesserCheckout,
      request.guesserCommit,
      "Reproduced Guesser repository",
    );
    await requireCleanCheckout(
      engineCheckout,
      request.engineCommit,
      "Reproduced Engine repository",
    );
    const entrypoints = builtExecutionEntrypoints(guesserCheckout);
    await assertEngineRuntimeCheckout(
      guesserCheckout,
      engineCheckout,
      entrypoints.parser,
    );
    return (await executingCodeManifest(
      guesserCheckout,
      entrypoints,
    )).identity;
  }, request.temporaryParent);
}


async function requireCommit(
  repository: string,
  commit: string,
): Promise<void> {
  await git(repository, ["cat-file", "-e", `${commit}^{commit}`]);
}

function exitCode(error: unknown): number | undefined {
  if (
    typeof error === "object"
    && error !== null
    && "code" in error
    && typeof error.code === "number"
  ) {
    return error.code;
  }
  return undefined;
}

export interface Schema9GitRepositoryVerifier extends Schema9RepositoryVerifier {
  readonly disposeExecutionGuard: () => void;
}

export interface Schema9GitRepositoryVerifierOptions {
  readonly temporaryParent?: string;
  readonly reproduceExecutionIdentity?: (
    request: Schema9ReproducibleBuildRequest,
  ) => Promise<Schema9ExecutionIdentity>;
}

export function createGitRepositoryVerifier(
  guesserRepository: string,
  engineRepository: string,
  options: Schema9GitRepositoryVerifierOptions = {},
): Schema9GitRepositoryVerifier {
  let executionGuard: ReturnType<typeof registerHooks> | undefined;
  let authenticatedGuesserCommit: string | undefined;
  let authenticatedEngineCommit: string | undefined;
  let reproducedIdentity: Promise<Schema9ExecutionIdentity> | undefined;
  return Object.freeze({
    async pinnedEngineCommitAt(guesserCommit: string): Promise<string> {
      const resolvedGuesser = await realpath(guesserRepository);
      const resolvedEngine = await realpath(engineRepository);
      await assertExplicitEngineSubmoduleCheckout(
        resolvedGuesser,
        resolvedEngine,
      );
      await requireCommit(resolvedGuesser, guesserCommit);
      await requireCleanCheckout(
        resolvedGuesser,
        guesserCommit,
        "Guesser repository",
      );
      const result = await git(resolvedGuesser, [
        "ls-tree",
        "--full-tree",
        guesserCommit,
        "--",
        "engine",
      ]);
      const match =
        /^160000 commit ([0-9a-f]{40}|[0-9a-f]{64})\tengine\n?$/u.exec(
          result.stdout,
        );
      if (match?.[1] === undefined) {
        throw new TypeError(
          "Guesser commit does not contain one pinned Engine gitlink.",
        );
      }
      await requireCommit(resolvedEngine, match[1]);
      await requireCleanCheckout(
        resolvedEngine,
        match[1],
        "Engine repository",
      );
      if (
        authenticatedGuesserCommit !== undefined
        && (
          authenticatedGuesserCommit !== guesserCommit
          || authenticatedEngineCommit !== match[1]
        )
      ) {
        throw new TypeError(
          "Schema-9 verifier cannot change authenticated repository commits.",
        );
      }
      authenticatedGuesserCommit ??= guesserCommit;
      authenticatedEngineCommit ??= match[1];
      return match[1];
    },
    async isEngineAncestor(
      ancestorCommit: string,
      descendantCommit: string,
    ): Promise<boolean> {
      const resolvedEngine = await realpath(engineRepository);
      await requireCommit(resolvedEngine, ancestorCommit);
      await requireCommit(resolvedEngine, descendantCommit);
      try {
        await git(resolvedEngine, [
          "merge-base",
          "--is-ancestor",
          ancestorCommit,
          descendantCommit,
        ]);
        return true;
      } catch (error: unknown) {
        if (exitCode(error) === 1) {
          return false;
        }
        throw error;
      }
    },
    async executingCodeIdentity() {
      if (
        authenticatedGuesserCommit === undefined
        || authenticatedEngineCommit === undefined
      ) {
        throw new TypeError(
          "Repository commits must be authenticated before execution code.",
        );
      }
      const resolvedGuesser = await realpath(guesserRepository);
      const resolvedEngine = await realpath(engineRepository);
      const entrypoints = currentExecutionEntrypoints();
      await assertEngineRuntimeCheckout(
        resolvedGuesser,
        resolvedEngine,
        entrypoints.parser,
      );
      const manifest = await executingCodeManifest(
        resolvedGuesser,
        entrypoints,
      );
      const reproduce = options.reproduceExecutionIdentity
        ?? reproduceSchema9ExecutionIdentity;
      reproducedIdentity ??= reproduce(Object.freeze({
        guesserRepository: resolvedGuesser,
        engineRepository: resolvedEngine,
        guesserCommit: authenticatedGuesserCommit,
        engineCommit: authenticatedEngineCommit,
        ...(options.temporaryParent === undefined
          ? {}
          : { temporaryParent: options.temporaryParent }),
      }));
      assertExactReproducibleExecutionIdentity(
        manifest.identity,
        await reproducedIdentity,
      );
      executionGuard ??= createSchema9ModuleLoadGuard(manifest.paths);
      return manifest.identity;
    },
    disposeExecutionGuard() {
      executionGuard?.deregister();
      executionGuard = undefined;
    },
  });
}

function verificationReceiptPath(ledgerPath: string, ledgerSha256: string) {
  return join(
    dirname(ledgerPath),
    `schema9-ledger-verification-${ledgerSha256}.json`,
  );
}

async function publishOrAuthenticateVerificationReceipt(
  ledgerPath: string,
  ledgerSha256: string,
  receipt: ReturnType<typeof createSchema9LedgerVerificationReceipt>,
): Promise<{ readonly bytes: number; readonly sha256: string }> {
  const outputPath = verificationReceiptPath(ledgerPath, ledgerSha256);
  try {
    return await writeSchema9LedgerVerificationReceiptAtomic(
      outputPath,
      receipt,
    );
  } catch (error: unknown) {
    const expected = canonicalJsonBytes(receipt);
    let existing: Buffer;
    try {
      existing = await readFile(outputPath);
    } catch {
      throw error;
    }
    if (!existing.equals(expected)) {
      throw new TypeError(
        "Existing schema-9 verification receipt is inconsistent.",
        { cause: error },
      );
    }
    return Object.freeze({
      bytes: existing.byteLength,
      sha256: createHash("sha256").update(existing).digest("hex"),
    });
  }
}

export async function runSchema9LedgerCli(
  options: Schema9LedgerCliOptions,
  io: Schema9LedgerCliIo = { stdout: process.stdout },
): Promise<void> {
  const repositoryVerifier = createGitRepositoryVerifier(
    options.guesserRepository,
    options.engineRepository,
  );
  try {
    const corpus: Schema9CorpusLedgerOptions = Object.freeze({
      ...options.corpus,
      repositoryVerifier,
      assignmentScheduler: schema9AssignmentScheduler,
    });
    if (options.operation === "create") {
      const written = await writeSchema9CorpusLedgerAtomic(
        options.ledgerPath,
        corpus,
      );
      const receipt = createSchema9LedgerVerificationReceipt(
        written.artifact,
        written.sha256,
      );
      const receiptWritten = await publishOrAuthenticateVerificationReceipt(
        options.ledgerPath,
        written.sha256,
        receipt,
      );
      io.stdout.write(
        `${JSON.stringify({
          format: SCHEMA9_CORPUS_LEDGER_FORMAT,
          version: SCHEMA9_CORPUS_LEDGER_VERSION,
          bytes: written.bytes,
          sha256: written.sha256,
          verificationReceiptSha256: receiptWritten.sha256,
        })}\n`,
      );
      return;
    }
    const artifact = await loadAndReauthenticateSchema9CorpusLedger(
      options.ledgerPath,
      corpus,
    );
    const sha256 = await schema9CorpusLedgerFileSha256(options.ledgerPath);
    const receipt = createSchema9LedgerVerificationReceipt(artifact, sha256);
    const receiptWritten = await publishOrAuthenticateVerificationReceipt(
      options.ledgerPath,
      sha256,
      receipt,
    );
    io.stdout.write(
      `${JSON.stringify({
        format: SCHEMA9_CORPUS_LEDGER_FORMAT,
        version: SCHEMA9_CORPUS_LEDGER_VERSION,
        verified: true,
        sha256,
        verificationReceiptSha256: receiptWritten.sha256,
      })}\n`,
    );
  } finally {
    repositoryVerifier.disposeExecutionGuard();
  }
}

async function main(): Promise<void> {
  try {
    const arguments_ = process.argv.slice(2).filter(
      (argument) => argument !== "--",
    );
    await runSchema9LedgerCli(
      parseSchema9LedgerCliArguments(arguments_),
    );
  } catch (error: unknown) {
    process.stderr.write(
      `Schema-9 corpus ledger failed: ${
        error instanceof Error ? error.message : String(error)
      }\n`,
    );
    process.exitCode = 1;
  }
}

const invokedPath = process.argv[1];
if (
  invokedPath !== undefined
  && import.meta.url === pathToFileURL(invokedPath).href
) {
  void main();
}
