import { createHash, randomUUID } from "node:crypto";
import { realpathSync, type BigIntStats } from "node:fs";
import {
  chmod,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  rename,
  rm,
  rmdir,
  stat,
} from "node:fs/promises";
import { registerHooks } from "node:module";
import { tmpdir } from "node:os";
import {
  basename,
  dirname,
  isAbsolute,
  join,
  parse,
  relative,
  resolve,
} from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  canonicalJsonBytes,
  createSchema9CorpusLedger,
  createSchema9LedgerVerificationReceipt,
  loadAndReauthenticateSchema9CorpusLedgerWithIdentity,
  publishOrAuthenticateSchema9CorpusLedgerArtifactAtomic,
  publishOrAuthenticateSchema9LedgerVerificationReceipt,
  removeSchema9StableFileIfOwned,
  runSchema9LinkedTaskGroup,
  schema9PublicationMayBeCommitted,
  schema9AssignmentScheduler,
  SCHEMA9_CORPUS_LEDGER_FORMAT,
  SCHEMA9_CORPUS_LEDGER_VERSION,
  SCHEMA9_EXECUTION_MANIFEST_ALGORITHM,
  SCHEMA9_LEDGER_SPLITS,
  SCHEMA9_PRODUCER_CONVERTER_POLICIES,
  type Schema9CorpusLedgerOptions,
  type Schema9ExecutionIdentity,
  type Schema9LedgerSplit,
  type Schema9ProducerConverterPolicy,
  type Schema9ProducerRuntimeIdentity,
  type Schema9RepositoryVerifier,
  type Schema9SplitFiles,
} from "@drawbackguesser/trace-to-dataset";
import {
  computeSchema9ProducerRuntimeIdentity,
  schema9ProducerRuntimeDescriptor,
} from "./schema9-producer-runtime-identity.js";
import { writeSchema9JsonLine } from "./schema9-json-line-writer.js";
import {
  findSchema9TerminationError,
  installSchema9TerminationSignal,
  runSchema9BoundedCommand,
  withSchema9CommandSignal,
} from "./schema9-process-lifecycle.js";

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

interface AuthenticatedGitExecutable {
  readonly path: string;
  readonly metadata: BigIntStats;
  readonly environment: NodeJS.ProcessEnv;
}

async function git(
  repository: string,
  arguments_: readonly string[],
  signal?: AbortSignal,
): Promise<GitResult> {
  assertNoSchema9GitConfigOverrides(arguments_);
  const executable = await authenticatedGitExecutable(signal);
  try {
    return await runSchema9BoundedCommand(
      executable.path,
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
        environment: executable.environment,
        timeoutMilliseconds: 2 * 60 * 1000,
        maxOutputBytes: 1024 * 1024,
        description: "Git command",
        ...(signal === undefined ? {} : { signal }),
      },
    );
  } finally {
    await assertAuthenticatedGitUnchanged(executable);
  }
}

export function assertNoSchema9GitConfigOverrides(
  arguments_: readonly string[],
): void {
  if (arguments_.some((argument) =>
    argument === "-c"
    || (argument.startsWith("-c") && argument.length > 2)
    || argument === "--config-env"
    || argument.startsWith("--config-env="))) {
    throw new TypeError("Schema-9 Git commands reject caller config overrides.");
  }
}

async function authenticatedGitExecutable(
  signal?: AbortSignal,
): Promise<AuthenticatedGitExecutable> {
  throwIfLedgerAborted(signal);
  let candidate: string;
  let systemRoot: string | undefined;
  if (process.platform === "win32") {
    const configuredSystemRoot = process.env["SystemRoot"]?.trim();
    if (configuredSystemRoot === undefined || !isAbsolute(configuredSystemRoot)) {
      throw new TypeError(
        "Schema-9 Git authentication requires an absolute SystemRoot.",
      );
    }
    systemRoot = await realpath(configuredSystemRoot);
    const programFiles = await realpath(join(parse(systemRoot).root, "Program Files"));
    candidate = await realpath(join(programFiles, "Git", "cmd", "git.exe"));
    const child = relative(programFiles, candidate);
    if (
      child === ""
      || child === ".."
      || child.startsWith("..\\")
      || isAbsolute(child)
    ) {
      throw new TypeError("Schema-9 Git escaped the fixed Program Files root.");
    }
  } else {
    const candidates = new Set<string>();
    for (const path of ["/usr/bin/git", "/bin/git"] as const) {
      try {
        const resolved = await realpath(path);
        const metadata = await lstat(resolved);
        if (metadata.isFile()) {
          candidates.add(resolved);
        }
      } catch (error: unknown) {
        if (!isNodeError(error, "ENOENT")) {
          throw error;
        }
      }
    }
    if (candidates.size !== 1) {
      throw new TypeError("Schema-9 requires one fixed system Git executable.");
    }
    candidate = [...candidates][0] as string;
  }
  throwIfLedgerAborted(signal);
  const metadata = await lstat(candidate, { bigint: true });
  if (!metadata.isFile()) {
    throw new TypeError("Schema-9 system Git is not a regular file.");
  }
  const environment: NodeJS.ProcessEnv = Object.fromEntries(
    Object.entries(process.env).filter(([name]) => {
      const canonicalName = name.toUpperCase();
      return canonicalName === "TEMP" || canonicalName === "TMP";
    }),
  );
  const nullDevice = process.platform === "win32" ? "NUL" : "/dev/null";
  environment["PATH"] = process.platform === "win32"
    ? [dirname(candidate), join(systemRoot as string, "System32"), systemRoot as string]
      .join(";")
    : "/usr/bin:/bin";
  if (process.platform === "win32") {
    environment["SystemRoot"] = systemRoot;
    environment["WINDIR"] = systemRoot;
    environment["ComSpec"] = join(systemRoot as string, "System32", "cmd.exe");
    environment["PATHEXT"] = ".COM;.EXE;.BAT;.CMD";
  }
  environment["GIT_ATTR_NOSYSTEM"] = "1";
  environment["GIT_CONFIG_COUNT"] = "0";
  environment["GIT_CONFIG_GLOBAL"] = nullDevice;
  environment["GIT_CONFIG_NOSYSTEM"] = "1";
  environment["GIT_OPTIONAL_LOCKS"] = "0";
  environment["GIT_PAGER"] = "cat";
  environment["GIT_TERMINAL_PROMPT"] = "0";
  environment["LC_ALL"] = "C";
  return Object.freeze({
    path: candidate,
    metadata,
    environment: Object.freeze(environment),
  });
}

async function assertAuthenticatedGitUnchanged(
  expected: AuthenticatedGitExecutable,
): Promise<void> {
  const actual = await lstat(expected.path, { bigint: true });
  if (
    !actual.isFile()
    || actual.dev !== expected.metadata.dev
    || actual.ino !== expected.metadata.ino
    || actual.size !== expected.metadata.size
    || actual.mtimeNs !== expected.metadata.mtimeNs
    || actual.ctimeNs !== expected.metadata.ctimeNs
  ) {
    throw new Error("Authenticated Schema-9 system Git changed during use.");
  }
}

async function optionalGitMatch(
  repository: string,
  arguments_: readonly string[],
  signal?: AbortSignal,
): Promise<string> {
  try {
    return (await git(repository, arguments_, signal)).stdout;
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
  signal?: AbortSignal,
): Promise<void> {
  const tagged = (await git(
    repository,
    ["ls-files", "-v", "-z"],
    signal,
  )).stdout;
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

function containsCommandFsmonitorValue(output: string): boolean {
  return output.split("\0").some((value) => {
    const normalized = value.trim().toLocaleLowerCase("en-US");
    return normalized.length > 0
      && normalized !== "true"
      && normalized !== "false";
  });
}

export async function requireNoCommandFsmonitorConfiguration(
  repository: string,
  label: string,
  signal?: AbortSignal,
): Promise<void> {
  const local = await optionalGitMatch(repository, [
    "config",
    "--local",
    "--null",
    "--get-all",
    "core.fsmonitor",
  ], signal);
  const worktreeConfigEnabled = (await optionalGitMatch(repository, [
    "config",
    "--local",
    "--type=bool",
    "--get",
    "extensions.worktreeConfig",
  ], signal)).trim() === "true";
  const worktree = worktreeConfigEnabled
    ? await optionalGitMatch(repository, [
      "config",
      "--worktree",
      "--null",
      "--get-all",
      "core.fsmonitor",
    ], signal)
    : "";
  if (
    containsCommandFsmonitorValue(local)
    || containsCommandFsmonitorValue(worktree)
  ) {
    throw new TypeError(
      `${label} contains forbidden command-bearing core.fsmonitor configuration.`,
    );
  }
}

function hasCommandFilterConfiguration(output: string): boolean {
  return output.split("\0").some((record) => record.trim().length > 0);
}

export async function requireNoCommandFilterConfiguration(
  repository: string,
  label: string,
  signal?: AbortSignal,
): Promise<void> {
  const filterPattern = "^filter\\..*\\.(clean|smudge|process)$";
  const local = await optionalGitMatch(repository, [
    "config",
    "--local",
    "--null",
    "--get-regexp",
    filterPattern,
  ], signal);
  const worktreeConfigEnabled = (await optionalGitMatch(repository, [
    "config",
    "--local",
    "--type=bool",
    "--get",
    "extensions.worktreeConfig",
  ], signal)).trim() === "true";
  const worktree = worktreeConfigEnabled
    ? await optionalGitMatch(repository, [
      "config",
      "--worktree",
      "--null",
      "--get-regexp",
      filterPattern,
    ], signal)
    : "";
  if (
    hasCommandFilterConfiguration(local)
    || hasCommandFilterConfiguration(worktree)
  ) {
    throw new TypeError(
      `${label} contains forbidden command-bearing Git filter configuration.`,
    );
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
  signal?: AbortSignal,
): Promise<void> {
  const local = await optionalGitMatch(repository, [
    "config",
    "--local",
    "--null",
    "--get-regexp",
    "^(submodule\\..*\\.ignore|diff\\.ignoreSubmodules)$",
  ], signal);
  const worktreeConfigEnabled = (await optionalGitMatch(repository, [
    "config",
    "--local",
    "--type=bool",
    "--get",
    "extensions.worktreeConfig",
  ], signal)).trim() === "true";
  const worktree = worktreeConfigEnabled
    ? await optionalGitMatch(repository, [
      "config",
      "--worktree",
      "--null",
      "--get-regexp",
      "^(submodule\\..*\\.ignore|diff\\.ignoreSubmodules)$",
    ], signal)
    : "";
  const gitmodulesEntry = (await git(repository, [
    "ls-tree",
    "--name-only",
    expectedHead,
    "--",
    ".gitmodules",
  ], signal)).stdout.trim();
  const committed = gitmodulesEntry === ".gitmodules"
    ? await optionalGitMatch(repository, [
      "config",
      "--blob",
      `${expectedHead}:.gitmodules`,
      "--null",
      "--get-regexp",
      "^submodule\\..*\\.ignore$",
    ], signal)
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

export async function requireCleanCheckout(
  repository: string,
  expectedHead: string,
  label: string,
  signal?: AbortSignal,
): Promise<void> {
  const head = (await git(
    repository,
    ["rev-parse", "HEAD"],
    signal,
  )).stdout.trim();
  if (head !== expectedHead) {
    throw new TypeError(`${label} HEAD does not match the declared commit.`);
  }
  await requireNoCommandFilterConfiguration(repository, label, signal);
  const status = await git(repository, [
    "status",
    "--porcelain=v1",
    "--untracked-files=no",
    "--ignore-submodules=none",
  ], signal);
  if (status.stdout.length !== 0) {
    throw new TypeError(`${label} tracked worktree or index is dirty.`);
  }
  await requireNoCommandFsmonitorConfiguration(repository, label, signal);
  const replaceRefs = await git(repository, [
    "for-each-ref",
    "--format=%(refname)",
    "refs/replace",
  ], signal);
  if (replaceRefs.stdout.trim().length !== 0) {
    throw new TypeError(`${label} contains forbidden Git replace refs.`);
  }
  await requireNoIndexIgnoreFlags(repository, label, signal);
  await requireNoSubmoduleIgnoreConfiguration(
    repository,
    expectedHead,
    label,
    signal,
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
  signal?: AbortSignal,
): Promise<void> {
  throwIfLedgerAborted(signal);
  const guesserRoot = await realpath(guesserRepository);
  const expectedEngineRoot = await realpath(join(guesserRoot, "engine"));
  const suppliedEngineRoot = await realpath(engineRepository);
  throwIfLedgerAborted(signal);
  if (relative(expectedEngineRoot, suppliedEngineRoot) !== "") {
    throw new TypeError(
      "Explicit Engine repository is not the Guesser engine submodule checkout.",
    );
  }
  if (!runtimeEngineEntrypoint.startsWith("file:")) {
    throw new TypeError("Engine runtime entrypoint must be a file URL.");
  }
  const runtimePath = await realpath(fileURLToPath(runtimeEngineEntrypoint));
  throwIfLedgerAborted(signal);
  if (pathEscapesRoot(expectedEngineRoot, runtimePath)) {
    throw new TypeError(
      "Loaded Engine runtime does not come from the Guesser engine submodule.",
    );
  }
}

async function assertExplicitEngineSubmoduleCheckout(
  guesserRepository: string,
  engineRepository: string,
  signal?: AbortSignal,
): Promise<void> {
  throwIfLedgerAborted(signal);
  const guesserRoot = await realpath(guesserRepository);
  const expectedEngineRoot = await realpath(join(guesserRoot, "engine"));
  const suppliedEngineRoot = await realpath(engineRepository);
  throwIfLedgerAborted(signal);
  if (relative(expectedEngineRoot, suppliedEngineRoot) !== "") {
    throw new TypeError(
      "Explicit Engine repository is not the Guesser engine submodule checkout.",
    );
  }
}

const REPRODUCIBLE_BUILD_PREFIX = "schema9-reproducible-build-";

export interface OwnedSchema9CheckoutIdentity {
  readonly dev: bigint;
  readonly ino: bigint;
  readonly birthtimeNs: bigint;
}

export interface OwnedSchema9CleanupState {
  readonly ownerPath: string;
  readonly ownerIdentity: OwnedSchema9CheckoutIdentity;
  readonly checkoutPath: string;
  readonly checkoutIdentity: OwnedSchema9CheckoutIdentity;
  readonly stage: "checkout" | "quarantine" | "checkout-removed";
}

class OwnedSchema9CleanupAttemptError extends Error {
  public constructor(
    message: string,
    public readonly state: OwnedSchema9CleanupState,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "OwnedSchema9CleanupAttemptError";
  }
}

export class IncompleteOwnedSchema9CleanupError extends AggregateError {
  public readonly cleanupComplete = false;
  readonly #state: OwnedSchema9CleanupState;
  readonly #failures: readonly unknown[];
  #activeRetry: Promise<void> | undefined;

  public constructor(
    failures: readonly unknown[],
    state: OwnedSchema9CleanupState,
  ) {
    super(
      [...failures],
      "Schema-9 reproducible checkout cleanup remains incomplete.",
    );
    this.name = "IncompleteOwnedSchema9CleanupError";
    this.#failures = [...failures];
    this.#state = state;
  }

  public cleanupOwnerIdentity(): OwnedSchema9CheckoutIdentity {
    return this.#state.ownerIdentity;
  }

  /** Retries deletion only through the exact retained checkout identity. */
  public retryCleanup(): Promise<void> {
    if (this.#activeRetry !== undefined) {
      return this.#activeRetry;
    }
    const retry = this.#retryOnce();
    this.#activeRetry = retry;
    void retry.then(
      () => {
        if (this.#activeRetry === retry) {
          this.#activeRetry = undefined;
        }
      },
      () => {
        if (this.#activeRetry === retry) {
          this.#activeRetry = undefined;
        }
      },
    );
    return retry;
  }

  async #retryOnce(): Promise<void> {
    try {
      await cleanupOwnedSchema9Boundary(this.#state);
    } catch (error: unknown) {
      const state = error instanceof OwnedSchema9CleanupAttemptError
        ? error.state
        : this.#state;
      throw new IncompleteOwnedSchema9CleanupError(
        [...this.#failures, error],
        state,
      );
    }
  }
}

const MAX_OWNED_CHECKOUT_CLEANUP_ATTEMPTS = 2;

export async function withSchema9TemporaryCheckout<T>(
  operation: (checkoutRoot: string) => Promise<T>,
  temporaryParent = tmpdir(),
  signal?: AbortSignal,
): Promise<T> {
  throwIfLedgerAborted(signal);
  const parent = await realpath(temporaryParent);
  const ownerPath = await mkdtemp(join(parent, REPRODUCIBLE_BUILD_PREFIX));
  const ownerIdentity = await ownedSchema9CheckoutIdentity(ownerPath);
  const checkoutPath = join(ownerPath, "checkout");
  let checkoutIdentity: OwnedSchema9CheckoutIdentity;
  try {
    await chmod(ownerPath, 0o700);
    await mkdir(checkoutPath, { mode: 0o700 });
    checkoutIdentity = await ownedSchema9CheckoutIdentity(checkoutPath);
  } catch (error: unknown) {
    try {
      await removeOwnedSchema9Checkout(ownerPath, ownerIdentity);
    } catch (cleanupError: unknown) {
      throw new AggregateError(
        [error, cleanupError],
        "Schema-9 temporary checkout initialization and cleanup failed.",
      );
    }
    throw error instanceof Error
      ? error
      : new Error("Schema-9 temporary checkout initialization failed.", {
        cause: error,
      });
  }
  let cleanupState: OwnedSchema9CleanupState = Object.freeze({
    ownerPath,
    ownerIdentity,
    checkoutPath,
    checkoutIdentity,
    stage: "checkout",
  });
  let outcome:
    | { readonly ok: true; readonly value: T }
    | { readonly ok: false; readonly error: unknown };
  try {
    throwIfLedgerAborted(signal);
    const checkoutRoot = await realpath(checkoutPath);
    if (
      relative(ownerPath, dirname(checkoutRoot)) !== ""
      || basename(checkoutRoot) !== "checkout"
    ) {
      throw new TypeError("Temporary schema-9 checkout escaped its owner.");
    }
    outcome = Object.freeze({ ok: true, value: await operation(checkoutRoot) });
  } catch (error: unknown) {
    outcome = Object.freeze({ ok: false, error });
  }
  if (
    relative(parent, dirname(ownerPath)) !== ""
    || !basename(ownerPath).startsWith(REPRODUCIBLE_BUILD_PREFIX)
  ) {
    throw new TypeError("Temporary schema-9 cleanup target is invalid.");
  }
  const cleanupFailures: unknown[] = [];
  let cleanupComplete = false;
  for (
    let attempt = 0;
    attempt < MAX_OWNED_CHECKOUT_CLEANUP_ATTEMPTS;
    attempt += 1
  ) {
    try {
      await cleanupOwnedSchema9Boundary(cleanupState);
      cleanupComplete = true;
      break;
    } catch (cleanupError: unknown) {
      cleanupFailures.push(cleanupError);
      if (cleanupError instanceof OwnedSchema9CleanupAttemptError) {
        cleanupState = cleanupError.state;
      }
    }
  }
  if (cleanupFailures.length > 0) {
    const failures = outcome.ok
      ? cleanupFailures
      : [outcome.error, ...cleanupFailures];
    if (!cleanupComplete) {
      throw new IncompleteOwnedSchema9CleanupError(
        failures,
        cleanupState,
      );
    }
    throw new AggregateError(
      failures,
      "Schema-9 reproducible checkout cleanup initially failed.",
    );
  }
  if (!outcome.ok) {
    throw outcome.error;
  }
  return outcome.value;
}

export async function ownedSchema9CheckoutIdentity(
  path: string,
): Promise<OwnedSchema9CheckoutIdentity> {
  const metadata = await lstat(path, { bigint: true });
  if (!metadata.isDirectory()) {
    throw new TypeError("Schema-9 temporary checkout owner is not a directory.");
  }
  return Object.freeze({
    dev: metadata.dev,
    ino: metadata.ino,
    birthtimeNs: metadata.birthtimeNs,
  });
}

export async function removeOwnedSchema9Checkout(
  target: string,
  expected: OwnedSchema9CheckoutIdentity,
): Promise<void> {
  const parent = dirname(target);
  const quarantine = join(
    parent,
    `.schema9-cleanup-${randomUUID()}`,
  );
  await assertOwnedSchema9Directory(target, expected, "cleanup target");
  await rename(target, quarantine);
  try {
    await assertOwnedSchema9Directory(quarantine, expected, "quarantine");
    await rm(quarantine, {
      recursive: true,
      maxRetries: 3,
      retryDelay: 50,
    });
  } catch (error: unknown) {
    throw new Error(
      "Schema-9 owned checkout cleanup remains incomplete.",
      { cause: error },
    );
  }
}

function sameOwnedSchema9Identity(
  metadata: BigIntStats,
  expected: OwnedSchema9CheckoutIdentity,
): boolean {
  return metadata.isDirectory()
    && metadata.dev === expected.dev
    && metadata.ino === expected.ino
    && metadata.birthtimeNs === expected.birthtimeNs;
}

async function assertOwnedSchema9Directory(
  path: string,
  expected: OwnedSchema9CheckoutIdentity,
  label: string,
): Promise<void> {
  let metadata: BigIntStats;
  try {
    metadata = await lstat(path, { bigint: true });
  } catch (error: unknown) {
    if (isNodeError(error, "ENOENT")) {
      throw new Error(
        `Schema-9 ${label} disappeared before deletion was proven.`,
        { cause: error },
      );
    }
    throw error;
  }
  if (!sameOwnedSchema9Identity(metadata, expected)) {
    throw new Error(`Schema-9 ${label} is no longer the owned directory.`);
  }
}

async function cleanupOwnedSchema9Boundary(
  initial: OwnedSchema9CleanupState,
): Promise<void> {
  let state = initial;
  try {
    await assertOwnedSchema9Directory(
      state.ownerPath,
      state.ownerIdentity,
      "cleanup owner",
    );
    if (state.stage === "checkout") {
      await assertOwnedSchema9Directory(
        state.checkoutPath,
        state.checkoutIdentity,
        "checkout",
      );
      const quarantinePath = join(
        state.ownerPath,
        `.quarantine-${randomUUID()}`,
      );
      await rename(state.checkoutPath, quarantinePath);
      state = Object.freeze({
        ...state,
        checkoutPath: quarantinePath,
        stage: "quarantine",
      });
    }
    if (state.stage === "quarantine") {
      await assertOwnedSchema9Directory(
        state.ownerPath,
        state.ownerIdentity,
        "cleanup owner",
      );
      await assertOwnedSchema9Directory(
        state.checkoutPath,
        state.checkoutIdentity,
        "quarantined checkout",
      );
      await rm(state.checkoutPath, {
        recursive: true,
        maxRetries: 3,
        retryDelay: 50,
      });
      state = Object.freeze({ ...state, stage: "checkout-removed" });
    }
    await assertOwnedSchema9Directory(
      state.ownerPath,
      state.ownerIdentity,
      "cleanup owner",
    );
    await rmdir(state.ownerPath);
  } catch (error: unknown) {
    throw new OwnedSchema9CleanupAttemptError(
      "Schema-9 reproducible checkout cleanup attempt failed.",
      state,
      { cause: error },
    );
  }
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
  "NODE_PRESERVE_SYMLINKS_MAIN",
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

export async function discoverRuntimeModulePaths(
  entryUrl: string,
  timeoutMilliseconds = 30_000,
  signal?: AbortSignal,
): Promise<readonly string[]> {
  throwIfLedgerAborted(signal);
  const environment: NodeJS.ProcessEnv = { ...process.env };
  delete environment["NODE_OPTIONS"];
  delete environment["NODE_PATH"];
  delete environment["NODE_PRESERVE_SYMLINKS"];
  delete environment["NODE_PRESERVE_SYMLINKS_MAIN"];
  environment["SCHEMA9_MODULE_GRAPH_ENTRY"] = entryUrl;
  const output = (await runSchema9BoundedCommand(
    process.execPath,
    [
      "--input-type=module",
      "--eval",
      MODULE_GRAPH_PROBE,
    ],
    {
      cwd: process.cwd(),
      environment,
      timeoutMilliseconds,
      maxOutputBytes: 16 * 1024 * 1024,
      description: "module graph probe",
      ...(signal === undefined ? {} : { signal }),
    },
  )).stdout;
  throwIfLedgerAborted(signal);
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
  signal?: AbortSignal,
): Promise<ModuleGraphManifest> {
  throwIfLedgerAborted(signal);
  const root = await realpath(repositoryRoot);
  const discovered = await discoverRuntimeModulePaths(
    entryUrl,
    30_000,
    signal,
  );
  const graphPaths = new Set<string>();
  for (const discoveredPath of discovered) {
    throwIfLedgerAborted(signal);
    const resolved = await realpath(discoveredPath);
    graphPaths.add(resolved);
    let directory = dirname(resolved);
    for (;;) {
      throwIfLedgerAborted(signal);
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
    throwIfLedgerAborted(signal);
    const current = await realpath(discoveredPath);
    const before = await stat(current);
    const bytes = await readFile(
      current,
      signal === undefined ? undefined : { signal },
    );
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
  signal?: AbortSignal,
): Promise<ModuleGraphManifest["identity"]> {
  return (await moduleGraphManifest(
    entrypoint,
    entryUrl,
    repositoryRoot,
    signal,
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
  signal?: AbortSignal,
) {
  throwIfLedgerAborted(signal);
  const runtime = schema9RuntimeIdentity();
  const manifests = await runSchema9LinkedTaskGroup([
    (taskSignal) => moduleGraphManifest(
      "simulation-trace",
      entrypoints.parser,
      repositoryRoot,
      taskSignal,
    ),
    (taskSignal) => moduleGraphManifest(
      "trace-to-dataset",
      entrypoints.converter,
      repositoryRoot,
      taskSignal,
    ),
    (taskSignal) => moduleGraphManifest(
      "schema9-schedule-replay",
      entrypoints.scheduler,
      repositoryRoot,
      taskSignal,
    ),
    (taskSignal) => moduleGraphManifest(
      "schema9-ledger-cli",
      entrypoints.verifier,
      repositoryRoot,
      taskSignal,
    ),
  ], signal, "Schema-9 module graph authentication");
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
  signal?: AbortSignal,
): Promise<CommandResult> {
  return runSchema9BoundedCommand(command, arguments_, {
    cwd,
    environment,
    timeoutMilliseconds: 10 * 60 * 1000,
    maxOutputBytes: 16 * 1024 * 1024,
    description: "reproducible build command",
    ...(signal === undefined ? {} : { signal }),
  });
}

function reproducibleBuildEnvironment(): NodeJS.ProcessEnv {
  const excluded = new Set([
    "NODE_OPTIONS",
    "NODE_PATH",
    "NODE_PRESERVE_SYMLINKS",
    "NODE_PRESERVE_SYMLINKS_MAIN",
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
    "GIT_EXEC_PATH",
    "GIT_NAMESPACE",
    "GIT_REPLACE_REF_BASE",
    "GIT_TEMPLATE_DIR",
  ]);
  const environment: NodeJS.ProcessEnv = Object.fromEntries(
    Object.entries(process.env).filter(([name]) => {
      const canonicalName = name.toUpperCase();
      return !excluded.has(canonicalName)
        && !canonicalName.startsWith("GIT_CONFIG_KEY_")
        && !canonicalName.startsWith("GIT_CONFIG_VALUE_")
        && !canonicalName.startsWith("NPM_CONFIG_")
        && !canonicalName.startsWith("PNPM_");
    }),
  );
  const nullDevice = process.platform === "win32" ? "NUL" : "/dev/null";
  environment["CI"] = "true";
  environment["COREPACK_ENABLE_DOWNLOAD_PROMPT"] = "0";
  environment["GIT_CONFIG_NOSYSTEM"] = "1";
  environment["GIT_CONFIG_GLOBAL"] = nullDevice;
  environment["npm_config_globalconfig"] = nullDevice;
  environment["npm_config_userconfig"] = nullDevice;
  environment["npm_config_offline"] = "true";
  environment["npm_config_ignore_scripts"] = "true";
  environment["npm_config_frozen_lockfile"] = "true";
  return environment;
}

async function pnpmEntrypoint(signal?: AbortSignal): Promise<string> {
  throwIfLedgerAborted(signal);
  const configured = process.env["npm_execpath"];
  if (configured === undefined || configured.length === 0) {
    throw new TypeError(
      "Schema-9 reproducible verification must be launched through pnpm.",
    );
  }
  const resolved = await realpath(configured);
  throwIfLedgerAborted(signal);
  if (!(await stat(resolved)).isFile()) {
    throw new TypeError("The active pnpm entrypoint is not a regular file.");
  }
  return resolved;
}

async function requiredPnpmVersion(
  repositoryRoot: string,
  signal?: AbortSignal,
): Promise<string> {
  const parsed: unknown = JSON.parse(
    await readFile(
      join(repositoryRoot, "package.json"),
      signal === undefined ? "utf8" : { encoding: "utf8", signal },
    ),
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
  signal?: AbortSignal,
): Promise<void> {
  const entrypoint = await pnpmEntrypoint(signal);
  const before = await stat(entrypoint);
  const beforeBytes = await readFile(
    entrypoint,
    signal === undefined ? undefined : { signal },
  );
  const environment = reproducibleBuildEnvironment();
  const invoke = (arguments_: readonly string[]) =>
    runBoundedCommand(
      process.execPath,
      [entrypoint, ...arguments_],
      repositoryRoot,
      environment,
      signal,
    );
  const requiredVersion = await requiredPnpmVersion(repositoryRoot, signal);
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
  throwIfLedgerAborted(signal);
  await invoke(["run", "engine:build"]);
  await invoke([
    "--filter",
    "@drawbackguesser/dataset-cli",
    "run",
    "build",
  ]);
  const after = await stat(entrypoint);
  const afterBytes = await readFile(
    entrypoint,
    signal === undefined ? undefined : { signal },
  );
  if (
    !sameFileSignature(before, after)
    || !beforeBytes.equals(afterBytes)
  ) {
    throw new TypeError("The pnpm entrypoint changed during reproduction.");
  }
}

async function installAndBuildEngineRuntimeCheckout(
  repositoryRoot: string,
  signal?: AbortSignal,
): Promise<void> {
  const entrypoint = await pnpmEntrypoint(signal);
  const before = await stat(entrypoint);
  const beforeBytes = await readFile(
    entrypoint,
    signal === undefined ? undefined : { signal },
  );
  const environment = reproducibleBuildEnvironment();
  const invoke = (arguments_: readonly string[]) =>
    runBoundedCommand(
      process.execPath,
      [entrypoint, ...arguments_],
      repositoryRoot,
      environment,
      signal,
    );
  const requiredVersion = await requiredPnpmVersion(repositoryRoot, signal);
  const actualVersion = (await invoke(["--version"])).stdout.trim();
  if (actualVersion !== requiredVersion) {
    throw new TypeError("Active pnpm version does not match Engine packageManager.");
  }
  await invoke([
    "install",
    "--offline",
    "--frozen-lockfile",
    "--ignore-scripts",
  ]);
  await invoke(["run", "build"]);
  throwIfLedgerAborted(signal);
  const after = await stat(entrypoint);
  const afterBytes = await readFile(
    entrypoint,
    signal === undefined ? undefined : { signal },
  );
  if (
    !sameFileSignature(before, after)
    || !beforeBytes.equals(afterBytes)
  ) {
    throw new TypeError("The pnpm entrypoint changed during Engine reproduction.");
  }
}

export async function cloneRepositoryAtCommit(
  sourceRepository: string,
  targetRepository: string,
  commit: string,
  signal?: AbortSignal,
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
  ], signal);
  await git(targetRepository, [
    "checkout",
    "--detach",
    "--force",
    commit,
  ], signal);
}

export interface Schema9ReproducibleBuildRequest {
  readonly guesserRepository: string;
  readonly engineRepository: string;
  readonly guesserCommit: string;
  readonly engineCommit: string;
  readonly temporaryParent?: string;
  readonly signal?: AbortSignal;
}

export interface Schema9ProducerRuntimeReproductionRequest {
  readonly engineRepository: string;
  readonly engineCommit: string;
  readonly temporaryParent?: string;
  readonly signal?: AbortSignal;
}

export async function reproduceSchema9ProducerRuntimeIdentity(
  request: Schema9ProducerRuntimeReproductionRequest,
): Promise<Schema9ProducerRuntimeIdentity> {
  const signal = request.signal;
  throwIfLedgerAborted(signal);
  const engineRepository = await realpath(request.engineRepository);
  await requireCommit(engineRepository, request.engineCommit, signal);
  return withSchema9TemporaryCheckout(async (checkoutRoot) => {
    const engineCheckout = join(checkoutRoot, "engine");
    await cloneRepositoryAtCommit(
      engineRepository,
      engineCheckout,
      request.engineCommit,
      signal,
    );
    await requireCleanCheckout(
      engineCheckout,
      request.engineCommit,
      "Reproduced producer Engine repository",
      signal,
    );
    await installAndBuildEngineRuntimeCheckout(engineCheckout, signal);
    await requireCleanCheckout(
      engineCheckout,
      request.engineCommit,
      "Reproduced producer Engine repository",
      signal,
    );
    return computeSchema9ProducerRuntimeIdentity(
      engineCheckout,
      schema9ProducerRuntimeDescriptor(),
      signal,
    );
  }, request.temporaryParent, signal);
}

export async function reproduceSchema9ExecutionIdentity(
  request: Schema9ReproducibleBuildRequest,
): Promise<Schema9ExecutionIdentity> {
  const signal = request.signal;
  throwIfLedgerAborted(signal);
  const guesserRepository = await realpath(request.guesserRepository);
  const engineRepository = await realpath(request.engineRepository);
  await requireCommit(guesserRepository, request.guesserCommit, signal);
  await requireCommit(engineRepository, request.engineCommit, signal);
  return withSchema9TemporaryCheckout(async (checkoutRoot) => {
    const guesserCheckout = join(checkoutRoot, "guesser");
    await cloneRepositoryAtCommit(
      guesserRepository,
      guesserCheckout,
      request.guesserCommit,
      signal,
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
      signal,
    );
    await requireCleanCheckout(
      guesserCheckout,
      request.guesserCommit,
      "Reproduced Guesser repository",
      signal,
    );
    await requireCleanCheckout(
      engineCheckout,
      request.engineCommit,
      "Reproduced Engine repository",
      signal,
    );
    await installAndBuildReproducibleCheckout(guesserCheckout, signal);
    await requireCleanCheckout(
      guesserCheckout,
      request.guesserCommit,
      "Reproduced Guesser repository",
      signal,
    );
    await requireCleanCheckout(
      engineCheckout,
      request.engineCommit,
      "Reproduced Engine repository",
      signal,
    );
    const entrypoints = builtExecutionEntrypoints(guesserCheckout);
    await assertEngineRuntimeCheckout(
      guesserCheckout,
      engineCheckout,
      entrypoints.parser,
      signal,
    );
    return (await executingCodeManifest(
      guesserCheckout,
      entrypoints,
      signal,
    )).identity;
  }, request.temporaryParent, signal);
}


async function requireCommit(
  repository: string,
  commit: string,
  signal?: AbortSignal,
): Promise<void> {
  await git(repository, ["cat-file", "-e", `${commit}^{commit}`], signal);
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

function isNodeError(error: unknown, code: string): boolean {
  return typeof error === "object"
    && error !== null
    && "code" in error
    && error.code === code;
}

function throwIfLedgerAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new Error("Schema-9 ledger operation was interrupted.");
  }
}

export interface Schema9GitRepositoryVerifier extends Schema9RepositoryVerifier {
  readonly disposeExecutionGuard: () => void;
}

export interface Schema9GitRepositoryVerifierOptions {
  readonly temporaryParent?: string;
  readonly reproduceExecutionIdentity?: (
    request: Schema9ReproducibleBuildRequest,
  ) => Promise<Schema9ExecutionIdentity>;
  readonly reproduceProducerRuntimeIdentity?: (
    request: Schema9ProducerRuntimeReproductionRequest,
  ) => Promise<Schema9ProducerRuntimeIdentity>;
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
  const reproducedProducerRuntimeIdentities = new Map<
    string,
    Promise<Schema9ProducerRuntimeIdentity>
  >();
  return Object.freeze({
    async pinnedEngineCommitAt(
      guesserCommit: string,
      signal?: AbortSignal,
    ): Promise<string> {
      throwIfLedgerAborted(signal);
      const resolvedGuesser = await realpath(guesserRepository);
      const resolvedEngine = await realpath(engineRepository);
      await assertExplicitEngineSubmoduleCheckout(
        resolvedGuesser,
        resolvedEngine,
        signal,
      );
      await requireCommit(resolvedGuesser, guesserCommit, signal);
      await requireCleanCheckout(
        resolvedGuesser,
        guesserCommit,
        "Guesser repository",
        signal,
      );
      const result = await git(resolvedGuesser, [
        "ls-tree",
        "--full-tree",
        guesserCommit,
        "--",
        "engine",
      ], signal);
      const match =
        /^160000 commit ([0-9a-f]{40}|[0-9a-f]{64})\tengine\n?$/u.exec(
          result.stdout,
        );
      if (match?.[1] === undefined) {
        throw new TypeError(
          "Guesser commit does not contain one pinned Engine gitlink.",
        );
      }
      await requireCommit(resolvedEngine, match[1], signal);
      await requireCleanCheckout(
        resolvedEngine,
        match[1],
        "Engine repository",
        signal,
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
      signal?: AbortSignal,
    ): Promise<boolean> {
      throwIfLedgerAborted(signal);
      const resolvedEngine = await realpath(engineRepository);
      await requireCommit(resolvedEngine, ancestorCommit, signal);
      await requireCommit(resolvedEngine, descendantCommit, signal);
      try {
        await git(resolvedEngine, [
          "merge-base",
          "--is-ancestor",
          ancestorCommit,
          descendantCommit,
        ], signal);
        return true;
      } catch (error: unknown) {
        if (exitCode(error) === 1) {
          return false;
        }
        throw error;
      }
    },
    async producerRuntimeIdentityAt(
      engineCommit: string,
      signal?: AbortSignal,
    ) {
      throwIfLedgerAborted(signal);
      if (
        authenticatedGuesserCommit === undefined
        || authenticatedEngineCommit === undefined
      ) {
        throw new TypeError(
          "Repository commits must be authenticated before producer runtime.",
        );
      }
      const resolvedEngine = await realpath(engineRepository);
      await requireCommit(resolvedEngine, engineCommit, signal);
      let reproduced = reproducedProducerRuntimeIdentities.get(engineCommit);
      if (reproduced === undefined) {
        const reproduce = options.reproduceProducerRuntimeIdentity
          ?? reproduceSchema9ProducerRuntimeIdentity;
        reproduced = reproduce(Object.freeze({
          engineRepository: resolvedEngine,
          engineCommit,
          ...(options.temporaryParent === undefined
            ? {}
            : { temporaryParent: options.temporaryParent }),
          ...(signal === undefined ? {} : { signal }),
        }));
        reproducedProducerRuntimeIdentities.set(engineCommit, reproduced);
      }
      return reproduced;
    },
    async executingCodeIdentity(signal?: AbortSignal) {
      throwIfLedgerAborted(signal);
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
        signal,
      );
      const manifest = await executingCodeManifest(
        resolvedGuesser,
        entrypoints,
        signal,
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
        ...(signal === undefined ? {} : { signal }),
      }));
      assertExactReproducibleExecutionIdentity(
        manifest.identity,
        await reproducedIdentity,
      );
      throwIfLedgerAborted(signal);
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

export async function runSchema9LedgerCli(
  options: Schema9LedgerCliOptions,
  io: Schema9LedgerCliIo = { stdout: process.stdout },
  signal?: AbortSignal,
): Promise<void> {
  const operation = (): Promise<void> => runSchema9LedgerCliOperation(
    options,
    io,
    signal,
  );
  if (signal === undefined) {
    return operation();
  }
  if (signal.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new Error("Schema-9 ledger operation was interrupted.");
  }
  return withSchema9CommandSignal(signal, operation);
}

async function runSchema9LedgerCliOperation(
  options: Schema9LedgerCliOptions,
  io: Schema9LedgerCliIo,
  signal: AbortSignal | undefined,
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
      ...(signal === undefined ? {} : { signal }),
    });
    if (options.operation === "create") {
      const artifact = await createSchema9CorpusLedger(corpus);
      const written = await publishOrAuthenticateSchema9CorpusLedgerArtifactAtomic(
        options.ledgerPath,
        artifact,
        signal,
      );
      const receipt = createSchema9LedgerVerificationReceipt(
        written.artifact,
        written.sha256,
      );
      let receiptWritten: Awaited<ReturnType<
        typeof publishOrAuthenticateSchema9LedgerVerificationReceipt
      >>;
      try {
        receiptWritten = await publishOrAuthenticateSchema9LedgerVerificationReceipt(
          verificationReceiptPath(options.ledgerPath, written.sha256),
          receipt,
          signal,
        );
      } catch (error: unknown) {
        if (written.created && !schema9PublicationMayBeCommitted(error)) {
          try {
            await removeSchema9StableFileIfOwned(
              written.publicationIdentity,
              written.sha256,
            );
          } catch (rollbackError: unknown) {
            throw new AggregateError(
              [error, rollbackError],
              "Schema-9 receipt publication failed and ledger rollback was incomplete.",
            );
          }
        }
        throw error;
      }
      await writeSchema9JsonLine(io.stdout, {
        format: SCHEMA9_CORPUS_LEDGER_FORMAT,
        version: SCHEMA9_CORPUS_LEDGER_VERSION,
        bytes: written.bytes,
        sha256: written.sha256,
        verificationReceiptSha256: receiptWritten.sha256,
      }, signal);
      return;
    }
    const authenticated = await loadAndReauthenticateSchema9CorpusLedgerWithIdentity(
      options.ledgerPath,
      corpus,
    );
    const { artifact, sha256 } = authenticated;
    const receipt = createSchema9LedgerVerificationReceipt(artifact, sha256);
    const receiptWritten = await publishOrAuthenticateSchema9LedgerVerificationReceipt(
      verificationReceiptPath(options.ledgerPath, sha256),
      receipt,
      signal,
    );
    await writeSchema9JsonLine(io.stdout, {
      format: SCHEMA9_CORPUS_LEDGER_FORMAT,
      version: SCHEMA9_CORPUS_LEDGER_VERSION,
      verified: true,
      sha256,
      verificationReceiptSha256: receiptWritten.sha256,
    }, signal);
  } finally {
    repositoryVerifier.disposeExecutionGuard();
  }
}

async function main(): Promise<void> {
  const termination = installSchema9TerminationSignal();
  const outputFailure = (error: Error): void => {
    termination.abort(error);
  };
  process.stdout.on("error", outputFailure);
  try {
    const arguments_ = process.argv.slice(2).filter(
      (argument) => argument !== "--",
    );
    await runSchema9LedgerCli(
      parseSchema9LedgerCliArguments(arguments_),
      { stdout: process.stdout },
      termination.signal,
    );
  } catch (error: unknown) {
    process.stderr.write(
      `Schema-9 corpus ledger failed: ${
        error instanceof Error ? error.message : String(error)
      }\n`,
    );
    process.exitCode = findSchema9TerminationError(error)?.exitCode ?? 1;
  } finally {
    process.stdout.removeListener("error", outputFailure);
    termination.dispose();
  }
}

const invokedPath = process.argv[1];
if (
  invokedPath !== undefined
  && import.meta.url === pathToFileURL(invokedPath).href
) {
  void main();
}
