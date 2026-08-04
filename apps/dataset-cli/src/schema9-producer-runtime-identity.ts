import { createHash } from "node:crypto";
import {
  lstat,
  readFile,
  readdir,
  realpath,
} from "node:fs/promises";
import { createRequire } from "node:module";
import {
  dirname,
  extname,
  join,
  relative,
  sep,
} from "node:path";
import {
  canonicalJsonBytes,
  checkedSchema9ProducerRuntimeIdentity,
  SCHEMA9_PRODUCER_RUNTIME_IDENTITY_FORMAT,
  SCHEMA9_PRODUCER_RUNTIME_IDENTITY_VERSION,
  SCHEMA9_PRODUCER_RUNTIME_MANIFEST_ALGORITHM,
  type Schema9ProducerRuntimeComponentIdentity,
  type Schema9ProducerRuntimeIdentity,
} from "@drawbackguesser/trace-to-dataset";

const COMPONENT_MANIFEST_FORMAT =
  "drawbackengine-schema9-runtime-component/v1";
const COORDINATOR_COMPONENT_ID = "schema9-coordinator/v1" as const;
const PARALLEL_WORKER_COMPONENT_ID =
  "player-private-parallel-worker/v1" as const;
const RUNTIME_EXTENSIONS = new Set([
  ".cjs",
  ".js",
  ".json",
  ".mjs",
  ".node",
  ".wasm",
]);
const WORKSPACE_RUNTIME_PACKAGES = Object.freeze([
  "packages/shared",
  "packages/probe-search",
  "packages/drawback-engine",
  "packages/chess-core",
  "packages/drawback-search",
  "packages/chess-evaluator",
  "packages/simulation-trace",
  "packages/simulation-arena",
] as const);
const EXTERNAL_RUNTIME_BINDINGS = Object.freeze([
  Object.freeze({ consumer: "packages/chess-core", packageName: "chess.js" }),
  Object.freeze({ consumer: "packages/chess-core", packageName: "chessops" }),
  Object.freeze({ consumer: "packages/chess-evaluator", packageName: "chess.js" }),
  Object.freeze({ consumer: "packages/drawback-engine", packageName: "chess.js" }),
  Object.freeze({ consumer: "packages/drawback-search", packageName: "chessops" }),
] as const);
const ROOT_RUNTIME_INPUTS = Object.freeze([
  "package.json",
  "pnpm-lock.yaml",
  "pnpm-workspace.yaml",
] as const);
const OPTIONAL_ROOT_RUNTIME_INPUTS = Object.freeze([
  ".npmrc",
  ".pnpmfile.cjs",
  "pnpmfile.cjs",
] as const);
const BLOCKED_NODE_ENVIRONMENT = Object.freeze([
  "NODE_OPTIONS",
  "NODE_PATH",
  "NODE_PRESERVE_SYMLINKS",
  "NODE_PRESERVE_SYMLINKS_MAIN",
] as const);
const RUNTIME_STRING = /^[0-9A-Za-z._-]+$/u;
const MAX_EXTERNAL_DEPENDENCY_DEPTH = 64;
const EXTERNAL_DEPENDENCY_CYCLE_FORMAT =
  "drawbackengine-schema9-external-dependency-cycle/v1";
const EXTERNAL_OPTIONAL_DEPENDENCY_FORMAT =
  "drawbackengine-schema9-optional-dependency/v1";

interface RuntimeFileIdentity {
  readonly id: string;
  readonly bytes: number;
  readonly sha256: string;
}

export function schema9ProducerRuntimeDescriptor(
  input: Readonly<{
    nodeVersion?: string;
    platform?: string;
    architecture?: string;
    execArgv?: readonly string[];
    environment?: NodeJS.ProcessEnv;
  }> = {},
): Schema9ProducerRuntimeIdentity["runtime"] {
  const execArgv = input.execArgv ?? process.execArgv;
  const environment = input.environment ?? process.env;
  if (execArgv.length !== 0) {
    throw new TypeError(
      "Schema-9 producer verification rejects Node execution arguments.",
    );
  }
  for (const name of BLOCKED_NODE_ENVIRONMENT) {
    if ((environment[name] ?? "").trim().length !== 0) {
      throw new TypeError(
        `Schema-9 producer verification rejects ${name}.`,
      );
    }
  }
  const runtime = Object.freeze({
    nodeVersion: input.nodeVersion ?? process.version,
    platform: input.platform ?? process.platform,
    architecture: input.architecture ?? process.arch,
    execArgv: Object.freeze([] as const),
  });
  if (
    !/^v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$/u.test(
      runtime.nodeVersion,
    )
    || !RUNTIME_STRING.test(runtime.platform)
    || !RUNTIME_STRING.test(runtime.architecture)
  ) {
    throw new TypeError("Schema-9 producer runtime is unsupported.");
  }
  return runtime;
}

export async function computeSchema9ProducerRuntimeIdentity(
  repository: string,
  runtime: Schema9ProducerRuntimeIdentity["runtime"] =
    schema9ProducerRuntimeDescriptor(),
  signal?: AbortSignal,
): Promise<Schema9ProducerRuntimeIdentity> {
  throwIfAborted(signal);
  const root = await realpath(repository);
  const sharedEntries = await collectSharedEntries(root, signal);
  const coordinatorEntries = [
    ...sharedEntries,
    ...await collectWorkspacePackageEntries(
      root,
      "apps/engine-cli",
      signal,
    ),
    ...await collectWorkspaceDependencyBindingEntries(
      root,
      "apps/engine-cli",
      signal,
    ),
  ].sort(compareEntries);
  const workerEntries = [...sharedEntries].sort(compareEntries);
  requireEntry(
    coordinatorEntries,
    "repo:apps/engine-cli/dist/schema9-player-private-cli.js",
  );
  requireEntry(
    workerEntries,
    "repo:packages/simulation-arena/dist/player-private-parallel-worker.js",
  );
  const coordinator = componentIdentity(
    COORDINATOR_COMPONENT_ID,
    coordinatorEntries,
  );
  const parallelWorker = componentIdentity(
    PARALLEL_WORKER_COMPONENT_ID,
    workerEntries,
  );
  const payload = Object.freeze({
    format: SCHEMA9_PRODUCER_RUNTIME_IDENTITY_FORMAT,
    version: SCHEMA9_PRODUCER_RUNTIME_IDENTITY_VERSION,
    algorithm: SCHEMA9_PRODUCER_RUNTIME_MANIFEST_ALGORITHM,
    runtime,
    coordinator,
    parallelWorker,
  });
  return checkedSchema9ProducerRuntimeIdentity(Object.freeze({
    ...payload,
    aggregateSha256: sha256(canonicalJsonBytes(payload)),
  }));
}

async function collectSharedEntries(
  repository: string,
  signal: AbortSignal | undefined,
): Promise<RuntimeFileIdentity[]> {
  const entries: RuntimeFileIdentity[] = [];
  for (const path of ROOT_RUNTIME_INPUTS) {
    entries.push(await trackedTextIdentity(
      join(repository, ...path.split("/")),
      `repo:${path}`,
      signal,
    ));
  }
  for (const path of OPTIONAL_ROOT_RUNTIME_INPUTS) {
    const absolutePath = join(repository, path);
    if (await regularFileExists(absolutePath)) {
      entries.push(path.endsWith(".cjs")
        ? identityFromBytes(
          `repo:${path}`,
          await stableFileBytes(absolutePath, signal),
        )
        : await trackedTextIdentity(
          absolutePath,
          `repo:${path}`,
          signal,
        ));
    }
  }
  for (const packagePath of WORKSPACE_RUNTIME_PACKAGES) {
    entries.push(...await collectWorkspacePackageEntries(
      repository,
      packagePath,
      signal,
    ));
    entries.push(...await collectWorkspaceDependencyBindingEntries(
      repository,
      packagePath,
      signal,
    ));
  }
  for (const binding of EXTERNAL_RUNTIME_BINDINGS) {
    entries.push(...await collectExternalPackageEntries(
      repository,
      binding.consumer,
      binding.packageName,
      signal,
    ));
  }
  assertUniqueIds(entries);
  return entries;
}

async function collectWorkspacePackageEntries(
  repository: string,
  packagePath: string,
  signal: AbortSignal | undefined,
): Promise<RuntimeFileIdentity[]> {
  const root = join(repository, ...packagePath.split("/"));
  return [
    await trackedTextIdentity(
      join(root, "package.json"),
      `repo:${packagePath}/package.json`,
      signal,
    ),
    ...await collectRuntimeFiles(
      join(root, "dist"),
      (path) => `repo:${packagePath}/dist/${path}`,
      signal,
    ),
  ];
}

async function collectExternalPackageEntries(
  repository: string,
  consumer: string,
  expectedName: string,
  signal: AbortSignal | undefined,
): Promise<RuntimeFileIdentity[]> {
  const installed = join(
    repository,
    ...consumer.split("/"),
    "node_modules",
    ...expectedName.split("/"),
  );
  const root = await realpath(installed);
  return collectExternalPackageClosure(
    root,
    expectedName,
    `npm-binding:${consumer}`,
    signal,
    0,
    new Set(),
  );
}

async function collectExternalPackageClosure(
  packageRoot: string,
  expectedName: string,
  chainPrefix: string,
  signal: AbortSignal | undefined,
  depth: number,
  ancestorRoots: ReadonlySet<string>,
): Promise<RuntimeFileIdentity[]> {
  if (depth > MAX_EXTERNAL_DEPENDENCY_DEPTH) {
    throw new Error("Schema-9 external dependency graph is too deep.");
  }
  throwIfAborted(signal);
  const root = await realpath(packageRoot);
  const manifestBytes = await stableFileBytes(
    join(root, "package.json"),
    signal,
  );
  const manifest = JSON.parse(manifestBytes.toString("utf8")) as unknown;
  if (
    !isRecord(manifest)
    || manifest["name"] !== expectedName
    || typeof manifest["version"] !== "string"
    || !RUNTIME_STRING.test(manifest["version"])
  ) {
    throw new TypeError(
      `Schema-9 runtime package ${expectedName} is invalid.`,
    );
  }
  const prefix = `${chainPrefix}:${expectedName}@${manifest["version"]}`;
  if (ancestorRoots.has(root)) {
    return [identityFromBytes(
      `${prefix}:dependency-cycle.json`,
      canonicalJsonBytes(Object.freeze({
        format: EXTERNAL_DEPENDENCY_CYCLE_FORMAT,
        packageName: expectedName,
        version: manifest["version"],
      })),
    )];
  }
  const entries = [identityFromBytes(
    `${prefix}:package.json`,
    manifestBytes,
  )];
  entries.push(...await collectRuntimeFiles(
    root,
    (path) => `${prefix}:${path}`,
    signal,
    new Set(["node_modules"]),
    new Set(["package.json"]),
  ));
  const nextAncestors = new Set(ancestorRoots);
  nextAncestors.add(root);
  for (const dependency of externalDependencySpecifications(manifest)) {
    throwIfAborted(signal);
    let dependencyRoot: string;
    try {
      dependencyRoot = await resolveExternalDependencyRoot(
        root,
        dependency.name,
        signal,
      );
    } catch (error: unknown) {
      if (dependency.optional && isNodeError(error, "MODULE_NOT_FOUND")) {
        entries.push(identityFromBytes(
          `${prefix}:optional:${dependency.name}:absent.json`,
          canonicalJsonBytes(Object.freeze({
            format: EXTERNAL_OPTIONAL_DEPENDENCY_FORMAT,
            packageName: dependency.name,
            requested: dependency.requested,
            state: "absent",
          })),
        ));
        continue;
      }
      throw new Error(
        `Schema-9 runtime dependency ${dependency.name} could not be resolved.`,
        { cause: error },
      );
    }
    entries.push(...await collectExternalPackageClosure(
      dependencyRoot,
      dependency.name,
      `${prefix}:dependency`,
      signal,
      depth + 1,
      nextAncestors,
    ));
  }
  return entries;
}

interface ExternalDependencySpecification {
  readonly name: string;
  readonly requested: string;
  readonly optional: boolean;
}

function externalDependencySpecifications(
  manifest: Readonly<Record<string, unknown>>,
): readonly ExternalDependencySpecification[] {
  const dependencies = dependencyMap(manifest["dependencies"], false);
  const optionalDependencies = dependencyMap(
    manifest["optionalDependencies"],
    true,
  );
  const merged = new Map<string, ExternalDependencySpecification>();
  for (const dependency of [...dependencies, ...optionalDependencies]) {
    merged.set(dependency.name, dependency);
  }
  return [...merged.values()].sort((left, right) => (
    compareStrings(left.name, right.name)
  ));
}

function dependencyMap(
  value: unknown,
  optional: boolean,
): readonly ExternalDependencySpecification[] {
  if (value === undefined) {
    return [];
  }
  if (!isRecord(value)) {
    throw new TypeError("Schema-9 external dependency map is malformed.");
  }
  return Object.keys(value).sort(compareStrings).map((name) => {
    const requested = value[name];
    if (typeof requested !== "string" || requested.length === 0) {
      throw new TypeError(
        "Schema-9 external dependency version is malformed.",
      );
    }
    return Object.freeze({ name, requested, optional });
  });
}

async function resolveExternalDependencyRoot(
  packageRoot: string,
  dependencyName: string,
  signal: AbortSignal | undefined,
): Promise<string> {
  const requireFromPackage = createRequire(join(packageRoot, "package.json"));
  const resolvedEntry = requireFromPackage.resolve(dependencyName);
  let candidate = dirname(await realpath(resolvedEntry));
  for (;;) {
    throwIfAborted(signal);
    const manifestPath = join(candidate, "package.json");
    if (await regularFileExists(manifestPath)) {
      const manifestBytes = await stableFileBytes(manifestPath, signal);
      const manifest = JSON.parse(manifestBytes.toString("utf8")) as unknown;
      if (isRecord(manifest) && manifest["name"] === dependencyName) {
        return realpath(candidate);
      }
    }
    const parent = dirname(candidate);
    if (parent === candidate) {
      throw new Error(
        `Schema-9 resolved dependency ${dependencyName} has no package root.`,
      );
    }
    candidate = parent;
  }
}

async function collectWorkspaceDependencyBindingEntries(
  repository: string,
  consumer: string,
  signal: AbortSignal | undefined,
): Promise<RuntimeFileIdentity[]> {
  const consumerRoot = join(repository, ...consumer.split("/"));
  const manifestBytes = await stableFileBytes(
    join(consumerRoot, "package.json"),
    signal,
  );
  const manifest = JSON.parse(manifestBytes.toString("utf8")) as unknown;
  if (!isRecord(manifest)) {
    throw new TypeError("Schema-9 workspace package manifest is malformed.");
  }
  const dependencies = manifest["dependencies"];
  if (dependencies === undefined) {
    return [];
  }
  if (!isRecord(dependencies)) {
    throw new TypeError("Schema-9 workspace dependencies are malformed.");
  }
  const entries: RuntimeFileIdentity[] = [];
  for (const dependencyName of Object.keys(dependencies).sort()) {
    const version = dependencies[dependencyName];
    if (typeof version !== "string" || !version.startsWith("workspace:")) {
      continue;
    }
    const root = await realpath(join(
      consumerRoot,
      "node_modules",
      ...dependencyName.split("/"),
    ));
    const dependencyManifestBytes = await stableFileBytes(
      join(root, "package.json"),
      signal,
    );
    const dependencyManifest = JSON.parse(
      dependencyManifestBytes.toString("utf8"),
    ) as unknown;
    if (
      !isRecord(dependencyManifest)
      || dependencyManifest["name"] !== dependencyName
    ) {
      throw new TypeError(
        "Schema-9 workspace dependency binding has the wrong package identity.",
      );
    }
    const prefix = `workspace-binding:${consumer}:${dependencyName}`;
    entries.push(await trackedTextIdentity(
      join(root, "package.json"),
      `${prefix}:package.json`,
      signal,
    ));
    entries.push(...await collectRuntimeFiles(
      join(root, "dist"),
      (path) => `${prefix}:dist/${path}`,
      signal,
    ));
  }
  return entries;
}

async function collectRuntimeFiles(
  root: string,
  logicalId: (path: string) => string,
  signal: AbortSignal | undefined,
  excludedDirectories: ReadonlySet<string> = new Set(),
  excludedFiles: ReadonlySet<string> = new Set(),
): Promise<RuntimeFileIdentity[]> {
  const result: RuntimeFileIdentity[] = [];
  const visit = async (directory: string): Promise<void> => {
    throwIfAborted(signal);
    const children = await readdir(directory, { withFileTypes: true });
    children.sort((left, right) => compareStrings(left.name, right.name));
    for (const child of children) {
      throwIfAborted(signal);
      const path = join(directory, child.name);
      if (child.isSymbolicLink()) {
        throw new TypeError("Schema-9 runtime trees may not contain symlinks.");
      }
      if (child.isDirectory()) {
        if (!excludedDirectories.has(child.name)) {
          await visit(path);
        }
        continue;
      }
      if (!child.isFile()) {
        throw new TypeError(
          "Schema-9 runtime tree contains an unsupported file type.",
        );
      }
      const relativePath = portableRelative(root, path);
      if (
        excludedFiles.has(relativePath)
        || !RUNTIME_EXTENSIONS.has(extname(child.name).toLowerCase())
      ) {
        continue;
      }
      result.push(identityFromBytes(
        logicalId(relativePath),
        await stableFileBytes(path, signal),
      ));
    }
  };
  await visit(root);
  if (result.length === 0) {
    throw new TypeError("Schema-9 runtime component is empty.");
  }
  return result;
}

async function trackedTextIdentity(
  path: string,
  id: string,
  signal: AbortSignal | undefined,
): Promise<RuntimeFileIdentity> {
  const raw = await stableFileBytes(path, signal);
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  } catch (error: unknown) {
    throw new TypeError(
      "Schema-9 runtime metadata must be UTF-8.",
      { cause: error },
    );
  }
  const canonical = text.replace(/\r\n/gu, "\n");
  if (canonical.includes("\r")) {
    throw new TypeError(
      "Schema-9 runtime metadata contains a lone carriage return.",
    );
  }
  return identityFromBytes(id, Buffer.from(canonical, "utf8"));
}

async function stableFileBytes(
  path: string,
  signal: AbortSignal | undefined,
): Promise<Buffer> {
  throwIfAborted(signal);
  const before = await lstat(path, { bigint: true });
  if (!before.isFile()) {
    throw new TypeError("Schema-9 runtime input is not a regular file.");
  }
  const bytes = await readFile(
    path,
    signal === undefined ? undefined : { signal },
  );
  throwIfAborted(signal);
  const after = await lstat(path, { bigint: true });
  if (
    !after.isFile()
    || before.dev !== after.dev
    || before.ino !== after.ino
    || before.size !== after.size
    || before.mtimeNs !== after.mtimeNs
    || BigInt(bytes.length) !== before.size
  ) {
    throw new Error("Schema-9 runtime input changed while it was hashed.");
  }
  return bytes;
}

function componentIdentity<
  Id extends Schema9ProducerRuntimeComponentIdentity["componentId"],
>(
  componentId: Id,
  entries: readonly RuntimeFileIdentity[],
): Schema9ProducerRuntimeComponentIdentity & Readonly<{ componentId: Id }> {
  const manifest = Object.freeze({
    format: COMPONENT_MANIFEST_FORMAT,
    componentId,
    entries,
  });
  return Object.freeze({
    componentId,
    files: entries.length,
    bytes: entries.reduce((sum, entry) => sum + entry.bytes, 0),
    sha256: sha256(canonicalJsonBytes(manifest)),
  });
}

function identityFromBytes(id: string, bytes: Buffer): RuntimeFileIdentity {
  if (id.length === 0 || id.includes("\\") || /(?:^|:)\.\.?\//u.test(id)) {
    throw new TypeError("Schema-9 runtime logical ID is invalid.");
  }
  return Object.freeze({ id, bytes: bytes.length, sha256: sha256(bytes) });
}

function portableRelative(root: string, path: string): string {
  const value = relative(root, path);
  if (value.length === 0 || value === ".." || value.startsWith(`..${sep}`)) {
    throw new TypeError("Schema-9 runtime file escaped its root.");
  }
  return value.split(sep).join("/");
}

function requireEntry(entries: readonly RuntimeFileIdentity[], id: string): void {
  if (!entries.some((entry) => entry.id === id)) {
    throw new TypeError(`Schema-9 runtime is missing required module ${id}.`);
  }
}

function assertUniqueIds(entries: readonly RuntimeFileIdentity[]): void {
  const ids = new Set<string>();
  for (const entry of entries) {
    if (ids.has(entry.id)) {
      throw new TypeError("Schema-9 runtime contains duplicate logical IDs.");
    }
    ids.add(entry.id);
  }
}

function compareEntries(
  left: RuntimeFileIdentity,
  right: RuntimeFileIdentity,
): number {
  return compareStrings(left.id, right.id);
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function sha256(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new Error("Schema-9 producer verification was interrupted.");
  }
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNodeError(error: unknown, code: string): boolean {
  return error instanceof Error
    && "code" in error
    && error.code === code;
}

async function regularFileExists(path: string): Promise<boolean> {
  try {
    const metadata = await lstat(path);
    if (!metadata.isFile()) {
      throw new TypeError(
        "Optional Schema-9 runtime input is not a regular file.",
      );
    }
    return true;
  } catch (error: unknown) {
    if (
      error instanceof Error
      && "code" in error
      && error.code === "ENOENT"
    ) {
      return false;
    }
    throw error;
  }
}
