import { createHash } from "node:crypto";

export const SCHEMA9_CORPUS_LEDGER_FORMAT =
  "drawbackguesser-schema9-corpus-ledger" as const;
export const SCHEMA9_CORPUS_LEDGER_VERSION = 3 as const;
export const SCHEMA9_EXECUTION_MANIFEST_ALGORITHM =
  "sha256-loaded-module-graph-v2" as const;
export const SCHEMA9_GENERATOR_LAUNCH_FORMAT =
  "drawbackengine-player-private-schedule-launch" as const;
export const SCHEMA9_GENERATOR_COMPLETION_FORMAT =
  "drawbackengine-player-private-schedule-completion" as const;
export const SCHEMA9_GENERATOR_RECEIPT_VERSION = 3 as const;
export const SCHEMA9_PRODUCER_RUNTIME_IDENTITY_VERSION = 1 as const;
export const SCHEMA9_PRODUCER_RUNTIME_IDENTITY_FORMAT =
  "drawbackengine-schema9-producer-runtime" as const;
export const SCHEMA9_PRODUCER_RUNTIME_MANIFEST_ALGORITHM =
  "sha256-engine-runtime-tree-v1" as const;
export const SCHEMA9_GENERATION_CONFIG = Object.freeze({
  maxPlies: 120,
  maxDepth: 2,
  maxNodes: 50_000,
  temperatureCp: 35,
  topK: 8,
  leafCacheEntries: 16_384,
  leafCacheHistoryMode: "full",
  opponentAggregation: "worst-case",
  evaluator: Object.freeze({
    kind: "material",
    version: 1,
    evaluatorId: "drawback-material/v1",
  } as const),
  opponentHypotheses: Object.freeze({
    kind: "unrestricted-baseline",
    version: 1,
  } as const),
} as const);
export const SCHEMA9_SCHEDULE_PROFILE = Object.freeze({
  id: "standard",
  policyId: "material-player-private-corpus/v1",
} as const);
export const SCHEMA9_LEDGER_SPLITS = Object.freeze([
  "train",
  "validation-a",
  "validation-b",
  "test",
] as const);
export const SCHEMA9_PRODUCER_CONVERTER_POLICIES = Object.freeze([
  "exact/v1",
  "converter-ancestor/v1",
] as const);
export const SCHEMA9_SEED_STREAMS = Object.freeze([
  "label",
  "gameplay",
  "parameters",
] as const);
export const SCHEMA9_SPLIT_SEED_ROOTS = Object.freeze({
  train: Object.freeze([
    1_261_462_769,
    242_269_024,
    1_837_697_911,
  ] as const),
  "validation-a": Object.freeze([
    2_069_246_597,
    1_391_196_133,
    2_739_675_947,
  ] as const),
  "validation-b": Object.freeze([
    3_786_384_219,
    3_547_865_132,
    2_689_552_677,
  ] as const),
  test: Object.freeze([
    2_033_321_041,
    1_354_035_545,
    4_189_758_462,
  ] as const),
});

export type Schema9LedgerSplit = (typeof SCHEMA9_LEDGER_SPLITS)[number];
export type Schema9GenerationConfig = typeof SCHEMA9_GENERATION_CONFIG;
export type Schema9ProducerConverterPolicy =
  (typeof SCHEMA9_PRODUCER_CONVERTER_POLICIES)[number];
export type Schema9SeedRoots = readonly [number, number, number];

export interface Schema9SplitFiles {
  readonly tracePath: string;
  readonly convertedPath: string;
  readonly launchReceiptPath: string;
  readonly completionReceiptPath: string;
  readonly scheduleId: string;
  readonly seedRoots: Schema9SeedRoots;
  readonly producerEngineCommit: string;
}

export interface Schema9RepositoryVerifier {
  readonly pinnedEngineCommitAt:
    (guesserCommit: string, signal?: AbortSignal) => Promise<string>;
  readonly isEngineAncestor:
    (
      ancestorCommit: string,
      descendantCommit: string,
      signal?: AbortSignal,
    ) => Promise<boolean>;
  /**
   * Returns a content-derived identity for the modules executing the parser,
   * converter, scheduler, and ledger verifier. Callers cannot supply these
   * hashes as commit-shaped assertions.
   */
  readonly executingCodeIdentity:
    (signal?: AbortSignal) => Promise<Schema9ExecutionIdentity>;
  /**
   * Rebuilds the exact producer commit in a fresh isolated checkout and
   * returns the runtime-tree identity derived from that build. Receipt values
   * are never accepted as the authority for this identity.
   */
  readonly producerRuntimeIdentityAt:
    (
      engineCommit: string,
      signal?: AbortSignal,
    ) => Promise<Schema9ProducerRuntimeIdentity>;
}

export interface Schema9ExpectedAssignment {
  readonly gameIndex: number;
  readonly gameId: string;
  readonly seed: number;
  readonly parameterSeeds: {
    readonly white: number;
    readonly black: number;
  };
  readonly whiteRuleId: string;
  readonly blackRuleId: string;
  readonly initialFen?: string;
  readonly initialReplaySha256: string;
}

export interface Schema9AssignmentScheduler {
  readonly assignments: (
    split: Schema9LedgerSplit,
    gameCount: number,
    seedRoots: Schema9SeedRoots,
    signal?: AbortSignal,
  ) => Iterable<Schema9ExpectedAssignment>;
}

export interface Schema9ExecutionComponentIdentity {
  readonly entrypoint: string;
  readonly files: number;
  readonly bytes: number;
  readonly sha256: string;
}

export interface Schema9ExecutionRuntimeIdentity {
  readonly nodeVersion: string;
  readonly platform: string;
  readonly architecture: string;
  readonly execArgv: readonly string[];
}

export interface Schema9ExecutionIdentity {
  readonly algorithm: typeof SCHEMA9_EXECUTION_MANIFEST_ALGORITHM;
  readonly runtime: Schema9ExecutionRuntimeIdentity;
  readonly parser: Schema9ExecutionComponentIdentity;
  readonly converter: Schema9ExecutionComponentIdentity;
  readonly scheduler: Schema9ExecutionComponentIdentity;
  readonly verifier: Schema9ExecutionComponentIdentity;
  readonly aggregateSha256: string;
}

export interface Schema9ProducerRuntimeIdentity {
  readonly format: typeof SCHEMA9_PRODUCER_RUNTIME_IDENTITY_FORMAT;
  readonly version: typeof SCHEMA9_PRODUCER_RUNTIME_IDENTITY_VERSION;
  readonly algorithm: typeof SCHEMA9_PRODUCER_RUNTIME_MANIFEST_ALGORITHM;
  readonly runtime: Schema9ProducerRuntimeDescriptor;
  readonly coordinator:
    Schema9ProducerRuntimeComponentIdentity<"schema9-coordinator/v1">;
  readonly parallelWorker:
    Schema9ProducerRuntimeComponentIdentity<
      "player-private-parallel-worker/v1"
    >;
  readonly aggregateSha256: string;
}

export interface Schema9ProducerRuntimeDescriptor {
  readonly nodeVersion: string;
  readonly platform: string;
  readonly architecture: string;
  readonly execArgv: readonly [];
}

export interface Schema9ProducerRuntimeComponentIdentity<
  ComponentId extends
    | "schema9-coordinator/v1"
    | "player-private-parallel-worker/v1" =
      | "schema9-coordinator/v1"
      | "player-private-parallel-worker/v1",
> {
  readonly componentId: ComponentId;
  readonly files: number;
  readonly bytes: number;
  readonly sha256: string;
}

export interface Schema9CorpusLedgerOptions {
  readonly guesserCommit: string;
  readonly converterEngineCommit: string;
  readonly producerConverterPolicy: Schema9ProducerConverterPolicy;
  readonly repositoryVerifier: Schema9RepositoryVerifier;
  readonly assignmentScheduler: Schema9AssignmentScheduler;
  readonly splits: Readonly<Record<Schema9LedgerSplit, Schema9SplitFiles>>;
  readonly signal?: AbortSignal;
}

export interface Schema9ReceiptIdentity {
  readonly sha256: string;
  readonly bytes: number;
}

export interface Schema9LabelCounts {
  readonly white: Readonly<Record<string, number>>;
  readonly black: Readonly<Record<string, number>>;
}

export interface Schema9SourceTraceIdentity {
  readonly sha256: string;
  readonly bytes: number;
  readonly games: number;
  readonly zeroPlyGames: number;
  readonly gameIds: readonly string[];
  readonly simulationSeeds: readonly number[];
  readonly parameterSeeds: readonly number[];
  readonly gameIdSetSha256: string;
  readonly simulationSeedSetSha256: string;
  readonly parameterSeedSetSha256: string;
  readonly labelCountsByColor: Schema9LabelCounts;
}

export interface Schema9ConvertedIdentity {
  readonly sha256: string;
  readonly bytes: number;
  readonly rows: number;
  readonly games: number;
  readonly gameIds: readonly string[];
  readonly simulationSeeds: readonly number[];
  readonly gameIdSetSha256: string;
  readonly simulationSeedSetSha256: string;
}

export interface Schema9SplitLedger {
  readonly split: Schema9LedgerSplit;
  readonly scheduleId: string;
  readonly seedRoots: Schema9SeedRoots;
  readonly producerEngineCommit: string;
  readonly producerRuntimeIdentity: Schema9ProducerRuntimeIdentity;
  readonly generatorReceipts: {
    readonly launch: Schema9ReceiptIdentity;
    readonly completion: Schema9ReceiptIdentity;
  };
  readonly scheduleProfile: typeof SCHEMA9_SCHEDULE_PROFILE;
  readonly sourceTrace: Schema9SourceTraceIdentity;
  readonly converted: Schema9ConvertedIdentity;
}

export interface Schema9OpportunityContract {
  readonly authorityId: "capturable-king/v1";
  readonly symbolicFeatureVersion: 9;
  readonly opportunityFeatureVersion: 1;
  readonly ruleIds: readonly string[];
  readonly fields: readonly string[];
  readonly shape: readonly [number, number];
}

export interface Schema9CorpusLedger {
  readonly format: typeof SCHEMA9_CORPUS_LEDGER_FORMAT;
  readonly version: typeof SCHEMA9_CORPUS_LEDGER_VERSION;
  readonly identity: {
    readonly guesserCommit: string;
    readonly converterEngineCommit: string;
    readonly producerConverterPolicy: Schema9ProducerConverterPolicy;
    readonly execution: Schema9ExecutionIdentity;
    readonly producerRuntimeIdentity: Schema9ProducerRuntimeIdentity;
  };
  readonly scheduleContract: {
    readonly authorityId: "capturable25-schema9-opportunity/v1";
    readonly seedStreams: typeof SCHEMA9_SEED_STREAMS;
  };
  readonly opportunityContract: Schema9OpportunityContract;
  readonly splits: readonly Schema9SplitLedger[];
  readonly partition: {
    readonly games: number;
    readonly gameIdAssignmentsSha256: string;
    readonly simulationSeedAssignmentsSha256: string;
    readonly parameterSeedAssignmentsSha256: string;
  };
  readonly contentSha256: string;
}

const FULL_GIT_COMMIT = /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/u;
const LOWER_SHA256 = /^[0-9a-f]{64}$/u;
const IDENTIFIER = /^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$/u;
const URL_SCHEME = /^[A-Za-z][A-Za-z0-9+.-]*:/u;
const VERSIONED_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*\/v[0-9]+$/u;
const WINDOWS_ABSOLUTE = /^[A-Za-z]:[\\/]/u;
const WINDOWS_UNC = /^(?:\\\\|\/\/)[^/\\]/u;
const USER_DIRECTORY =
  /(?:^|[/\\])(?:Users|home)[/\\][^/\\]+(?:[/\\]|$)/iu;
const WINDOWS_RESERVED =
  /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$/iu;
const PRIVATE_TOKEN = /(?:password|passwd|secret|credential|api[-_.]?key|token)/iu;
const IDENTITY_EMBEDDED_BEFORE = /[\p{L}\p{N}]$/u;
const IDENTITY_EMBEDDED_AFTER = /^[\p{L}\p{N}]/u;

export function checkedGitCommit(value: unknown, label: string): string {
  if (typeof value !== "string" || !FULL_GIT_COMMIT.test(value)) {
    throw new TypeError(`${label} must be a full lowercase Git commit.`);
  }
  return value;
}

export function checkedSha256(value: unknown, label: string): string {
  if (typeof value !== "string" || !LOWER_SHA256.test(value)) {
    throw new TypeError(`${label} must be a lowercase SHA-256.`);
  }
  return value;
}

export function throwIfSchema9Aborted(
  signal: AbortSignal | undefined,
  label = "Schema-9 operation",
): void {
  if (signal?.aborted !== true) {
    return;
  }
  if (signal.reason instanceof Error) {
    throw signal.reason;
  }
  throw new Error(`${label} was interrupted.`, { cause: signal.reason });
}

function exactObjectKeys(
  value: Readonly<Record<string, unknown>>,
  expected: readonly string[],
  label: string,
): void {
  const actual = Object.keys(value).sort();
  const canonical = [...expected].sort();
  if (
    actual.length !== canonical.length
    || actual.some((key, index) => key !== canonical[index])
  ) {
    throw new TypeError(`${label} has invalid fields.`);
  }
}

function checkedProducerRuntimeComponent<
  ComponentId extends Schema9ProducerRuntimeComponentIdentity["componentId"],
>(
  value: unknown,
  label: string,
  expectedComponentId: ComponentId,
): Schema9ProducerRuntimeComponentIdentity<ComponentId> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`);
  }
  const record = value as Readonly<Record<string, unknown>>;
  exactObjectKeys(
    record,
    ["componentId", "files", "bytes", "sha256"],
    label,
  );
  if (record["componentId"] !== expectedComponentId) {
    throw new TypeError(`${label} componentId is unsupported.`);
  }
  const files = record["files"];
  const bytes = record["bytes"];
  if (
    typeof files !== "number"
    || !Number.isSafeInteger(files)
    || Object.is(files, -0)
    || files <= 0
    || typeof bytes !== "number"
    || !Number.isSafeInteger(bytes)
    || Object.is(bytes, -0)
    || bytes <= 0
    || typeof record["sha256"] !== "string"
  ) {
    throw new TypeError(`${label} size or digest is invalid.`);
  }
  return Object.freeze({
    componentId: expectedComponentId,
    files,
    bytes,
    sha256: checkedSha256(record["sha256"], `${label} SHA-256`),
  });
}

export function checkedSchema9ProducerRuntimeIdentity(
  value: unknown,
  label = "schema-9 producer runtime identity",
): Schema9ProducerRuntimeIdentity {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`);
  }
  const record = value as Readonly<Record<string, unknown>>;
  exactObjectKeys(record, [
    "format",
    "version",
    "algorithm",
    "runtime",
    "coordinator",
    "parallelWorker",
    "aggregateSha256",
  ], label);
  if (
    record["format"] !== SCHEMA9_PRODUCER_RUNTIME_IDENTITY_FORMAT
    || record["version"] !== SCHEMA9_PRODUCER_RUNTIME_IDENTITY_VERSION
    || record["algorithm"] !== SCHEMA9_PRODUCER_RUNTIME_MANIFEST_ALGORITHM
  ) {
    throw new TypeError(`${label} version or algorithm is unsupported.`);
  }
  const runtimeValue = record["runtime"];
  if (
    typeof runtimeValue !== "object"
    || runtimeValue === null
    || Array.isArray(runtimeValue)
  ) {
    throw new TypeError(`${label} runtime must be an object.`);
  }
  const runtimeRecord = runtimeValue as Readonly<Record<string, unknown>>;
  exactObjectKeys(
    runtimeRecord,
    ["nodeVersion", "platform", "architecture", "execArgv"],
    `${label} runtime`,
  );
  if (
    typeof runtimeRecord["nodeVersion"] !== "string"
    || !/^v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$/u.test(
      runtimeRecord["nodeVersion"],
    )
    || typeof runtimeRecord["platform"] !== "string"
    || typeof runtimeRecord["architecture"] !== "string"
    || !Array.isArray(runtimeRecord["execArgv"])
    || runtimeRecord["execArgv"].length !== 0
  ) {
    throw new TypeError(`${label} runtime is invalid.`);
  }
  const runtime = Object.freeze({
    nodeVersion: runtimeRecord["nodeVersion"],
    platform: runtimeRecord["platform"],
    architecture: runtimeRecord["architecture"],
    execArgv: Object.freeze([] as const),
  });
  if (
    !/^[0-9A-Za-z._-]+$/u.test(runtime.platform)
    || !/^[0-9A-Za-z._-]+$/u.test(runtime.architecture)
  ) {
    throw new TypeError(`${label} runtime is invalid.`);
  }
  const coordinator = checkedProducerRuntimeComponent(
    record["coordinator"],
    `${label} coordinator`,
    "schema9-coordinator/v1",
  );
  const parallelWorker = checkedProducerRuntimeComponent(
    record["parallelWorker"],
    `${label} parallel worker`,
    "player-private-parallel-worker/v1",
  );
  const payload = Object.freeze({
    format: SCHEMA9_PRODUCER_RUNTIME_IDENTITY_FORMAT,
    version: SCHEMA9_PRODUCER_RUNTIME_IDENTITY_VERSION,
    algorithm: SCHEMA9_PRODUCER_RUNTIME_MANIFEST_ALGORITHM,
    runtime,
    coordinator,
    parallelWorker,
  });
  const aggregateSha256 = typeof record["aggregateSha256"] === "string"
    ? checkedSha256(
        record["aggregateSha256"],
        `${label} aggregate SHA-256`,
      )
    : "";
  const expectedAggregate = createHash("sha256")
    .update(canonicalJsonBytes(payload))
    .digest("hex");
  if (aggregateSha256 !== expectedAggregate) {
    throw new TypeError(`${label} aggregate is inconsistent.`);
  }
  const identity = Object.freeze({ ...payload, aggregateSha256 });
  assertPathFreeJson(identity, label);
  return identity;
}

export function checkedScheduleId(value: unknown, label: string): string {
  if (
    typeof value !== "string"
    || !IDENTIFIER.test(value)
    || value.includes("..")
    || value.includes("\\")
    || WINDOWS_RESERVED.test(value)
    || PRIVATE_TOKEN.test(value)
  ) {
    throw new TypeError(
      `${label} must be a canonical path-free schedule identifier.`,
    );
  }
  return value;
}

export function checkedSchema9SeedRoots(
  value: readonly number[],
  split: Schema9LedgerSplit,
): Schema9SeedRoots {
  const expected = SCHEMA9_SPLIT_SEED_ROOTS[split];
  if (
    value.length !== SCHEMA9_SEED_STREAMS.length
    || value.some(
      (seed, index) =>
        !Number.isSafeInteger(seed)
        || seed < 0
        || seed > 0xffff_ffff
        || seed !== expected[index],
    )
  ) {
    throw new TypeError(
      `${split} seedRoots must exactly match the frozen `
      + "label, gameplay, and parameters roots.",
    );
  }
  return Object.freeze([...expected]);
}

function environmentUserTokens(): readonly string[] {
  const tokens = new Set<string>();
  for (const value of [
    process.env["USERNAME"],
    process.env["USER"],
    process.env["LOGNAME"],
  ]) {
    const canonical = value?.trim().toLocaleLowerCase("en-US");
    if (canonical !== undefined && canonical.length >= 3) {
      tokens.add(canonical);
    }
  }
  return [...tokens];
}

function looksLikePath(value: string): boolean {
  if (
    value.startsWith("/")
    || value.startsWith("~/")
    || value.startsWith("./")
    || value.startsWith("../")
    || WINDOWS_ABSOLUTE.test(value)
    || WINDOWS_UNC.test(value)
    || USER_DIRECTORY.test(value)
    || value.toLocaleLowerCase("en-US").startsWith("file:")
    || URL_SCHEME.test(value)
    || value.includes("\\")
    || /(?:^|\/)\.\.?($|\/)/u.test(value)
  ) {
    return true;
  }
  if (
    value.includes("/")
    && !VERSIONED_ID.test(value)
  ) {
    return true;
  }
  return false;
}

function containsDelimitedIdentity(
  value: string,
  rawToken: string,
): boolean {
  const token = rawToken.trim().toLocaleLowerCase("en-US");
  if (token.length === 0) {
    return false;
  }
  let offset = 0;
  while (offset <= value.length - token.length) {
    const index = value.indexOf(token, offset);
    if (index === -1) {
      return false;
    }
    if (
      !IDENTITY_EMBEDDED_BEFORE.test(value.slice(0, index))
      && !IDENTITY_EMBEDDED_AFTER.test(value.slice(index + token.length))
    ) {
      return true;
    }
    offset = index + 1;
  }
  return false;
}

/**
 * Reject filesystem locations and delimited current-account identities in a
 * generator receipt without treating an account name embedded inside a fixed
 * schema word as private data. Receipt content is never copied to the ledger,
 * but path-free inputs prevent a future diagnostic or extension from
 * publishing private workstation metadata.
 */
export function assertPathFreeJson(
  value: unknown,
  label: string,
  privateTokens: readonly string[] = environmentUserTokens(),
): void {
  const seen = new Set<object>();
  const visit = (current: unknown, path: string, depth: number): void => {
    if (depth > 128) {
      throw new TypeError(`${label} exceeds the supported JSON depth.`);
    }
    if (typeof current === "string") {
      const lowered = current.toLocaleLowerCase("en-US");
      if (
        current.length > 4096
        || looksLikePath(current)
        || privateTokens.some((token) =>
          containsDelimitedIdentity(lowered, token)
        )
        || PRIVATE_TOKEN.test(current)
      ) {
        throw new TypeError(`${path} contains private path or user data.`);
      }
      return;
    }
    if (
      current === null
      || typeof current === "boolean"
    ) {
      return;
    }
    if (typeof current === "number") {
      if (!Number.isFinite(current) || Object.is(current, -0)) {
        throw new TypeError(`${path} contains a non-canonical JSON number.`);
      }
      return;
    }
    if (Array.isArray(current)) {
      if (seen.has(current)) {
        throw new TypeError(`${label} contains a JSON cycle.`);
      }
      seen.add(current);
      current.forEach((item, index) => {
        visit(item, `${path}[${String(index)}]`, depth + 1);
      });
      seen.delete(current);
      return;
    }
    if (typeof current === "object") {
      if (seen.has(current)) {
        throw new TypeError(`${label} contains a JSON cycle.`);
      }
      seen.add(current);
      for (const [key, item] of Object.entries(current)) {
        visit(key, `${path} key`, depth + 1);
        visit(item, `${path}.${key}`, depth + 1);
      }
      seen.delete(current);
      return;
    }
    throw new TypeError(`${label} contains a non-JSON value.`);
  };
  visit(value, label, 0);
}

class DuplicateKeyJsonScanner {
  readonly #text: string;
  #index = 0;

  constructor(text: string) {
    this.#text = text;
  }

  scan(): void {
    this.#value(0);
    this.#whitespace();
    if (this.#index !== this.#text.length) {
      throw new SyntaxError("JSON has trailing content.");
    }
  }

  #value(depth: number): void {
    if (depth > 128) {
      throw new SyntaxError("JSON exceeds the supported nesting depth.");
    }
    this.#whitespace();
    const token = this.#text[this.#index];
    if (token === "{") {
      this.#object(depth + 1);
      return;
    }
    if (token === "[") {
      this.#array(depth + 1);
      return;
    }
    if (token === "\"") {
      this.#string();
      return;
    }
    if (token === "t") {
      this.#literal("true");
      return;
    }
    if (token === "f") {
      this.#literal("false");
      return;
    }
    if (token === "n") {
      this.#literal("null");
      return;
    }
    this.#number();
  }

  #object(depth: number): void {
    this.#index += 1;
    this.#whitespace();
    const keys = new Set<string>();
    if (this.#text[this.#index] === "}") {
      this.#index += 1;
      return;
    }
    for (;;) {
      this.#whitespace();
      if (this.#text[this.#index] !== "\"") {
        throw new SyntaxError("JSON object key must be a string.");
      }
      const key = this.#string();
      if (keys.has(key)) {
        throw new SyntaxError(`JSON object contains duplicate key ${key}.`);
      }
      keys.add(key);
      this.#whitespace();
      if (this.#text[this.#index] !== ":") {
        throw new SyntaxError("JSON object key is missing a colon.");
      }
      this.#index += 1;
      this.#value(depth);
      this.#whitespace();
      const delimiter = this.#text[this.#index];
      if (delimiter === "}") {
        this.#index += 1;
        return;
      }
      if (delimiter !== ",") {
        throw new SyntaxError("JSON object entry is missing a delimiter.");
      }
      this.#index += 1;
    }
  }

  #array(depth: number): void {
    this.#index += 1;
    this.#whitespace();
    if (this.#text[this.#index] === "]") {
      this.#index += 1;
      return;
    }
    for (;;) {
      this.#value(depth);
      this.#whitespace();
      const delimiter = this.#text[this.#index];
      if (delimiter === "]") {
        this.#index += 1;
        return;
      }
      if (delimiter !== ",") {
        throw new SyntaxError("JSON array entry is missing a delimiter.");
      }
      this.#index += 1;
    }
  }

  #string(): string {
    const start = this.#index;
    this.#index += 1;
    for (;;) {
      const token = this.#text[this.#index];
      if (token === undefined) {
        throw new SyntaxError("JSON string is unterminated.");
      }
      if (token === "\"") {
        this.#index += 1;
        const parsed: unknown = JSON.parse(
          this.#text.slice(start, this.#index),
        );
        if (typeof parsed !== "string") {
          throw new SyntaxError("JSON string parser returned a non-string.");
        }
        return parsed;
      }
      if (token === "\\") {
        this.#index += 1;
        const escape = this.#text[this.#index];
        if (escape === "u") {
          const hex = this.#text.slice(this.#index + 1, this.#index + 5);
          if (!/^[0-9A-Fa-f]{4}$/u.test(hex)) {
            throw new SyntaxError("JSON unicode escape is invalid.");
          }
          this.#index += 5;
          continue;
        }
        if (
          escape === undefined
          || !"\"\\/bfnrt".includes(escape)
        ) {
          throw new SyntaxError("JSON string escape is invalid.");
        }
        this.#index += 1;
        continue;
      }
      if (token.charCodeAt(0) < 0x20) {
        throw new SyntaxError("JSON string contains a control character.");
      }
      this.#index += 1;
    }
  }

  #number(): void {
    const remainder = this.#text.slice(this.#index);
    const match =
      /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/u.exec(
        remainder,
      );
    if (match === null) {
      throw new SyntaxError("JSON value is invalid.");
    }
    this.#index += match[0].length;
  }

  #literal(literal: string): void {
    if (this.#text.slice(this.#index, this.#index + literal.length) !== literal) {
      throw new SyntaxError("JSON literal is invalid.");
    }
    this.#index += literal.length;
  }

  #whitespace(): void {
    for (;;) {
      const token = this.#text[this.#index];
      if (token === undefined || !" \t\r\n".includes(token)) {
        return;
      }
      this.#index += 1;
    }
  }
}

export function parseJsonWithoutDuplicateKeys(
  text: string,
  label: string,
): unknown {
  try {
    new DuplicateKeyJsonScanner(text).scan();
    return JSON.parse(text) as unknown;
  } catch (error: unknown) {
    throw new SyntaxError(`${label} is not strict JSON.`, { cause: error });
  }
}

function canonicalValue(value: unknown): unknown {
  if (
    value === null
    || typeof value === "string"
    || typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError("Canonical JSON cannot contain a non-finite number.");
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(canonicalValue);
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [
          key,
          canonicalValue((value as Readonly<Record<string, unknown>>)[key]),
        ]),
    );
  }
  throw new TypeError("Canonical JSON received a non-JSON value.");
}

export function canonicalJsonBytes(value: unknown): Buffer {
  return Buffer.from(
    `${JSON.stringify(canonicalValue(value))}\n`,
    "utf8",
  );
}
