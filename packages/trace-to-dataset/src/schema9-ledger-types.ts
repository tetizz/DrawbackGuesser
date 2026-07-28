export const SCHEMA9_CORPUS_LEDGER_FORMAT =
  "drawbackguesser-schema9-corpus-ledger" as const;
export const SCHEMA9_CORPUS_LEDGER_VERSION = 1 as const;
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
    (guesserCommit: string) => Promise<string>;
  readonly isEngineAncestor:
    (ancestorCommit: string, descendantCommit: string) => Promise<boolean>;
}

export interface Schema9CorpusLedgerOptions {
  readonly guesserCommit: string;
  readonly converterEngineCommit: string;
  readonly producerConverterPolicy: Schema9ProducerConverterPolicy;
  readonly repositoryVerifier: Schema9RepositoryVerifier;
  readonly splits: Readonly<Record<Schema9LedgerSplit, Schema9SplitFiles>>;
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
  readonly gameIdSetSha256: string;
  readonly simulationSeedSetSha256: string;
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
  readonly generatorReceipts: {
    readonly launch: Schema9ReceiptIdentity;
    readonly completion: Schema9ReceiptIdentity;
  };
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
  };
  readonly contentSha256: string;
}

const FULL_GIT_COMMIT = /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/u;
const LOWER_SHA256 = /^[0-9a-f]{64}$/u;
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,199}$/u;
const URL_SCHEME = /^[A-Za-z][A-Za-z0-9+.-]*:\/\//u;
const VERSIONED_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*\/v[0-9]+$/u;
const WINDOWS_ABSOLUTE = /^[A-Za-z]:[\\/]/u;
const WINDOWS_UNC = /^(?:\\\\|\/\/)[^/\\]/u;
const USER_DIRECTORY =
  /(?:^|[/\\])(?:Users|home)[/\\][^/\\]+(?:[/\\]|$)/iu;

export function checkedGitCommit(value: string, label: string): string {
  if (!FULL_GIT_COMMIT.test(value)) {
    throw new TypeError(`${label} must be a full lowercase Git commit.`);
  }
  return value;
}

export function checkedSha256(value: string, label: string): string {
  if (!LOWER_SHA256.test(value)) {
    throw new TypeError(`${label} must be a lowercase SHA-256.`);
  }
  return value;
}

export function checkedScheduleId(value: string, label: string): string {
  if (
    !IDENTIFIER.test(value)
    || value.includes("..")
    || value.includes("\\")
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
    || value.includes("\\")
    || /(?:^|\/)\.\.?($|\/)/u.test(value)
  ) {
    return true;
  }
  if (
    value.includes("/")
    && !URL_SCHEME.test(value)
    && !VERSIONED_ID.test(value)
  ) {
    return true;
  }
  return false;
}

/**
 * Reject filesystem locations and the current account identity anywhere in a
 * generator receipt. Receipt content is never copied to the ledger, but
 * path-free inputs prevent a future diagnostic or extension from publishing
 * private workstation metadata.
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
        looksLikePath(current)
        || privateTokens.some((token) => lowered === token)
      ) {
        throw new TypeError(`${path} contains private path or user data.`);
      }
      return;
    }
    if (
      current === null
      || typeof current === "boolean"
      || typeof current === "number"
    ) {
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
