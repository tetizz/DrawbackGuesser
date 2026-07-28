import { createHash, randomUUID } from "node:crypto";
import type { BigIntStats } from "node:fs";
import {
  link,
  lstat,
  open,
  readFile,
  realpath,
  rm,
  stat,
} from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { TextDecoder } from "node:util";
import {
  CAPTURABLE_HYPOTHESIS_RULE_IDS,
  RULE_OPPORTUNITY_FEATURE_FIELDS,
  RULE_OPPORTUNITY_FEATURE_VERSION,
} from "@drawbackguesser/predictor";
import {
  CAPTURABLE_SYMBOLIC_FEATURE_VERSION,
} from "./player-private-converter.js";
import {
  assertDistinctExplicitFiles,
  assertExactSchema9LabelBalance,
  assertScheduledConversionAccounting,
  authenticateSchema9Split,
  digestPartitionAssignments,
  type AuthenticatedSchema9Split,
} from "./schema9-ledger-authentication.js";
import {
  assertPathFreeJson,
  canonicalJsonBytes,
  checkedGitCommit,
  checkedSchema9SeedRoots,
  checkedScheduleId,
  checkedSha256,
  parseJsonWithoutDuplicateKeys,
  SCHEMA9_CORPUS_LEDGER_FORMAT,
  SCHEMA9_CORPUS_LEDGER_VERSION,
  SCHEMA9_LEDGER_SPLITS,
  SCHEMA9_PRODUCER_CONVERTER_POLICIES,
  SCHEMA9_SEED_STREAMS,
  type Schema9CorpusLedger,
  type Schema9CorpusLedgerOptions,
  type Schema9LedgerSplit,
} from "./schema9-ledger-types.js";

export type {
  Schema9CorpusLedger,
  Schema9CorpusLedgerOptions,
  Schema9LedgerSplit,
  Schema9ProducerConverterPolicy,
  Schema9RepositoryVerifier,
  Schema9SeedRoots,
  Schema9SplitFiles,
} from "./schema9-ledger-types.js";
export {
  SCHEMA9_CORPUS_LEDGER_FORMAT,
  SCHEMA9_CORPUS_LEDGER_VERSION,
  SCHEMA9_LEDGER_SPLITS,
  SCHEMA9_PRODUCER_CONVERTER_POLICIES,
  SCHEMA9_SEED_STREAMS,
  SCHEMA9_SPLIT_SEED_ROOTS,
} from "./schema9-ledger-types.js";

const MAX_LEDGER_BYTES = 8 * 1024 * 1024;
const UTF8 = new TextDecoder("utf-8", { fatal: true });

export interface WrittenSchema9CorpusLedger {
  readonly artifact: Schema9CorpusLedger;
  readonly bytes: number;
  readonly sha256: string;
}

function exactSplitRecord(
  value: Readonly<Record<Schema9LedgerSplit, unknown>>,
): void {
  const keys = Object.keys(value).sort();
  const expected = [...SCHEMA9_LEDGER_SPLITS].sort();
  if (
    keys.length !== expected.length
    || keys.some((key, index) => key !== expected[index])
  ) {
    throw new TypeError(
      "Schema-9 ledger requires exactly train, validation-a, "
      + "validation-b, and test inputs.",
    );
  }
}

function inputPaths(options: Schema9CorpusLedgerOptions): readonly string[] {
  return SCHEMA9_LEDGER_SPLITS.flatMap((split) => {
    const files = options.splits[split];
    return [
      files.tracePath,
      files.convertedPath,
      files.launchReceiptPath,
      files.completionReceiptPath,
    ];
  });
}

export async function verifySchema9RepositoryIdentity(
  options: Schema9CorpusLedgerOptions,
): Promise<{
  readonly guesserCommit: string;
  readonly converterEngineCommit: string;
}> {
  const guesserCommit = checkedGitCommit(
    options.guesserCommit,
    "guesserCommit",
  );
  const converterEngineCommit = checkedGitCommit(
    options.converterEngineCommit,
    "converterEngineCommit",
  );
  if (
    !SCHEMA9_PRODUCER_CONVERTER_POLICIES.includes(
      options.producerConverterPolicy,
    )
  ) {
    throw new TypeError("Producer/converter policy is unsupported.");
  }
  const pinned = checkedGitCommit(
    await options.repositoryVerifier.pinnedEngineCommitAt(guesserCommit),
    "pinned Engine submodule commit",
  );
  if (pinned !== converterEngineCommit) {
    throw new TypeError(
      "Guesser commit does not pin the declared converter Engine commit.",
    );
  }
  const producers = new Set(
    SCHEMA9_LEDGER_SPLITS.map((split) =>
      checkedGitCommit(
        options.splits[split].producerEngineCommit,
        `${split} producerEngineCommit`,
      )
    ),
  );
  if (options.producerConverterPolicy === "exact/v1") {
    if ([...producers].some((producer) => producer !== converterEngineCommit)) {
      throw new TypeError(
        "Exact producer/converter policy rejects an Engine commit mismatch.",
      );
    }
  } else {
    for (const producer of producers) {
      if (
        !(await options.repositoryVerifier.isEngineAncestor(
          converterEngineCommit,
          producer,
        ))
      ) {
        throw new TypeError(
          "Converter-ancestor policy rejects an unrelated producer commit.",
        );
      }
    }
  }
  return Object.freeze({ guesserCommit, converterEngineCommit });
}

function assertUniqueSchedules(options: Schema9CorpusLedgerOptions): void {
  const scheduleIds = SCHEMA9_LEDGER_SPLITS.map(
    (split) => options.splits[split].scheduleId,
  );
  if (new Set(scheduleIds).size !== scheduleIds.length) {
    throw new TypeError("Every corpus split must use a distinct schedule ID.");
  }
}

export function assertSchema9SplitsDisjoint(
  splits: readonly AuthenticatedSchema9Split[],
): void {
  const gameIds = new Set<string>();
  const seeds = new Set<number>();
  for (const split of splits) {
    for (const gameId of split.gameIds) {
      if (gameIds.has(gameId)) {
        throw new TypeError("Schema-9 splits overlap by game ID.");
      }
      gameIds.add(gameId);
    }
    for (const seed of split.simulationSeeds) {
      if (seeds.has(seed)) {
        throw new TypeError("Schema-9 splits overlap by simulation seed.");
      }
      seeds.add(seed);
    }
  }
}

function contentSha256(value: unknown): string {
  return createHash("sha256")
    .update(canonicalJsonBytes(value))
    .digest("hex");
}

function sortedValues<T extends string | number>(
  values: Iterable<T>,
): readonly T[] {
  return [...values].sort((left, right) => {
    if (typeof left === "number" && typeof right === "number") {
      return left - right;
    }
    const leftText = String(left);
    const rightText = String(right);
    return leftText < rightText ? -1 : leftText > rightText ? 1 : 0;
  });
}

function assertExactValues<T extends string | number>(
  declared: readonly T[],
  actual: Iterable<T>,
  label: string,
): void {
  const expected = sortedValues(actual);
  if (
    declared.length !== expected.length
    || declared.some((value, index) => value !== expected[index])
    || new Set(declared).size !== declared.length
  ) {
    throw new TypeError(`${label} must be the exact canonical set.`);
  }
}

function assertSplitLedgerIdentity(
  authenticated: AuthenticatedSchema9Split,
): void {
  const { ledger } = authenticated;
  checkedScheduleId(ledger.scheduleId, `${ledger.split} scheduleId`);
  checkedSchema9SeedRoots(ledger.seedRoots, ledger.split);
  checkedGitCommit(
    ledger.producerEngineCommit,
    `${ledger.split} producerEngineCommit`,
  );
  assertExactValues(
    ledger.sourceTrace.gameIds,
    authenticated.gameIds,
    `${ledger.split} source gameIds`,
  );
  assertExactValues(
    ledger.sourceTrace.simulationSeeds,
    authenticated.simulationSeeds,
    `${ledger.split} source simulationSeeds`,
  );
  if (
    ledger.sourceTrace.games !== ledger.sourceTrace.gameIds.length
    || ledger.sourceTrace.games
      !== ledger.sourceTrace.simulationSeeds.length
    || ledger.sourceTrace.gameIdSetSha256
      !== contentSha256(ledger.sourceTrace.gameIds)
    || ledger.sourceTrace.simulationSeedSetSha256
      !== contentSha256(ledger.sourceTrace.simulationSeeds)
  ) {
    throw new TypeError(
      `${ledger.split} source identity or set digest is inconsistent.`,
    );
  }
  assertExactSchema9LabelBalance(
    ledger.sourceTrace.games,
    ledger.sourceTrace.labelCountsByColor.white,
    ledger.sourceTrace.labelCountsByColor.black,
  );
  const convertedGameIds = new Set(ledger.converted.gameIds);
  const convertedSeeds = new Set(ledger.converted.simulationSeeds);
  assertExactValues(
    ledger.converted.gameIds,
    convertedGameIds,
    `${ledger.split} converted gameIds`,
  );
  assertExactValues(
    ledger.converted.simulationSeeds,
    convertedSeeds,
    `${ledger.split} converted simulationSeeds`,
  );
  if (
    ledger.converted.games !== convertedGameIds.size
    || ledger.converted.games !== convertedSeeds.size
    || ledger.converted.gameIdSetSha256
      !== contentSha256(ledger.converted.gameIds)
    || ledger.converted.simulationSeedSetSha256
      !== contentSha256(ledger.converted.simulationSeeds)
    || ledger.converted.gameIds.some(
      (gameId) => !authenticated.gameIds.has(gameId),
    )
    || ledger.converted.simulationSeeds.some(
      (seed) => !authenticated.simulationSeeds.has(seed),
    )
  ) {
    throw new TypeError(
      `${ledger.split} converted identity or set digest is inconsistent.`,
    );
  }
  assertScheduledConversionAccounting(
    ledger.sourceTrace.games,
    ledger.sourceTrace.zeroPlyGames,
    ledger.converted.games,
  );
}

/**
 * Authenticate the complete four-way schema-9 corpus without discovering
 * files from a directory. Every source trace is semantically replayed by the
 * pinned Engine parser and every converted row is reproduced byte-for-byte.
 */
export async function createSchema9CorpusLedger(
  options: Schema9CorpusLedgerOptions,
): Promise<Schema9CorpusLedger> {
  exactSplitRecord(options.splits);
  assertUniqueSchedules(options);
  await assertDistinctExplicitFiles(inputPaths(options));
  const identity = await verifySchema9RepositoryIdentity(options);
  const authenticated: AuthenticatedSchema9Split[] = [];
  for (const split of SCHEMA9_LEDGER_SPLITS) {
    authenticated.push(
      await authenticateSchema9Split(split, options.splits[split]),
    );
  }
  assertSchema9SplitsDisjoint(authenticated);
  return assembleSchema9CorpusLedger(
    identity,
    options.producerConverterPolicy,
    authenticated,
  );
}

export function assembleSchema9CorpusLedger(
  identity: Readonly<{
    readonly guesserCommit: string;
    readonly converterEngineCommit: string;
  }>,
  producerConverterPolicy:
    Schema9CorpusLedgerOptions["producerConverterPolicy"],
  authenticated: readonly AuthenticatedSchema9Split[],
): Schema9CorpusLedger {
  checkedGitCommit(identity.guesserCommit, "guesserCommit");
  checkedGitCommit(identity.converterEngineCommit, "converterEngineCommit");
  if (
    !SCHEMA9_PRODUCER_CONVERTER_POLICIES.includes(
      producerConverterPolicy,
    )
  ) {
    throw new TypeError("Producer/converter policy is unsupported.");
  }
  if (
    authenticated.length !== SCHEMA9_LEDGER_SPLITS.length
    || authenticated.some(
      (split, index) => split.ledger.split !== SCHEMA9_LEDGER_SPLITS[index],
    )
  ) {
    throw new TypeError(
      "Authenticated splits must use the exact canonical four-way order.",
    );
  }
  authenticated.forEach(assertSplitLedgerIdentity);
  assertSchema9SplitsDisjoint(authenticated);
  const payload = Object.freeze({
    format: SCHEMA9_CORPUS_LEDGER_FORMAT,
    version: SCHEMA9_CORPUS_LEDGER_VERSION,
    identity: Object.freeze({
      ...identity,
      producerConverterPolicy,
    }),
    scheduleContract: Object.freeze({
      authorityId: "capturable25-schema9-opportunity/v1" as const,
      seedStreams: SCHEMA9_SEED_STREAMS,
    }),
    opportunityContract: Object.freeze({
      authorityId: "capturable-king/v1" as const,
      symbolicFeatureVersion: CAPTURABLE_SYMBOLIC_FEATURE_VERSION,
      opportunityFeatureVersion: RULE_OPPORTUNITY_FEATURE_VERSION,
      ruleIds: Object.freeze([...CAPTURABLE_HYPOTHESIS_RULE_IDS]),
      fields: Object.freeze([...RULE_OPPORTUNITY_FEATURE_FIELDS]),
      shape: Object.freeze([
        CAPTURABLE_HYPOTHESIS_RULE_IDS.length,
        RULE_OPPORTUNITY_FEATURE_FIELDS.length,
      ] as const),
    }),
    splits: Object.freeze(authenticated.map((split) => split.ledger)),
    partition: Object.freeze({
      games: authenticated.reduce(
        (total, split) => total + split.ledger.sourceTrace.games,
        0,
      ),
      gameIdAssignmentsSha256: digestPartitionAssignments(
        authenticated,
        "gameIds",
      ),
      simulationSeedAssignmentsSha256: digestPartitionAssignments(
        authenticated,
        "simulationSeeds",
      ),
    }),
  });
  assertPathFreeJson(payload, "schema-9 corpus ledger");
  return Object.freeze({
    ...payload,
    contentSha256: contentSha256(payload),
  });
}

async function outputDestination(path: string): Promise<string> {
  if (path.length === 0) {
    throw new TypeError("Ledger output path must not be empty.");
  }
  try {
    await lstat(path);
    throw new FileExistsError("Schema-9 ledger output already exists.");
  } catch (error: unknown) {
    if (
      error instanceof FileExistsError
      || (
        typeof error === "object"
        && error !== null
        && "code" in error
        && error.code !== "ENOENT"
      )
    ) {
      throw error;
    }
  }
  const parent = await realpath(dirname(path));
  const parentInfo = await stat(parent);
  if (!parentInfo.isDirectory()) {
    throw new TypeError("Ledger output parent must be a directory.");
  }
  return join(parent, basename(path));
}

class FileExistsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FileExistsError";
  }
}

async function publishAtomicNoClobber(
  destination: string,
  payload: Buffer,
): Promise<void> {
  const temporary = join(
    dirname(destination),
    `${basename(destination)}.tmp-${String(process.pid)}-${randomUUID()}`,
  );
  let handle: Awaited<ReturnType<typeof open>> | undefined;
  try {
    handle = await open(temporary, "wx", 0o600);
    await handle.writeFile(payload);
    await handle.sync();
    await handle.close();
    handle = undefined;
    await link(temporary, destination);
    const published = await readFile(destination);
    if (!published.equals(payload)) {
      throw new Error("Published schema-9 ledger bytes changed.");
    }
  } finally {
    await handle?.close().catch(() => undefined);
    await rm(temporary, { force: true }).catch(() => undefined);
  }
}

export async function writeSchema9CorpusLedgerAtomic(
  outputPath: string,
  options: Schema9CorpusLedgerOptions,
): Promise<WrittenSchema9CorpusLedger> {
  const destination = await outputDestination(outputPath);
  const artifact = await createSchema9CorpusLedger(options);
  const bytes = canonicalJsonBytes(artifact);
  await publishAtomicNoClobber(destination, bytes);
  return writtenIdentity(artifact, bytes);
}

function writtenIdentity(
  artifact: Schema9CorpusLedger,
  bytes: Buffer,
): WrittenSchema9CorpusLedger {
  return Object.freeze({
    artifact,
    bytes: bytes.byteLength,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  });
}

export async function publishSchema9CorpusLedgerArtifactAtomic(
  outputPath: string,
  artifact: Schema9CorpusLedger,
): Promise<WrittenSchema9CorpusLedger> {
  const destination = await outputDestination(outputPath);
  const bytes = canonicalJsonBytes(artifact);
  verifySchema9CorpusLedgerReconstruction(bytes, artifact);
  await publishAtomicNoClobber(destination, bytes);
  return writtenIdentity(artifact, bytes);
}

function statSignature(value: BigIntStats): readonly bigint[] {
  return [
    value.dev,
    value.ino,
    value.size,
    value.mtimeNs,
    value.ctimeNs,
  ];
}

async function readStableLedger(path: string): Promise<Buffer> {
  const linkInfo = await lstat(path, { bigint: true });
  if (linkInfo.isSymbolicLink() || !linkInfo.isFile()) {
    throw new TypeError("Schema-9 ledger must be a regular non-symlink file.");
  }
  if (linkInfo.size <= 0n || linkInfo.size > BigInt(MAX_LEDGER_BYTES)) {
    throw new RangeError("Schema-9 ledger byte length is invalid.");
  }
  const resolved = await realpath(path);
  const before = await stat(resolved, { bigint: true });
  const bytes = await readFile(resolved);
  const after = await stat(resolved, { bigint: true });
  if (
    statSignature(before).some(
      (value, index) => value !== statSignature(after)[index],
    )
  ) {
    throw new Error("Schema-9 ledger changed while it was being read.");
  }
  return bytes;
}

function parsedCanonicalLedger(bytes: Buffer): unknown {
  let text: string;
  try {
    text = UTF8.decode(bytes);
  } catch (error: unknown) {
    throw new SyntaxError("Schema-9 ledger is not UTF-8.", { cause: error });
  }
  if (text.charCodeAt(0) === 0xfeff) {
    throw new SyntaxError("Schema-9 ledger must not contain a BOM.");
  }
  const value = parseJsonWithoutDuplicateKeys(text, "schema-9 ledger");
  if (!canonicalJsonBytes(value).equals(bytes)) {
    throw new TypeError("Schema-9 ledger JSON bytes are not canonical.");
  }
  if (
    typeof value !== "object"
    || value === null
    || Array.isArray(value)
  ) {
    throw new TypeError("Schema-9 ledger must be an object.");
  }
  const record = value as Readonly<Record<string, unknown>>;
  if (
    record["format"] !== SCHEMA9_CORPUS_LEDGER_FORMAT
    || record["version"] !== SCHEMA9_CORPUS_LEDGER_VERSION
    || typeof record["contentSha256"] !== "string"
  ) {
    throw new TypeError("Schema-9 ledger format or version is invalid.");
  }
  const declared = checkedSha256(
    record["contentSha256"],
    "schema-9 ledger contentSha256",
  );
  const payload = Object.fromEntries(
    Object.entries(record).filter(([key]) => key !== "contentSha256"),
  );
  if (contentSha256(payload) !== declared) {
    throw new TypeError("Schema-9 ledger content SHA-256 is invalid.");
  }
  return value;
}

/**
 * Re-open the canonical ledger, verify its self-digest, then independently
 * re-authenticate every caller-provided file and require byte-identical
 * reconstructed content.
 */
export async function loadAndReauthenticateSchema9CorpusLedger(
  ledgerPath: string,
  options: Schema9CorpusLedgerOptions,
): Promise<Schema9CorpusLedger> {
  const existingBytes = await readStableLedger(ledgerPath);
  const reconstructed = await createSchema9CorpusLedger(options);
  return verifySchema9CorpusLedgerReconstruction(
    existingBytes,
    reconstructed,
  );
}

export function verifySchema9CorpusLedgerReconstruction(
  existingBytes: Buffer,
  reconstructed: Schema9CorpusLedger,
): Schema9CorpusLedger {
  parsedCanonicalLedger(existingBytes);
  if (!canonicalJsonBytes(reconstructed).equals(existingBytes)) {
    throw new TypeError(
      "Schema-9 ledger no longer matches its authenticated corpus.",
    );
  }
  return reconstructed;
}

/**
 * Hash a canonical ledger file for a caller-managed outer receipt.
 */
export async function schema9CorpusLedgerFileSha256(
  ledgerPath: string,
): Promise<string> {
  const bytes = await readStableLedger(ledgerPath);
  parsedCanonicalLedger(bytes);
  return createHash("sha256").update(bytes).digest("hex");
}
