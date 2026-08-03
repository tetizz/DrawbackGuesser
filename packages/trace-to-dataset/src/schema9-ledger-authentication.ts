import { createHash } from "node:crypto";
import {
  createReadStream,
  type BigIntStats,
} from "node:fs";
import {
  lstat,
  readFile,
  realpath,
  stat,
} from "node:fs/promises";
import { TextDecoder } from "node:util";
import {
  parsePlayerPrivateSimulationTraceLine,
  type PlayerPrivateSimulationTraceRecord,
} from "@drawbackengine/simulation-trace";
import { CapturableKingPosition } from "@drawbackengine/chess-core";
import {
  CAPTURABLE_HYPOTHESIS_RULE_IDS,
} from "@drawbackguesser/predictor";
import {
  convertParsedPlayerPrivateTraceToDatasetRows,
} from "./player-private-converter.js";
import {
  assertPathFreeJson,
  canonicalJsonBytes,
  checkedGitCommit,
  checkedSchema9SeedRoots,
  checkedScheduleId,
  checkedSha256,
  parseJsonWithoutDuplicateKeys,
  SCHEMA9_GENERATOR_COMPLETION_FORMAT,
  SCHEMA9_GENERATOR_LAUNCH_FORMAT,
  SCHEMA9_GENERATOR_RECEIPT_VERSION,
  SCHEMA9_SCHEDULE_PROFILE,
  type Schema9AssignmentScheduler,
  type Schema9ConvertedIdentity,
  type Schema9LedgerSplit,
  type Schema9ReceiptIdentity,
  type Schema9SeedRoots,
  type Schema9SourceTraceIdentity,
  type Schema9SplitFiles,
  type Schema9SplitLedger,
} from "./schema9-ledger-types.js";

const MAX_TRACE_LINE_BYTES = 64 * 1024 * 1024;
const MAX_DATASET_LINE_BYTES = 8 * 1024 * 1024;
const MAX_RECEIPT_BYTES = 1024 * 1024;
const UTF8 = new TextDecoder("utf-8", { fatal: true });
const CAPTURABLE_RULE_ID_SET: ReadonlySet<string> = new Set(
  CAPTURABLE_HYPOTHESIS_RULE_IDS,
);
const DEFAULT_INITIAL_FEN = CapturableKingPosition.fromFen().fen;

interface StableFile {
  readonly requestedPath: string;
  readonly resolvedPath: string;
  readonly before: BigIntStats;
}

interface SourceGame {
  readonly gameId: string;
  readonly seed: number;
  readonly gameIndex: number;
  readonly parameterSeeds: {
    readonly white: number;
    readonly black: number;
  };
  readonly plies: number;
  readonly whiteRuleId: string;
  readonly blackRuleId: string;
  readonly initialFen: string;
  readonly policyId: string;
  readonly initialReplaySha256: string;
}

interface AuthenticatedReceipt {
  readonly identity: Schema9ReceiptIdentity;
  readonly value: Readonly<Record<string, unknown>>;
}

export interface AuthenticatedSchema9Split {
  readonly ledger: Schema9SplitLedger;
  readonly gameIds: ReadonlySet<string>;
  readonly simulationSeeds: ReadonlySet<number>;
  readonly parameterSeeds: ReadonlySet<number>;
}

function fileSignature(value: BigIntStats): readonly bigint[] {
  return Object.freeze([
    value.dev,
    value.ino,
    value.size,
    value.mtimeNs,
    value.ctimeNs,
  ]);
}

function sameSignature(left: BigIntStats, right: BigIntStats): boolean {
  const leftSignature = fileSignature(left);
  const rightSignature = fileSignature(right);
  return leftSignature.every(
    (value, index) => value === rightSignature[index],
  );
}

async function openStableFile(
  path: string,
  label: string,
): Promise<StableFile> {
  if (path.length === 0) {
    throw new TypeError(`${label} path must not be empty.`);
  }
  const linkInfo = await lstat(path, { bigint: true });
  if (linkInfo.isSymbolicLink()) {
    throw new TypeError(`${label} must not be a symbolic link.`);
  }
  if (!linkInfo.isFile()) {
    throw new TypeError(`${label} must be a regular file.`);
  }
  const resolvedPath = await realpath(path);
  const before = await stat(resolvedPath, { bigint: true });
  if (!before.isFile()) {
    throw new TypeError(`${label} must resolve to a regular file.`);
  }
  return {
    requestedPath: path,
    resolvedPath,
    before,
  };
}

async function assertUnchanged(file: StableFile, label: string): Promise<void> {
  const after = await stat(file.resolvedPath, { bigint: true });
  if (!sameSignature(file.before, after)) {
    throw new Error(`${label} changed while it was being authenticated.`);
  }
}

function fileObjectKey(file: StableFile): string {
  return `${file.before.dev.toString()}:${file.before.ino.toString()}`;
}

export async function assertDistinctExplicitFiles(
  paths: readonly string[],
): Promise<void> {
  const seenObjects = new Set<string>();
  const seenResolved = new Set<string>();
  for (const [index, path] of paths.entries()) {
    const file = await openStableFile(path, `input[${String(index)}]`);
    const objectKey = fileObjectKey(file);
    const normalizedPath = file.resolvedPath.toLocaleLowerCase("en-US");
    if (
      seenObjects.has(objectKey)
      || seenResolved.has(normalizedPath)
    ) {
      throw new TypeError(
        "Every ledger input must be an explicit, distinct file.",
      );
    }
    seenObjects.add(objectKey);
    seenResolved.add(normalizedPath);
  }
}

async function authenticateReceipt(
  path: string,
  label: string,
): Promise<AuthenticatedReceipt> {
  const file = await openStableFile(path, label);
  if (file.before.size <= 0n || file.before.size > BigInt(MAX_RECEIPT_BYTES)) {
    throw new RangeError(
      `${label} must be non-empty and at most ${String(MAX_RECEIPT_BYTES)} bytes.`,
    );
  }
  const bytes = await readFile(file.resolvedPath);
  await assertUnchanged(file, label);
  let text: string;
  try {
    text = UTF8.decode(bytes);
  } catch (error: unknown) {
    throw new SyntaxError(`${label} is not UTF-8.`, { cause: error });
  }
  if (text.charCodeAt(0) === 0xfeff) {
    throw new SyntaxError(`${label} must not contain a UTF-8 BOM.`);
  }
  const value = parseJsonWithoutDuplicateKeys(text, label);
  if (
    typeof value !== "object"
    || value === null
    || Array.isArray(value)
  ) {
    throw new TypeError(`${label} must contain one JSON object.`);
  }
  assertPathFreeJson(value, label);
  return Object.freeze({
    identity: Object.freeze({
      sha256: createHash("sha256").update(bytes).digest("hex"),
      bytes: bytes.byteLength,
    }),
    value: value as Readonly<Record<string, unknown>>,
  });
}

function receiptObject(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function exactReceiptKeys(
  value: object,
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

function receiptInteger(
  value: unknown,
  label: string,
  minimum = 0,
): number {
  if (
    typeof value !== "number"
    || !Number.isSafeInteger(value)
    || Object.is(value, -0)
    || value < minimum
  ) {
    throw new TypeError(`${label} must be a canonical safe integer.`);
  }
  return value;
}

function validateLaunchReceipt(
  receipt: AuthenticatedReceipt,
  split: Schema9LedgerSplit,
  scheduleId: string,
  seedRoots: readonly number[],
  producerEngineCommit: string,
): Readonly<Record<string, unknown>> {
  const value = receipt.value;
  exactReceiptKeys(value, [
    "format",
    "version",
    "scheduleAuthorityId",
    "scheduleId",
    "ledgerSplit",
    "engineSplit",
    "splitCounts",
    "seedRoots",
    "scheduleProfile",
    "producerEngineCommit",
  ], `${split} launch receipt`);
  if (
    value["format"] !== SCHEMA9_GENERATOR_LAUNCH_FORMAT
    || value["version"] !== SCHEMA9_GENERATOR_RECEIPT_VERSION
    || value["scheduleAuthorityId"]
      !== "capturable25-schema9-opportunity/v1"
    || value["scheduleId"] !== scheduleId
    || value["ledgerSplit"] !== split
    || value["engineSplit"] !== "train"
    || value["producerEngineCommit"] !== producerEngineCommit
  ) {
    throw new TypeError(`${split} launch receipt identity is inconsistent.`);
  }
  const declaredRoots = value["seedRoots"];
  if (
    !Array.isArray(declaredRoots)
    || declaredRoots.length !== seedRoots.length
    || declaredRoots.some((root, index) => root !== seedRoots[index])
  ) {
    throw new TypeError(`${split} launch receipt seed roots are inconsistent.`);
  }
  const profile = receiptObject(
    value["scheduleProfile"],
    `${split} launch schedule profile`,
  );
  exactReceiptKeys(profile, ["id", "policyId"], `${split} schedule profile`);
  if (
    profile["id"] !== SCHEMA9_SCHEDULE_PROFILE.id
    || profile["policyId"] !== SCHEMA9_SCHEDULE_PROFILE.policyId
  ) {
    throw new TypeError(`${split} launch receipt profile is unsupported.`);
  }
  const counts = receiptObject(
    value["splitCounts"],
    `${split} launch split counts`,
  );
  exactReceiptKeys(counts, ["train", "validation", "test"], `${split} split counts`);
  receiptInteger(counts["train"], `${split} train count`, 1);
  receiptInteger(counts["validation"], `${split} validation count`);
  receiptInteger(counts["test"], `${split} test count`);
  if (counts["validation"] !== 0 || counts["test"] !== 0) {
    throw new TypeError(
      `${split} launch receipt must use an isolated Engine train schedule.`,
    );
  }
  return counts;
}

function validateCompletionReceipt(
  receipt: AuthenticatedReceipt,
  split: Schema9LedgerSplit,
  scheduleId: string,
  producerEngineCommit: string,
  launchSha256: string,
  trace: Readonly<{ sha256: string; bytes: number; games: number }>,
): void {
  const value = receipt.value;
  exactReceiptKeys(value, [
    "format",
    "version",
    "scheduleId",
    "ledgerSplit",
    "state",
    "producerEngineCommit",
    "launchReceiptSha256",
    "output",
  ], `${split} completion receipt`);
  if (
    value["format"] !== SCHEMA9_GENERATOR_COMPLETION_FORMAT
    || value["version"] !== SCHEMA9_GENERATOR_RECEIPT_VERSION
    || value["scheduleId"] !== scheduleId
    || value["ledgerSplit"] !== split
    || value["state"] !== "completed"
    || value["producerEngineCommit"] !== producerEngineCommit
    || checkedSha256(
      String(value["launchReceiptSha256"]),
      `${split} completion launchReceiptSha256`,
    ) !== launchSha256
  ) {
    throw new TypeError(`${split} completion receipt identity is inconsistent.`);
  }
  const output = receiptObject(value["output"], `${split} completion output`);
  exactReceiptKeys(
    output,
    ["sha256", "bytes", "games", "firstGameIndex", "lastGameIndex"],
    `${split} completion output`,
  );
  if (
    checkedSha256(String(output["sha256"]), `${split} output SHA-256`)
      !== trace.sha256
    || receiptInteger(output["bytes"], `${split} output bytes`, 1)
      !== trace.bytes
    || receiptInteger(output["games"], `${split} output games`, 1)
      !== trace.games
    || receiptInteger(output["firstGameIndex"], `${split} first game index`)
      !== 0
    || receiptInteger(output["lastGameIndex"], `${split} last game index`)
      !== trace.games - 1
  ) {
    throw new TypeError(`${split} completion receipt output is inconsistent.`);
  }
}

async function* lfLines(
  file: StableFile,
  label: string,
  maximumLineBytes: number,
): AsyncGenerator<Buffer> {
  const stream = createReadStream(file.resolvedPath);
  let fragments: Buffer[] = [];
  let bufferedBytes = 0;
  let lineNumber = 0;
  for await (const rawChunk of stream) {
    const chunk = Buffer.isBuffer(rawChunk)
      ? rawChunk
      : Buffer.from(rawChunk as Uint8Array);
    let cursor = 0;
    let newline = chunk.indexOf(0x0a, cursor);
    while (newline >= 0) {
      const fragment = chunk.subarray(cursor, newline + 1);
      bufferedBytes += fragment.byteLength;
      if (bufferedBytes > maximumLineBytes) {
        throw new RangeError(
          `${label} line ${String(lineNumber + 1)} exceeds the byte limit.`,
        );
      }
      fragments.push(fragment);
      lineNumber += 1;
      yield Buffer.concat(fragments, bufferedBytes);
      fragments = [];
      bufferedBytes = 0;
      cursor = newline + 1;
      newline = chunk.indexOf(0x0a, cursor);
    }
    const tail = chunk.subarray(cursor);
    bufferedBytes += tail.byteLength;
    if (bufferedBytes > maximumLineBytes) {
      throw new RangeError(
        `${label} line ${String(lineNumber + 1)} exceeds the byte limit.`,
      );
    }
    if (tail.byteLength > 0) {
      fragments.push(tail);
    }
  }
  if (bufferedBytes > 0) {
    throw new SyntaxError(`${label} must end every record with LF.`);
  }
}

function decodedLine(raw: Buffer, label: string): string {
  if (
    raw.byteLength < 2
    || raw[raw.byteLength - 1] !== 0x0a
    || raw[raw.byteLength - 2] === 0x0d
  ) {
    throw new SyntaxError(`${label} must use canonical LF framing.`);
  }
  try {
    return UTF8.decode(raw.subarray(0, raw.byteLength - 1));
  } catch (error: unknown) {
    throw new SyntaxError(`${label} is not UTF-8.`, { cause: error });
  }
}

function sha256CanonicalSet(values: readonly (string | number)[]): string {
  const canonical = sortedCanonicalSet(values);
  return createHash("sha256")
    .update(canonicalJsonBytes(canonical))
    .digest("hex");
}

function sortedCanonicalSet<T extends string | number>(
  values: readonly T[],
): readonly T[] {
  return Object.freeze([...values].sort((left, right) => {
    if (typeof left === "number" && typeof right === "number") {
      return left - right;
    }
    const leftText = String(left);
    const rightText = String(right);
    return leftText < rightText ? -1 : leftText > rightText ? 1 : 0;
  }));
}

function checkedRuleContract(
  ruleIds: readonly string[],
): readonly string[] {
  if (
    ruleIds.length === 0
    || new Set(ruleIds).size !== ruleIds.length
    || ruleIds.some((ruleId) => !CAPTURABLE_RULE_ID_SET.has(ruleId))
  ) {
    throw new TypeError(
      "Expected rule IDs must be a non-empty unique subset of the "
      + "schema-9 25-rule contract.",
    );
  }
  return Object.freeze([...ruleIds]);
}

function emptyLabelCounts(
  ruleIds: readonly string[],
): Record<string, number> {
  return Object.fromEntries(
    ruleIds.map((ruleId) => [ruleId, 0]),
  );
}

function checkedSourceRuleId(
  value: string,
  label: string,
  expectedRuleIds: ReadonlySet<string>,
): string {
  if (!expectedRuleIds.has(value)) {
    throw new TypeError(`${label} is outside the exact 25-rule contract.`);
  }
  return value;
}

function sourceGame(
  trace: PlayerPrivateSimulationTraceRecord,
  expectedRuleIds: ReadonlySet<string>,
): SourceGame {
  if (trace.schemaVersion !== 2 || trace.ruleset.version !== 2) {
    throw new TypeError(
      "Schema-9 corpus source traces must use player-private schema 2.",
    );
  }
  if (
    trace.agents.white.searchPolicy.policyId
      !== trace.agents.black.searchPolicy.policyId
    || trace.hypothesisPolicy.kind !== "unrestricted-baseline"
  ) {
    throw new TypeError(
      "Schema-9 source traces must use the frozen standard profile.",
    );
  }
  return Object.freeze({
    gameId: trace.gameId,
    seed: trace.seed,
    gameIndex: trace.gameIndex,
    parameterSeeds: Object.freeze({ ...trace.parameterSeeds }),
    plies: trace.plies.length,
    whiteRuleId: checkedSourceRuleId(
      trace.secrets.initial.white.drawbackId,
      `source game ${trace.gameId} White drawback`,
      expectedRuleIds,
    ),
    blackRuleId: checkedSourceRuleId(
      trace.secrets.initial.black.drawbackId,
      `source game ${trace.gameId} Black drawback`,
      expectedRuleIds,
    ),
    initialFen: trace.initialPosition.fen,
    policyId: trace.agents.white.searchPolicy.policyId,
    initialReplaySha256: createHash("sha256")
      .update(canonicalJsonBytes({
        initialPosition: trace.initialPosition,
        initialSecrets: trace.secrets.initial,
      }))
      .digest("hex"),
  });
}

function assertExpectedSchedule(
  split: Schema9LedgerSplit,
  sourceGames: readonly SourceGame[],
  seedRoots: Schema9SeedRoots,
  scheduler: Schema9AssignmentScheduler,
): void {
  const expected = scheduler.assignments(
    split,
    sourceGames.length,
    seedRoots,
  )[Symbol.iterator]();
  for (const [index, source] of sourceGames.entries()) {
    const nextAssignment = expected.next();
    const assignment = nextAssignment.done ? undefined : nextAssignment.value;
    if (
      assignment === undefined
      || source.gameIndex !== index
      || assignment.gameIndex !== index
      || source.gameId !== assignment.gameId
      || source.seed !== assignment.seed
      || source.parameterSeeds.white !== assignment.parameterSeeds.white
      || source.parameterSeeds.black !== assignment.parameterSeeds.black
      || source.whiteRuleId !== assignment.whiteRuleId
      || source.blackRuleId !== assignment.blackRuleId
      || source.policyId !== SCHEMA9_SCHEDULE_PROFILE.policyId
      || source.initialFen !== (assignment.initialFen ?? DEFAULT_INITIAL_FEN)
      || source.initialReplaySha256
        !== checkedSha256(
          assignment.initialReplaySha256,
          `${split} expected initial replay SHA-256`,
        )
    ) {
      throw new TypeError(
        `${split} source game ${String(index)} differs from the frozen schedule.`,
      );
    }
  }
  if (!expected.next().done) {
    throw new TypeError(`${split} scheduler returned the wrong game count.`);
  }
}

export function assertExactSchema9LabelBalance(
  games: number,
  whiteCounts: Readonly<Record<string, number>>,
  blackCounts: Readonly<Record<string, number>>,
  ruleIds: readonly string[] = CAPTURABLE_HYPOTHESIS_RULE_IDS,
): void {
  const checkedRuleIds = checkedRuleContract(ruleIds);
  const expectedKeys = [...checkedRuleIds].sort();
  const whiteKeys = Object.keys(whiteCounts).sort();
  const blackKeys = Object.keys(blackCounts).sort();
  if (
    whiteKeys.length !== expectedKeys.length
    || blackKeys.length !== expectedKeys.length
    || whiteKeys.some((key, index) => key !== expectedKeys[index])
    || blackKeys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new TypeError(
      "Label balance must contain exactly the declared rule IDs.",
    );
  }
  if (games % checkedRuleIds.length !== 0) {
    throw new TypeError(
      "Source game count cannot be exactly balanced across 25 labels.",
    );
  }
  const expected = games / checkedRuleIds.length;
  for (const ruleId of checkedRuleIds) {
    if (
      whiteCounts[ruleId] !== expected
      || blackCounts[ruleId] !== expected
    ) {
      throw new TypeError(
        "Source trace is not exactly label-balanced for both colors.",
      );
    }
  }
}

function exactDatasetRowBytes(row: unknown): Buffer {
  return Buffer.from(`${JSON.stringify(row)}\n`, "utf8");
}

export function assertScheduledConversionAccounting(
  sourceGames: number,
  zeroPlyGames: number,
  convertedGames: number,
): void {
  if (
    !Number.isSafeInteger(sourceGames)
    || !Number.isSafeInteger(zeroPlyGames)
    || !Number.isSafeInteger(convertedGames)
    || sourceGames <= 0
    || zeroPlyGames < 0
    || convertedGames < 0
    || zeroPlyGames > sourceGames
    || convertedGames !== sourceGames - zeroPlyGames
  ) {
    throw new Error(
      "Converted game accounting lost scheduled zero-ply games.",
    );
  }
}

/**
 * Authenticate one explicit trace/dataset pair by replaying the pinned
 * converter and comparing every emitted row byte-for-byte.
 */
export async function authenticateSchema9Split(
  split: Schema9LedgerSplit,
  files: Schema9SplitFiles,
  scheduler: Schema9AssignmentScheduler,
): Promise<AuthenticatedSchema9Split> {
  return authenticateSchema9SplitWithRuleContract(
    split,
    files,
    CAPTURABLE_HYPOTHESIS_RULE_IDS,
    scheduler,
  );
}

/**
 * Internal contract seam used by focused tests and future explicitly-versioned
 * subset protocols. The public corpus ledger always calls the exact 25-rule
 * wrapper above.
 */
export async function authenticateSchema9SplitWithRuleContract(
  split: Schema9LedgerSplit,
  files: Schema9SplitFiles,
  ruleIds: readonly string[],
  scheduler: Schema9AssignmentScheduler,
): Promise<AuthenticatedSchema9Split> {
  const checkedRuleIds = checkedRuleContract(ruleIds);
  const expectedRuleIds = new Set(checkedRuleIds);
  const scheduleId = checkedScheduleId(
    files.scheduleId,
    `${split} scheduleId`,
  );
  const seedRoots = checkedSchema9SeedRoots(files.seedRoots, split);
  const producerEngineCommit = checkedGitCommit(
    files.producerEngineCommit,
    `${split} producerEngineCommit`,
  );
  const traceFile = await openStableFile(files.tracePath, `${split} trace`);
  const convertedFile = await openStableFile(
    files.convertedPath,
    `${split} converted dataset`,
  );
  const launch = await authenticateReceipt(
    files.launchReceiptPath,
    `${split} launch receipt`,
  );
  const completion = await authenticateReceipt(
    files.completionReceiptPath,
    `${split} completion receipt`,
  );
  const launchCounts = validateLaunchReceipt(
    launch,
    split,
    scheduleId,
    seedRoots,
    producerEngineCommit,
  );

  const traceHash = createHash("sha256");
  const datasetHash = createHash("sha256");
  const traceLines = lfLines(
    traceFile,
    `${split} trace`,
    MAX_TRACE_LINE_BYTES,
  );
  const datasetIterator = lfLines(
    convertedFile,
    `${split} converted dataset`,
    MAX_DATASET_LINE_BYTES,
  )[Symbol.asyncIterator]();
  const sourceGames: SourceGame[] = [];
  const gameIds = new Set<string>();
  const simulationSeeds = new Set<number>();
  const parameterSeeds = new Set<number>();
  const allSeeds = new Set<number>();
  const gameIndexes = new Set<number>();
  const convertedGameIds = new Set<string>();
  const convertedSeeds = new Set<number>();
  const whiteCounts = emptyLabelCounts(checkedRuleIds);
  const blackCounts = emptyLabelCounts(checkedRuleIds);
  let traceBytes = 0;
  let datasetBytes = 0;
  let rows = 0;
  let zeroPlyGames = 0;

  for await (const rawTrace of traceLines) {
    traceHash.update(rawTrace);
    traceBytes += rawTrace.byteLength;
    const lineNumber = sourceGames.length + 1;
    const trace = parsePlayerPrivateSimulationTraceLine(
      decodedLine(rawTrace, `${split} trace line ${String(lineNumber)}`),
    );
    const encodedTrace = Buffer.from(`${JSON.stringify(trace)}\n`, "utf8");
    if (!rawTrace.equals(encodedTrace)) {
      throw new TypeError(
        `${split} trace line ${String(lineNumber)} is not canonical Engine output.`,
      );
    }
    const source = sourceGame(trace, expectedRuleIds);
    if (gameIds.has(source.gameId)) {
      throw new TypeError(`${split} source contains a duplicate game ID.`);
    }
    if (simulationSeeds.has(source.seed)) {
      throw new TypeError(
        `${split} source contains a duplicate simulation seed.`,
      );
    }
    if (allSeeds.has(source.seed)) {
      throw new TypeError(`${split} reuses a seed across streams.`);
    }
    allSeeds.add(source.seed);
    for (const parameterSeed of [
      source.parameterSeeds.white,
      source.parameterSeeds.black,
    ]) {
      if (allSeeds.has(parameterSeed)) {
        throw new TypeError(`${split} reuses a seed across streams.`);
      }
      allSeeds.add(parameterSeed);
      parameterSeeds.add(parameterSeed);
    }
    if (gameIndexes.has(source.gameIndex)) {
      throw new TypeError(`${split} source contains a duplicate game index.`);
    }
    gameIds.add(source.gameId);
    simulationSeeds.add(source.seed);
    gameIndexes.add(source.gameIndex);
    sourceGames.push(source);
    whiteCounts[source.whiteRuleId] =
      (whiteCounts[source.whiteRuleId] ?? 0) + 1;
    blackCounts[source.blackRuleId] =
      (blackCounts[source.blackRuleId] ?? 0) + 1;

    const convertedRows =
      convertParsedPlayerPrivateTraceToDatasetRows(trace);
    if (convertedRows.length !== source.plies) {
      throw new Error(
        `${split} converter row count disagrees with source plies.`,
      );
    }
    if (source.plies === 0) {
      zeroPlyGames += 1;
    } else {
      convertedGameIds.add(source.gameId);
      convertedSeeds.add(source.seed);
    }
    for (const row of convertedRows) {
      const next = await datasetIterator.next();
      if (next.done) {
        throw new TypeError(
          `${split} converted dataset ended before pinned conversion output.`,
        );
      }
      const rawDataset = next.value;
      const expected = exactDatasetRowBytes(row);
      if (!rawDataset.equals(expected)) {
        throw new TypeError(
          `${split} converted dataset differs from pinned conversion at row `
          + `${String(rows + 1)}.`,
        );
      }
      datasetHash.update(rawDataset);
      datasetBytes += rawDataset.byteLength;
      rows += 1;
    }
  }
  const extra = await datasetIterator.next();
  if (extra.done === false) {
    throw new TypeError(
      `${split} converted dataset has rows outside the source trace.`,
    );
  }
  await assertUnchanged(traceFile, `${split} trace`);
  await assertUnchanged(convertedFile, `${split} converted dataset`);

  if (sourceGames.length === 0) {
    throw new TypeError(`${split} source trace must contain games.`);
  }
  if (rows === 0) {
    throw new TypeError(
      `${split} converted dataset must contain at least one observed move.`,
    );
  }
  if (launchCounts["train"] !== sourceGames.length) {
    throw new TypeError(`${split} launch receipt game count is inconsistent.`);
  }
  assertExpectedSchedule(split, sourceGames, seedRoots, scheduler);
  assertExactSchema9LabelBalance(
    sourceGames.length,
    whiteCounts,
    blackCounts,
    checkedRuleIds,
  );
  const sourceGameIds = sortedCanonicalSet([...gameIds]);
  const sourceSimulationSeeds = sortedCanonicalSet([...simulationSeeds]);
  const sourceParameterSeeds = sortedCanonicalSet([...parameterSeeds]);
  const convertedGameIdValues = sortedCanonicalSet([...convertedGameIds]);
  const convertedSeedValues = sortedCanonicalSet([...convertedSeeds]);
  const sourceTrace: Schema9SourceTraceIdentity = Object.freeze({
    sha256: traceHash.digest("hex"),
    bytes: traceBytes,
    games: sourceGames.length,
    zeroPlyGames,
    gameIds: sourceGameIds,
    simulationSeeds: sourceSimulationSeeds,
    parameterSeeds: sourceParameterSeeds,
    gameIdSetSha256: sha256CanonicalSet(sourceGameIds),
    simulationSeedSetSha256: sha256CanonicalSet(sourceSimulationSeeds),
    parameterSeedSetSha256: sha256CanonicalSet(sourceParameterSeeds),
    labelCountsByColor: Object.freeze({
      white: Object.freeze({ ...whiteCounts }),
      black: Object.freeze({ ...blackCounts }),
    }),
  });
  const converted: Schema9ConvertedIdentity = Object.freeze({
    sha256: datasetHash.digest("hex"),
    bytes: datasetBytes,
    rows,
    games: convertedGameIds.size,
    gameIds: convertedGameIdValues,
    simulationSeeds: convertedSeedValues,
    gameIdSetSha256: sha256CanonicalSet(convertedGameIdValues),
    simulationSeedSetSha256: sha256CanonicalSet(convertedSeedValues),
  });
  assertScheduledConversionAccounting(
    sourceGames.length,
    zeroPlyGames,
    converted.games,
  );
  if (convertedGameIds.size !== convertedSeeds.size) {
    throw new Error(
      `${split} converted game and seed accounting disagrees.`,
    );
  }
  validateCompletionReceipt(
    completion,
    split,
    scheduleId,
    producerEngineCommit,
    launch.identity.sha256,
    sourceTrace,
  );
  return Object.freeze({
    ledger: Object.freeze({
      split,
      scheduleId,
      seedRoots,
      producerEngineCommit,
      generatorReceipts: Object.freeze({
        launch: launch.identity,
        completion: completion.identity,
      }),
      scheduleProfile: SCHEMA9_SCHEDULE_PROFILE,
      sourceTrace,
      converted,
    }),
    gameIds,
    simulationSeeds,
    parameterSeeds,
  });
}

export function digestPartitionAssignments(
  splits: readonly AuthenticatedSchema9Split[],
  selector: "gameIds" | "simulationSeeds" | "parameterSeeds",
): string {
  const value = splits.map((split) => ({
    split: split.ledger.split,
    values: [...split[selector]].sort((left, right) => {
      if (typeof left === "number" && typeof right === "number") {
        return left - right;
      }
      const leftText = String(left);
      const rightText = String(right);
      return leftText < rightText ? -1 : leftText > rightText ? 1 : 0;
    }),
  }));
  return createHash("sha256")
    .update(canonicalJsonBytes(value))
    .digest("hex");
}
