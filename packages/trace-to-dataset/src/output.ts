import { createHash, randomUUID } from "node:crypto";
import type { BigIntStats } from "node:fs";
import { link, lstat, open, type FileHandle } from "node:fs/promises";
import { dirname } from "node:path";
import {
  convertTraceToDatasetRows,
  type TrainingDatasetRow,
} from "./converter.js";
import {
  convertParsedPlayerPrivateTraceToDatasetRows,
} from "./player-private-converter.js";
import {
  parseTrustedSimulationTraceRecord,
  type TrustedSimulationTraceRecord,
} from "./trusted-trace.js";
import {
  cleanupSchema9TemporaryPublication,
  Schema9AtomicPublicationCleanupError,
  Schema9AtomicPublicationError,
  schema9PublicationMayBeCommitted,
  syncSchema9PublicationParentDirectory,
  type Schema9TemporaryPublicationIdentity,
} from "./schema9-atomic-publication.js";

export interface DatasetOutputPolicy {
  /**
   * When provided, every game must use this evaluator coverage mode.
   * Release corpora should require `uniform`; research corpora may use `none`.
   */
  readonly expectedEvaluatorCoverage?: "none" | "uniform";
  readonly expectedAuthorityId?:
    | "standard-chess/v1"
    | "capturable-king/v1";
}

export interface WrittenTrainingDataset {
  readonly games: number;
  readonly rows: number;
  readonly bytes: number;
  readonly sha256: string;
  readonly evaluatorCoverage: "none" | "uniform" | null;
  readonly evaluatorPolicyId: string | null;
  readonly evaluatorEngineFingerprint: string | null;
  readonly authorityId:
    | "standard-chess/v1"
    | "capturable-king/v1"
    | null;
}

interface EvaluatorIdentity {
  readonly authorityId:
    | "standard-chess/v1"
    | "capturable-king/v1";
  readonly coverage: "none" | "uniform";
  readonly policyId: string | null;
  readonly engineFingerprint: string | null;
}

function evaluatorIdentity(
  trace: TrustedSimulationTraceRecord,
): EvaluatorIdentity {
  if (trace.authorityId === "capturable-king/v1") {
    return {
      authorityId: "capturable-king/v1",
      coverage: "none",
      policyId: null,
      engineFingerprint: null,
    };
  }
  if (trace.evaluatorCoverage === "none") {
    return {
      authorityId: "standard-chess/v1",
      coverage: "none",
      policyId: null,
      engineFingerprint: null,
    };
  }
  const firstConstraint = trace.plies[0]?.publicEvaluatorConstraint;
  if (firstConstraint === undefined || firstConstraint === null) {
    throw new TypeError(
      `Uniform evaluator trace ${trace.gameId} has no evaluator constraint.`,
    );
  }
  return {
    authorityId: "standard-chess/v1",
    coverage: "uniform",
    policyId: firstConstraint.policyId,
    engineFingerprint: firstConstraint.engineFingerprint,
  };
}

function assertEvaluatorConsistency(
  established: EvaluatorIdentity | null,
  current: EvaluatorIdentity,
  trace: TrustedSimulationTraceRecord,
  policy: DatasetOutputPolicy,
): EvaluatorIdentity {
  if (
    policy.expectedAuthorityId !== undefined
    && current.authorityId !== policy.expectedAuthorityId
  ) {
    throw new TypeError(
      `Trace ${trace.gameId} authority ${current.authorityId}`
      + ` does not match required ${policy.expectedAuthorityId}.`,
    );
  }
  if (
    policy.expectedEvaluatorCoverage !== undefined
    && current.coverage !== policy.expectedEvaluatorCoverage
  ) {
    throw new TypeError(
      `Trace ${trace.gameId} evaluator coverage ${current.coverage}`
      + ` does not match required ${policy.expectedEvaluatorCoverage}.`,
    );
  }
  if (established === null) {
    return current;
  }
  if (
    established.authorityId !== current.authorityId
    || established.coverage !== current.coverage
    || established.policyId !== current.policyId
    || established.engineFingerprint !== current.engineFingerprint
  ) {
    throw new TypeError(
      `Trace ${trace.gameId} evaluator identity differs from prior games.`,
    );
  }
  return established;
}

function convertTrustedTrace(
  trace: TrustedSimulationTraceRecord,
): readonly TrainingDatasetRow[] {
  return trace.authorityId === "capturable-king/v1"
    ? convertParsedPlayerPrivateTraceToDatasetRows(trace)
    : convertTraceToDatasetRows(trace);
}

function encodeDatasetRow(row: TrainingDatasetRow): string {
  return `${JSON.stringify(row)}\n`;
}

function temporaryIdentity(
  metadata: BigIntStats,
): Schema9TemporaryPublicationIdentity {
  return Object.freeze({
    dev: metadata.dev,
    ino: metadata.ino,
    birthtimeNs: metadata.birthtimeNs,
  });
}

function isSameTemporaryObject(
  metadata: BigIntStats,
  expected: Schema9TemporaryPublicationIdentity,
): boolean {
  return metadata.isFile()
    && !metadata.isSymbolicLink()
    && metadata.dev === expected.dev
    && metadata.ino === expected.ino
    && metadata.birthtimeNs === expected.birthtimeNs;
}

/** Exposed only so the replacement-race behavior can be regression tested. */
export async function publishTrainingDatasetNoClobberForTesting(
  temporaryPath: string,
  outputPath: string,
  expected: Schema9TemporaryPublicationIdentity,
  afterQuarantine?: (quarantine: string) => Promise<void>,
  afterDirectorySync?: (directory: string) => Promise<void>,
): Promise<void> {
  await link(temporaryPath, outputPath);
  try {
    const linked = await lstat(outputPath, { bigint: true });
    if (!isSameTemporaryObject(linked, expected)) {
      throw new Error("Published dataset is not the authenticated temporary file.");
    }
    await cleanupSchema9TemporaryPublication(
      temporaryPath,
      expected,
      "Dataset publication",
      afterQuarantine,
    );
    const published = await lstat(outputPath, { bigint: true });
    if (!isSameTemporaryObject(published, expected)) {
      throw new Error("Published dataset changed after temporary cleanup.");
    }
    await syncSchema9PublicationParentDirectory(
      outputPath,
      "Dataset publication",
    );
    await afterDirectorySync?.(dirname(outputPath));
  } catch (error: unknown) {
    throw error instanceof Schema9AtomicPublicationCleanupError
      ? error
      : new Schema9AtomicPublicationError(
        "Dataset publication committed but verification or cleanup failed.",
        true,
        { cause: error },
      );
  }
}

async function writeAll(
  handle: FileHandle,
  bytes: Buffer,
): Promise<void> {
  let offset = 0;
  while (offset < bytes.byteLength) {
    const result = await handle.write(
      bytes,
      offset,
      bytes.byteLength - offset,
      null,
    );
    if (result.bytesWritten <= 0) {
      throw new Error("Private dataset temporary output stopped accepting bytes.");
    }
    offset += result.bytesWritten;
  }
}

/**
 * Converts privileged Engine traces into private Guesser training rows.
 *
 * Public features are derived before labels are attached by the converter.
 * The destination is created with private permissions and is never replaced.
 */
export async function writeTrainingDatasetNdjsonFileAtomic(
  outputPath: string,
  traces: Iterable<unknown> | AsyncIterable<unknown>,
  policy: DatasetOutputPolicy = {},
): Promise<WrittenTrainingDataset> {
  const temporaryPath =
    `${outputPath}.tmp-${String(process.pid)}-${randomUUID()}`;
  let handle: FileHandle | undefined;
  let temporaryCreated = false;
  let ownedTemporary: Schema9TemporaryPublicationIdentity | undefined;
  const hash = createHash("sha256");
  const gameIds = new Set<string>();
  let games = 0;
  let rows = 0;
  let bytes = 0;
  let identity: EvaluatorIdentity | null = null;

  try {
    handle = await open(temporaryPath, "wx", 0o600);
    temporaryCreated = true;
    ownedTemporary = temporaryIdentity(await handle.stat({ bigint: true }));
    for await (const traceInput of traces) {
      const trace = parseTrustedSimulationTraceRecord(traceInput);
      if (gameIds.has(trace.gameId)) {
        throw new TypeError(`Duplicate trace gameId ${trace.gameId}.`);
      }
      gameIds.add(trace.gameId);
      identity = assertEvaluatorConsistency(
        identity,
        evaluatorIdentity(trace),
        trace,
        policy,
      );
      const converted = convertTrustedTrace(trace);
      for (const row of converted) {
        const chunk = encodeDatasetRow(row);
        const encoded = Buffer.from(chunk, "utf8");
        hash.update(encoded);
        bytes += encoded.byteLength;
        rows += 1;
        await writeAll(handle, encoded);
      }
      games += 1;
    }
    if (games === 0) {
      throw new TypeError("At least one private Engine trace is required.");
    }
    await handle.sync();
    await handle.close();
    handle = undefined;
    await publishTrainingDatasetNoClobberForTesting(
      temporaryPath,
      outputPath,
      ownedTemporary,
    );
    return {
      games,
      rows,
      bytes,
      sha256: hash.digest("hex"),
      evaluatorCoverage: identity?.coverage ?? null,
      evaluatorPolicyId: identity?.policyId ?? null,
      evaluatorEngineFingerprint: identity?.engineFingerprint ?? null,
      authorityId: identity?.authorityId ?? null,
    };
  } catch (error: unknown) {
    const cleanupFailures: unknown[] = [];
    if (handle !== undefined) {
      try {
        await handle.close();
      } catch (cleanupError: unknown) {
        cleanupFailures.push(cleanupError);
      }
    }
    if (
      temporaryCreated
      && !schema9PublicationMayBeCommitted(error)
    ) {
      try {
        if (ownedTemporary === undefined) {
          throw new Error(
            "Private dataset temporary identity is unavailable for cleanup.",
          );
        }
        await cleanupSchema9TemporaryPublication(
          temporaryPath,
          ownedTemporary,
          "Private dataset conversion",
        );
      } catch (cleanupError: unknown) {
        cleanupFailures.push(cleanupError);
      }
    }
    if (cleanupFailures.length > 0) {
      throw new AggregateError(
        [error, ...cleanupFailures],
        "Dataset conversion failed and its private temporary file could not be removed.",
      );
    }
    throw error instanceof Error
      ? error
      : new Error("Private dataset conversion failed.", { cause: error });
  }
}
