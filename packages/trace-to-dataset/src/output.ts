import { createHash, randomUUID } from "node:crypto";
import { once } from "node:events";
import { createWriteStream } from "node:fs";
import { link, rm } from "node:fs/promises";
import { finished } from "node:stream/promises";
import {
  parsePrivateSimulationTraceRecord,
  type PrivateSimulationTraceRecord,
} from "@drawbackengine/simulation-trace";
import {
  convertTraceToDatasetRows,
  type TrainingDatasetRow,
} from "./converter.js";

export interface DatasetOutputPolicy {
  /**
   * When provided, every game must use this evaluator coverage mode.
   * Release corpora should require `uniform`; research corpora may use `none`.
   */
  readonly expectedEvaluatorCoverage?: "none" | "uniform";
}

export interface WrittenTrainingDataset {
  readonly games: number;
  readonly rows: number;
  readonly bytes: number;
  readonly sha256: string;
  readonly evaluatorCoverage: "none" | "uniform" | null;
  readonly evaluatorPolicyId: string | null;
  readonly evaluatorEngineFingerprint: string | null;
}

interface EvaluatorIdentity {
  readonly coverage: "none" | "uniform";
  readonly policyId: string | null;
  readonly engineFingerprint: string | null;
}

function evaluatorIdentity(
  trace: PrivateSimulationTraceRecord,
): EvaluatorIdentity {
  if (trace.evaluatorCoverage === "none") {
    return {
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
    coverage: "uniform",
    policyId: firstConstraint.policyId,
    engineFingerprint: firstConstraint.engineFingerprint,
  };
}

function assertEvaluatorConsistency(
  established: EvaluatorIdentity | null,
  current: EvaluatorIdentity,
  trace: PrivateSimulationTraceRecord,
  policy: DatasetOutputPolicy,
): EvaluatorIdentity {
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
    established.coverage !== current.coverage
    || established.policyId !== current.policyId
    || established.engineFingerprint !== current.engineFingerprint
  ) {
    throw new TypeError(
      `Trace ${trace.gameId} evaluator identity differs from prior games.`,
    );
  }
  return established;
}

function encodeDatasetRow(row: TrainingDatasetRow): string {
  return `${JSON.stringify(row)}\n`;
}

async function removeIfPresent(path: string): Promise<void> {
  try {
    await rm(path);
  } catch (error: unknown) {
    if (
      typeof error === "object"
      && error !== null
      && "code" in error
      && error.code === "ENOENT"
    ) {
      return;
    }
    throw error;
  }
}

async function publishNoClobber(
  temporaryPath: string,
  outputPath: string,
): Promise<void> {
  await link(temporaryPath, outputPath);
  try {
    await rm(temporaryPath);
  } catch (error: unknown) {
    try {
      await removeIfPresent(outputPath);
    } catch (cleanupError: unknown) {
      throw new AggregateError(
        [error, cleanupError],
        "Dataset cleanup failed for both temporary and published paths.",
      );
    }
    throw error;
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
  const stream = createWriteStream(temporaryPath, {
    encoding: "utf8",
    flags: "wx",
    mode: 0o600,
  });
  const hash = createHash("sha256");
  const gameIds = new Set<string>();
  let games = 0;
  let rows = 0;
  let bytes = 0;
  let identity: EvaluatorIdentity | null = null;

  try {
    for await (const traceInput of traces) {
      const trace = parsePrivateSimulationTraceRecord(traceInput);
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
      const converted = convertTraceToDatasetRows(trace);
      for (const row of converted) {
        const chunk = encodeDatasetRow(row);
        hash.update(chunk, "utf8");
        bytes += Buffer.byteLength(chunk, "utf8");
        rows += 1;
        if (!stream.write(chunk, "utf8")) {
          await once(stream, "drain");
        }
      }
      games += 1;
    }
    if (games === 0) {
      throw new TypeError("At least one private Engine trace is required.");
    }
    stream.end();
    await finished(stream);
    await publishNoClobber(temporaryPath, outputPath);
    return {
      games,
      rows,
      bytes,
      sha256: hash.digest("hex"),
      evaluatorCoverage: identity?.coverage ?? null,
      evaluatorPolicyId: identity?.policyId ?? null,
      evaluatorEngineFingerprint: identity?.engineFingerprint ?? null,
    };
  } catch (error: unknown) {
    stream.destroy();
    await finished(stream).catch(() => undefined);
    try {
      await removeIfPresent(temporaryPath);
    } catch (cleanupError: unknown) {
      throw new AggregateError(
        [error, cleanupError],
        "Dataset conversion failed and its private temporary file could not be removed.",
      );
    }
    throw error;
  }
}
