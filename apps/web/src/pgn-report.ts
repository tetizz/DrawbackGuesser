import type { PlayerColor } from "@drawbackengine/shared";
import {
  tokenizePgn,
  type PgnAnalysisResult,
  type PgnGuess,
} from "./pgn-analysis.js";
import { classifyBrowserPredictorTrust } from "./model-trust.js";

export const PGN_REPORT_SCHEMA_VERSION = 7;
export const SYMBOLIC_PREDICTOR_ID = "symbolic-v2-standard";
export const SYMBOLIC_EVALUATOR_PREDICTOR_ID =
  "symbolic-v2-evaluator-enriched";
export const HYBRID_PREDICTOR_ID = "hybrid-v1-local";
export const HYBRID_V21_PREDICTOR_ID = "hybrid-v21-local";
export const HYBRID_V22_PREDICTOR_ID = "hybrid-v22-local";
export const HYBRID_V21_ENSEMBLE_PREDICTOR_ID =
  "hybrid-v21-ensemble-local-research";

export interface PgnReportTruth {
  readonly white: string;
  readonly black: string;
  readonly source: "user-entered";
}

export interface TruthScore {
  readonly drawbackId: string;
  readonly finalRank: number | null;
  readonly finalConfidence: number;
  readonly firstRankOnePly: number | null;
  readonly hardContradictionPly: number | null;
}

export interface PgnReportScoring {
  readonly white: TruthScore;
  readonly black: TruthScore;
}

export interface PgnAnalyticalPayload {
  readonly schemaVersion: typeof PGN_REPORT_SCHEMA_VERSION;
  readonly source: {
    readonly inputSha256: string;
    readonly headers: Readonly<Record<string, string>>;
    readonly normalizedMainline: readonly string[];
  };
  readonly replay: {
    readonly plyCount: number;
    readonly finalFen: string;
    readonly warnings: readonly string[];
  };
  readonly predictor: {
    readonly id:
      | typeof SYMBOLIC_PREDICTOR_ID
      | typeof SYMBOLIC_EVALUATOR_PREDICTOR_ID
      | typeof HYBRID_PREDICTOR_ID
      | typeof HYBRID_V21_PREDICTOR_ID
      | typeof HYBRID_V22_PREDICTOR_ID
      | typeof HYBRID_V21_ENSEMBLE_PREDICTOR_ID;
    readonly displayName:
      | "Symbolic v2 · standard-observation"
      | "Symbolic v2 · authenticated evaluator evidence"
      | "Hybrid v1 · local research artifact"
      | "Hybrid v21 · local research artifact"
      | "Hybrid v22 · local research artifact"
      | "Hybrid v21 ensemble · local research artifact";
    readonly source: "built-in-symbolic" | "manual-local-file";
    readonly trust: "built-in-code" | "unverified-local-research";
    readonly releaseApproved: false;
    readonly calibrationMetadata:
      | "none"
      | "artifact-declared-simulation-validation";
    readonly confidenceSemantics: string;
    readonly coverage: PgnAnalysisResult["coverage"];
    readonly representedDrawbackCount: number;
    readonly representedDrawbackIds: readonly string[];
    readonly catalogDrawbackCount: number;
    readonly unavailableDrawbacks: PgnAnalysisResult["unavailableDrawbacks"];
    readonly runtime: PgnAnalysisResult["predictor"];
  };
  readonly evaluatorEvidence: PgnAnalysisResult["evaluatorEvidence"];
  readonly timeline: PgnAnalysisResult["history"];
  readonly final: {
    readonly white: PgnAnalysisResult["finalWhite"];
    readonly black: PgnAnalysisResult["finalBlack"];
  };
}

export interface PgnAnalysisReport {
  readonly analyticalDigest: string;
  readonly analytical: PgnAnalyticalPayload;
  readonly truth?: PgnReportTruth;
  readonly scoring?: PgnReportScoring;
}

function sortedRecord(
  entries: Iterable<readonly [string, string]>,
): Readonly<Record<string, string>> {
  return Object.freeze(
    Object.fromEntries(
      [...entries].sort(([left], [right]) => left.localeCompare(right)),
    ),
  );
}

function parseHeaders(pgn: string): Readonly<Record<string, string>> {
  const headers: Array<readonly [string, string]> = [];
  const pattern = /^\s*\[([A-Za-z0-9_]+)\s+"((?:\\.|[^"])*)"\]\s*$/u;
  for (const line of pgn.split(/\r?\n/u)) {
    if (!line.trimStart().startsWith("[")) {
      continue;
    }
    const match = pattern.exec(line);
    if (match?.[1] !== undefined && match[2] !== undefined) {
      headers.push([
        match[1],
        match[2].replace(/\\(["\\])/gu, "$1"),
      ]);
    }
  }
  return sortedRecord(headers);
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, item]) => item !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, canonicalize(item)]),
    );
  }
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  throw new TypeError(`Unsupported canonical JSON value: ${typeof value}`);
}

export function canonicalJson(value: unknown): string {
  return JSON.stringify(canonicalize(value));
}

async function sha256(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function rankOf(guesses: readonly PgnGuess[], drawbackId: string): number | null {
  const index = guesses.findIndex((guess) => guess.id === drawbackId);
  return index === -1 ? null : index + 1;
}

function scoreColor(
  result: PgnAnalysisResult,
  color: PlayerColor,
  drawbackId: string,
): TruthScore {
  const final =
    color === "white" ? result.finalWhite : result.finalBlack;
  const finalGuess = final.find((guess) => guess.id === drawbackId);
  let firstRankOnePly: number | null = null;
  let hardContradictionPly: number | null = null;
  for (const point of result.history) {
    const guesses = color === "white" ? point.white : point.black;
    if (firstRankOnePly === null && guesses[0]?.id === drawbackId) {
      firstRankOnePly = point.ply;
    }
    const truth = guesses.find((guess) => guess.id === drawbackId);
    if (hardContradictionPly === null && truth?.eliminated === true) {
      hardContradictionPly = point.ply;
    }
  }
  return {
    drawbackId,
    finalRank: rankOf(final, drawbackId),
    finalConfidence: finalGuess?.confidence ?? 0,
    firstRankOnePly,
    hardContradictionPly,
  };
}

export function scoreReportTruth(
  result: PgnAnalysisResult,
  truth: PgnReportTruth,
): PgnReportScoring {
  return Object.freeze({
    white: scoreColor(result, "white", truth.white),
    black: scoreColor(result, "black", truth.black),
  });
}

export async function buildPgnAnalysisReport(
  pgn: string,
  result: PgnAnalysisResult,
  truth?: PgnReportTruth,
): Promise<PgnAnalysisReport> {
  const sourceHeaders = parseHeaders(pgn);
  const sourceMainline = Object.freeze([...tokenizePgn(pgn)]);
  if (
    canonicalJson(result.sourceBinding) !==
    canonicalJson({
      headers: sourceHeaders,
      normalizedMainline: sourceMainline,
    })
  ) {
    throw new Error(
      "PGN text no longer matches the analyzed result; analyze it again.",
    );
  }
  const predictorTrust = classifyBrowserPredictorTrust(
    result.predictor.mode === "symbolic-only"
      ? undefined
      : result.predictor.mode === "hybrid-v21-ensemble"
        ? "v21-hybrid-ensemble"
        : result.predictor.mode === "hybrid-v21"
          ? "v21-hybrid"
          : result.predictor.mode === "hybrid-v22"
            ? "v22-hybrid"
          : "v1",
  );
  const evaluatorEnriched =
    result.evaluatorEvidence.mode === "authenticated-sidecar";
  const representedScope = evaluatorEnriched
    ? "all 182 hypotheses using authenticated offline evaluator evidence"
    : "the 180 standard-PGN-reconstructible hypotheses";
  const analytical: PgnAnalyticalPayload = Object.freeze({
    schemaVersion: PGN_REPORT_SCHEMA_VERSION,
    source: Object.freeze({
      inputSha256: await sha256(pgn),
      headers: sourceHeaders,
      normalizedMainline: sourceMainline,
    }),
    replay: Object.freeze({
      plyCount: result.plyCount,
      finalFen: result.finalFen,
      warnings: Object.freeze([
        "Side variations and comments are not analyzed.",
        "Declared PGN result is preserved as a header but not used as evidence.",
      ]),
    }),
    predictor: Object.freeze({
      id: result.predictor.mode === "symbolic-only"
        ? evaluatorEnriched
          ? SYMBOLIC_EVALUATOR_PREDICTOR_ID
          : SYMBOLIC_PREDICTOR_ID
        : result.predictor.mode === "hybrid-v21-ensemble"
          ? HYBRID_V21_ENSEMBLE_PREDICTOR_ID
        : result.predictor.mode === "hybrid-v21"
          ? HYBRID_V21_PREDICTOR_ID
        : result.predictor.mode === "hybrid-v22"
          ? HYBRID_V22_PREDICTOR_ID
          : HYBRID_PREDICTOR_ID,
      displayName: result.predictor.mode === "symbolic-only"
        ? evaluatorEnriched
          ? "Symbolic v2 · authenticated evaluator evidence"
          : "Symbolic v2 · standard-observation"
        : result.predictor.mode === "hybrid-v21-ensemble"
          ? "Hybrid v21 ensemble · local research artifact"
        : result.predictor.mode === "hybrid-v21"
          ? "Hybrid v21 · local research artifact"
        : result.predictor.mode === "hybrid-v22"
          ? "Hybrid v22 · local research artifact"
          : "Hybrid v1 · local research artifact",
      ...predictorTrust,
      confidenceSemantics: result.predictor.mode === "symbolic-only"
        ? `Symbolic posterior mass conditioned on ${representedScope}; not a calibrated correctness probability.`
        : result.predictor.mode === "hybrid-v21-ensemble"
          ? `Local research ensemble posterior using artifact-declared simulation-validation calibration metadata with exact symbolic hard elimination, conditioned on ${representedScope}; the artifact is not release-approved and its calibration claims are not independently trusted.`
        : result.predictor.mode === "hybrid-v21"
          ? `Symbolic posterior plus a local neural residual, conditioned on ${representedScope}, with exact hard elimination; not a calibrated correctness probability.`
        : result.predictor.mode === "hybrid-v22"
          ? `Symbolic posterior plus a local neural residual using ${result.predictor.sequenceObservationMode}, conditioned on ${representedScope}, with exact hard elimination; not a calibrated correctness probability.`
          : `Symbolic posterior mass reweighted by an uncalibrated local neural artifact and conditioned on ${representedScope}; not a calibrated correctness probability.`,
      coverage: result.coverage,
      representedDrawbackCount: result.representedDrawbackCount,
      representedDrawbackIds: result.representedDrawbackIds,
      catalogDrawbackCount: result.catalogDrawbackCount,
      unavailableDrawbacks: result.unavailableDrawbacks,
      runtime: result.predictor,
    }),
    evaluatorEvidence: result.evaluatorEvidence,
    timeline: result.history,
    final: Object.freeze({
      white: result.finalWhite,
      black: result.finalBlack,
    }),
  });
  const analyticalDigest = await sha256(canonicalJson(analytical));
  return Object.freeze({
    analyticalDigest,
    analytical,
    ...(truth === undefined
      ? {}
      : {
          truth: Object.freeze({ ...truth }),
          scoring: scoreReportTruth(result, truth),
        }),
  });
}

export function serializePgnAnalysisReport(report: PgnAnalysisReport): string {
  return `${canonicalJson(report)}\n`;
}
