import { describe, expect, it } from "vitest";
import { analyzePgn } from "./pgn-analysis.js";
import {
  buildPgnAnalysisReport,
  canonicalJson,
  HYBRID_V21_ENSEMBLE_PREDICTOR_ID,
  HYBRID_V22_PREDICTOR_ID,
  HYBRID_PREDICTOR_ID,
  PGN_REPORT_SCHEMA_VERSION,
  serializePgnAnalysisReport,
  SYMBOLIC_PREDICTOR_ID,
} from "./pgn-report.js";

const PGN = `[Event "Report fixture"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0`;

describe("canonical PGN analysis reports", () => {
  it("is deterministic and identifies the actual symbolic provider", async () => {
    const result = analyzePgn(PGN);
    const first = await buildPgnAnalysisReport(PGN, result);
    const second = await buildPgnAnalysisReport(PGN, result);

    expect(serializePgnAnalysisReport(first)).toBe(
      serializePgnAnalysisReport(second),
    );
    expect(first.analyticalDigest).toMatch(/^[0-9a-f]{64}$/u);
    expect(first.analytical.predictor).toMatchObject({
      id: SYMBOLIC_PREDICTOR_ID,
      displayName: "Symbolic v2 · standard-observation",
      source: "built-in-symbolic",
      trust: "built-in-code",
      releaseApproved: false,
      calibrationMetadata: "none",
    });
    expect(first.analytical.evaluatorEvidence).toEqual({
      mode: "standard-pgn",
    });
    expect(first.analytical.schemaVersion).toBe(PGN_REPORT_SCHEMA_VERSION);
    expect(SYMBOLIC_PREDICTOR_ID).toBe("symbolic-v2-standard");
    expect(first.analytical.predictor.representedDrawbackIds).toHaveLength(180);
    expect(first.analytical.predictor.representedDrawbackIds).not.toContain(
      "hand-and-gigabrain",
    );
    expect(first.analytical.timeline).toHaveLength(6);
    expect(first.analytical.final.white).toHaveLength(180);
    expect(first.analytical.predictor).toMatchObject({
      representedDrawbackCount: 180,
      catalogDrawbackCount: 182,
      unavailableDrawbacks: [
        {
          id: "hand-and-gigabrain",
          rank: null,
          eliminated: false,
        },
        {
          id: "ichtyophobe",
          rank: null,
          eliminated: false,
        },
      ],
    });
    expect(first.analytical.source.headers).toEqual({
      Event: "Report fixture",
      Result: "1-0",
    });
  });

  it("keeps post-game truth outside predictions and the analytical digest", async () => {
    const result = analyzePgn(PGN);
    const withoutTruth = await buildPgnAnalysisReport(PGN, result);
    const firstTruth = await buildPgnAnalysisReport(PGN, result, {
      white: "vegan",
      black: "truant",
      source: "user-entered",
    });
    const changedTruth = await buildPgnAnalysisReport(PGN, result, {
      white: "checkers",
      black: "lame-duck",
      source: "user-entered",
    });

    expect(firstTruth.analytical).toEqual(withoutTruth.analytical);
    expect(changedTruth.analytical).toEqual(withoutTruth.analytical);
    expect(firstTruth.analyticalDigest).toBe(withoutTruth.analyticalDigest);
    expect(changedTruth.analyticalDigest).toBe(withoutTruth.analyticalDigest);
    expect(firstTruth.scoring?.white.drawbackId).toBe("vegan");
    expect(changedTruth.scoring?.white.drawbackId).toBe("checkers");

    const unavailableTruth = await buildPgnAnalysisReport(PGN, result, {
      white: "hand-and-gigabrain",
      black: "ichtyophobe",
      source: "user-entered",
    });
    expect(unavailableTruth.scoring?.white).toMatchObject({
      drawbackId: "hand-and-gigabrain",
    });
    expect(unavailableTruth.scoring?.white).toMatchObject({
      finalRank: null,
      finalConfidence: 0,
      firstRankOnePly: null,
      hardContradictionPly: null,
    });
    expect(unavailableTruth.scoring?.black).toMatchObject({
      finalRank: null,
      finalConfidence: 0,
      firstRankOnePly: null,
      hardContradictionPly: null,
    });
  });

  it("records local hybrid provenance under a distinct predictor identity", async () => {
    const symbolic = analyzePgn('[Result "1-0"]\n\n1. e4 e5 1-0');
    const result = {
      ...symbolic,
      predictor: {
        mode: "hybrid-v1" as const,
        modelFormatVersion: 1 as const,
        artifactSha256: "d".repeat(64),
        sourceCheckpointSha256: "c".repeat(64),
        featureSchemaVersion: 1 as const,
        neuralDrawbackVocabulary:
          symbolic.representedDrawbackIds.slice(0, 22),
        neuralCoveredDrawbackCount: 22,
        neuralEvidenceWeight: 0.35,
      },
    };
    const report = await buildPgnAnalysisReport(
      '[Result "1-0"]\n\n1. e4 e5 1-0',
      result,
    );

    expect(report.analytical.predictor).toMatchObject({
      id: HYBRID_PREDICTOR_ID,
      displayName: "Hybrid v1 · local research artifact",
      source: "manual-local-file",
      trust: "unverified-local-research",
      releaseApproved: false,
      calibrationMetadata: "none",
      runtime: {
        artifactSha256: "d".repeat(64),
        sourceCheckpointSha256: "c".repeat(64),
        featureSchemaVersion: 1,
        neuralDrawbackVocabulary: symbolic.representedDrawbackIds.slice(0, 22),
        neuralCoveredDrawbackCount: 22,
      },
    });
  });

  it("records self-declared ensemble metadata without trusting it", async () => {
    const symbolic = analyzePgn('[Result "1-0"]\n\n1. e4 e5 1-0');
    const result = {
      ...symbolic,
      predictor: {
        mode: "hybrid-v21-ensemble" as const,
        modelFormatVersion: 4 as const,
        artifactSha256: "1".repeat(64),
        sourceEnsembleReleaseSha256: "2".repeat(64),
        sourceFusionSelectionSha256: "4".repeat(64),
        sourceCalibrationSha256: "3".repeat(64),
        featureSchemaVersion: 1 as const,
        symbolicFeatureVersion: 6 as const,
        fusionMethod:
          "rank-preserving-bounded-residual-plus-symbolic-prior-v1" as const,
        selectedAlpha: 0.5,
        neuralDrawbackVocabulary: symbolic.representedDrawbackIds,
        neuralCoveredDrawbackCount: 182,
        unresolvedExternalConstraintIds: [
          "hand-and-gigabrain",
          "ichtyophobe",
        ],
        members: [20260811, 20260812, 20260813].map(
          (trainingSeed, index) => ({
            trainingSeed,
            sourceCheckpointSha256: String(index + 4).repeat(64),
            sourceSelectionSha256: String(index + 7).repeat(64),
            trainingRunId: ["a", "b", "c"][index]?.repeat(64) ?? "",
            selectedEpoch: index + 1,
          }),
        ),
        calibration: {
          preservesHardEliminations: true as const,
          white: {
            temperature: 0.9,
            exampleCount: 120,
            nllBefore: 4.5,
            nllAfter: 4.2,
          },
          black: {
            temperature: 1.1,
            exampleCount: 118,
            nllBefore: 4.4,
            nllAfter: 4.1,
          },
        },
      },
    };
    const report = await buildPgnAnalysisReport(
      '[Result "1-0"]\n\n1. e4 e5 1-0',
      result,
    );

    expect(report.analytical.schemaVersion).toBe(7);
    expect(report.analytical.predictor).toMatchObject({
      id: HYBRID_V21_ENSEMBLE_PREDICTOR_ID,
      displayName: "Hybrid v21 ensemble · local research artifact",
      source: "manual-local-file",
      trust: "unverified-local-research",
      releaseApproved: false,
      calibrationMetadata: "artifact-declared-simulation-validation",
      runtime: {
        sourceEnsembleReleaseSha256: "2".repeat(64),
        sourceFusionSelectionSha256: "4".repeat(64),
        sourceCalibrationSha256: "3".repeat(64),
        selectedAlpha: 0.5,
        members: result.predictor.members,
        calibration: result.predictor.calibration,
      },
    });
  });

  it("records the v22 public sequence observation mode", async () => {
    const symbolic = analyzePgn('[Result "1-0"]\n\n1. e4 e5 1-0');
    const result = {
      ...symbolic,
      predictor: {
        mode: "hybrid-v22" as const,
        modelFormatVersion: 3 as const,
        artifactSha256: "e".repeat(64),
        sourceCheckpointSha256: "f".repeat(64),
        featureSchemaVersion: 1 as const,
        symbolicFeatureVersion: 6 as const,
        sequenceObservationMode: "exact-current-v2" as const,
        neuralDrawbackVocabulary: symbolic.representedDrawbackIds,
        neuralCoveredDrawbackCount: 182,
        unresolvedExternalConstraintIds: [
          "hand-and-gigabrain",
          "ichtyophobe",
        ],
      },
    };
    const report = await buildPgnAnalysisReport(
      '[Result "1-0"]\n\n1. e4 e5 1-0',
      result,
    );

    expect(report.analytical.predictor).toMatchObject({
      id: HYBRID_V22_PREDICTOR_ID,
      displayName: "Hybrid v22 · local research artifact",
      source: "manual-local-file",
      trust: "unverified-local-research",
      runtime: {
        modelFormatVersion: 3,
        sequenceObservationMode: "exact-current-v2",
      },
    });
    expect(report.analytical.predictor.confidenceSemantics).toContain(
      "exact-current-v2",
    );
  });

  it("rejects a stale result from a different PGN source", async () => {
    const stale = analyzePgn('[Result "1-0"]\n\n1. e4 e5 1-0');
    await expect(
      buildPgnAnalysisReport("1. d4 d5", stale),
    ).rejects.toThrow("no longer matches");
  });

  it("canonicalizes object keys recursively", () => {
    expect(canonicalJson({ z: 1, a: { y: 2, b: 3 } })).toBe(
      '{"a":{"b":3,"y":2},"z":1}',
    );
  });
});
