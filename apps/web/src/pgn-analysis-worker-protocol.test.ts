import { describe, expect, it } from "vitest";
import {
  isPgnAnalysisWorkerRequest,
  isPgnAnalysisWorkerResponse,
} from "./pgn-analysis-worker-protocol.js";
import { analyzePgn } from "./pgn-analysis.js";
import { DEFAULT_HYPOTHESIS_RULE_IDS } from "@drawbackguesser/predictor";
import { replayCompletedPgn } from "@drawbackengine/chess-core";
import {
  createEvaluatorTurnConstraintRequest,
} from "@drawbackengine/drawback-engine";
import {
  buildCompletedPgnEvaluatorSidecar,
  loadAuthenticatedCompletedPgnEvaluatorSidecar,
  serializeCompletedPgnEvaluatorSidecar,
  type AuthenticatedCompletedPgnEvaluatorSidecar,
  type CompletedPgnEvaluatorPolicy,
} from "@drawbackengine/chess-evaluator/completed-pgn-sidecar";
import { createConstraintCacheRecord } from "@drawbackengine/chess-evaluator";
import { MAX_EVALUATOR_SIDECAR_BYTES } from "./pgn-analysis-worker-protocol.js";

const COMPLETED_PGN = '[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1';
const POLICY: CompletedPgnEvaluatorPolicy = {
  provider: "uci-best-move",
  id: "stockfish-bestmove-v1",
  version: 1,
  engine: {
    uciName: "Stockfish 18",
    engine: "stockfish",
    version: "18",
    executableSha256: "a".repeat(64),
    optionsDigest: "b".repeat(64),
    publicFingerprint:
      `stockfish:18:${"a".repeat(64)}:${"b".repeat(64)}`,
  },
  searchLimit: { kind: "nodes", value: 10_000 },
};

async function authenticatedEvidence():
Promise<AuthenticatedCompletedPgnEvaluatorSidecar> {
  const replay = replayCompletedPgn(COMPLETED_PGN);
  const records = await Promise.all(replay.steps.map((step) => {
    const request = createEvaluatorTurnConstraintRequest(
      {
        fen: step.fenBefore,
        turn: step.color,
        ply: step.ply - 1,
        history: step.historyBefore,
      },
      step.ordinaryLegalMoves,
    );
    return createConstraintCacheRecord(
      {
        policy: { id: POLICY.id, version: POLICY.version },
        fingerprint: {
          engine: POLICY.engine.engine,
          version: POLICY.engine.version,
          optionsDigest: POLICY.engine.optionsDigest,
        },
        fen: request.fen,
        rootMoves: request.ordinaryRootMoves,
        limit: { nodes: 10_000 },
      },
      request.ordinaryRootMoves[0] ?? null,
    );
  }));
  const built = await buildCompletedPgnEvaluatorSidecar({
    pgn: COMPLETED_PGN,
    policy: POLICY,
    records,
  });
  return loadAuthenticatedCompletedPgnEvaluatorSidecar(
    new TextEncoder().encode(
      serializeCompletedPgnEvaluatorSidecar(built.sidecar),
    ),
    COMPLETED_PGN,
    built.sha256,
  );
}

describe("PGN analysis worker protocol", () => {
  it("requires a bounded byte sidecar and lowercase digest as a pair", () => {
    const base = { type: "analyze", requestId: 1, pgn: COMPLETED_PGN };
    const bytes = new Uint8Array([1, 2, 3]);
    const digest = "a".repeat(64);

    expect(isPgnAnalysisWorkerRequest({
      ...base,
      evaluatorSidecarBytes: bytes,
      evaluatorSidecarSha256: digest,
    })).toBe(true);
    expect(isPgnAnalysisWorkerRequest({
      ...base,
      evaluatorSidecarBytes: bytes,
    })).toBe(false);
    expect(isPgnAnalysisWorkerRequest({
      ...base,
      evaluatorSidecarSha256: digest,
    })).toBe(false);
    expect(isPgnAnalysisWorkerRequest({
      ...base,
      evaluatorSidecarBytes: [1, 2, 3],
      evaluatorSidecarSha256: digest,
    })).toBe(false);
    expect(isPgnAnalysisWorkerRequest({
      ...base,
      evaluatorSidecarBytes:
        new Uint8Array(MAX_EVALUATOR_SIDECAR_BYTES + 1),
      evaluatorSidecarSha256: digest,
    })).toBe(false);
    expect(isPgnAnalysisWorkerRequest({
      ...base,
      evaluatorSidecarBytes: bytes,
      evaluatorSidecarSha256: "A".repeat(64),
    })).toBe(false);
  });

  it("accepts authenticated 182-rule results and rejects tampered evidence", async () => {
    const evidence = await authenticatedEvidence();
    const enriched = analyzePgn(COMPLETED_PGN, {
      evaluatorEvidence: evidence,
    });
    const standard = analyzePgn(COMPLETED_PGN);

    expect(enriched.representedDrawbackCount).toBe(182);
    if (enriched.evaluatorEvidence.mode !== "authenticated-sidecar") {
      throw new Error("Expected authenticated evaluator evidence.");
    }
    expect(isPgnAnalysisWorkerResponse({
      type: "result",
      requestId: 1,
      result: enriched,
    })).toBe(true);
    expect(isPgnAnalysisWorkerResponse({
      type: "result",
      requestId: 1,
      result: {
        ...enriched,
        evaluatorEvidence: {
          ...enriched.evaluatorEvidence,
          artifactSha256: "A".repeat(64),
        },
      },
    })).toBe(false);
    expect(isPgnAnalysisWorkerResponse({
      type: "result",
      requestId: 1,
      result: {
        ...enriched,
        evaluatorEvidence: {
          ...enriched.evaluatorEvidence,
          engine: {
            ...enriched.evaluatorEvidence.engine,
            publicFingerprint: "tampered",
          },
        },
      },
    })).toBe(false);

    expect(standard.representedDrawbackCount).toBe(180);
    expect(standard.evaluatorEvidence).toEqual({ mode: "standard-pgn" });
    expect(isPgnAnalysisWorkerResponse({
      type: "result",
      requestId: 2,
      result: standard,
    })).toBe(true);
  });

  it("accepts legal SetUp PGN fullmove numbers above the feature cap", () => {
    const result = analyzePgn(
      '[SetUp "1"]\n[FEN "8/8/8/8/8/8/6k1/K6R b - - 0 301"]\n' +
      '[Result "1/2-1/2"]\n\n301... Kf3 302. Kb1 1/2-1/2',
    );

    expect(result.history.map(({ moveNumber }) => moveNumber)).toEqual([
      301,
      302,
    ]);
    expect(isPgnAnalysisWorkerResponse({
      type: "result",
      requestId: 301,
      result,
    })).toBe(true);
  });

  it("accepts only bounded protocol shapes", () => {
    expect(
      isPgnAnalysisWorkerRequest({
        type: "load-model",
        requestId: 1,
        artifactText: "{}",
        expectedSha256: "a".repeat(64),
      }),
    ).toBe(true);
    expect(
      isPgnAnalysisWorkerRequest({
        type: "analyze",
        requestId: true,
        pgn: "1. e4",
      }),
    ).toBe(false);
    expect(
      isPgnAnalysisWorkerRequest({
        type: "analyze",
        requestId: 1,
        pgn: "1. e4",
        neuralArtifactSha256: "not-a-digest",
      }),
    ).toBe(false);
    expect(
      isPgnAnalysisWorkerRequest({
        type: "analyze",
        requestId: 1,
        pgn: "1. e4",
        neuralModel: {},
        neuralArtifactSha256: "a".repeat(64),
      }),
    ).toBe(false);
    expect(
      isPgnAnalysisWorkerResponse({
        type: "model-loaded",
        requestId: 1,
        model: {
          artifactSha256: "a".repeat(64),
          modelFormatVersion: 4,
          modelVariant: "v21-hybrid-ensemble",
          drawbackCount: 182,
        },
      }),
    ).toBe(true);
    expect(
      isPgnAnalysisWorkerResponse({
        type: "model-loaded",
        requestId: 2,
        model: {
          artifactSha256: "b".repeat(64),
          modelFormatVersion: 3,
          modelVariant: "v22-hybrid",
          drawbackCount: 182,
        },
      }),
    ).toBe(true);
    expect(
      isPgnAnalysisWorkerResponse({
        type: "model-loaded",
        requestId: 3,
        model: {
          artifactSha256: "b".repeat(64),
          modelFormatVersion: 3,
          modelVariant: "v21-hybrid",
          drawbackCount: 182,
        },
      }),
    ).toBe(false);
    expect(
      isPgnAnalysisWorkerResponse({
        type: "progress",
        requestId: 1,
        progress: { processedPlies: 2, totalPlies: 1 },
      }),
    ).toBe(false);
    expect(
      isPgnAnalysisWorkerResponse({
        type: "error",
        requestId: 1,
        error: {
          name: "PgnParseError",
          message: "bad move",
          ply: 1,
          token: "e9",
        },
      }),
    ).toBe(true);
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 1,
        result: {},
      }),
    ).toBe(false);
    const incompleteResult = structuredClone(
      analyzePgn('[Result "1-0"]\n\n1. e4 e5 1-0'),
    );
    const firstPoint = incompleteResult.history[0];
    if (firstPoint === undefined) {
      throw new Error("Expected the fixture to contain a prediction point.");
    }
    Reflect.deleteProperty(firstPoint, "color");
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 1,
        result: incompleteResult,
      }),
    ).toBe(false);

    const symbolicResult = analyzePgn('[Result "1-0"]\n\n1. e4 1-0');
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 2,
        result: symbolicResult,
      }),
    ).toBe(true);
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 2,
        result: {
          ...symbolicResult,
          representedDrawbackCount: 182,
        },
      }),
    ).toBe(false);
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 2,
        result: {
          ...symbolicResult,
          finalWhite: symbolicResult.finalWhite.slice(1),
        },
      }),
    ).toBe(false);

    const hybridResult = {
      ...analyzePgn('[Result "1-0"]\n\n1. e4 1-0'),
      predictor: {
        mode: "hybrid-v21",
        modelFormatVersion: 2,
        artifactSha256: "a".repeat(64),
        sourceCheckpointSha256: "b".repeat(64),
        featureSchemaVersion: 1,
        symbolicFeatureVersion: 6,
        neuralDrawbackVocabulary: DEFAULT_HYPOTHESIS_RULE_IDS,
        neuralCoveredDrawbackCount: 182,
        unresolvedExternalConstraintIds: [
          "hand-and-gigabrain",
          "ichtyophobe",
        ],
      },
    };
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 3,
        result: hybridResult,
      }),
    ).toBe(true);
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 3,
        result: {
          ...hybridResult,
          predictor: {
            ...hybridResult.predictor,
            symbolicFeatureVersion: 5,
          },
        },
      }),
    ).toBe(false);

    const v22Result = {
      ...analyzePgn('[Result "1-0"]\n\n1. e4 1-0'),
      predictor: {
        mode: "hybrid-v22",
        modelFormatVersion: 3,
        artifactSha256: "c".repeat(64),
        sourceCheckpointSha256: "d".repeat(64),
        featureSchemaVersion: 1,
        symbolicFeatureVersion: 6,
        sequenceObservationMode: "exact-current-v2",
        neuralDrawbackVocabulary: DEFAULT_HYPOTHESIS_RULE_IDS,
        neuralCoveredDrawbackCount: 182,
        unresolvedExternalConstraintIds: [
          "hand-and-gigabrain",
          "ichtyophobe",
        ],
      },
    };
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 4,
        result: v22Result,
      }),
    ).toBe(true);
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 4,
        result: {
          ...v22Result,
          predictor: {
            ...v22Result.predictor,
            hiddenLabel: "vegan",
          },
        },
      }),
    ).toBe(false);
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 4,
        result: {
          ...v22Result,
          predictor: {
            ...v22Result.predictor,
            sequenceObservationMode: "unmasked",
          },
        },
      }),
    ).toBe(false);

    const ensembleResult = {
      ...analyzePgn('[Result "1-0"]\n\n1. e4 1-0'),
      predictor: {
        mode: "hybrid-v21-ensemble",
        modelFormatVersion: 4,
        artifactSha256: "1".repeat(64),
        sourceEnsembleReleaseSha256: "2".repeat(64),
        sourceFusionSelectionSha256: "4".repeat(64),
        sourceCalibrationSha256: "3".repeat(64),
        featureSchemaVersion: 1,
        symbolicFeatureVersion: 6,
        fusionMethod:
          "rank-preserving-bounded-residual-plus-symbolic-prior-v1",
        selectedAlpha: 0.5,
        neuralDrawbackVocabulary: DEFAULT_HYPOTHESIS_RULE_IDS,
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
            trainingRunId: ["a", "b", "c"][index]?.repeat(64),
            selectedEpoch: index + 1,
          }),
        ),
        calibration: {
          preservesHardEliminations: true,
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
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 4,
        result: ensembleResult,
      }),
    ).toBe(true);
    const reordered = structuredClone(ensembleResult);
    const firstMember = reordered.predictor.members[0];
    const secondMember = reordered.predictor.members[1];
    if (firstMember === undefined || secondMember === undefined) {
      throw new Error("Expected three ensemble members.");
    }
    reordered.predictor.members[0] = secondMember;
    reordered.predictor.members[1] = firstMember;
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 4,
        result: reordered,
      }),
    ).toBe(false);

    const duplicateProvenance = structuredClone(ensembleResult);
    const duplicateFirst = duplicateProvenance.predictor.members[0];
    const duplicateSecond = duplicateProvenance.predictor.members[1];
    if (duplicateFirst === undefined || duplicateSecond === undefined) {
      throw new Error("Expected three ensemble members.");
    }
    duplicateSecond.sourceCheckpointSha256 =
      duplicateFirst.sourceCheckpointSha256;
    expect(
      isPgnAnalysisWorkerResponse({
        type: "result",
        requestId: 4,
        result: duplicateProvenance,
      }),
    ).toBe(false);
  });
});
