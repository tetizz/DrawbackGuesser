import {
  buildCompletedPgnEvaluatorSidecar,
  loadAuthenticatedCompletedPgnEvaluatorSidecar,
  serializeCompletedPgnEvaluatorSidecar,
  type AuthenticatedCompletedPgnEvaluatorSidecar,
  type CompletedPgnEvaluatorPolicy,
} from "@drawbackengine/chess-evaluator/completed-pgn-sidecar";
import { createConstraintCacheRecord } from "@drawbackengine/chess-evaluator";
import { replayCompletedPgn } from "@drawbackengine/chess-core";
import { createEvaluatorTurnConstraintRequest } from "@drawbackengine/drawback-engine";
import { beforeAll, describe, expect, it } from "vitest";
import hybridGoldenText from "./fixtures/v21-golden.json?raw";
import {
  MAX_PGN_INPUT_BYTES,
  MAX_PGN_PLIES,
  analyzePgn,
  buildPublicHybridObservation,
  PgnParseError,
  projectStandardPgnGuesses,
  tokenizePgn,
  type PgnGuess,
} from "./pgn-analysis.js";
import { STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS } from "./pgn-analysis-contract.js";
import {
  BROWSER_MODEL_FORMAT,
  BROWSER_MODEL_FORMAT_VERSION,
  NEURAL_FEATURE_DIMENSION,
  parseBrowserNeuralModel,
} from "./neural-model.js";
import type {
  HybridBrowserModel,
  HybridV22BrowserModel,
} from "./sequence-neural-model.js";
import {
  ENSEMBLE_CALIBRATION_METHOD,
  ENSEMBLE_FUSION_METHOD,
  ENSEMBLE_TRAINING_SEEDS,
  type EnsembleBrowserModel,
} from "./ensemble-neural-model.js";

function zeroTensor(shape: readonly number[]) {
  return {
    shape,
    values: Array(shape.reduce((left, right) => left * right, 1)).fill(0),
  };
}

function testNeuralModel() {
  const hidden = 2;
  return parseBrowserNeuralModel({
    format: BROWSER_MODEL_FORMAT,
    formatVersion: BROWSER_MODEL_FORMAT_VERSION,
    modelVariant: "v1",
    featureSchemaVersion: 1,
    sourceCheckpointSha256: "b".repeat(64),
    drawbackVocabulary: ["vegan", "checkers"],
    dimensions: {
      input: NEURAL_FEATURE_DIMENSION,
      hidden,
      drawbackClasses: 2,
    },
    tensors: {
      "encoder.0.weight": zeroTensor([hidden, NEURAL_FEATURE_DIMENSION]),
      "encoder.0.bias": zeroTensor([hidden]),
      "encoder.2.weight": zeroTensor([hidden, hidden]),
      "encoder.2.bias": zeroTensor([hidden]),
      "white_drawback.weight": zeroTensor([2, hidden]),
      "white_drawback.bias": { shape: [2], values: [-4, 4] },
      "black_drawback.weight": zeroTensor([2, hidden]),
      "black_drawback.bias": { shape: [2], values: [4, -4] },
    },
  });
}

function zeroResidualHybridModel(): HybridBrowserModel {
  const fixture = JSON.parse(hybridGoldenText) as {
    artifact: Record<string, unknown>;
  };
  const artifact = structuredClone(fixture.artifact);
  const tensors = artifact["tensors"] as Record<
    string,
    { data: string; shape: number[] }
  >;
  for (const tensor of Object.values(tensors)) {
    const byteLength = tensor.shape.reduce(
      (product, dimension) => product * dimension,
      4,
    );
    tensor.data = btoa("\0".repeat(byteLength));
  }
  const model = parseBrowserNeuralModel(artifact);
  if (model.modelVariant !== "v21-hybrid") {
    throw new Error("Hybrid test fixture has the wrong model variant.");
  }
  return model;
}

function zeroResidualV22Model(
  sequenceObservationMode:
    | "masked-current-v2"
    | "exact-current-v2",
): HybridV22BrowserModel {
  const fixture = JSON.parse(hybridGoldenText) as {
    artifact: Record<string, unknown>;
  };
  const artifact = structuredClone(fixture.artifact);
  artifact["formatVersion"] = 3;
  artifact["modelVariant"] = "v22-hybrid";
  artifact["sequenceObservationMode"] = sequenceObservationMode;
  artifact["tokenizer"] = {
    kind: "public-sequence-observation-token",
    version: 2,
    vocabulary: [
      "<pad>",
      "<unk>",
      "<unk-current-move>",
      "<current-move-masked>",
      "<move:e2e4>",
      "<move:e7e5>",
      "e4",
    ],
    max_sequence: 3,
    padding: "right",
    truncation: "keep-most-recent",
    current_move: "required-final-namespaced-uci",
  };
  const tensors = artifact["tensors"] as Record<
    string,
    { data: string; shape: number[] }
  >;
  for (const tensor of Object.values(tensors)) {
    const byteLength = tensor.shape.reduce(
      (product, dimension) => product * dimension,
      4,
    );
    tensor.data = btoa("\0".repeat(byteLength));
  }
  const model = parseBrowserNeuralModel(artifact);
  if (model.modelVariant !== "v22-hybrid") {
    throw new Error("v22 test fixture has the wrong model variant.");
  }
  return model;
}

function zeroResidualEnsembleModel(): EnsembleBrowserModel {
  const fixture = JSON.parse(hybridGoldenText) as {
    artifact: Record<string, unknown>;
  };
  const artifact = structuredClone(fixture.artifact);
  const baseTensors = artifact["tensors"] as Record<
    string,
    { data: string; shape: number[] }
  >;
  for (const tensor of Object.values(baseTensors)) {
    const byteLength = tensor.shape.reduce(
      (product, dimension) => product * dimension,
      4,
    );
    tensor.data = btoa("\0".repeat(byteLength));
  }
  const digest = (index: number): string =>
    index.toString(16).repeat(64);
  const ensembleArtifact = {
    format: artifact["format"],
    formatVersion: 4,
    modelVariant: "v21-hybrid-ensemble",
    featureSchemaVersion: artifact["featureSchemaVersion"],
    symbolicFeatureVersion: artifact["symbolicFeatureVersion"],
    drawbackVocabulary: artifact["drawbackVocabulary"],
    symbolicRuleIds: artifact["symbolicRuleIds"],
    tokenizer: artifact["tokenizer"],
    tensorEncoding: artifact["tensorEncoding"],
    dimensions: artifact["dimensions"],
    ensemble: {
      method: ENSEMBLE_FUSION_METHOD,
      memberCount: 3,
      seedOrder: [...ENSEMBLE_TRAINING_SEEDS],
      sourceEnsembleReleaseSha256: digest(10),
      sourceFusionSelectionSha256: digest(12),
      selectedAlpha: 0.5,
      members: ENSEMBLE_TRAINING_SEEDS.map((trainingSeed, index) => ({
        trainingSeed,
        selectedEpoch: index + 1,
        trainingRunId: digest(index + 1),
        sourceSelectionSha256: digest(index + 4),
        sourceCheckpointSha256: digest(index + 7),
        tensors: structuredClone(baseTensors),
      })),
    },
    calibration: {
      method: ENSEMBLE_CALIBRATION_METHOD,
      sourceCalibrationSha256: digest(11),
      preservesHardEliminations: true,
      white: {
        temperature: 1,
        exampleCount: 10,
        nllBefore: 2,
        nllAfter: 1,
      },
      black: {
        temperature: 1,
        exampleCount: 10,
        nllBefore: 2,
        nllAfter: 1,
      },
    },
  };
  const model = parseBrowserNeuralModel(ensembleArtifact);
  if (model.modelVariant !== "v21-hybrid-ensemble") {
    throw new Error("Ensemble test fixture has the wrong model variant.");
  }
  return model;
}

function completed(moves: string, result = "1-0"): string {
  return `[Result "${result}"]\n\n${moves} ${result}`;
}

const EVALUATOR_PGN = completed("1. e4 e5 2. Nf3 Nc6");
const EXECUTABLE_SHA = "ab".repeat(32);
const OPTIONS_SHA = "cd".repeat(32);
const EVALUATOR_POLICY: CompletedPgnEvaluatorPolicy = {
  provider: "uci-best-move",
  id: "stockfish-bestmove-v1",
  version: 1,
  engine: {
    uciName: "Stockfish 18",
    engine: "stockfish",
    version: "18",
    executableSha256: EXECUTABLE_SHA,
    optionsDigest: OPTIONS_SHA,
    publicFingerprint:
      `stockfish:18:${EXECUTABLE_SHA}:${OPTIONS_SHA}`,
  },
  searchLimit: { kind: "nodes", value: 10_000 },
};
let AUTHENTICATED_EVALUATOR:
  AuthenticatedCompletedPgnEvaluatorSidecar;

beforeAll(async () => {
  const replay = replayCompletedPgn(EVALUATOR_PGN);
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
    const observedUci =
      `${step.move.from}${step.move.to}${step.move.promotion?.[0] ?? ""}`;
    return createConstraintCacheRecord(
      {
        policy: {
          id: EVALUATOR_POLICY.id,
          version: EVALUATOR_POLICY.version,
        },
        fingerprint: {
          engine: EVALUATOR_POLICY.engine.engine,
          version: EVALUATOR_POLICY.engine.version,
          optionsDigest: EVALUATOR_POLICY.engine.optionsDigest,
        },
        fen: request.fen,
        rootMoves: request.ordinaryRootMoves,
        limit: { nodes: 10_000 },
      },
      observedUci,
    );
  }));
  const built = await buildCompletedPgnEvaluatorSidecar({
    pgn: EVALUATOR_PGN,
    policy: EVALUATOR_POLICY,
    records,
  });
  AUTHENTICATED_EVALUATOR =
    await loadAuthenticatedCompletedPgnEvaluatorSidecar(
      new TextEncoder().encode(
        serializeCompletedPgnEvaluatorSidecar(built.sidecar),
      ),
      EVALUATOR_PGN,
      built.sha256,
    );
});

describe("offline PGN analysis", () => {
  it("uses authenticated evaluator evidence for all 182 rules", () => {
    const result = analyzePgn(EVALUATOR_PGN, {
      evaluatorEvidence: AUTHENTICATED_EVALUATOR,
    });

    expect(result.representedDrawbackCount).toBe(182);
    expect(result.representedDrawbackIds).toHaveLength(182);
    expect(result.unavailableDrawbacks).toEqual([]);
    expect(result.evaluatorEvidence).toEqual({
      mode: "authenticated-sidecar",
      artifactSha256: AUTHENTICATED_EVALUATOR.artifactSha256,
      policy: { id: "stockfish-bestmove-v1", version: 1 },
      engine: EVALUATOR_POLICY.engine,
      searchLimit: { kind: "nodes", value: 10_000 },
    });
    expect(
      result.history[0]?.white.find(({ id }) => id === "ichtyophobe"),
    ).toMatchObject({ eliminated: true, confidence: 0 });
    expect(
      result.history[0]?.black.find(({ id }) => id === "ichtyophobe"),
    ).toMatchObject({ eliminated: false });
    expect(
      result.history[1]?.black.find(({ id }) => id === "ichtyophobe"),
    ).toMatchObject({ eliminated: true, confidence: 0 });
    expect(
      result.finalWhite.find(({ id }) => id === "hand-and-gigabrain"),
    ).toMatchObject({ eliminated: false });
    expect(
      result.finalWhite.findIndex(({ id }) => id === "hand-and-gigabrain"),
    ).toBeLessThan(
      result.finalWhite.findIndex(({ id }) => id === "ichtyophobe"),
    );
    expect(
      result.finalWhite.reduce((sum, guess) => sum + guess.confidence, 0),
    ).toBeCloseTo(1, 12);
  });

  it("rejects missing or malformed authenticated artifact digests", () => {
    const missing = {
      ...AUTHENTICATED_EVALUATOR,
    };
    Reflect.deleteProperty(missing, "artifactSha256");
    expect(() =>
      analyzePgn(EVALUATOR_PGN, {
        evaluatorEvidence: missing,
      })
    ).toThrow("authenticated sidecar SHA-256");
    expect(() =>
      analyzePgn(EVALUATOR_PGN, {
        evaluatorEvidence: {
          ...AUTHENTICATED_EVALUATOR,
          artifactSha256: "0".repeat(63),
        },
      })
    ).toThrow("authenticated sidecar SHA-256");
  });

  it("rejects partial authenticated evaluator evidence", () => {
    expect(() =>
      analyzePgn(EVALUATOR_PGN, {
        evaluatorEvidence: {
          ...AUTHENTICATED_EVALUATOR,
          constraints: AUTHENTICATED_EVALUATOR.constraints.slice(1),
        },
      })
    ).toThrow("constraint count must exactly match");
  });

  it("marks evaluator constraints resolved in enriched hybrid metadata", () => {
    const result = analyzePgn(EVALUATOR_PGN, {
      evaluatorEvidence: AUTHENTICATED_EVALUATOR,
      neuralModel: zeroResidualHybridModel(),
      neuralArtifactSha256: "ef".repeat(32),
    });

    expect(result.finalWhite).toHaveLength(182);
    expect(result.finalBlack).toHaveLength(182);
    expect(result.predictor).toMatchObject({
      mode: "hybrid-v21",
      unresolvedExternalConstraintIds: [],
    });
  });

  it("rejects ongoing games and legal prefixes without a terminal result", () => {
    expect(() =>
      analyzePgn('[Result "*"]\n\n1. e4 e5 *')
    ).toThrow("requires matching terminal PGN");
    expect(() => analyzePgn("1. e4 e5")).toThrow(
      "requires matching terminal PGN",
    );
    expect(() =>
      analyzePgn('[Result "1-0"]\n\n1. e4 e5 *')
    ).toThrow("requires matching terminal PGN");
  });

  it("replays a standard-legal game and records independent predictions per ply", () => {
    const result = analyzePgn(`
      [Event "Offline fixture"]
      [Result "1-0"]

      1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
    `);

    expect(result.plyCount).toBe(6);
    expect(result.history.map((point) => point.san)).toEqual([
      "e4",
      "e5",
      "Nf3",
      "Nc6",
      "Bb5",
      "a6",
    ]);
    expect(result.finalWhite).toHaveLength(180);
    expect(result.finalBlack).toHaveLength(180);
    expect(result).toMatchObject({
      representedDrawbackCount: 180,
      catalogDrawbackCount: 182,
      evaluatorEvidence: { mode: "standard-pgn" },
      unavailableDrawbacks: [
        {
          id: "hand-and-gigabrain",
          reason: "requires-public-evaluator-facts",
          rank: null,
          eliminated: false,
        },
        {
          id: "ichtyophobe",
          reason: "requires-public-evaluator-facts",
          rank: null,
          eliminated: false,
        },
      ],
    });
    expect(result.history[0]?.white).not.toEqual(result.history[0]?.black);
    for (const [index, point] of result.history.entries()) {
      const previous = index === 0 ? undefined : result.history[index - 1];
      for (const evidence of point.eliminations) {
        expect(evidence.color).toBe(point.color);
        expect(evidence.reason).toContain(`Observed ${point.san}`);
        const current =
          evidence.color === "white" ? point.white : point.black;
        const prior =
          evidence.color === "white" ? previous?.white : previous?.black;
        expect(
          current.find(({ id }) => id === evidence.drawbackId)?.eliminated,
        ).toBe(true);
        expect(
          prior?.find(({ id }) => id === evidence.drawbackId)?.eliminated ??
            false,
        ).toBe(false);
      }
    }
    expect(result.coverage).toHaveLength(12);
    expect(result.representedDrawbackIds).toHaveLength(180);
    expect(result.finalWhite.map(({ id }) => id).sort()).toEqual(
      [...result.representedDrawbackIds].sort(),
    );
    expect(
      result.finalWhite.some(({ id }) =>
        STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS.includes(
          id as (typeof STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS)[number],
        )
      ),
    ).toBe(false);
    expect(
      result.finalWhite.reduce((sum, guess) => sum + guess.confidence, 0),
    ).toBeCloseTo(1, 12);
    expect(
      result.finalBlack.reduce((sum, guess) => sum + guess.confidence, 0),
    ).toBeCloseTo(1, 12);
    expect(
      result.finalWhite.find(({ id }) => id === "untitled-duck-drawback")
        ?.parameters[0],
    ).toMatchObject({ name: "square" });
    expect(
      result.coverage.find(({ drawbackId }) => drawbackId === "gambler"),
    ).toMatchObject({ mode: "analytic", variantCount: 6 });
  });

  it("ignores headers, comments, annotations, NAGs, and side variations", () => {
    const tokens = tokenizePgn(`
      [Site "Local"]
      1.e4! {main line} e5 $1 (1... c5 2. Nf3)
      2.Nf3 Nc6 ; ignored to end of line
      3. Bb5 a6 1-0
    `);
    expect(tokens).toEqual(["e4!", "e5", "Nf3", "Nc6", "Bb5", "a6"]);
    expect(analyzePgn(completed(tokens.join(" "))).plyCount).toBe(6);
  });

  it("reports the exact illegal token and ply", () => {
    expect(() => analyzePgn(completed("1. e4 e5 2. Bh6"))).toThrow(PgnParseError);
    try {
      analyzePgn(completed("1. e4 e5 2. Bh6"));
    } catch (error) {
      expect(error).toMatchObject({ ply: 3, token: "Bh6" });
      expect((error as Error).message).toContain("not legal");
    }
  });

  it("honors a declared FEN starting position and side to move", () => {
    const result = analyzePgn(`
      [SetUp "1"]
      [FEN "7k/8/8/8/8/8/4K3/7R b - - 0 1"]
      [Result "1/2-1/2"]

      1... Kg7 2. Rh7+ 1/2-1/2
    `);

    expect(result.history.map(({ color, san }) => ({ color, san }))).toEqual([
      { color: "black", san: "Kg7" },
      { color: "white", san: "Rh7+" },
    ]);
  });

  it("rejects empty and structurally malformed PGN", () => {
    expect(() => analyzePgn("[Event \"No moves\"]")).toThrow(
      "at least one move",
    );
    expect(() => tokenizePgn("1. e4 {missing")).toThrow("Unterminated");
    expect(() =>
      analyzePgn('[SetUp "1"]\n[Result "1-0"]\n\n1. e4 1-0')
    ).toThrow("without a FEN");
    expect(() => analyzePgn('[Event "broken]\n1. e4')).toThrow(
      "Malformed PGN header",
    );
  });

  it("reports bounded per-ply progress", () => {
    const progress: Array<{ processedPlies: number; totalPlies: number }> = [];
    const result = analyzePgn(completed("1. e4 e5 2. Nf3"), {
      onProgress(update) {
        progress.push(update);
      },
    });

    expect(result.plyCount).toBe(3);
    expect(progress).toEqual([
      { processedPlies: 0, totalPlies: 3 },
      { processedPlies: 1, totalPlies: 3 },
      { processedPlies: 2, totalPlies: 3 },
      { processedPlies: 3, totalPlies: 3 },
    ]);
  });

  it("runs a local neural artifact in the worker-compatible analysis path", () => {
    const result = analyzePgn(completed("1. e4 e5"), {
      neuralModel: testNeuralModel(),
      neuralArtifactSha256: "d".repeat(64),
    });

    expect(result.predictor).toMatchObject({
      mode: "hybrid-v1",
      sourceCheckpointSha256: "b".repeat(64),
      artifactSha256: "d".repeat(64),
      neuralCoveredDrawbackCount: 2,
    });
    expect(result.finalWhite.find(({ id }) => id === "checkers")?.confidence)
      .toBeGreaterThan(
        result.finalWhite.find(({ id }) => id === "vegan")?.confidence ?? 0,
      );
    expect(result.finalBlack.find(({ id }) => id === "vegan")?.confidence)
      .toBeGreaterThan(
        result.finalBlack.find(({ id }) => id === "checkers")?.confidence ?? 0,
      );
  });

  it("encodes promotion moves with canonical UCI suffixes for neural inference", () => {
    const result = analyzePgn(`
      [SetUp "1"]
      [FEN "7k/4P3/8/8/8/8/8/K7 w - - 0 1"]
      [Result "1-0"]

      1. e8=Q+ 1-0
    `, {
      neuralModel: testNeuralModel(),
      neuralArtifactSha256: "d".repeat(64),
    });
    expect(result.plyCount).toBe(1);
    expect(result.predictor.mode).toBe("hybrid-v1");
  });

  it("runs v21 once over all public symbolic hypotheses without double fusion", () => {
    const pgn = completed("1. e4 e5 2. Nf3 Nc6");
    const symbolic = analyzePgn(pgn);
    const hybrid = analyzePgn(pgn, {
      neuralModel: zeroResidualHybridModel(),
      neuralArtifactSha256: "e".repeat(64),
    });

    expect(hybrid.predictor).toMatchObject({
      mode: "hybrid-v21",
      modelFormatVersion: 2,
      symbolicFeatureVersion: 6,
      neuralCoveredDrawbackCount: 182,
      unresolvedExternalConstraintIds: [
        "hand-and-gigabrain",
        "ichtyophobe",
      ],
    });
    expect(hybrid.finalWhite).toHaveLength(180);
    expect(hybrid.finalBlack).toHaveLength(180);
    for (const color of ["finalWhite", "finalBlack"] as const) {
      const symbolicById = new Map(
        symbolic[color].map((guess) => [guess.id, guess]),
      );
      for (const guess of hybrid[color]) {
        const baseline = symbolicById.get(guess.id);
        expect(baseline).toBeDefined();
        expect(guess.confidence).toBeCloseTo(
          baseline?.confidence ?? Number.NaN,
          12,
        );
        if (guess.eliminated) {
          expect(guess.confidence).toBe(0);
        }
      }
    }
  });

  it("routes exact-current v22 observations and reports their public mode", () => {
    const result = analyzePgn(completed("1. e4 e5 2. Nf3 Nc6"), {
      neuralModel: zeroResidualV22Model("exact-current-v2"),
      neuralArtifactSha256: "a".repeat(64),
    });

    expect(result.predictor).toMatchObject({
      mode: "hybrid-v22",
      modelFormatVersion: 3,
      symbolicFeatureVersion: 6,
      sequenceObservationMode: "exact-current-v2",
      neuralCoveredDrawbackCount: 182,
      unresolvedExternalConstraintIds: [
        "hand-and-gigabrain",
        "ichtyophobe",
      ],
    });
    expect(result.finalWhite).toHaveLength(180);
    expect(result.finalBlack).toHaveLength(180);
  });

  it("routes the masked-current v22 control without secret inputs", () => {
    const result = analyzePgn(completed("1. e4 e5"), {
      neuralModel: zeroResidualV22Model("masked-current-v2"),
      neuralArtifactSha256: "b".repeat(64),
    });

    expect(result.predictor).toMatchObject({
      mode: "hybrid-v22",
      sequenceObservationMode: "masked-current-v2",
    });
  });

  it("runs the calibrated three-member ensemble over 182 then projects 180", () => {
    const model = zeroResidualEnsembleModel();
    const symbolic = analyzePgn(completed("1. e4 e5"));
    const result = analyzePgn(completed("1. e4 e5"), {
      neuralModel: model,
      neuralArtifactSha256: "f".repeat(64),
    });

    expect(result.finalWhite).toHaveLength(180);
    expect(result.finalBlack).toHaveLength(180);
    expect(result.predictor).toMatchObject({
      mode: "hybrid-v21-ensemble",
      modelFormatVersion: 4,
      artifactSha256: "f".repeat(64),
      sourceEnsembleReleaseSha256:
        model.ensemble.sourceEnsembleReleaseSha256,
      sourceFusionSelectionSha256:
        model.ensemble.sourceFusionSelectionSha256,
      sourceCalibrationSha256:
        model.calibration.sourceCalibrationSha256,
      fusionMethod: ENSEMBLE_FUSION_METHOD,
      selectedAlpha: model.ensemble.selectedAlpha,
      neuralCoveredDrawbackCount: 182,
      members: model.ensemble.members.map((member) => ({
        trainingSeed: member.trainingSeed,
        sourceCheckpointSha256: member.sourceCheckpointSha256,
        sourceSelectionSha256: member.sourceSelectionSha256,
        trainingRunId: member.trainingRunId,
        selectedEpoch: member.selectedEpoch,
      })),
      calibration: {
        preservesHardEliminations: true,
        white: model.calibration.white,
        black: model.calibration.black,
      },
    });
    for (const color of ["finalWhite", "finalBlack"] as const) {
      const symbolicById = new Map(
        symbolic[color].map((guess) => [guess.id, guess]),
      );
      expect(
        result[color].reduce((sum, guess) => sum + guess.confidence, 0),
      ).toBeCloseTo(1, 12);
      for (const guess of result[color]) {
        expect(guess.confidence).toBeCloseTo(
          symbolicById.get(guess.id)?.confidence ?? Number.NaN,
          12,
        );
      }
    }
  });

  it("constructs v21 inputs solely from public replay observations", () => {
    const model = zeroResidualHybridModel();
    const symbolic = analyzePgn(completed("1. e4")).history[0];
    if (symbolic === undefined) {
      throw new Error("Symbolic fixture has no first ply.");
    }
    const completeWhite = [
      ...symbolic.white,
      ...STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS.map((id) => ({
        id,
        confidence: 0,
        eliminated: false,
        parameters: [],
      })),
    ];
    const completeBlack = [
      ...symbolic.black,
      ...STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS.map((id) => ({
        id,
        confidence: 0,
        eliminated: false,
        parameters: [],
      })),
    ];
    const publicObservation = buildPublicHybridObservation(
      model,
      {
        fenBefore: symbolic.fenBefore,
        move: "e2e4",
        moveNumber: 1,
        ply: 0,
        playerColor: "white",
        historySan: [],
        ordinaryLegalMoveCount: 20,
      },
      completeWhite,
      completeBlack,
    );

    expect(Object.keys(publicObservation).sort()).toEqual([
      "fenBefore",
      "historySan",
      "move",
      "moveNumber",
      "ordinaryLegalMoveCount",
      "playerColor",
      "ply",
      "symbolic",
    ]);
    expect(publicObservation.historySan).toEqual([]);
    expect(publicObservation.symbolic.ruleIds).toHaveLength(182);
    expect(JSON.stringify(publicObservation)).not.toMatch(
      /trueDrawback|hiddenParameters|internalState|gameId|seed/u,
    );
  });

  it("projects only evaluator-dependent rules and renormalizes their mass away", () => {
    const representedIds = analyzePgn(completed("1. e4")).representedDrawbackIds;
    const complete = (
      excludedMass: number,
      reverse = false,
    ): readonly PgnGuess[] => {
      const representedMass = 1 - excludedMass;
      const denominator = representedIds.length * (representedIds.length + 1) / 2;
      const represented = representedIds.map((id, index) => ({
        id,
        confidence:
          representedMass *
          ((reverse ? representedIds.length - index : index + 1) / denominator),
        eliminated: false,
        parameters: [],
      }));
      return [
        ...represented,
        {
          id: "hand-and-gigabrain",
          confidence: excludedMass * 0.99,
          eliminated: false,
          parameters: [],
        },
        {
          id: "ichtyophobe",
          confidence: excludedMass * 0.01,
          eliminated: false,
          parameters: [],
        },
      ];
    };

    const lowExcluded = projectStandardPgnGuesses(complete(0.01));
    const highExcluded = projectStandardPgnGuesses(complete(0.99));
    expect(lowExcluded.map(({ id }) => id).sort()).toEqual(
      [...representedIds].sort(),
    );
    expect(highExcluded.map(({ id }) => id)).toEqual(
      lowExcluded.map(({ id }) => id),
    );
    for (const [index, guess] of highExcluded.entries()) {
      expect(guess.confidence).toBeCloseTo(
        lowExcluded[index]?.confidence ?? Number.NaN,
        12,
      );
    }
    expect(
      highExcluded.reduce((sum, guess) => sum + guess.confidence, 0),
    ).toBeCloseTo(1, 12);

    const white = projectStandardPgnGuesses(complete(0.99));
    const black = projectStandardPgnGuesses(complete(0.99, true));
    expect(white[0]?.id).not.toBe(black[0]?.id);
  });

  it("rejects oversized input and excessive ply before replay", () => {
    expect(() => analyzePgn("x".repeat(MAX_PGN_INPUT_BYTES + 1))).toThrow(
      "byte analysis limit",
    );
    const excessive = Array.from(
      { length: MAX_PGN_PLIES + 1 },
      () => "e4",
    ).join(" ");
    expect(() => analyzePgn(completed(excessive))).toThrow("ply analysis limit");
  });
});
