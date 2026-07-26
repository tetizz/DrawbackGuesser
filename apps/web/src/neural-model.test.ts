import { describe, expect, it } from "vitest";
import goldenFixtureText from "./fixtures/v21-golden.json?raw";
import {
  BROWSER_MODEL_FORMAT,
  BROWSER_MODEL_FORMAT_VERSION,
  NEURAL_FEATURE_DIMENSION,
  buildNeuralFeatureVector,
  parseBrowserNeuralModel,
  runBrowserNeuralModel,
} from "./neural-model.js";
import type {
  HybridObservation,
  HybridOutput,
} from "./sequence-neural-model.js";
import {
  encodeHybridSequenceTokenIndices,
  runHybridBrowserResidualModel,
} from "./sequence-neural-model.js";
import { rankPreservingFusion } from "./rank-preserving-fusion.js";

function tensor(shape: readonly number[], fill = 0) {
  return { shape, values: Array(shape.reduce((a, b) => a * b, 1)).fill(fill) };
}

function fixture() {
  const hidden = 2;
  return {
    format: BROWSER_MODEL_FORMAT,
    formatVersion: BROWSER_MODEL_FORMAT_VERSION,
    modelVariant: "v1",
    featureSchemaVersion: 1,
    sourceCheckpointSha256: "a".repeat(64),
    drawbackVocabulary: ["vegan", "checkers"],
    dimensions: {
      input: NEURAL_FEATURE_DIMENSION,
      hidden,
      drawbackClasses: 2,
    },
    tensors: {
      "encoder.0.weight": tensor([hidden, NEURAL_FEATURE_DIMENSION]),
      "encoder.0.bias": tensor([hidden]),
      "encoder.2.weight": tensor([hidden, hidden]),
      "encoder.2.bias": tensor([hidden]),
      "white_drawback.weight": tensor([2, hidden]),
      "white_drawback.bias": { shape: [2], values: [2, 0] },
      "black_drawback.weight": tensor([2, hidden]),
      "black_drawback.bias": { shape: [2], values: [0, 2] },
    },
  };
}

const observation = {
  fenBefore: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  move: "e2e4",
  moveNumber: 1,
  ply: 0,
  playerColor: "white" as const,
  historySan: [],
  ordinaryLegalMoveCount: 20,
};

interface GoldenCase {
  readonly observation: HybridObservation;
  readonly expected: HybridOutput;
}

interface GoldenFixture {
  readonly artifact: unknown;
  readonly cases: readonly GoldenCase[];
}

const hybridGolden = JSON.parse(goldenFixtureText) as GoldenFixture;

function v22Fixture(
  sequenceObservationMode:
    | "masked-current-v2"
    | "exact-current-v2",
): Record<string, unknown> {
  const artifact = structuredClone(
    hybridGolden.artifact,
  ) as Record<string, unknown>;
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
      "<move:a7a8q>",
      "<move:a7a8r>",
      "<move:e2e4>",
    ],
    max_sequence: 3,
    padding: "right",
    truncation: "keep-most-recent",
    current_move: "required-final-namespaced-uci",
  };
  return artifact;
}

describe("browser neural model", () => {
  it("matches the Python v1 public feature dimension and scalar ordering", () => {
    const vector = buildNeuralFeatureVector(observation);
    expect(vector).toHaveLength(NEURAL_FEATURE_DIMENSION);
    expect(vector.slice(0, 64).reduce((sum, value) => sum + value, 0)).toBe(8);
    expect(vector.at(-7)).toBe(1);
    expect(vector.at(-3)).toBeCloseTo(20 / 218);
    expect(vector.at(-2)).toBeCloseTo(12 / 63);
    expect(vector.at(-1)).toBeCloseTo(28 / 63);
  });

  it("validates dimensions and runs independent White and Black heads", () => {
    const model = parseBrowserNeuralModel(fixture());
    const output = runBrowserNeuralModel(model, observation);
    expect(output.white["vegan"]).toBeGreaterThan(0.85);
    expect(output.black["checkers"]).toBeGreaterThan(0.85);
    expect(
      Object.values(output.white).reduce((sum, value) => sum + value, 0),
    ).toBeCloseTo(1);
  });

  it("fails closed on unknown tensors and schema mismatches", () => {
    const withUnknown = structuredClone(fixture());
    Object.assign(withUnknown.tensors, { secret_label: tensor([1]) });
    expect(() => parseBrowserNeuralModel(withUnknown)).toThrow(
      "unexpected tensors",
    );
    expect(() =>
      parseBrowserNeuralModel({
        ...fixture(),
        featureSchemaVersion: 99,
      }),
    ).toThrow("unsupported");
    expect(() =>
      parseBrowserNeuralModel({
        ...fixture(),
        drawbackVocabulary: ["vegan", "not-a-real-drawback"],
      }),
    ).toThrow("unknown drawback ID");
    expect(() =>
      parseBrowserNeuralModel({
        ...fixture(),
        dimensions: {
          ...fixture().dimensions,
          hidden: 257,
        },
      }),
    ).toThrow("runtime limit");
  });

  it("matches deterministic PyTorch v21 inference for public observations", () => {
    const model = parseBrowserNeuralModel(hybridGolden.artifact);
    expect(model.modelVariant).toBe("v21-hybrid");
    if (model.modelVariant !== "v21-hybrid") {
      throw new Error("Golden fixture did not parse as v21-hybrid.");
    }
    expect(model.drawbackVocabulary).toEqual(
      [...model.symbolicRuleIds].reverse(),
    );
    for (const testCase of hybridGolden.cases) {
      const actual = runBrowserNeuralModel(model, testCase.observation);
      for (const color of ["white", "black"] as const) {
        let maximumDifference = 0;
        for (const id of model.drawbackVocabulary) {
          const expected = testCase.expected[color][id];
          const received = actual[color][id];
          if (expected === undefined || received === undefined) {
            throw new Error(`Golden output omitted ${color} ${id}.`);
          }
          maximumDifference = Math.max(
            maximumDifference,
            Math.abs(received - expected),
          );
          const symbolicIndex = model.symbolicRuleIds.indexOf(id);
          if (testCase.observation.symbolic[
            color === "white" ? "whiteEliminated" : "blackEliminated"
          ][symbolicIndex]) {
            expect(received).toBe(0);
          }
        }
        expect(maximumDifference).toBeLessThanOrEqual(1e-5);
        expect(
          Object.values(actual[color]).reduce((sum, value) => sum + value, 0),
        ).toBeCloseTo(1, 10);
      }
    }
  });

  it("fails closed on malformed v21 tensors, tokenizer, and evidence", () => {
    const malformedBase64 = structuredClone(
      hybridGolden.artifact,
    ) as Record<string, unknown>;
    const malformedTensors = malformedBase64["tensors"] as Record<
      string,
      Record<string, unknown>
    >;
    const firstTensor = malformedTensors["board_encoder.0.bias"];
    if (firstTensor === undefined) {
      throw new Error("Golden fixture tensor is missing.");
    }
    firstTensor["data"] = "!!!!";
    expect(() => parseBrowserNeuralModel(malformedBase64)).toThrow("base64");

    const nonFinite = structuredClone(
      hybridGolden.artifact,
    ) as Record<string, unknown>;
    const nonFiniteTensors = nonFinite["tensors"] as Record<
      string,
      Record<string, unknown>
    >;
    const nonFiniteTensor = nonFiniteTensors["board_encoder.0.bias"];
    const finiteData = nonFiniteTensor?.["data"];
    if (nonFiniteTensor === undefined || typeof finiteData !== "string") {
      throw new Error("Golden fixture tensor data is missing.");
    }
    const decoded = Uint8Array.from(
      atob(finiteData),
      (character) => character.charCodeAt(0),
    );
    new DataView(decoded.buffer).setUint32(0, 0x7fc00000, true);
    nonFiniteTensor["data"] = btoa(
      String.fromCharCode(...decoded),
    );
    expect(() => parseBrowserNeuralModel(nonFinite)).toThrow("non-finite");

    const badTokenizer = structuredClone(
      hybridGolden.artifact,
    ) as Record<string, unknown>;
    const tokenizer = badTokenizer["tokenizer"] as Record<string, unknown>;
    tokenizer["padding"] = "left";
    expect(() => parseBrowserNeuralModel(badTokenizer)).toThrow("tokenizer");

    const ambiguousV21 = structuredClone(
      hybridGolden.artifact,
    ) as Record<string, unknown>;
    ambiguousV21["sequenceObservationMode"] = "exact-current-v2";
    expect(() => parseBrowserNeuralModel(ambiguousV21)).toThrow(
      "unsupported",
    );

    const v21TensorMetadata = structuredClone(
      hybridGolden.artifact,
    ) as Record<string, unknown>;
    const v21Tensors = v21TensorMetadata["tensors"] as Record<
      string,
      Record<string, unknown>
    >;
    const v21Bias = v21Tensors["board_encoder.0.bias"];
    if (v21Bias === undefined) {
      throw new Error("Golden v21 bias tensor is missing.");
    }
    v21Bias["legacyMetadata"] = true;
    expect(
      parseBrowserNeuralModel(v21TensorMetadata).modelVariant,
    ).toBe("v21-hybrid");

    const oversizedHistory = structuredClone(
      hybridGolden.artifact,
    ) as Record<string, unknown>;
    const oversizedTokenizer = oversizedHistory["tokenizer"] as Record<
      string,
      unknown
    >;
    oversizedTokenizer["max_history"] = 601;
    expect(() => parseBrowserNeuralModel(oversizedHistory)).toThrow(
      "runtime limit",
    );

    const model = parseBrowserNeuralModel(hybridGolden.artifact);
    const firstCase = hybridGolden.cases[0];
    if (firstCase === undefined) {
      throw new Error("Golden fixture case is missing.");
    }
    const mismatchedEvidence = {
      ...firstCase.observation,
      symbolic: {
        ...firstCase.observation.symbolic,
        ruleIds: [...firstCase.observation.symbolic.ruleIds].reverse(),
      },
    };
    expect(() => runBrowserNeuralModel(model, mismatchedEvidence)).toThrow(
      "symbolic evidence",
    );
    expect(() => runBrowserNeuralModel(model, observation)).toThrow(
      "requires symbolic evidence",
    );
  });

  it("parses strict Python-shaped v22 exact and masked artifacts", () => {
    const exact = parseBrowserNeuralModel(v22Fixture("exact-current-v2"));
    const masked = parseBrowserNeuralModel(v22Fixture("masked-current-v2"));
    expect(exact).toMatchObject({
      formatVersion: 3,
      modelVariant: "v22-hybrid",
      sequenceObservationMode: "exact-current-v2",
      tokenizer: {
        kind: "public-sequence-observation-token",
        version: 2,
        maxSequence: 3,
        currentMove: "required-final-namespaced-uci",
      },
    });
    expect(masked).toMatchObject({
      formatVersion: 3,
      modelVariant: "v22-hybrid",
      sequenceObservationMode: "masked-current-v2",
    });
    if (
      exact.modelVariant !== "v22-hybrid" ||
      masked.modelVariant !== "v22-hybrid"
    ) {
      throw new Error("Expected v22 models.");
    }
    expect(
      encodeHybridSequenceTokenIndices(exact, {
        historySan: ["old", "e4", "Nf3", "Nc6"],
        move: "e2e4",
      }),
    ).toEqual([1, 1, 6]);
    expect(
      encodeHybridSequenceTokenIndices(masked, {
        historySan: ["old", "e4", "Nf3", "Nc6"],
        move: "e2e4",
      }),
    ).toEqual([1, 1, 3]);

    const finalOnlyArtifact = v22Fixture("exact-current-v2");
    const finalOnlyTokenizer = finalOnlyArtifact["tokenizer"] as Record<
      string,
      unknown
    >;
    finalOnlyTokenizer["max_sequence"] = 1;
    const finalOnly = parseBrowserNeuralModel(finalOnlyArtifact);
    if (finalOnly.modelVariant !== "v22-hybrid") {
      throw new Error("Expected final-token v22 model.");
    }
    expect(
      encodeHybridSequenceTokenIndices(finalOnly, {
        historySan: Array<string>(600).fill("e4"),
        move: "e2e4",
      }),
    ).toEqual([6]);

    const fullPublicSequenceArtifact = v22Fixture("exact-current-v2");
    const fullPublicTokenizer =
      fullPublicSequenceArtifact["tokenizer"] as Record<string, unknown>;
    fullPublicTokenizer["max_sequence"] = 601;
    const fullPublicSequence = parseBrowserNeuralModel(
      fullPublicSequenceArtifact,
    );
    expect(
      fullPublicSequence.modelVariant === "v22-hybrid"
        ? fullPublicSequence.tokenizer.maxSequence
        : null,
    ).toBe(601);
  });

  it("distinguishes promotions and maps unknown exact current moves safely", () => {
    const exact = parseBrowserNeuralModel(v22Fixture("exact-current-v2"));
    if (exact.modelVariant !== "v22-hybrid") {
      throw new Error("Expected a v22 model.");
    }
    expect(
      encodeHybridSequenceTokenIndices(exact, {
        historySan: [],
        move: "a7a8q",
      }),
    ).toEqual([4]);
    expect(
      encodeHybridSequenceTokenIndices(exact, {
        historySan: [],
        move: "a7a8r",
      }),
    ).toEqual([5]);
    expect(
      encodeHybridSequenceTokenIndices(exact, {
        historySan: ["e4"],
        move: "h2h4",
      }),
    ).toEqual([1, 2]);
  });

  it("fails closed on malformed v22 artifacts and observations", () => {
    const extraRoot = v22Fixture("exact-current-v2");
    extraRoot["hiddenLabel"] = "vegan";
    expect(() => parseBrowserNeuralModel(extraRoot)).toThrow("canonical");

    const badMode = v22Fixture("exact-current-v2");
    badMode["sequenceObservationMode"] = "exact-current";
    expect(() => parseBrowserNeuralModel(badMode)).toThrow("unsupported");

    const badReserved = v22Fixture("exact-current-v2");
    const badReservedTokenizer = badReserved["tokenizer"] as Record<
      string,
      unknown
    >;
    badReservedTokenizer["vocabulary"] = [
      "<pad>",
      "<unk>",
      "<current-move-masked>",
      "<unk-current-move>",
      "e4",
      "<move:e2e4>",
      "<move:a7a8q>",
    ];
    expect(() => parseBrowserNeuralModel(badReserved)).toThrow(
      "reserved tokens",
    );

    const oversizedSequence = v22Fixture("exact-current-v2");
    const oversizedTokenizer = oversizedSequence["tokenizer"] as Record<
      string,
      unknown
    >;
    oversizedTokenizer["max_sequence"] = 602;
    expect(() => parseBrowserNeuralModel(oversizedSequence)).toThrow(
      "runtime limit",
    );

    const extraTensorField = v22Fixture("exact-current-v2");
    const extraTensors = extraTensorField["tensors"] as Record<
      string,
      Record<string, unknown>
    >;
    const extraBias = extraTensors["board_encoder.0.bias"];
    if (extraBias === undefined) {
      throw new Error("v22 bias tensor is missing.");
    }
    extraBias["values"] = [0, 0];
    expect(() => parseBrowserNeuralModel(extraTensorField)).toThrow(
      "canonical",
    );

    const badPromotionToken = v22Fixture("exact-current-v2");
    const badPromotionTokenizer = badPromotionToken["tokenizer"] as Record<
      string,
      unknown
    >;
    badPromotionTokenizer["vocabulary"] = [
      "<pad>",
      "<unk>",
      "<unk-current-move>",
      "<current-move-masked>",
      "<move:a7a6q>",
      "<move:a7a8r>",
      "<move:e2e4>",
    ];
    expect(() => parseBrowserNeuralModel(badPromotionToken)).toThrow(
      "canonical UCI",
    );

    const exact = parseBrowserNeuralModel(v22Fixture("exact-current-v2"));
    if (exact.modelVariant !== "v22-hybrid") {
      throw new Error("Expected a v22 model.");
    }
    for (const move of ["e2e2", "e2e4Q", "a7a6q", "e2e"]) {
      expect(() =>
        encodeHybridSequenceTokenIndices(exact, {
          historySan: [],
          move,
        })
      ).toThrow("canonical UCI");
    }
    expect(() =>
      encodeHybridSequenceTokenIndices(exact, {
        historySan: ["<move:e2e4>"],
        move: "e2e4",
      })
    ).toThrow("invalid prior SAN");
    expect(() =>
      encodeHybridSequenceTokenIndices(exact, {
        historySan: ["x".repeat(33)],
        move: "e2e4",
      })
    ).toThrow("invalid prior SAN");
  });

  it("routes v22 inference without changing v21 golden inference", () => {
    const model = parseBrowserNeuralModel(v22Fixture("exact-current-v2"));
    const firstCase = hybridGolden.cases[0];
    if (model.modelVariant !== "v22-hybrid" || firstCase === undefined) {
      throw new Error("Expected one v22 inference case.");
    }
    const output = runBrowserNeuralModel(model, firstCase.observation);
    const residuals = runHybridBrowserResidualModel(
      model,
      firstCase.observation,
      buildNeuralFeatureVector(firstCase.observation),
    );
    const expectedWhite = rankPreservingFusion(
      residuals.white,
      model.drawbackVocabulary.map((id) => {
        const index = model.symbolicRuleIds.indexOf(id);
        return firstCase.observation.symbolic.whiteProbabilities[index] ?? 0;
      }),
      model.drawbackVocabulary.map((id) => {
        const index = model.symbolicRuleIds.indexOf(id);
        return firstCase.observation.symbolic.whiteEliminated[index] ?? true;
      }),
      1,
    ).probabilities;
    for (const [index, id] of model.drawbackVocabulary.entries()) {
      expect(output.white[id]).toBeCloseTo(expectedWhite[index] ?? 0, 12);
    }
    expect(
      Object.values(output.white).reduce((sum, value) => sum + value, 0),
    ).toBeCloseTo(1, 10);
    expect(() => runBrowserNeuralModel(model, observation)).toThrow(
      "v22-hybrid inference requires symbolic evidence",
    );
  });
});
