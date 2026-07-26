import { describe, expect, it } from "vitest";
import {
  DatasetContractError,
  parseDatasetRow,
  parsePublicFeatureRecord,
} from "./index.js";

const schema = {
  symbolicFeatureVersion: 6,
  symbolicRuleCount: 2,
} as const;

function publicFeatures(): Record<string, unknown> {
  return {
    fenBefore: "8/8/8/8/8/8/4K3/7k w - - 0 1",
    move: "e2e3",
    moveNumber: 1,
    ply: 0,
    playerColor: "white",
    historySan: [],
    ordinaryLegalMoves: ["e2e3", "e2f2"],
    clockMs: null,
    symbolicFeatureVersion: 6,
    symbolicWhiteRuleProbabilities: [0.75, 0.25],
    symbolicBlackRuleProbabilities: [0.5, 0.5],
    symbolicWhiteEliminated: [false, false],
    symbolicBlackEliminated: [false, false],
    publicEvaluatorConstraint: null,
  };
}

function datasetRow(): Record<string, unknown> {
  return {
    ...publicFeatures(),
    gameId: "00000001-000000",
    seed: 1,
    san: "Ke3",
    botAgentId: "weak-human-like",
    botStyle: "weak",
    botStrength: 800,
    trueDrawback: "vegan",
    hiddenParameters: {},
    drawbackInternalState: { moves: 0 },
    drawbackLegalMoves: ["e2e3"],
    ruleTriggered: true,
    forced: true,
    result: { kind: "active" },
  };
}

describe("public feature boundary", () => {
  it("accepts the exact public allowlist", () => {
    const parsed = parsePublicFeatureRecord(publicFeatures(), schema);
    expect(parsed.playerColor).toBe("white");
    expect(parsed.symbolicWhiteRuleProbabilities).toEqual([0.75, 0.25]);
  });

  it.each([
    "trueDrawback",
    "hiddenParameters",
    "drawbackInternalState",
    "drawbackLegalMoves",
    "ruleTriggered",
    "forced",
    "result",
    "gameId",
    "botAgentId",
  ])("rejects non-feature key %s", (key) => {
    expect(() =>
      parsePublicFeatureRecord(
        { ...publicFeatures(), [key]: "leak" },
        schema,
      )
    ).toThrow(DatasetContractError);
  });

  it("rejects unknown future fields", () => {
    expect(() =>
      parsePublicFeatureRecord(
        { ...publicFeatures(), futureSecret: true },
        schema,
      )
    ).toThrow(/unknown futureSecret/u);
  });

  it("rejects invalid symbolic probability vectors", () => {
    expect(() =>
      parsePublicFeatureRecord(
        {
          ...publicFeatures(),
          symbolicWhiteRuleProbabilities: [0.8, 0.8],
        },
        schema,
      )
    ).toThrow(/sum to one or zero/u);
  });
});

describe("combined storage row", () => {
  it("splits public features, labels, and evaluation metadata", () => {
    const parsed = parseDatasetRow(datasetRow(), schema);
    expect(parsed.features).not.toHaveProperty("trueDrawback");
    expect(parsed.features).not.toHaveProperty("gameId");
    expect(parsed.labels.trueDrawback).toBe("vegan");
    expect(parsed.evaluation.gameId).toBe("00000001-000000");
  });

  it("rejects unknown storage fields instead of silently exposing them", () => {
    expect(() =>
      parseDatasetRow({ ...datasetRow(), futureTarget: "secret" }, schema)
    ).toThrow(/unknown futureTarget/u);
  });

  it("keeps hidden-parameter shape in parity with the Python trainer", () => {
    expect(() =>
      parseDatasetRow({ ...datasetRow(), hiddenParameters: 7 }, schema)
    ).toThrow(/hiddenParameters must be a plain object/u);
    expect(
      parseDatasetRow({ ...datasetRow(), hiddenParameters: null }, schema)
        .labels.hiddenParameters,
    ).toBeNull();
  });
});
