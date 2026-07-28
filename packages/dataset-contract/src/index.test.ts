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

const capturableSchema8 = {
  symbolicFeatureVersion: 8,
  symbolicRuleCount: 2,
  authorityId: "capturable-king/v1",
} as const;

const capturableSchema9 = {
  symbolicFeatureVersion: 9,
  symbolicRuleCount: 2,
  authorityId: "capturable-king/v1",
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

function capturablePublicFeatures(): Record<string, unknown> {
  return {
    ...publicFeatures(),
    symbolicFeatureVersion: 8,
    authorityId: "capturable-king/v1",
    publicAuthorityPositionBefore: {
      format: "drawbacktrainer-public-position",
      version: 1,
      authorityId: "capturable-king/v1",
      fen: publicFeatures()["fenBefore"],
      orthodoxCompatible: true,
      kingPassant: {
        victim: "white",
        kingSquare: "g1",
        targets: ["f1"],
      },
      terminal: null,
    },
  };
}

function capturableOpportunityPublicFeatures(): Record<string, unknown> {
  return {
    ...capturablePublicFeatures(),
    symbolicFeatureVersion: 9,
    opportunityFeatureVersion: 1,
    symbolicActiveRuleOpportunityFeatures: [
      1, 0.5, 1, 0,
      0.25, 0.125, 0, 0,
    ],
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

  it("accepts only the complete public capturable authority snapshot", () => {
    const parsed = parsePublicFeatureRecord(
      capturablePublicFeatures(),
      capturableSchema8,
    );
    expect(parsed.authorityId).toBe("capturable-king/v1");
    expect(parsed.publicAuthorityPositionBefore?.kingPassant).toEqual({
      victim: "white",
      kingSquare: "g1",
      targets: ["f1"],
    });

    const poisoned = capturablePublicFeatures();
    poisoned["publicAuthorityPositionBefore"] = {
      ...(poisoned["publicAuthorityPositionBefore"] as object),
      hiddenParameters: { square: "e4" },
    };
    expect(() =>
      parsePublicFeatureRecord(poisoned, capturableSchema8)
    ).toThrow(/unknown hiddenParameters/u);
  });

  it("does not permit capturable authority state in the orthodox schema", () => {
    expect(() =>
      parsePublicFeatureRecord(capturablePublicFeatures(), schema)
    ).toThrow(/unknown authorityId/u);
  });

  it("requires the exact schema-9 opportunity contract", () => {
    const parsed = parsePublicFeatureRecord(
      capturableOpportunityPublicFeatures(),
      capturableSchema9,
    );

    expect(parsed.opportunityFeatureVersion).toBe(1);
    expect(parsed.symbolicActiveRuleOpportunityFeatures).toEqual([
      1, 0.5, 1, 0,
      0.25, 0.125, 0, 0,
    ]);
    expect(Object.isFrozen(
      parsed.symbolicActiveRuleOpportunityFeatures,
    )).toBe(true);

    const missingVersion = capturableOpportunityPublicFeatures();
    delete missingVersion["opportunityFeatureVersion"];
    expect(() =>
      parsePublicFeatureRecord(missingVersion, capturableSchema9)
    ).toThrow(/missing opportunityFeatureVersion/u);

    expect(() =>
      parsePublicFeatureRecord(
        {
          ...capturableOpportunityPublicFeatures(),
          opportunityFeatureVersion: 2,
        },
        capturableSchema9,
      )
    ).toThrow(/opportunityFeatureVersion must be 1/u);
    expect(() =>
      parsePublicFeatureRecord(
        {
          ...capturableOpportunityPublicFeatures(),
          symbolicActiveRuleOpportunityFeatures: [
            1, 0.5, 1, 0,
            0.25, 0.125, Number.NaN, 0,
          ],
        },
        capturableSchema9,
      )
    ).toThrow(/8 finite values/u);
  });

  it("keeps schema 8 frozen and rejects schema-9 opportunity fields", () => {
    const legacy = capturablePublicFeatures();
    legacy["symbolicFeatureVersion"] = 8;
    expect(
      parsePublicFeatureRecord(legacy, capturableSchema8),
    ).not.toHaveProperty("opportunityFeatureVersion");
    expect(() =>
      parsePublicFeatureRecord(
        capturableOpportunityPublicFeatures(),
        capturableSchema8,
      )
    ).toThrow(/unknown opportunityFeatureVersion/u);
  });

  it.each([7, 10])(
    "rejects unsupported capturable symbolic schema %i",
    (symbolicFeatureVersion) => {
      expect(() =>
        parsePublicFeatureRecord(
          {
            ...capturablePublicFeatures(),
            symbolicFeatureVersion,
          },
          {
            ...capturableSchema8,
            symbolicFeatureVersion,
          },
        )
      ).toThrow(/capturable symbolicFeatureVersion must be 8 or 9/u);
    },
  );

  it.each([7, 10])(
    "keeps standard-authority symbolic schema %i generic",
    (symbolicFeatureVersion) => {
      const parsed = parsePublicFeatureRecord(
        { ...publicFeatures(), symbolicFeatureVersion },
        { ...schema, symbolicFeatureVersion },
      );
      expect(parsed.symbolicFeatureVersion).toBe(symbolicFeatureVersion);
    },
  );
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

  it("splits schema-9 opportunity features without exposing labels", () => {
    const parsed = parseDatasetRow(
      {
        ...datasetRow(),
        ...capturableOpportunityPublicFeatures(),
      },
      capturableSchema9,
    );

    expect(parsed.features.opportunityFeatureVersion).toBe(1);
    expect(parsed.features.symbolicActiveRuleOpportunityFeatures).toHaveLength(
      8,
    );
    expect(parsed.features).not.toHaveProperty("trueDrawback");
    expect(parsed.labels.trueDrawback).toBe("vegan");
  });
});
