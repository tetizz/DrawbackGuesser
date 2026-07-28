import { describe, expect, it } from "vitest";
import {
  CAPTURABLE_HYPOTHESIS_RULE_IDS,
} from "@drawbackguesser/predictor";
import {
  CAPTURABLE_SYMBOLIC_RULE_COUNT,
  convertPlayerPrivateTraceToDatasetRows,
  deriveCapturablePublicDatasetRows,
} from "./player-private-converter.js";
import {
  playerPrivateTraceFixture,
} from "./player-private-test-fixture.js";

describe("player-private Engine trace conversion", () => {
  it("derives capturable public features before attaching exact labels", () => {
    const trace = playerPrivateTraceFixture();
    const rows = convertPlayerPrivateTraceToDatasetRows(trace);

    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({
      authorityId: "capturable-king/v1",
      publicAuthorityPositionBefore: trace.initialPosition,
      fenBefore: trace.initialPosition.fen,
      move: "e2e4",
      moveNumber: 1,
      ply: 0,
      playerColor: "white",
      historySan: [],
      ordinaryLegalMoves: trace.plies[0]?.authorityLegalMoves,
      symbolicFeatureVersion: 9,
      opportunityFeatureVersion: 1,
      trueDrawback: "vegan",
      hiddenParameters: {},
      drawbackInternalState: { movesApplied: 0 },
      gameId: trace.gameId,
      seed: trace.seed,
    });
    expect(rows[1]).toMatchObject({
      playerColor: "black",
      trueDrawback: "checkers",
    });
    expect(rows[0]?.symbolicWhiteRuleProbabilities).toHaveLength(
      CAPTURABLE_SYMBOLIC_RULE_COUNT,
    );
    expect(CAPTURABLE_SYMBOLIC_RULE_COUNT).toBe(
      CAPTURABLE_HYPOTHESIS_RULE_IDS.length,
    );
    expect(rows[0]?.symbolicActiveRuleOpportunityFeatures).toHaveLength(
      CAPTURABLE_SYMBOLIC_RULE_COUNT * 4,
    );
    expect(
      rows[0]?.symbolicActiveRuleOpportunityFeatures?.every(
        (value) => Number.isFinite(value) && value >= 0 && value <= 1,
      ),
    ).toBe(true);
    expect(
      rows[0]?.symbolicWhiteRuleProbabilities.reduce(
        (sum, probability) => sum + probability,
        0,
      ),
    ).toBeCloseTo(1, 12);
  });

  it("keeps public rows invariant when both valid hidden rules change", () => {
    const first = playerPrivateTraceFixture({
      whiteRuleId: "vegan",
      blackRuleId: "checkers",
    });
    const second = playerPrivateTraceFixture({
      whiteRuleId: "lame-duck",
      blackRuleId: "spice-of-life",
    });

    const firstPublicRows = deriveCapturablePublicDatasetRows(first);
    const secondPublicRows = deriveCapturablePublicDatasetRows(second);
    expect(secondPublicRows).toEqual(firstPublicRows);
    expect(
      secondPublicRows[0]?.features
        .symbolicActiveRuleOpportunityFeatures,
    ).toEqual(
      firstPublicRows[0]?.features
        .symbolicActiveRuleOpportunityFeatures,
    );
    const serializedPublicRows = JSON.stringify(firstPublicRows);
    for (
      const privateOpportunityKey of [
        "hypothesisIndex",
        "observedMoveLegal",
        "allowedMoveCount",
        "drawbackId",
        "parameters",
        "internalState",
      ]
    ) {
      expect(serializedPublicRows).not.toContain(
        `"${privateOpportunityKey}"`,
      );
    }
    expect(
      convertPlayerPrivateTraceToDatasetRows(first).map(
        (row) => row.trueDrawback,
      ),
    ).toEqual(["vegan", "checkers"]);
    expect(
      convertPlayerPrivateTraceToDatasetRows(second).map(
        (row) => row.trueDrawback,
      ),
    ).toEqual(["lame-duck", "spice-of-life"]);
  });

  it("preserves the public one-reply king-passant authority state", () => {
    const trace = playerPrivateTraceFixture({
      whiteRuleId: "vegan",
      blackRuleId: "vegan",
      initialFen: "5r1k/8/8/8/8/8/8/4K2R w K - 0 1",
      moves: ["e1g1", "f8f1"],
    });
    const rows = convertPlayerPrivateTraceToDatasetRows(trace);

    expect(rows[1]?.publicAuthorityPositionBefore?.kingPassant).toEqual({
      victim: "white",
      kingSquare: "g1",
      targets: ["f1"],
    });
    expect(rows[1]?.move).toBe("f8f1");
    expect(rows[1]?.result).toEqual({
      kind: "king-capture",
      winner: "black",
      capturedKing: "white",
      method: "castling-en-passant",
    });
  });

  it("represents both Triple Play hidden-parameter particles", () => {
    const trace = playerPrivateTraceFixture({
      whiteRuleId: "triple-play",
    });
    const row = convertPlayerPrivateTraceToDatasetRows(trace)[0];

    expect(row?.trueDrawback).toBe("triple-play");
    const parameters = row?.hiddenParameters;
    expect(typeof parameters).toBe("object");
    expect(parameters).not.toBeNull();
    expect(
      (parameters as Readonly<Record<string, unknown>>)["requiredType"],
    ).toMatch(/^(bishop|knight)$/u);
  });

  it("rejects secret-state and authority-snapshot tampering", () => {
    const trace = playerPrivateTraceFixture();
    const first = trace.plies[0];
    if (first === undefined) {
      throw new Error("Expected fixture ply.");
    }
    expect(() => convertPlayerPrivateTraceToDatasetRows({
      ...trace,
      plies: [{
        ...first,
        activeSecret: {
          ...first.activeSecret,
          drawbackInternalState: { forged: true },
        },
      }, ...trace.plies.slice(1)],
    })).toThrow("does not match exact replay");
    expect(() => convertPlayerPrivateTraceToDatasetRows({
      ...trace,
      plies: [{
        ...first,
        positionBefore: {
          ...first.positionBefore,
          orthodoxCompatible: !first.positionBefore.orthodoxCompatible,
        },
      }, ...trace.plies.slice(1)],
    })).toThrow("must equal");
  });
});
