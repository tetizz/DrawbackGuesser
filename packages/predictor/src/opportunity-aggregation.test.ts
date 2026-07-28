import { describe, expect, it } from "vitest";
import {
  aggregateRuleOpportunityFeatures,
  RULE_OPPORTUNITY_FEATURE_FIELDS,
  RULE_OPPORTUNITY_FEATURE_VERSION,
} from "./opportunity.js";
import type {
  DrawbackHypothesis,
  HypothesisMoveOpportunity,
  PredictionOpportunitySnapshot,
  PredictionState,
} from "./types.js";

function hypothesis(
  hypothesisId: string,
  drawbackId: string,
  logProbability: number,
  eliminated = false,
): DrawbackHypothesis {
  return {
    hypothesisId,
    drawbackId,
    parameters: {},
    internalState: {},
    logProbability,
    eliminated,
    evidence: [],
  };
}

function state(
  white: readonly DrawbackHypothesis[],
  black: readonly DrawbackHypothesis[] = white,
): PredictionState {
  return {
    white: { hypotheses: white },
    black: { hypotheses: black },
  };
}

function known(
  hypothesisIndex: number,
  drawbackId: string,
  allowedMoveCount: number,
  options: {
    readonly ordinaryLegalMoveCount?: number;
    readonly triggered?: boolean;
    readonly forced?: boolean;
  } = {},
): HypothesisMoveOpportunity {
  const ordinaryLegalMoveCount = options.ordinaryLegalMoveCount ?? 4;
  return {
    status: "known",
    hypothesisIndex,
    drawbackId,
    ordinaryLegalMoveCount,
    allowedMoveCount,
    allowedMoveFraction: ordinaryLegalMoveCount === 0
      ? 0
      : allowedMoveCount / ordinaryLegalMoveCount,
    triggered: options.triggered ?? false,
    forced: options.forced ?? false,
    observedMoveLegal: true,
  };
}

function unavailable(
  hypothesisIndex: number,
  drawbackId: string,
  status: "unknown" | "eliminated",
  ordinaryLegalMoveCount = 4,
): HypothesisMoveOpportunity {
  return {
    status,
    hypothesisIndex,
    drawbackId,
    ordinaryLegalMoveCount,
    allowedMoveCount: null,
    allowedMoveFraction: null,
    triggered: null,
    forced: null,
    observedMoveLegal: null,
  };
}

function snapshot(
  hypotheses: readonly HypothesisMoveOpportunity[],
  color: "white" | "black" = "white",
): PredictionOpportunitySnapshot {
  return { color, hypotheses };
}

describe("rule opportunity feature aggregation", () => {
  it("uses the frozen version-one field order", () => {
    expect(RULE_OPPORTUNITY_FEATURE_VERSION).toBe(1);
    expect(RULE_OPPORTUNITY_FEATURE_FIELDS).toEqual([
      "knownMass",
      "allowedMoveFractionMass",
      "triggeredMass",
      "forcedMass",
    ]);
    expect(Object.isFrozen(RULE_OPPORTUNITY_FEATURE_FIELDS)).toBe(true);
  });

  it("emits fixed-order rule channels and zeros for eliminated rules", () => {
    const prediction = state([
      hypothesis("a", "rule-a", 0),
      hypothesis("b", "rule-b", Number.NEGATIVE_INFINITY, true),
    ]);
    const result = aggregateRuleOpportunityFeatures(
      prediction,
      snapshot([
        known(0, "rule-a", 2, { triggered: true, forced: true }),
        unavailable(1, "rule-b", "eliminated"),
      ]),
      ["rule-b", "rule-a"],
    );

    expect(result).toEqual([0, 0, 0, 0, 1, 0.5, 1, 1]);
    expect(Object.isFrozen(result)).toBe(true);
  });

  it("keeps unknown variants in the conditional denominator", () => {
    const prediction = state([
      hypothesis("known", "rule-a", Math.log(0.75)),
      hypothesis("unknown", "rule-a", Math.log(0.25)),
    ]);
    const result = aggregateRuleOpportunityFeatures(
      prediction,
      snapshot([
        known(0, "rule-a", 2, { triggered: true }),
        unavailable(1, "rule-a", "unknown"),
      ]),
      ["rule-a"],
    );

    expect(result).toEqual([0.75, 0.375, 0.75, 0]);
  });

  it("uses stable conditional log-sum-exp for extreme probabilities", () => {
    const prediction = state([
      hypothesis("known", "rule-a", -1_000),
      hypothesis("unknown", "rule-a", -1_001),
    ]);
    const result = aggregateRuleOpportunityFeatures(
      prediction,
      snapshot([
        known(0, "rule-a", 4),
        unavailable(1, "rule-a", "unknown"),
      ]),
      ["rule-a"],
    );
    const expectedKnownMass = 1 / (1 + Math.exp(-1));

    expect(result[0]).toBeCloseTo(expectedKnownMass, 12);
    expect(result[1]).toBeCloseTo(expectedKnownMass, 12);
  });

  it("reads only the snapshot's active color distribution", () => {
    const prediction = state(
      [hypothesis("white", "rule-a", 0)],
      [
        hypothesis("black-known", "rule-a", Math.log(0.25)),
        hypothesis("black-unknown", "rule-a", Math.log(0.75)),
      ],
    );

    expect(
      aggregateRuleOpportunityFeatures(
        prediction,
        snapshot([known(0, "rule-a", 4)], "white"),
        ["rule-a"],
      ),
    ).toEqual([1, 1, 0, 0]);
    const blackFeatures = aggregateRuleOpportunityFeatures(
      prediction,
      snapshot([
        known(0, "rule-a", 2),
        unavailable(1, "rule-a", "unknown"),
      ], "black"),
      ["rule-a"],
    );
    expect(blackFeatures[0]).toBeCloseTo(0.25, 12);
    expect(blackFeatures[1]).toBeCloseTo(0.125, 12);
    expect(blackFeatures.slice(2)).toEqual([0, 0]);
  });

  it("fails closed on index, rule, length, and ordering corruption", () => {
    const prediction = state([hypothesis("a", "rule-a", 0)]);

    expect(() =>
      aggregateRuleOpportunityFeatures(
        prediction,
        snapshot([known(1, "rule-a", 4)]),
        ["rule-a"],
      )
    ).toThrow(/misaligned/u);
    expect(() =>
      aggregateRuleOpportunityFeatures(
        prediction,
        snapshot([known(0, "rule-b", 4)]),
        ["rule-a"],
      )
    ).toThrow(/mismatched drawback ID/u);
    expect(() =>
      aggregateRuleOpportunityFeatures(
        prediction,
        snapshot([]),
        ["rule-a"],
      )
    ).toThrow(/length/u);
    expect(() =>
      aggregateRuleOpportunityFeatures(
        prediction,
        snapshot([known(0, "rule-a", 4)]),
        ["rule-a", "rule-a"],
      )
    ).toThrow(/non-empty and unique/u);
    expect(() =>
      aggregateRuleOpportunityFeatures(
        prediction,
        snapshot([known(0, "rule-a", 4)]),
        ["rule-a", "rule-b"],
      )
    ).toThrow(/unrepresented/u);
  });
});
