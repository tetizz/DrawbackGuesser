import { describe, expect, it } from "vitest";
import { DEFAULT_HYPOTHESIS_RULE_IDS } from "@drawbackguesser/predictor";
import {
  convertTraceToDatasetRows,
  derivePublicDatasetRows,
  SYMBOLIC_RULE_COUNT,
} from "./converter.js";
import { traceFixture } from "./test-fixture.js";

describe("Engine trace conversion", () => {
  it("derives exact public symbolic features before attaching labels", () => {
    const trace = traceFixture();
    const rows = convertTraceToDatasetRows(trace);

    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({
      fenBefore: trace.initialFen,
      move: "e2e4",
      moveNumber: 1,
      ply: 0,
      playerColor: "white",
      historySan: [],
      trueDrawback: "vegan",
      hiddenParameters: {},
      drawbackInternalState: { movesApplied: 0 },
      ruleTriggered: false,
      forced: false,
      gameId: trace.gameId,
      seed: trace.seed,
    });
    expect(rows[1]?.historySan).toEqual(["e4"]);
    expect(rows[0]?.symbolicWhiteRuleProbabilities).toHaveLength(
      SYMBOLIC_RULE_COUNT,
    );
    expect(rows[0]?.symbolicBlackRuleProbabilities).toHaveLength(
      SYMBOLIC_RULE_COUNT,
    );
    expect(rows[0]?.symbolicWhiteEliminated).toHaveLength(
      DEFAULT_HYPOTHESIS_RULE_IDS.length,
    );
    expect(
      rows[0]?.symbolicWhiteRuleProbabilities.reduce(
        (sum, probability) => sum + probability,
        0,
      ),
    ).toBeCloseTo(1, 12);
  });

  it("keeps the public projection invariant when private truth changes", () => {
    const vegan = traceFixture({ drawbackId: "vegan" });
    const checkers = traceFixture({ drawbackId: "checkers" });

    expect(derivePublicDatasetRows(checkers)).toEqual(
      derivePublicDatasetRows(vegan),
    );
    expect(convertTraceToDatasetRows(vegan)[0]?.trueDrawback).toBe("vegan");
    expect(convertTraceToDatasetRows(checkers)[0]?.trueDrawback).toBe(
      "checkers",
    );
  });

  it("uses the FEN fullmove number and preserves promotion UCI", () => {
    const trace = traceFixture({
      initialFen: "4k3/P7/8/8/8/8/8/4K3 w - - 0 17",
      moves: ["a7a8q"],
    });
    const row = convertTraceToDatasetRows(trace)[0];

    expect(row?.moveNumber).toBe(17);
    expect(row?.move).toBe("a7a8q");
    expect(row?.san).toContain("a8=Q");
  });

  it("fails closed when exact symbolic legality eliminates the trace truth", () => {
    const dishonestTrace = traceFixture({
      drawbackId: "checkers",
      initialFen: "4k3/8/8/8/8/p7/1P6/4K3 w - - 0 1",
      moves: ["b2b3"],
    });

    expect(() => convertTraceToDatasetRows(dishonestTrace)).toThrow(
      "drawbackLegalMoves does not match executable rule replay",
    );
  });

  it("rejects poisoned legal-mask and trigger labels for an otherwise legal move", () => {
    const poisoned = traceFixture({
      drawbackId: "checkers",
      initialFen: "4k3/8/8/8/8/p7/1P6/4K3 w - - 0 1",
      moves: ["b2a3"],
    });

    expect(() => convertTraceToDatasetRows(poisoned)).toThrow(
      "drawbackLegalMoves does not match executable rule replay",
    );
  });

  it("rejects parameter shapes that the Python trainer cannot encode", () => {
    const trace = traceFixture({ moves: ["e2e4"] });
    const poisoned = {
      ...trace,
      plies: trace.plies.map((ply, index) => ({
        ...ply,
        activeSecret: {
          ...ply.activeSecret,
          hiddenParameters:
            index === 0 ? 7 : ply.activeSecret.hiddenParameters,
        },
      })),
    };

    expect(() => convertTraceToDatasetRows(poisoned)).toThrow(
      "hiddenParameters must be a JSON object",
    );
  });

  it("rejects unknown parameter keys and values against the catalog schema", () => {
    const parameterless = traceFixture({ moves: ["e2e4"] });
    const extraKey = {
      ...parameterless,
      plies: parameterless.plies.map((ply) => ({
        ...ply,
        activeSecret: {
          ...ply.activeSecret,
          hiddenParameters: { secret: "poison" },
        },
      })),
    };
    expect(() => convertTraceToDatasetRows(extraKey)).toThrow(
      "hiddenParameters keys do not match rule vegan",
    );

    const squareRule = traceFixture({
      moves: ["e2e4"],
      drawbackId: "untitled-duck-drawback",
    });
    const invalidSquare = {
      ...squareRule,
      plies: squareRule.plies.map((ply) => ({
        ...ply,
        activeSecret: {
          ...ply.activeSecret,
          hiddenParameters: { square: "z9" },
        },
      })),
    };
    expect(() => convertTraceToDatasetRows(invalidSquare)).toThrow(
      "hiddenParameters.square is invalid",
    );
  });

  it("rejects a terminal result that disagrees with executable replay", () => {
    const trace = traceFixture();
    const poisoned = {
      ...trace,
      stoppedAtPlyLimit: false,
      result: {
        kind: "draw",
        reason: "tampered",
      } as const,
    };

    expect(() => convertTraceToDatasetRows(poisoned)).toThrow(
      "result does not match executable game replay",
    );
  });
});
