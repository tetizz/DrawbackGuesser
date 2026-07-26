import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  aggregateRulePosteriors,
  DEFAULT_HYPOTHESIS_RULE_IDS,
  SymbolicPredictor,
} from "@drawbackguesser/predictor";
import {
  createHypothesisSeeds,
  GAMBLER_OUTCOME_COUNT,
  HYPOTHESIS_COVERAGE,
} from "./hypothesis-catalog.js";

const INITIAL_POSITION = {
  fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  turn: "white",
  ply: 0,
  history: [],
} as const;

describe("web hypothesis catalog", () => {
  it("is deterministic, unique, and covers exact and sampled parameter sets", () => {
    const first = createHypothesisSeeds();
    const second = createHypothesisSeeds();
    expect(second.map((seed) =>
      seed.kind === "rerandomized" ? {} : seed.parameters
    )).toEqual(
      first.map((seed) =>
        seed.kind === "rerandomized" ? {} : seed.parameters
      ),
    );
    expect(first).toHaveLength(
      170 + 64 + 8 + 2 + 2 + 8 + 8 + 4 + 5,
    );

    const predictor = new SymbolicPredictor(
      { white: first, black: second },
      INITIAL_POSITION,
    );
    const rules = aggregateRulePosteriors(predictor.state.white);
    expect(rules).toHaveLength(182);
    expect(rules.every(({ probability }) => Math.abs(probability - 1 / 182) < 1e-12))
      .toBe(true);
    expect(
      new Set(first.map((seed) =>
        seed.kind === "rerandomized"
          ? `${seed.drawbackId}:{}`
          : `${seed.rule.id}:${JSON.stringify(seed.parameters)}`
      ))
        .size,
    ).toBe(first.length);
    const expectedVariants = {
      crenellations: ["dark", "light"],
      theocracy: ["even", "odd"],
      "active-volcano": ["c4", "c5", "d4", "d5", "e4", "e5", "f4", "f5"],
      "comfort-zone": ["c4", "c5", "d4", "d5", "e4", "e5", "f4", "f5"],
    } as const;
    for (const [drawbackId, expected] of Object.entries(expectedVariants)) {
      const variants = first.filter(
        (seed) =>
          seed.kind !== "rerandomized" && seed.rule.id === drawbackId,
      );
      const key = drawbackId === "crenellations"
        ? "squareColor"
        : drawbackId === "theocracy"
          ? "captureParity"
          : "square";
      expect(
        variants.map((seed) =>
          seed.kind === "rerandomized"
            ? ""
            : String(seed.parameters[key])
        ).sort(),
      ).toEqual([...expected].sort());
      expect(
        variants.every(
          (seed) =>
            seed.priorProbability !== undefined &&
            Math.abs(seed.priorProbability - 1 / variants.length) < 1e-12,
        ),
      ).toBe(true);
    }
  });

  it("marginalizes Gambler analytically without sampled hard elimination", () => {
    expect(
      HYPOTHESIS_COVERAGE.find(({ drawbackId }) => drawbackId === "gambler"),
    ).toMatchObject({
      mode: "analytic",
      variantCount: GAMBLER_OUTCOME_COUNT,
    });
  });

  it("represents rerandomized rules as one seed with analytic coverage", () => {
    const seeds = createHypothesisSeeds();
    const predictor = new SymbolicPredictor(
      { white: seeds, black: seeds },
      INITIAL_POSITION,
    );
    const expected = {
      colorblind: 2,
      gambler: 6,
      "hand-and-brainless": 6,
      obsession: 64,
      "winds-of-fate": 2,
    } as const;
    for (const [drawbackId, variantCount] of Object.entries(expected)) {
      const matching = seeds.filter(
        (seed) =>
          seed.kind === "rerandomized" &&
          seed.drawbackId === drawbackId,
      );
      expect(matching).toHaveLength(1);
      const publicHypotheses = predictor.state.white.hypotheses.filter(
        (hypothesis) => hypothesis.drawbackId === drawbackId,
      );
      expect(publicHypotheses).toHaveLength(1);
      expect(publicHypotheses[0]?.parameters).toEqual({});
      expect(publicHypotheses[0]?.hypothesisId).toBe(`${drawbackId}::{}`);
      expect(
        HYPOTHESIS_COVERAGE.find(
          (coverage) => coverage.drawbackId === drawbackId,
        ),
      ).toMatchObject({
        mode: "analytic",
        variantCount,
      });
    }
  });

  it("keeps the Python hybrid feature schema in exact predictor order", () => {
    const source = readFileSync(
      new URL(
        "../../../ml/training/drawback_ml/symbolic_schema.py",
        import.meta.url,
      ),
      "utf8",
    );
    const tuple = /SYMBOLIC_RULE_IDS = \(([\s\S]*?)\)/u.exec(source)?.[1];
    if (tuple === undefined) {
      throw new Error("Python symbolic rule schema tuple is missing.");
    }
    const pythonRuleIds = [...tuple.matchAll(/"([^"]+)"/gu)]
      .map((match) => match[1]);
    expect(pythonRuleIds).toEqual([...DEFAULT_HYPOTHESIS_RULE_IDS]);
    expect(source).toContain("SYMBOLIC_FEATURE_VERSION = 6");
  });
});
