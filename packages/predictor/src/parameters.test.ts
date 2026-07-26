import { describe, expect, it } from "vitest";
import type {
  DrawbackRule,
} from "@drawbackengine/drawback-engine";
import {
  aggregateParameterPosteriors,
  aggregateRulePosteriors,
  asHypothesisSeed,
  canonicalHypothesisId,
  expandHypothesisSeeds,
  probability,
  SymbolicPredictor,
} from "./index.js";
import { PredictorTestGame } from "./test-game.js";

interface State {
  readonly observations: number;
}

interface SquareParameters extends Record<string, unknown> {
  readonly forbiddenFrom: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

const parameterizedRule: DrawbackRule<State, SquareParameters> = {
  id: "hidden-square",
  name: "Hidden Square",
  description: "Test-only parameterized rule.",
  verification: "verified",
  generateParameters: () => ({ forbiddenFrom: "e2" }),
  initialize: () => ({ observations: 0 }),
  filterLegalMoves: (context, moves) =>
    moves.filter((move) => move.from !== context.parameters.forbiddenFrom),
  applyMove: (context) => ({
    observations: context.state.observations + 1,
  }),
  checkStartOfTurnLoss: () => null,
};

const otherRule: DrawbackRule<State, Record<string, never>> = {
  id: "other",
  name: "Other",
  description: "Test control.",
  verification: "verified",
  generateParameters: () => ({}),
  initialize: () => ({ observations: 0 }),
  filterLegalMoves: (_context, moves) => [...moves],
  applyMove: (context) => ({
    observations: context.state.observations + 1,
  }),
  checkStartOfTurnLoss: () => null,
};

describe("parameter hypotheses", () => {
  it("creates stable canonical IDs independent of object key order", () => {
    const first = canonicalHypothesisId("rule", {
      square: "e4",
      nested: { rank: 4, enabled: true },
    });
    const second = canonicalHypothesisId("rule", {
      nested: { enabled: true, rank: 4 },
      square: "e4",
    });
    expect(first).toBe(second);
    expect(first).not.toBe(
      canonicalHypothesisId("rule", { square: "d4" }),
    );
  });

  it("rejects sparse parameter arrays before they can collide canonically", () => {
    const sparseItems = Array<string>(1);
    expect(
      canonicalHypothesisId("rule", { items: [] }),
    ).toBe('rule::{"items":[]}');
    expect(
      () => canonicalHypothesisId("rule", { items: sparseItems }),
    ).toThrow(/parameter arrays must not contain holes/iu);

    const game = new PredictorTestGame();
    expect(
      () =>
        new SymbolicPredictor(
          {
            white: [
              asHypothesisSeed(parameterizedRule, {
                forbiddenFrom: "e2",
                metadata: { items: sparseItems },
              }),
            ],
            black: [asHypothesisSeed(otherRule, {})],
          },
          game.view(),
        ),
    ).toThrow(/parameter arrays must not contain holes/iu);
  });

  it("preserves a drawback prior while distributing it across variants", () => {
    const expanded = expandHypothesisSeeds(
      parameterizedRule,
      [
        { parameters: { forbiddenFrom: "e2" }, weight: 1 },
        { parameters: { forbiddenFrom: "d2" }, weight: 3 },
      ],
      0.6,
    );
    expect(expanded[0]?.priorProbability).toBeCloseTo(0.15);
    expect(expanded[1]?.priorProbability).toBeCloseTo(0.45);

    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [
          ...expanded,
          asHypothesisSeed(otherRule, {}, 0.4),
        ],
        black: [asHypothesisSeed(otherRule, {})],
      },
      game.view(),
    );
    const rules = aggregateRulePosteriors(predictor.state.white);
    const hiddenSquare = rules.find(
      (rule) => rule.drawbackId === "hidden-square",
    );
    expect(hiddenSquare).toMatchObject({
      variantCount: 2,
      liveVariantCount: 2,
    });
    expect(hiddenSquare?.probability).toBeCloseTo(0.6);
    expect(rules.find((rule) => rule.drawbackId === "other")?.probability)
      .toBeCloseTo(0.4);

    const parameter = aggregateParameterPosteriors(
      predictor.state.white,
      "hidden-square",
    )[0];
    expect(parameter?.parameter).toBe("forbiddenFrom");
    expect(parameter?.values.map((value) => ({
      value: value.value,
      conditional: value.conditionalProbability,
    }))).toEqual([
      { value: "d2", conditional: 0.75 },
      { value: "e2", conditional: 0.25 },
    ]);
  });

  it("eliminates variants independently and never restores one from a survivor", () => {
    const variants = expandHypothesisSeeds(parameterizedRule, [
      { parameters: { forbiddenFrom: "e2" } },
      { parameters: { forbiddenFrom: "d2" } },
    ]);
    const game = new PredictorTestGame();
    const initial = game.view();
    const predictor = new SymbolicPredictor(
      {
        white: variants,
        black: [asHypothesisSeed(otherRule, {})],
      },
      initial,
    );
    predictor.observe(game.play("e2", "e4"));
    const eliminatedId = canonicalHypothesisId("hidden-square", {
      forbiddenFrom: "e2",
    });
    expect(
      predictor.state.white.hypotheses.find(
        (hypothesis) => hypothesis.hypothesisId === eliminatedId,
      ),
    ).toMatchObject({
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });

    predictor.observe(game.play("e7", "e5"));

    const variantsAfter = predictor.state.white.hypotheses;
    const eliminated = variantsAfter.find(
      (hypothesis) => hypothesis.hypothesisId === eliminatedId,
    );
    const survivor = variantsAfter.find(
      (hypothesis) => hypothesis.hypothesisId !== eliminatedId,
    );
    expect(probability(eliminated?.logProbability ?? 0)).toBe(0);
    expect(eliminated?.eliminated).toBe(true);
    expect(survivor?.eliminated).toBe(false);
    expect(
      aggregateRulePosteriors(predictor.state.white)[0],
    ).toMatchObject({
      eliminated: false,
      liveVariantCount: 1,
      variantCount: 2,
    });
  });

  it("keeps parameter variants isolated between White and Black", () => {
    const variants = expandHypothesisSeeds(parameterizedRule, [
      { parameters: { forbiddenFrom: "e2" } },
      { parameters: { forbiddenFrom: "d2" } },
    ]);
    const game = new PredictorTestGame();
    const initial = game.view();
    const predictor = new SymbolicPredictor(
      { white: variants, black: variants },
      initial,
    );
    const blackBefore = predictor.state.black;
    predictor.observe(game.play("e2", "e4"));

    expect(predictor.state.black).toEqual(blackBefore);
    expect(predictor.state.black.hypotheses.every(
      (hypothesis) => !hypothesis.eliminated,
    )).toBe(true);
  });

  it("rejects duplicate variants instead of merging their state", () => {
    expect(() =>
      expandHypothesisSeeds(parameterizedRule, [
        { parameters: { forbiddenFrom: "e2" } },
        { parameters: { forbiddenFrom: "e2" } },
      ]),
    ).toThrow("Duplicate parameter variant");
  });
});
