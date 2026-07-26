import { describe, expect, it } from "vitest";
import type {
  ChessMove,
} from "@drawbackengine/drawback-engine";
import {
  asHypothesisSeed,
  asRerandomizedHypothesisSeed,
  probability,
  SymbolicPredictor,
  type RerandomizedHypothesisSeed,
} from "./index.js";
import { unrestrictedRule } from "@drawbackengine/drawback-engine";
import { PredictorTestGame } from "./test-game.js";

interface TestState {
  readonly movesApplied: number;
}

type TestOutcome = "allow-e4" | "allow-d4";

function rerandomizedSeed(
  probabilities: readonly [number, number] = [0.25, 0.75],
): RerandomizedHypothesisSeed<TestState, TestOutcome> {
  return {
    kind: "rerandomized",
    drawbackId: "rerandomized-test",
    name: "Rerandomized test",
    initialize: () => ({ movesApplied: 0 }),
    outcomes: () => [
      { outcome: "allow-e4", probability: probabilities[0] },
      { outcome: "allow-d4", probability: probabilities[1] },
    ],
    filterLegalMoves: (_context, outcome, moves) =>
      moves.filter((move) =>
        outcome === "allow-e4" ? move.to === "e4" : move.to === "d4"
      ),
    applyObservedMove: (context) => ({
      movesApplied: context.state.movesApplied + 1,
    }),
  };
}

describe("analytic rerandomized hypotheses", () => {
  it("marginalizes compatible outcome probability without exposing a seed", () => {
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(unrestrictedRule, {}),
          asRerandomizedHypothesisSeed(rerandomizedSeed()),
        ],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      game.view(),
      { scoreLogLikelihood: () => 0 },
    );
    const result = predictor.observe(game.play("e2", "e4"));
    const stochastic = result.white.hypotheses.find(
      ({ drawbackId }) => drawbackId === "rerandomized-test",
    );
    expect(probability(stochastic?.logProbability ?? 0)).toBeCloseTo(0.2);
    expect(stochastic).toMatchObject({
      eliminated: false,
      parameters: {},
      internalState: { movesApplied: 1 },
    });
    expect(JSON.stringify(stochastic)).not.toContain("seed");
  });

  it("hard-eliminates only when every exhaustive outcome is incompatible", () => {
    const game = new PredictorTestGame();
    const impossible = {
      ...rerandomizedSeed([0.5, 0.5]),
      filterLegalMoves: (
        _context: unknown,
        _outcome: TestOutcome,
        moves: readonly ChessMove[],
      ) => moves.filter((move) => move.to === "d4"),
    };
    const predictor = new SymbolicPredictor(
      {
        white: [
          asRerandomizedHypothesisSeed(impossible),
          asHypothesisSeed(unrestrictedRule, {}),
        ],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      game.view(),
      { scoreLogLikelihood: () => 0 },
    );
    const result = predictor.observe(game.play("e2", "e4"));
    expect(result.white.hypotheses[0]).toMatchObject({
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
  });

  it("survives repeated observations by collapsing ephemeral outcomes", () => {
    const game = new PredictorTestGame();
    const repeatedSeed = {
      ...rerandomizedSeed([0.5, 0.5]),
      filterLegalMoves: (
        _context: unknown,
        _outcome: TestOutcome,
        moves: readonly ChessMove[],
      ) => moves,
    };
    const predictor = new SymbolicPredictor(
      {
        white: [asRerandomizedHypothesisSeed(repeatedSeed)],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      game.view(),
      { scoreLogLikelihood: () => 0 },
    );
    for (let turn = 0; turn < 20; turn += 1) {
      const whiteFrom = turn % 2 === 0 ? "g1" : "f3";
      const whiteTo = turn % 2 === 0 ? "f3" : "g1";
      const blackFrom = turn % 2 === 0 ? "g8" : "f6";
      const blackTo = turn % 2 === 0 ? "f6" : "g8";
      const result = predictor.observe(game.play(whiteFrom, whiteTo));
      expect(result.white.hypotheses[0]?.eliminated).toBe(false);
      predictor.observe(game.play(blackFrom, blackTo));
    }
    expect(predictor.state.white.hypotheses[0]?.internalState).toEqual({
      movesApplied: 20,
    });
  });

  it("keeps the inactive color isolated", () => {
    const game = new PredictorTestGame();
    const seed = asRerandomizedHypothesisSeed(
      rerandomizedSeed([0.5, 0.5]),
    );
    const predictor = new SymbolicPredictor(
      { white: [seed], black: [seed] },
      game.view(),
      { scoreLogLikelihood: () => 0 },
    );
    const beforeBlack = predictor.state.black;
    predictor.observe(game.play("e2", "e4"));
    expect(predictor.state.black).toEqual(beforeBlack);
    expect(predictor.state.black.hypotheses[0]?.internalState).toEqual({
      movesApplied: 0,
    });
  });

  it("rejects malformed outcome distributions", () => {
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [
          asRerandomizedHypothesisSeed(rerandomizedSeed([0.4, 0.4])),
        ],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      game.view(),
    );
    expect(() => predictor.observe(game.play("e2", "e4"))).toThrow("sum to one");
  });
});
