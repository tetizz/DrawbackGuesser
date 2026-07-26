import { describe, expect, it } from "vitest";
import type {
  DrawbackRule,
} from "@drawbackengine/drawback-engine";
import { asHypothesisSeed, SymbolicPredictor } from "./index.js";
import { PredictorTestGame } from "./test-game.js";

interface State {
  readonly observed: number;
}

type Parameters = Record<string, never>;

function rule(id: string, forbiddenFrom: string | null): DrawbackRule<State, Parameters> {
  return {
    id,
    name: id,
    description: id,
    verification: "verified",
    generateParameters: () => ({}),
    initialize: () => ({ observed: 0 }),
    filterLegalMoves: (_context, moves) =>
      moves.filter((move) => move.from !== forbiddenFrom),
    applyMove: (context) => ({ observed: context.state.observed + 1 }),
    checkStartOfTurnLoss: () => null,
  };
}

describe("predictor bounded properties", () => {
  it("keeps hard-eliminated hypotheses at zero across later observations", () => {
    const open = rule("open", null);
    const forbidden = rule("forbidden-e2", "e2");
    const game = new PredictorTestGame();
    const initial = game.view();
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(open, {}),
          asHypothesisSeed(forbidden, {}),
        ],
        black: [
          asHypothesisSeed(open, {}),
          asHypothesisSeed(forbidden, {}),
        ],
      },
      initial,
    );
    predictor.observe(game.play("e2", "e4"));
    for (const hypothesis of predictor.state.white.hypotheses.filter(
      (candidate) => candidate.eliminated,
    )) {
      expect(hypothesis.logProbability).toBe(Number.NEGATIVE_INFINITY);
    }

    predictor.observe(game.play("e7", "e5"));
    predictor.observe(game.play("d2", "d4"));

    const eliminated = predictor.state.white.hypotheses.find(
      (candidate) => candidate.drawbackId === "forbidden-e2",
    );
    expect(eliminated?.eliminated).toBe(true);
    expect(eliminated?.logProbability).toBe(Number.NEGATIVE_INFINITY);
  });

  it("updates only the observed player's hypothesis states", () => {
    const open = rule("open", null);
    const game = new PredictorTestGame();
    const initial = game.view();
    const predictor = new SymbolicPredictor(
      {
        white: [asHypothesisSeed(open, {})],
        black: [asHypothesisSeed(open, {})],
      },
      initial,
    );
    const blackBefore = predictor.state.black;
    predictor.observe(game.play("e2", "e4"));

    expect(predictor.state.black).toEqual(blackBefore);
    expect(
      (predictor.state.white.hypotheses[0]?.internalState as State).observed,
    ).toBe(1);
    expect(
      (predictor.state.black.hypotheses[0]?.internalState as State).observed,
    ).toBe(0);
  });
});
