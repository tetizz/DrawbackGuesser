import { describe, expect, it } from "vitest";
import {
  handAndGigabrainRule,
  ichtyophobeRule,
  type ChessMove,
  type ExternalTurnConstraint,
  type PositionView,
} from "@drawbackengine/drawback-engine";
import {
  asExternalConstraintHypothesisSeed,
  SymbolicPredictor,
} from "./index.js";
import { PredictorTestGame } from "./test-game.js";

const INITIAL_FEN = "4k3/8/8/8/8/8/4P3/4K1N1 w - - 0 1";

function constraint(
  before: PositionView,
  ordinary: readonly ChessMove[],
): ExternalTurnConstraint {
  const request = ichtyophobeRule.requestTurnConstraint(
    {
      color: "white",
      parameters: {},
      state: { movesApplied: 0 },
      position: before,
    },
    ordinary,
  );
  return Object.freeze({
    provider: request.provider,
    policyId: request.policyId,
    positionKey: request.positionKey,
    bestMoveUci: "e2e4",
    requestDigest: "ab".repeat(32),
    engineFingerprint: "stockfish-public-test",
  });
}

function predictor(): SymbolicPredictor {
  const initial = new PredictorTestGame(INITIAL_FEN).view();
  const hypotheses = [
    asExternalConstraintHypothesisSeed(ichtyophobeRule, {}),
    asExternalConstraintHypothesisSeed(handAndGigabrainRule, {}),
  ];
  return new SymbolicPredictor(
    { white: hypotheses, black: hypotheses },
    initial,
  );
}

describe("external-constraint symbolic hypotheses", () => {
  it("lets exact public legality override likelihood scoring", () => {
    const game = new PredictorTestGame(INITIAL_FEN);
    const before = game.view();
    const result = predictor().observe(game.play("e2", "e4", {
      externalConstraint: constraint(before, game.legalMoves()),
    }));

    expect(result.white.hypotheses[0]).toMatchObject({
      drawbackId: "ichtyophobe",
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
    expect(result.white.hypotheses[0]?.evidence[0]?.kind).toBe("eliminated");
    expect(result.white.hypotheses[1]).toMatchObject({
      drawbackId: "hand-and-gigabrain",
      eliminated: false,
      logProbability: 0,
    });
  });

  it("treats an absent public constraint as unknown instead of eliminating", () => {
    const game = new PredictorTestGame(INITIAL_FEN);
    const result = predictor().observe(game.play("e2", "e4"));

    expect(result.white.hypotheses).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          drawbackId: "ichtyophobe",
          eliminated: false,
          evidence: [],
        }),
        expect.objectContaining({
          drawbackId: "hand-and-gigabrain",
          eliminated: false,
          evidence: [],
        }),
      ]),
    );
    expect(result.white.hypotheses.map(({ internalState }) => internalState))
      .toEqual([{ movesApplied: 1 }, { movesApplied: 1 }]);
  });

  it("updates only the observed player's external hypotheses", () => {
    const game = new PredictorTestGame(INITIAL_FEN);
    const prediction = predictor();
    const blackBefore = prediction.state.black;
    const result = prediction.observe(game.play("e2", "e4"));

    expect(result.white.hypotheses.map(({ internalState }) => internalState))
      .toEqual([{ movesApplied: 1 }, { movesApplied: 1 }]);
    expect(result.black).toEqual(blackBefore);
    expect(result.black.hypotheses.map(({ internalState }) => internalState))
      .toEqual([{ movesApplied: 0 }, { movesApplied: 0 }]);
  });
});
