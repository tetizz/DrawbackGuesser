import { describe, expect, it } from "vitest";
import type {
  ChessMove,
  DrawbackRule,
} from "@drawbackengine/drawback-engine";
import {
  asHypothesisSeed,
  probability,
  resolveLikelihoodWeights,
  scoreMoveLogLikelihood,
  SymbolicPredictor,
} from "./index.js";
import { PredictorTestGame } from "./test-game.js";

interface CounterState {
  readonly count: number;
}

type NoParameters = Record<string, never>;

function counterRule(
  id: string,
  allows: (move: ChessMove) => boolean,
): DrawbackRule<CounterState, NoParameters> {
  return {
    id,
    name: id,
    description: id,
    verification: "verified",
    generateParameters: () => ({}),
    initialize: () => ({ count: 0 }),
    filterLegalMoves: (_context, moves) => moves.filter(allows),
    applyMove: (context) => ({ count: context.state.count + 1 }),
    checkStartOfTurnLoss: () => null,
  };
}

describe("scoreMoveLogLikelihood", () => {
  it("uses allowed count, forced status, and optional observed signals", () => {
    const score = scoreMoveLogLikelihood(
      {
        allowedMoveCount: 1,
        ordinaryLegalMoveCount: 20,
        triggered: true,
        forced: true,
        signals: {
          humanMoveLogLikelihood: -0.2,
          engineQualityLogLikelihood: -0.3,
          playerStrengthLogLikelihood: -0.4,
          timeUsageLogLikelihood: -0.5,
        },
      },
      resolveLikelihoodWeights({ forcedMove: 0.1 }),
    );

    expect(score).toBeCloseTo(-1.3);
  });

  it("gives no-trigger turns only the configured evidence fraction", () => {
    const features = {
      allowedMoveCount: 8,
      ordinaryLegalMoveCount: 8,
      forced: false,
      signals: { humanMoveLogLikelihood: -0.5 },
    } as const;
    const withoutSignal = scoreMoveLogLikelihood({
      ...features,
      triggered: false,
      signals: {},
    });
    const inactive = scoreMoveLogLikelihood({ ...features, triggered: false });
    expect(inactive - withoutSignal).toBeCloseTo(-0.5 * 0.05);
  });

  it("rejects invalid counts, signals, and weights", () => {
    expect(() =>
      scoreMoveLogLikelihood({
        allowedMoveCount: 2,
        ordinaryLegalMoveCount: 1,
        triggered: true,
        forced: false,
      }),
    ).toThrow(RangeError);
    expect(() =>
      scoreMoveLogLikelihood({
        allowedMoveCount: 1,
        ordinaryLegalMoveCount: 1,
        triggered: false,
        forced: true,
        signals: { timeUsageLogLikelihood: 0.1 },
      }),
    ).toThrow(RangeError);
    expect(() => resolveLikelihoodWeights({ noTriggerEvidenceScale: 1.1 })).toThrow(
      RangeError,
    );
  });
});

describe("SymbolicPredictor likelihood integration", () => {
  const unrestricted = counterRule("unrestricted-likelihood", () => true);
  const forbidsE4 = counterRule(
    "forbids-e4-likelihood",
    (move) => move.from !== "e2" || move.to !== "e4",
  );

  it("updates surviving hypotheses with optional Bayesian likelihoods", () => {
    const onlyE4 = counterRule(
      "only-e4",
      (move) => move.from === "e2" && move.to === "e4",
    );
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(unrestricted, {}),
          asHypothesisSeed(onlyE4, {}),
        ],
        black: [asHypothesisSeed(unrestricted, {})],
      },
      game.view(),
      { likelihoodWeights: { forcedMove: 0.25 } },
    );

    const state = predictor.observe(game.play("e2", "e4"));

    const broad = state.white.hypotheses[0];
    const forced = state.white.hypotheses[1];
    expect(forced?.logProbability).toBeGreaterThan(broad?.logProbability ?? 0);
    expect(forced?.evidence.at(-1)).toMatchObject({
      kind: "likelihood",
      ruleId: "only-e4",
    });
  });

  it("uses historical frequencies as priors", () => {
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [
          {
            ...asHypothesisSeed(unrestricted, {}),
            historicalFrequency: 9,
          },
          {
            ...asHypothesisSeed(forbidsE4, {}),
            historicalFrequency: 1,
          },
        ],
        black: [asHypothesisSeed(unrestricted, {})],
      },
      game.view(),
    );

    expect(probability(predictor.state.white.hypotheses[0]?.logProbability ?? 0))
      .toBeCloseTo(0.9);
    expect(probability(predictor.state.white.hypotheses[1]?.logProbability ?? 0))
      .toBeCloseTo(0.1);
  });

  it("never scores or restores a hard-eliminated hypothesis", () => {
    let scored = 0;
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(unrestricted, {}),
          asHypothesisSeed(forbidsE4, {}),
        ],
        black: [asHypothesisSeed(unrestricted, {})],
      },
      game.view(),
      {
        scoreLogLikelihood: () => {
          scored += 1;
          return 0;
        },
      },
    );

    predictor.observe(game.play("e2", "e4"));
    expect(scored).toBe(1);
    expect(predictor.state.white.hypotheses[1]).toMatchObject({
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
  });
});
