import { describe, expect, it } from "vitest";
import {
  coveringFireRule,
  alwaysCheckRule,
  checkersRule,
  monkeySeeRule,
  reconnaissanceRule,
  type ChessMove,
  type DrawbackRule,
  type PositionView,
  type ReconnaissanceState,
} from "@drawbackengine/drawback-engine";
import {
  asHypothesisSeed,
  entropy,
  probability,
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
    description: `Test rule ${id}`,
    verification: "verified",
    generateParameters: () => ({}),
    initialize: () => ({ count: 0 }),
    filterLegalMoves: (_context, moves) => moves.filter(allows),
    applyMove: (context) => ({ count: context.state.count + 1 }),
    checkStartOfTurnLoss: () => null,
  };
}

const unrestricted = counterRule("unrestricted-test", () => true);
const forbidsE4 = counterRule(
  "forbids-e4",
  (move) => move.from !== "e2" || move.to !== "e4",
);

function createPredictor(initial: PositionView): SymbolicPredictor {
  return new SymbolicPredictor(
    {
      white: [
        asHypothesisSeed(unrestricted, {}),
        asHypothesisSeed(forbidsE4, {}),
      ],
      black: [
        asHypothesisSeed(unrestricted, {}),
        asHypothesisSeed(forbidsE4, {}),
      ],
    },
    initial,
  );
}

describe("SymbolicPredictor", () => {
  it("hard-eliminates impossible hypotheses and never restores them", () => {
    const game = new PredictorTestGame();
    const predictor = createPredictor(game.view());
    const first = predictor.observe(game.play("e2", "e4"));

    const eliminated = first.white.hypotheses[1];
    expect(eliminated).toMatchObject({
      drawbackId: "forbids-e4",
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
    expect(probability(eliminated?.logProbability ?? 0)).toBe(0);
    expect(eliminated?.evidence).toHaveLength(1);
    expect(Object.isFrozen(eliminated?.evidence)).toBe(true);

    predictor.observe(game.play("e7", "e5"));
    const stillEliminated = predictor.state.white.hypotheses[1];
    expect(stillEliminated?.eliminated).toBe(true);
    expect(stillEliminated?.logProbability).toBe(Number.NEGATIVE_INFINITY);
  });

  it("updates White and Black independently", () => {
    const game = new PredictorTestGame();
    const predictor = createPredictor(game.view());
    const blackBefore = predictor.state.black;

    predictor.observe(game.play("e2", "e4"));

    expect(predictor.state.black).toEqual(blackBefore);
    expect(
      (predictor.state.white.hypotheses[0]?.internalState as CounterState).count,
    ).toBe(1);
    expect(
      (predictor.state.black.hypotheses[0]?.internalState as CounterState).count,
    ).toBe(0);
  });

  it("normalizes priors in log space and computes entropy", () => {
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(unrestricted, {}, 3),
          asHypothesisSeed(forbidsE4, {}, 1),
        ],
        black: [asHypothesisSeed(unrestricted, {})],
      },
      game.view(),
    );

    const [first, second] = predictor.state.white.hypotheses;
    expect(probability(first?.logProbability ?? 0)).toBeCloseTo(0.75);
    expect(probability(second?.logProbability ?? 0)).toBeCloseTo(0.25);
    expect(entropy(predictor.state.white)).toBeCloseTo(0.811278, 5);
  });

  it("rejects inconsistent observations without changing state", () => {
    const game = new PredictorTestGame();
    const predictor = createPredictor(game.view());
    const previous = predictor.state;
    const valid = game.play("e2", "e4");
    expect(() =>
      predictor.observe({
        ...valid,
        color: "black",
      }),
    ).toThrow("Observation color must match");
    expect(predictor.state).toEqual(previous);
  });

  it("rejects discontinuous games and fabricated move metadata", () => {
    const game = new PredictorTestGame();
    const predictor = createPredictor(game.view());
    const first = game.play("e2", "e4");
    predictor.observe(first);
    const stateAfterFirst = predictor.state;

    const foreignGame = new PredictorTestGame();
    const foreign = foreignGame.play("d2", "d4");
    expect(() => predictor.observe(foreign)).toThrow("discontinuous");
    expect(predictor.state).toEqual(stateAfterFirst);

    const freshGame = new PredictorTestGame();
    const freshPredictor = createPredictor(freshGame.view());
    const valid = freshGame.play("e2", "e4");
    expect(() => freshPredictor.observe({
      ...valid,
      move: {
        ...valid.move,
        captured: "king",
      },
    })).toThrow("metadata does not match");
  });

  it("regenerates the complete authority set before Checkers elimination", () => {
    const fen = "4k3/8/8/8/3n4/2P5/8/4K3 w - - 0 1";
    const game = new PredictorTestGame(fen);
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(checkersRule, {}),
          asHypothesisSeed(unrestricted, {}),
        ],
        black: [asHypothesisSeed(unrestricted, {})],
      },
      game.view(),
    );
    const observation = game.play("e1", "f1");
    const result = predictor.observe(observation);
    expect(result.white.hypotheses[0]).toMatchObject({
      drawbackId: "checkers",
      eliminated: true,
    });

    const incompleteGame = new PredictorTestGame(fen);
    const incompletePredictor = new SymbolicPredictor(
      {
        white: [asHypothesisSeed(checkersRule, {})],
        black: [asHypothesisSeed(unrestricted, {})],
      },
      incompleteGame.view(),
    );
    const complete = incompleteGame.play("e1", "f1");
    expect(() => incompletePredictor.observe({
      ...complete,
      authorityLegalMoves: complete.authorityLegalMoves.filter(
        (move) => !(move.from === "c3" && move.to === "d4"),
      ),
    })).toThrow("complete position authority set");
  });

  it("advances Reconnaissance study state only for the observed color", () => {
    const game = new PredictorTestGame(
      "4k3/8/8/3p4/3Q3n/8/8/4K3 w - - 0 1",
    );
    const predictor = new SymbolicPredictor(
      {
        white: [asHypothesisSeed(reconnaissanceRule, {})],
        black: [asHypothesisSeed(reconnaissanceRule, {})],
      },
      game.view(),
    );

    predictor.observe(game.play("d4", "d3"));

    expect(
      predictor.state.white.hypotheses[0]?.internalState,
    ).toEqual<ReconnaissanceState>({
      movesApplied: 1,
      unlockedCapturedTypes: ["pawn", "knight"],
    });
    expect(
      predictor.state.black.hypotheses[0]?.internalState,
    ).toEqual<ReconnaissanceState>({
      movesApplied: 0,
      unlockedCapturedTypes: [],
    });
  });

  it("does not attach elimination evidence to a surviving contextual rule", () => {
    const game = new PredictorTestGame(
      "4k3/8/8/8/3r4/2P1P3/8/4K3 w - - 0 1",
    );
    const predictor = new SymbolicPredictor(
      {
        white: [asHypothesisSeed(coveringFireRule, {})],
        black: [asHypothesisSeed(unrestricted, {})],
      },
      game.view(),
    );
    const result = predictor.observe(game.play("c3", "d4"));
    const hypothesis = result.white.hypotheses[0];
    expect(hypothesis?.eliminated).toBe(false);
    expect(
      hypothesis?.evidence.some(({ kind }) => kind === "eliminated"),
    ).toBe(false);
  });

  it("hard-eliminates a hypothesis whose start-of-turn loss precludes the move", () => {
    const game = new PredictorTestGame(
      "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
    );
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(alwaysCheckRule, {}),
          asHypothesisSeed(unrestricted, {}),
        ],
        black: [asHypothesisSeed(unrestricted, {})],
      },
      game.view(),
    );
    const result = predictor.observe(game.play("e1", "e2"));
    expect(result.white.hypotheses[0]).toMatchObject({
      drawbackId: "always-check-it-might-be-mate",
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
    expect(result.white.hypotheses[1]?.eliminated).toBe(false);
  });

  it("defers loss-hypothesis elimination until play actually continues", () => {
    const game = new PredictorTestGame(
      "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1",
    );
    const predictor = new SymbolicPredictor(
      {
        white: [asHypothesisSeed(unrestricted, {})],
        black: [
          asHypothesisSeed(alwaysCheckRule, {}),
          asHypothesisSeed(unrestricted, {}),
        ],
      },
      game.view(),
    );
    const gameCouldHaveEnded = predictor.observe(game.play("e2", "e7"));
    expect(gameCouldHaveEnded.black.hypotheses[0]?.eliminated).toBe(false);

    const continued = predictor.observe(game.play("e8", "d8"));
    expect(continued.black.hypotheses[0]).toMatchObject({
      drawbackId: "always-check-it-might-be-mate",
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
  });

  it("uses public move history to activate a rule, hard-eliminate it, and never revive it", () => {
    const game = new PredictorTestGame(
      "r3k3/8/7p/8/8/8/P7/2B1K3 w - - 0 1",
    );
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(monkeySeeRule, {}),
          asHypothesisSeed(unrestricted, {}),
        ],
        black: [asHypothesisSeed(unrestricted, {})],
      },
      game.view(),
    );

    const beforeActivation = predictor.observe(game.play("e1", "f1"));
    expect(beforeActivation.white.hypotheses[0]).toMatchObject({
      drawbackId: "monkey-see",
      eliminated: false,
    });

    const activated = predictor.observe(game.play("a8", "a2"));
    expect(activated.white.hypotheses[0]).toMatchObject({
      drawbackId: "monkey-see",
      eliminated: false,
    });

    const contradicted = predictor.observe(game.play("c1", "h6"));
    expect(contradicted.white.hypotheses[0]).toMatchObject({
      drawbackId: "monkey-see",
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
    expect(probability(
      contradicted.white.hypotheses[0]?.logProbability ?? 0,
    )).toBe(0);

    predictor.observe(game.play("e8", "d7"));
    expect(predictor.state.white.hypotheses[0]).toMatchObject({
      drawbackId: "monkey-see",
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
  });
});
