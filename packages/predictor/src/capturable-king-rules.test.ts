import { describe, expect, it } from "vitest";
import {
  capturableKingIrresistibleRule,
  femmeFataleRule,
  nurturerRule,
  triplePlayRule,
  unrestrictedRule,
  youBestNotMissRule,
  type ChessMove,
  type PositionView,
} from "@drawbackengine/drawback-engine";
import {
  CapturableKingPosition,
  type MoveCommand,
} from "@drawbackengine/chess-core";
import {
  asHypothesisSeed,
  createPublicMoveObservation,
  SymbolicPredictor,
  type PredictionSeed,
  type PredictionState,
} from "./index.js";

function view(
  position: CapturableKingPosition,
  history: readonly ChessMove[],
): PositionView {
  return {
    fen: position.fen,
    turn: position.turn,
    ply: history.length,
    history: [...history],
  };
}

function predictorFor(
  position: CapturableKingPosition,
  white: readonly PredictionSeed[],
  black: readonly PredictionSeed[] = [
    asHypothesisSeed(unrestrictedRule, {}),
  ],
): SymbolicPredictor {
  return new SymbolicPredictor(
    { white, black },
    view(position, []),
    {
      authorityId: "capturable-king/v1",
      initialAuthorityPosition: position.snapshot(),
    },
  );
}

function observe(
  predictor: SymbolicPredictor,
  position: CapturableKingPosition,
  history: ChessMove[],
  command: MoveCommand,
): PredictionState {
  const authorityPositionBefore = position.snapshot();
  const positionBefore = view(position, history);
  const outcome = position.move(command);
  if (outcome === null) {
    throw new Error(
      `Predictor test command ${command.from}${command.to} is not authority legal.`,
    );
  }
  history.push(outcome.move);
  return predictor.observe(createPublicMoveObservation({
    authorityId: "capturable-king/v1",
    authorityPositionBefore,
    color: outcome.move.color,
    positionBefore,
    positionAfter: view(position, history),
    move: outcome.move,
  }));
}

function byId(
  state: PredictionState,
  color: "white" | "black",
  id: string,
) {
  return state[color].hypotheses.find(
    (hypothesis) => hypothesis.drawbackId === id,
  );
}

describe("capturable-king rule prediction", () => {
  it("hard-eliminates Irresistible after a quiet move declines adjacency", () => {
    const position = CapturableKingPosition.fromFen(
      "4k3/8/8/2N5/8/8/P7/4K3 w - - 0 1",
    );
    const predictor = predictorFor(position, [
      asHypothesisSeed(capturableKingIrresistibleRule, {}),
      asHypothesisSeed(unrestrictedRule, {}),
    ]);
    const state = observe(
      predictor,
      position,
      [],
      { from: "a2", to: "a3" },
    );
    expect(byId(state, "white", "irresistible")).toMatchObject({
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
    expect(byId(state, "white", "unrestricted")).toMatchObject({
      eliminated: false,
      logProbability: 0,
    });
  });

  it("preserves Irresistible for a non-adjacent king-passant capture", () => {
    const position = CapturableKingPosition.fromFen(
      "4q2k/8/8/8/8/3n4/8/4K2R w K - 0 1",
    );
    const predictor = predictorFor(
      position,
      [asHypothesisSeed(unrestrictedRule, {})],
      [
        asHypothesisSeed(capturableKingIrresistibleRule, {}),
        asHypothesisSeed(unrestrictedRule, {}),
      ],
    );
    const history: ChessMove[] = [];
    observe(
      predictor,
      position,
      history,
      { from: "e1", to: "g1" },
    );
    const state = observe(
      predictor,
      position,
      history,
      { from: "e8", to: "e1" },
    );
    expect(byId(state, "black", "irresistible")).toMatchObject({
      eliminated: false,
      internalState: { movesApplied: 1 },
    });
  });

  it("hard-eliminates Femme Fatale after a non-queen king capture", () => {
    const position = CapturableKingPosition.fromFen(
      "4k3/4R3/8/8/8/8/8/K7 w - - 0 1",
    );
    const predictor = predictorFor(position, [
      asHypothesisSeed(femmeFataleRule, {}),
      asHypothesisSeed(unrestrictedRule, {}),
    ]);
    const state = observe(
      predictor,
      position,
      [],
      { from: "e7", to: "e8" },
    );
    expect(byId(state, "white", "femme-fatale")).toMatchObject({
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
    expect(byId(state, "white", "unrestricted")).toMatchObject({
      eliminated: false,
      logProbability: 0,
    });
  });

  it("advances Nurturer promotion state from public moves without secret labels", () => {
    const position = CapturableKingPosition.fromFen(
      "4k3/Pp2R3/8/8/8/8/8/K7 w - - 0 1",
    );
    const predictor = predictorFor(position, [
      asHypothesisSeed(nurturerRule, {}),
      asHypothesisSeed(unrestrictedRule, {}),
    ]);
    const history: ChessMove[] = [];
    let state = observe(
      predictor,
      position,
      history,
      { from: "a7", to: "a8", promotion: "queen" },
    );
    expect(byId(state, "white", "nurturer")).toMatchObject({
      eliminated: false,
      internalState: {
        movesApplied: 1,
        hasPromotedPawn: true,
      },
    });
    state = observe(
      predictor,
      position,
      history,
      { from: "b7", to: "b6" },
    );
    state = observe(
      predictor,
      position,
      history,
      { from: "e7", to: "e8" },
    );
    expect(byId(state, "white", "nurturer")).toMatchObject({
      eliminated: false,
      internalState: {
        movesApplied: 2,
        hasPromotedPawn: true,
      },
    });
  });

  it("separates Triple Play parameter particles by exact material legality", () => {
    const position = CapturableKingPosition.fromFen(
      "4k3/4Q3/8/8/8/8/BBB5/K7 w - - 0 1",
    );
    const predictor = predictorFor(position, [
      asHypothesisSeed(triplePlayRule, { requiredType: "bishop" }),
      asHypothesisSeed(triplePlayRule, { requiredType: "knight" }),
    ]);
    const state = observe(
      predictor,
      position,
      [],
      { from: "e7", to: "e8" },
    );
    const particles = state.white.hypotheses.filter(
      (hypothesis) => hypothesis.drawbackId === "triple-play",
    );
    expect(particles).toEqual([
      expect.objectContaining({
        parameters: { requiredType: "bishop" },
        eliminated: false,
        logProbability: 0,
      }),
      expect.objectContaining({
        parameters: { requiredType: "knight" },
        eliminated: true,
        logProbability: Number.NEGATIVE_INFINITY,
      }),
    ]);
  });

  it("eliminates You Best Not Miss when an observed next move declines king capture", () => {
    const position = CapturableKingPosition.fromFen(
      "7k/1p6/8/8/8/8/8/R3K3 w - - 0 1",
    );
    const predictor = predictorFor(position, [
      asHypothesisSeed(youBestNotMissRule, {}),
      asHypothesisSeed(unrestrictedRule, {}),
    ]);
    const history: ChessMove[] = [];
    let state = observe(
      predictor,
      position,
      history,
      { from: "a1", to: "a8" },
    );
    expect(byId(state, "white", "you-best-not-miss")).toMatchObject({
      internalState: {
        movesApplied: 1,
        mustCaptureKingNextTurn: true,
      },
      eliminated: false,
    });
    state = observe(
      predictor,
      position,
      history,
      { from: "b7", to: "b6" },
    );
    expect(byId(state, "white", "you-best-not-miss")).toMatchObject({
      internalState: {
        movesApplied: 1,
        mustCaptureKingNextTurn: true,
      },
      eliminated: false,
    });
    state = observe(
      predictor,
      position,
      history,
      { from: "a8", to: "a7" },
    );
    expect(byId(state, "white", "you-best-not-miss")).toMatchObject({
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
    expect(byId(state, "white", "unrestricted")).toMatchObject({
      eliminated: false,
      logProbability: 0,
    });
  });

  it("keeps White and Black delayed-obligation hypotheses isolated", () => {
    const position = CapturableKingPosition.fromFen(
      "7k/1p6/8/8/8/8/8/R3K3 w - - 0 1",
    );
    const seed = asHypothesisSeed(youBestNotMissRule, {});
    const predictor = predictorFor(position, [seed], [seed]);
    const history: ChessMove[] = [];
    let state = observe(
      predictor,
      position,
      history,
      { from: "a1", to: "a8" },
    );
    expect(byId(state, "white", "you-best-not-miss")?.internalState).toEqual({
      movesApplied: 1,
      mustCaptureKingNextTurn: true,
    });
    expect(byId(state, "black", "you-best-not-miss")?.internalState).toEqual({
      movesApplied: 0,
      mustCaptureKingNextTurn: false,
    });
    state = observe(
      predictor,
      position,
      history,
      { from: "b7", to: "b6" },
    );
    expect(byId(state, "white", "you-best-not-miss")?.internalState).toEqual({
      movesApplied: 1,
      mustCaptureKingNextTurn: true,
    });
    expect(byId(state, "black", "you-best-not-miss")?.internalState).toEqual({
      movesApplied: 1,
      mustCaptureKingNextTurn: false,
    });
  });
});
