import { describe, expect, it } from "vitest";
import {
  advancePublicPositionAuthority,
  createStandardChessPositionSnapshot,
  publicAuthorityLegalMoves,
  type StandardChessPositionSnapshot,
} from "@drawbackengine/chess-core";
import type {
  ChessMove,
  PositionView,
} from "@drawbackengine/drawback-engine";
import {
  createPublicMoveObservation,
  SymbolicPredictor,
  type PredictionState,
} from "@drawbackguesser/predictor";
import { createHypothesisSeeds } from "./hypothesis-catalog.js";
import { LivePredictionWorkerRuntime } from "./live-prediction.worker.js";
import type {
  LivePredictionObservationInput,
  LivePredictionWorkerResponse,
} from "./live-prediction-worker-protocol.js";

const INITIAL_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const INITIAL_POSITION: PositionView = {
  fen: INITIAL_FEN,
  turn: "white",
  ply: 0,
  history: [],
};

function fixture(
  moves: readonly (readonly [from: string, to: string])[],
): readonly LivePredictionObservationInput[] {
  let authority: StandardChessPositionSnapshot =
    createStandardChessPositionSnapshot(INITIAL_FEN);
  let history: readonly ChessMove[] = [];
  return moves.map(([from, to]) => {
    const before: PositionView = {
      fen: authority.fen,
      turn: authority.fen.split(/\s+/u)[1] === "w" ? "white" : "black",
      ply: history.length,
      history,
    };
    const canonical = publicAuthorityLegalMoves(authority).find(
      (move) => move.from === from && move.to === to,
    );
    if (canonical === undefined) {
      throw new Error(`Fixture move ${from}${to} is not legal.`);
    }
    const transition = advancePublicPositionAuthority(authority, canonical);
    authority = transition.position as StandardChessPositionSnapshot;
    history = [...history, transition.move];
    return {
      authorityId: "standard-chess/v1",
      color: canonical.color,
      move: canonical,
      positionBefore: before,
      positionAfter: {
        fen: authority.fen,
        turn: authority.fen.split(/\s+/u)[1] === "w" ? "white" : "black",
        ply: history.length,
        history,
      },
    };
  });
}

function synchronousResult(
  observations: readonly LivePredictionObservationInput[],
): PredictionState {
  const seeds = createHypothesisSeeds();
  const predictor = new SymbolicPredictor(
    { white: seeds, black: seeds },
    INITIAL_POSITION,
  );
  let prediction = predictor.state;
  for (const observation of observations) {
    prediction = predictor.observe(
      createPublicMoveObservation(observation),
    );
  }
  return prediction;
}

describe("live prediction worker parity", () => {
  it.each([
    {
      name: "king pawn opening",
      moves: [
        ["e2", "e4"],
        ["e7", "e5"],
        ["g1", "f3"],
      ] as const,
    },
    {
      name: "queen pawn opening",
      moves: [
        ["d2", "d4"],
        ["d7", "d5"],
        ["c2", "c4"],
      ] as const,
    },
  ])("preserves exact synchronous results for $name", ({ moves }) => {
    const observations = fixture(moves);
    const responses: LivePredictionWorkerResponse[] = [];
    const runtime = new LivePredictionWorkerRuntime();
    runtime.handle(
      {
        type: "reconstruct",
        sessionId: "parity-game",
        revision: observations.length,
        initialPosition: INITIAL_POSITION,
        observations,
      },
      (message) => {
        responses.push(message);
      },
    );
    const response = responses[0];
    expect(response?.type).toBe("prediction");
    if (response?.type !== "prediction") {
      throw new Error("Expected worker prediction.");
    }
    expect(response.prediction).toEqual(synchronousResult(observations));
    expect(response.prediction.white).not.toBe(
      response.prediction.black,
    );
  });
});
