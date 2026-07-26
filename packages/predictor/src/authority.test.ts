import { describe, expect, it } from "vitest";
import type {
  ChessMove,
  DrawbackRule,
  PositionView,
} from "@drawbackengine/drawback-engine";
import {
  CapturableKingPosition,
} from "@drawbackengine/chess-core";
import {
  asHypothesisSeed,
  createPublicMoveObservation,
  SymbolicPredictor,
} from "./index.js";

interface State {
  readonly moves: number;
}

function rule(
  id: string,
  permits: (move: ChessMove) => boolean,
  capturable = true,
): DrawbackRule<State, Record<string, never>> {
  return {
    id,
    name: id,
    description: id,
    verification: "verified",
    ...(capturable
      ? { supportedAuthorities: ["capturable-king/v1"] as const }
      : {}),
    generateParameters: () => ({}),
    initialize: () => ({ moves: 0 }),
    filterLegalMoves: (_context, moves) => moves.filter(permits),
    applyMove: (context) => ({ moves: context.state.moves + 1 }),
    checkStartOfTurnLoss: () => null,
  };
}

const kingCapture: ChessMove = {
  from: "e7",
  to: "e8",
  color: "white",
  piece: "queen",
  captured: "king",
  san: "Qxe8",
  flags: "capture",
};

const before: PositionView = {
  fen: "4k3/4Q3/8/8/8/8/8/K7 w - - 0 1",
  turn: "white",
  ply: 0,
  history: [],
};

const after: PositionView = {
  fen: "4Q3/8/8/8/8/8/8/K7 b - - 0 1",
  turn: "black",
  ply: 1,
  history: [kingCapture],
};

const authorityBefore = CapturableKingPosition.fromFen(before.fen).snapshot();

describe("authority-aware public observations", () => {
  it("uses a capturable-king move as legal evidence without secret fields", () => {
    const allowsCapture = rule("allows-king-capture", () => true);
    const forbidsCapture = rule(
      "forbids-king-capture",
      (move) => move.captured !== "king",
    );
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(allowsCapture, {}),
          asHypothesisSeed(forbidsCapture, {}),
        ],
        black: [asHypothesisSeed(allowsCapture, {})],
      },
      before,
      {
        authorityId: "capturable-king/v1",
        initialAuthorityPosition: authorityBefore,
      },
    );
    const observation = createPublicMoveObservation({
      authorityId: "capturable-king/v1",
      authorityPositionBefore: authorityBefore,
      color: "white",
      positionBefore: before,
      positionAfter: after,
      move: kingCapture,
    });

    expect(Object.keys(observation).sort()).toEqual([
      "authorityId",
      "authorityLegalMoves",
      "authorityPositionBefore",
      "color",
      "move",
      "positionAfter",
      "positionBefore",
    ]);
    const result = predictor.observe(observation);
    expect(result.white.hypotheses).toEqual([
      expect.objectContaining({
        drawbackId: "allows-king-capture",
        eliminated: false,
      }),
      expect.objectContaining({
        drawbackId: "forbids-king-capture",
        eliminated: true,
      }),
    ]);
  });

  it("defensively clones and freezes public move data", () => {
    const observation = createPublicMoveObservation({
      authorityId: "capturable-king/v1",
      authorityPositionBefore: authorityBefore,
      color: "white",
      positionBefore: before,
      positionAfter: after,
      move: kingCapture,
    });
    expect(
      observation.authorityLegalMoves.some((move) => move.to === "e8"),
    ).toBe(true);
    expect(Object.isFrozen(observation)).toBe(true);
    expect(Object.isFrozen(observation.authorityLegalMoves)).toBe(true);
    expect(Object.isFrozen(observation.authorityLegalMoves[0])).toBe(true);
  });

  it("removes unsupported hypotheses before prior normalization", () => {
    const supported = rule("supported", () => true);
    const standardOnly = rule("standard-only", () => true, false);
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(supported, {}, 1),
          asHypothesisSeed(standardOnly, {}, 100),
        ],
        black: [asHypothesisSeed(supported, {})],
      },
      before,
      {
        authorityId: "capturable-king/v1",
        initialAuthorityPosition: authorityBefore,
      },
    );
    expect(predictor.state.white.hypotheses).toHaveLength(1);
    expect(predictor.state.white.hypotheses[0]).toMatchObject({
      drawbackId: "supported",
      logProbability: 0,
    });
  });

  it("rejects filters that manufacture non-authority moves", () => {
    const manufactured = rule("manufactured", () => true);
    manufactured.filterLegalMoves = () => [
      { ...kingCapture, from: "a1", to: "a8" },
    ];
    const predictor = new SymbolicPredictor(
      {
        white: [asHypothesisSeed(manufactured, {})],
        black: [asHypothesisSeed(rule("control", () => true), {})],
      },
      before,
      {
        authorityId: "capturable-king/v1",
        initialAuthorityPosition: authorityBefore,
      },
    );
    expect(() =>
      predictor.observe(createPublicMoveObservation({
        authorityId: "capturable-king/v1",
        authorityPositionBefore: authorityBefore,
        color: "white",
        positionBefore: before,
        positionAfter: after,
        move: kingCapture,
      }))
    ).toThrow("manufactured a move outside the authority set");
  });
});
