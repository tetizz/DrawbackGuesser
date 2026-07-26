import { describe, expect, it } from "vitest";
import {
  isLivePredictionWorkerRequest,
  isLivePredictionWorkerResponse,
  type LivePredictionObservationInput,
  type LivePredictionWorkerRequest,
} from "./live-prediction-worker-protocol.js";

const move = {
  from: "e2",
  to: "e4",
  color: "white",
  piece: "pawn",
  san: "e4",
  flags: "quiet",
} as const;

const observation: LivePredictionObservationInput = {
  authorityId: "standard-chess/v1",
  color: "white",
  move,
  positionBefore: {
    fen: "before",
    turn: "white",
    ply: 0,
    history: [],
  },
  positionAfter: {
    fen: "after",
    turn: "black",
    ply: 1,
    history: [move],
  },
};

function allKeys(value: unknown): readonly string[] {
  if (Array.isArray(value)) {
    return value.flatMap(allKeys);
  }
  if (typeof value !== "object" || value === null) {
    return [];
  }
  return [
    ...Object.keys(value),
    ...Object.values(value).flatMap(allKeys),
  ];
}

describe("live prediction worker protocol", () => {
  it("accepts the exact public observation payload without secret fields", () => {
    const request = {
      type: "observe",
      sessionId: "game-1",
      revision: 1,
      observation,
    } satisfies LivePredictionWorkerRequest;

    expect(isLivePredictionWorkerRequest(request)).toBe(true);
    expect(allKeys(request)).not.toEqual(
      expect.arrayContaining([
        "drawbackId",
        "parameters",
        "internalState",
        "whiteRule",
        "blackRule",
        "secret",
        "ruleTriggered",
        "drawbackLegalMoves",
      ]),
    );
  });

  it("rejects secret or extra fields and inconsistent revisions", () => {
    expect(
      isLivePredictionWorkerRequest({
        type: "observe",
        sessionId: "game-1",
        revision: 1,
        observation: {
          ...observation,
          secret: { drawbackId: "hidden" },
        },
      }),
    ).toBe(false);
    expect(
      isLivePredictionWorkerRequest({
        type: "observe",
        sessionId: "game-1",
        revision: 2,
        observation,
      }),
    ).toBe(false);
  });

  it("validates prediction and error responses", () => {
    const prediction = {
      white: { hypotheses: [] },
      black: { hypotheses: [] },
    };
    expect(
      isLivePredictionWorkerResponse({
        type: "prediction",
        sessionId: "game-1",
        revision: 0,
        prediction,
      }),
    ).toBe(true);
    expect(
      isLivePredictionWorkerResponse({
        type: "error",
        sessionId: "game-1",
        revision: 1,
        error: { name: "Error", message: "failed" },
      }),
    ).toBe(true);
    expect(
      isLivePredictionWorkerResponse({
        type: "prediction",
        sessionId: "game-1",
        revision: 0,
        prediction: { white: prediction.white },
      }),
    ).toBe(false);
  });
});
