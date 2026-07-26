import { describe, expect, it } from "vitest";
import type { PositionView } from "@drawbackengine/drawback-engine";
import type { PredictionState } from "@drawbackguesser/predictor";
import {
  LivePredictionController,
  type LivePredictionUpdate,
  type LivePredictionWorker,
} from "./live-prediction-controller.js";
import type {
  LivePredictionObservationInput,
  LivePredictionWorkerRequest,
} from "./live-prediction-worker-protocol.js";

const initialPosition: PositionView = {
  fen: "initial",
  turn: "white",
  ply: 0,
  history: [],
};

const firstMove = {
  from: "e2",
  to: "e4",
  color: "white",
  piece: "pawn",
  san: "e4",
  flags: "quiet",
} as const;

const secondMove = {
  from: "e7",
  to: "e5",
  color: "black",
  piece: "pawn",
  san: "e5",
  flags: "quiet",
} as const;

const firstObservation: LivePredictionObservationInput = {
  authorityId: "standard-chess/v1",
  color: "white",
  move: firstMove,
  positionBefore: initialPosition,
  positionAfter: {
    fen: "after-e4",
    turn: "black",
    ply: 1,
    history: [firstMove],
  },
};

const secondObservation: LivePredictionObservationInput = {
  authorityId: "standard-chess/v1",
  color: "black",
  move: secondMove,
  positionBefore: firstObservation.positionAfter,
  positionAfter: {
    fen: "after-e5",
    turn: "white",
    ply: 2,
    history: [firstMove, secondMove],
  },
};

const prediction: PredictionState = {
  white: { hypotheses: [] },
  black: { hypotheses: [] },
};

class FakeWorker implements LivePredictionWorker {
  public onmessage: ((event: MessageEvent<unknown>) => void) | null = null;
  public onerror: ((event: ErrorEvent) => void) | null = null;
  public readonly requests: LivePredictionWorkerRequest[] = [];
  public terminated = false;

  public postMessage(message: LivePredictionWorkerRequest): void {
    this.requests.push(structuredClone(message));
  }

  public terminate(): void {
    this.terminated = true;
  }

  public emit(data: unknown): void {
    this.onmessage?.(new MessageEvent("message", { data }));
  }

  public fail(): void {
    this.onerror?.({} as ErrorEvent);
  }
}

function setup(): {
  readonly controller: LivePredictionController;
  readonly workers: FakeWorker[];
  readonly updates: LivePredictionUpdate[];
  readonly errors: string[];
} {
  const workers: FakeWorker[] = [];
  const updates: LivePredictionUpdate[] = [];
  const errors: string[] = [];
  return {
    workers,
    updates,
    errors,
    controller: new LivePredictionController(
      () => {
        const worker = new FakeWorker();
        workers.push(worker);
        return worker;
      },
      {
        onPrediction(update) {
          updates.push(update);
        },
        onError(message) {
          errors.push(message);
        },
      },
    ),
  };
}

describe("live prediction controller", () => {
  it("orders correlated responses and marks only the newest revision latest", () => {
    const { controller, workers, updates } = setup();
    controller.startSession("game-1", initialPosition);
    controller.observe("game-1", 1, firstObservation);
    controller.observe("game-1", 2, secondObservation);
    const worker = workers[0];
    if (worker === undefined) {
      throw new Error("Expected worker.");
    }
    worker.emit({
      type: "prediction",
      sessionId: "game-1",
      revision: 2,
      prediction,
    });
    worker.emit({
      type: "prediction",
      sessionId: "game-1",
      revision: 0,
      prediction,
    });
    expect(updates.map(({ revision }) => revision)).toEqual([0]);
    worker.emit({
      type: "prediction",
      sessionId: "game-1",
      revision: 1,
      prediction,
    });

    expect(updates.map(({ revision }) => revision)).toEqual([0, 1, 2]);
    expect(updates.map(({ isLatest }) => isLatest)).toEqual([
      false,
      false,
      true,
    ]);
  });

  it("discards stale responses after reset", () => {
    const { controller, workers, updates } = setup();
    controller.startSession("game-1", initialPosition);
    const oldWorker = workers[0];
    const staleHandler = oldWorker?.onmessage;
    controller.startSession("game-2", initialPosition);
    if (oldWorker === undefined || staleHandler === null || staleHandler === undefined) {
      throw new Error("Expected old worker handler.");
    }
    expect(oldWorker.terminated).toBe(true);
    staleHandler(
      new MessageEvent("message", {
        data: {
          type: "prediction",
          sessionId: "game-1",
          revision: 0,
          prediction,
        },
      }),
    );
    expect(updates).toEqual([]);
    workers[1]?.emit({
      type: "prediction",
      sessionId: "game-2",
      revision: 0,
      prediction,
    });
    expect(updates.map(({ sessionId }) => sessionId)).toEqual(["game-2"]);
  });

  it("suspends replay work, discards its stale response, and reconstructs", () => {
    const { controller, workers, updates } = setup();
    controller.startSession("game-1", initialPosition);
    const staleWorker = workers[0];
    const staleHandler = staleWorker?.onmessage;
    controller.suspend();
    if (staleWorker === undefined || staleHandler === null || staleHandler === undefined) {
      throw new Error("Expected replay worker handler.");
    }
    staleHandler(
      new MessageEvent("message", {
        data: {
          type: "prediction",
          sessionId: "game-1",
          revision: 0,
          prediction,
        },
      }),
    );
    expect(updates).toEqual([]);

    controller.reconstruct(
      "game-1",
      initialPosition,
      [firstObservation, secondObservation],
    );
    expect(workers[1]?.requests).toEqual([
      {
        type: "reconstruct",
        sessionId: "game-1",
        revision: 2,
        initialPosition,
        observations: [firstObservation, secondObservation],
      },
    ]);
    workers[1]?.emit({
      type: "prediction",
      sessionId: "game-1",
      revision: 2,
      prediction,
    });
    expect(updates.map(({ revision }) => revision)).toEqual([2]);
  });

  it("reports worker protocol errors and runtime failures", () => {
    const malformed = setup();
    malformed.controller.startSession("game-1", initialPosition);
    malformed.workers[0]?.emit({});
    expect(malformed.errors).toEqual([
      "Live prediction worker returned invalid data.",
    ]);
    expect(malformed.workers[0]?.terminated).toBe(true);

    const explicit = setup();
    explicit.controller.startSession("game-2", initialPosition);
    explicit.workers[0]?.emit({
      type: "error",
      sessionId: "game-2",
      revision: 0,
      error: { name: "Error", message: "predictor exploded" },
    });
    expect(explicit.errors).toEqual(["predictor exploded"]);

    const crashed = setup();
    crashed.controller.startSession("game-3", initialPosition);
    crashed.workers[0]?.fail();
    expect(crashed.errors).toEqual(["Live prediction worker failed."]);
  });

  it("refuses out-of-order posting and disposes idempotently", () => {
    const { controller, workers } = setup();
    controller.startSession("game-1", initialPosition);
    expect(() => {
      controller.observe("game-1", 2, secondObservation);
    }).toThrow("in order");
    controller.dispose();
    controller.dispose();
    expect(workers[0]?.terminated).toBe(true);
    expect(() => {
      controller.startSession("game-2", initialPosition);
    }).toThrow("disposed");
  });
});
