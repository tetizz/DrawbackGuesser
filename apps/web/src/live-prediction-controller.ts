import type { PositionView } from "@drawbackengine/drawback-engine";
import type { PredictionState } from "@drawbackguesser/predictor";
import {
  isLivePredictionWorkerRequest,
  isLivePredictionWorkerResponse,
  type LivePredictionObservationInput,
  type LivePredictionWorkerRequest,
} from "./live-prediction-worker-protocol.js";

export interface LivePredictionWorker {
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage(message: LivePredictionWorkerRequest): void;
  terminate(): void;
}

export interface LivePredictionUpdate {
  readonly sessionId: string;
  readonly revision: number;
  readonly prediction: PredictionState;
  readonly isLatest: boolean;
}

interface LivePredictionControllerCallbacks {
  readonly onPrediction: (update: LivePredictionUpdate) => void;
  readonly onError: (message: string) => void;
}

export class LivePredictionController {
  readonly #workerFactory: () => LivePredictionWorker;
  readonly #callbacks: LivePredictionControllerCallbacks;
  #worker: LivePredictionWorker | null = null;
  #generation = 0;
  #sessionId: string | null = null;
  #highestRequestedRevision = -1;
  #nextDeliveryRevision = -1;
  #buffer = new Map<number, PredictionState>();
  #disposed = false;

  public constructor(
    workerFactory: () => LivePredictionWorker,
    callbacks: LivePredictionControllerCallbacks,
  ) {
    this.#workerFactory = workerFactory;
    this.#callbacks = callbacks;
  }

  public startSession(
    sessionId: string,
    initialPosition: PositionView,
  ): void {
    this.#assertAvailable();
    this.#replaceWorker(sessionId, 0);
    this.#post({
      type: "initialize",
      sessionId,
      revision: 0,
      initialPosition,
    });
  }

  public observe(
    sessionId: string,
    revision: number,
    observation: LivePredictionObservationInput,
  ): void {
    this.#assertAvailable();
    if (this.#worker === null || this.#sessionId !== sessionId) {
      throw new Error("Live prediction session is not active.");
    }
    if (revision !== this.#highestRequestedRevision + 1) {
      throw new Error("Live prediction revisions must be posted in order.");
    }
    this.#highestRequestedRevision = revision;
    this.#post({
      type: "observe",
      sessionId,
      revision,
      observation,
    });
  }

  public reconstruct(
    sessionId: string,
    initialPosition: PositionView,
    observations: readonly LivePredictionObservationInput[],
  ): void {
    this.#assertAvailable();
    const revision = observations.length;
    this.#replaceWorker(sessionId, revision);
    this.#post({
      type: "reconstruct",
      sessionId,
      revision,
      initialPosition,
      observations,
    });
  }

  public suspend(): void {
    if (this.#disposed) {
      return;
    }
    this.#generation += 1;
    this.#worker?.terminate();
    this.#worker = null;
    this.#sessionId = null;
    this.#highestRequestedRevision = -1;
    this.#nextDeliveryRevision = -1;
    this.#buffer.clear();
  }

  public dispose(): void {
    if (this.#disposed) {
      return;
    }
    this.suspend();
    this.#disposed = true;
  }

  #replaceWorker(sessionId: string, firstDeliveryRevision: number): void {
    this.suspend();
    this.#sessionId = sessionId;
    this.#highestRequestedRevision = firstDeliveryRevision;
    this.#nextDeliveryRevision = firstDeliveryRevision;
    const generation = this.#generation;
    const worker = this.#workerFactory();
    this.#worker = worker;
    worker.onmessage = (event): void => {
      if (
        generation !== this.#generation ||
        worker !== this.#worker
      ) {
        return;
      }
      this.#handleMessage(event.data);
    };
    worker.onerror = (): void => {
      if (
        generation !== this.#generation ||
        worker !== this.#worker
      ) {
        return;
      }
      this.#fail("Live prediction worker failed.");
    };
  }

  #post(message: LivePredictionWorkerRequest): void {
    if (!isLivePredictionWorkerRequest(message)) {
      throw new TypeError("Refusing to post invalid live prediction data.");
    }
    this.#worker?.postMessage(message);
  }

  #handleMessage(data: unknown): void {
    if (!isLivePredictionWorkerResponse(data)) {
      this.#fail("Live prediction worker returned invalid data.");
      return;
    }
    if (data.sessionId !== this.#sessionId) {
      return;
    }
    if (data.type === "error") {
      this.#fail(data.error.message);
      return;
    }
    if (
      data.revision < this.#nextDeliveryRevision ||
      data.revision > this.#highestRequestedRevision
    ) {
      return;
    }
    this.#buffer.set(data.revision, data.prediction);
    this.#drain();
  }

  #drain(): void {
    for (;;) {
      const prediction = this.#buffer.get(this.#nextDeliveryRevision);
      if (prediction === undefined || this.#sessionId === null) {
        return;
      }
      const revision = this.#nextDeliveryRevision;
      this.#buffer.delete(revision);
      this.#nextDeliveryRevision += 1;
      this.#callbacks.onPrediction({
        sessionId: this.#sessionId,
        revision,
        prediction,
        isLatest: revision === this.#highestRequestedRevision,
      });
    }
  }

  #fail(message: string): void {
    this.suspend();
    this.#callbacks.onError(message);
  }

  #assertAvailable(): void {
    if (this.#disposed) {
      throw new Error("Live prediction controller has been disposed.");
    }
  }
}
