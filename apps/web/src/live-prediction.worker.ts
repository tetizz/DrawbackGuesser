import {
  createPublicMoveObservation,
  SymbolicPredictor,
  type PredictionState,
} from "@drawbackguesser/predictor";
import { createHypothesisSeeds } from "./hypothesis-catalog.js";
import {
  isLivePredictionWorkerRequest,
  serializeLivePredictionError,
  type LivePredictionObservationInput,
  type LivePredictionWorkerResponse,
} from "./live-prediction-worker-protocol.js";

type PostResponse = (message: LivePredictionWorkerResponse) => void;

function createPredictor(
  initialPosition: LivePredictionObservationInput["positionBefore"],
): SymbolicPredictor {
  const seeds = createHypothesisSeeds();
  return new SymbolicPredictor(
    { white: seeds, black: seeds },
    initialPosition,
  );
}

function observe(
  predictor: SymbolicPredictor,
  input: LivePredictionObservationInput,
): PredictionState {
  return predictor.observe(createPublicMoveObservation(input));
}

export class LivePredictionWorkerRuntime {
  #sessionId: string | null = null;
  #revision = -1;
  #predictor: SymbolicPredictor | null = null;

  public handle(data: unknown, postMessage: PostResponse): void {
    if (!isLivePredictionWorkerRequest(data)) {
      return;
    }
    const { sessionId, revision } = data;
    try {
      if (data.type === "initialize") {
        this.#sessionId = sessionId;
        this.#revision = 0;
        this.#predictor = createPredictor(data.initialPosition);
        postMessage({
          type: "prediction",
          sessionId,
          revision: 0,
          prediction: this.#predictor.state,
        });
        return;
      }
      if (data.type === "reconstruct") {
        const predictor = createPredictor(data.initialPosition);
        let prediction = predictor.state;
        for (const input of data.observations) {
          prediction = observe(predictor, input);
        }
        this.#sessionId = sessionId;
        this.#revision = revision;
        this.#predictor = predictor;
        postMessage({
          type: "prediction",
          sessionId,
          revision,
          prediction,
        });
        return;
      }
      if (
        this.#predictor === null ||
        this.#sessionId !== sessionId ||
        revision !== this.#revision + 1
      ) {
        throw new Error(
          "Live prediction observation does not follow the active session revision.",
        );
      }
      const prediction = observe(this.#predictor, data.observation);
      this.#revision = revision;
      postMessage({
        type: "prediction",
        sessionId,
        revision,
        prediction,
      });
    } catch (error) {
      postMessage({
        type: "error",
        sessionId,
        revision,
        error: serializeLivePredictionError(error),
      });
    }
  }
}

if (typeof self !== "undefined" && "postMessage" in self) {
  const scope = self as unknown as {
    onmessage: ((event: MessageEvent<unknown>) => void) | null;
    postMessage(message: LivePredictionWorkerResponse): void;
  };
  const runtime = new LivePredictionWorkerRuntime();
  scope.onmessage = (event): void => {
    runtime.handle(event.data, (message) => {
      scope.postMessage(message);
    });
  };
}
