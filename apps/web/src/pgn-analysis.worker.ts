import { analyzePgn, PgnParseError } from "./pgn-analysis.js";
import {
  loadAuthenticatedCompletedPgnEvaluatorSidecar,
} from "@drawbackengine/chess-evaluator/completed-pgn-sidecar";
import {
  MAX_MODEL_ARTIFACT_BYTES,
  isPgnAnalysisWorkerRequest,
  type PgnAnalysisWorkerResponse,
  type SerializedPgnAnalysisError,
} from "./pgn-analysis-worker-protocol.js";
import { parseBrowserNeuralModel } from "./neural-model.js";

interface AnalysisWorkerScope {
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  postMessage(message: PgnAnalysisWorkerResponse): void;
}

const scope = globalThis as unknown as AnalysisWorkerScope;

async function sha256Text(value: string): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function serializeError(error: unknown): SerializedPgnAnalysisError {
  if (error instanceof PgnParseError) {
    return {
      name: error.name,
      message: error.message,
      ply: error.ply,
      token: error.token,
    };
  }
  return {
    name: error instanceof Error ? error.name : "Error",
    message:
      error instanceof Error ? error.message : "Unknown PGN analysis error.",
    ply: null,
    token: null,
  };
}

export class PgnAnalysisWorkerRuntime {
  readonly #parseModel: typeof parseBrowserNeuralModel;
  readonly #analyze: typeof analyzePgn;
  readonly #authenticateEvaluator:
    typeof loadAuthenticatedCompletedPgnEvaluatorSidecar;
  #cachedModel:
    | {
        readonly artifactSha256: string;
        readonly model: ReturnType<typeof parseBrowserNeuralModel>;
      }
    | undefined;

  public constructor(
    parseModel: typeof parseBrowserNeuralModel = parseBrowserNeuralModel,
    analyze: typeof analyzePgn = analyzePgn,
    authenticateEvaluator:
      typeof loadAuthenticatedCompletedPgnEvaluatorSidecar =
        loadAuthenticatedCompletedPgnEvaluatorSidecar,
  ) {
    this.#parseModel = parseModel;
    this.#analyze = analyze;
    this.#authenticateEvaluator = authenticateEvaluator;
  }

  public async handle(
    request: unknown,
    postMessage: (message: PgnAnalysisWorkerResponse) => void,
  ): Promise<void> {
    if (!isPgnAnalysisWorkerRequest(request)) {
      return;
    }
    const { requestId } = request;
    try {
      if (request.type === "load-model") {
      const byteLength = new TextEncoder().encode(
        request.artifactText,
      ).byteLength;
      if (byteLength > MAX_MODEL_ARTIFACT_BYTES) {
        throw new RangeError(
          `Model artifact exceeds the ${String(MAX_MODEL_ARTIFACT_BYTES)} byte limit.`,
        );
      }
      const actualSha256 = await sha256Text(request.artifactText);
      if (actualSha256 !== request.expectedSha256) {
        throw new Error("Model artifact SHA-256 does not match.");
      }
      if (this.#cachedModel?.artifactSha256 !== actualSha256) {
        const parsedJson = JSON.parse(request.artifactText) as unknown;
        this.#cachedModel = {
          artifactSha256: actualSha256,
          model: this.#parseModel(parsedJson),
        };
      }
      postMessage({
        type: "model-loaded",
        requestId,
        model: {
          artifactSha256: this.#cachedModel.artifactSha256,
          modelFormatVersion: this.#cachedModel.model.formatVersion,
          modelVariant: this.#cachedModel.model.modelVariant,
          drawbackCount: this.#cachedModel.model.drawbackVocabulary.length,
        },
      });
      return;
    }
    const {
      pgn,
      neuralArtifactSha256,
      evaluatorSidecarBytes,
      evaluatorSidecarSha256,
    } = request;
    if (
      neuralArtifactSha256 !== undefined &&
      this.#cachedModel?.artifactSha256 !== neuralArtifactSha256
    ) {
      throw new Error(
        "Requested neural artifact is not loaded in this worker.",
      );
    }
    const selectedModel =
      neuralArtifactSha256 === undefined
        ? undefined
        : this.#cachedModel?.model;
    if (neuralArtifactSha256 !== undefined && selectedModel === undefined) {
      throw new Error(
        "Requested neural artifact is not loaded in this worker.",
      );
    }
    const evaluatorEvidence =
      evaluatorSidecarBytes === undefined ||
        evaluatorSidecarSha256 === undefined
        ? undefined
        : await this.#authenticateEvaluator(
          evaluatorSidecarBytes,
          pgn,
          evaluatorSidecarSha256,
        );
    const result = this.#analyze(pgn, {
      ...(selectedModel === undefined
        ? {}
        : {
            neuralModel: selectedModel,
            neuralArtifactSha256,
          }),
      ...(evaluatorEvidence === undefined ? {} : { evaluatorEvidence }),
      onProgress(progress) {
        postMessage({ type: "progress", requestId, progress });
      },
    });
    postMessage({ type: "result", requestId, result });
  } catch (error) {
    postMessage({
      type: "error",
      requestId,
      error: serializeError(error),
    });
  }
  }
}

const runtime = new PgnAnalysisWorkerRuntime();
scope.onmessage = (event): void => {
  void runtime.handle(event.data, (message) => {
    scope.postMessage(message);
  });
};
