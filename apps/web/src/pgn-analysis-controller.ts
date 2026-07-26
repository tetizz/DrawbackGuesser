import {
  MAX_PGN_INPUT_BYTES,
  PgnParseError,
  type PgnAnalysisProgress,
  type PgnAnalysisResult,
} from "./pgn-analysis.js";
import {
  MAX_EVALUATOR_SIDECAR_BYTES,
  MAX_MODEL_ARTIFACT_BYTES,
  isPgnAnalysisWorkerResponse,
  type LoadedPgnAnalysisModel,
  type PgnAnalysisWorkerRequest,
} from "./pgn-analysis-worker-protocol.js";

export interface PgnAnalysisWorker {
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage(message: PgnAnalysisWorkerRequest): void;
  terminate(): void;
}

export type PgnAnalysisWorkerFactory = () => PgnAnalysisWorker;

export interface AsyncPgnAnalysisOptions {
  readonly onProgress?: (progress: PgnAnalysisProgress) => void;
  readonly neuralArtifactSha256?: string;
  readonly evaluatorSidecarBytes?: Uint8Array;
  readonly evaluatorSidecarSha256?: string;
}

export class PgnAnalysisCancelledError extends Error {
  public constructor(message = "PGN analysis was cancelled.") {
    super(message);
    this.name = "PgnAnalysisCancelledError";
  }
}

interface ActiveAnalysis {
  readonly kind: "analysis";
  readonly requestId: number;
  readonly resolve: (result: PgnAnalysisResult) => void;
  readonly reject: (error: unknown) => void;
  readonly onProgress: ((progress: PgnAnalysisProgress) => void) | undefined;
  readonly pgn: string;
  readonly neuralArtifactSha256: string | undefined;
  readonly evaluatorSidecarBytes: Uint8Array | undefined;
  readonly evaluatorSidecarSha256: string | undefined;
}

interface ActiveModelLoad {
  readonly kind: "model-load";
  readonly requestId: number;
  readonly artifactText: string;
  readonly expectedSha256: string;
  readonly resolve: (model: LoadedPgnAnalysisModel) => void;
  readonly reject: (error: unknown) => void;
}

function defaultWorkerFactory(): PgnAnalysisWorker {
  return new Worker(
    new URL("./pgn-analysis.worker.ts", import.meta.url),
    { type: "module", name: "drawback-pgn-analysis" },
  );
}

export class PgnAnalysisController {
  readonly #factory: PgnAnalysisWorkerFactory;
  #worker: PgnAnalysisWorker | null = null;
  #active: ActiveAnalysis | ActiveModelLoad | null = null;
  #loadedWorkerDigest: string | null = null;
  #selectedModel: LoadedPgnAnalysisModel | null = null;
  #nextRequestId = 1;
  #disposed = false;

  public constructor(factory: PgnAnalysisWorkerFactory = defaultWorkerFactory) {
    this.#factory = factory;
  }

  public loadModel(
    artifactText: string,
    expectedSha256: string,
  ): Promise<LoadedPgnAnalysisModel> {
    if (this.#disposed) {
      return Promise.reject(
        new Error("PGN analysis controller has been disposed."),
      );
    }
    if (!/^[0-9a-f]{64}$/u.test(expectedSha256)) {
      return Promise.reject(
        new TypeError("Model artifact SHA-256 must be a lowercase digest."),
      );
    }
    if (
      new TextEncoder().encode(artifactText).byteLength >
      MAX_MODEL_ARTIFACT_BYTES
    ) {
      return Promise.reject(
        new RangeError(
          `Model artifact exceeds the ${String(MAX_MODEL_ARTIFACT_BYTES)} byte limit.`,
        ),
      );
    }
    this.cancel("Model loading superseded the active analysis.");
    const worker = this.#worker ?? this.#createWorker();
    const requestId = this.#nextRequestId++;
    return new Promise<LoadedPgnAnalysisModel>((resolve, reject) => {
      this.#active = {
        kind: "model-load",
        requestId,
        artifactText,
        expectedSha256,
        resolve,
        reject,
      };
      worker.postMessage({
        type: "load-model",
        requestId,
        artifactText,
        expectedSha256,
      });
    });
  }

  public clearModel(): void {
    this.cancel("Loaded model was cleared.");
    this.#selectedModel = null;
    this.#terminateWorker();
  }

  public analyze(
    pgn: string,
    options: AsyncPgnAnalysisOptions = {},
  ): Promise<PgnAnalysisResult> {
    if (this.#disposed) {
      return Promise.reject(
        new Error("PGN analysis controller has been disposed."),
      );
    }
    if (new TextEncoder().encode(pgn).byteLength > MAX_PGN_INPUT_BYTES) {
      return Promise.reject(
        new PgnParseError(
          `PGN exceeds the ${String(MAX_PGN_INPUT_BYTES)} byte analysis limit.`,
          0,
          null,
        ),
      );
    }
    if (
      options.neuralArtifactSha256 !== undefined &&
      !/^[0-9a-f]{64}$/u.test(options.neuralArtifactSha256)
    ) {
      return Promise.reject(
        new TypeError("Neural artifact SHA-256 must be a lowercase digest."),
      );
    }
    const hasEvaluatorBytes = options.evaluatorSidecarBytes !== undefined;
    const hasEvaluatorDigest = options.evaluatorSidecarSha256 !== undefined;
    if (hasEvaluatorBytes !== hasEvaluatorDigest) {
      return Promise.reject(
        new TypeError(
          "Evaluator sidecar bytes and SHA-256 digest must be supplied together.",
        ),
      );
    }
    if (
      options.evaluatorSidecarBytes !== undefined &&
      !(options.evaluatorSidecarBytes instanceof Uint8Array)
    ) {
      return Promise.reject(
        new TypeError("Evaluator sidecar bytes must be a Uint8Array."),
      );
    }
    if (
      options.evaluatorSidecarBytes !== undefined &&
      options.evaluatorSidecarBytes.byteLength > MAX_EVALUATOR_SIDECAR_BYTES
    ) {
      return Promise.reject(
        new RangeError(
          `Evaluator sidecar exceeds the ${String(MAX_EVALUATOR_SIDECAR_BYTES)} byte limit.`,
        ),
      );
    }
    if (
      options.evaluatorSidecarSha256 !== undefined &&
      !/^[0-9a-f]{64}$/u.test(options.evaluatorSidecarSha256)
    ) {
      return Promise.reject(
        new TypeError(
          "Evaluator sidecar SHA-256 must be a lowercase digest.",
        ),
      );
    }
    if (
      options.neuralArtifactSha256 !== undefined
    ) {
      this.cancel("PGN analysis was superseded by a newer request.");
      if (
        this.#selectedModel?.artifactSha256 !==
          options.neuralArtifactSha256 ||
        this.#loadedWorkerDigest !== options.neuralArtifactSha256
      ) {
        return Promise.reject(
          new Error(
            "Requested neural artifact is not loaded; load the model again.",
          ),
        );
      }
    } else {
      this.cancel("PGN analysis was superseded by a newer request.");
    }
    const worker = this.#worker ?? this.#createWorker();
    const requestId = this.#nextRequestId;
    this.#nextRequestId += 1;
    return new Promise<PgnAnalysisResult>((resolve, reject) => {
      this.#active = {
        kind: "analysis",
        requestId,
        resolve,
        reject,
        onProgress: options.onProgress,
        pgn,
        neuralArtifactSha256: options.neuralArtifactSha256,
        evaluatorSidecarBytes: options.evaluatorSidecarBytes === undefined
          ? undefined
          : new Uint8Array(options.evaluatorSidecarBytes),
        evaluatorSidecarSha256: options.evaluatorSidecarSha256,
      };
      this.#postAnalysis(worker, this.#active);
    });
  }

  public cancel(message?: string): boolean {
    const active = this.#active;
    if (active === null) {
      return false;
    }
    this.#active = null;
    this.#terminateWorker();
    active.reject(new PgnAnalysisCancelledError(message));
    return true;
  }

  public dispose(): void {
    if (this.#disposed) {
      return;
    }
    this.cancel("PGN analysis controller was disposed.");
    this.#terminateWorker();
    this.#disposed = true;
  }

  #createWorker(): PgnAnalysisWorker {
    if (this.#worker !== null) {
      return this.#worker;
    }
    const worker = this.#factory();
    worker.onmessage = (event): void => {
      this.#handleMessage(worker, event.data);
    };
    worker.onerror = (event): void => {
      if (worker !== this.#worker) {
        return;
      }
      const active = this.#active;
      this.#active = null;
      this.#terminateWorker();
      const workerMessage =
        typeof event.message === "string" ? event.message.trim() : "";
      active?.reject(
        new Error(
          workerMessage.length === 0
            ? "PGN analysis worker failed."
            : `PGN analysis worker failed: ${workerMessage}`,
        ),
      );
    };
    this.#worker = worker;
    return worker;
  }

  #handleMessage(worker: PgnAnalysisWorker, value: unknown): void {
    if (worker !== this.#worker) {
      return;
    }
    const active = this.#active;
    if (!isPgnAnalysisWorkerResponse(value)) {
      if (
        active !== null &&
        typeof value === "object" &&
        value !== null &&
        "requestId" in value &&
        value.requestId === active.requestId
      ) {
        this.#active = null;
        this.#terminateWorker();
        active.reject(new Error("PGN analysis worker returned invalid data."));
      }
      return;
    }
    if (active === null || value.requestId !== active.requestId) {
      return;
    }
    if (value.type === "model-loaded") {
      if (active.kind === "model-load") {
        if (
          value.model.artifactSha256 !== active.expectedSha256
        ) {
          this.#active = null;
          this.#terminateWorker();
          active.reject(new Error("Worker loaded the wrong model artifact."));
          return;
        }
        this.#selectedModel = value.model;
        this.#loadedWorkerDigest = value.model.artifactSha256;
        this.#active = null;
        active.resolve(value.model);
        return;
      }
      this.#active = null;
      this.#terminateWorker();
      active.reject(new Error("Worker loaded an unexpected model artifact."));
      return;
    }
    if (value.type === "error") {
      this.#active = null;
      active.reject(
        value.error.name === "PgnParseError" && value.error.ply !== null
          ? new PgnParseError(
              value.error.message,
              value.error.ply,
              value.error.token,
            )
          : new Error(value.error.message),
      );
      return;
    }
    if (active.kind !== "analysis") {
      this.#active = null;
      this.#terminateWorker();
      active.reject(new Error("PGN analysis worker returned invalid data."));
      return;
    }
    if (value.type === "progress") {
      try {
        active.onProgress?.(value.progress);
      } catch (error) {
        this.#active = null;
        this.#terminateWorker();
        active.reject(error);
      }
      return;
    }
    this.#active = null;
    active.resolve(value.result);
  }

  #terminateWorker(): void {
    const worker = this.#worker;
    this.#worker = null;
    this.#loadedWorkerDigest = null;
    this.#selectedModel = null;
    if (worker !== null) {
      worker.onmessage = null;
      worker.onerror = null;
      worker.terminate();
    }
  }

  #postAnalysis(worker: PgnAnalysisWorker, active: ActiveAnalysis): void {
    worker.postMessage({
      type: "analyze",
      requestId: active.requestId,
      pgn: active.pgn,
      ...(active.neuralArtifactSha256 === undefined
        ? {}
        : { neuralArtifactSha256: active.neuralArtifactSha256 }),
      ...(active.evaluatorSidecarBytes === undefined ||
          active.evaluatorSidecarSha256 === undefined
        ? {}
        : {
          evaluatorSidecarBytes: active.evaluatorSidecarBytes,
          evaluatorSidecarSha256: active.evaluatorSidecarSha256,
        }),
    });
  }
}
