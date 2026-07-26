import { describe, expect, it } from "vitest";
import {
  PgnAnalysisCancelledError,
  PgnAnalysisController,
  type PgnAnalysisWorker,
} from "./pgn-analysis-controller.js";
import { analyzePgn, MAX_PGN_INPUT_BYTES } from "./pgn-analysis.js";
import type {
  PgnAnalysisWorkerRequest,
  PgnAnalysisWorkerResponse,
} from "./pgn-analysis-worker-protocol.js";
import { MAX_EVALUATOR_SIDECAR_BYTES } from "./pgn-analysis-worker-protocol.js";

class FakeWorker implements PgnAnalysisWorker {
  public onmessage: ((event: MessageEvent<unknown>) => void) | null = null;
  public onerror: ((event: ErrorEvent) => void) | null = null;
  public readonly requests: PgnAnalysisWorkerRequest[] = [];
  public terminated = false;

  public postMessage(message: PgnAnalysisWorkerRequest): void {
    this.requests.push(message);
  }

  public terminate(): void {
    this.terminated = true;
  }

  public emit(message: unknown): void {
    this.onmessage?.(new MessageEvent("message", { data: message }));
  }

  public fail(): void {
    this.onerror?.({} as ErrorEvent);
  }
}

function workerFactory(workers: FakeWorker[]): () => FakeWorker {
  return () => {
    const worker = new FakeWorker();
    workers.push(worker);
    return worker;
  };
}

describe("asynchronous PGN analysis controller", () => {
  it("delivers legal SetUp results beginning after fullmove 300", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const pgn =
      '[SetUp "1"]\n[FEN "8/8/8/8/8/8/6k1/K6R b - - 0 301"]\n' +
      '[Result "1/2-1/2"]\n\n301... Kf3 302. Kb1 1/2-1/2';
    const pending = controller.analyze(pgn);
    const worker = workers[0];
    const request = worker?.requests[0];
    if (worker === undefined || request?.type !== "analyze") {
      throw new Error("Expected analysis worker request.");
    }
    worker.emit({
      type: "result",
      requestId: request.requestId,
      result: analyzePgn(pgn),
    });

    await expect(pending).resolves.toMatchObject({
      history: [
        { moveNumber: 301 },
        { moveNumber: 302 },
      ],
    });
  });

  it("validates evaluator sidecar inputs before creating a worker", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const bytes = new Uint8Array([1, 2, 3]);
    const digest = "a".repeat(64);

    await expect(controller.analyze("1. e4", {
      evaluatorSidecarBytes: bytes,
    })).rejects.toThrow("supplied together");
    await expect(controller.analyze("1. e4", {
      evaluatorSidecarSha256: digest,
    })).rejects.toThrow("supplied together");
    await expect(controller.analyze("1. e4", {
      evaluatorSidecarBytes: [1, 2, 3] as unknown as Uint8Array,
      evaluatorSidecarSha256: digest,
    })).rejects.toThrow("Uint8Array");
    await expect(controller.analyze("1. e4", {
      evaluatorSidecarBytes:
        new Uint8Array(MAX_EVALUATOR_SIDECAR_BYTES + 1),
      evaluatorSidecarSha256: digest,
    })).rejects.toThrow("byte limit");
    await expect(controller.analyze("1. e4", {
      evaluatorSidecarBytes: bytes,
      evaluatorSidecarSha256: "A".repeat(64),
    })).rejects.toThrow("lowercase digest");
    expect(workers).toHaveLength(0);
  });

  it("snapshots evaluator bytes before posting them to the worker", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const source = new Uint8Array([1, 2, 3]);
    const pending = controller.analyze("1. e4", {
      evaluatorSidecarBytes: source,
      evaluatorSidecarSha256: "a".repeat(64),
    });
    const worker = workers[0];
    const request = worker?.requests[0];
    if (worker === undefined || request?.type !== "analyze") {
      throw new Error("Expected evaluator analysis worker request.");
    }
    source[0] = 99;
    expect(request.evaluatorSidecarBytes).toEqual(
      new Uint8Array([1, 2, 3]),
    );
    expect(request.evaluatorSidecarBytes).not.toBe(source);

    worker.emit({
      type: "result",
      requestId: request.requestId,
      result: analyzePgn('[Result "1-0"]\n\n1. e4 1-0'),
    });
    await pending;
  });

  it("delivers progress and keeps the worker alive for reuse", async () => {
    const workers: FakeWorker[] = [];
    const progress: number[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const pending = controller.analyze("1. e4", {
      onProgress(update) {
        progress.push(update.processedPlies);
      },
    });
    const worker = workers[0];
    const request = worker?.requests[0];
    if (worker === undefined || request === undefined) {
      throw new Error("Expected one analysis worker request.");
    }
    worker.emit({
      type: "progress",
      requestId: request.requestId,
      progress: { processedPlies: 0, totalPlies: 1 },
    });
    const result = analyzePgn('[Result "1-0"]\n\n1. e4 1-0');
    worker.emit({ type: "result", requestId: request.requestId, result });

    await expect(pending).resolves.toEqual(result);
    expect(progress).toEqual([0]);
    expect(worker.terminated).toBe(false);
  });

  it("loads model text once and analyzes by cached digest only", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const artifactText = '{"format":"drawbacktrainer-browser-model"}\n';
    const digest = "a".repeat(64);
    const loading = controller.loadModel(artifactText, digest);
    const worker = workers[0];
    const loadRequest = worker?.requests[0];
    if (
      worker === undefined ||
      loadRequest === undefined ||
      loadRequest.type !== "load-model"
    ) {
      throw new Error("Expected model load request.");
    }
    expect(loadRequest.artifactText).toBe(artifactText);
    worker.emit({
      type: "model-loaded",
      requestId: loadRequest.requestId,
      model: {
        artifactSha256: digest,
        modelFormatVersion: 2,
        modelVariant: "v21-hybrid",
        drawbackCount: 182,
      },
    });
    await expect(loading).resolves.toMatchObject({
      artifactSha256: digest,
    });

    const pending = controller.analyze("1. e4", {
      neuralArtifactSha256: "a".repeat(64),
    });
    const request = worker.requests[1];
    if (request === undefined || request.type !== "analyze") {
      throw new Error("Expected model analysis worker request.");
    }
    expect(request.neuralArtifactSha256).toBe("a".repeat(64));
    expect("artifactText" in request).toBe(false);
    const result = analyzePgn('[Result "1-0"]\n\n1. e4 1-0');
    worker.emit({ type: "result", requestId: request.requestId, result });
    await expect(pending).resolves.toEqual(result);
    expect(workers).toHaveLength(1);
  });

  it("accepts matching v22 model metadata from the worker", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const digest = "c".repeat(64);
    const loading = controller.loadModel("{}", digest);
    const worker = workers[0];
    const request = worker?.requests[0];
    if (
      worker === undefined ||
      request === undefined ||
      request.type !== "load-model"
    ) {
      throw new Error("Expected v22 model load request.");
    }
    worker.emit({
      type: "model-loaded",
      requestId: request.requestId,
      model: {
        artifactSha256: digest,
        modelFormatVersion: 3,
        modelVariant: "v22-hybrid",
        drawbackCount: 182,
      },
    });

    await expect(loading).resolves.toEqual({
      artifactSha256: digest,
      modelFormatVersion: 3,
      modelVariant: "v22-hybrid",
      drawbackCount: 182,
    });
  });

  it("invalidates the digest when its worker is lost", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const digest = "a".repeat(64);
    const loading = controller.loadModel("{}", digest);
    const worker = workers[0];
    const request = worker?.requests[0];
    if (
      worker === undefined ||
      request === undefined ||
      request.type !== "load-model"
    ) {
      throw new Error("Expected model load request.");
    }
    worker.emit({
      type: "model-loaded",
      requestId: request.requestId,
      model: {
        artifactSha256: digest,
        modelFormatVersion: 4,
        modelVariant: "v21-hybrid-ensemble",
        drawbackCount: 182,
      },
    });
    await loading;
    worker.fail();

    await expect(
      controller.analyze("1. e4", {
        neuralArtifactSha256: digest,
      }),
    ).rejects.toThrow("load the model again");
    expect(workers).toHaveLength(1);
  });

  it("cancels by terminating and recreates for a superseding request", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const first = controller.analyze("1. e4");
    const firstRejection = expect(first).rejects.toBeInstanceOf(
      PgnAnalysisCancelledError,
    );
    const second = controller.analyze("1. d4");

    expect(workers).toHaveLength(2);
    expect(workers[0]?.terminated).toBe(true);
    await firstRejection;
    const secondWorker = workers[1];
    const secondRequest = secondWorker?.requests[0];
    if (secondWorker === undefined || secondRequest === undefined) {
      throw new Error("Expected replacement worker request.");
    }
    const result = analyzePgn('[Result "1-0"]\n\n1. d4 1-0');
    secondWorker.emit({
      type: "result",
      requestId: secondRequest.requestId,
      result,
    });
    await expect(second).resolves.toEqual(result);
  });

  it("ignores stale messages from a terminated worker generation", async () => {
    const workers: FakeWorker[] = [];
    const progress: number[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const first = controller.analyze("1. e4");
    const staleWorker = workers[0];
    const staleHandler = staleWorker?.onmessage;
    const firstRejection = expect(first).rejects.toBeInstanceOf(
      PgnAnalysisCancelledError,
    );
    const second = controller.analyze("1. d4", {
      onProgress(update) {
        progress.push(update.processedPlies);
      },
    });
    await firstRejection;
    const staleRequest = staleWorker?.requests[0];
    if (staleWorker === undefined || staleHandler === null || staleHandler === undefined || staleRequest === undefined) {
      throw new Error("Expected stale worker handler and request.");
    }
    staleHandler(
      new MessageEvent("message", {
        data: {
          type: "result",
          requestId: staleRequest.requestId,
          result: analyzePgn('[Result "1-0"]\n\n1. e4 1-0'),
        } satisfies PgnAnalysisWorkerResponse,
      }),
    );
    expect(progress).toEqual([]);

    const currentWorker = workers[1];
    const currentRequest = currentWorker?.requests[0];
    if (currentWorker === undefined || currentRequest === undefined) {
      throw new Error("Expected current worker request.");
    }
    const result = analyzePgn('[Result "1-0"]\n\n1. d4 1-0');
    currentWorker.emit({
      type: "result",
      requestId: currentRequest.requestId,
      result,
    });
    await expect(second).resolves.toEqual(result);
  });

  it("reconstructs parse errors and rejects worker failures", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const invalid = controller.analyze("invalid");
    const invalidWorker = workers[0];
    const invalidRequest = invalidWorker?.requests[0];
    if (invalidWorker === undefined || invalidRequest === undefined) {
      throw new Error("Expected invalid analysis request.");
    }
    invalidWorker.emit({
      type: "error",
      requestId: invalidRequest.requestId,
      error: {
        name: "PgnParseError",
        message: "bad move",
        ply: 1,
        token: "invalid",
      },
    });
    await expect(invalid).rejects.toMatchObject({
      name: "PgnParseError",
      ply: 1,
      token: "invalid",
    });

    const failed = controller.analyze("1. e4");
    workers[0]?.fail();
    await expect(failed).rejects.toThrow("worker failed");
  });

  it("rejects a malformed result from the active worker", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const pending = controller.analyze("1. e4");
    const worker = workers[0];
    const request = worker?.requests[0];
    if (worker === undefined || request === undefined) {
      throw new Error("Expected active worker request.");
    }
    worker.emit({
      type: "result",
      requestId: request.requestId,
      result: {},
    });
    await expect(pending).rejects.toThrow("invalid data");
    expect(worker.terminated).toBe(true);
  });

  it("disposes idempotently and refuses future work", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    const pending = controller.analyze("1. e4");
    const rejection = expect(pending).rejects.toBeInstanceOf(
      PgnAnalysisCancelledError,
    );
    controller.dispose();
    controller.dispose();

    await rejection;
    expect(workers[0]?.terminated).toBe(true);
    await expect(controller.analyze("1. d4")).rejects.toThrow("disposed");
    expect(workers).toHaveLength(1);
  });

  it("rejects oversized text before creating or posting to a worker", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    await expect(
      controller.analyze("x".repeat(MAX_PGN_INPUT_BYTES + 1)),
    ).rejects.toThrow("byte analysis limit");
    expect(workers).toHaveLength(0);
  });

  it("rejects oversized model text and a mismatched worker digest", async () => {
    const workers: FakeWorker[] = [];
    const controller = new PgnAnalysisController(workerFactory(workers));
    await expect(
      controller.loadModel("x".repeat(32 * 1024 * 1024 + 1), "a".repeat(64)),
    ).rejects.toThrow("byte limit");
    expect(workers).toHaveLength(0);

    const loading = controller.loadModel("{}", "a".repeat(64));
    const worker = workers[0];
    const request = worker?.requests[0];
    if (
      worker === undefined ||
      request === undefined ||
      request.type !== "load-model"
    ) {
      throw new Error("Expected model load request.");
    }
    worker.emit({
      type: "model-loaded",
      requestId: request.requestId,
      model: {
        artifactSha256: "b".repeat(64),
        modelFormatVersion: 4,
        modelVariant: "v21-hybrid-ensemble",
        drawbackCount: 182,
      },
    });
    await expect(loading).rejects.toThrow("wrong model artifact");
    expect(worker.terminated).toBe(true);
  });
});
