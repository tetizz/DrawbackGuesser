import { describe, expect, it } from "vitest";
import type {
  AuthenticatedCompletedPgnEvaluatorSidecar,
} from "@drawbackengine/chess-evaluator/completed-pgn-sidecar";
import type { BrowserNeuralModel } from "./neural-model.js";
import {
  analyzePgn,
  type PgnAnalysisOptions,
  type PgnAnalysisResult,
} from "./pgn-analysis.js";
import {
  PgnAnalysisWorkerRuntime,
} from "./pgn-analysis.worker.js";
import type { PgnAnalysisWorkerResponse } from "./pgn-analysis-worker-protocol.js";

async function sha256Text(value: string): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function fakeModel(): BrowserNeuralModel {
  return {
    format: "drawbacktrainer-browser-model",
    formatVersion: 4,
    modelVariant: "v21-hybrid-ensemble",
    drawbackVocabulary: ["vegan"],
  } as unknown as BrowserNeuralModel;
}

function fakeV22Model(): BrowserNeuralModel {
  return {
    format: "drawbacktrainer-browser-model",
    formatVersion: 3,
    modelVariant: "v22-hybrid",
    drawbackVocabulary: ["vegan"],
  } as unknown as BrowserNeuralModel;
}

describe("PGN analysis worker model cache", () => {
  it("authenticates evaluator bytes before invoking the analyzer", async () => {
    const order: string[] = [];
    const authenticated = { artifactSha256: "a".repeat(64) };
    const analyze = (
      pgn: string,
      options: PgnAnalysisOptions = {},
    ): PgnAnalysisResult => {
      order.push("analyze");
      expect(pgn).toContain("e4");
      expect(options).toMatchObject({ evaluatorEvidence: authenticated });
      return analyzePgn('[Result "1-0"]\n\n1. e4 1-0');
    };
    const authenticate = (
      bytes: Uint8Array,
      pgn: string,
      digest: string,
    ): Promise<AuthenticatedCompletedPgnEvaluatorSidecar> => {
      order.push("authenticate");
      expect(bytes).toEqual(new Uint8Array([1, 2, 3]));
      expect(pgn).toContain("e4");
      expect(digest).toBe("b".repeat(64));
      return Promise.resolve(
        authenticated as unknown as AuthenticatedCompletedPgnEvaluatorSidecar,
      );
    };
    const runtime = new PgnAnalysisWorkerRuntime(
      undefined,
      analyze,
      authenticate,
    );
    const responses: PgnAnalysisWorkerResponse[] = [];

    await runtime.handle({
      type: "analyze",
      requestId: 7,
      pgn: '[Result "1-0"]\n\n1. e4 1-0',
      evaluatorSidecarBytes: new Uint8Array([1, 2, 3]),
      evaluatorSidecarSha256: "b".repeat(64),
    }, (response) => responses.push(response));

    expect(order).toEqual(["authenticate", "analyze"]);
    const response = responses.at(-1);
    if (response?.type === "error") {
      throw new Error(response.error.message);
    }
    expect(response?.type).toBe("result");
  });

  it("propagates evaluator authentication failure without analyzing", async () => {
    let analyzeCount = 0;
    const analyze = (pgn: string): PgnAnalysisResult => {
      analyzeCount += 1;
      return analyzePgn(pgn);
    };
    const rejectAuthentication =
      (): Promise<AuthenticatedCompletedPgnEvaluatorSidecar> =>
        Promise.reject(new Error("Evaluator sidecar SHA-256 does not match."));
    const runtime = new PgnAnalysisWorkerRuntime(
      undefined,
      analyze,
      rejectAuthentication,
    );
    const responses: PgnAnalysisWorkerResponse[] = [];

    await runtime.handle({
      type: "analyze",
      requestId: 8,
      pgn: '[Result "1-0"]\n\n1. e4 1-0',
      evaluatorSidecarBytes: new Uint8Array([1, 2, 3]),
      evaluatorSidecarSha256: "b".repeat(64),
    }, (response) => responses.push(response));

    expect(analyzeCount).toBe(0);
    expect(responses).toEqual([{
      type: "error",
      requestId: 8,
      error: {
        name: "Error",
        message: "Evaluator sidecar SHA-256 does not match.",
        ply: null,
        token: null,
      },
    }]);
  });

  it("rejects ongoing PGN analysis in the worker", async () => {
    const runtime = new PgnAnalysisWorkerRuntime();
    const responses: PgnAnalysisWorkerResponse[] = [];
    await runtime.handle({
      type: "analyze",
      requestId: 1,
      pgn: '[Result "*"]\n\n1. e4 e5 *',
    }, (response) => {
      responses.push(response);
    });
    expect(responses).toHaveLength(1);
    const response = responses[0];
    expect(response?.type).toBe("error");
    if (response?.type !== "error") {
      throw new Error("Expected the worker to reject the ongoing PGN.");
    }
    expect(response.error.name).toBe("PgnParseError");
    expect(response.error.message).toContain("matching terminal PGN");
  });

  it("preserves structured illegal-move errors across the worker boundary", async () => {
    const runtime = new PgnAnalysisWorkerRuntime();
    const responses: PgnAnalysisWorkerResponse[] = [];
    await runtime.handle({
      type: "analyze",
      requestId: 2,
      pgn: '[Result "1-0"]\n\n1. e5 1-0',
    }, (response) => {
      responses.push(response);
    });

    expect(responses).toHaveLength(1);
    const response = responses[0];
    expect(response?.type).toBe("error");
    if (response?.type !== "error") {
      throw new Error("Expected the worker to reject the illegal move.");
    }
    expect(response.error).toEqual({
      name: "PgnParseError",
      message: "Move 1 (e5) is not legal in the current position.",
      ply: 1,
      token: "e5",
    });
  });

  it("hashes and parses one artifact once, then analyzes by digest only", async () => {
    let parseCount = 0;
    const runtime = new PgnAnalysisWorkerRuntime(() => {
      parseCount += 1;
      return fakeModel();
    });
    const artifactText = "{}";
    const digest = await sha256Text(artifactText);
    const responses: PgnAnalysisWorkerResponse[] = [];
    const post = (response: PgnAnalysisWorkerResponse): void => {
      responses.push(response);
    };

    await runtime.handle({
      type: "load-model",
      requestId: 1,
      artifactText,
      expectedSha256: digest,
    }, post);
    await runtime.handle({
      type: "load-model",
      requestId: 2,
      artifactText,
      expectedSha256: digest,
    }, post);

    expect(parseCount).toBe(1);
    expect(responses.filter(({ type }) => type === "model-loaded"))
      .toHaveLength(2);
  });

  it("reports authenticated v22 model metadata across the worker boundary", async () => {
    const runtime = new PgnAnalysisWorkerRuntime(() => fakeV22Model());
    const artifactText = "{}";
    const digest = await sha256Text(artifactText);
    const responses: PgnAnalysisWorkerResponse[] = [];

    await runtime.handle({
      type: "load-model",
      requestId: 22,
      artifactText,
      expectedSha256: digest,
    }, (response) => responses.push(response));

    expect(responses).toEqual([{
      type: "model-loaded",
      requestId: 22,
      model: {
        artifactSha256: digest,
        modelFormatVersion: 3,
        modelVariant: "v22-hybrid",
        drawbackCount: 1,
      },
    }]);
  });

  it("rejects digest mismatch, oversize text, and a missing cache", async () => {
    let parseCount = 0;
    const runtime = new PgnAnalysisWorkerRuntime(() => {
      parseCount += 1;
      return fakeModel();
    });
    const responses: PgnAnalysisWorkerResponse[] = [];
    const post = (response: PgnAnalysisWorkerResponse): void => {
      responses.push(response);
    };

    await runtime.handle({
      type: "load-model",
      requestId: 1,
      artifactText: "{}",
      expectedSha256: "0".repeat(64),
    }, post);
    await runtime.handle({
      type: "load-model",
      requestId: 2,
      artifactText: "x".repeat(32 * 1024 * 1024 + 1),
      expectedSha256: "0".repeat(64),
    }, post);
    await runtime.handle({
      type: "analyze",
      requestId: 3,
      pgn: "1. e4",
      neuralArtifactSha256: "a".repeat(64),
    }, post);

    expect(parseCount).toBe(0);
    expect(
      responses.map((response) =>
        response.type === "error" ? response.error.message : response.type
      ),
    ).toEqual([
      "Model artifact SHA-256 does not match.",
      expect.stringContaining("byte limit"),
      "Requested neural artifact is not loaded in this worker.",
    ]);
  });
});
