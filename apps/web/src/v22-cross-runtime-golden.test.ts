import { describe, expect, it } from "vitest";
import fixtureText from "./fixtures/v22-cross-runtime-golden.json?raw";
import exporterSource from "../../../ml/training/drawback_ml/browser_artifact.py?raw";
import generatorSource from "../../../scripts/generate-v22-cross-runtime-fixture.py?raw";
import {
  buildNeuralFeatureVector,
  parseBrowserNeuralModel,
  runBrowserNeuralModel,
} from "./neural-model.js";
import type {
  HybridObservation,
  HybridOutput,
  HybridV22BrowserModel,
} from "./sequence-neural-model.js";
import {
  encodeHybridSequenceTokenIndices,
} from "./sequence-neural-model.js";

interface GoldenExpected {
  readonly white: readonly number[];
  readonly black: readonly number[];
}

interface GoldenCase {
  readonly observation: HybridObservation;
  readonly expectedTokenIndices: {
    readonly exact: readonly number[];
    readonly masked: readonly number[];
  };
  readonly expected: {
    readonly exact: GoldenExpected;
    readonly masked: GoldenExpected;
  };
}

interface CrossRuntimeFixture {
  readonly format: string;
  readonly formatVersion: number;
  readonly bindings: {
    readonly checkpointProvenance: {
      readonly algorithm: "sha256";
      readonly encoding: "portable-torch-checkpoint-zip-v1";
      readonly exactSha256: string;
      readonly maskedSha256: string;
    };
    readonly exactArtifactSha256: string;
    readonly inputSpecSha256: string;
    readonly maskedArtifactSha256: string;
    readonly sourceSha256: Readonly<Record<string, string>>;
  };
  readonly inputSpec: unknown;
  readonly artifact: unknown;
  readonly maskedArtifactDelta: {
    readonly sequenceObservationMode: "masked-current-v2";
    readonly sourceCheckpointSha256: string;
  };
  readonly cases: readonly GoldenCase[];
}

const fixture = JSON.parse(fixtureText) as CrossRuntimeFixture;

function sortedJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortedJsonValue);
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) =>
          left < right ? -1 : left > right ? 1 : 0
        )
        .map(([key, item]) => [key, sortedJsonValue(item)]),
    );
  }
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "boolean" ||
    (typeof value === "number" && Number.isFinite(value))
  ) {
    return value;
  }
  throw new TypeError("Golden fixture contains non-canonical JSON.");
}

function canonicalJson(value: unknown): string {
  return `${JSON.stringify(sortedJsonValue(value))}\n`;
}

function normalizedSource(value: string): string {
  return value.replace(/\r\n?/gu, "\n");
}

async function sha256Text(value: string): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function maskedArtifact(): unknown {
  if (
    fixture.artifact === null ||
    typeof fixture.artifact !== "object" ||
    Array.isArray(fixture.artifact)
  ) {
    throw new TypeError("Golden artifact must be an object.");
  }
  return {
    ...structuredClone(fixture.artifact as Record<string, unknown>),
    ...fixture.maskedArtifactDelta,
  };
}

function orderedOutput(
  model: HybridV22BrowserModel,
  output: HybridOutput,
  color: "white" | "black",
): readonly number[] {
  return model.drawbackVocabulary.map((id) => {
    const probability = output[color][id];
    if (probability === undefined) {
      throw new Error(`Browser output omitted ${color} ${id}.`);
    }
    return probability;
  });
}

function expectPythonParity(
  model: HybridV22BrowserModel,
  observation: HybridObservation,
  expected: GoldenExpected,
): HybridOutput {
  const output = runBrowserNeuralModel(model, observation);
  for (const color of ["white", "black"] as const) {
    const actual = orderedOutput(model, output, color);
    expect(actual).toHaveLength(expected[color].length);
    let maximumDifference = 0;
    for (const [index, received] of actual.entries()) {
      maximumDifference = Math.max(
        maximumDifference,
        Math.abs(received - (expected[color][index] ?? Number.NaN)),
      );
      const id = model.drawbackVocabulary[index];
      if (id === undefined) {
        throw new Error("Golden vocabulary index is missing.");
      }
      const symbolicIndex = model.symbolicRuleIds.indexOf(id);
      const eliminated = observation.symbolic[
        color === "white" ? "whiteEliminated" : "blackEliminated"
      ][symbolicIndex];
      if (eliminated) {
        expect(received).toBe(0);
      }
    }
    expect(maximumDifference).toBeLessThanOrEqual(1e-5);
    expect(actual.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
  }
  return output;
}

describe("v22 Python/browser cross-runtime golden", () => {
  it("binds the generated artifacts to normalized sources and inputs", async () => {
    expect(fixture).toMatchObject({
      format: "drawbacktrainer-v22-cross-runtime-golden",
      formatVersion: 1,
    });
    await expect(
      sha256Text(normalizedSource(generatorSource)),
    ).resolves.toBe(
      fixture.bindings.sourceSha256[
        "scripts/generate-v22-cross-runtime-fixture.py"
      ],
    );
    await expect(
      sha256Text(normalizedSource(exporterSource)),
    ).resolves.toBe(
      fixture.bindings.sourceSha256[
        "ml/training/drawback_ml/browser_artifact.py"
      ],
    );
    await expect(sha256Text(canonicalJson(fixture.inputSpec))).resolves.toBe(
      fixture.bindings.inputSpecSha256,
    );
    await expect(sha256Text(canonicalJson(fixture.artifact))).resolves.toBe(
      fixture.bindings.exactArtifactSha256,
    );
    await expect(sha256Text(canonicalJson(maskedArtifact()))).resolves.toBe(
      fixture.bindings.maskedArtifactSha256,
    );
    expect(fixture.bindings.checkpointProvenance).toMatchObject({
      algorithm: "sha256",
      encoding: "portable-torch-checkpoint-zip-v1",
      exactSha256: (
        fixture.artifact as Record<string, unknown>
      )["sourceCheckpointSha256"],
      maskedSha256: fixture.maskedArtifactDelta.sourceCheckpointSha256,
    });
  });

  it("matches Python tokenization and inference in exact and masked modes", () => {
    const exact = parseBrowserNeuralModel(fixture.artifact);
    const masked = parseBrowserNeuralModel(maskedArtifact());
    if (
      exact.modelVariant !== "v22-hybrid" ||
      masked.modelVariant !== "v22-hybrid"
    ) {
      throw new Error("Golden artifacts did not parse as v22 models.");
    }
    expect(exact.sequenceObservationMode).toBe("exact-current-v2");
    expect(masked.sequenceObservationMode).toBe("masked-current-v2");
    expect(exact.sourceCheckpointSha256).not.toBe(
      masked.sourceCheckpointSha256,
    );
    expect(exact.tensors).toEqual(masked.tensors);

    const goldenCase = fixture.cases[0];
    if (goldenCase === undefined) {
      throw new Error("Golden fixture contains no inference case.");
    }
    expect(
      encodeHybridSequenceTokenIndices(exact, goldenCase.observation),
    ).toEqual(goldenCase.expectedTokenIndices.exact);
    expect(
      encodeHybridSequenceTokenIndices(masked, goldenCase.observation),
    ).toEqual(goldenCase.expectedTokenIndices.masked);
    expect(goldenCase.expectedTokenIndices.exact.at(-1)).not.toBe(
      goldenCase.expectedTokenIndices.masked.at(-1),
    );

    const exactOutput = expectPythonParity(
      exact,
      goldenCase.observation,
      goldenCase.expected.exact,
    );
    const maskedOutput = expectPythonParity(
      masked,
      goldenCase.observation,
      goldenCase.expected.masked,
    );
    const exactWhite = orderedOutput(exact, exactOutput, "white");
    const maskedWhite = orderedOutput(masked, maskedOutput, "white");
    expect(
      Math.max(
        ...exactWhite.map((value, index) =>
          Math.abs(value - (maskedWhite[index] ?? Number.NaN))
        ),
      ),
    ).toBeGreaterThan(1e-7);

    expect(
      buildNeuralFeatureVector(goldenCase.observation),
    ).toHaveLength(exact.dimensions.input);
  });
});
