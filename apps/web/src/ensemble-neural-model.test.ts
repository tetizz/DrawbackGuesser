import { describe, expect, it } from "vitest";
import hybridGoldenText from "./fixtures/v21-golden.json?raw";
import ensembleGoldenText from "./fixtures/ensemble-v4-golden.json?raw";
import {
  ENSEMBLE_BROWSER_MODEL_FORMAT_VERSION,
  ENSEMBLE_BROWSER_MODEL_VARIANT,
  ENSEMBLE_CALIBRATION_METHOD,
  ENSEMBLE_FUSION_METHOD,
  ENSEMBLE_TRAINING_SEEDS,
} from "./ensemble-neural-model.js";
import {
  parseBrowserNeuralModel,
  runBrowserNeuralModel,
} from "./neural-model.js";
import type { HybridObservation } from "./sequence-neural-model.js";

interface Formula {
  readonly modulus: number;
  readonly center: number;
  readonly scale: number;
  readonly memberOffsets: readonly number[];
}

interface Golden {
  readonly caseIndex: number;
  readonly fusionAlpha: number;
  readonly calibration: {
    readonly whiteTemperature: number;
    readonly blackTemperature: number;
  };
  readonly residualFormula: {
    readonly white: Formula;
    readonly black: Formula;
  };
  readonly expected: Record<
    "white" | "black",
    {
      readonly topIds: readonly string[];
      readonly zeroCount: number;
      readonly probabilities: Readonly<Record<string, number>>;
    }
  >;
}

function encode(values: readonly number[]): string {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => {
    view.setFloat32(index * 4, value, true);
  });
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

function digest(index: number): string {
  return index.toString(16).repeat(64);
}

function required<T>(value: T | undefined, name: string): T {
  if (value === undefined) {
    throw new Error(`${name} is missing.`);
  }
  return value;
}

function artifactFixture(): {
  artifact: Record<string, unknown>;
  observation: Record<string, unknown>;
  golden: Golden;
} {
  const hybrid = JSON.parse(hybridGoldenText) as {
    artifact: Record<string, unknown>;
    cases: Array<{ observation: Record<string, unknown> }>;
  };
  const golden = JSON.parse(ensembleGoldenText) as Golden;
  const vocabulary = hybrid.artifact["drawbackVocabulary"];
  if (!Array.isArray(vocabulary)) {
    throw new Error("Hybrid golden vocabulary is missing.");
  }
  const baseTensors = hybrid.artifact["tensors"] as Record<
    string,
    { shape: number[]; data: string }
  >;
  const members = ENSEMBLE_TRAINING_SEEDS.map((trainingSeed, memberIndex) => {
    const tensors = structuredClone(baseTensors);
    for (const tensor of Object.values(tensors)) {
      const size = tensor.shape.reduce(
        (product, dimension) => product * dimension,
        1,
      );
      tensor.data = encode(Array<number>(size).fill(0));
    }
    for (const color of ["white", "black"] as const) {
      const formula = golden.residualFormula[color];
      const offset = formula.memberOffsets[memberIndex];
      if (offset === undefined) {
        throw new Error("Ensemble golden member offset is missing.");
      }
      required(
        tensors[`${color}_drawback.bias`],
        `${color} drawback bias`,
      ).data = encode(
        vocabulary.map(
          (_, index) =>
            ((index % formula.modulus) - formula.center) * formula.scale +
            offset,
        ),
      );
    }
    return {
      trainingSeed,
      selectedEpoch: memberIndex + 2,
      trainingRunId: digest(memberIndex + 1),
      sourceSelectionSha256: digest(memberIndex + 4),
      sourceCheckpointSha256: digest(memberIndex + 7),
      tensors,
    };
  });
  const artifact = {
    format: hybrid.artifact["format"],
    formatVersion: ENSEMBLE_BROWSER_MODEL_FORMAT_VERSION,
    modelVariant: ENSEMBLE_BROWSER_MODEL_VARIANT,
    featureSchemaVersion: hybrid.artifact["featureSchemaVersion"],
    symbolicFeatureVersion: hybrid.artifact["symbolicFeatureVersion"],
    drawbackVocabulary: hybrid.artifact["drawbackVocabulary"],
    symbolicRuleIds: hybrid.artifact["symbolicRuleIds"],
    tokenizer: hybrid.artifact["tokenizer"],
    tensorEncoding: hybrid.artifact["tensorEncoding"],
    dimensions: hybrid.artifact["dimensions"],
    ensemble: {
      method: ENSEMBLE_FUSION_METHOD,
      memberCount: 3,
      seedOrder: [...ENSEMBLE_TRAINING_SEEDS],
      sourceEnsembleReleaseSha256: digest(10),
      sourceFusionSelectionSha256: digest(12),
      selectedAlpha: golden.fusionAlpha,
      members,
    },
    calibration: {
      method: ENSEMBLE_CALIBRATION_METHOD,
      sourceCalibrationSha256: digest(11),
      preservesHardEliminations: true,
      white: {
        temperature: golden.calibration.whiteTemperature,
        exampleCount: 500,
        nllBefore: 4,
        nllAfter: 3,
      },
      black: {
        temperature: golden.calibration.blackTemperature,
        exampleCount: 500,
        nllBefore: 4,
        nllAfter: 3,
      },
    },
  };
  const observation = hybrid.cases[golden.caseIndex]?.observation;
  if (observation === undefined) {
    throw new Error("Ensemble golden observation is missing.");
  }
  return { artifact, observation, golden };
}

describe("browser v4 calibrated ensemble", () => {
  it("matches the deterministic Python fusion and calibration golden", () => {
    const fixture = artifactFixture();
    const model = parseBrowserNeuralModel(fixture.artifact);
    expect(model.modelVariant).toBe(ENSEMBLE_BROWSER_MODEL_VARIANT);
    if (model.modelVariant !== ENSEMBLE_BROWSER_MODEL_VARIANT) {
      throw new Error("Ensemble fixture parsed as the wrong model variant.");
    }
    expect(model.ensemble.members.map(({ trainingSeed }) => trainingSeed))
      .toEqual(ENSEMBLE_TRAINING_SEEDS);
    const output = runBrowserNeuralModel(
      model,
      fixture.observation as unknown as HybridObservation,
    );
    for (const color of ["white", "black"] as const) {
      const ordered = Object.entries(output[color])
        .sort(([leftId, left], [rightId, right]) =>
          right - left || leftId.localeCompare(rightId)
        )
        .slice(0, 5)
        .map(([id]) => id);
      expect(ordered).toEqual(fixture.golden.expected[color].topIds);
      expect(
        Object.values(output[color]).filter((probability) => probability === 0),
      ).toHaveLength(fixture.golden.expected[color].zeroCount);
      expect(
        Object.values(output[color]).reduce(
          (sum, probability) => sum + probability,
          0,
        ),
      ).toBeCloseTo(1, 12);
      for (
        const [id, expected] of Object.entries(
          fixture.golden.expected[color].probabilities,
        )
      ) {
        expect(output[color][id]).toBeCloseTo(expected, 6);
      }
    }
  });

  it("keeps White and Black calibration and masks isolated", () => {
    const fixture = artifactFixture();
    const baseline = runBrowserNeuralModel(
      parseBrowserNeuralModel(fixture.artifact),
      fixture.observation as unknown as HybridObservation,
    );
    const changed = structuredClone(fixture.artifact);
    const calibration = changed["calibration"] as Record<string, unknown>;
    const white = calibration["white"] as Record<string, unknown>;
    white["temperature"] = 2.5;
    const received = runBrowserNeuralModel(
      parseBrowserNeuralModel(changed),
      fixture.observation as unknown as HybridObservation,
    );
    expect(received.black).toEqual(baseline.black);
    expect(received.white).not.toEqual(baseline.white);
  });

  it("rejects noncanonical fields, member order, and reused identities", () => {
    const legacy = artifactFixture().artifact;
    legacy["formatVersion"] = 3;
    expect(() => parseBrowserNeuralModel(legacy)).toThrow("unsupported");

    const v22Root = artifactFixture().artifact;
    v22Root["modelVariant"] = "v22-hybrid";
    expect(() => parseBrowserNeuralModel(v22Root)).toThrow("unsupported");

    const v22Member = artifactFixture().artifact;
    const v22Members = (
      v22Member["ensemble"] as Record<string, unknown>
    )["members"] as Array<Record<string, unknown>>;
    required(v22Members[0], "first member")["modelVariant"] = "v22-hybrid";
    expect(() => parseBrowserNeuralModel(v22Member)).toThrow("canonical");

    const extra = artifactFixture().artifact;
    extra["unexpected"] = true;
    expect(() => parseBrowserNeuralModel(extra)).toThrow("not canonical");

    const short = artifactFixture().artifact;
    const shortEnsemble = short["ensemble"] as Record<string, unknown>;
    (shortEnsemble["members"] as unknown[]).pop();
    expect(() => parseBrowserNeuralModel(short)).toThrow("exactly three");

    const reordered = artifactFixture().artifact;
    const reorderedEnsemble = reordered["ensemble"] as Record<string, unknown>;
    const reorderedMembers = reorderedEnsemble["members"] as Array<
      Record<string, unknown>
    >;
    const reorderedFirst = required(reorderedMembers[0], "first member");
    const reorderedSecond = required(reorderedMembers[1], "second member");
    [
      reorderedFirst["trainingSeed"],
      reorderedSecond["trainingSeed"],
    ] = [
      reorderedSecond["trainingSeed"],
      reorderedFirst["trainingSeed"],
    ];
    expect(() => parseBrowserNeuralModel(reordered)).toThrow("seed order");

    const reused = artifactFixture().artifact;
    const reusedMembers = (
      reused["ensemble"] as Record<string, unknown>
    )["members"] as Array<Record<string, unknown>>;
    const reusedFirst = required(reusedMembers[0], "first member");
    const reusedSecond = required(reusedMembers[1], "second member");
    reusedSecond["sourceCheckpointSha256"] =
      reusedFirst["sourceCheckpointSha256"];
    expect(() => parseBrowserNeuralModel(reused)).toThrow(
      "reuse sourceCheckpointSha256",
    );
  });

  it("rejects oversized tensors and invalid calibration", () => {
    const oversized = artifactFixture().artifact;
    const members = (
      oversized["ensemble"] as Record<string, unknown>
    )["members"] as Array<Record<string, unknown>>;
    const tensors = required(members[0], "first member")["tensors"] as Record<
      string,
      Record<string, unknown>
    >;
    required(
      tensors["board_encoder.0.bias"],
      "board encoder bias",
    )["data"] =
      "A".repeat(4 * Math.ceil((8 * 1024 * 1024) / 3) + 4);
    expect(() => parseBrowserNeuralModel(oversized)).toThrow(
      "8 MiB runtime limit",
    );

    for (const mutate of [
      (calibration: Record<string, unknown>) => {
        (calibration["white"] as Record<string, unknown>)["temperature"] = 0;
      },
      (calibration: Record<string, unknown>) => {
        calibration["preservesHardEliminations"] = false;
      },
      (calibration: Record<string, unknown>) => {
        const black = calibration["black"] as Record<string, unknown>;
        black["nllAfter"] = black["nllBefore"];
      },
    ]) {
      const invalid = artifactFixture().artifact;
      mutate(invalid["calibration"] as Record<string, unknown>);
      expect(() => parseBrowserNeuralModel(invalid)).toThrow(
        /calibration|temperature|NLL/u,
      );
    }
  });
});
