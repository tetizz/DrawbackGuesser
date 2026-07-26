import {
  applyHybridResidualPosterior,
  parseHybridBrowserModel,
  runHybridBrowserResidualModel,
  type HybridBrowserModel,
  type HybridObservation,
  type HybridOutput,
} from "./sequence-neural-model.js";
import {
  RANK_PRESERVING_FUSION_METHOD,
} from "./rank-preserving-fusion.js";

export const ENSEMBLE_BROWSER_MODEL_FORMAT_VERSION = 4;
export const ENSEMBLE_BROWSER_MODEL_VARIANT = "v21-hybrid-ensemble";
export const ENSEMBLE_FUSION_METHOD =
  RANK_PRESERVING_FUSION_METHOD;
export const ENSEMBLE_CALIBRATION_METHOD =
  "per-head-multiclass-temperature-scaling";
export const ENSEMBLE_TRAINING_SEEDS = Object.freeze([
  20260811,
  20260812,
  20260813,
] as const);

const MINIMUM_TEMPERATURE = 0.05;
const MAXIMUM_TEMPERATURE = 10;
const DIGEST_PATTERN = /^[0-9a-f]{64}$/u;

interface EnsembleCalibrationHead {
  readonly temperature: number;
  readonly exampleCount: number;
  readonly nllBefore: number;
  readonly nllAfter: number;
}

interface EnsembleMember {
  readonly trainingSeed: (typeof ENSEMBLE_TRAINING_SEEDS)[number];
  readonly selectedEpoch: number;
  readonly trainingRunId: string;
  readonly sourceSelectionSha256: string;
  readonly sourceCheckpointSha256: string;
  readonly model: HybridBrowserModel;
}

export interface EnsembleBrowserModel {
  readonly format: "drawbacktrainer-browser-model";
  readonly formatVersion: typeof ENSEMBLE_BROWSER_MODEL_FORMAT_VERSION;
  readonly modelVariant: typeof ENSEMBLE_BROWSER_MODEL_VARIANT;
  readonly featureSchemaVersion: 1;
  readonly symbolicFeatureVersion: 6;
  readonly drawbackVocabulary: readonly string[];
  readonly symbolicRuleIds: readonly string[];
  readonly tokenizer: HybridBrowserModel["tokenizer"];
  readonly tensorEncoding: HybridBrowserModel["tensorEncoding"];
  readonly dimensions: HybridBrowserModel["dimensions"];
  readonly ensemble: {
    readonly method: typeof ENSEMBLE_FUSION_METHOD;
    readonly memberCount: 3;
    readonly seedOrder: typeof ENSEMBLE_TRAINING_SEEDS;
    readonly sourceEnsembleReleaseSha256: string;
    readonly sourceFusionSelectionSha256: string;
    readonly selectedAlpha: number;
    readonly members: readonly EnsembleMember[];
  };
  readonly calibration: {
    readonly method: typeof ENSEMBLE_CALIBRATION_METHOD;
    readonly sourceCalibrationSha256: string;
    readonly preservesHardEliminations: true;
    readonly white: EnsembleCalibrationHead;
    readonly black: EnsembleCalibrationHead;
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  name: string,
): void {
  const actual = Object.keys(value).sort();
  const canonical = [...expected].sort();
  if (
    actual.length !== canonical.length ||
    actual.some((key, index) => key !== canonical[index])
  ) {
    throw new TypeError(`${name} fields are not canonical.`);
  }
}

function digest(value: unknown, name: string): string {
  if (typeof value !== "string" || !DIGEST_PATTERN.test(value)) {
    throw new TypeError(`${name} must be a lowercase SHA-256 digest.`);
  }
  return value;
}

function positiveInteger(value: unknown, name: string): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value <= 0
  ) {
    throw new TypeError(`${name} must be a positive safe integer.`);
  }
  return value;
}

function fusionAlpha(value: unknown): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < 0 ||
    value > 1
  ) {
    throw new TypeError("Browser ensemble selectedAlpha is invalid.");
  }
  return value;
}

function finiteNonnegative(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new TypeError(`${name} must be finite and non-negative.`);
  }
  return value;
}

function calibrationHead(
  value: unknown,
  name: string,
): EnsembleCalibrationHead {
  if (!isRecord(value)) {
    throw new TypeError(`${name} must be an object.`);
  }
  exactKeys(
    value,
    ["temperature", "exampleCount", "nllBefore", "nllAfter"],
    name,
  );
  const temperature = finiteNonnegative(
    value["temperature"],
    `${name}.temperature`,
  );
  if (
    temperature < MINIMUM_TEMPERATURE ||
    temperature > MAXIMUM_TEMPERATURE
  ) {
    throw new TypeError(`${name}.temperature is outside release bounds.`);
  }
  const exampleCount = positiveInteger(
    value["exampleCount"],
    `${name}.exampleCount`,
  );
  const nllBefore = finiteNonnegative(
    value["nllBefore"],
    `${name}.nllBefore`,
  );
  const nllAfter = finiteNonnegative(
    value["nllAfter"],
    `${name}.nllAfter`,
  );
  if (nllAfter >= nllBefore) {
    throw new TypeError(`${name} did not improve NLL.`);
  }
  return Object.freeze({
    temperature,
    exampleCount,
    nllBefore,
    nllAfter,
  });
}

function sharedHybridArtifact(
  root: Record<string, unknown>,
  member: Record<string, unknown>,
): Record<string, unknown> {
  return {
    format: "drawbacktrainer-browser-model",
    formatVersion: 2,
    modelVariant: "v21-hybrid",
    featureSchemaVersion: root["featureSchemaVersion"],
    symbolicFeatureVersion: root["symbolicFeatureVersion"],
    sourceCheckpointSha256: member["sourceCheckpointSha256"],
    drawbackVocabulary: root["drawbackVocabulary"],
    symbolicRuleIds: root["symbolicRuleIds"],
    tokenizer: root["tokenizer"],
    tensorEncoding: root["tensorEncoding"],
    dimensions: root["dimensions"],
    tensors: member["tensors"],
  };
}

export function parseEnsembleBrowserModel(
  value: Record<string, unknown>,
  featureDimension: number,
): EnsembleBrowserModel {
  exactKeys(
    value,
    [
      "format",
      "formatVersion",
      "modelVariant",
      "featureSchemaVersion",
      "symbolicFeatureVersion",
      "drawbackVocabulary",
      "symbolicRuleIds",
      "tokenizer",
      "tensorEncoding",
      "dimensions",
      "ensemble",
      "calibration",
    ],
    "Browser ensemble model",
  );
  if (
    value["format"] !== "drawbacktrainer-browser-model" ||
    value["formatVersion"] !== ENSEMBLE_BROWSER_MODEL_FORMAT_VERSION ||
    value["modelVariant"] !== ENSEMBLE_BROWSER_MODEL_VARIANT
  ) {
    throw new TypeError("Browser ensemble model format is unsupported.");
  }
  const tokenizer = value["tokenizer"];
  if (!isRecord(tokenizer)) {
    throw new TypeError("Browser ensemble SAN tokenizer is missing.");
  }
  exactKeys(
    tokenizer,
    [
      "kind",
      "version",
      "vocabulary",
      "max_history",
      "padding",
      "truncation",
    ],
    "Browser ensemble SAN tokenizer",
  );
  const dimensions = value["dimensions"];
  if (!isRecord(dimensions)) {
    throw new TypeError("Browser ensemble dimensions are missing.");
  }
  exactKeys(
    dimensions,
    [
      "input",
      "boardHidden",
      "sanVocabulary",
      "sanEmbedding",
      "sequenceHidden",
      "symbolicInput",
      "symbolicHidden",
      "drawbackClasses",
    ],
    "Browser ensemble dimensions",
  );
  const ensemble = value["ensemble"];
  if (!isRecord(ensemble)) {
    throw new TypeError("Browser ensemble metadata is missing.");
  }
  exactKeys(
    ensemble,
    [
      "method",
      "memberCount",
      "seedOrder",
      "sourceEnsembleReleaseSha256",
      "sourceFusionSelectionSha256",
      "selectedAlpha",
      "members",
    ],
    "Browser ensemble metadata",
  );
  if (
    ensemble["method"] !== ENSEMBLE_FUSION_METHOD ||
    ensemble["memberCount"] !== 3 ||
    !Array.isArray(ensemble["seedOrder"]) ||
    ensemble["seedOrder"].length !== ENSEMBLE_TRAINING_SEEDS.length ||
    ensemble["seedOrder"].some(
      (seed, index) => seed !== ENSEMBLE_TRAINING_SEEDS[index],
    )
  ) {
    throw new TypeError("Browser ensemble method or seed order is invalid.");
  }
  const sourceEnsembleReleaseSha256 = digest(
    ensemble["sourceEnsembleReleaseSha256"],
    "sourceEnsembleReleaseSha256",
  );
  const sourceFusionSelectionSha256 = digest(
    ensemble["sourceFusionSelectionSha256"],
    "sourceFusionSelectionSha256",
  );
  const selectedAlpha = fusionAlpha(ensemble["selectedAlpha"]);
  const rawMembers = ensemble["members"];
  if (!Array.isArray(rawMembers) || rawMembers.length !== 3) {
    throw new TypeError("Browser ensemble requires exactly three members.");
  }
  const members = rawMembers.map((rawMember, index): EnsembleMember => {
    if (!isRecord(rawMember)) {
      throw new TypeError("Browser ensemble member must be an object.");
    }
    exactKeys(
      rawMember,
      [
        "trainingSeed",
        "selectedEpoch",
        "trainingRunId",
        "sourceSelectionSha256",
        "sourceCheckpointSha256",
        "tensors",
      ],
      "Browser ensemble member",
    );
    const expectedSeed = ENSEMBLE_TRAINING_SEEDS[index];
    if (expectedSeed === undefined) {
      throw new Error("Browser ensemble seed-order invariant violated.");
    }
    if (rawMember["trainingSeed"] !== expectedSeed) {
      throw new TypeError("Browser ensemble members are not in fixed seed order.");
    }
    const model = parseHybridBrowserModel(
      sharedHybridArtifact(value, rawMember),
      featureDimension,
    );
    return Object.freeze({
      trainingSeed: expectedSeed,
      selectedEpoch: positiveInteger(
        rawMember["selectedEpoch"],
        "selectedEpoch",
      ),
      trainingRunId: digest(rawMember["trainingRunId"], "trainingRunId"),
      sourceSelectionSha256: digest(
        rawMember["sourceSelectionSha256"],
        "sourceSelectionSha256",
      ),
      sourceCheckpointSha256: model.sourceCheckpointSha256,
      model,
    });
  });
  for (const field of [
    "trainingRunId",
    "sourceSelectionSha256",
    "sourceCheckpointSha256",
  ] as const) {
    if (new Set(members.map((member) => member[field])).size !== 3) {
      throw new TypeError(`Browser ensemble members reuse ${field}.`);
    }
  }
  const calibration = value["calibration"];
  if (!isRecord(calibration)) {
    throw new TypeError("Browser ensemble calibration is missing.");
  }
  exactKeys(
    calibration,
    [
      "method",
      "sourceCalibrationSha256",
      "preservesHardEliminations",
      "white",
      "black",
    ],
    "Browser ensemble calibration",
  );
  if (
    calibration["method"] !== ENSEMBLE_CALIBRATION_METHOD ||
    calibration["preservesHardEliminations"] !== true
  ) {
    throw new TypeError("Browser ensemble calibration method is invalid.");
  }
  const first = members[0];
  if (first === undefined) {
    throw new Error("Browser ensemble member invariant violated.");
  }
  return Object.freeze({
    format: "drawbacktrainer-browser-model",
    formatVersion: ENSEMBLE_BROWSER_MODEL_FORMAT_VERSION,
    modelVariant: ENSEMBLE_BROWSER_MODEL_VARIANT,
    featureSchemaVersion: first.model.featureSchemaVersion,
    symbolicFeatureVersion: first.model.symbolicFeatureVersion,
    drawbackVocabulary: first.model.drawbackVocabulary,
    symbolicRuleIds: first.model.symbolicRuleIds,
    tokenizer: first.model.tokenizer,
    tensorEncoding: first.model.tensorEncoding,
    dimensions: first.model.dimensions,
    ensemble: Object.freeze({
      method: ENSEMBLE_FUSION_METHOD,
      memberCount: 3,
      seedOrder: ENSEMBLE_TRAINING_SEEDS,
      sourceEnsembleReleaseSha256,
      sourceFusionSelectionSha256,
      selectedAlpha,
      members: Object.freeze(members),
    }),
    calibration: Object.freeze({
      method: ENSEMBLE_CALIBRATION_METHOD,
      sourceCalibrationSha256: digest(
        calibration["sourceCalibrationSha256"],
        "sourceCalibrationSha256",
      ),
      preservesHardEliminations: true,
      white: calibrationHead(calibration["white"], "calibration.white"),
      black: calibrationHead(calibration["black"], "calibration.black"),
    }),
  });
}

export function runEnsembleBrowserModel(
  model: EnsembleBrowserModel,
  observation: HybridObservation,
  boardFeatures: readonly number[],
): HybridOutput {
  const first = model.ensemble.members[0];
  if (first === undefined) {
    throw new Error("Browser ensemble member invariant violated.");
  }
  const residuals = model.ensemble.members.map((member) =>
    runHybridBrowserResidualModel(member.model, observation, boardFeatures)
  );
  const mean = (color: "white" | "black"): readonly number[] =>
    Object.freeze(
      model.drawbackVocabulary.map((_, index) =>
        residuals.reduce(
          (sum, residual) => sum + (residual[color][index] ?? 0),
          0,
        ) / residuals.length
      ),
    );
  return applyHybridResidualPosterior(
    first.model,
    observation.symbolic,
    { white: mean("white"), black: mean("black") },
    {
      white: model.calibration.white.temperature,
      black: model.calibration.black.temperature,
    },
    model.ensemble.selectedAlpha,
  );
}
