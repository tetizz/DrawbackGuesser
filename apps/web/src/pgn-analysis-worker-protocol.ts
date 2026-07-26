import type {
  PgnAnalysisProgress,
  PgnAnalysisResult,
} from "./pgn-analysis.js";
import {
  MAX_COMPLETED_PGN_EVALUATOR_SIDECAR_BYTES,
} from "@drawbackengine/chess-evaluator/completed-pgn-sidecar";
import { STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS } from "./pgn-analysis-contract.js";
import { DEFAULT_HYPOTHESIS_RULE_IDS } from "@drawbackguesser/predictor";

export const MAX_MODEL_ARTIFACT_BYTES = 32 * 1024 * 1024;
export const MAX_EVALUATOR_SIDECAR_BYTES =
  MAX_COMPLETED_PGN_EVALUATOR_SIDECAR_BYTES;

const CATALOG_RULE_IDS = new Set<string>(DEFAULT_HYPOTHESIS_RULE_IDS);
const REPRESENTED_RULE_IDS = new Set<string>(
  DEFAULT_HYPOTHESIS_RULE_IDS.filter(
    (id) => !STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS.includes(
      id as (typeof STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS)[number],
    ),
  ),
);

export type PgnAnalysisWorkerRequest =
  | {
      readonly type: "load-model";
      readonly requestId: number;
      readonly artifactText: string;
      readonly expectedSha256: string;
    }
  | {
      readonly type: "analyze";
      readonly requestId: number;
      readonly pgn: string;
      readonly neuralArtifactSha256?: string;
      readonly evaluatorSidecarBytes?: Uint8Array;
      readonly evaluatorSidecarSha256?: string;
    };

export interface LoadedPgnAnalysisModel {
  readonly artifactSha256: string;
  readonly modelFormatVersion: 1 | 2 | 3 | 4;
  readonly modelVariant:
    | "v1"
    | "v21-hybrid"
    | "v22-hybrid"
    | "v21-hybrid-ensemble";
  readonly drawbackCount: number;
}

export interface SerializedPgnAnalysisError {
  readonly name: string;
  readonly message: string;
  readonly ply: number | null;
  readonly token: string | null;
}

export type PgnAnalysisWorkerResponse =
  | {
      readonly type: "model-loaded";
      readonly requestId: number;
      readonly model: LoadedPgnAnalysisModel;
    }
  | {
      readonly type: "progress";
      readonly requestId: number;
      readonly progress: PgnAnalysisProgress;
    }
  | {
      readonly type: "result";
      readonly requestId: number;
      readonly result: PgnAnalysisResult;
    }
  | {
      readonly type: "error";
      readonly requestId: number;
      readonly error: SerializedPgnAnalysisError;
    };

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null;
}

function hasExactKeys(
  value: Readonly<Record<string, unknown>>,
  keys: readonly string[],
): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return (
    actual.length === expected.length &&
    actual.every((key, index) => key === expected[index])
  );
}

function isStringArray(value: unknown, maximum: number): value is string[] {
  return (
    Array.isArray(value) &&
    value.length <= maximum &&
    value.every((item) => typeof item === "string")
  );
}

function isExactRuleSet(
  value: unknown,
  expected: ReadonlySet<string>,
): value is string[] {
  return (
    isStringArray(value, expected.size) &&
    value.length === expected.size &&
    new Set(value).size === expected.size &&
    value.every((id) => expected.has(id))
  );
}

function isGuess(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value["id"] === "string" &&
    typeof value["confidence"] === "number" &&
    Number.isFinite(value["confidence"]) &&
    value["confidence"] >= 0 &&
    value["confidence"] <= 1 &&
    typeof value["eliminated"] === "boolean" &&
    Array.isArray(value["parameters"]) &&
    value["parameters"].length <= 16 &&
    value["parameters"].every(
      (parameter) =>
        isRecord(parameter) &&
        typeof parameter["name"] === "string" &&
        Object.hasOwn(parameter, "value") &&
        typeof parameter["confidence"] === "number" &&
        Number.isFinite(parameter["confidence"]) &&
        parameter["confidence"] >= 0 &&
        parameter["confidence"] <= 1,
    )
  );
}

function isGuessArray(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.length <= 200 &&
    value.every(isGuess)
  );
}

function isExactGuessArray(
  value: unknown,
  expected: ReadonlySet<string>,
): boolean {
  return (
    isGuessArray(value) &&
    Array.isArray(value) &&
    value.length === expected.size &&
    new Set(
      value.flatMap((guess) =>
        isRecord(guess) && typeof guess["id"] === "string"
          ? [guess["id"]]
          : []
      ),
    ).size === expected.size &&
    value.every(
      (guess) =>
        isRecord(guess) &&
        typeof guess["id"] === "string" &&
        expected.has(guess["id"]),
    )
  );
}

function isExactUnavailableDrawbacks(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.length === STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS.length &&
    value.every(
      (item, index) =>
        isRecord(item) &&
        item["id"] === STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS[index] &&
        typeof item["name"] === "string" &&
        item["name"].length > 0 &&
        item["reason"] === "requires-public-evaluator-facts" &&
        item["rank"] === null &&
        item["eliminated"] === false,
    )
  );
}

function isDigest(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/u.test(value);
}

function isCalibrationHead(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      "temperature",
      "exampleCount",
      "nllBefore",
      "nllAfter",
    ]) &&
    typeof value["temperature"] === "number" &&
    Number.isFinite(value["temperature"]) &&
    value["temperature"] >= 0.05 &&
    value["temperature"] <= 10 &&
    typeof value["exampleCount"] === "number" &&
    Number.isSafeInteger(value["exampleCount"]) &&
    value["exampleCount"] > 0 &&
    typeof value["nllBefore"] === "number" &&
    Number.isFinite(value["nllBefore"]) &&
    value["nllBefore"] >= 0 &&
    typeof value["nllAfter"] === "number" &&
    Number.isFinite(value["nllAfter"]) &&
    value["nllAfter"] >= 0 &&
    value["nllAfter"] < value["nllBefore"]
  );
}

function hasUniqueMemberProvenance(members: readonly unknown[]): boolean {
  const records = members.filter(isRecord);
  if (records.length !== members.length) {
    return false;
  }
  return (
    new Set(records.map((member) => member["sourceCheckpointSha256"])).size ===
      members.length &&
    new Set(records.map((member) => member["sourceSelectionSha256"])).size ===
      members.length &&
    new Set(records.map((member) => member["trainingRunId"])).size ===
      members.length
  );
}

function hasExpectedUnresolvedConstraints(
  value: unknown,
  enriched: boolean,
): boolean {
  return (
    isStringArray(value, 2) &&
    value.length === (enriched ? 0 : STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS.length) &&
    value.every(
      (id, index) => id === STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS[index],
    )
  );
}

function isEvaluatorEvidence(
  value: unknown,
): value is PgnAnalysisResult["evaluatorEvidence"] {
  if (!isRecord(value)) {
    return false;
  }
  if (value["mode"] === "standard-pgn") {
    return hasExactKeys(value, ["mode"]);
  }
  const policy = value["policy"];
  const engine = value["engine"];
  const limit = value["searchLimit"];
  return (
    value["mode"] === "authenticated-sidecar" &&
    hasExactKeys(value, [
      "mode",
      "artifactSha256",
      "policy",
      "engine",
      "searchLimit",
    ]) &&
    isDigest(value["artifactSha256"]) &&
    isRecord(policy) &&
    hasExactKeys(policy, ["id", "version"]) &&
    typeof policy["id"] === "string" &&
    policy["id"].length > 0 &&
    Number.isSafeInteger(policy["version"]) &&
    typeof policy["version"] === "number" &&
    policy["version"] > 0 &&
    isRecord(engine) &&
    hasExactKeys(engine, [
      "uciName",
      "engine",
      "version",
      "executableSha256",
      "optionsDigest",
      "publicFingerprint",
    ]) &&
    typeof engine["uciName"] === "string" &&
    engine["uciName"].length > 0 &&
    typeof engine["engine"] === "string" &&
    engine["engine"].length > 0 &&
    typeof engine["version"] === "string" &&
    engine["version"].length > 0 &&
    isDigest(engine["executableSha256"]) &&
    isDigest(engine["optionsDigest"]) &&
    engine["publicFingerprint"] === [
      engine["engine"],
      engine["version"],
      engine["executableSha256"],
      engine["optionsDigest"],
    ].join(":") &&
    isRecord(limit) &&
    hasExactKeys(limit, ["kind", "value"]) &&
    (
      limit["kind"] === "depth" ||
      limit["kind"] === "move-time-ms" ||
      limit["kind"] === "nodes"
    ) &&
    Number.isSafeInteger(limit["value"]) &&
    typeof limit["value"] === "number" &&
    limit["value"] > 0
  );
}

function isEnsemblePredictor(
  value: Readonly<Record<string, unknown>>,
  enriched: boolean,
): boolean {
  const members = value["members"];
  const calibration = value["calibration"];
  return (
    hasExactKeys(value, [
      "mode",
      "modelFormatVersion",
      "artifactSha256",
      "sourceEnsembleReleaseSha256",
      "sourceFusionSelectionSha256",
      "sourceCalibrationSha256",
      "featureSchemaVersion",
      "symbolicFeatureVersion",
      "fusionMethod",
      "selectedAlpha",
      "neuralDrawbackVocabulary",
      "neuralCoveredDrawbackCount",
      "unresolvedExternalConstraintIds",
      "members",
      "calibration",
    ]) &&
    value["mode"] === "hybrid-v21-ensemble" &&
    value["modelFormatVersion"] === 4 &&
    isDigest(value["artifactSha256"]) &&
    isDigest(value["sourceEnsembleReleaseSha256"]) &&
    isDigest(value["sourceFusionSelectionSha256"]) &&
    isDigest(value["sourceCalibrationSha256"]) &&
    value["featureSchemaVersion"] === 1 &&
    value["symbolicFeatureVersion"] === 6 &&
    value["fusionMethod"] ===
      "rank-preserving-bounded-residual-plus-symbolic-prior-v1" &&
    typeof value["selectedAlpha"] === "number" &&
    Number.isFinite(value["selectedAlpha"]) &&
    value["selectedAlpha"] >= 0 &&
    value["selectedAlpha"] <= 1 &&
    isExactRuleSet(value["neuralDrawbackVocabulary"], CATALOG_RULE_IDS) &&
    value["neuralCoveredDrawbackCount"] === CATALOG_RULE_IDS.size &&
    hasExpectedUnresolvedConstraints(
      value["unresolvedExternalConstraintIds"],
      enriched,
    ) &&
    Array.isArray(members) &&
    members.length === 3 &&
    members.every(
      (member, index) =>
        isRecord(member) &&
        hasExactKeys(member, [
          "trainingSeed",
          "sourceCheckpointSha256",
          "sourceSelectionSha256",
          "trainingRunId",
          "selectedEpoch",
        ]) &&
        member["trainingSeed"] ===
          [20260811, 20260812, 20260813][index] &&
        isDigest(member["sourceCheckpointSha256"]) &&
        isDigest(member["sourceSelectionSha256"]) &&
        isDigest(member["trainingRunId"]) &&
        typeof member["selectedEpoch"] === "number" &&
        Number.isSafeInteger(member["selectedEpoch"]) &&
        member["selectedEpoch"] > 0,
    ) &&
    hasUniqueMemberProvenance(members) &&
    isRecord(calibration) &&
    hasExactKeys(calibration, [
      "preservesHardEliminations",
      "white",
      "black",
    ]) &&
    calibration["preservesHardEliminations"] === true &&
    isCalibrationHead(calibration["white"]) &&
    isCalibrationHead(calibration["black"])
  );
}

function isResult(value: unknown): value is PgnAnalysisResult {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "sourceBinding",
      "plyCount",
      "finalFen",
      "finalWhite",
      "finalBlack",
      "history",
      "coverage",
      "representedDrawbackCount",
      "representedDrawbackIds",
      "catalogDrawbackCount",
      "unavailableDrawbacks",
      "evaluatorEvidence",
      "predictor",
    ])
  ) {
    return false;
  }
  const source = value["sourceBinding"];
  const predictor = value["predictor"];
  const history = value["history"];
  const evaluatorEvidence = value["evaluatorEvidence"];
  if (!isEvaluatorEvidence(evaluatorEvidence)) {
    return false;
  }
  const enriched = evaluatorEvidence.mode === "authenticated-sidecar";
  const expectedRules = enriched ? CATALOG_RULE_IDS : REPRESENTED_RULE_IDS;
  return (
    isRecord(source) &&
    hasExactKeys(source, ["headers", "normalizedMainline"]) &&
    isRecord(source["headers"]) &&
    Object.values(source["headers"]).every(
      (item) => typeof item === "string",
    ) &&
    isStringArray(source["normalizedMainline"], 600) &&
    typeof value["plyCount"] === "number" &&
    Number.isSafeInteger(value["plyCount"]) &&
    value["plyCount"] >= 1 &&
    value["plyCount"] <= 600 &&
    typeof value["finalFen"] === "string" &&
    isExactGuessArray(value["finalWhite"], expectedRules) &&
    isExactGuessArray(value["finalBlack"], expectedRules) &&
    Array.isArray(history) &&
    history.length === value["plyCount"] &&
    history.every(
      (point, index) =>
        isRecord(point) &&
        hasExactKeys(point, [
          "ply",
          "moveNumber",
          "color",
          "san",
          "fenBefore",
          "white",
          "black",
          "eliminations",
        ]) &&
        typeof point["ply"] === "number" &&
        Number.isSafeInteger(point["ply"]) &&
        point["ply"] === index + 1 &&
        typeof point["moveNumber"] === "number" &&
        Number.isSafeInteger(point["moveNumber"]) &&
        point["moveNumber"] >= 1 &&
        (point["color"] === "white" || point["color"] === "black") &&
        typeof point["san"] === "string" &&
        typeof point["fenBefore"] === "string" &&
        isExactGuessArray(point["white"], expectedRules) &&
        isExactGuessArray(point["black"], expectedRules) &&
        Array.isArray(point["eliminations"]) &&
        point["eliminations"].every(
          (item) =>
            isRecord(item) &&
            hasExactKeys(item, ["color", "drawbackId", "reason"]) &&
            (item["color"] === "white" || item["color"] === "black") &&
            item["color"] === point["color"] &&
            typeof item["drawbackId"] === "string" &&
            expectedRules.has(item["drawbackId"]) &&
            typeof item["reason"] === "string" &&
            item["reason"].length > 0,
        ),
    ) &&
    Array.isArray(value["coverage"]) &&
    value["representedDrawbackCount"] === expectedRules.size &&
    isExactRuleSet(value["representedDrawbackIds"], expectedRules) &&
    value["catalogDrawbackCount"] === CATALOG_RULE_IDS.size &&
    (
      enriched
        ? Array.isArray(value["unavailableDrawbacks"]) &&
          value["unavailableDrawbacks"].length === 0
        : isExactUnavailableDrawbacks(value["unavailableDrawbacks"])
    ) &&
    isRecord(predictor) &&
    (
      predictor["mode"] === "symbolic-only" ||
      (
        predictor["mode"] === "hybrid-v1" &&
        predictor["modelFormatVersion"] === 1 &&
        typeof predictor["artifactSha256"] === "string" &&
        /^[0-9a-f]{64}$/u.test(predictor["artifactSha256"]) &&
        typeof predictor["sourceCheckpointSha256"] === "string" &&
        /^[0-9a-f]{64}$/u.test(predictor["sourceCheckpointSha256"]) &&
        predictor["featureSchemaVersion"] === 1 &&
        isStringArray(predictor["neuralDrawbackVocabulary"], 200) &&
        typeof predictor["neuralEvidenceWeight"] === "number" &&
        predictor["neuralEvidenceWeight"] === 0.35
      ) ||
      (
        predictor["mode"] === "hybrid-v21" &&
        predictor["modelFormatVersion"] === 2 &&
        typeof predictor["artifactSha256"] === "string" &&
        /^[0-9a-f]{64}$/u.test(predictor["artifactSha256"]) &&
        typeof predictor["sourceCheckpointSha256"] === "string" &&
        /^[0-9a-f]{64}$/u.test(predictor["sourceCheckpointSha256"]) &&
        predictor["featureSchemaVersion"] === 1 &&
        predictor["symbolicFeatureVersion"] === 6 &&
        isExactRuleSet(
          predictor["neuralDrawbackVocabulary"],
          CATALOG_RULE_IDS,
        ) &&
        typeof predictor["neuralCoveredDrawbackCount"] === "number" &&
        predictor["neuralCoveredDrawbackCount"] ===
          DEFAULT_HYPOTHESIS_RULE_IDS.length &&
        hasExpectedUnresolvedConstraints(
          predictor["unresolvedExternalConstraintIds"],
          enriched,
        )
      ) ||
      (
        predictor["mode"] === "hybrid-v22" &&
        hasExactKeys(predictor, [
          "mode",
          "modelFormatVersion",
          "artifactSha256",
          "sourceCheckpointSha256",
          "featureSchemaVersion",
          "symbolicFeatureVersion",
          "sequenceObservationMode",
          "neuralDrawbackVocabulary",
          "neuralCoveredDrawbackCount",
          "unresolvedExternalConstraintIds",
        ]) &&
        predictor["modelFormatVersion"] === 3 &&
        typeof predictor["artifactSha256"] === "string" &&
        /^[0-9a-f]{64}$/u.test(predictor["artifactSha256"]) &&
        typeof predictor["sourceCheckpointSha256"] === "string" &&
        /^[0-9a-f]{64}$/u.test(predictor["sourceCheckpointSha256"]) &&
        predictor["featureSchemaVersion"] === 1 &&
        predictor["symbolicFeatureVersion"] === 6 &&
        (
          predictor["sequenceObservationMode"] === "masked-current-v2" ||
          predictor["sequenceObservationMode"] === "exact-current-v2"
        ) &&
        isExactRuleSet(
          predictor["neuralDrawbackVocabulary"],
          CATALOG_RULE_IDS,
        ) &&
        typeof predictor["neuralCoveredDrawbackCount"] === "number" &&
        predictor["neuralCoveredDrawbackCount"] ===
          DEFAULT_HYPOTHESIS_RULE_IDS.length &&
        hasExpectedUnresolvedConstraints(
          predictor["unresolvedExternalConstraintIds"],
          enriched,
        )
      ) ||
      (
        predictor["mode"] === "hybrid-v21-ensemble" &&
        isEnsemblePredictor(predictor, enriched)
      )
    )
  );
}

export function isPgnAnalysisWorkerRequest(
  value: unknown,
): value is PgnAnalysisWorkerRequest {
  if (
    !isRecord(value) ||
    !Number.isSafeInteger(value["requestId"]) ||
    typeof value["requestId"] !== "number" ||
    value["requestId"] <= 0
  ) {
    return false;
  }
  if (value["type"] === "load-model") {
    return (
      hasExactKeys(value, [
        "type",
        "requestId",
        "artifactText",
        "expectedSha256",
      ]) &&
      typeof value["artifactText"] === "string" &&
      typeof value["expectedSha256"] === "string" &&
      /^[0-9a-f]{64}$/u.test(value["expectedSha256"])
    );
  }
  if (value["type"] !== "analyze") {
    return false;
  }
  const hasModel = value["neuralArtifactSha256"] !== undefined;
  const hasEvaluatorBytes = value["evaluatorSidecarBytes"] !== undefined;
  const hasEvaluatorDigest = value["evaluatorSidecarSha256"] !== undefined;
  if (hasEvaluatorBytes !== hasEvaluatorDigest) {
    return false;
  }
  const expectedKeys = [
    "type",
    "requestId",
    "pgn",
    ...(hasModel ? ["neuralArtifactSha256"] : []),
    ...(hasEvaluatorBytes
      ? ["evaluatorSidecarBytes", "evaluatorSidecarSha256"]
      : []),
  ];
  return (
    hasExactKeys(value, expectedKeys) &&
    typeof value["pgn"] === "string" &&
    (
      !hasModel ||
      (
        typeof value["neuralArtifactSha256"] === "string" &&
        /^[0-9a-f]{64}$/u.test(value["neuralArtifactSha256"])
      )
    ) &&
    (
      !hasEvaluatorBytes ||
      (
        value["evaluatorSidecarBytes"] instanceof Uint8Array &&
        value["evaluatorSidecarBytes"].byteLength <=
          MAX_EVALUATOR_SIDECAR_BYTES &&
        isDigest(value["evaluatorSidecarSha256"])
      )
    )
  );
}

export function isPgnAnalysisWorkerResponse(
  value: unknown,
): value is PgnAnalysisWorkerResponse {
  if (
    !isRecord(value) ||
    !Number.isSafeInteger(value["requestId"]) ||
    typeof value["requestId"] !== "number" ||
    value["requestId"] <= 0
  ) {
    return false;
  }
  if (value["type"] === "progress") {
    const progress = value["progress"];
    return (
      hasExactKeys(value, ["type", "requestId", "progress"]) &&
      isRecord(progress) &&
      hasExactKeys(progress, ["processedPlies", "totalPlies"]) &&
      Number.isSafeInteger(progress["processedPlies"]) &&
      Number.isSafeInteger(progress["totalPlies"]) &&
      typeof progress["processedPlies"] === "number" &&
      typeof progress["totalPlies"] === "number" &&
      progress["processedPlies"] >= 0 &&
      progress["totalPlies"] > 0 &&
      progress["processedPlies"] <= progress["totalPlies"]
    );
  }
  if (value["type"] === "model-loaded") {
    const model = value["model"];
    return (
      hasExactKeys(value, ["type", "requestId", "model"]) &&
      isRecord(model) &&
      hasExactKeys(model, [
        "artifactSha256",
        "modelFormatVersion",
        "modelVariant",
        "drawbackCount",
      ]) &&
      typeof model["artifactSha256"] === "string" &&
      /^[0-9a-f]{64}$/u.test(model["artifactSha256"]) &&
      (
        model["modelFormatVersion"] === 1 ||
        model["modelFormatVersion"] === 2 ||
        model["modelFormatVersion"] === 3 ||
        model["modelFormatVersion"] === 4
      ) &&
      (
        model["modelVariant"] === "v1" ||
        model["modelVariant"] === "v21-hybrid" ||
        model["modelVariant"] === "v22-hybrid" ||
        model["modelVariant"] === "v21-hybrid-ensemble"
      ) &&
      (
        (model["modelFormatVersion"] === 1 &&
          model["modelVariant"] === "v1") ||
        (model["modelFormatVersion"] === 2 &&
          model["modelVariant"] === "v21-hybrid") ||
        (model["modelFormatVersion"] === 3 &&
          model["modelVariant"] === "v22-hybrid") ||
        (model["modelFormatVersion"] === 4 &&
          model["modelVariant"] === "v21-hybrid-ensemble")
      ) &&
      typeof model["drawbackCount"] === "number" &&
      Number.isSafeInteger(model["drawbackCount"]) &&
      model["drawbackCount"] > 0 &&
      model["drawbackCount"] <= CATALOG_RULE_IDS.size
    );
  }
  if (value["type"] === "result") {
    return (
      hasExactKeys(value, ["type", "requestId", "result"]) &&
      isResult(value["result"])
    );
  }
  if (value["type"] === "error") {
    const error = value["error"];
    return (
      hasExactKeys(value, ["type", "requestId", "error"]) &&
      isRecord(error) &&
      hasExactKeys(error, ["name", "message", "ply", "token"]) &&
      typeof error["name"] === "string" &&
      typeof error["message"] === "string" &&
      (error["ply"] === null ||
        (typeof error["ply"] === "number" &&
          Number.isSafeInteger(error["ply"]) &&
          error["ply"] >= 0)) &&
      (error["token"] === null || typeof error["token"] === "string")
    );
  }
  return false;
}
