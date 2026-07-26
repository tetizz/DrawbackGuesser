import type {
  ChessMove,
  PositionAuthorityId,
  PositionView,
} from "@drawbackengine/drawback-engine";
import type { MoveObservation, PredictionState } from "@drawbackguesser/predictor";

const MAX_LIVE_PREDICTION_PLIES = 600;
const MAX_HYPOTHESES_PER_COLOR = 20_000;

export interface LivePredictionObservationInput {
  readonly authorityId: PositionAuthorityId;
  readonly color: MoveObservation["color"];
  readonly positionBefore: PositionView;
  readonly positionAfter: PositionView;
  readonly move: ChessMove;
}

export type LivePredictionWorkerRequest =
  | {
      readonly type: "initialize";
      readonly sessionId: string;
      readonly revision: 0;
      readonly initialPosition: PositionView;
    }
  | {
      readonly type: "observe";
      readonly sessionId: string;
      readonly revision: number;
      readonly observation: LivePredictionObservationInput;
    }
  | {
      readonly type: "reconstruct";
      readonly sessionId: string;
      readonly revision: number;
      readonly initialPosition: PositionView;
      readonly observations: readonly LivePredictionObservationInput[];
    };

export interface SerializedLivePredictionError {
  readonly name: string;
  readonly message: string;
}

export type LivePredictionWorkerResponse =
  | {
      readonly type: "prediction";
      readonly sessionId: string;
      readonly revision: number;
      readonly prediction: PredictionState;
    }
  | {
      readonly type: "error";
      readonly sessionId: string;
      readonly revision: number;
      readonly error: SerializedLivePredictionError;
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

function isSessionId(value: unknown): value is string {
  return typeof value === "string" && /^[a-zA-Z0-9_-]{1,96}$/u.test(value);
}

function isRevision(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isSafeInteger(value) &&
    value >= 0 &&
    value <= MAX_LIVE_PREDICTION_PLIES
  );
}

function isMove(value: unknown): value is ChessMove {
  if (!isRecord(value)) {
    return false;
  }
  const optionalKeys = [
    ...(value["captured"] === undefined ? [] : ["captured"]),
    ...(value["promotion"] === undefined ? [] : ["promotion"]),
  ];
  return (
    hasExactKeys(value, [
      "from",
      "to",
      "color",
      "piece",
      "san",
      "flags",
      ...optionalKeys,
    ]) &&
    typeof value["from"] === "string" &&
    typeof value["to"] === "string" &&
    (value["color"] === "white" || value["color"] === "black") &&
    typeof value["piece"] === "string" &&
    typeof value["san"] === "string" &&
    typeof value["flags"] === "string" &&
    (value["captured"] === undefined || typeof value["captured"] === "string") &&
    (value["promotion"] === undefined ||
      typeof value["promotion"] === "string")
  );
}

function isPosition(value: unknown): value is PositionView {
  return (
    isRecord(value) &&
    hasExactKeys(value, ["fen", "turn", "ply", "history"]) &&
    typeof value["fen"] === "string" &&
    (value["turn"] === "white" || value["turn"] === "black") &&
    isRevision(value["ply"]) &&
    Array.isArray(value["history"]) &&
    value["history"].length === value["ply"] &&
    value["history"].every(isMove)
  );
}

export function isLivePredictionObservationInput(
  value: unknown,
): value is LivePredictionObservationInput {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      "authorityId",
      "color",
      "positionBefore",
      "positionAfter",
      "move",
    ]) &&
    (value["authorityId"] === "standard-chess/v1" ||
      value["authorityId"] === "capturable-king/v1") &&
    (value["color"] === "white" || value["color"] === "black") &&
    isPosition(value["positionBefore"]) &&
    isPosition(value["positionAfter"]) &&
    isMove(value["move"]) &&
    value["move"].color === value["color"] &&
    value["positionBefore"].turn === value["color"] &&
    value["positionAfter"].ply === value["positionBefore"].ply + 1
  );
}

export function isLivePredictionWorkerRequest(
  value: unknown,
): value is LivePredictionWorkerRequest {
  if (
    !isRecord(value) ||
    !isSessionId(value["sessionId"]) ||
    !isRevision(value["revision"])
  ) {
    return false;
  }
  if (value["type"] === "initialize") {
    return (
      hasExactKeys(value, [
        "type",
        "sessionId",
        "revision",
        "initialPosition",
      ]) &&
      value["revision"] === 0 &&
      isPosition(value["initialPosition"])
    );
  }
  if (value["type"] === "observe") {
    return (
      hasExactKeys(value, [
        "type",
        "sessionId",
        "revision",
        "observation",
      ]) &&
      value["revision"] > 0 &&
      isLivePredictionObservationInput(value["observation"]) &&
      value["observation"].positionAfter.ply === value["revision"]
    );
  }
  return (
    value["type"] === "reconstruct" &&
    hasExactKeys(value, [
      "type",
      "sessionId",
      "revision",
      "initialPosition",
      "observations",
    ]) &&
    isPosition(value["initialPosition"]) &&
    Array.isArray(value["observations"]) &&
    value["observations"].length === value["revision"] &&
    value["observations"].every(isLivePredictionObservationInput)
  );
}

function isEvidence(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  const optionalKeys = [
    ...(value["move"] === undefined ? [] : ["move"]),
    ...(value["weight"] === undefined ? [] : ["weight"]),
  ];
  return (
    hasExactKeys(value, ["ruleId", "kind", "message", ...optionalKeys]) &&
    typeof value["ruleId"] === "string" &&
    ["allowed", "eliminated", "triggered", "forced", "likelihood"].includes(
      String(value["kind"]),
    ) &&
    typeof value["message"] === "string" &&
    (value["move"] === undefined || isMove(value["move"])) &&
    (value["weight"] === undefined ||
      (typeof value["weight"] === "number" &&
        Number.isFinite(value["weight"])))
  );
}

function isDistribution(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasExactKeys(value, ["hypotheses"]) &&
    Array.isArray(value["hypotheses"]) &&
    value["hypotheses"].length <= MAX_HYPOTHESES_PER_COLOR &&
    value["hypotheses"].every(
      (hypothesis) =>
        isRecord(hypothesis) &&
        hasExactKeys(hypothesis, [
          "hypothesisId",
          "drawbackId",
          "parameters",
          "internalState",
          "logProbability",
          "eliminated",
          "evidence",
        ]) &&
        typeof hypothesis["hypothesisId"] === "string" &&
        typeof hypothesis["drawbackId"] === "string" &&
        isRecord(hypothesis["parameters"]) &&
        typeof hypothesis["logProbability"] === "number" &&
        !Number.isNaN(hypothesis["logProbability"]) &&
        typeof hypothesis["eliminated"] === "boolean" &&
        Array.isArray(hypothesis["evidence"]) &&
        hypothesis["evidence"].every(isEvidence),
    )
  );
}

function isPredictionState(value: unknown): value is PredictionState {
  return (
    isRecord(value) &&
    hasExactKeys(value, ["white", "black"]) &&
    isDistribution(value["white"]) &&
    isDistribution(value["black"])
  );
}

export function isLivePredictionWorkerResponse(
  value: unknown,
): value is LivePredictionWorkerResponse {
  if (
    !isRecord(value) ||
    !isSessionId(value["sessionId"]) ||
    !isRevision(value["revision"])
  ) {
    return false;
  }
  if (value["type"] === "prediction") {
    return (
      hasExactKeys(value, [
        "type",
        "sessionId",
        "revision",
        "prediction",
      ]) && isPredictionState(value["prediction"])
    );
  }
  return (
    value["type"] === "error" &&
    hasExactKeys(value, ["type", "sessionId", "revision", "error"]) &&
    isRecord(value["error"]) &&
    hasExactKeys(value["error"], ["name", "message"]) &&
    typeof value["error"]["name"] === "string" &&
    typeof value["error"]["message"] === "string"
  );
}

export function serializeLivePredictionError(
  error: unknown,
): SerializedLivePredictionError {
  return error instanceof Error
    ? { name: error.name, message: error.message }
    : { name: "Error", message: String(error) };
}
