export type PlayerColor = "white" | "black";

export interface PublicEvaluatorConstraint {
  readonly provider: "uci-best-move";
  readonly policyId: string;
  readonly positionKey: string;
  readonly requestDigest: string;
  readonly bestMoveUci: string;
  readonly engineFingerprint: string;
}

export interface PublicFeatureRecord {
  readonly fenBefore: string;
  readonly move: string;
  readonly moveNumber: number;
  readonly ply: number;
  readonly playerColor: PlayerColor;
  readonly historySan: readonly string[];
  readonly ordinaryLegalMoves: readonly string[];
  readonly clockMs: number | null;
  readonly symbolicFeatureVersion: number;
  readonly symbolicWhiteRuleProbabilities: readonly number[];
  readonly symbolicBlackRuleProbabilities: readonly number[];
  readonly symbolicWhiteEliminated: readonly boolean[];
  readonly symbolicBlackEliminated: readonly boolean[];
  readonly publicEvaluatorConstraint: PublicEvaluatorConstraint | null;
}

export interface TrainingLabelRecord {
  readonly playerColor: PlayerColor;
  readonly trueDrawback: string;
  readonly hiddenParameters: unknown;
  readonly drawbackInternalState: unknown;
  readonly drawbackLegalMoves: readonly string[];
  readonly ruleTriggered: boolean;
  readonly forced: boolean;
  readonly result: unknown;
}

export interface EvaluationMetadata {
  readonly gameId: string;
  readonly seed: number;
  readonly san: string;
  readonly botAgentId: string;
  readonly botStyle: string | null;
  readonly botStrength: number | null;
}

export interface ParsedDatasetRow {
  readonly features: PublicFeatureRecord;
  readonly labels: TrainingLabelRecord;
  readonly evaluation: EvaluationMetadata;
}

export interface DatasetSchema {
  readonly symbolicFeatureVersion: number;
  readonly symbolicRuleCount: number;
}

export const PUBLIC_FEATURE_KEYS = Object.freeze([
  "fenBefore",
  "move",
  "moveNumber",
  "ply",
  "playerColor",
  "historySan",
  "ordinaryLegalMoves",
  "clockMs",
  "symbolicFeatureVersion",
  "symbolicWhiteRuleProbabilities",
  "symbolicBlackRuleProbabilities",
  "symbolicWhiteEliminated",
  "symbolicBlackEliminated",
  "publicEvaluatorConstraint",
] as const);

export const FORBIDDEN_FEATURE_KEYS = Object.freeze([
  "trueDrawback",
  "hiddenParameters",
  "drawbackInternalState",
  "drawbackLegalMoves",
  "ruleTriggered",
  "forced",
  "result",
] as const);

export const EVALUATION_ONLY_KEYS = Object.freeze([
  "gameId",
  "seed",
  "san",
  "botAgentId",
  "botStyle",
  "botStrength",
] as const);

const DATASET_ROW_KEYS = Object.freeze([
  ...PUBLIC_FEATURE_KEYS,
  ...FORBIDDEN_FEATURE_KEYS,
  ...EVALUATION_ONLY_KEYS,
] as const);

const UCI_MOVE = /^[a-h][1-8][a-h][1-8][nbrq]?$/u;
const SHA256 = /^[0-9a-f]{64}$/u;

export class DatasetContractError extends TypeError {
  public constructor(message: string) {
    super(message);
    this.name = "DatasetContractError";
  }
}

function record(
  value: unknown,
  label: string,
): Readonly<Record<string, unknown>> {
  if (
    typeof value !== "object"
    || value === null
    || Array.isArray(value)
    || Object.getPrototypeOf(value) !== Object.prototype
  ) {
    throw new DatasetContractError(`${label} must be a plain object`);
  }
  return value as Readonly<Record<string, unknown>>;
}

function assertExactKeys(
  value: Readonly<Record<string, unknown>>,
  expected: readonly string[],
  label: string,
): void {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  if (
    actual.length !== sortedExpected.length
    || actual.some((key, index) => key !== sortedExpected[index])
  ) {
    const missing = sortedExpected.filter((key) => !actual.includes(key));
    const unknown = actual.filter((key) => !sortedExpected.includes(key));
    throw new DatasetContractError(
      `${label} keys are invalid`
      + (missing.length === 0 ? "" : `; missing ${missing.join(", ")}`)
      + (unknown.length === 0 ? "" : `; unknown ${unknown.join(", ")}`),
    );
  }
}

function requiredString(
  value: Readonly<Record<string, unknown>>,
  key: string,
): string {
  const item = value[key];
  if (typeof item !== "string" || item.length === 0) {
    throw new DatasetContractError(`${key} must be a non-empty string`);
  }
  return item;
}

function nonNegativeInteger(
  value: unknown,
  label: string,
): number {
  if (
    typeof value !== "number"
    || !Number.isSafeInteger(value)
    || value < 0
  ) {
    throw new DatasetContractError(
      `${label} must be a non-negative safe integer`,
    );
  }
  return value;
}

function stringArray(value: unknown, label: string): readonly string[] {
  if (!Array.isArray(value)) {
    throw new DatasetContractError(`${label} must be an array of strings`);
  }
  const strings: string[] = [];
  for (const item of value as readonly unknown[]) {
    if (typeof item !== "string") {
      throw new DatasetContractError(`${label} must be an array of strings`);
    }
    strings.push(item);
  }
  return Object.freeze(strings);
}

function probabilityArray(
  value: unknown,
  ruleCount: number,
  label: string,
): readonly number[] {
  if (
    !Array.isArray(value)
    || value.length !== ruleCount
    || value.some(
      (item) =>
        typeof item !== "number"
        || !Number.isFinite(item)
        || item < 0
        || item > 1,
    )
  ) {
    throw new DatasetContractError(
      `${label} must contain ${String(ruleCount)} finite probabilities`,
    );
  }
  const probabilities = value as number[];
  const sum = probabilities.reduce((total, item) => total + item, 0);
  if (sum !== 0 && Math.abs(sum - 1) > 1e-6) {
    throw new DatasetContractError(`${label} must sum to one or zero`);
  }
  return Object.freeze([...probabilities]);
}

function booleanArray(
  value: unknown,
  ruleCount: number,
  label: string,
): readonly boolean[] {
  if (!Array.isArray(value) || value.length !== ruleCount) {
    throw new DatasetContractError(
      `${label} must contain ${String(ruleCount)} booleans`,
    );
  }
  const booleans: boolean[] = [];
  for (const item of value as readonly unknown[]) {
    if (typeof item !== "boolean") {
      throw new DatasetContractError(
        `${label} must contain ${String(ruleCount)} booleans`,
      );
    }
    booleans.push(item);
  }
  return Object.freeze(booleans);
}

function evaluatorConstraint(value: unknown): PublicEvaluatorConstraint | null {
  if (value === null) {
    return null;
  }
  const item = record(value, "publicEvaluatorConstraint");
  assertExactKeys(
    item,
    [
      "provider",
      "policyId",
      "positionKey",
      "requestDigest",
      "bestMoveUci",
      "engineFingerprint",
    ],
    "publicEvaluatorConstraint",
  );
  if (item["provider"] !== "uci-best-move") {
    throw new DatasetContractError(
      "publicEvaluatorConstraint.provider must be uci-best-move",
    );
  }
  const bestMoveUci = requiredString(item, "bestMoveUci");
  const requestDigest = requiredString(item, "requestDigest");
  if (!UCI_MOVE.test(bestMoveUci)) {
    throw new DatasetContractError(
      "publicEvaluatorConstraint.bestMoveUci must be a UCI move",
    );
  }
  if (!SHA256.test(requestDigest)) {
    throw new DatasetContractError(
      "publicEvaluatorConstraint.requestDigest must be lowercase SHA-256",
    );
  }
  return Object.freeze({
    provider: "uci-best-move",
    policyId: requiredString(item, "policyId"),
    positionKey: requiredString(item, "positionKey"),
    requestDigest,
    bestMoveUci,
    engineFingerprint: requiredString(item, "engineFingerprint"),
  });
}

function checkedSchema(schema: DatasetSchema): DatasetSchema {
  if (
    !Number.isSafeInteger(schema.symbolicFeatureVersion)
    || schema.symbolicFeatureVersion <= 0
    || !Number.isSafeInteger(schema.symbolicRuleCount)
    || schema.symbolicRuleCount <= 0
  ) {
    throw new DatasetContractError(
      "symbolic feature version and rule count must be positive safe integers",
    );
  }
  return schema;
}

export function parsePublicFeatureRecord(
  input: unknown,
  schemaInput: DatasetSchema,
): PublicFeatureRecord {
  const schema = checkedSchema(schemaInput);
  const value = record(input, "public feature record");
  const leaked = FORBIDDEN_FEATURE_KEYS.filter((key) =>
    Object.hasOwn(value, key)
  );
  const evaluation = EVALUATION_ONLY_KEYS.filter((key) =>
    Object.hasOwn(value, key)
  );
  if (leaked.length > 0 || evaluation.length > 0) {
    throw new DatasetContractError(
      `non-feature fields cannot enter model input: ${[
        ...leaked,
        ...evaluation,
      ].join(", ")}`,
    );
  }
  assertExactKeys(value, PUBLIC_FEATURE_KEYS, "public feature record");

  const playerColor = value["playerColor"];
  if (playerColor !== "white" && playerColor !== "black") {
    throw new DatasetContractError("playerColor must be white or black");
  }
  const move = requiredString(value, "move");
  if (!UCI_MOVE.test(move)) {
    throw new DatasetContractError("move must be canonical UCI");
  }
  const moveNumber = nonNegativeInteger(value["moveNumber"], "moveNumber");
  if (moveNumber < 1) {
    throw new DatasetContractError("moveNumber must be at least one");
  }
  const symbolicFeatureVersion = nonNegativeInteger(
    value["symbolicFeatureVersion"],
    "symbolicFeatureVersion",
  );
  if (symbolicFeatureVersion !== schema.symbolicFeatureVersion) {
    throw new DatasetContractError(
      `symbolicFeatureVersion must be ${String(schema.symbolicFeatureVersion)}`,
    );
  }
  const rawClock = value["clockMs"];
  const clockMs = rawClock === null
    ? null
    : nonNegativeInteger(rawClock, "clockMs");

  return Object.freeze({
    fenBefore: requiredString(value, "fenBefore"),
    move,
    moveNumber,
    ply: nonNegativeInteger(value["ply"], "ply"),
    playerColor,
    historySan: stringArray(value["historySan"], "historySan"),
    ordinaryLegalMoves: stringArray(
      value["ordinaryLegalMoves"],
      "ordinaryLegalMoves",
    ),
    clockMs,
    symbolicFeatureVersion,
    symbolicWhiteRuleProbabilities: probabilityArray(
      value["symbolicWhiteRuleProbabilities"],
      schema.symbolicRuleCount,
      "symbolicWhiteRuleProbabilities",
    ),
    symbolicBlackRuleProbabilities: probabilityArray(
      value["symbolicBlackRuleProbabilities"],
      schema.symbolicRuleCount,
      "symbolicBlackRuleProbabilities",
    ),
    symbolicWhiteEliminated: booleanArray(
      value["symbolicWhiteEliminated"],
      schema.symbolicRuleCount,
      "symbolicWhiteEliminated",
    ),
    symbolicBlackEliminated: booleanArray(
      value["symbolicBlackEliminated"],
      schema.symbolicRuleCount,
      "symbolicBlackEliminated",
    ),
    publicEvaluatorConstraint: evaluatorConstraint(
      value["publicEvaluatorConstraint"],
    ),
  });
}

export function parseDatasetRow(
  input: unknown,
  schema: DatasetSchema,
): ParsedDatasetRow {
  const row = record(input, "dataset row");
  assertExactKeys(row, DATASET_ROW_KEYS, "dataset row");
  const featureInput = Object.fromEntries(
    PUBLIC_FEATURE_KEYS.map((key) => [key, row[key]]),
  );
  const features = parsePublicFeatureRecord(featureInput, schema);
  const trueDrawback = requiredString(row, "trueDrawback");
  const ruleTriggered = row["ruleTriggered"];
  const forced = row["forced"];
  if (typeof ruleTriggered !== "boolean" || typeof forced !== "boolean") {
    throw new DatasetContractError("ruleTriggered and forced must be booleans");
  }
  const botStyle = row["botStyle"];
  const botStrength = row["botStrength"];
  if (botStyle !== null && (typeof botStyle !== "string" || botStyle === "")) {
    throw new DatasetContractError(
      "botStyle must be a non-empty string or null",
    );
  }
  if (
    botStrength !== null
    && (
      typeof botStrength !== "number"
      || !Number.isSafeInteger(botStrength)
      || botStrength < 0
    )
  ) {
    throw new DatasetContractError(
      "botStrength must be a non-negative safe integer or null",
    );
  }
  const seed = nonNegativeInteger(row["seed"], "seed");
  if (seed > 0xffff_ffff) {
    throw new DatasetContractError("seed must be an unsigned 32-bit integer");
  }

  return Object.freeze({
    features,
    labels: Object.freeze({
      playerColor: features.playerColor,
      trueDrawback,
      hiddenParameters: structuredClone(row["hiddenParameters"]),
      drawbackInternalState: structuredClone(row["drawbackInternalState"]),
      drawbackLegalMoves: stringArray(
        row["drawbackLegalMoves"],
        "drawbackLegalMoves",
      ),
      ruleTriggered,
      forced,
      result: structuredClone(row["result"]),
    }),
    evaluation: Object.freeze({
      gameId: requiredString(row, "gameId"),
      seed,
      san: requiredString(row, "san"),
      botAgentId: requiredString(row, "botAgentId"),
      botStyle,
      botStrength,
    }),
  });
}
