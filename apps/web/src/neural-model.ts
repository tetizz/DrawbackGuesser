import type { PlayerColor } from "@drawbackengine/shared";
import { DEFAULT_HYPOTHESIS_RULE_IDS } from "@drawbackguesser/predictor";
import {
  parseHybridBrowserModel,
  parseHybridV22BrowserModel,
  runHybridBrowserModel,
  type HybridBrowserModel,
  type HybridObservation,
  type HybridV22BrowserModel,
} from "./sequence-neural-model.js";
import {
  parseEnsembleBrowserModel,
  runEnsembleBrowserModel,
  type EnsembleBrowserModel,
} from "./ensemble-neural-model.js";

export const BROWSER_MODEL_FORMAT = "drawbacktrainer-browser-model";
export const BROWSER_MODEL_FORMAT_VERSION = 1;
export const NEURAL_FEATURE_SCHEMA_VERSION = 1;
export const MAX_BROWSER_HIDDEN_DIMENSION = 256;

const PIECES = "PNBRQKpnbrqk";
const BOARD_FEATURE_COUNT = 64 * PIECES.length;
const SCALAR_FEATURE_COUNT = 24;
export const NEURAL_FEATURE_DIMENSION =
  BOARD_FEATURE_COUNT + SCALAR_FEATURE_COUNT;

type DenseTensorName =
  | "encoder.0.weight"
  | "encoder.0.bias"
  | "encoder.2.weight"
  | "encoder.2.bias"
  | "white_drawback.weight"
  | "white_drawback.bias"
  | "black_drawback.weight"
  | "black_drawback.bias";

export interface BrowserModelTensor {
  readonly shape: readonly number[];
  readonly values: readonly number[];
}

export interface BrowserV1NeuralModel {
  readonly format: typeof BROWSER_MODEL_FORMAT;
  readonly formatVersion: typeof BROWSER_MODEL_FORMAT_VERSION;
  readonly modelVariant: "v1";
  readonly featureSchemaVersion: typeof NEURAL_FEATURE_SCHEMA_VERSION;
  readonly sourceCheckpointSha256: string;
  readonly drawbackVocabulary: readonly string[];
  readonly dimensions: {
    readonly input: number;
    readonly hidden: number;
    readonly drawbackClasses: number;
  };
  readonly tensors: Readonly<Record<DenseTensorName, BrowserModelTensor>>;
}

export type BrowserNeuralModel =
  | BrowserV1NeuralModel
  | HybridBrowserModel
  | HybridV22BrowserModel
  | EnsembleBrowserModel;

export interface NeuralObservation {
  readonly fenBefore: string;
  readonly move: string;
  readonly moveNumber: number;
  readonly ply: number;
  readonly playerColor: PlayerColor;
  readonly historySan: readonly string[];
  readonly ordinaryLegalMoveCount: number;
}

export interface NeuralDrawbackProbabilities {
  readonly white: Readonly<Record<string, number>>;
  readonly black: Readonly<Record<string, number>>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function finiteNumberArray(value: unknown): value is readonly number[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) =>
        typeof item === "number" &&
        Number.isFinite(item),
    )
  );
}

function nonEmptyStringArray(value: unknown): value is readonly string[] {
  return (
    Array.isArray(value) &&
    value.every((item) => typeof item === "string" && item.length > 0)
  );
}

function requireInteger(
  value: unknown,
  name: string,
  minimum = 1,
): number {
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < minimum
  ) {
    throw new TypeError(`${name} must be an integer of at least ${String(minimum)}.`);
  }
  return value;
}

function tensorSize(shape: readonly number[]): number {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function parseTensor(
  tensors: Record<string, unknown>,
  name: DenseTensorName,
  expectedShape: readonly number[],
): BrowserModelTensor {
  const raw = tensors[name];
  if (!isRecord(raw) || !finiteNumberArray(raw["shape"])) {
    throw new TypeError(`Model tensor ${name} has an invalid shape.`);
  }
  const shape = raw["shape"];
  if (
    shape.length !== expectedShape.length ||
    shape.some(
      (dimension, index) =>
        !Number.isInteger(dimension) ||
        dimension !== expectedShape[index],
    )
  ) {
    throw new TypeError(`Model tensor ${name} has an unexpected shape.`);
  }
  const values = raw["values"];
  if (!finiteNumberArray(values) || values.length !== tensorSize(shape)) {
    throw new TypeError(`Model tensor ${name} has invalid values.`);
  }
  return Object.freeze({
    shape: Object.freeze([...shape]),
    values: Object.freeze([...values]),
  });
}

export function parseBrowserNeuralModel(value: unknown): BrowserNeuralModel {
  if (!isRecord(value)) {
    throw new TypeError("Browser model must be an object.");
  }
  if (
    value["formatVersion"] === 4 ||
    value["modelVariant"] === "v21-hybrid-ensemble"
  ) {
    return parseEnsembleBrowserModel(value, NEURAL_FEATURE_DIMENSION);
  }
  if (
    value["formatVersion"] === 3 ||
    value["modelVariant"] === "v22-hybrid"
  ) {
    return parseHybridV22BrowserModel(value, NEURAL_FEATURE_DIMENSION);
  }
  if (
    value["formatVersion"] === 2 ||
    value["modelVariant"] === "v21-hybrid"
  ) {
    return parseHybridBrowserModel(value, NEURAL_FEATURE_DIMENSION);
  }
  if (
    value["format"] !== BROWSER_MODEL_FORMAT ||
    value["formatVersion"] !== BROWSER_MODEL_FORMAT_VERSION ||
    value["modelVariant"] !== "v1" ||
    value["featureSchemaVersion"] !== NEURAL_FEATURE_SCHEMA_VERSION
  ) {
    throw new TypeError("Browser model format or feature schema is unsupported.");
  }
  const digest = value["sourceCheckpointSha256"];
  if (
    typeof digest !== "string" ||
    !/^[0-9a-f]{64}$/u.test(digest)
  ) {
    throw new TypeError("Browser model checkpoint digest is invalid.");
  }
  const vocabulary = value["drawbackVocabulary"];
  if (
    !nonEmptyStringArray(vocabulary) ||
    vocabulary.length === 0 ||
    new Set(vocabulary).size !== vocabulary.length
  ) {
    throw new TypeError("Browser model drawback vocabulary is invalid.");
  }
  const supportedIds = new Set(DEFAULT_HYPOTHESIS_RULE_IDS);
  for (const id of vocabulary) {
    if (!supportedIds.has(id as (typeof DEFAULT_HYPOTHESIS_RULE_IDS)[number])) {
      throw new TypeError(`Browser model contains unknown drawback ID: ${id}.`);
    }
  }
  const dimensions = value["dimensions"];
  if (!isRecord(dimensions)) {
    throw new TypeError("Browser model dimensions are missing.");
  }
  const input = requireInteger(dimensions["input"], "dimensions.input");
  const hidden = requireInteger(dimensions["hidden"], "dimensions.hidden");
  if (hidden > MAX_BROWSER_HIDDEN_DIMENSION) {
    throw new TypeError("Browser model hidden dimension exceeds the runtime limit.");
  }
  const drawbackClasses = requireInteger(
    dimensions["drawbackClasses"],
    "dimensions.drawbackClasses",
  );
  if (
    input !== NEURAL_FEATURE_DIMENSION ||
    drawbackClasses !== vocabulary.length
  ) {
    throw new TypeError("Browser model dimensions do not match its schema.");
  }
  const rawTensors = value["tensors"];
  if (!isRecord(rawTensors)) {
    throw new TypeError("Browser model tensors are missing.");
  }
  const expectedNames: readonly DenseTensorName[] = [
    "encoder.0.weight",
    "encoder.0.bias",
    "encoder.2.weight",
    "encoder.2.bias",
    "white_drawback.weight",
    "white_drawback.bias",
    "black_drawback.weight",
    "black_drawback.bias",
  ];
  if (
    Object.keys(rawTensors).length !== expectedNames.length ||
    Object.keys(rawTensors).some(
      (name) => !expectedNames.includes(name as DenseTensorName),
    )
  ) {
    throw new TypeError("Browser model contains unexpected tensors.");
  }
  const tensors = Object.freeze({
    "encoder.0.weight": parseTensor(
      rawTensors,
      "encoder.0.weight",
      [hidden, input],
    ),
    "encoder.0.bias": parseTensor(
      rawTensors,
      "encoder.0.bias",
      [hidden],
    ),
    "encoder.2.weight": parseTensor(
      rawTensors,
      "encoder.2.weight",
      [hidden, hidden],
    ),
    "encoder.2.bias": parseTensor(
      rawTensors,
      "encoder.2.bias",
      [hidden],
    ),
    "white_drawback.weight": parseTensor(
      rawTensors,
      "white_drawback.weight",
      [drawbackClasses, hidden],
    ),
    "white_drawback.bias": parseTensor(
      rawTensors,
      "white_drawback.bias",
      [drawbackClasses],
    ),
    "black_drawback.weight": parseTensor(
      rawTensors,
      "black_drawback.weight",
      [drawbackClasses, hidden],
    ),
    "black_drawback.bias": parseTensor(
      rawTensors,
      "black_drawback.bias",
      [drawbackClasses],
    ),
  });
  return Object.freeze({
    format: BROWSER_MODEL_FORMAT,
    formatVersion: BROWSER_MODEL_FORMAT_VERSION,
    modelVariant: "v1",
    featureSchemaVersion: NEURAL_FEATURE_SCHEMA_VERSION,
    sourceCheckpointSha256: digest,
    drawbackVocabulary: Object.freeze([...vocabulary]),
    dimensions: Object.freeze({ input, hidden, drawbackClasses }),
    tensors,
  });
}

function squareIndex(square: string): number {
  return (Number(square[1]) - 1) * 8 + square.charCodeAt(0) - 97;
}

function moveIndex(move: string): number {
  if (!/^[a-h][1-8][a-h][1-8][nbrq]?$/u.test(move)) {
    throw new TypeError(`Invalid UCI move: ${move}`);
  }
  const promotion = new Map([
    ["n", 1],
    ["b", 2],
    ["r", 3],
    ["q", 4],
  ]).get(move.slice(4)) ?? 0;
  return (
    (squareIndex(move.slice(0, 2)) * 64 + squareIndex(move.slice(2, 4))) * 5 +
    promotion
  );
}

export function buildNeuralFeatureVector(
  observation: NeuralObservation,
): readonly number[] {
  const fields = observation.fenBefore.split(" ");
  if (fields.length !== 6) {
    throw new TypeError("FEN must contain six fields.");
  }
  const [board, turn, castling, enPassant, halfmove, fullmove] = fields;
  if (
    board === undefined ||
    turn === undefined ||
    castling === undefined ||
    enPassant === undefined ||
    halfmove === undefined ||
    fullmove === undefined
  ) {
    throw new TypeError("FEN fields are incomplete.");
  }
  const features = Array<number>(BOARD_FEATURE_COUNT).fill(0);
  const ranks = board.split("/");
  if (ranks.length !== 8) {
    throw new TypeError("FEN board must contain eight ranks.");
  }
  for (const [fenRank, contents] of ranks.entries()) {
    let file = 0;
    for (const token of contents) {
      if (/^[1-8]$/u.test(token)) {
        file += Number(token);
        continue;
      }
      const piece = PIECES.indexOf(token);
      if (piece < 0 || file >= 8) {
        throw new TypeError("FEN contains an invalid board token.");
      }
      const square = (7 - fenRank) * 8 + file;
      features[piece * 64 + square] = 1;
      file += 1;
    }
    if (file !== 8) {
      throw new TypeError("FEN rank does not contain eight squares.");
    }
  }
  if (turn !== "w" && turn !== "b") {
    throw new TypeError("FEN side to move is invalid.");
  }
  const halfmoveValue = Number(halfmove);
  const fullmoveValue = Number(fullmove);
  if (
    !Number.isInteger(halfmoveValue) ||
    halfmoveValue < 0 ||
    !Number.isInteger(fullmoveValue) ||
    fullmoveValue < 1
  ) {
    throw new TypeError("FEN counters are invalid.");
  }
  const enPassantFeatures = Array<number>(9).fill(0);
  if (enPassant === "-") {
    enPassantFeatures[8] = 1;
  } else if (/^[a-h][36]$/u.test(enPassant)) {
    enPassantFeatures[enPassant.charCodeAt(0) - 97] = 1;
  } else {
    throw new TypeError("FEN en-passant square is invalid.");
  }
  const encodedMove = moveIndex(observation.move);
  const from = Math.floor(encodedMove / (64 * 5));
  const to = Math.floor(encodedMove / 5) % 64;
  features.push(
    turn === "w" ? 1 : 0,
    turn === "b" ? 1 : 0,
    ..."KQkq".split("").map((right) => castling.includes(right) ? 1 : 0),
    ...enPassantFeatures,
    Math.min(halfmoveValue, 100) / 100,
    Math.min(fullmoveValue, 300) / 300,
    observation.playerColor === "white" ? 1 : 0,
    Math.min(observation.ply, 600) / 600,
    Math.min(observation.moveNumber, 300) / 300,
    Math.min(observation.historySan.length, 600) / 600,
    Math.min(observation.ordinaryLegalMoveCount, 218) / 218,
    from / 63,
    to / 63,
  );
  if (features.length !== NEURAL_FEATURE_DIMENSION) {
    throw new Error("Neural feature dimension invariant violated.");
  }
  return Object.freeze(features);
}

function dense(
  input: readonly number[],
  weight: BrowserModelTensor,
  bias: BrowserModelTensor,
  relu: boolean,
): number[] {
  const rows = weight.shape[0];
  const columns = weight.shape[1];
  if (rows === undefined || columns === undefined) {
    throw new Error("Dense tensor shape invariant violated.");
  }
  const output = Array<number>(rows);
  for (let row = 0; row < rows; row += 1) {
    let value = bias.values[row] ?? 0;
    const offset = row * columns;
    for (let column = 0; column < columns; column += 1) {
      value += (weight.values[offset + column] ?? 0) * (input[column] ?? 0);
    }
    output[row] = relu && value < 0 ? 0 : value;
  }
  return output;
}

function softmax(logits: readonly number[]): number[] {
  const maximum = Math.max(...logits);
  const exponentials = logits.map((logit) => Math.exp(logit - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    throw new Error("Neural model produced an invalid distribution.");
  }
  return exponentials.map((value) => value / total);
}

export function runBrowserNeuralModel(
  model: BrowserNeuralModel,
  observation: NeuralObservation | HybridObservation,
): NeuralDrawbackProbabilities {
  const input = buildNeuralFeatureVector(observation);
  if (model.modelVariant === "v21-hybrid-ensemble") {
    if (!("symbolic" in observation)) {
      throw new TypeError(
        "v21-hybrid-ensemble inference requires symbolic evidence.",
      );
    }
    return runEnsembleBrowserModel(model, observation, input);
  }
  if (
    model.modelVariant === "v21-hybrid" ||
    model.modelVariant === "v22-hybrid"
  ) {
    if (!("symbolic" in observation)) {
      throw new TypeError(
        `${model.modelVariant} inference requires symbolic evidence.`,
      );
    }
    return runHybridBrowserModel(model, observation, input);
  }
  const first = dense(
    input,
    model.tensors["encoder.0.weight"],
    model.tensors["encoder.0.bias"],
    true,
  );
  const encoded = dense(
    first,
    model.tensors["encoder.2.weight"],
    model.tensors["encoder.2.bias"],
    true,
  );
  const probabilities = (color: PlayerColor): Readonly<Record<string, number>> => {
    const logits = dense(
      encoded,
      model.tensors[`${color}_drawback.weight`],
      model.tensors[`${color}_drawback.bias`],
      false,
    );
    const distribution = softmax(logits);
    return Object.freeze(
      Object.fromEntries(
        model.drawbackVocabulary.map((id, index) => [
          id,
          distribution[index] ?? 0,
        ]),
      ),
    );
  };
  return Object.freeze({
    white: probabilities("white"),
    black: probabilities("black"),
  });
}
