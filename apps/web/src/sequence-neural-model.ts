import { DEFAULT_HYPOTHESIS_RULE_IDS } from "@drawbackguesser/predictor";
import type { PlayerColor } from "@drawbackengine/shared";
import { rankPreservingFusion } from "./rank-preserving-fusion.js";

export const HYBRID_BROWSER_MODEL_FORMAT_VERSION = 2;
export const HYBRID_V22_BROWSER_MODEL_FORMAT_VERSION = 3;
export const HYBRID_SYMBOLIC_FEATURE_VERSION = 6;
export const HYBRID_TENSOR_ENCODING = "float32-le-base64";
export const HYBRID_V22_MODEL_VARIANT = "v22-hybrid";
export const PUBLIC_SEQUENCE_TOKENIZER_KIND =
  "public-sequence-observation-token";
export const PUBLIC_SEQUENCE_TOKENIZER_VERSION = 2;
export const UNKNOWN_CURRENT_MOVE_TOKEN = "<unk-current-move>";
export const MASKED_CURRENT_MOVE_TOKEN = "<current-move-masked>";

export type SequenceObservationMode =
  | "masked-current-v2"
  | "exact-current-v2";

const MAX_DIMENSION = 256;
const MAX_HISTORY = 600;
const MAX_SEQUENCE = MAX_HISTORY + 1;
const MAX_VOCABULARY = 65_536;
const MAX_TENSOR_BYTES = 8 * 1024 * 1024;

const HYBRID_TENSOR_NAMES = [
  "board_encoder.0.weight",
  "board_encoder.0.bias",
  "board_encoder.2.weight",
  "board_encoder.2.bias",
  "san_embedding.weight",
  "history_encoder.weight_ih_l0",
  "history_encoder.weight_hh_l0",
  "history_encoder.bias_ih_l0",
  "history_encoder.bias_hh_l0",
  "symbolic_encoder.0.weight",
  "symbolic_encoder.0.bias",
  "symbolic_encoder.2.weight",
  "symbolic_encoder.2.bias",
  "white_drawback.weight",
  "white_drawback.bias",
  "black_drawback.weight",
  "black_drawback.bias",
] as const;

type HybridTensorName = (typeof HYBRID_TENSOR_NAMES)[number];

export interface HybridTensor {
  readonly shape: readonly number[];
  readonly values: Float32Array;
}

export interface HybridBrowserModel {
  readonly format: "drawbacktrainer-browser-model";
  readonly formatVersion: typeof HYBRID_BROWSER_MODEL_FORMAT_VERSION;
  readonly modelVariant: "v21-hybrid";
  readonly featureSchemaVersion: 1;
  readonly symbolicFeatureVersion: typeof HYBRID_SYMBOLIC_FEATURE_VERSION;
  readonly sourceCheckpointSha256: string;
  readonly drawbackVocabulary: readonly string[];
  readonly symbolicRuleIds: readonly string[];
  readonly tokenizer: {
    readonly kind: "exact-san-token";
    readonly version: 1;
    readonly vocabulary: readonly string[];
    readonly maxHistory: number;
    readonly padding: "right";
    readonly truncation: "keep-most-recent";
  };
  readonly tensorEncoding: typeof HYBRID_TENSOR_ENCODING;
  readonly dimensions: {
    readonly input: number;
    readonly boardHidden: number;
    readonly sanVocabulary: number;
    readonly sanEmbedding: number;
    readonly sequenceHidden: number;
    readonly symbolicInput: number;
    readonly symbolicHidden: number;
    readonly drawbackClasses: number;
  };
  readonly tensors: Readonly<Record<HybridTensorName, HybridTensor>>;
}

export interface HybridV22BrowserModel {
  readonly format: "drawbacktrainer-browser-model";
  readonly formatVersion: typeof HYBRID_V22_BROWSER_MODEL_FORMAT_VERSION;
  readonly modelVariant: typeof HYBRID_V22_MODEL_VARIANT;
  readonly featureSchemaVersion: 1;
  readonly symbolicFeatureVersion: typeof HYBRID_SYMBOLIC_FEATURE_VERSION;
  readonly sequenceObservationMode: SequenceObservationMode;
  readonly sourceCheckpointSha256: string;
  readonly drawbackVocabulary: readonly string[];
  readonly symbolicRuleIds: readonly string[];
  readonly tokenizer: {
    readonly kind: typeof PUBLIC_SEQUENCE_TOKENIZER_KIND;
    readonly version: typeof PUBLIC_SEQUENCE_TOKENIZER_VERSION;
    readonly vocabulary: readonly string[];
    readonly maxSequence: number;
    readonly padding: "right";
    readonly truncation: "keep-most-recent";
    readonly currentMove: "required-final-namespaced-uci";
  };
  readonly tensorEncoding: typeof HYBRID_TENSOR_ENCODING;
  readonly dimensions: HybridBrowserModel["dimensions"];
  readonly tensors: Readonly<Record<HybridTensorName, HybridTensor>>;
}

export type SingleHybridBrowserModel =
  | HybridBrowserModel
  | HybridV22BrowserModel;

export interface HybridSymbolicEvidence {
  readonly ruleIds: readonly string[];
  readonly whiteProbabilities: readonly number[];
  readonly blackProbabilities: readonly number[];
  readonly whiteEliminated: readonly boolean[];
  readonly blackEliminated: readonly boolean[];
}

export interface HybridObservation {
  readonly fenBefore: string;
  readonly move: string;
  readonly moveNumber: number;
  readonly ply: number;
  readonly playerColor: PlayerColor;
  readonly historySan: readonly string[];
  readonly ordinaryLegalMoveCount: number;
  readonly symbolic: HybridSymbolicEvidence;
}

export interface HybridOutput {
  readonly white: Readonly<Record<string, number>>;
  readonly black: Readonly<Record<string, number>>;
}

export interface HybridResidualOutput {
  readonly white: ArrayLike<number>;
  readonly black: ArrayLike<number>;
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

const CURRENT_MOVE_PATTERN = /^[a-h][1-8][a-h][1-8][nbrq]?$/u;
const CURRENT_MOVE_PREFIX = "<move:";

export function currentMoveObservationToken(move: string): string {
  if (
    typeof move !== "string" ||
    !CURRENT_MOVE_PATTERN.test(move) ||
    move.slice(0, 2) === move.slice(2, 4) ||
    (move.length === 5 && !["1", "8"].includes(move[3] ?? ""))
  ) {
    throw new TypeError("Current move must be canonical UCI.");
  }
  return `${CURRENT_MOVE_PREFIX}${move}>`;
}

function isValidPriorSan(value: unknown): value is readonly string[] {
  return (
    Array.isArray(value) &&
    value.every(
      (token: unknown) =>
        typeof token === "string" &&
        token.length > 0 &&
        token.length <= 32 &&
        token !== "<pad>" &&
        token !== "<unk>" &&
        token !== UNKNOWN_CURRENT_MOVE_TOKEN &&
        token !== MASKED_CURRENT_MOVE_TOKEN &&
        !token.startsWith(CURRENT_MOVE_PREFIX) &&
        !token.includes("<") &&
        !token.includes(">"),
    )
  );
}

function integer(value: unknown, name: string, maximum: number): number {
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value <= 0 ||
    value > maximum
  ) {
    throw new TypeError(`${name} is outside the browser runtime limit.`);
  }
  return value;
}

function stringList(value: unknown, name: string): readonly string[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new TypeError(`${name} must be a unique non-empty string list.`);
  }
  const items: string[] = [];
  for (const item of value) {
    if (typeof item !== "string" || item.length === 0) {
      throw new TypeError(`${name} must be a unique non-empty string list.`);
    }
    items.push(item);
  }
  if (new Set(items).size !== items.length) {
    throw new TypeError(`${name} must be a unique non-empty string list.`);
  }
  return items;
}

function tensorSize(shape: readonly number[]): number {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function decodeTensor(
  rawTensors: Record<string, unknown>,
  name: HybridTensorName,
  expectedShape: readonly number[],
): HybridTensor {
  const raw = rawTensors[name];
  if (!isRecord(raw) || !Array.isArray(raw["shape"])) {
    throw new TypeError(`Model tensor ${name} has an invalid shape.`);
  }
  const shape = raw["shape"];
  if (
    shape.length !== expectedShape.length ||
    shape.some(
      (dimension, index) =>
        typeof dimension !== "number" ||
        !Number.isInteger(dimension) ||
        dimension !== expectedShape[index],
    )
  ) {
    throw new TypeError(`Model tensor ${name} has an unexpected shape.`);
  }
  const data = raw["data"];
  const byteLength = tensorSize(expectedShape) * 4;
  const maximumEncodedLength = 4 * Math.ceil(MAX_TENSOR_BYTES / 3);
  if (typeof data === "string" && data.length > maximumEncodedLength) {
    throw new TypeError(
      `Model tensor ${name} exceeds the 8 MiB runtime limit.`,
    );
  }
  if (
    typeof data !== "string" ||
    byteLength > MAX_TENSOR_BYTES ||
    data.length !== 4 * Math.ceil(byteLength / 3) ||
    !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u.test(
      data,
    )
  ) {
    throw new TypeError(`Model tensor ${name} has invalid base64 data.`);
  }
  let decoded: string;
  try {
    decoded = atob(data);
  } catch {
    throw new TypeError(`Model tensor ${name} has invalid base64 data.`);
  }
  if (decoded.length !== byteLength) {
    throw new TypeError(`Model tensor ${name} has an invalid byte length.`);
  }
  const bytes = Uint8Array.from(decoded, (character) => character.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const values = new Float32Array(byteLength / 4);
  for (let index = 0; index < values.length; index += 1) {
    const value = view.getFloat32(index * 4, true);
    if (!Number.isFinite(value)) {
      throw new TypeError(`Model tensor ${name} contains a non-finite value.`);
    }
    values[index] = value;
  }
  return Object.freeze({
    shape: Object.freeze([...expectedShape]),
    values,
  });
}

export function parseHybridBrowserModel(
  value: Record<string, unknown>,
  featureDimension: number,
): HybridBrowserModel {
  if (
    value["format"] !== "drawbacktrainer-browser-model" ||
    value["formatVersion"] !== HYBRID_BROWSER_MODEL_FORMAT_VERSION ||
    value["modelVariant"] !== "v21-hybrid" ||
    value["featureSchemaVersion"] !== 1 ||
    value["symbolicFeatureVersion"] !== HYBRID_SYMBOLIC_FEATURE_VERSION ||
    value["tensorEncoding"] !== HYBRID_TENSOR_ENCODING ||
    Object.hasOwn(value, "sequenceObservationMode")
  ) {
    throw new TypeError("Browser hybrid model format or schema is unsupported.");
  }
  const digest = value["sourceCheckpointSha256"];
  if (typeof digest !== "string" || !/^[0-9a-f]{64}$/u.test(digest)) {
    throw new TypeError("Browser model checkpoint digest is invalid.");
  }
  const symbolicRuleIds = stringList(
    value["symbolicRuleIds"],
    "symbolicRuleIds",
  );
  if (
    symbolicRuleIds.length !== DEFAULT_HYPOTHESIS_RULE_IDS.length ||
    symbolicRuleIds.some(
      (id, index) => id !== DEFAULT_HYPOTHESIS_RULE_IDS[index],
    )
  ) {
    throw new TypeError("Browser hybrid model symbolic rule order is incompatible.");
  }
  const drawbackVocabulary = stringList(
    value["drawbackVocabulary"],
    "drawbackVocabulary",
  );
  if (
    drawbackVocabulary.length !== symbolicRuleIds.length ||
    drawbackVocabulary.some((id) => !symbolicRuleIds.includes(id))
  ) {
    throw new TypeError(
      "Browser hybrid drawback vocabulary must match all symbolic rules.",
    );
  }
  const rawTokenizer = value["tokenizer"];
  if (
    !isRecord(rawTokenizer) ||
    rawTokenizer["kind"] !== "exact-san-token" ||
    rawTokenizer["version"] !== 1 ||
    rawTokenizer["padding"] !== "right" ||
    rawTokenizer["truncation"] !== "keep-most-recent"
  ) {
    throw new TypeError("Browser hybrid SAN tokenizer is incompatible.");
  }
  const sanVocabulary = stringList(
    rawTokenizer["vocabulary"],
    "tokenizer.vocabulary",
  );
  if (
    sanVocabulary.length > MAX_VOCABULARY ||
    sanVocabulary.some((token) => token.length > 32)
  ) {
    throw new TypeError("Browser hybrid SAN tokenizer exceeds runtime limits.");
  }
  if (sanVocabulary[0] !== "<pad>" || sanVocabulary[1] !== "<unk>") {
    throw new TypeError("Browser hybrid SAN tokenizer reserved tokens are invalid.");
  }
  const maxHistory = integer(
    rawTokenizer["max_history"],
    "tokenizer.max_history",
    MAX_HISTORY,
  );
  const rawDimensions = value["dimensions"];
  if (!isRecord(rawDimensions)) {
    throw new TypeError("Browser hybrid model dimensions are missing.");
  }
  const dimensions = {
    input: integer(rawDimensions["input"], "dimensions.input", featureDimension),
    boardHidden: integer(
      rawDimensions["boardHidden"],
      "dimensions.boardHidden",
      MAX_DIMENSION,
    ),
    sanVocabulary: integer(
      rawDimensions["sanVocabulary"],
      "dimensions.sanVocabulary",
      MAX_VOCABULARY,
    ),
    sanEmbedding: integer(
      rawDimensions["sanEmbedding"],
      "dimensions.sanEmbedding",
      MAX_DIMENSION,
    ),
    sequenceHidden: integer(
      rawDimensions["sequenceHidden"],
      "dimensions.sequenceHidden",
      MAX_DIMENSION,
    ),
    symbolicInput: integer(
      rawDimensions["symbolicInput"],
      "dimensions.symbolicInput",
      DEFAULT_HYPOTHESIS_RULE_IDS.length * 4,
    ),
    symbolicHidden: integer(
      rawDimensions["symbolicHidden"],
      "dimensions.symbolicHidden",
      MAX_DIMENSION,
    ),
    drawbackClasses: integer(
      rawDimensions["drawbackClasses"],
      "dimensions.drawbackClasses",
      DEFAULT_HYPOTHESIS_RULE_IDS.length,
    ),
  };
  if (
    dimensions.input !== featureDimension ||
    dimensions.sanVocabulary !== sanVocabulary.length ||
    dimensions.symbolicInput !== symbolicRuleIds.length * 4 ||
    dimensions.drawbackClasses !== drawbackVocabulary.length
  ) {
    throw new TypeError("Browser hybrid model dimensions do not match its schema.");
  }
  const rawTensors = value["tensors"];
  if (
    !isRecord(rawTensors) ||
    Object.keys(rawTensors).length !== HYBRID_TENSOR_NAMES.length ||
    Object.keys(rawTensors).some(
      (name) => !HYBRID_TENSOR_NAMES.includes(name as HybridTensorName),
    )
  ) {
    throw new TypeError("Browser hybrid model contains unexpected tensors.");
  }
  const combined =
    dimensions.boardHidden +
    dimensions.sequenceHidden +
    dimensions.symbolicHidden;
  const shapes: Record<HybridTensorName, readonly number[]> = {
    "board_encoder.0.weight": [dimensions.boardHidden, dimensions.input],
    "board_encoder.0.bias": [dimensions.boardHidden],
    "board_encoder.2.weight": [
      dimensions.boardHidden,
      dimensions.boardHidden,
    ],
    "board_encoder.2.bias": [dimensions.boardHidden],
    "san_embedding.weight": [
      dimensions.sanVocabulary,
      dimensions.sanEmbedding,
    ],
    "history_encoder.weight_ih_l0": [
      dimensions.sequenceHidden * 3,
      dimensions.sanEmbedding,
    ],
    "history_encoder.weight_hh_l0": [
      dimensions.sequenceHidden * 3,
      dimensions.sequenceHidden,
    ],
    "history_encoder.bias_ih_l0": [dimensions.sequenceHidden * 3],
    "history_encoder.bias_hh_l0": [dimensions.sequenceHidden * 3],
    "symbolic_encoder.0.weight": [
      dimensions.symbolicHidden,
      dimensions.symbolicInput,
    ],
    "symbolic_encoder.0.bias": [dimensions.symbolicHidden],
    "symbolic_encoder.2.weight": [
      dimensions.symbolicHidden,
      dimensions.symbolicHidden,
    ],
    "symbolic_encoder.2.bias": [dimensions.symbolicHidden],
    "white_drawback.weight": [dimensions.drawbackClasses, combined],
    "white_drawback.bias": [dimensions.drawbackClasses],
    "black_drawback.weight": [dimensions.drawbackClasses, combined],
    "black_drawback.bias": [dimensions.drawbackClasses],
  };
  const tensors = Object.fromEntries(
    HYBRID_TENSOR_NAMES.map((name) => [
      name,
      decodeTensor(rawTensors, name, shapes[name]),
    ]),
  ) as unknown as Readonly<Record<HybridTensorName, HybridTensor>>;
  return Object.freeze({
    format: "drawbacktrainer-browser-model",
    formatVersion: HYBRID_BROWSER_MODEL_FORMAT_VERSION,
    modelVariant: "v21-hybrid",
    featureSchemaVersion: 1,
    symbolicFeatureVersion: HYBRID_SYMBOLIC_FEATURE_VERSION,
    sourceCheckpointSha256: digest,
    drawbackVocabulary: Object.freeze([...drawbackVocabulary]),
    symbolicRuleIds: Object.freeze([...symbolicRuleIds]),
    tokenizer: Object.freeze({
      kind: "exact-san-token",
      version: 1,
      vocabulary: Object.freeze([...sanVocabulary]),
      maxHistory,
      padding: "right",
      truncation: "keep-most-recent",
    }),
    tensorEncoding: HYBRID_TENSOR_ENCODING,
    dimensions: Object.freeze(dimensions),
    tensors: Object.freeze(tensors),
  });
}

export function parseHybridV22BrowserModel(
  value: Record<string, unknown>,
  featureDimension: number,
): HybridV22BrowserModel {
  exactKeys(
    value,
    [
      "format",
      "formatVersion",
      "modelVariant",
      "featureSchemaVersion",
      "symbolicFeatureVersion",
      "sequenceObservationMode",
      "sourceCheckpointSha256",
      "drawbackVocabulary",
      "symbolicRuleIds",
      "tokenizer",
      "tensorEncoding",
      "dimensions",
      "tensors",
    ],
    "Browser v22 hybrid model",
  );
  if (
    value["format"] !== "drawbacktrainer-browser-model" ||
    value["formatVersion"] !== HYBRID_V22_BROWSER_MODEL_FORMAT_VERSION ||
    value["modelVariant"] !== HYBRID_V22_MODEL_VARIANT ||
    (
      value["sequenceObservationMode"] !== "masked-current-v2" &&
      value["sequenceObservationMode"] !== "exact-current-v2"
    )
  ) {
    throw new TypeError("Browser v22 hybrid model format is unsupported.");
  }
  const sequenceObservationMode = value["sequenceObservationMode"];
  const rawTokenizer = value["tokenizer"];
  if (!isRecord(rawTokenizer)) {
    throw new TypeError("Browser v22 sequence tokenizer is missing.");
  }
  exactKeys(
    rawTokenizer,
    [
      "kind",
      "version",
      "vocabulary",
      "max_sequence",
      "padding",
      "truncation",
      "current_move",
    ],
    "Browser v22 sequence tokenizer",
  );
  if (
    rawTokenizer["kind"] !== PUBLIC_SEQUENCE_TOKENIZER_KIND ||
    rawTokenizer["version"] !== PUBLIC_SEQUENCE_TOKENIZER_VERSION ||
    rawTokenizer["padding"] !== "right" ||
    rawTokenizer["truncation"] !== "keep-most-recent" ||
    rawTokenizer["current_move"] !== "required-final-namespaced-uci"
  ) {
    throw new TypeError("Browser v22 sequence tokenizer is incompatible.");
  }
  const vocabulary = stringList(
    rawTokenizer["vocabulary"],
    "tokenizer.vocabulary",
  );
  if (
    vocabulary.length < 4 ||
    vocabulary.length > MAX_VOCABULARY ||
    vocabulary[0] !== "<pad>" ||
    vocabulary[1] !== "<unk>" ||
    vocabulary[2] !== UNKNOWN_CURRENT_MOVE_TOKEN ||
    vocabulary[3] !== MASKED_CURRENT_MOVE_TOKEN
  ) {
    throw new TypeError(
      "Browser v22 sequence tokenizer reserved tokens are invalid.",
    );
  }
  for (const token of vocabulary.slice(4)) {
    if (token.startsWith(CURRENT_MOVE_PREFIX)) {
      if (!token.endsWith(">")) {
        throw new TypeError(
          "Browser v22 sequence tokenizer contains an invalid current-move token.",
        );
      }
      const move = token.slice(CURRENT_MOVE_PREFIX.length, -1);
      if (currentMoveObservationToken(move) !== token) {
        throw new TypeError(
          "Browser v22 sequence tokenizer contains an invalid current-move token.",
        );
      }
    } else if (
      token.length > 32 ||
      token.includes("<") ||
      token.includes(">")
    ) {
      throw new TypeError(
        "Browser v22 sequence tokenizer contains an invalid SAN token.",
      );
    }
  }
  const maxSequence = integer(
    rawTokenizer["max_sequence"],
    "tokenizer.max_sequence",
    MAX_SEQUENCE,
  );
  const rawDimensions = value["dimensions"];
  if (!isRecord(rawDimensions)) {
    throw new TypeError("Browser v22 hybrid model dimensions are missing.");
  }
  exactKeys(
    rawDimensions,
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
    "Browser v22 hybrid dimensions",
  );
  const rawTensors = value["tensors"];
  if (!isRecord(rawTensors)) {
    throw new TypeError("Browser v22 hybrid model tensors are missing.");
  }
  for (const name of HYBRID_TENSOR_NAMES) {
    const rawTensor = rawTensors[name];
    if (!isRecord(rawTensor)) {
      throw new TypeError(`Model tensor ${name} is missing.`);
    }
    exactKeys(rawTensor, ["shape", "data"], `Model tensor ${name}`);
  }
  const parsed = parseHybridBrowserModel(
    {
      format: value["format"],
      formatVersion: HYBRID_BROWSER_MODEL_FORMAT_VERSION,
      modelVariant: "v21-hybrid",
      featureSchemaVersion: value["featureSchemaVersion"],
      symbolicFeatureVersion: value["symbolicFeatureVersion"],
      sourceCheckpointSha256: value["sourceCheckpointSha256"],
      drawbackVocabulary: value["drawbackVocabulary"],
      symbolicRuleIds: value["symbolicRuleIds"],
      tokenizer: {
        kind: "exact-san-token",
        version: 1,
        vocabulary,
        max_history: Math.min(maxSequence, MAX_HISTORY),
        padding: "right",
        truncation: "keep-most-recent",
      },
      tensorEncoding: value["tensorEncoding"],
      dimensions: value["dimensions"],
      tensors: value["tensors"],
    },
    featureDimension,
  );
  return Object.freeze({
    format: "drawbacktrainer-browser-model",
    formatVersion: HYBRID_V22_BROWSER_MODEL_FORMAT_VERSION,
    modelVariant: HYBRID_V22_MODEL_VARIANT,
    featureSchemaVersion: parsed.featureSchemaVersion,
    symbolicFeatureVersion: parsed.symbolicFeatureVersion,
    sequenceObservationMode,
    sourceCheckpointSha256: parsed.sourceCheckpointSha256,
    drawbackVocabulary: parsed.drawbackVocabulary,
    symbolicRuleIds: parsed.symbolicRuleIds,
    tokenizer: Object.freeze({
      kind: PUBLIC_SEQUENCE_TOKENIZER_KIND,
      version: PUBLIC_SEQUENCE_TOKENIZER_VERSION,
      vocabulary: Object.freeze([...vocabulary]),
      maxSequence,
      padding: "right",
      truncation: "keep-most-recent",
      currentMove: "required-final-namespaced-uci",
    }),
    tensorEncoding: parsed.tensorEncoding,
    dimensions: parsed.dimensions,
    tensors: parsed.tensors,
  });
}

function dense(
  input: ArrayLike<number>,
  weight: HybridTensor,
  bias: HybridTensor,
  relu: boolean,
): Float32Array {
  const rows = weight.shape[0] ?? 0;
  const columns = weight.shape[1] ?? 0;
  const output = new Float32Array(rows);
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

function sigmoid(value: number): number {
  return value >= 0
    ? 1 / (1 + Math.exp(-value))
    : Math.exp(value) / (1 + Math.exp(value));
}

export function encodeHybridSequenceTokenIndices(
  model: SingleHybridBrowserModel,
  observation: Pick<HybridObservation, "historySan" | "move">,
): readonly number[] {
  const tokenIndices = new Map(
    model.tokenizer.vocabulary.map((token, index) => [token, index]),
  );
  if (model.modelVariant === "v21-hybrid") {
    return Object.freeze(
      observation.historySan
        .slice(-model.tokenizer.maxHistory)
        .map((token) => tokenIndices.get(token) ?? 1),
    );
  }
  if (
    !isValidPriorSan(observation.historySan)
  ) {
    throw new TypeError(
      "Browser v22 sequence observation contains invalid prior SAN.",
    );
  }
  const exactCurrent = currentMoveObservationToken(observation.move);
  const current =
    model.sequenceObservationMode === "masked-current-v2"
      ? MASKED_CURRENT_MOVE_TOKEN
      : exactCurrent;
  const bounded = [...observation.historySan, current].slice(
    -model.tokenizer.maxSequence,
  );
  return Object.freeze(
    bounded.map((token) =>
      tokenIndices.get(token) ??
        (token.startsWith(CURRENT_MOVE_PREFIX) ? 2 : 1)
    ),
  );
}

function encodeHistory(
  model: SingleHybridBrowserModel,
  observation: Pick<HybridObservation, "historySan" | "move">,
): Float32Array {
  const hiddenSize = model.dimensions.sequenceHidden;
  let hidden = new Float32Array(hiddenSize);
  const encoded = encodeHybridSequenceTokenIndices(model, observation);
  if (encoded.length === 0) {
    return hidden;
  }
  const embedding = model.tensors["san_embedding.weight"];
  const weightInput = model.tensors["history_encoder.weight_ih_l0"];
  const weightHidden = model.tensors["history_encoder.weight_hh_l0"];
  const biasInput = model.tensors["history_encoder.bias_ih_l0"];
  const biasHidden = model.tensors["history_encoder.bias_hh_l0"];
  const embeddingSize = model.dimensions.sanEmbedding;
  for (const tokenIndex of encoded) {
    const embedded = embedding.values.subarray(
      tokenIndex * embeddingSize,
      (tokenIndex + 1) * embeddingSize,
    );
    const inputGates = dense(
      embedded,
      weightInput,
      biasInput,
      false,
    );
    const hiddenGates = dense(
      hidden,
      weightHidden,
      biasHidden,
      false,
    );
    const next = new Float32Array(hiddenSize);
    for (let index = 0; index < hiddenSize; index += 1) {
      const reset = sigmoid(
        (inputGates[index] ?? 0) + (hiddenGates[index] ?? 0),
      );
      const update = sigmoid(
        (inputGates[hiddenSize + index] ?? 0) +
          (hiddenGates[hiddenSize + index] ?? 0),
      );
      const candidate = Math.tanh(
        (inputGates[hiddenSize * 2 + index] ?? 0) +
          reset * (hiddenGates[hiddenSize * 2 + index] ?? 0),
      );
      next[index] = (1 - update) * candidate + update * (hidden[index] ?? 0);
    }
    hidden = next;
  }
  return hidden;
}

function checkedSymbolic(
  model: SingleHybridBrowserModel,
  evidence: HybridSymbolicEvidence,
): {
  readonly vector: Float32Array;
  readonly whitePrior: readonly number[];
  readonly blackPrior: readonly number[];
  readonly whiteEliminated: readonly boolean[];
  readonly blackEliminated: readonly boolean[];
} {
  const count = model.symbolicRuleIds.length;
  if (
    evidence.ruleIds.length !== count ||
    evidence.ruleIds.some((id, index) => id !== model.symbolicRuleIds[index]) ||
    evidence.whiteProbabilities.length !== count ||
    evidence.blackProbabilities.length !== count ||
    evidence.whiteEliminated.length !== count ||
    evidence.blackEliminated.length !== count ||
    evidence.whiteEliminated.some((value) => typeof value !== "boolean") ||
    evidence.blackEliminated.some((value) => typeof value !== "boolean")
  ) {
    throw new TypeError("Hybrid symbolic evidence does not match model schema.");
  }
  for (const probabilities of [
    evidence.whiteProbabilities,
    evidence.blackProbabilities,
  ]) {
    const total = probabilities.reduce((sum, probability) => {
      if (!Number.isFinite(probability) || probability < 0) {
        throw new TypeError("Hybrid symbolic probabilities are invalid.");
      }
      return sum + probability;
    }, 0);
    if (Math.abs(total - 1) > 1e-6) {
      throw new TypeError("Hybrid symbolic probabilities must sum to one.");
    }
  }
  if (
    evidence.whiteEliminated.every(Boolean) ||
    evidence.blackEliminated.every(Boolean)
  ) {
    throw new TypeError("Hybrid symbolic evidence eliminated every drawback.");
  }
  const vector = Float32Array.from([
    ...evidence.whiteProbabilities,
    ...evidence.blackProbabilities,
    ...evidence.whiteEliminated.map(Number),
    ...evidence.blackEliminated.map(Number),
  ]);
  return {
    vector,
    whitePrior: evidence.whiteProbabilities,
    blackPrior: evidence.blackProbabilities,
    whiteEliminated: evidence.whiteEliminated,
    blackEliminated: evidence.blackEliminated,
  };
}

function maskedPosterior(
  model: SingleHybridBrowserModel,
  residual: ArrayLike<number>,
  prior: readonly number[],
  eliminated: readonly boolean[],
  temperature: number,
  fusionAlpha: number,
): Readonly<Record<string, number>> {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new TypeError("Hybrid calibration temperature must be positive.");
  }
  const orderedPrior = model.drawbackVocabulary.map((id) => {
    const symbolicIndex = model.symbolicRuleIds.indexOf(id);
    if (symbolicIndex < 0) {
      throw new TypeError("Hybrid drawback vocabulary is not symbolic.");
    }
    return prior[symbolicIndex] ?? 0;
  });
  const orderedEliminated = model.drawbackVocabulary.map((id) => {
    const symbolicIndex = model.symbolicRuleIds.indexOf(id);
    if (symbolicIndex < 0) {
      throw new TypeError("Hybrid drawback vocabulary is not symbolic.");
    }
    return eliminated[symbolicIndex] ?? true;
  });
  const fusion = rankPreservingFusion(
    residual,
    orderedPrior,
    orderedEliminated,
    fusionAlpha,
  );
  const logits = fusion.logits.map((logit, index) => {
    if (orderedEliminated[index]) {
      return Number.NEGATIVE_INFINITY;
    }
    return logit / temperature;
  });
  const maximum = Math.max(...logits);
  const masses = logits.map((logit) =>
    logit === Number.NEGATIVE_INFINITY ? 0 : Math.exp(logit - maximum)
  );
  const total = masses.reduce((sum, mass) => sum + mass, 0);
  if (!Number.isFinite(total) || total <= 0) {
    throw new Error("Hybrid model produced an invalid posterior.");
  }
  return Object.freeze(
    Object.fromEntries(
      model.drawbackVocabulary.map((id, index) => [
        id,
        (masses[index] ?? 0) / total,
      ]),
    ),
  );
}

function legacyMaskedPosterior(
  model: SingleHybridBrowserModel,
  residual: ArrayLike<number>,
  prior: readonly number[],
  eliminated: readonly boolean[],
  temperature: number,
): Readonly<Record<string, number>> {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new TypeError("Hybrid calibration temperature must be positive.");
  }
  const logits = model.drawbackVocabulary.map((id, classIndex) => {
    const symbolicIndex = model.symbolicRuleIds.indexOf(id);
    if (symbolicIndex < 0 || eliminated[symbolicIndex]) {
      return Number.NEGATIVE_INFINITY;
    }
    return (
      (residual[classIndex] ?? 0) +
      Math.log(Math.max(prior[symbolicIndex] ?? 0, 1e-12))
    ) / temperature;
  });
  const maximum = Math.max(...logits);
  const masses = logits.map((logit) =>
    logit === Number.NEGATIVE_INFINITY ? 0 : Math.exp(logit - maximum)
  );
  const total = masses.reduce((sum, mass) => sum + mass, 0);
  if (!Number.isFinite(total) || total <= 0) {
    throw new Error("Hybrid model produced an invalid posterior.");
  }
  return Object.freeze(
    Object.fromEntries(
      model.drawbackVocabulary.map((id, index) => [
        id,
        (masses[index] ?? 0) / total,
      ]),
    ),
  );
}

export function runHybridBrowserModel(
  model: SingleHybridBrowserModel,
  observation: HybridObservation,
  boardFeatures: readonly number[],
): HybridOutput {
  const residuals = runHybridBrowserResidualModel(
    model,
    observation,
    boardFeatures,
  );
  const symbolic = checkedSymbolic(model, observation.symbolic);
  const posterior =
    model.modelVariant === HYBRID_V22_MODEL_VARIANT
      ? (
        residual: ArrayLike<number>,
        prior: readonly number[],
        eliminated: readonly boolean[],
      ): Readonly<Record<string, number>> =>
        maskedPosterior(
          model,
          residual,
          prior,
          eliminated,
          1,
          1,
        )
      : (
        residual: ArrayLike<number>,
        prior: readonly number[],
        eliminated: readonly boolean[],
      ): Readonly<Record<string, number>> =>
        legacyMaskedPosterior(model, residual, prior, eliminated, 1);
  return Object.freeze({
    white: posterior(
      residuals.white,
      symbolic.whitePrior,
      symbolic.whiteEliminated,
    ),
    black: posterior(
      residuals.black,
      symbolic.blackPrior,
      symbolic.blackEliminated,
    ),
  });
}

export function runHybridBrowserResidualModel(
  model: SingleHybridBrowserModel,
  observation: HybridObservation,
  boardFeatures: readonly number[],
): HybridResidualOutput {
  const symbolic = checkedSymbolic(model, observation.symbolic);
  const boardFirst = dense(
    boardFeatures,
    model.tensors["board_encoder.0.weight"],
    model.tensors["board_encoder.0.bias"],
    true,
  );
  const board = dense(
    boardFirst,
    model.tensors["board_encoder.2.weight"],
    model.tensors["board_encoder.2.bias"],
    true,
  );
  const history = encodeHistory(model, observation);
  const symbolicFirst = dense(
    symbolic.vector,
    model.tensors["symbolic_encoder.0.weight"],
    model.tensors["symbolic_encoder.0.bias"],
    true,
  );
  const symbolicEncoded = dense(
    symbolicFirst,
    model.tensors["symbolic_encoder.2.weight"],
    model.tensors["symbolic_encoder.2.bias"],
    true,
  );
  const combined = Float32Array.from([
    ...board,
    ...history,
    ...symbolicEncoded,
  ]);
  const whiteResidual = dense(
    combined,
    model.tensors["white_drawback.weight"],
    model.tensors["white_drawback.bias"],
    false,
  );
  const blackResidual = dense(
    combined,
    model.tensors["black_drawback.weight"],
    model.tensors["black_drawback.bias"],
    false,
  );
  return Object.freeze({
    white: whiteResidual,
    black: blackResidual,
  });
}

export function applyHybridResidualPosterior(
  model: SingleHybridBrowserModel,
  evidence: HybridSymbolicEvidence,
  residuals: HybridResidualOutput,
  temperatures: {
    readonly white: number;
    readonly black: number;
  },
  fusionAlpha: number,
): HybridOutput {
  const symbolic = checkedSymbolic(model, evidence);
  for (const residual of [residuals.white, residuals.black]) {
    if (
      residual.length !== model.drawbackVocabulary.length ||
      Array.from(residual).some((value) => !Number.isFinite(value))
    ) {
      throw new TypeError("Hybrid residual logits are invalid.");
    }
  }
  return Object.freeze({
    white: maskedPosterior(
      model,
      residuals.white,
      symbolic.whitePrior,
      symbolic.whiteEliminated,
      temperatures.white,
      fusionAlpha,
    ),
    black: maskedPosterior(
      model,
      residuals.black,
      symbolic.blackPrior,
      symbolic.blackEliminated,
      temperatures.black,
      fusionAlpha,
    ),
  });
}
