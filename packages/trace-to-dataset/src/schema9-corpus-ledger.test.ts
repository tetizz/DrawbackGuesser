import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import {
  appendFile,
  mkdtemp,
  readdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resolve } from "node:path";
import { promisify } from "node:util";
import {
  CAPTURABLE_HYPOTHESIS_RULE_IDS,
  RULE_OPPORTUNITY_FEATURE_FIELDS,
} from "@drawbackguesser/predictor";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("./player-private-converter.js", () => ({
  CAPTURABLE_SYMBOLIC_FEATURE_VERSION: 9,
  convertParsedPlayerPrivateTraceToDatasetRows: (
    trace: {
      readonly gameId: string;
      readonly plies: readonly { readonly ply: number }[];
    },
  ) =>
    trace.plies.map((ply) =>
      Object.freeze({
        authenticatedGameId: trace.gameId,
        authenticatedPly: ply.ply,
      })
    ),
}));
vi.mock("@drawbackengine/simulation-trace", () => ({
  parsePlayerPrivateSimulationTraceLine: (line: string) =>
    JSON.parse(line) as unknown,
}));
import {
  assembleSchema9CorpusLedger,
  assertSchema9CorpusLedgerByteLength,
  assertSchema9SplitsDisjoint,
  createSchema9CorpusLedger,
  publishOrAuthenticateSchema9CorpusLedgerArtifactAtomic,
  publishSchema9CorpusLedgerArtifactAtomic,
  schema9CorpusLedgerFileSha256,
  SCHEMA9_CORPUS_LEDGER_VERSION,
  SCHEMA9_LEDGER_SPLITS,
  SCHEMA9_EXECUTION_MANIFEST_ALGORITHM,
  SCHEMA9_GENERATION_CONFIG,
  SCHEMA9_GENERATOR_COMPLETION_FORMAT,
  SCHEMA9_GENERATOR_LAUNCH_FORMAT,
  SCHEMA9_GENERATOR_RECEIPT_VERSION,
  SCHEMA9_PRODUCER_RUNTIME_IDENTITY_FORMAT,
  SCHEMA9_PRODUCER_RUNTIME_IDENTITY_VERSION,
  SCHEMA9_PRODUCER_RUNTIME_MANIFEST_ALGORITHM,
  SCHEMA9_SCHEDULE_PROFILE,
  SCHEMA9_SEED_STREAMS,
  SCHEMA9_SPLIT_SEED_ROOTS,
  verifySchema9CorpusLedgerReconstruction,
  verifySchema9RepositoryIdentity,
  type Schema9CorpusLedgerOptions,
  type Schema9CorpusLedger,
  type Schema9AssignmentScheduler,
  type Schema9ExecutionIdentity,
  type Schema9LedgerSplit,
  type Schema9ProducerRuntimeIdentity,
  type Schema9RepositoryVerifier,
  type Schema9SplitFiles,
} from "./schema9-corpus-ledger.js";
import {
  assertExactSchema9LabelBalance,
  assertScheduledConversionAccounting,
  authenticateSchema9SplitWithRuleContract,
  type AuthenticatedSchema9Split,
} from "./schema9-ledger-authentication.js";
import {
  assertPathFreeJson,
  canonicalJsonBytes,
  checkedGitCommit,
  checkedSchema9ProducerRuntimeIdentity,
  checkedSchema9SeedRoots,
  checkedScheduleId,
  checkedSha256,
  parseJsonWithoutDuplicateKeys,
} from "./schema9-ledger-types.js";
import {
  createSchema9LedgerVerificationReceipt,
  publishOrAuthenticateSchema9LedgerVerificationReceipt,
  SCHEMA9_LEDGER_VERIFICATION_RECEIPT_VERSION,
  writeSchema9LedgerVerificationReceiptAtomic,
} from "./schema9-ledger-verification-receipt.js";

const GUESSER_COMMIT = "a".repeat(40);
const CONVERTER_ENGINE_COMMIT = "b".repeat(40);
const DESCENDANT_ENGINE_COMMIT = "c".repeat(40);
const EXECUTION_COMPONENT = Object.freeze({
  entrypoint: "fixture-code",
  files: 1,
  bytes: 1,
  sha256: "8".repeat(64),
});
const EXECUTION_PAYLOAD = Object.freeze({
  algorithm: SCHEMA9_EXECUTION_MANIFEST_ALGORITHM,
  runtime: Object.freeze({
    nodeVersion: "v24.0.0",
    platform: "win32",
    architecture: "x64",
    execArgv: Object.freeze([]),
  }),
  parser: EXECUTION_COMPONENT,
  converter: EXECUTION_COMPONENT,
  scheduler: EXECUTION_COMPONENT,
  verifier: EXECUTION_COMPONENT,
});
const EXECUTION_IDENTITY: Schema9ExecutionIdentity = Object.freeze({
  ...EXECUTION_PAYLOAD,
  aggregateSha256: createHash("sha256")
    .update(canonicalJsonBytes(EXECUTION_PAYLOAD))
    .digest("hex"),
});
const PRODUCER_RUNTIME_PAYLOAD = Object.freeze({
  format: SCHEMA9_PRODUCER_RUNTIME_IDENTITY_FORMAT,
  version: SCHEMA9_PRODUCER_RUNTIME_IDENTITY_VERSION,
  algorithm: SCHEMA9_PRODUCER_RUNTIME_MANIFEST_ALGORITHM,
  runtime: Object.freeze({
    nodeVersion: "v24.0.0",
    platform: "win32",
    architecture: "x64",
    execArgv: Object.freeze([] as const),
  }),
  coordinator: Object.freeze({
    componentId: "schema9-coordinator/v1" as const,
    files: 11,
    bytes: 12_345,
    sha256: "6".repeat(64),
  }),
  parallelWorker: Object.freeze({
    componentId: "player-private-parallel-worker/v1" as const,
    files: 7,
    bytes: 8_765,
    sha256: "7".repeat(64),
  }),
});
const PRODUCER_RUNTIME_IDENTITY: Schema9ProducerRuntimeIdentity =
  Object.freeze({
    ...PRODUCER_RUNTIME_PAYLOAD,
    aggregateSha256: createHash("sha256")
      .update(canonicalJsonBytes(PRODUCER_RUNTIME_PAYLOAD))
      .digest("hex"),
  });
const ALTERNATE_PRODUCER_RUNTIME_PAYLOAD = Object.freeze({
  ...PRODUCER_RUNTIME_PAYLOAD,
  parallelWorker: Object.freeze({
    ...PRODUCER_RUNTIME_PAYLOAD.parallelWorker,
    sha256: "9".repeat(64),
  }),
});
const ALTERNATE_PRODUCER_RUNTIME_IDENTITY: Schema9ProducerRuntimeIdentity =
  Object.freeze({
    ...ALTERNATE_PRODUCER_RUNTIME_PAYLOAD,
    aggregateSha256: createHash("sha256")
      .update(canonicalJsonBytes(ALTERNATE_PRODUCER_RUNTIME_PAYLOAD))
      .digest("hex"),
  });
const PRODUCER_RUNTIME_GOLDEN_PAYLOAD = Object.freeze({
  format: SCHEMA9_PRODUCER_RUNTIME_IDENTITY_FORMAT,
  version: SCHEMA9_PRODUCER_RUNTIME_IDENTITY_VERSION,
  algorithm: SCHEMA9_PRODUCER_RUNTIME_MANIFEST_ALGORITHM,
  runtime: Object.freeze({
    nodeVersion: "v22.17.0",
    platform: "win32",
    architecture: "x64",
    execArgv: Object.freeze([] as const),
  }),
  coordinator: Object.freeze({
    componentId: "schema9-coordinator/v1" as const,
    files: 17,
    bytes: 1_234,
    sha256: "1".repeat(64),
  }),
  parallelWorker: Object.freeze({
    componentId: "player-private-parallel-worker/v1" as const,
    files: 13,
    bytes: 987,
    sha256: "2".repeat(64),
  }),
});
const PRODUCER_RUNTIME_GOLDEN_SHA256 =
  "8ae516a9c7dd38ec645f79036806fceb9f75e9e4860426d53b83befee5a0347d";
const cleanupDirectories: string[] = [];
const execFileAsync = promisify(execFile);

interface FixtureTrace {
  readonly schemaVersion: 2;
  readonly ruleset: { readonly version: 2 };
  readonly gameId: string;
  readonly seed: number;
  readonly gameIndex: number;
  readonly parameterSeeds: { readonly white: number; readonly black: number };
  readonly plyLimit: number;
  readonly initialPosition: { readonly fen: string };
  readonly hypothesisPolicy: {
    readonly kind: "unrestricted-baseline";
    readonly version: 1;
  };
  readonly agents: {
    readonly white: { readonly searchPolicy: FixtureSearchPolicy };
    readonly black: { readonly searchPolicy: FixtureSearchPolicy };
  };
  readonly secrets: {
    readonly initial: {
      readonly white: { readonly drawbackId: string };
      readonly black: { readonly drawbackId: string };
    };
  };
  readonly plies: readonly { readonly ply: number }[];
}

interface FixtureSearchPolicy {
  readonly policyId: string;
  readonly evaluatorId: string;
  readonly maxDepth: number;
  readonly maxNodes: number;
  readonly temperatureCp: number;
  readonly topK: number;
  readonly leafCacheEntries: number;
  readonly leafCacheHistoryMode: string;
  readonly opponentAggregation: string;
}

interface ReceiptMutationCase {
  readonly name: string;
  readonly expectedError: string;
  readonly mutate: (receipt: Record<string, unknown>) => void;
}

interface GenerationConfigValueMutation {
  readonly name: string;
  readonly path: readonly string[];
  readonly replacement: unknown;
}

const GENERATION_CONFIG_VALUE_MUTATIONS = Object.freeze([
  { name: "maxPlies", path: ["maxPlies"], replacement: 121 },
  { name: "maxDepth", path: ["maxDepth"], replacement: 3 },
  { name: "maxNodes", path: ["maxNodes"], replacement: 49_999 },
  { name: "temperatureCp", path: ["temperatureCp"], replacement: 36 },
  { name: "topK", path: ["topK"], replacement: 9 },
  {
    name: "leafCacheEntries",
    path: ["leafCacheEntries"],
    replacement: 16_385,
  },
  {
    name: "leafCacheHistoryMode",
    path: ["leafCacheHistoryMode"],
    replacement: "fen-only",
  },
  {
    name: "opponentAggregation",
    path: ["opponentAggregation"],
    replacement: "average",
  },
  {
    name: "evaluator.kind",
    path: ["evaluator", "kind"],
    replacement: "stockfish",
  },
  {
    name: "evaluator.version",
    path: ["evaluator", "version"],
    replacement: 2,
  },
  {
    name: "evaluator.evaluatorId",
    path: ["evaluator", "evaluatorId"],
    replacement: "alternate-material/v1",
  },
  {
    name: "opponentHypotheses.kind",
    path: ["opponentHypotheses", "kind"],
    replacement: "catalog",
  },
  {
    name: "opponentHypotheses.version",
    path: ["opponentHypotheses", "version"],
    replacement: 2,
  },
] satisfies readonly GenerationConfigValueMutation[]);

const SEARCH_POLICY_VALUE_MUTATIONS = Object.freeze([
  {
    name: "policyId",
    path: ["policyId"],
    replacement: "alternate-policy/v1",
  },
  {
    name: "evaluatorId",
    path: ["evaluatorId"],
    replacement: "alternate-material/v1",
  },
  { name: "maxDepth", path: ["maxDepth"], replacement: 3 },
  { name: "maxNodes", path: ["maxNodes"], replacement: 49_999 },
  {
    name: "temperatureCp",
    path: ["temperatureCp"],
    replacement: 36,
  },
  { name: "topK", path: ["topK"], replacement: 9 },
  {
    name: "leafCacheEntries",
    path: ["leafCacheEntries"],
    replacement: 16_385,
  },
  {
    name: "leafCacheHistoryMode",
    path: ["leafCacheHistoryMode"],
    replacement: "ignore",
  },
  {
    name: "opponentAggregation",
    path: ["opponentAggregation"],
    replacement: "posterior-expected",
  },
] satisfies readonly GenerationConfigValueMutation[]);

const SOURCE_TRACE_CONFIG_MUTATIONS = Object.freeze([
  {
    name: "plyLimit",
    path: ["plyLimit"],
    replacement: 119,
  },
  ...(["white", "black"] as const).flatMap((color) => [
    ...SEARCH_POLICY_VALUE_MUTATIONS.map((mutation) => ({
      name: `${color}.${mutation.name}`,
      path: ["agents", color, "searchPolicy", ...mutation.path],
      replacement: mutation.replacement,
    })),
    {
      name: `${color}.missing-opponentAggregation`,
      path: ["agents", color, "searchPolicy", "opponentAggregation"],
      replacement: undefined,
    },
  ]),
  {
    name: "hypothesisPolicy.kind",
    path: ["hypothesisPolicy", "kind"],
    replacement: "audited-uniform",
  },
  {
    name: "hypothesisPolicy.version",
    path: ["hypothesisPolicy", "version"],
    replacement: 2,
  },
] satisfies readonly GenerationConfigValueMutation[]);

afterEach(async () => {
  await Promise.all(
    cleanupDirectories.splice(0).map((path) =>
      rm(path, { recursive: true, force: true })
    ),
  );
});

function repositoryVerifier(
  ancestorAccepted = true,
): Schema9RepositoryVerifier {
  return Object.freeze({
    pinnedEngineCommitAt: () => Promise.resolve(CONVERTER_ENGINE_COMMIT),
    isEngineAncestor: () => Promise.resolve(ancestorAccepted),
    executingCodeIdentity: () => Promise.resolve(EXECUTION_IDENTITY),
    producerRuntimeIdentityAt: () =>
      Promise.resolve(PRODUCER_RUNTIME_IDENTITY),
  });
}

async function splitFixture(): Promise<{
  readonly root: string;
  readonly trace: FixtureTrace;
  readonly files: Schema9SplitFiles;
}> {
  const root = await mkdtemp(
    join(tmpdir(), "drawback-guesser-schema9-split-"),
  );
  cleanupDirectories.push(root);
  const searchPolicy: FixtureSearchPolicy = Object.freeze({
    policyId: SCHEMA9_SCHEDULE_PROFILE.policyId,
    evaluatorId: SCHEMA9_GENERATION_CONFIG.evaluator.evaluatorId,
    maxDepth: SCHEMA9_GENERATION_CONFIG.maxDepth,
    maxNodes: SCHEMA9_GENERATION_CONFIG.maxNodes,
    temperatureCp: SCHEMA9_GENERATION_CONFIG.temperatureCp,
    topK: SCHEMA9_GENERATION_CONFIG.topK,
    leafCacheEntries: SCHEMA9_GENERATION_CONFIG.leafCacheEntries,
    leafCacheHistoryMode: SCHEMA9_GENERATION_CONFIG.leafCacheHistoryMode,
    opponentAggregation: SCHEMA9_GENERATION_CONFIG.opponentAggregation,
  });
  const trace: FixtureTrace = Object.freeze({
    schemaVersion: 2,
    ruleset: Object.freeze({ version: 2 }),
    gameId: "schema9-ledger-fixture-game",
    seed: 3_145_926,
    gameIndex: 0,
    parameterSeeds: Object.freeze({ white: 101, black: 102 }),
    plyLimit: SCHEMA9_GENERATION_CONFIG.maxPlies,
    initialPosition: Object.freeze({ fen: "fixture-fen" }),
    hypothesisPolicy: Object.freeze({
      kind: "unrestricted-baseline",
      version: 1,
    }),
    agents: Object.freeze({
      white: Object.freeze({
        searchPolicy,
      }),
      black: Object.freeze({
        searchPolicy,
      }),
    }),
    secrets: Object.freeze({
      initial: Object.freeze({
        white: Object.freeze({ drawbackId: "vegan" }),
        black: Object.freeze({ drawbackId: "vegan" }),
      }),
    }),
    plies: Object.freeze([Object.freeze({ ply: 0 })]),
  });
  const tracePath = join(root, "source.ndjson");
  const convertedPath = join(root, "converted.ndjson");
  const launchReceiptPath = join(root, "launch.json");
  const completionReceiptPath = join(root, "completion.json");
  const tracePayload = Buffer.from(`${JSON.stringify(trace)}\n`, "utf8");
  await writeFile(tracePath, tracePayload);
  await writeFile(
    convertedPath,
    trace.plies
      .map((ply) => ({
        authenticatedGameId: trace.gameId,
        authenticatedPly: ply.ply,
      }))
      .map((row) => `${JSON.stringify(row)}\n`)
      .join(""),
    "utf8",
  );
  const launchPayload = Buffer.from(`${JSON.stringify({
    format: SCHEMA9_GENERATOR_LAUNCH_FORMAT,
    version: SCHEMA9_GENERATOR_RECEIPT_VERSION,
    scheduleAuthorityId: "capturable25-schema9-opportunity/v1",
    scheduleId: "schema9-fixture-train",
    ledgerSplit: "train",
    engineSplit: "train",
    splitCounts: { train: 1, validation: 0, test: 0 },
    seedRoots: SCHEMA9_SPLIT_SEED_ROOTS.train,
    scheduleProfile: SCHEMA9_SCHEDULE_PROFILE,
    generationConfig: SCHEMA9_GENERATION_CONFIG,
    producerEngineCommit: CONVERTER_ENGINE_COMMIT,
    producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
  })}\n`, "utf8");
  await writeFile(launchReceiptPath, launchPayload);
  await writeFile(
    completionReceiptPath,
    `${JSON.stringify({
      format: SCHEMA9_GENERATOR_COMPLETION_FORMAT,
      version: SCHEMA9_GENERATOR_RECEIPT_VERSION,
      scheduleId: "schema9-fixture-train",
      ledgerSplit: "train",
      state: "completed",
      producerEngineCommit: CONVERTER_ENGINE_COMMIT,
      producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
      launchReceiptSha256: createHash("sha256")
        .update(launchPayload)
        .digest("hex"),
      output: {
        sha256: createHash("sha256").update(tracePayload).digest("hex"),
        bytes: tracePayload.byteLength,
        games: 1,
        firstGameIndex: 0,
        lastGameIndex: 0,
      },
    })}\n`,
    "utf8",
  );
  return {
    root,
    trace,
    files: Object.freeze({
      tracePath,
      convertedPath,
      launchReceiptPath,
      completionReceiptPath,
      scheduleId: "schema9-fixture-train",
      seedRoots: SCHEMA9_SPLIT_SEED_ROOTS.train,
      producerEngineCommit: CONVERTER_ENGINE_COMMIT,
    }),
  };
}

function mutableFixtureObject(
  value: unknown,
  label: string,
): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} fixture must be an object.`);
  }
  return value as Record<string, unknown>;
}

function generationConfigOf(
  receipt: Record<string, unknown>,
): Record<string, unknown> {
  return mutableFixtureObject(
    receipt["generationConfig"],
    "generationConfig",
  );
}

function producerRuntimeIdentityOf(
  receipt: Record<string, unknown>,
): Record<string, unknown> {
  return mutableFixtureObject(
    receipt["producerRuntimeIdentity"],
    "producerRuntimeIdentity",
  );
}

function setFixturePath(
  root: Record<string, unknown>,
  path: readonly string[],
  replacement: unknown,
): void {
  if (path.length === 0) {
    throw new TypeError("Fixture mutation path must not be empty.");
  }
  let parent = root;
  for (const key of path.slice(0, -1)) {
    parent = mutableFixtureObject(parent[key], key);
  }
  const leaf = path.at(-1);
  if (leaf === undefined) {
    throw new TypeError("Fixture mutation path must contain a leaf.");
  }
  parent[leaf] = replacement;
}

async function mutateLaunchReceipt(
  files: Schema9SplitFiles,
  mutate: (receipt: Record<string, unknown>) => void,
): Promise<void> {
  const receipt = mutableFixtureObject(
    JSON.parse(await readFile(files.launchReceiptPath, "utf8")) as unknown,
    "launch receipt",
  );
  mutate(receipt);
  await writeFile(
    files.launchReceiptPath,
    `${JSON.stringify(receipt)}\n`,
    "utf8",
  );
}

async function mutateCompletionReceipt(
  files: Schema9SplitFiles,
  mutate: (receipt: Record<string, unknown>) => void,
): Promise<void> {
  const receipt = mutableFixtureObject(
    JSON.parse(await readFile(files.completionReceiptPath, "utf8")) as unknown,
    "completion receipt",
  );
  mutate(receipt);
  await writeFile(
    files.completionReceiptPath,
    `${JSON.stringify(receipt)}\n`,
    "utf8",
  );
}

async function mutateSourceTrace(
  files: Schema9SplitFiles,
  mutate: (trace: Record<string, unknown>) => void,
): Promise<void> {
  const trace = mutableFixtureObject(
    JSON.parse(await readFile(files.tracePath, "utf8")) as unknown,
    "source trace",
  );
  mutate(trace);
  await writeFile(files.tracePath, `${JSON.stringify(trace)}\n`, "utf8");
}

const GENERATION_CONFIG_FIELD_MUTATIONS = Object.freeze([
  {
    name: "legacy version-2 receipt",
    expectedError: "launch receipt identity is inconsistent",
    mutate: (receipt) => {
      receipt["version"] = 2;
    },
  },
  {
    name: "missing generationConfig",
    expectedError: "launch receipt has invalid fields",
    mutate: (receipt) => {
      delete receipt["generationConfig"];
    },
  },
  {
    name: "extra launch field",
    expectedError: "launch receipt has invalid fields",
    mutate: (receipt) => {
      receipt["unexpectedConfig"] = true;
    },
  },
  {
    name: "non-object generationConfig",
    expectedError: "launch generation config must be an object",
    mutate: (receipt) => {
      receipt["generationConfig"] = null;
    },
  },
  {
    name: "missing generationConfig field",
    expectedError: "launch generation config has invalid fields",
    mutate: (receipt) => {
      delete generationConfigOf(receipt)["maxPlies"];
    },
  },
  {
    name: "extra generationConfig field",
    expectedError: "launch generation config has invalid fields",
    mutate: (receipt) => {
      generationConfigOf(receipt)["unexpected"] = true;
    },
  },
  {
    name: "missing evaluator field",
    expectedError: "launch generation evaluator has invalid fields",
    mutate: (receipt) => {
      const evaluator = mutableFixtureObject(
        generationConfigOf(receipt)["evaluator"],
        "evaluator",
      );
      delete evaluator["evaluatorId"];
    },
  },
  {
    name: "extra evaluator field",
    expectedError: "launch generation evaluator has invalid fields",
    mutate: (receipt) => {
      const evaluator = mutableFixtureObject(
        generationConfigOf(receipt)["evaluator"],
        "evaluator",
      );
      evaluator["unexpected"] = true;
    },
  },
  {
    name: "missing opponentHypotheses field",
    expectedError:
      "launch generation opponent hypotheses has invalid fields",
    mutate: (receipt) => {
      const hypotheses = mutableFixtureObject(
        generationConfigOf(receipt)["opponentHypotheses"],
        "opponentHypotheses",
      );
      delete hypotheses["version"];
    },
  },
  {
    name: "extra opponentHypotheses field",
    expectedError:
      "launch generation opponent hypotheses has invalid fields",
    mutate: (receipt) => {
      const hypotheses = mutableFixtureObject(
        generationConfigOf(receipt)["opponentHypotheses"],
        "opponentHypotheses",
      );
      hypotheses["unexpected"] = true;
    },
  },
] satisfies readonly ReceiptMutationCase[]);

const PRODUCER_RUNTIME_IDENTITY_MUTATIONS = Object.freeze([
  {
    name: "missing identity",
    mutate: (receipt: Record<string, unknown>) => {
      delete receipt["producerRuntimeIdentity"];
    },
  },
  {
    name: "unknown identity field",
    mutate: (receipt: Record<string, unknown>) => {
      producerRuntimeIdentityOf(receipt)["unexpected"] = true;
    },
  },
  {
    name: "wrong format",
    mutate: (receipt: Record<string, unknown>) => {
      producerRuntimeIdentityOf(receipt)["format"] = "alternate-runtime";
    },
  },
  {
    name: "wrong version",
    mutate: (receipt: Record<string, unknown>) => {
      producerRuntimeIdentityOf(receipt)["version"] = 2;
    },
  },
  {
    name: "wrong algorithm",
    mutate: (receipt: Record<string, unknown>) => {
      producerRuntimeIdentityOf(receipt)["algorithm"] = "sha256-source-tree-v1";
    },
  },
  {
    name: "runtime field missing",
    mutate: (receipt: Record<string, unknown>) => {
      delete mutableFixtureObject(
        producerRuntimeIdentityOf(receipt)["runtime"],
        "runtime",
      )["architecture"];
    },
  },
  {
    name: "runtime field added",
    mutate: (receipt: Record<string, unknown>) => {
      mutableFixtureObject(
        producerRuntimeIdentityOf(receipt)["runtime"],
        "runtime",
      )["hook"] = "enabled";
    },
  },
  {
    name: "runtime flags active",
    mutate: (receipt: Record<string, unknown>) => {
      mutableFixtureObject(
        producerRuntimeIdentityOf(receipt)["runtime"],
        "runtime",
      )["execArgv"] = ["--inspect"];
    },
  },
  ...(["coordinator", "parallelWorker"] as const).flatMap((component) => [
    {
      name: `${component} field missing`,
      mutate: (receipt: Record<string, unknown>) => {
        delete mutableFixtureObject(
          producerRuntimeIdentityOf(receipt)[component],
          component,
        )["sha256"];
      },
    },
    {
      name: `${component} field added`,
      mutate: (receipt: Record<string, unknown>) => {
        mutableFixtureObject(
          producerRuntimeIdentityOf(receipt)[component],
          component,
        )["path"] = "module.js";
      },
    },
    {
      name: `${component} id changed`,
      mutate: (receipt: Record<string, unknown>) => {
        mutableFixtureObject(
          producerRuntimeIdentityOf(receipt)[component],
          component,
        )["componentId"] = "alternate-component/v1";
      },
    },
    {
      name: `${component} files zero`,
      mutate: (receipt: Record<string, unknown>) => {
        mutableFixtureObject(
          producerRuntimeIdentityOf(receipt)[component],
          component,
        )["files"] = 0;
      },
    },
    {
      name: `${component} bytes zero`,
      mutate: (receipt: Record<string, unknown>) => {
        mutableFixtureObject(
          producerRuntimeIdentityOf(receipt)[component],
          component,
        )["bytes"] = 0;
      },
    },
    {
      name: `${component} digest changed`,
      mutate: (receipt: Record<string, unknown>) => {
        mutableFixtureObject(
          producerRuntimeIdentityOf(receipt)[component],
          component,
        )["sha256"] = "0".repeat(64);
      },
    },
  ]),
  {
    name: "aggregate changed",
    mutate: (receipt: Record<string, unknown>) => {
      producerRuntimeIdentityOf(receipt)["aggregateSha256"] = "0".repeat(64);
    },
  },
]);

const COMPLETION_DIGEST_TYPE_MUTATIONS = Object.freeze(
  ([
    ["array", ["a".repeat(64)]],
    ["object", { digest: "a".repeat(64) }],
    ["number", 123],
  ] as const).flatMap(([valueType, replacement]) =>
    (["launchReceiptSha256", "output.sha256"] as const).map((field) => ({
      name: `${field} ${valueType}`,
      field,
      replacement,
    }))
  ),
);

function fixtureScheduler(trace: FixtureTrace): Schema9AssignmentScheduler {
  return Object.freeze({
    assignments: () => Object.freeze([Object.freeze({
      gameIndex: trace.gameIndex,
      gameId: trace.gameId,
      seed: trace.seed,
      parameterSeeds: trace.parameterSeeds,
      whiteRuleId: trace.secrets.initial.white.drawbackId,
      blackRuleId: trace.secrets.initial.black.drawbackId,
      initialFen: trace.initialPosition.fen,
      initialReplaySha256: createHash("sha256")
        .update(canonicalJsonBytes({
          initialPosition: trace.initialPosition,
          initialSecrets: trace.secrets.initial,
        }))
        .digest("hex"),
    })]),
  });
}

function fakeAuthenticatedSplit(
  split: Schema9LedgerSplit,
  gameId: string,
  seed: number,
): AuthenticatedSchema9Split {
  const labelCounts = Object.freeze(
    Object.fromEntries(
      CAPTURABLE_HYPOTHESIS_RULE_IDS.map((ruleId) => [ruleId, 1]),
    ),
  );
  const gameIds = Array.from(
    { length: 25 },
    (_, index) => `${gameId}-${String(index)}`,
  ).sort((left, right) => left < right ? -1 : left > right ? 1 : 0);
  const simulationSeeds = Array.from(
    { length: 25 },
    (_, index) => seed + index,
  );
  const parameterSeeds = Array.from(
    { length: 50 },
    (_, index) => seed + 1_000 + index,
  );
  const convertedGameIds = Object.freeze([gameIds[0] as string]);
  const convertedSimulationSeeds = Object.freeze([simulationSeeds[0] as number]);
  const setDigest = (values: readonly (string | number)[]): string =>
    createHash("sha256")
      .update(`${JSON.stringify(values)}\n`)
      .digest("hex");
  return Object.freeze({
    gameIds: new Set(gameIds),
    simulationSeeds: new Set(simulationSeeds),
    parameterSeeds: new Set(parameterSeeds),
    ledger: Object.freeze({
      split,
      scheduleId: `schema9-${split}`,
      seedRoots: SCHEMA9_SPLIT_SEED_ROOTS[split],
      producerEngineCommit: CONVERTER_ENGINE_COMMIT,
      producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
      generatorReceipts: Object.freeze({
        launch: Object.freeze({ sha256: "d".repeat(64), bytes: 10 }),
        completion: Object.freeze({ sha256: "e".repeat(64), bytes: 11 }),
      }),
      scheduleProfile: SCHEMA9_SCHEDULE_PROFILE,
      sourceTrace: Object.freeze({
        sha256: "f".repeat(64),
        bytes: 100,
        games: 25,
        zeroPlyGames: 24,
        gameIds: Object.freeze(gameIds),
        simulationSeeds: Object.freeze(simulationSeeds),
        parameterSeeds: Object.freeze(parameterSeeds),
        gameIdSetSha256: setDigest(gameIds),
        simulationSeedSetSha256: setDigest(simulationSeeds),
        parameterSeedSetSha256: setDigest(parameterSeeds),
        labelCountsByColor: Object.freeze({
          white: labelCounts,
          black: labelCounts,
        }),
      }),
      converted: Object.freeze({
        sha256: "3".repeat(64),
        bytes: 20,
        rows: 1,
        games: 1,
        gameIds: convertedGameIds,
        simulationSeeds: convertedSimulationSeeds,
        gameIdSetSha256: setDigest(convertedGameIds),
        simulationSeedSetSha256: setDigest(convertedSimulationSeeds),
      }),
    }),
  });
}

function canonicalAuthenticatedSplits(): readonly AuthenticatedSchema9Split[] {
  return SCHEMA9_LEDGER_SPLITS.map((split, index) =>
    fakeAuthenticatedSplit(split, `game-${String(index)}`, (index * 100) + 1)
  );
}

function pythonCompatibleAuthenticatedSplit(
  split: Schema9LedgerSplit,
  splitIndex: number,
): AuthenticatedSchema9Split {
  const gameIds = Array.from(
    { length: 2_500 },
    (_, index) => `${split}-cross-language-${String(index).padStart(4, "0")}`,
  );
  const simulationSeeds = Array.from(
    { length: 2_500 },
    (_, index) => ((splitIndex + 1) * 100_000) + index,
  );
  const parameterSeeds = Array.from(
    { length: 5_000 },
    (_, index) => 1_000_000 + (splitIndex * 10_000) + index,
  );
  const setDigest = (values: readonly (string | number)[]): string =>
    createHash("sha256")
      .update(canonicalJsonBytes(values))
      .digest("hex");
  const labelCounts = Object.freeze(Object.fromEntries(
    CAPTURABLE_HYPOTHESIS_RULE_IDS.map((ruleId) => [ruleId, 100]),
  ));
  return Object.freeze({
    gameIds: new Set(gameIds),
    simulationSeeds: new Set(simulationSeeds),
    parameterSeeds: new Set(parameterSeeds),
    ledger: Object.freeze({
      split,
      scheduleId: `schema9-${split}`,
      seedRoots: SCHEMA9_SPLIT_SEED_ROOTS[split],
      producerEngineCommit: CONVERTER_ENGINE_COMMIT,
      producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
      generatorReceipts: Object.freeze({
        launch: Object.freeze({ sha256: "d".repeat(64), bytes: 100 }),
        completion: Object.freeze({ sha256: "e".repeat(64), bytes: 120 }),
      }),
      scheduleProfile: SCHEMA9_SCHEDULE_PROFILE,
      sourceTrace: Object.freeze({
        sha256: "f".repeat(64),
        bytes: 100_000,
        games: 2_500,
        zeroPlyGames: 0,
        gameIds: Object.freeze(gameIds),
        simulationSeeds: Object.freeze(simulationSeeds),
        parameterSeeds: Object.freeze(parameterSeeds),
        gameIdSetSha256: setDigest(gameIds),
        simulationSeedSetSha256: setDigest(simulationSeeds),
        parameterSeedSetSha256: setDigest(parameterSeeds),
        labelCountsByColor: Object.freeze({
          white: labelCounts,
          black: labelCounts,
        }),
      }),
      converted: Object.freeze({
        sha256: String(splitIndex + 1).repeat(64),
        bytes: 200_000,
        rows: 2_500,
        games: 2_500,
        gameIds: Object.freeze(gameIds),
        simulationSeeds: Object.freeze(simulationSeeds),
        gameIdSetSha256: setDigest(gameIds),
        simulationSeedSetSha256: setDigest(simulationSeeds),
      }),
    }),
  });
}

function metadataOptions(
  producerEngineCommit: string,
  policy: "exact/v1" | "converter-ancestor/v1",
  ancestorAccepted = true,
): Schema9CorpusLedgerOptions {
  const splits = Object.fromEntries(
    SCHEMA9_LEDGER_SPLITS.map((split) => [
      split,
      {
        tracePath: `${split}-trace`,
        convertedPath: `${split}-converted`,
        launchReceiptPath: `${split}-launch`,
        completionReceiptPath: `${split}-completion`,
        scheduleId: `schema9-${split}`,
        seedRoots: SCHEMA9_SPLIT_SEED_ROOTS[split],
        producerEngineCommit,
      },
    ]),
  ) as Readonly<Record<Schema9LedgerSplit, Schema9SplitFiles>>;
  return Object.freeze({
    guesserCommit: GUESSER_COMMIT,
    converterEngineCommit: CONVERTER_ENGINE_COMMIT,
    producerConverterPolicy: policy,
    repositoryVerifier: repositoryVerifier(ancestorAccepted),
    assignmentScheduler: Object.freeze({ assignments: () => [] }),
    splits,
  });
}

describe("schema-9 corpus ledger", () => {
  it("freezes the public generator receipt v3 configuration", () => {
    expect(SCHEMA9_CORPUS_LEDGER_VERSION).toBe(3);
    expect(SCHEMA9_GENERATOR_RECEIPT_VERSION).toBe(3);
    expect(SCHEMA9_LEDGER_VERIFICATION_RECEIPT_VERSION).toBe(2);
    expect(SCHEMA9_GENERATION_CONFIG).toStrictEqual({
      maxPlies: 120,
      maxDepth: 2,
      maxNodes: 50_000,
      temperatureCp: 35,
      topK: 8,
      leafCacheEntries: 16_384,
      leafCacheHistoryMode: "full",
      opponentAggregation: "worst-case",
      evaluator: {
        kind: "material",
        version: 1,
        evaluatorId: "drawback-material/v1",
      },
      opponentHypotheses: {
        kind: "unrestricted-baseline",
        version: 1,
      },
    });
  });

  it.each(GENERATION_CONFIG_VALUE_MUTATIONS)(
    "rejects generationConfig value mutation: $name",
    async ({ path, replacement }) => {
      const fixture = await splitFixture();
      await mutateLaunchReceipt(fixture.files, (receipt) => {
        setFixturePath(generationConfigOf(receipt), path, replacement);
      });
      await expect(authenticateSchema9SplitWithRuleContract(
        "train",
        fixture.files,
        ["vegan"],
        fixtureScheduler(fixture.trace),
      )).rejects.toThrow(
        "launch receipt generation config is unsupported",
      );
    },
  );

  it.each(GENERATION_CONFIG_FIELD_MUTATIONS)(
    "rejects receipt field drift: $name",
    async ({ expectedError, mutate }) => {
      const fixture = await splitFixture();
      await mutateLaunchReceipt(fixture.files, mutate);
      await expect(authenticateSchema9SplitWithRuleContract(
        "train",
        fixture.files,
        ["vegan"],
        fixtureScheduler(fixture.trace),
      )).rejects.toThrow(expectedError);
    },
  );

  it.each(PRODUCER_RUNTIME_IDENTITY_MUTATIONS)(
    "rejects launch producer runtime identity drift: $name",
    async ({ mutate }) => {
      const fixture = await splitFixture();
      await mutateLaunchReceipt(fixture.files, mutate);
      await expect(authenticateSchema9SplitWithRuleContract(
        "train",
        fixture.files,
        ["vegan"],
        fixtureScheduler(fixture.trace),
      )).rejects.toThrow();
    },
  );

  it.each(PRODUCER_RUNTIME_IDENTITY_MUTATIONS)(
    "rejects completion producer runtime identity drift: $name",
    async ({ mutate }) => {
      const fixture = await splitFixture();
      await mutateCompletionReceipt(fixture.files, mutate);
      await expect(authenticateSchema9SplitWithRuleContract(
        "train",
        fixture.files,
        ["vegan"],
        fixtureScheduler(fixture.trace),
      )).rejects.toThrow();
    },
  );

  it("rejects a well-formed completion runtime that differs from launch", async () => {
    const fixture = await splitFixture();
    await mutateCompletionReceipt(fixture.files, (completion) => {
      completion["producerRuntimeIdentity"] =
        ALTERNATE_PRODUCER_RUNTIME_IDENTITY;
    });
    await expect(authenticateSchema9SplitWithRuleContract(
      "train",
      fixture.files,
      ["vegan"],
      fixtureScheduler(fixture.trace),
    )).rejects.toThrow("completion producer runtime identity is inconsistent");
  });

  it.each(COMPLETION_DIGEST_TYPE_MUTATIONS)(
    "rejects non-string completion digest: $name",
    async ({ field, replacement }) => {
      const fixture = await splitFixture();
      await mutateCompletionReceipt(fixture.files, (completion) => {
        if (field === "launchReceiptSha256") {
          completion[field] = replacement;
          return;
        }
        mutableFixtureObject(completion["output"], "completion output")[
          "sha256"
        ] = replacement;
      });
      await expect(authenticateSchema9SplitWithRuleContract(
        "train",
        fixture.files,
        ["vegan"],
        fixtureScheduler(fixture.trace),
      )).rejects.toThrow("must be a lowercase SHA-256");
    },
  );

  it("matches the Engine producer-runtime golden aggregate", () => {
    expect(createHash("sha256")
      .update(canonicalJsonBytes(PRODUCER_RUNTIME_GOLDEN_PAYLOAD))
      .digest("hex"))
      .toBe(PRODUCER_RUNTIME_GOLDEN_SHA256);
    expect(checkedSchema9ProducerRuntimeIdentity({
      ...PRODUCER_RUNTIME_GOLDEN_PAYLOAD,
      aggregateSha256: PRODUCER_RUNTIME_GOLDEN_SHA256,
    })).toEqual({
      ...PRODUCER_RUNTIME_GOLDEN_PAYLOAD,
      aggregateSha256: PRODUCER_RUNTIME_GOLDEN_SHA256,
    });
  });

  it.each(SOURCE_TRACE_CONFIG_MUTATIONS)(
    "rejects realized source config mutation: $name",
    async ({ path, replacement }) => {
      const fixture = await splitFixture();
      await mutateSourceTrace(fixture.files, (trace) => {
        setFixturePath(trace, path, replacement);
      });
      await expect(authenticateSchema9SplitWithRuleContract(
        "train",
        fixture.files,
        ["vegan"],
        fixtureScheduler(fixture.trace),
      )).rejects.toThrow(
        "source trace differs from the authenticated generation config",
      );
    },
  );

  it("rejects a version-2 completion receipt", async () => {
    const fixture = await splitFixture();
    const completion = mutableFixtureObject(
      JSON.parse(
        await readFile(fixture.files.completionReceiptPath, "utf8"),
      ) as unknown,
      "completion receipt",
    );
    completion["version"] = 2;
    await writeFile(
      fixture.files.completionReceiptPath,
      `${JSON.stringify(completion)}\n`,
      "utf8",
    );
    await expect(authenticateSchema9SplitWithRuleContract(
      "train",
      fixture.files,
      ["vegan"],
      fixtureScheduler(fixture.trace),
    )).rejects.toThrow("completion receipt identity is inconsistent");
  });

  it("authenticates exact converter bytes and rejects file or receipt tampering", async () => {
    const fixture = await splitFixture();
    const authenticated = await authenticateSchema9SplitWithRuleContract(
      "train",
      fixture.files,
      ["vegan"],
      fixtureScheduler(fixture.trace),
    );
    expect(authenticated.ledger.sourceTrace).toMatchObject({
      games: 1,
      zeroPlyGames: 0,
      labelCountsByColor: {
        white: { vegan: 1 },
        black: { vegan: 1 },
      },
    });
    expect(authenticated.ledger.sourceTrace.gameIds).toEqual([
      fixture.trace.gameId,
    ]);
    expect(authenticated.ledger.sourceTrace.simulationSeeds).toEqual([
      fixture.trace.seed,
    ]);
    expect(authenticated.ledger.converted).toMatchObject({
      rows: 1,
      games: 1,
      gameIds: [fixture.trace.gameId],
      simulationSeeds: [fixture.trace.seed],
    });

    await appendFile(fixture.files.convertedPath, " ");
    await expect(
      authenticateSchema9SplitWithRuleContract(
        "train",
        fixture.files,
        ["vegan"],
        fixtureScheduler(fixture.trace),
      ),
    ).rejects.toThrow(/converted dataset/u);

    await writeFile(
      fixture.files.launchReceiptPath,
      `${JSON.stringify({
        format: "fixture-launch",
        privateLocation: "../private/source.ndjson",
      })}\n`,
      "utf8",
    );
    await expect(
      authenticateSchema9SplitWithRuleContract(
        "train",
        fixture.files,
        ["vegan"],
        fixtureScheduler(fixture.trace),
      ),
    ).rejects.toThrow("private path or user data");
  }, 60_000);

  it("assembles and atomically publishes the closed path-free artifact", async () => {
    const artifact = assembleSchema9CorpusLedger(
      {
        guesserCommit: GUESSER_COMMIT,
        converterEngineCommit: CONVERTER_ENGINE_COMMIT,
        execution: EXECUTION_IDENTITY,
        producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
      },
      "exact/v1",
      canonicalAuthenticatedSplits(),
    );
    expect(artifact.opportunityContract).toEqual({
      authorityId: "capturable-king/v1",
      symbolicFeatureVersion: 9,
      opportunityFeatureVersion: 1,
      ruleIds: CAPTURABLE_HYPOTHESIS_RULE_IDS,
      fields: RULE_OPPORTUNITY_FEATURE_FIELDS,
      shape: [25, 4],
    });
    expect(artifact.scheduleContract).toEqual({
      authorityId: "capturable25-schema9-opportunity/v1",
      seedStreams: SCHEMA9_SEED_STREAMS,
    });
    expect(artifact.splits.map((split) => split.seedRoots)).toEqual(
      SCHEMA9_LEDGER_SPLITS.map(
        (split) => SCHEMA9_SPLIT_SEED_ROOTS[split],
      ),
    );
    const root = await mkdtemp(
      join(tmpdir(), "drawback-guesser-schema9-ledger-"),
    );
    cleanupDirectories.push(root);
    const output = join(root, "ledger.json");
    const written = await publishSchema9CorpusLedgerArtifactAtomic(
      output,
      artifact,
    );
    const bytes = await readFile(output);
    expect(written.sha256).toBe(
      createHash("sha256").update(bytes).digest("hex"),
    );
    expect(await schema9CorpusLedgerFileSha256(output)).toBe(written.sha256);
    expect(
      verifySchema9CorpusLedgerReconstruction(bytes, artifact),
    ).toEqual(artifact);
    expect(bytes.toString("utf8")).not.toContain(root);
    expect(bytes.toString("utf8")).not.toContain(".ndjson");
    await expect(
      publishSchema9CorpusLedgerArtifactAtomic(output, artifact),
    ).rejects.toThrow("already exists");
    const recovered = await publishOrAuthenticateSchema9CorpusLedgerArtifactAtomic(
      output,
      artifact,
    );
    expect(recovered).toMatchObject({
      created: false,
      sha256: written.sha256,
    });
    expect(await readFile(output)).toEqual(bytes);
    expect(() =>
      verifySchema9CorpusLedgerReconstruction(
        Buffer.concat([bytes, Buffer.from(" ")]),
        artifact,
      )
    ).toThrow("not canonical");
  });

  it("recovers an exact ledger and receipt pair but rejects mismatches", async () => {
    const artifact = assembleSchema9CorpusLedger(
      {
        guesserCommit: GUESSER_COMMIT,
        converterEngineCommit: CONVERTER_ENGINE_COMMIT,
        execution: EXECUTION_IDENTITY,
        producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
      },
      "exact/v1",
      canonicalAuthenticatedSplits(),
    );
    const root = await mkdtemp(join(tmpdir(), "schema9-pair-recovery-"));
    cleanupDirectories.push(root);
    const ledgerPath = join(root, "ledger.json");
    const ledger = await publishOrAuthenticateSchema9CorpusLedgerArtifactAtomic(
      ledgerPath,
      artifact,
    );
    expect(ledger.created).toBe(true);
    const receipt = createSchema9LedgerVerificationReceipt(
      artifact,
      ledger.sha256,
    );
    const receiptPath = join(root, "receipt.json");
    const firstReceipt = await publishOrAuthenticateSchema9LedgerVerificationReceipt(
      receiptPath,
      receipt,
    );
    expect(firstReceipt.created).toBe(true);
    await expect(
      publishOrAuthenticateSchema9CorpusLedgerArtifactAtomic(
        ledgerPath,
        artifact,
      ),
    ).resolves.toMatchObject({ created: false, sha256: ledger.sha256 });
    await expect(
      publishOrAuthenticateSchema9LedgerVerificationReceipt(
        receiptPath,
        receipt,
      ),
    ).resolves.toMatchObject({
      created: false,
      sha256: firstReceipt.sha256,
    });

    await writeFile(join(root, "mismatch.json"), "{}", "utf8");
    await expect(
      publishOrAuthenticateSchema9CorpusLedgerArtifactAtomic(
        join(root, "mismatch.json"),
        artifact,
      ),
    ).rejects.toThrow("inconsistent");
    expect(await readFile(join(root, "mismatch.json"), "utf8")).toBe("{}");
  });

  it("stops corpus authentication after delayed in-process cancellation", async () => {
    const controller = new AbortController();
    const cancellation = new Error("fixture corpus cancellation");
    const options = metadataOptions(CONVERTER_ENGINE_COMMIT, "exact/v1");
    const verifier: Schema9RepositoryVerifier = Object.freeze({
      ...options.repositoryVerifier,
      pinnedEngineCommitAt: async () => {
        await new Promise<void>((resolveDelay) => {
          setTimeout(resolveDelay, 0);
        });
        controller.abort(cancellation);
        return CONVERTER_ENGINE_COMMIT;
      },
    });

    await expect(createSchema9CorpusLedger({
      ...options,
      repositoryVerifier: verifier,
      signal: controller.signal,
    })).rejects.toBe(cancellation);
  });

  it("does not publish a ledger after delayed in-process cancellation", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-ledger-cancel-"));
    cleanupDirectories.push(root);
    const output = join(root, "ledger.json");
    const artifact = assembleSchema9CorpusLedger(
      {
        guesserCommit: GUESSER_COMMIT,
        converterEngineCommit: CONVERTER_ENGINE_COMMIT,
        execution: EXECUTION_IDENTITY,
        producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
      },
      "exact/v1",
      canonicalAuthenticatedSplits(),
    );
    const controller = new AbortController();
    const cancellation = new Error("fixture ledger publication cancellation");
    const publication = publishSchema9CorpusLedgerArtifactAtomic(
      output,
      artifact,
      controller.signal,
    );
    queueMicrotask(() => {
      controller.abort(cancellation);
    });

    await expect(publication).rejects.toBe(cancellation);
    await expect(readFile(output)).rejects.toThrow();
    expect((await readdir(root)).filter((entry) => entry.includes(".tmp-")))
      .toEqual([]);
  });

  it("does not publish a verification receipt after delayed cancellation", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-receipt-cancel-"));
    cleanupDirectories.push(root);
    const output = join(root, "verification.json");
    const artifact = assembleSchema9CorpusLedger(
      {
        guesserCommit: GUESSER_COMMIT,
        converterEngineCommit: CONVERTER_ENGINE_COMMIT,
        execution: EXECUTION_IDENTITY,
        producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
      },
      "exact/v1",
      canonicalAuthenticatedSplits(),
    );
    const receipt = createSchema9LedgerVerificationReceipt(
      artifact,
      "9".repeat(64),
    );
    const controller = new AbortController();
    const cancellation = new Error("fixture receipt publication cancellation");
    const publication = writeSchema9LedgerVerificationReceiptAtomic(
      output,
      receipt,
      controller.signal,
    );
    queueMicrotask(() => {
      controller.abort(cancellation);
    });

    await expect(publication).rejects.toBe(cancellation);
    await expect(readFile(output)).rejects.toThrow();
    expect((await readdir(root)).filter((entry) => entry.includes(".tmp-")))
      .toEqual([]);
  });

  it("rejects any split runtime that differs from the reproduced producer", () => {
    const splits = [...canonicalAuthenticatedSplits()];
    const first = splits[0];
    if (first === undefined) {
      throw new Error("Canonical split fixture is empty.");
    }
    splits[0] = Object.freeze({
      ...first,
      ledger: Object.freeze({
        ...first.ledger,
        producerRuntimeIdentity: ALTERNATE_PRODUCER_RUNTIME_IDENTITY,
      }),
    });
    expect(() => assembleSchema9CorpusLedger(
      {
        guesserCommit: GUESSER_COMMIT,
        converterEngineCommit: CONVERTER_ENGINE_COMMIT,
        execution: EXECUTION_IDENTITY,
        producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
      },
      "exact/v1",
      splits,
    )).toThrow("does not match the reproduced producer build");
  });

  it("rejects cross-split game IDs and every seed-stream overlap", () => {
    expect(() => {
      assertSchema9SplitsDisjoint([
        fakeAuthenticatedSplit("train", "game-a", 1),
        fakeAuthenticatedSplit("validation-a", "game-a", 2),
      ]);
    }).toThrow("overlap by game ID");
    expect(() => {
      assertSchema9SplitsDisjoint([
        fakeAuthenticatedSplit("train", "game-a", 1),
        fakeAuthenticatedSplit("validation-a", "game-b", 1),
      ]);
    }).toThrow("overlap by simulation seed");
    const parameterOverlap = fakeAuthenticatedSplit(
      "validation-a",
      "game-b",
      100,
    );
    expect(() => {
      assertSchema9SplitsDisjoint([
        fakeAuthenticatedSplit("train", "game-a", 1),
        {
          ...parameterOverlap,
          parameterSeeds: new Set([1_001]),
        },
      ]);
    }).toThrow("seed streams overlap");
  });

  it("rejects either color's exact 25-label imbalance", () => {
    const balanced = Object.fromEntries(
      CAPTURABLE_HYPOTHESIS_RULE_IDS.map((ruleId) => [ruleId, 1]),
    );
    expect(() => {
      assertExactSchema9LabelBalance(25, balanced, balanced);
    }).not.toThrow();
    const imbalanced = { ...balanced, vegan: 2, irresistible: 0 };
    expect(() => {
      assertExactSchema9LabelBalance(25, imbalanced, balanced);
    }).toThrow("not exactly label-balanced");
    expect(() => {
      assertExactSchema9LabelBalance(25, balanced, imbalanced);
    }).toThrow("not exactly label-balanced");
    expect(() => {
      assertExactSchema9LabelBalance(
        25,
        { ...balanced, unexpected: 0 },
        balanced,
      );
    }).toThrow("exactly the declared rule IDs");
  });

  it("fails closed on producer mismatch unless ancestry is authenticated", async () => {
    await expect(
      verifySchema9RepositoryIdentity(
        metadataOptions(DESCENDANT_ENGINE_COMMIT, "exact/v1"),
      ),
    ).rejects.toThrow("Exact producer/converter policy rejects");
    await expect(
      verifySchema9RepositoryIdentity(
        metadataOptions(
          DESCENDANT_ENGINE_COMMIT,
          "converter-ancestor/v1",
        ),
      ),
    ).resolves.toEqual({
      guesserCommit: GUESSER_COMMIT,
      converterEngineCommit: CONVERTER_ENGINE_COMMIT,
      execution: EXECUTION_IDENTITY,
      producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
    });
    await expect(
      verifySchema9RepositoryIdentity(
        metadataOptions(
          DESCENDANT_ENGINE_COMMIT,
          "converter-ancestor/v1",
          false,
        ),
      ),
    ).rejects.toThrow("unrelated producer commit");

    const mixed = metadataOptions(
      CONVERTER_ENGINE_COMMIT,
      "converter-ancestor/v1",
    );
    const mixedSplits = {
      ...mixed.splits,
      test: Object.freeze({
        ...mixed.splits.test,
        producerEngineCommit: DESCENDANT_ENGINE_COMMIT,
      }),
    };
    await expect(verifySchema9RepositoryIdentity(Object.freeze({
      ...mixed,
      splits: mixedSplits,
      repositoryVerifier: Object.freeze({
        ...mixed.repositoryVerifier,
        producerRuntimeIdentityAt: (engineCommit: string) =>
          Promise.resolve(
            engineCommit === DESCENDANT_ENGINE_COMMIT
              ? ALTERNATE_PRODUCER_RUNTIME_IDENTITY
              : PRODUCER_RUNTIME_IDENTITY,
          ),
      }),
    }))).rejects.toThrow(
      "splits do not reproduce one exact producer runtime identity",
    );
  });

  it("rejects paths, user tokens, duplicate keys, and zero-ply drift", () => {
    expect(() => checkedSha256(["a".repeat(64)], "digest"))
      .toThrow("must be a lowercase SHA-256");
    expect(() => checkedGitCommit([GUESSER_COMMIT], "commit"))
      .toThrow("must be a full lowercase Git commit");
    expect(() => checkedScheduleId(["schema9-fixture"], "schedule"))
      .toThrow("must be a canonical path-free schedule identifier");
    expect(() =>
      checkedSchema9SeedRoots(
        SCHEMA9_SPLIT_SEED_ROOTS["validation-b"],
        "validation-b",
      )
    ).not.toThrow();
    expect(() =>
      checkedSchema9SeedRoots([1, 2, 3], "validation-b")
    ).toThrow("must exactly match the frozen");
    expect(() => {
      assertPathFreeJson(
        { privateLocation: "../private/corpus.ndjson" },
        "receipt",
      );
    }).toThrow("private path or user data");
    expect(() => {
      assertPathFreeJson(
        { "../private/corpus.ndjson": true },
        "receipt",
      );
    }).toThrow("private path or user data");
    expect(() => {
      assertPathFreeJson(
        { account: "fixture-private-user" },
        "receipt",
        ["fixture-private-user"],
      );
    }).toThrow("private path or user data");
    expect(() =>
      parseJsonWithoutDuplicateKeys(
        "{\"format\":\"one\",\"format\":\"two\"}",
        "receipt",
      )
    ).toThrow("not strict JSON");
    expect(() => {
      const overflow = parseJsonWithoutDuplicateKeys(
        '{"value":1e400}',
        "receipt",
      );
      assertPathFreeJson(overflow, "receipt");
    }).toThrow("non-canonical JSON number");
    expect(() => {
      assertPathFreeJson({ value: -0 }, "receipt");
    })
      .toThrow("non-canonical JSON number");
    for (const identifier of [
      "http://internal/run",
      "c:private",
      "folder/run",
      "api-token-run",
      "con",
      "a".repeat(65),
    ]) {
      expect(() => checkedScheduleId(identifier, "scheduleId")).toThrow(
        "path-free schedule identifier",
      );
    }
    expect(() => {
      assertScheduledConversionAccounting(25, 24, 1);
    }).not.toThrow();
    expect(() => {
      assertScheduledConversionAccounting(25, 1, 25);
    }).toThrow("lost scheduled zero-ply games");
  });

  it("matches environment identities only at token boundaries", () => {
    const rootIdentity = ["root"] as const;
    expect(() => {
      assertPathFreeJson(
        {
          seedRoots: [1, 2, 3],
          deeplyRooted: true,
          "𐐀root": true,
          "root𐐀": true,
          "𝟘root": true,
          "root𝟘": true,
        },
        "receipt",
        rootIdentity,
      );
    }).not.toThrow();
    for (const value of [
      { account: "root" },
      { account: "run-root-v1" },
      { root: true },
      { "run-root-v1": true },
      { account: "/home/root/corpus.ndjson" },
    ]) {
      expect(() => {
        assertPathFreeJson(value, "receipt", rootIdentity);
      }).toThrow("private path or user data");
    }
  });

  it("rejects false schedule assignments and typed receipt drift", async () => {
    const fixture = await splitFixture();
    const falseScheduler: Schema9AssignmentScheduler = Object.freeze({
      assignments: () => [Object.freeze({
        gameIndex: fixture.trace.gameIndex,
        gameId: fixture.trace.gameId,
        seed: fixture.trace.seed + 1,
        parameterSeeds: fixture.trace.parameterSeeds,
        whiteRuleId: "vegan",
        blackRuleId: "vegan",
        initialFen: fixture.trace.initialPosition.fen,
        initialReplaySha256: "0".repeat(64),
      })],
    });
    await expect(authenticateSchema9SplitWithRuleContract(
      "train",
      fixture.files,
      ["vegan"],
      falseScheduler,
    )).rejects.toThrow("differs from the frozen schedule");

    const launch = JSON.parse(
      await readFile(fixture.files.launchReceiptPath, "utf8"),
    ) as Record<string, unknown>;
    launch["scheduleId"] = "another-schedule";
    await writeFile(
      fixture.files.launchReceiptPath,
      `${JSON.stringify(launch)}\n`,
      "utf8",
    );
    await expect(authenticateSchema9SplitWithRuleContract(
      "train",
      fixture.files,
      ["vegan"],
      fixtureScheduler(fixture.trace),
    )).rejects.toThrow("launch receipt identity is inconsistent");
  });

  it("enforces the same ledger byte bound before create and verify", async () => {
    expect(() => {
      assertSchema9CorpusLedgerByteLength(Buffer.alloc(1));
    })
      .not.toThrow();
    expect(() => {
      assertSchema9CorpusLedgerByteLength(Buffer.alloc(8 * 1024 * 1024));
    }).not.toThrow();
    expect(() => {
      assertSchema9CorpusLedgerByteLength(
        Buffer.alloc((8 * 1024 * 1024) + 1),
      );
    }).toThrow("must be from 1 through");
    const root = await mkdtemp(join(tmpdir(), "schema9-oversize-"));
    cleanupDirectories.push(root);
    const output = join(root, "oversized.json");
    const artifact = assembleSchema9CorpusLedger(
      {
        guesserCommit: GUESSER_COMMIT,
        converterEngineCommit: CONVERTER_ENGINE_COMMIT,
        execution: EXECUTION_IDENTITY,
        producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
      },
      "exact/v1",
      canonicalAuthenticatedSplits(),
    );
    const oversized = {
      ...artifact,
      padding: "x".repeat(8 * 1024 * 1024),
    } as unknown as Schema9CorpusLedger;
    await expect(publishSchema9CorpusLedgerArtifactAtomic(output, oversized))
      .rejects.toThrow("must be from 1 through");
    await expect(readFile(output)).rejects.toThrow();
  });

  it("round-trips a TypeScript ledger receipt through the Python loader", async () => {
    const artifact = assembleSchema9CorpusLedger(
      {
        guesserCommit: GUESSER_COMMIT,
        converterEngineCommit: CONVERTER_ENGINE_COMMIT,
        execution: EXECUTION_IDENTITY,
        producerRuntimeIdentity: PRODUCER_RUNTIME_IDENTITY,
      },
      "exact/v1",
      SCHEMA9_LEDGER_SPLITS.map(pythonCompatibleAuthenticatedSplit),
    );
    const root = await mkdtemp(join(tmpdir(), "schema9-cross-language-"));
    cleanupDirectories.push(root);
    const ledgerPath = join(root, "schema9-corpus-ledger.json");
    const written = await publishSchema9CorpusLedgerArtifactAtomic(
      ledgerPath,
      artifact,
    );
    const receipt = createSchema9LedgerVerificationReceipt(
      artifact,
      written.sha256,
    );
    const receiptPath = join(
      root,
      `schema9-ledger-verification-${written.sha256}.json`,
    );
    const receiptWritten = await writeSchema9LedgerVerificationReceiptAtomic(
      receiptPath,
      receipt,
    );
    const python = [
      "import sys",
      "from pathlib import Path",
      "sys.path.insert(0, sys.argv[4])",
      "from drawback_ml.capturable_opportunity_workflow import _load_corpus_ledger",
      "loaded = _load_corpus_ledger(Path(sys.argv[1]), sys.argv[2], sys.argv[3])",
      "print(loaded.sha256)",
    ].join("; ");
    const result = await execFileAsync(
      "python",
      [
        "-c",
        python,
        ledgerPath,
        written.sha256,
        receiptWritten.sha256,
        resolve("ml/training"),
      ],
      { windowsHide: true },
    );
    expect(result.stdout.trim()).toBe(written.sha256);

    const tampered = {
      ...receipt,
      inputSetSha256: "0".repeat(64),
    };
    const tamperedPayload = {
      ...tampered,
      contentSha256: createHash("sha256")
        .update(canonicalJsonBytes(Object.fromEntries(
          Object.entries(tampered).filter(([key]) => key !== "contentSha256"),
        )))
        .digest("hex"),
    };
    await writeFile(receiptPath, canonicalJsonBytes(tamperedPayload));
    await expect(execFileAsync(
      "python",
      [
        "-c",
        python,
        ledgerPath,
        written.sha256,
        receiptWritten.sha256,
        resolve("ml/training"),
      ],
      { windowsHide: true },
    )).rejects.toThrow();
  }, 60_000);
});
