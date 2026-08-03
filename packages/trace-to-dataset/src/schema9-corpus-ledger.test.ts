import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import {
  appendFile,
  mkdtemp,
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
  publishSchema9CorpusLedgerArtifactAtomic,
  schema9CorpusLedgerFileSha256,
  SCHEMA9_LEDGER_SPLITS,
  SCHEMA9_EXECUTION_MANIFEST_ALGORITHM,
  SCHEMA9_GENERATOR_COMPLETION_FORMAT,
  SCHEMA9_GENERATOR_LAUNCH_FORMAT,
  SCHEMA9_GENERATOR_RECEIPT_VERSION,
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
  checkedSchema9SeedRoots,
  checkedScheduleId,
  parseJsonWithoutDuplicateKeys,
} from "./schema9-ledger-types.js";
import {
  createSchema9LedgerVerificationReceipt,
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
const cleanupDirectories: string[] = [];
const execFileAsync = promisify(execFile);

interface FixtureTrace {
  readonly schemaVersion: 2;
  readonly ruleset: { readonly version: 2 };
  readonly gameId: string;
  readonly seed: number;
  readonly gameIndex: number;
  readonly parameterSeeds: { readonly white: number; readonly black: number };
  readonly initialPosition: { readonly fen: string };
  readonly hypothesisPolicy: {
    readonly kind: "unrestricted-baseline";
    readonly version: 1;
  };
  readonly agents: {
    readonly white: { readonly searchPolicy: { readonly policyId: string } };
    readonly black: { readonly searchPolicy: { readonly policyId: string } };
  };
  readonly secrets: {
    readonly initial: {
      readonly white: { readonly drawbackId: string };
      readonly black: { readonly drawbackId: string };
    };
  };
  readonly plies: readonly { readonly ply: number }[];
}

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
  const trace: FixtureTrace = Object.freeze({
    schemaVersion: 2,
    ruleset: Object.freeze({ version: 2 }),
    gameId: "schema9-ledger-fixture-game",
    seed: 3_145_926,
    gameIndex: 0,
    parameterSeeds: Object.freeze({ white: 101, black: 102 }),
    initialPosition: Object.freeze({ fen: "fixture-fen" }),
    hypothesisPolicy: Object.freeze({
      kind: "unrestricted-baseline",
      version: 1,
    }),
    agents: Object.freeze({
      white: Object.freeze({
        searchPolicy: Object.freeze({
          policyId: SCHEMA9_SCHEDULE_PROFILE.policyId,
        }),
      }),
      black: Object.freeze({
        searchPolicy: Object.freeze({
          policyId: SCHEMA9_SCHEDULE_PROFILE.policyId,
        }),
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
    producerEngineCommit: CONVERTER_ENGINE_COMMIT,
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
    expect(await readFile(output)).toEqual(bytes);
    expect(() =>
      verifySchema9CorpusLedgerReconstruction(
        Buffer.concat([bytes, Buffer.from(" ")]),
        artifact,
      )
    ).toThrow("not canonical");
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
  });

  it("rejects paths, user tokens, duplicate keys, and zero-ply drift", () => {
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
