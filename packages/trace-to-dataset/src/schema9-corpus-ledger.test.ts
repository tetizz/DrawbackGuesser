import { createHash } from "node:crypto";
import {
  appendFile,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
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
  assertSchema9SplitsDisjoint,
  publishSchema9CorpusLedgerArtifactAtomic,
  schema9CorpusLedgerFileSha256,
  SCHEMA9_LEDGER_SPLITS,
  SCHEMA9_SEED_STREAMS,
  SCHEMA9_SPLIT_SEED_ROOTS,
  verifySchema9CorpusLedgerReconstruction,
  verifySchema9RepositoryIdentity,
  type Schema9CorpusLedgerOptions,
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
  checkedSchema9SeedRoots,
  parseJsonWithoutDuplicateKeys,
} from "./schema9-ledger-types.js";

const GUESSER_COMMIT = "a".repeat(40);
const CONVERTER_ENGINE_COMMIT = "b".repeat(40);
const DESCENDANT_ENGINE_COMMIT = "c".repeat(40);
const cleanupDirectories: string[] = [];

interface FixtureTrace {
  readonly schemaVersion: 2;
  readonly ruleset: { readonly version: 2 };
  readonly gameId: string;
  readonly seed: number;
  readonly gameIndex: number;
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
  await writeFile(tracePath, `${JSON.stringify(trace)}\n`, "utf8");
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
  await writeFile(
    launchReceiptPath,
    `${JSON.stringify({
      format: "fixture-launch",
      version: 1,
      scheduleId: "fixture-train",
    }, null, 2)}\n`,
    "utf8",
  );
  await writeFile(
    completionReceiptPath,
    `${JSON.stringify({
      format: "fixture-completion",
      version: 1,
      completed: true,
    }, null, 2)}\n`,
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
  const convertedGameIds = Object.freeze([gameIds[0] as string]);
  const convertedSimulationSeeds = Object.freeze([simulationSeeds[0] as number]);
  const setDigest = (values: readonly (string | number)[]): string =>
    createHash("sha256")
      .update(`${JSON.stringify(values)}\n`)
      .digest("hex");
  return Object.freeze({
    gameIds: new Set(gameIds),
    simulationSeeds: new Set(simulationSeeds),
    ledger: Object.freeze({
      split,
      scheduleId: `schema9-${split}`,
      seedRoots: SCHEMA9_SPLIT_SEED_ROOTS[split],
      producerEngineCommit: CONVERTER_ENGINE_COMMIT,
      generatorReceipts: Object.freeze({
        launch: Object.freeze({ sha256: "d".repeat(64), bytes: 10 }),
        completion: Object.freeze({ sha256: "e".repeat(64), bytes: 11 }),
      }),
      sourceTrace: Object.freeze({
        sha256: "f".repeat(64),
        bytes: 100,
        games: 25,
        zeroPlyGames: 24,
        gameIds: Object.freeze(gameIds),
        simulationSeeds: Object.freeze(simulationSeeds),
        gameIdSetSha256: setDigest(gameIds),
        simulationSeedSetSha256: setDigest(simulationSeeds),
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
      ),
    ).rejects.toThrow("private path or user data");
  }, 60_000);

  it("assembles and atomically publishes the closed path-free artifact", async () => {
    const artifact = assembleSchema9CorpusLedger(
      {
        guesserCommit: GUESSER_COMMIT,
        converterEngineCommit: CONVERTER_ENGINE_COMMIT,
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

  it("rejects cross-split game-ID and simulation-seed overlap", () => {
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
      assertScheduledConversionAccounting(25, 24, 1);
    }).not.toThrow();
    expect(() => {
      assertScheduledConversionAccounting(25, 1, 25);
    }).toThrow("lost scheduled zero-ply games");
  });
});
