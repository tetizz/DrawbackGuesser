import { createHash, randomUUID } from "node:crypto";
import {
  lstat,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  publishTrainingDatasetNoClobberForTesting,
  writeTrainingDatasetNdjsonFileAtomic,
} from "./output.js";
import { traceFixture } from "./test-fixture.js";
import {
  playerPrivateTraceFixture,
} from "./player-private-test-fixture.js";

const cleanupPaths: string[] = [];

afterEach(async () => {
  await Promise.all(
    cleanupPaths.splice(0).map((path) => rm(path, { force: true })),
  );
});

function temporaryDataset(label: string): string {
  const path = join(
    tmpdir(),
    `drawback-guesser-${label}-${randomUUID()}.ndjson`,
  );
  cleanupPaths.push(path);
  return path;
}

describe("private training dataset output", () => {
  it("rejects an empty source instead of publishing an empty corpus", async () => {
    const path = temporaryDataset("empty");
    await expect(
      writeTrainingDatasetNdjsonFileAtomic(path, []),
    ).rejects.toThrow("At least one private Engine trace is required");
    await expect(readFile(path)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("publishes deterministic content-addressed bytes without clobbering", async () => {
    const path = temporaryDataset("atomic");
    const traces = [
      traceFixture({ gameIndex: 0 }),
      traceFixture({ gameIndex: 1 }),
    ];
    const written = await writeTrainingDatasetNdjsonFileAtomic(path, traces);
    const bytes = await readFile(path);

    expect(written).toEqual({
      games: 2,
      rows: 4,
      bytes: bytes.byteLength,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      evaluatorCoverage: "none",
      evaluatorPolicyId: null,
      evaluatorEngineFingerprint: null,
      authorityId: "standard-chess/v1",
    });
    await expect(
      writeTrainingDatasetNdjsonFileAtomic(path, traces),
    ).rejects.toMatchObject({ code: "EEXIST" });
    expect(await readFile(path)).toEqual(bytes);
    if (process.platform !== "win32") {
      expect((await stat(path)).mode & 0o777).toBe(0o600);
    }
    expect(
      (await readdir(dirname(path))).some((entry) =>
        entry.startsWith(`${basename(path)}.tmp-`)
      ),
    ).toBe(false);
  });

  it("never deletes a replacement raced into its cleanup quarantine", async () => {
    const temporary = temporaryDataset("cleanup-race-temporary");
    const output = temporaryDataset("cleanup-race-output");
    const original = Buffer.from("authenticated dataset\n", "utf8");
    const replacement = Buffer.from("attacker replacement\n", "utf8");
    await writeFile(temporary, original, { flag: "wx", mode: 0o600 });
    const metadata = await lstat(temporary, { bigint: true });
    let quarantine = "";
    let displaced = "";

    await expect(publishTrainingDatasetNoClobberForTesting(
      temporary,
      output,
      Object.freeze({
        dev: metadata.dev,
        ino: metadata.ino,
        birthtimeNs: metadata.birthtimeNs,
      }),
      async (candidate) => {
        quarantine = candidate;
        displaced = `${candidate}.displaced`;
        cleanupPaths.push(quarantine, displaced);
        await rename(candidate, displaced);
        await writeFile(candidate, replacement, { flag: "wx", mode: 0o600 });
      },
    )).rejects.toMatchObject({ committed: true });

    expect(await readFile(output)).toEqual(original);
    expect(await readFile(quarantine)).toEqual(replacement);
    expect(await readFile(displaced)).toEqual(original);
  });

  it("syncs the parent only after the final dataset entry is authenticated", async () => {
    const temporary = temporaryDataset("directory-sync-temporary");
    const output = temporaryDataset("directory-sync-output");
    const original = Buffer.from("authenticated dataset\n", "utf8");
    await writeFile(temporary, original, { flag: "wx", mode: 0o600 });
    const metadata = await lstat(temporary, { bigint: true });
    const events: string[] = [];

    await publishTrainingDatasetNoClobberForTesting(
      temporary,
      output,
      Object.freeze({
        dev: metadata.dev,
        ino: metadata.ino,
        birthtimeNs: metadata.birthtimeNs,
      }),
      () => {
        events.push("quarantined");
        return Promise.resolve();
      },
      async (directory) => {
        events.push("directory-synced");
        expect(directory).toBe(dirname(output));
        await expect(readFile(temporary)).rejects.toMatchObject({
          code: "ENOENT",
        });
        await expect(readFile(output)).resolves.toEqual(original);
      },
    );

    expect(events).toEqual(["quarantined", "directory-synced"]);
  });

  it("accepts async traces and preserves monolithic shard byte identity", async () => {
    const monolithic = temporaryDataset("monolithic");
    const firstShard = temporaryDataset("first-shard");
    const secondShard = temporaryDataset("second-shard");
    const first = traceFixture({ gameIndex: 0 });
    const second = traceFixture({ gameIndex: 1 });
    async function* traces(): AsyncIterableIterator<unknown> {
      yield first;
      await Promise.resolve();
      yield second;
    }

    await writeTrainingDatasetNdjsonFileAtomic(monolithic, traces());
    await writeTrainingDatasetNdjsonFileAtomic(firstShard, [first]);
    await writeTrainingDatasetNdjsonFileAtomic(secondShard, [second]);

    expect(await readFile(monolithic)).toEqual(
      Buffer.concat([
        await readFile(firstShard),
        await readFile(secondShard),
      ]),
    );
  });

  it("rejects duplicate games and removes partial private output", async () => {
    const path = temporaryDataset("duplicate");
    const trace = traceFixture();

    await expect(
      writeTrainingDatasetNdjsonFileAtomic(path, [trace, trace]),
    ).rejects.toThrow(`Duplicate trace gameId ${trace.gameId}`);
    await expect(readFile(path)).rejects.toMatchObject({ code: "ENOENT" });
    expect(
      (await readdir(dirname(path))).some((entry) =>
        entry.startsWith(`${basename(path)}.tmp-`)
      ),
    ).toBe(false);
  });

  it("enforces one evaluator policy and optional release coverage", async () => {
    const requiredPath = temporaryDataset("required-coverage");
    await expect(
      writeTrainingDatasetNdjsonFileAtomic(
        requiredPath,
        [traceFixture()],
        { expectedEvaluatorCoverage: "uniform" },
      ),
    ).rejects.toThrow("does not match required uniform");

    const mixedPath = temporaryDataset("mixed-evaluator");
    await expect(
      writeTrainingDatasetNdjsonFileAtomic(mixedPath, [
        traceFixture({
          gameIndex: 0,
          evaluatorCoverage: "uniform",
          evaluatorFingerprint: "engine-a",
        }),
        traceFixture({
          gameIndex: 1,
          evaluatorCoverage: "uniform",
          evaluatorFingerprint: "engine-b",
        }),
      ]),
    ).rejects.toThrow("evaluator identity differs");
  });

  it("streams capturable traces and rejects mixed authorities", async () => {
    const path = temporaryDataset("capturable");
    const trace = playerPrivateTraceFixture();
    const written = await writeTrainingDatasetNdjsonFileAtomic(
      path,
      [trace],
      {
        expectedAuthorityId: "capturable-king/v1",
        expectedEvaluatorCoverage: "none",
      },
    );
    const rows = (await readFile(path, "utf8")).trimEnd().split("\n");

    expect(written).toMatchObject({
      authorityId: "capturable-king/v1",
      evaluatorCoverage: "none",
      games: 1,
      rows: 2,
    });
    expect(JSON.parse(rows[0] ?? "{}")).toMatchObject({
      authorityId: "capturable-king/v1",
      trueDrawback: "vegan",
    });

    const mixedPath = temporaryDataset("mixed-authority");
    await expect(
      writeTrainingDatasetNdjsonFileAtomic(mixedPath, [
        traceFixture(),
        trace,
      ]),
    ).rejects.toThrow("evaluator identity differs");
    await expect(readFile(mixedPath)).rejects.toMatchObject({
      code: "ENOENT",
    });
  });

  it("preserves capturable schema-9 bytes across monolithic and sharded output", async () => {
    const monolithic = temporaryDataset("capturable-monolithic");
    const firstShard = temporaryDataset("capturable-first-shard");
    const secondShard = temporaryDataset("capturable-second-shard");
    const first = playerPrivateTraceFixture({
      seed: 1,
      parameterSeeds: { white: 11, black: 12 },
    });
    const second = playerPrivateTraceFixture({
      seed: 2,
      parameterSeeds: { white: 21, black: 22 },
    });
    const options = {
      expectedAuthorityId: "capturable-king/v1" as const,
      expectedEvaluatorCoverage: "none" as const,
    };

    await writeTrainingDatasetNdjsonFileAtomic(
      monolithic,
      [first, second],
      options,
    );
    await writeTrainingDatasetNdjsonFileAtomic(firstShard, [first], options);
    await writeTrainingDatasetNdjsonFileAtomic(secondShard, [second], options);

    const monolithicBytes = await readFile(monolithic);
    expect(monolithicBytes).toEqual(
      Buffer.concat([
        await readFile(firstShard),
        await readFile(secondShard),
      ]),
    );
    const firstRow = JSON.parse(
      monolithicBytes.toString("utf8").split("\n")[0] ?? "{}",
    ) as Readonly<Record<string, unknown>>;
    expect(firstRow["opportunityFeatureVersion"]).toBe(1);
    expect(firstRow["symbolicActiveRuleOpportunityFeatures"]).toHaveLength(
      100,
    );
  });
});
