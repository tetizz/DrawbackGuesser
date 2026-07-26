import { createHash, randomUUID } from "node:crypto";
import { readFile, readdir, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  writeTrainingDatasetNdjsonFileAtomic,
} from "./output.js";
import { traceFixture } from "./test-fixture.js";

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
});
