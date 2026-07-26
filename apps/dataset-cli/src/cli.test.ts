import { randomUUID } from "node:crypto";
import { readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  encodePlayerPrivateSimulationTraceRecord,
  encodePrivateSimulationTraceRecord,
} from "@drawbackengine/simulation-trace";
import { traceFixture } from "../../../packages/trace-to-dataset/src/test-fixture.js";
import {
  playerPrivateTraceFixture,
} from "../../../packages/trace-to-dataset/src/player-private-test-fixture.js";
import {
  parseDatasetCliArguments,
  runDatasetCli,
} from "./cli.js";
import { readPrivateTraceNdjson } from "./trace-input.js";

const cleanupPaths: string[] = [];

afterEach(async () => {
  await Promise.all(
    cleanupPaths.splice(0).map((path) => rm(path, { force: true })),
  );
});

function temporaryPath(label: string): string {
  const path = join(tmpdir(), `${label}-${randomUUID()}.ndjson`);
  cleanupPaths.push(path);
  return path;
}

describe("dataset CLI", () => {
  it("requires explicit distinct input and output paths", () => {
    expect(() => parseDatasetCliArguments([])).toThrow("--input");
    expect(() =>
      parseDatasetCliArguments([
        "--input",
        "same.ndjson",
        "--output",
        "same.ndjson",
      ], "C:\\fixture")
    ).toThrow("must be different");
    expect(() =>
      parseDatasetCliArguments([
        "--input",
        "input.ndjson",
        "--output",
        "output.ndjson",
        "--require-evaluator",
        "sometimes",
      ])
    ).toThrow("none or uniform");
    expect(() =>
      parseDatasetCliArguments([
        "--input",
        "input.ndjson",
        "--output",
        "output.ndjson",
        "--require-authority",
        "unknown/v1",
      ])
    ).toThrow("standard-chess/v1 or capturable-king/v1");
    expect(() =>
      parseDatasetCliArguments([
        "--input",
        "first.ndjson",
        "--input",
        "second.ndjson",
        "--output",
        "output.ndjson",
      ])
    ).toThrow("may appear only once");
  });

  it("streams private Engine traces into contract-checked rows", async () => {
    const inputPath = temporaryPath("drawback-engine-trace");
    const outputPath = temporaryPath("drawback-guesser-dataset");
    const trace = traceFixture();
    await writeFile(inputPath, encodePrivateSimulationTraceRecord(trace), {
      encoding: "utf8",
      mode: 0o600,
    });
    let stdout = "";

    await runDatasetCli(
      { inputPath, outputPath, expectedEvaluatorCoverage: "none" },
      {
        stdout: { write: (chunk) => { stdout += chunk; } },
        stderr: { write: () => undefined },
      },
    );

    const rows = (await readFile(outputPath, "utf8")).trimEnd().split("\n");
    expect(rows).toHaveLength(2);
    expect(JSON.parse(rows[0] ?? "{}")).toMatchObject({
      gameId: trace.gameId,
      move: "e2e4",
      trueDrawback: "vegan",
    });
    expect(stdout).toContain("2 private training rows from 1 games");
  });

  it("reports line numbers and bounds untrusted input lines", async () => {
    const invalidPath = temporaryPath("invalid-trace");
    await writeFile(invalidPath, "{}\nnot-json\n", "utf8");
    const invalid = readPrivateTraceNdjson(invalidPath);
    await expect(invalid.next()).rejects.toThrow(
      "Private trace line 1 is invalid",
    );

    const largePath = temporaryPath("large-trace");
    await writeFile(largePath, "123456789\n", "utf8");
    const large = readPrivateTraceNdjson(largePath, { maxLineBytes: 4 });
    await expect(large.next()).rejects.toThrow(
      "Private trace line 1 exceeds 4 bytes",
    );

    const utf8Path = temporaryPath("invalid-utf8");
    await writeFile(utf8Path, Buffer.from([0xff, 0x0a]));
    const invalidUtf8 = readPrivateTraceNdjson(utf8Path);
    await expect(invalidUtf8.next()).rejects.toThrow(
      "Private trace line 1 is not valid UTF-8",
    );
  });

  it("converts replay-verified capturable traces through the CLI", async () => {
    const inputPath = temporaryPath("player-private-trace");
    const outputPath = temporaryPath("capturable-dataset");
    const trace = playerPrivateTraceFixture({
      whiteRuleId: "triple-play",
    });
    await writeFile(
      inputPath,
      encodePlayerPrivateSimulationTraceRecord(trace),
      {
        encoding: "utf8",
        mode: 0o600,
      },
    );

    await runDatasetCli({
      inputPath,
      outputPath,
      expectedEvaluatorCoverage: "none",
      expectedAuthorityId: "capturable-king/v1",
    });

    const rows = (await readFile(outputPath, "utf8")).trimEnd().split("\n");
    expect(rows).toHaveLength(2);
    expect(JSON.parse(rows[0] ?? "{}")).toMatchObject({
      authorityId: "capturable-king/v1",
      trueDrawback: "triple-play",
      publicAuthorityPositionBefore: {
        authorityId: "capturable-king/v1",
      },
    });
  });
});
