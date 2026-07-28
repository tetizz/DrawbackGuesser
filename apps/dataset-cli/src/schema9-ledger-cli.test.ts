import { resolve } from "node:path";
import {
  SCHEMA9_SPLIT_SEED_ROOTS,
} from "@drawbackguesser/trace-to-dataset";
import { describe, expect, it } from "vitest";
import {
  parseSchema9LedgerCliArguments,
} from "./schema9-ledger-cli.js";

const SPLITS = ["train", "validation-a", "validation-b", "test"] as const;

function completeArguments(): string[] {
  const values = [
    "--operation",
    "create",
    "--ledger",
    "receipts/schema9-ledger.json",
    "--guesser-repository",
    "guesser",
    "--engine-repository",
    "engine",
    "--guesser-commit",
    "a".repeat(40),
    "--converter-engine-commit",
    "b".repeat(40),
    "--producer-converter-policy",
    "converter-ancestor/v1",
  ];
  for (const split of SPLITS) {
    const [labelRoot, gameplayRoot, parametersRoot] =
      SCHEMA9_SPLIT_SEED_ROOTS[split];
    values.push(
      `--${split}-trace`,
      `private/${split}.trace.ndjson`,
      `--${split}-converted`,
      `private/${split}.schema9.ndjson`,
      `--${split}-launch-receipt`,
      `private/${split}.launch.json`,
      `--${split}-completion-receipt`,
      `private/${split}.completion.json`,
      `--${split}-schedule-id`,
      `schema9-${split}`,
      `--${split}-label-seed-root`,
      String(labelRoot),
      `--${split}-gameplay-seed-root`,
      String(gameplayRoot),
      `--${split}-parameters-seed-root`,
      String(parametersRoot),
      `--${split}-producer-engine-commit`,
      "c".repeat(40),
    );
  }
  return values;
}

describe("schema-9 corpus ledger CLI arguments", () => {
  it("requires every explicit file and preserves the four split names", () => {
    const invocationDirectory = resolve("fixture-invocation");
    const parsed = parseSchema9LedgerCliArguments(
      completeArguments(),
      invocationDirectory,
    );

    expect(parsed.operation).toBe("create");
    expect(parsed.corpus.producerConverterPolicy)
      .toBe("converter-ancestor/v1");
    expect(Object.keys(parsed.corpus.splits)).toEqual(SPLITS);
    expect(parsed.corpus.splits["validation-b"]).toMatchObject({
      tracePath: resolve(
        invocationDirectory,
        "private/validation-b.trace.ndjson",
      ),
      scheduleId: "schema9-validation-b",
      seedRoots: SCHEMA9_SPLIT_SEED_ROOTS["validation-b"],
      producerEngineCommit: "c".repeat(40),
    });
  });

  it("rejects missing, duplicate, unsupported, and weak policy flags", () => {
    const complete = completeArguments();
    expect(() =>
      parseSchema9LedgerCliArguments(complete.slice(0, -2))
    ).toThrow("Missing schema-9 ledger flags");
    expect(() =>
      parseSchema9LedgerCliArguments([
        ...complete,
        "--operation",
        "verify",
      ])
    ).toThrow("--operation may appear only once");
    expect(() =>
      parseSchema9LedgerCliArguments([
        ...complete.slice(0, -2),
        "--unknown",
        "value",
      ])
    ).toThrow("Unsupported or incomplete");
    const policyIndex = complete.indexOf("--producer-converter-policy") + 1;
    const weakPolicy = [...complete];
    weakPolicy[policyIndex] = "allow-any";
    expect(() => parseSchema9LedgerCliArguments(weakPolicy))
      .toThrow("must be exact/v1 or converter-ancestor/v1");
  });
});
