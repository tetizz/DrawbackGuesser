import { createHash } from "node:crypto";
import {
  mkdtemp,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  CAPTURABLE_HYPOTHESIS_RULE_IDS,
} from "@drawbackguesser/predictor";
import type {
  AuditedCapturableKingRuleId,
} from "@drawbackengine/drawback-engine";
import { afterEach, describe, expect, it } from "vitest";
import {
  convertParsedPlayerPrivateTraceToDatasetRows,
} from "./player-private-converter.js";
import { playerPrivateTraceFixture } from "./player-private-test-fixture.js";
import {
  authenticateSchema9SplitWithRuleContract,
} from "./schema9-ledger-authentication.js";
import {
  SCHEMA9_GENERATOR_COMPLETION_FORMAT,
  SCHEMA9_GENERATOR_LAUNCH_FORMAT,
  SCHEMA9_GENERATOR_RECEIPT_VERSION,
  SCHEMA9_SCHEDULE_PROFILE,
  SCHEMA9_SPLIT_SEED_ROOTS,
  type Schema9SplitFiles,
} from "./schema9-ledger-types.js";
import { schema9AssignmentScheduler } from "./schema9-schedule-replay.js";

const ENGINE_COMMIT = "b".repeat(40);
const cleanup: string[] = [];

afterEach(async () => {
  await Promise.all(cleanup.splice(0).map((path) =>
    rm(path, { recursive: true, force: true })
  ));
});

describe("schema-9 real parser/converter integration", () => {
  it("replays one complete Latin label cycle with real Engine contracts", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-real-replay-"));
    cleanup.push(root);
    const assignments = [...schema9AssignmentScheduler.assignments(
      "train",
      CAPTURABLE_HYPOTHESIS_RULE_IDS.length,
      SCHEMA9_SPLIT_SEED_ROOTS.train,
    )];
    const traces = assignments.map((assignment) =>
      playerPrivateTraceFixture({
        whiteRuleId: assignment.whiteRuleId as AuditedCapturableKingRuleId,
        blackRuleId: assignment.blackRuleId as AuditedCapturableKingRuleId,
        seed: assignment.seed,
        gameIndex: assignment.gameIndex,
        parameterSeeds: assignment.parameterSeeds,
        policyId: SCHEMA9_SCHEDULE_PROFILE.policyId,
        autoLegalPlies: 1,
      })
    );
    const tracePayload = Buffer.from(
      traces.map((trace) => `${JSON.stringify(trace)}\n`).join(""),
      "utf8",
    );
    const convertedPayload = Buffer.from(
      traces.flatMap(convertParsedPlayerPrivateTraceToDatasetRows)
        .map((row) => `${JSON.stringify(row)}\n`)
        .join(""),
      "utf8",
    );
    const tracePath = join(root, "source.ndjson");
    const convertedPath = join(root, "converted.ndjson");
    const launchReceiptPath = join(root, "launch.json");
    const completionReceiptPath = join(root, "completion.json");
    await writeFile(tracePath, tracePayload);
    await writeFile(convertedPath, convertedPayload);
    const scheduleId = "schema9-real-replay";
    const launchPayload = Buffer.from(`${JSON.stringify({
      format: SCHEMA9_GENERATOR_LAUNCH_FORMAT,
      version: SCHEMA9_GENERATOR_RECEIPT_VERSION,
      scheduleAuthorityId: "capturable25-schema9-opportunity/v1",
      scheduleId,
      ledgerSplit: "train",
      engineSplit: "train",
      splitCounts: { train: traces.length, validation: 0, test: 0 },
      seedRoots: SCHEMA9_SPLIT_SEED_ROOTS.train,
      scheduleProfile: SCHEMA9_SCHEDULE_PROFILE,
      producerEngineCommit: ENGINE_COMMIT,
    })}\n`, "utf8");
    await writeFile(launchReceiptPath, launchPayload);
    await writeFile(completionReceiptPath, `${JSON.stringify({
      format: SCHEMA9_GENERATOR_COMPLETION_FORMAT,
      version: SCHEMA9_GENERATOR_RECEIPT_VERSION,
      scheduleId,
      ledgerSplit: "train",
      state: "completed",
      producerEngineCommit: ENGINE_COMMIT,
      launchReceiptSha256: createHash("sha256")
        .update(launchPayload)
        .digest("hex"),
      output: {
        sha256: createHash("sha256").update(tracePayload).digest("hex"),
        bytes: tracePayload.byteLength,
        games: traces.length,
        firstGameIndex: 0,
        lastGameIndex: traces.length - 1,
      },
    })}\n`, "utf8");
    const files: Schema9SplitFiles = Object.freeze({
      tracePath,
      convertedPath,
      launchReceiptPath,
      completionReceiptPath,
      scheduleId,
      seedRoots: SCHEMA9_SPLIT_SEED_ROOTS.train,
      producerEngineCommit: ENGINE_COMMIT,
    });
    const authenticated = await authenticateSchema9SplitWithRuleContract(
      "train",
      files,
      CAPTURABLE_HYPOTHESIS_RULE_IDS,
      schema9AssignmentScheduler,
    );
    expect(authenticated.ledger.sourceTrace).toMatchObject({
      games: 25,
      zeroPlyGames: 0,
    });
    expect(authenticated.ledger.converted.games).toBe(25);
  }, 60_000);
});
