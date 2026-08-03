import { createHash, randomUUID } from "node:crypto";
import {
  link,
  open,
  readFile,
  rm,
} from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import {
  assertPathFreeJson,
  canonicalJsonBytes,
  checkedSha256,
  type Schema9CorpusLedger,
} from "./schema9-ledger-types.js";

export const SCHEMA9_LEDGER_VERIFICATION_RECEIPT_FORMAT =
  "drawbackguesser-schema9-ledger-verification" as const;
export const SCHEMA9_LEDGER_VERIFICATION_RECEIPT_VERSION = 1 as const;
const MAX_VERIFICATION_RECEIPT_BYTES = 1024 * 1024;

export interface Schema9LedgerVerificationReceipt {
  readonly format: typeof SCHEMA9_LEDGER_VERIFICATION_RECEIPT_FORMAT;
  readonly version: typeof SCHEMA9_LEDGER_VERIFICATION_RECEIPT_VERSION;
  readonly ledger: {
    readonly sha256: string;
    readonly contentSha256: string;
  };
  readonly repository: Schema9CorpusLedger["identity"];
  readonly inputSetSha256: string;
  readonly verificationPolicy: {
    readonly repository: "head-clean-content-manifest/v1";
    readonly schedule: "engine-scheduler-replay/v1";
    readonly corpus: "full-byte-reauthentication/v1";
  };
  readonly contentSha256: string;
}

function sha256(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function contentSha256(value: unknown): string {
  return sha256(canonicalJsonBytes(value));
}

function inputSet(ledger: Schema9CorpusLedger): unknown {
  return ledger.splits.map((split) => ({
    split: split.split,
    scheduleId: split.scheduleId,
    seedRoots: split.seedRoots,
    producerEngineCommit: split.producerEngineCommit,
    generatorReceipts: split.generatorReceipts,
    sourceTrace: {
      sha256: split.sourceTrace.sha256,
      bytes: split.sourceTrace.bytes,
      games: split.sourceTrace.games,
      gameIdSetSha256: split.sourceTrace.gameIdSetSha256,
      simulationSeedSetSha256: split.sourceTrace.simulationSeedSetSha256,
      parameterSeedSetSha256: split.sourceTrace.parameterSeedSetSha256,
    },
    converted: {
      sha256: split.converted.sha256,
      bytes: split.converted.bytes,
      rows: split.converted.rows,
      games: split.converted.games,
      gameIdSetSha256: split.converted.gameIdSetSha256,
      simulationSeedSetSha256: split.converted.simulationSeedSetSha256,
    },
  }));
}

export function createSchema9LedgerVerificationReceipt(
  ledger: Schema9CorpusLedger,
  ledgerSha256: string,
): Schema9LedgerVerificationReceipt {
  const payload = Object.freeze({
    format: SCHEMA9_LEDGER_VERIFICATION_RECEIPT_FORMAT,
    version: SCHEMA9_LEDGER_VERIFICATION_RECEIPT_VERSION,
    ledger: Object.freeze({
      sha256: checkedSha256(ledgerSha256, "verified ledger SHA-256"),
      contentSha256: checkedSha256(
        ledger.contentSha256,
        "verified ledger content SHA-256",
      ),
    }),
    repository: ledger.identity,
    inputSetSha256: contentSha256(inputSet(ledger)),
    verificationPolicy: Object.freeze({
      repository: "head-clean-content-manifest/v1" as const,
      schedule: "engine-scheduler-replay/v1" as const,
      corpus: "full-byte-reauthentication/v1" as const,
    }),
  });
  assertPathFreeJson(payload, "schema-9 ledger verification receipt");
  return Object.freeze({
    ...payload,
    contentSha256: contentSha256(payload),
  });
}

export function schema9LedgerVerificationReceiptSha256(
  receipt: Schema9LedgerVerificationReceipt,
): string {
  return sha256(canonicalJsonBytes(receipt));
}

export async function writeSchema9LedgerVerificationReceiptAtomic(
  outputPath: string,
  receipt: Schema9LedgerVerificationReceipt,
): Promise<{ readonly bytes: number; readonly sha256: string }> {
  const payload = canonicalJsonBytes(receipt);
  if (
    payload.byteLength <= 0
    || payload.byteLength > MAX_VERIFICATION_RECEIPT_BYTES
  ) {
    throw new RangeError("Schema-9 verification receipt byte length is invalid.");
  }
  const temporary = join(
    dirname(outputPath),
    `${basename(outputPath)}.tmp-${String(process.pid)}-${randomUUID()}`,
  );
  let handle: Awaited<ReturnType<typeof open>> | undefined;
  try {
    handle = await open(temporary, "wx", 0o600);
    await handle.writeFile(payload);
    await handle.sync();
    await handle.close();
    handle = undefined;
    await link(temporary, outputPath);
    if (!(await readFile(outputPath)).equals(payload)) {
      throw new Error("Published schema-9 verification receipt changed.");
    }
  } finally {
    await handle?.close().catch(() => undefined);
    await rm(temporary, { force: true }).catch(() => undefined);
  }
  return Object.freeze({
    bytes: payload.byteLength,
    sha256: sha256(payload),
  });
}
