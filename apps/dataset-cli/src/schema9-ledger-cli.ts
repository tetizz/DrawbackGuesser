import { execFile } from "node:child_process";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  loadAndReauthenticateSchema9CorpusLedger,
  schema9CorpusLedgerFileSha256,
  SCHEMA9_CORPUS_LEDGER_FORMAT,
  SCHEMA9_LEDGER_SPLITS,
  SCHEMA9_PRODUCER_CONVERTER_POLICIES,
  writeSchema9CorpusLedgerAtomic,
  type Schema9CorpusLedgerOptions,
  type Schema9LedgerSplit,
  type Schema9ProducerConverterPolicy,
  type Schema9RepositoryVerifier,
  type Schema9SplitFiles,
} from "@drawbackguesser/trace-to-dataset";

export interface Schema9LedgerCliOptions {
  readonly operation: "create" | "verify";
  readonly ledgerPath: string;
  readonly guesserRepository: string;
  readonly engineRepository: string;
  readonly corpus: Omit<Schema9CorpusLedgerOptions, "repositoryVerifier">;
}

export interface Schema9LedgerCliIo {
  readonly stdout: { write(chunk: string): unknown };
}

const GLOBAL_FLAGS = Object.freeze([
  "--operation",
  "--ledger",
  "--guesser-repository",
  "--engine-repository",
  "--guesser-commit",
  "--converter-engine-commit",
  "--producer-converter-policy",
] as const);
const SPLIT_FIELDS = Object.freeze([
  "trace",
  "converted",
  "launch-receipt",
  "completion-receipt",
  "schedule-id",
  "label-seed-root",
  "gameplay-seed-root",
  "parameters-seed-root",
  "producer-engine-commit",
] as const);

function splitFlag(split: Schema9LedgerSplit, field: string): string {
  return `--${split}-${field}`;
}

function valueAfter(
  values: ReadonlyMap<string, string>,
  flag: string,
): string {
  const value = values.get(flag);
  if (value === undefined) {
    throw new TypeError(`${flag} is required.`);
  }
  return value;
}

function splitFiles(
  values: ReadonlyMap<string, string>,
  split: Schema9LedgerSplit,
  invocationDirectory: string,
): Schema9SplitFiles {
  const unsignedRoot = (field: string): number => {
    const raw = valueAfter(values, splitFlag(split, field));
    if (!/^(?:0|[1-9][0-9]{0,9})$/u.test(raw)) {
      throw new TypeError(
        `${splitFlag(split, field)} must be a canonical unsigned integer.`,
      );
    }
    const parsed = Number(raw);
    if (!Number.isSafeInteger(parsed) || parsed > 0xffff_ffff) {
      throw new RangeError(
        `${splitFlag(split, field)} must fit an unsigned 32-bit integer.`,
      );
    }
    return parsed;
  };
  return Object.freeze({
    tracePath: resolve(
      invocationDirectory,
      valueAfter(values, splitFlag(split, "trace")),
    ),
    convertedPath: resolve(
      invocationDirectory,
      valueAfter(values, splitFlag(split, "converted")),
    ),
    launchReceiptPath: resolve(
      invocationDirectory,
      valueAfter(values, splitFlag(split, "launch-receipt")),
    ),
    completionReceiptPath: resolve(
      invocationDirectory,
      valueAfter(values, splitFlag(split, "completion-receipt")),
    ),
    scheduleId: valueAfter(values, splitFlag(split, "schedule-id")),
    seedRoots: Object.freeze([
      unsignedRoot("label-seed-root"),
      unsignedRoot("gameplay-seed-root"),
      unsignedRoot("parameters-seed-root"),
    ] as const),
    producerEngineCommit: valueAfter(
      values,
      splitFlag(split, "producer-engine-commit"),
    ),
  });
}

export function parseSchema9LedgerCliArguments(
  arguments_: readonly string[],
  invocationDirectory = process.cwd(),
): Schema9LedgerCliOptions {
  if (arguments_.length % 2 !== 0) {
    throw new TypeError("Every schema-9 ledger flag requires one value.");
  }
  const supported = new Set<string>(GLOBAL_FLAGS);
  for (const split of SCHEMA9_LEDGER_SPLITS) {
    for (const field of SPLIT_FIELDS) {
      supported.add(splitFlag(split, field));
    }
  }
  const values = new Map<string, string>();
  for (let index = 0; index < arguments_.length; index += 2) {
    const flag = arguments_[index];
    const value = arguments_[index + 1];
    if (
      flag === undefined
      || !supported.has(flag)
      || value === undefined
      || value.startsWith("--")
    ) {
      throw new TypeError(
        `Unsupported or incomplete schema-9 ledger flag: ${flag ?? ""}.`,
      );
    }
    if (values.has(flag)) {
      throw new TypeError(`${flag} may appear only once.`);
    }
    values.set(flag, value);
  }
  if (values.size !== supported.size) {
    const missing = [...supported].filter((flag) => !values.has(flag));
    throw new TypeError(
      `Missing schema-9 ledger flags: ${missing.join(", ")}.`,
    );
  }
  const operation = valueAfter(values, "--operation");
  if (operation !== "create" && operation !== "verify") {
    throw new TypeError("--operation must be create or verify.");
  }
  const rawPolicy = valueAfter(values, "--producer-converter-policy");
  if (
    !SCHEMA9_PRODUCER_CONVERTER_POLICIES.includes(
      rawPolicy as Schema9ProducerConverterPolicy,
    )
  ) {
    throw new TypeError(
      "--producer-converter-policy must be exact/v1 "
      + "or converter-ancestor/v1.",
    );
  }
  const producerConverterPolicy =
    rawPolicy as Schema9ProducerConverterPolicy;
  const splits = Object.fromEntries(
    SCHEMA9_LEDGER_SPLITS.map((split) => [
      split,
      splitFiles(values, split, invocationDirectory),
    ]),
  ) as Readonly<Record<Schema9LedgerSplit, Schema9SplitFiles>>;
  return Object.freeze({
    operation,
    ledgerPath: resolve(
      invocationDirectory,
      valueAfter(values, "--ledger"),
    ),
    guesserRepository: resolve(
      invocationDirectory,
      valueAfter(values, "--guesser-repository"),
    ),
    engineRepository: resolve(
      invocationDirectory,
      valueAfter(values, "--engine-repository"),
    ),
    corpus: Object.freeze({
      guesserCommit: valueAfter(values, "--guesser-commit"),
      converterEngineCommit: valueAfter(
        values,
        "--converter-engine-commit",
      ),
      producerConverterPolicy,
      splits,
    }),
  });
}

interface GitResult {
  readonly stdout: string;
}

function git(
  repository: string,
  arguments_: readonly string[],
): Promise<GitResult> {
  return new Promise((accept, reject) => {
    execFile(
      "git",
      ["-C", repository, ...arguments_],
      {
        encoding: "utf8",
        windowsHide: true,
        maxBuffer: 1024 * 1024,
      },
      (error, stdout) => {
        if (error !== null) {
          reject(
            error instanceof Error
              ? error
              : new Error("Git process failed.", { cause: error }),
          );
          return;
        }
        accept({ stdout });
      },
    );
  });
}

async function requireCommit(
  repository: string,
  commit: string,
): Promise<void> {
  await git(repository, ["cat-file", "-e", `${commit}^{commit}`]);
}

function exitCode(error: unknown): number | undefined {
  if (
    typeof error === "object"
    && error !== null
    && "code" in error
    && typeof error.code === "number"
  ) {
    return error.code;
  }
  return undefined;
}

export function createGitRepositoryVerifier(
  guesserRepository: string,
  engineRepository: string,
): Schema9RepositoryVerifier {
  return Object.freeze({
    async pinnedEngineCommitAt(guesserCommit: string): Promise<string> {
      await requireCommit(guesserRepository, guesserCommit);
      const result = await git(guesserRepository, [
        "ls-tree",
        "--full-tree",
        guesserCommit,
        "--",
        "engine",
      ]);
      const match =
        /^160000 commit ([0-9a-f]{40}|[0-9a-f]{64})\tengine\n?$/u.exec(
          result.stdout,
        );
      if (match?.[1] === undefined) {
        throw new TypeError(
          "Guesser commit does not contain one pinned Engine gitlink.",
        );
      }
      await requireCommit(engineRepository, match[1]);
      return match[1];
    },
    async isEngineAncestor(
      ancestorCommit: string,
      descendantCommit: string,
    ): Promise<boolean> {
      await requireCommit(engineRepository, ancestorCommit);
      await requireCommit(engineRepository, descendantCommit);
      try {
        await git(engineRepository, [
          "merge-base",
          "--is-ancestor",
          ancestorCommit,
          descendantCommit,
        ]);
        return true;
      } catch (error: unknown) {
        if (exitCode(error) === 1) {
          return false;
        }
        throw error;
      }
    },
  });
}

export async function runSchema9LedgerCli(
  options: Schema9LedgerCliOptions,
  io: Schema9LedgerCliIo = { stdout: process.stdout },
): Promise<void> {
  const repositoryVerifier = createGitRepositoryVerifier(
    options.guesserRepository,
    options.engineRepository,
  );
  const corpus: Schema9CorpusLedgerOptions = Object.freeze({
    ...options.corpus,
    repositoryVerifier,
  });
  if (options.operation === "create") {
    const written = await writeSchema9CorpusLedgerAtomic(
      options.ledgerPath,
      corpus,
    );
    io.stdout.write(
      `${JSON.stringify({
        format: SCHEMA9_CORPUS_LEDGER_FORMAT,
        version: 1,
        bytes: written.bytes,
        sha256: written.sha256,
      })}\n`,
    );
    return;
  }
  await loadAndReauthenticateSchema9CorpusLedger(
    options.ledgerPath,
    corpus,
  );
  const sha256 = await schema9CorpusLedgerFileSha256(options.ledgerPath);
  io.stdout.write(
    `${JSON.stringify({
      format: SCHEMA9_CORPUS_LEDGER_FORMAT,
      version: 1,
      verified: true,
      sha256,
    })}\n`,
  );
}

async function main(): Promise<void> {
  try {
    const arguments_ = process.argv.slice(2).filter(
      (argument) => argument !== "--",
    );
    await runSchema9LedgerCli(
      parseSchema9LedgerCliArguments(arguments_),
    );
  } catch (error: unknown) {
    process.stderr.write(
      `Schema-9 corpus ledger failed: ${
        error instanceof Error ? error.message : String(error)
      }\n`,
    );
    process.exitCode = 1;
  }
}

const invokedPath = process.argv[1];
if (
  invokedPath !== undefined
  && import.meta.url === pathToFileURL(invokedPath).href
) {
  void main();
}
