import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  writeTrainingDatasetNdjsonFileAtomic,
  type DatasetOutputPolicy,
} from "@drawbackguesser/trace-to-dataset";
import { readPrivateTraceNdjson } from "./trace-input.js";

export interface DatasetCliOptions {
  readonly inputPath: string;
  readonly outputPath: string;
  readonly expectedEvaluatorCoverage?: "none" | "uniform";
}

export interface DatasetCliIo {
  readonly stdout: { write(chunk: string): unknown };
  readonly stderr: { write(chunk: string): unknown };
}

function valueAfter(arguments_: readonly string[], flag: string): string {
  const index = arguments_.indexOf(flag);
  const value = arguments_[index + 1];
  if (index < 0 || value === undefined || value.startsWith("--")) {
    throw new TypeError(`${flag} requires a value.`);
  }
  return value;
}

export function parseDatasetCliArguments(
  arguments_: readonly string[],
  invocationDirectory = process.cwd(),
): DatasetCliOptions {
  if (arguments_.length % 2 !== 0) {
    throw new TypeError("Every dataset argument requires one value.");
  }
  const supported = new Set([
    "--input",
    "--output",
    "--require-evaluator",
  ]);
  const seen = new Set<string>();
  for (let index = 0; index < arguments_.length; index += 2) {
    const flag = arguments_[index];
    if (flag === undefined || !supported.has(flag)) {
      throw new TypeError(`Unsupported dataset argument: ${flag ?? ""}.`);
    }
    if (seen.has(flag)) {
      throw new TypeError(`Dataset argument ${flag} may appear only once.`);
    }
    seen.add(flag);
    const value = arguments_[index + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new TypeError(`${flag} requires a value.`);
    }
  }
  const inputPath = resolve(
    invocationDirectory,
    valueAfter(arguments_, "--input"),
  );
  const outputPath = resolve(
    invocationDirectory,
    valueAfter(arguments_, "--output"),
  );
  if (inputPath === outputPath) {
    throw new TypeError("Input and output paths must be different.");
  }
  const evaluator = arguments_.includes("--require-evaluator")
    ? valueAfter(arguments_, "--require-evaluator")
    : undefined;
  if (
    evaluator !== undefined
    && evaluator !== "none"
    && evaluator !== "uniform"
  ) {
    throw new TypeError(
      "--require-evaluator must be either none or uniform.",
    );
  }
  return {
    inputPath,
    outputPath,
    ...(evaluator === undefined
      ? {}
      : { expectedEvaluatorCoverage: evaluator }),
  };
}

export async function runDatasetCli(
  options: DatasetCliOptions,
  io: DatasetCliIo = {
    stdout: process.stdout,
    stderr: process.stderr,
  },
): Promise<void> {
  await mkdir(dirname(options.outputPath), { recursive: true });
  const policy: DatasetOutputPolicy =
    options.expectedEvaluatorCoverage === undefined
      ? {}
      : {
          expectedEvaluatorCoverage: options.expectedEvaluatorCoverage,
        };
  const written = await writeTrainingDatasetNdjsonFileAtomic(
    options.outputPath,
    readPrivateTraceNdjson(options.inputPath),
    policy,
  );
  io.stdout.write(
    `Wrote ${String(written.rows)} private training rows from `
    + `${String(written.games)} games (${String(written.bytes)} bytes, `
    + `sha256 ${written.sha256}) to ${options.outputPath}\n`,
  );
}

async function main(): Promise<void> {
  try {
    const arguments_ = process.argv.slice(2).filter(
      (argument) => argument !== "--",
    );
    await runDatasetCli(parseDatasetCliArguments(arguments_));
  } catch (error: unknown) {
    process.stderr.write(
      `Dataset conversion failed: ${
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
