import { createReadStream } from "node:fs";
import { TextDecoder } from "node:util";
import {
  parseTrustedSimulationTraceLine,
  type TrustedSimulationTraceRecord,
} from "@drawbackguesser/trace-to-dataset";

export const DEFAULT_MAX_TRACE_LINE_BYTES = 64 * 1024 * 1024;

export interface TraceInputOptions {
  readonly maxLineBytes?: number;
}

const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });

function checkedMaximum(value: number | undefined): number {
  const maximum = value ?? DEFAULT_MAX_TRACE_LINE_BYTES;
  if (!Number.isSafeInteger(maximum) || maximum <= 0) {
    throw new RangeError("Maximum trace line bytes must be a positive integer.");
  }
  return maximum;
}

function parseLine(
  bytes: Buffer,
  lineNumber: number,
  maximum: number,
): TrustedSimulationTraceRecord {
  if (bytes.byteLength > maximum) {
    throw new RangeError(
      `Private trace line ${String(lineNumber)} exceeds ${String(maximum)} bytes.`,
    );
  }
  let line: string;
  try {
    line = UTF8_DECODER.decode(bytes).replace(/\r$/u, "");
  } catch (error: unknown) {
    throw new SyntaxError(
      `Private trace line ${String(lineNumber)} is not valid UTF-8.`,
      { cause: error },
    );
  }
  if (line.trim().length === 0) {
    throw new SyntaxError(
      `Private trace line ${String(lineNumber)} must not be blank.`,
    );
  }
  try {
    return parseTrustedSimulationTraceLine(line);
  } catch (error: unknown) {
    throw new SyntaxError(
      `Private trace line ${String(lineNumber)} is invalid: ${
        error instanceof Error ? error.message : String(error)
      }`,
      { cause: error },
    );
  }
}

export async function* readPrivateTraceNdjson(
  path: string,
  options: TraceInputOptions = {},
): AsyncIterableIterator<TrustedSimulationTraceRecord> {
  const maximum = checkedMaximum(options.maxLineBytes);
  const stream = createReadStream(path);
  let fragments: Buffer[] = [];
  let bufferedBytes = 0;
  let lineNumber = 0;

  for await (const rawChunk of stream) {
    const chunk = Buffer.isBuffer(rawChunk)
      ? rawChunk
      : Buffer.from(rawChunk as Uint8Array);
    let cursor = 0;
    let newline = chunk.indexOf(0x0a, cursor);
    while (newline >= 0) {
      const fragment = chunk.subarray(cursor, newline);
      bufferedBytes += fragment.byteLength;
      if (bufferedBytes > maximum) {
        throw new RangeError(
          `Private trace line ${String(lineNumber + 1)} exceeds ${String(maximum)} bytes.`,
        );
      }
      fragments.push(fragment);
      lineNumber += 1;
      yield parseLine(
        Buffer.concat(fragments, bufferedBytes),
        lineNumber,
        maximum,
      );
      fragments = [];
      bufferedBytes = 0;
      cursor = newline + 1;
      newline = chunk.indexOf(0x0a, cursor);
    }
    const tail = chunk.subarray(cursor);
    bufferedBytes += tail.byteLength;
    if (bufferedBytes > maximum) {
      throw new RangeError(
        `Private trace line ${String(lineNumber + 1)} exceeds ${String(maximum)} bytes.`,
      );
    }
    fragments.push(tail);
  }
  if (bufferedBytes > 0) {
    lineNumber += 1;
    yield parseLine(
      Buffer.concat(fragments, bufferedBytes),
      lineNumber,
      maximum,
    );
  }
}
