import type { Writable } from "node:stream";

interface WritableLike {
  write(chunk: string): unknown;
}

/** Writes one result line while honoring stream backpressure and EPIPE. */
export async function writeSchema9JsonLine(
  stream: WritableLike,
  value: unknown,
  signal?: AbortSignal,
): Promise<void> {
  throwIfAborted(signal);
  const line = `${JSON.stringify(value)}\n`;
  if (!isNodeWritable(stream)) {
    const result = stream.write(line);
    if (isPromiseLike(result)) {
      await result;
    }
    return;
  }
  if (stream.destroyed || stream.writableEnded) {
    throw new Error("Schema-9 output stream is not writable.");
  }
  await new Promise<void>((accept, reject) => {
    let settled = false;
    const cleanup = (retainErrorListener: boolean): void => {
      if (!retainErrorListener) {
        stream.removeListener("error", onError);
      }
      signal?.removeEventListener("abort", onAbort);
    };
    const finish = (
      error?: Error | null,
      retainErrorListener = false,
    ): void => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup(retainErrorListener);
      if (error === undefined || error === null) {
        accept();
      } else {
        reject(error);
      }
    };
    const onError = (error: Error): void => {
      finish(error);
    };
    const onAbort = (): void => {
      finish(abortFailure(signal), true);
    };
    stream.once("error", onError);
    signal?.addEventListener("abort", onAbort, { once: true });
    try {
      stream.write(line, "utf8", (error?: Error | null) => {
        if (settled) {
          signal?.removeEventListener("abort", onAbort);
          if (error === undefined || error === null) {
            stream.removeListener("error", onError);
          }
          return;
        }
        finish(error, error !== undefined && error !== null);
      });
    } catch (error: unknown) {
      finish(error instanceof Error
        ? error
        : new Error("Schema-9 output failed.", { cause: error }));
    }
  });
}

function isNodeWritable(value: WritableLike): value is Writable {
  return "once" in value
    && typeof value.once === "function"
    && "removeListener" in value
    && typeof value.removeListener === "function";
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return typeof value === "object"
    && value !== null
    && "then" in value
    && typeof value.then === "function";
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw abortFailure(signal);
  }
}

function abortFailure(signal: AbortSignal | undefined): Error {
  return signal?.reason instanceof Error
    ? signal.reason
    : new Error("Schema-9 output was interrupted.");
}
