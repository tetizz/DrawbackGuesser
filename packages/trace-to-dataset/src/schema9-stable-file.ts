import { createHash, randomUUID } from "node:crypto";
import { constants, type BigIntStats } from "node:fs";
import {
  lstat,
  open,
  realpath,
  rename,
  rm,
  stat,
  type FileHandle,
} from "node:fs/promises";
import {
  isAbsolute,
  basename,
  dirname,
  join,
  parse,
  relative,
  resolve,
  sep,
} from "node:path";
import { throwIfSchema9Aborted } from "./schema9-ledger-types.js";

export interface Schema9StableFileRequest {
  readonly path: string;
  readonly label: string;
}

export interface Schema9StableFileIdentity {
  readonly path: string;
  readonly resolvedPath: string;
  readonly dev: bigint;
  readonly ino: bigint;
  readonly size: bigint;
  readonly birthtimeNs: bigint;
  readonly mtimeNs: bigint;
  readonly ctimeNs: bigint;
}

export interface Schema9OwnedStableFile extends Schema9StableFileIdentity {
  readonly handle: FileHandle;
}

export interface Schema9OwnedStableFileSet {
  readonly files: readonly Schema9OwnedStableFile[];
  readonly byPath: ReadonlyMap<string, Schema9OwnedStableFile>;
}

export interface ReadSchema9StableFile {
  readonly bytes: Buffer;
  readonly identity: Schema9StableFileIdentity;
}

function signature(value: BigIntStats): readonly bigint[] {
  return Object.freeze([
    value.dev,
    value.ino,
    value.size,
    value.birthtimeNs,
    value.mtimeNs,
    value.ctimeNs,
  ]);
}

function sameObject(left: BigIntStats, right: BigIntStats): boolean {
  return left.dev === right.dev && left.ino === right.ino;
}

function fileIdentity(
  path: string,
  resolvedPath: string,
  metadata: BigIntStats,
): Schema9StableFileIdentity {
  return Object.freeze({
    path,
    resolvedPath,
    dev: metadata.dev,
    ino: metadata.ino,
    size: metadata.size,
    birthtimeNs: metadata.birthtimeNs,
    mtimeNs: metadata.mtimeNs,
    ctimeNs: metadata.ctimeNs,
  });
}

async function assertNoLinkComponents(path: string, label: string): Promise<void> {
  const absolute = resolve(path);
  const root = parse(absolute).root;
  const child = relative(root, absolute);
  if (child === "" || child === ".." || child.startsWith(`..${sep}`)) {
    throw new TypeError(`${label} path is invalid.`);
  }
  let current = root;
  for (const part of child.split(sep)) {
    current = join(current, part);
    const metadata = await lstat(current, { bigint: true });
    if (metadata.isSymbolicLink()) {
      throw new TypeError(`${label} path may not contain symbolic links or junctions.`);
    }
  }
}

export async function openSchema9OwnedStableFile(
  request: Schema9StableFileRequest,
  signal?: AbortSignal,
): Promise<Schema9OwnedStableFile> {
  const { path, label } = request;
  throwIfSchema9Aborted(signal, label);
  if (path.length === 0 || !isAbsolute(path)) {
    throw new TypeError(`${label} path must be absolute and non-empty.`);
  }
  await assertNoLinkComponents(path, label);
  throwIfSchema9Aborted(signal, label);
  const pathBefore = await lstat(path, { bigint: true });
  if (pathBefore.isSymbolicLink() || !pathBefore.isFile()) {
    throw new TypeError(`${label} must be a regular non-link file.`);
  }
  const noFollow = typeof constants.O_NOFOLLOW === "number"
    ? constants.O_NOFOLLOW
    : 0;
  const handle = await open(path, constants.O_RDONLY | noFollow);
  try {
    throwIfSchema9Aborted(signal, label);
    await assertNoLinkComponents(path, label);
    const opened = await handle.stat({ bigint: true });
    if (!opened.isFile() || !sameObject(pathBefore, opened)) {
      throw new TypeError(`${label} changed while it was opened.`);
    }
    const resolvedPath = await realpath(path);
    const resolved = await stat(resolvedPath, { bigint: true });
    const pathAfter = await lstat(path, { bigint: true });
    if (
      pathAfter.isSymbolicLink()
      || !pathAfter.isFile()
      || !sameObject(opened, resolved)
      || !sameObject(opened, pathAfter)
    ) {
      throw new TypeError(`${label} changed while its identity was pinned.`);
    }
    await assertNoLinkComponents(path, label);
    throwIfSchema9Aborted(signal, label);
    return Object.freeze({
      ...fileIdentity(path, resolvedPath, opened),
      handle,
    });
  } catch (error: unknown) {
    try {
      await handle.close();
    } catch (closeError: unknown) {
      throw new AggregateError(
        [error, closeError],
        `${label} authentication and handle cleanup both failed.`,
      );
    }
    throw error;
  }
}

export async function assertSchema9OwnedStableFileUnchanged(
  file: Schema9OwnedStableFile,
  label: string,
  signal?: AbortSignal,
): Promise<void> {
  throwIfSchema9Aborted(signal, label);
  await assertNoLinkComponents(file.path, label);
  const opened = await file.handle.stat({ bigint: true });
  const pathInfo = await lstat(file.path, { bigint: true });
  const resolvedPath = await realpath(file.path);
  const resolvedInfo = await stat(resolvedPath, { bigint: true });
  const expected = Object.freeze([
    file.dev,
    file.ino,
    file.size,
    file.birthtimeNs,
    file.mtimeNs,
    file.ctimeNs,
  ]);
  if (
    pathInfo.isSymbolicLink()
    || !pathInfo.isFile()
    || resolvedPath !== file.resolvedPath
    || !sameObject(opened, pathInfo)
    || !sameObject(opened, resolvedInfo)
    || signature(opened).some((value, index) => value !== expected[index])
  ) {
    throw new Error(`${label} changed while it was authenticated.`);
  }
  await assertNoLinkComponents(file.path, label);
  throwIfSchema9Aborted(signal, label);
}

export async function* readSchema9OwnedStableFileChunks(
  file: Schema9OwnedStableFile,
  chunkBytes = 1024 * 1024,
  signal?: AbortSignal,
): AsyncGenerator<Buffer> {
  if (!Number.isSafeInteger(chunkBytes) || chunkBytes <= 0) {
    throw new RangeError("Schema-9 stable chunk size is invalid.");
  }
  let position = 0;
  while (position < file.size) {
    throwIfSchema9Aborted(signal, "Schema-9 stable input stream");
    const length = Number(
      BigInt(chunkBytes) < file.size - BigInt(position)
        ? BigInt(chunkBytes)
        : file.size - BigInt(position),
    );
    const chunk = Buffer.allocUnsafe(length);
    const result = await file.handle.read({
      buffer: chunk,
      offset: 0,
      length,
      position,
    });
    if (result.bytesRead === 0) {
      throw new Error("Schema-9 stable input ended before its authenticated size.");
    }
    position += result.bytesRead;
    yield result.bytesRead === length
      ? chunk
      : Buffer.from(chunk.subarray(0, result.bytesRead));
  }
}

export async function readSchema9OwnedStableFileBytes(
  file: Schema9OwnedStableFile,
  maximumBytes: number,
  label: string,
  signal?: AbortSignal,
): Promise<Buffer> {
  if (!Number.isSafeInteger(maximumBytes) || maximumBytes <= 0) {
    throw new RangeError("Schema-9 stable-file byte limit is invalid.");
  }
  if (file.size <= 0n || file.size > BigInt(maximumBytes)) {
    throw new RangeError(
      `${label} must be non-empty and at most ${String(maximumBytes)} bytes.`,
    );
  }
  const expectedBytes = Number(file.size);
  const bytes = Buffer.allocUnsafe(expectedBytes);
  let offset = 0;
  while (offset < expectedBytes) {
    throwIfSchema9Aborted(signal, label);
    const read = await file.handle.read({
      buffer: bytes,
      offset,
      length: expectedBytes - offset,
      position: offset,
    });
    if (read.bytesRead === 0) {
      break;
    }
    offset += read.bytesRead;
  }
  const overflow = Buffer.allocUnsafe(1);
  const extra = await file.handle.read({
    buffer: overflow,
    offset: 0,
    length: 1,
    position: expectedBytes,
  });
  if (offset !== expectedBytes || extra.bytesRead !== 0) {
    throw new Error(`${label} changed while it was read.`);
  }
  await assertSchema9OwnedStableFileUnchanged(file, label, signal);
  return bytes;
}

async function closeFiles(
  files: readonly Schema9OwnedStableFile[],
): Promise<readonly unknown[]> {
  const settled = await Promise.allSettled(
    [...files].reverse().map(async (file) => file.handle.close()),
  );
  const failures: unknown[] = [];
  for (const result of settled) {
    if (result.status === "rejected") {
      failures.push(result.reason as unknown);
    }
  }
  return failures;
}

export async function withSchema9OwnedStableFiles<T>(
  requests: readonly Schema9StableFileRequest[],
  operation: (files: Schema9OwnedStableFileSet) => Promise<T>,
  signal?: AbortSignal,
): Promise<T> {
  const files: Schema9OwnedStableFile[] = [];
  let result: T | undefined;
  let primaryFailure: unknown;
  try {
    const objectKeys = new Set<string>();
    const byPath = new Map<string, Schema9OwnedStableFile>();
    for (const request of requests) {
      const file = await openSchema9OwnedStableFile(request, signal);
      files.push(file);
      const objectKey = `${file.dev.toString()}:${file.ino.toString()}`;
      if (objectKeys.has(objectKey) || byPath.has(request.path)) {
        throw new TypeError(
          "Every Schema-9 input must be an explicit, distinct file object.",
        );
      }
      objectKeys.add(objectKey);
      byPath.set(request.path, file);
    }
    result = await operation(Object.freeze({
      files: Object.freeze([...files]),
      byPath,
    }));
    for (const [index, file] of files.entries()) {
      await assertSchema9OwnedStableFileUnchanged(
        file,
        requests[index]?.label ?? `input[${String(index)}]`,
        signal,
      );
    }
  } catch (error: unknown) {
    primaryFailure = error;
  }
  const closeFailures = await closeFiles(files);
  if (primaryFailure !== undefined) {
    if (closeFailures.length > 0) {
      throw new AggregateError(
        [primaryFailure, ...closeFailures],
        "Schema-9 input authentication and handle cleanup both failed.",
      );
    }
    throw primaryFailure instanceof Error
      ? primaryFailure
      : new Error("Schema-9 input authentication failed.", {
        cause: primaryFailure,
      });
  }
  if (closeFailures.length > 0) {
    throw new AggregateError(
      closeFailures,
      "Schema-9 stable input handle cleanup failed.",
    );
  }
  return result as T;
}

export async function readSchema9StableFileBytes(
  path: string,
  maximumBytes: number,
  label: string,
  signal?: AbortSignal,
): Promise<ReadSchema9StableFile> {
  return withSchema9OwnedStableFiles(
    [Object.freeze({ path, label })],
    async ({ files }) => {
      const file = files[0];
      if (file === undefined) {
        throw new Error("Schema-9 stable file was not opened.");
      }
      const bytes = await readSchema9OwnedStableFileBytes(
        file,
        maximumBytes,
        label,
        signal,
      );
      return Object.freeze({
        bytes,
        identity: fileIdentity(file.path, file.resolvedPath, await file.handle.stat({
          bigint: true,
        })),
      });
    },
    signal,
  );
}

/**
 * Remove only the exact stable file object previously authenticated.
 *
 * Node does not expose an fd-relative unlink on every supported platform. A
 * same-user process that can mutate the quarantine directory can therefore
 * race the final identity check and unlink. Schema-9 output directories must
 * not be shared with concurrently untrusted code running as the same OS user.
 */
export async function removeSchema9StableFileIfOwned(
  expected: Schema9StableFileIdentity,
  expectedSha256: string,
): Promise<void> {
  if (!/^[0-9a-f]{64}$/u.test(expectedSha256) || expected.size <= 0n) {
    throw new TypeError("Schema-9 rollback digest or size is invalid.");
  }
  const before = await lstat(expected.path, { bigint: true });
  const matches = before.isFile()
    && !before.isSymbolicLink()
    && before.dev === expected.dev
    && before.ino === expected.ino
    && before.size === expected.size
    && before.birthtimeNs === expected.birthtimeNs
    && before.mtimeNs === expected.mtimeNs
    && before.ctimeNs === expected.ctimeNs;
  if (!matches) {
    throw new Error("Schema-9 rollback target is no longer the published file.");
  }
  const quarantine = join(
    dirname(expected.path),
    `.${basename(expected.path)}.rollback-${randomUUID()}`,
  );
  await rename(expected.path, quarantine);
  const authenticated = await readSchema9StableFileBytes(
    quarantine,
    Number(expected.size),
    "Schema-9 rollback quarantine",
  );
  const after = await lstat(quarantine, { bigint: true });
  if (
    !after.isFile()
    || after.isSymbolicLink()
    || after.dev !== expected.dev
    || after.ino !== expected.ino
    || after.size !== expected.size
    || after.birthtimeNs !== expected.birthtimeNs
    || after.mtimeNs !== expected.mtimeNs
    || authenticated.identity.dev !== expected.dev
    || authenticated.identity.ino !== expected.ino
    || authenticated.identity.birthtimeNs !== expected.birthtimeNs
    || after.size !== authenticated.identity.size
    || after.mtimeNs !== authenticated.identity.mtimeNs
    || after.ctimeNs !== authenticated.identity.ctimeNs
    || createHash("sha256").update(authenticated.bytes).digest("hex")
      !== expectedSha256
  ) {
    throw new Error("Schema-9 rollback quarantine identity is inconsistent.");
  }
  await rm(quarantine);
}
