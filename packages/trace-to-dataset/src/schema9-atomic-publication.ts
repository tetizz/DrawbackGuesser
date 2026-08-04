import { createHash, randomUUID } from "node:crypto";
import type { BigIntStats } from "node:fs";
import {
  lstat,
  link,
  open,
  realpath,
  rename,
  rm,
  stat,
} from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { throwIfSchema9Aborted } from "./schema9-ledger-types.js";
import {
  readSchema9StableFileBytes,
  type Schema9StableFileIdentity,
} from "./schema9-stable-file.js";

export interface Schema9AtomicPublicationResult {
  readonly bytes: number;
  readonly sha256: string;
  readonly publicationIdentity: Schema9StableFileIdentity;
}

export class Schema9AtomicPublicationCleanupError extends AggregateError {
  public constructor(
    failures: readonly unknown[],
    public readonly committed: boolean,
    label: string,
  ) {
    super(
      [...failures],
      `${label} ${committed ? "committed but " : ""}cleanup failed.`,
    );
    this.name = "Schema9AtomicPublicationCleanupError";
  }
}

export class Schema9AtomicPublicationError extends Error {
  public constructor(
    message: string,
    public readonly committed: boolean,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "Schema9AtomicPublicationError";
  }
}

export interface Schema9TemporaryPublicationIdentity {
  readonly dev: bigint;
  readonly ino: bigint;
  readonly birthtimeNs: bigint;
}

export interface Schema9AtomicPublicationHooks {
  readonly afterLink?: (destination: string) => Promise<void>;
  readonly afterDirectorySync?: (directory: string) => Promise<void>;
  readonly afterQuarantine?: (quarantine: string) => Promise<void>;
}

function isWindowsReadOnlyDirectorySyncFailure(error: unknown): boolean {
  if (process.platform !== "win32" || !(error instanceof Error)) {
    return false;
  }
  const code = (error as NodeJS.ErrnoException).code;
  return code === "EACCES"
    || code === "EBADF"
    || code === "EINVAL"
    || code === "EISDIR"
    || code === "EPERM";
}

async function syncDirectoryWithFlags(
  directory: string,
  flags: "r" | "r+",
  label: string,
): Promise<void> {
  let handle: Awaited<ReturnType<typeof open>> | undefined;
  let primaryFailure: unknown;
  try {
    handle = await open(directory, flags);
    await handle.sync();
  } catch (error: unknown) {
    primaryFailure = error;
  }

  let closeFailure: unknown;
  if (handle !== undefined) {
    try {
      await handle.close();
    } catch (error: unknown) {
      closeFailure = error;
    }
  }
  if (primaryFailure !== undefined && closeFailure !== undefined) {
    throw new AggregateError(
      [primaryFailure, closeFailure],
      `${label} parent directory sync and handle cleanup failed.`,
    );
  }
  if (primaryFailure !== undefined) {
    throw primaryFailure instanceof Error
      ? primaryFailure
      : new Error(`${label} parent directory sync failed.`, {
        cause: primaryFailure,
      });
  }
  if (closeFailure !== undefined) {
    throw closeFailure instanceof Error
      ? closeFailure
      : new Error(`${label} parent directory handle cleanup failed.`, {
        cause: closeFailure,
      });
  }
}

/**
 * Make a newly-created destination name durable before publication succeeds.
 *
 * POSIX accepts fsync on a read-only directory handle. Node on Windows can
 * open that handle but reports EPERM from fsync, so Windows retries the same
 * parent directory with a read/write handle. Failure of both strategies is
 * fatal; there is no silent best-effort success path.
 */
export async function syncSchema9PublicationParentDirectory(
  destination: string,
  label: string,
): Promise<void> {
  const directory = dirname(destination);
  try {
    await syncDirectoryWithFlags(directory, "r", label);
    return;
  } catch (error: unknown) {
    if (!isWindowsReadOnlyDirectorySyncFailure(error)) {
      throw error;
    }
    try {
      await syncDirectoryWithFlags(directory, "r+", label);
    } catch (fallbackError: unknown) {
      throw new AggregateError(
        [error, fallbackError],
        `${label} parent directory could not be synced.`,
      );
    }
  }
}

function temporaryIdentity(
  metadata: BigIntStats,
): Schema9TemporaryPublicationIdentity {
  return Object.freeze({
    dev: metadata.dev,
    ino: metadata.ino,
    birthtimeNs: metadata.birthtimeNs,
  });
}

function sameTemporaryIdentity(
  metadata: BigIntStats,
  expected: Schema9TemporaryPublicationIdentity,
): boolean {
  return metadata.isFile()
    && !metadata.isSymbolicLink()
    && metadata.dev === expected.dev
    && metadata.ino === expected.ino
    && metadata.birthtimeNs === expected.birthtimeNs;
}

export async function schema9PublicationDestination(
  path: string,
  label: string,
): Promise<string> {
  if (path.length === 0) {
    throw new TypeError(`${label} output path must not be empty.`);
  }
  const parent = await realpath(dirname(path));
  const parentInfo = await stat(parent);
  if (!parentInfo.isDirectory()) {
    throw new TypeError(`${label} output parent must be a directory.`);
  }
  return join(parent, basename(path));
}

/** Publish exact bytes without overwrite and surface every owned-temp failure. */
export async function publishSchema9BytesAtomicNoClobber(
  outputPath: string,
  payload: Buffer,
  maximumBytes: number,
  label: string,
  signal?: AbortSignal,
  hooks: Schema9AtomicPublicationHooks = {},
): Promise<Schema9AtomicPublicationResult> {
  throwIfSchema9Aborted(signal, `${label} publication`);
  if (
    payload.byteLength <= 0
    || payload.byteLength > maximumBytes
  ) {
    throw new RangeError(`${label} byte length is invalid.`);
  }
  const destination = await schema9PublicationDestination(outputPath, label);
  const temporary = join(
    dirname(destination),
    `.${basename(destination)}.tmp-${String(process.pid)}-${randomUUID()}`,
  );
  let handle: Awaited<ReturnType<typeof open>> | undefined;
  let ownedTemporary: Schema9TemporaryPublicationIdentity | undefined;
  let temporaryCreated = false;
  let committed = false;
  let primaryFailure: unknown;
  try {
    handle = await open(temporary, "wx", 0o600);
    temporaryCreated = true;
    ownedTemporary = temporaryIdentity(await handle.stat({ bigint: true }));
    throwIfSchema9Aborted(signal, `${label} publication`);
    await handle.writeFile(payload);
    await handle.sync();
    throwIfSchema9Aborted(signal, `${label} publication`);
    await handle.close();
    handle = undefined;
    await link(temporary, destination);
    committed = true;
    await hooks.afterLink?.(destination);
    const published = await readSchema9StableFileBytes(
      destination,
      maximumBytes,
      label,
      signal,
    );
    if (!published.bytes.equals(payload)) {
      throw new Error(`${label} published bytes changed.`);
    }
  } catch (error: unknown) {
    primaryFailure = error;
  }

  const cleanupFailures: unknown[] = [];
  if (handle !== undefined) {
    try {
      await handle.close();
    } catch (error: unknown) {
      cleanupFailures.push(error);
    }
  }
  if (temporaryCreated) {
    try {
      if (ownedTemporary === undefined) {
        throw new Error(`${label} temporary publication identity is unavailable.`);
      }
      await cleanupSchema9TemporaryPublication(
        temporary,
        ownedTemporary,
        label,
        hooks.afterQuarantine,
      );
    } catch (error: unknown) {
      cleanupFailures.push(error);
    }
  }
  if (cleanupFailures.length > 0) {
    throw new Schema9AtomicPublicationCleanupError(
      primaryFailure === undefined
        ? cleanupFailures
        : [primaryFailure, ...cleanupFailures],
      committed,
      label,
    );
  }
  if (primaryFailure !== undefined) {
    const failure = primaryFailure instanceof Error
      ? primaryFailure
      : new Error(`${label} publication failed.`, { cause: primaryFailure });
    if (committed) {
      throw new Schema9AtomicPublicationError(
        `${label} committed but post-link verification failed.`,
        true,
        { cause: failure },
      );
    }
    throw failure;
  }

  let published: Awaited<ReturnType<typeof readSchema9StableFileBytes>>;
  try {
    published = await readSchema9StableFileBytes(
      destination,
      maximumBytes,
      label,
      signal,
    );
    if (!published.bytes.equals(payload)) {
      throw new Error(`${label} changed after temporary cleanup.`);
    }
    await syncSchema9PublicationParentDirectory(destination, label);
    await hooks.afterDirectorySync?.(dirname(destination));
  } catch (error: unknown) {
    throw new Schema9AtomicPublicationError(
      `${label} committed but final durability verification failed.`,
      true,
      { cause: error },
    );
  }
  return Object.freeze({
    bytes: payload.byteLength,
    sha256: createHash("sha256").update(payload).digest("hex"),
    publicationIdentity: published.identity,
  });
}

/**
 * Quarantine and remove the exact temporary object observed by the caller.
 *
 * The final lstat-to-rm interval is not fd-bound because Node has no portable
 * unlink-at API. Callers must not share the parent directory with concurrently
 * untrusted code running as the same OS user; the architecture document records
 * this local-host trust boundary.
 */
export async function cleanupSchema9TemporaryPublication(
  temporary: string,
  expected: Schema9TemporaryPublicationIdentity,
  label: string,
  afterQuarantine?: (quarantine: string) => Promise<void>,
): Promise<void> {
  const before = await lstat(temporary, { bigint: true });
  if (!sameTemporaryIdentity(before, expected)) {
    throw new Error(`${label} temporary publication object changed.`);
  }
  const quarantine = join(
    dirname(temporary),
    `.${basename(temporary)}.cleanup-${randomUUID()}`,
  );
  await rename(temporary, quarantine);
  await afterQuarantine?.(quarantine);
  const finalInfo = await lstat(quarantine, { bigint: true });
  if (!sameTemporaryIdentity(finalInfo, expected)) {
    throw new Error(`${label} temporary quarantine changed.`);
  }
  await rm(quarantine);
}

export function schema9PublicationMayBeCommitted(error: unknown): boolean {
  if (
    error instanceof Schema9AtomicPublicationCleanupError
    || error instanceof Schema9AtomicPublicationError
  ) {
    return error.committed;
  }
  if (error instanceof AggregateError) {
    return error.errors.some(schema9PublicationMayBeCommitted);
  }
  if (error instanceof Error && error.cause !== undefined) {
    return schema9PublicationMayBeCommitted(error.cause);
  }
  return false;
}
