import {
  lstat,
  mkdtemp,
  readFile,
  readdir,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  cleanupSchema9TemporaryPublication,
  publishSchema9BytesAtomicNoClobber,
  schema9PublicationMayBeCommitted,
} from "./schema9-atomic-publication.js";

describe("schema-9 atomic publication", () => {
  it("syncs its parent after the create-only entry and cleanup are final", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-directory-sync-"));
    const output = join(root, "artifact.json");
    const payload = Buffer.from("{\"artifact\":true}\n", "utf8");
    const syncedDirectories: string[] = [];
    try {
      const written = await publishSchema9BytesAtomicNoClobber(
        output,
        payload,
        1024,
        "Schema-9 fixture",
        undefined,
        {
          afterDirectorySync: async (directory) => {
            syncedDirectories.push(directory);
            expect(
              (await readdir(root)).filter((entry) => entry.includes(".tmp-")),
            ).toEqual([]);
          },
        },
      );

      expect(written.bytes).toBe(payload.byteLength);
      expect(syncedDirectories).toEqual([root]);
      await expect(readFile(output)).resolves.toEqual(payload);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("keeps an existing destination and never reaches directory sync", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-no-clobber-sync-"));
    const output = join(root, "artifact.json");
    const original = Buffer.from("original\n", "utf8");
    let directorySyncReached = false;
    try {
      await writeFile(output, original);
      await expect(publishSchema9BytesAtomicNoClobber(
        output,
        Buffer.from("replacement\n", "utf8"),
        1024,
        "Schema-9 fixture",
        undefined,
        {
          afterDirectorySync: () => {
            directorySyncReached = true;
            return Promise.resolve();
          },
        },
      )).rejects.toMatchObject({ code: "EEXIST" });

      expect(directorySyncReached).toBe(false);
      await expect(readFile(output)).resolves.toEqual(original);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("retains committed provenance when post-link verification aborts", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-committed-abort-"));
    const output = join(root, "artifact.json");
    const payload = Buffer.from("{\"artifact\":true}\n", "utf8");
    const controller = new AbortController();
    try {
      const failure = await publishSchema9BytesAtomicNoClobber(
        output,
        payload,
        1024,
        "Schema-9 fixture",
        controller.signal,
        {
          afterLink: () => {
            controller.abort(new Error("post-link cancellation"));
            return Promise.resolve();
          },
        },
      ).catch((error: unknown) => error);

      expect(failure).toBeInstanceOf(Error);
      expect(schema9PublicationMayBeCommitted(failure)).toBe(true);
      await expect(readFile(output)).resolves.toEqual(payload);
      expect((await readdir(root)).filter((entry) => entry.includes(".tmp-")))
        .toEqual([]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("never removes a replacement installed after quarantine", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-cleanup-swap-"));
    const temporary = join(root, "temporary");
    const displaced = join(root, "displaced");
    const payload = Buffer.from("owned bytes\n", "utf8");
    try {
      await writeFile(temporary, payload);
      const metadata = await lstat(temporary, { bigint: true });
      await expect(cleanupSchema9TemporaryPublication(
        temporary,
        {
          dev: metadata.dev,
          ino: metadata.ino,
          birthtimeNs: metadata.birthtimeNs,
        },
        "Schema-9 fixture",
        async (quarantine) => {
          await rename(quarantine, displaced);
          await writeFile(quarantine, "replacement\n", "utf8");
        },
      )).rejects.toThrow();
      await expect(readFile(displaced)).resolves.toEqual(payload);
      const replacements = (await readdir(root)).filter(
        (entry) => entry !== "displaced",
      );
      expect(replacements).toHaveLength(1);
      await expect(readFile(join(root, replacements[0] as string), "utf8"))
        .resolves.toBe("replacement\n");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
