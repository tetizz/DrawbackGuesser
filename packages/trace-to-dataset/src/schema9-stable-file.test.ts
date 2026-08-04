import {
  link,
  mkdir,
  mkdtemp,
  lstat,
  rename,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { withSchema9OwnedStableFiles } from "./schema9-stable-file.js";
import { runSchema9LinkedTaskGroup } from "./schema9-task-group.js";

describe("schema-9 stable file ownership", () => {
  it("rejects two pathnames for the same file object", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-hardlink-"));
    try {
      const first = join(root, "first.txt");
      const second = join(root, "second.txt");
      await writeFile(first, "stable\n", "utf8");
      await link(first, second);
      await expect(withSchema9OwnedStableFiles(
        [
          { path: first, label: "first" },
          { path: second, label: "second" },
        ],
        () => Promise.resolve(undefined),
      )).rejects.toThrow("distinct file object");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("reads through the retained handle and rejects a pathname swap", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-path-swap-"));
    const input = join(root, "input.txt");
    const displaced = join(root, "displaced.txt");
    let observed = "";
    try {
      await writeFile(input, "authenticated\n", "utf8");
      await expect(withSchema9OwnedStableFiles(
        [{ path: input, label: "fixture input" }],
        async ({ files }) => {
          const file = files[0];
          if (file === undefined) {
            throw new Error("Stable fixture file was not retained.");
          }
          await rename(input, displaced);
          await writeFile(input, "replacement\n", "utf8");
          const bytes = Buffer.alloc(Number(file.size));
          const result = await file.handle.read({
            buffer: bytes,
            position: 0,
          });
          observed = bytes.subarray(0, result.bytesRead).toString("utf8");
        },
      )).rejects.toThrow("changed while it was authenticated");
      expect(observed).toBe("authenticated\n");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("rejects a link or junction in an input path", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-link-component-"));
    const target = join(root, "target");
    const linked = join(root, "linked");
    try {
      await mkdir(target);
      await writeFile(join(target, "input.txt"), "stable\n", "utf8");
      await symlink(
        target,
        linked,
        process.platform === "win32" ? "junction" : "dir",
      );
      expect((await lstat(linked)).isSymbolicLink()).toBe(true);
      await expect(withSchema9OwnedStableFiles(
        [{ path: join(linked, "input.txt"), label: "linked input" }],
        () => Promise.resolve(undefined),
      )).rejects.toThrow("symbolic links or junctions");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});

describe("schema-9 linked task groups", () => {
  it("aborts and settles siblings before reporting the primary failure", async () => {
    const primary = new Error("primary task failure");
    let siblingSettled = false;
    const operation = runSchema9LinkedTaskGroup([
      async () => {
        await new Promise((accept) => setTimeout(accept, 10));
        throw primary;
      },
      async (signal) => new Promise<void>((accept) => {
        signal.addEventListener("abort", () => {
          siblingSettled = true;
          accept();
        }, { once: true });
      }),
    ], undefined, "fixture task group");

    await expect(operation).rejects.toBe(primary);
    expect(siblingSettled).toBe(true);
  });

  it("normalizes a non-Error task rejection", async () => {
    const failure = await runSchema9LinkedTaskGroup([
      () => {
        // Deliberately violate the Promise convention to test normalization.
        // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
        return Promise.reject("string failure");
      },
    ], undefined, "fixture task group").catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(Error);
    expect(failure).toMatchObject({
      message: "fixture task group task failed.",
      cause: "string failure",
    });
  });
});
