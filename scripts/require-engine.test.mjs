import { execFile } from "node:child_process";
import {
  copyFile,
  mkdir,
  mkdtemp,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import { URL } from "node:url";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";

const execFileAsync = promisify(execFile);

async function git(repository, ...arguments_) {
  return execFileAsync("git", ["-C", repository, ...arguments_], {
    encoding: "utf8",
    timeout: 30_000,
    windowsHide: true,
  });
}

describe("DrawbackEngine workspace boundary", () => {
  it("rejects an Engine HEAD that differs from the index gitlink", async () => {
    const root = await mkdtemp(join(tmpdir(), "require-engine-gitlink-"));
    const scripts = join(root, "scripts");
    const engine = join(root, "engine");
    const packages = [
      ["shared", "@drawbackengine/shared"],
      ["drawback-engine", "@drawbackengine/drawback-engine"],
      ["chess-core", "@drawbackengine/chess-core"],
      ["chess-evaluator", "@drawbackengine/chess-evaluator"],
      ["simulation-trace", "@drawbackengine/simulation-trace"],
    ];
    try {
      await mkdir(scripts);
      await mkdir(engine);
      await copyFile(
        new URL("./require-engine.mjs", import.meta.url),
        join(scripts, "require-engine.mjs"),
      );
      await git(engine, "init", "--quiet");
      await git(engine, "config", "user.name", "tetizz");
      await git(
        engine,
        "config",
        "user.email",
        "104690265+tetizz@users.noreply.github.com",
      );
      for (const [directory, name] of packages) {
        const packageRoot = join(engine, "packages", directory);
        await mkdir(packageRoot, { recursive: true });
        await writeFile(
          join(packageRoot, "package.json"),
          `${JSON.stringify({ name })}\n`,
          "utf8",
        );
      }
      await git(engine, "add", ".");
      await git(engine, "commit", "--quiet", "-m", "Engine fixture");

      await git(root, "init", "--quiet");
      await git(root, "config", "user.name", "tetizz");
      await git(
        root,
        "config",
        "user.email",
        "104690265+tetizz@users.noreply.github.com",
      );
      await writeFile(join(root, "README.md"), "fixture\n", "utf8");
      await git(root, "add", "README.md", "engine");
      await git(root, "commit", "--quiet", "-m", "Guesser fixture");

      await expect(execFileAsync(
        process.execPath,
        [join(scripts, "require-engine.mjs")],
        {
          cwd: root,
          encoding: "utf8",
          timeout: 30_000,
          windowsHide: true,
        },
      )).resolves.toMatchObject({
        stdout: expect.stringContaining("gitlink and workspace package"),
      });

      await writeFile(join(engine, "changed.txt"), "new commit\n", "utf8");
      await git(engine, "add", "changed.txt");
      await git(engine, "commit", "--quiet", "-m", "Engine drift");
      const failure = await execFileAsync(
        process.execPath,
        [join(scripts, "require-engine.mjs")],
        {
          cwd: root,
          encoding: "utf8",
          timeout: 30_000,
          windowsHide: true,
        },
      ).catch((error) => error);
      expect(failure).toBeInstanceOf(Error);
      expect(failure).toMatchObject({
        code: 2,
        stderr: expect.stringContaining("does not match the commit"),
      });

      await git(root, "add", "engine");
      await expect(execFileAsync(
        process.execPath,
        [join(scripts, "require-engine.mjs")],
        {
          cwd: root,
          encoding: "utf8",
          timeout: 30_000,
          windowsHide: true,
        },
      )).resolves.toMatchObject({
        stdout: expect.stringContaining("gitlink and workspace package"),
      });
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  }, 60_000);
});
