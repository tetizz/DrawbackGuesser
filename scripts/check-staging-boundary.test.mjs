import { execFile } from "node:child_process";
import { access, chmod, copyFile, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import { URL } from "node:url";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";

const execFileAsync = promisify(execFile);
const roots = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, {
    force: true,
    recursive: true,
  })));
});

async function git(repository, ...arguments_) {
  return execFileAsync("git", arguments_, {
    cwd: repository,
    encoding: "utf8",
    timeout: 30_000,
    windowsHide: true,
  });
}

describe("staging boundary Git isolation", () => {
  it("does not execute a repository core.fsmonitor command", async () => {
    const root = await mkdtemp(join(tmpdir(), "drawback-boundary-git-"));
    roots.push(root);
    const scripts = join(root, "scripts");
    const marker = join(root, "fsmonitor-executed.txt");
    const monitor = join(root, "malicious-fsmonitor.sh");
    await mkdir(scripts);
    await copyFile(
      new URL("./check-staging-boundary.mjs", import.meta.url),
      join(scripts, "check-staging-boundary.mjs"),
    );
    await git(root, "init", "--quiet");
    await git(root, "config", "user.name", "tetizz");
    await git(
      root,
      "config",
      "user.email",
      "104690265+tetizz@users.noreply.github.com",
    );
    await writeFile(join(root, "tracked.txt"), "one\n", "utf8");
    await git(root, "add", "tracked.txt");
    await git(root, "commit", "--quiet", "-m", "Boundary fixture");
    const shellMarker = marker.replaceAll("\\", "/").replaceAll("'", "'\\''");
    await writeFile(
      monitor,
      `#!/bin/sh\nprintf executed > '${shellMarker}'\n`,
      "utf8",
    );
    await chmod(monitor, 0o755);
    await git(root, "config", "core.fsmonitor", monitor.replaceAll("\\", "/"));

    await expect(execFileAsync(
      process.execPath,
      [join(scripts, "check-staging-boundary.mjs")],
      {
        cwd: root,
        encoding: "utf8",
        timeout: 30_000,
        windowsHide: true,
      },
    )).resolves.toMatchObject({
      stdout: expect.stringContaining("Staging boundary is clean"),
    });
    await expect(access(marker)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("does not execute a Git command injected through PATH", async () => {
    const root = await mkdtemp(join(tmpdir(), "drawback-boundary-path-"));
    roots.push(root);
    const scripts = join(root, "scripts");
    const shadow = join(root, "shadow");
    const marker = join(root, "shadow-git-executed.txt");
    await mkdir(scripts);
    await mkdir(shadow);
    await copyFile(
      new URL("./check-staging-boundary.mjs", import.meta.url),
      join(scripts, "check-staging-boundary.mjs"),
    );
    await git(root, "init", "--quiet");
    await git(root, "config", "user.name", "tetizz");
    await git(
      root,
      "config",
      "user.email",
      "104690265+tetizz@users.noreply.github.com",
    );
    await writeFile(join(root, "tracked.txt"), "one\n", "utf8");
    await git(root, "add", "tracked.txt");
    await git(root, "commit", "--quiet", "-m", "Boundary fixture");
    if (process.platform === "win32") {
      await writeFile(
        join(shadow, "git.cmd"),
        `@echo executed>"${marker}"\r\n@exit /b 91\r\n`,
        "utf8",
      );
    } else {
      const executable = join(shadow, "git");
      await writeFile(
        executable,
        `#!/bin/sh\nprintf executed > '${marker}'\nexit 91\n`,
        "utf8",
      );
      await chmod(executable, 0o755);
    }

    await expect(execFileAsync(
      process.execPath,
      [join(scripts, "check-staging-boundary.mjs")],
      {
        cwd: root,
        encoding: "utf8",
        env: { ...process.env, PATH: shadow },
        timeout: 30_000,
        windowsHide: true,
      },
    )).resolves.toMatchObject({
      stdout: expect.stringContaining("Staging boundary is clean"),
    });
    await expect(access(marker)).rejects.toMatchObject({ code: "ENOENT" });
  });
});
