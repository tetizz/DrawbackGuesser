import { EventEmitter } from "node:events";
import { spawn, type ChildProcess } from "node:child_process";
import {
  access,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { Writable } from "node:stream";
import { pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import { writeSchema9JsonLine } from "./schema9-json-line-writer.js";
import { withSchema9TemporaryCheckout } from "./schema9-ledger-cli.js";
import {
  installSchema9TerminationSignal,
  preserveSchema9PrimaryWithCleanup,
  runSchema9BoundedCommand,
  Schema9CommandFailure,
  Schema9TerminationError,
} from "./schema9-process-lifecycle.js";

interface ProcessIds {
  readonly child: number;
  readonly grandchild: number;
}

describe("schema-9 process lifecycle", () => {
  it("propagates EPIPE from the result destination", async () => {
    const output = new Writable({
      write(_chunk, _encoding, callback) {
        const failure = new Error("injected EPIPE");
        Object.assign(failure, { code: "EPIPE" });
        callback(failure);
      },
    });
    await expect(writeSchema9JsonLine(output, { verified: true }))
      .rejects.toMatchObject({ code: "EPIPE" });
  });

  it("keeps excessive output primary when cleanup also fails", () => {
    const primary = new Error("produced excessive output");
    const cleanup = new Error("cleanup failed");
    const failure = preserveSchema9PrimaryWithCleanup(
      primary,
      cleanup,
      "output cleanup failed",
    );
    expect(failure).toBeInstanceOf(AggregateError);
    expect((failure as AggregateError).errors).toEqual([primary, cleanup]);
  });

  it("keeps a launcher failure primary when cleanup also fails", () => {
    const primary = new Schema9CommandFailure(
      "injected launcher failure",
      null,
      { cause: new Error("injected stdin failure") },
    );
    const cleanup = new Error("injected cleanup failure");
    const failure = preserveSchema9PrimaryWithCleanup(
      primary,
      cleanup,
      "launcher and cleanup failed",
    );
    expect(failure).toBeInstanceOf(AggregateError);
    expect((failure as AggregateError).errors).toEqual([primary, cleanup]);
  });

  it("terminates a real command that exceeds its output budget", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-output-limit-"));
    try {
      await expect(runSchema9BoundedCommand(
        process.execPath,
        [
          "--eval",
          "process.stdout.write('x'.repeat(65536));"
            + "setInterval(() => undefined, 1000);",
        ],
        {
          cwd: root,
          environment: { ...process.env },
          timeoutMilliseconds: 20_000,
          maxOutputBytes: 128,
          description: "output fixture",
        },
      )).rejects.toThrow("produced excessive output");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  }, 30_000);

  it.each([
    ["SIGINT", 130],
    ["SIGTERM", 143],
  ] as const)("maps %s to a cooperative exit code", (signal, exitCode) => {
    const source = new EventEmitter();
    const installed = installSchema9TerminationSignal(source);
    try {
      source.emit(signal);
      expect(installed.signal.aborted).toBe(true);
      expect(installed.signal.reason).toBeInstanceOf(Schema9TerminationError);
      expect(installed.signal.reason).toMatchObject({ signal, exitCode });
    } finally {
      installed.dispose();
    }
    expect(source.listenerCount("SIGINT")).toBe(0);
    expect(source.listenerCount("SIGTERM")).toBe(0);
  });

  it("kills a real child and grandchild and cleans its owned checkout", async () => {
    const parent = await mkdtemp(join(tmpdir(), "schema9-tree-parent-"));
    const controller = new AbortController();
    let ids: ProcessIds | undefined;
    try {
      const operation = withSchema9TemporaryCheckout(async (checkout) => {
        const marker = join(checkout, "processes.json");
        const wrapper = String.raw`
const { spawn } = require("node:child_process");
const { writeFileSync } = require("node:fs");
const marker = process.argv[1];
const grandchild = spawn(process.execPath, [
  "--eval",
  "setInterval(() => undefined, 1000)",
], { stdio: "ignore", windowsHide: true });
writeFileSync(marker, JSON.stringify({
  child: process.pid,
  grandchild: grandchild.pid,
}));
setInterval(() => undefined, 1000);
`;
        return runSchema9BoundedCommand(
          process.execPath,
          ["--eval", wrapper, marker],
          {
            cwd: checkout,
            environment: { ...process.env },
            signal: controller.signal,
            timeoutMilliseconds: 20_000,
            description: "lifecycle fixture",
          },
        );
      }, parent);
      const markerPath = await waitForOneProcessMarker(parent);
      ids = JSON.parse(await readFile(markerPath, "utf8")) as ProcessIds;
      expect(ids.child).toBeGreaterThan(0);
      expect(ids.grandchild).toBeGreaterThan(0);
      controller.abort(new Error("injected cancellation"));
      await expect(operation).rejects.toThrow("injected cancellation");
      await expect(waitForProcessExit(ids.child)).resolves.toBeUndefined();
      await expect(waitForProcessExit(ids.grandchild)).resolves.toBeUndefined();
      await expect(readdir(parent)).resolves.toEqual([]);
    } finally {
      if (ids !== undefined) {
        killFixtureProcess(ids.child);
        killFixtureProcess(ids.grandchild);
      }
      await rm(parent, { recursive: true, force: true });
    }
  }, 30_000);

  it("kills a real process tree after a command timeout", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-tree-timeout-"));
    const marker = join(root, "processes.json");
    let ids: ProcessIds | undefined;
    try {
      const wrapper = String.raw`
const { spawn } = require("node:child_process");
const { writeFileSync } = require("node:fs");
const marker = process.argv[1];
const grandchild = spawn(process.execPath, [
  "--eval",
  "setInterval(() => undefined, 1000)",
], { stdio: "ignore", windowsHide: true });
writeFileSync(marker, JSON.stringify({
  child: process.pid,
  grandchild: grandchild.pid,
}));
setInterval(() => undefined, 1000);
`;
      const operation = runSchema9BoundedCommand(
        process.execPath,
        ["--eval", wrapper, marker],
        {
          cwd: root,
          environment: { ...process.env },
          timeoutMilliseconds: 750,
          description: "timeout fixture",
        },
      );
      await waitForPath(marker);
      ids = JSON.parse(await readFile(marker, "utf8")) as ProcessIds;
      await expect(operation).rejects.toThrow("exceeded its time limit");
      await expect(waitForProcessExit(ids.child)).resolves.toBeUndefined();
      await expect(waitForProcessExit(ids.grandchild)).resolves.toBeUndefined();
    } finally {
      if (ids !== undefined) {
        killFixtureProcess(ids.child);
        killFixtureProcess(ids.grandchild);
      }
      await rm(root, { recursive: true, force: true });
    }
  }, 30_000);

  it.skipIf(process.platform !== "win32")(
    "kills an inherited-stdio grandchild after its direct parent is already dead",
    async () => {
      const root = await mkdtemp(join(tmpdir(), "schema9-fast-parent-"));
      const marker = join(root, "processes.json");
      let ids: ProcessIds | undefined;
      try {
        const wrapper = String.raw`
const { spawn } = require("node:child_process");
const { writeFileSync } = require("node:fs");
const grandchild = spawn(process.execPath, [
  "--eval",
  "setInterval(() => undefined, 1000)",
], {
  detached: true,
  stdio: ["ignore", "inherit", "inherit"],
  windowsHide: true,
});
grandchild.unref();
writeFileSync(process.argv[1], JSON.stringify({
  child: process.pid,
  grandchild: grandchild.pid,
}));
`;
        const operation = runSchema9BoundedCommand(
          process.execPath,
          ["--eval", wrapper, marker],
          {
            cwd: root,
            environment: { ...process.env },
            timeoutMilliseconds: 20_000,
            description: "fast-parent fixture",
          },
        );
        await waitForPath(marker);
        ids = JSON.parse(await readFile(marker, "utf8")) as ProcessIds;
        await expect(waitForProcessExit(ids.child)).resolves.toBeUndefined();
        await expect(operation).resolves.toEqual({ stdout: "", stderr: "" });
        await expect(waitForProcessExit(ids.grandchild)).resolves.toBeUndefined();
      } finally {
        if (ids !== undefined) {
          killFixtureProcess(ids.child);
          killFixtureProcess(ids.grandchild);
        }
        await rm(root, { recursive: true, force: true });
      }
    },
    30_000,
  );

  it.skipIf(process.platform !== "win32")(
    "keeps launcher injection variables out of the supervisor environment",
    async () => {
      const root = await mkdtemp(join(tmpdir(), "schema9-host-env-"));
      try {
        const environment = {
          SCHEMA9_TARGET_PROBE: "target-visible",
          COR_ENABLE_PROFILING: "1",
          COR_PROFILER: "{11111111-1111-1111-1111-111111111111}",
          COMPlus_ReadyToRun: "0",
          PSModulePath: join(root, "hostile-modules"),
        };
        const source = String.raw`
process.stdout.write(JSON.stringify({
  probe: process.env.SCHEMA9_TARGET_PROBE,
  cor: process.env.COR_ENABLE_PROFILING,
  plus: process.env.COMPlus_ReadyToRun,
  modules: process.env.PSModulePath,
}));
`;
        const result = await runSchema9BoundedCommand(
          process.execPath,
          ["--eval", source],
          {
            cwd: root,
            environment,
            timeoutMilliseconds: 20_000,
            description: "closed-host-environment fixture",
          },
        );
        expect(JSON.parse(result.stdout)).toEqual({
          probe: "target-visible",
          cor: "1",
          plus: "0",
          modules: join(root, "hostile-modules"),
        });
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    },
    30_000,
  );

  it(
    "rejects null bytes before native command-line classification",
    async () => {
      await expect(runSchema9BoundedCommand(
        `${process.execPath}\0--must-not-be-truncated`,
        ["--version"],
        {
          cwd: process.cwd(),
          environment: { ...process.env },
          timeoutMilliseconds: 20_000,
          description: "null executable fixture",
        },
      )).rejects.toThrow("executable contains a null byte");
      await expect(runSchema9BoundedCommand(
        process.execPath,
        [`--version\0--must-not-be-truncated`],
        {
          cwd: process.cwd(),
          environment: { ...process.env },
          timeoutMilliseconds: 20_000,
          description: "null argument fixture",
        },
      )).rejects.toThrow("argument 0 contains a null byte");
    },
  );

  it.skipIf(process.platform !== "win32")(
    "ignores a poisoned SystemRoot and cleans the real process tree",
    async () => {
      const root = await mkdtemp(join(tmpdir(), "schema9-taskkill-failure-"));
      const previousSystemRoot = process.env["SystemRoot"];
      let ids: ProcessIds | undefined;
      try {
        const childEnvironment = { ...process.env };
        process.env["SystemRoot"] = root;
        const controller = new AbortController();
        const marker = join(root, "failure-processes.json");
        const wrapper = String.raw`
const { spawn } = require("node:child_process");
const { writeFileSync } = require("node:fs");
const marker = process.argv[1];
const grandchild = spawn(process.execPath, [
  "--eval",
  "setInterval(() => undefined, 1000)",
], { stdio: ["ignore", "inherit", "inherit"], windowsHide: true });
writeFileSync(marker, JSON.stringify({
  child: process.pid,
  grandchild: grandchild.pid,
}));
setInterval(() => undefined, 1000);
`;
        const operation = runSchema9BoundedCommand(
          process.execPath,
          ["--eval", wrapper, marker],
          {
            cwd: root,
            environment: childEnvironment,
            signal: controller.signal,
            timeoutMilliseconds: 20_000,
            cleanupSettlementMilliseconds: 300,
            description: "cleanup failure fixture",
          },
        );
        const outcome = operation.then(
          () => Object.freeze({ ok: true as const }),
          (error: unknown) => Object.freeze({ ok: false as const, error }),
        );
        await waitForPath(marker);
        ids = JSON.parse(await readFile(marker, "utf8")) as ProcessIds;
        controller.abort(new Error("injected cancellation"));
        const result = await outcome;
        expect(result.ok).toBe(false);
        if (result.ok) {
          throw new Error("Cleanup-failure fixture unexpectedly succeeded.");
        }
        expect(result.error).toBeInstanceOf(Error);
        expect(String(result.error)).toContain("injected cancellation");
        await expect(waitForProcessExit(ids.child)).resolves.toBeUndefined();
        await expect(waitForProcessExit(ids.grandchild)).resolves.toBeUndefined();
      } finally {
        if (previousSystemRoot === undefined) {
          delete process.env["SystemRoot"];
        } else {
          process.env["SystemRoot"] = previousSystemRoot;
        }
        if (ids !== undefined) {
          killFixtureProcess(ids.child);
          killFixtureProcess(ids.grandchild);
        }
        await rm(root, { recursive: true, force: true });
      }
    },
    30_000,
  );

  it.skipIf(process.platform !== "win32")(
    "lets a poisoned wrapper settle after authenticated tree cleanup",
    async () => {
      const root = await mkdtemp(join(tmpdir(), "schema9-wrapper-exit-"));
      const processMarker = join(root, "processes.json");
      const caughtMarker = join(root, "caught.txt");
      const wrapperPath = join(root, "wrapper.mjs");
      let wrapper: ChildProcess | undefined;
      let ids: ProcessIds | undefined;
      const wrapperStderr: Buffer[] = [];
      try {
        const lifecycleUrl = pathToFileURL(resolve(
          "apps/dataset-cli/src/schema9-process-lifecycle.ts",
        )).href;
        const tsxLoaderUrl = pathToFileURL(resolve(
          "apps/dataset-cli/node_modules/tsx/dist/loader.mjs",
        )).href;
        const childSource = [
          'const { spawn } = require("node:child_process");',
          'const { writeFileSync } = require("node:fs");',
          "const marker = process.argv[1];",
          "const grandchild = spawn(process.execPath, [",
          '  "--eval",',
          '  "setInterval(() => undefined, 1000)",',
          '], { stdio: ["ignore", "inherit", "inherit"], windowsHide: true });',
          "writeFileSync(marker, JSON.stringify({",
          "  child: process.pid,",
          "  grandchild: grandchild.pid,",
          "}));",
          "setInterval(() => undefined, 1000);",
          "",
        ].join("\n");
        await writeFile(wrapperPath, String.raw`
import { access, writeFile } from "node:fs/promises";
import { runSchema9BoundedCommand } from ${JSON.stringify(lifecycleUrl)};
const [root, processMarker, caughtMarker] = process.argv.slice(2);
const childEnvironment = { ...process.env };
process.env.SystemRoot = root;
const controller = new AbortController();
const childSource = ${JSON.stringify(childSource)};
const operation = runSchema9BoundedCommand(
  process.execPath,
  ["--eval", childSource, processMarker],
  {
    cwd: root,
    environment: childEnvironment,
    signal: controller.signal,
    timeoutMilliseconds: 20_000,
    cleanupSettlementMilliseconds: 250,
    description: "wrapper cleanup failure fixture",
  },
);
for (;;) {
  try {
    await access(processMarker);
    break;
  } catch {
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 10));
  }
}
controller.abort(new Error("injected wrapper cancellation"));
try {
  await operation;
  throw new Error("cleanup failure fixture unexpectedly succeeded");
} catch (error) {
  await writeFile(caughtMarker, String(error));
}
`, "utf8");
        wrapper = spawn(
          process.execPath,
          [
            "--import",
            tsxLoaderUrl,
            wrapperPath,
            root,
            processMarker,
            caughtMarker,
          ],
          {
            cwd: process.cwd(),
            windowsHide: true,
            stdio: ["ignore", "pipe", "pipe"],
          },
        );
        wrapper.stderr?.on("data", (chunk: Buffer) => {
          wrapperStderr.push(chunk);
        });
        const wrapperClose = waitForChildClose(wrapper, 12_000);
        await Promise.race([
          waitForPath(processMarker),
          wrapperClose.then((code) => {
            throw new Error(
              `Lifecycle wrapper exited ${String(code)} before its marker: ${
                Buffer.concat(wrapperStderr).toString("utf8").trim()
              }`,
            );
          }),
        ]);
        ids = JSON.parse(await readFile(processMarker, "utf8")) as ProcessIds;
        await waitForPath(caughtMarker);
        await expect(wrapperClose).resolves.toBe(0);
        await expect(readFile(caughtMarker, "utf8"))
          .resolves.toContain("injected wrapper cancellation");
        await expect(waitForProcessExit(ids.child)).resolves.toBeUndefined();
        await expect(waitForProcessExit(ids.grandchild)).resolves.toBeUndefined();
      } finally {
        if (wrapper?.pid !== undefined) {
          killFixtureProcess(wrapper.pid);
        }
        if (ids !== undefined) {
          killFixtureProcess(ids.child);
          killFixtureProcess(ids.grandchild);
        }
        await rm(root, { recursive: true, force: true });
      }
    },
    30_000,
  );
});

async function waitForOneProcessMarker(parent: string): Promise<string> {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    for (const entry of await readdir(parent)) {
      const marker = join(parent, entry, "checkout", "processes.json");
      try {
        await access(marker);
        return marker;
      } catch {
        // The process has not published its PID pair yet.
      }
    }
    await delay(25);
  }
  throw new Error("Lifecycle fixture did not publish process IDs.");
}

async function waitForPath(path: string): Promise<void> {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    try {
      await access(path);
      return;
    } catch {
      await delay(25);
    }
  }
  throw new Error("Lifecycle fixture did not publish its marker.");
}

async function waitForProcessExit(pid: number): Promise<void> {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    if (!processExists(pid)) {
      return;
    }
    await delay(25);
  }
  throw new Error(`Process ${String(pid)} remained alive after cleanup.`);
}

function processExists(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error: unknown) {
    return !(
      error instanceof Error
      && "code" in error
      && error.code === "ESRCH"
    );
  }
}

function killFixtureProcess(pid: number): void {
  if (!processExists(pid)) {
    return;
  }
  try {
    process.kill(pid, "SIGKILL");
  } catch {
    // Best effort after a failed regression assertion.
  }
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((accept) => setTimeout(accept, milliseconds));
}

function waitForChildClose(
  child: ChildProcess,
  timeoutMilliseconds: number,
): Promise<number | null> {
  return new Promise((accept, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error("Lifecycle wrapper did not exit after forced settlement."));
    }, timeoutMilliseconds);
    child.once("close", (code) => {
      clearTimeout(timeout);
      accept(code);
    });
    child.once("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
  });
}
