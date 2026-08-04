import {
  mkdir,
  mkdtemp,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import {
  computeSchema9ProducerRuntimeIdentity,
  schema9ProducerRuntimeDescriptor,
} from "./schema9-producer-runtime-identity.js";

const WORKSPACE_PACKAGES = Object.freeze([
  "packages/shared",
  "packages/probe-search",
  "packages/drawback-engine",
  "packages/chess-core",
  "packages/drawback-search",
  "packages/chess-evaluator",
  "packages/simulation-trace",
  "packages/simulation-arena",
  "apps/engine-cli",
] as const);
const EXTERNAL_BINDINGS = Object.freeze([
  ["packages/chess-core", "chess.js"],
  ["packages/chess-core", "chessops"],
  ["packages/chess-evaluator", "chess.js"],
  ["packages/drawback-engine", "chess.js"],
  ["packages/drawback-search", "chessops"],
] as const);
const WORKSPACE_BINDING_CONSUMERS = Object.freeze([
  "packages/chess-core",
  "apps/engine-cli",
] as const);
const RUNTIME = Object.freeze({
  nodeVersion: "v24.0.0",
  platform: "win32",
  architecture: "x64",
  execArgv: Object.freeze([] as const),
});

async function fixture(
  root: string,
  endOfLine: "\n" | "\r\n",
): Promise<void> {
  const text = (value: string) => value.replace(/\n/gu, endOfLine);
  await writeFile(
    join(root, "package.json"),
    text('{"name":"engine","packageManager":"pnpm@11.9.0"}\n'),
  );
  await writeFile(join(root, "pnpm-lock.yaml"), text("lockfileVersion: '9.0'\n"));
  await writeFile(join(root, "pnpm-workspace.yaml"), text("packages: []\n"));
  await writeFile(join(root, ".npmrc"), text("strict-peer-dependencies=true\n"));
  for (const packagePath of WORKSPACE_PACKAGES) {
    const packageRoot = join(root, ...packagePath.split("/"));
    await mkdir(join(packageRoot, "dist"), { recursive: true });
    const packageName = packagePath === "apps/engine-cli"
      ? "@drawbackengine/cli"
      : `@drawbackengine/${packagePath.split("/").at(-1) ?? "invalid"}`;
    const dependencies = WORKSPACE_BINDING_CONSUMERS.includes(
      packagePath as (typeof WORKSPACE_BINDING_CONSUMERS)[number],
    )
      ? ',"dependencies":{"@drawbackengine/shared":"workspace:*"}'
      : "";
    await writeFile(
      join(packageRoot, "package.json"),
      text(`{"name":"${packageName}"${dependencies}}\n`),
    );
    await writeFile(
      join(packageRoot, "dist", "index.js"),
      `export const id = ${JSON.stringify(packagePath)};\n`,
    );
  }
  for (const consumer of WORKSPACE_BINDING_CONSUMERS) {
    const bindingRoot = join(
      root,
      ...consumer.split("/"),
      "node_modules",
      "@drawbackengine",
      "shared",
    );
    await mkdir(join(bindingRoot, "dist"), { recursive: true });
    await writeFile(
      join(bindingRoot, "package.json"),
      text('{"name":"@drawbackengine/shared"}\n'),
    );
    await writeFile(
      join(bindingRoot, "dist", "index.js"),
      "export const binding = true;\n",
    );
  }
  await writeFile(
    join(root, "apps", "engine-cli", "dist", "schema9-player-private-cli.js"),
    "export const coordinator = true;\n",
  );
  await writeFile(
    join(
      root,
      "packages",
      "simulation-arena",
      "dist",
      "player-private-parallel-worker.js",
    ),
    "export const worker = true;\n",
  );
  for (const [consumer, dependency] of EXTERNAL_BINDINGS) {
    const dependencyRoot = join(
      root,
      ...consumer.split("/"),
      "node_modules",
      dependency,
    );
    await mkdir(dependencyRoot, { recursive: true });
    const manifest = dependency === "chessops"
      ? {
        name: dependency,
        version: "1.0.0",
        main: "index.js",
        dependencies: { "@badrap/result": "1.0.0" },
        optionalDependencies: { "missing-optional-runtime": "1.0.0" },
      }
      : { name: dependency, version: "1.0.0", main: "index.js" };
    await writeFile(
      join(dependencyRoot, "package.json"),
      `${JSON.stringify(manifest)}\n`,
    );
    await writeFile(
      join(dependencyRoot, "index.js"),
      `export const dependency = ${JSON.stringify(dependency)};\n`,
    );
    if (dependency === "chessops") {
      const transitiveRoot = join(
        dependencyRoot,
        "node_modules",
        "@badrap",
        "result",
      );
      await mkdir(transitiveRoot, { recursive: true });
      await writeFile(
        join(transitiveRoot, "package.json"),
        '{"name":"@badrap/result","version":"1.0.0","main":"index.js"}\n',
      );
      await writeFile(
        join(transitiveRoot, "index.js"),
        "export const result = true;\n",
      );
    }
  }
}

describe("schema-9 producer runtime identity", () => {
  it("is path-independent and canonicalizes tracked text line endings", async () => {
    const parent = await mkdtemp(join(tmpdir(), "schema9-runtime-fixture-"));
    const lf = join(parent, "lf");
    const crlf = join(parent, "crlf");
    try {
      await mkdir(lf);
      await mkdir(crlf);
      await fixture(lf, "\n");
      await fixture(crlf, "\r\n");
      const first = await computeSchema9ProducerRuntimeIdentity(lf, RUNTIME);
      const second = await computeSchema9ProducerRuntimeIdentity(crlf, RUNTIME);
      expect(second).toEqual(first);
      expect(first.format).toBe("drawbackengine-schema9-producer-runtime");
      expect(first.version).toBe(1);
      expect(first.algorithm).toBe("sha256-engine-runtime-tree-v1");
      expect(first.aggregateSha256).toBe(
        "d9edd0d078bae33abce9b19605f7688a74ac87e32b4fabafe452867cd7effcde",
      );
      expect(first.coordinator.files).toBeGreaterThan(
        first.parallelWorker.files,
      );
    } finally {
      await rm(parent, { recursive: true, force: true });
    }
  });

  it("changes both closures when shared executable bytes change", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-runtime-mutation-"));
    try {
      await fixture(root, "\n");
      const before = await computeSchema9ProducerRuntimeIdentity(root, RUNTIME);
      await writeFile(
        join(root, "packages", "shared", "dist", "index.js"),
        "export const changed = true;\n",
      );
      const after = await computeSchema9ProducerRuntimeIdentity(root, RUNTIME);
      expect(after.coordinator.sha256).not.toBe(before.coordinator.sha256);
      expect(after.parallelWorker.sha256).not.toBe(
        before.parallelWorker.sha256,
      );
      expect(after.aggregateSha256).not.toBe(before.aggregateSha256);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("stops a pure runtime walk when its programmatic signal aborts", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-runtime-abort-"));
    try {
      await fixture(root, "\n");
      const controller = new AbortController();
      const cancellation = new Error("injected runtime walk cancellation");
      const operation = computeSchema9ProducerRuntimeIdentity(
        root,
        RUNTIME,
        controller.signal,
      );
      queueMicrotask(() => {
        controller.abort(cancellation);
      });
      await expect(operation).rejects.toBe(cancellation);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("binds transitive packages and fails closed for required gaps", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-runtime-transitive-"));
    const chessopsRoot = join(
      root,
      "packages",
      "chess-core",
      "node_modules",
      "chessops",
    );
    const transitiveRoot = join(
      chessopsRoot,
      "node_modules",
      "@badrap",
      "result",
    );
    try {
      await fixture(root, "\n");
      const before = await computeSchema9ProducerRuntimeIdentity(root, RUNTIME);
      await writeFile(
        join(transitiveRoot, "index.js"),
        "export const result = 'changed';\n",
      );
      const changed = await computeSchema9ProducerRuntimeIdentity(root, RUNTIME);
      expect(changed.coordinator.sha256).not.toBe(before.coordinator.sha256);
      expect(changed.parallelWorker.sha256).not.toBe(
        before.parallelWorker.sha256,
      );

      await rm(transitiveRoot, { recursive: true, force: true });
      await writeFile(
        join(chessopsRoot, "package.json"),
        '{"name":"chessops","version":"1.0.0","main":"index.js","dependencies":{"@badrap/result":"1.0.0"}}\n',
      );
      await expect(
        computeSchema9ProducerRuntimeIdentity(root, RUNTIME),
      ).rejects.toThrow(/could not be resolved/u);

      const optionalRoot = join(root, "optional-fixture");
      await mkdir(optionalRoot);
      await fixture(optionalRoot, "\n");
      const optionalChessopsRoot = join(
        optionalRoot,
        "packages",
        "chess-core",
        "node_modules",
        "chessops",
      );
      await rm(join(
        optionalChessopsRoot,
        "node_modules",
        "@badrap",
        "result",
      ), { recursive: true, force: true });
      await writeFile(
        join(optionalChessopsRoot, "package.json"),
        '{"name":"chessops","version":"1.0.0","main":"index.js","optionalDependencies":{"@badrap/result":"1.0.0"}}\n',
      );
      await expect(
        computeSchema9ProducerRuntimeIdentity(optionalRoot, RUNTIME),
      ).resolves.toBeDefined();

      const cycleRoot = join(root, "cycle-fixture");
      await mkdir(cycleRoot);
      await fixture(cycleRoot, "\n");
      const cycleResultManifest = join(
        cycleRoot,
        "packages",
        "chess-core",
        "node_modules",
        "chessops",
        "node_modules",
        "@badrap",
        "result",
        "package.json",
      );
      await writeFile(
        cycleResultManifest,
        '{"name":"@badrap/result","version":"1.0.0","main":"index.js","dependencies":{"chessops":"1.0.0"}}\n',
      );
      await expect(
        computeSchema9ProducerRuntimeIdentity(cycleRoot, RUNTIME),
      ).resolves.toBeDefined();
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("matches the pinned Engine runtime implementation exactly", async () => {
    const engineRoot = resolve("engine");
    const moduleUrl = pathToFileURL(join(
      engineRoot,
      "apps",
      "engine-cli",
      "dist",
      "schema9-runtime-identity.js",
    )).href;
    const engineModule: unknown = await import(moduleUrl);
    const computeEngineIdentity = runtimeIdentityComputer(engineModule);
    const expected = await computeEngineIdentity(engineRoot, RUNTIME);
    const actual = await computeSchema9ProducerRuntimeIdentity(
      engineRoot,
      RUNTIME,
    );
    expect(actual).toEqual(expected);
  }, 30_000);

  it("rejects preload flags and a missing deferred worker", async () => {
    expect(() => schema9ProducerRuntimeDescriptor({
      execArgv: ["--inspect"],
      environment: {},
    })).toThrow(/execution arguments/u);
    expect(() => schema9ProducerRuntimeDescriptor({
      execArgv: [],
      environment: { NODE_OPTIONS: "--require injected.js" },
    })).toThrow(/NODE_OPTIONS/u);

    const root = await mkdtemp(join(tmpdir(), "schema9-runtime-worker-"));
    try {
      await fixture(root, "\n");
      await rm(join(
        root,
        "packages",
        "simulation-arena",
        "dist",
        "player-private-parallel-worker.js",
      ));
      await expect(
        computeSchema9ProducerRuntimeIdentity(root, RUNTIME),
      ).rejects.toThrow(/parallel-worker/u);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});

type RuntimeIdentityComputer = (
  repository: string,
  runtime: typeof RUNTIME,
) => Promise<unknown>;

function runtimeIdentityComputer(value: unknown): RuntimeIdentityComputer {
  if (typeof value !== "object" || value === null) {
    throw new TypeError("Pinned Engine runtime module is invalid.");
  }
  if (!("computeSchema9ProducerRuntimeIdentity" in value)) {
    throw new TypeError("Pinned Engine runtime computer is missing.");
  }
  const candidate: unknown = value.computeSchema9ProducerRuntimeIdentity;
  if (typeof candidate !== "function") {
    throw new TypeError("Pinned Engine runtime computer is missing.");
  }
  return candidate as RuntimeIdentityComputer;
}
