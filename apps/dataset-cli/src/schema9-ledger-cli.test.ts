import { execFile } from "node:child_process";
import {
  access,
  mkdir,
  mkdtemp,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { pathToFileURL } from "node:url";
import {
  SCHEMA9_EXECUTION_MANIFEST_ALGORITHM,
  SCHEMA9_SPLIT_SEED_ROOTS,
  type Schema9ExecutionIdentity,
} from "@drawbackguesser/trace-to-dataset";
import { describe, expect, it } from "vitest";
import {
  assertEngineRuntimeCheckout,
  assertExactReproducibleExecutionIdentity,
  cloneRepositoryAtCommit,
  createGitRepositoryVerifier,
  moduleGraphIdentity,
  parseSchema9LedgerCliArguments,
  requireNoSubmoduleIgnoreConfiguration,
  schema9RuntimeIdentity,
  withSchema9TemporaryCheckout,
} from "./schema9-ledger-cli.js";

const SPLITS = ["train", "validation-a", "validation-b", "test"] as const;
const execFileAsync = promisify(execFile);

async function git(repository: string, ...arguments_: string[]) {
  return execFileAsync(
    "git",
    ["-C", repository, ...arguments_],
    { windowsHide: true },
  );
}

function completeArguments(): string[] {
  const values = [
    "--operation",
    "create",
    "--ledger",
    "receipts/schema9-ledger.json",
    "--guesser-repository",
    "guesser",
    "--engine-repository",
    "engine",
    "--guesser-commit",
    "a".repeat(40),
    "--converter-engine-commit",
    "b".repeat(40),
    "--producer-converter-policy",
    "converter-ancestor/v1",
  ];
  for (const split of SPLITS) {
    const [labelRoot, gameplayRoot, parametersRoot] =
      SCHEMA9_SPLIT_SEED_ROOTS[split];
    values.push(
      `--${split}-trace`,
      `private/${split}.trace.ndjson`,
      `--${split}-converted`,
      `private/${split}.schema9.ndjson`,
      `--${split}-launch-receipt`,
      `private/${split}.launch.json`,
      `--${split}-completion-receipt`,
      `private/${split}.completion.json`,
      `--${split}-schedule-id`,
      `schema9-${split}`,
      `--${split}-label-seed-root`,
      String(labelRoot),
      `--${split}-gameplay-seed-root`,
      String(gameplayRoot),
      `--${split}-parameters-seed-root`,
      String(parametersRoot),
      `--${split}-producer-engine-commit`,
      "c".repeat(40),
    );
  }
  return values;
}

describe("schema-9 corpus ledger CLI arguments", () => {
  it("requires every explicit file and preserves the four split names", () => {
    const invocationDirectory = resolve("fixture-invocation");
    const parsed = parseSchema9LedgerCliArguments(
      completeArguments(),
      invocationDirectory,
    );

    expect(parsed.operation).toBe("create");
    expect(parsed.corpus.producerConverterPolicy)
      .toBe("converter-ancestor/v1");
    expect(Object.keys(parsed.corpus.splits)).toEqual(SPLITS);
    expect(parsed.corpus.splits["validation-b"]).toMatchObject({
      tracePath: resolve(
        invocationDirectory,
        "private/validation-b.trace.ndjson",
      ),
      scheduleId: "schema9-validation-b",
      seedRoots: SCHEMA9_SPLIT_SEED_ROOTS["validation-b"],
      producerEngineCommit: "c".repeat(40),
    });
  });

  it("rejects missing, duplicate, unsupported, and weak policy flags", () => {
    const complete = completeArguments();
    expect(() =>
      parseSchema9LedgerCliArguments(complete.slice(0, -2))
    ).toThrow("Missing schema-9 ledger flags");
    expect(() =>
      parseSchema9LedgerCliArguments([
        ...complete,
        "--operation",
        "verify",
      ])
    ).toThrow("--operation may appear only once");
    expect(() =>
      parseSchema9LedgerCliArguments([
        ...complete.slice(0, -2),
        "--unknown",
        "value",
      ])
    ).toThrow("Unsupported or incomplete");
    const policyIndex = complete.indexOf("--producer-converter-policy") + 1;
    const weakPolicy = [...complete];
    weakPolicy[policyIndex] = "allow-any";
    expect(() => parseSchema9LedgerCliArguments(weakPolicy))
      .toThrow("must be exact/v1 or converter-ancestor/v1");
  });

  it("rejects a different or dirty executing checkout and replace refs", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-git-identity-"));
    const engineSource = join(root, "engine-repository");
    const guesser = join(root, "guesser-repository");
    try {
      await execFileAsync("git", ["init", engineSource], { windowsHide: true });
      await git(engineSource, "config", "user.name", "tetizz");
      await git(
        engineSource,
        "config",
        "user.email",
        "104690265+tetizz@users.noreply.github.com",
      );
      await git(engineSource, "config", "core.autocrlf", "false");
      await writeFile(
        join(engineSource, "engine.txt"),
        "engine-one\n",
        "utf8",
      );
      await git(engineSource, "add", "engine.txt");
      await git(engineSource, "commit", "-m", "Engine fixture");

      await execFileAsync("git", ["init", guesser], { windowsHide: true });
      await git(guesser, "config", "user.name", "tetizz");
      await git(
        guesser,
        "config",
        "user.email",
        "104690265+tetizz@users.noreply.github.com",
      );
      await git(guesser, "config", "core.autocrlf", "false");
      await writeFile(join(guesser, "README.md"), "fixture\n", "utf8");
      await git(guesser, "add", "README.md");
      await git(
        guesser,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        engineSource,
        "engine",
      );
      const engineCheckout = join(guesser, "engine");
      await git(engineCheckout, "config", "user.name", "tetizz");
      await git(
        engineCheckout,
        "config",
        "user.email",
        "104690265+tetizz@users.noreply.github.com",
      );
      await git(engineCheckout, "config", "core.autocrlf", "false");
      await writeFile(
        join(engineCheckout, "engine.txt"),
        "engine-checkout\n",
        "utf8",
      );
      await git(engineCheckout, "add", "engine.txt");
      await git(engineCheckout, "commit", "-m", "Engine checkout fixture");
      const engineHead = (await git(engineCheckout, "rev-parse", "HEAD"))
        .stdout.trim();
      await git(guesser, "add", ".gitmodules", "engine");
      await git(guesser, "commit", "-m", "Guesser fixture");
      await expect(
        git(engineCheckout, "status", "--porcelain=v1", "--untracked-files=all"),
      ).resolves.toMatchObject({ stdout: "" });
      await expect(
        git(
          guesser,
          "status",
          "--porcelain=v1",
          "--untracked-files=no",
          "--ignore-submodules=none",
        ),
      ).resolves.toMatchObject({ stdout: "" });
      const guesserHead = (await git(guesser, "rev-parse", "HEAD"))
        .stdout.trim();
      const verifier = createGitRepositoryVerifier(guesser, engineCheckout);
      await expect(verifier.pinnedEngineCommitAt(guesserHead))
        .resolves.toBe(engineHead);
      await expect(
        createGitRepositoryVerifier(guesser, engineSource)
          .pinnedEngineCommitAt(guesserHead),
      ).rejects.toThrow("not the Guesser engine submodule checkout");
      await expect(assertEngineRuntimeCheckout(
        guesser,
        engineCheckout,
        pathToFileURL(join(engineCheckout, "engine.txt")).href,
      )).resolves.toBeUndefined();
      await expect(assertEngineRuntimeCheckout(
        guesser,
        engineCheckout,
        pathToFileURL(join(guesser, "README.md")).href,
      )).rejects.toThrow("Loaded Engine runtime");

      await git(guesser, "config", "submodule.engine.ignore", "all");
      await expect(verifier.pinnedEngineCommitAt(guesserHead))
        .rejects.toThrow("submodule ignore configuration");
      await git(guesser, "config", "--unset", "submodule.engine.ignore");

      await git(guesser, "config", "diff.ignoreSubmodules", "all");
      await expect(verifier.pinnedEngineCommitAt(guesserHead))
        .rejects.toThrow("submodule ignore configuration");
      await git(guesser, "config", "--unset", "diff.ignoreSubmodules");

      await git(guesser, "update-index", "--skip-worktree", "README.md");
      await expect(verifier.pinnedEngineCommitAt(guesserHead))
        .rejects.toThrow("skip-worktree or assume-unchanged");
      await git(guesser, "update-index", "--no-skip-worktree", "README.md");

      await git(guesser, "update-index", "--assume-unchanged", "README.md");
      await expect(verifier.pinnedEngineCommitAt(guesserHead))
        .rejects.toThrow("skip-worktree or assume-unchanged");
      await git(
        guesser,
        "update-index",
        "--no-assume-unchanged",
        "README.md",
      );

      await git(
        engineCheckout,
        "update-index",
        "--assume-unchanged",
        "engine.txt",
      );
      await expect(verifier.pinnedEngineCommitAt(guesserHead))
        .rejects.toThrow("skip-worktree or assume-unchanged");
      await git(
        engineCheckout,
        "update-index",
        "--no-assume-unchanged",
        "engine.txt",
      );

      await writeFile(join(guesser, "README.md"), "dirty\n", "utf8");
      await expect(verifier.pinnedEngineCommitAt(guesserHead))
        .rejects.toThrow("tracked worktree or index is dirty");
      await git(guesser, "add", "README.md");
      await git(guesser, "commit", "-m", "Different head");
      await expect(verifier.pinnedEngineCommitAt(guesserHead))
        .rejects.toThrow("HEAD does not match");

      const latest = (await git(guesser, "rev-parse", "HEAD")).stdout.trim();
      await expect(verifier.pinnedEngineCommitAt(latest))
        .rejects.toThrow("cannot change authenticated repository commits");
      await git(guesser, "update-ref", `refs/replace/${latest}`, latest);
      await expect(verifier.pinnedEngineCommitAt(latest))
        .rejects.toThrow("replace refs");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  }, 60_000);

  it("rejects a module graph that differs from its reproduced build", () => {
    const component = Object.freeze({
      entrypoint: "fixture",
      files: 1,
      bytes: 10,
      sha256: "a".repeat(64),
    });
    const executing: Schema9ExecutionIdentity = Object.freeze({
      algorithm: SCHEMA9_EXECUTION_MANIFEST_ALGORITHM,
      runtime: schema9RuntimeIdentity({}, []),
      parser: component,
      converter: component,
      scheduler: component,
      verifier: component,
      aggregateSha256: "b".repeat(64),
    });
    const reproduced: Schema9ExecutionIdentity = Object.freeze({
      ...executing,
      verifier: Object.freeze({
        ...component,
        sha256: "c".repeat(64),
      }),
    });
    expect(() => {
      assertExactReproducibleExecutionIdentity(executing, reproduced);
    }).toThrow("does not match the reproducible build");
    expect(() => {
      assertExactReproducibleExecutionIdentity(executing, executing);
    }).not.toThrow();
  });

  it("handles linked-worktree config only when Git enables it", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-linked-worktree-"));
    const repository = join(root, "repository");
    const linked = join(root, "linked");
    try {
      await execFileAsync("git", ["init", repository], { windowsHide: true });
      await git(repository, "config", "user.name", "tetizz");
      await git(
        repository,
        "config",
        "user.email",
        "104690265+tetizz@users.noreply.github.com",
      );
      await writeFile(join(repository, "README.md"), "fixture\n", "utf8");
      await git(repository, "add", "README.md");
      await git(repository, "commit", "-m", "Worktree fixture");
      const head = (await git(repository, "rev-parse", "HEAD")).stdout.trim();
      await git(repository, "worktree", "add", "--detach", linked, head);

      await expect(requireNoSubmoduleIgnoreConfiguration(
        linked,
        head,
        "Linked repository",
      )).resolves.toBeUndefined();

      await git(repository, "config", "extensions.worktreeConfig", "true");
      await git(
        linked,
        "config",
        "--worktree",
        "diff.ignoreSubmodules",
        "all",
      );
      await expect(requireNoSubmoduleIgnoreConfiguration(
        linked,
        head,
        "Linked repository",
      )).rejects.toThrow("submodule ignore configuration");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  }, 30_000);

  it("reproduces an exact module graph from a clean isolated checkout", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-reproduction-fixture-"));
    const source = join(root, "source");
    const temporaryParent = join(root, "temporary");
    const buildScript = join(source, "build.mjs");
    try {
      await mkdir(join(source, "src"), { recursive: true });
      await mkdir(temporaryParent, { recursive: true });
      await execFileAsync("git", ["init", source], { windowsHide: true });
      await git(source, "config", "user.name", "tetizz");
      await git(
        source,
        "config",
        "user.email",
        "104690265+tetizz@users.noreply.github.com",
      );
      await git(source, "config", "core.autocrlf", "false");
      await writeFile(
        join(source, "package.json"),
        '{"name":"schema9-reproduction-fixture","type":"module"}\n',
        "utf8",
      );
      await writeFile(
        join(source, "src", "entry.mjs"),
        "export const value = 7;\n",
        "utf8",
      );
      await writeFile(
        buildScript,
        'import { mkdir, readFile, writeFile } from "node:fs/promises";\n'
        + 'import { dirname, join } from "node:path";\n'
        + 'import { fileURLToPath } from "node:url";\n'
        + "const root = dirname(fileURLToPath(import.meta.url));\n"
        + 'await mkdir(join(root, "dist"), { recursive: true });\n'
        + "await writeFile(\n"
        + '  join(root, "dist", "entry.mjs"),\n'
        + '  await readFile(join(root, "src", "entry.mjs")),\n'
        + ");\n",
        "utf8",
      );
      await git(source, "add", "package.json", "src/entry.mjs", "build.mjs");
      await git(source, "commit", "-m", "Reproduction fixture");
      const commit = (await git(source, "rev-parse", "HEAD")).stdout.trim();
      await execFileAsync(process.execPath, [buildScript], {
        cwd: source,
        windowsHide: true,
      });
      const executing = await moduleGraphIdentity(
        "fixture-reproduction",
        pathToFileURL(join(source, "dist", "entry.mjs")).href,
        source,
      );
      let isolatedRepository: string | undefined;
      const reproduced = await withSchema9TemporaryCheckout(
        async (checkout) => {
          isolatedRepository = join(checkout, "repository");
          await cloneRepositoryAtCommit(
            source,
            isolatedRepository,
            commit,
          );
          await execFileAsync(
            process.execPath,
            [join(isolatedRepository, "build.mjs")],
            { cwd: isolatedRepository, windowsHide: true },
          );
          return moduleGraphIdentity(
            "fixture-reproduction",
            pathToFileURL(join(
              isolatedRepository,
              "dist",
              "entry.mjs",
            )).href,
            isolatedRepository,
          );
        },
        temporaryParent,
      );
      expect(reproduced).toEqual(executing);
      if (isolatedRepository === undefined) {
        throw new Error("Isolated repository was not created.");
      }
      await expect(access(isolatedRepository))
        .rejects.toMatchObject({ code: "ENOENT" });
      await expect(readdir(temporaryParent)).resolves.toEqual([]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  }, 60_000);

  it("removes an isolated checkout when verification fails", async () => {
    const parent = await mkdtemp(join(tmpdir(), "schema9-cleanup-parent-"));
    let checkout: string | undefined;
    try {
      await expect(withSchema9TemporaryCheckout(async (path) => {
        checkout = path;
        await writeFile(join(path, "sentinel"), "temporary\n", "utf8");
        throw new Error("injected build failure");
      }, parent)).rejects.toThrow("injected build failure");
      if (checkout === undefined) {
        throw new Error("Temporary checkout was not created.");
      }
      await expect(access(checkout)).rejects.toMatchObject({ code: "ENOENT" });
      await expect(readdir(parent)).resolves.toEqual([]);
    } finally {
      await rm(parent, { recursive: true, force: true });
    }
  });

  it("binds reachable bare-package dependencies into the execution graph", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-module-graph-"));
    const dependency = join(root, "node_modules", "fixture-dependency");
    const entry = join(root, "entry.mjs");
    const nested = join(dependency, "nested.mjs");
    try {
      await mkdir(dependency, { recursive: true });
      await writeFile(
        join(dependency, "package.json"),
        '{"name":"fixture-dependency","type":"module","exports":"./index.mjs"}\n',
        "utf8",
      );
      await writeFile(
        join(dependency, "index.mjs"),
        'import { createRequire } from "node:module";\n'
        + 'const target = "./nested.mjs";\n'
        + 'const requiredTarget = "./runtime.cjs";\n'
        + "const imported = await import(target);\n"
        + "const required = createRequire(import.meta.url)(requiredTarget);\n"
        + "export const value = imported.value + required;\n",
        "utf8",
      );
      await writeFile(nested, 'export const value = "first";\n', "utf8");
      await writeFile(
        join(dependency, "runtime.cjs"),
        'module.exports = "-runtime";\n',
        "utf8",
      );
      await writeFile(
        entry,
        'import { value } from "fixture-dependency"; export default value;\n',
        "utf8",
      );
      const first = await moduleGraphIdentity(
        "fixture-graph",
        pathToFileURL(entry).href,
        root,
      );
      expect(first.files).toBe(5);
      await writeFile(nested, 'export const value = "second";\n', "utf8");
      const second = await moduleGraphIdentity(
        "fixture-graph",
        pathToFileURL(entry).href,
        root,
      );
      expect(second.files).toBe(5);
      expect(second.sha256).not.toBe(first.sha256);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("normalizes package manifest line endings without masking changes", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-manifest-eol-"));
    const lf = join(root, "lf");
    const crlf = join(root, "crlf");
    try {
      await mkdir(lf, { recursive: true });
      await mkdir(crlf, { recursive: true });
      const manifest = '{\n  "name": "manifest-fixture",\n  "type": "module"\n}\n';
      await writeFile(join(lf, "package.json"), manifest, "utf8");
      await writeFile(
        join(crlf, "package.json"),
        manifest.replaceAll("\n", "\r\n"),
        "utf8",
      );
      await writeFile(join(lf, "entry.mjs"), "export const value = 1;\n", "utf8");
      await writeFile(join(crlf, "entry.mjs"), "export const value = 1;\n", "utf8");
      const lfIdentity = await moduleGraphIdentity(
        "manifest-fixture",
        pathToFileURL(join(lf, "entry.mjs")).href,
        lf,
      );
      const crlfIdentity = await moduleGraphIdentity(
        "manifest-fixture",
        pathToFileURL(join(crlf, "entry.mjs")).href,
        crlf,
      );
      expect(crlfIdentity).toEqual(lfIdentity);

      await writeFile(
        join(crlf, "package.json"),
        manifest.replace("manifest-fixture", "changed-fixture")
          .replaceAll("\n", "\r\n"),
        "utf8",
      );
      const changed = await moduleGraphIdentity(
        "manifest-fixture",
        pathToFileURL(join(crlf, "entry.mjs")).href,
        crlf,
      );
      expect(changed.sha256).not.toBe(lfIdentity.sha256);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("rejects execution hooks and search-path overrides", () => {
    expect(() => schema9RuntimeIdentity({}, ["--import=hook.mjs"]))
      .toThrow("without runtime hooks");
    expect(() => schema9RuntimeIdentity({ NODE_PATH: "elsewhere" }, []))
      .toThrow("without runtime hooks");
    expect(schema9RuntimeIdentity({}, [])).toMatchObject({
      nodeVersion: process.version,
      platform: process.platform,
      architecture: process.arch,
      execArgv: [],
    });
  });

  it("rejects deferred file and non-file module loads", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-load-guard-"));
    const entry = join(root, "deferred-entry.mjs");
    try {
      await writeFile(
        entry,
        'export const loadFile = () => import("./deferred-target.mjs");\n'
        + "export const loadData = () => "
        + 'import("data:text/javascript,export default 1");\n',
        "utf8",
      );
      await writeFile(
        join(root, "deferred-target.mjs"),
        "export default 1;\n",
        "utf8",
      );
      const cliUrl = pathToFileURL(resolve(
        "apps/dataset-cli/dist/schema9-ledger-cli.js",
      )).href;
      const entryUrl = pathToFileURL(entry).href;
      const { stdout } = await execFileAsync(
        process.execPath,
        [
          "--input-type=module",
          "--eval",
          `
import { createSchema9ModuleLoadGuard } from ${JSON.stringify(cliUrl)};
import { fileURLToPath } from "node:url";
const entryUrl = ${JSON.stringify(entryUrl)};
const guard = createSchema9ModuleLoadGuard([fileURLToPath(entryUrl)]);
const messages = [];
try {
  const loaded = await import(entryUrl + "?guard=deferred");
  for (const load of [loaded.loadFile, loaded.loadData]) {
    try {
      await load();
      messages.push("resolved");
    } catch (error) {
      messages.push(error instanceof Error ? error.message : String(error));
    }
  }
} finally {
  guard.deregister();
}
process.stdout.write(JSON.stringify(messages));
`,
        ],
        { windowsHide: true },
      );
      expect(JSON.parse(stdout)).toEqual([
        "Schema-9 verification rejected a deferred runtime module load.",
        "Schema-9 verification rejected a non-file runtime module.",
      ]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
