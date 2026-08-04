import {
  access,
  mkdtemp,
  mkdir,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import process from "node:process";
import { afterEach, describe, expect, it } from "vitest";
import { runIsolatedPython } from "./isolated-python.mjs";

const cleanupRoots = [];

afterEach(async () => {
  await Promise.all(cleanupRoots.splice(0).map(async (root) => rm(root, {
    force: true,
    recursive: true,
  })));
});

describe("isolated Python inspection", () => {
  it("ignores a hostile caller PYTHONPATH", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-python-path-"));
    cleanupRoots.push(root);
    const trusted = join(root, "trusted");
    const hostile = join(root, "hostile");
    const hostileAppData = join(hostile, "AppData", "Roaming");
    const siteMarker = join(root, "sitecustomize-ran");
    const userMarker = join(root, "usercustomize-ran");
    await mkdir(trusted);
    await mkdir(hostile);
    await mkdir(hostileAppData, { recursive: true });
    await writeFile(join(trusted, "schema9_probe.py"), "VALUE = 'trusted'\n");
    await writeFile(join(hostile, "schema9_probe.py"), "VALUE = 'hostile'\n");
    await writeFile(join(hostile, "schema9_hostile_only.py"), "VALUE = 'hostile'\n");
    await writeFile(
      join(hostile, "sitecustomize.py"),
      `from pathlib import Path\nPath(${JSON.stringify(siteMarker)}).write_text('ran')\n`,
    );
    await writeFile(
      join(hostile, "usercustomize.py"),
      `from pathlib import Path\nPath(${JSON.stringify(userMarker)}).write_text('ran')\n`,
    );
    const hostileEnvironment = {
      APPDATA: hostileAppData,
      PYTHONDONTWRITEBYTECODE: "0",
      PYTHONHOME: hostile,
      PYTHONPATH: hostile,
      PYTHONUSERBASE: hostile,
    };
    const previous = Object.fromEntries(Object.keys(hostileEnvironment).map(
      (name) => [name, process.env[name]],
    ));
    try {
      Object.assign(process.env, hostileEnvironment);
      await expect(runIsolatedPython(
        [
          "import importlib.util",
          "from pathlib import Path",
          "import schema9_probe",
          "import sys",
          "assert importlib.util.find_spec('schema9_hostile_only') is None",
          `_schema9_hostile_root = Path(${JSON.stringify(hostile)}).resolve()`,
          "assert not any(Path(path).resolve().is_relative_to(_schema9_hostile_root) for path in sys.path if path)",
          "assert 'sitecustomize' not in sys.modules",
          "assert 'usercustomize' not in sys.modules",
          "print(schema9_probe.VALUE)",
        ].join("\n"),
        [trusted],
        root,
      )).resolves.toMatch(/^trusted\r?\n$/u);
      await expect(access(siteMarker)).rejects.toThrow();
      await expect(access(userMarker)).rejects.toThrow();
    } finally {
      for (const [name, value] of Object.entries(previous)) {
        if (value === undefined) {
          Reflect.deleteProperty(process.env, name);
        } else {
          process.env[name] = value;
        }
      }
    }
  });

  it("applies the closed deterministic Python startup contract", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-python-policy-"));
    cleanupRoots.push(root);
    const command = [
      "import json",
      "import os",
      "import sys",
      "print(json.dumps({",
      "  'hash': hash('schema9-closed-python-v1'),",
      "  'pythonControls': {name: value for name, value in os.environ.items() if name.upper().startswith('PYTHON')},",
      "  'flags': {",
      "    'isolated': sys.flags.isolated,",
      "    'ignoreEnvironment': sys.flags.ignore_environment,",
      "    'noSite': sys.flags.no_site,",
      "    'noUserSite': sys.flags.no_user_site,",
      "    'safePath': sys.flags.safe_path,",
      "    'dontWriteBytecode': sys.flags.dont_write_bytecode,",
      "    'hashRandomization': sys.flags.hash_randomization,",
      "  },",
      "  'customizations': [name for name in ('sitecustomize', 'usercustomize') if name in sys.modules],",
      "}, sort_keys=True))",
    ].join("\n");

    const first = await runIsolatedPython(command, [], root);
    const second = await runIsolatedPython(command, [], root);
    expect(second).toBe(first);
    const observed = JSON.parse(first);
    expect(observed.pythonControls).toEqual({ PYTHONHASHSEED: "0" });
    expect(observed.flags).toEqual({
      dontWriteBytecode: 1,
      hashRandomization: 0,
      ignoreEnvironment: 0,
      isolated: 0,
      noSite: 1,
      noUserSite: 1,
      safePath: true,
    });
    expect(observed.customizations).toEqual([]);
    expect((await readdir(root, { recursive: true })).some(
      (entry) => entry.split(/[\\/]/u).includes("__pycache__"),
    )).toBe(false);
  });

  it("executes the resolved interpreter instead of a launcher", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-python-identity-"));
    cleanupRoots.push(root);

    const executable = (await runIsolatedPython(
      "import sys; print(sys.executable)",
      [],
      root,
    )).trim();
    expect(executable).not.toBe("");
    if (process.platform === "win32") {
      expect(basename(executable).toLocaleLowerCase("en-US")).not.toBe("py.exe");
    }
  });

  it("does not expose ambient package directories", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-python-packages-"));
    cleanupRoots.push(root);

    await expect(runIsolatedPython(
      [
        "from pathlib import Path",
        "import sys",
        "assert not any({'site-packages', 'dist-packages'} & {part.lower() for part in Path(path).parts} for path in sys.path if path)",
        "assert 'sitecustomize' not in sys.modules",
        "assert 'usercustomize' not in sys.modules",
        "print('isolated')",
      ].join("\n"),
      [],
      root,
    )).resolves.toMatch(/^isolated\r?\n$/u);
  });

  it("terminates a Python inspection that exceeds its deadline", async () => {
    const root = await mkdtemp(join(tmpdir(), "schema9-python-timeout-"));
    cleanupRoots.push(root);

    await expect(runIsolatedPython(
      "while True: pass",
      [],
      root,
      250,
    )).rejects.toThrow("exceeded its time limit");
  });
});
