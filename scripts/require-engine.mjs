import console from "node:console";
import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import process from "node:process";

const expected = Object.freeze([
  ["engine/packages/shared/package.json", "@drawbackengine/shared"],
  [
    "engine/packages/drawback-engine/package.json",
    "@drawbackengine/drawback-engine",
  ],
  ["engine/packages/chess-core/package.json", "@drawbackengine/chess-core"],
  [
    "engine/packages/chess-evaluator/package.json",
    "@drawbackengine/chess-evaluator",
  ],
  [
    "engine/packages/simulation-trace/package.json",
    "@drawbackengine/simulation-trace",
  ],
]);

const root = resolve(import.meta.dirname, "..");
const failures = [];

for (const [relativePath, packageName] of expected) {
  const path = resolve(root, relativePath);
  try {
    await access(path);
    const manifest = JSON.parse(await readFile(path, "utf8"));
    if (manifest.name !== packageName) {
      failures.push(
        `${relativePath} has package name ${String(manifest.name)}; expected ${packageName}`,
      );
    }
  } catch (error) {
    failures.push(
      `${relativePath}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

if (failures.length > 0) {
  console.error(
    [
      "DrawbackEngine workspace dependency is not ready.",
      ...failures.map((failure) => `- ${failure}`),
      "Populate engine/ with the pinned DrawbackEngine submodule; do not copy rule logic here.",
    ].join("\n"),
  );
  process.exitCode = 2;
} else {
  console.log("DrawbackEngine workspace package boundary is present.");
}
