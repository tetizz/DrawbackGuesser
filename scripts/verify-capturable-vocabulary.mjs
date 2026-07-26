import console from "node:console";
import { delimiter, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import process from "node:process";
import {
  AUDITED_CAPTURABLE_KING_RULE_IDS,
} from "../engine/packages/drawback-engine/dist/index.js";
import {
  CAPTURABLE_SYMBOLIC_FEATURE_VERSION,
  CAPTURABLE_SYMBOLIC_RULE_COUNT,
} from "../packages/trace-to-dataset/dist/index.js";

const repositoryRoot = resolve(import.meta.dirname, "..");
const pythonPath = resolve(repositoryRoot, "ml", "training");
const existingPythonPath = process.env["PYTHONPATH"];
const command = [
  "import json",
  "from drawback_ml.capturable_records import (",
  "  CAPTURABLE_RULE_IDS,",
  "  CAPTURABLE_SYMBOLIC_FEATURE_VERSION,",
  ")",
  "print(json.dumps({",
  "  'ruleIds': CAPTURABLE_RULE_IDS,",
  "  'featureVersion': CAPTURABLE_SYMBOLIC_FEATURE_VERSION,",
  "}, separators=(',', ':')))",
].join("\n");
const result = spawnSync("python", ["-c", command], {
  cwd: repositoryRoot,
  encoding: "utf8",
  windowsHide: true,
  env: {
    ...process.env,
    PYTHONPATH:
      existingPythonPath === undefined || existingPythonPath.length === 0
        ? pythonPath
        : `${pythonPath}${delimiter}${existingPythonPath}`,
  },
});
if (result.error !== undefined) {
  throw result.error;
}
if (result.status !== 0) {
  throw new Error(
    `Python capturable vocabulary inspection failed: ${result.stderr.trim()}`,
  );
}

let python;
try {
  python = JSON.parse(result.stdout);
} catch (error) {
  throw new Error("Python capturable vocabulary output is not valid JSON.", {
    cause: error,
  });
}
const engineRuleIds = [...AUDITED_CAPTURABLE_KING_RULE_IDS];
if (
  !Array.isArray(python.ruleIds)
  || JSON.stringify(python.ruleIds) !== JSON.stringify(engineRuleIds)
) {
  throw new Error(
    "Python capturable vocabulary is out of sync with DrawbackEngine.",
  );
}
if (
  python.featureVersion !== CAPTURABLE_SYMBOLIC_FEATURE_VERSION
  || CAPTURABLE_SYMBOLIC_RULE_COUNT !== engineRuleIds.length
) {
  throw new Error(
    "Capturable symbolic feature version or rule count is inconsistent.",
  );
}
console.log(
  `Verified ${String(engineRuleIds.length)} capturable labels at symbolic schema ${String(CAPTURABLE_SYMBOLIC_FEATURE_VERSION)} across Engine, TypeScript, and Python.`,
);
