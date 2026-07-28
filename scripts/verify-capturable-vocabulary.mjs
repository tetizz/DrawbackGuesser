import console from "node:console";
import { delimiter, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import process from "node:process";
import {
  AUDITED_CAPTURABLE_KING_RULE_IDS,
} from "../engine/packages/drawback-engine/dist/index.js";
import {
  CAPTURABLE_LEGACY_SYMBOLIC_FEATURE_VERSION,
  CAPTURABLE_OPPORTUNITY_FEATURE_VERSION,
  CAPTURABLE_OPPORTUNITY_SYMBOLIC_FEATURE_VERSION,
  CAPTURABLE_RULE_OPPORTUNITY_FEATURE_WIDTH,
} from "../packages/dataset-contract/dist/index.js";
import {
  RULE_OPPORTUNITY_FEATURE_FIELDS,
  RULE_OPPORTUNITY_FEATURE_VERSION,
  RULE_OPPORTUNITY_FEATURE_WIDTH,
} from "../packages/predictor/dist/index.js";
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
  "  CAPTURABLE_OPPORTUNITY_FEATURE_VERSION,",
  "  CAPTURABLE_OPPORTUNITY_FIELDS,",
  "  CAPTURABLE_OPPORTUNITY_SHAPE,",
  "  CAPTURABLE_OPPORTUNITY_SYMBOLIC_FEATURE_VERSION,",
  "  CAPTURABLE_RULE_IDS,",
  "  CAPTURABLE_SYMBOLIC_FEATURE_VERSION,",
  ")",
  "print(json.dumps({",
  "  'ruleIds': CAPTURABLE_RULE_IDS,",
  "  'legacyFeatureVersion': CAPTURABLE_SYMBOLIC_FEATURE_VERSION,",
  "  'opportunitySymbolicFeatureVersion': CAPTURABLE_OPPORTUNITY_SYMBOLIC_FEATURE_VERSION,",
  "  'opportunityFeatureVersion': CAPTURABLE_OPPORTUNITY_FEATURE_VERSION,",
  "  'opportunityFields': CAPTURABLE_OPPORTUNITY_FIELDS,",
  "  'opportunityShape': CAPTURABLE_OPPORTUNITY_SHAPE,",
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
const opportunityFields = [...RULE_OPPORTUNITY_FEATURE_FIELDS];
if (
  python.legacyFeatureVersion
    !== CAPTURABLE_LEGACY_SYMBOLIC_FEATURE_VERSION
  || python.opportunitySymbolicFeatureVersion
    !== CAPTURABLE_OPPORTUNITY_SYMBOLIC_FEATURE_VERSION
  || python.opportunitySymbolicFeatureVersion
    !== CAPTURABLE_SYMBOLIC_FEATURE_VERSION
  || python.opportunityFeatureVersion
    !== CAPTURABLE_OPPORTUNITY_FEATURE_VERSION
  || python.opportunityFeatureVersion
    !== RULE_OPPORTUNITY_FEATURE_VERSION
  || JSON.stringify(python.opportunityFields)
    !== JSON.stringify(opportunityFields)
  || JSON.stringify(python.opportunityShape)
    !== JSON.stringify([
      engineRuleIds.length,
      RULE_OPPORTUNITY_FEATURE_WIDTH,
    ])
  || RULE_OPPORTUNITY_FEATURE_WIDTH
    !== CAPTURABLE_RULE_OPPORTUNITY_FEATURE_WIDTH
  || CAPTURABLE_SYMBOLIC_RULE_COUNT !== engineRuleIds.length
) {
  throw new Error(
    "Capturable symbolic opportunity contract is inconsistent.",
  );
}
console.log(
  `Verified ${String(engineRuleIds.length)} capturable labels, legacy schema ${String(CAPTURABLE_LEGACY_SYMBOLIC_FEATURE_VERSION)}, and opportunity schema ${String(CAPTURABLE_SYMBOLIC_FEATURE_VERSION)} across Engine, TypeScript, and Python.`,
);
