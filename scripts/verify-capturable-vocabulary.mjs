import console from "node:console";
import { join, resolve } from "node:path";
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
import { runIsolatedPython } from "./isolated-python.mjs";

const repositoryRoot = resolve(import.meta.dirname, "..");
const pythonPath = resolve(repositoryRoot, "ml", "training");
const drawbackPackagePath = join(pythonPath, "drawback_ml");
const command = [
  "import importlib.machinery",
  "import json",
  "import sys",
  "import types",
  `_drawback_package_path = ${JSON.stringify(drawbackPackagePath)}`,
  "_drawback_package = types.ModuleType('drawback_ml')",
  "_drawback_package.__package__ = 'drawback_ml'",
  "_drawback_package.__path__ = [_drawback_package_path]",
  "_drawback_package.__spec__ = importlib.machinery.ModuleSpec('drawback_ml', loader=None, is_package=True)",
  "sys.modules['drawback_ml'] = _drawback_package",
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
const pythonOutput = await runIsolatedPython(
  command,
  [],
  repositoryRoot,
);

let python;
try {
  python = JSON.parse(pythonOutput);
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
