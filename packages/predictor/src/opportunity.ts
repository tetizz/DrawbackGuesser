import { logSumExp } from "./math.js";
import type {
  DrawbackHypothesis,
  HypothesisDistribution,
  HypothesisMoveOpportunity,
  PredictionOpportunitySnapshot,
  PredictionState,
  RuleOpportunityFeatureField,
} from "./types.js";

export const RULE_OPPORTUNITY_FEATURE_VERSION = 1 as const;
export const RULE_OPPORTUNITY_FEATURE_FIELDS = Object.freeze([
  "knownMass",
  "allowedMoveFractionMass",
  "triggeredMass",
  "forcedMass",
] as const satisfies readonly RuleOpportunityFeatureField[]);
export const RULE_OPPORTUNITY_FEATURE_WIDTH =
  RULE_OPPORTUNITY_FEATURE_FIELDS.length;

const UNIT_INTERVAL_TOLERANCE = 1e-12;

function activeDistribution(
  state: PredictionState,
  snapshot: PredictionOpportunitySnapshot,
): HypothesisDistribution {
  return snapshot.color === "white" ? state.white : state.black;
}

function assertOrderedRuleIds(ruleIds: readonly string[]): void {
  if (
    ruleIds.length === 0
    || ruleIds.some((ruleId) => ruleId.length === 0)
    || new Set(ruleIds).size !== ruleIds.length
  ) {
    throw new TypeError(
      "Ordered opportunity rule IDs must be non-empty and unique.",
    );
  }
}

function assertCount(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new TypeError(`${label} must be a non-negative safe integer.`);
  }
}

function assertUnitInterval(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new TypeError(`${label} must be finite and between zero and one.`);
  }
}

function assertAlignedOpportunity(
  hypothesis: DrawbackHypothesis,
  opportunity: HypothesisMoveOpportunity,
  index: number,
  ordinaryLegalMoveCount: number | undefined,
): number {
  if (opportunity.hypothesisIndex !== index) {
    throw new TypeError(
      `Opportunity hypothesis index ${String(index)} is misaligned.`,
    );
  }
  if (opportunity.drawbackId !== hypothesis.drawbackId) {
    throw new TypeError(
      `Opportunity hypothesis ${String(index)} has a mismatched drawback ID.`,
    );
  }
  assertCount(
    opportunity.ordinaryLegalMoveCount,
    `Opportunity hypothesis ${String(index)} ordinary move count`,
  );
  if (
    ordinaryLegalMoveCount !== undefined
    && opportunity.ordinaryLegalMoveCount !== ordinaryLegalMoveCount
  ) {
    throw new TypeError(
      "Opportunity hypotheses disagree on the ordinary legal move count.",
    );
  }
  if (hypothesis.eliminated) {
    if (opportunity.status !== "eliminated") {
      throw new TypeError(
        `Eliminated hypothesis ${String(index)} has live opportunity data.`,
      );
    }
    return opportunity.ordinaryLegalMoveCount;
  }
  if (!Number.isFinite(hypothesis.logProbability)) {
    throw new TypeError(
      `Live hypothesis ${String(index)} must have a finite log probability.`,
    );
  }
  if (opportunity.status === "eliminated") {
    throw new TypeError(
      `Live hypothesis ${String(index)} is marked eliminated in its opportunity.`,
    );
  }
  if (opportunity.status === "known") {
    if (
      typeof opportunity.triggered !== "boolean"
      || typeof opportunity.forced !== "boolean"
      || typeof opportunity.observedMoveLegal !== "boolean"
    ) {
      throw new TypeError(
        `Known opportunity hypothesis ${String(index)} must contain boolean evidence fields.`,
      );
    }
    assertCount(
      opportunity.allowedMoveCount,
      `Opportunity hypothesis ${String(index)} allowed move count`,
    );
    if (opportunity.allowedMoveCount > opportunity.ordinaryLegalMoveCount) {
      throw new TypeError(
        `Opportunity hypothesis ${String(index)} allows too many moves.`,
      );
    }
    assertUnitInterval(
      opportunity.allowedMoveFraction,
      `Opportunity hypothesis ${String(index)} allowed move fraction`,
    );
    const expectedFraction = opportunity.ordinaryLegalMoveCount === 0
      ? 0
      : opportunity.allowedMoveCount / opportunity.ordinaryLegalMoveCount;
    if (
      Math.abs(opportunity.allowedMoveFraction - expectedFraction)
      > UNIT_INTERVAL_TOLERANCE
    ) {
      throw new TypeError(
        `Opportunity hypothesis ${String(index)} has an inconsistent allowed move fraction.`,
      );
    }
  }
  return opportunity.ordinaryLegalMoveCount;
}

function boundedUnit(value: number, label: string): number {
  if (
    !Number.isFinite(value)
    || value < -UNIT_INTERVAL_TOLERANCE
    || value > 1 + UNIT_INTERVAL_TOLERANCE
  ) {
    throw new RangeError(`${label} is outside the unit interval.`);
  }
  return Math.min(1, Math.max(0, value));
}

/**
 * Aggregates pre-observation hypothesis opportunity evidence into a fixed,
 * label-blind vector. Unknown variants remain in each rule's conditional
 * denominator but contribute zero to all four public channels.
 */
export function aggregateRuleOpportunityFeatures(
  stateBefore: PredictionState,
  snapshot: PredictionOpportunitySnapshot,
  orderedRuleIds: readonly string[],
): readonly number[] {
  assertOrderedRuleIds(orderedRuleIds);
  const distribution = activeDistribution(stateBefore, snapshot);
  if (distribution.hypotheses.length !== snapshot.hypotheses.length) {
    throw new TypeError(
      "Opportunity snapshot length does not match the active distribution.",
    );
  }

  const allowedRuleIds = new Set(orderedRuleIds);
  const liveByRule = new Map<
    string,
    {
      readonly hypothesis: DrawbackHypothesis;
      readonly opportunity: HypothesisMoveOpportunity;
    }[]
  >();
  let ordinaryLegalMoveCount: number | undefined;
  distribution.hypotheses.forEach((hypothesis, index) => {
    const opportunity = snapshot.hypotheses[index];
    if (opportunity === undefined) {
      throw new TypeError(`Opportunity hypothesis ${String(index)} is missing.`);
    }
    ordinaryLegalMoveCount = assertAlignedOpportunity(
      hypothesis,
      opportunity,
      index,
      ordinaryLegalMoveCount,
    );
    if (!allowedRuleIds.has(hypothesis.drawbackId)) {
      throw new TypeError(
        `Opportunity contains unordered rule ${hypothesis.drawbackId}.`,
      );
    }
    if (!hypothesis.eliminated) {
      const variants = liveByRule.get(hypothesis.drawbackId) ?? [];
      variants.push({ hypothesis, opportunity });
      liveByRule.set(hypothesis.drawbackId, variants);
    }
  });

  const representedRules = new Set(
    distribution.hypotheses.map((hypothesis) => hypothesis.drawbackId),
  );
  for (const ruleId of orderedRuleIds) {
    if (!representedRules.has(ruleId)) {
      throw new TypeError(`Ordered opportunity rule ${ruleId} is unrepresented.`);
    }
  }

  const features: number[] = [];
  for (const ruleId of orderedRuleIds) {
    const liveVariants = liveByRule.get(ruleId) ?? [];
    if (liveVariants.length === 0) {
      features.push(0, 0, 0, 0);
      continue;
    }
    const normalizer = logSumExp(
      liveVariants.map(({ hypothesis }) => hypothesis.logProbability),
    );
    if (!Number.isFinite(normalizer)) {
      throw new TypeError(
        `Live opportunity rule ${ruleId} has no finite probability mass.`,
      );
    }
    let knownMass = 0;
    let allowedMoveFractionMass = 0;
    let triggeredMass = 0;
    let forcedMass = 0;
    for (const { hypothesis, opportunity } of liveVariants) {
      if (opportunity.status !== "known") {
        continue;
      }
      const weight = Math.exp(hypothesis.logProbability - normalizer);
      knownMass += weight;
      allowedMoveFractionMass += weight * opportunity.allowedMoveFraction;
      triggeredMass += weight * Number(opportunity.triggered);
      forcedMass += weight * Number(opportunity.forced);
    }
    features.push(
      boundedUnit(knownMass, `${ruleId} known mass`),
      boundedUnit(
        allowedMoveFractionMass,
        `${ruleId} allowed move fraction mass`,
      ),
      boundedUnit(triggeredMass, `${ruleId} triggered mass`),
      boundedUnit(forcedMass, `${ruleId} forced mass`),
    );
  }
  return Object.freeze(features);
}
