import type {
  LikelihoodFeatures,
  LikelihoodWeights,
  MoveLikelihoodSignals,
} from "./types.js";

export const DEFAULT_LIKELIHOOD_WEIGHTS: Readonly<LikelihoodWeights> =
  Object.freeze({
    allowedMoveCount: 1,
    forcedMove: 0,
    humanMove: 1,
    engineQuality: 1,
    playerStrength: 1,
    timeUsage: 1,
    noTriggerEvidenceScale: 0.05,
  });

function finiteLogSignal(
  name: keyof MoveLikelihoodSignals,
  value: number | undefined,
): number {
  if (value === undefined) {
    return 0;
  }
  if (!Number.isFinite(value) || value > 0) {
    throw new RangeError(`${name} must be a finite log likelihood at most zero`);
  }
  return value;
}

function finiteWeight(name: keyof LikelihoodWeights, value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${name} must be finite and non-negative`);
  }
  return value;
}

export function resolveLikelihoodWeights(
  overrides: Partial<LikelihoodWeights> = {},
): Readonly<LikelihoodWeights> {
  const resolved = {
    ...DEFAULT_LIKELIHOOD_WEIGHTS,
    ...overrides,
  };
  for (const [name, value] of Object.entries(resolved)) {
    finiteWeight(name as keyof LikelihoodWeights, value);
  }
  if (resolved.noTriggerEvidenceScale > 1) {
    throw new RangeError("noTriggerEvidenceScale must be at most one");
  }
  return Object.freeze(resolved);
}

/**
 * Scores only information observable from the move and a candidate's legal
 * set. All optional signal values are log likelihoods produced without access
 * to the true drawback or its secret state.
 */
export function scoreMoveLogLikelihood(
  features: LikelihoodFeatures,
  weights: Readonly<LikelihoodWeights> = DEFAULT_LIKELIHOOD_WEIGHTS,
): number {
  if (!Number.isSafeInteger(features.allowedMoveCount) || features.allowedMoveCount <= 0) {
    throw new RangeError("allowedMoveCount must be a positive safe integer");
  }
  if (
    !Number.isSafeInteger(features.ordinaryLegalMoveCount) ||
    features.ordinaryLegalMoveCount <= 0
  ) {
    throw new RangeError("ordinaryLegalMoveCount must be a positive safe integer");
  }
  if (features.allowedMoveCount > features.ordinaryLegalMoveCount) {
    throw new RangeError("allowedMoveCount cannot exceed ordinaryLegalMoveCount");
  }

  const checkedWeights = resolveLikelihoodWeights(weights);
  const signals = features.signals ?? {};
  const choiceLikelihood =
    -Math.log(features.allowedMoveCount) * checkedWeights.allowedMoveCount;
  const forcedAdjustment = features.forced ? checkedWeights.forcedMove : 0;
  const observedSignals =
    finiteLogSignal("humanMoveLogLikelihood", signals.humanMoveLogLikelihood) *
      checkedWeights.humanMove +
    finiteLogSignal(
      "engineQualityLogLikelihood",
      signals.engineQualityLogLikelihood,
    ) *
      checkedWeights.engineQuality +
    finiteLogSignal(
      "playerStrengthLogLikelihood",
      signals.playerStrengthLogLikelihood,
    ) *
      checkedWeights.playerStrength +
    finiteLogSignal("timeUsageLogLikelihood", signals.timeUsageLogLikelihood) *
      checkedWeights.timeUsage;
  const signalScale = features.triggered
    ? 1
    : checkedWeights.noTriggerEvidenceScale;
  return choiceLikelihood + forcedAdjustment + observedSignals * signalScale;
}
