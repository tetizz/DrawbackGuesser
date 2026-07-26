import type { HypothesisDistribution } from "./types.js";

export function logSumExp(values: readonly number[]): number {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) {
    return Number.NEGATIVE_INFINITY;
  }
  const maximum = Math.max(...finite);
  return maximum + Math.log(finite.reduce((sum, value) => sum + Math.exp(value - maximum), 0));
}

export function normalizeLogProbabilities(
  values: readonly number[],
): readonly number[] {
  const normalizer = logSumExp(values);
  if (!Number.isFinite(normalizer)) {
    return values.map(() => Number.NEGATIVE_INFINITY);
  }
  return values.map((value) =>
    Number.isFinite(value) ? value - normalizer : Number.NEGATIVE_INFINITY,
  );
}

export function probability(logProbability: number): number {
  return Number.isFinite(logProbability) ? Math.exp(logProbability) : 0;
}

export function entropy(distribution: HypothesisDistribution): number {
  return distribution.hypotheses.reduce((total, hypothesis) => {
    const p = probability(hypothesis.logProbability);
    return p === 0 ? total : total - p * Math.log2(p);
  }, 0);
}
