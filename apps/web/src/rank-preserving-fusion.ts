export const RANK_PRESERVING_FUSION_METHOD =
  "rank-preserving-bounded-residual-plus-symbolic-prior-v1";
export const MAXIMUM_NEURAL_SCALE = 1;
export const ZERO_PRIOR_SEPARATION = 1_000_000;

export interface RankPreservingFusionResult {
  readonly logits: readonly number[];
  readonly probabilities: readonly number[];
  readonly boundedNeuralSignal: readonly number[];
  readonly neuralScales: readonly number[];
  readonly alpha: number;
}

function checkedAlpha(value: number): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new TypeError("Fusion alpha must be between zero and one.");
  }
  return value;
}

function nextUp(value: number): number {
  if (value === Number.POSITIVE_INFINITY) {
    return value;
  }
  if (Object.is(value, -0) || value === 0) {
    return Number.MIN_VALUE;
  }
  const buffer = new ArrayBuffer(8);
  const view = new DataView(buffer);
  view.setFloat64(0, value, false);
  const bits = view.getBigUint64(0, false);
  view.setBigUint64(
    0,
    value > 0 ? bits + 1n : bits - 1n,
    false,
  );
  return view.getFloat64(0, false);
}

function monotonicLogTiers(
  positiveProbabilities: readonly number[],
): Map<number, number> {
  const probabilities = [...new Set(positiveProbabilities)]
    .sort((left, right) => left - right);
  const tiers = new Map<number, number>();
  let previous: number | undefined;
  for (const probability of probabilities) {
    let score = Math.log(probability);
    if (previous !== undefined && score <= previous) {
      score = nextUp(previous);
    }
    if (
      !Number.isFinite(score) ||
      (previous !== undefined && score <= previous)
    ) {
      throw new TypeError(
        "Symbolic prior tiers lack a representable strict gap.",
      );
    }
    tiers.set(probability, score);
    previous = score;
  }
  return tiers;
}

function localNeuralScales(
  bases: ReadonlyMap<number, number>,
): Map<number, number> {
  const ordered = [...bases.entries()]
    .sort((left, right) => left[1] - right[1]);
  const scales = new Map<number, number>();
  ordered.forEach(([probability, base], index) => {
    const previous = ordered[index - 1]?.[1];
    const next = ordered[index + 1]?.[1];
    const headroom = [
      MAXIMUM_NEURAL_SCALE,
      ...(previous === undefined ? [] : [(base - previous) / 4]),
      ...(next === undefined ? [] : [(next - base) / 4]),
    ];
    const scale = Math.min(...headroom);
    if (!Number.isFinite(scale) || scale < 0) {
      throw new TypeError("Symbolic prior tiers have invalid neural headroom.");
    }
    scales.set(probability, scale);
  });
  return scales;
}

function boundedSignal(
  residuals: readonly number[],
  survivors: readonly number[],
): readonly number[] {
  const maximum = Math.max(...survivors.map((index) => residuals[index] ?? 0));
  const weights = survivors.map(
    (index) => Math.exp((residuals[index] ?? 0) - maximum),
  );
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    throw new TypeError(
      "Neural residuals produced an invalid survivor distribution.",
    );
  }
  const uniform = 1 / survivors.length;
  const result = Array<number>(residuals.length).fill(0);
  survivors.forEach((classIndex, survivorIndex) => {
    result[classIndex] = (weights[survivorIndex] ?? 0) / total - uniform;
  });
  return Object.freeze(result);
}

function assertRankPreserved(
  logits: readonly number[],
  prior: readonly number[],
  eliminated: readonly boolean[],
): void {
  const tiers = new Map<number, number[]>();
  eliminated.forEach((isEliminated, index) => {
    if (!isEliminated) {
      const probability = prior[index] ?? 0;
      const scores = tiers.get(probability) ?? [];
      scores.push(logits[index] ?? 0);
      tiers.set(probability, scores);
    }
  });
  let previousMaximum: number | undefined;
  for (
    const [, scores] of [...tiers.entries()]
      .sort((left, right) => left[0] - right[0])
  ) {
    const minimum = Math.min(...scores);
    if (previousMaximum !== undefined && minimum <= previousMaximum) {
      throw new TypeError(
        "Fusion failed to preserve symbolic survivor ordering.",
      );
    }
    previousMaximum = Math.max(...scores);
  }
}

export function rankPreservingFusion(
  residual: ArrayLike<number>,
  prior: readonly number[],
  eliminated: readonly boolean[],
  alpha: number,
): RankPreservingFusionResult {
  const residuals = Array.from(residual);
  if (
    residuals.length === 0 ||
    prior.length !== residuals.length ||
    eliminated.length !== residuals.length
  ) {
    throw new TypeError(
      "Fusion residual, prior, and hard-mask dimensions must match.",
    );
  }
  if (
    residuals.some((value) => !Number.isFinite(value)) ||
    prior.some((value) => !Number.isFinite(value) || value < 0) ||
    eliminated.some((value) => typeof value !== "boolean")
  ) {
    throw new TypeError("Fusion inputs are invalid.");
  }
  const alphaValue = checkedAlpha(alpha);
  const survivors = eliminated
    .map((value, index) => value ? -1 : index)
    .filter((index) => index >= 0);
  if (survivors.length === 0) {
    throw new TypeError("Symbolic engine eliminated every drawback.");
  }
  const positiveProbabilities = survivors
    .map((index) => prior[index] ?? 0)
    .filter((probability) => probability > 0);
  if (positiveProbabilities.length === 0) {
    throw new TypeError(
      "Surviving symbolic hypotheses must contain positive mass.",
    );
  }

  const bases = monotonicLogTiers(positiveProbabilities);
  const minimumPositiveBase = Math.min(...bases.values());
  if (
    survivors.some((index) => (prior[index] ?? 0) === 0)
  ) {
    bases.set(0, minimumPositiveBase - ZERO_PRIOR_SEPARATION);
  }
  const scales = localNeuralScales(bases);
  const signal = boundedSignal(residuals, survivors);
  const neuralScales = residuals.map((_, index) =>
    eliminated[index] ? 0 : (scales.get(prior[index] ?? 0) ?? 0)
  );
  const logits = residuals.map((_, index) => {
    if (eliminated[index]) {
      return 0;
    }
    const probability = prior[index] ?? 0;
    const base = bases.get(probability);
    if (base === undefined) {
      throw new TypeError("Symbolic prior tier is missing.");
    }
    return base + alphaValue * (neuralScales[index] ?? 0) *
      (signal[index] ?? 0);
  });
  assertRankPreserved(logits, prior, eliminated);
  const maximum = Math.max(
    ...logits.filter((_, index) => !(eliminated[index] ?? false)),
  );
  const masses = logits.map((logit, index) =>
    eliminated[index] ? 0 : Math.exp(logit - maximum)
  );
  const total = masses.reduce((sum, mass) => sum + mass, 0);
  if (!Number.isFinite(total) || total <= 0) {
    throw new TypeError("Fusion produced an invalid probability distribution.");
  }
  return Object.freeze({
    logits: Object.freeze(logits),
    probabilities: Object.freeze(masses.map((mass) => mass / total)),
    boundedNeuralSignal: signal,
    neuralScales: Object.freeze(neuralScales),
    alpha: alphaValue,
  });
}
