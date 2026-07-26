import type { PgnGuess } from "./pgn-analysis.js";

export const DEFAULT_NEURAL_EVIDENCE_WEIGHT = 0.35;

export interface HybridFusionResult {
  readonly guesses: readonly PgnGuess[];
  readonly neuralCoveredDrawbackCount: number;
}

/**
 * Reweight the exact symbolic posterior with a bounded neural likelihood.
 *
 * The symbolic engine remains authoritative: eliminated hypotheses stay at
 * exactly zero and a neural vocabulary omission is neutral. Dividing by the
 * uniform neural prior makes an uninformative model a no-op.
 */
export function fuseSymbolicAndNeural(
  symbolic: readonly PgnGuess[],
  neural: Readonly<Record<string, number>>,
  evidenceWeight = DEFAULT_NEURAL_EVIDENCE_WEIGHT,
): HybridFusionResult {
  if (
    !Number.isFinite(evidenceWeight) ||
    evidenceWeight < 0 ||
    evidenceWeight > 1
  ) {
    throw new RangeError("Neural evidence weight must be between zero and one.");
  }
  const covered = symbolic.filter(
    (guess) =>
      !guess.eliminated &&
      Object.hasOwn(neural, guess.id),
  );
  if (covered.length === 0 || evidenceWeight === 0) {
    return {
      guesses: symbolic,
      neuralCoveredDrawbackCount: covered.length,
    };
  }
  const neuralEntries = Object.entries(neural);
  const neuralTotal = neuralEntries.reduce((total, [id, probability]) => {
    if (!Number.isFinite(probability) || probability < 0) {
      throw new TypeError(`Invalid neural probability for ${id}.`);
    }
    return total + probability;
  }, 0);
  if (!Number.isFinite(neuralTotal) || Math.abs(neuralTotal - 1) > 1e-6) {
    throw new TypeError("Neural probabilities must sum to one.");
  }
  const vocabularySize = neuralEntries.length;
  const weighted = symbolic.map((guess) => {
    if (guess.eliminated) {
      return { guess, mass: 0 };
    }
    const probability = neural[guess.id];
    if (probability === undefined) {
      return { guess, mass: guess.confidence };
    }
    const likelihoodRatio = Math.max(probability * vocabularySize, 1e-12);
    return {
      guess,
      mass: guess.confidence * likelihoodRatio ** evidenceWeight,
    };
  });
  const total = weighted.reduce((sum, item) => sum + item.mass, 0);
  if (!Number.isFinite(total) || total <= 0) {
    throw new Error("Hybrid posterior normalization failed.");
  }
  return {
    guesses: Object.freeze(
      weighted
        .map(({ guess, mass }) => Object.freeze({
          ...guess,
          confidence: mass / total,
        }))
        .sort(
          (left, right) =>
            Number(left.eliminated) - Number(right.eliminated) ||
            right.confidence - left.confidence ||
            left.id.localeCompare(right.id),
        ),
    ),
    neuralCoveredDrawbackCount: covered.length,
  };
}
