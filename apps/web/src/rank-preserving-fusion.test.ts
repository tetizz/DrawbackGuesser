import { describe, expect, it } from "vitest";
import parityFixtureText from
  "./fixtures/rank-preserving-fusion-v1.json?raw";
import {
  RANK_PRESERVING_FUSION_METHOD,
  rankPreservingFusion,
} from "./rank-preserving-fusion.js";

describe("rank-preserving fusion", () => {
  it("matches the shared Python parity vectors", () => {
    const fixture = JSON.parse(parityFixtureText) as {
      method: string;
      cases: Array<{
        name: string;
        residuals: number[];
        prior: number[];
        eliminated: boolean[];
        alpha: number;
        expected: {
          logits: number[];
          probabilities: number[];
          boundedNeuralSignal: number[];
          neuralScales: number[];
        };
      }>;
    };
    expect(fixture.method).toBe(RANK_PRESERVING_FUSION_METHOD);
    for (const testCase of fixture.cases) {
      const result = rankPreservingFusion(
        testCase.residuals,
        testCase.prior,
        testCase.eliminated,
        testCase.alpha,
      );
      for (const key of [
        "logits",
        "probabilities",
        "boundedNeuralSignal",
        "neuralScales",
      ] as const) {
        result[key].forEach((actual, index) => {
          expect(
            actual,
            `${testCase.name}.${key}[${String(index)}]`,
          ).toBeCloseTo(testCase.expected[key][index] ?? Number.NaN, 13);
        });
      }
    }
  });

  it("cannot reverse strict symbolic order or restore eliminations", () => {
    const result = rankPreservingFusion(
      [-1000, 1000, Number.MAX_VALUE],
      [0.9, 0.1, 0],
      [false, false, true],
      1,
    );
    expect(result.logits[0]).toBeGreaterThan(result.logits[1] ?? 0);
    expect(result.probabilities[0]).toBeGreaterThan(
      result.probabilities[1] ?? 0,
    );
    expect(result.probabilities[2]).toBe(0);
    expect(result.boundedNeuralSignal[2]).toBe(0);
  });

  it("discards stale symbolic mass behind a hard mask", () => {
    const result = rankPreservingFusion(
      [0, 1000],
      [0.6, 0.4],
      [false, true],
      1,
    );
    expect(result.probabilities).toEqual([1, 0]);
  });

  it("uses neural evidence inside ties without global near-tie suppression", () => {
    const result = rankPreservingFusion(
      [-1000, 1000, 1000, -1000],
      [0.4, 0.4, 0.100000000000001, 0.1],
      [false, false, false, false],
      1,
    );
    expect(result.neuralScales[0]).toBeGreaterThan(0.1);
    expect(result.neuralScales[0]).toBe(result.neuralScales[1]);
    expect(result.probabilities[1]).toBeGreaterThan(
      result.probabilities[0] ?? 0,
    );
    expect(result.logits[0]).toBeGreaterThan(result.logits[2] ?? 0);
  });

  it("leaves symbolic evidence unchanged for constant residuals", () => {
    const prior = [0.5, 0.3, 0.2];
    const result = rankPreservingFusion(
      [7, 7, 7],
      prior,
      [false, false, false],
      1,
    );
    expect(result.boundedNeuralSignal).toEqual([0, 0, 0]);
    result.probabilities.forEach((probability, index) => {
      expect(probability).toBeCloseTo(prior[index] ?? 0, 14);
    });
  });

  it("projects logarithmically collapsed valid priors monotonically", () => {
    const result = rankPreservingFusion(
      [0, 0, 0],
      [0.9999999998, 9.999999998e-11, 9.999999997999999e-11],
      [false, false, false],
      1,
    );
    expect(result.logits[0]).toBeGreaterThan(result.logits[1] ?? 0);
    expect(result.logits[1]).toBeGreaterThan(result.logits[2] ?? 0);
  });

  it("is shift invariant and validates the exact authority contract", () => {
    const first = rankPreservingFusion(
      [-2, 3, 1],
      [0.4, 0.4, 0.2],
      [false, false, false],
      0.5,
    );
    const shifted = rankPreservingFusion(
      [998, 1003, 1001],
      [0.4, 0.4, 0.2],
      [false, false, false],
      0.5,
    );
    first.probabilities.forEach((probability, index) => {
      expect(probability).toBeCloseTo(shifted.probabilities[index] ?? 0, 14);
    });
    expect(() =>
      rankPreservingFusion([0], [1], [false], Number.NaN)
    ).toThrow("between zero and one");
    expect(() =>
      rankPreservingFusion([0], [1], [true], 1)
    ).toThrow("eliminated every");
    expect(() =>
      rankPreservingFusion([0], [0], [false], 1)
    ).toThrow("positive mass");
  });
});
