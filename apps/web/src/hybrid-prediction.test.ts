import { describe, expect, it } from "vitest";
import { fuseSymbolicAndNeural } from "./hybrid-prediction.js";
import type { PgnGuess } from "./pgn-analysis.js";

const guesses: readonly PgnGuess[] = [
  { id: "vegan", confidence: 0.5, eliminated: false, parameters: [] },
  { id: "checkers", confidence: 0.3, eliminated: false, parameters: [] },
  { id: "truant", confidence: 0.2, eliminated: false, parameters: [] },
  { id: "pacman", confidence: 0, eliminated: true, parameters: [] },
];

describe("hybrid drawback fusion", () => {
  it("lets neural evidence rank covered live hypotheses", () => {
    const result = fuseSymbolicAndNeural(
      guesses,
      { vegan: 0.05, checkers: 0.95 },
      1,
    );
    expect(result.guesses[0]?.id).toBe("checkers");
    expect(result.neuralCoveredDrawbackCount).toBe(2);
    expect(
      result.guesses.reduce((sum, guess) => sum + guess.confidence, 0),
    ).toBeCloseTo(1);
  });

  it("never restores a hard-eliminated hypothesis", () => {
    const result = fuseSymbolicAndNeural(
      guesses,
      { vegan: 0.0001, checkers: 0.0001, pacman: 0.9998 },
    );
    expect(result.guesses.find(({ id }) => id === "pacman")).toMatchObject({
      confidence: 0,
      eliminated: true,
    });
  });

  it("does not redistribute eliminated neural mass into covered survivors", () => {
    const result = fuseSymbolicAndNeural(
      guesses,
      { vegan: 0.1, checkers: 0.1, pacman: 0.8 },
      1,
    );
    const byId = new Map(result.guesses.map((guess) => [guess.id, guess]));
    expect(byId.get("vegan")?.confidence).toBeCloseTo(0.15 / 0.44);
    expect(byId.get("checkers")?.confidence).toBeCloseTo(0.09 / 0.44);
    expect(byId.get("truant")?.confidence).toBeCloseTo(0.2 / 0.44);
    expect(byId.get("pacman")?.confidence).toBe(0);
  });

  it("is neutral for uniform evidence and vocabulary omissions", () => {
    const result = fuseSymbolicAndNeural(
      guesses,
      { vegan: 0.5, checkers: 0.5 },
      1,
    );
    expect(result.guesses.find(({ id }) => id === "vegan")?.confidence).toBeCloseTo(
      0.5,
    );
    expect(result.guesses.find(({ id }) => id === "truant")?.confidence).toBeCloseTo(
      0.2,
    );
  });

  it("rejects malformed neural probabilities", () => {
    expect(() =>
      fuseSymbolicAndNeural(guesses, { vegan: Number.NaN }),
    ).toThrow("Invalid neural probability");
  });
});
