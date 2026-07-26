import { describe, expect, it } from "vitest";
import { parseFenBoard, rankedHypotheses } from "./model.js";

describe("web model helpers", () => {
  it("parses an ordinary starting board", () => {
    const squares = parseFenBoard(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    );
    expect(squares).toHaveLength(64);
    expect(squares[0]).toMatchObject({ square: "a8", piece: "♜" });
    expect(squares[63]).toMatchObject({ square: "h1", piece: "♖" });
  });

  it("ranks surviving posterior hypotheses ahead of eliminated ones", () => {
    const ranked = rankedHypotheses([
      {
        hypothesisId: "low::{}",
        drawbackId: "low",
        parameters: {},
        internalState: {},
        logProbability: Math.log(0.2),
        eliminated: false,
        evidence: [],
      },
      {
        hypothesisId: "gone::{}",
        drawbackId: "gone",
        parameters: {},
        internalState: {},
        logProbability: Number.NEGATIVE_INFINITY,
        eliminated: true,
        evidence: [],
      },
      {
        hypothesisId: "high::{}",
        drawbackId: "high",
        parameters: {},
        internalState: {},
        logProbability: Math.log(0.8),
        eliminated: false,
        evidence: [],
      },
    ]);
    expect(ranked.map((item) => item.id)).toEqual(["high", "low", "gone"]);
  });

  it("aggregates parameter variants into one rule-level ranking", () => {
    const ranked = rankedHypotheses([
      {
        hypothesisId: 'duck::{"square":"a1"}',
        drawbackId: "duck",
        parameters: { square: "a1" },
        internalState: {},
        logProbability: Math.log(0.25),
        eliminated: false,
        evidence: [],
      },
      {
        hypothesisId: 'duck::{"square":"b2"}',
        drawbackId: "duck",
        parameters: { square: "b2" },
        internalState: {},
        logProbability: Math.log(0.35),
        eliminated: false,
        evidence: [],
      },
      {
        hypothesisId: "other::{}",
        drawbackId: "other",
        parameters: {},
        internalState: {},
        logProbability: Math.log(0.4),
        eliminated: false,
        evidence: [],
      },
    ]);
    expect(ranked).toEqual([
      { id: "duck", confidence: 0.6, eliminated: false },
      { id: "other", confidence: 0.4, eliminated: false },
    ]);
  });

  it("rejects malformed boards", () => {
    expect(() => parseFenBoard("8/8/8")).toThrow("eight ranks");
  });
});
