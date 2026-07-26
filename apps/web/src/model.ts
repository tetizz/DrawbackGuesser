import {
  aggregateRulePosteriors,
  type DrawbackHypothesis,
} from "@drawbackguesser/predictor";

export interface BoardSquare {
  readonly square: string;
  readonly piece: string | null;
  readonly dark: boolean;
}

const FILES = "abcdefgh";
const PIECES: Readonly<Record<string, string>> = {
  K: "♔",
  Q: "♕",
  R: "♖",
  B: "♗",
  N: "♘",
  P: "♙",
  k: "♚",
  q: "♛",
  r: "♜",
  b: "♝",
  n: "♞",
  p: "♟",
};

export function parseFenBoard(fen: string): readonly BoardSquare[] {
  const board = fen.split(" ")[0];
  if (board === undefined) {
    throw new Error("FEN must contain a board field.");
  }
  const ranks = board.split("/");
  if (ranks.length !== 8) {
    throw new Error("FEN board must contain eight ranks.");
  }

  return ranks.flatMap((rank, rankIndex) => {
    const expanded: (string | null)[] = [];
    for (const token of rank) {
      const empty = Number(token);
      if (Number.isInteger(empty) && empty >= 1 && empty <= 8) {
        expanded.push(...Array.from<null>({ length: empty }).fill(null));
      } else {
        const piece = PIECES[token];
        if (piece === undefined) {
          throw new Error(`Unsupported FEN piece token: ${token}`);
        }
        expanded.push(piece);
      }
    }
    if (expanded.length !== 8) {
      throw new Error("Every FEN rank must contain eight squares.");
    }
    return expanded.map((piece, fileIndex) => ({
      square: `${FILES[fileIndex] ?? ""}${String(8 - rankIndex)}`,
      piece,
      dark: (rankIndex + fileIndex) % 2 === 1,
    }));
  });
}

export function rankedHypotheses(
  hypotheses: readonly DrawbackHypothesis[],
): readonly { readonly id: string; readonly confidence: number; readonly eliminated: boolean }[] {
  return aggregateRulePosteriors({ hypotheses }).map((posterior) => ({
    id: posterior.drawbackId,
    confidence: posterior.probability,
    eliminated: posterior.eliminated,
  }));
}
