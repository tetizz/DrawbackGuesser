import {
  advancePublicPositionAuthority,
  createStandardChessPositionSnapshot,
  publicAuthorityLegalMoves,
  type StandardChessPositionSnapshot,
} from "@drawbackengine/chess-core";
import type {
  ChessMove,
  ExternalTurnConstraint,
  PositionView,
} from "@drawbackengine/drawback-engine";
import type { MoveLikelihoodSignals, MoveObservation } from "./types.js";
import { createPublicMoveObservation } from "./observation.js";

const INITIAL_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export class PredictorTestGame {
  #position: StandardChessPositionSnapshot;
  #history: readonly ChessMove[] = [];

  public constructor(fen = INITIAL_FEN) {
    this.#position = createStandardChessPositionSnapshot(fen);
  }

  public view(): PositionView {
    return Object.freeze({
      fen: this.#position.fen,
      turn: fenTurn(this.#position.fen),
      ply: this.#history.length,
      history: Object.freeze(structuredClone(this.#history)),
    });
  }

  public legalMoves(): readonly ChessMove[] {
    return publicAuthorityLegalMoves(this.#position);
  }

  public play(
    from: string,
    to: string,
    options: {
      readonly promotion?: "knight" | "bishop" | "rook" | "queen";
      readonly externalConstraint?: ExternalTurnConstraint;
      readonly likelihoodSignals?: Readonly<MoveLikelihoodSignals>;
    } = {},
  ): MoveObservation {
    const before = this.view();
    const authorityBefore = this.#position;
    const canonical = this.legalMoves().find((move) =>
      move.from === from
      && move.to === to
      && move.promotion === options.promotion
    );
    if (canonical === undefined) {
      throw new RangeError(`Test move ${from}${to} is not legal.`);
    }
    const transition = advancePublicPositionAuthority(authorityBefore, {
      from,
      to,
      ...(options.promotion === undefined
        ? {}
        : { promotion: options.promotion }),
    });
    this.#position =
      transition.position as StandardChessPositionSnapshot;
    this.#history = Object.freeze([...this.#history, transition.move]);
    return createPublicMoveObservation({
      authorityId: "standard-chess/v1",
      authorityPositionBefore: authorityBefore,
      color: canonical.color,
      positionBefore: before,
      positionAfter: this.view(),
      move: canonical,
      ...(options.externalConstraint === undefined
        ? {}
        : { externalConstraint: options.externalConstraint }),
      ...(options.likelihoodSignals === undefined
        ? {}
        : { likelihoodSignals: options.likelihoodSignals }),
    });
  }
}

function fenTurn(fen: string): "white" | "black" {
  return fen.split(/\s+/u)[1] === "w" ? "white" : "black";
}
