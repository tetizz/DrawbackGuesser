import {
  advancePublicPositionAuthority,
  createStandardChessPositionSnapshot,
  publicAuthorityLegalMoves,
  validatePublicPositionAuthoritySnapshot,
  type PublicPositionAuthoritySnapshot,
} from "@drawbackengine/chess-core";
import type {
  ChessMove,
  ExternalTurnConstraint,
  PositionAuthorityId,
  PositionView,
} from "@drawbackengine/drawback-engine";
import type { PlayerColor } from "@drawbackengine/shared";
import type {
  MoveLikelihoodSignals,
  MoveObservation,
} from "./types.js";

export interface PublicMoveObservationInput {
  readonly authorityId: PositionAuthorityId;
  /**
   * Required for capturable-king/v1 because FEN cannot encode the public
   * one-reply castling king-passant right.
   */
  readonly authorityPositionBefore?: PublicPositionAuthoritySnapshot;
  readonly color: PlayerColor;
  readonly positionBefore: PositionView;
  readonly positionAfter: PositionView;
  readonly move: ChessMove;
  readonly externalConstraint?: ExternalTurnConstraint;
  readonly likelihoodSignals?: Readonly<MoveLikelihoodSignals>;
}

/**
 * Constructs and verifies the predictor's deliberately public observation.
 *
 * The complete legal set is regenerated from a public position-authority
 * snapshot. Callers cannot omit alternatives to weaken hard elimination, and
 * move metadata is checked against the authority's canonical move.
 */
export function createPublicMoveObservation(
  input: PublicMoveObservationInput,
): MoveObservation {
  const authorityPositionBefore = resolveAuthorityPosition(input);
  const authorityLegalMoves = publicAuthorityLegalMoves(
    authorityPositionBefore,
  );
  const canonicalMove = authorityLegalMoves.find((candidate) =>
    sameMove(candidate, input.move)
  );
  if (canonicalMove === undefined) {
    throw new RangeError(
      "Observed move is outside the complete authority move set.",
    );
  }
  if (!sameMoveDetails(canonicalMove, input.move)) {
    throw new TypeError(
      "Observed move metadata does not match the position authority.",
    );
  }
  const transition = advancePublicPositionAuthority(
    authorityPositionBefore,
    input.move,
  );
  if (transition.position.fen !== input.positionAfter.fen) {
    throw new TypeError(
      "After-position FEN does not match the authority transition.",
    );
  }
  return Object.freeze({
    authorityId: input.authorityId,
    authorityPositionBefore,
    color: input.color,
    positionBefore: immutablePosition(input.positionBefore),
    positionAfter: immutablePosition(input.positionAfter),
    move: immutableMove(canonicalMove),
    authorityLegalMoves: Object.freeze(
      authorityLegalMoves.map(immutableMove),
    ),
    ...(input.externalConstraint === undefined
      ? {}
      : { externalConstraint: immutableClone(input.externalConstraint) }),
    ...(input.likelihoodSignals === undefined
      ? {}
      : { likelihoodSignals: immutableClone(input.likelihoodSignals) }),
  });
}

function resolveAuthorityPosition(
  input: PublicMoveObservationInput,
): PublicPositionAuthoritySnapshot {
  const supplied = input.authorityPositionBefore;
  if (input.authorityId === "standard-chess/v1") {
    const snapshot = supplied === undefined
      ? createStandardChessPositionSnapshot(input.positionBefore.fen)
      : validatePublicPositionAuthoritySnapshot(supplied);
    if (
      snapshot.authorityId !== "standard-chess/v1"
      || snapshot.fen !== input.positionBefore.fen
    ) {
      throw new TypeError(
        "Observation authority position does not match positionBefore.",
      );
    }
    return snapshot;
  }
  if (supplied === undefined) {
    throw new TypeError(
      "capturable-king/v1 observations require a complete public authority snapshot.",
    );
  }
  const snapshot = validatePublicPositionAuthoritySnapshot(supplied);
  if (
    snapshot.authorityId !== "capturable-king/v1"
    || snapshot.fen !== input.positionBefore.fen
  ) {
    throw new TypeError(
      "Observation authority position does not match positionBefore.",
    );
  }
  return snapshot;
}

function immutablePosition(position: PositionView): PositionView {
  return Object.freeze({
    fen: position.fen,
    turn: position.turn,
    ply: position.ply,
    history: Object.freeze(position.history.map(immutableMove)),
  });
}

function immutableMove(move: ChessMove): ChessMove {
  return Object.freeze(structuredClone(move));
}

function immutableClone<Value extends object>(value: Value): Readonly<Value> {
  return Object.freeze(structuredClone(value));
}

function sameMove(
  left: Pick<ChessMove, "from" | "to" | "promotion">,
  right: Pick<ChessMove, "from" | "to" | "promotion">,
): boolean {
  return (
    left.from === right.from
    && left.to === right.to
    && left.promotion === right.promotion
  );
}

function sameMoveDetails(left: ChessMove, right: ChessMove): boolean {
  return (
    sameMove(left, right)
    && left.color === right.color
    && left.piece === right.piece
    && left.captured === right.captured
    && left.san === right.san
    && left.flags === right.flags
  );
}
