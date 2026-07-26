import {
  advancePublicPositionAuthority,
  createStandardChessPositionSnapshot,
  publicAuthorityLegalMoves,
  type PublicPositionAuthoritySnapshot,
} from "@drawbackengine/chess-core";
import {
  canonicalMoveUci,
  createEvaluatorTurnConstraintRequest,
  type ChessMove,
  type PositionView,
} from "@drawbackengine/drawback-engine";
import {
  parsePrivateSimulationTraceRecord,
  simulationGameId,
  type PrivateSimulationTraceRecord,
  type PrivateSimulationTracePly,
} from "@drawbackengine/simulation-trace";
import type { PlayerColor } from "@drawbackengine/shared";

const STANDARD_INITIAL_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export interface TraceFixtureOptions {
  readonly seed?: number;
  readonly gameIndex?: number;
  readonly initialFen?: string;
  readonly moves?: readonly string[];
  readonly drawbackId?: string;
  readonly evaluatorCoverage?: "none" | "uniform";
  readonly evaluatorFingerprint?: string;
}

function turnFromFen(fen: string): PlayerColor {
  return fen.split(" ")[1] === "w" ? "white" : "black";
}

function commandFromUci(uci: string): {
  readonly from: string;
  readonly to: string;
  readonly promotion?: "knight" | "bishop" | "rook" | "queen";
} {
  const promotion = {
    b: "bishop",
    n: "knight",
    q: "queen",
    r: "rook",
  } as const;
  const symbol = uci[4] as keyof typeof promotion | undefined;
  return {
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    ...(symbol === undefined ? {} : { promotion: promotion[symbol] }),
  };
}

export function traceFixture(
  options: TraceFixtureOptions = {},
): PrivateSimulationTraceRecord {
  const seed = options.seed ?? 7;
  const gameIndex = options.gameIndex ?? 0;
  const initialFen = options.initialFen ?? STANDARD_INITIAL_FEN;
  const requestedMoves = options.moves ?? ["e2e4", "e7e5"];
  const drawbackId = options.drawbackId ?? "vegan";
  const evaluatorCoverage = options.evaluatorCoverage ?? "none";
  const history: ChessMove[] = [];
  const plies: PrivateSimulationTracePly[] = [];
  let snapshot: PublicPositionAuthoritySnapshot =
    createStandardChessPositionSnapshot(initialFen);

  for (const [ply, uci] of requestedMoves.entries()) {
    if (snapshot.authorityId !== "standard-chess/v1") {
      throw new Error("Fixture requires standard chess authority.");
    }
    const color = turnFromFen(snapshot.fen);
    const ordinaryMoves = publicAuthorityLegalMoves(snapshot);
    const position: PositionView = {
      fen: snapshot.fen,
      turn: color,
      ply,
      history: [...history],
    };
    const request = createEvaluatorTurnConstraintRequest(
      position,
      ordinaryMoves,
    );
    const transition = advancePublicPositionAuthority(
      snapshot,
      commandFromUci(uci),
    );
    const ordinaryUci = ordinaryMoves.map(canonicalMoveUci).sort();
    plies.push({
      ply,
      color,
      fenBefore: snapshot.fen,
      fenAfter: transition.position.fen,
      move: {
        uci: canonicalMoveUci(transition.move),
        san: transition.move.san,
      },
      ordinaryLegalMoves: ordinaryUci,
      drawbackLegalMoves: ordinaryUci,
      ruleTriggered: false,
      forced: ordinaryUci.length === 1,
      publicEvaluatorConstraint:
        evaluatorCoverage === "none"
          ? null
          : {
              provider: request.provider,
              policyId: request.policyId,
              positionKey: request.positionKey,
              requestDigest: "a".repeat(64),
              bestMoveUci: ordinaryUci[0] ?? uci,
              engineFingerprint:
                options.evaluatorFingerprint ?? "fixture-engine-sha256",
            },
      activeSecret: {
        drawbackId,
        hiddenParameters: {},
        drawbackInternalState: {
          movesApplied: history.filter((move) => move.color === color).length,
        },
      },
    });
    history.push(transition.move);
    snapshot = transition.position;
  }

  return parsePrivateSimulationTraceRecord({
    format: "drawbackengine-private-simulation-trace",
    schemaVersion: 1,
    authorityId: "standard-chess/v1",
    gameIndex,
    gameId: simulationGameId(seed, gameIndex),
    seed,
    plyLimit: requestedMoves.length,
    initialFen,
    finalFen: snapshot.fen,
    result: { kind: "active" },
    stoppedAtPlyLimit: true,
    evaluatorCoverage,
    drawbacks: {
      white: drawbackId,
      black: drawbackId,
    },
    agents: {
      white: {
        id: "fixture-agent",
        style: "test",
        strength: 100,
      },
      black: {
        id: "fixture-agent",
        style: "test",
        strength: 100,
      },
    },
    plies,
  });
}
