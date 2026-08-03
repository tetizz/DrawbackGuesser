import {
  DrawbackGameSession,
} from "@drawbackengine/chess-core";
import {
  canonicalMoveUci,
  resolveAuditedCapturableKingRule,
  type AuditedCapturableKingRuleId,
  type PromotionPiece,
} from "@drawbackengine/drawback-engine";
import {
  PLAYER_PRIVATE_SIMULATION_TRACE_FORMAT,
  PLAYER_PRIVATE_SIMULATION_TRACE_SCHEMA_VERSION,
  parsePlayerPrivateSimulationTraceRecord,
  playerPrivateSimulationGameId,
  type PlayerPrivateSimulationTraceRecord,
} from "@drawbackengine/simulation-trace";
import {
  Mulberry32,
  SIMULATION_RANDOM_POLICY,
} from "@drawbackengine/shared";

export interface PlayerPrivateTraceFixtureOptions {
  readonly whiteRuleId?: AuditedCapturableKingRuleId;
  readonly blackRuleId?: AuditedCapturableKingRuleId;
  readonly seed?: number;
  readonly gameIndex?: number;
  readonly policyId?: string;
  readonly parameterSeeds?: {
    readonly white: number;
    readonly black: number;
  };
  readonly initialFen?: string;
  readonly moves?: readonly string[];
  readonly autoLegalPlies?: number;
}

const AGENT = Object.freeze({
  id: "fixture-drawback-search",
  style: "drawback-search",
  strength: 1_200,
  searchPolicy: Object.freeze({
    policyId: "fixture-drawback-search/v1",
    evaluatorId: "drawback-material/v1",
    maxDepth: 1,
    maxNodes: 2_000,
    leafCacheEntries: 1_024,
    leafCacheHistoryMode: "full",
    opponentAggregation: "worst-case",
    temperatureCp: 1,
    topK: 1,
  }),
});

export function playerPrivateTraceFixture(
  options: PlayerPrivateTraceFixtureOptions = {},
): PlayerPrivateSimulationTraceRecord {
  const whiteRuleId = options.whiteRuleId ?? "vegan";
  const blackRuleId = options.blackRuleId ?? "checkers";
  const seed = options.seed ?? 0x1234_5678;
  const parameterSeeds = options.parameterSeeds ?? {
    white: 0x1111_1111,
    black: 0x2222_2222,
  };
  if (options.moves !== undefined && options.autoLegalPlies !== undefined) {
    throw new TypeError("Fixture moves and autoLegalPlies are mutually exclusive.");
  }
  const moves = options.moves ?? (
    options.autoLegalPlies === undefined ? ["e2e4", "e7e5"] : []
  );
  const plyLimit = options.autoLegalPlies ?? Math.max(1, moves.length);
  if (!Number.isSafeInteger(plyLimit) || plyLimit <= 0) {
    throw new RangeError("Fixture ply limit must be a positive safe integer.");
  }
  const gameIndex = options.gameIndex ?? 0;
  const agent = options.policyId === undefined
    ? AGENT
    : Object.freeze({
        ...AGENT,
        searchPolicy: Object.freeze({
          ...AGENT.searchPolicy,
          policyId: options.policyId,
        }),
      });
  const session = DrawbackGameSession.create(
    {
      white: resolveAuditedCapturableKingRule(whiteRuleId),
      black: resolveAuditedCapturableKingRule(blackRuleId),
    },
    {
      white: new Mulberry32(parameterSeeds.white),
      black: new Mulberry32(parameterSeeds.black),
    },
    options.initialFen,
  );
  const initialPosition = session.publicPositionSnapshot();
  const initialSecrets = session.exportSecretSnapshot();
  const plies = Array.from({ length: plyLimit }, (_, ply) => {
    const positionBefore = session.publicPositionSnapshot();
    const color = session.turn;
    const authorityLegalMoves =
      session.authorityLegalMoves().map(canonicalMoveUci);
    const drawbackLegalMoves = session.legalMoves().map(canonicalMoveUci);
    const activeSecret =
      session.turn === "white"
        ? session.exportSecretSnapshot().white
        : session.exportSecretSnapshot().black;
    const configured = moves[ply];
    const uci = configured ?? canonicalMoveUci(
      session.legalMoves()[0] ?? (() => {
        throw new Error("Fixture position has no legal move.");
      })(),
    );
    const outcome = session.move(commandFromUci(uci));
    if (!outcome.ok) {
      throw new Error(`Fixture move ${uci} was rejected: ${outcome.reason}.`);
    }
    return {
      ply,
      color,
      positionBefore,
      positionAfter: session.publicPositionSnapshot(),
      move: {
        uci: canonicalMoveUci(outcome.observation.move),
        san: outcome.observation.move.san,
      },
      authorityLegalMoves,
      drawbackLegalMoves,
      ruleTriggered: outcome.observation.ruleTriggered,
      forced: outcome.observation.forced,
      activeSecret: {
        drawbackId: activeSecret.drawbackId,
        hiddenParameters: activeSecret.parameters,
        drawbackInternalState: activeSecret.state,
      },
    };
  });
  const finalSecrets = session.exportSecretSnapshot();
  return parsePlayerPrivateSimulationTraceRecord({
    format: PLAYER_PRIVATE_SIMULATION_TRACE_FORMAT,
    schemaVersion: PLAYER_PRIVATE_SIMULATION_TRACE_SCHEMA_VERSION,
    authorityId: "capturable-king/v1",
    ruleset: {
      kind: "audited-player-private",
      version: PLAYER_PRIVATE_SIMULATION_TRACE_SCHEMA_VERSION,
    },
    randomPolicy: SIMULATION_RANDOM_POLICY,
    gameIndex,
    gameId: playerPrivateSimulationGameId(
      seed,
      gameIndex,
      parameterSeeds,
    ),
    seed,
    parameterSeeds,
    plyLimit,
    initialPosition,
    finalPosition: session.publicPositionSnapshot(),
    result: session.result,
    stoppedAtPlyLimit: session.result.kind === "active",
    hypothesisPolicy: {
      kind: "unrestricted-baseline",
      version: 1,
    },
    secrets: {
      initial: secretPair(initialSecrets),
      final: secretPair(finalSecrets),
    },
    agents: {
      white: agent,
      black: agent,
    },
    plies,
  });
}

function secretPair(
  secrets: {
    readonly white: {
      readonly drawbackId: string;
      readonly parameters: unknown;
      readonly state: unknown;
    };
    readonly black: {
      readonly drawbackId: string;
      readonly parameters: unknown;
      readonly state: unknown;
    };
  },
) {
  return {
    white: {
      drawbackId: secrets.white.drawbackId,
      hiddenParameters: secrets.white.parameters,
      drawbackInternalState: secrets.white.state,
    },
    black: {
      drawbackId: secrets.black.drawbackId,
      hiddenParameters: secrets.black.parameters,
      drawbackInternalState: secrets.black.state,
    },
  };
}

function commandFromUci(uci: string): {
  readonly from: string;
  readonly to: string;
  readonly promotion?: PromotionPiece;
} {
  const promotionBySymbol: Readonly<Record<string, PromotionPiece>> = {
    b: "bishop",
    n: "knight",
    q: "queen",
    r: "rook",
  };
  const symbol = uci[4];
  const promotion = symbol === undefined
    ? undefined
    : promotionBySymbol[symbol];
  if (symbol !== undefined && promotion === undefined) {
    throw new TypeError(`Fixture promotion ${symbol} is unsupported.`);
  }
  return {
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    ...(promotion === undefined ? {} : { promotion }),
  };
}
