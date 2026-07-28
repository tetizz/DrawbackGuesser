import {
  publicAuthorityLegalMoves,
  type CapturableKingPositionSnapshot,
} from "@drawbackengine/chess-core";
import {
  canonicalMoveUci,
  type ChessMove,
  type PositionView,
} from "@drawbackengine/drawback-engine";
import type { PlayerColor } from "@drawbackengine/shared";
import {
  parsePlayerPrivateSimulationTraceRecord,
  type PlayerPrivateSimulationTraceRecord,
} from "@drawbackengine/simulation-trace";
import {
  parseDatasetRow,
  type CapturablePublicPositionSnapshot,
  type EvaluationMetadata,
  type PublicFeatureRecord,
} from "@drawbackguesser/dataset-contract";
import {
  aggregateRuleOpportunityFeatures,
  aggregateRulePosteriors,
  CAPTURABLE_HYPOTHESIS_RULE_IDS,
  createCapturableHypothesisSeeds,
  createPublicMoveObservation,
  RULE_OPPORTUNITY_FEATURE_VERSION,
  SymbolicPredictor,
  type PredictionState,
} from "@drawbackguesser/predictor";
import type {
  DerivedPublicDatasetRow,
  TrainingDatasetRow,
} from "./converter.js";

export const CAPTURABLE_SYMBOLIC_FEATURE_VERSION = 9 as const;
export const CAPTURABLE_SYMBOLIC_RULE_COUNT =
  CAPTURABLE_HYPOTHESIS_RULE_IDS.length;

interface PublicCapturablePly {
  readonly ply: number;
  readonly color: PlayerColor;
  readonly positionBefore: CapturableKingPositionSnapshot;
  readonly positionAfter: CapturableKingPositionSnapshot;
  readonly move: {
    readonly uci: string;
    readonly san: string;
  };
  readonly authorityLegalMoves: readonly string[];
}

interface PublicCapturableTraceProjection {
  readonly gameId: string;
  readonly seed: number;
  readonly initialPosition: CapturableKingPositionSnapshot;
  readonly agents: PlayerPrivateSimulationTraceRecord["agents"];
  readonly plies: readonly PublicCapturablePly[];
}

/**
 * Deliberately removes every private label before symbolic inference starts.
 */
function publicTraceProjection(
  trace: PlayerPrivateSimulationTraceRecord,
): PublicCapturableTraceProjection {
  return {
    gameId: trace.gameId,
    seed: trace.seed,
    initialPosition: structuredClone(trace.initialPosition),
    agents: structuredClone(trace.agents),
    plies: trace.plies.map((ply) => ({
      ply: ply.ply,
      color: ply.color,
      positionBefore: structuredClone(ply.positionBefore),
      positionAfter: structuredClone(ply.positionAfter),
      move: structuredClone(ply.move),
      authorityLegalMoves: [...ply.authorityLegalMoves],
    })),
  };
}

function turnFromFen(fen: string): PlayerColor {
  const turn = fen.split(" ")[1];
  if (turn === "w") {
    return "white";
  }
  if (turn === "b") {
    return "black";
  }
  throw new TypeError("Trace FEN has an invalid side to move.");
}

function moveNumberFromFen(fen: string): number {
  const value = Number(fen.split(" ")[5]);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new TypeError("Trace FEN has an invalid fullmove number.");
  }
  return value;
}

function canonicalObservedMove(ply: PublicCapturablePly): ChessMove {
  const move = publicAuthorityLegalMoves(ply.positionBefore).find(
    (candidate) => canonicalMoveUci(candidate) === ply.move.uci,
  );
  if (move === undefined || move.san !== ply.move.san) {
    throw new RangeError(
      `Trace ply ${String(ply.ply)} move does not match capturable authority.`,
    );
  }
  return move;
}

function publicPositionFeature(
  snapshot: CapturableKingPositionSnapshot,
): CapturablePublicPositionSnapshot {
  if (snapshot.terminal !== null) {
    throw new TypeError("A training ply cannot begin after king capture.");
  }
  return {
    format: snapshot.format,
    version: snapshot.version,
    authorityId: snapshot.authorityId,
    fen: snapshot.fen,
    orthodoxCompatible: snapshot.orthodoxCompatible,
    kingPassant: structuredClone(snapshot.kingPassant),
    terminal: null,
  };
}

function posteriorVector(
  state: PredictionState,
  color: PlayerColor,
): {
  readonly probabilities: readonly number[];
  readonly eliminated: readonly boolean[];
} {
  const distribution = color === "white" ? state.white : state.black;
  const byRule = new Map(
    aggregateRulePosteriors(distribution).map((posterior) => [
      posterior.drawbackId,
      posterior,
    ]),
  );
  return {
    probabilities: CAPTURABLE_HYPOTHESIS_RULE_IDS.map(
      (id) => byRule.get(id)?.probability ?? 0,
    ),
    eliminated: CAPTURABLE_HYPOTHESIS_RULE_IDS.map(
      (id) => byRule.get(id)?.eliminated ?? true,
    ),
  };
}

function parsedPublicRows(
  trace: PlayerPrivateSimulationTraceRecord,
): readonly DerivedPublicDatasetRow[] {
  const source = publicTraceProjection(trace);
  const history: ChessMove[] = [];
  const historySan: string[] = [];
  const initialPosition: PositionView = {
    fen: source.initialPosition.fen,
    turn: turnFromFen(source.initialPosition.fen),
    ply: 0,
    history: [],
  };
  const seeds = createCapturableHypothesisSeeds();
  const predictor = new SymbolicPredictor(
    { white: seeds, black: seeds },
    initialPosition,
    {
      authorityId: "capturable-king/v1",
      initialAuthorityPosition: source.initialPosition,
    },
  );

  return source.plies.map((ply) => {
    const move = canonicalObservedMove(ply);
    const positionBefore: PositionView = {
      fen: ply.positionBefore.fen,
      turn: ply.color,
      ply: ply.ply,
      history: [...history],
    };
    const nextHistory = [...history, move];
    const positionAfter: PositionView = {
      fen: ply.positionAfter.fen,
      turn: turnFromFen(ply.positionAfter.fen),
      ply: ply.ply + 1,
      history: nextHistory,
    };
    const stateBefore = predictor.state;
    const prediction = predictor.observeWithOpportunities(
      createPublicMoveObservation({
        authorityId: "capturable-king/v1",
        authorityPositionBefore: ply.positionBefore,
        color: ply.color,
        positionBefore,
        positionAfter,
        move,
      }),
    );
    const opportunityFeatures = aggregateRuleOpportunityFeatures(
      stateBefore,
      prediction.opportunity,
      CAPTURABLE_HYPOTHESIS_RULE_IDS,
    );
    const white = posteriorVector(prediction.state, "white");
    const black = posteriorVector(prediction.state, "black");
    const agent =
      ply.color === "white" ? source.agents.white : source.agents.black;
    const features: PublicFeatureRecord = {
      authorityId: "capturable-king/v1",
      publicAuthorityPositionBefore: publicPositionFeature(
        ply.positionBefore,
      ),
      fenBefore: ply.positionBefore.fen,
      move: ply.move.uci,
      moveNumber: moveNumberFromFen(ply.positionBefore.fen),
      ply: ply.ply,
      playerColor: ply.color,
      historySan: [...historySan],
      ordinaryLegalMoves: [...ply.authorityLegalMoves],
      clockMs: null,
      symbolicFeatureVersion: CAPTURABLE_SYMBOLIC_FEATURE_VERSION,
      opportunityFeatureVersion: RULE_OPPORTUNITY_FEATURE_VERSION,
      symbolicActiveRuleOpportunityFeatures: opportunityFeatures,
      symbolicWhiteRuleProbabilities: white.probabilities,
      symbolicBlackRuleProbabilities: black.probabilities,
      symbolicWhiteEliminated: white.eliminated,
      symbolicBlackEliminated: black.eliminated,
      publicEvaluatorConstraint: null,
    };
    const evaluation: EvaluationMetadata = {
      gameId: source.gameId,
      seed: source.seed,
      san: ply.move.san,
      botAgentId: agent.id,
      botStyle: agent.style,
      botStrength: agent.strength,
    };
    history.push(move);
    historySan.push(move.san);
    return Object.freeze({ features, evaluation });
  });
}

export function deriveCapturablePublicDatasetRows(
  traceInput: unknown,
): readonly DerivedPublicDatasetRow[] {
  return parsedPublicRows(
    parsePlayerPrivateSimulationTraceRecord(traceInput),
  );
}

export function convertPlayerPrivateTraceToDatasetRows(
  traceInput: unknown,
): readonly TrainingDatasetRow[] {
  return convertParsedPlayerPrivateTraceToDatasetRows(
    parsePlayerPrivateSimulationTraceRecord(traceInput),
  );
}

/**
 * Internal fast path for a record already returned by the Engine parser.
 * Callers at trust boundaries must parse first; this avoids repeating full
 * semantic replay while authenticating large corpora.
 */
export function convertParsedPlayerPrivateTraceToDatasetRows(
  trace: PlayerPrivateSimulationTraceRecord,
): readonly TrainingDatasetRow[] {
  const publicRows = parsedPublicRows(trace);
  return publicRows.map((publicRow, index) => {
    const ply = trace.plies[index];
    if (ply === undefined) {
      throw new Error("Public trace projection lost private-label alignment.");
    }
    const trueRuleIndex = CAPTURABLE_HYPOTHESIS_RULE_IDS.indexOf(
      ply.activeSecret.drawbackId,
    );
    if (trueRuleIndex < 0) {
      throw new RangeError(
        `Trace truth ${ply.activeSecret.drawbackId} is outside the capturable catalog.`,
      );
    }
    const activeEliminated =
      ply.color === "white"
        ? publicRow.features.symbolicWhiteEliminated[trueRuleIndex]
        : publicRow.features.symbolicBlackEliminated[trueRuleIndex];
    if (activeEliminated !== false) {
      throw new Error(
        `Symbolic inference contradicted true drawback ${ply.activeSecret.drawbackId} at ply ${String(ply.ply)}.`,
      );
    }
    const row: TrainingDatasetRow = {
      ...publicRow.features,
      trueDrawback: ply.activeSecret.drawbackId,
      hiddenParameters: structuredClone(ply.activeSecret.hiddenParameters),
      drawbackInternalState: structuredClone(
        ply.activeSecret.drawbackInternalState,
      ),
      drawbackLegalMoves: [...ply.drawbackLegalMoves],
      ruleTriggered: ply.ruleTriggered,
      forced: ply.forced,
      result: structuredClone(trace.result),
      ...publicRow.evaluation,
    };
    parseDatasetRow(row, {
      authorityId: "capturable-king/v1",
      symbolicFeatureVersion: CAPTURABLE_SYMBOLIC_FEATURE_VERSION,
      symbolicRuleCount: CAPTURABLE_SYMBOLIC_RULE_COUNT,
    });
    return Object.freeze(row);
  });
}
