import {
  createStandardChessPositionSnapshot,
  publicAuthorityLegalMoves,
} from "@drawbackengine/chess-core";
import type {
  ChessMove,
  PositionView,
} from "@drawbackengine/drawback-engine";
import { canonicalMoveUci } from "@drawbackengine/drawback-engine";
import type { PlayerColor } from "@drawbackengine/shared";
import {
  parsePrivateSimulationTraceRecord,
  type PrivateSimulationTraceRecord,
} from "@drawbackengine/simulation-trace";
import {
  parseDatasetRow,
  type EvaluationMetadata,
  type PublicFeatureRecord,
  type TrainingLabelRecord,
} from "@drawbackguesser/dataset-contract";
import {
  aggregateRulePosteriors,
  createDefaultHypothesisSeeds,
  createPublicMoveObservation,
  DEFAULT_HYPOTHESIS_RULE_IDS,
  SymbolicPredictor,
  type PredictionState,
} from "@drawbackguesser/predictor";
import { validatePrivateTraceLabels } from "./private-label-validation.js";

export const SYMBOLIC_FEATURE_VERSION = 6 as const;
export const SYMBOLIC_RULE_COUNT = DEFAULT_HYPOTHESIS_RULE_IDS.length;

export type TrainingDatasetRow =
  & PublicFeatureRecord
  & Omit<TrainingLabelRecord, "playerColor">
  & EvaluationMetadata;

export interface DerivedPublicDatasetRow {
  readonly features: PublicFeatureRecord;
  readonly evaluation: EvaluationMetadata;
}

interface PublicTracePly {
  readonly ply: number;
  readonly color: PlayerColor;
  readonly fenBefore: string;
  readonly fenAfter: string;
  readonly move: {
    readonly uci: string;
    readonly san: string;
  };
  readonly ordinaryLegalMoves: readonly string[];
  readonly publicEvaluatorConstraint:
    PrivateSimulationTraceRecord["plies"][number]["publicEvaluatorConstraint"];
}

interface PublicTraceProjection {
  readonly gameId: string;
  readonly seed: number;
  readonly initialFen: string;
  readonly agents: PrivateSimulationTraceRecord["agents"];
  readonly plies: readonly PublicTracePly[];
}

function publicTraceProjection(
  trace: PrivateSimulationTraceRecord,
): PublicTraceProjection {
  return {
    gameId: trace.gameId,
    seed: trace.seed,
    initialFen: trace.initialFen,
    agents: trace.agents,
    plies: trace.plies.map((ply) => ({
      ply: ply.ply,
      color: ply.color,
      fenBefore: ply.fenBefore,
      fenAfter: ply.fenAfter,
      move: ply.move,
      ordinaryLegalMoves: ply.ordinaryLegalMoves,
      publicEvaluatorConstraint: ply.publicEvaluatorConstraint,
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

function canonicalObservedMove(ply: PublicTracePly): ChessMove {
  const snapshot = createStandardChessPositionSnapshot(ply.fenBefore);
  const move = publicAuthorityLegalMoves(snapshot).find(
    (candidate) => canonicalMoveUci(candidate) === ply.move.uci,
  );
  if (move === undefined) {
    throw new RangeError(
      `Trace ply ${String(ply.ply)} move ${ply.move.uci} is not authority-legal.`,
    );
  }
  return move;
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
    probabilities: DEFAULT_HYPOTHESIS_RULE_IDS.map(
      (id) => byRule.get(id)?.probability ?? 0,
    ),
    eliminated: DEFAULT_HYPOTHESIS_RULE_IDS.map(
      (id) => byRule.get(id)?.eliminated ?? true,
    ),
  };
}

function deriveParsedPublicRows(
  trace: PrivateSimulationTraceRecord,
): readonly DerivedPublicDatasetRow[] {
  const source = publicTraceProjection(trace);
  const history: ChessMove[] = [];
  const historySan: string[] = [];
  const initialPosition: PositionView = {
    fen: source.initialFen,
    turn: turnFromFen(source.initialFen),
    ply: 0,
    history: [],
  };
  const predictor = new SymbolicPredictor(
    {
      white: createDefaultHypothesisSeeds(),
      black: createDefaultHypothesisSeeds(),
    },
    initialPosition,
  );

  return source.plies.map((ply) => {
    const move = canonicalObservedMove(ply);
    const positionBefore: PositionView = {
      fen: ply.fenBefore,
      turn: ply.color,
      ply: ply.ply,
      history: [...history],
    };
    const nextHistory = [...history, move];
    const positionAfter: PositionView = {
      fen: ply.fenAfter,
      turn: turnFromFen(ply.fenAfter),
      ply: ply.ply + 1,
      history: nextHistory,
    };
    const prediction = predictor.observe(createPublicMoveObservation({
      authorityId: "standard-chess/v1",
      color: ply.color,
      positionBefore,
      positionAfter,
      move,
      ...(ply.publicEvaluatorConstraint === null
        ? {}
        : { externalConstraint: ply.publicEvaluatorConstraint }),
    }));
    const white = posteriorVector(prediction, "white");
    const black = posteriorVector(prediction, "black");
    const agent =
      ply.color === "white" ? source.agents.white : source.agents.black;
    const row: DerivedPublicDatasetRow = {
      features: {
        fenBefore: ply.fenBefore,
        move: ply.move.uci,
        moveNumber: moveNumberFromFen(ply.fenBefore),
        ply: ply.ply,
        playerColor: ply.color,
        historySan: [...historySan],
        ordinaryLegalMoves: [...ply.ordinaryLegalMoves],
        clockMs: null,
        symbolicFeatureVersion: SYMBOLIC_FEATURE_VERSION,
        symbolicWhiteRuleProbabilities: white.probabilities,
        symbolicBlackRuleProbabilities: black.probabilities,
        symbolicWhiteEliminated: white.eliminated,
        symbolicBlackEliminated: black.eliminated,
        publicEvaluatorConstraint: ply.publicEvaluatorConstraint,
      },
      evaluation: {
        gameId: source.gameId,
        seed: source.seed,
        san: ply.move.san,
        botAgentId: agent.id,
        botStyle: agent.style,
        botStrength: agent.strength,
      },
    };
    history.push(move);
    historySan.push(move.san);
    return row;
  });
}

export function derivePublicDatasetRows(
  traceInput: unknown,
): readonly DerivedPublicDatasetRow[] {
  const trace = parsePrivateSimulationTraceRecord(traceInput);
  validatePrivateTraceLabels(trace);
  return deriveParsedPublicRows(trace);
}

export function convertTraceToDatasetRows(
  traceInput: unknown,
): readonly TrainingDatasetRow[] {
  const trace = parsePrivateSimulationTraceRecord(traceInput);
  validatePrivateTraceLabels(trace);
  const publicRows = deriveParsedPublicRows(trace);
  return publicRows.map((publicRow, index) => {
    const ply = trace.plies[index];
    if (ply === undefined) {
      throw new Error("Public trace projection lost alignment with private labels.");
    }
    const trueRuleIndex = DEFAULT_HYPOTHESIS_RULE_IDS.findIndex(
      (ruleId) => ruleId === ply.activeSecret.drawbackId,
    );
    if (trueRuleIndex < 0) {
      throw new RangeError(
        `Trace truth ${ply.activeSecret.drawbackId} is outside the symbolic catalog.`,
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
      symbolicFeatureVersion: SYMBOLIC_FEATURE_VERSION,
      symbolicRuleCount: SYMBOLIC_RULE_COUNT,
    });
    return Object.freeze(row);
  });
}
