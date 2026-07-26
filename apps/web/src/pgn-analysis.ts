import {
  CompletedPgnParseError,
  MAX_COMPLETED_PGN_INPUT_BYTES,
  MAX_COMPLETED_PGN_PLIES,
  replayCompletedPgn,
  tokenizeCompletedPgn,
} from "@drawbackengine/chess-core";
import {
  createEvaluatorTurnConstraintRequest,
  type ChessMove,
} from "@drawbackengine/drawback-engine";
import type {
  AuthenticatedCompletedPgnEvaluatorSidecar,
} from "@drawbackengine/chess-evaluator/completed-pgn-sidecar";
import {
  aggregateParameterPosteriors,
  createPublicMoveObservation,
  SymbolicPredictor,
  type PredictionState,
} from "@drawbackguesser/predictor";
import type { PlayerColor } from "@drawbackengine/shared";
import {
  createHypothesisSeeds,
  HYPOTHESIS_COVERAGE,
  type HypothesisCoverage,
} from "./hypothesis-catalog.js";
import { rankedHypotheses } from "./model.js";
import { fuseSymbolicAndNeural } from "./hybrid-prediction.js";
import {
  runBrowserNeuralModel,
  type BrowserNeuralModel,
} from "./neural-model.js";
import type {
  HybridObservation,
  SingleHybridBrowserModel,
} from "./sequence-neural-model.js";
import type {
  ENSEMBLE_FUSION_METHOD,
  EnsembleBrowserModel,
} from "./ensemble-neural-model.js";
import { STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS } from "./pgn-analysis-contract.js";

export interface PgnParameterGuess {
  readonly name: string;
  readonly value: unknown;
  readonly confidence: number;
}

export interface PgnGuess {
  readonly id: string;
  readonly confidence: number;
  readonly eliminated: boolean;
  readonly parameters: readonly PgnParameterGuess[];
}

export interface PgnPredictionPoint {
  readonly ply: number;
  readonly moveNumber: number;
  readonly color: PlayerColor;
  readonly san: string;
  readonly fenBefore: string;
  readonly white: readonly PgnGuess[];
  readonly black: readonly PgnGuess[];
  readonly eliminations: readonly PgnEliminationEvidence[];
}

export interface PgnEliminationEvidence {
  readonly color: PlayerColor;
  readonly drawbackId: string;
  readonly reason: string;
}

export interface PgnAnalysisResult {
  readonly sourceBinding: {
    readonly headers: Readonly<Record<string, string>>;
    readonly normalizedMainline: readonly string[];
  };
  readonly plyCount: number;
  readonly finalFen: string;
  readonly finalWhite: readonly PgnGuess[];
  readonly finalBlack: readonly PgnGuess[];
  readonly history: readonly PgnPredictionPoint[];
  readonly coverage: readonly HypothesisCoverage[];
  readonly representedDrawbackCount: number;
  readonly representedDrawbackIds: readonly string[];
  readonly catalogDrawbackCount: number;
  readonly unavailableDrawbacks: readonly {
    readonly id: string;
    readonly name: string;
    readonly reason: "requires-public-evaluator-facts";
    readonly rank: null;
    readonly eliminated: false;
  }[];
  readonly evaluatorEvidence:
    | { readonly mode: "standard-pgn" }
    | {
        readonly mode: "authenticated-sidecar";
        readonly artifactSha256: string;
        readonly policy: {
          readonly id: string;
          readonly version: number;
        };
        readonly engine: {
          readonly uciName: string;
          readonly engine: string;
          readonly version: string;
          readonly executableSha256: string;
          readonly optionsDigest: string;
          readonly publicFingerprint: string;
        };
        readonly searchLimit:
          | { readonly kind: "depth"; readonly value: number }
          | { readonly kind: "move-time-ms"; readonly value: number }
          | { readonly kind: "nodes"; readonly value: number };
      };
  readonly predictor:
    | { readonly mode: "symbolic-only" }
    | {
        readonly mode: "hybrid-v1";
        readonly modelFormatVersion: 1;
        readonly artifactSha256: string;
        readonly sourceCheckpointSha256: string;
        readonly featureSchemaVersion: 1;
        readonly neuralDrawbackVocabulary: readonly string[];
        readonly neuralCoveredDrawbackCount: number;
        readonly neuralEvidenceWeight: number;
      }
    | {
        readonly mode: "hybrid-v21";
        readonly modelFormatVersion: 2;
        readonly artifactSha256: string;
        readonly sourceCheckpointSha256: string;
        readonly featureSchemaVersion: 1;
        readonly symbolicFeatureVersion: 6;
        readonly neuralDrawbackVocabulary: readonly string[];
        readonly neuralCoveredDrawbackCount: number;
        readonly unresolvedExternalConstraintIds: readonly string[];
      }
    | {
        readonly mode: "hybrid-v22";
        readonly modelFormatVersion: 3;
        readonly artifactSha256: string;
        readonly sourceCheckpointSha256: string;
        readonly featureSchemaVersion: 1;
        readonly symbolicFeatureVersion: 6;
        readonly sequenceObservationMode:
          | "masked-current-v2"
          | "exact-current-v2";
        readonly neuralDrawbackVocabulary: readonly string[];
        readonly neuralCoveredDrawbackCount: number;
        readonly unresolvedExternalConstraintIds: readonly string[];
      }
    | {
        readonly mode: "hybrid-v21-ensemble";
        readonly modelFormatVersion: 4;
        readonly artifactSha256: string;
        readonly sourceEnsembleReleaseSha256: string;
        readonly sourceFusionSelectionSha256: string;
        readonly sourceCalibrationSha256: string;
        readonly featureSchemaVersion: 1;
        readonly symbolicFeatureVersion: 6;
        readonly fusionMethod: typeof ENSEMBLE_FUSION_METHOD;
        readonly selectedAlpha: number;
        readonly neuralDrawbackVocabulary: readonly string[];
        readonly neuralCoveredDrawbackCount: number;
        readonly unresolvedExternalConstraintIds: readonly string[];
        readonly members: readonly {
          readonly trainingSeed: number;
          readonly sourceCheckpointSha256: string;
          readonly sourceSelectionSha256: string;
          readonly trainingRunId: string;
          readonly selectedEpoch: number;
        }[];
        readonly calibration: {
          readonly preservesHardEliminations: true;
          readonly white: {
            readonly temperature: number;
            readonly exampleCount: number;
            readonly nllBefore: number;
            readonly nllAfter: number;
          };
          readonly black: {
            readonly temperature: number;
            readonly exampleCount: number;
            readonly nllBefore: number;
            readonly nllAfter: number;
          };
        };
      };
}

export interface PgnAnalysisProgress {
  readonly processedPlies: number;
  readonly totalPlies: number;
}

export interface PgnAnalysisOptions {
  readonly onProgress?: (progress: PgnAnalysisProgress) => void;
  readonly neuralModel?: BrowserNeuralModel;
  readonly neuralArtifactSha256?: string;
  readonly evaluatorEvidence?: AuthenticatedCompletedPgnEvaluatorSidecar;
}

export const MAX_PGN_INPUT_BYTES = MAX_COMPLETED_PGN_INPUT_BYTES;
export const MAX_PGN_PLIES = MAX_COMPLETED_PGN_PLIES;
export { CompletedPgnParseError as PgnParseError };
const ALL_HYPOTHESIS_SEEDS = createHypothesisSeeds();
const STANDARD_PGN_UNAVAILABLE_DRAWBACK_ID_SET = new Set<string>(
  STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS,
);
const HYPOTHESIS_NAME_BY_ID = new Map(
  ALL_HYPOTHESIS_SEEDS.map((seed) => [
    seed.kind === "rerandomized" ? seed.drawbackId : seed.rule.id,
    seed.kind === "rerandomized" ? seed.name : seed.rule.name,
  ]),
);
const UNAVAILABLE_DRAWBACKS = Object.freeze(
  STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS.map((id) => {
    const name = HYPOTHESIS_NAME_BY_ID.get(id);
    if (name === undefined) {
      throw new Error(`Standard-PGN unavailable drawback is unknown: ${id}`);
    }
    return Object.freeze({
      id,
      name,
      reason: "requires-public-evaluator-facts" as const,
      rank: null,
      eliminated: false as const,
    });
  }),
);
const CATALOG_DRAWBACK_COUNT = new Set(
  ALL_HYPOTHESIS_SEEDS.map((seed) =>
    seed.kind === "rerandomized" ? seed.drawbackId : seed.rule.id
  ),
).size;
const REPRESENTED_DRAWBACK_IDS = Object.freeze(
  [...new Set(
    ALL_HYPOTHESIS_SEEDS.map((seed) =>
      seed.kind === "rerandomized" ? seed.drawbackId : seed.rule.id
    ),
  )]
    .filter((id) => !STANDARD_PGN_UNAVAILABLE_DRAWBACK_ID_SET.has(id))
    .sort(),
);
const REPRESENTED_DRAWBACK_COUNT = REPRESENTED_DRAWBACK_IDS.length;
const ALL_DRAWBACK_IDS = Object.freeze(
  [...new Set(
    ALL_HYPOTHESIS_SEEDS.map((seed) =>
      seed.kind === "rerandomized" ? seed.drawbackId : seed.rule.id
    ),
  )].sort(),
);

if (
  CATALOG_DRAWBACK_COUNT !== 182 ||
  REPRESENTED_DRAWBACK_COUNT !== 180 ||
  CATALOG_DRAWBACK_COUNT - REPRESENTED_DRAWBACK_COUNT !==
    STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS.length
) {
  throw new Error(
    "Standard-PGN browser view must represent exactly 180 of 182 catalog drawbacks.",
  );
}

function createPredictor(
  initialFen: string,
  initialTurn: PlayerColor,
): SymbolicPredictor {
  return new SymbolicPredictor(
    {
      white: ALL_HYPOTHESIS_SEEDS,
      black: ALL_HYPOTHESIS_SEEDS,
    },
    {
      fen: initialFen,
      turn: initialTurn,
      ply: 0,
      history: [],
    },
  );
}

export function tokenizePgn(pgn: string): readonly string[] {
  return tokenizeCompletedPgn(pgn);
}

function guesses(state: PredictionState, color: PlayerColor): readonly PgnGuess[] {
  const distribution = state[color];
  return rankedHypotheses(distribution.hypotheses).map((hypothesis) => ({
    id: hypothesis.id,
    confidence: hypothesis.confidence,
    eliminated: hypothesis.eliminated,
    parameters: aggregateParameterPosteriors(distribution, hypothesis.id).flatMap(
      (parameter) => {
        const top = parameter.values[0];
        return top === undefined
          ? []
          : [
              {
                name: parameter.parameter,
                value: top.value,
                confidence: top.conditionalProbability,
              },
            ];
      },
    ),
  }));
}

/**
 * Projects the complete internal posterior onto rules whose constraints can be
 * reconstructed from an ordinary PGN. The two evaluator-dependent rules remain
 * internal evidence inputs, but are unavailable rather than hard-eliminated in
 * the public standard-PGN view.
 */
export function projectStandardPgnGuesses(
  complete: readonly PgnGuess[],
): readonly PgnGuess[] {
  const byId = new Map(complete.map((guess) => [guess.id, guess]));
  if (
    complete.length !== CATALOG_DRAWBACK_COUNT ||
    byId.size !== CATALOG_DRAWBACK_COUNT ||
    REPRESENTED_DRAWBACK_IDS.some((id) => !byId.has(id)) ||
    STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS.some((id) => !byId.has(id)) ||
    complete.some(
      ({ confidence, eliminated }) =>
        !Number.isFinite(confidence) ||
        confidence < 0 ||
        (eliminated && confidence !== 0),
    )
  ) {
    throw new Error(
      "Complete posterior does not match the 182-rule drawback catalog.",
    );
  }
  const represented = complete.filter(
    ({ id }) => !STANDARD_PGN_UNAVAILABLE_DRAWBACK_ID_SET.has(id),
  );
  const representedMass = represented.reduce(
    (sum, { confidence }) => sum + confidence,
    0,
  );
  if (!Number.isFinite(representedMass) || representedMass <= 0) {
    throw new Error(
      "Standard-PGN represented posterior has no finite positive mass.",
    );
  }
  return Object.freeze(
    represented
      .map((guess) =>
        Object.freeze({
          ...guess,
          confidence: guess.eliminated
            ? 0
            : guess.confidence / representedMass,
        })
      )
      .sort(
        (left, right) =>
          Number(left.eliminated) - Number(right.eliminated) ||
          right.confidence - left.confidence ||
          left.id.localeCompare(right.id),
      ),
  );
}

function exposeGuesses(
  complete: readonly PgnGuess[],
  enriched: boolean,
): readonly PgnGuess[] {
  return enriched
    ? Object.freeze([...complete])
    : projectStandardPgnGuesses(complete);
}

function orderedSymbolicValues(
  guessesForColor: readonly PgnGuess[],
  ruleIds: readonly string[],
): {
  readonly probabilities: readonly number[];
  readonly eliminated: readonly boolean[];
} {
  const byId = new Map(guessesForColor.map((guess) => [guess.id, guess]));
  if (
    byId.size !== guessesForColor.length ||
    byId.size !== ruleIds.length ||
    ruleIds.some((id) => !byId.has(id))
  ) {
    throw new Error("Public symbolic state does not match the hybrid schema.");
  }
  return {
    probabilities: Object.freeze(
      ruleIds.map((id) => byId.get(id)?.confidence ?? 0),
    ),
    eliminated: Object.freeze(
      ruleIds.map((id) => byId.get(id)?.eliminated ?? true),
    ),
  };
}

export function buildPublicHybridObservation(
  model: SingleHybridBrowserModel | EnsembleBrowserModel,
  observation: Omit<HybridObservation, "symbolic">,
  symbolicWhite: readonly PgnGuess[],
  symbolicBlack: readonly PgnGuess[],
): HybridObservation {
  const white = orderedSymbolicValues(
    symbolicWhite,
    model.symbolicRuleIds,
  );
  const black = orderedSymbolicValues(
    symbolicBlack,
    model.symbolicRuleIds,
  );
  return Object.freeze({
    ...observation,
    symbolic: Object.freeze({
      ruleIds: model.symbolicRuleIds,
      whiteProbabilities: white.probabilities,
      blackProbabilities: black.probabilities,
      whiteEliminated: white.eliminated,
      blackEliminated: black.eliminated,
    }),
  });
}

function applyHybridPosterior(
  symbolic: readonly PgnGuess[],
  neural: Readonly<Record<string, number>>,
): readonly PgnGuess[] {
  if (
    Object.keys(neural).length !== symbolic.length ||
    symbolic.some((guess) => !Object.hasOwn(neural, guess.id))
  ) {
    throw new Error("Hybrid posterior does not match the symbolic state.");
  }
  return Object.freeze(
    symbolic
      .map((guess) => {
        const confidence = neural[guess.id];
        if (
          confidence === undefined ||
          !Number.isFinite(confidence) ||
          confidence < 0 ||
          (guess.eliminated && confidence !== 0)
        ) {
          throw new Error("Hybrid posterior violates exact symbolic legality.");
        }
        return Object.freeze({ ...guess, confidence });
      })
      .sort(
        (left, right) =>
          Number(left.eliminated) - Number(right.eliminated) ||
          right.confidence - left.confidence ||
          left.id.localeCompare(right.id),
      ),
  );
}

function moveCode(move: ChessMove): string {
  const promotion = move.promotion === undefined
    ? ""
    : {
        knight: "n",
        bishop: "b",
        rook: "r",
        queen: "q",
      }[move.promotion];
  return `${move.from}${move.to}${promotion}`;
}

function boundHeaders(
  headers: ReadonlyMap<string, string>,
): Readonly<Record<string, string>> {
  return Object.freeze(
    Object.fromEntries(
      [...headers.entries()].sort(([left], [right]) =>
        left.localeCompare(right)
      ),
    ),
  );
}

export function analyzePgn(
  pgn: string,
  options: PgnAnalysisOptions = {},
): PgnAnalysisResult {
  const evaluatorEvidence = options.evaluatorEvidence;
  if (
    evaluatorEvidence !== undefined &&
    !/^[0-9a-f]{64}$/u.test(evaluatorEvidence.artifactSha256)
  ) {
    throw new TypeError(
      "Evaluator-enriched analysis requires the authenticated sidecar SHA-256 digest.",
    );
  }
  const neuralArtifactSha256 = options.neuralArtifactSha256;
  if (
    options.neuralModel !== undefined &&
    (
      neuralArtifactSha256 === undefined ||
      !/^[0-9a-f]{64}$/u.test(neuralArtifactSha256)
    )
  ) {
    throw new TypeError(
      "Hybrid analysis requires the selected artifact SHA-256 digest.",
    );
  }
  if (
    options.neuralModel !== undefined &&
    !options.neuralModel.drawbackVocabulary.some((id) =>
      REPRESENTED_DRAWBACK_IDS.includes(id)
    )
  ) {
    throw new TypeError(
      "Neural artifact has no overlap with represented browser hypotheses.",
    );
  }
  const replay = replayCompletedPgn(pgn);
  if (
    evaluatorEvidence !== undefined &&
    evaluatorEvidence.constraints.length !== replay.steps.length
  ) {
    throw new TypeError(
      "Evaluator sidecar constraint count must exactly match the completed PGN.",
    );
  }
  if (evaluatorEvidence !== undefined) {
    for (const [index, step] of replay.steps.entries()) {
      const constraint = evaluatorEvidence.constraints[index];
      const expected = createEvaluatorTurnConstraintRequest(
        {
          fen: step.fenBefore,
          turn: step.color,
          ply: index,
          history: step.historyBefore,
        },
        step.ordinaryLegalMoves,
      );
      if (
        constraint === undefined ||
        constraint.positionKey !== expected.positionKey
      ) {
        throw new TypeError(
          "Evaluator sidecar constraints do not match the completed PGN replay.",
        );
      }
    }
  }
  options.onProgress?.({
    processedPlies: 0,
    totalPlies: replay.steps.length,
  });

  const firstStep = replay.steps[0];
  if (firstStep === undefined) {
    throw new Error("Completed PGN replay unexpectedly contains no moves.");
  }
  const predictor = createPredictor(replay.initialFen, firstStep.color);
  const history: PgnPredictionPoint[] = [];
  const enriched = evaluatorEvidence !== undefined;
  let previousWhite = exposeGuesses(guesses(predictor.state, "white"), enriched);
  let previousBlack = exposeGuesses(guesses(predictor.state, "black"), enriched);
  let neuralCoveredDrawbackCount = 0;

  for (const [index, step] of replay.steps.entries()) {
    const nextTurn: PlayerColor =
      step.color === "white" ? "black" : "white";
    const state = predictor.observe(createPublicMoveObservation({
      authorityId: "standard-chess/v1",
      color: step.color,
      move: step.move,
      positionBefore: {
        fen: step.fenBefore,
        turn: step.color,
        ply: index,
        history: step.historyBefore,
      },
      positionAfter: {
        fen: step.fenAfter,
        turn: nextTurn,
        ply: index + 1,
        history: Object.freeze([...step.historyBefore, step.move]),
      },
      ...(evaluatorEvidence === undefined
        ? {}
        : {
          externalConstraint:
            evaluatorEvidence.constraints[index],
        }),
    }));
    const symbolicWhite = guesses(state, "white");
    const symbolicBlack = guesses(state, "black");
    const publicObservation = {
      fenBefore: step.fenBefore,
      move: moveCode(step.move),
      moveNumber: step.moveNumber,
      ply: index,
      playerColor: step.color,
      historySan: step.historyBefore.map(
        (historicMove) => historicMove.san,
      ),
      ordinaryLegalMoveCount: step.ordinaryLegalMoves.length,
    };
    const neural = options.neuralModel === undefined
      ? undefined
      : options.neuralModel.modelVariant !== "v1"
        ? runBrowserNeuralModel(
            options.neuralModel,
            buildPublicHybridObservation(
              options.neuralModel,
              publicObservation,
              symbolicWhite,
              symbolicBlack,
            ),
          )
        : runBrowserNeuralModel(options.neuralModel, publicObservation);
    const whiteFusion = neural === undefined
      ? undefined
      : options.neuralModel?.modelVariant !== "v1"
        ? {
            guesses: applyHybridPosterior(symbolicWhite, neural.white),
            neuralCoveredDrawbackCount: Object.keys(neural.white).length,
          }
        : fuseSymbolicAndNeural(symbolicWhite, neural.white);
    const blackFusion = neural === undefined
      ? undefined
      : options.neuralModel?.modelVariant !== "v1"
        ? {
            guesses: applyHybridPosterior(symbolicBlack, neural.black),
            neuralCoveredDrawbackCount: Object.keys(neural.black).length,
          }
        : fuseSymbolicAndNeural(symbolicBlack, neural.black);
    neuralCoveredDrawbackCount = Math.max(
      neuralCoveredDrawbackCount,
      whiteFusion?.neuralCoveredDrawbackCount ?? 0,
      blackFusion?.neuralCoveredDrawbackCount ?? 0,
    );
    const currentWhite = exposeGuesses(
      whiteFusion?.guesses ?? symbolicWhite,
      enriched,
    );
    const currentBlack = exposeGuesses(
      blackFusion?.guesses ?? symbolicBlack,
      enriched,
    );
    const previousForColor =
      step.color === "white" ? previousWhite : previousBlack;
    const currentForColor =
      step.color === "white" ? currentWhite : currentBlack;
    const previousById = new Map(
      previousForColor.map((guess) => [guess.id, guess]),
    );
    const eliminations = currentForColor
      .filter(
        (guess) =>
          guess.eliminated &&
          previousById.get(guess.id)?.eliminated === false,
      )
      .map((guess) =>
        Object.freeze({
          color: step.color,
          drawbackId: guess.id,
          reason: `Observed ${step.san} was impossible under this drawback's reconstructed legal-move constraint.`,
        })
      );
    history.push({
      ply: step.ply,
      moveNumber: step.moveNumber,
      color: step.color,
      san: step.san,
      fenBefore: step.fenBefore,
      white: currentWhite,
      black: currentBlack,
      eliminations: Object.freeze(eliminations),
    });
    previousWhite = currentWhite;
    previousBlack = currentBlack;
    options.onProgress?.({
      processedPlies: index + 1,
      totalPlies: replay.steps.length,
    });
  }

  const finalState = predictor.state;
  const finalPoint = history.at(-1);
  return {
    sourceBinding: {
      headers: boundHeaders(replay.headers),
      normalizedMainline: replay.normalizedMainline,
    },
    plyCount: history.length,
    finalFen: replay.finalFen,
    finalWhite:
      finalPoint?.white ??
      exposeGuesses(guesses(finalState, "white"), enriched),
    finalBlack:
      finalPoint?.black ??
      exposeGuesses(guesses(finalState, "black"), enriched),
    history,
    coverage: HYPOTHESIS_COVERAGE,
    representedDrawbackCount: enriched
      ? CATALOG_DRAWBACK_COUNT
      : REPRESENTED_DRAWBACK_COUNT,
    representedDrawbackIds: enriched
      ? ALL_DRAWBACK_IDS
      : REPRESENTED_DRAWBACK_IDS,
    catalogDrawbackCount: CATALOG_DRAWBACK_COUNT,
    unavailableDrawbacks: enriched ? Object.freeze([]) : UNAVAILABLE_DRAWBACKS,
    evaluatorEvidence: evaluatorEvidence === undefined
      ? Object.freeze({ mode: "standard-pgn" as const })
      : Object.freeze({
        mode: "authenticated-sidecar" as const,
        artifactSha256: evaluatorEvidence.artifactSha256,
        policy: Object.freeze({
          id: evaluatorEvidence.sidecar.policy.id,
          version: evaluatorEvidence.sidecar.policy.version,
        }),
        engine: Object.freeze({
          ...evaluatorEvidence.sidecar.policy.engine,
        }),
        searchLimit: Object.freeze({
          ...evaluatorEvidence.sidecar.policy.searchLimit,
        }),
      }),
    predictor:
      options.neuralModel === undefined || neuralArtifactSha256 === undefined
      ? { mode: "symbolic-only" }
      : options.neuralModel.modelVariant === "v21-hybrid-ensemble"
        ? {
          mode: "hybrid-v21-ensemble",
          modelFormatVersion: 4,
          artifactSha256: neuralArtifactSha256,
          sourceEnsembleReleaseSha256:
            options.neuralModel.ensemble.sourceEnsembleReleaseSha256,
          sourceFusionSelectionSha256:
            options.neuralModel.ensemble.sourceFusionSelectionSha256,
          sourceCalibrationSha256:
            options.neuralModel.calibration.sourceCalibrationSha256,
          featureSchemaVersion: 1,
          symbolicFeatureVersion: 6,
          fusionMethod: options.neuralModel.ensemble.method,
          selectedAlpha: options.neuralModel.ensemble.selectedAlpha,
          neuralDrawbackVocabulary:
            options.neuralModel.drawbackVocabulary,
          neuralCoveredDrawbackCount,
          unresolvedExternalConstraintIds: enriched
            ? Object.freeze([])
            : UNAVAILABLE_DRAWBACKS.map(({ id }) => id),
          members: options.neuralModel.ensemble.members.map((member) => ({
            trainingSeed: member.trainingSeed,
            sourceCheckpointSha256: member.sourceCheckpointSha256,
            sourceSelectionSha256: member.sourceSelectionSha256,
            trainingRunId: member.trainingRunId,
            selectedEpoch: member.selectedEpoch,
          })),
          calibration: {
            preservesHardEliminations:
              options.neuralModel.calibration.preservesHardEliminations,
            white: { ...options.neuralModel.calibration.white },
            black: { ...options.neuralModel.calibration.black },
          },
        }
      : options.neuralModel.modelVariant === "v21-hybrid"
        ? {
          mode: "hybrid-v21",
          modelFormatVersion: 2,
          artifactSha256: neuralArtifactSha256,
          sourceCheckpointSha256:
            options.neuralModel.sourceCheckpointSha256,
          featureSchemaVersion: 1,
          symbolicFeatureVersion: 6,
          neuralDrawbackVocabulary:
            options.neuralModel.drawbackVocabulary,
          neuralCoveredDrawbackCount,
          unresolvedExternalConstraintIds: enriched
            ? Object.freeze([])
            : UNAVAILABLE_DRAWBACKS.map(({ id }) => id),
        }
      : options.neuralModel.modelVariant === "v22-hybrid"
        ? {
          mode: "hybrid-v22",
          modelFormatVersion: 3,
          artifactSha256: neuralArtifactSha256,
          sourceCheckpointSha256:
            options.neuralModel.sourceCheckpointSha256,
          featureSchemaVersion: 1,
          symbolicFeatureVersion: 6,
          sequenceObservationMode:
            options.neuralModel.sequenceObservationMode,
          neuralDrawbackVocabulary:
            options.neuralModel.drawbackVocabulary,
          neuralCoveredDrawbackCount,
          unresolvedExternalConstraintIds: enriched
            ? Object.freeze([])
            : UNAVAILABLE_DRAWBACKS.map(({ id }) => id),
        }
        : {
          mode: "hybrid-v1",
          modelFormatVersion: 1,
          artifactSha256: neuralArtifactSha256,
          sourceCheckpointSha256:
            options.neuralModel.sourceCheckpointSha256,
          featureSchemaVersion: 1,
          neuralDrawbackVocabulary:
            options.neuralModel.drawbackVocabulary,
          neuralCoveredDrawbackCount,
          neuralEvidenceWeight: 0.35,
        },
  };
}
