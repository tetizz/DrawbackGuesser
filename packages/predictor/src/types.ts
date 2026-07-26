import type {
  ChessMove,
  DrawbackRule,
  ExternalConstraintDrawbackRule,
  ExternalTurnConstraint,
  PositionAuthorityId,
  PositionView,
  RuleEvidence,
} from "@drawbackengine/drawback-engine";
import type {
  PublicPositionAuthoritySnapshot,
} from "@drawbackengine/chess-core";
import type { PlayerColor } from "@drawbackengine/shared";

export interface DrawbackHypothesis {
  readonly hypothesisId: string;
  readonly drawbackId: string;
  readonly parameters: Readonly<Record<string, unknown>>;
  readonly internalState: unknown;
  readonly logProbability: number;
  readonly eliminated: boolean;
  readonly evidence: readonly RuleEvidence[];
}

export interface ParameterVariant<Parameters extends object> {
  readonly parameters: Readonly<Parameters>;
  readonly weight?: number;
}

export interface RulePosterior {
  readonly drawbackId: string;
  readonly logProbability: number;
  readonly probability: number;
  readonly eliminated: boolean;
  readonly liveVariantCount: number;
  readonly variantCount: number;
}

export interface ParameterValuePosterior {
  readonly canonicalValue: string;
  readonly value: unknown;
  readonly probability: number;
  readonly conditionalProbability: number;
  readonly variantCount: number;
}

export interface ParameterPosterior {
  readonly drawbackId: string;
  readonly parameter: string;
  readonly drawbackProbability: number;
  readonly coveredProbability: number;
  readonly values: readonly ParameterValuePosterior[];
}

export interface HypothesisDistribution {
  readonly hypotheses: readonly DrawbackHypothesis[];
}

export interface PredictionState {
  readonly white: HypothesisDistribution;
  readonly black: HypothesisDistribution;
}

/**
 * Public move data available to a real predictor. It deliberately contains no
 * true drawback, secret parameters, or authoritative drawback state.
 */
export interface MoveObservation {
  readonly authorityId: PositionAuthorityId;
  readonly authorityPositionBefore: PublicPositionAuthoritySnapshot;
  readonly color: PlayerColor;
  readonly positionBefore: PositionView;
  readonly positionAfter: PositionView;
  readonly move: ChessMove;
  readonly authorityLegalMoves: readonly ChessMove[];
  /**
   * Public, reproducible evaluator output for this position. Its absence means
   * "not observed", not that an evaluator-backed hypothesis is impossible.
   */
  readonly externalConstraint?: ExternalTurnConstraint;
  readonly likelihoodSignals?: Readonly<MoveLikelihoodSignals>;
}

export interface MoveLikelihoodSignals {
  readonly humanMoveLogLikelihood?: number;
  readonly engineQualityLogLikelihood?: number;
  readonly playerStrengthLogLikelihood?: number;
  readonly timeUsageLogLikelihood?: number;
}

export interface LikelihoodFeatures {
  readonly allowedMoveCount: number;
  readonly ordinaryLegalMoveCount: number;
  readonly triggered: boolean;
  readonly forced: boolean;
  readonly signals?: Readonly<MoveLikelihoodSignals>;
}

export interface LikelihoodWeights {
  readonly allowedMoveCount: number;
  readonly forcedMove: number;
  readonly humanMove: number;
  readonly engineQuality: number;
  readonly playerStrength: number;
  readonly timeUsage: number;
  readonly noTriggerEvidenceScale: number;
}

export interface PredictorOptions {
  readonly authorityId?: PositionAuthorityId;
  readonly initialAuthorityPosition?: PublicPositionAuthoritySnapshot;
  readonly likelihoodWeights?: Partial<LikelihoodWeights>;
  readonly scoreLogLikelihood?: (features: LikelihoodFeatures) => number;
}

export interface HypothesisSeed<State, Parameters extends Record<string, unknown>> {
  readonly kind?: "exact";
  readonly rule: DrawbackRule<State, Parameters>;
  readonly parameters: Readonly<Parameters>;
  readonly priorProbability?: number;
  readonly historicalFrequency?: number;
}

export interface ExternalConstraintHypothesisSeed<
  State,
  Parameters extends Record<string, unknown>,
> {
  readonly kind: "external-turn-constraint";
  readonly rule: ExternalConstraintDrawbackRule<State, Parameters>;
  readonly parameters: Readonly<Parameters>;
  readonly priorProbability?: number;
  readonly historicalFrequency?: number;
}

export interface RerandomizedOutcome<Outcome> {
  readonly outcome: Readonly<Outcome>;
  readonly probability: number;
}

export interface RerandomizedContext<State> {
  readonly color: PlayerColor;
  readonly state: Readonly<State>;
  readonly position: PositionView;
}

export interface RerandomizedTransitionContext<State>
  extends RerandomizedContext<State> {
  readonly positionAfterMove: PositionView;
}

export interface RerandomizedHypothesisSeed<State, Outcome> {
  readonly kind: "rerandomized";
  readonly drawbackId: string;
  readonly name: string;
  readonly supportedAuthorities?: readonly PositionAuthorityId[];
  readonly priorProbability?: number;
  readonly historicalFrequency?: number;
  initialize(context: {
    readonly color: PlayerColor;
    readonly position: PositionView;
  }): State;
  outcomes(
    context: RerandomizedContext<State>,
  ): readonly RerandomizedOutcome<Outcome>[];
  filterLegalMoves(
    context: RerandomizedContext<State>,
    outcome: Readonly<Outcome>,
    moves: readonly ChessMove[],
  ): readonly ChessMove[];
  applyObservedMove(
    context: RerandomizedTransitionContext<State>,
    move: ChessMove,
  ): State;
}

export type PredictionSeed =
  | HypothesisSeed<unknown, Record<string, unknown>>
  | ExternalConstraintHypothesisSeed<unknown, Record<string, unknown>>
  | RerandomizedHypothesisSeed<unknown, unknown>;

export interface PredictionSeeds {
  readonly white: readonly PredictionSeed[];
  readonly black: readonly PredictionSeed[];
}
