import type {
  ChessMove,
  DrawbackRule,
  ExternalConstraintDrawbackRule,
  PositionAuthorityId,
  RuleEvidence,
} from "@drawbackengine/drawback-engine";
import {
  advancePublicPositionAuthority,
  createStandardChessPositionSnapshot,
  publicAuthorityLegalMoves,
  validatePublicPositionAuthoritySnapshot,
  type PublicPositionAuthoritySnapshot,
} from "@drawbackengine/chess-core";
import type { PlayerColor } from "@drawbackengine/shared";
import {
  resolveLikelihoodWeights,
  scoreMoveLogLikelihood,
} from "./likelihood.js";
import { logSumExp, normalizeLogProbabilities } from "./math.js";
import { canonicalHypothesisId } from "./parameters.js";
import type {
  DrawbackHypothesis,
  ExternalConstraintHypothesisSeed,
  HypothesisDistribution,
  HypothesisMoveOpportunity,
  HypothesisSeed,
  LikelihoodFeatures,
  MoveObservation,
  PredictionObservationResult,
  PredictionOpportunitySnapshot,
  PredictorOptions,
  PredictionSeeds,
  PredictionSeed,
  PredictionState,
  RerandomizedHypothesisSeed,
  RerandomizedOutcome,
} from "./types.js";

interface RuntimeHypothesis {
  readonly publicState: DrawbackHypothesis;
  readonly observe: (
    observation: MoveObservation,
    scoreLogLikelihood: (features: LikelihoodFeatures) => number,
  ) => RuntimeObservation;
}

interface RuntimeObservation {
  /**
   * `null` means public evidence is insufficient to know this mask. An empty
   * array is a known mask with no permitted continuation.
   */
  readonly legalMoves: readonly ChessMove[] | null;
  readonly next: RuntimeHypothesis;
}

function sameMove(left: ChessMove, right: ChessMove): boolean {
  return (
    left.from === right.from &&
    left.to === right.to &&
    left.promotion === right.promotion
  );
}

function immutableEvidence(evidence: readonly RuleEvidence[]): readonly RuleEvidence[] {
  return Object.freeze([...evidence]);
}

function assertSupportedParameterShape(
  value: unknown,
  label: string,
  ancestors = new WeakSet(),
): void {
  if (
    value === null
    || value === undefined
    || typeof value === "string"
    || typeof value === "boolean"
  ) {
    return;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError(`Unsupported ${label} shape: numbers must be finite`);
    }
    return;
  }
  if (typeof value !== "object") {
    throw new TypeError(
      `Unsupported ${label} shape: ${typeof value} values cannot be published`,
    );
  }
  if (ancestors.has(value)) {
    throw new TypeError(`Unsupported ${label} shape: cyclic references`);
  }
  if (Object.getOwnPropertySymbols(value).length !== 0) {
    throw new TypeError(`Unsupported ${label} shape: symbol properties`);
  }

  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      const propertyNames = Object.getOwnPropertyNames(value).filter(
        (propertyName) => propertyName !== "length",
      );
      for (const propertyName of propertyNames) {
        const index = Number(propertyName);
        if (
          !Number.isInteger(index)
          || index < 0
          || index >= 4_294_967_295
          || String(index) !== propertyName
        ) {
          throw new TypeError(
            `Unsupported ${label} shape: arrays cannot have named properties`,
          );
        }
      }
      if (propertyNames.length !== value.length) {
        throw new TypeError(
          `Unsupported ${label} shape: arrays cannot contain holes`,
        );
      }
      for (let index = 0; index < value.length; index += 1) {
        const descriptor = Object.getOwnPropertyDescriptor(
          value,
          String(index),
        );
        if (
          descriptor === undefined
          || !descriptor.enumerable
          || !("value" in descriptor)
        ) {
          throw new TypeError(
            `Unsupported ${label} shape: arrays require enumerable data entries`,
          );
        }
        assertSupportedParameterShape(descriptor.value, label, ancestors);
      }
      return;
    }

    const prototype = Object.getPrototypeOf(value) as object | null;
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError(
        `Unsupported ${label} shape: only plain objects and arrays are allowed`,
      );
    }
    for (const propertyName of Object.getOwnPropertyNames(value)) {
      const descriptor = Object.getOwnPropertyDescriptor(value, propertyName);
      if (
        descriptor === undefined
        || !descriptor.enumerable
        || !("value" in descriptor)
      ) {
        throw new TypeError(
          `Unsupported ${label} shape: objects require enumerable data properties`,
        );
      }
      assertSupportedParameterShape(descriptor.value, label, ancestors);
    }
  } finally {
    ancestors.delete(value);
  }
}

function deepFreezeParameter<T>(
  value: T,
  seen = new WeakSet(),
): T {
  if (value === null || typeof value !== "object" || seen.has(value)) {
    return value;
  }
  seen.add(value);
  for (const nested of Array.isArray(value) ? value : Object.values(value)) {
    deepFreezeParameter(nested, seen);
  }
  return Object.freeze(value);
}

function immutableParameterClone<T>(value: T): T {
  // Parameter values participate in canonical hypothesis IDs, so their
  // supported domain remains deliberately JSON-like and deeply immutable.
  const label = "predictor parameter";
  assertSupportedParameterShape(value, label);
  try {
    return deepFreezeParameter(structuredClone(value));
  } catch (error) {
    throw new TypeError(
      `Unsupported ${label} shape: structured cloning failed`,
      { cause: error },
    );
  }
}

function freezePlainStateSnapshot<T>(
  value: T,
  seen = new WeakSet(),
): T {
  if (value === null || typeof value !== "object" || seen.has(value)) {
    return value;
  }
  seen.add(value);
  if (Array.isArray(value)) {
    for (const nested of value) {
      freezePlainStateSnapshot(nested, seen);
    }
    return Object.freeze(value);
  }
  if (value instanceof Map) {
    for (const [key, nested] of value) {
      freezePlainStateSnapshot(key, seen);
      freezePlainStateSnapshot(nested, seen);
    }
    return value;
  }
  if (value instanceof Set) {
    for (const nested of value) {
      freezePlainStateSnapshot(nested, seen);
    }
    return value;
  }
  const prototype = Object.getPrototypeOf(value) as object | null;
  if (prototype === Object.prototype || prototype === null) {
    for (const nested of Object.values(value)) {
      freezePlainStateSnapshot(nested, seen);
    }
    return Object.freeze(value);
  }
  return value;
}

function detachedPublicState<T>(state: T): T {
  // Rule State is intentionally unconstrained. A detached structured clone
  // protects the private runtime even when a built-in container cannot be
  // made meaningfully immutable with Object.freeze.
  let detached: T;
  try {
    detached = structuredClone(state);
  } catch (error) {
    throw new TypeError(
      "Unable to publish predictor state: structured cloning failed",
      { cause: error },
    );
  }
  return freezePlainStateSnapshot(detached);
}

function appendEvidence(
  existing: readonly RuleEvidence[],
  additions: readonly RuleEvidence[],
): readonly RuleEvidence[] {
  return immutableEvidence([...existing, ...additions]);
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

function moveKey(move: ChessMove): string {
  return `${move.from}${move.to}${move.promotion ?? ""}`;
}

function validatedFilteredMoves(
  authorityMoves: readonly ChessMove[],
  filteredMoves: readonly ChessMove[],
  label: string,
): readonly ChessMove[] {
  const authorityByKey = new Map(
    authorityMoves.map((move) => [moveKey(move), move] as const),
  );
  const seen = new Set<string>();
  return Object.freeze(filteredMoves.map((move) => {
    const key = moveKey(move);
    const authorityMove = authorityByKey.get(key);
    if (authorityMove === undefined) {
      throw new Error(`${label} manufactured a move outside the authority set`);
    }
    if (seen.has(key)) {
      throw new Error(`${label} returned a duplicate authority move`);
    }
    seen.add(key);
    return authorityMove;
  }));
}

function supportsAuthority(
  seed: PredictionSeed,
  authorityId: PositionAuthorityId,
): boolean {
  const supported = seed.kind === "rerandomized"
    ? seed.supportedAuthorities
    : seed.kind === "external-turn-constraint"
      ? undefined
      : seed.rule.supportedAuthorities;
  return (supported ?? ["standard-chess/v1"]).includes(authorityId);
}

function validateObservation(
  observation: MoveObservation,
  authorityId: PositionAuthorityId,
  expectedPosition: MoveObservation["positionBefore"],
  expectedAuthorityPosition: PublicPositionAuthoritySnapshot,
): PublicPositionAuthoritySnapshot {
  if (observation.authorityId !== authorityId) {
    throw new Error("Observation authority does not match the predictor authority");
  }
  const authorityPosition = validatePublicPositionAuthoritySnapshot(
    observation.authorityPositionBefore,
  );
  if (
    authorityPosition.authorityId !== authorityId
    || authorityPosition.fen !== observation.positionBefore.fen
  ) {
    throw new Error(
      "Observation authority position does not match its public position",
    );
  }
  if (!sameAuthorityPosition(authorityPosition, expectedAuthorityPosition)) {
    throw new Error(
      "Observation authority position is discontinuous with predictor state",
    );
  }
  if (!samePosition(observation.positionBefore, expectedPosition)) {
    throw new Error("Observation is discontinuous with predictor state");
  }
  if (observation.positionBefore.turn !== observation.color) {
    throw new Error("Observation color must match the position side to move");
  }
  if (observation.move.color !== observation.color) {
    throw new Error("Observed move color must match observation color");
  }
  if (observation.positionAfter.ply !== observation.positionBefore.ply + 1) {
    throw new Error("Observation positions must advance exactly one ply");
  }
  if (
    observation.positionBefore.history.length !==
      observation.positionBefore.ply
    || observation.positionAfter.history.length !==
      observation.positionAfter.ply
  ) {
    throw new Error("Observation history length must match its position ply");
  }
  if (
    observation.positionAfter.turn === observation.color
    || observation.positionAfter.history.some((move, index) => {
      if (index < observation.positionBefore.history.length) {
        const priorMove = observation.positionBefore.history[index];
        return priorMove === undefined || !sameMove(move, priorMove);
      }
      return index === observation.positionBefore.history.length
        ? !sameMove(move, observation.move)
        : true;
    })
  ) {
    throw new Error("Observation after-position does not extend the public move");
  }
  const keys = new Set<string>();
  const canonicalMoves = publicAuthorityLegalMoves(authorityPosition);
  const canonicalByKey = new Map(
    canonicalMoves.map((move) => [moveKey(move), move] as const),
  );
  for (const move of observation.authorityLegalMoves) {
    if (move.color !== observation.color) {
      throw new Error("Authority move color must match observation color");
    }
    const key = moveKey(move);
    if (keys.has(key)) {
      throw new Error("Authority legal moves contain a duplicate move");
    }
    const canonical = canonicalByKey.get(key);
    if (canonical === undefined || !sameMoveDetails(canonical, move)) {
      throw new Error(
        "Authority legal moves do not match the complete position authority set",
      );
    }
    keys.add(key);
  }
  if (keys.size !== canonicalByKey.size) {
    throw new Error(
      "Authority legal moves do not match the complete position authority set",
    );
  }
  if (!keys.has(moveKey(observation.move))) {
    throw new Error("Observed move is outside the authority legal move set");
  }
  const canonicalObserved = canonicalByKey.get(moveKey(observation.move));
  if (
    canonicalObserved === undefined
    || !sameMoveDetails(canonicalObserved, observation.move)
  ) {
    throw new Error(
      "Observed move metadata does not match the position authority",
    );
  }
  const transition = advancePublicPositionAuthority(
    authorityPosition,
    observation.move,
  );
  if (transition.position.fen !== observation.positionAfter.fen) {
    throw new Error(
      "Observation after-position does not match the authority transition",
    );
  }
  return transition.position;
}

function samePosition(
  left: MoveObservation["positionBefore"],
  right: MoveObservation["positionBefore"],
): boolean {
  return (
    left.fen === right.fen
    && left.turn === right.turn
    && left.ply === right.ply
    && left.history.length === right.history.length
    && left.history.every((move, index) => {
      const other = right.history[index];
      return other !== undefined && sameMoveDetails(move, other);
    })
  );
}

function sameAuthorityPosition(
  left: PublicPositionAuthoritySnapshot,
  right: PublicPositionAuthoritySnapshot,
): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function immutablePosition(
  position: MoveObservation["positionBefore"],
): MoveObservation["positionBefore"] {
  return Object.freeze({
    fen: position.fen,
    turn: position.turn,
    ply: position.ply,
    history: Object.freeze(
      position.history.map((move) =>
        Object.freeze(structuredClone(move))
      ),
    ),
  });
}

function validatedOutcomes<Outcome>(
  outcomes: readonly RerandomizedOutcome<Outcome>[],
): readonly RerandomizedOutcome<Outcome>[] {
  if (outcomes.length === 0) {
    throw new RangeError("Rerandomized hypotheses require at least one outcome");
  }
  let total = 0;
  for (const { probability: outcomeProbability } of outcomes) {
    if (!Number.isFinite(outcomeProbability) || outcomeProbability <= 0) {
      throw new RangeError(
        "Rerandomized outcome probabilities must be finite and positive",
      );
    }
    total += outcomeProbability;
  }
  if (Math.abs(total - 1) > 1e-12) {
    throw new RangeError(
      "Rerandomized outcome probabilities must sum to one",
    );
  }
  return outcomes;
}

function consensusMoves(
  moveSets: readonly (readonly ChessMove[])[],
): readonly ChessMove[] | null {
  const first = moveSets[0];
  if (first === undefined) {
    throw new Error("Rerandomized opportunity requires at least one outcome");
  }
  return moveSets.every(
    (moves) =>
      moves.length === first.length
      && moves.every((move) =>
        first.some((candidate) => sameMove(candidate, move))
      ),
  )
    ? first
    : null;
}

function makeRerandomizedRuntime<State, Outcome>(
  color: PlayerColor,
  seed: RerandomizedHypothesisSeed<State, Outcome>,
  state: State,
  logProbability: number,
  evidence: readonly RuleEvidence[] = [],
  eliminated = false,
): RuntimeHypothesis {
  const parameters = Object.freeze({});
  const publicState: DrawbackHypothesis = Object.freeze({
    hypothesisId: canonicalHypothesisId(seed.drawbackId, parameters),
    drawbackId: seed.drawbackId,
    parameters,
    internalState: detachedPublicState(state),
    logProbability,
    eliminated,
    evidence: immutableEvidence(evidence),
  });

  return {
    publicState,
    observe: (observation, scoreLogLikelihood) => {
      if (publicState.eliminated) {
        return {
          legalMoves: null,
          next: makeRerandomizedRuntime(
            color,
            seed,
            state,
            Number.NEGATIVE_INFINITY,
            publicState.evidence,
            true,
          ),
        };
      }
      const context = {
        color,
        state,
        position: observation.positionBefore,
      };
      const outcomes = validatedOutcomes(seed.outcomes(context));
      const branchMoveSets: (readonly ChessMove[])[] = [];
      const compatibleLogs: number[] = [];
      let compatibleCount = 0;
      for (const { outcome, probability: outcomeProbability } of outcomes) {
        const allowed = validatedFilteredMoves(
          observation.authorityLegalMoves,
          seed.filterLegalMoves(
            context,
            outcome,
            [...observation.authorityLegalMoves],
          ),
          seed.name,
        );
        branchMoveSets.push(allowed);
        if (!allowed.some((candidate) => sameMove(candidate, observation.move))) {
          continue;
        }
        compatibleCount += 1;
        const branchLikelihood = scoreLogLikelihood({
          allowedMoveCount: allowed.length,
          ordinaryLegalMoveCount: observation.authorityLegalMoves.length,
          triggered: allowed.length !== observation.authorityLegalMoves.length,
          forced: allowed.length === 1,
          ...(observation.likelihoodSignals === undefined
            ? {}
            : { signals: observation.likelihoodSignals }),
        });
        if (!Number.isFinite(branchLikelihood)) {
          throw new RangeError("Likelihood scorer must return a finite number");
        }
        compatibleLogs.push(Math.log(outcomeProbability) + branchLikelihood);
      }
      if (compatibleLogs.length === 0) {
        const elimination: RuleEvidence = Object.freeze({
          ruleId: seed.drawbackId,
          kind: "eliminated",
          message:
            `${observation.move.san} is impossible under every current ` +
            `${seed.name} outcome.`,
          move: observation.move,
        });
        return {
          legalMoves: consensusMoves(branchMoveSets),
          next: makeRerandomizedRuntime(
            color,
            seed,
            state,
            Number.NEGATIVE_INFINITY,
            appendEvidence(publicState.evidence, [elimination]),
            true,
          ),
        };
      }
      const marginalizedLogLikelihood = logSumExp(compatibleLogs);
      const nextState = seed.applyObservedMove(
        {
          color,
          state,
          position: observation.positionBefore,
          positionAfterMove: observation.positionAfter,
        },
        observation.move,
      );
      const likelihoodEvidence: RuleEvidence = Object.freeze({
        ruleId: seed.drawbackId,
        kind: "likelihood",
        message:
          `${observation.move.san} is compatible with ` +
          `${String(compatibleCount)} of ${String(outcomes.length)} ` +
          `${seed.name} outcomes.`,
        move: observation.move,
        weight: marginalizedLogLikelihood,
      });
      return {
        legalMoves: consensusMoves(branchMoveSets),
        next: makeRerandomizedRuntime(
          color,
          seed,
          nextState,
          publicState.logProbability + marginalizedLogLikelihood,
          appendEvidence(publicState.evidence, [likelihoodEvidence]),
        ),
      };
    },
  };
}

function makeRuntime<State, Parameters extends Record<string, unknown>>(
  color: PlayerColor,
  seed: HypothesisSeed<State, Parameters>,
  initialState: State,
  parameters: Readonly<Parameters>,
  logProbability: number,
  evidence: readonly RuleEvidence[] = [],
  eliminated = false,
): RuntimeHypothesis {
  const publicParameters = immutableParameterClone(parameters);
  const publicState: DrawbackHypothesis = Object.freeze({
    hypothesisId: canonicalHypothesisId(seed.rule.id, publicParameters),
    drawbackId: seed.rule.id,
    parameters: publicParameters,
    internalState: detachedPublicState(initialState),
    logProbability,
    eliminated,
    evidence: immutableEvidence(evidence),
  });

  return {
    publicState,
    observe: (observation, scoreLogLikelihood) => {
      if (publicState.eliminated) {
        return {
          legalMoves: null,
          next: makeRuntime(
            color,
            seed,
            initialState,
            parameters,
            Number.NEGATIVE_INFINITY,
            publicState.evidence,
            true,
          ),
        };
      }

      const moveContext = {
        color,
        parameters,
        state: initialState,
        position: observation.positionBefore,
      };
      const startOfTurnLoss = seed.rule.checkStartOfTurnLoss(moveContext);
      if (startOfTurnLoss !== null) {
        const elimination: RuleEvidence = Object.freeze({
          ruleId: seed.rule.id,
          kind: "eliminated",
          message:
            `${observation.move.san} could not occur because ${seed.rule.name} ` +
            `would already have lost: ${startOfTurnLoss.reason}`,
          move: observation.move,
        });
        return {
          legalMoves: Object.freeze([]),
          next: makeRuntime(
            color,
            seed,
            initialState,
            parameters,
            Number.NEGATIVE_INFINITY,
            appendEvidence(publicState.evidence, [elimination]),
            true,
          ),
        };
      }

      const allowed = validatedFilteredMoves(
        observation.authorityLegalMoves,
        seed.rule.filterLegalMoves(
          moveContext,
          [...observation.authorityLegalMoves],
        ),
        seed.rule.name,
      );
      if (!allowed.some((candidate) => sameMove(candidate, observation.move))) {
        const elimination: RuleEvidence = Object.freeze({
          ruleId: seed.rule.id,
          kind: "eliminated",
          message: `${observation.move.san} is impossible under ${seed.rule.name}.`,
          move: observation.move,
        });
        return {
          legalMoves: allowed,
          next: makeRuntime(
            color,
            seed,
            initialState,
            parameters,
            Number.NEGATIVE_INFINITY,
            appendEvidence(publicState.evidence, [elimination]),
            true,
          ),
        };
      }

      const explanation =
        seed.rule.explainMove?.(
          moveContext,
          observation.move,
        ) ?? [];
      const triggered = allowed.length !== observation.authorityLegalMoves.length;
      const forced = allowed.length === 1;
      const logLikelihood = scoreLogLikelihood({
        allowedMoveCount: allowed.length,
        ordinaryLegalMoveCount: observation.authorityLegalMoves.length,
        triggered,
        forced,
        ...(observation.likelihoodSignals === undefined
          ? {}
          : { signals: observation.likelihoodSignals }),
      });
      if (!Number.isFinite(logLikelihood)) {
        throw new RangeError("Likelihood scorer must return a finite number");
      }
      const likelihoodEvidence: readonly RuleEvidence[] =
        logLikelihood === 0
          ? []
          : [
              Object.freeze({
                ruleId: seed.rule.id,
                kind: "likelihood" as const,
                message: triggered
                  ? `Observed move likelihood under ${seed.rule.name}.`
                  : `Low-weight no-trigger likelihood under ${seed.rule.name}.`,
                move: observation.move,
                weight: logLikelihood,
              }),
            ];
      const nextState = seed.rule.applyMove(
        {
          color,
          parameters,
          state: initialState,
          position: observation.positionBefore,
          positionAfterMove: observation.positionAfter,
        },
        observation.move,
      );
      return {
        legalMoves: allowed,
        next: makeRuntime(
          color,
          seed,
          nextState,
          parameters,
          publicState.logProbability + logLikelihood,
          appendEvidence(publicState.evidence, [
            ...explanation,
            ...likelihoodEvidence,
          ]),
        ),
      };
    },
  };
}

function makeExternalRuntime<
  State,
  Parameters extends Record<string, unknown>,
>(
  color: PlayerColor,
  seed: ExternalConstraintHypothesisSeed<State, Parameters>,
  state: State,
  parameters: Readonly<Parameters>,
  logProbability: number,
  evidence: readonly RuleEvidence[] = [],
  eliminated = false,
): RuntimeHypothesis {
  const publicParameters = immutableParameterClone(parameters);
  const publicState: DrawbackHypothesis = Object.freeze({
    hypothesisId: canonicalHypothesisId(seed.rule.id, publicParameters),
    drawbackId: seed.rule.id,
    parameters: publicParameters,
    internalState: detachedPublicState(state),
    logProbability,
    eliminated,
    evidence: immutableEvidence(evidence),
  });
  const context = (observation: MoveObservation) => ({
    color,
    parameters,
    state,
    position: observation.positionBefore,
  });

  return {
    publicState,
    observe: (observation, scoreLogLikelihood) => {
      if (publicState.eliminated) {
        return {
          legalMoves: null,
          next: makeExternalRuntime(
            color,
            seed,
            state,
            parameters,
            Number.NEGATIVE_INFINITY,
            publicState.evidence,
            true,
          ),
        };
      }
      const moveContext = context(observation);
      const loss = seed.rule.checkStartOfTurnLoss(moveContext);
      if (loss !== null) {
        const elimination: RuleEvidence = Object.freeze({
          ruleId: seed.rule.id,
          kind: "eliminated",
          message:
            `${observation.move.san} could not occur because ${seed.rule.name} ` +
            `would already have lost: ${loss.reason}`,
          move: observation.move,
        });
        return {
          legalMoves: Object.freeze([]),
          next: makeExternalRuntime(
            color,
            seed,
            state,
            parameters,
            Number.NEGATIVE_INFINITY,
            appendEvidence(publicState.evidence, [elimination]),
            true,
          ),
        };
      }

      const nextState = seed.rule.applyMove(
        {
          color,
          parameters,
          state,
          position: observation.positionBefore,
          positionAfterMove: observation.positionAfter,
        },
        observation.move,
      );
      const constraint = observation.externalConstraint;
      // Missing public evaluator data is unknown evidence. Advance only public
      // state so history stays aligned; do not eliminate or change probability.
      if (constraint === undefined) {
        return {
          legalMoves: null,
          next: makeExternalRuntime(
            color,
            seed,
            nextState,
            parameters,
            publicState.logProbability,
            publicState.evidence,
          ),
        };
      }

      const legalMoves = validatedFilteredMoves(
        observation.authorityLegalMoves,
        seed.rule.filterLegalMovesWithConstraint(
          moveContext,
          [...observation.authorityLegalMoves],
          constraint,
        ),
        seed.rule.name,
      );
      if (!legalMoves.some((candidate) => sameMove(candidate, observation.move))) {
        const elimination: RuleEvidence = Object.freeze({
          ruleId: seed.rule.id,
          kind: "eliminated",
          message: `${observation.move.san} is impossible under ${seed.rule.name}.`,
          move: observation.move,
        });
        return {
          legalMoves,
          next: makeExternalRuntime(
            color,
            seed,
            state,
            parameters,
            Number.NEGATIVE_INFINITY,
            appendEvidence(publicState.evidence, [elimination]),
            true,
          ),
        };
      }

      const triggered =
        legalMoves.length !== observation.authorityLegalMoves.length;
      const moveLogLikelihood = scoreLogLikelihood({
        allowedMoveCount: legalMoves.length,
        ordinaryLegalMoveCount: observation.authorityLegalMoves.length,
        triggered,
        forced: legalMoves.length === 1,
        ...(observation.likelihoodSignals === undefined
          ? {}
          : { signals: observation.likelihoodSignals }),
      });
      if (!Number.isFinite(moveLogLikelihood)) {
        throw new RangeError("Likelihood scorer must return a finite number");
      }
      const explanation =
        seed.rule.explainMove?.(moveContext, observation.move, constraint) ?? [];
      const likelihoodEvidence: readonly RuleEvidence[] =
        moveLogLikelihood === 0
          ? []
          : [
              Object.freeze({
                ruleId: seed.rule.id,
                kind: "likelihood" as const,
                message: triggered
                  ? `Observed move likelihood under ${seed.rule.name}.`
                  : `Low-weight no-trigger likelihood under ${seed.rule.name}.`,
                move: observation.move,
                weight: moveLogLikelihood,
              }),
            ];
      return {
        legalMoves,
        next: makeExternalRuntime(
          color,
          seed,
          nextState,
          parameters,
          publicState.logProbability + moveLogLikelihood,
          appendEvidence(publicState.evidence, [
            ...explanation,
            ...likelihoodEvidence,
          ]),
        ),
      };
    },
  };
}

function initializeDistribution(
  color: PlayerColor,
  seeds: readonly PredictionSeed[],
  position: MoveObservation["positionBefore"],
): readonly RuntimeHypothesis[] {
  if (seeds.length === 0) {
    throw new RangeError(`${color} requires at least one drawback hypothesis`);
  }
  const hypothesisIds = new Set<string>();
  for (const seed of seeds) {
    const id = seed.kind === "rerandomized"
      ? canonicalHypothesisId(seed.drawbackId, {})
      : canonicalHypothesisId(seed.rule.id, seed.parameters);
    if (hypothesisIds.has(id)) {
      throw new RangeError(`Duplicate hypothesis: ${id}`);
    }
    hypothesisIds.add(id);
  }
  const rawLogs = seeds.map((seed) => {
    if (
      seed.priorProbability !== undefined &&
      seed.historicalFrequency !== undefined
    ) {
      throw new RangeError(
        "Specify either priorProbability or historicalFrequency, not both",
      );
    }
    const prior = seed.historicalFrequency ?? seed.priorProbability ?? 1;
    if (!Number.isFinite(prior) || prior <= 0) {
      throw new RangeError("priorProbability must be finite and greater than zero");
    }
    return Math.log(prior);
  });
  const normalized = normalizeLogProbabilities(rawLogs);
  return seeds.map((seed, index) => {
    const logProbability = normalized[index];
    if (logProbability === undefined) {
      throw new Error("Probability normalization did not preserve hypothesis count");
    }
    if (seed.kind === "rerandomized") {
      const state = seed.initialize({ color, position });
      return makeRerandomizedRuntime(
        color,
        seed,
        state,
        logProbability,
      );
    }
    if (seed.kind === "external-turn-constraint") {
      const parameters = immutableParameterClone(seed.parameters);
      const state = seed.rule.initialize({
        color,
        parameters,
        position,
      });
      return makeExternalRuntime(
        color,
        seed,
        state,
        parameters,
        logProbability,
      );
    }
    const parameters = immutableParameterClone(seed.parameters);
    const state = seed.rule.initialize({
      color,
      parameters,
      position,
    });
    return makeRuntime(
      color,
      seed,
      state,
      parameters,
      logProbability,
    );
  });
}

function publicDistribution(
  runtimes: readonly RuntimeHypothesis[],
): HypothesisDistribution {
  return Object.freeze({
    hypotheses: Object.freeze(runtimes.map((runtime) => runtime.publicState)),
  });
}

function publicOpportunity(
  observation: MoveObservation,
  runtimes: readonly RuntimeHypothesis[],
  runtimeObservations: readonly RuntimeObservation[],
): PredictionOpportunitySnapshot {
  if (runtimes.length !== runtimeObservations.length) {
    throw new Error("Opportunity evaluation did not preserve hypothesis count");
  }
  const ordinaryLegalMoveCount = observation.authorityLegalMoves.length;
  const hypotheses = runtimes.map((runtime, index): HypothesisMoveOpportunity => {
    const runtimeObservation = runtimeObservations[index];
    if (runtimeObservation === undefined) {
      throw new Error("Opportunity evaluation did not preserve hypothesis order");
    }
    const identity = {
      hypothesisIndex: index,
      drawbackId: runtime.publicState.drawbackId,
      ordinaryLegalMoveCount,
    };
    if (runtime.publicState.eliminated) {
      return Object.freeze({
        ...identity,
        status: "eliminated",
        allowedMoveCount: null,
        allowedMoveFraction: null,
        triggered: null,
        forced: null,
        observedMoveLegal: null,
      });
    }
    if (runtimeObservation.legalMoves === null) {
      return Object.freeze({
        ...identity,
        status: "unknown",
        allowedMoveCount: null,
        allowedMoveFraction: null,
        triggered: null,
        forced: null,
        observedMoveLegal: null,
      });
    }

    const allowedMoves = runtimeObservation.legalMoves;
    const allowedMoveCount = allowedMoves.length;
    return Object.freeze({
      ...identity,
      status: "known",
      allowedMoveCount,
      allowedMoveFraction:
        ordinaryLegalMoveCount === 0
          ? 0
          : allowedMoveCount / ordinaryLegalMoveCount,
      triggered: allowedMoveCount !== ordinaryLegalMoveCount,
      forced: allowedMoveCount === 1,
      observedMoveLegal: allowedMoves.some((candidate) =>
        sameMove(candidate, observation.move)
      ),
    });
  });
  return Object.freeze({
    color: observation.color,
    hypotheses: Object.freeze(hypotheses),
  });
}

function renormalize(runtimes: readonly RuntimeHypothesis[]): readonly RuntimeHypothesis[] {
  const normalized = normalizeLogProbabilities(
    runtimes.map((runtime) => runtime.publicState.logProbability),
  );
  return runtimes.map((runtime, index) => {
    const nextLogProbability = normalized[index];
    if (nextLogProbability === undefined) {
      throw new Error("Probability normalization did not preserve hypothesis count");
    }
    if (nextLogProbability === runtime.publicState.logProbability) {
      return runtime;
    }
    return {
      ...runtime,
      publicState: Object.freeze({
        ...runtime.publicState,
        logProbability: nextLogProbability,
      }),
    };
  });
}

export class SymbolicPredictor {
  #white: readonly RuntimeHypothesis[];
  #black: readonly RuntimeHypothesis[];
  #expectedPosition: MoveObservation["positionBefore"];
  #expectedAuthorityPosition: PublicPositionAuthoritySnapshot;
  readonly #scoreLogLikelihood: (features: LikelihoodFeatures) => number;
  readonly #authorityId: PositionAuthorityId;

  public constructor(
    seeds: PredictionSeeds,
    initialPosition: MoveObservation["positionBefore"],
    options: PredictorOptions = {},
  ) {
    this.#authorityId = options.authorityId ?? "standard-chess/v1";
    if (initialPosition.history.length !== initialPosition.ply) {
      throw new TypeError(
        "Initial predictor history length must match its public ply.",
      );
    }
    const fenTurn = initialPosition.fen.trim().split(/\s+/u)[1];
    if (
      (fenTurn === "w" ? "white" : fenTurn === "b" ? "black" : null)
      !== initialPosition.turn
    ) {
      throw new TypeError(
        "Initial predictor turn must match its public FEN.",
      );
    }
    const authorityPosition = options.initialAuthorityPosition === undefined
      ? this.#authorityId === "standard-chess/v1"
        ? createStandardChessPositionSnapshot(initialPosition.fen)
        : null
      : validatePublicPositionAuthoritySnapshot(
          options.initialAuthorityPosition,
        );
    if (authorityPosition === null) {
      throw new TypeError(
        "capturable-king/v1 predictors require initialAuthorityPosition.",
      );
    }
    if (
      authorityPosition.authorityId !== this.#authorityId
      || authorityPosition.fen !== initialPosition.fen
    ) {
      throw new TypeError(
        "Initial authority position does not match predictor authority and FEN.",
      );
    }
    this.#expectedPosition = immutablePosition(initialPosition);
    this.#expectedAuthorityPosition = authorityPosition;
    this.#white = initializeDistribution(
      "white",
      seeds.white.filter((seed) => supportsAuthority(seed, this.#authorityId)),
      initialPosition,
    );
    this.#black = initializeDistribution(
      "black",
      seeds.black.filter((seed) => supportsAuthority(seed, this.#authorityId)),
      initialPosition,
    );
    if (options.scoreLogLikelihood !== undefined) {
      this.#scoreLogLikelihood = options.scoreLogLikelihood;
    } else {
      const weights = resolveLikelihoodWeights(options.likelihoodWeights);
      this.#scoreLogLikelihood = (features) =>
        scoreMoveLogLikelihood(features, weights);
    }
  }

  public get state(): PredictionState {
    return Object.freeze({
      white: publicDistribution(this.#white),
      black: publicDistribution(this.#black),
    });
  }

  public observe(observation: MoveObservation): PredictionState {
    return this.observeWithOpportunities(observation).state;
  }

  public observeWithOpportunities(
    observation: MoveObservation,
  ): PredictionObservationResult {
    const nextAuthorityPosition = validateObservation(
      observation,
      this.#authorityId,
      this.#expectedPosition,
      this.#expectedAuthorityPosition,
    );
    const activeRuntimes =
      observation.color === "white" ? this.#white : this.#black;
    const runtimeObservations = activeRuntimes.map((runtime) =>
      runtime.observe(observation, this.#scoreLogLikelihood)
    );
    const opportunity = publicOpportunity(
      observation,
      activeRuntimes,
      runtimeObservations,
    );
    const nextRuntimes = renormalize(
      runtimeObservations.map(({ next }) => next),
    );
    if (observation.color === "white") {
      this.#white = nextRuntimes;
    } else {
      this.#black = nextRuntimes;
    }
    this.#expectedPosition = immutablePosition(observation.positionAfter);
    this.#expectedAuthorityPosition = nextAuthorityPosition;
    return Object.freeze({
      state: this.state,
      opportunity,
    });
  }
}

export function asHypothesisSeed<State, Parameters extends Record<string, unknown>>(
  rule: DrawbackRule<State, Parameters>,
  parameters: Readonly<Parameters>,
  priorProbability?: number,
  historicalFrequency?: number,
): HypothesisSeed<unknown, Record<string, unknown>> {
  const erasedRule = rule as DrawbackRule<unknown, Record<string, unknown>>;
  return {
    rule: erasedRule,
    parameters,
    ...(priorProbability === undefined ? {} : { priorProbability }),
    ...(historicalFrequency === undefined ? {} : { historicalFrequency }),
  };
}

export function asExternalConstraintHypothesisSeed<
  State,
  Parameters extends Record<string, unknown>,
>(
  rule: ExternalConstraintDrawbackRule<State, Parameters>,
  parameters: Readonly<Parameters>,
  priorProbability?: number,
  historicalFrequency?: number,
): ExternalConstraintHypothesisSeed<unknown, Record<string, unknown>> {
  const erasedRule = rule as ExternalConstraintDrawbackRule<
    unknown,
    Record<string, unknown>
  >;
  return {
    kind: "external-turn-constraint",
    rule: erasedRule,
    parameters,
    ...(priorProbability === undefined ? {} : { priorProbability }),
    ...(historicalFrequency === undefined ? {} : { historicalFrequency }),
  };
}

export function asRerandomizedHypothesisSeed<State, Outcome>(
  seed: RerandomizedHypothesisSeed<State, Outcome>,
): RerandomizedHypothesisSeed<unknown, unknown> {
  return seed;
}
