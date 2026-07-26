import { isDeepStrictEqual } from "node:util";
import {
  advancePublicPositionAuthority,
  createStandardChessPositionSnapshot,
  GameSession,
  publicAuthorityLegalMoves,
  type PublicPositionAuthoritySnapshot,
  type SessionResult,
} from "@drawbackengine/chess-core";
import {
  canonicalMoveUci,
  OBSERVED_BLINDED_SQUARES,
  OBSERVED_CENTRAL_SQUARES,
  resolvePreparedExecutableRule,
  unrestrictedRule,
  type ChessMove,
  type ExternalExecutableDrawbackRule,
  type PositionView,
  type PreparedExecutableDrawbackRule,
  type PromotionPiece,
} from "@drawbackengine/drawback-engine";
import type {
  PrivateSimulationTraceRecord,
  TraceActiveSecret,
} from "@drawbackengine/simulation-trace";
import { Mulberry32, type PlayerColor } from "@drawbackengine/shared";

const SEED_PARAMETER_RULES = new Set([
  "gambler",
  "colorblind",
  "hand-and-brainless",
  "obsession",
  "winds-of-fate",
]);
const SQUARE_PARAMETER_RULES = new Set([
  "untitled-duck-drawback",
  "active-volcano",
  "comfort-zone",
  "blinded-by-the-sun",
]);
const BOARD_SQUARE = /^[a-h][1-8]$/u;

interface PrivateRuleRuntime {
  readonly color: PlayerColor;
  readonly rule: PreparedExecutableDrawbackRule;
  readonly parameters: Readonly<unknown>;
  readonly state: Readonly<unknown>;
}

function isExternalRule(
  rule: PreparedExecutableDrawbackRule,
): rule is ExternalExecutableDrawbackRule {
  return "kind" in rule;
}

function turnFromFen(fen: string): PlayerColor {
  return fen.split(" ")[1] === "w" ? "white" : "black";
}

function firstSecret(
  trace: PrivateSimulationTraceRecord,
  color: PlayerColor,
): TraceActiveSecret | null {
  return trace.plies.find((ply) => ply.color === color)?.activeSecret ?? null;
}

function expectedParameterKeys(ruleId: string): readonly string[] {
  return SQUARE_PARAMETER_RULES.has(ruleId)
    ? ["square"]
    : ruleId === "just-passing-through"
      ? ["rank"]
      : SEED_PARAMETER_RULES.has(ruleId)
        ? ["seed"]
        : ruleId === "crenellations"
          ? ["squareColor"]
          : ruleId === "theocracy"
            ? ["captureParity"]
            : [];
}

function assertParameterShape(
  value: unknown,
  ruleId: string,
  color: PlayerColor,
): void {
  if (
    typeof value !== "object"
    || value === null
    || Array.isArray(value)
  ) {
    throw new TypeError(
      `${color} hiddenParameters must be a JSON object.`,
    );
  }
  const parameters = value as Readonly<Record<string, unknown>>;
  const keys = Object.keys(parameters).sort();
  const expectedKeys = expectedParameterKeys(ruleId);
  if (
    keys.length !== expectedKeys.length
    || keys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new TypeError(
      `${color} hiddenParameters keys do not match rule ${ruleId}.`,
    );
  }
  if (SQUARE_PARAMETER_RULES.has(ruleId)) {
    const square = parameters["square"];
    const validDomain =
      ruleId === "active-volcano" || ruleId === "comfort-zone"
        ? (OBSERVED_CENTRAL_SQUARES as readonly unknown[]).includes(square)
        : ruleId === "blinded-by-the-sun"
          ? (OBSERVED_BLINDED_SQUARES as readonly unknown[]).includes(square)
          : typeof square === "string" && BOARD_SQUARE.test(square);
    if (!validDomain) {
      throw new TypeError(
        `${color} hiddenParameters.square is invalid for rule ${ruleId}.`,
      );
    }
  } else if (
    ruleId === "just-passing-through"
    && (
      !Number.isSafeInteger(parameters["rank"])
      || Number(parameters["rank"]) < 1
      || Number(parameters["rank"]) > 8
    )
  ) {
    throw new TypeError(`${color} hiddenParameters.rank must be 1 through 8.`);
  } else if (
    SEED_PARAMETER_RULES.has(ruleId)
    && (
      !Number.isSafeInteger(parameters["seed"])
      || Number(parameters["seed"]) < 0
      || Number(parameters["seed"]) > 0xffff_ffff
    )
  ) {
    throw new TypeError(`${color} hiddenParameters.seed must be uint32.`);
  } else if (
    ruleId === "crenellations"
    && parameters["squareColor"] !== "dark"
    && parameters["squareColor"] !== "light"
  ) {
    throw new TypeError(
      `${color} hiddenParameters.squareColor is invalid.`,
    );
  } else if (
    ruleId === "theocracy"
    && parameters["captureParity"] !== "odd"
    && parameters["captureParity"] !== "even"
  ) {
    throw new TypeError(
      `${color} hiddenParameters.captureParity is invalid.`,
    );
  }
}

function initializeRuntime(
  trace: PrivateSimulationTraceRecord,
  color: PlayerColor,
): PrivateRuleRuntime | null {
  const secret = firstSecret(trace, color);
  const revealedId =
    color === "white" ? trace.drawbacks.white : trace.drawbacks.black;
  const rule = resolvePreparedExecutableRule(revealedId);
  if (secret === null && expectedParameterKeys(rule.id).length > 0) {
    return null;
  }
  const hiddenParameters = secret?.hiddenParameters ?? {};
  assertParameterShape(hiddenParameters, rule.id, color);
  const parameters =
    isExternalRule(rule) || rule.validateParameters === undefined
      ? structuredClone(hiddenParameters)
      : rule.validateParameters(hiddenParameters);
  const position: PositionView = {
    fen: trace.initialFen,
    turn: turnFromFen(trace.initialFen),
    ply: 0,
    history: [],
  };
  return {
    color,
    rule,
    parameters: parameters as Readonly<unknown>,
    state: rule.initialize({
      color,
      parameters: parameters as Readonly<unknown>,
      position,
    }) as Readonly<unknown>,
  };
}

function commandFromMove(move: ChessMove): {
  readonly from: string;
  readonly to: string;
  readonly promotion?: PromotionPiece;
} {
  return {
    from: move.from,
    to: move.to,
    ...(move.promotion === undefined ? {} : { promotion: move.promotion }),
  };
}

function observedMove(
  snapshot: PublicPositionAuthoritySnapshot,
  uci: string,
): ChessMove {
  const move = publicAuthorityLegalMoves(snapshot).find(
    (candidate) => canonicalMoveUci(candidate) === uci,
  );
  if (move === undefined) {
    throw new RangeError(`Trace move ${uci} is not authority-legal.`);
  }
  return move;
}

function exactMoveMask(
  rule: PreparedExecutableDrawbackRule,
  runtime: PrivateRuleRuntime,
  position: PositionView,
  ordinaryMoves: readonly ChessMove[],
  constraint:
    PrivateSimulationTraceRecord["plies"][number]["publicEvaluatorConstraint"],
): readonly ChessMove[] {
  const context = {
    color: runtime.color,
    parameters: runtime.parameters,
    state: runtime.state,
    position,
  };
  if (isExternalRule(rule)) {
    if (constraint === null) {
      throw new TypeError(
        `Trace rule ${rule.id} requires a public evaluator constraint.`,
      );
    }
    return rule.filterLegalMovesWithConstraint(
      context,
      ordinaryMoves,
      constraint,
    );
  }
  return rule.filterLegalMoves(context, ordinaryMoves);
}

function sortedMask(moves: readonly ChessMove[]): readonly string[] {
  const mask = moves.map(canonicalMoveUci).sort();
  if (new Set(mask).size !== mask.length) {
    throw new TypeError("Drawback rule returned duplicate legal moves.");
  }
  return mask;
}

function assertEqualJson(
  actual: unknown,
  expected: unknown,
  label: string,
): void {
  if (!isDeepStrictEqual(actual, expected)) {
    throw new TypeError(`${label} does not match executable rule replay.`);
  }
}

function assertEqualMask(
  actual: readonly string[],
  expected: readonly string[],
  label: string,
): void {
  if (
    actual.length !== expected.length
    || actual.some((move, index) => move !== expected[index])
  ) {
    throw new TypeError(`${label} does not match executable rule replay.`);
  }
}

function applyToRuntime(
  runtime: PrivateRuleRuntime,
  position: PositionView,
  positionAfterMove: PositionView,
  move: ChessMove,
): PrivateRuleRuntime {
  return {
    ...runtime,
    state: runtime.rule.applyMove(
      {
        color: runtime.color,
        parameters: runtime.parameters,
        state: runtime.state,
        position,
        positionAfterMove,
      },
      move,
    ) as Readonly<unknown>,
  };
}

function assertFinalResult(
  trace: PrivateSimulationTraceRecord,
  standardResult: SessionResult,
  white: PrivateRuleRuntime | null,
  black: PrivateRuleRuntime | null,
  snapshot: PublicPositionAuthoritySnapshot,
  history: readonly ChessMove[],
): void {
  const color = turnFromFen(trace.finalFen);
  const runtime = color === "white" ? white : black;
  if (runtime === null) {
    throw new TypeError(
      `Trace ${trace.gameId} final result cannot be authenticated without ${color} parameters.`,
    );
  }
  if (isExternalRule(runtime.rule)) {
    throw new TypeError(
      `Trace ${trace.gameId} final result requires an unrecorded evaluator constraint for ${runtime.rule.id}.`,
    );
  }
  const position: PositionView = {
    fen: trace.finalFen,
    turn: color,
    ply: history.length,
    history,
  };
  let loss = runtime.rule.checkStartOfTurnLoss({
    color,
    parameters: runtime.parameters,
    state: runtime.state,
    position,
  });
  const ordinaryMoves = publicAuthorityLegalMoves(snapshot);
  if (loss === null && ordinaryMoves.length > 0) {
    const drawbackMoves = runtime.rule.filterLegalMoves(
      {
        color,
        parameters: runtime.parameters,
        state: runtime.state,
        position,
      },
      ordinaryMoves,
    );
    if (drawbackMoves.length === 0) {
      loss = {
        ruleId: runtime.rule.id,
        color,
        reason: "The drawback forbids every otherwise legal move.",
      };
    }
  }
  const expected: SessionResult =
    loss === null ? standardResult : { kind: "drawback-loss", loss };
  if (!isDeepStrictEqual(trace.result, expected)) {
    throw new TypeError(
      `Trace ${trace.gameId} result does not match executable game replay.`,
    );
  }
}

/**
 * Recomputes every private per-ply label from the pinned executable catalog.
 *
 * The Engine trace parser proves the public chess replay. This second replay
 * proves that hidden parameters/state, drawback masks, and trigger/forced
 * labels agree with the claimed rules before those values enter training.
 */
export function validatePrivateTraceLabels(
  trace: PrivateSimulationTraceRecord,
): void {
  let white = initializeRuntime(trace, "white");
  let black = initializeRuntime(trace, "black");
  let snapshot: PublicPositionAuthoritySnapshot =
    createStandardChessPositionSnapshot(trace.initialFen);
  const history: ChessMove[] = [];
  const standardSession = new GameSession(
    {
      white: unrestrictedRule,
      black: unrestrictedRule,
    },
    new Mulberry32(0),
    trace.initialFen,
  );

  for (const ply of trace.plies) {
    const runtime = ply.color === "white" ? white : black;
    if (runtime === null) {
      throw new Error(`Trace ${trace.gameId} lacks an active rule runtime.`);
    }
    const position: PositionView = {
      fen: ply.fenBefore,
      turn: ply.color,
      ply: ply.ply,
      history: [...history],
    };
    assertEqualJson(
      ply.activeSecret.hiddenParameters,
      runtime.parameters,
      `Trace ply ${String(ply.ply)} hiddenParameters`,
    );
    assertEqualJson(
      ply.activeSecret.drawbackInternalState,
      runtime.state,
      `Trace ply ${String(ply.ply)} drawbackInternalState`,
    );
    if (
      runtime.rule.checkStartOfTurnLoss({
        color: runtime.color,
        parameters: runtime.parameters,
        state: runtime.state,
        position,
      }) !== null
    ) {
      throw new TypeError(
        `Trace ply ${String(ply.ply)} continues after a drawback loss.`,
      );
    }
    const ordinaryMoves = publicAuthorityLegalMoves(snapshot);
    const exactMoves = exactMoveMask(
      runtime.rule,
      runtime,
      position,
      ordinaryMoves,
      ply.publicEvaluatorConstraint,
    );
    const exactMask = sortedMask(exactMoves);
    assertEqualMask(
      [...ply.drawbackLegalMoves].sort(),
      exactMask,
      `Trace ply ${String(ply.ply)} drawbackLegalMoves`,
    );
    const triggered = exactMask.length !== ordinaryMoves.length;
    if (ply.ruleTriggered !== triggered) {
      throw new TypeError(
        `Trace ply ${String(ply.ply)} ruleTriggered does not match executable rule replay.`,
      );
    }
    if (ply.forced !== (exactMask.length === 1)) {
      throw new TypeError(
        `Trace ply ${String(ply.ply)} forced does not match executable rule replay.`,
      );
    }
    const move = observedMove(snapshot, ply.move.uci);
    if (!exactMask.includes(ply.move.uci)) {
      throw new TypeError(
        `Trace ply ${String(ply.ply)} observed move is not drawback-legal.`,
      );
    }
    const transition = advancePublicPositionAuthority(
      snapshot,
      commandFromMove(move),
    );
    const standardOutcome = standardSession.move(commandFromMove(move));
    if (!standardOutcome.ok) {
      throw new TypeError(
        `Trace ply ${String(ply.ply)} continues after a standard chess ending.`,
      );
    }
    const nextHistory = [...history, move];
    const positionAfterMove: PositionView = {
      fen: transition.position.fen,
      turn: turnFromFen(transition.position.fen),
      ply: ply.ply + 1,
      history: nextHistory,
    };
    if (ply.color === "white") {
      white = applyToRuntime(runtime, position, positionAfterMove, move);
    } else {
      black = applyToRuntime(runtime, position, positionAfterMove, move);
    }
    history.push(move);
    snapshot = transition.position;
  }
  assertFinalResult(
    trace,
    standardSession.result,
    white,
    black,
    snapshot,
    history,
  );
}
