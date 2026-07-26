import { describe, expect, it } from "vitest";
import type {
  ChessMove,
  DrawbackRule,
  ExternalConstraintDrawbackRule,
} from "@drawbackengine/drawback-engine";
import {
  asExternalConstraintHypothesisSeed,
  asHypothesisSeed,
  asRerandomizedHypothesisSeed,
  SymbolicPredictor,
} from "./index.js";
import { PredictorTestGame } from "./test-game.js";

interface NestedState {
  readonly nested: {
    readonly allowed: boolean;
    readonly count: number;
  };
  readonly trail: readonly { readonly label: string }[];
}

interface NestedParameters extends Record<string, unknown> {
  readonly policy: {
    readonly allowed: boolean;
    readonly step: number;
  };
  readonly tags: readonly string[];
}

interface CyclicMarker {
  self?: CyclicMarker;
}

interface CloneableState {
  readonly facts: Map<string, { allowed: boolean; count: number }>;
  readonly labels: Set<string>;
  readonly timestamp: Date;
  readonly bytes: Uint8Array;
  readonly cyclic: CyclicMarker;
  readonly nonFinite: number;
  readonly large: bigint;
}

const initialNestedState = (): NestedState => ({
  nested: { allowed: true, count: 0 },
  trail: [{ label: "initial" }],
});

function allowedE4(move: ChessMove): boolean {
  return move.from === "e2" && move.to === "e4";
}

function exactIsolationRule(
  initialized: (state: NestedState) => void,
): DrawbackRule<NestedState, NestedParameters> {
  return {
    id: "exact-isolation",
    name: "Exact isolation",
    description: "Predictor regression rule.",
    verification: "verified",
    generateParameters: () => ({
      policy: { allowed: true, step: 1 },
      tags: ["generated"],
    }),
    initialize: () => {
      const state = initialNestedState();
      initialized(state);
      return state;
    },
    filterLegalMoves: (context, moves) =>
      context.state.nested.allowed && context.parameters.policy.allowed
        ? moves
        : moves.filter((move) => !allowedE4(move)),
    applyMove: (context) => ({
      nested: {
        allowed: context.state.nested.allowed,
        count:
          context.state.nested.count + context.parameters.policy.step,
      },
      trail: [...context.state.trail, { label: "observed" }],
    }),
    checkStartOfTurnLoss: () => null,
  };
}

function externalIsolationRule(
  initialized: (state: NestedState) => void,
): ExternalConstraintDrawbackRule<NestedState, NestedParameters> {
  return {
    kind: "external-turn-constraint",
    id: "external-isolation",
    name: "External isolation",
    description: "Predictor regression rule.",
    verification: "verified",
    generateParameters: () => ({
      policy: { allowed: true, step: 1 },
      tags: ["generated"],
    }),
    initialize: () => {
      const state = initialNestedState();
      initialized(state);
      return state;
    },
    requestTurnConstraint: (context) => ({
      provider: "uci-best-move",
      policyId: "isolation-test",
      positionKey: context.position.fen,
      fen: context.position.fen,
      ordinaryRootMoves: [],
    }),
    filterLegalMovesWithConstraint: (context, moves) =>
      context.state.nested.allowed && context.parameters.policy.allowed
        ? moves
        : moves.filter((move) => !allowedE4(move)),
    applyMove: (context) => ({
      nested: {
        allowed: context.state.nested.allowed,
        count:
          context.state.nested.count + context.parameters.policy.step,
      },
      trail: [...context.state.trail, { label: "observed" }],
    }),
    checkStartOfTurnLoss: () => null,
  };
}

function assertDeeplyFrozenPublicSnapshot(
  publicState: unknown,
  publicParameters: Readonly<Record<string, unknown>>,
): void {
  const state = publicState as NestedState;
  const parameters = publicParameters as NestedParameters;
  expect(Object.isFrozen(state)).toBe(true);
  expect(Object.isFrozen(state.nested)).toBe(true);
  expect(Object.isFrozen(state.trail)).toBe(true);
  expect(Object.isFrozen(state.trail[0])).toBe(true);
  expect(Object.isFrozen(parameters)).toBe(true);
  expect(Object.isFrozen(parameters.policy)).toBe(true);
  expect(Object.isFrozen(parameters.tags)).toBe(true);
  expect(() => Object.assign(state.nested, { allowed: false })).toThrow(
    TypeError,
  );
  expect(() => Object.assign(parameters.policy, { allowed: false })).toThrow(
    TypeError,
  );
}

describe("predictor public state isolation", () => {
  it("does not expose or re-read exact runtime state and nested parameters", () => {
    const game = new PredictorTestGame();
    let privateInitialState: NestedState | undefined;
    const sourceParameters: NestedParameters = {
      policy: { allowed: true, step: 1 },
      tags: ["source"],
    };
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(
            exactIsolationRule((state) => {
              privateInitialState = state;
            }),
            sourceParameters,
          ),
        ],
        black: [
          asHypothesisSeed(
            exactIsolationRule(() => undefined),
            {
              policy: { allowed: true, step: 1 },
              tags: ["black"],
            },
          ),
        ],
      },
      game.view(),
    );

    const exposed = predictor.state.white.hypotheses[0];
    expect(exposed).toBeDefined();
    expect(exposed?.internalState).not.toBe(privateInitialState);
    expect(
      (exposed?.parameters as NestedParameters).policy,
    ).not.toBe(sourceParameters.policy);
    assertDeeplyFrozenPublicSnapshot(
      exposed?.internalState,
      exposed?.parameters ?? {},
    );

    Object.assign(sourceParameters.policy, { allowed: false, step: 50 });
    predictor.observe(game.play("e2", "e4"));

    expect(predictor.state.white.hypotheses[0]).toMatchObject({
      eliminated: false,
      internalState: {
        nested: { allowed: true, count: 1 },
      },
    });
  });

  it("does not expose rerandomized runtime state", () => {
    const game = new PredictorTestGame();
    let privateInitialState: NestedState | undefined;
    const rerandomized = asRerandomizedHypothesisSeed({
      kind: "rerandomized",
      drawbackId: "rerandomized-isolation",
      name: "Rerandomized isolation",
      initialize: () => {
        const state = initialNestedState();
        privateInitialState = state;
        return state;
      },
      outcomes: () => [{ outcome: { allows: true }, probability: 1 }],
      filterLegalMoves: (context, outcome, moves) =>
        context.state.nested.allowed && outcome.allows
          ? moves
          : moves.filter((move) => !allowedE4(move)),
      applyObservedMove: (context) => ({
        nested: {
          allowed: context.state.nested.allowed,
          count: context.state.nested.count + 1,
        },
        trail: [...context.state.trail, { label: "observed" }],
      }),
    });
    const predictor = new SymbolicPredictor(
      {
        white: [rerandomized],
        black: [rerandomized],
      },
      game.view(),
    );

    const exposed = predictor.state.white.hypotheses[0];
    expect(exposed?.internalState).not.toBe(privateInitialState);
    const state = exposed?.internalState as NestedState;
    expect(Object.isFrozen(state)).toBe(true);
    expect(Object.isFrozen(state.nested)).toBe(true);
    expect(() => Object.assign(state.nested, { allowed: false })).toThrow(
      TypeError,
    );

    predictor.observe(game.play("e2", "e4"));
    expect(predictor.state.white.hypotheses[0]).toMatchObject({
      eliminated: false,
      internalState: {
        nested: { allowed: true, count: 1 },
      },
    });
  });

  it("does not expose or re-read external runtime state and parameters", () => {
    const game = new PredictorTestGame();
    let privateInitialState: NestedState | undefined;
    const sourceParameters: NestedParameters = {
      policy: { allowed: true, step: 1 },
      tags: ["source"],
    };
    const predictor = new SymbolicPredictor(
      {
        white: [
          asExternalConstraintHypothesisSeed(
            externalIsolationRule((state) => {
              privateInitialState = state;
            }),
            sourceParameters,
          ),
        ],
        black: [
          asExternalConstraintHypothesisSeed(
            externalIsolationRule(() => undefined),
            {
              policy: { allowed: true, step: 1 },
              tags: ["black"],
            },
          ),
        ],
      },
      game.view(),
    );

    const exposed = predictor.state.white.hypotheses[0];
    expect(exposed?.internalState).not.toBe(privateInitialState);
    expect(
      (exposed?.parameters as NestedParameters).policy,
    ).not.toBe(sourceParameters.policy);
    assertDeeplyFrozenPublicSnapshot(
      exposed?.internalState,
      exposed?.parameters ?? {},
    );

    Object.assign(sourceParameters.policy, { allowed: false, step: 50 });
    predictor.observe(game.play("e2", "e4"));
    expect(predictor.state.white.hypotheses[0]).toMatchObject({
      eliminated: false,
      internalState: {
        nested: { allowed: true, count: 1 },
      },
    });
  });

  it("detaches structured-cloneable state shapes without restricting the state contract", () => {
    const game = new PredictorTestGame();
    let privateInitialState: CloneableState | undefined;
    const cloneableRule: DrawbackRule<
      CloneableState,
      Record<string, never>
    > = {
      id: "cloneable-state-shapes",
      name: "Cloneable state shapes",
      description: "Predictor regression rule.",
      verification: "verified",
      generateParameters: () => ({}),
      initialize: () => {
        const cyclic: CyclicMarker = {};
        cyclic.self = cyclic;
        const state: CloneableState = {
          facts: new Map([
            ["runtime", { allowed: true, count: 0 }],
          ]),
          labels: new Set(["trusted"]),
          timestamp: new Date(0),
          bytes: new Uint8Array([1, 2, 3]),
          cyclic,
          nonFinite: Number.POSITIVE_INFINITY,
          large: 9_007_199_254_740_993n,
        };
        privateInitialState = state;
        return state;
      },
      filterLegalMoves: (context, moves) => {
        const fact = context.state.facts.get("runtime");
        return (
          fact?.allowed === true
          && context.state.labels.has("trusted")
          && context.state.timestamp.getTime() === 0
          && context.state.bytes[0] === 1
        )
          ? moves
          : moves.filter((move) => !allowedE4(move));
      },
      applyMove: (context) => {
        const fact = context.state.facts.get("runtime");
        return {
          ...context.state,
          facts: new Map([
            [
              "runtime",
              {
                allowed: fact?.allowed ?? false,
                count: (fact?.count ?? 0) + 1,
              },
            ],
          ]),
        };
      },
      checkStartOfTurnLoss: () => null,
    };
    const predictor = new SymbolicPredictor(
      {
        white: [asHypothesisSeed(cloneableRule, {})],
        black: [asHypothesisSeed(cloneableRule, {})],
      },
      game.view(),
    );

    const exposed = predictor.state.white.hypotheses[0]
      ?.internalState as CloneableState;
    expect(exposed).not.toBe(privateInitialState);
    expect(exposed.facts).not.toBe(privateInitialState?.facts);
    expect(exposed.cyclic.self).toBe(exposed.cyclic);
    expect(Object.isFrozen(exposed)).toBe(true);
    expect(Object.isFrozen(exposed.facts.get("runtime"))).toBe(true);
    expect(Object.isFrozen(exposed.cyclic)).toBe(true);
    expect(exposed.nonFinite).toBe(Number.POSITIVE_INFINITY);
    expect(exposed.large).toBe(9_007_199_254_740_993n);

    exposed.facts.set("runtime", { allowed: false, count: 99 });
    exposed.labels.clear();
    exposed.timestamp.setTime(10);
    exposed.bytes[0] = 0;
    predictor.observe(game.play("e2", "e4"));

    expect(predictor.state.white.hypotheses[0]).toMatchObject({
      eliminated: false,
      internalState: {
        facts: new Map([
          ["runtime", { allowed: true, count: 1 }],
        ]),
      },
    });
  });

  it("fails closed for unclonable state and unsupported parameter shapes", () => {
    const game = new PredictorTestGame();
    const unsupportedStateRule: DrawbackRule<unknown, Record<string, unknown>> = {
      id: "unsupported-state-shape",
      name: "Unsupported state shape",
      description: "Predictor regression rule.",
      verification: "verified",
      generateParameters: () => ({}),
      initialize: () => ({ callback: () => true }),
      filterLegalMoves: (_context, moves) => moves,
      applyMove: (context) => context.state,
      checkStartOfTurnLoss: () => null,
    };
    const safeRule = exactIsolationRule(() => undefined);

    expect(
      () =>
        new SymbolicPredictor(
          {
            white: [asHypothesisSeed(unsupportedStateRule, {})],
            black: [
              asHypothesisSeed(safeRule, {
                policy: { allowed: true, step: 1 },
                tags: [],
              }),
            ],
          },
          game.view(),
        ),
    ).toThrow(/state.*structured cloning failed/iu);

    expect(
      () =>
        new SymbolicPredictor(
          {
            white: [
              asHypothesisSeed(safeRule, {
                policy: { allowed: true, step: 1 },
                tags: [],
                unsupported: new Date(0),
              }),
            ],
            black: [
              asHypothesisSeed(safeRule, {
                policy: { allowed: true, step: 1 },
                tags: [],
              }),
            ],
          },
          game.view(),
        ),
    ).toThrow(/unsupported.*parameter.*shape/iu);
  });
});
