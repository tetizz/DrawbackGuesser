import { describe, expect, it } from "vitest";
import {
  ichtyophobeRule,
  unrestrictedRule,
  type ChessMove,
  type DrawbackRule,
  type ExternalTurnConstraint,
  type PositionView,
} from "@drawbackengine/drawback-engine";
import {
  asExternalConstraintHypothesisSeed,
  asHypothesisSeed,
  asRerandomizedHypothesisSeed,
  SymbolicPredictor,
  type RerandomizedHypothesisSeed,
} from "./index.js";
import { PredictorTestGame } from "./test-game.js";

interface CounterState {
  readonly movesApplied: number;
}

type NoParameters = Record<string, never>;

function counterRule(
  id: string,
  permits: (move: ChessMove) => boolean,
  onFilter: () => void = () => undefined,
  onApply: () => void = () => undefined,
): DrawbackRule<CounterState, NoParameters> {
  return {
    id,
    name: id,
    description: id,
    verification: "verified",
    generateParameters: () => ({}),
    initialize: () => ({ movesApplied: 0 }),
    filterLegalMoves: (_context, moves) => {
      onFilter();
      return moves.filter(permits);
    },
    applyMove: (context) => {
      onApply();
      return {
        movesApplied: context.state.movesApplied + 1,
      };
    },
    checkStartOfTurnLoss: () => null,
  };
}

const unrestricted = counterRule("opportunity-unrestricted", () => true);
const onlyE4 = counterRule(
  "opportunity-only-e4",
  (move) => move.from === "e2" && move.to === "e4",
);

function exactPredictor(initial: PositionView): SymbolicPredictor {
  return new SymbolicPredictor(
    {
      white: [
        asHypothesisSeed(unrestricted, {}),
        asHypothesisSeed(onlyE4, {}),
      ],
      black: [asHypothesisSeed(unrestricted, {})],
    },
    initial,
    { scoreLogLikelihood: () => 0 },
  );
}

function publicConstraint(
  before: PositionView,
  ordinaryMoves: readonly ChessMove[],
): ExternalTurnConstraint {
  const request = ichtyophobeRule.requestTurnConstraint(
    {
      color: "white",
      parameters: {},
      state: { movesApplied: 0 },
      position: before,
    },
    ordinaryMoves,
  );
  return Object.freeze({
    provider: request.provider,
    policyId: request.policyId,
    positionKey: request.positionKey,
    requestDigest: "ab".repeat(32),
    bestMoveUci: "e2e4",
    engineFingerprint: "public-opportunity-test",
  });
}

interface RerandomizedState {
  readonly movesApplied: number;
}

type RerandomizedOutcome = "e4" | "d4";

function rerandomizedSeed(): RerandomizedHypothesisSeed<
  RerandomizedState,
  RerandomizedOutcome
> {
  return {
    kind: "rerandomized",
    drawbackId: "opportunity-rerandomized",
    name: "Opportunity rerandomized",
    initialize: () => ({ movesApplied: 0 }),
    outcomes: () => [
      { outcome: "e4", probability: 0.5 },
      { outcome: "d4", probability: 0.5 },
    ],
    filterLegalMoves: (_context, outcome, moves) =>
      moves.filter((move) => move.to === outcome),
    applyObservedMove: (context) => ({
      movesApplied: context.state.movesApplied + 1,
    }),
  };
}

type ConsensusOutcome = "likely" | "unlikely";

function consensusRerandomizedSeed(): RerandomizedHypothesisSeed<
  RerandomizedState,
  ConsensusOutcome
> {
  return {
    kind: "rerandomized",
    drawbackId: "opportunity-rerandomized-consensus",
    name: "Opportunity rerandomized consensus",
    initialize: () => ({ movesApplied: 0 }),
    outcomes: () => [
      { outcome: "likely", probability: 0.99 },
      { outcome: "unlikely", probability: 0.01 },
    ],
    filterLegalMoves: (_context, _outcome, moves) =>
      moves.filter((move) => move.from === "e2" && move.to === "e4"),
    applyObservedMove: (context) => ({
      movesApplied: context.state.movesApplied + 1,
    }),
  };
}

describe("public pre-observation opportunity snapshots", () => {
  it("keeps observe source-compatible and behavior-identical", () => {
    const observeGame = new PredictorTestGame();
    const opportunityGame = new PredictorTestGame();
    const observed = exactPredictor(observeGame.view()).observe(
      observeGame.play("e2", "e4"),
    );
    const withOpportunity = exactPredictor(
      opportunityGame.view(),
    ).observeWithOpportunities(opportunityGame.play("e2", "e4"));

    expect(withOpportunity.state).toEqual(observed);
  });

  it("reports pre-observation counts, activation, forcing, and legality", () => {
    const game = new PredictorTestGame();
    const result = exactPredictor(game.view()).observeWithOpportunities(
      game.play("e2", "e4"),
    );

    expect(result.opportunity).toEqual({
      color: "white",
      hypotheses: [
        {
          hypothesisIndex: 0,
          drawbackId: "opportunity-unrestricted",
          status: "known",
          ordinaryLegalMoveCount: 20,
          allowedMoveCount: 20,
          allowedMoveFraction: 1,
          triggered: false,
          forced: false,
          observedMoveLegal: true,
        },
        {
          hypothesisIndex: 1,
          drawbackId: "opportunity-only-e4",
          status: "known",
          ordinaryLegalMoveCount: 20,
          allowedMoveCount: 1,
          allowedMoveFraction: 0.05,
          triggered: true,
          forced: true,
          observedMoveLegal: true,
        },
      ],
    });
  });

  it("marks an external hypothesis unknown without a public constraint and known with one", () => {
    const unknownGame = new PredictorTestGame(
      "4k3/8/8/8/8/8/4P3/4K1N1 w - - 0 1",
    );
    const unknownPredictor = new SymbolicPredictor(
      {
        white: [asExternalConstraintHypothesisSeed(ichtyophobeRule, {})],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      unknownGame.view(),
    );
    const unknown = unknownPredictor.observeWithOpportunities(
      unknownGame.play("e2", "e4"),
    ).opportunity.hypotheses[0];
    expect(unknown).toMatchObject({
      drawbackId: "ichtyophobe",
      status: "unknown",
      allowedMoveCount: null,
      allowedMoveFraction: null,
      triggered: null,
      forced: null,
      observedMoveLegal: null,
    });

    const knownGame = new PredictorTestGame(
      "4k3/8/8/8/8/8/4P3/4K1N1 w - - 0 1",
    );
    const knownPredictor = new SymbolicPredictor(
      {
        white: [asExternalConstraintHypothesisSeed(ichtyophobeRule, {})],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      knownGame.view(),
    );
    const before = knownGame.view();
    const constraint = publicConstraint(before, knownGame.legalMoves());
    const known = knownPredictor.observeWithOpportunities(
      knownGame.play("e2", "e4", { externalConstraint: constraint }),
    ).opportunity.hypotheses[0];
    expect(known).toMatchObject({
      drawbackId: "ichtyophobe",
      status: "known",
      observedMoveLegal: false,
    });
    expect(known?.allowedMoveCount).toBeTypeOf("number");
  });

  it("marks differing rerandomized outcome masks unknown", () => {
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [asRerandomizedHypothesisSeed(rerandomizedSeed())],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      game.view(),
      { scoreLogLikelihood: () => 0 },
    );
    const opportunity = predictor.observeWithOpportunities(
      game.play("e2", "e4"),
    ).opportunity.hypotheses[0];

    expect(opportunity).toMatchObject({
      drawbackId: "opportunity-rerandomized",
      hypothesisIndex: 0,
      status: "unknown",
      ordinaryLegalMoveCount: 20,
      allowedMoveCount: null,
      allowedMoveFraction: null,
      triggered: null,
      forced: null,
      observedMoveLegal: null,
    });
  });

  it("reports a rerandomized mask only when every outcome agrees", () => {
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [
          asRerandomizedHypothesisSeed(consensusRerandomizedSeed()),
        ],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      game.view(),
      { scoreLogLikelihood: () => 0 },
    );
    const opportunity = predictor.observeWithOpportunities(
      game.play("e2", "e4"),
    ).opportunity.hypotheses[0];

    expect(opportunity).toMatchObject({
      drawbackId: "opportunity-rerandomized-consensus",
      status: "known",
      ordinaryLegalMoveCount: 20,
      allowedMoveCount: 1,
      allowedMoveFraction: 0.05,
      triggered: true,
      forced: true,
      observedMoveLegal: true,
    });
  });

  it("does not serialize parameter values in opportunity identities", () => {
    const parameterizedRule: DrawbackRule<
      CounterState,
      { readonly hiddenSquare: string }
    > = {
      id: "opportunity-hidden-square",
      name: "Opportunity hidden square",
      description: "Parameter privacy test rule",
      verification: "verified",
      generateParameters: () => ({ hiddenSquare: "a1" }),
      initialize: () => ({ movesApplied: 0 }),
      filterLegalMoves: (_context, moves) => moves,
      applyMove: (context) => ({
        movesApplied: context.state.movesApplied + 1,
      }),
      checkStartOfTurnLoss: () => null,
    };
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(parameterizedRule, { hiddenSquare: "e4" }),
        ],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      game.view(),
    );
    const opportunity = predictor.observeWithOpportunities(
      game.play("e2", "e4"),
    ).opportunity;
    const serialized = JSON.stringify(opportunity);

    expect(opportunity.hypotheses[0]).toMatchObject({
      hypothesisIndex: 0,
      drawbackId: "opportunity-hidden-square",
    });
    expect(serialized).not.toContain("hiddenSquare");
    expect(serialized).not.toContain('"e4"');
  });

  it("reports a public external start-of-turn loss without evaluator data", () => {
    const externalLossRule: typeof ichtyophobeRule = {
      ...ichtyophobeRule,
      id: "opportunity-external-loss",
      name: "Opportunity external loss",
      checkStartOfTurnLoss: (context) => ({
        ruleId: "opportunity-external-loss",
        color: context.color,
        reason: "the public loss condition is already met",
      }),
    };
    const game = new PredictorTestGame(
      "4k3/8/8/8/8/8/4P3/4K1N1 w - - 0 1",
    );
    const predictor = new SymbolicPredictor(
      {
        white: [
          asExternalConstraintHypothesisSeed(externalLossRule, {}),
        ],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      game.view(),
    );
    const result = predictor.observeWithOpportunities(
      game.play("e2", "e4"),
    );

    expect(result.opportunity.hypotheses[0]).toMatchObject({
      status: "known",
      allowedMoveCount: 0,
      allowedMoveFraction: 0,
      triggered: true,
      forced: false,
      observedMoveLegal: false,
    });
    expect(result.state.white.hypotheses[0]?.eliminated).toBe(true);
  });

  it("executes active exact callbacks exactly once", () => {
    let filterCalls = 0;
    let applyCalls = 0;
    let scoreCalls = 0;
    const counted = counterRule(
      "opportunity-counted",
      () => true,
      () => {
        filterCalls += 1;
      },
      () => {
        applyCalls += 1;
      },
    );
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [asHypothesisSeed(counted, {})],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      game.view(),
      {
        scoreLogLikelihood: () => {
          scoreCalls += 1;
          return 0;
        },
      },
    );

    predictor.observeWithOpportunities(game.play("e2", "e4"));

    expect(filterCalls).toBe(1);
    expect(applyCalls).toBe(1);
    expect(scoreCalls).toBe(1);
  });

  it("evaluates each rerandomized branch once and advances once", () => {
    let outcomesCalls = 0;
    let filterCalls = 0;
    let applyCalls = 0;
    let scoreCalls = 0;
    const counted: RerandomizedHypothesisSeed<
      RerandomizedState,
      RerandomizedOutcome
    > = {
      ...rerandomizedSeed(),
      outcomes: () => {
        outcomesCalls += 1;
        return [
          { outcome: "e4", probability: 0.5 },
          { outcome: "d4", probability: 0.5 },
        ];
      },
      filterLegalMoves: (_context, outcome, moves) => {
        filterCalls += 1;
        return moves.filter((move) => move.to === outcome);
      },
      applyObservedMove: (context) => {
        applyCalls += 1;
        return {
          movesApplied: context.state.movesApplied + 1,
        };
      },
    };
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [asRerandomizedHypothesisSeed(counted)],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      game.view(),
      {
        scoreLogLikelihood: () => {
          scoreCalls += 1;
          return 0;
        },
      },
    );

    predictor.observeWithOpportunities(game.play("e2", "e4"));

    expect(outcomesCalls).toBe(1);
    expect(filterCalls).toBe(2);
    expect(applyCalls).toBe(1);
    expect(scoreCalls).toBe(1);
  });

  it("does not inspect or reactivate an already eliminated hypothesis", () => {
    let filterCalls = 0;
    const forbidsE4 = counterRule(
      "opportunity-forbids-e4",
      (move) => move.to !== "e4",
      () => {
        filterCalls += 1;
      },
    );
    const game = new PredictorTestGame();
    const predictor = new SymbolicPredictor(
      {
        white: [
          asHypothesisSeed(forbidsE4, {}),
          asHypothesisSeed(unrestrictedRule, {}),
        ],
        black: [asHypothesisSeed(unrestrictedRule, {})],
      },
      game.view(),
      { scoreLogLikelihood: () => 0 },
    );

    predictor.observeWithOpportunities(game.play("e2", "e4"));
    predictor.observe(game.play("e7", "e5"));
    const callsBeforeEliminatedTurn = filterCalls;
    const result = predictor.observeWithOpportunities(game.play("g1", "f3"));

    expect(result.opportunity.hypotheses[0]).toMatchObject({
      drawbackId: "opportunity-forbids-e4",
      status: "eliminated",
      allowedMoveCount: null,
      allowedMoveFraction: null,
      triggered: null,
      forced: null,
      observedMoveLegal: null,
    });
    expect(filterCalls).toBe(callsBeforeEliminatedTurn);
    expect(result.state.white.hypotheses[0]).toMatchObject({
      eliminated: true,
      logProbability: Number.NEGATIVE_INFINITY,
    });
  });

  it("leaves the inactive color unchanged while advancing the active color once", () => {
    const game = new PredictorTestGame();
    const predictor = exactPredictor(game.view());
    const blackBefore = predictor.state.black;
    const result = predictor.observeWithOpportunities(game.play("e2", "e4"));

    expect(result.state.black).toEqual(blackBefore);
    expect(result.state.black.hypotheses[0]?.internalState).toEqual({
      movesApplied: 0,
    });
    expect(result.state.white.hypotheses[0]?.internalState).toEqual({
      movesApplied: 1,
    });
  });

  it("deeply freezes the deterministic snapshot and exposes no private-shaped fields", () => {
    const firstGame = new PredictorTestGame();
    const secondGame = new PredictorTestGame();
    const first = exactPredictor(firstGame.view()).observeWithOpportunities(
      firstGame.play("e2", "e4"),
    ).opportunity;
    const second = exactPredictor(secondGame.view()).observeWithOpportunities(
      secondGame.play("e2", "e4"),
    ).opportunity;

    expect(second).toEqual(first);
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(first.hypotheses)).toBe(true);
    expect(first.hypotheses.every((entry) => Object.isFrozen(entry))).toBe(true);
    expect(Reflect.set(first, "color", "black")).toBe(false);
    expect(Reflect.set(first.hypotheses, "0", {})).toBe(false);

    const serialized = JSON.stringify(first);
    for (const privateField of [
      "parameters",
      "internalState",
      "evidence",
      "logProbability",
      "probability",
      "trueDrawback",
      "secret",
      "label",
      "result",
      "seed",
      "bot",
      "allowedMoves",
      "authorityLegalMoves",
      "hypothesisId",
    ]) {
      expect(serialized).not.toContain(`"${privateField}"`);
    }
  });
});
