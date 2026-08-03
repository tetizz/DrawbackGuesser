import { describe, expect, it } from "vitest";
import {
  createPlayerPrivateAssignmentSchedule,
} from "@drawbackengine/simulation-arena";
import {
  CAPTURABLE_HYPOTHESIS_RULE_IDS,
} from "@drawbackguesser/predictor";
import {
  SCHEMA9_LEDGER_SPLITS,
  SCHEMA9_SPLIT_SEED_ROOTS,
  type Schema9ExpectedAssignment,
  type Schema9LedgerSplit,
  type Schema9SeedRoots,
} from "./schema9-ledger-types.js";
import { schema9AssignmentScheduler } from "./schema9-schedule-replay.js";

interface AuthoritativeBoundaryVector {
  readonly index: number;
  readonly whiteRuleId: string;
  readonly blackRuleId: string;
  readonly seed: number;
  readonly parameterSeeds: Readonly<{ white: number; black: number }>;
  readonly gameId: string;
}

const AUTHORITATIVE_ENGINE_BOUNDARIES = Object.freeze({
  train: Object.freeze([
    Object.freeze({
      index: 0,
      whiteRuleId: "battle-fatigue",
      blackRuleId: "false-prophets",
      seed: 53_114_731,
      parameterSeeds: Object.freeze({
        white: 2_530_891_679,
        black: 519_356_932,
      }),
      gameId: "player-private-v1-032a776b-000000-96da579f-1ef4c204",
    }),
    Object.freeze({
      index: 624,
      whiteRuleId: "even-keeled",
      blackRuleId: "forward-march",
      seed: 1_501_970_941,
      parameterSeeds: Object.freeze({
        white: 2_886_564_798,
        black: 139_019_728,
      }),
      gameId: "player-private-v1-598641fd-000624-ac0d7bbe-084945d0",
    }),
  ]),
  "validation-a": Object.freeze([
    Object.freeze({
      index: 0,
      whiteRuleId: "pacman",
      blackRuleId: "femme-fatale",
      seed: 2_681_234_919,
      parameterSeeds: Object.freeze({
        white: 3_607_390_157,
        black: 2_402_665_756,
      }),
      gameId: "player-private-v1-9fd065e7-000000-d70467cd-8f35c51c",
    }),
    Object.freeze({
      index: 624,
      whiteRuleId: "horse-tranquilizer",
      blackRuleId: "irresistible",
      seed: 977_313_404,
      parameterSeeds: Object.freeze({
        white: 3_531_641_376,
        black: 3_931_694_435,
      }),
      gameId: "player-private-v1-3a409e7c-000624-d2809220-ea58e563",
    }),
  ]),
  "validation-b": Object.freeze([
    Object.freeze({
      index: 0,
      whiteRuleId: "pacman",
      blackRuleId: "forward-march",
      seed: 1_775_353_909,
      parameterSeeds: Object.freeze({
        white: 2_783_502_898,
        black: 551_951_680,
      }),
      gameId: "player-private-v1-69d1c035-000000-a5e8e232-20e61d40",
    }),
    Object.freeze({
      index: 624,
      whiteRuleId: "truant",
      blackRuleId: "oddball",
      seed: 104_751_319,
      parameterSeeds: Object.freeze({
        white: 1_852_407_637,
        black: 3_694_535_774,
      }),
      gameId: "player-private-v1-063e60d7-000624-6e697f55-dc36245e",
    }),
  ]),
  test: Object.freeze([
    Object.freeze({
      index: 0,
      whiteRuleId: "truant",
      blackRuleId: "femme-fatale",
      seed: 2_113_582_989,
      parameterSeeds: Object.freeze({
        white: 3_280_375_812,
        black: 2_735_646_158,
      }),
      gameId: "player-private-v1-7dfab78d-000000-c3869004-a30ea5ce",
    }),
    Object.freeze({
      index: 624,
      whiteRuleId: "lame-duck",
      blackRuleId: "nurturer",
      seed: 2_761_140_428,
      parameterSeeds: Object.freeze({
        white: 3_517_909_109,
        black: 1_156_780_054,
      }),
      gameId: "player-private-v1-a493a8cc-000624-d1af0875-44f31016",
    }),
  ]),
}) satisfies Readonly<
  Record<Schema9LedgerSplit, readonly AuthoritativeBoundaryVector[]>
>;

function boundaryIdentity(assignment: Schema9ExpectedAssignment) {
  return Object.freeze({
    index: assignment.gameIndex,
    whiteRuleId: assignment.whiteRuleId,
    blackRuleId: assignment.blackRuleId,
    seed: assignment.seed,
    parameterSeeds: assignment.parameterSeeds,
    gameId: assignment.gameId,
  });
}

describe("schema-9 authoritative schedule replay", () => {
  it.each(SCHEMA9_LEDGER_SPLITS)(
    "matches the pinned Engine boundary vectors for %s",
    (split) => {
      const assignments = [...schema9AssignmentScheduler.assignments(
        split,
        625,
        SCHEMA9_SPLIT_SEED_ROOTS[split],
      )];

      expect(assignments).toHaveLength(625);
      for (const expected of AUTHORITATIVE_ENGINE_BOUNDARIES[split]) {
        const assignment = assignments[expected.index];
        expect(assignment).toBeDefined();
        expect(boundaryIdentity(assignment as Schema9ExpectedAssignment))
          .toEqual(expected);
      }
    },
    60_000,
  );

  it("rejects invalid seed roots before uint32 coercion", () => {
    const invalidGameplayRoots = [
      [SCHEMA9_SPLIT_SEED_ROOTS.train[0], -1,
        SCHEMA9_SPLIT_SEED_ROOTS.train[2]],
      [SCHEMA9_SPLIT_SEED_ROOTS.train[0], 0x1_0000_0000,
        SCHEMA9_SPLIT_SEED_ROOTS.train[2]],
    ] as const;

    for (const roots of invalidGameplayRoots) {
      expect(() => schema9AssignmentScheduler.assignments(
        "train",
        1,
        roots as unknown as Schema9SeedRoots,
      )).toThrow("must exactly match the frozen");
    }
  });

  it("enforces the uint32 game-index ceiling and returns a lazy iterator", () => {
    expect(() => schema9AssignmentScheduler.assignments(
      "train",
      0x1_0000_0000 + 1,
      SCHEMA9_SPLIT_SEED_ROOTS.train,
    )).toThrow("cannot exceed the uint32 index space");

    const assignments = schema9AssignmentScheduler.assignments(
      "train",
      0x1_0000_0000,
      SCHEMA9_SPLIT_SEED_ROOTS.train,
    );
    expect(Array.isArray(assignments)).toBe(false);
    expect(assignments[Symbol.iterator]()).toBe(assignments);
  });

  it.each(SCHEMA9_LEDGER_SPLITS)(
    "matches the executing Engine scheduler across sampled %s assignments",
    (split) => {
      const roots = SCHEMA9_SPLIT_SEED_ROOTS[split];
      const gameCount = 625;
      const replay = [...schema9AssignmentScheduler.assignments(
        split,
        gameCount,
        roots,
      )];
      const engine = [...createPlayerPrivateAssignmentSchedule({
        splitCounts: { train: gameCount, validation: 0, test: 0 },
        labelSeed: roots[0],
        gameplaySeed: roots[1],
        parameterSeed: roots[2],
        ruleIds: CAPTURABLE_HYPOTHESIS_RULE_IDS,
      })];
      const sampledIndices = [0, 1, 24, 25, 26, 127, 311, 312, 623, 624];

      expect(engine).toHaveLength(gameCount);
      for (const index of sampledIndices) {
        const replayed = replay[index];
        const scheduled = engine[index];
        expect(replayed).toBeDefined();
        expect(scheduled).toBeDefined();
        expect({
          gameIndex: replayed?.gameIndex,
          seed: replayed?.seed,
          parameterSeeds: replayed?.parameterSeeds,
          whiteRuleId: replayed?.whiteRuleId,
          blackRuleId: replayed?.blackRuleId,
        }).toEqual({
          gameIndex: scheduled?.splitIndex,
          seed: scheduled?.assignment.seed,
          parameterSeeds: scheduled?.assignment.parameterSeeds,
          whiteRuleId: scheduled?.assignment.whiteRuleId,
          blackRuleId: scheduled?.assignment.blackRuleId,
        });
      }
    },
    60_000,
  );
});
