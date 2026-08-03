import { createHash } from "node:crypto";
import { DrawbackGameSession } from "@drawbackengine/chess-core";
import {
  resolveAuditedCapturableKingRule,
  type AuditedCapturableKingRuleId,
} from "@drawbackengine/drawback-engine";
import {
  playerPrivateSimulationGameId,
} from "@drawbackengine/simulation-trace";
import {
  deriveSimulationStreamSeed,
  Mulberry32,
} from "@drawbackengine/shared";
import {
  CAPTURABLE_HYPOTHESIS_RULE_IDS,
} from "@drawbackguesser/predictor";
import type {
  Schema9AssignmentScheduler,
  Schema9ExpectedAssignment,
  Schema9LedgerSplit,
  Schema9SeedRoots,
} from "./schema9-ledger-types.js";
import {
  canonicalJsonBytes,
  checkedSchema9SeedRoots,
} from "./schema9-ledger-types.js";

const SCHEDULE_DOMAINS = Object.freeze({
  whiteLabels: 0xa91f_0b21,
  blackLabels: 0x76e3_4cd5,
  whiteParameters: 0x1c69_ae77,
  blackParameters: 0xd432_508b,
});

function deriveGameSeed(batchSeed: number, gameIndex: number): number {
  let value = (batchSeed ^ Math.imul(gameIndex + 1, 0x9e37_79b9)) >>> 0;
  value ^= value >>> 16;
  value = Math.imul(value, 0x21f0_aaad);
  value ^= value >>> 15;
  value = Math.imul(value, 0x735a_2d97);
  return (value ^ (value >>> 15)) >>> 0;
}

function shuffledRules(seed: number): readonly string[] {
  const result = [...CAPTURABLE_HYPOTHESIS_RULE_IDS];
  const random = new Mulberry32(seed);
  for (let index = result.length - 1; index > 0; index -= 1) {
    const other = random.integer(index + 1);
    const current = result[index];
    const replacement = result[other];
    if (current === undefined || replacement === undefined) {
      throw new Error("Schema-9 rule permutation lost a rule.");
    }
    result[index] = replacement;
    result[other] = current;
  }
  return Object.freeze(result);
}

function initialReplaySha256(
  whiteRuleId: string,
  blackRuleId: string,
  parameterSeeds: Readonly<{ white: number; black: number }>,
): string {
  const session = DrawbackGameSession.create(
    {
      white: resolveAuditedCapturableKingRule(
        whiteRuleId as AuditedCapturableKingRuleId,
      ),
      black: resolveAuditedCapturableKingRule(
        blackRuleId as AuditedCapturableKingRuleId,
      ),
    },
    {
      white: new Mulberry32(parameterSeeds.white),
      black: new Mulberry32(parameterSeeds.black),
    },
  );
  const secrets = session.exportSecretSnapshot();
  return createHash("sha256")
    .update(canonicalJsonBytes({
      initialPosition: session.publicPositionSnapshot(),
      initialSecrets: {
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
      },
    }))
    .digest("hex");
}

export const schema9AssignmentScheduler: Schema9AssignmentScheduler =
  Object.freeze({
    assignments(
      split: Schema9LedgerSplit,
      gameCount: number,
      seedRoots: Schema9SeedRoots,
    ) {
      if (!Number.isSafeInteger(gameCount) || gameCount <= 0) {
        throw new RangeError("Schema-9 schedule game count must be positive.");
      }
      if (gameCount > 0x1_0000_0000) {
        throw new RangeError(
          "Schema-9 schedule game count cannot exceed the uint32 index space.",
        );
      }
      const checkedRoots = checkedSchema9SeedRoots(seedRoots, split);
      const whiteRules = shuffledRules(
        deriveSimulationStreamSeed(
          checkedRoots[0],
          SCHEDULE_DOMAINS.whiteLabels,
          0,
        ),
      );
      const blackRules = shuffledRules(
        deriveSimulationStreamSeed(
          checkedRoots[0],
          SCHEDULE_DOMAINS.blackLabels,
          0,
        ),
      );
      return (function* assignmentIterator(): IterableIterator<Schema9ExpectedAssignment> {
        for (let gameIndex = 0; gameIndex < gameCount; gameIndex += 1) {
          const slot = gameIndex % whiteRules.length;
          const round = Math.floor(gameIndex / whiteRules.length)
            % whiteRules.length;
          const whiteRuleId = whiteRules[slot];
          const blackRuleId = blackRules[(slot + round) % blackRules.length];
          if (whiteRuleId === undefined || blackRuleId === undefined) {
            throw new Error("Schema-9 schedule lost a rule assignment.");
          }
          const seed = deriveGameSeed(checkedRoots[1], gameIndex);
          const parameterSeeds = Object.freeze({
            white: deriveSimulationStreamSeed(
              checkedRoots[2],
              SCHEDULE_DOMAINS.whiteParameters,
              gameIndex,
            ),
            black: deriveSimulationStreamSeed(
              checkedRoots[2],
              SCHEDULE_DOMAINS.blackParameters,
              gameIndex,
            ),
          });
          yield Object.freeze({
            gameIndex,
            gameId: playerPrivateSimulationGameId(
              seed,
              gameIndex,
              parameterSeeds,
            ),
            seed,
            parameterSeeds,
            whiteRuleId,
            blackRuleId,
            initialReplaySha256: initialReplaySha256(
              whiteRuleId,
              blackRuleId,
              parameterSeeds,
            ),
          });
        }
      }());
    },
  });
