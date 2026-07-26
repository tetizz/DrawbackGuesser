import {
  AUDITED_CAPTURABLE_KING_RULE_IDS,
  activeVolcanoRule,
  barbarianRageRule,
  battleFatigueRule,
  boardRelativeRules,
  bridgeOverTroubledWaterRule,
  cessRule,
  checkersRule,
  colorblindRule,
  comfortZoneRule,
  conscientiousObjectorsRule,
  crenellationsRule,
  evenKeeledRule,
  expeditionRule,
  eyeForAnEyeRule,
  eyeOfSauronRule,
  filterColorblindMoves,
  filterHandAndBrainlessMoves,
  filterObsessionMoves,
  filterWindsOfFateMoves,
  forwardMarchRule,
  dragRule,
  gamblerRule,
  handAndBrainlessRule,
  handAndGigabrainRule,
  historyFilterRules,
  horseTranquilizerRule,
  entrenchedRule,
  justPassingThroughRule,
  lameDuckRule,
  noShufflingRule,
  numberOfTheBeastRule,
  OBSERVED_CENTRAL_SQUARES,
  oddballRule,
  obsessionRule,
  pacmanRule,
  quitHorsingAroundRule,
  remorsefulRule,
  spiceOfLifeRule,
  shadowQueenRule,
  stopStallingRule,
  trophyWifeRule,
  theocracyRule,
  truantRule,
  trueGentlemanRule,
  untitledDuckDrawbackRule,
  veganRule,
  communityRules,
  communityRulesTwo,
  falseProphetsRule,
  lossRules,
  observedRulesThree,
  preparedExecutableRules,
  geometricObservedRules,
  responseHistoryRules,
  nextStatefulRules,
  attackObservedRules,
  remainingResponseRules,
  remainingStatefulRules,
  bishopFanClubRule,
  blindedByTheSunRule,
  fischerRandomRule,
  OBSERVED_BLINDED_SQUARES,
  respectfulRule,
  reflectiveRule,
  reconnaissanceRule,
  rookFanClubRule,
  shapeshifterRule,
  unspoolingRule,
  oohShinyRule,
  ichtyophobeRule,
  windsOfFateRule,
  capturableKingIrresistibleRule,
  femmeFataleRule,
  nurturerRule,
  OBSERVED_TRIPLE_PLAY_TYPES,
  triplePlayRule,
  youBestNotMissRule,
} from "@drawbackengine/drawback-engine";
import type {
  ChessMove,
  PieceType,
} from "@drawbackengine/drawback-engine";
import type { PlayerColor } from "@drawbackengine/shared";
import {
  asExternalConstraintHypothesisSeed,
  asHypothesisSeed,
  asRerandomizedHypothesisSeed,
} from "./predictor.js";
import { expandHypothesisSeeds } from "./parameters.js";
import type {
  PredictionSeed,
  RerandomizedHypothesisSeed,
} from "./types.js";

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"] as const;

export const DEFAULT_HYPOTHESIS_RULE_IDS = [
  "vegan",
  "lame-duck",
  "checkers",
  "truant",
  "spice-of-life",
  "true-gentleman",
  "trophy-wife",
  "cess",
  "forward-march",
  "pacman",
  "oddball",
  "even-keeled",
  "quit-horsing-around",
  "remorseful",
  "battle-fatigue",
  "eye-for-an-eye",
  "barbarian-rage",
  "conscientious-objectors",
  "horse-tranquilizer",
  "untitled-duck-drawback",
  "just-passing-through",
  "gambler",
  "number-of-the-beast",
  "shadow-queen",
  "entrenched",
  "no-shuffling",
  "stop-stalling",
  "greedy",
  "professional-courtesy",
  "snipers",
  "stay-at-home-mom",
  "elephants-fear-mice",
  "far-sighted",
  "whites-of-their-eyes",
  "champing-at-the-bit",
  "scent-of-blood",
  "indecisive",
  "control-center",
  "out-of-breath",
  "queen-bee",
  "alternator",
  "hopscotch",
  "bottled-lighting",
  "chivalry",
  "covering-fire",
  "escort-mission",
  "evil-twin",
  "exclusivity-clause",
  "leaps-and-bounds",
  "left-for-dead",
  "outflanked",
  "punching-down",
  "simplifier",
  "bipartisanship",
  "false-prophets",
  "abstinence",
  "always-check-it-might-be-mate",
  "boastful",
  "closed-book",
  "hold-them-back",
  "homeland-security",
  "ivory-tower",
  "king-of-the-hill",
  "modest",
  "simp",
  "tower-defense",
  "warlord",
  "lucky",
  "eisoptrophobia",
  "gloomstalker",
  "noblesse-oblige",
  "bongcloud",
  "eat-your-vegetables",
  "horse-eats-first",
  "messy-divorce",
  "body-snatcher",
  "castle-doctrine",
  "my-kingdom-for-a-horse",
  "octomom",
  "pawn-battle",
  "edgelord",
  "botez-gambit",
  "cheerleaders",
  "noble-steed",
  "pack-mentality",
  "separation-anxiety",
  "separation-of-church-and-state",
  "sibling-rivalry",
  "social-distancing",
  "spread-out",
  "torchlight",
  "royal-berth",
  "peons-first",
  "power-cells",
  "leading-the-charge",
  "scouting-ahead",
  "diplomatic-immunity",
  "flatterer",
  "hipster",
  "hedonic-treadmill",
  "ladies-first",
  "centralized-command",
  "royal-jubilee",
  "monkey-see",
  "haunted",
  "scorched-earth",
  "turn-the-other-cheek",
  "velociraptor",
  "windup-toys",
  "doctor-octopus",
  "cowering-in-fear",
  "crenellations",
  "theocracy",
  "active-volcano",
  "comfort-zone",
  "crossing-the-rubicon",
  "true-love",
  "lethal-attraction",
  "thunderdome",
  "irresistible",
  "prima-donna",
  "inside-the-lines",
  "boxing-with-shadow",
  "cowardly",
  "going-the-distance",
  "left-to-right",
  "relay-race",
  "religious-dispute",
  "simon-says",
  "superstitious",
  "torpedos",
  "stir-crazy",
  "bloodthirsty",
  "fixation",
  "leveling-up",
  "quicksand",
  "absolution",
  "moving-day",
  "siege",
  "deer-in-the-headlights",
  "jumpy",
  "medusa",
  "stand-your-ground",
  "unrequited-love",
  "helicopter-parent",
  "paranoid",
  "rook-buddies",
  "atomic-bomb",
  "get-down-mr-president",
  "guerilla-tactics",
  "prince-charming",
  "savior-complex",
  "shellshocked",
  "skittish",
  "sleepy-king",
  "three-check",
  "friendly-fire",
  "protected-pawns",
  "rook-on-the-seventh",
  "rising-water",
  "queen-disguise",
  "now-kiss",
  "bishop-fan-club",
  "rook-fan-club",
  "respectful",
  "shapeshifter",
  "fischer-random",
  "unspooling",
  "blinded-by-the-sun",
  "colorblind",
  "hand-and-brainless",
  "obsession",
  "winds-of-fate",
  "expedition",
  "reflective",
  "eye-of-sauron",
  "drag",
  "ooh-shiny",
  "bridge-over-troubled-water",
  "reconnaissance",
  "hand-and-gigabrain",
  "ichtyophobe",
] as const;

export const CAPTURABLE_HYPOTHESIS_RULE_IDS =
  AUDITED_CAPTURABLE_KING_RULE_IDS;

if (
  preparedExecutableRules.length !== DEFAULT_HYPOTHESIS_RULE_IDS.length ||
  new Set(DEFAULT_HYPOTHESIS_RULE_IDS).size !==
    DEFAULT_HYPOTHESIS_RULE_IDS.length ||
  DEFAULT_HYPOTHESIS_RULE_IDS.some(
    (id) => !preparedExecutableRules.some((rule) => rule.id === id),
  ) ||
  preparedExecutableRules.some(
    (rule) => !DEFAULT_HYPOTHESIS_RULE_IDS.includes(
      rule.id as (typeof DEFAULT_HYPOTHESIS_RULE_IDS)[number],
    ),
  )
) {
  throw new Error("Predictor rule IDs are out of sync with the executable catalog.");
}

export const GAMBLER_OUTCOME_COUNT = 6;

export interface HypothesisCoverage {
  readonly drawbackId: string;
  readonly mode: "analytic" | "exact" | "sampled";
  readonly variantCount: number;
  readonly note: string;
}

export const HYPOTHESIS_COVERAGE: readonly HypothesisCoverage[] =
  Object.freeze([
    {
      drawbackId: "untitled-duck-drawback",
      mode: "exact",
      variantCount: 64,
      note: "All 64 hidden squares are enumerated.",
    },
    {
      drawbackId: "just-passing-through",
      mode: "exact",
      variantCount: 8,
      note: "All eight hidden ranks are enumerated.",
    },
    {
      drawbackId: "gambler",
      mode: "analytic",
      variantCount: GAMBLER_OUTCOME_COUNT,
      note:
        "Conservative per-turn approximation: all six forbidden piece-type " +
        "outcomes are marginalized independently, so sampled seeds cannot " +
        "create false hard elimination. Hidden 32-bit seed correlations are " +
        "not inferred.",
    },
    {
      drawbackId: "crenellations",
      mode: "exact",
      variantCount: 2,
      note: "Both hidden destination-square colors are enumerated.",
    },
    {
      drawbackId: "theocracy",
      mode: "exact",
      variantCount: 2,
      note: "Both hidden fullmove parities are enumerated.",
    },
    {
      drawbackId: "active-volcano",
      mode: "exact",
      variantCount: 8,
      note:
        "All eight middle-board candidates inferred from observations are " +
        "enumerated; the official parameter domain remains unverified.",
    },
    {
      drawbackId: "comfort-zone",
      mode: "exact",
      variantCount: 8,
      note:
        "All eight middle-board candidates inferred from observations are " +
        "enumerated; the official parameter domain remains unverified.",
    },
    {
      drawbackId: "blinded-by-the-sun",
      mode: "exact",
      variantCount: 4,
      note:
        "All four central squares observed in the site corpus are enumerated; " +
        "the official parameter domain remains unverified.",
    },
    {
      drawbackId: "colorblind",
      mode: "analytic",
      variantCount: 2,
      note:
        "Both independently rerandomized destination-square colors are " +
        "marginalized on every affected-player turn.",
    },
    {
      drawbackId: "hand-and-brainless",
      mode: "analytic",
      variantCount: 6,
      note:
        "All six independently rerandomized primary mover types are " +
        "marginalized on every affected-player turn.",
    },
    {
      drawbackId: "obsession",
      mode: "analytic",
      variantCount: 64,
      note:
        "All 64 independently rerandomized target squares are marginalized " +
        "on every affected-player turn.",
    },
    {
      drawbackId: "winds-of-fate",
      mode: "analytic",
      variantCount: 2,
      note:
        "Both independently rerandomized player-relative directions are " +
        "marginalized on every affected-player turn.",
    },
  ]);

interface AnalyticState {
  readonly movesApplied: number;
}

function analyticRerandomizedSeed<Outcome>(configuration: {
  readonly drawbackId: string;
  readonly name: string;
  readonly outcomes: readonly Outcome[];
  readonly filter: (
    color: PlayerColor,
    outcome: Readonly<Outcome>,
    moves: readonly ChessMove[],
  ) => readonly ChessMove[];
}): RerandomizedHypothesisSeed<unknown, unknown> {
  const probability = 1 / configuration.outcomes.length;
  return asRerandomizedHypothesisSeed<AnalyticState, Outcome>({
    kind: "rerandomized",
    drawbackId: configuration.drawbackId,
    name: configuration.name,
    initialize: ({ color, position }) => ({
      movesApplied: position.history.filter(
        (move) => move.color === color,
      ).length,
    }),
    outcomes: () =>
      configuration.outcomes.map((outcome) => ({
        outcome,
        probability,
      })),
    filterLegalMoves: (context, outcome, moves) =>
      configuration.filter(context.color, outcome, moves),
    applyObservedMove: (context) => ({
      movesApplied: context.state.movesApplied + 1,
    }),
  });
}

const ANALYTIC_RERANDOMIZED_SEEDS = Object.freeze([
  analyticRerandomizedSeed({
    drawbackId: gamblerRule.id,
    name: gamblerRule.name,
    outcomes: [
      "pawn",
      "knight",
      "bishop",
      "rook",
      "queen",
      "king",
    ] as const satisfies readonly PieceType[],
    filter: (_color, forbidden, moves) =>
      moves.filter((move) => move.piece !== forbidden),
  }),
  analyticRerandomizedSeed({
    drawbackId: colorblindRule.id,
    name: colorblindRule.name,
    outcomes: ["dark", "light"] as const,
    filter: (_color, forbidden, moves) =>
      filterColorblindMoves(forbidden, moves),
  }),
  analyticRerandomizedSeed({
    drawbackId: handAndBrainlessRule.id,
    name: handAndBrainlessRule.name,
    outcomes: [
      "pawn",
      "knight",
      "bishop",
      "rook",
      "queen",
      "king",
    ] as const satisfies readonly PieceType[],
    filter: (_color, required, moves) =>
      filterHandAndBrainlessMoves(required, moves),
  }),
  analyticRerandomizedSeed({
    drawbackId: obsessionRule.id,
    name: obsessionRule.name,
    outcomes: FILES.flatMap((file) =>
      Array.from(
        { length: 8 },
        (_, index) => `${file}${String(index + 1)}`,
      ),
    ),
    filter: (_color, target, moves) =>
      filterObsessionMoves(target, moves),
  }),
  analyticRerandomizedSeed({
    drawbackId: windsOfFateRule.id,
    name: windsOfFateRule.name,
    outcomes: ["left", "right"] as const,
    filter: (color, forbidden, moves) =>
      filterWindsOfFateMoves(color, forbidden, moves),
  }),
]);

export function createDefaultHypothesisSeeds(): readonly PredictionSeed[] {
  const parameterless = [
    asHypothesisSeed(veganRule, {}, 1),
    asHypothesisSeed(lameDuckRule, {}, 1),
    asHypothesisSeed(checkersRule, {}, 1),
    asHypothesisSeed(truantRule, {}, 1),
    asHypothesisSeed(spiceOfLifeRule, {}, 1),
    asHypothesisSeed(trueGentlemanRule, {}, 1),
    asHypothesisSeed(trophyWifeRule, {}, 1),
    asHypothesisSeed(cessRule, {}, 1),
    asHypothesisSeed(forwardMarchRule, {}, 1),
    asHypothesisSeed(pacmanRule, {}, 1),
    asHypothesisSeed(oddballRule, {}, 1),
    asHypothesisSeed(evenKeeledRule, {}, 1),
    asHypothesisSeed(quitHorsingAroundRule, {}, 1),
    asHypothesisSeed(remorsefulRule, {}, 1),
    asHypothesisSeed(battleFatigueRule, {}, 1),
    asHypothesisSeed(eyeForAnEyeRule, {}, 1),
    asHypothesisSeed(barbarianRageRule, {}, 1),
    asHypothesisSeed(conscientiousObjectorsRule, {}, 1),
    asHypothesisSeed(horseTranquilizerRule, {}, 1),
    asHypothesisSeed(numberOfTheBeastRule, {}, 1),
    asHypothesisSeed(shadowQueenRule, {}, 1),
    asHypothesisSeed(entrenchedRule, {}, 1),
    asHypothesisSeed(noShufflingRule, {}, 1),
    asHypothesisSeed(stopStallingRule, {}, 1),
    ...communityRules.map((rule) => asHypothesisSeed(rule, {}, 1)),
    ...communityRulesTwo.map((rule) => asHypothesisSeed(rule, {}, 1)),
    asHypothesisSeed(falseProphetsRule, {}, 1),
    ...lossRules.map((rule) => asHypothesisSeed(rule, {}, 1)),
    ...observedRulesThree.map((rule) => asHypothesisSeed(rule, {}, 1)),
    ...boardRelativeRules.map((rule) => asHypothesisSeed(rule, {}, 1)),
    ...historyFilterRules.map((rule) => asHypothesisSeed(rule, {}, 1)),
    ...geometricObservedRules.map((rule) => asHypothesisSeed(rule, {}, 1)),
    ...responseHistoryRules.map((rule) => asHypothesisSeed(rule, {}, 1)),
    ...nextStatefulRules.map((rule) => asHypothesisSeed(rule, {}, 1)),
    ...attackObservedRules.map((rule) => asHypothesisSeed(rule, {}, 1)),
    ...remainingResponseRules.map((rule) => asHypothesisSeed(rule, {}, 1)),
    ...remainingStatefulRules.map((rule) => asHypothesisSeed(rule, {}, 1)),
    asHypothesisSeed(bishopFanClubRule, {}, 1),
    asHypothesisSeed(rookFanClubRule, {}, 1),
    asHypothesisSeed(respectfulRule, {}, 1),
    asHypothesisSeed(shapeshifterRule, {}, 1),
    asHypothesisSeed(fischerRandomRule, {}, 1),
    asHypothesisSeed(unspoolingRule, {}, 1),
    asHypothesisSeed(expeditionRule, {}, 1),
    asHypothesisSeed(reflectiveRule, {}, 1),
    asHypothesisSeed(eyeOfSauronRule, {}, 1),
    asHypothesisSeed(dragRule, {}, 1),
    asHypothesisSeed(oohShinyRule, {}, 1),
    asHypothesisSeed(bridgeOverTroubledWaterRule, {}, 1),
    asHypothesisSeed(reconnaissanceRule, {}, 1),
  ];
  const squares = FILES.flatMap((file) =>
    Array.from({ length: 8 }, (_, index) => ({
      parameters: { square: `${file}${String(index + 1)}` },
    })),
  );
  const ranks = Array.from({ length: 8 }, (_, index) => ({
    parameters: { rank: index + 1 },
  }));
  const observedCentralSquares = OBSERVED_CENTRAL_SQUARES.map((square) => ({
    parameters: { square },
  }));
  return Object.freeze([
    ...parameterless,
    ...expandHypothesisSeeds(untitledDuckDrawbackRule, squares, 1),
    ...expandHypothesisSeeds(justPassingThroughRule, ranks, 1),
    ...expandHypothesisSeeds(
      crenellationsRule,
      [
        { parameters: { squareColor: "light" as const } },
        { parameters: { squareColor: "dark" as const } },
      ],
      1,
    ),
    ...expandHypothesisSeeds(
      theocracyRule,
      [
        { parameters: { captureParity: "odd" as const } },
        { parameters: { captureParity: "even" as const } },
      ],
      1,
    ),
    ...expandHypothesisSeeds(activeVolcanoRule, observedCentralSquares, 1),
    ...expandHypothesisSeeds(comfortZoneRule, observedCentralSquares, 1),
    ...expandHypothesisSeeds(
      blindedByTheSunRule,
      OBSERVED_BLINDED_SQUARES.map((square) => ({
        parameters: { square },
      })),
      1,
    ),
    ...ANALYTIC_RERANDOMIZED_SEEDS,
    asExternalConstraintHypothesisSeed(handAndGigabrainRule, {}, 1),
    asExternalConstraintHypothesisSeed(ichtyophobeRule, {}, 1),
  ]);
}

/**
 * Exact hypothesis particles for the audited capturable-king authority.
 *
 * Triple Play has two observed hidden parameter values, while the other nine
 * rules are parameterless. The public rule order is owned by DrawbackEngine's
 * audited authority allowlist so labels cannot drift from executable support.
 */
export function createCapturableHypothesisSeeds():
  readonly PredictionSeed[] {
  const parameterless = [
    asHypothesisSeed(veganRule, {}, 1),
    asHypothesisSeed(lameDuckRule, {}, 1),
    asHypothesisSeed(checkersRule, {}, 1),
    asHypothesisSeed(truantRule, {}, 1),
    asHypothesisSeed(spiceOfLifeRule, {}, 1),
    asHypothesisSeed(femmeFataleRule, {}, 1),
    asHypothesisSeed(nurturerRule, {}, 1),
    asHypothesisSeed(youBestNotMissRule, {}, 1),
    asHypothesisSeed(capturableKingIrresistibleRule, {}, 1),
  ];
  const seeds = Object.freeze([
    ...parameterless,
    ...expandHypothesisSeeds(
      triplePlayRule,
      OBSERVED_TRIPLE_PLAY_TYPES.map((requiredType) => ({
        parameters: { requiredType },
      })),
      1,
    ),
  ]);
  const represented = new Set(seeds.map((seed) => seed.rule.id));
  if (
    represented.size !== CAPTURABLE_HYPOTHESIS_RULE_IDS.length
    || CAPTURABLE_HYPOTHESIS_RULE_IDS.some((id) => !represented.has(id))
  ) {
    throw new Error(
      "Capturable hypothesis seeds are out of sync with the audited Engine catalog.",
    );
  }
  return seeds;
}
