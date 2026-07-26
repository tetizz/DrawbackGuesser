/**
 * Drawbacks whose legality depends on evaluator facts that an ordinary PGN
 * does not contain. They remain internal model classes, but are not ranked by
 * the standard-PGN browser view.
 */
export const STANDARD_PGN_UNAVAILABLE_DRAWBACK_IDS = Object.freeze([
  "hand-and-gigabrain",
  "ichtyophobe",
] as const);
