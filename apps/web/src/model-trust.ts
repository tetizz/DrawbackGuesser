export type BrowserModelVariant =
  | "v1"
  | "v21-hybrid"
  | "v22-hybrid"
  | "v21-hybrid-ensemble";

export type BrowserPredictorTrust =
  | {
      readonly source: "built-in-symbolic";
      readonly trust: "built-in-code";
      readonly releaseApproved: false;
      readonly calibrationMetadata: "none";
    }
  | {
      readonly source: "manual-local-file";
      readonly trust: "unverified-local-research";
      readonly releaseApproved: false;
      readonly calibrationMetadata:
        | "none"
        | "artifact-declared-simulation-validation";
    };

export function classifyBrowserPredictorTrust(
  modelVariant?: BrowserModelVariant,
): BrowserPredictorTrust {
  if (modelVariant === undefined) {
    return Object.freeze({
      source: "built-in-symbolic",
      trust: "built-in-code",
      releaseApproved: false,
      calibrationMetadata: "none",
    });
  }
  return Object.freeze({
    source: "manual-local-file",
    trust: "unverified-local-research",
    releaseApproved: false,
    calibrationMetadata:
      modelVariant === "v21-hybrid-ensemble"
        ? "artifact-declared-simulation-validation"
        : "none",
  });
}
