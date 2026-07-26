import { describe, expect, it } from "vitest";
import { classifyBrowserPredictorTrust } from "./model-trust.js";

describe("browser predictor trust classification", () => {
  it("classifies symbolic-only analysis without inventing release approval", () => {
    expect(classifyBrowserPredictorTrust()).toEqual({
      source: "built-in-symbolic",
      trust: "built-in-code",
      releaseApproved: false,
      calibrationMetadata: "none",
    });
  });

  it("keeps a manually opened model unverified", () => {
    expect(classifyBrowserPredictorTrust("v21-hybrid")).toEqual({
      source: "manual-local-file",
      trust: "unverified-local-research",
      releaseApproved: false,
      calibrationMetadata: "none",
    });
    expect(classifyBrowserPredictorTrust("v22-hybrid")).toEqual({
      source: "manual-local-file",
      trust: "unverified-local-research",
      releaseApproved: false,
      calibrationMetadata: "none",
    });
  });

  it("does not trust crafted ensemble calibration metadata", () => {
    expect(classifyBrowserPredictorTrust("v21-hybrid-ensemble")).toEqual({
      source: "manual-local-file",
      trust: "unverified-local-research",
      releaseApproved: false,
      calibrationMetadata: "artifact-declared-simulation-validation",
    });
  });
});
