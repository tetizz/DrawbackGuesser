import type { DrawbackRule } from "@drawbackengine/drawback-engine";
import { logSumExp, probability } from "./math.js";
import type {
  HypothesisDistribution,
  HypothesisSeed,
  ParameterPosterior,
  ParameterVariant,
  ParameterValuePosterior,
  RulePosterior,
} from "./types.js";

function canonicalize(
  value: unknown,
  ancestors: ReadonlySet<object>,
): string {
  if (value === null) {
    return "null";
  }
  switch (typeof value) {
    case "string":
      return JSON.stringify(value);
    case "boolean":
      return value ? "true" : "false";
    case "number":
      if (!Number.isFinite(value)) {
        throw new TypeError("Hypothesis parameters must contain finite numbers");
      }
      return JSON.stringify(Object.is(value, -0) ? 0 : value);
    case "object": {
      if (ancestors.has(value)) {
        throw new TypeError("Hypothesis parameters must not contain cycles");
      }
      const nextAncestors = new Set(ancestors);
      nextAncestors.add(value);
      if (Array.isArray(value)) {
        const propertyNames = Object.getOwnPropertyNames(value).filter(
          (propertyName) => propertyName !== "length",
        );
        for (const propertyName of propertyNames) {
          const index = Number(propertyName);
          if (
            !Number.isInteger(index)
            || index < 0
            || index >= value.length
            || String(index) !== propertyName
          ) {
            throw new TypeError(
              "Hypothesis parameter arrays must not contain named properties",
            );
          }
        }
        if (propertyNames.length !== value.length) {
          throw new TypeError(
            "Hypothesis parameter arrays must not contain holes",
          );
        }
        const items: string[] = [];
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
              "Hypothesis parameter arrays require own enumerable data entries",
            );
          }
          items.push(canonicalize(descriptor.value, nextAncestors));
        }
        return `[${items.join(",")}]`;
      }
      const objectValue = value as Record<string, unknown>;
      return `{${Object.keys(objectValue)
        .sort()
        .map(
          (key) =>
            `${JSON.stringify(key)}:${canonicalize(
              objectValue[key],
              nextAncestors,
            )}`,
        )
        .join(",")}}`;
    }
    case "undefined":
    case "bigint":
    case "function":
    case "symbol":
      throw new TypeError(
        `Unsupported hypothesis parameter value: ${typeof value}`,
      );
  }
  throw new TypeError("Unsupported hypothesis parameter value");
}

export function canonicalParameterValue(value: unknown): string {
  return canonicalize(value, new Set());
}

export function canonicalHypothesisId(
  drawbackId: string,
  parameters: Readonly<Record<string, unknown>>,
): string {
  if (drawbackId.length === 0) {
    throw new TypeError("drawbackId must not be empty");
  }
  return `${drawbackId}::${canonicalParameterValue(parameters)}`;
}

/**
 * Expands one drawback into parameter variants while preserving the configured
 * total prior assigned to that drawback.
 */
export function expandHypothesisSeeds<
  State,
  Parameters extends object,
>(
  rule: DrawbackRule<State, Parameters>,
  variants: readonly ParameterVariant<Parameters>[],
  drawbackPrior = 1,
): readonly HypothesisSeed<unknown, Record<string, unknown>>[] {
  if (!Number.isFinite(drawbackPrior) || drawbackPrior <= 0) {
    throw new RangeError("drawbackPrior must be finite and greater than zero");
  }
  if (variants.length === 0) {
    throw new RangeError("At least one parameter variant is required");
  }
  const weights = variants.map((variant) => {
    const weight = variant.weight ?? 1;
    if (!Number.isFinite(weight) || weight <= 0) {
      throw new RangeError("Variant weights must be finite and greater than zero");
    }
    return weight;
  });
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const ids = new Set<string>();
  return Object.freeze(
    variants.map((variant, index) => {
      const parameters = variant.parameters as Readonly<
        Record<string, unknown>
      >;
      const id = canonicalHypothesisId(rule.id, parameters);
      if (ids.has(id)) {
        throw new RangeError(`Duplicate parameter variant: ${id}`);
      }
      ids.add(id);
      const weight = weights[index];
      if (weight === undefined) {
        throw new Error("Variant weight expansion changed length");
      }
      return Object.freeze({
        rule: rule as DrawbackRule<unknown, Record<string, unknown>>,
        parameters: Object.freeze({ ...parameters }),
        priorProbability: drawbackPrior * (weight / totalWeight),
      });
    }),
  );
}

export function aggregateRulePosteriors(
  distribution: HypothesisDistribution,
): readonly RulePosterior[] {
  const grouped = new Map<
    string,
    { logs: number[]; live: number; variants: number }
  >();
  for (const hypothesis of distribution.hypotheses) {
    const group = grouped.get(hypothesis.drawbackId) ?? {
      logs: [],
      live: 0,
      variants: 0,
    };
    group.logs.push(hypothesis.logProbability);
    group.live += hypothesis.eliminated ? 0 : 1;
    group.variants += 1;
    grouped.set(hypothesis.drawbackId, group);
  }
  return Object.freeze(
    [...grouped.entries()]
      .map(([drawbackId, group]) => {
        const logProbability = logSumExp(group.logs);
        return Object.freeze({
          drawbackId,
          logProbability,
          probability: probability(logProbability),
          eliminated: group.live === 0,
          liveVariantCount: group.live,
          variantCount: group.variants,
        });
      })
      .sort(
        (left, right) =>
          right.probability - left.probability ||
          left.drawbackId.localeCompare(right.drawbackId),
      ),
  );
}

export function aggregateParameterPosteriors(
  distribution: HypothesisDistribution,
  drawbackId: string,
): readonly ParameterPosterior[] {
  const variants = distribution.hypotheses.filter(
    (hypothesis) => hypothesis.drawbackId === drawbackId,
  );
  const drawbackProbability = variants.reduce(
    (sum, hypothesis) => sum + probability(hypothesis.logProbability),
    0,
  );
  const parameterNames = new Set(
    variants.flatMap((hypothesis) => Object.keys(hypothesis.parameters)),
  );

  return Object.freeze(
    [...parameterNames].sort().map((parameter) => {
      const values = new Map<
        string,
        { value: unknown; probability: number; variants: number }
      >();
      for (const hypothesis of variants) {
        if (!Object.hasOwn(hypothesis.parameters, parameter)) {
          continue;
        }
        const value = hypothesis.parameters[parameter];
        const canonicalValue = canonicalParameterValue(value);
        const current = values.get(canonicalValue) ?? {
          value,
          probability: 0,
          variants: 0,
        };
        current.probability += probability(hypothesis.logProbability);
        current.variants += 1;
        values.set(canonicalValue, current);
      }
      const valuePosteriors: readonly ParameterValuePosterior[] = Object.freeze(
        [...values.entries()]
          .map(([canonicalValue, value]) =>
            Object.freeze({
              canonicalValue,
              value: value.value,
              probability: value.probability,
              conditionalProbability:
                drawbackProbability === 0
                  ? 0
                  : value.probability / drawbackProbability,
              variantCount: value.variants,
            }),
          )
          .sort(
            (left, right) =>
              right.probability - left.probability ||
              left.canonicalValue.localeCompare(right.canonicalValue),
          ),
      );
      const coveredProbability = valuePosteriors.reduce(
        (sum, value) => sum + value.probability,
        0,
      );
      const normalizedValues = Object.freeze(
        valuePosteriors.map((value) =>
          Object.freeze({
            ...value,
            conditionalProbability:
              coveredProbability === 0
                ? 0
                : value.probability / coveredProbability,
          }),
        ),
      );
      return Object.freeze({
        drawbackId,
        parameter,
        drawbackProbability,
        coveredProbability,
        values: normalizedValues,
      });
    }),
  );
}
