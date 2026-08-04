import { throwIfSchema9Aborted } from "./schema9-ledger-types.js";

function asError(value: unknown, label: string): Error {
  return value instanceof Error ? value : new Error(label, { cause: value });
}

/** Aborts siblings on first failure and waits for every cleanup to settle. */
export async function runSchema9LinkedTaskGroup<T>(
  tasks: readonly ((signal: AbortSignal) => Promise<T>)[],
  parentSignal: AbortSignal | undefined,
  label: string,
): Promise<readonly T[]> {
  throwIfSchema9Aborted(parentSignal, label);
  const controller = new AbortController();
  const signal = parentSignal === undefined
    ? controller.signal
    : AbortSignal.any([parentSignal, controller.signal]);
  let primaryFailure: Error | undefined;
  const pending = tasks.map(async (task) => {
    try {
      return await task(signal);
    } catch (error: unknown) {
      const failure = asError(error, `${label} task failed.`);
      primaryFailure ??= failure;
      if (!controller.signal.aborted) {
        controller.abort(failure);
      }
      throw failure;
    }
  });
  const settled = await Promise.allSettled(pending);
  const failures: unknown[] = [];
  for (const result of settled) {
    if (
      result.status === "rejected"
      && !failures.some((failure) => failure === result.reason)
    ) {
      failures.push(asError(result.reason, `${label} task failed.`));
    }
  }
  if (parentSignal?.aborted === true) {
    const parentFailure = asError(
      parentSignal.reason,
      `${label} was interrupted.`,
    );
    if (!failures.some((failure) => failure === parentFailure)) {
      failures.unshift(parentFailure);
    }
  } else if (
    primaryFailure !== undefined
    && !failures.some((failure) => failure === primaryFailure)
  ) {
    failures.unshift(primaryFailure);
  }
  if (failures.length === 1) {
    throw failures[0];
  }
  if (failures.length > 1) {
    throw new AggregateError(failures, `${label} failed and sibling cleanup settled.`);
  }
  return Object.freeze(settled.map((result) => {
    if (result.status !== "fulfilled") {
      throw new Error(`${label} settled without a result.`);
    }
    return result.value;
  }));
}
