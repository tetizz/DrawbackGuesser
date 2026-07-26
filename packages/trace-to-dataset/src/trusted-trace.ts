import {
  PLAYER_PRIVATE_SIMULATION_TRACE_FORMAT,
  PRIVATE_SIMULATION_TRACE_FORMAT,
  parsePlayerPrivateSimulationTraceRecord,
  parsePrivateSimulationTraceRecord,
  type PlayerPrivateSimulationTraceRecord,
  type PrivateSimulationTraceRecord,
} from "@drawbackengine/simulation-trace";

export type TrustedSimulationTraceRecord =
  | PrivateSimulationTraceRecord
  | PlayerPrivateSimulationTraceRecord;

export function parseTrustedSimulationTraceRecord(
  input: unknown,
): TrustedSimulationTraceRecord {
  if (
    typeof input !== "object"
    || input === null
    || Array.isArray(input)
  ) {
    throw new TypeError("Trusted Engine trace must be an object.");
  }
  const format = (input as Readonly<Record<string, unknown>>)["format"];
  if (format === PRIVATE_SIMULATION_TRACE_FORMAT) {
    return parsePrivateSimulationTraceRecord(input);
  }
  if (format === PLAYER_PRIVATE_SIMULATION_TRACE_FORMAT) {
    return parsePlayerPrivateSimulationTraceRecord(input);
  }
  throw new TypeError("Trusted Engine trace format is unsupported.");
}

export function parseTrustedSimulationTraceLine(
  line: string,
): TrustedSimulationTraceRecord {
  let input: unknown;
  try {
    input = JSON.parse(line) as unknown;
  } catch (error: unknown) {
    throw new SyntaxError("Trusted Engine trace line is not valid JSON.", {
      cause: error,
    });
  }
  return parseTrustedSimulationTraceRecord(input);
}
