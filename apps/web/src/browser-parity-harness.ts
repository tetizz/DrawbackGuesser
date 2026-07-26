import type {
  PgnAnalysisWorkerRequest,
  PgnAnalysisWorkerResponse,
} from "./pgn-analysis-worker-protocol.js";

interface ExpectedHead {
  readonly probabilities: Readonly<Record<string, number>>;
  readonly topIds: readonly string[];
  readonly hardZeroIds: readonly string[];
}

interface ParityCase {
  readonly id: string;
  readonly pgn: string;
  readonly pgnSha256: string;
  readonly expected: {
    readonly white: ExpectedHead;
    readonly black: ExpectedHead;
  };
}

interface ParityInput {
  readonly format: "drawbacktrainer-browser-parity-input";
  readonly version: 1;
  readonly browserArtifactSha256: string;
  readonly fixtureSha256: string;
  readonly protocolId: string;
  readonly partition: Readonly<Record<string, unknown>>;
  readonly bindings: Readonly<Record<string, unknown>>;
  readonly publicFixture: Readonly<Record<string, unknown>>;
  readonly cases: readonly ParityCase[];
}

interface CaseResult {
  readonly id: string;
  readonly maximumAbsoluteDifference: number;
  readonly topKIdentical: boolean;
  readonly hardZeroSetsIdentical: boolean;
  readonly analysis: unknown;
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null;
}

const resultElement = document.querySelector<HTMLPreElement>(
  "#browser-parity-result",
);
if (resultElement === null) {
  throw new Error("Browser parity result element is absent.");
}
const output: HTMLPreElement = resultElement;

function post(
  worker: Worker,
  request: PgnAnalysisWorkerRequest,
): Promise<PgnAnalysisWorkerResponse> {
  return new Promise((resolve, reject) => {
    const listener = (event: MessageEvent<PgnAnalysisWorkerResponse>): void => {
      if (
        event.data.requestId !== request.requestId ||
        event.data.type === "progress"
      ) {
        return;
      }
      worker.removeEventListener("message", listener);
      if (event.data.type === "error") {
        reject(new Error(event.data.error.message));
        return;
      }
      resolve(event.data);
    };
    worker.addEventListener("message", listener);
    worker.postMessage(request);
  });
}

function sortedIds(probabilities: Readonly<Record<string, number>>): string[] {
  return Object.entries(probabilities)
    .sort(([leftId, left], [rightId, right]) =>
      right - left || leftId.localeCompare(rightId)
    )
    .map(([id]) => id);
}

async function sha256Utf8(value: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function compareHead(
  guesses: readonly {
    readonly id: string;
    readonly confidence: number;
  }[],
  expected: ExpectedHead,
): Omit<CaseResult, "id" | "analysis"> {
  const actual = Object.fromEntries(
    guesses.map(({ id, confidence }) => [id, confidence]),
  );
  const expectedIds = Object.keys(expected.probabilities).sort();
  if (
    Object.keys(actual).sort().join("\n") !== expectedIds.join("\n")
  ) {
    throw new Error("Browser and Python drawback vocabularies differ.");
  }
  const maximumAbsoluteDifference = Math.max(
    0,
    ...expectedIds.map((id) =>
      Math.abs((actual[id] ?? Number.NaN) - (expected.probabilities[id] ?? 0))
    ),
  );
  const hardZeroIds = expectedIds
    .filter((id) => actual[id] === 0)
    .sort();
  return {
    maximumAbsoluteDifference,
    topKIdentical:
      sortedIds(actual).slice(0, expected.topIds.length).join("\n") ===
      expected.topIds.join("\n"),
    hardZeroSetsIdentical:
      hardZeroIds.join("\n") === [...expected.hardZeroIds].sort().join("\n"),
  };
}

async function run(): Promise<void> {
  const [inputResponse, artifactResponse] = await Promise.all([
    fetch("./browser-parity-input.json", { cache: "no-store" }),
    fetch("./browser-model.json", { cache: "no-store" }),
  ]);
  if (!inputResponse.ok || !artifactResponse.ok) {
    throw new Error("Browser parity inputs could not be loaded.");
  }
  const rawInput: unknown = await inputResponse.json();
  const artifactText = await artifactResponse.text();
  if (
    !isRecord(rawInput) ||
    rawInput["format"] !== "drawbacktrainer-browser-parity-input" ||
    rawInput["version"] !== 1 ||
    !Array.isArray(rawInput["cases"]) ||
    rawInput["cases"].length === 0
  ) {
    throw new Error("Browser parity input contract is invalid.");
  }
  const input = rawInput as unknown as ParityInput;
  const worker = new Worker(
    new URL("./pgn-analysis.worker.ts", import.meta.url),
    { type: "module" },
  );
  const loaded = await post(worker, {
    type: "load-model",
    requestId: 1,
    artifactText,
    expectedSha256: input.browserArtifactSha256,
  });
  if (loaded.type !== "model-loaded") {
    throw new Error("Browser Worker did not load the model.");
  }
  const cases: CaseResult[] = [];
  for (const [index, testCase] of input.cases.entries()) {
    if ((await sha256Utf8(testCase.pgn)) !== testCase.pgnSha256) {
      throw new Error("Browser parity PGN digest differs.");
    }
    const response = await post(worker, {
      type: "analyze",
      requestId: index + 2,
      pgn: testCase.pgn,
      neuralArtifactSha256: input.browserArtifactSha256,
    });
    if (response.type !== "result") {
      throw new Error("Browser Worker did not return an analysis.");
    }
    const white = compareHead(response.result.finalWhite, testCase.expected.white);
    const black = compareHead(response.result.finalBlack, testCase.expected.black);
    cases.push({
      id: testCase.id,
      maximumAbsoluteDifference: Math.max(
        white.maximumAbsoluteDifference,
        black.maximumAbsoluteDifference,
      ),
      topKIdentical: white.topKIdentical && black.topKIdentical,
      hardZeroSetsIdentical:
        white.hardZeroSetsIdentical && black.hardZeroSetsIdentical,
      analysis: response.result,
    });
  }
  worker.terminate();
  output.textContent = JSON.stringify({
    format: "drawbacktrainer-browser-worker-transcript",
    version: 1,
    protocolId: input.protocolId,
    browserArtifactSha256: input.browserArtifactSha256,
    fixtureSha256: input.fixtureSha256,
    partition: input.partition,
    bindings: input.bindings,
    publicFixture: input.publicFixture,
    workerE2ePassed: true,
    maximumAbsoluteDifference: Math.max(
      ...cases.map(({ maximumAbsoluteDifference }) =>
        maximumAbsoluteDifference
      ),
    ),
    topKIdentical: cases.every(({ topKIdentical }) => topKIdentical),
    hardZeroSetsIdentical: cases.every(
      ({ hardZeroSetsIdentical }) => hardZeroSetsIdentical,
    ),
    cases,
  });
  document.documentElement.dataset["parityComplete"] = "true";
}

void run().catch((error: unknown) => {
  output.textContent = JSON.stringify({
    format: "drawbacktrainer-browser-worker-transcript",
    version: 1,
    workerE2ePassed: false,
    error: error instanceof Error ? error.message : String(error),
  });
  document.documentElement.dataset["parityComplete"] = "true";
});
