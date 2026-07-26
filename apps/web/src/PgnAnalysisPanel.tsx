import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  Copy,
  Download,
  FileSearch,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import {
  MAX_PGN_INPUT_BYTES,
  PgnParseError,
  type PgnAnalysisResult,
  type PgnGuess,
} from "./pgn-analysis.js";
import {
  PgnAnalysisCancelledError,
  PgnAnalysisController,
} from "./pgn-analysis-controller.js";
import {
  MAX_EVALUATOR_SIDECAR_BYTES,
  MAX_MODEL_ARTIFACT_BYTES,
  type LoadedPgnAnalysisModel,
} from "./pgn-analysis-worker-protocol.js";
import {
  buildPgnAnalysisReport,
  serializePgnAnalysisReport,
  type PgnAnalysisReport,
  type PgnReportTruth,
} from "./pgn-report.js";
import { classifyBrowserPredictorTrust } from "./model-trust.js";
const SAMPLE_PGN = `[Event "Completed offline example"]
[Site "Local"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0`;

async function sha256Bytes(value: Uint8Array): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    Uint8Array.from(value).buffer,
  );
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function sha256Text(value: string): Promise<string> {
  return sha256Bytes(new TextEncoder().encode(value));
}

interface LocalEvaluatorSidecar {
  readonly bytes: Uint8Array;
  readonly fileName: string;
  readonly computedSha256: string;
}

function GuessList({
  title,
  guesses,
  nameFor,
}: {
  readonly title: string;
  readonly guesses: readonly PgnGuess[];
  readonly nameFor: (id: string) => string;
}): React.JSX.Element {
  const displayParameter = (value: unknown): string =>
    typeof value === "string" || typeof value === "number"
      ? String(value)
      : JSON.stringify(value);
  return (
    <div className="pgn-guesses">
      <h4>{title}</h4>
      {guesses.slice(0, 5).map((guess, index) => (
        <div className={guess.eliminated ? "eliminated" : ""} key={guess.id}>
          <span>{String(index + 1).padStart(2, "0")}</span>
          <strong>
            {nameFor(guess.id)}
            {guess.parameters.map((parameter) => (
              <small key={parameter.name}>
                {parameter.name} {displayParameter(parameter.value)} ·{" "}
                {String(Math.round(parameter.confidence * 100))}%
              </small>
            ))}
          </strong>
          <em>
            {guess.eliminated
              ? "OUT"
              : `${String(Math.round(guess.confidence * 100))}%`}
          </em>
        </div>
      ))}
    </div>
  );
}

export function PgnAnalysisPanel({
  nameFor,
}: {
  readonly nameFor: (id: string) => string;
}): React.JSX.Element {
  const [pgn, setPgn] = useState("");
  const [result, setResult] = useState<PgnAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [truthWhite, setTruthWhite] = useState("");
  const [truthBlack, setTruthBlack] = useState("");
  const [report, setReport] = useState<PgnAnalysisReport | null>(null);
  const [reportStatus, setReportStatus] = useState<string | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<{
    readonly processedPlies: number;
    readonly totalPlies: number;
  } | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [loadingFile, setLoadingFile] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [neuralModel, setNeuralModel] = useState<LoadedPgnAnalysisModel | null>(
    null,
  );
  const [modelArtifactSha256, setModelArtifactSha256] = useState<string | null>(
    null,
  );
  const [modelStatus, setModelStatus] = useState<string | null>(null);
  const [evaluatorSidecar, setEvaluatorSidecar] =
    useState<LocalEvaluatorSidecar | null>(null);
  const [evaluatorSidecarSha256, setEvaluatorSidecarSha256] = useState("");
  const [evaluatorStatus, setEvaluatorStatus] = useState<string | null>(null);
  const [loadingEvaluator, setLoadingEvaluator] = useState(false);
  const controller = useRef<PgnAnalysisController | null>(null);
  const analysisGeneration = useRef(0);
  const modelLoadGeneration = useRef(0);
  const evaluatorLoadGeneration = useRef(0);

  useEffect(() => {
    const activeController = new PgnAnalysisController();
    controller.current = activeController;
    return () => {
      analysisGeneration.current += 1;
      if (controller.current === activeController) {
        controller.current = null;
      }
      activeController.dispose();
    };
  }, []);

  function selectedTruth(): PgnReportTruth | undefined {
    return truthWhite === "" || truthBlack === ""
      ? undefined
      : {
          white: truthWhite,
          black: truthBlack,
          source: "user-entered",
        };
  }

  async function currentReport(): Promise<PgnAnalysisReport> {
    if (result === null) {
      throw new Error("Analyze a PGN before creating a report.");
    }
    const next = await buildPgnAnalysisReport(pgn, result, selectedTruth());
    setReport(next);
    return next;
  }

  async function downloadReport(): Promise<void> {
    try {
      const next = await currentReport();
      const blob = new Blob([serializePgnAnalysisReport(next)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `drawback-analysis-${next.analyticalDigest.slice(0, 12)}.json`;
      anchor.click();
      URL.revokeObjectURL(url);
      setReportStatus("Report downloaded");
    } catch {
      setReportStatus("Report export failed");
    }
  }

  async function copySummary(): Promise<void> {
    try {
      const next = await currentReport();
      const white = next.analytical.final.white[0];
      const black = next.analytical.final.black[0];
      const summary = [
        `DrawbackGuesser · ${next.analytical.predictor.displayName}`,
        `White: ${white === undefined ? "no guess" : `${nameFor(white.id)} (${String(Math.round(white.confidence * 100))}%)`}`,
        `Black: ${black === undefined ? "no guess" : `${nameFor(black.id)} (${String(Math.round(black.confidence * 100))}%)`}`,
        `Ply: ${String(next.analytical.replay.plyCount)}`,
        `Digest: ${next.analyticalDigest}`,
      ].join("\n");
      await navigator.clipboard.writeText(summary);
      setReportStatus("Summary copied");
    } catch {
      setReportStatus("Clipboard unavailable");
    }
  }

  async function analyze(sourcePgn = pgn): Promise<void> {
    const generation = analysisGeneration.current + 1;
    analysisGeneration.current = generation;
    setError(null);
    setResult(null);
    setReport(null);
    setReportStatus(null);
    setTruthWhite("");
    setTruthBlack("");
    setAnalysisProgress(null);
    setAnalyzing(true);
    try {
      const activeController = controller.current;
      if (activeController === null) {
        throw new Error("PGN analysis worker is not ready.");
      }
      const next = await activeController.analyze(sourcePgn, {
        ...(modelArtifactSha256 === null
          ? {}
          : { neuralArtifactSha256: modelArtifactSha256 }),
        ...(evaluatorSidecar === null
          ? {}
          : {
              evaluatorSidecarBytes: evaluatorSidecar.bytes,
              evaluatorSidecarSha256: evaluatorSidecarSha256.trim().toLowerCase(),
            }),
        onProgress(progress) {
          if (analysisGeneration.current === generation) {
            setAnalysisProgress(progress);
          }
        },
      });
      if (analysisGeneration.current === generation) {
        setResult(next);
      }
    } catch (caught) {
      if (caught instanceof PgnAnalysisCancelledError) {
        return;
      }
      if (caught instanceof PgnParseError) {
        setError(
          caught.ply > 0
            ? `Ply ${String(caught.ply)}: ${caught.message}`
            : caught.message,
        );
      } else {
        setError(
          caught instanceof Error
            ? caught.message
            : "The PGN could not be analyzed.",
        );
      }
    } finally {
      if (analysisGeneration.current === generation) {
        setAnalyzing(false);
        setAnalysisProgress(null);
      }
    }
  }

  async function loadModelFile(file: File): Promise<void> {
    const generation = modelLoadGeneration.current + 1;
    modelLoadGeneration.current = generation;
    setModelStatus(null);
    if (file.size > MAX_MODEL_ARTIFACT_BYTES) {
      controller.current?.clearModel();
      setLoadingModel(false);
      setNeuralModel(null);
      setModelArtifactSha256(null);
      setResult(null);
      setReport(null);
      setModelStatus("Model artifact exceeds the 32 MiB local limit.");
      return;
    }
    setLoadingModel(true);
    try {
      const text = await file.text();
      const artifactSha256 = await sha256Text(text);
      const activeController = controller.current;
      if (activeController === null) {
        throw new Error("PGN analysis worker is not ready.");
      }
      const parsed = await activeController.loadModel(text, artifactSha256);
      if (modelLoadGeneration.current !== generation) {
        return;
      }
      setNeuralModel(parsed);
      setModelArtifactSha256(artifactSha256);
      const trust = classifyBrowserPredictorTrust(parsed.modelVariant);
      setModelStatus(
        `Loaded unverified local research ${parsed.modelVariant === "v21-hybrid-ensemble" ? "v21 ensemble" : parsed.modelVariant === "v22-hybrid" ? "v22" : parsed.modelVariant === "v21-hybrid" ? "v21" : "v1"} model · ${String(parsed.drawbackCount)} rules · artifact ${artifactSha256.slice(0, 12)}${trust.calibrationMetadata === "artifact-declared-simulation-validation" ? " · contains self-declared simulation-validation calibration metadata" : ""}`,
      );
      setResult(null);
      setReport(null);
    } catch (caught) {
      if (modelLoadGeneration.current !== generation) {
        return;
      }
      setNeuralModel(null);
      setModelArtifactSha256(null);
      controller.current?.clearModel();
      setResult(null);
      setReport(null);
      setModelStatus(
        caught instanceof Error
          ? `Model rejected: ${caught.message}`
          : "Model artifact could not be parsed.",
      );
    } finally {
      if (modelLoadGeneration.current === generation) {
        setLoadingModel(false);
      }
    }
  }

  async function loadEvaluatorSidecarFile(file: File): Promise<void> {
    const generation = evaluatorLoadGeneration.current + 1;
    evaluatorLoadGeneration.current = generation;
    setEvaluatorStatus(null);
    setResult(null);
    setReport(null);
    if (file.size > MAX_EVALUATOR_SIDECAR_BYTES) {
      setEvaluatorSidecar(null);
      setLoadingEvaluator(false);
      setEvaluatorStatus("Evaluator sidecar exceeds the 8 MiB local limit.");
      return;
    }
    setLoadingEvaluator(true);
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const computedSha256 = await sha256Bytes(bytes);
      if (evaluatorLoadGeneration.current !== generation) {
        return;
      }
      setEvaluatorSidecar({
        bytes,
        fileName: file.name,
        computedSha256,
      });
      setEvaluatorStatus(
        `Loaded local evaluator sidecar ${file.name} · calculated digest ${computedSha256.slice(0, 12)} · enter its independently supplied SHA-256 before analysis.`,
      );
    } catch {
      if (evaluatorLoadGeneration.current === generation) {
        setEvaluatorSidecar(null);
        setEvaluatorStatus("The evaluator sidecar could not be read.");
      }
    } finally {
      if (evaluatorLoadGeneration.current === generation) {
        setLoadingEvaluator(false);
      }
    }
  }

  function cancelWorkerAnalysis(): boolean {
    const cancelled = controller.current?.cancel() ?? false;
    if (cancelled && neuralModel !== null) {
      modelLoadGeneration.current += 1;
      setLoadingModel(false);
      setNeuralModel(null);
      setModelArtifactSha256(null);
      setModelStatus(
        "Analysis cancelled; reload the local model before neural analysis.",
      );
    }
    return cancelled;
  }

  function cancelAnalysis(): void {
    analysisGeneration.current += 1;
    cancelWorkerAnalysis();
    setAnalyzing(false);
    setAnalysisProgress(null);
    setError("Analysis cancelled.");
  }

  async function loadPgnFile(file: File): Promise<void> {
    const generation = analysisGeneration.current + 1;
    analysisGeneration.current = generation;
    cancelWorkerAnalysis();
    setAnalyzing(false);
    setAnalysisProgress(null);
    setResult(null);
    setReport(null);
    setReportStatus(null);
    setError(null);
    if (file.size > MAX_PGN_INPUT_BYTES) {
      setError(
        `PGN exceeds the ${String(MAX_PGN_INPUT_BYTES)} byte analysis limit.`,
      );
      return;
    }
    setLoadingFile(true);
    try {
      const text = await file.text();
      if (analysisGeneration.current === generation) {
        setPgn(text);
      }
    } catch {
      if (analysisGeneration.current === generation) {
        setError("The selected PGN file could not be read.");
      }
    } finally {
      if (analysisGeneration.current === generation) {
        setLoadingFile(false);
      }
    }
  }

  const truthChoices = result === null
    ? []
    : [
        ...result.finalWhite.map((guess) => ({
          id: guess.id,
          name: nameFor(guess.id),
          available: true,
        })),
        ...result.unavailableDrawbacks.map((drawback) => ({
          id: drawback.id,
          name: drawback.name,
          available: result.finalWhite.some((guess) => guess.id === drawback.id),
        })),
      ]
        .filter(
          (choice, index, choices) =>
            choices.findIndex(({ id }) => id === choice.id) === index,
        )
        .sort((left, right) => left.name.localeCompare(right.name));
  const normalizedEvaluatorSha256 = evaluatorSidecarSha256.trim().toLowerCase();
  const evaluatorDigestValid =
    evaluatorSidecar === null
      ? normalizedEvaluatorSha256.length === 0
      : /^[0-9a-f]{64}$/u.test(normalizedEvaluatorSha256) &&
        normalizedEvaluatorSha256 === evaluatorSidecar.computedSha256;
  const evaluatorConfigurationIncomplete =
    (evaluatorSidecar !== null || normalizedEvaluatorSha256.length > 0) &&
    (evaluatorSidecar === null || !evaluatorDigestValid);
  const busy =
    analyzing || loadingFile || loadingModel || loadingEvaluator;

  return (
    <section className="pgn-lab" aria-labelledby="pgn-lab-title">
      <div className="pgn-lab-heading">
        <div>
          <FileSearch size={19} />
          <span>
            <strong id="pgn-lab-title">
              Post-game PGN drawback analysis
            </strong>
            <small>
              {neuralModel === null
                ? evaluatorSidecar === null
                  ? "Symbolic v2 · 180 ranked from standard PGN · 2 explicitly unavailable"
                  : "Symbolic v2 · evaluator facts loaded · trusted digest required for all 182 rules"
                : `Unverified local research ${neuralModel.modelVariant === "v21-hybrid-ensemble" ? "hybrid v21 ensemble" : neuralModel.modelVariant === "v22-hybrid" ? "hybrid v22" : neuralModel.modelVariant === "v21-hybrid" ? "hybrid v21" : "hybrid v1"} · ${String(neuralModel.drawbackCount)} neural rules · no external connection`}
            </small>
          </span>
        </div>
        <span className="offline-badge">
          <ShieldCheck size={13} /> OFFLINE ONLY
        </span>
      </div>

      <div className="pgn-input">
        <label htmlFor="pgn-text">Paste completed PGN</label>
        <textarea
              id="pgn-text"
              value={pgn}
              disabled={busy}
              onChange={(event) => {
                setPgn(event.target.value);
                setError(null);
                setResult(null);
                setReport(null);
                setReportStatus(null);
                setTruthWhite("");
                setTruthBlack("");
              }}
              placeholder={SAMPLE_PGN}
              spellCheck={false}
        />
        <div>
          <label className="pgn-file-button">
            Open .pgn
            <input
              type="file"
              disabled={busy}
              accept=".pgn,application/x-chess-pgn,text/plain"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file !== undefined) {
                  void loadPgnFile(file);
                }
                event.target.value = "";
              }}
            />
          </label>
          <label className="pgn-file-button">
            Open model
            <input
              type="file"
              disabled={busy}
              accept=".json,application/json"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file !== undefined) {
                  void loadModelFile(file);
                }
                event.target.value = "";
              }}
            />
          </label>
          <label className="pgn-file-button">
            Open evaluator facts
            <input
              type="file"
              disabled={busy}
              accept=".json,application/json"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file !== undefined) {
                  void loadEvaluatorSidecarFile(file);
                }
                event.target.value = "";
              }}
            />
          </label>
          {neuralModel === null ? null : (
            <button
              className="secondary"
              type="button"
              disabled={analyzing}
              onClick={() => {
                modelLoadGeneration.current += 1;
                setLoadingModel(false);
                setNeuralModel(null);
                setModelArtifactSha256(null);
                controller.current?.clearModel();
                setModelStatus("Local model removed; symbolic-only mode active.");
                setResult(null);
                setReport(null);
              }}
            >
              Remove model
            </button>
          )}
          {evaluatorSidecar === null ? null : (
            <button
              className="secondary"
              type="button"
              disabled={analyzing}
              onClick={() => {
                evaluatorLoadGeneration.current += 1;
                setLoadingEvaluator(false);
                setEvaluatorSidecar(null);
                setEvaluatorSidecarSha256("");
                setEvaluatorStatus("Evaluator sidecar removed; standard-PGN mode active.");
                setResult(null);
                setReport(null);
              }}
            >
              Remove evaluator facts
            </button>
          )}
          <button
            className="secondary"
            type="button"
            disabled={busy}
            onClick={() => {
              setPgn(SAMPLE_PGN);
              setError(null);
              setResult(null);
              setAnalysisProgress(null);
            }}
          >
            Load example
          </button>
          {analyzing ? (
            <button className="secondary" type="button" onClick={cancelAnalysis}>
              Cancel analysis
            </button>
          ) : (
            <button
              type="button"
              disabled={
                pgn.trim().length === 0 ||
                busy ||
                evaluatorConfigurationIncomplete
              }
              onClick={() => void analyze()}
            >
              <Sparkles size={14} /> Analyze locally
            </button>
          )}
        </div>
        <div className="pgn-evaluator-trust">
          <label htmlFor="evaluator-sidecar-sha256">
            Trusted evaluator sidecar SHA-256
          </label>
          <input
            id="evaluator-sidecar-sha256"
            type="text"
            value={evaluatorSidecarSha256}
            disabled={busy}
            inputMode="text"
            autoComplete="off"
            spellCheck={false}
            placeholder="Paste the digest supplied separately from the sidecar file"
            aria-invalid={evaluatorConfigurationIncomplete}
            onChange={(event) => {
              setEvaluatorSidecarSha256(event.target.value);
              setResult(null);
              setReport(null);
            }}
          />
          <small>
            The file is accepted only when this independently obtained digest
            exactly matches its bytes. Selecting a file does not trust it.
          </small>
          {evaluatorSidecar === null ? null : (
            <code>
              {evaluatorSidecar.fileName} · calculated{" "}
              {evaluatorSidecar.computedSha256}
            </code>
          )}
        </div>
      </div>

      {modelStatus === null ? null : (
        <div className="pgn-report-status" role="status">
          <ShieldCheck size={13} /> {modelStatus}
        </div>
      )}
      {evaluatorStatus === null ? null : (
        <div className="pgn-report-status" role="status">
          <ShieldCheck size={13} /> {evaluatorStatus}
        </div>
      )}

      {analysisProgress === null ? null : (
        <div
          className="pgn-analysis-progress"
          role="progressbar"
          aria-label="PGN analysis progress"
          aria-valuemin={0}
          aria-valuemax={analysisProgress.totalPlies}
          aria-valuenow={analysisProgress.processedPlies}
        >
          <span
            style={{
              width: `${String(
                analysisProgress.totalPlies === 0
                  ? 0
                  : (analysisProgress.processedPlies /
                      analysisProgress.totalPlies) *
                      100,
              )}%`,
            }}
          />
          <strong>
            Replaying {String(analysisProgress.processedPlies)} of{" "}
            {String(analysisProgress.totalPlies)} ply
          </strong>
        </div>
      )}

      {error === null ? null : (
        <div className="pgn-error" role="alert">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      {result === null ? (
        <div className="pgn-empty">
          Paste a completed PGN with a terminal Result header to review
          independent White and Black guesses after every ply. This analyzer
          does not accept ongoing games.
        </div>
      ) : (
        <div className="pgn-results">
          <div className="pgn-summary">
            <div>
              <span>
                {result.predictor.mode === "symbolic-only"
                  ? `SYMBOLIC V2 · ${String(result.representedDrawbackCount)}/${String(result.catalogDrawbackCount)} RANKED · NOT CALIBRATED`
                  : `UNVERIFIED LOCAL RESEARCH ${result.predictor.mode === "hybrid-v21-ensemble" ? "HYBRID V21 ENSEMBLE · ARTIFACT-DECLARED CALIBRATION METADATA" : result.predictor.mode === "hybrid-v22" ? "HYBRID V22" : result.predictor.mode === "hybrid-v21" ? "HYBRID V21" : "HYBRID V1"} · ${String(result.predictor.neuralCoveredDrawbackCount)} NEURAL RULES · NOT RELEASE-APPROVED`}
              </span>
              <strong>{result.plyCount} legal ply analyzed</strong>
              <small>
                {String(result.representedDrawbackCount)}/
                {String(result.catalogDrawbackCount)} drawbacks represented
              </small>
              <small>{result.finalFen}</small>
              <p>
                Confidence is {result.predictor.mode === "symbolic-only"
                  ? "symbolic posterior"
                  : result.predictor.mode === "hybrid-v21-ensemble"
                    ? "local research ensemble posterior using self-declared simulation-validation calibration metadata with exact symbolic elimination"
                    : result.predictor.mode === "hybrid-v22"
                    ? `symbolic posterior plus the selected model residual using ${result.predictor.sequenceObservationMode}`
                    : result.predictor.mode === "hybrid-v21"
                    ? "symbolic posterior plus the selected model residual"
                    : "symbolic posterior reweighted by the selected local model"}{" "}
                mass within the {String(result.representedDrawbackCount)} represented rules.
                {result.predictor.mode === "hybrid-v21-ensemble"
                  ? " The metadata is preserved for research but is not independently verified, release-approved, or established on real-world games."
                  : " It is not a calibrated probability of correctness."}
              </p>
              {result.evaluatorEvidence.mode === "authenticated-sidecar" ? (
                <p className="pgn-evaluator-evidence">
                  All 182 rules ranked with authenticated offline evaluator
                  evidence · {result.evaluatorEvidence.engine.uciName}{" "}
                  {result.evaluatorEvidence.engine.version} ·{" "}
                  {result.evaluatorEvidence.searchLimit.kind}{" "}
                  {String(result.evaluatorEvidence.searchLimit.value)} ·
                  artifact{" "}
                  <code>
                    {result.evaluatorEvidence.artifactSha256.slice(0, 16)}
                  </code>
                </p>
              ) : null}
              {result.predictor.mode === "symbolic-only" ? null : (
                <p className="pgn-availability-warning">
                  Local model checkpoint{" "}
                  <code>
                    {result.predictor.mode === "hybrid-v21-ensemble"
                      ? result.predictor.sourceEnsembleReleaseSha256.slice(0, 12)
                      : result.predictor.sourceCheckpointSha256.slice(0, 12)}
                  </code>
                  {result.predictor.mode === "hybrid-v21-ensemble"
                    ? " · the local file self-declares three checkpoint digests and per-head simulation-validation calibration; these claims are not independently authenticated or release-approved."
                    : ". DrawbackGuesser has not promoted a bundled neural model; only use artifacts whose evaluation you trust."}
                </p>
              )}
              {result.unavailableDrawbacks.length === 0 ? null : (
                <p className="pgn-availability-warning">
                  Not ranked from standard PGN because evaluator facts are
                  unavailable:{" "}
                  {result.unavailableDrawbacks
                    .map((drawback) => drawback.name)
                    .join(", ")}.
                </p>
              )}
            </div>
            <GuessList
              title="Final White guess"
              guesses={result.finalWhite}
              nameFor={nameFor}
            />
            <GuessList
              title="Final Black guess"
              guesses={result.finalBlack}
              nameFor={nameFor}
            />
          </div>
          <div className="pgn-report-tools">
            <div>
              <strong>Post-game reveal</strong>
              <small>
                Optional truth is scored after analysis and never enters the predictor.
              </small>
            </div>
            <label>
              White truth
              <select
                value={truthWhite}
                onChange={(event) => {
                  setTruthWhite(event.target.value);
                  setReport(null);
                  setReportStatus(null);
                }}
              >
                <option value="">Not entered</option>
                {truthChoices.map((choice) => (
                  <option key={choice.id} value={choice.id}>
                    {choice.name}{choice.available ? "" : " · unavailable to predictor"}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Black truth
              <select
                value={truthBlack}
                onChange={(event) => {
                  setTruthBlack(event.target.value);
                  setReport(null);
                  setReportStatus(null);
                }}
              >
                <option value="">Not entered</option>
                {truthChoices.map((choice) => (
                  <option key={choice.id} value={choice.id}>
                    {choice.name}{choice.available ? "" : " · unavailable to predictor"}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" onClick={() => void downloadReport()}>
              <Download size={14} /> Download JSON
            </button>
            <button type="button" onClick={() => void copySummary()}>
              <Copy size={14} /> Copy summary
            </button>
          </div>
          {report?.scoring === undefined ? null : (
            <div className="pgn-truth-score">
              <span>
                White truth rank{" "}
                <strong>{report.scoring.white.finalRank ?? "not represented"}</strong>
              </span>
              <span>
                Black truth rank{" "}
                <strong>{report.scoring.black.finalRank ?? "not represented"}</strong>
              </span>
              <span>
                Digest <code>{report.analyticalDigest.slice(0, 16)}</code>
              </span>
            </div>
          )}
          {reportStatus === null ? null : (
            <div className="pgn-report-status" role="status">
              <Check size={13} /> {reportStatus}
            </div>
          )}
          <div className="pgn-coverage">
            {result.coverage.map((item) => (
              <span key={item.drawbackId}>
                <strong>{nameFor(item.drawbackId)}</strong>
                {item.mode === "exact"
                  ? `exact · ${String(item.variantCount)} variants`
                  : item.mode === "analytic"
                    ? `analytic · ${String(item.variantCount)} outcomes`
                    : `sampled · ${String(item.variantCount)} particles`}
              </span>
            ))}
          </div>
          <div className="pgn-history">
            <div className="section-heading">
              <div>
                <FileSearch size={15} />
                <h3>Per-ply prediction history</h3>
              </div>
              <span>{result.plyCount} snapshots</span>
            </div>
            <div className="pgn-history-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Ply</th>
                    <th>Move</th>
                    <th>White top guess</th>
                    <th>Black top guess</th>
                    <th>New symbolic eliminations</th>
                  </tr>
                </thead>
                <tbody>
                  {result.history.map((point) => {
                    const white = point.white[0];
                    const black = point.black[0];
                    return (
                      <tr key={`${String(point.ply)}-${point.san}`}>
                        <td>{String(point.ply)}</td>
                        <td>
                          {point.color === "white"
                            ? `${String(point.moveNumber)}.`
                            : `${String(point.moveNumber)}…`}{" "}
                          <strong>{point.san}</strong>
                        </td>
                        <td>
                          {white === undefined ? "—" : nameFor(white.id)}
                          <span>
                            {white === undefined
                              ? ""
                              : `${String(Math.round(white.confidence * 100))}%`}
                          </span>
                        </td>
                        <td>
                          {black === undefined ? "—" : nameFor(black.id)}
                          <span>
                            {black === undefined
                              ? ""
                              : `${String(Math.round(black.confidence * 100))}%`}
                          </span>
                        </td>
                        <td>
                          {point.eliminations.length === 0
                            ? "—"
                            : point.eliminations.map((evidence) => (
                                <span key={`${evidence.color}-${evidence.drawbackId}`}>
                                  <strong>
                                    {evidence.color === "white" ? "White" : "Black"}:{" "}
                                    {nameFor(evidence.drawbackId)}
                                  </strong>
                                  {" — "}
                                  {evidence.reason}
                                </span>
                              ))}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
