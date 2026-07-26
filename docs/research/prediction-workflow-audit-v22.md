# Prediction workflow audit v2.2

Status: implementation audit and prioritized specification. This document
describes the repository at revision
`972da1ba5375c99473f83d0820ccc3148556fd5e` and does not claim that proposed
behavior already exists.

Scope is manual/offline PGN prediction, the web secrecy boundary, model
loading, report export, and readiness for post-game drawback guessing. No
external chess site was accessed or integrated.

## Decision

The current browser flow is a useful symbolic research demo, not a
production-ready post-game guessing workflow.

What is already strong:

- PGN replay and prediction execute locally in the browser.
- Standard legality is reconstructed independently with an unrestricted
  `GameSession`.
- White and Black symbolic hypotheses are updated independently.
- Hard-impossible hypotheses are eliminated by executable candidate rules.
- Hidden-square and hidden-rank variants are exhaustive, and Gambler's
  deterministic approximation is disclosed.
- The Python checkpoint loader has meaningful format, metadata, tensor-shape,
  and strict-state validation.

What blocks a production claim:

- The web application does not load or invoke a neural checkpoint. Its PGN
  output is symbolic/Bayesian only.
- No neural checkpoint is published as a supported model. The documented
  v2.1 hybrid failed its preregistered validation gates.
- Analysis is synchronous on the UI thread, unbounded, paste-only, and has no
  progress or cancellation.
- Results cannot be exported with provenance, reproduced later, or compared
  against post-game revealed truth.
- “Offline” is presented as interface text rather than enforced and tested as
  a deployable network/content-security policy.

The right v2.2 target is an explicit **offline post-game analysis product**:
import a completed or partial PGN, run a versioned predictor locally, inspect
per-ply evidence, optionally enter the two true drawbacks after the game, and
export a reproducible report. It must not add live capture from competitive
sites.

## Evidence map

| Area | Existing implementation | Audit finding |
| --- | --- | --- |
| PGN replay | `apps/web/src/pgn-analysis.ts` | Local, deterministic mainline replay with standard legality and per-ply symbolic updates. |
| PGN interface | `apps/web/src/PgnAnalysisPanel.tsx` | Paste textarea, example, synchronous Analyze button, top-five final guesses, coverage badges, and a per-ply top-one table. |
| Symbolic predictor | `packages/predictor/src/predictor.ts` | Exact hard elimination precedes likelihood scoring; only the moving color's distribution transitions. |
| Hypothesis catalog | `packages/predictor/src/catalog.ts` | 27 rule IDs; exhaustive 64-square and eight-rank variants; 192 deterministic Gambler particles. |
| Browser model path | `apps/web/src` and `apps/web/package.json` | No checkpoint loader, inference adapter, ONNX/WebAssembly runtime, model fetch, or model-selection state. |
| Python model path | `ml/training/drawback_ml/inference.py` | Reconstructs format-3 PyTorch checkpoints and rejects incompatible metadata and shapes. This path is not connected to the browser. |
| Model release state | `ml/models/README.md`; `docs/research/baseline-v21-hybrid-experiment.md` | No supported checkpoint is published. The v2.1 hybrid improved Top-1 but failed Top-3, NLL, and Brier validation gates. |
| Live secrecy | `apps/web/src/App.tsx`; predictor `MoveObservation` type | Predictor observations contain public positions, observed move, and ordinary legal moves. The local game controller still necessarily holds both secrets. |
| Report handoff | `PgnAnalysisResult` and `PgnAnalysisPanel` | Rich in-memory history, but no download/copy/schema/provenance/source hash or persisted artifact. |
| CI | `.github/workflows/ci.yml` | Typecheck, lint, tests, builds, and Python suites run; no browser end-to-end offline/network or large-PGN responsiveness gate. |

## Current manual/offline behavior

### Import and parsing

Existing behavior:

- The user pastes text into a textarea or loads a six-ply sample.
- Header lines are parsed and unescaped.
- `SetUp "1"` plus `FEN` initializes a nonstandard position.
- Brace comments, semicolon comments, nested side variations, NAGs, move
  numbers, result tokens, and trailing `!?` annotations are ignored.
- Each SAN token is matched against ordinary legal moves in the reconstructed
  position. An invalid token reports its token and one-based ply.
- Completed and partial mainlines are accepted.

Limitations:

- There is no `.pgn` file picker, drag-and-drop, clipboard-read button, batch
  import, or CLI equivalent.
- The parser is intentionally small rather than a complete PGN interchange
  implementation. It ignores all variations and does not preserve comments,
  headers, declared result, or source text in `PgnAnalysisResult`.
- Multiple games in one input are not represented as separate analyses.
- There is no explicit input byte, token, or ply limit.
- The declared result is discarded rather than checked against the replayed
  terminal state. That is acceptable for partial analysis but should be
  disclosed in the report.
- Synchronous parsing, replay, hypothesis filtering, and snapshot construction
  happen in the click handler. Long or adversarial input can freeze the page.

### Prediction

Existing behavior:

- Replay uses `unrestrictedRule` for both colors, so no unknown drawback is
  accidentally enforced as truth.
- Before every move, ordinary legal moves are generated from the public board.
- `SymbolicPredictor.observe` receives public before/after positions, the
  observed move, and ordinary moves.
- Only the moving color's hypothesis distribution changes.
- A candidate is permanently eliminated when the observed move is absent from
  that candidate's filtered move set. Survivors update candidate-private
  symbolic state.
- Rule posterior ranking aggregates parameter variants. The final result and
  every ply snapshot include independent White and Black rankings and leading
  parameter guesses.

Limitations:

- The default likelihood is largely uniform-choice pressure based on candidate
  allowed-move count. PGN analysis supplies none of the optional human-move,
  engine-quality, strength, clock, or time-use likelihood signals.
- Confidence is a model posterior under the implemented hypothesis catalog and
  priors, not an empirically calibrated probability of correctness. The UI
  does not state that distinction.
- Equal seed priors and the deterministic Gambler particles are fixed in code,
  not recorded in an exported run manifest.
- The UI shows only five final rule guesses and only the top rule per color per
  ply. It does not show why a rule moved, which move eliminated it, surviving
  variant counts over time, or normalized semantic confusion groups.
- The PGN analyzer is not wired to the v2.1 hybrid or any neural model.

### Result presentation

Existing behavior:

- Final White and Black top-five lists show rounded rule confidence and the
  leading value for parameterized rules.
- Coverage badges disclose exact versus sampled parameter coverage.
- The per-ply table shows move, White top guess, Black top guess, and rounded
  confidence.
- Final FEN and analyzed ply count are displayed.

Limitations:

- There is no PGN replay board in the offline panel, selectable timeline row,
  confidence chart, evidence drawer, or side-by-side hypothesis comparison.
- No true drawback can be entered after the game, so the workflow cannot show
  rank of truth, first rank-one ply, calibration error, or exact elimination
  mistakes for a known training game.
- Nothing is exportable. Reloading loses PGN text and all results.
- There is no report schema version, application revision, catalog version,
  predictor version/configuration, input hash, analysis timestamp, locale,
  warnings, or integrity checksum.

## Secrecy and offline boundary

### What is actually protected

`MoveObservation` deliberately excludes true drawback, hidden parameters, and
authoritative drawback state. The offline PGN path creates an unrestricted
session and never possesses true drawbacks at all. The symbolic predictor
maintains candidate-private state derived from public history. This is a real
separation, not merely a hidden UI element.

The offline panel contains no fetch, WebSocket, external-site adapter, browser
extension interface, or remote chess account integration. Its current analysis
function is pure local TypeScript execution.

### What is not a security boundary

The live training application holds both assigned rules inside one browser
controller. It shows the active side's true rule card and, after termination,
both rules. In a pass-and-play setting, the same person or anyone with
developer tools can inspect client memory and observe both sides across turns.
This is appropriate for a local trainer but is not secret isolation between
mutually distrustful players.

The live UI also displays authoritative `ruleTriggered` notices and move
markers. Those come from the game engine, not the public predictor. They are
training feedback and must not be described as information available in an
unknown real-world PGN.

The “OFFLINE ONLY” badge is descriptive. Vite has no committed content security
policy, service-worker offline shell, network-deny harness, or production
deployment configuration proving that the shipped application makes no
requests. Current source inspection supports a local-only claim for this flow;
production enforcement and regression evidence do not yet exist.

### Required product boundary

V2.2 should use two clearly separated modes:

1. **Local trainer:** engine knows both assigned rules, active player sees their
   own rule, and training feedback may include authoritative trigger markers.
2. **Offline PGN guesser:** engine knows no true rule, accepts only user-provided
   PGN plus optional post-game truth, labels all predictions as inferred, and
   never displays authoritative trigger claims.

Post-game truth entry must be stored separately from predictor inputs. Mutation
tests must prove that adding or changing truth leaves every prediction byte
identical and affects only scoring/reveal sections.

## Model loading audit

### Existing Python loader

`load_checkpoint_predictor` is a credible offline evaluation loader:

- uses `torch.load(..., weights_only=True)`;
- requires checkpoint format 3;
- validates model/training metadata and feature-schema version;
- checks vocabulary, tensor dimensions, tokenizer metadata, and v2.1 symbolic
  metadata;
- loads state strictly, selects a declared device, enables deterministic
  algorithms, and switches to evaluation mode; and
- exposes an inference API that accepts the public `FeatureRecord`.

This is production-oriented validation for a Python process, but it is not a
distribution story. There is no signature/trust policy, maximum artifact size,
dependency/environment attestation, model registry, calibration artifact, or
browser-compatible serialization.

### Browser reality

The browser imports `@drawbackguesser/predictor`, not the Python predictor.
There is no server endpoint and no local subprocess bridge. Therefore wording
such as “model confidence” would be misleading for the PGN page today: the
confidence shown is symbolic/Bayesian posterior mass.

No checkpoint should be silently bundled merely to fill this gap. The current
model documentation says generated `.pt` files are ignored and no v1 model is
published; the v2.1 report records a failed validation decision. Until a model
passes a preregistered validation release gate, production should ship
`symbolic-v1` as an explicit predictor kind and avoid implying neural quality.

### Recommended loading contract

Define a versioned `PredictorProvider` boundary usable by both symbolic-only and
future hybrid implementations:

```text
describe() -> id, version, catalog hash, parameter coverage, calibration status
analyzePrefix(publicPrefix) -> independent White/Black distributions + evidence
dispose() -> release worker/runtime resources
```

A future browser artifact should include:

- immutable model and preprocessing hashes;
- exact drawback vocabulary and catalog hash;
- feature/token schema versions;
- calibration transform and validation report reference;
- supported runtime/version and memory estimate;
- parameter-coverage declaration;
- license and provenance; and
- a signature or pinned digest verified before activation.

Load it inside a dedicated Web Worker. Fail closed to the named symbolic
provider when integrity, schema, memory, or runtime checks fail. Surface the
active provider and fallback reason in both UI and exports. Neural scores may
rerank only exact symbolic survivors and must never restore eliminated rules.

## Exportability audit

`PgnAnalysisResult` is close to a useful internal payload but not a durable
report. It contains final rankings, all per-ply rankings, FENs, and coverage;
it omits the original PGN, headers, predictor identity/config, evidence,
warnings, and provenance.

V2.2 should export a canonical UTF-8 JSON report first. CSV and annotated PGN
can be derived later. Required top-level fields:

```text
schemaVersion
applicationRevision
createdAt
source: inputSha256, preservedHeaders, normalizedMainline, declaredResult
replay: initialFen, finalFen, plyCount, replayStatus, warnings
predictor: id, version, catalogSha256, configuration, calibration, coverage
timeline[]: ply, color, san, fenBefore, topK, survivorCount, eliminations
final: whiteDistribution, blackDistribution, parameterPosteriors
truth?: white, black, parameters, source=user-entered
scoring?: finalRanks, firstRankOnePly, hardContradictions
```

The report must not embed a checkpoint, secret local path, browser storage
identifier, or unrelated live-session state. Export should use a stable key
order or canonical serializer so identical input/configuration produces the
same analytical payload; place wall-clock time in an envelope excluded from
the analytical digest.

UI actions should include Download JSON and Copy summary, with a visible
provider/catalog hash. Importing a prior report must validate schema and hashes
and render it read-only unless the user explicitly reruns analysis.

## Prioritized implementation specification

### P0 — honest, reproducible offline analysis

1. **Name the predictor.** Display `Symbolic v1` and its catalog/coverage
   version everywhere confidence is shown. Replace generic model wording.
2. **Move analysis to a Web Worker.** Add bounded input bytes and plies,
   progress by parsed/replayed ply, cancellation, deterministic errors, and a
   worker termination timeout.
3. **Create a report schema and JSON export.** Include source hash, preserved
   headers, normalized line, predictor/config/catalog identity, all warnings,
   full distributions, evidence deltas, and analytical digest.
4. **Add post-game truth entry.** Accept White/Black rule and supported
   parameters only after analysis. Keep truth outside predictor input and
   produce ranks/timeline scoring without recomputing predictions.
5. **Make confidence semantics explicit.** State “posterior within represented
   hypotheses; not calibrated correctness probability,” especially for sampled
   Gambler parameters.
6. **Enforce the offline build.** Add a restrictive CSP (`connect-src 'none'`
   for the offline artifact), no remote assets, and an end-to-end network
   interception test that fails on any request after initial local assets.

P0 acceptance:

- a 200-ply fixture does not block UI interaction and can be cancelled;
- oversize input fails before analysis without partial stale results;
- same PGN/provider/config produces byte-identical analytical JSON;
- modifying user-entered truth changes no stored prediction or digest;
- hard-eliminated hypotheses always remain probability zero;
- no runtime request escapes the packaged offline origin; and
- no failed/unpublished neural model is presented as active.

### P1 — usable post-game investigation

1. Add `.pgn` file selection, drag-and-drop, explicit single-game selection for
   multi-game files, and preserved headers/comments in source metadata.
2. Add a replay board synchronized to a confidence timeline and evidence pane.
3. Show full searchable distributions, survivor/variant counts, parameter
   coverage, and exact elimination reason/move.
4. Validate declared PGN result when possible while supporting partial games
   with a clear warning.
5. Add report import/read-only rendering and annotated-PGN export.
6. Add batch CLI analysis using the same report schema and symbolic provider,
   with no network behavior.

P1 acceptance:

- UI, worker, and CLI produce the same analytical digest for the same fixture;
- unsupported PGN constructs generate explicit warnings or errors rather than
  silent reinterpretation;
- selecting any timeline ply reconstructs its public FEN and displayed
  posterior; and
- sampled parameter families cannot be labeled exhaustive.

### P2 — gated hybrid model

1. Promote a hybrid artifact only after a new validation protocol passes its
   preregistered gates; do not use the rejected v2.1 checkpoint.
2. Implement the provider contract in a worker-compatible runtime with pinned
   artifact digest and schema validation.
3. Apply exact symbolic elimination after neural scoring and test the invariant
   across every exported prefix.
4. Ship calibration metadata and display calibrated versus uncalibrated status.
5. Add deterministic symbolic-versus-hybrid comparison to exported reports.

P2 acceptance:

- the release record links artifact, code, catalog, validation, calibration,
  and license hashes;
- corrupt/incompatible artifacts fail closed to Symbolic v1 with a visible
  reason;
- neural inference never revives a symbolic elimination;
- browser and reference Python inference agree within declared tolerance on
  public fixtures; and
- performance, memory, startup, and long-game budgets pass on the supported
  browser matrix.

## Production readiness checklist

| Capability | Current status | Release requirement |
| --- | --- | --- |
| Local standard-legal replay | Implemented | Expand parser conformance and bound resources. |
| Separate White/Black inference | Implemented | Preserve in provider/report contracts. |
| Exact symbolic elimination | Implemented | Export evidence and retain hard-zero invariant. |
| Parameter coverage disclosure | Partial | Include all families and report-level provenance. |
| Neural inference in web | Absent | Do not claim until gated provider ships. |
| Supported model artifact | Absent | Validation-approved, hashed, licensed release only. |
| Post-game truth/reveal scoring | Absent for PGN | Add input-isolated truth and scoring. |
| Durable report | Absent | Canonical JSON plus import and copy summary. |
| Responsive long-game analysis | Absent | Worker, limits, progress, cancel, benchmarks. |
| Enforced offline policy | Partial | CSP, local assets, network-deny E2E. |
| Accessibility/keyboard QA | Unverified | File/input/status/table/worker flows audited. |
| Cross-browser/package QA | Unverified | Defined support matrix and offline artifact smoke. |

## Recommended delivery slices

1. **v2.2a:** provider naming, worker boundary, limits/progress/cancel, canonical
   report export, and offline network test.
2. **v2.2b:** file/multi-game import, replay/timeline/evidence UI, post-game
   truth scoring, report import, and shared CLI.
3. **v2.2c:** only after model approval, signed hybrid provider, calibration,
   fallback, and cross-runtime parity.

This sequence makes the current symbolic value production-usable without
waiting for a neural breakthrough, while preventing an unvalidated model or
client-side secret from being presented as stronger than it is.
