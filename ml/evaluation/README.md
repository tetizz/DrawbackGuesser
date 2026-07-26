# Model evaluation

`ml.evaluation` calculates classification, calibration, time-to-identification,
parameter, entropy, diagnostic, and confusion metrics from explicit prediction
records. It has no third-party runtime dependency and contains no claimed model
results.

## Legal-set identifiability

`ml.evaluation.identifiability` reports exact public hard-elimination state and
a separate oracle-only mask diagnostic. Each offline observation supplies:

- the ordinary legal-set trajectory reconstructed from public positions;
- one unique hypothesis ID, drawback label, permitted-set trajectory, and
  elimination flag for every parameter hypothesis;
- an optional public posterior for label-level tie scoring; and
- the true drawback only as an evaluation label.

Exact public identifiability uses only distinct drawback labels left after hard
elimination. A prefix is exact-identifiable only when one label survives.
Uniform hard-legal Top-k tie credit is `min(k, surviving_labels) /
surviving_labels`; it is a comparator, never an accuracy ceiling.

Full permitted masks for every counterfactual parameter variant are not
observed in a PGN. The evaluator retains them only as
`full_mask_diagnostic_separability`: it partitions surviving variants by their
complete caller-supplied mask histories and reports variant mask-partition
entropy and the labels sharing the true variant's mask class. These diagnostics
describe potential contrast under the rule engine; they do not establish
public recoverability.

Variant posterior masses are summed by drawback label in canonical
`hypothesis_id` order. Reports distinguish fractional top-max tie credit,
an explicit count/rate for top-max sets that exclude the truth, unique-top
correctness, and unique-top error. A unique-top error is an error on an
exact-identifiable prefix only when one hard-legal label survives. The
descriptive `top_max_labels_disjoint_from_true_variant_mask_labels` count is not
called a model error because the counterfactual mask class is oracle-only.

Opportunity compares the labeled true variant's permitted mask with ordinary
legality. Reports include current opportunity, cumulative opportunity
count/rate, first opportunity turn index, and restriction/addition
counts/fractions. Explicit turn indices must be strictly increasing, align with
every history, and end at or before the reported horizon. Restriction fraction
uses the cumulative ordinary-move count as denominator; addition fraction uses
the cumulative true-variant permitted-move count.

Reports include overall, horizon, rule, color, and color/horizon/rule slices.
The accumulator rejects duplicate game/color/horizon rows, changing player-game
truth, changing hypothesis identities, restored hard eliminations, and any
ordinary, turn-index, or per-hypothesis history that is not an exact
longitudinal prefix. Longitudinal validation sorts each player-game by horizon,
so valid reports are invariant to caller input order.

This module validates consistency, not provenance. A caller could still supply
masks derived with secret state. Inputs therefore require an external,
content-addressed, label-blind replay receipt proving reconstruction from
public FEN, ordinary moves, and public history. Until such a receipt is bound,
this report is diagnostic only and must not be used as model-release evidence.
The true hypothesis ID selects the labeled variant only during offline scoring;
it is never a model input or candidate-trajectory input.

Accuracy at 5, 10, 15, and 20 uses predictions recorded at exactly that
observed ply for each independently scored White or Black head.
When a cutoff has no records, the report returns `None` instead of inventing a
value. Mean first rank-one move includes games that reach rank one and returns
`None` when none do. Multiclass Brier score is the per-example sum of squared
class errors, averaged across examples.

`SplitManifest` rejects duplicate seeds within a split and overlap between
training, validation, and test splits.

The current diagnostic CLI evaluates the complete validation split and
requires the content-addressed schema-6 manifest plus its exact bound file:

```powershell
py -3.11 -m ml.evaluation.cli `
  data/generated/current/checkpoint.pt `
  data/generated/current/validation.ndjson `
  data/generated/current/manifest.json `
  --split validation --batch-size 256 `
  --output data/generated/current/validation-report.json
```

Before checkpoint inference, the command runs the same strict corpus audit as
training, requires the checkpoint's ordered vocabulary to equal the canonical
182-rule schema, and rejects a checkpoint whose authenticated training-corpus,
engine, evaluator, rule-catalog, or symbolic-schema identity differs from the
supplied manifest. Rows and metrics are processed with bounded memory; the
report is published atomically and an existing report is never overwritten.
Checkpoint predictors evaluate bounded batches (256 examples by default);
predictors without a batch API retain the scalar compatibility path.
The JSON output is an envelope containing measured metrics plus exact
checkpoint, manifest, and held-out dataset SHA-256 provenance. Test evaluation
remains a separate, one-shot promotion action after the frozen validation
protocol passes.

After selecting one checkpoint for each fixed training seed, publish the
authenticated three-member ensemble in seed order (`20260811`, `20260812`,
`20260813`):

```powershell
py -3.11 -m ml.evaluation.cli create-ensemble-release `
  data/generated/release/ensemble.json `
  --selection data/generated/release/20260811/selection.json `
  --selection-sha256 <selection-20260811-sha256> `
  --selection data/generated/release/20260812/selection.json `
  --selection-sha256 <selection-20260812-sha256> `
  --selection data/generated/release/20260813/selection.json `
  --selection-sha256 <selection-20260813-sha256> `
  --training-run data/generated/release/20260811/run.claim.json `
  --training-run-sha256 <run-20260811-sha256> `
  --training-run data/generated/release/20260812/run.claim.json `
  --training-run-sha256 <run-20260812-sha256> `
  --training-run data/generated/release/20260813/run.claim.json `
  --training-run-sha256 <run-20260813-sha256>
```

The command recursively verifies every selection, training claim, and selected
checkpoint, rejects mixed provenance or incorrect seed order, refuses to
overwrite an existing release, and prints the release filename and SHA-256 as
JSON. Each repeated path is paired positionally with its corresponding digest.

The envelope declares `serialization.nonFiniteMetricPolicy = "null"`:
mathematically infinite losses are represented as JSON `null`, never the
non-standard `Infinity` or `NaN` tokens.

## Post-training release orchestration

The release operations below are deliberately separate CLIs; there is no
implicit command that searches a training directory or opens the sealed test.
First create a canonical builder lock containing the exact SHA-256 references
for the three completed checkpoint indexes, current validation release,
training-frequency comparator, public parity fixture, and four external tools.
All inputs are explicit; the lock writer does not search model directories:

```powershell
py -3.11 -m ml.evaluation.release_workflow_lock `
  data/generated/release/post-training-builder-lock.json `
  --repository . `
  --source-revision <full-git-sha> `
  --browser <browser-executable> `
  --git <git-executable> `
  --node <node-executable> `
  --pnpm <pnpm-executable> `
  --dataset data/generated/current/validation.ndjson `
  --public-root data/releases/current/public/manifest.json `
  --private-validation data/releases/current/private/validation/manifest.json `
  --checkpoint-index data/generated/models/20260811/checkpoint-index.json `
  --checkpoint-index data/generated/models/20260812/checkpoint-index.json `
  --checkpoint-index data/generated/models/20260813/checkpoint-index.json `
  --training-frequency data/generated/release/training-frequency.json `
  --browser-fixture data/generated/release/public-parity-fixture.json `
  --output-root data/generated/release/frozen
```

The three repeated `--checkpoint-index` options are interpreted in occurrence
order and must be supplied in fixed seed order. The writer authenticates each
index and rejects a seed or eight-epoch mismatch. It hashes every regular file, rejects paths outside
the repository except for the four explicitly typed tools, restricts generated
outputs to `data/generated`, and atomically refuses to overwrite an existing
lock. Then build the closed workflow without scanning a model directory:

```powershell
py -3.11 -m ml.evaluation.release_workflow_builder `
  data/generated/release/post-training-builder-lock.json `
  data/generated/release/post-training-workflow.json `
  --repository .
```

The builder accepts only fixed seeds `20260811`, `20260812`, and `20260813`,
each with eight checkpoints authenticated by its no-clobber checkpoint index.
It verifies every index, training claim, and checkpoint hash; requires the
three claims to share one training-corpus and public-release identity; and
requires validation data to be bound by the current public and private
validation manifests. The training-frequency comparator must match that exact
training corpus. The builder never selects a latest file and its lock schema
contains no sealed-test reference. It publishes canonical JSON atomically,
refuses to overwrite an existing file, and rejects a workflow filename that
collides with any declared release input or output. Generate the
training-frequency comparator after GPU training finishes, before constructing
the final builder lock.

Use the manifest-driven orchestrator to bind them into one reviewed sequence:

```powershell
py -3.11 -m ml.evaluation.release_workflow `
  data/generated/release/post-training-workflow.json `
  --repository .
```

Dry-run is the default. It validates and prints the complete plan without
starting a process or writing release data. Version 2 is a closed typed schema,
not a list of commands. It declares:

1. selection-fit evaluation and selection-summary emission for every epoch
   1–8 of seeds `20260811`, `20260812`, and `20260813`;
2. one epoch selection for each seed in that seed order;
3. ensemble release, calibration evaluation and fitting, validation gate,
   browser artifact export, parity-input production, and browser parity;
4. authenticated training-frequency, public parity fixture, and external
   browser references required by downstream gates.

Each seed has exactly eight typed epoch records with distinct `checkpoint`,
`report`, and `summary` paths. The orchestrator constructs the approved
`ml.evaluation` module and subcommand arguments internally, always requests
the frozen `selection` partition, and binds later stages to hashes of the
declared outputs. Every pre-existing project input—including manifests,
validation data, training claims, checkpoints, training-frequency evidence,
and the public parity fixture—is a closed
`{"path": "...", "sha256": "..."}` reference authenticated before execution.
No command or module name is accepted from the manifest.
All project paths are normalized repository-relative POSIX paths without
backslashes, drive prefixes, or `..`; execution rejects symlink traversal and
resolved paths outside the repository. Inputs and outputs must be globally
distinct, so a release output cannot overwrite an authenticated input.
External browser, Git, Node.js, and pnpm executables are separately typed
absolute-path exceptions and must include expected SHA-256 values. Execution
uses those authenticated tools through a closed PATH and launches the approved
Python modules with isolated startup, so repository `sitecustomize.py` and
ambient `PYTHONPATH` cannot alter them.

After review, add `--execute` to run the validated workflow. The 24 independent
selection-fit evaluations run in a fixed four-worker wave, with results recorded
in canonical seed/epoch order. Selection summaries and every dependent stage
remain sequential. This uses bounded CPU/RAM while preserving the same
authenticated inputs, no-clobber outputs, selection logic, and transcript order.
The first worker failure stops queued evaluations immediately; at most the other
three already-running reports may finish afterward. All reports published by a
failed wave must be archived before retrying.
The orchestrator rejects incomplete or reordered 3×8 identities, duplicate
artifact paths, unknown fields, and any output that already exists. After each
successful stage it requires every declared no-clobber output to be a confined
regular file and records its SHA-256. Execution publishes a canonical
transcript bound to the canonical workflow SHA-256 and exact source revision.
Pre-existing inputs are reauthenticated immediately before and after every
stage that consumes them. Generated reports, summaries, selections, ensemble,
calibration, and parity artifacts are likewise hashed as typed dependencies,
checked before and after their consuming stage, and recorded in the transcript.
The transcript explicitly contains `evidence:false`: orchestration receipts
are not promotion evidence or a filesystem sandbox. Generated workflow
manifests and outputs belong under ignored `data/generated/` release
directories.

Real-domain Stage A/B currently expose audited Python library functions in
`ml.evaluation.real_domain_benchmark`, not standalone CLIs. They are a hard
external handoff and cannot be executed by this runner. Stage A requires
separate provisioning and an independently authenticated no-label
mount/access-domain receipt; Stage B requires separate label-aware
authorization. Neither the dry-run nor transcript claims that labels were
unmounted. The sealed test is absent from the workflow schema.
The sealed-test opener is implemented by `ml.evaluation.sealed_test`; this diagnostic command deliberately rejects
`--split test`. Its whole-validation report is useful for research and
debugging but is not by itself a promotion decision.

The identity audit verifies exact bytes, labels, evaluator request facts, and
catalog coverage. Release-safe private split audits also run the independent
streaming semantic replay verifier: it replays FEN, turn, move number, SAN
history, ordinary legal moves, the observed move, and final FEN while checking
the symbolic hard-mask invariants. The verifier remains label-blind and bounded
to one game. Exact TypeScript posterior weights are not duplicated in Python;
their normalization, elimination, isolation, and known-truth invariants are
checked independently.

## Authenticated training-frequency comparator

The frozen validation protocol compares the candidate against a per-color
training-frequency prior. Build that prior from the exact authenticated
release training union, never from validation rows or from move-row
frequencies:

```powershell
py -3.11 -m ml.evaluation.training_frequency `
  data/releases/current/evidence/training-frequency.json `
  --public-root data/releases/current/public/manifest.json `
  --private-train data/releases/current/private/train/manifest.json `
  --primary-dataset data/generated/current/train.ndjson `
  --expected-training-corpus-set-sha256 <sha256> `
  --hard-negative checkers-pacman <manifest> <dataset> <plan> `
  --hard-negative truant-spice-of-life <manifest> <dataset> <plan> `
  --hard-negative oddball-even-keeled <manifest> <dataset> <plan> `
  --hard-negative quit-horsing-around-forward-march <manifest> <dataset> <plan> `
  --hard-negative horse-tranquilizer-conscientious-objectors <manifest> <dataset> <plan> `
  --hard-negative gambler-truant <manifest> <dataset> <plan>
```

Construction pins and audits the primary training split and all six frozen
hard-negative supplements. Each observed White or Black player contributes
exactly one count per game regardless of game length. The canonical,
no-clobber artifact records the ordered 182-rule counts, complete corpus-set
identity, and content bindings for every root, manifest, dataset, and plan.
`load_training_frequency_artifact` rejects noncanonical JSON, an altered count
total, source-binding drift, rule-order drift, or a corpus-set mismatch.
Release review can call `verify_training_frequency_sources` with the same
seven explicit source bindings to re-audit them and reproduce the canonical
artifact byte for byte.

## Frozen validation gate

The promotion candidate is evaluated only on the preregistered
`validation-gate` subpartition:

```powershell
py -3.11 -m ml.evaluation.validation_gate `
  <ensemble-release.json> <calibration.json> <training-frequency.json> `
  <validation.ndjson> `
  --ensemble-sha256 <sha256> `
  --calibration-sha256 <sha256> `
  --training-frequency-sha256 <sha256> `
  --public-root <public-manifest.json> `
  --private-validation <private-validation-manifest.json> `
  --report-output <validation-gate-report.json> `
  --decision-output <validation-gate-decision.json>
```

The partition, bootstrap seed (`20260814`), 10,000 complete-game paired
replicates, class views, comparator systems, and every threshold are fixed in
code and cannot be overridden from the command line. The command publishes a
canonical no-clobber report and a separately hashed exhaustive decision.
Every frozen protocol requirement has a stable gate ID. Missing, null,
non-finite, or unsupported evidence is recorded with status `missing` and
forces `passed` to remain false; it is never inferred from an adjacent metric.
The sealed test split is not accepted by this command.

## Independent validation reproduction

After every validation gate passes, reproduce it from the same authenticated
inputs in a fresh Python process:

```powershell
py -3.11 -m ml.evaluation.validation_reproduction `
  <validation-gate-report.json> <validation-gate-decision.json> `
  <ensemble-release.json> <calibration.json> <training-frequency.json> `
  <validation.ndjson> `
  --original-report-sha256 <sha256> `
  --original-decision-sha256 <sha256> `
  --ensemble-sha256 <sha256> `
  --calibration-sha256 <sha256> `
  --training-frequency-sha256 <sha256> `
  --public-root <public-manifest.json> `
  --private-validation <private-validation-manifest.json> `
  --repository-root <clean-checkout> `
  --expected-source-revision <full-git-sha> `
  --pnpm-lock-sha256 <sha256> `
  --python-requirements-sha256 <sha256> `
  --python-project-sha256 <sha256> `
  --engine-binary-sha256 <sha256> `
  --engine-fingerprint <fingerprint> `
  --receipt-output <validation-reproduction-receipt.json>
```

The parent requires a clean approved source revision and exact dependency
locks, pins the authenticated validation split, and launches
`ml.evaluation.validation_gate` in a separate process with deterministic
Python settings. Candidate and input hashes plus the inference transcript must
match exactly; every reproduced metric must be within `1e-6`. The canonical
no-clobber receipt binds the original and reproduced evidence, validation
corpus, catalogs, source, locks, Python executable, evaluator identity, and
exact child command. This command exposes no test partition or private-test
path and rejects a non-passing original or reproduced decision.

## Production browser parity

The standalone split preserves the authenticated public replay-fixture
consumer, but the pinned Engine does not yet expose the former fixture
generator as a public CLI command. Do not promote a release until a fixture has
been generated by a versioned Engine producer and passed the checks below.

Protocol v1 freezes the seed domain, root seed, eight-game count, 320-ply
bound, and random/weak/greedy agent schedule. The canonical no-clobber fixture
declares `candidateInputs: []` and contains only public PGNs, standard PGN
results, seeds, FENs, ply counts, and replay hashes. It contains no drawback
identity, hidden parameter, internal state, label, or evaluator fact.
`load_public_parity_fixture` content-authenticates the manifest and uses
python-chess to replay every move independently before it can be used.
Each game also carries its final public `HybridObservation`: the ordinary
position, observed move, SAN history, complete legal-move list, ordered
182-rule symbolic probabilities, and hard masks. Python independently checks
the chess fields against replay and checks vocabulary, dimensions,
normalization, finite values, and exact masked zeros. These symbolic arrays are
public derived evidence, not labels. Python expected values intentionally share
this frozen TS symbolic boundary; the production Worker recomputes it from PGN,
while the predictor legality and PGN-analysis suites independently test the
symbolic transition boundary.
Python does not reimplement or independently recompute the symbolic rule
posterior; it authenticates and validates the TS-produced symbolic evidence.

Build the content-addressed Python expectation input with:

```powershell
py -3.11 -m ml.evaluation.browser_parity_input `
  <public-parity-fixture.json> <ensemble-release.json> <calibration.json> `
  <browser-model.json> `
  --fixture-sha256 <sha256> `
  --ensemble-sha256 <sha256> `
  --calibration-sha256 <sha256> `
  --browser-artifact-sha256 <sha256> `
  --repository <clean-checkout> `
  --output <validation-parity-input.json>
```

This invokes the shared two-head promotion inference helper, projects the two
explicitly unavailable evaluator-backed rules out of the 182-rule prepared
distribution, renormalizes the browser's 180-rule view, and publishes full
probabilities, exact Top-5, and hard-zero IDs. Every sanitized PGN string sent
to the browser has its UTF-8 SHA-256 recomputed in both fixture loading and
parity-input validation; an expected digest never substitutes for hashing the
actual string.

The parity gate runs the exported calibrated ensemble through the production
Vite bundle and an actual browser `Worker`; instantiating the Worker runtime
class inside Vitest is not accepted as release evidence:

```powershell
py -3.11 -m ml.evaluation.browser_parity `
  --repository <clean-checkout> `
  --browser "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" `
  --browser-artifact <browser-model.json> `
  --calibration <calibration.json> `
  --input <validation-parity-input.json> `
  --input-sha256 <sha256> `
  --transcript-output <browser-worker-transcript.json> `
  --evidence-output <browser-parity-evidence.json>
```

The authenticated input is a deterministic `validation-parity` subpartition
containing public PGNs and Python posterior vectors. Its closed schema does not
accept true drawbacks, hidden parameters, evaluator facts, sealed-test metrics,
or a test-partition path. It binds the ensemble, calibration, source revision,
dependency lock, partition selection, browser artifact, and fixture digests.
The partition ID must be nonempty, its selection digest must be canonical, its
declared example count must equal the cases, case IDs must be unique, and every
head supplies exactly the protocol's five ordered Top-k IDs.

The command performs a production web build, serves it only on loopback, loads
the exact content-addressed artifact, and drives `load-model` and `analyze`
messages through `pgn-analysis.worker.ts` in headless Edge or Chrome. It
requires exact hard-zero sets and Top-k ordering for both colors and a maximum
absolute probability difference of `1e-6`. The full canonical transcript
includes the Worker reports and fixture binding; the canonical no-clobber
evidence projection has the exact schema consumed by sealed-review
authorization. Existing output paths are never overwritten.
The transcript and evidence also bind the SHA-256 of the exact browser
executable and its reported version. Tracked changes or untracked source under
`apps`, `packages`, `ml`, or `scripts` invalidate the clean-source attestation;
ignored release data does not.
