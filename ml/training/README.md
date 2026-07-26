# Drawback model training

## Capturable-king research baseline

Schema-8 capturable datasets have a separate, non-release baseline. Selection
accepts already-converted, game-disjoint train and validation files and trains
from a deterministic random initialization:

```powershell
py -m ml.training.drawback_ml.capturable_experiment select `
  --train ..\DrawbackTrainingData\capturable-train.ndjson `
  --train ..\DrawbackTrainingData\capturable-train-extra.ndjson `
  --validation ..\DrawbackTrainingData\capturable-validation.ndjson `
  --output ..\DrawbackTrainingData\capturable-selection-run `
  --seed 3235776257 --epochs 8 --batch-size 256 `
  --hidden-dimension 128 --torch-threads 14
```

The selection command has no test-path argument and publishes
`sealedTestStatus: "unopened"`. After the model seed, architecture, selected
epoch, fusion alpha, and prior smoothing are frozen, compare multiple
predeclared candidates using validation only:

```powershell
py -m ml.training.drawback_ml.capturable_experiment choose `
  --candidate ..\DrawbackTrainingData\candidate-a\selection.json `
  --candidate ..\DrawbackTrainingData\candidate-b\selection.json `
  --output ..\DrawbackTrainingData\candidate-selection.json
```

The chooser requires byte-canonical reports, identical train/validation
inputs, unique candidate identities, and matching checkpoint hashes. It
preserves `sealedTestStatus: "unopened"`. Evaluate only the selected checkpoint:

When an intervention deliberately adds a train-only corpus, compare its frozen
selection reports against the frozen control with the unchanged validation
identity:

```powershell
py -m ml.training.drawback_ml.capturable_experiment compare-treatment `
  --control ..\DrawbackTrainingData\control\selection.json `
  --treatment ..\DrawbackTrainingData\treatment-a\selection.json `
  --treatment ..\DrawbackTrainingData\treatment-b\selection.json `
  --output ..\DrawbackTrainingData\treatment-comparison.json
```

This command permits different training inputs, but requires an exact
validation identity and identical model/training configuration apart from the
random seed and trigger-row multiplier. It authenticates every checkpoint,
uses validation metrics only, and leaves the sealed test unopened. Parameter
tie-breaks may select between treatments, but cannot promote a treatment whose
Top-1, Top-3, and NLL tuple merely ties the control.

After a treatment wins validation and a new test split has been preregistered,
evaluate the authenticated control and treatment together:

```powershell
py -m ml.training.drawback_ml.capturable_experiment evaluate-treatment `
  --comparison ..\DrawbackTrainingData\treatment-comparison.json `
  --test ..\DrawbackTrainingData\fresh-test.ndjson `
  --output ..\DrawbackTrainingData\paired-sealed-evaluation.json
```

The paired evaluator rejects altered comparison/checkpoint identities and
train/validation overlap, loads the fresh test once, evaluates both frozen
models without an intervening decision, and publishes one no-clobber report.

```powershell
py -m ml.training.drawback_ml.capturable_experiment evaluate `
  --checkpoint ..\DrawbackTrainingData\capturable-selection-run\model.pt `
  --test ..\DrawbackTrainingData\capturable-test.ndjson `
  --output ..\DrawbackTrainingData\capturable-sealed-evaluation.json
```

The checkpoint records every train/validation game ID. Sealed evaluation
rejects any overlapping test game, authenticates the checkpoint and test
bytes, loads tensors with PyTorch's restricted weights-only loader, and
refuses to overwrite an existing report.

`--train` may be repeated for independently generated training corpora. The
loader requires canonical UTF-8/LF records, exact schema keys, complete
public capturable-authority state, and a surviving true symbolic hypothesis.
It separates public features from labels before feature construction and
rejects overlapping game IDs. Validation game-normalized Top-1, then Top-3,
then game-normalized NLL jointly select the epoch and frozen fusion settings; the test split is
evaluated only after that choice. Hard-mask fusion keeps every exact
elimination irreversible while
allowing learned public move patterns to rerank the surviving soft
hypotheses. Validation also selects bounded prior smoothing so a non-eliminated
hypothesis with tiny soft mass is not mistaken for a mathematical
contradiction. Training losses weight every player-game equally, so a long game
cannot dominate simply by contributing more move rows. Optional
`--trigger-row-multiplier` weighting reallocates mass inside a player-game
toward exact restriction events without changing that player-game's total
weight or using the trigger label at inference time. Public behavior features
cover current mover/capture/castling facts, authority-move composition, per-side
piece and tactical history, recent piece types, and repeated-piece streaks.

The selection output directory must not already contain `model.pt` or
`selection.json`. Both files bind the input hashes and record
`freshStart: true`; all selection and sealed-test artifacts belong outside the
repository. The training code now targets the Engine's 25-rule capturable
catalog. No 25-label accuracy is claimed until a fresh, disjoint schema-8
experiment is complete. Published schema-7 results remain a historical
10-label baseline, not a promoted browser model or a claim about the full
catalog.

## Manifest-bound current-catalog training

Release-candidate training must start from the public release root plus only
the authorized train-private manifest and train dataset. The training process
must not receive validation/test private manifests:

```powershell
py -3.11 -m ml.training.drawback_ml.cli train-release `
  data/releases/current/public/manifest.json `
  data/releases/current/private/train/manifest.json `
  data/generated/current/train.ndjson `
  data/generated/current/model-seed-20260811 `
  --execution-source-revision <40-character-clean-HEAD-SHA> `
  --seed 20260811 --model-variant v1 --epochs 5 --device cuda `
  --player-game-examples-per-epoch 8
```

Release training refuses to start unless the supplied revision is the current
clean Git `HEAD`, records that revision in the run claim, and checks the same
clean source state again immediately before atomic publication.

The audit verifies canonical public/private manifest bytes, the public root's
commitment to the private train manifest, run and split binding, the exact
dataset SHA-256 and byte length, declared and observed seeds/games/rows, the
per-game outcome ledger, hash-isolated split assignment, all public evaluator
requests, uniform evaluator identity, symbolic schema, and empirical
per-rule/per-color observed-row coverage. Zero-ply games remain authenticated
scheduled assignments but produce no move examples; one-sided games use the
schedule-bound White and Black class labels without inventing missing hidden
parameter targets. `train-corpus` additionally fixes
the output heads to the canonical ordered 182-rule vocabulary and stores the
audited corpus provenance in every checkpoint and `run.json`.

The manifest-bound command is a bounded-memory multi-pass trainer. Its memory
use is proportional to the authenticated game schedule and discovered
vocabularies, plus one game, the configured shuffle window, and one batch; it
does not materialize all NDJSON move rows. `--shuffle-buffer-size` controls the
deterministic streaming shuffle. The output directory must not already exist.
Training writes to a unique same-parent staging directory, re-audits the corpus,
and publishes the completed directory with a no-clobber rename. Failed runs and
corpus changes therefore leave no checkpoint at the requested output path.
After final validation, the staging directory receives
`checkpoint-index.claim.json`: a canonical, content-addressed byte inventory
that binds the semantically validated run claim to every expected epoch in
order. Checkpoint hashes are calculated immediately after each save so index
publication does not add one large end-of-run reread. The index verifier
authenticates the claim's run ID, seed, and epoch count plus every referenced
file. It deliberately does not deserialize PyTorch checkpoint payloads;
selection and ensemble loading remain responsible for restricted checkpoint
loading and embedded seed/epoch/model-contract validation.

Before the global shuffle, every row-bearing game contributes exactly the
configured number of moves per epoch. Long games are sampled without
replacement; short games are sampled deterministically with replacement.
Sampling uses a versioned SHA-256-derived seed for each game and epoch, never
Python's randomized `hash()`. This prevents long games from dominating the
loss while preserving one-sided games' missing-parameter masks. Checkpoints and
`run.json` record the policy/version, K, raw rows, row-bearing games, and
effective examples per epoch. A drawback head is supervised only after that
color has made at least one publicly observed move, so short games cannot teach
an opponent label from behavior that never occurred.

Drawback classification is the primary objective. Trigger and hidden-parameter
losses default to weight `0.1`, while the sparse legal-mask auxiliary defaults
to `0.05`; all three weights are explicit CLI options and checkpoint metadata.
Hybrid training fails closed if public symbolic evidence eliminates the known
true class. Symbolic feature version 6 conservatively marginalizes Gambler's six
per-turn forbidden piece types instead of treating sampled seeds as exhaustive;
it does not claim to infer correlations from Gambler's hidden 32-bit seed.

CUDA training is opt-in. The selected PyTorch build must contain the GPU's
compute architecture, and deterministic CUDA runs require
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`) before launch. Device, Python,
PyTorch, CUDA, cuDNN, GPU capability, deterministic settings, corpus identity,
and run identity are stored with the run. Runtime metadata participates in the
run identity, so different supported Python, PyTorch, or accelerator stacks do
not claim the same identity. Use `--device cpu` when those CUDA
requirements are not satisfied.

`--allow-incomplete-catalog` exists only for bounded pipeline smoke audits. A
corpus accepted with that flag cannot enter `train-corpus` or model promotion.
The older bare-file `train` command remains available for historical research
reproduction and does not produce a promotable checkpoint.

The older `train-corpus` command consumes the monolithic manifest and remains
available only for explicitly labeled research reproduction. It must not
produce a release candidate because parsing that manifest exposes withheld
split metadata. The current private audit proves corpus identity, the public
evaluator boundary, and every serialized standard-chess transition. Its
independent `python-chess` replay holds only one bounded game at a time and
verifies the pre-move FEN, side and move number, complete prior SAN history,
exact ordinary legal-move set, observed UCI move, SAN, and outcome-ledger final
FEN. Only an explicit projection of public fields reaches the replay verifier;
drawback labels, hidden parameters, authoritative drawback state,
drawback-legal moves, and results do not cross that boundary.

The replay also validates the version-6 symbolic array shape, finite normalized
probabilities, zero probability for eliminated hypotheses, monotonic hard
elimination, and isolation of the non-moving player's symbolic head. It does
not duplicate the TypeScript drawback hypothesis engine in Python, so it
cannot independently recalculate the exact surviving posterior weights.
Authenticated manifests, exact evaluator requests, and the true-drawback
non-elimination check remain separate fail-closed audit layers.

The default `v1` model is the measured feed-forward control. It consumes only
the existing public board and move feature vector, and its architecture and
loss remain unchanged.

`v2-gru` is opt-in:

```powershell
python -m drawback_ml.cli train dataset.ndjson artifacts `
  --seed 7 --model-variant v2-gru
```

It combines the same board feature encoder with a bounded GRU over public
`historySan`. SAN is tokenized exactly—there is no hashing—and the vocabulary
is fitted only from training-split examples. `<pad>` and `<unk>` are stable
reserved tokens. The vocabulary, maximum history, padding, truncation, model
dimensions, objective, feature schema, and split configuration are recorded in
each checkpoint so held-out inference reconstructs the same preprocessing and
architecture.

`v21-hybrid` adds versioned public symbolic posteriors and elimination masks:

```powershell
python -m drawback_ml.cli train dataset.ndjson artifacts `
  --seed 7 --model-variant v21-hybrid
```

Its neural logits are residuals on the symbolic log-prior. Exact elimination
is applied after the neural head, so a contradicted hypothesis always receives
zero probability. Hybrid corpora must contain all four symbolic vectors with
the supported feature version; older corpora remain valid for `v1` and
`v2-gru`.

The sequence model's legal-mask loss gives legal and illegal cells equal total
weight. This prevents the much larger illegal class from dominating the
objective. Other output losses retain the v1 definitions.

No model accepts drawback labels, hidden parameters, internal rule state,
drawback-legal moves, triggers, or results through its inference API. Those
values remain labels used only after the public feature boundary.

## Browser artifact export

Feed-forward `v1` and sequence-aware `v21-hybrid` checkpoints can be exported
to deterministic, versioned JSON artifacts:

```powershell
python -m drawback_ml.cli export-browser `
  artifacts/baseline-seed-0000000007-epoch-0003.pt `
  artifacts/browser-model.json
```

The exporter loads checkpoints with PyTorch's restricted `weights_only` mode,
validates format and feature versions, exact model-state keys, tensor shapes,
dense float32 types, finite values, and vocabulary dimensions, then writes
canonical UTF-8 JSON atomically. The artifact records the SHA-256 of the exact
checkpoint byte snapshot used for export.

Format version 1 contains the two-layer feed-forward encoder and White/Black
drawback classification heads. Format version 2 contains the public board and
symbolic encoders, exact-SAN embedding and GRU, and both residual drawback
heads. The exporter validates every checkpoint tensor, including auxiliary
parameter, trigger, and legal-mask heads, but omits those auxiliary tensors from
the browser artifact. The browser reconstructs the same symbolic prior and hard
mask contract; it never receives labels or hidden engine state. Plain
`v2-gru` checkpoints remain unsupported because they do not provide the hybrid
legality boundary. Generated artifacts are release outputs and should not be
committed without an explicit model-release review.
