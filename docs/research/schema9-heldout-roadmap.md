# Schema 9 held-out roadmap

## Decision

The next measurable model improvement is the existing Schema 9 public rule-opportunity residual, evaluated against its zero-opportunity ablation on a fresh, authenticated corpus.

This is the highest-value next experiment because the implementation already isolates one causal question: does exact public information about which rules had a chance to constrain the current turn improve hidden-drawback inference? The treatment changes only the `25 x 4` opportunity tensor. The control uses the same Schema 9 rows and the same model, but replaces that tensor with zeros after strict parsing. Adding a larger temporal or context-conditioned network before answering this question would confound feature value with architecture value.

No fresh held-out accuracy is established by this document. Until the experiment below completes, the only defensible statement is that the path is implemented and contract-tested, not that it improves guessing accuracy.

## Audited implementation boundary

The roadmap is grounded in the current public code rather than an assumed pipeline:

- [`packages/dataset-contract/src/index.ts`](../../packages/dataset-contract/src/index.ts) defines an exact public allowlist, rejects label and evaluation fields at the feature boundary, and parses combined storage rows into separate feature, label, and evaluation objects.
- [`packages/trace-to-dataset/src/player-private-converter.ts`](../../packages/trace-to-dataset/src/player-private-converter.ts) derives the public projection before attaching trusted labels. Its tests mutate valid hidden rules and require the public projection to remain unchanged.
- [`ml/training/drawback_ml/capturable_records.py`](../../ml/training/drawback_ml/capturable_records.py) strictly parses Schema 9, validates the `25 x 4` unit-interval tensor, and passes only `CapturablePublicFeatures` to feature construction.
- [`ml/training/drawback_ml/capturable_baseline.py`](../../ml/training/drawback_ml/capturable_baseline.py) implements the zero-initialized per-rule residual and the explicit `public-exact` and `zero-ablation` tensor modes.
- [`ml/training/drawback_ml/capturable_opportunity_workflow.py`](../../ml/training/drawback_ml/capturable_opportunity_workflow.py) authenticates matched pairs, binds disjoint corpus identities, freezes selection after validation A, gates validation B, and prevents early final-test access.
- [`ml/evaluation/metrics.py`](../../ml/evaluation/metrics.py) provides tie-aware row and game-normalized Top-1/3/5, NLL, Brier, calibration, move-horizon, slice, confusion, and hard-mask diagnostics.
- [`ml/evaluation/splits.py`](../../ml/evaluation/splits.py) rejects overlapping seed manifests, while [`ml/evaluation/runner.py`](../../ml/evaluation/runner.py) restricts evaluation to declared validation or test seeds and scores labels after inference.

The remaining evidence gap is a completed fresh-corpus run and an uncertainty report. Code-path tests cannot substitute for those measurements.

## Candidate and control

The experiment is limited to the 25-rule `capturable-king/v1` catalog and Schema 9.

- **Control — `zero-ablation`:** parse the exact Schema 9 rows, retain the base public feature vector and symbolic inputs, and replace every opportunity tensor with zeros.
- **Treatment — `public-exact`:** use the same parsed rows and the exact public opportunity tensor in catalog order. Its four fields are `knownMass`, `allowedMoveFractionMass`, `triggeredMass`, and `forcedMass`.
- **Shared model:** both arms use the same base encoder, drawback heads, auxiliary heads, training configuration, and zero-initialized `25 x 4` rule-specific residual parameters. A zero opportunity tensor must therefore reproduce the control logits exactly at initialization and must never alter the auxiliary heads.
- **Authority:** exact symbolic elimination is applied after neural scoring. A neural residual can rank surviving hypotheses but cannot give nonzero probability to a hard-eliminated rule.

The treatment under test is intentionally small. For rule `r`, its four public opportunity values are multiplied by the four learned weights for `r` and summed into that rule's logit. The same residual implementation exists in both arms; only the arm input differs.

## Frozen deterministic design

Use the existing `capturable25-schema9-opportunity-v1` protocol and version every change to it before any experiment data is opened.

1. Generate four disjoint splits: train, validation A, validation B, and final held-out test. The corpus ledger must bind the producer commit, converter commit, schedule identity, row count, game count, byte digest, game-ID set, simulation-seed set, and the independently derived label, gameplay, and parameter seed streams for every split.
2. Use exactly 2,500 games in validation A, 2,500 in validation B, and 2,500 in the final held-out test. Do not reuse game IDs, simulation seeds, parameter seeds, schedules, or dataset bytes across splits.
3. Train exactly three matched control/treatment pairs with model seeds `3685459371`, `480184104`, and `3192956725`. Within each pair, use identical train bytes, validation-A bytes, initialization seed, row order, batches, optimizer, epoch budget, loss weights, fusion grid, prior-smoothing grid, and all other configuration values.
4. Keep the frozen common configuration at 8 epochs, batch size 256, hidden dimension 128, and trigger-row multiplier 1.0. A protocol version bump is required to change any of these values.
5. Each arm may select its own epoch, fusion alpha, and prior smoothing on validation A only. Selection order is game-normalized Top-1, then game-normalized Top-3, then lower game-normalized NLL. Record every candidate, not only the winner.
6. Stage A promotes only when at least two of the three matched pairs pass their complete pair gate, at least two are eligible, mean game-normalized Top-1 delta is positive, and every aggregate non-regression gate below passes. Freeze the lower-median eligible pair by Top-1 delta; break ties by lower NLL delta and then lower model seed.
7. Evaluate that frozen pair once on validation B. Do not retrain, recalibrate, retune fusion, select another seed, or change preprocessing after Stage A.
8. Make the final held-out path available only after validation B authorizes it. Consume that split once. A failed validation stage ends the experiment; it does not authorize another candidate search on the same split.

The implemented consumption registry is a trusted-operator safeguard under one
Git common directory. It prevents accidental or repeated use only in worktrees
sharing that directory; its user-deletable markers do not follow another clone
and cannot enforce a global one-shot claim. Any public global one-shot claim
requires an external append-only authority that issues a signed, single-use
lease keyed by the sealed-corpus identity. That authority is not implemented in
this repository.

Every generated report must be canonical, content-addressed, no-clobber, and path-minimized. Replaying a decision from its bound inputs must reproduce the same bytes.

## Success metrics and gates

The primary metric is treatment minus control game-normalized Top-1 accuracy. Game normalization first averages all observed plies within each player-game and then gives every player-game equal weight, preventing long games from dominating the result.

### Existing deterministic promotion gate

The treatment must satisfy all of these conditions. There is no discretionary override.

- Game-normalized Top-1 delta is strictly greater than zero.
- Game-normalized Top-3 and Top-5 deltas are greater than or equal to zero.
- Game-normalized NLL and multiclass Brier deltas are less than or equal to zero.
- Expected calibration error delta is less than or equal to zero.
- Top-1 at 5, 10, 15, and 20 observed moves does not regress at any horizon.
- White and Black each have non-regressing game-normalized Top-1, Top-3, and NLL.
- Trigger and forced heads do not regress in accuracy, NLL, or Brier score.
- Hidden-parameter accuracy does not regress.
- No individual drawback loses more than 0.01 absolute Top-1 accuracy.
- Every scored row has a hard mask, the mask is checked, hard-elimination violations equal zero, missing-mask count equals zero, and maximum probability assigned to any eliminated hypothesis equals exactly `0.0`.

Stage A applies those pair gates and also requires positive mean Top-1 across the three seeds with no aggregate regression in Top-3, Top-5, NLL, Brier, calibration, all four move horizons, or either color. Validation B and the final held-out test each require the frozen treatment to beat the frozen control on Top-1 and pass the complete reliability gate independently.

### Uncertainty required for an accuracy claim

Before publishing a claim that Schema 9 improves accuracy, add an evaluation-only paired confidence report. It must not participate in model or hyperparameter selection.

- Resampling unit: whole game ID, preserving both colors, all plies, and the control/treatment pairing.
- Replicates: exactly 10,000.
- Random seed: the first unsigned 32 bits of `SHA-256("capturable25-schema9-opportunity-v1/paired-bootstrap/v1")`, interpreted big-endian.
- Statistic: treatment minus control game-normalized Top-1.
- Interval: two-sided 95% percentile interval using the sorted replicate values and nearest-rank endpoints.
- Claim gate: the lower endpoint must be strictly greater than zero on validation B and independently on the one-time final held-out result.

Always report the point estimate, interval, number of games, number of player-games, number of move rows, and all three training-seed outcomes. A promotion decision without the confidence report may justify further research, but not a public accuracy-improvement claim.

Secondary reporting must include Top-3, Top-5, NLL, Brier, expected calibration error, accuracy at 5/10/15/20 moves, mean and median first rank-one move, per-drawback and per-family slices with support, confusion counts, trigger and forced metrics, hidden-parameter accuracy, and exact hard-mask diagnostics. Do not substitute row-weighted accuracy for the primary game-normalized metric.

## Selection and test isolation

The following boundary is mandatory:

```text
train -> validation A selection -> freeze one matched pair
      -> validation B confirmation -> authorize one final held-out evaluation
```

- Training and epoch/fusion/calibration selection may consume train and validation A only.
- Validation B may confirm or reject the frozen pair. It may not choose a new checkpoint, seed, threshold, feature set, or metric.
- The final held-out split may not be resolved, opened, hashed, counted, or otherwise inspected before validation-B authorization is replayed successfully.
- Final held-out labels and metrics must never flow back into training, selection, calibration, feature generation, corpus targeting, or a second evaluation attempt.
- If implementation work changes the model, converter, feature contract, schedule, metric semantics, or corpus, start a new versioned experiment with fresh split seeds.

## Public-feature and leakage checks

All checks below must pass before Stage A. Fail closed on a missing check.

1. **Exact allowlist:** model inputs contain only the public board snapshot, FEN before the move, observed move, move number, ply, player color, causal SAN prefix, ordinary authority-legal moves, optional public clock, symbolic posterior and elimination vectors, and the Schema 9 opportunity tensor. Reject unknown fields.
2. **Forbidden fields:** true drawback, hidden parameters, drawback internal state, drawback-legal moves, rule-trigger label, forced label, result, game ID, seed, SAN label, bot identity/style/strength, and any provenance field must not enter feature construction or inference.
3. **Chronology:** compute the opportunity tensor from the public position and symbolic state before observing the move. Only after recording that tensor may the observed move update symbolic probabilities and elimination masks. `historySan` must contain exactly the prior `ply` moves and no future move.
4. **No per-hypothesis answer key:** opportunity features must not encode the observed move's legality under a hypothesis, private allowed-move counts or lists, hypothesis parameters/state, truth identity, or outcome.
5. **Private-mutation invariance:** mutate every valid private label and evaluation-only field while holding the public trace fixed. Parsed public features, serialized feature bytes, tensors, and pre-loss model outputs must remain byte-identical.
6. **Ablation identity:** control and treatment must have identical ordered row identities and identical non-opportunity tensors. Zeroing occurs only after the row passes the strict Schema 9 parser. The opportunity input must be the sole tensor difference.
7. **Causal vocabulary:** any vocabulary, normalization statistic, class prior, source weight, or calibration parameter is fit on training or the authorized selection split only. Unknown validation/test values must follow a frozen fallback path.
8. **Split isolation:** allow the expected multiple move rows for one game, but reject duplicate game assignments or schedules and reject any game-ID or seed-stream overlap across splits. Report set digests without exposing private row contents.
9. **Color isolation:** a White observation updates and scores White's hidden-drawback head; a Black observation updates and scores Black's head. Swapping private labels between colors must not change either head's public inputs.
10. **Symbolic supremacy:** deliberately apply an extreme positive neural residual to an eliminated rule and verify its final probability remains exactly zero.
11. **Deterministic replay:** two runs with the same corpus bytes, code, environment lock, seed, and configuration must produce byte-identical selection decisions and numerically identical metrics; record hardware and library versions when checkpoint bytes are not portable across devices.
12. **Prediction-label separation:** evaluation may use labels to score outputs only after inference. Mutating labels or legal-mask targets must not change the predictor output passed to the scorer.

## Evidence bundle

A completed experiment is reviewable only when it contains:

- the exact source commit and clean-worktree statement;
- environment and dependency lock identities;
- authenticated corpus-ledger and split identities;
- all three matched-pair configurations and checkpoint identities;
- validation-A candidate histories and the deterministic Stage A decision;
- the frozen-pair identity and validation-B decision;
- the one-time final held-out report and consumption evidence;
- raw metric bundles, paired deltas, the fixed-bootstrap report, per-rule support, and hard-mask diagnostics;
- logs for dataset-contract, leakage, determinism, evaluation, and workflow state-machine tests.

Do not summarize the result as a single accuracy number. At minimum, name the domain, rule count, data source, split size, game-normalized Top-1/Top-3/Top-5, NLL, Brier, calibration, uncertainty interval, and hard-mask result.

## Stop conditions and claim limits

Stop without opening a later split if any corpus identity, seed stream, chronology check, label-mutation invariant, matched-pair identity, replay digest, or hard-mask check fails. A failed gate is evidence against the candidate under this protocol; it is not permission to tune against validation B or the final held-out set.

Even a successful result supports only this claim:

> On the authenticated synthetic `capturable-king/v1` 25-rule corpus and the frozen Schema 9 protocol, the public-exact opportunity residual outperformed its matched zero-opportunity control by the reported held-out amount.

It does not establish accuracy on humans, live games, rules outside the 25-rule catalog, a different engine policy, a different drawback authority, or an external website. It does not validate covert live assistance. A human or strong-engine domain claim requires a separately preregistered, post-game-only dataset with fresh participants or games and its own untouched test split.

## Follow-on only after this experiment

If the treatment passes, the next architecture experiment should compare the frozen linear opportunity residual with a zero-initialized context-conditioned residual that can model interactions between the observed move, ordinary legal-move composition, symbolic prior, and each rule's four public opportunity values. That experiment must keep the linear treatment as its baseline and include a zero-input equivalence test.

If the treatment fails, use validation-only confusion and opportunity-support slices to determine whether the issue is feature sparsity, converter chronology, or insufficient model interaction. Do not inspect the final held-out split to choose the revision.
