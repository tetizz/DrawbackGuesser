# Model release readiness

Audit date: 2026-07-24
Repository base: `0f1f1cd4f0d202ed3354fd01e59c960c6a4f1dcc`

## Decision

**The post-game guessing workflow is operational, but no neural model is ready
for promotion.**

The browser can replay a pasted or local PGN, maintain independent White and
Black symbolic distributions, load an explicitly selected local v1 JSON model,
run both neural heads in the analysis worker, preserve exact symbolic
eliminations, show a per-ply timeline, accept optional truth only after
analysis, and export a provenance-bearing report. This is a usable research
delivery path.

It is not evidence that an accurate Drawback Chess guesser has been released.
There is no bundled model, no current-catalog checkpoint that passed a frozen
validation protocol, no calibration artifact, no one-shot current test report,
and no content-addressed release manifest. The only browser-compatible measured
checkpoint is the approximately-random 22-class v1 baseline. The stronger
27-class v2.1 experiment failed its preregistered gates and is incompatible
with the current rule universe.

The safe product boundary remains **offline, post-game analysis of user-supplied
public moves**. Promotion must not add scraping, live board observation, or
competitive external-site assistance.

## What is implemented and authoritative

| Capability | Current evidence | Readiness |
| --- | --- | --- |
| Observed catalog | `data/catalog/observed-drawbacks.json` records 194 observed, 182 executable, and 12 unsupported titles. `packages/drawback-engine/src/rules/executable-rules.test.ts` asserts the same counts. | Complete as an inventory, not as full rule support. |
| Offline hypothesis coverage | `apps/web/src/pgn-analysis.ts` excludes the two evaluator-backed rules when no uniform public evaluator stream exists. The UI declares 180/182 represented and names the unavailable rules. | Correct fail-closed behavior. |
| Browser model format | `docs/architecture/browser-hybrid-inference.md`, `ml/training/drawback_ml/browser_artifact.py`, and `apps/web/src/neural-model.ts` share format `drawbacktrainer-browser-model`, version 1, feature schema 1, exact tensor names, dimensions, and ordered vocabulary. | Implemented for feed-forward v1 only. |
| Browser execution | `apps/web/src/neural-model.ts` validates and executes the two-layer encoder and separate White/Black heads. `pgn-analysis.worker.ts` reparses the artifact inside the worker before analysis. | Implemented. |
| Symbolic authority | `apps/web/src/hybrid-prediction.ts` assigns zero mass to eliminated rules before normalization. `hybrid-prediction.test.ts` proves a dominant neural score cannot restore one. | Implemented and mandatory. |
| Local artifact trust display | `PgnAnalysisPanel.tsx` computes an artifact SHA-256, labels the model unpromoted and uncalibrated, limits local input to 8 MiB, and displays source-checkpoint provenance separately. | Suitable for research artifacts. |
| Post-game guesses | `PgnAnalysisPanel.tsx` displays independent final White and Black guesses and the prediction timeline. It labels confidence as mass within represented rules rather than probability of correctness. | Implemented with honest limitations. |
| Truth isolation | Truth selectors are enabled after analysis. `pgn-report.ts` separates analytical content from scoring content, and tests require truth changes not to alter the analytical digest. | Implemented. |
| Worker lifecycle | `pgn-analysis-controller.ts` gives each request an identity, terminates superseded workers, rejects stale results, and supports cancellation. | Implemented. |
| Report provenance | Reports distinguish symbolic and local-hybrid predictor identities and record model/checkpoint and artifact provenance when present. | Implemented for local analysis; not a release attestation. |

An integration audit against the existing v1 epoch-3 checkpoint also
established current numerical compatibility:

- checkpoint SHA-256:
  `9363d87c2509330272174c9c47c5bec00512fe6f346011ae0bf374f10c5a664e`;
- exported JSON size: 2,569,496 bytes;
- dimensions: 792 inputs, 128 hidden units, 22 drawback classes;
- maximum absolute difference across 44 Python/browser White and Black
  probabilities on one real public training record: approximately
  `1.5e-8`;
- the real artifact completed worker-compatible four-ply hybrid PGN analysis.

These are runtime-parity observations, not model-quality results.

## Why no current candidate can be promoted

### The browser-compatible v1 is not useful enough

`docs/research/baseline-v1-results.md` reports approximately random
classification over only 22 classes. Test Top-1 was 4.39% for White and 5.63%
for Black. The report explicitly rejects the feature/model design as an
effective classifier, and `ml/models/README.md` intentionally publishes no
checkpoint.

The v1 artifact path proves delivery mechanics. It must not be mistaken for a
candidate model.

### The v2.1 candidate failed and cannot use the v1 browser runtime

`docs/research/model-promotion-audit.md` records that the three-seed 27-class
v2.1 hybrid missed its frozen Top-3, NLL, and Brier gates. Its sealed test split
was therefore not opened. It covers only 27 of the current 182 executable
rules and uses an obsolete symbolic schema.

The format-v1 browser exporter deliberately rejects `v2-gru` and
`v21-hybrid`. Relaxing that rejection would be unsafe: their SAN tokenizer,
GRU state, symbolic encoder, feature version, and fusion contract are not part
of browser artifact version 1.

### Current catalog coverage has no trained release artifact

The simulation and training schemas can represent the current 182-rule
prepared domain, including uniformly generated evaluator facts. There is no
corresponding multi-seed trained checkpoint, validation report, calibration
record, browser artifact, or untouched test report. Twelve observed titles
remain unsupported by the executable rule engine, so no system can honestly
claim prediction coverage over all 194 observed titles.

## Required promotion artifacts

Promotion is a content-addressed release, not a copied JSON file. One immutable
release directory must contain all of the following:

1. **Browser model artifact**
   - a versioned runtime-compatible artifact below the declared browser size
     limit;
   - SHA-256 of its canonical bytes;
   - source checkpoint SHA-256;
   - exact ordered drawback vocabulary and represented rule intersection;
   - model, feature, symbolic, fusion, and calibration versions.
2. **Source checkpoint**
   - reconstructable format and exact model/training metadata;
   - content hash, selected epoch, training seed, and selection rule;
   - no optimizer state or unrelated heads in the browser delivery artifact.
3. **Frozen corpus manifest**
   - content hashes and row/game counts for train, validation, and sealed test;
   - disjoint seed lists and split algorithm/version;
   - current catalog hash and executable-rule IDs;
   - simulator revision, agent mixture, parameter sampling, maximum plies, and
     evaluator coverage policy;
   - proof that evaluator facts are uniformly present or uniformly absent,
     independent of true labels.
4. **Training run records**
   - at least the preregistered independent training seeds;
   - source commit and clean-tree status;
   - dependency lock hashes, Python/PyTorch versions, device and deterministic
     settings;
   - complete hyperparameters and epoch-selection evidence.
5. **Validation protocol and reports**
   - committed before candidate evaluation;
   - per-color, per-rule, per-family, per-parameter, and per-horizon metrics;
   - symbolic-only, uniform, class-prior, and previous-promoted comparators;
   - paired game-seed confidence intervals where the protocol requires them;
   - confusion matrix and hard-negative cohort results.
6. **Calibration artifact**
   - fitted using validation data only;
   - temperature or successor method, feature/version binding, validation
     report hash, and hard-elimination preservation;
   - calibration metrics before and after fitting.
7. **One-shot test report**
   - generated only after every frozen validation gate passes;
   - exact command, environment, selected checkpoint and manifest hashes;
   - no model selection, tuning, or calibration on test labels.
8. **Browser parity evidence**
   - permanent Python-to-TypeScript feature and logit golden fixtures;
   - tolerance and deterministic runtime declaration;
   - real Worker execution using the release artifact;
   - malformed, oversized, stale, cancelled, and wrong-version negative tests.
9. **Release manifest and approval**
   - hashes of every item above plus source revision and license notices;
   - explicit status `promoted`, supported product boundary, limitations, and
     rollback target;
   - reviewer sign-off that no ignored local path is the only copy of evidence.

Until this directory exists and verifies independently, a local artifact is
`experimental`, regardless of whether it parses successfully.

## Required gates

### G0 — rule universe and claim accuracy

- Freeze the release catalog and its hash before corpus generation.
- Every claimed class must have executable rule semantics and predictor
  hypotheses. Unsupported titles are reported as unsupported, never folded
  into an “other” class without a separately evaluated contract.
- Coverage claims distinguish 194 observed, 182 executable, 180 standard-PGN
  observable, and any smaller neural vocabulary.
- Evaluator-dependent rules are available only with a uniform, pinned public
  evaluator stream. Otherwise they remain explicitly unavailable.

### G1 — leakage and secrecy

- Feature construction remains an allowlist of public pre-move observations.
- Mutation of truth, hidden parameters, authoritative rule state, result,
  drawback-legal moves, trigger labels, and future moves leaves inference
  inputs unchanged or is rejected.
- White and Black secret state never crosses into the browser worker or
  report analytical payload.
- Post-game truth changes only scoring metadata and never reruns or mutates
  predictions.
- Hot-seat play retains the handoff curtain; model promotion does not weaken
  local secret isolation.

### G2 — deterministic corpus and split integrity

- Fixed root seeds reproduce byte-identical public examples across worker
  counts.
- Train, validation, and test are disjoint by complete game seed.
- Training vocabulary and parameter heads are fit from training only.
- All claimed rules, colors, parameter strata, and required edge cases are
  represented in validation and test or explicitly marked unscorable.
- Targeted hard negatives cover the largest known confusions without leaking
  test outcomes into generation.

### G3 — preregistered validation quality

A new current-catalog protocol must freeze numerical thresholds before reading
candidate validation reports. At minimum it must gate, independently for both
colors:

- Top-1, Top-3, Top-5, NLL, Brier, and expected calibration error;
- accuracy after 5, 10, 15, and 20 plies;
- mean ply when truth first reaches rank one;
- per-rule and per-family accuracy with minimum support;
- hidden-parameter accuracy and unscorable rate;
- entropy reduction and diagnostic information gain where claimed;
- zero probability for every hard-eliminated hypothesis;
- non-regression versus symbolic-only and the last promoted model.

The failed v2.1 thresholds cannot be silently relaxed after observing its
results. A successor protocol may choose different thresholds only with a
written rationale committed before training/evaluation.

### G4 — calibration and confidence language

- Calibration is selected and fitted on validation only, then frozen.
- The untouched test report demonstrates the calibration gate.
- Every UI and report confidence value identifies predictor, artifact,
  calibration, represented rule universe, and unavailable classes.
- If calibration is missing or fails, the UI continues to say “not
  calibrated”; the model is not promoted as a probability-of-correctness
  provider.

### G5 — browser/runtime equivalence

- Canonical artifact export and browser parsing agree on all names, dimensions,
  vocabulary order, tensor orientation, and feature order.
- A permanent golden fixture compares Python and browser logits/probabilities
  within a frozen tolerance.
- The exact release artifact runs in a real built Worker, not only a direct
  function or mocked transport.
- Export and load share one maximum-byte and maximum-dimension policy.
- Invalid neural inference fails visibly or switches to a separately identified
  symbolic-only result; it never retains a hybrid label.
- Symbolic hard masks remain authoritative after inference, calibration,
  report reload, and truth selection.

### G6 — real-domain usefulness

Synthetic self-play is necessary but not sufficient for the final
Drawback Chess guessing objective. Before a claim of real-game usefulness:

- evaluate on a legally and ethically acquired, consented set of completed
  Drawback Chess games with post-game revealed drawbacks;
- keep that domain set disjoint from simulated training and tuning;
- report coverage when an observed title is unsupported or absent from the
  model vocabulary;
- stratify by rule, player color, game length, opportunity-to-trigger, and
  available public metadata;
- compare against symbolic-only and frequency baselines;
- document distribution shift rather than combining synthetic and real-domain
  metrics into one headline.

This gate concerns post-game research evaluation. It does not authorize live
collection, screen reading, browser injection, or assistance during play.

### G7 — release provenance, deployment, and rollback

- Verify the release manifest from a clean checkout and isolated environment.
- Build the web application with the exact dependency lock.
- Record deployed asset hashes and ensure the artifact served is the approved
  canonical artifact.
- Run browser smoke tests for model load, analysis, cancellation, report
  download, truth isolation, and symbolic fallback.
- Keep the prior predictor available as an explicit rollback.
- Publish limitations and the exact promotion decision; never infer success
  from a green build alone.

## Authoritative missing items

| Missing item | Why it blocks promotion | Evidence required to close |
| --- | --- | --- |
| Current-catalog candidate checkpoint | Existing v1 covers 22 rules and is approximately random; v2.1 covers 27 and failed. | Multi-seed selected checkpoint(s) trained against the frozen current catalog. |
| Frozen current validation protocol | No approved thresholds bind a 182-rule successor. | Pre-evaluation committed protocol with baselines, thresholds, selection, and test-opening rule. |
| Passing validation report | No current candidate has passed. | Content-hashed per-seed and aggregate report satisfying every frozen gate. |
| Calibration artifact | Browser correctly reports “not calibrated.” | Validation-only fitted artifact plus before/after calibration report. |
| One-shot current test report | v2.1 test remained sealed; v1 test is irrelevant to a successor. | Test report produced once after validation approval. |
| Browser artifact for an approved model | Export/runtime compatibility exists only for v1. | Versioned artifact/runtime for the approved architecture, within shared bounds. |
| Permanent cross-runtime golden | Manual parity is not a CI guarantee. | Python-exported feature/logit fixture exercised by Python and browser tests. |
| Full release manifest | Current hashes are distributed across ignored local files and audit prose. | Single verifiable content-addressed manifest binding code, data, model, evaluation, calibration, and artifact. |
| Real-domain post-game evaluation | Synthetic accuracy does not prove usefulness on drawbackchess.com games. | Consented held-out completed-game benchmark with revealed truth and coverage accounting. |
| Complete observed-title semantics | Twelve of 194 observed titles remain unsupported. | Evidence-backed implementations and verification tests, or a permanently narrower claim. |

## Promotion state machine

```text
research
  -> protocol-frozen
  -> corpus-frozen
  -> candidates-trained
  -> validation-passed
  -> calibration-frozen
  -> test-opened-once
  -> runtime-parity-passed
  -> release-manifest-verified
  -> promoted
```

Any failed gate returns the candidate to `research`. Test data is not reopened
for iterative repair. An artifact that merely loads in the browser remains
`local experimental`. A promoted model that later fails provenance, secrecy,
runtime, or calibration checks is withdrawn and the UI rolls back to the
explicit symbolic-only predictor.

## Next authoritative work

1. Freeze a successor protocol for the current executable catalog, including
   symbolic-only and frequency baselines and a real-domain post-game benchmark.
2. Generate and hash a current uniform-evaluator corpus with isolated seeds.
3. Train the preregistered multi-seed sequence/hybrid candidates and evaluate
   validation only.
4. Stop if any gate fails. If all pass, freeze calibration and open the current
   test split exactly once.
5. Implement a new browser artifact version for the approved architecture,
   rather than forcing sequence/hybrid weights into v1.
6. Add permanent Python/browser golden parity and real-Worker release-artifact
   tests.
7. Assemble and independently verify the content-addressed release manifest.
8. Promote only the exact approved artifact, with post-game-only limitations
   and a symbolic rollback.

The present browser workflow is the correct delivery shell for that future
artifact. The remaining work is authoritative model evidence and release
provenance, not another way to bypass the gates.
