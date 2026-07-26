# Current-catalog model validation protocol

Status: **preregistered and frozen before candidate evaluation**

Protocol version: `current-catalog-182-v2`

Freeze date: 2026-07-24

Repository base: `b7d1c7b3b2d95f7a83afde519b7c37be9f66372d`

## Decision governed by this protocol

This protocol decides whether one current-catalog neural ensemble may advance
from research to a one-time sealed-test evaluation and, if that test passes,
to a post-game browser release review.

It does not authorize:

- a live-game assistant;
- observation, scraping, or injection on drawbackchess.com or another site;
- use of future moves, revealed drawbacks, hidden parameters, authoritative
  rule state, or drawback-filtered moves as inference inputs;
- replacement of executable symbolic hard elimination by a model prediction;
- support claims for the 12 observed titles whose rules remain unsupported.

The intended release accepts a user-supplied completed or partial PGN after or
outside play and independently guesses White's and Black's drawbacks. Truth
may be added after analysis for scoring. It never feeds back into inference.

No candidate result, ignored generated checkpoint, or future test aggregate
may be inspected to revise this document. A change to any frozen item below
creates protocol version 2 and requires new corpus, candidates, validation,
and sealed test artifacts.

## Frozen rule universe

The canonical model class order is the exact 182-entry tuple
`SYMBOLIC_RULE_IDS` from
`ml/training/drawback_ml/symbolic_schema.py` at this protocol's repository
base. It must be byte-for-byte equal to the predictor's current default
hypothesis rule IDs. The following catalogs must contain the same unique ID
set, but their selection or presentation order is not a model-head contract:

- `PREPARED_EXECUTABLE_RULE_IDS` in
  `packages/simulation/src/prepared-catalog.ts`;
- the non-unsupported IDs in `data/catalog/observed-drawbacks.json`.

The corpus manifest must record both the unique prepared sampling order in
`ruleIds` and the ordered model-head contract in `symbolicRuleIds`, plus the
SHA-256 of each canonical UTF-8 JSON array. Duplicate, missing, or additional
IDs invalidate either field. Reordering `symbolicRuleIds` invalidates the run;
`ruleIds` may use its frozen prepared-catalog order.

The release universe has two declared views:

1. **Prepared evaluation view — 182 rules.** Every eligible position contains
   a public evaluator fact under one pinned policy, so `Hand and Gigabrain`
   and `Ichtyophobe` are evaluable.
2. **Standard-PGN browser view — 180 rules.** Without a uniform evaluator
   stream, those two rules are unavailable. They receive no probability and
   are not treated as eliminated or unrestricted.

All aggregate prepared-view metrics use 182 classes. Browser-view metrics use
the fixed 180-rule intersection and report the two exclusions. Neither view
includes the 12 unsupported observed titles.

Uniform 182-class reference values are frozen as:

| Reference | Value |
| --- | ---: |
| Top-1 | `1 / 182 = 0.0054945055` |
| Top-3 | `3 / 182 = 0.0164835165` |
| Top-5 | `5 / 182 = 0.0274725275` |
| NLL | `ln(182) = 5.2040066871` |
| Brier | `181 / 182 = 0.9945054945` |

## Frozen corpus

### Generation

The general corpus is generated once with:

| Field | Frozen value |
| --- | --- |
| Generator | `@drawbacktrainer/simulation` evaluator corpus schema 6 |
| Root seed | `20260810` |
| Train games | `18,200` |
| Validation games | `9,100` |
| Sealed test games | `9,100` |
| Maximum plies | `80` |
| Rule IDs | all 182 canonical IDs |
| Agents | random legal, greedy material, weak human-like, medium human-like, strong human-like |
| Symbolic feature version | `6` |
| Evaluator coverage | `uniform-required` |
| Evaluator request schema | `1` |
| Evaluator cache schema | `1` |
| Rule assignment policy | `balanced-symmetric-v1` |
| Evaluator policy ID | `stockfish-bestmove-v1` |
| Evaluator policy version | `1` |
| Search limit | exactly `10,000` nodes |
| Threads | `1` |
| Ponder | `false` |
| Hash | `16` MiB |
| Reset | `ucinewgame`, Clear Hash, `isready` before each uncached constraint |

The exact UCI name, engine version, executable SHA-256, option digest, source
commit, lockfile hash, worker count, and generated-file hashes are filled into
the corpus manifest before training. They are provenance values, not tunable
variables. Failure to acquire or verify them stops generation.

Worker count may change throughput only. Generation is accepted only if runs
with one worker and the production worker count produce byte-identical
fixtures for a fixed 20-seed audit subset.

The balanced scheduler assigns every rule exactly 100 times as White and 100
times as Black in train, and exactly 50 times per color in each held-out split.
Signed cyclic offsets pair every scheduled orientation with its color-reversed
orientation, exclude self-pairs, and balance each of the five agents
conditionally on every rule and color (20 appearances per agent in train and
10 in each held-out split). Rule and agent permutations are derived from the
root seed and split domain before play; worker completion order cannot change
labels.

Every scheduled assignment is represented exactly once in the authenticated
outcome ledger. A game may contain moves from both colors, one color, or no
moves when a legal start-of-turn drawback loss ends it before observation.
Only a color that has made at least one publicly observed move contributes
classification or parameter supervision. Zero-ply and one-sided outcomes
remain in coverage and attrition reporting and are never silently replaced,
because replacing them would change the drawback distribution. The model must
not receive the unobserved color's label through batching, loss construction,
tokenizer fitting, calibration, or evaluation.

### Hard-negative training supplement

Only the training split receives a supplement. The six existing named profiles
are each generated through an evaluator-enabled hard-negative path with the
same pinned evaluator policy and uniform public fact coverage as the general
corpus. That path does not exist at the frozen base and must be implemented and
tested before generation; using the current synchronous hard-negative output
would invalidate the run. Each profile uses 1,000 training games, a root seed
formed by adding the following offset to `20260810`, and maximum 80 plies:

| Profile | Offset |
| --- | ---: |
| Checkers / Pacman | `101` |
| Truant / Spice of Life | `102` |
| Oddball / Even Keeled | `103` |
| Quit Horsing Around / Forward March | `104` |
| Horse Tranquilizer / Conscientious Objectors | `105` |
| Gambler / Truant | `106` |

No validation or test hard-negative row is appended. The general validation
and test sets remain representative of their frozen generator. Unsupported
counterfactual profiles are not approximated with mislabeled data. Every
supplement row must use symbolic feature version 6 and contain the same public
evaluator fields and policy identity as the general training rows; field
presence may not reveal that a row came from a profile.

### Split and validation sub-splits

Complete game seeds are the indivisible unit. The existing
`BLAKE2b-64(drawbacktrainer-v1:gameSeed)` split assignment must place each game
in exactly one general split.

Validation is deterministically subdivided by
`BLAKE2b-64(current-catalog-182-v2:validation:gameSeed)`:

- values in `[0.00, 0.70)` are **selection**;
- values in `[0.70, 0.85)` are **calibration-fit**;
- values in `[0.85, 1.00)` are **validation-gate**.

These are disjoint by complete game seed. Epoch selection and architecture
diagnostics use selection only. Temperature fitting uses calibration-fit only.
Every promotion gate is evaluated on validation-gate only. The sealed test is
not read by corpus audits, tokenizer fitting, parameter-vocabulary fitting,
epoch selection, calibration, threshold selection, or error analysis.

## Frozen candidate

### Architecture

The candidate is a three-member ensemble of the existing `v21-hybrid`
architecture, with these exact per-member settings:

| Setting | Value |
| --- | ---: |
| Public board feature schema | `1`, 792 values |
| Symbolic feature schema | `6`, ordered 182-rule probabilities and masks |
| Model variant | `v21-hybrid` |
| Numeric precision | FP32 |
| Hidden dimension | `256` |
| SAN embedding dimension | `32` |
| GRU sequence hidden dimension | `128` |
| Symbolic hidden dimension | `128` |
| Maximum SAN history | `80` |
| Epochs | `8` |
| Batch size | `1024` |
| Examples sampled per game per epoch | `16` |
| Streaming shuffle buffer | `16,384` |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Classification loss weights | White `1.0`, Black `1.0` |
| Parameter loss weights | White `0.1`, Black `0.1` |
| Trigger loss weight | `0.1` |
| Legal-mask loss weight | `0.05` |
| Legal-mask objective | balanced positive/negative BCE |

Training seeds are exactly:

```text
20260811
20260812
20260813
```

The SAN tokenizer and parameter vocabulary are fit from the general and
hard-negative training rows only. Their ordered values and hashes are stored
in every checkpoint. No validation or test token may expand a head; unseen
values are unscorable and counted.

This protocol evaluates the implementation as it exists at the frozen base.
Changing current-move encoding, per-color supervision grain, loss composition,
network depth, or batching is a new architecture and requires protocol v2.

### Epoch selection

For each training seed independently:

1. Evaluate epochs 1 through 8 on validation-selection.
2. Compute White NLL and Black NLL after the model's symbolic prior and hard
   mask, before calibration.
3. Select the epoch with minimum arithmetic mean of the two NLLs.
4. If two means differ by at most `0.005`, select the earlier epoch.

All five reports remain in the release evidence. The selected epoch cannot be
changed using calibration-fit, validation-gate, browser-view, real-domain, or
test results.

### Ensemble equation

For each color and class, take the arithmetic mean of the three selected
members' raw neural residual logits. Add that mean to the exact symbolic log
prior for the same color and class. Apply the executable symbolic elimination
mask as negative infinity after neural scoring. Softmax covers surviving
classes only.

No member weighting, seed dropping, per-rule weighting, or stacking model is
allowed. If one member fails to load or infer, the ensemble fails and is
reported unavailable; it does not silently run with two members.

## Frozen comparators

Every metric table includes these systems over the identical game seeds,
plies, class universe, and truth labels:

1. **Uniform.** Equal mass over all available non-eliminated classes.
2. **Training frequency.** Per-color player-game class frequencies from the
   general plus hard-negative training set, renormalized after hard masks.
3. **Symbolic only.** Exact current predictor posterior and elimination mask,
   with no neural residual.
4. **Single-seed members.** Each selected member under the same symbolic prior,
   reported for robustness but not eligible alone.
5. **Uncalibrated ensemble.** The frozen three-member equation.
6. **Calibrated ensemble.** The sole promotion candidate.

Comparators do not receive post-game truth, model labels, assigned parameters,
or drawback-legal move sets.

## Frozen calibration

Fit two scalar temperatures, one for the White head and one for the Black
head, using calibration-fit examples only. Each temperature minimizes
multiclass NLL over surviving logits with:

- search interval `[0.05, 10.0]`;
- deterministic optimizer/tie handling from
  `ml/evaluation/calibration.py`;
- exact symbolic hard masks reapplied after scaling;
- one shared White temperature across all White rules and horizons;
- one shared Black temperature across all Black rules and horizons.

No rule-specific, family-specific, horizon-specific, seed-specific, or
real-domain temperature is permitted. The calibration artifact records both
temperatures, optimizer version, calibration-fit seed hash, uncalibrated
ensemble hash, and symbolic schema hash.

Calibration is accepted only if it lowers mean White/Black NLL on
calibration-fit and does not assign positive probability to an eliminated
class. Promotion gates use the frozen temperatures on validation-gate.

## Metrics and decision units

Classification metrics are computed independently for White and Black at every
eligible observed ply, then also macro-averaged by complete player-game so long
games do not dominate the headline.

Required reports:

- Top-1, Top-3, Top-5;
- NLL, multiclass Brier, and 15-bin expected calibration error;
- Top-1 and Top-3 after exactly 5, 10, 15, and 20 observed plies, using the
  last available prefix at or before the horizon;
- mean and median observed ply when truth first reaches rank one;
- per-rule and per-family Top-1, Top-3, NLL, Brier, support, trigger
  opportunity, and unscorable count;
- hidden-parameter whole-object accuracy, component accuracy, coverage, and
  unscorable rate;
- entropy reduction per observed move;
- hard-elimination violations;
- confusion matrices for both colors;
- metrics by agent profile and evaluator-backed versus synchronous family.

Paired differences use 10,000 deterministic bootstrap replicates of complete
game seeds with bootstrap seed `20260814`. White and Black trajectories for a
game remain together. Intervals are percentile 95% confidence intervals.
Intervals support non-regression gates; they cannot rescue a point estimate
that misses an absolute gate.

## Validation-gate thresholds

Every hard gate must pass. “Both colors” means White and Black separately, not
their mean.

### Exactness and coverage

1. Hard-elimination violations: exactly `0`.
2. Probability normalization error: at most `1e-6` per distribution.
3. Unknown/duplicate class IDs, wrong symbolic version, non-finite outputs, or
   cross-color head swaps: exactly `0`.
4. Truth class scorable for at least `99.5%` of player-games and `99.5%` of
   move examples. Every excluded example is enumerated by reason.
5. All 182 prepared rules meet the frozen split support minima.

### Prepared 182-rule quality

For the calibrated ensemble, both colors must meet:

| Metric | Absolute gate |
| --- | ---: |
| Macro player-game Top-1 | at least `0.10` |
| Macro player-game Top-3 | at least `0.25` |
| Macro player-game Top-5 | at least `0.35` |
| Move-example NLL | at most `4.50` |
| Move-example Brier | at most `0.970` |
| 15-bin ECE | at most `0.080` |
| Top-1 at 10 plies | at least `0.040` |
| Top-1 at 15 plies | at least `0.070` |
| Top-1 at 20 plies | at least `0.100` |
| Top-3 at 20 plies | at least `0.250` |

Additionally:

- macro per-rule Top-3 is at least `0.18` for both colors;
- at least 80% of the 182 rules have per-rule Top-5 above the uniform Top-5
  reference for both colors;
- no rule with at least 25 supported player-games has Top-5 equal to zero;
- parameter whole-object accuracy is at least `0.20` on scorable parameterized
  examples, with at least `0.95` parameter coverage;
- each individual selected seed has mean White/Black Top-3 at least `0.18` and
  mean White/Black NLL at most `4.90`.

### Comparator gates

For both colors, the calibrated ensemble must:

- exceed training-frequency Top-1 by at least `0.05`;
- exceed training-frequency Top-3 by at least `0.10`;
- have NLL at least `0.20` lower than training frequency;
- have Brier at least `0.010` lower than training frequency;
- have Top-1 no more than `0.02` below symbolic only;
- have Top-3 no more than `0.03` below symbolic only;
- have NLL at least `0.15` lower than symbolic only;
- have Brier at least `0.010` lower than symbolic only;
- have a 95% paired interval whose lower bound is above `-0.02` for Top-1
  difference from symbolic only;
- have a 95% paired interval whose upper bound is below `0` for NLL difference
  from symbolic only.

Calibration must not change Top-k rankings. Calibrated ECE must be no worse
than uncalibrated ECE for either color, and calibrated NLL must be lower for
both colors on validation-gate.

### Standard-PGN 180-rule browser view

After excluding `Hand and Gigabrain` and `Ichtyophobe` and renormalizing every
system over the same 180 classes, both colors must meet:

- Top-1 at least `0.10`;
- Top-3 at least `0.25`;
- Top-5 at least `0.35`;
- NLL at most `4.50`;
- Brier at most `0.970`;
- ECE at most `0.080`;
- the same symbolic-only non-regression gates as the prepared view.

The two unavailable rules remain visible in coverage metadata with rank
`null`. Their removal may not be counted as a correct elimination.

## Test-opening rule

The sealed test split may be opened exactly once only after a committed
validation approval record proves all of the following:

1. Every absolute, comparator, calibration, exactness, coverage, individual
   seed, prepared-view, and browser-view validation gate passed.
2. Corpus manifest, all split files, training supplement, selected
   checkpoints, tokenizers, calibration artifact, comparator outputs,
   validation reports, source revision, dependency locks, evaluator binary,
   and evaluator options have recorded SHA-256 hashes.
3. A clean independent process reconstructs all three members and reproduces
   fixed validation-gate logits and metrics within `1e-6`.
4. The release-candidate browser artifact contract is frozen and rejects wrong
   versions, tensors, vocabularies, symbolic schemas, and calibration.
5. The exact test command, report schema, bootstrap seed, output path, and
   expected input hashes are committed before execution.
6. Two reviewers sign the approval record. Neither may alter a threshold or
   candidate after seeing validation-gate results.

The one test execution evaluates the selected ensemble and all frozen
comparators. No test-derived retraining, epoch change, temperature change,
threshold change, class exclusion, feature change, or artifact change is
allowed.

Test promotion requires every validation absolute threshold again on test,
every hard exactness gate, and these stability gates for both colors:

- test Top-1 is no more than `0.03` below validation-gate Top-1;
- test Top-3 is no more than `0.04` below validation-gate Top-3;
- test NLL is no more than `0.20` above validation-gate NLL;
- test Brier is no more than `0.020` above validation-gate Brier;
- test ECE is no more than `0.030` above validation-gate ECE.

Any failure keeps the model unpromoted. The test split is not reopened for the
same protocol. A successor starts with new isolated seeds and protocol version.

## Browser parity and artifact gates

The approved architecture requires a new browser artifact version; it must not
be forced into feed-forward artifact version 1. Before test opening:

1. The artifact records all three source checkpoint hashes, selected epochs,
   ordered 182-rule vocabulary, tokenizer, feature schema 1, symbolic schema 6,
   ensemble equation, calibration temperatures, and fusion version.
2. Canonical artifact SHA-256 and byte size are recorded. Export and UI enforce
   the same maximum size and dimension limits.
3. Python and TypeScript use committed golden fixtures for:
   - board and scalar features;
   - current observed move;
   - SAN prefix tokens and lengths;
   - symbolic probabilities and hard masks;
   - member residual logits;
   - ensemble logits;
   - calibrated prepared-view and browser-view probabilities.
4. Maximum Python/browser absolute difference is at most `1e-6`; Top-k order
   and hard-zero sets are identical.
5. A production Vite build runs the exact artifact inside a real Worker for
   PGNs covering both colors, promotion, castling, en passant, custom FEN,
   cancellation, supersession, malformed input, and maximum supported length.
6. An eliminated hypothesis remains exactly zero after every member, ensemble,
   calibration, report serialization/reload, and truth selection.
7. White and Black heads and symbolic masks remain isolated.
8. Missing or partial evaluator facts never make the two evaluator rules
   appear available.
9. Artifact or inference failure is visible and produces a separately
   identified symbolic-only report; it never retains a hybrid label.
10. The report binds model artifact SHA-256, three checkpoint hashes,
    calibration hash, ordered vocabulary hash, symbolic schema, fusion weight,
    evaluator policy or explicit absence, and analytical digest.

## Real-domain post-game gate

Synthetic test passage is not sufficient for a claim about completed
drawbackchess.com games.

Before public promotion, evaluate once on a consented, legally acquired
post-game benchmark that was not used for training, protocol design,
calibration, or threshold selection. It must contain:

- at least 2,000 completed games;
- revealed White and Black drawback truth;
- at least 10 player-games for every claimed standard-PGN rule;
- no live capture or data collected before the corresponding game ended;
- public PGN/move history only at inference;
- a manifest of consent/license, deduplication, source date range, rule mapping,
  and unsupported/unavailable outcomes.

The promoted claim is limited to the benchmark's represented standard-PGN
rules. Both colors must meet:

- Top-1 at least `0.08`;
- Top-3 at least `0.20`;
- Top-5 at least `0.30`;
- NLL below the real-domain training-frequency comparator by at least `0.10`;
- Brier below that comparator by at least `0.005`;
- ECE at most `0.10`;
- Top-3 no more than `0.04` below symbolic only;
- hard-elimination violations exactly `0`.

Synthetic and real-domain results remain separate tables. Missing rules,
unsupported titles, ambiguous title mappings, and evaluator-dependent rules
without uniform facts receive rank `null` and count against coverage, not
accuracy.

## Release boundary

Passing validation and test permits a release review, not automatic deployment.
A promoted package must bind:

- source revision and clean-tree attestation;
- corpus, split, supplement, and real-domain manifest hashes;
- all three checkpoint and tokenizer hashes;
- validation, calibration, test, parity, and real-domain report hashes;
- browser artifact and calibration hashes;
- rule-universe, feature, symbolic, ensemble, and fusion versions;
- dependency and evaluator binary/option hashes;
- model license, data-use statement, limitations, and rollback artifact.

The web UI must:

- default to the post-game offline workflow;
- identify symbolic-only versus promoted hybrid results;
- show calibrated status, represented count, unavailable rules, and artifact
  identity;
- preserve the hot-seat handoff curtain in local games;
- enable truth only after inference;
- never connect to or monitor an external live game;
- retain symbolic-only as an explicit rollback.

If no candidate passes, the correct release remains symbolic-only with optional
clearly labeled unpromoted local artifacts. Gates are not lowered to ensure a
neural release.

## Frozen command order

The exact CLI spelling may be implemented after this protocol, but the
semantic order is frozen:

```text
generate general corpus and training-only hard negatives
hash and audit corpus without model access
train seeds 20260811, 20260812, 20260813 for eight epochs
select one epoch per seed on validation-selection
form the unweighted three-member ensemble
fit White and Black temperatures on calibration-fit
evaluate all frozen systems once on validation-gate
stop unless every validation gate passes
freeze browser artifact and parity evidence
commit two-reviewer test-opening record and exact test command
evaluate sealed synthetic test once
stop unless every test and stability gate passes
evaluate consented post-game real-domain benchmark once
assemble content-addressed release package
conduct independent release review
```

This protocol is intentionally capable of producing no promoted model.
