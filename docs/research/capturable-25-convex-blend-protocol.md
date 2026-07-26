# Capturable 25-label convex-blend protocol

## Objective

This experiment tests whether the selected balanced-diagnostics model can
improve calibration without replacing the stronger ranking behavior of the
retained control. It is a deterministic post-training ensemble experiment:
neither checkpoint is retrained or modified.

The previous balanced treatment remains rejected on its own. It improved
validation Top-1, negative log likelihood, Brier score, calibration, and
trigger accuracy, but regressed Top-3, Top-5, three move horizons, and several
drawback-level results. No fresh test was generated or opened.

## Frozen inputs

All generated data and checkpoints remain outside Git under the private
training-data directory. The experiment binds these exact bytes:

| Role | Private artifact | SHA-256 |
| --- | --- | --- |
| control selection | `capturable25-v3-control-s3235776259-t2/selection.json` | `889e8a22f812f6359b7aa36d66436f077bbbbd80fbe7ef6ac393533eb47fbf06` |
| control checkpoint | `capturable25-v3-control-s3235776259-t2/model.pt` | `b314ae8e0c020490363237a025c8f6291dc24073542e00c17091a87df9c65469` |
| treatment selection | `capturable25-v3-balanced-s3235776257-t1/selection.json` | `3a6e66e46002d037e4a1e599b7accf7d9ede753b155a8d12fc3cd1ad3086ce57` |
| treatment checkpoint | `capturable25-v3-balanced-s3235776257-t1/model.pt` | `8f821461304bdc03786aea354676c0dd8fccadaecbad89496d0322dc095c424f` |
| selection validation | `capturable25-v3-balanced-validation-schema8.ndjson` | `09d5d9a4991d76e9fb564ac1fbc64c10212f80b75b0e28fbb9f74b4a93bbaf3d` |
| prior comparison | `capturable25-v3-balanced-treatment-comparison.json` | `2535b9916f75a5eff075570295911e738108a3a92ddcf4ab2e209fe65a096a3d` |

The validation corpus has 30,352 rows from 625 games. It contains every
ordered pair of the 25 audited capturable drawbacks exactly once, with 25
player-games in every label/color cell. Both checkpoints were selected on
this validation corpus, so this experiment is a bounded secondary selection
and requires a genuinely fresh confirmation set before promotion.

The implementation must authenticate each selection report against its
checkpoint, verify every recorded hash above, require the same rule
vocabulary and feature dimension, require the same validation identity, and
reject changed, overlapping, reordered, missing, or extra inputs.

## Frozen blend

For every row, each checkpoint first produces its own published posterior
using its selected symbolic-fusion alpha and prior-smoothing value. The
candidate posterior is then:

```text
P_blend = (1 - w) * P_control + w * P_treatment
```

This is an arithmetic mixture of final probabilities, not a mixture of raw
logits. The same weight is applied independently to:

- the 25-label drawback posterior;
- the trigger probability;
- the forced-move probability;
- the two-value Triple Play parameter posterior.

Both component drawback posteriors must expose exactly the same hard
elimination mask. An eliminated hypothesis must have probability exactly
zero in both components and in the blend. A mismatch aborts the experiment;
the implementation may not intersect or otherwise weaken the masks.

The frozen treatment-weight grid is:

```text
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
```

Weight `0.0` is evaluated once as the authenticated control comparator but is
not a blend candidate. The winning nonzero weight is selected by:

1. highest validation game-normalized Top-1;
2. highest validation game-normalized Top-3;
3. lowest validation game-normalized negative log likelihood;
4. lower treatment weight.

All model inference is CPU-only and deterministic. Row order, vocabulary
order, dataset bytes, checkpoints, component posterior values, and candidate
metrics are content-addressed in a canonical no-clobber comparison artifact.

## Validation release gate

The selected blend may reserve the fresh test only if every check passes:

1. its Top-1/Top-3/NLL selection tuple is strictly better than control;
2. game-normalized Top-1, Top-3, and Top-5 do not regress;
3. game-normalized NLL and Brier score do not regress;
4. expected calibration error does not regress;
5. accuracy at 5, 10, 15, and 20 observed moves does not regress;
6. White and Black Top-1, Top-3, and NLL each do not regress;
7. trigger and forced accuracy, NLL, and Brier each do not regress;
8. Triple Play hidden-parameter accuracy does not regress;
9. no audited drawback loses more than one absolute Top-1 percentage point;
10. exact symbolic authority has zero missing masks, zero violations, and
    exactly zero probability on every eliminated hypothesis.

Failure retains the control and consumes no new test.

## Reserved fresh confirmation

Only after a committed passing validation artifact may the previously
unconsumed balanced-protocol test schedule be generated:

- split counts: 0 train / 0 validation / 625 test;
- label seed: 633442320;
- gameplay seed: 633446417;
- parameter seed: 633450514;
- profile: `standard`;
- workers/window: 15 / 30;
- maximum plies: 60;
- search depth/node budget: 1 / 5,000;
- temperature/top-K: 35 centipawns / 8;
- evaluator: material v1;
- opponent policy/aggregation: unrestricted baseline / worst case.

The schedule must contain all 625 ordered label pairs once and 25
player-games in every label/color cell. Exact replay, schema-8 loading,
true-hypothesis survival, row-bearing accounting, and disjointness from all
earlier corpora are mandatory.

The paired sealed evaluator must authenticate the committed validation
decision before reading the fresh dataset. It evaluates only control and the
single frozen blend weight, applies the same complete release gate, and
cannot retune the weight. A failed confirmation retains control permanently
for this experiment.

## Claim boundary

Passing both stages would establish an improvement only on deterministic
synthetic self-play over the current 25-label capturable catalog. It would
not establish reliability on human games, unsupported drawbacks, or external
live play. The public application remains an offline completed-game training
and research tool.
