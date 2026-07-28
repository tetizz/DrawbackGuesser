# Capturable 25-label source-weighting research protocol

## Question

The balanced king-diagnostic supplement contains the same number of scheduled
games as the standard baseline corpus, but far fewer observed moves. Equal
player-game normalization therefore gave the supplement about half of the
classification loss mass. The earlier treatment improved a few targeted
rules and calibration while regressing Top-3, Top-5, later horizons, Black,
Checkers, and Barbarian Rage.

This bounded experiment asks whether retaining the diagnostic examples at a
smaller source weight preserves their useful signal without displacing broad
standard-game behavior.

## Frozen research inputs

Only the already-consulted pre-v4 training and validation artifacts may be
used:

- standard train: `capturable25-v1-train-schema8.ndjson`, source weight `1.0`;
- balanced diagnostic train:
  `capturable25-v3-balanced-train-schema8.ndjson`, source weight `0.1`;
- validation: `capturable25-v3-balanced-validation-schema8.ndjson`;
- seed `3235776259`;
- trigger-row multiplier `2`;
- 12 epochs, hidden dimension 128, batch size 256, and seven Torch threads.

The weight is applied after each player-game's rows have been normalized.
With equal scheduled player-game counts, weights `1.0` and `0.1` target a
diagnostic loss share of approximately 9.09%, rather than 50%.
Source multipliers are divided by their training-wide, player-game-weighted
mean before float32 conversion, while the denominator retains the existing
player-game normalization. This keeps relative source weights active in mixed,
source-homogeneous, and single-row minibatches and is invariant to multiplying
every source weight by the same positive constant. Ratios too extreme to remain
strictly positive and finite in the training tensor fail before training.
Weighted artifacts bind objective identifier
`global-source-mean-player-game/v1`.

## Command

```powershell
$trainingRoot = Resolve-Path '..\DrawbackTrainingData'

py -3.11 -B -E -s -m ml.training.drawback_ml.capturable_experiment select `
  --train "$trainingRoot\capturable25-v1-train-schema8.ndjson" `
  --train-source-weight 1.0 `
  --train "$trainingRoot\capturable25-v3-balanced-train-schema8.ndjson" `
  --train-source-weight 0.1 `
  --validation "$trainingRoot\capturable25-v3-balanced-validation-schema8.ndjson" `
  --output "$trainingRoot\capturable25-v3-weight010-s3235776259-t2-global-source-mean-v1" `
  --seed 3235776259 --epochs 12 --batch-size 256 `
  --hidden-dimension 128 --torch-threads 7 --trigger-row-multiplier 2
```

## Interpretation boundary

This run is diagnostic only. Its validation split has already been consulted,
so a favorable result cannot promote a model or authorize access to any
consumed test. It is useful only for deciding whether source weighting belongs
in the next preregistered multi-seed experiment.

The research result is favorable only if it improves the frozen control's
31.9432% game-normalized Top-1 without regressing Top-3, Top-5, any
5/10/15/20-move horizon, either color, or exact hard-elimination authority.
A later release attempt must repeat the intervention across paired seeds on
new independent validation folds, freeze the selected candidate, and then use
one newly generated disjoint sealed test exactly once.

## Invalidated preliminary run

The first local execution of this protocol normalized every minibatch by the
sum of its source-weighted rows. That canceled the `0.1` multiplier whenever a
minibatch contained only diagnostic rows, so the run did not implement this
protocol and its checkpoint and metrics are invalid. It cannot be compared,
promoted, or used for model selection. The experiment must be rerun with the
versioned objective above in the new no-clobber output directory. Preserve the
invalid artifacts for audit only:

- output directory:
  `capturable25-v3-weight010-s3235776259-t2-research`;
- invalid report SHA-256:
  `a02adb8e5bb94808fdadeb80473212cebdedb644e7b364f75506e9da7fccf0ea`;
- invalid checkpoint SHA-256:
  `24872b343c1bb362664e6dc3a91254a9bd09c09d6fa67d105e4c0322f1ed1af2`.

## Corrected validation result

The corrected run completed with the declared
`global-source-mean-player-game/v1` objective. It selected epoch 2 with alpha
`2.0` and smoothing `0.0`.

| Metric | Frozen control | Weighted treatment | Change |
| --- | ---: | ---: | ---: |
| Top-1 | 31.9432% | 32.6027% | +0.6594 percentage points |
| Top-3 | 52.3559% | 52.5830% | +0.2270 percentage points |
| Top-5 | 64.2221% | 63.8832% | -0.3389 percentage points |
| NLL | 3.153120 | 3.154055 | +0.000936 |
| Brier score | 0.910303 | 0.904664 | -0.005638 |
| ECE | 0.260714 | 0.250483 | -0.010230 |
| Top-1 after 5 moves | 11.76% | 13.20% | +1.44 percentage points |
| Top-1 after 10 moves | 26.32% | 27.12% | +0.80 percentage points |
| Top-1 after 15 moves | 36.24% | 37.28% | +1.04 percentage points |
| Top-1 after 20 moves | 44.96% | 44.88% | -0.08 percentage points |

Exact hard-elimination authority passed: the maximum probability assigned to
an eliminated hypothesis was zero, with zero violations.

The treatment **fails the preregistered reliability gate** because Top-5,
NLL, and the 20-move horizon regressed. It is not eligible for promotion, does
not replace the retained control, and does not change the reported fresh
held-out result. Opportunity-aware features and source weighting must be
retested on newly generated, independent validation folds before any sealed
evaluation.

Audit identities:

- corrected report SHA-256:
  `af7e999de6e7ea03c181a4b3a1e011b1eae76467342936d9b28f1e444e50f95a`;
- corrected checkpoint SHA-256:
  `6822996b6af888a41bd518c64335ec251022038cab324f27cd81e0bb63fa3318`.
