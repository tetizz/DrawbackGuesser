# Capturable diagnostic augmentation protocol

## Scope

This protocol freezes the second 10-rule `capturable-king/v1` hybrid
experiment. It tests whether label-independent diagnostic starting positions
and trigger-aware training improve ordinary-start drawback identification.
It is not a browser-model promotion or evidence about live human games.

The Engine authority is DrawbackEngine revision
`1406224aadde35b45efd777eda83693e4afced38`. Diagnostic scenario choice comes
only from the gameplay seed domain and cannot depend on either hidden drawback
or hidden parameter seed.

## Training sources

The ordinary source is unchanged from the first baseline:

- 300 games and 14,960 rows;
- dataset SHA-256
  `1728657bd97bf70b013090c2be21e47c7820133e975501ba58e7f3d5c539f17d`.

The selected diagnostic supplement uses:

- profile: `king-capture-diagnostics-v1`;
- 250 games and 549 rows;
- label seed root: `324508639`;
- gameplay seed root: `610839776`;
- parameter seed root: `253635900`;
- maximum plies: 4;
- search depth: 1;
- maximum nodes: 5,000;
- temperature: 35 centipawns;
- top K: 8;
- trace SHA-256
  `e9c977dc94faed07856efa5784c385441d9bcecaa38e80db531d3c6592c948de`;
- dataset SHA-256
  `9c748c416cc15bfd75acb0cc3d50d30cbabe9295d9e9f079003d40a971c6a93f`.

The supplement contains only the five audited king-capture drawbacks. It is
combined with, never substituted for, the balanced ordinary source.

## Validation and selected model

Selection uses a new ordinary-start validation corpus:

- 100 games and 5,012 rows;
- label seed root: `2882400001`;
- gameplay seed root: `271136839`;
- parameter seed root: `1447508009`;
- maximum plies: 60;
- search depth: 1;
- maximum nodes: 5,000;
- temperature: 35 centipawns;
- top K: 8;
- trace SHA-256
  `86acc9876e740ca9f0e886d008be382fe94cd99e3ccb183ddde8f1fa08f01e17`;
- dataset SHA-256
  `45131b663220279f24cf8def2749447b43e5023707d8eb9c74be350dbc0e2951`.

The frozen model settings are:

- model seed: `3235776257`;
- epochs: 8;
- batch size: 256;
- hidden dimension: 128;
- feature dimension: 892;
- trigger-row multiplier: 2;
- training prior smoothing: 0.10;
- fusion alpha grid: 0, 0.25, 0.5, 1, 2;
- validation prior-smoothing grid: 0, 0.01, 0.05, 0.10, 0.20.

Trigger weighting reallocates mass only among rows from the same player-game.
Every player-game retains total loss weight one, and the trigger label is not
an inference input.

Validation selects game-normalized Top-1, then game-normalized Top-3, then
game-normalized negative log likelihood. Remaining ties prefer smaller fusion
alpha and smaller prior smoothing. The selected model uses epoch 3, fusion
alpha 2, and smoothing 0.

On validation, the selected candidate measured:

- game-normalized Top-1: 40.2284%;
- game-normalized Top-3: 67.9792%;
- game-normalized Top-5: 84.7445%;
- game-normalized negative log likelihood: 2.6782%.

The unchanged 300-game control measured 39.6075% Top-1 and 66.4033% Top-3 on
the same validation corpus. The selected candidate therefore wins the frozen
primary metric, but its calibration is worse and the gain is small.

Repeated training produced byte-identical checkpoint and report files:

- checkpoint SHA-256
  `022b68e92f7b802f3de98df698c896773ee543cd5bc8c4e7dc876d839629ca88`;
- exploratory report SHA-256
  `657435bd3fedbc32f130e103cf3ad575cc5f86eb608947ee8e4cf37e727ca318`.

The report's old 60-game test partition is exploratory only and is not final
evidence for this protocol.

## Rejected candidates

Validation rejected:

- 100-, 250-, and 500-game diagnostic doses without trigger weighting;
- 100- and 500-game doses with a 2x trigger multiplier;
- 250 games with 1.5x and 4x trigger multipliers;
- class-balanced loss weighting;
- residual ensembles with target weights 0.25, 0.50, and 0.75.

These choices may not be revisited after the final test is opened.

## Unopened final test

The final test is fixed at 150 newly simulated ordinary-start games:

- label seed root: `3405691582`;
- gameplay seed root: `3735928559`;
- parameter seed root: `2343432205`;
- maximum plies: 60;
- search depth: 1;
- maximum nodes: 5,000;
- temperature: 35 centipawns;
- top K: 8.

The final trace may be generated, converted, and evaluated only after this
protocol, Engine pin, feature implementation, trigger weighting, and tests are
published. The final result cannot change the model, epoch, fusion settings,
training sources, or claim boundary.

## Required report

The final report must include source hashes, row and player-game counts,
Top-1/3/5, negative log likelihood, Brier score, expected calibration error,
accuracy after 5/10/15/20 observed moves, per-rule accuracy, hidden-parameter
accuracy, trigger and forced-move accuracy, and hard-elimination diagnostics.
No value may be reported unless it is read from the content-addressed final
evaluation artifact.

The completed, rejected result is recorded in
`capturable-diagnostic-augmentation-results.md`.
