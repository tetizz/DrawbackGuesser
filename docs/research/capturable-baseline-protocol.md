# Capturable baseline protocol

## Scope

This protocol freezes the first hybrid neural baseline for the 10 audited
`capturable-king/v1` drawbacks. It is a research result, not a promoted browser
model and not evidence for unsupported catalog rules.

The source authority is DrawbackEngine revision
`6994b727a4425b72423525cfe28767d4567c593d`. Every source trace and converted
dataset row must pass exact semantic replay through that revision.

## Frozen training decision

The final model is selected without access to the final test corpus:

- training games: 300;
- converted training rows: 14,960;
- training dataset SHA-256:
  `1728657bd97bf70b013090c2be21e47c7820133e975501ba58e7f3d5c539f17d`;
- validation games: 60;
- converted validation rows: 3,040;
- validation dataset SHA-256:
  `7dd87658bdbf2a83fd8a723296049e9bdbf954294e61b5b424593fb1accdb657`;
- model seed: `3235776257`;
- epochs: 8;
- batch size: 256;
- hidden dimension: 128;
- public feature dimension: 892;
- player-game inverse-frequency loss weighting;
- training prior smoothing: 0.10;
- validation fusion alpha grid: 0, 0.25, 0.5, 1, 2;
- validation prior-smoothing grid: 0, 0.01, 0.05, 0.10, 0.20.

Validation selects by game-normalized Top-1, then game-normalized Top-3, then
game-normalized NLL. Remaining ties prefer the smaller fusion alpha and smaller
prior smoothing. This selected epoch 2, fusion alpha 2, and smoothing 0.
Repeated training produced bit-identical tensors.

Two additional model seeds, a 700-game supplemental training source,
128/256-unit 1,000-game models, and all tested ensembles were rejected using
validation only. Their exploratory test reports are not final evidence.

## Unopened final test

The final test is fixed at 100 newly simulated games with:

- label seed root: `3562944537`;
- gameplay seed root: `666289313`;
- parameter seed root: `2399397741`;
- maximum plies: 60;
- search depth: 1;
- maximum nodes: 5,000;
- temperature: 35 centipawns;
- top K: 8.

These roots are disjoint from all training, validation, and exploratory-test
roots. The final test may be converted and evaluated only after the source,
features, model configuration, selected epoch, and fusion settings above are
frozen. No result from it may change those choices.

## Required report

The final report must include row and player-game counts, Top-1/3/5, NLL,
Brier score, expected calibration error, accuracy after 5/10/15/20 observed
moves, per-rule results, parameter accuracy, and hard-elimination diagnostics.
No accuracy value may be reported unless it is read from the content-addressed
evaluation artifact.

The completed held-out evaluation is documented in
`capturable-baseline-results.md`.
