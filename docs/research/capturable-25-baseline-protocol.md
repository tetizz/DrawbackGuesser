# Twenty-five-label capturable baseline protocol

## Scope

This protocol freezes the first schema-8 experiment over all 25 drawbacks
audited for `capturable-king/v1`. It measures synthetic player-private
self-play only. It is not a claim about live human games and it does not
promote a browser model.

DrawbackEngine revision
`62f5a3ed3343e6472587b1e7b14893704b0c455f` owns move legality, the balanced
assignment schedule, private search, and trace replay. DrawbackGuesser
revision is frozen by the eventual selection artifact.

## Frozen corpus

All three splits use one schedule with:

- train games: 1,250;
- validation games: 625;
- sealed test games: 625;
- label seed root: `633380865`;
- gameplay seed root: `633384962`;
- parameter seed root: `633389059`;
- workers: 15;
- maximum plies: 60;
- bounded scheduling window: 30;
- search depth: 1;
- maximum search nodes: 5,000;
- temperature: 35 centipawns;
- top K: 8;
- evaluator: material version 1;
- training profile: `standard`;
- opponent model: unrestricted baseline;
- opponent aggregation: worst case.

The 25-by-25 ordered-pair scheduler completes two cycles in train and one
cycle in each held-out split. Each drawback therefore occurs exactly 50 times
per color in train and 25 times per color in validation and test before
terminal zero-ply accounting. Split game indexes and gameplay seeds are
disjoint.

The source traces and converted datasets remain outside Git. Their byte
lengths, SHA-256 digests, observed row counts, and replay results are recorded
after generation; an incomplete or replay-invalid split aborts the experiment.

## Measured throughput decision

On the generation machine, 50 depth-1 games capped at 20 plies completed in
14.769 seconds. A 20-game, 10-ply depth-2 sample completed in 44.802 seconds.
Depth 2 was therefore about 15 times slower per generated ply. An initial run
using 60-assignment execution windows reported an opaque worker failure. A
30-assignment diagnostic rerun localized it to game 684: a non-orthodox
capturable-king position retained a pseudo-legal en-passant target internally
but dropped it from the public FEN reconstructed by search. Engine revision
`62f5a3ed3343e6472587b1e7b14893704b0c455f` preserves the target and includes
both position-level and exact 39-ply corpus regressions. Production generation
uses 30-assignment windows so any future failure has a smaller retry and audit
scope without changing deterministic assignments or game bytes. The balanced
bulk corpus uses depth 1 for coverage and reproducibility. Deeper,
drawback-aware games may later form a separately labeled training supplement,
but cannot be mixed into this baseline after the protocol is frozen.

## Frozen model candidates

Every candidate starts from a new deterministic random initialization. The
candidate grid is:

- model seeds: `3235776257`, `3235776258`, `3235776259`;
- hidden dimension: 128;
- epochs: 12;
- batch size: 256;
- Torch CPU threads: 14;
- trigger-row multiplier: 1 or 2.

This produces six candidates. Each candidate selects its epoch, fusion alpha,
and symbolic-prior smoothing from validation only. Candidate selection uses,
in order:

1. validation game-normalized Top-1 accuracy;
2. validation game-normalized Top-3 accuracy;
3. lowest validation game-normalized negative log likelihood;
4. lower trigger-row multiplier;
5. lower model seed.

The two-stage CLI has no test-path argument during selection. It publishes a
checkpoint with `sealedTestStatus: "unopened"` and records every train and
validation game ID.

## Sealed evaluation

The test trace is not converted or loaded until all six selection artifacts
exist and the winning checkpoint is frozen. The sealed command accepts only
that checkpoint and the test dataset, rejects any train/validation game
overlap, authenticates both inputs, and writes a no-clobber report.

The report must include Top-1, Top-3, Top-5, negative log likelihood, Brier
score, expected calibration error, move-horizon accuracy, per-rule results,
parameter accuracy, trigger and forced-event metrics, and hard-elimination
diagnostics. No 25-label accuracy is reported before that artifact exists.
