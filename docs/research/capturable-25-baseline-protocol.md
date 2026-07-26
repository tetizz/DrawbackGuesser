# Twenty-five-label capturable baseline protocol

## Scope

This protocol freezes the first schema-8 experiment over all 25 drawbacks
audited for `capturable-king/v1`. It measures synthetic player-private
self-play only. It is not a claim about live human games and it does not
promote a browser model.

DrawbackEngine revision
`436407b51b983ba9c173f93f6c6d08920a36825f` owns move legality, the balanced
assignment schedule, private search, and trace replay. DrawbackGuesser
selection executed from clean revision
`b19d8f5e153ea9cb616fb7b915dcab6d753a2384`. Selection-artifact version 1
binds the exact input bytes, model configuration, validation metrics, and
checkpoint bytes, but does not embed the Git revision. This limitation is
recorded rather than silently overstating the artifact's provenance.

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
`436407b51b983ba9c173f93f6c6d08920a36825f` preserves the target and includes
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

## Completed baseline receipts

The frozen run completed on 2026-07-26. No generated trace, dataset, model, or
report is committed to Git.

| Artifact | Games or rows | SHA-256 |
| --- | ---: | --- |
| train trace | 1,250 games | `a7a9088886cf025ebfb8c42bce178378cbc28fe3f4a89074cb02c5f9886d1a83` |
| validation trace | 625 games | `8ee4fca10c4bd03bc78beee6e05ed8ba62db49d5a372d882b57fb55ed5fe9072` |
| sealed test trace | 625 games | `d8fc6734cc8aa59373d55fee95ba837c447ac3538f70014da7aed32c554f79bb` |
| train schema-8 dataset | 61,117 rows | `d419dce002af469fd6663185378a8cd06aa03f2faa35a382980d27f7adee1977` |
| validation schema-8 dataset | 30,736 rows | `3a2724f753e6f8c07a95d9ea0a7ea7f558b0359dfaa863b9e7facc62e0ae3092` |
| sealed test schema-8 dataset | 30,838 rows | `3b9c7c1b7585ddea8edfabb7dcc7a881de49438564859e94894fbd552a7fb200` |
| candidate selection | 6 candidates | `29d6e2239cedda917757c349ab17f80108c16fc60e079a8d5c65939c2457ffa9` |
| selected checkpoint | seed 3235776257, trigger multiplier 2 | `e5f58efe34d13110d4742fe352e064c70e537e2f8536b17dc49aa87886b8738e` |
| sealed evaluation | 1,250 player-games | `c3fb71271ccd5c70cfa082af111d913759e69a269091d8eaf673ebbd84c2017f` |

Every split contained all 625 ordered drawback pairs. Every label appeared
for both colors in exactly 50 train player-games and 25 validation/test
player-games. All 2,500 games produced model rows, the three splits had zero
game-ID overlap, and the strict loader confirmed that exact symbolic logic
never eliminated the known truth.

## Sealed result

The primary equal-player-game metrics on the single sealed test evaluation
are:

- Top-1: 32.8633%;
- Top-3: 52.2399%;
- Top-5: 63.7741%;
- negative log likelihood: 2.39948;
- Brier score: 0.81946.

Move-row-weighted Top-1 was 34.0521%. Top-1 accuracy at the last available
prefix after 5, 10, 15, and 20 observed moves was 14.32%, 26.16%, 37.20%, and
45.04%, respectively. The expected calibration error was 0.17397. Hidden
Triple Play parameter accuracy was 47.03%.

The exact-symbolic control achieved 16.0924% equal-player-game Top-1 and
46.2271% Top-3. The neural residual therefore improved ranking without ever
restoring a hard-eliminated hypothesis.

This baseline is not accurate enough to call reliable across the full
catalog. Move-row Top-1 ranged from 84.48% for Spice of Life to 0.31% for
Femme Fatale. Validation independently identified the same rare-trigger
weaknesses before the sealed test was opened: Femme Fatale, Nurturer, and
Triple Play were at or below 4% Top-1. A future targeted-training phase may
use that validation evidence, but this consumed test split cannot be reused
to select or claim an improved model.
