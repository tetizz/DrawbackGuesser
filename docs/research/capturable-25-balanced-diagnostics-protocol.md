# Capturable 25-label balanced diagnostics protocol

## Objective

This experiment tests an all-label hard-negative correction for the rejected
five-label king-diagnostic treatment. The previous treatment produced a tiny
Top-1 gain but worse NLL, calibration, early-move accuracy, trigger accuracy,
and several rule-level results. It remains rejected for release.

The correction uses the same public king-capture scenarios for every audited
label. A diagnostic starting position therefore cannot itself reveal that the
true drawback belongs to a five-rule subgroup.

## Frozen source and data plan

Generation uses DrawbackEngine revision
`74eb6fc95571994bd96b7a351278f3f74f0972e3`, including the
`catalog-balanced-king-diagnostics-v1` profile.

### Balanced train-only supplement

- split counts: 1,250 train / 0 validation / 0 test;
- label seed: 633417738;
- gameplay seed: 633421835;
- parameter seed: 633425932;
- profile: `catalog-balanced-king-diagnostics-v1`;
- workers/window: 15 / 30;
- maximum plies: 12;
- search depth/node budget: 1 / 5,000;
- temperature/top-K: 35 centipawns / 8;
- evaluator: material v1;
- opponent policy/aggregation: unrestricted baseline / worst case.

The expected schedule contains all 625 ordered pairs exactly twice and 50
player-games in every label/color cell.

### New selection validation

- split counts: 0 train / 625 validation / 0 test;
- label seed: 633430029;
- gameplay seed: 633434126;
- parameter seed: 633438223;
- profile: `standard`;
- workers/window: 15 / 30;
- maximum plies: 60;
- search depth/node budget: 1 / 5,000;
- temperature/top-K: 35 centipawns / 8;
- evaluator: material v1;
- opponent policy/aggregation: unrestricted baseline / worst case.

The expected schedule contains all 625 ordered pairs exactly once and 25
player-games in every label/color cell.

Both corpora must pass exact replay conversion, canonical schema-8 loading,
true-hypothesis survival, row-bearing-game accounting, and game-ID
disjointness from every prior train, validation, diagnostic, and test corpus.

## Frozen model matrix

The control arm uses only
`capturable25-v1-train-schema8.ndjson`. The treatment arm uses that same
primary train file plus the new balanced supplement. Both use the new
validation split.

Each arm trains the full Cartesian matrix:

- seeds: 3235776257, 3235776258, 3235776259;
- trigger-row multipliers: 1 and 2;
- fresh initialization: required;
- epochs: 12;
- hidden dimension: 128;
- batch size: 256;
- Torch threads: 7;
- all remaining optimizer, fusion, smoothing, and auxiliary-loss settings:
  existing `CapturableTrainingConfig` defaults.

Two seven-thread runs may execute concurrently on the 16-logical-processor
host. The six candidates within each arm are selected on game-normalized
Top-1, then Top-3, then lowest NLL, with the existing deterministic parameter
tie-breaks. The selected control and treatment are then compared on the exact
same validation identity.

## Validation release gate

The treatment may reserve a fresh test only if all of these validation checks
pass:

1. the primary Top-1/Top-3/NLL tuple is strictly better;
2. Top-1 and Top-3 do not regress;
3. game-normalized NLL and Brier score do not regress;
4. expected calibration error does not regress;
5. accuracy at 5, 10, 15, and 20 observed moves does not regress;
6. trigger and forced-move accuracy do not regress;
7. no target rule loses more than one absolute Top-1 percentage point;
8. exact symbolic hard elimination remains authoritative.

Failure retains the selected control and consumes no new test.

## Reserved test plan

Only after a committed passing validation decision:

- split counts: 0 train / 0 validation / 625 test;
- label seed: 633442320;
- gameplay seed: 633446417;
- parameter seed: 633450514;
- all other simulation settings: identical to the new standard validation.

The paired sealed evaluator must authenticate both selected checkpoints and
the validation comparison before reading that test. Its release decision uses
the same non-regression checks as validation. No metric from the consumed v1
or five-label-treatment tests may select or tune this experiment.

## Claim boundary

Passing would show improvement only on deterministic synthetic self-play over
the current 25-label capturable catalog. It would not establish accuracy on
human games, unsupported drawbacks, or external live play.

## Executed experiment

The balanced training trace contains 1,250 games and all 625 ordered label
pairs exactly twice. Every label/color cell has 50 player-games, and all eight
diagnostic scenarios occur for every label/color cell. Its SHA-256 is
`a33471b94cdf7ce774b0a9195c267f0159c5e4137ffaecf755faf2e7cb61f61c`.
The converted 4,269-row schema-8 corpus has SHA-256
`84b32242bdc84eb3fe69dd8d6f7d0ebfca4665b434961048325846232f7920e8`.

The new validation trace contains 625 row-bearing games and all 625 ordered
label pairs exactly once. Every label/color cell has 25 player-games. Its
SHA-256 is
`788fa649011d114c5bbe2937ab8c98d2ae3c1a7b57c1ba18b85b5fac23802967`.
The converted 30,352-row schema-8 corpus has SHA-256
`09d5d9a4991d76e9fb564ac1fbc64c10212f80b75b0e28fbb9f74b4a93bbaf3d`.
Both corpora passed strict loading, true-hypothesis survival, and disjointness
checks against each other and every earlier v1/v2 train, validation,
diagnostic, and test corpus.

All 12 preregistered models completed from fresh initialization. The
authenticated arm choosers selected:

- control: seed 3235776259, trigger multiplier 2, epoch 2, checkpoint
  `b314ae8e0c020490363237a025c8f6291dc24073542e00c17091a87df9c65469`;
- treatment: seed 3235776257, trigger multiplier 1, epoch 2, checkpoint
  `8f821461304bdc03786aea354676c0dd8fccadaecbad89496d0322dc095c424f`.

The control chooser has SHA-256
`dc0b62bb6ae608b30142f29dd9861fed8d06666ecf7c1a7183098051e2e7063d`.
The treatment chooser has SHA-256
`ece48646bfd4759b7e88c1eb7f2b67ad0d5d0c3dcd80ee14d6124b0203177e0d`.

## Validation result

| Metric | Control | Treatment | Delta |
| --- | ---: | ---: | ---: |
| Game-normalized Top-1 | 31.9432% | 32.0611% | +0.1179 pp |
| Game-normalized Top-3 | 52.3559% | 51.5386% | -0.8173 pp |
| Game-normalized Top-5 | 64.2221% | 63.0429% | -1.1792 pp |
| Game-normalized NLL | 3.1531 | 2.4468 | -0.7063 |
| Game-normalized Brier | 0.9103 | 0.8385 | -0.0718 |
| Expected calibration error | 0.2607 | 0.1693 | -0.0914 |
| Trigger accuracy | 71.4187% | 73.5998% | +2.1811 pp |
| Forced-move accuracy | 97.4960% | 97.4960% | 0.0000 pp |

Five-move accuracy improved from 11.76% to 13.52%, but accuracy regressed at
10 moves (26.32% to 26.00%), 15 moves (36.24% to 35.84%), and 20 moves
(44.96% to 42.64%). Target-rule Top-1 also regressed by 1.12 points for Femme
Fatale, 5.99 points for Irresistible, and 1.02 points for You Best Not Miss.
Nurturer improved by 6.59 points and Triple Play improved by 0.15 points.

Both selected reports recorded zero hard-elimination violations, zero missing
hard masks, and zero probability assigned to eliminated hypotheses.

The reusable comparison gate conservatively applies the one-point Top-1
regression ceiling to all 25 supported drawbacks, not only the five target
rules required by this protocol. This stronger post-protocol safety check
cannot turn a failed treatment into a passing one.

The primary Top-1/Top-3/NLL ordering prefers the treatment, but the release
gate rejects it because Top-3, multiple move horizons, and three target rules
regressed beyond the preregistered limits. The authenticated comparison has
SHA-256
`2535b9916f75a5eff075570295911e738108a3a92ddcf4ab2e209fe65a096a3d`
and records `primaryDecision: "confirm-treatment"` separately from
`releaseDecision: "retain-control"`.

The selected control remains the release candidate. The reserved test split
was not generated or opened.
