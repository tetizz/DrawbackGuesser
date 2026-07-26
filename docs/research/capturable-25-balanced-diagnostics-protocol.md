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
