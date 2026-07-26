# Capturable 25-label fixed-blend confirmation protocol

## Objective and selection disclosure

This experiment performs one fresh confirmation of a fixed treatment weight
of `0.1`. The weight is not preregistered independently of all prior evidence:
it was identified after the completed convex-grid validation as the only
nonzero grid value that passed every release check while improving Top-1,
Top-3, and Top-5.

That validation set is now consumed for development. It cannot confirm this
candidate. The present protocol therefore freezes one weight before any fresh
test is generated and prohibits all test-time weight selection.

## Frozen candidate

The component inputs remain:

| Role | Private artifact | SHA-256 |
| --- | --- | --- |
| control selection | `capturable25-v3-control-s3235776259-t2/selection.json` | `889e8a22f812f6359b7aa36d66436f077bbbbd80fbe7ef6ac393533eb47fbf06` |
| control checkpoint | `capturable25-v3-control-s3235776259-t2/model.pt` | `b314ae8e0c020490363237a025c8f6291dc24073542e00c17091a87df9c65469` |
| treatment selection | `capturable25-v3-balanced-s3235776257-t1/selection.json` | `3a6e66e46002d037e4a1e599b7accf7d9ede753b155a8d12fc3cd1ad3086ce57` |
| treatment checkpoint | `capturable25-v3-balanced-s3235776257-t1/model.pt` | `8f821461304bdc03786aea354676c0dd8fccadaecbad89496d0322dc095c424f` |
| convex-grid decision | `capturable25-v3-convex-blend-validation.json` | `e7721460b8ab94e2bd4e8ee293efdc36e25d49c92c2f9660dae3dba532ad6375` |
| prior-corpus registry | `capturable25-prior-corpus-registry-v1.json` | `af97da7cf0e790fc50747898141d348cf016855e6158e2bc1cd6a835c66aa1a0` |

The grid decision binds clean DrawbackGuesser revision
`02ee847f9d6791a5eb09a281026ce537f33e922c`. Its weight-`0.1`,
row-ordered validation prediction stream has SHA-256
`86299d8cb6c79973f7e675d495254a7a7f265a2bcf218758d46cfd048b7f223b`.
The registry canonically binds all 54 pre-generation NDJSON sources and the
union of their 9,037 game IDs.

The fixed candidate is:

```text
P_fixed = 0.9 * P_control + 0.1 * P_treatment
```

The same arithmetic probability mixture applies to the drawback, trigger,
forced-move, and Triple Play parameter outputs after each component's own
selected symbolic fusion. Component hard masks must match exactly, and every
eliminated drawback must retain probability exactly zero.

For context only, its consumed-development validation metrics were:

- game-normalized Top-1: 32.1651% versus 31.9432% control;
- game-normalized Top-3: 52.6161% versus 52.3559%;
- game-normalized Top-5: 64.2853% versus 64.2221%;
- NLL: 2.5740 versus 3.1531;
- Brier: 0.8890 versus 0.9103;
- ECE: 0.2416 versus 0.2607.

These values selected the fixed hypothesis; they are not confirmation
evidence.

## Fresh confirmation corpus

Generation is pinned to the same validation-era DrawbackEngine revision
`74eb6fc95571994bd96b7a351278f3f74f0972e3` and the production `worst-case`
opponent aggregation. This avoids changing the synthetic move distribution
between model selection and confirmation. The schedule is:

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

The private trace basename is
`capturable25-v4-fixed-blend-confirmation-trace.ndjson`; the converted test
basename is `capturable25-v4-fixed-blend-confirmation-schema8.ndjson`.
The canonical preparation receipt basename is
`capturable25-v4-fixed-blend-confirmation-corpus-receipt.json`.
The schedule must contain every one of the 625 ordered label pairs exactly
once, with 25 player-games in every label/color cell. Every scheduled game
must be emitted and row-bearing for both colors, retaining both player
trajectories. A game still active at exactly the 60-ply limit is retained as
a censored active trace, matching the validation distribution; it must not be
dropped, extended, or relabeled as a completed game. Earlier terminal games
retain their actual result. The trace and converted schema-8 corpus must pass:

- deterministic schedule and replay verification;
- exact public-authority reconstruction;
- strict UTF-8/LF and closed-schema loading;
- true-hypothesis survival;
- zero game-ID overlap with every earlier train, validation, diagnostic, or
  test corpus recorded in the pre-generation audited corpus registry;
- byte hashing before and after loading.

The trace, dataset, checkpoints, and evaluation report remain outside Git.

## Corpus preparation receipt

Before sealed evaluation, one committed audit command must validate the
generated trace and converted dataset without loading either model. The
command must:

1. authenticate a clean DrawbackGuesser revision containing the audit tool;
2. authenticate the detached generator worktree at Engine revision
   `74eb6fc95571994bd96b7a351278f3f74f0972e3` and the frozen Engine lockfile;
3. independently reconstruct every scheduled assignment, including game,
   parameter, and label seeds;
4. strictly parse and semantically replay all 625 schema-v2 trace records;
5. require the standard initial position, exact unrestricted-baseline
   opponent hypotheses, exact worst-case material search policy, exact
   censoring semantics, and game indexes `0` through `624`;
6. regenerate every schema-8 row from the trace and require byte-for-byte
   identity with the converted dataset;
7. authenticate the frozen prior-corpus registry and prove zero game-ID
   overlap; and
8. hash the trace and dataset both before and after verification, failing if
   either changes.

The command publishes one create-only canonical UTF-8/LF JSON receipt. The
receipt uses closed format
`drawbackguesser-capturable-fixed-confirmation-corpus-receipt` version 1. It
contains only protocol and clean audit-revision identity; generator commit,
lockfile hash, and complete frozen schedule; trace filename, byte hash, byte
count, game/ply/result counts, schema/authority/random-policy identity, replay
status, pair/marginal counts, and index bounds; converter and Engine-submodule
identity; dataset filename, byte hash, byte count, row/game/schema/authority
counts and true-hypothesis-survival status; and prior-registry
filename/hash/count/overlap status. It contains no moves, positions, hidden
parameters, private states, or per-game labels.

Receipt preparation is part of deterministic corpus construction, after the
candidate and protocol are frozen and before any model inference. It may not
load checkpoints, calculate predictions, or expose evaluation metrics.

## One-pass sealed evaluator

The evaluator must be committed and the source tree clean before it may read
the fresh dataset. It must:

1. authenticate this protocol, the convex-grid decision, both selection
   reports, and both checkpoints;
2. require the grid artifact's weight-`0.1` validation candidate to pass the
   complete gate, without using the grid's rejected selected weight;
3. authenticate the fixed-basename corpus receipt as metadata and bind its
   exact SHA-256 without opening the trace or test dataset;
4. durably publish a no-clobber consumption marker before opening either the
   trace or test bytes, so any later audit, load, inference, or publication
   failure still leaves the test irreversibly marked consumed;
5. rerun the committed semantic replay and byte-for-byte trace-to-dataset
   audit after marker publication, and require exact agreement with the bound
   receipt;
6. authenticate the pre-generation corpus registry and reject any registered
   game-ID overlap before inference;
7. run each component once over the same ordered test rows;
8. evaluate only control and fixed weight `0.1`;
9. write one canonical, content-addressed, no-clobber report on successful
   completion and bind the consumption marker, receipt, trace, and dataset
   SHA-256 values.

The evaluator accepts no weight argument and cannot enumerate alternatives.

The consumption marker has the fixed basename
`capturable25-v4-fixed-blend-confirmation-consumption.json`. It is canonical
JSON with the closed format
`drawbackguesser-capturable-fixed-blend-consumption` version 1 and contains
only: state `consumed`; protocol file/commit/hash; clean execution revision;
the frozen weight; the grid, selection, checkpoint, and registry bindings;
the fixed test basename; and the complete generation schedule above. It does
not contain or require the trace or test hash because it must be durably
created before either file is opened. It additionally contains only the
fixed receipt basename and receipt SHA-256. The marker's same-directory
temporary file is flushed and fsynced before a no-replace link is created,
then the containing directory or equivalent platform metadata barrier is
flushed before evaluation continues. Failure to establish that durability
barrier fails closed before trace or test access. The marker is never removed
after a later exception.

The CLI accepts a private output directory, not a caller-selected report
filename. A successful evaluation is canonical JSON named
`capturable25-v4-fixed-blend-confirmation-report-<sha256>.json`, where the
suffix is the SHA-256 of the exact report bytes. Publication is create-only,
durable, and the report binds the marker filename and hash, receipt filename
and hash, trace filename and hash, and dataset filename and hash.

## Paired uncertainty gate

The aggregate metrics below are deterministic benchmark results, but a tiny
positive delta on one finite corpus is not enough to claim an improvement.
The evaluator must therefore calculate one frozen whole-game paired bootstrap
for game-normalized Top-1:

- cluster unit: the complete game, retaining its White and Black trajectories;
- within-player-game value: mean tie-aware Top-1 credit across that player's
  observed rows;
- within-game value: mean of the White and Black player-game values;
- paired observation: fixed-blend within-game value minus control;
- bootstrap replicates: 20,000;
- each replicate draws exactly 625 physical game IDs with replacement;
- bootstrap seed: 633454611;
- sampler: SplitMix64 with unsigned 64-bit wraparound, using rejection
  sampling before modulo reduction to select a game index;
- interval: the one-sided 95% percentile lower bound, defined as the 1,000th
  smallest replicate (zero-based sorted index `999`).

The sampler contract is `splitmix64-rejection/v1`. Starting with the seed as
the unsigned 64-bit state, each draw adds `0x9E3779B97F4A7C15`, then applies
the standard SplitMix64 xor/multiply sequence with
`0xBF58476D1CE4E5B9` and `0x94D049BB133111EB`, masking to 64 bits after
each addition and multiplication. Let `u` be the final xor-shifted unsigned
value and `limit = 2^64 - (2^64 mod 625)`. Values `u >= limit` are discarded;
an accepted draw selects index `u mod 625`. Sampler state continues across
replicates rather than restarting.

The canonical `math.fsum` mean of the unresampled game-delta vector is the
reported game-normalized Top-1 delta; it must not be recomputed through a
second summation path. Promotion requires both an observed Top-1 gain of at
least `0.001` (0.1 absolute percentage point) and a bootstrap lower bound
strictly greater than zero. The fixed seed, replicate count, sampling
algorithm, cluster definition, percentile index, and threshold are not CLI
arguments.

Because the schedule contains one game per ordered drawback pair, this bound
measures robustness across the 625 physical games; it does not estimate
conditional gameplay-seed variance within each ordered pair or prove a
population effect of `0.001`. Failure retains control rather than weakening
the gate.

## Confirmation release gate

Promotion requires every check below on the fresh corpus:

1. the fixed blend's Top-1/Top-3/NLL tuple is strictly better than control;
2. the paired Top-1 observed gain and lower confidence bound pass the frozen
   minimum-effect and uncertainty gate;
3. game-normalized Top-1, Top-3, and Top-5 do not regress;
4. game-normalized NLL and Brier do not regress;
5. expected calibration error does not regress;
6. accuracy at 5, 10, 15, and 20 observed moves does not regress;
7. White and Black Top-1, Top-3, and NLL each do not regress;
8. trigger and forced accuracy, NLL, and Brier each do not regress;
9. Triple Play hidden-parameter accuracy does not regress;
10. no audited drawback loses more than one absolute Top-1 percentage point;
11. hard masks are complete and exact, with zero restored probability.

There are no tolerances beyond the one-point per-drawback limit. Any failed
check retains control. A passing result promotes only this exact two-member
weight-`0.1` ensemble; it does not authorize another weight, retraining run,
or unmeasured model.

## Claim boundary

Passing would confirm a synthetic 25-label capturable self-play improvement
on one fresh balanced corpus. It would not establish full-catalog, human-game,
or external live-game reliability. The product remains an offline
completed-game research and training tool.
