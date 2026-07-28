# Capturable 25-label rule-opportunity ablation protocol

## Question

Does public, exact knowledge of when each surviving drawback could bind improve
hidden-drawback ranking beyond an otherwise identical model that receives the
same schema-9 rows with the opportunity tensor zeroed?

This protocol tests only the opportunity residual. It does not test a larger
network, a different sampling policy, new private inputs, or a relaxed
symbolic mask.

## Preregistered treatments

For every candidate seed, train a matched pair:

| Arm | Opportunity mode | Input |
| --- | --- | --- |
| Control | `zero-ablation` | strict schema 9, authenticated tensor replaced by zeros |
| Treatment | `public-exact` | strict schema 9, authenticated public tensor retained |

The pair must share:

- the exact train and validation bytes;
- model architecture and zero initialization of the opportunity residual;
- seed and every other parameter initialization;
- epoch count, batches, shuffle order, optimizer, and learning rate;
- trigger-row multiplier and source-weight objective;
- symbolic fusion grid; and
- evaluation code and ordering.

The opportunity mode is part of candidate identity. Same-seed control and
treatment candidates are required and are not duplicates.

## Fresh corpus boundary

Only newly generated schema-9 corpora may decide this intervention. Previously
consulted schema-8 validation or test artifacts cannot select, confirm, or
reject it.

The first balanced v5 training shard is generated from a deterministic
25-label schedule with independent label, gameplay, and parameter seed
streams. Subsequent validation A, validation B, and test schedules must use
new disjoint streams and prove:

- no game-ID or seed overlap across any split;
- exact balance by true drawback and color;
- deterministic public-authority replay;
- true-hypothesis survival;
- exact converter byte hashes;
- identical row sets for both ablation arms; and
- no access to the test rows before one candidate is frozen.

Before training, all four splits must be sealed in one canonical schema-9
corpus ledger as defined in
[`schema9-corpus-ledger.md`](../architecture/schema9-corpus-ledger.md). The
ledger must authenticate the frozen seed roots, exact source and converted
game-ID/seed arrays, zero-ply accounting, both-color label counts, generator
receipt identities, producer and converter Engine commits, the pinned Guesser
gitlink, and the exact `[25, 4]` opportunity contract. Training and evaluation
must re-authenticate the ledger against the explicitly supplied files; a
self-hash alone is not sufficient.

The current material-search self-play shard can establish the feature
pipeline, but cannot by itself prove real-player or strong-engine
generalization. Stockfish/Fairy-backed self-play and a separately governed
real-domain post-game corpus remain required before a production claim.

## Selection sequence

1. Train matched `zero-ablation` and `public-exact` pairs across at least three
   preregistered seeds on the training split.
2. Use validation A to select epoch, symbolic fusion, and any allowed
   calibration setting independently inside each arm.
3. Freeze one paired comparison before opening validation B.
4. Require the same directional result on validation B.
5. Freeze one final checkpoint and configuration.
6. Evaluate that frozen candidate once on a newly generated sealed test.

No result may be selected by test performance. A failed gate consumes the
validation evidence but does not authorize another look at the sealed test.

## Promotion gate

The treatment is eligible for sealed evaluation only if, on both validation
folds:

- game-normalized Top-1 improves;
- NLL and Brier score do not regress;
- Top-3 and Top-5 do not regress;
- Top-1 after 5, 10, 15, and 20 moves does not regress;
- neither White nor Black Top-1 regresses;
- no rule with adequate support has a material unexplained regression;
- expected calibration error does not materially regress;
- every hard-eliminated probability is exactly zero; and
- results reproduce from the authenticated checkpoint and report bytes.

The report must also include per-rule accuracy, per-family accuracy, confusion
matrix, entropy reduction per move, mean move at which truth reaches rank one,
trigger/forced accuracy, and hidden-parameter accuracy where defined.

If the treatment passes the sealed test, it is still described as performance
on fresh synthetic games. A claim that it works on live or human games
requires a disjoint real-domain post-game benchmark with no secret inputs and
the same exact hard-legality boundary.

## Reporting boundary

Until the full sequence passes, the retained public result remains the last
honestly measured fresh held-out result. Validation improvements are reported
as validation results, never promoted to a new accuracy claim.
