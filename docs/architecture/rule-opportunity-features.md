# Public rule-opportunity features

## Purpose

An observed move is most informative when a candidate drawback had a real
chance to change the legal move set. A quiet turn with no binding opportunity
should not be treated like evidence that the player deliberately avoided a
restricted move.

Capturable symbolic schema 9 exposes that distinction to the learned model
without exposing either player's secret. Exact symbolic elimination remains
authoritative; these features can rank surviving rules but cannot restore an
eliminated rule.

## Frozen public contract

The contract is additive to the existing 922-value board and history vector:

```text
opportunityFeatureVersion = 1
symbolicActiveRuleOpportunityFeatures = 25 rules x 4 fields
```

Rule order is exactly the frozen capturable 25-label vocabulary. Field order
for every rule is:

1. `knownMass`
2. `allowedMoveFractionMass`
3. `triggeredMass`
4. `forcedMass`

The 100 values are finite numbers in the closed interval `[0, 1]`. Dataset
parsers require the exact length, version, authority, key set, and symbolic
feature version. Legacy schema 8 remains readable only through its legacy
loader and never receives fabricated opportunity values.

## Chronology

For each public move, conversion performs these operations exactly once:

1. retain the active color's symbolic distribution immediately before the
   move;
2. reconstruct the public authority position and ordinary legal moves;
3. ask every live hypothesis for its permitted moves and public trigger/forced
   evidence;
4. observe the move, applying exact hard elimination and state transitions;
5. aggregate the pre-observation opportunity evidence; and
6. serialize the opportunity vector beside the post-observation White and
   Black symbolic distributions.

Using pre-observation probability mass prevents the observed move from
retroactively changing how much weight its own opportunity evidence receives.
Using the post-observation symbolic distribution preserves the existing hard
legality contract.

## Aggregation

Each drawback can have several hidden-parameter hypotheses. Aggregation first
normalizes the live pre-observation log probabilities within one rule with a
stable log-sum-exp calculation.

Known variants contribute their conditional probability mass to
`knownMass`. They also contribute probability-weighted allowed-move fraction,
trigger, and forced indicators. An unknown variant remains in the conditional
denominator but contributes zero to all four fields. A fully eliminated rule
emits four zeros.

This representation intentionally excludes:

- the observed-move-legality flag, because it duplicates hard elimination;
- hypothesis indexes or hidden parameter values;
- internal drawback state;
- private drawback-legal move lists;
- the true drawback label; and
- any game result or evaluator output unavailable at prediction time.

Changing private truth while holding the public game trace fixed must leave
the complete feature record byte-identical.

## Learned residual

The opportunity-aware baseline adds one trainable `25 x 4` weight matrix,
initialized to zero. The dot product for each rule is added only to that
rule's White and Black drawback logits. Trigger, forced-move, and hidden
parameter heads are unchanged.

Two explicit modes share the same schema-9 parser and architecture:

- `public-exact` uses the authenticated opportunity tensor;
- `zero-ablation` replaces it with zeros after strict parsing.

Paired candidates must use identical seeds, initialization, shuffle order, and
training configuration. The mode is part of candidate identity and checkpoint
metadata. Checkpoint loading fails closed on a mismatched feature version,
rule order, field order, tensor shape, or mode.

Before and after the learned residual, the symbolic hard mask is exact:
eliminated rules receive probability zero and cannot be revived by a neural
score.
