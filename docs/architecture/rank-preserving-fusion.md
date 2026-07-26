# Rank-preserving symbolic and neural fusion

The executable symbolic predictor is authoritative. The neural model is a
residual signal: it may order hypotheses that have equal symbolic probability,
but it may not reverse a strict symbolic ordering or restore a hypothesis that
symbolic legality eliminated.

The production ensemble uses the versioned method
`rank-preserving-bounded-residual-plus-symbolic-prior-v1`.

For each color and observed move:

1. Validate finite residuals, a finite non-negative symbolic prior, an exact
   boolean hard mask, and at least one positive-mass survivor.
2. Convert survivor residuals to a centered softmax signal. Equal residuals
   produce an exact zero signal.
3. Group equal symbolic probabilities into tiers.
4. Give each tier local neural headroom of at most one quarter of either
   adjacent log-probability gap, capped at one.
5. Scale the bounded signal by the selected alpha.
6. Assert that every higher symbolic tier still has a greater score than every
   lower tier.
7. Apply the hard mask and normalize.

An eliminated hypothesis always has probability zero. A non-eliminated
zero-prior hypothesis is placed in a finite bottom tier far enough below the
positive tiers that its normalized probability remains exactly zero. This is
intentional: neural evidence cannot resurrect a Bayesian zero.

## Alpha selection

Alpha is selected only on the authenticated validation `selection` partition
from the frozen grid:

```text
0, 0.125, 0.25, 0.5, 1
```

Every candidate uses the same ensemble residuals and symbolic evidence. NLL is
averaged within each player-game, then across player-games for White and Black,
then across the two colors. This prevents long games from dominating. Exact
ties select the smaller alpha.

The content-addressed selection artifact binds:

- the ensemble release;
- private validation manifest and dataset;
- selection seed set and partition identity;
- training corpus set;
- symbolic schema and 182-class order;
- the complete candidate grid and scores;
- the selected alpha.

Calibration, promotion evaluation, browser export, and browser parity receive
no alpha override. They recursively authenticate this artifact and use its
selected value.

## Artifact migration

The new policy is carried by ensemble browser artifact format 4 and ensemble
calibration format 3. Older ensemble artifacts are rejected and must be
regenerated.

Standalone browser model format 2 retains its original additive log-prior
equation for compatibility with its Python checkpoint inference contract. It
is not silently reinterpreted under the new method. A production ensemble must
use format 4, which declares the fusion method, selection artifact digest, and
selected alpha.

Python and TypeScript share committed parity vectors. Both implementations
also test extreme residuals, exact hard zeros, constant-residual no-op,
permutation equivariance, representable near-ties, stale eliminated mass, and
zero-prior survivors.

## Fusion-aware training objective

New `v21-hybrid` checkpoints optimize
`rank-preserving-fusion-grid-nll-v1`, not the legacy additive log-prior
equation. Training uses the same centered survivor softmax, symbolic tier
bases, local headroom, and exact hard mask as production fusion. It averages
cross-entropy over the frozen nonzero alpha grid:

```text
0.125, 0.25, 0.5, 1
```

Alpha zero remains available to validation-time selection as the symbolic-only
control, but contributes no neural gradient and is therefore not a training
objective. Every training row is built only from public symbolic priors and
hard masks plus its supervised drawback label. Training fails closed when the
true label is eliminated or has zero symbolic prior.

Both the in-memory and bounded-memory trainers call the same differentiable
helper. Checkpoint and run metadata bind the loss method, version, production
fusion method, alpha grid, and aggregation rule. This metadata is written only
for newly trained v21 checkpoints; older additive-objective checkpoints retain
their original semantics and must not be reinterpreted as fusion-aware.

Selection reports for new checkpoints use player-game-normalized NLL over the
same nonzero alpha grid. Moves are averaged inside each player-game, player
games are averaged inside each color, and the White and Black head scores are
then averaged. This prevents long games from dominating checkpoint selection.
Version 4 selection bundles recursively bind each checkpoint to its run
identifier, seed, epoch, objective, and authenticated training-corpus set.
Historical version 3 bundles remain byte-addressed for compatibility, while
new version 3 summary emission still requires a safely loadable checkpoint.

Prepared tier bases, fused logits, and posterior probabilities remain binary64
through the checkpoint inference boundary. Casting them to a float32 model
dtype could collapse adjacent representable symbolic priors into an artificial
tie.
