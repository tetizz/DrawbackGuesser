# Predictor architecture

The predictor owns independent hypothesis distributions for White and Black.
Its public observation contains the position before and after a move, the
observed move, and ordinary legal moves. It never accepts the true drawback,
true hidden parameters, or authoritative game-engine drawback state.

## Update order

For each hypothesis belonging to the player who moved:

1. Filter a defensive copy of the ordinary legal moves through the candidate
   rule.
2. Permanently eliminate the candidate when the observed move is absent.
3. For a survivor, calculate optional move log likelihood from allowed-move
   count, whether the rule triggered, whether the move was forced, and
   observation-only human, engine-quality, strength, and time signals.
4. Apply the observed move to that hypothesis's private symbolic state.
5. Normalize survivors in log space.

Hard elimination runs before likelihood scoring. An eliminated candidate keeps
negative-infinity log probability and cannot be restored by a scorer.

## Priors and likelihood signals

`historicalFrequency` or `priorProbability` on a hypothesis seed establishes its
prior. Supplying both is rejected. Values are relative positive weights and are
normalized in log space.

Optional move signals are already-calculated log likelihoods at most zero:

- human move likelihood;
- engine-quality likelihood;
- player-strength likelihood;
- time-usage likelihood.

They must be derived only from public observation data. The default scorer adds
weighted signals to a uniform-choice term based on the candidate's allowed move
count. A forced-move adjustment is configurable. When a candidate does not
restrict the ordinary move set, its auxiliary human, engine, strength, and time
signals are scaled to five percent by default, because a no-trigger turn
provides little candidate-specific evidence. The uniform legal-choice term
remains shared across no-trigger candidates and therefore cancels on
normalization.

Callers can inject a custom scorer, but it receives only candidate legal-set
features and observation-safe signals. It does not receive the true label or
secret engine state.
