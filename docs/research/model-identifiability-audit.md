# Model identifiability audit

Status: validation-only research audit. No test NDJSON, test aggregate, or test
report was opened.

## Executive finding

The current public-input boundary is appropriately strict, but the v1 and v2
models do not yet present the classifier with rule evidence at the right grain.
Validation performance remains close to a 22-class uniform classifier, and v2
made likelihood and calibration worse despite adding SAN history. That result
does not prove any single cause. It does reject the proposition that the
current GRU/history change is sufficient.

The next experiment should be called v2.1 and should change evidence alignment
before increasing model size:

1. construct one prefix sequence per player and update that player's drawback
   loss only after that player moves;
2. include the complete current observed move in the sequence;
3. add public, candidate-rule legality features computed independently for
   every hypothesis;
4. train a classification-only control before reintroducing auxiliary losses;
5. balance sampling and reporting at the player-game level, with
   trigger-opportunity and agent strata; and
6. evaluate these changes as preregistered ablations on validation only.

## Evidence inspected

Committed sources:

- `ml/training/drawback_ml/records.py`, `features.py`, `sequence.py`,
  `model.py`, and `training.py`;
- `packages/simulation/src/dataset.ts`, `simulation.ts`, `agents.ts`, and
  `catalog.ts`;
- `docs/research/baseline-v2-experiment.md` and
  `baseline-v2-validation-results.md`; and
- only the validation summary rows from the older v1 results document.

Local ignored evidence:

- the frozen corpus manifest;
- `train.ndjson` and `validation.ndjson`; and
- the selected epoch-1 validation JSON for v1 and v2-GRU seeds `20260725`,
  `20260726`, and `20260727`.

All local measurements below use the manifest identified in the committed v2
validation report by SHA-256
`E536A140EE0E57832819BCBC3243A8FBD610C339DF909DCFD1A92C3993407DD4`.
They are descriptive audits of that frozen corpus, not new model selection.

## What the model can actually see

`records.py` admits only FEN before the observed move, UCI move, move number,
ply, player color, prior SAN history, ordinary legal moves, and clock. It
rejects true drawbacks, hidden parameters/state, drawback-legal moves,
trigger/forced labels, and results as features. Agent ID/style/strength,
game ID, and seed are also absent from `FeatureRecord`. This is a sound secret
and simulator-metadata boundary.

The numerical encoder uses:

- a 12-piece, 64-square one-hot board from pre-move FEN;
- side to move, castling rights, en-passant file, and FEN counters;
- player color, ply, move number, prior-history length, and ordinary legal-move
  count; and
- only normalized origin and destination squares for the current move.

Although `encode_move` retains promotion for legal-mask indexing,
`build_feature_vector` discards that promotion component after calculating
origin and destination. The current move's piece identity, capture, promotion,
check, and SAN token are not supplied directly. Some are inferable from the
pre-move board and squares, but the network must learn that reconstruction.

V2 adds an exact-token GRU over `historySan`. Dataset construction appends the
current SAN only after emitting its row, so this history ends immediately
before the current observed move. Consequently v2's recurrent path does not
encode the very move whose legality is the newest evidence; that move remains
only the two numerical squares used by v1.

The public `ordinaryLegalMoves` set is parsed but encoded only as its count.
The identities of legal alternatives—which are necessary to decide whether an
observed move was forced or excluded under a candidate rule—are not inputs.

These are code facts. Whether each omission explains the validation result
requires the ablations proposed below.

## Supervision grain and evidence alignment

`group_training_examples` attaches both game-level drawback labels to every
move row. The training loop then applies both White and Black classification
losses on every row.

This means:

- after a White move, the White head has one new White action but the Black
  head receives another loss for the same Black label without a new Black
  action;
- each player-game contributes in proportion to total game length, not one
  equally weighted trajectory; and
- adjacent prefixes from one game appear as many highly correlated optimizer
  examples.

This is not label leakage, and it is not proven to harm accuracy. It is a
plausible mismatch between the causal evidence unit (a player's observed
choices) and the optimization unit (every board prefix for both heads). The
v2 protocol already identifies the player-game trajectory as the decision
unit; training should match it.

## Simulator signal and nuisance variation

Each simulated agent receives only moves already filtered by its own drawback.
The random agent samples uniformly. The other four agents share one compact
material/promotion/check/center score; the three “human-like” levels differ
only in softmax temperature. Therefore a drawback is observable through:

1. an actual restriction of the legal set; and
2. the selected move under a simple agent policy once that set is restricted.

On turns where a rule removes no ordinary move, the agent policy has no direct
rule-specific input. The resulting move can still contain historical evidence,
but a no-trigger turn alone should not be treated as strong positive evidence.

Agent selection is independent of drawback selection, which avoids a direct
label shortcut. Agent metadata is correctly excluded from model inputs.
However, the policy mixture is a nuisance variable. Measured move-row shares
were:

| Agent | Train | Validation |
| --- | ---: | ---: |
| greedy material | 19.0% | 20.2% |
| human-like weak | 20.2% | 17.5% |
| human-like medium | 22.2% | 20.0% |
| human-like strong | 19.2% | 27.2% |
| random legal | 19.4% | 15.1% |

The split difference is descriptive, not evidence that agent mix caused the
model failure. V2.1 should stratify validation metrics by agent and use
player-game-balanced training batches so this hypothesis can be tested.

## Trigger prevalence and identifiability

From 15,075 training move rows, 6,278 were marked triggered (41.65%) and 497
were forced (3.30%). From 3,038 validation rows, 1,176 were triggered (38.71%)
and 104 were forced (3.42%).

Validation trigger prevalence varied sharply by true drawback:

| Drawback | Rows | Triggered | Rate |
| --- | ---: | ---: | ---: |
| Quit Horsing Around | 80 | 5 | 6.25% |
| Eye for an Eye | 116 | 20 | 17.24% |
| Remorseful | 80 | 16 | 20.00% |
| Just Passing Through | 231 | 49 | 21.21% |
| Trophy Wife | 135 | 29 | 21.48% |
| Horse Tranquilizer | 221 | 51 | 23.08% |
| Pacman | 60 | 15 | 25.00% |
| True Gentleman | 152 | 39 | 25.66% |
| Even Keeled | 114 | 31 | 27.19% |
| Vegan | 201 | 61 | 30.35% |
| Cess | 80 | 71 | 88.75% |
| Spice of Life | 157 | 130 | 82.80% |
| Truant | 148 | 112 | 75.68% |
| Gambler | 197 | 133 | 67.51% |
| Forward March | 139 | 83 | 59.71% |

The table shows the low and high ends rather than selectively claiming that
trigger frequency determines accuracy. “Triggered” means the rule changed the
legal-set cardinality on that turn; it is not a complete measure of
identifiability. A restriction can be redundant with ordinary legality, and
two rules can produce the same surviving set. Conversely, repeated
non-triggering opportunities may still be useful negative evidence if the
candidate rule was actually applicable.

V2.1 needs two public diagnostics that are not currently in the report:

- **opportunity:** whether ordinary legal moves include at least one move that
  a candidate hypothesis would remove; and
- **contrast:** how many surviving hypotheses produce a different legal set at
  this prefix.

Both are computed from public position/history plus candidate rule definitions.
Neither uses the assigned secret.

## Validation result audit

Uniform 22-class reference values are 4.55% Top-1, NLL 3.0910, and Brier
0.9545.

The older v1 validation summary reported:

| Head | Top-1 | Top-3 | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| White | 4.54% | 14.81% | 3.4152 | 0.9954 | 0.1381 |
| Black | 5.89% | 16.69% | 3.5179 | 1.0030 | 0.1435 |

The later paired three-seed validation rerun is the correct v2 comparator:

| Model/head | Top-1 | Top-3 | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 White | 5.05% | 17.35% | 3.2125 | 0.9737 | 0.0996 |
| v1 Black | 4.96% | 15.88% | 3.2992 | 0.9808 | 0.1073 |
| v2 White | 5.26% | 15.59% | 3.3018 | 0.9852 | 0.1179 |
| v2 Black | 5.27% | 16.50% | 3.4260 | 0.9884 | 0.1188 |

V2 improved Top-1 by only 0.21 and 0.31 percentage points while worsening NLL,
Brier, and ECE for both heads; White Top-3 also fell 1.76 points. The mask head
remained at 0% exact match and fell from 4.40% to 4.05% micro F1. These are
measured validation outcomes. They show that the GRU plus balanced dense-mask
loss did not solve identification on this corpus; they do not isolate which
component failed.

Epoch 1 had the best mean White/Black validation NLL for every reported seed.
That makes “train the same model longer” a weak next proposal unless a new
learning-curve experiment shows underfitting rather than rapid fitting to
unhelpful objectives.

## Confusion behavior

I pooled the committed-selection epoch-1 confusion matrices over the three v2
training seeds and both color heads. This is a move-row-weighted diagnostic:
each validation row contributes to each corresponding head in each seed, so
counts are not independent player-game trials.

Large v2 off-diagonal flows included:

| Actual | Predicted | Pooled count |
| --- | --- | ---: |
| Vegan | Remorseful | 290 |
| Horse Tranquilizer | Truant | 237 |
| Vegan | Gambler | 183 |
| Just Passing Through | Gambler | 175 |
| Untitled Duck Drawback | Trophy Wife | 173 |
| Just Passing Through | Remorseful | 147 |
| Oddball | Gambler | 142 |
| Barbarian Rage | Truant | 142 |
| Checkers | Cess | 137 |
| Vegan | Quit Horsing Around | 136 |
| Spice of Life | Cess | 135 |

Predictions were concentrated: across the same pooled reports, Remorseful,
Trophy Wife, Gambler, and Truant received 1,726, 1,685, 1,625, and 1,570
predictions respectively. Some true classes had extremely low pooled recall:
Pacman 0/360, Battle Fatigue 4/636, Eye for an Eye 5/696, Remorseful 7/480,
and Even Keeled 14/681.

These flows are not stable semantic “confusion pairs” by themselves. For
example, a model that overpredicts a few labels creates many large
off-diagonals. Targeted corpora should therefore be chosen from rule semantics
and validated opportunity/contrast measurements, not just raw confusion
counts.

## Auxiliary objectives

V1 and v2 assign weight 1.0 to each of six losses: White classification, Black
classification, White parameters, Black parameters, trigger, and legal mask.
Their raw scales and gradient magnitudes are not reported. The legal mask has
20,480 outputs; v2 balances positive and negative cells, but validation mask
quality regressed. Trigger prevalence is 38.71% on validation, yet the report
contains accuracy/NLL/Brier only, not the required prevalence, balanced
accuracy, precision, recall, or rule-conditioned baseline.

It is therefore unmeasured whether auxiliary gradients help, compete with, or
are simply ignored by the shared encoder. V2.1 should not assume causation.
Run:

1. classification only;
2. classification plus candidate-set trigger/opportunity;
3. classification plus sparse candidate legal-mask loss; and
4. the full objective with measured per-loss gradient norms.

The dense 20,480 mask should be replaced with scoring over ordinary-legal
candidates, with the standard-legal mask applied exactly.

## Concrete v2.1 design

### Representation

- Build one chronological public prefix with typed UCI move tokens, color,
  moving piece, capture/promotion flags, and ordinary-chess check status.
  Derive all annotations from public chess state.
- Include the current observed move in the sequence before producing the
  posterior.
- Maintain separate White and Black evidence streams, plus a shared board
  context. An opponent move can update shared position context, but only the
  moving player's action should produce that player's classification loss.
- Encode the actual ordinary-legal candidate set, not only its size.
- For every candidate drawback, compute exact public compatibility,
  candidate-legal-set size, observed-move rank within that set, forced status,
  and whether the rule had an opportunity to bind. Hard incompatibility must
  mask the neural logit rather than become a learned suggestion.

### Optimization

- Sample player-game trajectories uniformly, then sample a prefix/horizon
  within the trajectory. Do not let long games silently dominate.
- Use class-balanced and trigger-opportunity-stratified batches, reporting the
  effective weights.
- Start with a single moving-player classification loss. Add auxiliary heads
  only when an ablation improves validation NLL for both colors.
- Replace whole-vocabulary legal-mask BCE with contrastive or binary scoring
  over ordinary-legal candidates.
- Condition parameter heads on a candidate/true rule during training and
  report oracle-rule separately from predicted-rule performance.

### Data

- Freeze a new manifest with per-split player-game counts by rule, agent, and
  trigger-opportunity bucket.
- Add named hard-negative training profiles for rule pairs whose candidate
  legal sets overlap. Keep the existing validation split fixed for the v2.1
  protocol or create a new sealed protocol; never regenerate validation in
  response to individual mistakes.
- Expand agents only with declared behavior models. Report results separately
  by agent, and include an agent-held-out validation slice to test whether the
  model learned rule constraints rather than one policy's preferences.
- Do not claim “natural capture” or “no capture opportunity” negatives without
  explicit public acceptance predicates and acceptance-rate metadata.

### Validation gates and diagnostics

Before any test access:

- compare uniform, training prior, paired v1, classification-only v2.1, and
  full v2.1 over the same three seeds;
- report player-game-bootstrap intervals, both colors separately;
- report per-rule accuracy/NLL and semantic pair confusion normalized by true
  class;
- report Top-1/NLL by horizon, agent, trigger status, opportunity status, and
  number of surviving symbolic hypotheses;
- report trigger balanced accuracy and NLL against always-negative,
  prevalence, and rule-conditioned prevalence baselines;
- report candidate-mask average precision, macro F1, exact match, and forced
  accuracy against the ordinary-legal-copy baseline; and
- measure gradient norms and validation deltas for each auxiliary objective.

## Falsifiable sequence of experiments

1. **Current-move ablation:** add the full current move token while leaving all
   else fixed. If paired validation NLL does not improve, missing current SAN
   is not sufficient.
2. **Moving-player loss:** train only the active player's head at each row,
   then train trajectory-balanced prefixes. If neither improves both colors,
   supervision alignment is not sufficient.
3. **Symbolic compatibility:** apply exact candidate elimination with uniform
   probability over survivors, without a neural model. This measures how much
   identification is available from legality alone.
4. **Neural-on-survivors:** rank only surviving hypotheses from public
   sequences. Improvement over symbolic uniform measures learned preference
   signal without allowing contradictions.
5. **Auxiliary-loss ladder:** add trigger and sparse mask objectives one at a
   time. Retain an objective only if it improves preregistered validation
   metrics for both colors.
6. **Agent robustness:** compare in-mixture and held-out-agent validation.
   A large gap would support, but not prove, policy overfitting.

The test split remains sealed until a newly preregistered v2.1 configuration
passes its validation gates. This audit makes no test-performance claim.
