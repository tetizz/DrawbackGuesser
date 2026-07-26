# Capturable diagnostic augmentation final results

## Decision

The diagnostic augmentation candidate is rejected for promotion. On its
preregistered 150-game final test it reached 39.5989% game-normalized Top-1
accuracy. That is not reliable drawback identification, and the targeted
king-capture family remained the weakest family at 7.2022% row-weighted Top-1.

This result covers only the 10 audited `capturable-king/v1` drawbacks in newly
simulated ordinary-start games. It is not evidence about live human games,
unsupported rules, or the full Drawback Chess catalog.

## Frozen evaluation

The model and selection policy were published in
`capturable-diagnostic-augmentation-protocol.md` before the final seeds were
generated. The final run used:

- Engine revision:
  `1406224aadde35b45efd777eda83693e4afced38`;
- ordinary training source: 14,960 rows from 300 games, SHA-256
  `1728657bd97bf70b013090c2be21e47c7820133e975501ba58e7f3d5c539f17d`;
- diagnostic training source: 549 rows from 250 games, SHA-256
  `9c748c416cc15bfd75acb0cc3d50d30cbabe9295d9e9f079003d40a971c6a93f`;
- validation source: 5,012 rows from 100 games, SHA-256
  `45131b663220279f24cf8def2749447b43e5023707d8eb9c74be350dbc0e2951`;
- final trace: 150 games, SHA-256
  `a4de4a7e2a8f2c2c81daa88e0ca180bae1fdb125ecbc12cd7a71290bad10216d`;
- final dataset: 7,596 rows and 300 player-games, SHA-256
  `0cf366cc7dbf10d72012179b7cb2a63679d0cbc1ee8b67267b812dd2e1980822`;
- final checkpoint SHA-256
  `a487e591ec2de58ee80190522a3339a3d2d83e534e623874ec96d82d63497fa5`;
- final evaluation report SHA-256
  `7190cc66e1e78f37fb4f46bf65b314d08552d3b5809a7fd24099f097df996e2a`.

The trace passed exact public-position and private-rule replay. The ordinary
training, diagnostic training, validation, and final test game IDs are
pairwise disjoint. The report records `freshStart: true`, format version 2,
epoch 3, fusion alpha 2, prior smoothing 0, and a trigger-row multiplier of 2.

The final checkpoint file has a different hash from the preregistered
exploratory checkpoint because checkpoint metadata binds the final test path,
row count, and hash. All 14 state tensors are bit-for-bit identical to the
frozen candidate.

Generated traces, datasets, checkpoints, and full reports remain outside the
repository.

## Aggregate results

The game-normalized metrics give each player-game equal weight and are the
headline results.

| Metric | Hybrid, row-weighted | Hybrid, game-normalized | Symbolic-only, row-weighted | Symbolic-only, game-normalized |
| --- | ---: | ---: | ---: | ---: |
| Top-1 accuracy | 41.6272% | 39.5989% | 41.1859% | 38.5175% |
| Top-3 accuracy | 69.7867% | 68.1223% | 71.2953% | 68.3731% |
| Top-5 accuracy | 86.8352% | 85.6715% | 86.7105% | 84.6966% |
| Negative log likelihood | 2.7631 | 2.7544 | 3.9435 | 3.9808 |
| Brier score | 0.7947 | 0.8185 | 0.8912 | 0.9264 |

Hybrid expected calibration error is 0.2327. White player-games reached
38.4719% game-normalized Top-1 and Black player-games reached 40.7259%.
The hybrid reached rank one at least once in 219 of 300 player-games; 81 never
reached rank one. Among successful player-games, the mean first rank-one move
was 8.49 and the median was 5.

## Accuracy by observed player moves

| Observed moves | Top-1 | Top-3 |
| ---: | ---: | ---: |
| 5 | 25.33% | 53.00% |
| 10 | 36.33% | 62.33% |
| 15 | 43.67% | 74.00% |
| 20 | 48.00% | 85.00% |

## Per-drawback results

| Drawback | Rows | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| Checkers | 739 | 78.48% | 82.00% | 82.95% |
| Femme Fatale | 835 | 2.51% | 51.50% | 90.06% |
| Irresistible | 693 | 36.08% | 55.56% | 71.86% |
| Lame Duck | 704 | 32.81% | 67.76% | 88.78% |
| Nurturer | 871 | 16.07% | 65.33% | 87.60% |
| Spice of Life | 847 | 92.68% | 96.69% | 97.64% |
| Triple Play | 821 | 2.56% | 36.30% | 75.15% |
| Truant | 723 | 74.83% | 93.91% | 98.89% |
| Vegan | 839 | 51.25% | 70.68% | 79.74% |
| You Best Not Miss | 524 | 31.11% | 84.92% | 98.66% |

| Rule family | Rows | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| Forbidden mover or capture | 1,543 | 42.84% | 69.35% | 83.86% |
| Forced move | 1,432 | 57.96% | 69.20% | 77.58% |
| History restriction | 2,094 | 71.11% | 92.79% | 98.33% |
| King-capture restriction | 2,527 | 7.20% | 51.33% | 84.37% |

The added diagnostic data did not solve the intended family. Femme Fatale and
Triple Play remained close to chance at Top-1, while many of their positions
were ranked correctly only in the wider Top-3 or Top-5 set.

## Auxiliary outputs and exactness

- Triple Play `requiredType` parameter accuracy: 52.86%
- Last-move trigger accuracy: 74.17%
- Forced-move accuracy: 96.54%
- Hard-elimination violations: 0 across 7,596 checked rows
- Maximum probability assigned to an eliminated hypothesis: 0
- Missing hard masks: 0
- Maximum probability-sum error: `2.220446049250313e-16`

The legal-rule engine continued to override the neural distribution exactly.

## Comparison and next work

The first baseline reported 41.3564% game-normalized Top-1 on a different
100-game final test. This candidate reported 39.5989% on the new 150-game
final test. Because the final sets are independent, the numerical difference
is not a paired estimate of regression, but the new result supplies no basis
for claiming an improvement.

The next experiment should not add more copies of the same short diagnostic
positions. It should first measure trigger coverage at ordinary-game time
horizons, then generate label-independent continuations that preserve
king-capture opportunities for longer. Model work should target the
observational equivalence among Femme Fatale, Nurturer, and Triple Play and
must use a new validation split. This final test is sealed and may not be used
for further selection.
