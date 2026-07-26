# Capturable baseline final results

## Claim boundary

This is the final held-out result for the protocol in
`capturable-baseline-protocol.md`. It measures a fresh-start hybrid model on
100 newly simulated games covering the 10 audited `capturable-king/v1`
drawbacks. It does not measure live human games, unsupported catalog rules, or
the complete Drawback Chess catalog.

The model, epoch, and fusion settings were frozen and published before the
final test dataset was converted or evaluated. The final checkpoint's 14 state
tensors are bit-for-bit identical to the pre-test frozen reproduction.

## Bound artifacts

- Engine revision:
  `6994b727a4425b72423525cfe28767d4567c593d`
- Final trace: 100 games, SHA-256
  `cd6438df2dbcfabf6af7e449d240e5421e179613cefb8fb4ed34efec0a0810f9`
- Final dataset: 5,381 rows, 100 games, SHA-256
  `1f8cef369bff52dfeee1fe65873b410c8caada83c6dc587ef97e2db80597dedb`
- Checkpoint SHA-256:
  `c078dd867c49e3098316e4a68b0ca075174edb5be497d67915a86d07068781b2`
- Evaluation report SHA-256:
  `65a911327a02d37115eb5f2938b3a8ce1716580c27009b31dc6540bcf5b55db3`
- Selected epoch: 2
- Fusion alpha: 2
- Prior smoothing: 0
- Fresh start: true

Generated datasets, checkpoints, and full reports remain outside the
repository.

## Aggregate results

The row-weighted result gives every observed move equal weight. The
game-normalized result gives every player-game equal weight and is the safer
headline result because long games cannot dominate it.

| Metric | Hybrid, row-weighted | Hybrid, game-normalized | Symbolic-only, row-weighted | Symbolic-only, game-normalized |
| --- | ---: | ---: | ---: | ---: |
| Top-1 accuracy | 42.6501% | 41.3564% | 40.9360% | 39.4332% |
| Top-3 accuracy | 70.3773% | 69.5904% | 68.6314% | 66.6301% |
| Top-5 accuracy | 87.6975% | 87.0540% | 85.8697% | 84.2307% |
| Negative log likelihood | 2.4552 | 2.4934 | 3.5084 | 3.5969 |
| Brier score | 0.7742 | 0.7923 | 0.8796 | 0.9057 |

Hybrid expected calibration error is 0.2237. There are 200 player-games:
150 reached a unique rank-one prediction at least once and 50 never did. For
those that reached rank one, the mean first rank-one move is 8.53 and the
median is 5.

## Accuracy by observed player moves

| Observed moves | Top-1 | Top-3 |
| ---: | ---: | ---: |
| 5 | 27.0% | 49.0% |
| 10 | 33.0% | 64.0% |
| 15 | 43.0% | 76.0% |
| 20 | 51.0% | 84.5% |

## Per-drawback results

| Drawback | Rows | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| Checkers | 547 | 81.17% | 82.63% | 83.00% |
| Femme Fatale | 554 | 2.17% | 26.53% | 67.69% |
| Irresistible | 483 | 30.85% | 59.83% | 79.92% |
| Lame Duck | 498 | 31.33% | 88.35% | 99.40% |
| Nurturer | 588 | 21.43% | 58.67% | 83.16% |
| Spice of Life | 561 | 91.44% | 98.22% | 98.22% |
| Triple Play | 579 | 2.76% | 37.31% | 78.24% |
| Truant | 549 | 73.77% | 97.63% | 99.64% |
| Vegan | 585 | 66.32% | 81.54% | 90.94% |
| You Best Not Miss | 437 | 19.68% | 76.43% | 100.00% |

The rule-family Top-1 results are 64.90% for history restrictions, 57.57% for
forced moves, 50.23% for forbidden mover or capture rules, and only 8.95% for
king-capture restrictions. That last family is the main accuracy bottleneck.

## Auxiliary outputs and exactness

- Triple Play `requiredType` parameter accuracy: 54.23%
- Last-move trigger accuracy: 76.19%
- Forced-move accuracy: 96.78%
- Hard-elimination violations: 0 across 5,381 checked rows
- Maximum probability assigned to an eliminated hypothesis: 0
- Missing hard masks: 0
- Maximum probability-sum error: `2.220446049250313e-16`

The neural model therefore never restores a drawback that the exact rule
engine has eliminated.

## Interpretation

The hybrid is substantially better calibrated than the symbolic-only
distribution and improves both Top-1 and Top-3 accuracy, but 41.36%
game-normalized Top-1 is not enough to call the guesser generally reliable.
The model is already useful as a ranked research assistant for these 10
simulated rules, especially after 15 to 20 observed moves. It is not ready for
a claim of accurate live-game identification.

The next protocol should target the confused king-capture rules with
trigger-rich simulations and diagnostic probing, then evaluate once on a new
sealed seed set. The current final test must not be reused for model selection.
