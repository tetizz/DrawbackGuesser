# Capturable 25-label convex-blend result

## Decision

The preregistered convex blend was rejected for release. The retained control
remains the selected 25-label checkpoint, and the reserved fresh test was not
generated or opened.

The validation run used clean DrawbackGuesser revision
`02ee847f9d6791a5eb09a281026ce537f33e922c`. Its private canonical artifact has
SHA-256
`e7721460b8ab94e2bd4e8ee293efdc36e25d49c92c2f9660dae3dba532ad6375`.
The strict loader recomputed its selection and `retain-control` decision
without error.

## Frozen-grid result

The primary Top-1/Top-3/NLL ordering selected treatment weight `0.7`.

| Weight | Top-1 | Top-3 | Top-5 | NLL | Brier | ECE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 31.9432% | 52.3559% | 64.2221% | 3.1531 | 0.9103 | 0.2607 |
| 0.1 | 32.1651% | 52.6161% | 64.2853% | 2.5740 | 0.8890 | 0.2416 |
| 0.2 | 32.3119% | 52.6943% | 64.3497% | 2.4977 | 0.8708 | 0.2233 |
| 0.3 | 32.5419% | 52.8371% | 64.4086% | 2.4517 | 0.8558 | 0.2056 |
| 0.4 | 32.6084% | 52.6758% | 64.3560% | 2.4210 | 0.8439 | 0.1908 |
| 0.5 | 32.7079% | 52.8078% | 64.2649% | 2.4004 | 0.8352 | 0.1774 |
| 0.6 | 32.8108% | 52.7102% | 64.0685% | 2.3878 | 0.8296 | 0.1660 |
| **0.7** | **32.8750%** | **52.7175%** | 63.9209% | **2.3827** | 0.8271 | 0.1589 |
| 0.8 | 32.7571% | 52.5054% | 63.7909% | 2.3856 | 0.8277 | 0.1565 |
| 0.9 | 32.4268% | 52.3063% | 63.6987% | 2.3997 | 0.8315 | 0.1617 |
| 1.0 | 32.0611% | 51.5386% | 63.0429% | 2.4468 | 0.8385 | 0.1693 |

The selected blend improved Top-1 by 0.9318 percentage points, Top-3 by
0.3615 points, NLL by 0.7704, Brier by 0.0832, and ECE by 0.1018. Both color
slices, trigger accuracy, forced accuracy, and Triple Play parameter accuracy
also passed their non-regression checks.

## Failed release checks

Weight `0.7` failed three mandatory checks:

1. Top-5 fell from 64.2221% to 63.9209%.
2. Accuracy after 20 observed moves fell from 44.96% to 44.88%, despite
   improvements at 5, 10, and 15 moves.
3. Eleven drawbacks lost more than one absolute Top-1 percentage point:
   Barbarian Rage, Cess, Checkers, Eye for an Eye, Femme Fatale,
   Irresistible, Lame Duck, Quit Horsing Around, Triple Play, Truant, and You
   Best Not Miss.

The largest rule losses were Checkers (-10.12 points), Barbarian Rage
(-8.22), Irresistible (-4.68), Eye for an Eye (-3.90), Quit Horsing Around
(-3.56), and Lame Duck (-3.14). Those regressions are material and cannot be
hidden by the aggregate Top-1 increase.

Both component streams and every candidate recorded complete hard masks,
zero hard-elimination violations, and exactly zero probability on eliminated
hypotheses.

## Post-result development observation

Weight `0.1` passed every release check on this already-consumed validation
corpus and improved all three Top-k metrics. It was not selected by the
preregistered primary ordering, so this observation cannot alter the
experiment's rejection.

It may be tested only as a single fixed hypothesis under a new protocol
committed before any fresh corpus is generated. Such a protocol must perform
no weight selection on the fresh corpus and must retain the same complete
release gate. Until a fresh confirmation passes, weight `0.1` is exploratory
and the control remains authoritative.

## Claim boundary

These are deterministic synthetic self-play validation results over 25
capturable drawbacks. They are not human-game accuracy, full-catalog
accuracy, or evidence for external live-game use.
