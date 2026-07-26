# Real-domain post-game benchmark

Last reviewed: 2026-07-24

## Purpose and safety boundary

`ml.evaluation.real_domain_benchmark` measures the approved predictor on
consented, completed DrawbackChess games. It is an offline evaluation boundary,
not live assistance:

- corpus consent must be explicit;
- every game must be marked complete and have a terminal PGN result;
- `liveCollection` must be `false`;
- simulation seeds are forbidden;
- the analyzer is invoked only after the complete corpus has passed replay and
  overlap validation;
- Stage A has no label argument. Label isolation is an external deployment
  control, not something this Python module can enforce;
- Stage B receives that bundle and the revealed labels, computes aggregates,
  and has no analyzer parameter or inference path;
- the published report contains aggregate metrics and content hashes, never
  PGNs, player identifiers, per-game rows, or per-title results.

No real game data is committed to this repository.

## Authenticated inputs

All benchmark inputs are passed as a path plus expected SHA-256. The corpus,
revealed labels, and candidate provenance must use canonical JSON: UTF-8,
sorted object keys, two-space indentation, and one trailing newline. Duplicate
JSON keys are rejected.

The corpus and candidate provenance artifacts use canonical JSON. The existing
194-title catalog is authenticated by its frozen file hash. Its
historic key order is preserved rather than rewritten, so using this benchmark
does not alter the catalog or the 182-class model identity.

### Completed PGN corpus

```json
{
  "consent": {
    "basis": "explicit",
    "completedOnly": true,
    "liveCollection": false
  },
  "format": "drawbacktrainer-real-domain-completed-pgn-corpus",
  "games": [
    {
      "completed": true,
      "pgn": "<complete PGN>",
      "pgnSha256": "<sha256 of exact UTF-8 PGN>",
      "result": "1-0",
      "simulationSeed": null
    }
  ],
  "version": 1
}
```

Each PGN is legally replayed with `python-chess`. It must contain exactly one
non-empty game and a terminal `1-0`, `0-1`, or `1/2-1/2` result matching the
manifest. Exact PGN hashes and normalized starting-position/move hashes
must both be unique.
The semantic hash deliberately excludes the declared result: changing only
`Result` changes the exact-byte identity but preserves replay identity.

Comments, NAGs, and headers whose names mention drawbacks, handicaps,
parameters, or seeds are rejected. Before analysis, all identifying headers
are removed. The analyzer receives only:

- `Result`;
- `SetUp` and `FEN` when required;
- the comment-free, variation-free legal mainline.

This ensures names, sites, events, and collection metadata cannot become model
features even though they may remain inside the access-controlled input.

### Separately revealed truth

```json
{
  "format": "drawbacktrainer-real-domain-revealed-labels",
  "labels": [
    {
      "black": {
        "parameters": {},
        "title": "Checkers"
      },
      "pgnSha256": "<matching corpus PGN hash>",
      "white": {
        "parameters": {},
        "title": "Vegan"
      }
    }
  ],
  "revealTiming": "after-game-completion",
  "version": 1
}
```

Every corpus game has exactly one White and Black reveal. Titles are
Unicode-normalized, whitespace-normalized, case-insensitive matches against all
194 observed catalog titles. Unknown or ambiguous titles fail closed.
Parameters remain evaluation labels and are never supplied to the analyzer.

### Recomputed training audit

Stage A authenticates the canonical candidate training-corpus-set artifact and
verifies its exact one-primary/six-supplement schema and internal digest. It
requires the seven content-addressed dataset files in canonical primary-then-six
supplement order, plus all six authenticated supplement manifests and plans.
Every dataset, manifest, and plan SHA must match the verified candidate
identity. The full schema-6 manifests bind the hard-negative profile/rules,
source revision, generation run, corpus configuration, plan hash, train
dataset/counts, and frozen policies. The full schema-1 plans bind the same
profile/rules, source revision, run ID, corpus configuration, and schedule.

The evaluator streams every NDJSON dataset, checks exact byte hashes and
candidate byte/row/game counts, enforces contiguous games and stable seeds,
checks ply/FEN continuity, replays every UCI move with `python-chess`, and
derives the complete semantic set from `{startingFen, moves}`. Benchmark
semantic identities are compared directly to that recomputed set. There is no
self-attested PGN, semantic, seed, or “zero overlap” exclusion list.

## Approved analyzer contract

Production supplies an `ApprovedPostGameAnalyzer` adapter that invokes the
production browser Worker with the authenticated calibrated ensemble artifact.
`ApprovedSubprocessAnalyzer` authenticates the runtime, launcher, runtime
dependencies, browser artifact, ensemble release, calibration, and approval
evidence. It retains those authenticated bytes, stages those exact bytes in a
private temporary directory, and rehashes staged executables and predictor
artifacts before launch. The environment is minimal and has no ambient `PATH`.
The sanitized PGN is a private temporary input file; only its SHA-256 is placed
in the environment. The evaluator-only original digest used for label joining
is never passed across the analyzer boundary.

Stdout is drained incrementally through a bounded pipe rather than buffered by
`communicate()`. Timeout, stdout, and output-file bounds fail closed, and
timeout/oversize termination targets the entire process tree (a POSIX process
group or a Windows Job Object configured with kill-on-close). Windows Job
creation and process assignment fail closed before analyzer execution;
the assigned Windows bootstrap keeps its root PID alive after the launcher
returns until the controller has recorded launcher status and terminated the
tree. This prevents a fast parent from exiting while a descendant retains the
stdout pipe. `taskkill /T` from explicit `SystemRoot` remains a direct cleanup
attempt. Returned
predictor identity must match the independently authenticated artifacts.

The adapter receives the sanitized completed PGN and returns:

```json
{
  "classIds": ["<the approved 180 standard-PGN class IDs>"],
  "completed": true,
  "format": "drawbacktrainer-approved-postgame-analysis",
  "pgnSha256": "<input PGN hash>",
  "plyCount": 42,
  "predictor": {
    "approvalEvidenceSha256": "<sha256>",
    "browserArtifactSha256": "<sha256>",
    "calibrationSha256": "<sha256>",
    "ensembleReleaseSha256": "<sha256>",
    "mode": "hybrid-v21-ensemble"
  },
  "snapshots": [
    {
      "black": {
        "<drawback id>": 0.005
      },
      "ply": 1,
      "white": {
        "<drawback id>": 0.005
      }
    }
  ],
  "unavailableSupportedIds": [
    "hand-and-gigabrain",
    "ichtyophobe"
  ],
  "version": 1
}
```

The complete posterior domain must be the exact approved 180-class
standard-PGN view. The two evaluator-fact rules must be explicitly unavailable;
the 180 plus two unavailable IDs must equal the frozen 182 supported IDs.
Every posterior is finite, nonnegative, exact-domain, and normalized within
`1e-9`. Snapshots increase by completed ply and include the final ply.

All games must bind the same browser artifact, ensemble release, calibration,
and approval evidence. Symbolic-only, single-model, mixed-artifact, incomplete,
wrong-PGN, or wrong-class analyses are rejected.

The harness accepts an analyzer interface instead of importing browser code
into Python. The production launcher remains responsible for the already
approved browser execution and its content-addressed transcript. Tests use a
synthetic analyzer; synthetic outputs are never reported as real metrics.

## Metrics and coverage

The report records complete catalog mapping:

- 194 observed titles;
- 182 executable/supported titles;
- 12 explicitly unsupported titles;
- 180 standard-PGN-analyzable supported titles;
- two supported evaluator-fact titles unavailable from an ordinary PGN.

For the actual labels it reports aggregate counts and ratios for supported,
unsupported, analyzable supported, and unavailable supported examples.
Unsupported and unavailable truths are coverage observations, not silently
scored predictions.

For all scorable examples, combined and independently for White and Black, it
reports:

- Top-1, Top-3, and Top-5 accuracy;
- negative log likelihood;
- multiclass Brier score;
- 15-bin expected calibration error;
- final-position metrics;
- metrics at the latest completed prefix at or before plies 5, 10, 15, and 20.

Empty slices are JSON `null`. Infinite NLL is also `null` under the declared
non-finite metric policy. The report never fills missing evidence with an
invented value.

## Two-stage publication

Stage A, `create_real_domain_prediction_bundle`, accepts corpus and complete
training provenance, the catalog, and analyzer. It intentionally has no label
argument or reference. The bundle and report explicitly record
`external-access-domain`; code does not claim it enforced mount isolation.
Real release metrics require a separately provisioned no-label mount and its
operational receipt. Its canonical prediction bundle uses
create-exclusive, no-clobber publication and privately retains the original
join digest; the analyzer receives only the sanitized PGN digest.

Stage B, `publish_real_domain_benchmark_report`, accepts the authenticated
bundle, revealed labels, catalog, and output path. It has no analyzer argument
and cannot run inference. It publishes only canonical aggregate output with
the same no-clobber behavior.

Stage B defaults to `BenchmarkClaimConfig(mode="research")`. Research reports
remain useful for development, but explicitly record that they are not passing
release claims. A public release claim must instead provide
`mode="release-claim"` and an explicit list of claimed drawback IDs. That mode
refuses publication unless the corpus contains at least 2,000 distinct
completed games, every claimed standard-PGN-analyzable drawback has at least
10 player-games, and every scored truth retains nonzero posterior probability.
The report records the claim scope, per-claim support, thresholds, failure
codes, and explicit finite-NLL/zero-truth-probability counters so JSON `null`
cannot conceal an infinite loss.

The report records:

- completed-games-only scope;
- no live integration;
- aggregate-only output;
- no raw PGNs;
- no player identifiers;
- aggregate coverage and metrics derived from validated inputs.

Duplicates and training overlaps are rejected before inference; they are not
reported as measured zero-count fields. No unmeasured
`sourceLeakageCount` or `trainingSeedOverlapCount` claim is emitted.

Keep the input corpus and revealed labels in an access-controlled,
non-repository location. A report is publishable only after independent review
confirms the analyzer adapter invokes the exact browser Worker and approval
evidence named in its predictor identity.

## Verification

Synthetic tests cover:

- successful aggregate metrics and every color/horizon projection;
- all 194 title mappings and 182/12 coverage;
- supported-but-browser-unavailable and unsupported truths;
- exact and semantic duplicate rejection;
- training-corpus overlap rejection;
- live, incomplete, seeded, and label-leaking corpus rejection;
- removal of player identifiers before analysis;
- a Stage A API with no reveal input and explicit external isolation status;
- Stage B scoring without analyzer invocation;
- complete one-primary/six-hard-negative provenance and omitted-source
  rejection;
- result-independent semantic hashing;
- staged-copy tamper and bounded streamed-stdout rejection;
- approved artifact and posterior-domain enforcement;
- canonical report content and no-clobber publication.

Run:

```powershell
.venv\Scripts\python.exe -m unittest `
  ml.evaluation.tests.test_real_domain_benchmark -v
```
