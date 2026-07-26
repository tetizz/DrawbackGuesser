# Completed-PGN evaluator sidecars

Two executable drawbacks depend on a reproducible Stockfish best-move fact.
Ordinary PGN text does not contain that fact, so standard post-game analysis
deliberately ranks 180 of the 182 executable rules. A completed-PGN evaluator
sidecar supplies the missing public evidence without adding evaluator calls to
the pure rule engine or exposing hidden drawback labels.

This path is offline and post-game only. It does not read live games, connect to
competitive chess websites, or provide in-game external assistance.

## Generation

Configure a pinned Stockfish executable:

```powershell
$env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"
$env:STOCKFISH_UCI_NAME = "Stockfish 18"
$env:STOCKFISH_VERSION = "18"

pnpm --dir engine --filter @drawbackengine/cli pgn:evaluator-sidecar -- `
  completed-game.pgn `
  completed-game.sidecar.json
```

The generator:

1. rejects ongoing, malformed, or illegal PGNs;
2. verifies the exact executable SHA-256 and reported UCI engine name;
3. uses the fixed `stockfish-bestmove-v1` policy, 10,000 nodes, one thread,
   16 MiB hash, no pondering, one principal variation, and standard chess;
4. reconstructs every pre-move FEN and complete ordinary legal root mask;
5. resolves exactly one evaluator fact per replay ply;
6. validates every content-addressed cache record;
7. writes one canonical UTF-8 JSON artifact with no BOM using atomic
   no-clobber publication; and
8. prints the exact artifact SHA-256.

The output path must not already exist.

## Trust boundary

Semantic sidecar validation proves that cache records agree with the PGN,
policy, engine fingerprint, search limit, legal roots, and one another. That
alone does not prove who generated the artifact: all hashes are intentionally
reproducible.

A consumer must obtain the expected sidecar SHA-256 from an independently
trusted manifest or channel and call
`loadAuthenticatedCompletedPgnEvaluatorSidecar` with the exact artifact bytes.
The loader verifies the byte digest before parsing JSON, rejects oversized,
non-UTF-8, BOM-prefixed, non-canonical, duplicate-key, partial, reordered, or
unknown-field content, and then performs semantic replay validation.

An adjacent, user-editable receipt is not a trust root. The CLI therefore emits
one atomic artifact and prints its digest instead of publishing a two-file
sidecar/receipt pair that could be stranded by a crash.

## Browser analysis

The completed-game analysis panel accepts a local sidecar and an expected
SHA-256 supplied separately. Selecting a file calculates and displays its
digest but does not trust it or populate the expected-digest field. Analysis
remains disabled until the independently entered digest exactly matches the
selected bytes.

The browser sends an immutable byte snapshot and the expected digest to its
analysis worker. The worker authenticates and semantically binds the artifact
to the exact completed PGN before invoking the predictor. Standard PGN analysis
continues to rank 180 rules. A successfully authenticated sidecar supplies the
per-ply evaluator constraints needed to rank all 182 rules. The downloaded
schema-7 report preserves the sidecar digest, policy, engine fingerprint, and
search limit in its analytical payload.

Sidecar input is limited to 8 MiB. A missing pair member, malformed digest,
wrong byte type, oversized artifact, failed authentication, or replay mismatch
is rejected rather than falling back silently to 180-rule analysis.

## Data boundary

The sidecar contains only:

- exact PGN and normalized-mainline hashes;
- public evaluator policy and executable provenance;
- ordered ply numbers; and
- canonical evaluator cache records.

It contains no true drawback, hidden parameter, private rule state, player
identity, model label, or prediction. Unknown fields are rejected recursively
at the sidecar and cache-record boundaries.
