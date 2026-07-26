# Engine trace to dataset bridge

## Purpose

`@drawbackguesser/trace-to-dataset` is the trusted boundary between privileged
DrawbackEngine simulations and Guesser training data. It accepts version 1
`drawbackengine-private-simulation-trace` records from the pinned Engine
submodule and emits one strict dataset row per observed move.

Both sides of this bridge are private. Engine traces contain hidden drawback
parameters and internal state, while the derived rows contain supervised
labels. Neither format is a browser or player observation format.

## Leakage boundary

Conversion happens in two phases:

1. Parse and semantically replay the Engine trace under
   `standard-chess/v1`.
2. Project only the game ID, seed, public agent metadata, FEN chain, observed
   moves, complete ordinary legal masks, and public evaluator facts.
3. Reconstruct the White and Black symbolic hypothesis distributions from
   that public projection.
4. Create and validate all public feature records.
5. Re-run the claimed executable rules to verify exact parameter schemas,
   pre-move state, drawback masks, trigger/forced flags, and terminal result.
6. Attach the active player's true drawback, hidden parameters, pre-move
   internal state, drawback-legal mask, trigger/forced flags, and game result
   as labels.

Changing a trace's private drawback truth without changing its public moves
must not change the derived public features. Tests enforce this invariant.
The converter also fails if exact symbolic legality has eliminated the trace's
claimed true drawback; a learned model can never override that contradiction.

Trace V1 does not carry the evaluator fact needed to recompute an
evaluator-backed rule's legal mask at the final post-move turn. The bridge
therefore rejects a game whose final side has one of those rules instead of
trusting an unauthenticated result label. A later trace schema must record that
final public constraint before such games can enter a release corpus.

## Determinism and storage

The CLI reads bounded, strict UTF-8 NDJSON without loading the full corpus into
memory. Each input record is validated by the Engine trace parser, including
canonical FEN/SAN/UCI replay and the complete ordinary legal move set.

Rows use a fixed insertion order and one newline per record. Ordered shards
therefore concatenate to exactly the same bytes as a monolithic conversion.
The writer reports the exact byte count and SHA-256 digest.

Output is written to a same-directory temporary file with mode `0600` on
platforms that honor POSIX permissions. Publication uses a no-clobber link, so
an existing output is never replaced. Invalid later records remove the
partial temporary file.

Within one output, every game must share one evaluator coverage mode. Uniform
coverage additionally requires the same evaluator policy ID and engine
fingerprint. Release-candidate generation must pass
`--require-evaluator uniform`; `none` exists only for local research.

## Current release status

This bridge establishes the dataset producer and its public/private boundary.
It does not turn the historical schema-6 corpus protocol into a new release.
Schema 6 remains reproducible legacy research data. A schema-7 release
manifest must separately bind the exact Engine trace bytes, Engine/catalog
revision, bridge artifact and source revision, derived dataset bytes, split
ledger, and evaluator identity before any trained model may be promoted.
