# Schema-9 corpus ledger

The schema-9 corpus ledger is the fail-closed handoff between deterministic
Engine generation, Guesser conversion, and the opportunity-ablation
workflow. It authenticates exactly four caller-supplied splits:
`train`, `validation-a`, `validation-b`, and `test`.

The ledger command receives every trace, converted dataset, launch receipt,
and completion receipt as an explicit file argument. It does not search a
directory or choose the newest matching file.

## Frozen schedule contract

The schedule authority is
`capturable25-schema9-opportunity/v1`. Each split records its unique,
path-free schedule ID and three unsigned 32-bit seed roots in this declared
order:

1. `label`
2. `gameplay`
3. `parameters`

| Ledger split | Workflow domain | Seed roots |
| --- | --- | --- |
| `train` | `train` | `1261462769, 242269024, 1837697911` |
| `validation-a` | `validation-a` | `2069246597, 1391196133, 2739675947` |
| `validation-b` | `validation-b` | `3786384219, 3547865132, 2689552677` |
| `test` | `sealed-test` | `2033321041, 1354035545, 4189758462` |

The command rejects any other value. A free-form schedule ID is therefore
only a run identity; it cannot weaken or replace the frozen seed roots.

## Authenticated identities

For every split, the ledger binds:

- the source trace SHA-256, byte count, scheduled game count, and zero-ply
  game count;
- the exact canonical source game-ID and simulation-seed arrays and their
  independent set hashes;
- exact White and Black counts for all 25 frozen labels, counted once per
  scheduled source game, including zero-ply games;
- the converted dataset SHA-256, byte count, row count, and represented game
  count;
- the exact canonical converted game-ID and simulation-seed arrays and their
  independent set hashes;
- the path-free launch and completion receipt SHA-256 and byte identities;
- the producer Engine commit; and
- the schedule ID and frozen seed roots.

The source arrays contain every scheduled game. The converted arrays contain
exactly the source games with at least one emitted row. They must be strict
source subsets when zero-ply games exist, and:

```text
converted games = source games - zero-ply games
```

The four source game-ID sets and four source simulation-seed sets must be
pairwise disjoint.

## Converter and repository binding

The Guesser commit must pin the declared converter Engine commit at its
`engine` gitlink. Under `exact/v1`, every producer commit must equal that
converter commit. Under `converter-ancestor/v1`, the converter commit must be
an authenticated Git ancestor of every producer commit.

Every source line is parsed and semantically replayed by the pinned Engine
schema-2 parser. The pinned converter then regenerates each dataset row, and
the supplied converted file must match those rows byte for byte. A neural
training process cannot override this check.

## Canonical hashes

Ledger JSON is UTF-8, recursively key-sorted, compact JSON with one trailing
LF. `contentSha256` is the SHA-256 of that canonical payload before the
`contentSha256` field is added.

Game IDs are the lowercase ASCII identifiers reconstructed by the Engine
parser. Game-ID sets use locale-independent code-unit lexical order, which is
the same as Python's default order for this ASCII domain. Seed sets are
numerically sorted. Each set hash is:

```text
SHA-256(canonical compact JSON array + LF)
```

Partition hashes use the fixed four-split order and hash arrays of
`{"split": ..., "values": [...]}` objects with the same canonical encoding.

## Publication and re-authentication

Creation writes a mode-`0600` temporary file, flushes it, and publishes with a
create-only hard link. An existing destination is never overwritten.

Verification reopens a regular non-symlink ledger, checks canonical bytes and
the self-hash, re-authenticates all explicitly supplied source files and
repository identities, reconstructs the complete artifact, and requires
byte-identical equality.

No local file path, user-directory name, or receipt body is copied into the
ledger. Receipt JSON must itself be strict, duplicate-key-free, and path-free;
only its SHA-256 and byte count are retained.

## Command inputs

`drawback-guesser-schema9-ledger` supports `create` and `verify`. Both modes
require the same complete input contract:

- ledger output/input, Guesser repository, Engine repository, and their
  commits;
- one producer/converter policy; and
- for each of the four splits, trace, converted dataset, launch receipt,
  completion receipt, schedule ID, all three seed roots, and producer Engine
  commit.

Omitted, duplicate, or unknown flags are rejected. Successful output contains
only the artifact format, version, byte/hash identity, and verification
status.
