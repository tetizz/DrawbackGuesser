# Schema-9 corpus ledger

The version-3 schema-9 corpus ledger is the fail-closed handoff between deterministic
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
Schedule IDs are lowercase opaque identifiers of at most 64 characters. URL,
drive, slash, reserved-device, credential-like, and user-derived values are
rejected.

Every split is replayed as one isolated Engine `train` schedule using the
standard `material-player-private-corpus/v1` profile. The authenticated source
must contain contiguous indexes beginning at zero and must exactly match the
Engine scheduler's gameplay seed, parameter seeds, White/Black labels,
deterministic game ID, and initial position at every index.

## Authenticated identities

For every split, the ledger binds:

- the source trace SHA-256, byte count, scheduled game count, and zero-ply
  game count;
- the exact canonical source game-ID, simulation-seed, and combined
  White/Black parameter-seed arrays and their independent set hashes;
- exact White and Black counts for all 25 frozen labels, counted once per
  scheduled source game, including zero-ply games;
- the converted dataset SHA-256, byte count, row count, and represented game
  count;
- the exact canonical converted game-ID and simulation-seed arrays and their
  independent set hashes;
- the path-free typed launch and completion receipt SHA-256 and byte identities;
- the producer Engine commit and exact producer runtime-tree identity; and
- the schedule ID and frozen seed roots.

The source arrays contain every scheduled game. The converted arrays contain
exactly the source games with at least one emitted row. They must be strict
source subsets when zero-ply games exist, and:

```text
converted games = source games - zero-ply games
```

The four source game-ID sets must be pairwise disjoint. Simulation and
parameter seed values must be globally unique across every stream and split.

Launch receipts use
`drawbackengine-player-private-schedule-launch` version 3 and bind the schedule
authority, ledger split, isolated Engine split counts, roots, profile, schedule
ID, producer commit, producer runtime identity, and exact `generationConfig`.
The generation configuration
is recursively exact-key checked and must equal:

```json
{
  "maxPlies": 120,
  "maxDepth": 2,
  "maxNodes": 50000,
  "temperatureCp": 35,
  "topK": 8,
  "leafCacheEntries": 16384,
  "leafCacheHistoryMode": "full",
  "opponentAggregation": "worst-case",
  "evaluator": {
    "kind": "material",
    "version": 1,
    "evaluatorId": "drawback-material/v1"
  },
  "opponentHypotheses": {
    "kind": "unrestricted-baseline",
    "version": 1
  }
}
```

Completion receipts use
`drawbackengine-player-private-schedule-completion` version 3 and bind the
launch digest, completed state, producer, and exact trace digest, bytes, game
count, and index bounds. It repeats the exact launch producer runtime identity.
Earlier receipts, unknown fields, missing fields,
mutated generation values, unsafe integers, negative zero, overflowing numbers,
and non-object receipt roots are rejected.

The declaration is also bound to realized source semantics for every game.
Each schema-2 trace must use `plyLimit: 120`; both White and Black search
policies must independently match the frozen profile ID, evaluator ID, depth,
node budget, temperature, top-K, cache size, cache-history mode, and worst-case
opponent aggregation. The trace hypothesis policy must contain exactly
`{"kind":"unrestricted-baseline","version":1}`. Any mismatch rejects the
split before its trace or converted identities can enter the ledger.

## Converter and repository binding

The Guesser commit must pin the declared converter Engine commit at its
`engine` gitlink. Under `exact/v1`, every producer commit must equal that
converter commit. Under `converter-ancestor/v1`, the converter commit must be
an authenticated Git ancestor of every producer commit.

The commit relationship is necessary but insufficient. Both generator receipts
must contain this exact path-free `producerRuntimeIdentity` contract:

```text
format: drawbackengine-schema9-producer-runtime
version: 1
algorithm: sha256-engine-runtime-tree-v1
runtime: exact Node version, platform, architecture, and empty execArgv
coordinator: schema9-coordinator/v1 runtime-tree count, bytes, and SHA-256
parallelWorker: player-private-parallel-worker/v1 runtime-tree count, bytes,
  and SHA-256
aggregateSha256: canonical SHA-256 of every preceding identity field
```

All object keys are exact; both component counts are positive safe integers;
digests are lowercase SHA-256 values; and path-like or private values are
rejected. The deferred parallel worker is a separate required component because
it is not loaded by importing the coordinator module alone. The ledger verifier
rebuilds every distinct recorded producer commit in a fresh isolated Engine
checkout and recomputes this identity. All four splits must reproduce one exact
identity, every launch must declare it, every completion must repeat it, and the
ledger stores it both globally and per split. Commit-only or receipt-only claims
therefore fail closed.

Every source line is parsed and semantically replayed by the executing Engine
schema-2 parser. The executing converter then regenerates each dataset row, and
the supplied converted file must match those rows byte for byte. A neural
training process cannot override this check.

The ledger records a `sha256-loaded-module-graph-v2` execution manifest for the
actual parser, converter, schedule replay, and verifier module graphs. A
sanitized child traces Node's synchronous ESM and CommonJS loader hooks,
including computed dynamic imports and `createRequire` loads that run while the
entry graph initializes. It hashes every loaded file byte and governing package
manifest; callers cannot substitute commit-shaped claims for it. The manifest
also binds the exact Node version, platform, architecture, and empty
execution-argument contract. The CLI rejects preload/loader flags,
`NODE_OPTIONS`, `NODE_PATH`, and symlink-preservation overrides. After tracing,
the same loader hooks remain as a guard for the complete create/verify operation:
loads outside the authenticated file set, including deferred file or non-file
modules, fail closed. The CLI also requires the declared Guesser and pinned Engine
commits to be the checked-out `HEAD`, rejects tracked index/worktree changes and
Git replace refs, rejects `skip-worktree`, `assume-unchanged`, and submodule-ignore
configuration, and runs Git with repository-affecting environment variables
removed. The caller-supplied Engine repository must realpath to the checked-out
`engine` submodule, and the Engine package actually resolved by Node must remain
inside that same checkout.

Ignored `dist` output is not trusted merely because its bytes were hashed. Before
any corpus file is opened, the CLI clones the declared Guesser and Engine commits
into an isolated temporary checkout, installs the frozen lockfile offline with
lifecycle scripts disabled, rebuilds the exact parser/converter/scheduler/verifier
package closure with the repository-pinned pnpm version, and requires the rebuilt loaded
module graph to equal the executing graph byte for byte. A mismatch fails closed;
the temporary checkout is removed on success or failure. This is reproducible-build
provenance under the trusted local Node/pnpm runtime, not a hostile-host signature
or remote attestation.

Repository and execution identities are checked again after the full corpus
replay, and verification must reproduce the same code-content manifest.

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
Separate commitments cover game IDs, simulation seeds, and parameter seeds.

## Publication and re-authentication

Creation writes a mode-`0600` temporary file, flushes it, and publishes with a
create-only hard link. An existing destination is never overwritten.
Input path walks reject Node-recognized symbolic links and Windows junctions;
the Windows regression asserts that `lstat().isSymbolicLink()` identifies the
junction form used by the CLI. Stable inputs are then held by open handles and
rechecked against their path, resolved path, object identity, size, and times.
Creation and verification enforce the same 8 MiB canonical byte limit before
publication; larger ledgers require a future chunked/Merkle protocol version.
The caller's optional cancellation signal is checked across repository
verification, split streaming/replay, stable-file reads, and every temporary
file suspension point. The final check occurs immediately before the hard-link
commit. Cancellation before that boundary removes the temporary file and
publishes neither ledger nor verification receipt; after the link succeeds,
the publication is committed and only its byte-for-byte durability check runs.

Temporary and rollback cleanup first renames the authenticated object to a
random quarantine name and rechecks its identity and, for rollback, its exact
content hash. This blocks every replacement installed before that recheck.
Node does not provide a portable fd-relative unlink, so the final
`lstat`-to-`rm` interval cannot exclude a concurrently malicious process running
as the same OS user with write access to the quarantine directory. Schema-9
publication directories therefore must not be shared with such a process; this
is an explicit local-host trust boundary, not a claim of hostile-host safety.

Verification reopens a regular non-symlink ledger, checks canonical bytes and
the self-hash, re-authenticates all explicitly supplied source files and
repository identities, reconstructs the complete artifact, and requires
byte-identical equality.

After that replay, the CLI publishes or authenticates a canonical create-only
verification receipt version 2 named
`schema9-ledger-verification-<ledger-file-sha256>.json`. It binds the ledger
file and content hashes, every input identity, repository policy, and the
content-derived execution manifest. Every Python workflow stage requires and
reauthenticates this direct-sibling receipt; a caller digest or self-consistent
ledger alone cannot enter the workflow.

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
only the artifact format, version, byte/hash identity, verification-receipt
SHA-256, and verification status.
