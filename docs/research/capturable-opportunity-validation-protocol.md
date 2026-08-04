# Capturable Opportunity Validation Protocol

Status: preregistered workflow contract, version 2.

This protocol governs the schema-9 opportunity ablation only. It preserves
the three model seeds, training configuration, two public validation stages,
and one sealed-test access. It does not authorize a release by itself.

## Frozen experiment

- Domain: `capturable25-schema9-opportunity-v1`
- Model seeds: `3685459371`, `480184104`, `3192956725`
- Training: 8 epochs, batch size 256, hidden dimension 128, trigger-row
  multiplier 1.0
- Seed stream order: `label`, `gameplay`, `parameters`
- Train roots: `1261462769`, `242269024`, `1837697911`
- Validation-A roots: `2069246597`, `1391196133`, `2739675947`
- Validation-B roots: `3786384219`, `3547865132`, `2689552677`
- Sealed-test roots: `2033321041`, `1354035545`, `4189758462`
- Validation-A, validation-B, and test each contain exactly 2,500 converted
  games.

The train roots were derived with the same frozen rule as every other split:
the first four bytes of
`SHA-256(capturable25-schema9-opportunity-v1/<split>/<stream>)`, interpreted
as an unsigned big-endian integer. The corpus-ledger split name `test` maps
to the workflow domain token `sealed-test`.

## Required corpus ledger

Every command requires an explicit corpus-ledger artifact, its
caller-authenticated file SHA-256, and the caller-authenticated SHA-256 of the
TypeScript verification receipt. Digest strings without both artifacts are
not sufficient.

The accepted ledger is canonical
`drawbackguesser-schema9-corpus-ledger` version 3 with:

- a valid self-hash over the canonical payload excluding `contentSha256`;
- exact top-level and nested fields, with unknown fields rejected;
- the exact schema-9 opportunity rule IDs, fields, shape, and authority;
- the exact schedule authority, seed-stream order, and roots above;
- canonical train, validation-A, validation-B, and test ordering;
- exact enumerable source and converted game-ID and simulation-seed sets,
  plus matching canonical set hashes;
- exact converted dataset SHA-256, byte, row, and game counts;
- authenticated launch and completion receipt identities;
- exact per-label game counts for both colors, including zero-ply games;
- source/converted accounting in which converted games equal source games
  minus zero-ply games;
- pairwise game-ID disjointness plus global uniqueness of simulation and both
  player-parameter seed streams across all four source splits, with
  independently recomputed partition commitments; and
- full Guesser, converter, and producer commit identities;
- one exact producer runtime-tree identity, repeated by every launch and
  completion receipt and reproduced from each recorded Engine commit;
- a content-derived parser/converter/scheduler/verifier execution manifest;
  and
- exact Engine-scheduler replay of every game index, seed, parameter seed,
  label pair, game ID, profile, and initial position.

Every stage also requires the direct-sibling TypeScript verification receipt
version 2,
`schema9-ledger-verification-<ledger-file-sha256>.json`. Python verifies its
actual file bytes against the caller-supplied receipt SHA-256, then verifies
its canonical self-hash, ledger file/content binding, complete input-set
commitment, repository policy, and execution identity before accepting the
ledger. The authenticated receipt digest is bound into Stage A and therefore
propagates through Stage B, consumption markers, and sealed reports.

The Python workflow accepts `producerConverterPolicy: exact/v1` only and
requires every producer commit to equal the converter commit. The ledger
producer also supports `converter-ancestor/v1`, but this workflow rejects
that policy because it does not receive an independently authenticated
Engine repository with which to replay ancestry. Adding that support
requires an explicit versioned protocol change; it must never be inferred
from the ledger's assertion alone.

Stage A binds the authenticated train and validation-A ledger identities to
the three selection comparisons. Stage B and the sealed test additionally
recompute the loaded dataset's exact game-ID and simulation-seed sets and
compare them with the ledger. A dataset using a claimed frozen root but
containing another seed, including seed 7, is rejected.

## Stage transitions

Stage A authenticates exactly three same-seed control/treatment comparisons,
one for each frozen model seed. At least two pairs must be eligible and
promote the treatment. Aggregate gates require:

- positive mean Top-1;
- non-regressing Top-3 and Top-5;
- non-regressing negative log likelihood, Brier score, calibration, and all
  move horizons; and
- for both White and Black independently, Top-1 at least 0, Top-3 at least
  0, and negative-log-likelihood delta at most 0.

The selected pair is the lower median of eligible promoted pairs by Top-1
delta. Ties use lower negative-log-likelihood delta and then lower model
seed.

Stage B reauthenticates Stage A, the ledger, the frozen checkpoints, and the
validation-B bytes. Only the exact authorization
`sealed-test-authorized` permits sealed-test access. A blocked Stage B is
also rejected when loading an existing sealed report.

## One sealed-test access

The consumption marker is a create-only durable artifact named:

`sealed-test-consumption-<sealed-corpus-identity-sha256>.json`

Its identity hashes the protocol/version, authenticated ledger file/content
hashes, test converted digest and set commitments, and test schedule identity.
It is independent of Stage-A/Stage-B filenames, serializations, or the selected
frozen pair. Its content records the authorizing Stage-B SHA-256, exact
authorization, ledger, protocol, and frozen pair. The marker is published before the test
path is resolved, opened, hashed, or inspected. Publication races are
resolved by create-only filesystem semantics.

Once marker publication succeeds, access is consumed even if inference or
final-report publication fails. Changing a report name, aliasing Stage A,
serializing a second Stage B, or freezing another otherwise valid pair against
the same ledger/test cannot create another marker and cannot reopen the test.

By default the trusted local consumption registry is
`<Git common directory>/drawbackguesser/sealed-corpus-consumption-v1`; callers
may explicitly provide another registry. It prevents accidental or repeated
use only while trusted evaluators share and preserve that exact directory. A
repository owner can delete or replace the marker, and another clone has a
different Git common directory. Global one-shot enforcement requires an
external append-only authority, or a signed single-use lease issued by that
authority, keyed by sealed-corpus identity. This local workflow does not claim
that guarantee.

## Final report authentication

`verify-sealed-test` requires `--report-sha256`. The value must come from the
create operation's `artifactSha256` result or another caller-controlled
receipt. Verification compares this digest before trusting or recomputing
the report's metrics. Rewriting control/treatment metrics and all derived
deltas into a new internally consistent report therefore fails against the
original caller-authenticated digest.

The verifier still replays the Stage-B, marker, frozen-pair, input-identity,
metric, reliability, and decision contracts. The caller digest is an
additional outer integrity anchor, not a replacement for semantic checks.

## Command shape

All paths shown below must name explicit files. The ledger, workflow
artifacts, and comparisons are direct siblings.

```text
python -m drawback_ml.capturable_opportunity_workflow stage-a \
  --comparison pair-1.json --comparison pair-2.json --comparison pair-3.json \
  --corpus-ledger schema9-corpus-ledger.json \
  --corpus-ledger-sha256 <ledger-sha256> \
  --corpus-ledger-verification-receipt-sha256 <receipt-sha256> \
  --output stage-a.json

python -m drawback_ml.capturable_opportunity_workflow stage-b \
  --stage-a stage-a.json --validation-b validation-b.ndjson \
  --corpus-ledger schema9-corpus-ledger.json \
  --corpus-ledger-sha256 <ledger-sha256> \
  --corpus-ledger-verification-receipt-sha256 <receipt-sha256> \
  --output stage-b.json

python -m drawback_ml.capturable_opportunity_workflow sealed-test \
  --stage-b stage-b.json --validation-b validation-b.ndjson \
  --test sealed-test.ndjson \
  --corpus-ledger schema9-corpus-ledger.json \
  --corpus-ledger-sha256 <ledger-sha256> \
  --corpus-ledger-verification-receipt-sha256 <receipt-sha256> \
  --output sealed-test-report.json

python -m drawback_ml.capturable_opportunity_workflow verify-sealed-test \
  --report sealed-test-report.json --report-sha256 <report-sha256> \
  --stage-b stage-b.json --validation-b validation-b.ndjson \
  --corpus-ledger schema9-corpus-ledger.json \
  --corpus-ledger-sha256 <ledger-sha256> \
  --corpus-ledger-verification-receipt-sha256 <receipt-sha256>
```

CLI JSON emits basenames only. Published artifacts recursively reject
absolute Windows paths, absolute POSIX paths, and directory-bearing Windows
or POSIX reference fields. Windows alternate-data-stream syntax, control
characters, trailing dots/spaces, and reserved device names are also
rejected. This applies on creation and every load/verify boundary.

## Compatibility

Workflow version-1 Stage-A, Stage-B, sealed-test, and consumption artifacts
are intentionally incompatible with version 2. The ledger path is now
mandatory, sealed-report verification now requires the caller's final
report SHA-256, the consumption-marker name is Stage-B keyed, and bound
dataset identities include schedule and set commitments. There is no
silent legacy fallback.
