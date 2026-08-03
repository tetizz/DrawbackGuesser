# Schema 9 execution readiness

Status: **READY for an authenticated contract smoke; HOLD for accuracy-scale
claims**. The Engine now publishes each frozen split as an atomic trace,
launch-receipt, and completion-receipt bundle. The deterministic scheduler,
trace converter, corpus authenticator, and ledger verifier consume that bundle
without a hand-written provenance step. A 25-game-per-split run proves the
boundary only; it is too small for a trustworthy model-accuracy claim.

This document is derived from public source and tests. It deliberately does
not depend on, inspect, or describe any pre-existing non-public artifact.

## Executable path

The intended data path is:

1. Pin a clean Guesser commit and its exact `engine` gitlink commit.
2. For each ledger split, use the Engine Schema 9 bundle CLI to create a
   canonical NDJSON simulation trace and both authenticated receipts.
3. Convert each trace with the Guesser dataset CLI under
   `capturable-king/v1` authority and evaluator `none`.
4. Supply the Engine-owned launch and completion receipts that authenticate
   the declared schedule, complete generation configuration, producer commit,
   and exact trace bytes.
5. Create the four-split ledger. Its verifier reconstructs the schedule,
   reparses every source game, reconverts every represented game, checks exact
   converted bytes, checks split disjointness, and reproduces the executing
   code identity in isolated checkouts.
6. Preserve the ledger SHA-256 and the TypeScript verification-receipt SHA-256
   printed by the CLI as the public handoff to downstream training.

All six steps have checked-in command paths. Corpus scale and downstream model
claims remain separately gated by measured held-out runs.

## Authenticated inputs and outputs

The ledger CLI requires seven global values and nine values for every one of
`train`, `validation-a`, `validation-b`, and `test`.

Global values:

- operation: `create` or `verify`;
- output/existing ledger path;
- Guesser repository path and immutable commit;
- Engine repository path and the Engine commit used for conversion;
- converter relationship policy, preferably `exact/v1`.

Per-split values:

- canonical Engine trace;
- converted training NDJSON;
- launch receipt and completion receipt;
- schedule ID;
- label, gameplay, and parameter seed roots;
- producer Engine commit.

All sixteen artifact paths must be distinct across the four splits. Receipts
are authenticated by raw byte identity, not only parsed values. The version-2
launch receipt must declare authority
`capturable25-schema9-opportunity/v1`, Engine split `train`, a positive train
count, zero validation/test counts, and the exact frozen generation config.
Every schema-2/ruleset-2 source trace must realize that config for both
players. The completion receipt binds the launch receipt hash plus the exact
output trace hash, byte count, game count, and contiguous index range.

The public validators impose these per-file limits:

- trace line: 64 MiB;
- converted dataset line: 8 MiB;
- launch or completion receipt: 1 MiB;
- complete ledger: 8 MiB.

These are rejection limits, not expected file-size estimates.

## Deterministic lineage

The schedule contract is `material-player-private-corpus/v1` with stream order
`label`, `gameplay`, `parameters`. The four roots are fixed:

| Ledger split | Label root | Gameplay root | Parameter root |
| --- | ---: | ---: | ---: |
| `train` | 1261462769 | 242269024 | 1837697911 |
| `validation-a` | 2069246597 | 1391196133 | 2739675947 |
| `validation-b` | 3786384219 | 3547865132 | 2689552677 |
| `test` | 2033321041 | 1354035545 | 4189758462 |

For each split, the scheduler deterministically derives the game ID, gameplay
seed, separate White and Black parameter seeds, ordered drawback assignments,
and initial replay hash from the root tuple and zero-based game index. A
25-game block balances both marginal drawback labels. A 625-game block covers
every ordered White/Black drawback pair. Seed and game-ID sets are required to
be disjoint across ledger splits.

The ledger intentionally invokes the Engine CLI split named `train` for every
ledger split. Dataset separation comes from the four disjoint root tuples and
four separate output paths; the receipt authenticates that Engine invocation.

## Public-safe command sequence

Run from the Guesser repository root in PowerShell. Keep all generated data in
an explicit directory outside both repositories. The commands below neither
select nor inspect a pre-existing dataset.

### 1. Pin and build the public code

```powershell
$repo = (Resolve-Path '.').Path
$engine = (Resolve-Path (Join-Path $repo 'engine')).Path

if ([string]::IsNullOrWhiteSpace($env:SCHEMA9_RUN_ROOT)) {
  throw 'Set SCHEMA9_RUN_ROOT to a new directory outside the repositories.'
}
$runRoot = [IO.Path]::GetFullPath($env:SCHEMA9_RUN_ROOT)
$repoPrefix = $repo.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if ($runRoot -eq $repo -or $runRoot.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
  throw 'SCHEMA9_RUN_ROOT must be outside the Guesser and Engine repositories.'
}
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$guesserCommit = (git -C $repo rev-parse HEAD).Trim()
$engineCommit = (git -C $engine rev-parse HEAD).Trim()
$pinnedEngineCommit = ((git -C $repo ls-tree $guesserCommit -- engine) -split '\s+')[2]
if ($engineCommit -ne $pinnedEngineCommit) {
  throw "Engine HEAD $engineCommit does not match gitlink $pinnedEngineCommit."
}

if (git -C $repo status --porcelain=v1 --untracked-files=no --ignore-submodules=none) {
  throw 'Guesser tracked worktree or index is dirty.'
}
if (git -C $engine status --porcelain=v1 --untracked-files=no --ignore-submodules=none) {
  throw 'Engine tracked worktree or index is dirty.'
}

pnpm engine:build
if ($LASTEXITCODE -ne 0) { throw 'Engine build failed.' }
pnpm --filter '@drawbackguesser/trace-to-dataset' build
if ($LASTEXITCODE -ne 0) { throw 'Trace converter build failed.' }
```

The ledger verifier also rejects a mismatched `HEAD`, Git replacement refs,
skip-worktree or assume-unchanged index flags, weakened submodule-ignore
configuration, and a non-submodule Engine path.

### 2. Generate one split deterministically

The Engine owns the roots and complete generation configuration. The caller
chooses only a ledger split, a complete 25-label game-cycle count, bounded
worker concurrency, a split-specific path-free schedule ID, and an absent
bundle path.

```powershell
$scheduleIds = @{
  'train' = 'schema9-smoke-v2-train'
  'validation-a' = 'schema9-smoke-v2-validation-a'
  'validation-b' = 'schema9-smoke-v2-validation-b'
  'test' = 'schema9-smoke-v2-test'
}
$seedRoots = @{
  'train' = [uint32[]]@(1261462769, 242269024, 1837697911)
  'validation-a' = [uint32[]]@(2069246597, 1391196133, 2739675947)
  'validation-b' = [uint32[]]@(3786384219, 3547865132, 2689552677)
  'test' = [uint32[]]@(2033321041, 1354035545, 4189758462)
}

function Invoke-Schema9Split {
  param(
    [Parameter(Mandatory)] [string] $Name,
    [Parameter(Mandatory)] [int] $Games,
    [Parameter(Mandatory)] [int] $Workers
  )

  if ($Games -lt 25 -or ($Games % 25) -ne 0) {
    throw 'Games must be a positive multiple of 25.'
  }
  if ($Workers -lt 1 -or $Workers -gt 256) {
    throw 'Workers must be between 1 and 256.'
  }
  if (-not $seedRoots.ContainsKey($Name)) {
    throw "Unknown Schema 9 split $Name."
  }
  if (-not $scheduleIds.ContainsKey($Name)) {
    throw "Missing Schema 9 schedule ID for $Name."
  }

  $bundle = Join-Path $runRoot "$Name.bundle"
  $trace = Join-Path $bundle 'trace.ndjson'
  $converted = Join-Path $runRoot "$Name.converted.ndjson"
  $scheduleId = $scheduleIds[$Name]

  pnpm --dir $engine --filter '@drawbackengine/cli' player-private:schema9 -- `
    --ledger-split $Name `
    --games $Games `
    --workers $Workers `
    --schedule-id $scheduleId `
    --bundle $bundle `
    --engine-repository $engine
  if ($LASTEXITCODE -ne 0) { throw "Simulation failed for $Name." }

  pnpm --filter '@drawbackguesser/dataset-cli' start -- `
    --input $trace `
    --output $converted `
    --require-authority capturable-king/v1 `
    --require-evaluator none
  if ($LASTEXITCODE -ne 0) { throw "Conversion failed for $Name." }

  $roots = $seedRoots[$Name]
  [pscustomobject]@{
    Name = $Name
    Games = $Games
    Trace = $trace
    Converted = $converted
    LaunchReceipt = Join-Path $bundle 'launch.json'
    CompletionReceipt = Join-Path $bundle 'completion.json'
    ScheduleId = $scheduleId
    LabelRoot = $roots[0]
    GameplayRoot = $roots[1]
    ParameterRoot = $roots[2]
  }
}

$parallelism = [int](node -p "require('node:os').availableParallelism()")
$workers = [Math]::Min(256, [Math]::Max(1, $parallelism - 1))
$train = Invoke-Schema9Split train 25 $workers
$validationA = Invoke-Schema9Split validation-a 25 $workers
$validationB = Invoke-Schema9Split validation-b 25 $workers
$test = Invoke-Schema9Split test 25 $workers
```

The four 25-game calls are the smallest contract-valid smoke corpus. Replace
only the game counts for a larger run; do not change the roots or policy
arguments, and keep all four schedule IDs distinct. The producer requires an
existing trusted parent directory, refuses
output inside the Engine checkout, revalidates the same clean Engine commit
before publication, hashes the closed trace again, and publishes the
three-file bundle at one irreversible rename commit point.

### 3. Create and verify the ledger

The following argument assembly matches every required CLI flag and consumes
the Engine-owned receipts directly.

```powershell
$splits = @($train, $validationA, $validationB, $test)
$ledger = Join-Path $runRoot 'schema9-corpus-ledger.json'

function Get-LedgerArguments {
  param([Parameter(Mandatory)] [ValidateSet('create', 'verify')] [string] $Operation)

  $arguments = [System.Collections.Generic.List[string]]::new()
  $arguments.AddRange([string[]]@(
    '--operation', $Operation,
    '--ledger', $ledger,
    '--guesser-repository', $repo,
    '--engine-repository', $engine,
    '--guesser-commit', $guesserCommit,
    '--converter-engine-commit', $engineCommit,
    '--producer-converter-policy', 'exact/v1'
  ))
  foreach ($split in $splits) {
    $prefix = "--$($split.Name)"
    $arguments.AddRange([string[]]@(
      "$prefix-trace", $split.Trace,
      "$prefix-converted", $split.Converted,
      "$prefix-launch-receipt", $split.LaunchReceipt,
      "$prefix-completion-receipt", $split.CompletionReceipt,
      "$prefix-schedule-id", $split.ScheduleId,
      "$prefix-label-seed-root", [string]$split.LabelRoot,
      "$prefix-gameplay-seed-root", [string]$split.GameplayRoot,
      "$prefix-parameters-seed-root", [string]$split.ParameterRoot,
      "$prefix-producer-engine-commit", $engineCommit
    ))
  }
  return $arguments.ToArray()
}

pnpm --filter '@drawbackguesser/dataset-cli' start:ledger -- @(Get-LedgerArguments create)
if ($LASTEXITCODE -ne 0) { throw 'Schema 9 ledger creation failed.' }

pnpm --filter '@drawbackguesser/dataset-cli' start:ledger -- @(Get-LedgerArguments verify)
if ($LASTEXITCODE -ne 0) { throw 'Schema 9 ledger verification failed.' }
```

The successful JSON output contains the ledger digest and verification-receipt
digest. Downstream training must pin both, along with the immutable Guesser and
Engine commits, rather than trusting path names.

## Code-derived resource envelope

There is no checked-in representative throughput or serialized-size benchmark,
so wall-clock, RAM, and disk forecasts would be invented. The following are
safe upper-bound calculations from the configured command path; a node budget
is work allowed to the search, not proof that every node is visited.

| Plan | Source games | Maximum trace plies / converted rows | Maximum search-node budget |
| --- | ---: | ---: | ---: |
| Small contract smoke, 25 per split | 100 | 12,000 | 600,000,000 |
| Ordered-pair coverage, 625 per split | 2,500 | 300,000 | 15,000,000,000 |
| Symmetric planning example, 2,500 per split | 10,000 | 1,200,000 | 60,000,000,000 |

The downstream validation workflow requires exactly 2,500 represented games
for each of `validation-a`, `validation-b`, and `test`: 7,500 represented games
in total. It does not freeze a train count. If `T` is the chosen train source
count and the three validation/test source counts are each 2,500, the CLI
settings imply at most:

```text
rows or plies = 120 * (T + 7,500)
search-node budget = 50,000 * rows = 6,000,000 * (T + 7,500)
```

The symmetric 2,500-per-split row in the table is therefore an example, not a
protocol requirement. A zero-ply source game produces no converted training
game, so an execution plan also needs a deterministic top-up/resume policy if
zero-ply games are possible; none is currently exposed by the batch CLI.

Concurrency is bounded as follows:

- the producer requires an explicit worker count; this helper selects
  `max(1, available parallelism - 1)`, capped at 256;
- default/current coordinator window: four times the worker count;
- coordinator retains at most one window of assignments/results;
- each worker policy configures a 16,384-entry leaf cache, but JavaScript entry
  size is not specified, so its byte footprint cannot be stated honestly;
- a failed worker shard may be attempted up to three times;
- ledger authentication retains game IDs, seeds, replay hashes, and source
  game records, making verifier memory grow with game count;
- isolated reproducibility runs four child commands with individual ten-minute
  timeouts. That is not a total forty-minute guarantee because cloning,
  parsing, authentication, and replay are outside those command timeouts.

Before a full run, benchmark the 25-game smoke on the target machine and record
elapsed time, peak resident memory, trace bytes, converted bytes, rows, and
games. Scale planning from that receipt-backed measurement, then confirm with
a 625-game run because search positions and game lengths are not uniform.

## Release blockers

1. **No permanent CLI-to-ledger integration gate.** The focused schema 9
   integration test builds traces from an in-memory fixture and constructs
   receipts in the test. It does not execute the Engine batch CLI, streaming
   trace writer, conversion CLI, or receipt-producing bundle command. The
   release procedure must retain one measured external four-split smoke until
   this becomes a bounded automated test.
2. **Full train size is not frozen.** Validation/test represented counts are
   fixed at 2,500 each; train is only required to be a positive label-balanced
   multiple. There is no single reproducible definition of a “full” corpus or
   honest full-run duration/disk estimate yet.
3. **Zero-ply top-up is unspecified.** Source counts and represented converted
   counts can diverge. The CLI has no deterministic continuation range or
   receipt-bound top-up workflow.
4. **Offline verification has operational prerequisites.** Reproduction needs
   the pinned pnpm runtime and all dependencies in the offline store, plus clean
   exact commits and the real Engine submodule checkout. Missing cached
   dependencies block verification even when the corpus bytes are correct.

## Required release gate

An accuracy-scale corpus remains on HOLD until all of the following pass from
a clean, pinned checkout:

- the four-split 25-game Engine/convert/ledger smoke completes and its exact
  commits, digests, throughput, memory, and byte measurements are retained;
- a bounded regression executes the real producer-to-consumer path in CI;
- tampering with any receipt, trace byte, converted byte, seed, schedule item,
  policy value, commit, or split membership fails closed;
- interrupted execution leaves no accepted receipt or partial published file;
- the train count and deterministic zero-ply continuation policy are frozen
  before the larger validation/test execution is authorized.

## Current focused verification

With public dependencies built first, the schema 9 suite command is:

```powershell
pnpm engine:build
pnpm --filter '@drawbackguesser/trace-to-dataset' build
pnpm exec vitest run `
  packages/trace-to-dataset/src/schema9-corpus-ledger.test.ts `
  packages/trace-to-dataset/src/schema9-schedule-replay.test.ts `
  packages/trace-to-dataset/src/schema9-real-integration.test.ts
```

At the receipt-v2 contract commit, the focused suite passes 3 files and 68
tests, while the complete trace-to-dataset package passes 6 files and 89 tests.
Those tests validate tamper rejection and exact trace-policy binding. The
external four-split smoke remains the proof of the actual child-process path
until a bounded end-to-end CI regression is added.

Primary implementation evidence:

- `packages/trace-to-dataset/src/schema9-ledger-types.ts`
- `packages/trace-to-dataset/src/schema9-schedule-replay.ts`
- `packages/trace-to-dataset/src/schema9-ledger-authentication.ts`
- `packages/trace-to-dataset/src/schema9-corpus-ledger.ts`
- `apps/dataset-cli/src/schema9-ledger-cli.ts`
- `engine/apps/engine-cli/src/schema9-player-private-cli.ts`
- `engine/apps/engine-cli/src/schema9-player-private-bundle.ts`
- `engine/packages/simulation-arena/src/player-private-assignment-scheduler.ts`
- `engine/packages/simulation-arena/src/player-private-stream.ts`
- `ml/training/drawback_ml/capturable_opportunity_workflow.py`
