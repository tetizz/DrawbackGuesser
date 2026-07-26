# Evaluator-backed turn constraints

## Status and scope

This document defines the architecture required for drawbacks whose legal-move
set depends on a deterministic chess-engine evaluation. It is a design contract,
not an implementation claim. The three catalog entries reviewed for this work
remain `unsupported` until the implementation, tests, and rule-source
requirements in this document are satisfied.

The observed descriptions currently available in
`data/catalog/observed-drawbacks.json` are:

- **Hand and Gigabrain:** “You must move the piece type that Stockfish
  recommends.”
- **Ichtyophobe:** “You can't make the move that stockfish would make.”
- **Devil on Your Shoulder:** “A devil is suggesting terrible moves for you to
  make. If you disobey it 7 turns in a row, you must obey in the 8th.”

Those descriptions are the only rule semantics assumed here. In particular,
this design does not equate “the devil” with Stockfish, define “terrible,” choose
one suggestion versus a set of suggestions, or define how suggestions persist.
Devil on Your Shoulder must therefore remain unsupported until a better source
answers those questions.

The first two descriptions also leave engine version, search budget, option
settings, and tie handling unspecified. DrawbackEngine must publish a versioned
platform interpretation for those operational details and mark the rules
`implemented-unverified`, not `verified`, until rule-specific evidence confirms
the interpretation.

## Architectural decision

Do not make `DrawbackRule` asynchronous.

The existing synchronous interface is the correct boundary for rules whose
legal masks are pure functions of position, parameters, and state. Adding a
promise to `filterLegalMoves` would force asynchronous behavior into every rule,
the symbolic predictor, diagnostic search, the current game loop, and all
synchronous simulations.

Instead, add a second, explicitly discriminated rule capability and an
asynchronous session orchestrator:

```ts
export interface ExternalTurnConstraintRequest {
  readonly provider: "uci-best-move";
  readonly policyId: string;
  readonly fen: string;
  readonly ordinaryRootMoves: readonly string[];
}

export interface ExternalTurnConstraint {
  readonly provider: "uci-best-move";
  readonly policyId: string;
  readonly positionKey: string;
  readonly bestMoveUci: string;
  readonly engineFingerprint: string;
}

export interface ExternalConstraintDrawbackRule<State, Parameters> {
  readonly kind: "external-turn-constraint";
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly verification:
    | "verified"
    | "implemented-unverified"
    | "partial";

  generateParameters(rng: RandomSource): Parameters;
  initialize(context: RuleInitializationContext<Parameters>): State;

  requestTurnConstraint(
    context: RuleMoveContext<State, Parameters>,
    ordinaryMoves: readonly ChessMove[],
  ): ExternalTurnConstraintRequest;

  filterLegalMovesWithConstraint(
    context: RuleMoveContext<State, Parameters>,
    ordinaryMoves: readonly ChessMove[],
    constraint: ExternalTurnConstraint,
  ): readonly ChessMove[];

  applyMove(
    context: RuleTransitionContext<State, Parameters>,
    move: ChessMove,
  ): State;

  checkStartOfTurnLoss(
    context: RuleLossContext<State, Parameters>,
  ): DrawbackLoss | null;

  explainMove?(
    context: RuleMoveContext<State, Parameters>,
    move: ChessMove,
    constraint: ExternalTurnConstraint,
  ): RuleEvidence[];
}
```

The generic contract belongs in `drawback-engine`; it must not import a Node
process transport, Stockfish implementation, React, or browser code. The
concrete evaluator provider belongs beside `chess-evaluator`. The executable
catalog must distinguish synchronous rules from external-constraint rules so a
synchronous caller can never accidentally run an evaluator-backed rule as an
unrestricted fallback.

The provider boundary is:

```ts
export interface TurnConstraintProvider {
  resolve(
    request: ExternalTurnConstraintRequest,
    options?: { readonly signal?: AbortSignal },
  ): Promise<ExternalTurnConstraint>;

  dispose(): Promise<void>;
}
```

The provider receives only public position data and an ordinary-legal root
mask. It does not receive a drawback ID, rule parameters, rule state, player
identity, prediction state, or true label.

## Prepared-turn session sequence

Introduce an `AsyncGameSession` (or an equivalently named facade over a shared
session kernel) for sessions that can contain external-constraint rules.
Existing `GameSession` remains synchronous and rejects external-constraint rules
at construction.

For each turn, the asynchronous session performs exactly this sequence:

1. Evaluate start-of-turn drawback losses whose evaluation does not need the
   external constraint.
2. Evaluate ordinary standard-chess terminal conditions.
3. Generate ordinary legal moves once.
4. If the active rule is synchronous, apply its normal filter.
5. If the active rule needs an external constraint, construct the request from
   the pre-move FEN and the ordinary legal moves, then await the provider.
6. Validate that the returned policy and position keys match the request and
   that `bestMoveUci` is one of the ordinary root moves.
7. Apply the rule's pure prepared filter to a fresh move array.
8. If the resulting drawback-legal set is empty, apply the existing drawback
   no-legal-move loss behavior.
9. Accept only a move in the prepared legal set.
10. Apply the standard chess move, update drawback state, switch turns, and
    invalidate the prepared turn.
11. Prepare the next turn before reporting its legal moves or accepting another
    move.
12. Emit the public observation and update prediction hypotheses.

Steps 1 and 2 must retain the current engine's established precedence after a
focused compatibility review. In particular, no evaluator process should be
started when ordinary chess already has no legal moves.

Preparation is cached by immutable session revision, exact pre-move FEN, active
rule policy, and sorted ordinary root mask. Repeated or concurrent
`legalMoves()` calls for the same revision share one promise. `move()` awaits
that same promise and rechecks the revision before applying the move. A result
for an earlier FEN or revision is stale and must be discarded, never applied.

React rendering must not call a method that starts an evaluation. A controller
prepares the turn, records `loading | ready | error`, and then renders the
prepared immutable snapshot.

## Rule filtering semantics

### Hand and Gigabrain

The evaluator searches the complete ordinary-legal root mask. It must not search
a mask already restricted by the hidden drawback.

After matching the canonical `bestMoveUci` to its `ChessMove`, retain every
ordinary legal move whose `piece` equals the matched move's `piece`. The rule
description says piece **type**, so it does not require the exact recommended
piece or exact recommended move.

For this comparison:

- A promotion is a pawn move; its mover type is `pawn`.
- Castling is a king move.
- En passant is a pawn move.
- Captured-piece type and promotion target type do not affect the mover type.

Because the evaluator's best move is itself in the ordinary root mask, this
filter cannot produce an empty set if the provider result is valid.

The interpretation above remains `implemented-unverified` until corroborating
rule evidence confirms that “piece type” has this meaning.

### Ichtyophobe

The evaluator searches the complete ordinary-legal root mask. Remove exactly
the move matching `bestMoveUci`, including its promotion suffix when present.
Do not remove all moves by the same piece, all moves with equal score, or all
engine principal-variation alternatives.

If the recommended move was the only ordinary legal move, the prepared legal
set is empty and the existing drawback no-legal-move loss behavior applies.

The single canonical best-move interpretation remains
`implemented-unverified` because the source does not define tied evaluations.

### Devil on Your Shoulder

Do not register an executable rule. The current description is insufficient to
define:

- how a “terrible” suggestion is produced;
- whether there is one suggested move or a set;
- whether the suggestion must be standard-chess legal;
- what counts as obeying or disobeying;
- when the seven-turn counter resets;
- whether “8th” means the affected player's eighth turn;
- what happens when the suggestion cannot be obeyed.

No evaluator configuration can resolve these rule-semantic questions. They
require source research before implementation design.

## Deterministic UCI policy

Every evaluator constraint references an immutable policy ID, initially
`stockfish-bestmove-v1`. A dataset, replay, and cache entry must record the full
policy fingerprint, not only the friendly ID.

The fingerprint covers:

- exact engine binary or WASM SHA-256;
- engine-reported name and version;
- evaluator adapter schema version;
- all required UCI options and their values;
- search-limit kind and value;
- root-move ordering rule;
- FEN normalization rule;
- cache schema version.

The initial reproducible policy must use:

- a pinned Stockfish build;
- `Threads = 1`;
- a fixed `Hash` value;
- `Ponder = false`;
- `MultiPV = 1`;
- standard chess (`UCI_Chess960 = false`);
- a fixed node limit, not wall-clock `movetime`;
- lexicographically sorted, deduplicated UCI root moves;
- exact normalized six-field FEN as the position input.

The exact node and hash values are project policy choices, not inferred drawback
semantics. They must be chosen once with performance evidence, documented in
the policy manifest, and changed only under a new policy ID.

Position evaluation must not depend on earlier games, worker assignment, or
cache warm-up. Before every uncached query, clear engine state using
`ucinewgame`, `setoption name Clear Hash`, and an `isready` barrier. A provider
that cannot establish the required options and readiness must fail startup.

`UciClient` therefore needs typed, validated support for required `setoption`
commands, option discovery from the handshake, engine identity capture,
`AbortSignal`, and a readiness barrier. Option names and values come only from
the trusted policy manifest. They are never constructed from user FEN, move
text, or drawback parameters.

Stockfish can change its selected best move across versions even under identical
limits. Such a change is a policy change and must invalidate caches and separate
simulation/train/validation data by fingerprint.

## Cache contract

The content-addressed key is a hash of:

```text
cache schema
policy fingerprint
exact normalized FEN
sorted ordinary root-move list
```

Cache filenames contain only the hash, never raw FEN or move text. Cached values
contain the request digest, full policy fingerprint, best move, optional public
score metadata, adapter version, and a checksum. Readers validate all fields and
verify that the best move belongs to the requested root mask.

Required behavior:

- one process coalesces concurrent identical misses onto one promise;
- writes use a temporary file plus atomic rename;
- corrupt, truncated, wrong-policy, or wrong-request entries are rejected;
- a cache hit performs no UCI search;
- cold and warm cache runs produce identical game records;
- worker count and worker scheduling do not change results;
- two writers producing different values for the same key cause a deterministic
  integrity error rather than last-writer-wins behavior.

For a browser build, an IndexedDB cache may use the same logical schema and
validation. Cache persistence is an optimization; deleting it must not change
results.

## Failure and cancellation behavior

Evaluator-backed legality fails closed. It never silently becomes unrestricted,
uses a random alternative, restores an eliminated hypothesis, or substitutes a
different engine policy.

Use typed failures for at least:

- provider unavailable;
- unsupported required UCI option;
- process or worker startup failure;
- timeout;
- cancellation;
- malformed protocol output;
- best move outside the ordinary root mask;
- policy or position mismatch;
- corrupt cache entry;
- provider disposal.

On timeout or malformed protocol state, terminate and recreate the poisoned
client before a retry. If a retry is enabled, its maximum count and timeout are
fixed policy values and the failure/retry is recorded. Direct UCI search
cancellation issues `stop`, drains the matching `bestmove` where possible, and
otherwise terminates the client so cancelled output cannot satisfy a later
request. A caller cancelling its wait on a shared deterministic cache fill does
not cancel that fill for other coalesced callers; it detaches only that waiter.

The web UI blocks move input, shows an explicit evaluator-unavailable state, and
offers retry. A simulator aborts the affected game with seed, FEN, policy
fingerprint, and typed failure. It must not commit partial game rows as a
completed labeled game. Batch policy determines whether other independent games
continue.

## Prediction and diagnostic search

The predictor remains synchronous after an asynchronous analysis stage has
attached a validated public evaluator fact to the pre-move observation. That
fact is computed from the public FEN and ordinary legal moves under a configured
policy, independently of either player's true drawback.

For each Hand and Gigabrain or Ichtyophobe hypothesis:

1. Require a fact matching the observation FEN, root mask, and policy.
2. Reconstruct the hypothesis's prepared legal mask with the same pure filter
   used by the game engine.
3. Hard-eliminate the hypothesis if the observed move is outside that mask.
4. Otherwise update its state and likelihood normally.

If the fact is absent or unavailable, the hypothesis is **unevaluable**, not
unrestricted and not eliminated. The prediction API needs an explicit
availability state and evidence explaining that exact legality was deferred.
An evaluator failure cannot increase or decrease that hypothesis's probability.

The same public evaluator fact must be generated for both colors and regardless
of the true rule. If it is stored in training data only when the true drawback
is evaluator-backed, field presence leaks the label. Either compute it uniformly
for every eligible position or exclude it from model inputs.

Diagnostic search needs an asynchronous, batched counterpart because each
candidate position may require a new public evaluator fact. Until that exists,
recommendations must state that evaluator-backed hypotheses were excluded from
information-gain coverage. They must not model those rules as unrestricted.

## Simulation and worker integration

The current asynchronous agent path is not sufficient: agents can await move
selection, but the current session still calculates legality synchronously.
Evaluator-backed games must use the asynchronous prepared session.

Each simulation worker owns its provider lifecycle. Worker requests contain only
serializable policy IDs and trusted evaluator configuration references, never a
live client. A worker initializes one provider, calls the deterministic reset
sequence for every uncached position, and disposes it on normal completion,
abort, and error.

Game seeds continue to determine drawback choice, parameters, bot behavior, and
temperature sampling. Evaluator output is not randomized and consumes no game
RNG values. Adding or removing a cache hit therefore cannot shift the RNG
sequence.

Datasets and replay records include:

- evaluator policy ID and fingerprint;
- public request digest;
- canonical best move or a public prepared-constraint digest;
- cache hit/miss for diagnostics only;
- typed evaluator failure for aborted games.

Cache hit/miss, elapsed time, process ID, and worker ID are not model features.
Validation and test splits must reject records created under an incompatible
policy fingerprint unless the evaluation explicitly studies cross-policy drift.

## Web integration and secrecy

The browser implementation uses a dedicated Web Worker and pinned Stockfish
WASM asset. It does not expose Node process spawning to the browser bundle.
Required controls include a restrictive content security policy, no evaluator
network access, pinned asset integrity, bounded worker lifetime, and explicit
cancellation when a game or replay changes.

The evaluator request contains only public information. Logs, cache entries,
worker messages, and errors must not contain the true drawback, hidden
parameters, private rule state, or posterior labels.

Although a best move is derived from public information, displaying it can give
one player unintended chess assistance. During normal play:

- do not display raw engine best moves, scores, or principal variations;
- show Hand and Gigabrain's required piece type only to the player who owns that
  drawback, as part of their private turn instruction;
- do not reveal why the opponent's legal set was restricted;
- keep training-mode engine information behind an explicit mode that is not
  presented as assistance for external competitive play.

Executable paths are trusted CLI/server configuration only. Continue using
`shell: false`; do not download binaries at runtime. Bound and sanitize stderr.
Cache roots are trusted fixed directories and cache keys cannot influence path
traversal.

## Migration sequence

1. Add a sourced rule note and versioned evaluator-policy manifest. Keep all
   three catalog entries unsupported.
2. Extend `chess-evaluator` with option discovery/configuration, identity and
   binary fingerprinting, abort/stop handling, reset barriers, and the
   deterministic content-addressed provider/cache.
3. Add the external-constraint rule contract to `drawback-engine`, plus pure
   Hand and Gigabrain and Ichtyophobe prepared filters. A missing constraint
   throws a typed error; there is no unrestricted implementation.
4. Refactor the session transition internals so synchronous `GameSession` and
   `AsyncGameSession` share sequencing without duplicating the rules. Reject
   external rules in the synchronous constructor.
5. Add the asynchronous public-fact analysis stage and unevaluable state to the
   symbolic predictor. Preserve the invariant that hard-eliminated hypotheses
   never regain probability.
6. Route evaluator-backed self-play through asynchronous sessions. Add provider
   lifecycle and policy configuration to workers, then prove determinism across
   cache state and worker counts.
7. Add the browser WASM worker, controller loading/error states, cancellation,
   and secrecy tests. Do not evaluate during React rendering.
8. Add asynchronous batched evaluator coverage to diagnostic search, or expose
   an exact coverage limitation.
9. Only after all required rule tests and replay fixtures pass, register Hand
   and Gigabrain and Ichtyophobe as executable
   `implemented-unverified` rules. Devil on Your Shoulder remains unsupported
   pending source clarification.

## Required tests

### Evaluator protocol and policy

- Exact `uci`, required `setoption`, `isready`, reset, position, search, and
  shutdown command ordering.
- Startup rejects a missing or unsupported required option.
- Root moves are deduplicated and sorted before search and keying.
- Fixed policy plus fixture position produces the expected canonical best move.
- Promotion UCI strings remain distinct and are validated exactly.
- A best move outside the root mask is rejected.
- Mock-UCI tests run in normal CI; a pinned-binary golden integration test runs
  wherever the licensed binary is explicitly provisioned.

### Cache and determinism

- A valid hit performs no engine search.
- Concurrent identical requests perform one search.
- FEN, root-mask, policy, and binary-fingerprint changes create distinct keys.
- Corrupt, truncated, mismatched, and wrong-policy entries fail validation.
- Conflicting concurrent writers produce an integrity error.
- Cold-cache and warm-cache results are byte-identical.

### Rule behavior

- Hand and Gigabrain retains all and only moves of the recommended mover type.
- Hand handles pawn promotion, castling as king, en passant as pawn, captures,
  and multiple pieces of the same type.
- Ichtyophobe removes exactly one from/to/promotion move and preserves all
  others.
- Ichtyophobe with one ordinary legal move produces the established drawback
  no-legal-move loss.
- Both filters leave their ordinary input arrays and move objects unchanged.
- Every returned move is ordinary-chess legal.

### Session behavior

- Repeated and concurrent legal-move reads share one preparation.
- `move()` waits for preparation and rejects a stale result after revision
  change, reset, replay navigation, or cancellation.
- Standard terminal positions start no evaluator search.
- Start-of-turn drawback losses and no-legal-move losses preserve documented
  precedence.
- White and Black prepared results, private state, and cancellation remain
  isolated.

### Failure behavior

- Timeout, malformed output, process exit, worker crash, invalid best move,
  abort, and disposal never return unrestricted moves.
- A poisoned client is recreated before a fixed retry.
- UI input remains blocked during failure and retry.
- A failed simulation writes no completed labeled game.

### Predictor and leakage

- Both colors use only the matching pre-move public fact.
- A missing fact marks a hypothesis unevaluable without probability evidence.
- An impossible observed move hard-eliminates the corresponding prepared
  hypothesis.
- A hard-eliminated hypothesis cannot be restored by neural scores.
- Provider requests, cache entries, logs, and browser messages contain no rule
  ID, true label, hidden parameters, or private internal state.
- Training features cannot infer the label from evaluator-field presence.

### Simulation and web

- Fixed seeds produce byte-identical games with one worker and multiple workers.
- Fixed seeds produce byte-identical games with cold and warm caches.
- Provider initialization, per-query reset, abort, and disposal are exercised
  in worker smoke tests.
- Browser component tests cover loading, ready, error, retry, cancellation, and
  unmount.
- No evaluation starts during React render.
- Normal-play UI never exposes an opponent's recommended move, piece type,
  engine score, or principal variation.
- Browser evaluator tests reject network fetches outside the pinned local WASM
  asset.

## Review evidence

This design was based on direct inspection of:

- `packages/drawback-engine/src/types.ts`
- `packages/chess-core/src/game-session.ts`
- `packages/chess-evaluator/src/client.ts`
- `packages/chess-evaluator/src/node-process-transport.ts`
- `packages/simulation/src/stockfish-agent.ts`
- `packages/simulation/src/async-simulation.ts`
- `packages/simulation/src/simulation.ts`
- `packages/simulation/src/parallel.ts`
- `packages/simulation/src/worker-entry.ts`
- `packages/predictor/src/predictor.ts`
- `packages/probe-search/src/probe-search.ts`
- `apps/web/src/App.tsx`
- `data/catalog/observed-drawbacks.json`

Representative inspection commands:

```powershell
rg -n '"hand-and-gigabrain"|"ichtyophobe"|"devil-on-your-shoulder"' data packages docs apps
Get-Content packages/drawback-engine/src/types.ts
Get-Content packages/chess-core/src/game-session.ts
Get-Content packages/chess-evaluator/src/client.ts
Get-Content packages/simulation/src/async-simulation.ts
Get-Content packages/predictor/src/predictor.ts
Get-Content packages/probe-search/src/probe-search.ts
rg -n 'new GameSession|legalMoves\(|\.move\(' apps/web/src packages/simulation/src
```
