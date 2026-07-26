# Variant move and termination authority

## Status and scope

This document specifies the engine boundary required for **Death Wish** and
drawbacks whose observed text literally refers to capturing a king. It does
not normalize king capture to checkmate.

An initial executable `capturable-king/v1` milestone now lives in
`packages/chess-core/src/capturable-king-position.ts`,
`packages/chess-core/src/drawback-game-session.ts`, and
`packages/drawback-search`. It implements the public site's global geometric
move model, direct king capture, castling-en-passant king capture, exact session
forks, and an omniscient offline drawback-aware tree search. The authority
snapshot, predictor, browser, asynchronous-rule, and dataset migrations
described below remain open; see `docs/architecture/drawback-search.md`.
Therefore this document is still partly a migration contract, not a claim that
every consumer has moved.

The current engine cannot implement those rules faithfully:

- `GameSession` asks `chess.js` for ordinary legal moves and permits drawbacks
  only to remove moves from that set.
- `chess.js` excludes moves that leave the moving king attacked, rejects those
  moves when applied, never generates a king capture, and expects both kings in
  positions it parses.
- `SessionResult` has no king-capture terminal result.
- predictor observations contain a standard FEN and a standard-legal move set.
- replay fixtures reject every declared move that is not legal according to
  `chess.js`.

The goal is one authoritative move implementation shared by live sessions,
simulation, predictor replay, diagnostic probing, and fixtures. UI code and
individual drawback rules must not independently reconstruct variant legality.

## Design principles

1. **One authority owns position mutation.** `GameSession` must not apply a
   move through `chess.js` after another component generated it.
2. **Standard chess remains the default authority.** Existing rules, FENs,
   datasets, and tests continue through a `StandardChessAuthority` adapter.
3. **Variant moves are explicit.** A move outside standard chess carries a
   declared availability class. It must never appear accidentally under a rule
   that did not request that capability.
4. **King capture is a real terminal event.** It is serialized and evaluated
   directly; it is not inferred from checkmate notation.
5. **Public variant state is not secret drawback state.** The board, side to
   move, and prior moves remain public predictor inputs. Hidden parameters and
   rule runtime state remain outside the authority snapshot.
6. **Every consumer reuses the same authority contract.** A predictor or probe
   result produced with different move semantics is invalid.
7. **Unsupported combinations fail closed.** Authority selection, snapshot
   decoding, and move application return typed errors rather than silently
   falling back to standard chess.

## Terminology

- **Position authority**: implementation that decodes, validates, generates
  moves for, mutates, and evaluates one family of positions.
- **Authority requirement**: static capability requested by a drawback.
- **Availability class**: why a generated move is present.
- **Base move**: a move admitted by the selected authority before the active
  drawback's normal filtering.
- **Standard-compatible move**: a base move also legal under orthodox chess.
- **Synthetic move**: a base move that `chess.js` would not admit, such as a
  king move into attack.

## Public contracts

The following TypeScript is normative pseudocode. Names may change during
implementation, but ownership and data flow must not.

```ts
type AuthorityId = "standard-chess/v1" | "capturable-king/v1";

type AuthorityCapability =
  | "king-may-enter-attack"
  | "enemy-king-capture";

interface AuthorityRequirement {
  readonly authorityFamily: "standard-chess" | "capturable-king";
  readonly capabilities: readonly AuthorityCapability[];
}

interface SerializedAuthorityPosition {
  readonly authorityId: AuthorityId;
  readonly schemaVersion: 1;
  /** Canonical JSON data owned and validated by the named authority. */
  readonly state: Readonly<Record<string, unknown>>;
  readonly turn: PlayerColor;
}

type MoveAvailability =
  | { readonly kind: "base" }
  | {
      readonly kind: "capability";
      readonly capability: AuthorityCapability;
    };

interface AuthorityMove extends ChessMove {
  /** Stable identity; coordinate notation includes promotion when present. */
  readonly moveId: string;
  readonly availability: MoveAvailability;
  readonly terminalIntent?: "capture-king";
  /**
   * Human display only. Variant notation is not required to be valid SAN and
   * must never be parsed to recover move identity.
   */
  readonly display: string;
}

interface MoveGenerationRequest {
  readonly activeColor: PlayerColor;
  readonly activeCapabilities: readonly AuthorityCapability[];
}

type AuthorityTerminal =
  | {
      readonly kind: "king-capture";
      readonly winner: PlayerColor;
      readonly capturedKing: PlayerColor;
      readonly moveId: string;
    }
  | { readonly kind: "checkmate"; readonly winner: PlayerColor }
  | { readonly kind: "stalemate" }
  | { readonly kind: "draw"; readonly reason: string };

interface AppliedAuthorityMove {
  readonly position: SerializedAuthorityPosition;
  readonly move: AuthorityMove;
  readonly terminal: AuthorityTerminal | null;
}

interface PositionAuthority {
  readonly id: AuthorityId;

  initialPosition(input?: SerializedAuthorityPosition): SerializedAuthorityPosition;

  decode(serialized: SerializedAuthorityPosition): AuthorityPosition;
  encode(position: AuthorityPosition): SerializedAuthorityPosition;

  generateMoves(
    position: AuthorityPosition,
    request: MoveGenerationRequest,
  ): readonly AuthorityMove[];

  applyMove(
    position: AuthorityPosition,
    moveId: string,
    request: MoveGenerationRequest,
  ): AppliedAuthorityMove;

  evaluate(position: AuthorityPosition): AuthorityTerminal | null;

  /** Optional only when the position is losslessly representable as FEN. */
  toFen(position: AuthorityPosition): string | null;
}
```

`applyMove` must regenerate or otherwise validate the move against the same
request before mutation. Accepting an arbitrary `AuthorityMove` object supplied
by a client would permit forged synthetic moves.

### Rule declaration

Authority needs are static rule metadata, not hidden parameters:

```ts
interface DrawbackRule<State, Parameters> {
  // Existing fields omitted.
  readonly authorityRequirement?: AuthorityRequirement;
}
```

Rules without the field require `standard-chess` and no synthetic capability.
Death Wish declares the `capturable-king` family and
`king-may-enter-attack`. Literal king-capture restrictions declare the
`capturable-king` family but do not automatically gain
`king-may-enter-attack`.

The authority resolver receives both players' requirements before a session
starts:

```ts
resolveAuthority(
  white: AuthorityRequirement | undefined,
  black: AuthorityRequirement | undefined,
): PositionAuthority
```

Compatible requirements produce the least permissive authority that satisfies
their union. An incompatible pair throws `AuthorityCompatibilityError` before
parameters or agents consume random numbers. This preserves deterministic RNG
streams on failed setup.

## Standard and variant coexistence

### Standard authority

`StandardChessAuthority` wraps the current `chess.js` calls:

- input/output uses `standard-chess/v1`;
- its canonical state contains FEN;
- every move has `{kind: "base"}`;
- `toFen` always succeeds;
- terminal evaluation maps the existing checkmate, stalemate, repetition,
  insufficient-material, and fifty-move results.

Existing `GameSession` callers may continue to pass a FEN during migration. The
constructor translates it once into a standard authority snapshot.

### Capturable-king authority

`CapturableKingAuthority` must be a board implementation independent of
`chess.js` mutation. It may use `chess.js` only as a differential oracle for
positions and moves that are demonstrably orthodox.

Its state must represent:

- all occupied squares, including piece color and type;
- side to move;
- castling rights;
- en-passant target;
- halfmove/fullmove counters where applicable;
- repetition identity/history required by the chosen draw policy;
- whether each original king is present;
- whether a present king is attacked.

The board remains serializable after one king is removed. Such a position is
terminal and is never passed to `chess.js`.

Move generation has two layers:

1. Generate moves legal for the capturable-king family, including an attack on
   the opposing king as an actual capture.
2. When the active request contains `king-may-enter-attack`, additionally emit
   otherwise geometrically valid moves of that player's king whose destination
   is attacked. Mark them with the capability availability class.

No other synthetic self-check move is enabled. A pinned rook, for example,
does not become movable merely because the king has a Death Wish.

The exact behavior after a Death Wish king enters attack needs a written rule
specification before implementation. The authority must be capable of the
following defensible model without hard-coding it into `GameSession`:

- the exposed king remains on the board;
- the opponent may capture it as a terminal move;
- if it is not captured, the exposed player is considered in check on their
  next turn and receives the authority's normal check-evasion base moves;
- Death Wish's “and aren't already in check” condition is evaluated from the
  authority's attack state.

Whether king capture is compulsory for the opponent is a drawback rule
question, not an authority assumption.

### Capability gating

`generateMoves` receives capabilities only from the active player's rule.
Therefore a player without Death Wish cannot move its king into attack merely
because the opponent selected a capturable-king drawback.

The active rule then filters the generated base set through its existing
`filterLegalMoves` method. Death Wish must:

1. inspect whether the active king is already attacked;
2. find moves marked `king-may-enter-attack`;
3. force that subset when nonempty and the king was not already attacked;
4. otherwise preserve the authority's ordinary base set.

The generic session pipeline validates that the rule returns only moves from
the authority-generated array and cannot manufacture a new move or change its
availability marker.

## Session sequence and termination precedence

The revised sequence is:

1. Decode the current authority position.
2. Generate authority base moves using only the active rule's declared
   capabilities.
3. Pass an immutable copy through the active drawback filter.
4. Reject a command not identified in the authority set or removed by the
   drawback.
5. Ask the same authority to apply the selected `moveId`.
6. Append the normalized authority move to public history.
7. Update the moving player's drawback state.
8. If application returned `king-capture`, end immediately.
9. Otherwise switch turns as encoded by the authority.
10. Evaluate the affected player's start-of-turn drawback loss, including the
    generic zero-filter loss using authority moves.
11. Evaluate remaining authority terminal conditions.
12. Update prediction hypotheses from the public observation.

King capture has immediate precedence because there is no affected king left
to begin a turn. It cannot be superseded by a later start-of-turn drawback
loss. Standard checkmate keeps the project's existing ordering relative to
start-of-turn losses unless a separate reviewed migration changes it.

Extend the result type:

```ts
type SessionResult =
  | { readonly kind: "active" }
  | { readonly kind: "drawback-loss"; readonly loss: DrawbackLoss }
  | { readonly kind: "king-capture"; readonly winner: PlayerColor;
      readonly capturedKing: PlayerColor }
  | { readonly kind: "checkmate"; readonly winner: PlayerColor }
  | { readonly kind: "draw"; readonly reason: string };
```

Move rejection should become authority-neutral:

```ts
reason: "game-over" | "not-authority-legal" | "drawback-forbidden"
```

User messages may still say “not legal in standard chess” only when the active
authority is standard.

## Position and move serialization

FEN and SAN are insufficient as the canonical variant formats:

- common FEN validators reject a missing king and may reject a side that left
  its king attacked through a synthetic move;
- SAN assumes orthodox legality and checkmate rather than king capture.

Dataset, replay, and API records must add:

```ts
interface SerializedMoveObservation {
  readonly authorityId: AuthorityId;
  readonly positionBefore: SerializedAuthorityPosition;
  readonly positionAfter: SerializedAuthorityPosition;
  readonly moveId: string;
  readonly move: AuthorityMove;
  readonly authorityMoves: readonly AuthorityMove[];
  readonly drawbackLegalMoves: readonly AuthorityMove[];
  readonly terminal: AuthorityTerminal | null;
  /** Compatibility field; null when no lossless FEN exists. */
  readonly fenBefore: string | null;
  readonly fenAfter: string | null;
}
```

Canonical JSON rules are required: sorted object keys, stable square ordering,
finite integer counters, and a versioned authority ID. Snapshot decoders reject
unknown fields or versions. `moveId` is coordinate notation
`from + to + promotion` for all currently contemplated moves; the contract
allows a future authority-specific suffix without changing display text.

Dataset schema and symbolic feature versions must be bumped when these fields
or candidate semantics become active. Standard rows may retain non-null FENs
for compatibility, but all new readers use the authority snapshot first.

Authority snapshots contain no hidden drawback ID, parameters, random seed, or
rule state. Training feature parsers must forbid those secret fields exactly as
they do today.

## Predictor parity

The current predictor trusts an externally supplied standard
`ordinaryLegalMoves` array and asks a rule only to filter it. Variant
hypotheses require authority-aware seeds:

```ts
interface AuthorityHypothesisRuntime {
  readonly authorityRequirement: AuthorityRequirement;
  readonly authority: PositionAuthority;
  readonly authorityPosition: SerializedAuthorityPosition;
}
```

For every hypothesis and observed move:

1. decode or replay its public authority position;
2. generate base moves with that hypothesis's active capabilities;
3. apply its rule filter;
4. eliminate it if the observed `moveId` is absent;
5. apply the observed move through that hypothesis's authority;
6. compare the resulting public snapshot and terminal event with the observed
   snapshot/event;
7. transition rule state only after both checks pass.

The legal-rule/authority result remains a hard mask. A likelihood or neural
model cannot restore a hypothesis contradicted by move generation, transition,
or terminal parity.

### Authority-family compatibility

A standard hypothesis presented with a synthetic Death Wish move is
eliminated because its authority cannot generate that `moveId`. It is not an
engine error. Conversely, a capturable-king hypothesis can survive orthodox
moves while maintaining its own replay state.

Hypotheses must not read the true hidden drawback's authority requirement from
secret session data. The observation exposes only the public serialized board
and the move. If the selected match protocol publicly declares its authority
family, that protocol value may be an input; otherwise each hypothesis replays
its own compatible authority.

Authority replay state is distinct from per-color drawback internal state.
Public board transitions affect both White and Black hypotheses even though
only the mover's drawback state transitions. The current predictor optimization
that updates only one color's rule state may remain, but both distributions
must validate the shared public move against their authority families when
authority behavior can persist across turns.

### Evidence

Elimination evidence distinguishes:

- move not generated by authority;
- move generated but forbidden by drawback;
- authority transition disagrees with observed position;
- terminal event disagrees with observation.

This prevents a synthetic-move contradiction from being mislabeled as an
ordinary drawback filter rejection.

## Diagnostic probe parity

Probe search is generic today and already requires callers to represent
terminal reply outcomes explicitly when a hypothesis has no replies. Define a
chess adapter with:

```ts
type AuthorityReplyOutcome =
  | { readonly kind: "move"; readonly move: AuthorityMove;
      readonly position: SerializedAuthorityPosition }
  | { readonly kind: "terminal"; readonly terminal: AuthorityTerminal;
      readonly position: SerializedAuthorityPosition };
```

For each candidate diagnostic move and opponent hypothesis, the adapter:

1. applies the candidate with the current player's authority/runtime;
2. emits an immediate terminal branch for king capture;
3. otherwise generates opponent authority moves with the opponent
   hypothesis's capabilities;
4. filters them through that hypothesis;
5. applies each reply and emits move or terminal outcomes;
6. keys branches by canonical authority ID, move ID, terminal kind, and
   serialized resulting position.

An empty reply array remains an error. Checkmate, king capture, drawback loss,
and stalemate are explicit branches with probability mass. Chess-quality
assessment must either support the authority family or report
`unsupported-assessment`; Stockfish must not receive a non-orthodox snapshot
or silently evaluate a nearby FEN.

## Replay fixture evolution

Extend fixtures without weakening standard-fixture validation:

```json
{
  "ruleId": "death-wish",
  "authorityId": "capturable-king/v1",
  "position": {
    "authorityId": "capturable-king/v1",
    "schemaVersion": 1,
    "turn": "white",
    "state": {}
  },
  "authorityMoves": ["e1e2", "e1f2"],
  "allowedMoves": ["e1f2"],
  "forbiddenMoves": ["e1e2"]
}
```

- Fixtures with no `authorityId` remain standard and continue through the
  current `chess.js` legal-move assertion.
- Variant fixtures resolve the named authority, validate the snapshot, and
  assert declared moves against its generated set.
- A fixture may not label a synthetic move as `ordinaryLegalMoves`.
- Terminal fixtures assert the exact result kind, winner, captured king,
  resulting snapshot, and move ID.
- Chronological fixtures replay from an initial authority snapshot rather than
  injecting a final FEN plus context-only history.

## Migration plan

### Phase 1: introduce adapters without behavior changes

1. Add the versioned authority contracts in a package independent of React and
   predictor.
2. Implement `StandardChessAuthority` as a thin wrapper around the existing
   adapter and terminal calls.
3. Refactor `GameSession` to store a serialized authority position instead of
   a mutable `Chess` instance.
4. Keep `fen`, standard move observations, and existing rejection text as
   compatibility projections.
5. Differential-test every existing game/property fixture against the pre-
   refactor implementation with fixed seeds and byte-identical standard rows.

### Phase 2: propagate public serialization

1. Add authority snapshots and move IDs to observations, simulations, NDJSON,
   PGN analysis output, and replay viewers.
2. Teach predictor and probe adapters to consume authority positions.
3. Bump dataset, report, and symbolic schema versions; reject incompatible
   checkpoints rather than padding vectors or positions.
4. Retain FEN as nullable compatibility metadata.

### Phase 3: implement capturable-king authority

1. Implement board decoding/encoding and orthodox-compatible generation.
2. Differential-test its orthodox subset against `chess.js`.
3. Add explicit king-capture generation, application, serialization, and
   terminal evaluation.
4. Add capability-gated king-into-attack moves.
5. Add predictor, probe, worker, dataset, and UI support before registering any
   executable drawback that requires it.

### Phase 4: register reviewed rules

1. Write the Death Wish rule specification and fixtures from attributable
   evidence, including already-in-check behavior.
2. Register Death Wish only after its synthetic moves work through every
   consumer.
3. Implement literal king-capture restrictions on terminal-intent moves; do
   not reuse checkmate surrogates.
4. Mark rules `implemented-unverified` until the full verification checklist
   and source ambiguities are resolved.

## Test obligations

### Authority contract tests

- encode/decode canonical round trip and defensive immutability;
- reject wrong authority IDs, versions, malformed boards, duplicate kings, and
  invalid counters;
- stable move IDs and deterministic move ordering;
- `applyMove` rejects forged, stale, unavailable-capability, wrong-turn, and
  wrong-promotion moves;
- standard adapter differential parity for legal moves, FEN, repetition, all
  draw conditions, checkmate, castling, en passant, and promotions;
- capturable authority differential parity on positions without synthetic
  moves;
- a captured-king snapshot round trips although it has no valid standard FEN.

### Death Wish tests

- king-into-attack moves appear only with the capability;
- when not already attacked and at least one exists, all ordinary alternatives
  are filtered out;
- multiple suicidal destinations remain legal;
- already-in-check positions do not trigger the force;
- pinned geometric attackers and enemy-king adjacency follow the written
  attack definition;
- castling into, out of, or through attack remains forbidden unless explicitly
  evidenced otherwise;
- opponent king capture applies and terminates immediately;
- declining a possible king capture follows the authority/rule specification;
- White and Black capability/state remain isolated.

### Literal king-capture restriction tests

- a king-capture terminal-intent move is admitted or rejected by mover type,
  hidden parameter, material prerequisite, or historical state as specified;
- checks and checkmates that do not capture a king do not satisfy the literal
  obligation;
- promotion mover versus resulting type is explicit;
- discovered attacks, castling, and en-passant do not masquerade as king
  capture;
- no legal required king capture produces the documented drawback loss.

### Session and precedence tests

- exact sequence from generation through transition and immediate terminal;
- king capture outranks next-turn drawback loss;
- rejected moves do not mutate authority or rule state;
- history contains one immutable normalized move per accepted command;
- secret snapshots contain no authority internals beyond public position;
- fixed seeds preserve White/Black parameter generation order;
- standard sessions remain byte-for-byte deterministic.

### Predictor tests

- synthetic observation eliminates standard hypotheses;
- compatible variant hypothesis transitions to the exact observed snapshot;
- impossible transition and terminal mismatch hard-eliminate;
- neural likelihood never restores eliminated mass;
- both color distributions validate persistent public authority state while
  only mover rule state changes;
- parameter particles retain one total drawback prior;
- chronological replay equals live session at every ply;
- late-start snapshots either decode exactly or fail explicitly.

### Probe tests

- immediate king capture is a terminal reply branch, never an empty array;
- entropy and elimination grouping key terminal kinds and snapshots
  deterministically;
- worst-case assessment reports unsupported for non-orthodox positions unless
  a variant evaluator is installed;
- serial and async probe paths agree on structural branches.

### Fixture, simulation, and UI tests

- standard fixture behavior remains unchanged;
- variant fixture moves are validated by their authority, not `chess.js`;
- serial and worker simulation produce identical authority snapshots and
  terminal rows;
- dataset rows never leak hidden rule parameters/state;
- board UI renders exposed and captured kings from authority state without
  parsing invalid FEN;
- move input sends `moveId`, not display notation;
- replay and post-game analysis reproduce king capture exactly;
- Stockfish-backed agents fail closed on unsupported variant positions.

## Non-goals

- Treating standard checkmate as a literal king capture.
- Allowing a drawback filter to mutate the board.
- Encoding hidden drawback identity or RNG seed into authority state.
- Sending invalid or approximate FENs to Stockfish.
- Enabling every pseudo-legal self-check move when only Death Wish king moves
  require an exception.
- Registering a partial implementation that silently falls back to orthodox
  legality.
