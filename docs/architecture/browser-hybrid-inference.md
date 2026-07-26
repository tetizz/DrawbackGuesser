# Browser hybrid inference contract

## Purpose

This document defines the minimum safe boundary for adding neural drawback
ranking to the offline browser PGN analyzer. The neural model is advisory. The
executable symbolic rule engine remains the authority for legality and hard
elimination.

The first browser release covered by this contract is post-game, offline
analysis of a public standard-chess PGN. It must not scrape, connect to, or
observe a live external game. Adding a model does not change that product
boundary.

## Trust boundaries

The pipeline has four distinct domains:

1. The PGN replay worker reconstructs public positions and observed moves.
2. The symbolic predictor evaluates exact rule hypotheses from those public
   observations.
3. The neural runtime consumes a versioned public feature record and returns
   finite logits in a declared rule order.
4. The fusion layer applies symbolic probability and hard masks to those
   logits, then publishes independent White and Black distributions.

Labels used during offline training are outside the browser inference domain.
They may be read by the training loss, but they must be separated before
feature construction. A trained artifact contains weights and public schemas,
not training rows, labels, hidden parameters, or simulation state.

The browser worker must receive only public replay data and immutable model
artifacts. React receives only the resulting public analysis snapshots. Neither
the worker protocol nor the rendered result may contain an authoritative game
session, rule secret snapshot, or label record.

## Versioned model artifact

The browser loads one immutable artifact manifest before inference:

```ts
interface BrowserNeuralModelV1 {
  readonly format: "drawbacktrainer-browser-model";
  readonly formatVersion: 1;
  readonly modelVariant: "v1";
  readonly featureSchemaVersion: 1;
  readonly sourceCheckpointSha256: string;
  readonly drawbackVocabulary: readonly string[];
  readonly dimensions: {
    readonly input: 792;
    readonly hidden: number;
    readonly drawbackClasses: number;
  };
  readonly tensors: Readonly<Record<
    | "encoder.0.weight" | "encoder.0.bias"
    | "encoder.2.weight" | "encoder.2.bias"
    | "white_drawback.weight" | "white_drawback.bias"
    | "black_drawback.weight" | "black_drawback.bias",
    { readonly shape: readonly number[]; readonly values: readonly number[] }
  >>;
}
```

All identifiers are exact, case-sensitive values. `drawbackVocabulary` defines
the output order. The artifact is rejected before inference when:

- its feature schema is unsupported;
- its drawback IDs are missing, duplicated, or unknown;
- its checkpoint digest is not lowercase SHA-256;
- any tensor name, shape, count, or value is unexpected or non-finite.

Changing the input features, architecture, rule universe, or fusion equation
creates a new artifact schema or version. A report records the checkpoint
digest, feature version, ordered drawback vocabulary, represented intersection,
and fusion weight.

Subset vocabularies are intentional for research artifacts. Head values are
joined to symbolic hypotheses by the artifact's exact ordered vocabulary, not
by catalog position. The report preserves that order for reproducibility.

The artifact must be bundled or selected explicitly from a trusted local file.
The runtime does not download arbitrary model URLs and does not execute custom
operators or scripts supplied by a report.

## Allowed browser features

Feature construction is an allowlist. The minimal allowed record is derived
only from information that can be reconstructed from the pasted PGN:

- exact pre-move FEN for the offline public replay;
- observed move in canonical UCI form;
- move number, ply, and mover color;
- prior observed SAN or canonical move history;
- the complete ordinary standard-chess legal move set before the move;
- simulated clock time only when it is explicitly present as public input;
- versioned symbolic White and Black rule posterior vectors;
- versioned symbolic White and Black hard-elimination masks;
- a validated public evaluator constraint only when it was computed uniformly
  for every eligible ply under one pinned policy.

The two colors use the same public observation sequence, but have separate
symbolic states, neural heads, masks, and output distributions. A White move
updates White evidence without mutating Black hypothesis state, and vice versa.

Symbolic vectors are permitted because they are derived from public
observations. If the selected neural architecture consumes them, the artifact
must declare that dependency and training and browser inference must use the
same symbolic feature version and ordered rule IDs. The fusion layer still
reapplies an independently retained hard mask after neural inference. It never
trusts a model-provided mask.

An evaluator constraint is either present for every eligible observation or is
absent for the entire analysis. Field presence, policy choice, cache status,
latency, process identity, worker identity, and failure status are not model
features. A missing uniform evaluator stream makes evaluator-dependent
hypotheses unavailable; it does not make them unrestricted or eliminated.

## Forbidden inputs

The browser feature builder and neural adapter must reject, rather than ignore,
any record containing:

- the true White or Black drawback;
- hidden drawback parameters;
- authoritative drawback internal state;
- the drawback-filtered legal move set;
- rule-trigger or forced-move labels;
- the authoritative result when it encodes a drawback loss;
- training split, simulation seed, game ID, bot identity, or worker identity;
- future moves or the final position when scoring an earlier ply;
- post-game truth selected for report scoring;
- neural targets, legal-mask targets, parameter targets, or trigger targets;
- cache-hit status, evaluator timing, engine process output, scores, or
  principal variations not declared as public inputs.

In particular, the following dataset fields are label-side data and never cross
into the model input:

```text
trueDrawback
hiddenParameters
drawbackInternalState
drawbackLegalMoves
ruleTriggered
forced
result
```

The current Python boundary already names these as forbidden feature keys.
Browser code must maintain an independent explicit allowlist rather than
serializing a `DatasetMove` and deleting known labels. Unknown keys fail
decoding so a future dataset addition cannot silently become a feature.

The ordinary standard-chess legal set is allowed. The drawback-legal set is
forbidden because it reveals the exact restriction the model is supposed to
infer. Trigger and forced labels are forbidden even if the neural model has
auxiliary heads that learned to predict them; predicted auxiliary outputs may
be displayed or used by a declared model architecture, while authoritative
targets remain training-only.

## Per-ply inference

For each observed ply, the worker performs this sequence:

1. Reconstruct the pre-move public position and ordinary legal moves.
2. Attach a matching public evaluator fact when uniform evaluator coverage is
   configured; otherwise declare evaluator coverage absent.
3. Advance the symbolic predictor with the observed move.
4. Snapshot both colors' symbolic log probabilities and hard-elimination
   masks.
5. Build and strictly validate the public neural feature record.
6. Run one deterministic model inference and validate its two rule-logit
   vectors.
7. Fuse each color independently.
8. Store the fused distributions, symbolic evidence, model provenance, and
   availability in the per-ply analysis snapshot.

The neural runtime output is invalid if a vector has the wrong length or
contains `NaN`, positive infinity, or an undeclared rule. Negative infinity is
not accepted from the neural runtime; only the trusted fusion layer applies it
for hard elimination. Invalid inference fails the neural stage visibly. It
does not fall back while continuing to label the result as hybrid.

An explicitly configured symbolic-only fallback may return the existing
symbolic report, but must use a different predictor ID and record the neural
failure or unavailability. It must not reuse hybrid confidence labels.

## Fusion equations

Standalone format-1 and format-2 artifacts retain their versioned legacy
equations so previously exported artifacts keep Python/browser parity. They
still apply the trusted hard-elimination mask exactly.

Production ensemble format 4 uses
`rank-preserving-bounded-residual-plus-symbolic-prior-v1`. It converts the
three-member mean residual to a centered bounded signal and limits each
symbolic probability tier by its adjacent log-probability gaps. Neural evidence
may order equal symbolic tiers, but cannot reverse a strict symbolic order.
Hard-eliminated and zero-prior hypotheses remain exactly zero.

The ensemble alpha is chosen from a frozen grid on the authenticated validation
selection partition. It is not a UI or evaluator argument. The
content-addressed selection artifact and selected alpha are recursively bound
through calibration and browser export. See
[rank-preserving fusion](rank-preserving-fusion.md) for the exact algorithm and
artifact chain.

## Hard-elimination invariants

These invariants are mandatory and tested at the fusion boundary:

1. A symbolically eliminated rule has exactly zero fused probability.
2. A neural logit, calibration transform, later move, report reload, or truth
   selection can never restore an eliminated hypothesis.
3. Hard masks come from the executable symbolic predictor, not model output.
4. White and Black masks are applied only to their corresponding heads.
5. Mask indexing is joined by the manifest's ordered rule IDs, never by object
   enumeration or display rank.
6. A rule absent from the represented model/hypothesis intersection is marked
   unavailable and receives no probability; it is not described as eliminated.
7. Missing evaluator facts leave evaluator-dependent rules unavailable. A
   neural score cannot make them available.
8. If every represented rule is eliminated, fusion fails with a typed
   consistency error instead of returning a uniform distribution.
9. Parameter-particle elimination aggregates to the rule level only when every
   represented particle is eliminated. Neural rule logits do not restore an
   eliminated particle or invent a hidden parameter value.
10. The exported report retains symbolic elimination evidence separately from
    neural ranking so every zero and major posterior change is explainable.

Neural legal-mask and trigger heads are predictions, not authorities. They
cannot change game legality, symbolic evidence, or the hard mask.

## Leakage invariants

Training and browser parity is proven by tests at both boundaries:

1. The same public feature record produces byte-identical encoded features in
   Python and the browser.
2. Adding, removing, or changing any forbidden label field does not change
   encoded features; strict browser decoding normally rejects such a record
   before encoding.
3. Post-game truth may change scoring metadata but never the analytical payload,
   model inputs, fused timeline, or analytical digest.
4. Evaluator facts are uniformly present or uniformly absent regardless of the
   true drawback and player color.
5. The model input contains ordinary legal moves, never drawback-legal moves.
6. Auxiliary target values are available only inside the training loss and are
   not serialized into the browser artifact.
7. Feature construction at ply \(n\) cannot read moves or positions after ply
   \(n\).
8. White and Black labels used for training never enter the opposite or same
   color feature encoder.
9. Model confidence is identified by artifact and calibration provenance and is
   never presented as calibrated when calibration is absent.
10. A symbolic-only analysis and a hybrid analysis use distinct predictor IDs
    and report provenance.

Required negative tests inject each forbidden field individually, including
plausible aliases, into the browser feature decoder and assert failure. A
separate influence test mutates every training label while holding public
observations fixed and asserts identical neural inputs and pre-truth outputs.

## Worker and React boundary

Model loading and inference run inside the dedicated analysis worker. Each
request has a generation-bound request ID. Cancellation terminates the active
worker; stale progress, model-load completion, or inference results cannot
update a newer request.

The protocol is versioned and strictly decodes:

- request ID and bounded PGN input;
- selected trusted model ID;
- progress phase and bounded counts;
- final analysis shape and model provenance;
- typed parse, model, schema, cancellation, and consistency failures.

The worker reports distinct phases for replay, model loading, feature encoding,
and inference. React does not receive raw model tensors or authoritative
dataset rows. It displays whether the result is symbolic-only or hybrid,
represented and unavailable rules, calibration status, and model digest.

The offline truth controls become enabled only after analysis. Selecting truth
does not rerun inference. Reports can score unavailable truth as rank `null`
without adding it to the probability distribution.

## Minimum acceptance tests

The browser hybrid path is not complete until tests prove:

- artifact digest, schema, rule order, represented IDs, fusion mode, evaluator
  policy, and calibration validation;
- Python/browser public feature parity on fixed fixtures;
- exact White/Black output ordering and isolation;
- zero fused probability for every hard-eliminated rule;
- neural logits cannot restore a hard-eliminated hypothesis;
- unavailable evaluator-dependent rules remain unavailable without facts;
- uniform public evaluator facts make those rules eligible under the matching
  policy only;
- wrong-policy and partially present evaluator facts fail closed;
- every forbidden input is rejected by the browser feature decoder;
- label mutation and truth selection cannot change features or analytical
  predictions;
- malformed, wrong-length, and non-finite model outputs fail visibly;
- cancellation and superseding requests ignore stale model results;
- symbolic-only fallback has distinct provenance;
- report serialization is deterministic and includes exact model, feature,
  rule-universe, evaluator, fusion, and calibration identities.

## Current implementation boundary

The default offline browser analyzer is `symbolic-v2-standard`. It intentionally
represents the standard-observation rules and marks evaluator-dependent rules
unavailable when uniform public evaluator facts are absent. A user may
explicitly select a compatible local standalone artifact. The worker validates
and executes it under its declared format. The version-4 ensemble path
additionally requires calibrated, content-addressed fusion policy evidence.

No artifact is bundled or promoted because the audited checkpoints do not meet
the current coverage and evaluation gates. A future bundled release still
requires the remaining promotion, provenance, calibration, and current-catalog
acceptance gates above.
