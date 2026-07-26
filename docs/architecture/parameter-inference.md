# Hidden-parameter inference

Parameterized drawbacks are represented as separate symbolic hypotheses for
each parameter candidate. A hypothesis variant is identified by the drawback
ID plus a canonical serialization of its parameters. Variants transition and
eliminate independently; aggregating variants for display must never revive an
eliminated variant.

## Priors

The configured prior belongs to the drawback, not to every parameter variant.
For a drawback prior \(P(D)\) and \(n\) equally likely variants, each seed
starts with \(P(D) / n\). This prevents a rule with 64 square candidates from
receiving 64 times the prior mass of a parameterless rule.

Rule-level posterior probability is the sum of its surviving variant
probabilities. A parameter posterior is normalized conditionally within that
rule:

```text
P(parameter | drawback, moves)
  = P(drawback, parameter | moves) / P(drawback | moves)
```

The UI may collapse variants only after the exact legality update.

## Coverage policy

- **Untitled Duck Drawback:** exact enumeration of all 64 squares.
- **Just Passing Through:** exact enumeration of ranks 1 through 8.
- **Gambler:** a deterministic particle approximation over hidden 32-bit
  seeds. Exhaustive enumeration is impossible. The particle count, particle
  generator seed, coverage limitations, and effective sample size must be
  visible in analysis metadata. A zero posterior means all represented
  particles were eliminated; it does not prove that every one of the
  \(2^{32}\) original seeds was impossible.

The default particle set is for offline training and research. It is not an
exact reconstruction of Gambler's secret seed distribution and must not be
presented as one.

## Neural targets

Hidden parameters are labels, never features. Dataset parsing separates them
before feature construction. Parameter vocabularies are derived only from
training labels, stored in checkpoint metadata, and decoded through separate
White and Black outputs. Examples without a parameter target use an explicit
no-parameter class or a masked parameter loss; they are never assigned an
arbitrary hidden value.
