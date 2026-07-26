# Model promotion audit

Audit date: 2026-07-24
Repository base: `784ac4b602a0e96ba0f97ae45f8b055684e065ab`

## Finding

**No generated checkpoint is promotable.**

The strongest inspected candidate is the three-seed v2.1 hybrid experiment in
`data/generated/baseline-v21-hybrid-27-covered-20260723`. It has genuine
held-out validation measurements, but it failed its preregistered Top-3, NLL,
and Brier gates. Under the frozen test policy, the hybrid test split was not
opened and there is no hybrid test report. The only measured test report found
belongs to the older v1 baseline. Validation-only performance cannot promote a
model.

The generated corpus, checkpoints, and reports are ignored local files. This
audit records their observed state but does not convert them into release
artifacts.

## Candidate and held-out evidence

The v2.1 protocol selected epoch 1 for each training seed by mean White/Black
validation NLL. Each validation report contains 9,383 move examples.

| Training seed | White Top-1 | Black Top-1 | White Top-3 | Black Top-3 | White Top-5 | Black Top-5 | White NLL | Black NLL | White Brier | Black Brier | White ECE | Black ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260731 | 0.1478205265 | 0.1927954812 | 0.3022487477 | 0.3729084515 | 0.4279015240 | 0.4953639561 | 3.1662194578 | 3.0209618463 | 0.9886832134 | 0.9347221569 | 0.1948766350 | 0.1459055304 |
| 20260801 | 0.1429180433 | 0.1874666951 | 0.3024618992 | 0.3552168816 | 0.4333368859 | 0.4845998082 | 3.1430287219 | 3.0246245298 | 0.9987511350 | 0.9362552149 | 0.2010377264 | 0.1362849582 |
| 20260802 | 0.1490994351 | 0.1821379090 | 0.3208994991 | 0.3474368539 | 0.4512416072 | 0.4691463285 | 3.1829184041 | 3.0983567144 | 0.9874058231 | 0.9404434677 | 0.1905426428 | 0.1573108517 |
| Three-seed mean | 0.1466126683 | 0.1874666951 | 0.3085367153 | 0.3585207290 | 0.4374933390 | 0.4830366976 | 3.1640555279 | 3.0479810302 | 0.9916133905 | 0.9371402798 | 0.1954856681 | 0.1465004468 |

The mean across both color heads and all three seeds is Top-1 `0.1670396817`,
Top-3 `0.3335287222`, Top-5 `0.4602650183`, NLL `3.1060182791`, and Brier
`0.9643768352`. The recorded horizon Top-1 means at plies 5, 10, 15, and 20
are 0.0605, 0.0956, 0.1212, and 0.1703.

The frozen gates required Top-3 at least `0.35`, NLL at most `3.00`, and Brier
below the 27-class uniform value `26/27 = 0.9629629630`. The candidate misses
all three. Its Top-3 also trails the symbolic-only comparator by 0.0402, beyond
the allowed 0.03 regression.

The only test report found is
`data/generated/baseline-v1-fixed-20260723/test-report.json`. It covers 2,984
move examples for the older v1 model:

| Head | Top-1 | Top-3 | Top-5 | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| White | 0.0439008043 | 0.1430965147 | 0.2402815013 | 3.4285469188 | 0.9921543620 | 0.1326592505 |
| Black | 0.0563002681 | 0.1735924933 | 0.2701072386 | 3.5123412149 | 1.0005170167 | 0.1440275447 |

These v1 test measurements do not validate or promote the v2.1 hybrid.

## Coverage

The hybrid corpus manifest records:

- schema version 1 and root seed `20260730`;
- 27 drawback classes and five agent profiles;
- 900 training games / 32,170 rows;
- 270 validation games / 9,383 rows;
- 270 sealed test games / 9,044 rows;
- 1,440 games / 50,597 rows total;
- maximum 40 plies and eight generation workers.

The generated rows contain Symbolic feature version 1 with 27 White
probabilities, 27 Black probabilities, and matching elimination masks. The run
metadata agrees on the exact 27-class vocabulary.

That coverage is only 27 of the repository's current 182 executable prepared
rules (14.84%), and it excludes the evaluator-backed rules. The current ML
schema uses Symbolic feature version 5. Consequently these checkpoints are
historical experiments, not current-catalog release candidates.

## Provenance and artifacts

Observed content hashes and sizes:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Corpus manifest | 29,933 | `514370a9d2e2665c236d4ec49cf98f9dc082d31cd92d16f037920420599ea1e3` |
| Training NDJSON | 70,041,249 | `49a71d16e82cb6b0765cde023bce955187ae6a028ff7d37347bf8fb021507695` |
| Validation NDJSON | 20,405,015 | `53c50ad49698939bc621890df2c3aae7c1105bf2ee7b85abbfd0943586785774` |
| Sealed test NDJSON | 19,791,655 | `4ad62e31e8031e669d6736895e9fed21722460c6bc3aa7a88d6e7fabf1b6c288` |
| Seed 20260731 epoch-1 checkpoint | 66,335,498 | `66ca1b83cc125eb41e65b24fcbd7f4893f999421bb3312b8f920ebb9a0de7b35` |
| Seed 20260801 epoch-1 checkpoint | 66,335,498 | `731c3e6928768972fd7640f5925c55eb3d92cec3481b017185887121d0cf3b22` |
| Seed 20260802 epoch-1 checkpoint | 66,335,498 | `20ca49d9f63a0c72a63040d144f8af24420340ac1256d4f0d505568768256089` |

Each selected checkpoint is approximately 63.26 MiB. Run metadata declares
checkpoint format 3, model variant `v21-hybrid`, feature schema 1, Symbolic
feature version 1, a 792-element board feature vector, 20,480 legal-mask
outputs, three epochs, and the fixed training seed. It also records the exact
drawback and parameter vocabularies and SAN tokenizer metadata.

The corpus manifest does not record the source revision, dependency lock hash,
Python/PyTorch versions, hardware, or hashes of the generated corpus files.
The run metadata does not bind itself to the corpus manifest hash or a source
revision. The hashes above were computed during this audit and are not an
embedded provenance chain.

PyTorch is not installed in the audit environment, so checkpoint deserialization
could not be independently executed. Static metadata inspection remains
sufficient to establish non-promotability; it is not sufficient to establish
runtime compatibility.

## Promotion blockers

1. The v2.1 hybrid fails preregistered validation gates for Top-3, NLL, and
   Brier.
2. No v2.1 hybrid test report exists. The sealed test data cannot be treated as
   evaluated evidence.
3. The model covers 27 of 182 currently executable rules and omits
   evaluator-backed public constraints.
4. The model and corpus use Symbolic feature version 1 while current training
   and inference require version 5.
5. No checkpoint is published in `ml/models`, and the ignored 63.26 MiB
   PyTorch artifact has no browser-compatible release format.
6. Corpus, code, environment, model, and report provenance are not bound by a
   single signed or content-addressed release manifest.
7. Runtime checkpoint reconstruction was not verified in this environment
   because PyTorch is absent.

Promotion requires a current 182-rule schema-5 corpus with uniform evaluator
coverage, a preregistered multi-seed training run, passing validation gates, a
one-shot untouched test evaluation, complete content-addressed provenance, and
a verified deployable inference artifact. Until all of those exist, the web
application must continue to describe its predictions as symbolic and
uncalibrated rather than as a promoted hybrid model.
