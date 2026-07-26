"""Authenticated player-game frequency comparator for the release corpus union.

The comparator prior is deliberately materialized before validation-gate
evaluation.  It counts each observed player color at most once per game, so a
long game cannot contribute more prior mass than a short game.  Construction
accepts only the frozen release-safe primary corpus and the six authenticated
training-only hard-negative supplements.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import BinaryIO, Mapping, Sequence

from ml.training.drawback_ml.corpus_contract import (
    open_audited_hard_negative_train_corpus,
    open_audited_private_corpus_split,
)
from ml.training.drawback_ml.records import DatasetSchemaError, parse_dataset_row
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS
from ml.training.drawback_ml.training_corpus_set import (
    FROZEN_SUPPLEMENT_PROFILES,
    CorpusIdentity,
    SupplementIdentity,
    create_training_corpus_set,
    verify_training_corpus_set,
)


FORMAT = "drawbacktrainer-training-frequency-comparator"
VERSION = 1
COUNTING_UNIT = "observed-player-game-color-v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PROFILE_OFFSETS = {
    profile_id: offset
    for offset, profile_id, _rule_ids in FROZEN_SUPPLEMENT_PROFILES
}


@dataclass(frozen=True)
class ContentAddressedFile:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("file sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class HardNegativeBinding:
    profile_id: str
    manifest: Path
    dataset: Path
    plan: Path


@dataclass(frozen=True)
class LoadedTrainingFrequency:
    source: ContentAddressedFile
    training_corpus_set_sha256: str
    rule_ids: tuple[str, ...]
    white_counts: tuple[int, ...]
    black_counts: tuple[int, ...]
    white_total: int
    black_total: int
    sources: Mapping[str, object]

    def probabilities(self, color: str) -> tuple[float, ...]:
        """Return the unsmoothed authenticated player-game prior."""

        if color == "white":
            counts, total = self.white_counts, self.white_total
        elif color == "black":
            counts, total = self.black_counts, self.black_total
        else:
            raise ValueError("color must be white or black")
        return tuple(count / total for count in counts)

    def promotion_priors(self) -> Mapping[str, Mapping[str, float]]:
        """Return the exact mapping contract consumed by promotion evaluation."""

        return {
            color: dict(
                zip(self.rule_ids, self.probabilities(color), strict=True)
            )
            for color in ("white", "black")
        }


def _canonical_compact(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields are not canonical")


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"{label} contains non-finite constant {token}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    _require_finite(value, label)
    return value


def _require_finite(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite(item, label)
    elif isinstance(value, list):
        for item in value:
            _require_finite(item, label)


def _verified_bytes(reference: ContentAddressedFile, label: str) -> bytes:
    try:
        payload = reference.path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label}: {reference.path}") from error
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError(f"{label} sha256 does not match")
    return payload


def _binding(path: Path, sha256: str, *, byte_count: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "file": path.name,
        "sha256": _digest(sha256, f"{path.name} sha256"),
    }
    if byte_count is not None:
        value["bytes"] = _nonnegative_int(byte_count, f"{path.name} bytes")
    return value


def _read_pinned_json(source: BinaryIO, label: str) -> Mapping[str, object]:
    source.seek(0)
    payload = source.read()
    source.seek(0)
    return _strict_json(payload, label)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{label} must be a string array")
    return tuple(value)


def _primary_identity(lease: object) -> CorpusIdentity:
    audited = lease.audited
    root = _read_pinned_json(lease.root, "pinned public release root")
    corpus = _object(root.get("corpus"), "public release corpus")
    limit = _object(corpus.get("evaluatorSearchLimit"), "evaluator search limit")
    return CorpusIdentity(
        release_root_sha256=_string(
            audited.release_root_sha256, "release root sha256"
        ),
        corpus_run_id=_string(audited.corpus_run_id, "corpus run id"),
        private_train_manifest_sha256=audited.manifest_sha256,
        outcomes_sha256=audited.outcomes_sha256,
        dataset_sha256=audited.dataset_sha256,
        dataset_bytes=audited.dataset_bytes,
        games=audited.games,
        rows=audited.rows,
        schema_version=_integer(corpus.get("schemaVersion"), "schema version"),
        symbolic_feature_version=_integer(
            corpus.get("symbolicFeatureVersion"), "symbolic feature version"
        ),
        max_plies=_integer(corpus.get("maxPlies"), "max plies"),
        observation_policy=_string(
            corpus.get("observationPolicy"), "observation policy"
        ),
        evaluator_policy_id=_string(
            corpus.get("evaluatorPolicyId"), "evaluator policy id"
        ),
        evaluator_policy_version=_integer(
            corpus.get("evaluatorPolicyVersion"), "evaluator policy version"
        ),
        evaluator_nodes=_integer(limit.get("value"), "evaluator node limit"),
        engine_binary_sha256=_string(
            corpus.get("engineBinarySha256"), "engine binary sha256"
        ),
        engine_fingerprint=audited.engine_fingerprint,
        agent_domain=_string_tuple(corpus.get("agentIds"), "agent ids"),
    )


def _supplement_identity(
    lease: object, offset: int, profile_id: str
) -> tuple[SupplementIdentity, Mapping[str, object]]:
    audited = lease.audited
    manifest = _read_pinned_json(
        lease.manifest, f"pinned {profile_id} manifest"
    )
    profile = _object(
        manifest.get("hardNegativeProfile"),
        f"{profile_id} hardNegativeProfile",
    )
    identity = SupplementIdentity(
        source_revision=audited.source_revision,
        generation_run_id=audited.run_id,
        manifest_sha256=audited.manifest_sha256,
        plan_sha256=audited.plan_sha256,
        outcomes_sha256=audited.outcomes_sha256,
        dataset_sha256=audited.dataset_sha256,
        dataset_bytes=audited.dataset_bytes,
        games=audited.games,
        rows=audited.rows,
        schema_version=6,
        symbolic_feature_version=6,
        max_plies=audited.max_plies,
        observation_policy=audited.observation_policy,
        evaluator_policy_id=audited.evaluator_policy_id,
        evaluator_policy_version=audited.evaluator_policy_version,
        evaluator_nodes=audited.evaluator_nodes,
        engine_binary_sha256=audited.engine_binary_sha256,
        engine_fingerprint=audited.engine_fingerprint,
        agent_domain=audited.agent_ids,
        profile_offset=offset,
        profile_id=audited.profile_id,
        rule_ids=audited.rule_ids,
        profile_sha256=hashlib.sha256(_canonical_compact(profile)).hexdigest(),
    )
    return identity, manifest


def _count_observed_player_games(
    source: BinaryIO,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Count one label per observed game/color using constant per-game state."""

    index = {rule_id: position for position, rule_id in enumerate(SYMBOLIC_RULE_IDS)}
    white = [0] * len(SYMBOLIC_RULE_IDS)
    black = [0] * len(SYMBOLIC_RULE_IDS)
    current_game: str | None = None
    seen_colors: set[str] = set()
    source.seek(0)
    try:
        for line_number, raw_line in enumerate(source, start=1):
            try:
                value = json.loads(raw_line)
                if not isinstance(value, Mapping):
                    raise DatasetSchemaError("dataset row must be an object")
                row = parse_dataset_row(value)
            except (UnicodeDecodeError, json.JSONDecodeError, DatasetSchemaError) as error:
                raise ValueError(
                    f"authenticated training row {line_number} is invalid"
                ) from error
            if row.game_id != current_game:
                current_game = row.game_id
                seen_colors.clear()
            color = row.labels.player_color
            if color in seen_colors:
                continue
            try:
                rule_index = index[row.labels.true_drawback]
            except KeyError as error:
                raise ValueError("training row label is outside the 182-rule catalog") from error
            (white if color == "white" else black)[rule_index] += 1
            seen_colors.add(color)
    finally:
        source.seek(0)
    return tuple(white), tuple(black)


def _sum_counts(rows: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    if not rows:
        raise ValueError("at least one count source is required")
    return tuple(
        sum(row[index] for row in rows)
        for index in range(len(SYMBOLIC_RULE_IDS))
    )


def _source_value(
    primary_lease: object,
    primary: CorpusIdentity,
    public_root: Path,
    supplement_entries: Sequence[
        tuple[HardNegativeBinding, object, SupplementIdentity]
    ],
) -> dict[str, object]:
    return {
        "primary": {
            "public_root": _binding(
                public_root,
                primary.release_root_sha256,
            ),
            "private_manifest": _binding(
                primary_lease.audited.manifest_path,
                primary.private_train_manifest_sha256,
            ),
            "dataset": _binding(
                primary_lease.audited.dataset_path,
                primary.dataset_sha256,
                byte_count=primary.dataset_bytes,
            ),
            "corpus_run_id": primary.corpus_run_id,
            "outcomes_sha256": primary.outcomes_sha256,
        },
        "supplements": [
            {
                "profile_id": identity.profile_id,
                "profile_offset": identity.profile_offset,
                "manifest": _binding(binding.manifest, identity.manifest_sha256),
                "dataset": _binding(
                    binding.dataset,
                    identity.dataset_sha256,
                    byte_count=identity.dataset_bytes,
                ),
                "plan": _binding(binding.plan, identity.plan_sha256),
                "generation_run_id": identity.generation_run_id,
                "outcomes_sha256": identity.outcomes_sha256,
            }
            for binding, _lease, identity in supplement_entries
        ],
    }


def _write_atomic_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(
                f"refusing to overwrite training-frequency artifact: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def write_training_frequency_artifact(
    output: Path,
    *,
    public_root: Path,
    private_train: Path,
    primary_dataset: Path,
    hard_negatives: Sequence[HardNegativeBinding],
    expected_training_corpus_set_sha256: str,
) -> ContentAddressedFile:
    """Audit the exact release corpus union and publish its frequency prior."""

    expected = _digest(
        expected_training_corpus_set_sha256,
        "expected training corpus set sha256",
    )
    if len(hard_negatives) != len(FROZEN_SUPPLEMENT_PROFILES):
        raise ValueError("exactly six hard-negative bindings are required")
    ordered = sorted(
        hard_negatives, key=lambda item: PROFILE_OFFSETS.get(item.profile_id, -1)
    )
    if tuple(item.profile_id for item in ordered) != tuple(
        profile_id for _offset, profile_id, _rules in FROZEN_SUPPLEMENT_PROFILES
    ):
        raise ValueError("hard-negative profiles must equal the frozen six")

    with ExitStack() as stack:
        primary_lease = stack.enter_context(
            open_audited_private_corpus_split(
                public_root, private_train, primary_dataset, "train"
            )
        )
        primary = _primary_identity(primary_lease)
        count_rows = [_count_observed_player_games(primary_lease.dataset)]
        supplement_entries: list[
            tuple[HardNegativeBinding, object, SupplementIdentity]
        ] = []
        supplement_identities: list[SupplementIdentity] = []
        for binding in ordered:
            offset = PROFILE_OFFSETS[binding.profile_id]
            lease = stack.enter_context(
                open_audited_hard_negative_train_corpus(
                    binding.manifest,
                    binding.dataset,
                    binding.plan,
                    binding.profile_id,
                )
            )
            identity, _manifest = _supplement_identity(
                lease, offset, binding.profile_id
            )
            supplement_identities.append(identity)
            supplement_entries.append((binding, lease, identity))
            count_rows.append(_count_observed_player_games(lease.dataset))

        corpus_set = verify_training_corpus_set(
            create_training_corpus_set(primary, supplement_identities)
        )
        if corpus_set.get("sha256") != expected:
            raise ValueError(
                "authenticated corpus union disagrees with expected training set"
            )
        white = _sum_counts(tuple(row[0] for row in count_rows))
        black = _sum_counts(tuple(row[1] for row in count_rows))
        value = {
            "format": FORMAT,
            "version": VERSION,
            "counting_unit": COUNTING_UNIT,
            "training_corpus_set": corpus_set,
            "training_corpus_set_sha256": expected,
            "rule_ids": list(SYMBOLIC_RULE_IDS),
            "rule_ids_sha256": hashlib.sha256(
                _canonical_compact(list(SYMBOLIC_RULE_IDS))
            ).hexdigest(),
            "counts": {
                "white": list(white),
                "black": list(black),
                "white_total": sum(white),
                "black_total": sum(black),
            },
            "sources": _source_value(
                primary_lease, primary, public_root, supplement_entries
            ),
        }
        payload = _canonical_pretty(value)
        _write_atomic_no_clobber(output, payload)
        reference = ContentAddressedFile(
            output, hashlib.sha256(payload).hexdigest()
        )
        load_training_frequency_artifact(reference, expected)
        return reference


def _parse_binding(
    value: object, label: str, *, with_bytes: bool
) -> Mapping[str, object]:
    binding = _object(value, label)
    expected = {"file", "sha256", "bytes"} if with_bytes else {"file", "sha256"}
    _exact_keys(binding, expected, label)
    filename = binding.get("file")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError(f"{label}.file must be a basename")
    _digest(binding.get("sha256"), f"{label}.sha256")
    if with_bytes:
        _nonnegative_int(binding.get("bytes"), f"{label}.bytes")
    return binding


def load_training_frequency_artifact(
    reference: ContentAddressedFile,
    expected_training_corpus_set_sha256: str | None = None,
) -> LoadedTrainingFrequency:
    """Strictly verify canonical artifact bytes and every internal binding."""

    payload = _verified_bytes(reference, "training-frequency artifact")
    value = _strict_json(payload, "training-frequency artifact")
    if payload != _canonical_pretty(value):
        raise ValueError("training-frequency artifact JSON is not canonical")
    _exact_keys(
        value,
        {
            "format",
            "version",
            "counting_unit",
            "training_corpus_set",
            "training_corpus_set_sha256",
            "rule_ids",
            "rule_ids_sha256",
            "counts",
            "sources",
        },
        "training-frequency artifact",
    )
    if (
        value.get("format") != FORMAT
        or value.get("version") != VERSION
        or value.get("counting_unit") != COUNTING_UNIT
    ):
        raise ValueError("unsupported training-frequency artifact")
    corpus_set = verify_training_corpus_set(value.get("training_corpus_set"))
    corpus_sha = _digest(
        value.get("training_corpus_set_sha256"),
        "training corpus set sha256",
    )
    if corpus_set.get("sha256") != corpus_sha:
        raise ValueError("training corpus set hash is inconsistent")
    if (
        expected_training_corpus_set_sha256 is not None
        and corpus_sha
        != _digest(
            expected_training_corpus_set_sha256,
            "expected training corpus set sha256",
        )
    ):
        raise ValueError("training-frequency artifact uses a different corpus set")
    rule_ids = value.get("rule_ids")
    if rule_ids != list(SYMBOLIC_RULE_IDS):
        raise ValueError("training-frequency rule order is not canonical")
    if value.get("rule_ids_sha256") != hashlib.sha256(
        _canonical_compact(rule_ids)
    ).hexdigest():
        raise ValueError("training-frequency rule-order hash is invalid")
    counts = _object(value.get("counts"), "counts")
    _exact_keys(
        counts, {"white", "black", "white_total", "black_total"}, "counts"
    )
    parsed_counts: dict[str, tuple[int, ...]] = {}
    for color in ("white", "black"):
        raw = counts.get(color)
        if not isinstance(raw, list) or len(raw) != len(SYMBOLIC_RULE_IDS):
            raise ValueError(f"{color} counts must contain exactly 182 entries")
        parsed_counts[color] = tuple(
            _nonnegative_int(item, f"{color} count") for item in raw
        )
    white_total = _positive_int(counts.get("white_total"), "white total")
    black_total = _positive_int(counts.get("black_total"), "black total")
    if sum(parsed_counts["white"]) != white_total:
        raise ValueError("White counts do not sum to White total")
    if sum(parsed_counts["black"]) != black_total:
        raise ValueError("Black counts do not sum to Black total")

    sources = _object(value.get("sources"), "sources")
    _exact_keys(sources, {"primary", "supplements"}, "sources")
    primary = _object(sources.get("primary"), "primary source")
    _exact_keys(
        primary,
        {
            "public_root",
            "private_manifest",
            "dataset",
            "corpus_run_id",
            "outcomes_sha256",
        },
        "primary source",
    )
    root_binding = _parse_binding(
        primary.get("public_root"), "primary public root", with_bytes=False
    )
    private_binding = _parse_binding(
        primary.get("private_manifest"),
        "primary private manifest",
        with_bytes=False,
    )
    dataset_binding = _parse_binding(
        primary.get("dataset"), "primary dataset", with_bytes=True
    )
    primary_set = _object(corpus_set.get("primary"), "corpus-set primary")
    if (
        root_binding["sha256"] != primary_set.get("release_root_sha256")
        or private_binding["sha256"]
        != primary_set.get("private_train_manifest_sha256")
        or dataset_binding["sha256"] != primary_set.get("dataset_sha256")
        or dataset_binding["bytes"] != primary_set.get("dataset_bytes")
        or primary.get("corpus_run_id") != primary_set.get("corpus_run_id")
        or primary.get("outcomes_sha256") != primary_set.get("outcomes_sha256")
    ):
        raise ValueError("primary source bindings disagree with corpus set")
    supplements = sources.get("supplements")
    corpus_supplements = corpus_set.get("supplements")
    if (
        not isinstance(supplements, list)
        or not isinstance(corpus_supplements, list)
        or len(supplements) != 6
        or len(corpus_supplements) != 6
    ):
        raise ValueError("source bindings require exactly six supplements")
    for index, (source_item, corpus_item) in enumerate(
        zip(supplements, corpus_supplements, strict=True)
    ):
        source = _object(source_item, f"supplement source {index}")
        corpus = _object(corpus_item, f"corpus supplement {index}")
        _exact_keys(
            source,
            {
                "profile_id",
                "profile_offset",
                "manifest",
                "dataset",
                "plan",
                "generation_run_id",
                "outcomes_sha256",
            },
            f"supplement source {index}",
        )
        manifest = _parse_binding(
            source.get("manifest"),
            f"supplement {index} manifest",
            with_bytes=False,
        )
        dataset = _parse_binding(
            source.get("dataset"),
            f"supplement {index} dataset",
            with_bytes=True,
        )
        plan = _parse_binding(
            source.get("plan"),
            f"supplement {index} plan",
            with_bytes=False,
        )
        if (
            source.get("profile_id") != corpus.get("profile_id")
            or source.get("profile_offset") != corpus.get("profile_offset")
            or manifest["sha256"] != corpus.get("manifest_sha256")
            or dataset["sha256"] != corpus.get("dataset_sha256")
            or dataset["bytes"] != corpus.get("dataset_bytes")
            or plan["sha256"] != corpus.get("plan_sha256")
            or source.get("generation_run_id")
            != corpus.get("generation_run_id")
            or source.get("outcomes_sha256") != corpus.get("outcomes_sha256")
        ):
            raise ValueError(
                f"supplement source {index} disagrees with corpus set"
            )
    return LoadedTrainingFrequency(
        source=reference,
        training_corpus_set_sha256=corpus_sha,
        rule_ids=tuple(SYMBOLIC_RULE_IDS),
        white_counts=parsed_counts["white"],
        black_counts=parsed_counts["black"],
        white_total=white_total,
        black_total=black_total,
        sources=sources,
    )


def verify_training_frequency_sources(
    reference: ContentAddressedFile,
    *,
    public_root: Path,
    private_train: Path,
    primary_dataset: Path,
    hard_negatives: Sequence[HardNegativeBinding],
    expected_training_corpus_set_sha256: str,
) -> LoadedTrainingFrequency:
    """Re-audit all seven sources and reproduce the artifact byte for byte."""

    original_payload = _verified_bytes(
        reference, "training-frequency artifact"
    )
    loaded = load_training_frequency_artifact(
        reference, expected_training_corpus_set_sha256
    )
    if (
        _verified_bytes(reference, "training-frequency artifact")
        != original_payload
    ):
        raise ValueError("training-frequency artifact changed during verification")
    with tempfile.TemporaryDirectory(
        prefix="drawbacktrainer-frequency-verification-"
    ) as raw:
        reproduced = write_training_frequency_artifact(
            Path(raw) / reference.path.name,
            public_root=public_root,
            private_train=private_train,
            primary_dataset=primary_dataset,
            hard_negatives=hard_negatives,
            expected_training_corpus_set_sha256=(
                expected_training_corpus_set_sha256
            ),
        )
        if reproduced.path.read_bytes() != original_payload:
            raise ValueError(
                "training-frequency artifact is not reproducible from its sources"
            )
    if (
        _verified_bytes(reference, "training-frequency artifact")
        != original_payload
    ):
        raise ValueError("training-frequency artifact changed during verification")
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.training_frequency"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-train", type=Path, required=True)
    parser.add_argument("--primary-dataset", type=Path, required=True)
    parser.add_argument(
        "--hard-negative",
        action="append",
        nargs=4,
        metavar=("PROFILE", "MANIFEST", "DATASET", "PLAN"),
        required=True,
    )
    parser.add_argument(
        "--expected-training-corpus-set-sha256", required=True
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    bindings = tuple(
        HardNegativeBinding(
            profile_id=profile,
            manifest=Path(manifest),
            dataset=Path(dataset),
            plan=Path(plan),
        )
        for profile, manifest, dataset, plan in arguments.hard_negative
    )
    output = write_training_frequency_artifact(
        arguments.output,
        public_root=arguments.public_root,
        private_train=arguments.private_train,
        primary_dataset=arguments.primary_dataset,
        hard_negatives=bindings,
        expected_training_corpus_set_sha256=(
            arguments.expected_training_corpus_set_sha256
        ),
    )
    print(
        json.dumps(
            {"file": output.path.name, "sha256": output.sha256},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
