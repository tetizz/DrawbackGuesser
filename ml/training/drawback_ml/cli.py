"""Command-line entrypoint for auditing datasets and training the baseline."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, BinaryIO, Iterable, Mapping

from .browser_artifact import export_browser_artifact
from .corpus_contract import (
    audit_corpus_split,
    open_audited_hard_negative_train_corpus,
    open_audited_private_corpus_split,
)
from .records import group_training_examples
from .splits import assign_split
from .symbolic_schema import SYMBOLIC_RULE_IDS
from .training import TrainingConfig, train_baseline
from .streaming import (
    PinnedExampleSource,
    example_factory,
    pinned_multi_source_example_factory,
)
from .streaming_training import train_streaming_baseline
from .training_corpus_set import (
    FROZEN_SUPPLEMENT_PROFILES,
    SCHEMA_VERSION,
    SYMBOLIC_FEATURE_VERSION,
    CorpusIdentity,
    SupplementIdentity,
    create_training_corpus_set,
    verify_training_corpus_set,
)


def _read_ndjson(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield value


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument(
        "--model-variant",
        choices=("v1", "v2-gru", "v21-hybrid", "v22-hybrid"),
        default="v1",
        help=(
            "v1 is the measured feed-forward control; v21-hybrid uses the "
            "frozen rank-preserving fusion-grid objective; v22-hybrid is the "
            "preregistered current-move ablation architecture"
        ),
    )
    parser.add_argument(
        "--sequence-observation-mode",
        choices=("masked-current-v2", "exact-current-v2"),
        default=None,
        help=(
            "required for v22-hybrid; the paired A1 arms differ only in "
            "whether the final public current-move token is masked"
        ),
    )
    parser.add_argument("--max-history", type=int, default=128)
    parser.add_argument("--san-embedding-dimension", type=int, default=32)
    parser.add_argument("--sequence-hidden-dimension", type=int, default=64)
    parser.add_argument("--symbolic-hidden-dimension", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--shuffle-buffer-size", type=int, default=4096)
    parser.add_argument("--trigger-loss-weight", type=float, default=0.1)
    parser.add_argument("--parameter-loss-weight", type=float, default=0.1)
    parser.add_argument("--legal-mask-loss-weight", type=float, default=0.05)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drawback-guesser-train")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="audit and count an NDJSON dataset")
    inspect.add_argument("dataset", type=Path)
    train = subparsers.add_parser("train", help="train a v1 or opt-in v2 model")
    train.add_argument("dataset", type=Path)
    train.add_argument("output", type=Path)
    _add_training_arguments(train)
    audit = subparsers.add_parser(
        "audit-corpus",
        help="verify a content-addressed schema-6 corpus split",
    )
    audit.add_argument("manifest", type=Path)
    audit.add_argument("split", choices=("train", "validation", "test"))
    audit.add_argument(
        "--allow-incomplete-catalog",
        action="store_true",
        help="permit bounded pipeline smoke corpora that cannot train a 182-class head",
    )
    train_corpus = subparsers.add_parser(
        "train-corpus",
        help="legacy research only: train from a monolithic schema-6 manifest",
    )
    train_corpus.add_argument("manifest", type=Path)
    train_corpus.add_argument("output", type=Path)
    _add_training_arguments(train_corpus)
    train_corpus.add_argument(
        "--game-examples-per-epoch",
        type=int,
        default=None,
        help=(
            "deprecated total-per-game setting; must be even and is divided "
            "equally between observed White and Black player-games"
        ),
    )
    train_corpus.add_argument(
        "--player-game-examples-per-epoch",
        type=int,
        default=None,
        help=(
            "move examples sampled per observed drawback/color/player-game "
            "after deterministic stratum balancing (default: 8)"
        ),
    )
    train_release = subparsers.add_parser(
        "train-release",
        help="train from an authenticated public root and train-private manifest",
    )
    train_release.add_argument("root_manifest", type=Path)
    train_release.add_argument("private_manifest", type=Path)
    train_release.add_argument("dataset", type=Path)
    train_release.add_argument("output", type=Path)
    train_release.add_argument(
        "--execution-source-revision",
        required=True,
        help="exact clean repository HEAD used for release training",
    )
    _add_training_arguments(train_release)
    train_release.add_argument(
        "--game-examples-per-epoch",
        type=int,
        default=None,
        help=(
            "deprecated total-per-game setting; must be even and is divided "
            "equally between observed White and Black player-games"
        ),
    )
    train_release.add_argument(
        "--player-game-examples-per-epoch",
        type=int,
        default=None,
        help=(
            "move examples sampled per observed drawback/color/player-game "
            "after deterministic stratum balancing (default: 8)"
        ),
    )
    train_release.add_argument(
        "--hard-negative",
        action="append",
        nargs=4,
        metavar=("PROFILE", "MANIFEST", "DATASET", "PLAN"),
        default=[],
        help=(
            "authenticated hard-negative supplement; repeat exactly once for "
            "each frozen profile"
        ),
    )
    export = subparsers.add_parser(
        "export-browser",
        help=(
            "export a validated v1, v21-hybrid, or v22-hybrid checkpoint "
            "as canonical JSON"
        ),
    )
    export.add_argument("checkpoint", type=Path)
    export.add_argument("output", type=Path)
    return parser


def _validate_unchanged_corpus(manifest: Path, expected: object) -> None:
    if audit_corpus_split(manifest, "train") != expected:
        raise ValueError("corpus changed during streaming training")


def _read_pinned_json(source: BinaryIO, label: str) -> dict[str, Any]:
    source.seek(0)
    try:
        value = json.load(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    finally:
        source.seek(0)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_pinned_sha256(
    source: BinaryIO,
    expected: str,
    label: str,
) -> None:
    source.seek(0)
    digest = hashlib.sha256()
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
    source.seek(0)
    if digest.hexdigest() != expected:
        raise ValueError(f"pinned {label} changed after authentication")


def _hard_negative_bindings(
    raw_bindings: object,
) -> tuple[tuple[int, str, Path, Path, Path], ...]:
    if not isinstance(raw_bindings, list):
        raise ValueError("--hard-negative bindings must be a list")
    if len(raw_bindings) != len(FROZEN_SUPPLEMENT_PROFILES):
        raise ValueError(
            "train-release requires exactly six --hard-negative bindings"
        )
    profiles = {
        profile_id: offset
        for offset, profile_id, _rule_ids in FROZEN_SUPPLEMENT_PROFILES
    }
    parsed: list[tuple[int, str, Path, Path, Path]] = []
    seen: set[str] = set()
    for raw in raw_bindings:
        if (
            not isinstance(raw, list)
            or len(raw) != 4
            or any(not isinstance(item, str) or not item for item in raw)
        ):
            raise ValueError(
                "--hard-negative requires PROFILE MANIFEST DATASET PLAN"
            )
        profile_id, manifest, dataset, plan = raw
        offset = profiles.get(profile_id)
        if offset is None:
            raise ValueError(f"unknown hard-negative profile: {profile_id}")
        if profile_id in seen:
            raise ValueError(f"duplicate hard-negative profile: {profile_id}")
        seen.add(profile_id)
        parsed.append(
            (offset, profile_id, Path(manifest), Path(dataset), Path(plan))
        )
    return tuple(sorted(parsed))


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{label} must be a string array")
    return tuple(value)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "export-browser":
        output = export_browser_artifact(arguments.checkpoint, arguments.output)
        print(str(output))
        return 0
    if arguments.command == "audit-corpus":
        audited = audit_corpus_split(
            arguments.manifest,
            arguments.split,
            require_complete_catalog=not arguments.allow_incomplete_catalog,
        )
        value = asdict(audited)
        for key in ("manifest_path", "dataset_path"):
            value[key] = str(value[key])
        print(json.dumps(value, sort_keys=True))
        return 0
    if arguments.command in {"train-corpus", "train-release"}:
        if arguments.command == "train-release":
            bindings = _hard_negative_bindings(arguments.hard_negative)
            arguments.execution_source_revision = (
                _verify_clean_execution_revision(
                    arguments.execution_source_revision
                )
            )
            with ExitStack() as stack:
                primary_lease = stack.enter_context(
                    open_audited_private_corpus_split(
                        arguments.root_manifest,
                        arguments.private_manifest,
                        arguments.dataset,
                        "train",
                    )
                )
                primary = primary_lease.audited
                root_value = _read_pinned_json(
                    primary_lease.root, "pinned public release root"
                )
                corpus = _object(
                    root_value.get("corpus"),
                    "pinned public release root corpus",
                )
                search_limit = _object(
                    corpus.get("evaluatorSearchLimit"),
                    "public evaluator search limit",
                )
                primary_identity = CorpusIdentity(
                    release_root_sha256=_string(
                        primary.release_root_sha256,
                        "release root sha256",
                    ),
                    corpus_run_id=_string(
                        primary.corpus_run_id,
                        "corpus run id",
                    ),
                    private_train_manifest_sha256=primary.manifest_sha256,
                    outcomes_sha256=primary.outcomes_sha256,
                    dataset_sha256=primary.dataset_sha256,
                    dataset_bytes=primary.dataset_bytes,
                    games=primary.games,
                    rows=primary.rows,
                    schema_version=_integer(
                        corpus.get("schemaVersion"), "public schemaVersion"
                    ),
                    symbolic_feature_version=_integer(
                        corpus.get("symbolicFeatureVersion"),
                        "public symbolicFeatureVersion",
                    ),
                    max_plies=_integer(
                        corpus.get("maxPlies"), "public maxPlies"
                    ),
                    observation_policy=_string(
                        corpus.get("observationPolicy"),
                        "public observationPolicy",
                    ),
                    evaluator_policy_id=_string(
                        corpus.get("evaluatorPolicyId"),
                        "public evaluatorPolicyId",
                    ),
                    evaluator_policy_version=_integer(
                        corpus.get("evaluatorPolicyVersion"),
                        "public evaluatorPolicyVersion",
                    ),
                    evaluator_nodes=_integer(
                        search_limit.get("value"),
                        "public evaluator node limit",
                    ),
                    engine_binary_sha256=_string(
                        corpus.get("engineBinarySha256"),
                        "public engineBinarySha256",
                    ),
                    engine_fingerprint=primary.engine_fingerprint,
                    agent_domain=_string_tuple(
                        corpus.get("agentIds"), "public agentIds"
                    ),
                )
                sources = [
                    PinnedExampleSource(
                        namespace="primary",
                        source=primary_lease.dataset,
                        assignments={
                            game_id: (white, black)
                            for game_id, white, black
                            in primary.game_assignments
                        },
                        max_rows_per_game=primary_identity.max_plies,
                    )
                ]
                supplement_identities: list[SupplementIdentity] = []
                supplement_leases = []
                for (
                    offset,
                    profile_id,
                    manifest_path,
                    dataset_path,
                    plan_path,
                ) in bindings:
                    supplement_lease = stack.enter_context(
                        open_audited_hard_negative_train_corpus(
                            manifest_path,
                            dataset_path,
                            plan_path,
                            profile_id,
                        )
                    )
                    supplement_leases.append(supplement_lease)
                    supplement = supplement_lease.audited
                    manifest = _read_pinned_json(
                        supplement_lease.manifest,
                        f"pinned {profile_id} manifest",
                    )
                    profile = _object(
                        manifest.get("hardNegativeProfile"),
                        f"{profile_id} hardNegativeProfile",
                    )
                    supplement_identities.append(
                        SupplementIdentity(
                            source_revision=supplement.source_revision,
                            generation_run_id=supplement.run_id,
                            manifest_sha256=supplement.manifest_sha256,
                            plan_sha256=supplement.plan_sha256,
                            outcomes_sha256=supplement.outcomes_sha256,
                            dataset_sha256=supplement.dataset_sha256,
                            dataset_bytes=supplement.dataset_bytes,
                            games=supplement.games,
                            rows=supplement.rows,
                            schema_version=SCHEMA_VERSION,
                            symbolic_feature_version=SYMBOLIC_FEATURE_VERSION,
                            max_plies=supplement.max_plies,
                            observation_policy=supplement.observation_policy,
                            evaluator_policy_id=supplement.evaluator_policy_id,
                            evaluator_policy_version=(
                                supplement.evaluator_policy_version
                            ),
                            evaluator_nodes=supplement.evaluator_nodes,
                            engine_binary_sha256=(
                                supplement.engine_binary_sha256
                            ),
                            engine_fingerprint=supplement.engine_fingerprint,
                            agent_domain=supplement.agent_ids,
                            profile_offset=offset,
                            profile_id=supplement.profile_id,
                            rule_ids=supplement.rule_ids,
                            profile_sha256=_canonical_sha256(profile),
                        )
                    )
                    sources.append(
                        PinnedExampleSource(
                            namespace=f"hard-negative-{offset}-{profile_id}",
                            source=supplement_lease.dataset,
                            assignments={
                                game_id: (white, black)
                                for game_id, white, black
                                in supplement.game_assignments
                            },
                            max_rows_per_game=supplement.max_plies,
                        )
                    )
                corpus_set = verify_training_corpus_set(
                    create_training_corpus_set(
                        primary_identity,
                        supplement_identities,
                    )
                )
                corpus_set_sha256 = _string(
                    corpus_set.get("sha256"),
                    "training corpus set sha256",
                )

                def verify_all_leases_unchanged(
                    staging_directory: Path,
                ) -> None:
                    _verify_clean_execution_revision(
                        arguments.execution_source_revision,
                        ignored_untracked_paths=(staging_directory,),
                    )
                    _verify_pinned_sha256(
                        primary_lease.root,
                        primary_identity.release_root_sha256,
                        "public release root",
                    )
                    _verify_pinned_sha256(
                        primary_lease.private_manifest,
                        primary_identity.private_train_manifest_sha256,
                        "private train manifest",
                    )
                    primary_lease.verify_dataset_unchanged()
                    for supplement_lease in supplement_leases:
                        supplement_lease.verify_unchanged()

                train_streaming_baseline(
                    pinned_multi_source_example_factory(sources),
                    arguments.output,
                    _training_config(
                        arguments,
                        {
                            "training_corpus_set": corpus_set,
                            "training_corpus_set_sha256": corpus_set_sha256,
                        },
                    ),
                    final_validation=verify_all_leases_unchanged,
                )
            print(str(arguments.output))
            return 0
        else:
            audited = audit_corpus_split(arguments.manifest, "train")
            manifest_value = json.loads(
                arguments.manifest.read_text(encoding="utf-8")
            )
        assignments = {
            game_id: (white, black)
            for game_id, white, black in audited.game_assignments
        }
        max_plies = manifest_value.get("maxPlies")
        if (
            isinstance(max_plies, bool)
            or not isinstance(max_plies, int)
            or max_plies <= 0
        ):
            raise ValueError("schema-6 training manifest maxPlies must be positive")
        train_streaming_baseline(
            example_factory(
                audited.dataset_path,
                assignments,
                max_rows_per_game=max_plies,
            ),
            arguments.output,
            _training_config(arguments, audited.provenance()),
            final_validation=lambda _staging_directory: _validate_unchanged_corpus(
                arguments.manifest, audited
            ),
        )
        print(str(arguments.output))
        return 0
    examples = group_training_examples(_read_ndjson(arguments.dataset))
    if arguments.command == "inspect":
        counts = {"train": 0, "validation": 0, "test": 0}
        for example in examples:
            counts[assign_split(example.seed).value] += 1
        print(json.dumps({"examples": len(examples), "splits": counts}, sort_keys=True))
        return 0
    train_baseline(
        examples,
        arguments.output,
        TrainingConfig(
            seed=arguments.seed,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            hidden_dimension=arguments.hidden_dimension,
            model_variant=arguments.model_variant,
            sequence_observation_mode=arguments.sequence_observation_mode,
            max_history=arguments.max_history,
            san_embedding_dimension=arguments.san_embedding_dimension,
            sequence_hidden_dimension=arguments.sequence_hidden_dimension,
            symbolic_hidden_dimension=arguments.symbolic_hidden_dimension,
            device=arguments.device,
            shuffle_buffer_size=arguments.shuffle_buffer_size,
            trigger_loss_weight=arguments.trigger_loss_weight,
            parameter_loss_weight=arguments.parameter_loss_weight,
            legal_mask_loss_weight=arguments.legal_mask_loss_weight,
        ),
    )
    print(str(arguments.output))
    return 0


def _training_config(
    arguments: argparse.Namespace, provenance: dict[str, object]
) -> TrainingConfig:
    legacy_examples = arguments.game_examples_per_epoch
    player_game_examples = arguments.player_game_examples_per_epoch
    if legacy_examples is not None and player_game_examples is not None:
        raise ValueError(
            "choose either --game-examples-per-epoch or "
            "--player-game-examples-per-epoch"
        )
    return TrainingConfig(
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        hidden_dimension=arguments.hidden_dimension,
        model_variant=arguments.model_variant,
        sequence_observation_mode=arguments.sequence_observation_mode,
        max_history=arguments.max_history,
        san_embedding_dimension=arguments.san_embedding_dimension,
        sequence_hidden_dimension=arguments.sequence_hidden_dimension,
        symbolic_hidden_dimension=arguments.symbolic_hidden_dimension,
        required_drawback_vocabulary=tuple(SYMBOLIC_RULE_IDS),
        corpus_provenance=provenance,
        device=arguments.device,
        shuffle_buffer_size=arguments.shuffle_buffer_size,
        game_examples_per_epoch=(
            16 if legacy_examples is None else legacy_examples
        ),
        player_game_examples_per_epoch=player_game_examples,
        trigger_loss_weight=arguments.trigger_loss_weight,
        parameter_loss_weight=arguments.parameter_loss_weight,
        legal_mask_loss_weight=arguments.legal_mask_loss_weight,
        execution_source_revision=getattr(
            arguments,
            "execution_source_revision",
            None,
        ),
    )


def _verify_clean_execution_revision(
    expected: object,
    *,
    ignored_untracked_paths: tuple[Path, ...] = (),
) -> str:
    if (
        not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-f]{40}", expected) is None
    ):
        raise ValueError(
            "execution source revision must be a full lowercase Git SHA"
        )
    source_repository_root = Path(__file__).resolve().parents[3]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        cwd=source_repository_root,
    ).stdout.strip()
    if head != expected:
        raise ValueError(
            "execution source revision differs from repository HEAD"
        )
    reported_repository_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            shell=False,
            cwd=source_repository_root,
        ).stdout.strip()
    ).resolve()
    if reported_repository_root != source_repository_root:
        raise ValueError(
            "loaded training source is not rooted in its reported Git "
            "repository"
        )
    repository_root = source_repository_root
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        cwd=repository_root,
    ).stdout
    final_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        cwd=repository_root,
    ).stdout.strip()
    if final_head != expected:
        raise ValueError(
            "execution source revision changed during source verification"
        )
    ignored = tuple(
        relative.as_posix()
        for path in ignored_untracked_paths
        if (
            relative := _relative_to_repository(
                path.resolve(),
                repository_root,
            )
        )
        is not None
    )
    dirty_entries = []
    for entry in status.split("\0"):
        if not entry:
            continue
        if entry.startswith("?? "):
            untracked = entry[3:]
            if any(
                untracked == root or untracked.startswith(f"{root}/")
                for root in ignored
            ):
                continue
        dirty_entries.append(entry)
    if dirty_entries:
        raise ValueError(
            "release training requires a clean repository source state"
        )
    return expected


def _relative_to_repository(
    path: Path,
    repository_root: Path,
) -> Path | None:
    try:
        return path.relative_to(repository_root)
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
