"""Canonical Python/browser parity evidence using a production Web Worker."""

from __future__ import annotations

import argparse
import chess.pgn
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Thread
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator, Mapping, Sequence
from ml.training.drawback_ml.records import FeatureRecord
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS
from ml.training.drawback_ml.ensemble import load_hybrid_ensemble

from .validation_gate import PROTOCOL_ID, _canonical_pretty
from .ensemble_calibration import ContentAddressedFile, load_ensemble_calibration
from .release_selection_bundle import ContentAddressedJson
from .ensemble_release import verify_ensemble_release
from .promotion_evaluator import (
    PromotionTemperatures,
    UNAVAILABLE_BROWSER_RULE_IDS,
    _calibration_fusion_policy,
    _calibration_temperature,
    _open_checkpoint_sources,
    _verified_checkpoint_bytes,
    _verify_calibration_binding,
    predict_calibrated_two_heads,
)


INPUT_FORMAT = "drawbacktrainer-browser-parity-input"
TRANSCRIPT_FORMAT = "drawbacktrainer-browser-worker-transcript"
EVIDENCE_FORMAT = "drawbacktrainer-browser-parity-evidence"
VERSION = 1
TOLERANCE = 1e-6
TOP_K = 5
SOURCE_PATHS = (
    "apps",
    "packages",
    "ml",
    "scripts",
    "engine",
    "package.json",
    "pnpm-lock.yaml",
)
SHA256_KEYS = {
    "ensembleSha256",
    "calibrationSha256",
    "fusionSelectionSha256",
    "pnpmLockSha256",
}
PUBLIC_FIXTURE_FORMAT = "drawbacktrainer-public-pgn-parity-fixture"
PUBLIC_FIXTURE_PROTOCOL_ID = "drawbacktrainer-public-pgn-parity-v1"
PUBLIC_FIXTURE_SEED_DOMAIN = "public-parity-v1"
PUBLIC_FIXTURE_ROOT_SEED = 0x5A17_2026
PUBLIC_FIXTURE_GAME_COUNT = 8
PUBLIC_FIXTURE_MAX_PLIES = 320
PUBLIC_FIXTURE_AGENTS = (
    "random-legal",
    "human-like-weak",
    "greedy-material",
)


@dataclass(frozen=True)
class PublicParityGame:
    game_id: str
    seed: int
    pgn: str
    final_fen: str
    ply_count: int
    result: str
    features: FeatureRecord


def load_public_parity_fixture(
    path: Path, expected_sha256: str
) -> tuple[Mapping[str, object], tuple[PublicParityGame, ...]]:
    """Authenticate and independently replay the candidate-independent PGNs."""

    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError("public parity fixture SHA-256 does not match")
    value = _strict_json(payload, "public parity fixture")
    if payload != _canonical_pretty(value):
        raise ValueError("public parity fixture is not canonical")
    if set(value) != {
        "format",
        "version",
        "protocol",
        "candidateInputs",
        "games",
    } or value.get("format") != PUBLIC_FIXTURE_FORMAT or value.get("version") != 1:
        raise ValueError("public parity fixture contract is invalid")
    if value.get("candidateInputs") != []:
        raise ValueError("public parity fixture selection is candidate-dependent")
    protocol = value.get("protocol")
    expected_protocol = {
        "id": PUBLIC_FIXTURE_PROTOCOL_ID,
        "seedDomain": PUBLIC_FIXTURE_SEED_DOMAIN,
        "rootSeed": PUBLIC_FIXTURE_ROOT_SEED,
        "gameCount": PUBLIC_FIXTURE_GAME_COUNT,
        "maxPlies": PUBLIC_FIXTURE_MAX_PLIES,
        "agentSchedule": list(PUBLIC_FIXTURE_AGENTS),
    }
    if protocol != expected_protocol:
        raise ValueError("public parity generator protocol differs")
    games = value.get("games")
    if not isinstance(games, list) or len(games) != PUBLIC_FIXTURE_GAME_COUNT:
        raise ValueError("public parity game count differs")
    replayed: list[PublicParityGame] = []
    ids: set[str] = set()
    for raw in games:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "seed",
            "pgn",
            "pgnSha256",
            "plyCount",
            "initialFen",
            "finalFen",
            "result",
            "finalPublicObservation",
        }:
            raise ValueError("public parity game fields are invalid")
        game_id = raw.get("id")
        pgn = raw.get("pgn")
        if (
            not isinstance(game_id, str)
            or not game_id
            or game_id in ids
            or not isinstance(pgn, str)
            or _sha256(pgn.encode("utf-8")) != raw.get("pgnSha256")
        ):
            raise ValueError("public parity game identity or PGN hash is invalid")
        parsed = chess.pgn.read_game(__import__("io").StringIO(pgn))
        if parsed is None:
            raise ValueError("public parity PGN cannot be parsed")
        board = parsed.board()
        plies = 0
        final_before = ""
        final_move = ""
        final_history: tuple[str, ...] = ()
        final_legal: tuple[str, ...] = ()
        history_san: list[str] = []
        try:
            for move in parsed.mainline_moves():
                if move not in board.legal_moves:
                    raise ValueError("public parity PGN contains an illegal move")
                final_before = board.fen(en_passant="legal")
                final_move = move.uci()
                final_history = tuple(history_san)
                final_legal = tuple(item.uci() for item in board.legal_moves)
                history_san.append(board.san(move))
                board.push(move)
                plies += 1
        except (ValueError, AssertionError) as error:
            raise ValueError("public parity PGN replay failed") from error
        if (
            parsed.errors
            or parsed.headers.get("Result") != raw.get("result")
            or parsed.headers.get("Result") not in {"1-0", "0-1", "1/2-1/2", "*"}
            or raw.get("initialFen")
            != "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            or raw.get("finalFen") != board.fen(en_passant="legal")
            or raw.get("plyCount") != plies
            or plies > PUBLIC_FIXTURE_MAX_PLIES
        ):
            raise ValueError("public parity replay metadata differs")
        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("public parity seed is invalid")
        observation = _public_feature_record(
            raw.get("finalPublicObservation"),
            final_before=final_before,
            final_move=final_move,
            final_history=final_history,
            final_legal=final_legal,
            final_ply=plies - 1,
        )
        ids.add(game_id)
        replayed.append(
            PublicParityGame(
                game_id=game_id,
                seed=seed,
                pgn=pgn,
                final_fen=board.fen(en_passant="legal"),
                ply_count=plies,
                result=str(raw["result"]),
                features=observation,
            )
        )
    return value, tuple(replayed)


def _public_feature_record(
    value: object,
    *,
    final_before: str,
    final_move: str,
    final_history: tuple[str, ...],
    final_legal: tuple[str, ...],
    final_ply: int,
) -> FeatureRecord:
    if not isinstance(value, Mapping) or set(value) != {
        "fenBefore",
        "move",
        "moveNumber",
        "ply",
        "playerColor",
        "historySan",
        "ordinaryLegalMoves",
        "symbolicFeatureVersion",
        "symbolic",
    }:
        raise ValueError("public parity observation fields are invalid")
    if (
        value.get("fenBefore") != final_before
        or value.get("move") != final_move
        or value.get("historySan") != list(final_history)
        or not isinstance(value.get("ordinaryLegalMoves"), list)
        or set(value["ordinaryLegalMoves"]) != set(final_legal)
        or len(value["ordinaryLegalMoves"]) != len(final_legal)
        or value.get("ply") != final_ply
        or value.get("moveNumber") != final_ply // 2 + 1
        or value.get("playerColor") != ("white" if final_ply % 2 == 0 else "black")
        or value.get("symbolicFeatureVersion") != 6
    ):
        raise ValueError("public parity observation disagrees with PGN replay")
    symbolic = value.get("symbolic")
    if not isinstance(symbolic, Mapping) or set(symbolic) != {
        "ruleIds",
        "whiteProbabilities",
        "blackProbabilities",
        "whiteEliminated",
        "blackEliminated",
    } or symbolic.get("ruleIds") != list(SYMBOLIC_RULE_IDS):
        raise ValueError("public parity symbolic vocabulary differs")
    heads: dict[str, tuple[tuple[float, ...], tuple[bool, ...]]] = {}
    for color in ("white", "black"):
        probabilities = symbolic.get(f"{color}Probabilities")
        eliminated = symbolic.get(f"{color}Eliminated")
        if (
            not isinstance(probabilities, list)
            or not isinstance(eliminated, list)
            or len(probabilities) != len(SYMBOLIC_RULE_IDS)
            or len(eliminated) != len(SYMBOLIC_RULE_IDS)
            or any(not isinstance(item, bool) for item in eliminated)
        ):
            raise ValueError("public parity symbolic dimensions differ")
        converted: list[float] = []
        for probability, masked in zip(probabilities, eliminated, strict=True):
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(float(probability))
                or not 0 <= float(probability) <= 1
                or (masked and probability != 0)
            ):
                raise ValueError("public parity symbolic probability is invalid")
            converted.append(float(probability))
        if abs(math.fsum(converted) - 1.0) > TOLERANCE:
            raise ValueError("public parity symbolic probabilities are not normalized")
        heads[color] = (tuple(converted), tuple(eliminated))
    return FeatureRecord(
        fen_before=final_before,
        move=final_move,
        move_number=final_ply // 2 + 1,
        ply=final_ply,
        player_color="white" if final_ply % 2 == 0 else "black",
        history_san=final_history,
        ordinary_legal_moves=tuple(value["ordinaryLegalMoves"]),
        clock_ms=None,
        symbolic_feature_version=6,
        symbolic_white_rule_probabilities=heads["white"][0],
        symbolic_black_rule_probabilities=heads["black"][0],
        symbolic_white_eliminated=heads["white"][1],
        symbolic_black_eliminated=heads["black"][1],
        public_evaluator_constraint=None,
    )


def build_public_parity_input(
    *,
    fixture: ContentAddressedFile,
    ensemble_release: ContentAddressedJson,
    calibration: ContentAddressedFile,
    browser_artifact: ContentAddressedFile,
    repository: Path,
    output: Path,
) -> Mapping[str, object]:
    """Publish Python ensemble expectations over the immutable public fixture."""

    fixture_value, games = load_public_parity_fixture(
        fixture.path, fixture.sha256
    )
    release = verify_ensemble_release(ensemble_release)
    calibration_value = load_ensemble_calibration(calibration)
    _verify_calibration_binding(calibration_value, ensemble_release)
    fusion_alpha, fusion_selection_sha256 = _calibration_fusion_policy(
        calibration_value
    )
    artifact_payload = browser_artifact.path.read_bytes()
    if _sha256(artifact_payload) != browser_artifact.sha256:
        raise ValueError("browser artifact SHA-256 does not match")
    artifact = _strict_json(artifact_payload, "browser artifact")
    artifact_ensemble = artifact.get("ensemble")
    artifact_calibration = artifact.get("calibration")
    if (
        not isinstance(artifact_ensemble, Mapping)
        or artifact_ensemble.get("sourceEnsembleReleaseSha256")
        != ensemble_release.sha256
        or artifact_ensemble.get("sourceFusionSelectionSha256")
        != fusion_selection_sha256
        or artifact_ensemble.get("selectedAlpha") != fusion_alpha
        or not isinstance(artifact_calibration, Mapping)
        or artifact_calibration.get("sourceCalibrationSha256")
        != calibration.sha256
    ):
        raise ValueError("browser artifact binds a different candidate")
    temperatures = PromotionTemperatures(
        white=_calibration_temperature(calibration_value, "white"),
        black=_calibration_temperature(calibration_value, "black"),
    )
    cases: list[Mapping[str, object]] = []
    with ExitStack() as stack:
        checkpoints = _open_checkpoint_sources(stack, ensemble_release, release)
        payloads = tuple(
            _verified_checkpoint_bytes(source, member.checkpoint_sha256)
            for source, member in zip(checkpoints, release.members, strict=True)
        )
        loaded = load_hybrid_ensemble(
            checkpoints,
            device="cpu",
            fusion_alpha=fusion_alpha,
            required_corpus_provenance={
                "training_corpus_set_sha256": release.training_corpus_set_sha256
            },
        )
        represented = tuple(
            rule_id
            for rule_id in SYMBOLIC_RULE_IDS
            if rule_id not in UNAVAILABLE_BROWSER_RULE_IDS
        )
        represented_indices = tuple(
            index
            for index, rule_id in enumerate(SYMBOLIC_RULE_IDS)
            if rule_id in represented
        )
        for game in games:
            prepared = predict_calibrated_two_heads(
                members=loaded.members,
                features=game.features,
                temperatures=temperatures,
                fusion_alpha=fusion_alpha,
            )
            expected: dict[str, object] = {}
            for color in ("white", "black"):
                values = [prepared[color][index] for index in represented_indices]
                mass = math.fsum(values)
                probabilities = {
                    rule_id: value / mass
                    for rule_id, value in zip(represented, values, strict=True)
                }
                ordered = sorted(
                    represented,
                    key=lambda rule_id: (-probabilities[rule_id], rule_id),
                )
                expected[color] = {
                    "probabilities": probabilities,
                    "topIds": ordered[:TOP_K],
                    "hardZeroIds": sorted(
                        rule_id
                        for rule_id in represented
                        if probabilities[rule_id] == 0
                    ),
                }
            cases.append(
                {
                    "id": game.game_id,
                    "pgn": game.pgn,
                    "pgnSha256": _sha256(game.pgn.encode("utf-8")),
                    "expected": expected,
                }
            )
        for source, expected_payload in zip(checkpoints, payloads, strict=True):
            source.seek(0)
            if source.read() != expected_payload:
                raise ValueError("ensemble checkpoint changed during parity inference")
    partition: Mapping[str, object] = {
        "id": PUBLIC_FIXTURE_PROTOCOL_ID,
        "split": "validation-parity",
        "selectionSha256": _sha256(
            _canonical_pretty([game.game_id for game in games])
        ),
        "publicExampleCount": len(cases),
    }
    fixture_digest = _sha256(
        _canonical_pretty({"partition": partition, "cases": cases})
    )
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    source_status = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *SOURCE_PATHS,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if source_status:
        raise ValueError("parity input production requires a clean source HEAD")
    value: Mapping[str, object] = {
        "format": INPUT_FORMAT,
        "version": VERSION,
        "protocolId": PROTOCOL_ID,
        "browserArtifactSha256": browser_artifact.sha256,
        "fixtureSha256": fixture_digest,
        "partition": partition,
        "bindings": {
            "ensembleSha256": ensemble_release.sha256,
            "calibrationSha256": calibration.sha256,
            "fusionSelectionSha256": fusion_selection_sha256,
            "sourceRevision": revision,
            "pnpmLockSha256": _digest_file(repository / "pnpm-lock.yaml"),
        },
        "publicFixture": {
            "file": fixture.path.name,
            "sha256": fixture.sha256,
            "generatorProtocol": fixture_value["protocol"],
        },
        "cases": cases,
    }
    payload = _canonical_pretty(value)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(payload)
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    return value


def load_authenticated_input(
    path: Path, expected_sha256: str, artifact_sha256: str
) -> Mapping[str, object]:
    """Load a canonical, content-addressed validation-parity input."""

    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError("parity input SHA-256 does not match")
    value = _strict_json(payload, "parity input")
    if payload != _canonical_pretty(value):
        raise ValueError("parity input is not canonical")
    if set(value) != {
        "format",
        "version",
        "protocolId",
        "browserArtifactSha256",
        "fixtureSha256",
        "partition",
        "bindings",
        "publicFixture",
        "cases",
    }:
        raise ValueError("parity input fields are invalid")
    if (
        value.get("format") != INPUT_FORMAT
        or value.get("version") != VERSION
        or value.get("protocolId") != PROTOCOL_ID
        or value.get("browserArtifactSha256") != artifact_sha256
    ):
        raise ValueError("parity input identity is invalid")
    fixture_sha256 = value.get("fixtureSha256")
    if (
        not isinstance(fixture_sha256, str)
        or len(fixture_sha256) != 64
        or any(character not in "0123456789abcdef" for character in fixture_sha256)
    ):
        raise ValueError("parity fixture SHA-256 is invalid")
    partition = value.get("partition")
    bindings = value.get("bindings")
    public_fixture = value.get("publicFixture")
    cases = value.get("cases")
    if (
        not isinstance(partition, Mapping)
        or set(partition) != {
            "id",
            "split",
            "selectionSha256",
            "publicExampleCount",
        }
        or partition.get("split") != "validation-parity"
        or not isinstance(partition.get("publicExampleCount"), int)
        or int(partition["publicExampleCount"]) <= 0
        or not isinstance(bindings, Mapping)
        or set(bindings) != {
            "ensembleSha256",
            "calibrationSha256",
            "fusionSelectionSha256",
            "sourceRevision",
            "pnpmLockSha256",
        }
        or not isinstance(cases, list)
        or not cases
        or not isinstance(public_fixture, Mapping)
        or set(public_fixture) != {"file", "sha256", "generatorProtocol"}
        or public_fixture.get("generatorProtocol")
        != {
            "id": PUBLIC_FIXTURE_PROTOCOL_ID,
            "seedDomain": PUBLIC_FIXTURE_SEED_DOMAIN,
            "rootSeed": PUBLIC_FIXTURE_ROOT_SEED,
            "gameCount": PUBLIC_FIXTURE_GAME_COUNT,
            "maxPlies": PUBLIC_FIXTURE_MAX_PLIES,
            "agentSchedule": list(PUBLIC_FIXTURE_AGENTS),
        }
        or not isinstance(public_fixture.get("file"), str)
        or Path(str(public_fixture["file"])).name != public_fixture["file"]
        or not isinstance(public_fixture.get("sha256"), str)
        or len(str(public_fixture["sha256"])) != 64
    ):
        raise ValueError("parity input authentication metadata is invalid")
    partition_id = partition.get("id")
    selection_sha256 = partition.get("selectionSha256")
    if (
        not isinstance(partition_id, str)
        or not partition_id.strip()
        or not isinstance(selection_sha256, str)
        or len(selection_sha256) != 64
        or any(character not in "0123456789abcdef" for character in selection_sha256)
        or partition.get("publicExampleCount") != len(cases)
    ):
        raise ValueError("parity partition identity or count is invalid")
    for key in SHA256_KEYS:
        digest = bindings.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"parity input {key} is invalid")
    revision = bindings.get("sourceRevision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("parity input sourceRevision is invalid")
    expected_fixture = _sha256(
        _canonical_pretty({"partition": partition, "cases": cases})
    )
    if fixture_sha256 != expected_fixture:
        raise ValueError("parity fixture SHA-256 does not match partition and cases")
    # Inputs contain public PGN and Python posteriors only. Ground-truth
    # drawback labels, parameters, evaluator facts, and sealed metrics are
    # deliberately not accepted by this schema.
    forbidden = {"truth", "label", "hiddenParameters", "evaluatorFacts"}
    if any(key in forbidden for case in cases if isinstance(case, Mapping) for key in case):
        raise ValueError("parity input exposes sealed-test or hidden data")
    case_ids: set[str] = set()
    for case in cases:
        _validate_case(case)
        case_id = str(case["id"])
        if case_id in case_ids:
            raise ValueError("parity case ids are not unique")
        case_ids.add(case_id)
    return value


def _validate_case(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "id",
        "pgn",
        "pgnSha256",
        "expected",
    }:
        raise ValueError("parity case fields are invalid")
    if (
        not isinstance(value.get("id"), str)
        or not value["id"]
        or not isinstance(value.get("pgn"), str)
        or not value["pgn"].strip()
        or value.get("pgnSha256")
        != _sha256(str(value.get("pgn")).encode("utf-8"))
    ):
        raise ValueError("parity case identity or PGN is invalid")
    expected = value.get("expected")
    if not isinstance(expected, Mapping) or set(expected) != {"white", "black"}:
        raise ValueError("parity case expected heads are invalid")
    for color in ("white", "black"):
        head = expected.get(color)
        if (
            not isinstance(head, Mapping)
            or set(head) != {"probabilities", "topIds", "hardZeroIds"}
        ):
            raise ValueError(f"parity {color} head fields are invalid")
        probabilities = head.get("probabilities")
        top_ids = head.get("topIds")
        zero_ids = head.get("hardZeroIds")
        if (
            not isinstance(probabilities, Mapping)
            or not probabilities
            or not isinstance(top_ids, list)
            or len(top_ids) != TOP_K
            or not isinstance(zero_ids, list)
        ):
            raise ValueError(f"parity {color} head is invalid")
        ids = set(probabilities)
        if (
            any(not isinstance(rule_id, str) or not rule_id for rule_id in ids)
            or len(top_ids) != len(set(top_ids))
            or len(zero_ids) != len(set(zero_ids))
            or any(rule_id not in ids for rule_id in top_ids + zero_ids)
        ):
            raise ValueError(f"parity {color} rule ids are invalid")
        total = 0.0
        for rule_id, probability in probabilities.items():
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not 0 <= float(probability) <= 1
            ):
                raise ValueError(f"parity {color} probability is invalid")
            total += float(probability)
            if (probability == 0) != (rule_id in zero_ids):
                raise ValueError(f"parity {color} hard-zero set is inconsistent")
        if abs(total - 1.0) > TOLERANCE:
            raise ValueError(f"parity {color} probabilities do not sum to one")
        ordered = sorted(ids, key=lambda rule_id: (-float(probabilities[rule_id]), rule_id))
        if top_ids != ordered[: len(top_ids)]:
            raise ValueError(f"parity {color} Top-k order is inconsistent")


def authenticate_runtime_bindings(
    repository: Path,
    artifact: Path,
    calibration: Path,
    parity_input: Mapping[str, object],
) -> None:
    bindings = parity_input["bindings"]
    if not isinstance(bindings, Mapping):
        raise ValueError("parity input bindings are invalid")
    artifact_value = _strict_json(artifact.read_bytes(), "browser artifact")
    ensemble = artifact_value.get("ensemble")
    calibration_section = artifact_value.get("calibration")
    if (
        not isinstance(ensemble, Mapping)
        or ensemble.get("sourceEnsembleReleaseSha256")
        != bindings.get("ensembleSha256")
        or ensemble.get("sourceFusionSelectionSha256")
        != bindings.get("fusionSelectionSha256")
        or not isinstance(calibration_section, Mapping)
        or calibration_section.get("sourceCalibrationSha256")
        != bindings.get("calibrationSha256")
        or _digest_file(calibration) != bindings.get("calibrationSha256")
        or _digest_file(repository / "pnpm-lock.yaml")
        != bindings.get("pnpmLockSha256")
    ):
        raise ValueError("candidate or dependency binding differs")
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *SOURCE_PATHS,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if (
        revision != bindings.get("sourceRevision")
        or dirty.returncode != 0
        or dirty.stdout.strip()
    ):
        raise ValueError("source revision is not the bound clean HEAD")


class _ResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inside = False
        self.fragments: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "pre" and ("id", "browser-parity-result") in attrs:
            self.inside = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self.inside = False

    def handle_data(self, data: str) -> None:
        if self.inside:
            self.fragments.append(data)


@contextmanager
def _serve(directory: Path) -> Iterator[str]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/browser-parity.html"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def run_real_worker(
    repository: Path,
    browser: Path,
    artifact: Path,
    parity_input: Path,
) -> bytes:
    """Build the production app and capture a real-browser Worker transcript."""

    browser_payload = browser.read_bytes()
    browser_sha256 = _sha256(browser_payload)
    browser_version_process = subprocess.run(
        [str(browser), "--version"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    browser_version = browser_version_process.stdout.strip()
    if not browser_version:
        raise ValueError("browser did not report a version")
    subprocess.run(
        ["pnpm", "--filter", "@drawbackguesser/web", "build"],
        cwd=repository,
        check=True,
    )
    dist = repository / "apps" / "web" / "dist"
    with tempfile.TemporaryDirectory(prefix="drawback-parity-") as temporary:
        webroot = Path(temporary) / "web"
        shutil.copytree(dist, webroot)
        shutil.copy2(artifact, webroot / "browser-model.json")
        shutil.copy2(parity_input, webroot / "browser-parity-input.json")
        with _serve(webroot) as url:
            completed = subprocess.run(
                [
                    str(browser),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    "--disable-background-networking",
                    "--virtual-time-budget=120000",
                    "--dump-dom",
                    url,
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=180,
            )
    if completed.returncode != 0:
        raise ValueError(
            f"headless browser failed with exit code {completed.returncode}"
        )
    parser = _ResultParser()
    parser.feed(completed.stdout)
    if not parser.fragments:
        raise ValueError("headless browser returned no parity transcript")
    value = dict(
        _strict_json("".join(parser.fragments).encode(), "worker transcript")
    )
    value["browserRuntime"] = {
        "binarySha256": browser_sha256,
        "version": browser_version,
    }
    return _canonical_pretty(value)


def publish_evidence(
    *,
    transcript_payload: bytes,
    browser_artifact_sha256: str,
    calibration_sha256: str,
    parity_input: Mapping[str, object],
    parity_input_sha256: str,
    output: Path,
) -> Mapping[str, object]:
    """Verify a Worker transcript and atomically publish review evidence."""

    transcript = _strict_json(transcript_payload, "worker transcript")
    if transcript_payload != _canonical_pretty(transcript):
        raise ValueError("worker transcript is not canonical")
    maximum = transcript.get("maximumAbsoluteDifference")
    browser_runtime = transcript.get("browserRuntime")
    passed = (
        transcript.get("format") == TRANSCRIPT_FORMAT
        and transcript.get("version") == VERSION
        and transcript.get("browserArtifactSha256")
        == browser_artifact_sha256
        and transcript.get("workerE2ePassed") is True
        and transcript.get("topKIdentical") is True
        and transcript.get("hardZeroSetsIdentical") is True
        and isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and 0 <= float(maximum) <= TOLERANCE
        and isinstance(browser_runtime, Mapping)
        and set(browser_runtime) == {"binarySha256", "version"}
        and isinstance(browser_runtime.get("binarySha256"), str)
        and len(str(browser_runtime["binarySha256"])) == 64
        and isinstance(browser_runtime.get("version"), str)
        and bool(str(browser_runtime["version"]).strip())
    )
    if not passed:
        raise ValueError("browser Worker parity transcript is not passing")
    evidence: Mapping[str, object] = {
        "format": EVIDENCE_FORMAT,
        "version": VERSION,
        "protocol_id": PROTOCOL_ID,
        "browser_artifact_sha256": browser_artifact_sha256,
        "calibration_sha256": calibration_sha256,
        "passed": True,
        "max_absolute_difference": float(maximum),
        "top_k_identical": True,
        "hard_zero_sets_identical": True,
        "worker_e2e_passed": True,
        "parity_input_sha256": parity_input_sha256,
        "transcript_sha256": _sha256(transcript_payload),
        "fixture_sha256": parity_input["fixtureSha256"],
        "partition_selection_sha256": parity_input["partition"][
            "selectionSha256"
        ],
        "ensemble_sha256": parity_input["bindings"]["ensembleSha256"],
        "source_revision": parity_input["bindings"]["sourceRevision"],
        "pnpm_lock_sha256": parity_input["bindings"]["pnpmLockSha256"],
        "browser_binary_sha256": browser_runtime["binarySha256"],
        "browser_version": browser_runtime["version"],
        "public_fixture_sha256": parity_input["publicFixture"]["sha256"],
    }
    payload = _canonical_pretty(evidence)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o644)
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(payload)
    return evidence


def verify_transcript_bindings(
    transcript_payload: bytes, parity_input: Mapping[str, object]
) -> None:
    """Require the browser transcript to echo authenticated public bindings."""

    transcript = _strict_json(transcript_payload, "worker transcript")
    for transcript_key, input_key in (
        ("protocolId", "protocolId"),
        ("browserArtifactSha256", "browserArtifactSha256"),
        ("fixtureSha256", "fixtureSha256"),
        ("partition", "partition"),
        ("bindings", "bindings"),
        ("publicFixture", "publicFixture"),
    ):
        if transcript.get(transcript_key) != parity_input.get(input_key):
            raise ValueError(
                f"worker transcript {transcript_key} binding differs"
            )


def _digest_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run production browser Worker parity and publish evidence."
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--browser-artifact", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--transcript-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_sha256 = _digest_file(args.browser_artifact)
    if args.transcript_output.exists() or args.evidence_output.exists():
        raise FileExistsError("parity output already exists")
    parity_input = load_authenticated_input(
        args.input, args.input_sha256, artifact_sha256
    )
    authenticate_runtime_bindings(
        args.repository.resolve(),
        args.browser_artifact.resolve(),
        args.calibration.resolve(),
        parity_input,
    )
    transcript = run_real_worker(
        args.repository.resolve(),
        args.browser.resolve(),
        args.browser_artifact.resolve(),
        args.input.resolve(),
    )
    verify_transcript_bindings(transcript, parity_input)
    # Both transcript and approval projection are canonical and no-clobber.
    try:
        descriptor = os.open(
            args.transcript_output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(transcript)
        publish_evidence(
            transcript_payload=transcript,
            browser_artifact_sha256=artifact_sha256,
            calibration_sha256=_digest_file(args.calibration),
            parity_input=parity_input,
            parity_input_sha256=args.input_sha256,
            output=args.evidence_output,
        )
    except BaseException:
        args.transcript_output.unlink(missing_ok=True)
        args.evidence_output.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
