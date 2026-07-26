"""Bounded-memory iteration over an authenticated schema-6 training split."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sqlite3
import tempfile
from types import MappingProxyType
from typing import Any, BinaryIO, TypeVar

from .records import TrainingExample, group_training_examples


T = TypeVar("T")
GAME_BALANCED_POLICY = "equal-game-moves"
GAME_BALANCED_POLICY_VERSION = 1
GAME_OPPORTUNITY_PAIRED_POLICY = "equal-game-opportunity-paired-v2"
GAME_OPPORTUNITY_PAIRED_POLICY_VERSION = 2
PLAYER_GAME_BALANCED_POLICY = "observed-player-drawback-color-game-balanced"
PLAYER_GAME_BALANCED_POLICY_VERSION = 1
HARD_NEGATIVE_PLAYER_GAME_FRACTION_CAP = 0.25
OPPORTUNITY_PAIRED_UNIFORM_ANCHORS = 8
OPPORTUNITY_PAIRED_MAX_PAIRS = 4
OPPORTUNITY_PAIRED_HORIZON_BANDS = (
    (0, 4),
    (5, 9),
    (10, 19),
    (20, 39),
    (40, None),
)
_SAFE_SOURCE_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


@dataclass(frozen=True, eq=False)
class PinnedExampleSource:
    """One authenticated input in a canonical multi-source stream.

    The assignment mapping is copied into a read-only snapshot so callers
    cannot change label authentication between repeated passes. The open
    binary handle itself pins the authenticated bytes against pathname
    replacement.
    """

    namespace: str
    source: BinaryIO
    assignments: Mapping[str, tuple[str, str]]
    max_rows_per_game: int

    def __post_init__(self) -> None:
        if (
            _SAFE_SOURCE_NAMESPACE.fullmatch(self.namespace) is None
            or self.namespace in {".", ".."}
        ):
            raise ValueError(
                "source namespace must be 1-64 lowercase ASCII letters, "
                "digits, dots, underscores, or hyphens, start with a letter "
                "or digit, and cannot be . or .."
            )
        if (
            isinstance(self.max_rows_per_game, bool)
            or not isinstance(self.max_rows_per_game, int)
            or self.max_rows_per_game <= 0
        ):
            raise ValueError("max_rows_per_game must be a positive integer")
        if self.source.closed:
            raise ValueError("source handle must be open")
        if not self.source.readable() or not self.source.seekable():
            raise ValueError("source handle must be readable and seekable")
        assignment_snapshot = dict(self.assignments)
        if any(
            not isinstance(game_id, str)
            or not game_id
            or not isinstance(drawbacks, tuple)
            or len(drawbacks) != 2
            or any(
                not isinstance(drawback, str) or not drawback
                for drawback in drawbacks
            )
            for game_id, drawbacks in assignment_snapshot.items()
        ):
            raise ValueError(
                "assignments must map non-empty game IDs to two non-empty drawbacks"
            )
        object.__setattr__(
            self,
            "assignments",
            MappingProxyType(assignment_snapshot),
        )


@dataclass(eq=False)
class PlayerGameSamplingPlan:
    """Disk-backed inventory for balanced observed-player supervision.

    Labels are used only to choose player-game rows. They are never copied into
    ``FeatureRecord`` or otherwise exposed to model inputs. Corpus-sized
    identity and selection indexes live in a private SQLite database so the
    trainer's Python heap remains bounded.
    """

    labels: tuple[str, ...]
    player_games_per_stratum: int
    hard_negative_player_games_per_stratum: int
    hard_negative_player_games_per_epoch: int
    raw_examples: int
    row_bearing_games: int
    database_path: Path
    temporary_directory: Path
    _closed: bool = False

    def __enter__(self) -> PlayerGameSamplingPlan:
        self._ensure_open()
        return self

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort fallback for exceptional exits. Production callers close
        # explicitly; this prevents abandoned private indexes if setup fails.
        self.close()

    def _ensure_open(self) -> None:
        if self._closed or not self.database_path.is_file():
            raise RuntimeError("player-game sampling plan is closed")

    def connect(self) -> sqlite3.Connection:
        self._ensure_open()
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA temp_store = FILE")
        return connection

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self.temporary_directory, ignore_errors=True)

    @property
    def stratum_count(self) -> int:
        return len(self.labels) * 2

    def metadata(self, examples_per_player_game: int) -> dict[str, object]:
        self._ensure_open()
        effective_player_games = (
            self.stratum_count * self.player_games_per_stratum
        )
        return {
            "policy": PLAYER_GAME_BALANCED_POLICY,
            "version": PLAYER_GAME_BALANCED_POLICY_VERSION,
            "balance_unit": "observed-player-drawback-color-game",
            "examples_per_player_game": examples_per_player_game,
            "player_games_per_stratum": self.player_games_per_stratum,
            "strata": self.stratum_count,
            "effective_player_games_per_epoch": effective_player_games,
            "effective_examples_per_epoch": (
                effective_player_games * examples_per_player_game
            ),
            "hard_negative_fraction_cap": (
                HARD_NEGATIVE_PLAYER_GAME_FRACTION_CAP
            ),
            "hard_negative_player_games_per_stratum_cap": (
                self.hard_negative_player_games_per_stratum
            ),
            "hard_negative_player_games_per_epoch": (
                self.hard_negative_player_games_per_epoch
            ),
            "raw_rows": self.raw_examples,
            "row_bearing_games": self.row_bearing_games,
            "drawback_supervision": "observed-color-moves-v1",
            "label_feature_boundary": "sampling-only-no-feature-mutation",
        }


def _is_hard_negative_game(game_id: str) -> bool:
    return game_id.startswith("hard-negative-")


def build_player_game_sampling_plan(
    values: Iterable[TrainingExample],
    *,
    labels: tuple[str, ...],
) -> PlayerGameSamplingPlan:
    """Inventory authenticated player-games and freeze an equal-stratum quota."""

    if not labels or len(set(labels)) != len(labels):
        raise ValueError("sampling labels must be non-empty and unique")
    allowed = frozenset(labels)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="drawback-player-game-plan-")
    )
    database_path = temporary_directory / "sampling.sqlite3"
    raw_examples = 0
    connection: sqlite3.Connection | None = None
    try:
        with sqlite3.connect(database_path) as connection:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute(
                """
                CREATE TABLE player_games (
                    game_id TEXT NOT NULL,
                    color TEXT NOT NULL,
                    label TEXT NOT NULL,
                    hard_negative INTEGER NOT NULL,
                    PRIMARY KEY (game_id, color)
                )
                """
            )
            connection.execute(
                "CREATE TABLE games (game_id TEXT PRIMARY KEY)"
            )
            for example in values:
                raw_examples += 1
                color = example.features.player_color
                if color not in {"white", "black"}:
                    raise ValueError("training example player color is invalid")
                label = (
                    example.white_drawback
                    if color == "white"
                    else example.black_drawback
                )
                if label not in allowed:
                    raise ValueError(
                        "observed drawback is outside the frozen vocabulary: "
                        f"{label}"
                    )
                hard_negative = int(
                    _is_hard_negative_game(example.game_id)
                )
                connection.execute(
                    "INSERT OR IGNORE INTO games(game_id) VALUES (?)",
                    (example.game_id,),
                )
                inserted = connection.execute(
                    """
                    INSERT OR IGNORE INTO player_games(
                        game_id, color, label, hard_negative
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (example.game_id, color, label, hard_negative),
                )
                if inserted.rowcount == 0:
                    prior = connection.execute(
                        """
                        SELECT label, hard_negative
                        FROM player_games
                        WHERE game_id = ? AND color = ?
                        """,
                        (example.game_id, color),
                    ).fetchone()
                    if prior != (label, hard_negative):
                        raise ValueError(
                            "player-game sampling identity changes labels "
                            "or source"
                        )
            connection.execute(
                """
                CREATE INDEX player_games_stratum
                ON player_games(label, color, hard_negative)
                """
            )
            primary_counts = [
                int(row[0])
                for row in connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM player_games
                    WHERE hard_negative = 0
                    GROUP BY label, color
                    """
                )
            ]
            expected_strata = len(labels) * 2
            if (
                len(primary_counts) != expected_strata
                or min(primary_counts, default=0) <= 0
            ):
                raise ValueError(
                    "every drawback/color stratum requires a primary "
                    "player-game"
                )
            quota = min(primary_counts)
            hard_cap = int(quota * HARD_NEGATIVE_PLAYER_GAME_FRACTION_CAP)
            hard_negative_per_epoch = sum(
                min(int(row[0]), hard_cap)
                for row in connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM player_games
                    WHERE hard_negative = 1
                    GROUP BY label, color
                    """
                )
            )
            row_bearing_games = int(
                connection.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            )
        connection.close()
        return PlayerGameSamplingPlan(
            labels=labels,
            player_games_per_stratum=quota,
            hard_negative_player_games_per_stratum=hard_cap,
            hard_negative_player_games_per_epoch=hard_negative_per_epoch,
            raw_examples=raw_examples,
            row_bearing_games=row_bearing_games,
            database_path=database_path,
            temporary_directory=temporary_directory,
        )
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise


def _player_game_rank(
    seed: int,
    epoch: int,
    label: str,
    color: str,
    source: str,
    game_id: str,
) -> bytes:
    material = json.dumps(
        [
            PLAYER_GAME_BALANCED_POLICY,
            PLAYER_GAME_BALANCED_POLICY_VERSION,
            seed,
            epoch,
            label,
            color,
            source,
            game_id,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).digest()


def _prepare_selected_player_games(
    plan: PlayerGameSamplingPlan,
    *,
    connection: sqlite3.Connection,
    seed: int,
    epoch: int,
) -> int:
    connection.execute("DROP TABLE IF EXISTS selected_player_games")
    connection.execute(
        """
        CREATE TEMP TABLE selected_player_games (
            game_id TEXT NOT NULL,
            color TEXT NOT NULL,
            PRIMARY KEY (game_id, color)
        )
        """
    )
    connection.create_function(
        "player_game_rank",
        6,
        lambda label, color, source, game_id, local_seed, local_epoch: (
            _player_game_rank(
                int(local_seed),
                int(local_epoch),
                str(label),
                str(color),
                str(source),
                str(game_id),
            )
        ),
        deterministic=True,
    )
    for label in plan.labels:
        for color in ("white", "black"):
            supplemental_available = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM player_games
                    WHERE label = ? AND color = ? AND hard_negative = 1
                    """,
                    (label, color),
                ).fetchone()[0]
            )
            supplemental_count = min(
                supplemental_available,
                plan.hard_negative_player_games_per_stratum,
            )
            primary_count = plan.player_games_per_stratum - supplemental_count
            for hard_negative, source, count in (
                (0, "primary", primary_count),
                (1, "hard-negative", supplemental_count),
            ):
                connection.execute(
                    """
                    INSERT INTO selected_player_games(game_id, color)
                    SELECT game_id, color
                    FROM player_games
                    WHERE label = ? AND color = ? AND hard_negative = ?
                    ORDER BY player_game_rank(
                        label, color, ?, game_id, ?, ?
                    ), game_id
                    LIMIT ?
                    """,
                    (
                        label,
                        color,
                        hard_negative,
                        source,
                        seed,
                        epoch,
                        count,
                    ),
                )
    connection.commit()
    return int(
        connection.execute(
            "SELECT COUNT(*) FROM selected_player_games"
        ).fetchone()[0]
    )


def player_game_balanced_examples(
    values: Iterable[TrainingExample],
    *,
    plan: PlayerGameSamplingPlan,
    seed: int,
    epoch: int,
    examples_per_player_game: int,
) -> Iterator[TrainingExample]:
    """Yield equal samples per drawback/color player-game with source capping."""

    if seed < 0 or epoch <= 0 or examples_per_player_game <= 0:
        raise ValueError("sampling seed, epoch, and example count are invalid")
    connection = plan.connect()
    try:
        selected_count = _prepare_selected_player_games(
            plan,
            connection=connection,
            seed=seed,
            epoch=epoch,
        )
    except BaseException:
        connection.close()
        raise
    observed_selected = 0
    raw_count = 0
    game_count = 0
    current_game: str | None = None
    expected_identities: dict[str, tuple[str, bool]] = {}
    selected_colors: frozenset[str] = frozenset()
    observed_colors: set[str] = set()
    game_rows: dict[str, list[TrainingExample]] = {
        "white": [],
        "black": [],
    }

    def flush() -> Iterator[TrainingExample]:
        nonlocal game_count, observed_selected
        if current_game is None:
            return
        game_count += 1
        if observed_colors != set(expected_identities):
            raise RuntimeError(
                "training player-game identities changed between streaming "
                "passes"
            )
        for color in ("white", "black"):
            if color not in selected_colors:
                continue
            examples = game_rows[color]
            if not examples:
                raise RuntimeError(
                    "selected player-game contains no observed moves"
                )
            observed_selected += 1
            rng = random.Random(
                int.from_bytes(
                    _player_game_rank(
                        seed, epoch, "", color, "row", current_game
                    )[:16],
                    "big",
                )
            )
            if len(examples) >= examples_per_player_game:
                indices = rng.sample(
                    range(len(examples)), examples_per_player_game
                )
            else:
                indices = [
                    rng.randrange(len(examples))
                    for _ in range(examples_per_player_game)
                ]
            for index in indices:
                yield examples[index]

    try:
        for example in values:
            raw_count += 1
            if current_game is None or example.game_id != current_game:
                if current_game is not None:
                    yield from flush()
                    game_rows["white"].clear()
                    game_rows["black"].clear()
                    observed_colors.clear()
                current_game = example.game_id
                expected_identities = {
                    str(color): (str(label), bool(hard_negative))
                    for color, label, hard_negative in connection.execute(
                        """
                        SELECT color, label, hard_negative
                        FROM player_games
                        WHERE game_id = ?
                        """,
                        (current_game,),
                    )
                }
                selected_colors = frozenset(
                    str(row[0])
                    for row in connection.execute(
                        """
                        SELECT color FROM selected_player_games
                        WHERE game_id = ?
                        """,
                        (current_game,),
                    )
                )
            color = example.features.player_color
            label = (
                example.white_drawback
                if color == "white"
                else example.black_drawback
            )
            actual_identity = (
                label,
                _is_hard_negative_game(example.game_id),
            )
            if expected_identities.get(color) != actual_identity:
                raise RuntimeError(
                    "training player-game label or source identity changed "
                    "between streaming passes"
                )
            observed_colors.add(color)
            if color in selected_colors:
                game_rows[color].append(example)
        if current_game is not None:
            yield from flush()
        if (
            raw_count != plan.raw_examples
            or game_count != plan.row_bearing_games
        ):
            raise RuntimeError(
                "training corpus changed between streaming passes"
            )
        if observed_selected != selected_count:
            raise RuntimeError(
                "selected player-game disappeared from training corpus"
            )
    finally:
        connection.close()


def _stable_game_seed(seed: int, epoch: int, game_id: str) -> int:
    material = json.dumps(
        [
            GAME_BALANCED_POLICY,
            GAME_BALANCED_POLICY_VERSION,
            seed,
            epoch,
            game_id,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def game_balanced_examples(
    values: Iterable[TrainingExample],
    *,
    seed: int,
    epoch: int,
    examples_per_game: int,
    expected_raw_examples: int | None = None,
    expected_games: int | None = None,
) -> Iterator[TrainingExample]:
    """Yield exactly K deterministic samples from every contiguous game."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if examples_per_game <= 0:
        raise ValueError("examples_per_game must be positive")
    current_game: str | None = None
    game: list[TrainingExample] = []
    raw_count = 0
    game_count = 0

    def sample() -> Iterator[TrainingExample]:
        if not game:
            return
        rng = random.Random(
            _stable_game_seed(seed, epoch, game[0].game_id)
        )
        if len(game) >= examples_per_game:
            indices = rng.sample(range(len(game)), examples_per_game)
        else:
            indices = [
                rng.randrange(len(game)) for _ in range(examples_per_game)
            ]
        for index in indices:
            yield game[index]

    for value in values:
        raw_count += 1
        if current_game is not None and value.game_id != current_game:
            yield from sample()
            game.clear()
            game_count += 1
        current_game = value.game_id
        game.append(value)
    if game:
        yield from sample()
        game_count += 1
    if (
        expected_raw_examples is not None
        and raw_count != expected_raw_examples
    ) or (
        expected_games is not None
        and game_count != expected_games
    ):
        raise RuntimeError("training corpus changed between streaming passes")


def _opportunity_paired_game_seed(
    seed: int, epoch: int, game_id: str, purpose: str
) -> int:
    material = json.dumps(
        [
            GAME_OPPORTUNITY_PAIRED_POLICY,
            GAME_OPPORTUNITY_PAIRED_POLICY_VERSION,
            seed,
            epoch,
            game_id,
            purpose,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def _opportunity_paired_tiebreak(
    seed: int,
    epoch: int,
    game_id: str,
    purpose: str,
    *indices: int,
) -> bytes:
    material = json.dumps(
        [
            GAME_OPPORTUNITY_PAIRED_POLICY,
            GAME_OPPORTUNITY_PAIRED_POLICY_VERSION,
            seed,
            epoch,
            game_id,
            purpose,
            *indices,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).digest()


def _rotated_opportunity_bands(
    seed: int,
    epoch: int,
    game_id: str,
) -> tuple[int, ...]:
    material = json.dumps(
        [
            GAME_OPPORTUNITY_PAIRED_POLICY,
            GAME_OPPORTUNITY_PAIRED_POLICY_VERSION,
            seed,
            game_id,
            "rotating-band-order",
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    band_count = len(OPPORTUNITY_PAIRED_HORIZON_BANDS)
    start = (int.from_bytes(hashlib.sha256(material).digest()[:8], "big") + epoch - 1) % band_count
    return tuple((start + offset) % band_count for offset in range(band_count))


def _opportunity_horizon_band(ply: int) -> int:
    for index, (lower, upper) in enumerate(OPPORTUNITY_PAIRED_HORIZON_BANDS):
        if ply >= lower and (upper is None or ply <= upper):
            return index
    raise ValueError("training example ply must be non-negative")


def _is_rule_opportunity(example: TrainingExample) -> bool:
    ordinary = frozenset(example.features.ordinary_legal_moves)
    drawback = frozenset(example.drawback_legal_moves)
    return example.rule_triggered or ordinary != drawback


def _sample_opportunity_paired_game(
    game: list[TrainingExample],
    *,
    seed: int,
    epoch: int,
    examples_per_game: int,
) -> Iterator[TrainingExample]:
    if len(game) < examples_per_game:
        for example in game:
            yield example
        rng = random.Random(
            _opportunity_paired_game_seed(
                seed, epoch, game[0].game_id, "short-game-replacement"
            )
        )
        for _ in range(examples_per_game - len(game)):
            yield game[rng.randrange(len(game))]
        return

    selected: list[int] = []
    selected_set: set[int] = set()
    anchor_count = min(
        OPPORTUNITY_PAIRED_UNIFORM_ANCHORS,
        examples_per_game,
    )
    anchor_rng = random.Random(
        _opportunity_paired_game_seed(
            seed, epoch, game[0].game_id, "uniform-anchors"
        )
    )
    for index in anchor_rng.sample(range(len(game)), anchor_count):
        selected.append(index)
        selected_set.add(index)

    remaining_slots = examples_per_game - len(selected)
    maximum_pairs = min(
        OPPORTUNITY_PAIRED_MAX_PAIRS,
        remaining_slots // 2,
    )
    opportunity_by_band: dict[int, list[int]] = {
        index: [] for index in range(len(OPPORTUNITY_PAIRED_HORIZON_BANDS))
    }
    for index, example in enumerate(game):
        if index not in selected_set and _is_rule_opportunity(example):
            opportunity_by_band[_opportunity_horizon_band(example.features.ply)].append(
                index
            )
    for candidates in opportunity_by_band.values():
        candidates.sort(
            key=lambda index: _opportunity_paired_tiebreak(
                seed,
                epoch,
                game[0].game_id,
                "opportunity",
                index,
            )
        )

    pairs: list[tuple[int, int]] = []
    while len(pairs) < maximum_pairs:
        added = False
        for band in _rotated_opportunity_bands(
            seed,
            epoch,
            game[0].game_id,
        ):
            candidates = opportunity_by_band[band]
            while candidates:
                opportunity_index = candidates.pop(0)
                opportunity = game[opportunity_index]
                controls = [
                    index
                    for index, candidate in enumerate(game)
                    if index not in selected_set
                    and index != opportunity_index
                    and not _is_rule_opportunity(candidate)
                    and candidate.features.player_color
                    == opportunity.features.player_color
                    and _opportunity_horizon_band(candidate.features.ply) == band
                ]
                if not controls:
                    continue
                control_index = min(
                    controls,
                    key=lambda index: (
                        abs(
                            game[index].features.ply
                            - opportunity.features.ply
                        ),
                        _opportunity_paired_tiebreak(
                            seed,
                            epoch,
                            game[0].game_id,
                            "control",
                            opportunity_index,
                            index,
                        ),
                    ),
                )
                pairs.append((opportunity_index, control_index))
                selected_set.add(opportunity_index)
                selected_set.add(control_index)
                added = True
                break
            if len(pairs) == maximum_pairs:
                break
        if not added:
            break

    for opportunity_index, control_index in pairs:
        selected.extend((opportunity_index, control_index))

    fill_count = examples_per_game - len(selected)
    if fill_count:
        fill_candidates = [
            index for index in range(len(game)) if index not in selected_set
        ]
        fill_rng = random.Random(
            _opportunity_paired_game_seed(
                seed, epoch, game[0].game_id, "uniform-fill"
            )
        )
        selected.extend(fill_rng.sample(fill_candidates, fill_count))

    for index in selected:
        yield game[index]


def game_opportunity_paired_examples(
    values: Iterable[TrainingExample],
    *,
    seed: int,
    epoch: int,
    examples_per_game: int,
    expected_raw_examples: int | None = None,
    expected_games: int | None = None,
) -> Iterator[TrainingExample]:
    """Yield deterministic equal-game samples with opportunity/control pairs.

    Selection may inspect truth-side move masks, but it returns the original
    immutable examples and never adds truth-derived model features.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if examples_per_game <= 0:
        raise ValueError("examples_per_game must be positive")
    current_game: str | None = None
    game: list[TrainingExample] = []
    raw_count = 0
    game_count = 0

    def sample() -> Iterator[TrainingExample]:
        if game:
            yield from _sample_opportunity_paired_game(
                game,
                seed=seed,
                epoch=epoch,
                examples_per_game=examples_per_game,
            )

    for value in values:
        raw_count += 1
        if current_game is not None and value.game_id != current_game:
            yield from sample()
            game.clear()
            game_count += 1
        current_game = value.game_id
        game.append(value)
    if game:
        yield from sample()
        game_count += 1
    if (
        expected_raw_examples is not None
        and raw_count != expected_raw_examples
    ) or (
        expected_games is not None
        and game_count != expected_games
    ):
        raise RuntimeError("training corpus changed between streaming passes")


def iter_authenticated_examples_from_binary(
    source: BinaryIO,
    game_assignments: Mapping[str, tuple[str, str]],
    *,
    max_rows_per_game: int,
    source_name: str = "authenticated training dataset",
) -> Iterator[TrainingExample]:
    """Yield one contiguous game's examples at a time.

    The schema-6 audit has already authenticated canonical game ordering. This
    reader deliberately retains at most one game's raw rows.
    """

    if max_rows_per_game <= 0:
        raise ValueError("max_rows_per_game must be positive")
    current_game: str | None = None
    rows: list[dict[str, Any]] = []

    def flush() -> Iterator[TrainingExample]:
        if rows:
            yield from group_training_examples(rows, game_assignments)

    source.seek(0)
    for line_number, payload in enumerate(source, start=1):
        try:
            line = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"{source_name}:{line_number} is not valid UTF-8"
            ) from error
        if not line.strip():
            raise ValueError(f"{source_name}:{line_number} is blank")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source_name}:{line_number} must contain an object")
        game_id = value.get("gameId")
        if not isinstance(game_id, str) or not game_id:
            raise ValueError(f"{source_name}:{line_number} has an invalid gameId")
        if current_game is not None and game_id != current_game:
            yield from flush()
            rows.clear()
        current_game = game_id
        rows.append(value)
        if len(rows) > max_rows_per_game:
            raise ValueError(
                f"game {game_id} exceeds the authenticated maximum ply count"
            )
    yield from flush()


def iter_authenticated_examples(
    path: Path,
    game_assignments: Mapping[str, tuple[str, str]],
    *,
    max_rows_per_game: int,
) -> Iterator[TrainingExample]:
    """Legacy pathname reader for monolithic research corpora."""

    with path.open("rb") as source:
        yield from iter_authenticated_examples_from_binary(
            source,
            game_assignments,
            max_rows_per_game=max_rows_per_game,
            source_name=str(path),
        )


def deterministic_shuffle_buffer(
    values: Iterable[T],
    *,
    seed: int,
    buffer_size: int,
) -> Iterator[T]:
    """Apply a deterministic bounded-memory streaming shuffle."""

    if buffer_size <= 0:
        raise ValueError("shuffle buffer size must be positive")
    rng = random.Random(seed)
    buffer: list[T] = []
    for value in values:
        if len(buffer) < buffer_size:
            buffer.append(value)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = value
    rng.shuffle(buffer)
    yield from buffer


def iter_batches(values: Iterable[T], batch_size: int) -> Iterator[tuple[T, ...]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[T] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def example_factory(
    path: Path,
    game_assignments: Mapping[str, tuple[str, str]],
    *,
    max_rows_per_game: int,
) -> Callable[[], Iterator[TrainingExample]]:
    return lambda: iter_authenticated_examples(
        path,
        game_assignments,
        max_rows_per_game=max_rows_per_game,
    )


def pinned_example_factory(
    source: BinaryIO,
    game_assignments: Mapping[str, tuple[str, str]],
    *,
    max_rows_per_game: int,
    source_name: str = "authenticated training dataset",
) -> Callable[[], Iterator[TrainingExample]]:
    """Build repeated passes over an already authenticated, pinned handle."""

    return lambda: iter_authenticated_examples_from_binary(
        source,
        game_assignments,
        max_rows_per_game=max_rows_per_game,
        source_name=source_name,
    )


def pinned_multi_source_example_factory(
    sources: Iterable[PinnedExampleSource],
) -> Callable[[], Iterator[TrainingExample]]:
    """Compose pinned sources without allowing game identities to collide.

    Source order is deliberately neither sorted nor normalized: the caller's
    canonical order is part of training reproducibility. Only the game ID is
    namespaced. In particular, the public ``FeatureRecord`` object is reused
    exactly and receives no source or profile field.
    """

    pinned_sources = tuple(sources)
    namespaces: set[str] = set()
    handle_ids: set[int] = set()
    for source in pinned_sources:
        if source.namespace in namespaces:
            raise ValueError(
                f"duplicate source namespace: {source.namespace}"
            )
        handle_id = id(source.source)
        if handle_id in handle_ids:
            raise ValueError("duplicate pinned source handle")
        namespaces.add(source.namespace)
        handle_ids.add(handle_id)

    def factory() -> Iterator[TrainingExample]:
        for source in pinned_sources:
            for example in iter_authenticated_examples_from_binary(
                source.source,
                source.assignments,
                max_rows_per_game=source.max_rows_per_game,
                source_name=source.namespace,
            ):
                yield replace(
                    example,
                    game_id=f"{source.namespace}:{example.game_id}",
                )

    return factory
