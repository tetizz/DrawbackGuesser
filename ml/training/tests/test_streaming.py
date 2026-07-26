from __future__ import annotations

from dataclasses import fields, replace
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import tracemalloc
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.streaming import (
    GAME_OPPORTUNITY_PAIRED_POLICY,
    GAME_OPPORTUNITY_PAIRED_POLICY_VERSION,
    HARD_NEGATIVE_PLAYER_GAME_FRACTION_CAP,
    PLAYER_GAME_BALANCED_POLICY,
    PinnedExampleSource,
    _rotated_opportunity_bands,
    build_player_game_sampling_plan,
    deterministic_shuffle_buffer,
    game_balanced_examples,
    game_opportunity_paired_examples,
    iter_authenticated_examples_from_binary,
    iter_authenticated_examples,
    iter_batches,
    player_game_balanced_examples,
    pinned_example_factory,
    pinned_multi_source_example_factory,
)
from drawback_ml.streaming_training import _scan
from drawback_ml.records import group_training_examples
from test_records import row


class StreamingTests(unittest.TestCase):
    @staticmethod
    def _binary_rows(values: list[dict[str, object]]) -> BytesIO:
        return BytesIO(
            "".join(json.dumps(value) + "\n" for value in values).encode(
                "utf-8"
            )
        )

    @staticmethod
    def _examples(game_id: str, count: int):
        rows = [
            {
                **row("white" if index % 2 == 0 else "black", "vegan"),
                "gameId": game_id,
                "ply": index,
            }
            for index in range(max(count, 2))
        ]
        base = group_training_examples(rows)
        return [
            replace(
                base[index % len(base)],
                game_id=game_id,
                features=replace(
                    base[index % len(base)].features,
                    ply=index,
                    move=f"a{index}",
                ),
            )
            for index in range(count)
        ]

    def test_game_balancing_is_exact_deterministic_and_epoch_specific(self) -> None:
        values = self._examples("long", 20) + self._examples("other", 9)
        first = list(
            game_balanced_examples(
                values, seed=17, epoch=1, examples_per_game=6
            )
        )
        repeated = list(
            game_balanced_examples(
                iter(values), seed=17, epoch=1, examples_per_game=6
            )
        )
        second_epoch = list(
            game_balanced_examples(
                values, seed=17, epoch=2, examples_per_game=6
            )
        )
        self.assertEqual(first, repeated)
        self.assertEqual(
            {
                game_id: sum(item.game_id == game_id for item in first)
                for game_id in ("long", "other")
            },
            {"long": 6, "other": 6},
        )
        for game_id in ("long", "other"):
            selected = [
                item.features.ply for item in first if item.game_id == game_id
            ]
            self.assertEqual(len(selected), len(set(selected)))
        self.assertNotEqual(
            [item.features.ply for item in first],
            [item.features.ply for item in second_epoch],
        )

    def test_short_games_are_sampled_deterministically_with_replacement(self) -> None:
        values = [
            replace(
                item,
                black_parameters=None,
                black_parameters_observed=False,
            )
            for item in self._examples("short", 2)
        ]
        sampled = list(
            game_balanced_examples(
                values, seed=4, epoch=1, examples_per_game=7
            )
        )
        self.assertEqual(len(sampled), 7)
        self.assertLess(len({item.features.ply for item in sampled}), 7)
        self.assertTrue(all(item in values for item in sampled))
        self.assertTrue(
            all(
                item.black_parameters is None
                and not item.black_parameters_observed
                for item in sampled
            )
        )

    def test_player_game_balancing_caps_hard_negatives_and_preserves_examples(
        self,
    ) -> None:
        values = []
        for label in ("alpha", "beta"):
            for index in range(4):
                values.extend(
                    replace(
                        item,
                        white_drawback=label,
                        black_drawback=label,
                    )
                    for item in self._examples(
                        f"primary:{label}-{index}", 6
                    )
                )
            for index in range(3):
                values.extend(
                    replace(
                        item,
                        white_drawback=label,
                        black_drawback=label,
                    )
                    for item in self._examples(
                        f"hard-negative-101:{label}-{index}", 6
                    )
                )
        plan = build_player_game_sampling_plan(
            values,
            labels=("alpha", "beta"),
        )
        self.addCleanup(plan.close)
        before = tuple(values)
        first = list(
            player_game_balanced_examples(
                values,
                plan=plan,
                seed=91,
                epoch=1,
                examples_per_player_game=2,
            )
        )
        repeated = list(
            player_game_balanced_examples(
                iter(values),
                plan=plan,
                seed=91,
                epoch=1,
                examples_per_player_game=2,
            )
        )
        second_epoch = list(
            player_game_balanced_examples(
                values,
                plan=plan,
                seed=91,
                epoch=2,
                examples_per_player_game=2,
            )
        )

        self.assertEqual(PLAYER_GAME_BALANCED_POLICY,
                         "observed-player-drawback-color-game-balanced")
        self.assertEqual(HARD_NEGATIVE_PLAYER_GAME_FRACTION_CAP, 0.25)
        self.assertEqual(plan.player_games_per_stratum, 4)
        self.assertEqual(plan.hard_negative_player_games_per_stratum, 1)
        self.assertEqual(first, repeated)
        self.assertEqual(tuple(values), before)
        self.assertTrue(all(item in values for item in first))
        self.assertEqual(len(first), 32)
        self.assertEqual(
            sum(item.game_id.startswith("hard-negative-") for item in first),
            8,
        )
        for label in ("alpha", "beta"):
            for color in ("white", "black"):
                self.assertEqual(
                    sum(
                        item.features.player_color == color
                        and (
                            item.white_drawback
                            if color == "white"
                            else item.black_drawback
                        )
                        == label
                        for item in first
                    ),
                    8,
                )
        self.assertNotEqual(
            [(item.game_id, item.features.ply) for item in first],
            [(item.game_id, item.features.ply) for item in second_epoch],
        )
        metadata = plan.metadata(2)
        self.assertEqual(metadata["effective_examples_per_epoch"], 32)
        self.assertEqual(metadata["hard_negative_player_games_per_epoch"], 4)
        self.assertEqual(
            metadata["label_feature_boundary"],
            "sampling-only-no-feature-mutation",
        )

        epoch_one = player_game_balanced_examples(
            values,
            plan=plan,
            seed=91,
            epoch=1,
            examples_per_player_game=2,
        )
        prefix = [next(epoch_one), next(epoch_one)]
        list(
            player_game_balanced_examples(
                values,
                plan=plan,
                seed=91,
                epoch=2,
                examples_per_player_game=2,
            )
        )
        self.assertEqual(prefix + list(epoch_one), first)

    def test_player_game_plan_requires_primary_coverage_for_every_stratum(
        self,
    ) -> None:
        values = [
            replace(item, white_drawback="alpha", black_drawback="alpha")
            for item in self._examples("primary:only-alpha", 4)
        ]
        with self.assertRaisesRegex(ValueError, "every drawback/color"):
            build_player_game_sampling_plan(
                values,
                labels=("alpha", "beta"),
            )

    def test_player_game_plan_keeps_large_identity_index_off_python_heap(
        self,
    ) -> None:
        templates = self._examples("template", 2)

        def values():
            for index in range(10_000):
                label = "alpha" if index % 2 == 0 else "beta"
                game_id = f"primary:bounded-{index}"
                for template in templates:
                    yield replace(
                        template,
                        game_id=game_id,
                        white_drawback=label,
                        black_drawback=label,
                    )

        tracemalloc.start()
        plan = build_player_game_sampling_plan(
            values(),
            labels=("alpha", "beta"),
        )
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.addCleanup(plan.close)

        self.assertLess(peak, 8 * 1024 * 1024)
        self.assertFalse(hasattr(plan, "identities"))
        self.assertFalse(hasattr(plan, "primary"))
        with plan.connect() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM player_games"
                ).fetchone()[0],
                20_000,
            )

    def test_player_game_sampler_rejects_same_count_identity_drift(
        self,
    ) -> None:
        values = []
        for label in ("alpha", "beta"):
            values.extend(
                replace(
                    item,
                    white_drawback=label,
                    black_drawback=label,
                )
                for item in self._examples(f"primary:{label}", 4)
            )
        plan = build_player_game_sampling_plan(
            values,
            labels=("alpha", "beta"),
        )
        self.addCleanup(plan.close)
        changed = list(values)
        changed[0] = replace(changed[0], white_drawback="beta")
        with self.assertRaisesRegex(
            RuntimeError, "label or source identity changed"
        ):
            list(
                player_game_balanced_examples(
                    changed,
                    plan=plan,
                    seed=5,
                    epoch=1,
                    examples_per_player_game=2,
                )
            )

        with plan.connect() as connection:
            identity = connection.execute(
                """
                SELECT game_id, color FROM player_games
                ORDER BY game_id, color LIMIT 1
                """
            ).fetchone()
            self.assertIsNotNone(identity)
            connection.execute(
                """
                UPDATE player_games
                SET hard_negative = 1 - hard_negative
                WHERE game_id = ? AND color = ?
                """,
                identity,
            )
        with self.assertRaisesRegex(
            RuntimeError, "label or source identity changed"
        ):
            list(
                player_game_balanced_examples(
                    values,
                    plan=plan,
                    seed=5,
                    epoch=1,
                    examples_per_player_game=2,
                )
            )

    def test_opportunity_paired_sampling_is_exact_deterministic_and_immutable(
        self,
    ) -> None:
        values = self._examples("paired", 50)
        opportunities = {4, 8, 12, 16, 24, 32, 42, 46}
        prepared = [
            replace(
                item,
                rule_triggered=item.features.ply == 46,
                drawback_legal_moves=(
                    ("a1a2",)
                    if item.features.ply in opportunities
                    else ("a1a2", "b1b2")
                ),
                features=replace(
                    item.features,
                    ordinary_legal_moves=("b1b2", "a1a2", "a1a2"),
                ),
            )
            for item in values
        ]
        before = tuple(prepared)
        first = list(
            game_opportunity_paired_examples(
                prepared, seed=23, epoch=1, examples_per_game=16
            )
        )
        repeated = list(
            game_opportunity_paired_examples(
                iter(prepared), seed=23, epoch=1, examples_per_game=16
            )
        )
        second_epoch = list(
            game_opportunity_paired_examples(
                prepared, seed=23, epoch=2, examples_per_game=16
            )
        )

        self.assertEqual(
            GAME_OPPORTUNITY_PAIRED_POLICY,
            "equal-game-opportunity-paired-v2",
        )
        self.assertEqual(GAME_OPPORTUNITY_PAIRED_POLICY_VERSION, 2)
        self.assertEqual(first, repeated)
        self.assertEqual(len(first), 16)
        self.assertEqual(len({item.features.ply for item in first}), 16)
        self.assertNotEqual(
            [item.features.ply for item in first],
            [item.features.ply for item in second_epoch],
        )
        selected_opportunities = [
            item for item in first if item.features.ply in opportunities
        ]
        self.assertGreaterEqual(len(selected_opportunities), 4)
        self.assertEqual(tuple(prepared), before)
        self.assertTrue(all(item in prepared for item in first))

    def test_opportunity_pairs_use_same_color_band_and_closest_control(
        self,
    ) -> None:
        values = self._examples("controls", 48)
        opportunities = {6, 14, 22, 42}
        prepared = [
            replace(
                item,
                rule_triggered=item.features.ply in opportunities,
                drawback_legal_moves=("a1a2", "b1b2"),
                features=replace(
                    item.features,
                    ordinary_legal_moves=("a1a2", "b1b2"),
                ),
            )
            for item in values
        ]
        sampled = list(
            game_opportunity_paired_examples(
                prepared, seed=9, epoch=1, examples_per_game=16
            )
        )
        paired_tail = sampled[8:]
        observed_pairs = []
        for index in range(0, len(paired_tail) - 1, 2):
            opportunity = paired_tail[index]
            control = paired_tail[index + 1]
            if opportunity.features.ply not in opportunities:
                break
            observed_pairs.append((opportunity, control))

        self.assertTrue(observed_pairs)
        for opportunity, control in observed_pairs:
            opportunity_ply = opportunity.features.ply
            same_color_same_band = {
                candidate.features.ply
                for candidate in prepared
                if candidate.features.player_color
                == prepared[opportunity_ply].features.player_color
                and (
                    (5 <= candidate.features.ply <= 9 and opportunity_ply == 6)
                    or (
                        10 <= candidate.features.ply <= 19
                        and opportunity_ply == 14
                    )
                    or (
                        20 <= candidate.features.ply <= 39
                        and opportunity_ply == 22
                    )
                    or (candidate.features.ply >= 40 and opportunity_ply == 42)
                )
                and candidate.features.ply not in opportunities
            }
            closest_distance = min(
                abs(ply - opportunity_ply) for ply in same_color_same_band
            )
            self.assertEqual(
                control.features.player_color,
                opportunity.features.player_color,
            )
            self.assertEqual(
                abs(control.features.ply - opportunity_ply),
                closest_distance,
            )

    def test_opportunity_paired_shortage_fills_and_replaces_only_short_games(
        self,
    ) -> None:
        long_game = self._examples("no-opportunities", 20)
        long_sample = list(
            game_opportunity_paired_examples(
                long_game, seed=4, epoch=1, examples_per_game=16
            )
        )
        short_game = [
            replace(
                item,
                rule_triggered=item.features.ply == 1,
                drawback_legal_moves=(
                    ("a1a2",)
                    if item.features.ply == 1
                    else ("a1a2", "b1b2")
                ),
                features=replace(
                    item.features,
                    ordinary_legal_moves=("a1a2", "b1b2"),
                ),
            )
            for item in self._examples("short-paired", 3)
        ]
        short_first = list(
            game_opportunity_paired_examples(
                short_game, seed=4, epoch=1, examples_per_game=7
            )
        )
        short_repeated = list(
            game_opportunity_paired_examples(
                short_game, seed=4, epoch=1, examples_per_game=7
            )
        )

        self.assertEqual(len(long_sample), 16)
        self.assertEqual(len({item.features.ply for item in long_sample}), 16)
        self.assertEqual(short_first, short_repeated)
        self.assertEqual(len(short_first), 7)
        self.assertLess(len({item.features.ply for item in short_first}), 7)
        self.assertEqual(
            {item.features.ply for item in short_first},
            {0, 1, 2},
        )
        self.assertTrue(any(item.rule_triggered for item in short_first))

    def test_opportunity_band_priority_rotates_across_epochs(self) -> None:
        orders = [
            _rotated_opportunity_bands(31, epoch, "rotation")
            for epoch in range(1, 6)
        ]

        self.assertEqual(len(set(orders)), 5)
        self.assertEqual({order[0] for order in orders}, {0, 1, 2, 3, 4})
        self.assertTrue(all(set(order) == {0, 1, 2, 3, 4} for order in orders))

    def test_game_balancing_does_not_read_multiple_future_games(self) -> None:
        first = self._examples("one", 5)
        second = self._examples("two", 4)

        def guarded():
            yield from first
            yield second[0]
            raise AssertionError("sampler read beyond the next game boundary")

        stream = game_balanced_examples(
            guarded(), seed=1, epoch=1, examples_per_game=3
        )
        self.assertEqual(next(stream).game_id, "one")

    def test_groups_lazily_with_one_game_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.ndjson"
            values = [
                {**row("white", "vegan"), "gameId": "one", "ply": 0},
                {**row("black", "checkers"), "gameId": "one", "ply": 1},
                {**row("white", "truant"), "gameId": "two", "ply": 0},
            ]
            path.write_text(
                "".join(json.dumps(value) + "\n" for value in values),
                encoding="utf-8",
            )
            stream = iter_authenticated_examples(
                path,
                {"one": ("vegan", "checkers"), "two": ("truant", "vegan")},
                max_rows_per_game=2,
            )
            first = next(stream)
            self.assertEqual(first.game_id, "one")
            self.assertEqual([item.game_id for item in stream], ["one", "two"])

    def test_rejects_a_game_larger_than_the_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.ndjson"
            values = [
                {**row("white", "vegan"), "gameId": "one", "ply": 0},
                {**row("black", "checkers"), "gameId": "one", "ply": 1},
            ]
            path.write_text(
                "".join(json.dumps(value) + "\n" for value in values),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "maximum ply"):
                list(
                    iter_authenticated_examples(
                        path,
                        {"one": ("vegan", "checkers")},
                        max_rows_per_game=1,
                    )
                )

    def test_pinned_factory_ignores_authenticated_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.ndjson"
            values = [
                {**row("white", "vegan"), "gameId": "one", "ply": 0},
                {**row("black", "checkers"), "gameId": "one", "ply": 1},
            ]
            path.write_text(
                "".join(json.dumps(value) + "\n" for value in values),
                encoding="utf-8",
            )
            replacement = path.with_name("replacement.ndjson")
            replacement.write_text(
                json.dumps(
                    {**row("white", "truant"), "gameId": "malicious", "ply": 0}
                )
                + "\n",
                encoding="utf-8",
            )
            with path.open("rb") as source:
                factory = pinned_example_factory(
                    source,
                    {"one": ("vegan", "checkers")},
                    max_rows_per_game=2,
                    source_name=path.name,
                )
                before = [(item.game_id, item.features.ply) for item in factory()]
                try:
                    os.replace(replacement, path)
                except PermissionError:
                    self.skipTest(
                        "platform sharing policy disallows replacing an open file"
                    )
                after = [(item.game_id, item.features.ply) for item in factory()]
            self.assertEqual(before, [("one", 0), ("one", 1)])
            self.assertEqual(after, before)
            self.assertIn("malicious", path.read_text(encoding="utf-8"))

    def test_multi_source_factory_preserves_order_and_repeated_passes(self) -> None:
        first_handle = self._binary_rows(
            [{**row("white", "vegan"), "gameId": "same", "historySan": ["e4"]}]
        )
        second_handle = self._binary_rows(
            [
                {
                    **row("black", "checkers"),
                    "gameId": "same",
                    "historySan": ["d4"],
                }
            ]
        )
        first = PinnedExampleSource(
            "base",
            first_handle,
            {"same": ("vegan", "checkers")},
            1,
        )
        second = PinnedExampleSource(
            "targeted",
            second_handle,
            {"same": ("truant", "checkers")},
            1,
        )
        factory = pinned_multi_source_example_factory((first, second))
        expected = [
            ("base:same", ("e4",)),
            ("targeted:same", ("d4",)),
        ]
        self.assertEqual(
            [(item.game_id, item.features.history_san) for item in factory()],
            expected,
        )
        self.assertEqual(
            [(item.game_id, item.features.history_san) for item in factory()],
            expected,
        )
        reversed_factory = pinned_multi_source_example_factory((second, first))
        self.assertEqual(
            [item.game_id for item in reversed_factory()],
            ["targeted:same", "base:same"],
        )

    def test_namespacing_changes_only_game_id_and_prevents_collision(self) -> None:
        payload = [{**row("white", "vegan"), "gameId": "collision"}]
        direct_handle = self._binary_rows(payload)
        direct = list(
            iter_authenticated_examples_from_binary(
                direct_handle,
                {"collision": ("vegan", "checkers")},
                max_rows_per_game=1,
            )
        )[0]
        combined_handle = self._binary_rows(payload)
        combined = list(
            pinned_multi_source_example_factory(
                (
                    PinnedExampleSource(
                        "clean",
                        combined_handle,
                        {"collision": ("vegan", "checkers")},
                        1,
                    ),
                )
            )()
        )[0]
        self.assertEqual(combined, replace(direct, game_id="clean:collision"))
        self.assertEqual(combined.features, direct.features)
        self.assertEqual(
            {field.name for field in fields(combined.features)},
            {field.name for field in fields(direct.features)},
        )
        self.assertNotIn("clean", repr(combined.features))

        other_handle = self._binary_rows(payload)
        colliding = list(
            pinned_multi_source_example_factory(
                (
                    PinnedExampleSource(
                        "one",
                        combined_handle,
                        {"collision": ("vegan", "checkers")},
                        1,
                    ),
                    PinnedExampleSource(
                        "two",
                        other_handle,
                        {"collision": ("vegan", "checkers")},
                        1,
                    ),
                )
            )()
        )
        self.assertEqual(
            [item.game_id for item in colliding],
            ["one:collision", "two:collision"],
        )

    def test_multi_source_balancing_counts_only_row_bearing_games(self) -> None:
        populated = self._binary_rows(
            [
                {**row("white", "vegan"), "gameId": "one", "ply": 0},
                {**row("black", "checkers"), "gameId": "one", "ply": 1},
                {**row("white", "vegan"), "gameId": "two", "ply": 0},
            ]
        )
        empty = self._binary_rows([])
        factory = pinned_multi_source_example_factory(
            (
                PinnedExampleSource(
                    "games",
                    populated,
                    {
                        "one": ("vegan", "checkers"),
                        "two": ("vegan", "checkers"),
                    },
                    2,
                ),
                PinnedExampleSource("empty", empty, {}, 2),
            )
        )
        sampled = list(
            game_balanced_examples(
                factory(),
                seed=12,
                epoch=1,
                examples_per_game=4,
                expected_raw_examples=3,
                expected_games=2,
            )
        )
        self.assertEqual(len(sampled), 8)
        self.assertEqual(
            {
                game_id: sum(item.game_id == game_id for item in sampled)
                for game_id in ("games:one", "games:two")
            },
            {"games:one": 4, "games:two": 4},
        )
        self.assertFalse(any(item.game_id.startswith("empty:") for item in sampled))

    def test_multi_source_scan_observes_union_without_source_features(self) -> None:
        def source_rows(
            first_drawback: str,
            second_drawback: str,
            token: str,
            parameter: str,
        ) -> tuple[BytesIO, dict[str, tuple[str, str]]]:
            values = []
            assignments = {}
            for game_id, white, black in (
                ("forward", first_drawback, second_drawback),
                ("reverse", second_drawback, first_drawback),
            ):
                values.append(
                    {
                        **row("white", white),
                        "gameId": game_id,
                        "historySan": [token],
                        "hiddenParameters": {"kind": parameter},
                    }
                )
                assignments[game_id] = (white, black)
            return self._binary_rows(values), assignments

        base_handle, base_assignments = source_rows(
            "vegan", "checkers", "e4", "base"
        )
        extra_handle, extra_assignments = source_rows(
            "truant", "spice-of-life", "Nf3", "extra"
        )
        factory = pinned_multi_source_example_factory(
            (
                PinnedExampleSource(
                    "base", base_handle, base_assignments, 1
                ),
                PinnedExampleSource(
                    "extra", extra_handle, extra_assignments, 1
                ),
            )
        )
        parameter_vocabulary, tokenizer, count, game_count = _scan(
            factory,
            ("checkers", "spice-of-life", "truant", "vegan"),
            sequence=True,
            max_history=8,
        )
        self.assertEqual(count, 4)
        self.assertEqual(game_count, 4)
        self.assertEqual(
            set(parameter_vocabulary.tokens),
            {'{"kind":"base"}', '{"kind":"extra"}'},
        )
        self.assertIsNotNone(tokenizer)
        assert tokenizer is not None
        self.assertTrue({"e4", "Nf3"}.issubset(tokenizer.vocabulary))
        for example in factory():
            feature_repr = repr(example.features)
            self.assertNotIn("base:", feature_repr)
            self.assertNotIn("extra:", feature_repr)
            self.assertNotIn("kind", feature_repr)

    def test_multi_source_rejects_unsafe_namespaces_and_duplicate_inputs(self) -> None:
        for namespace in ("", "UPPER", "has space", "colon:name", ".", ".."):
            with self.subTest(namespace=namespace):
                with self.assertRaisesRegex(ValueError, "namespace"):
                    PinnedExampleSource(
                        namespace,
                        self._binary_rows([]),
                        {},
                        1,
                    )

        with self.assertRaisesRegex(ValueError, "positive integer"):
            PinnedExampleSource("bad-count", self._binary_rows([]), {}, True)
        with self.assertRaisesRegex(ValueError, "assignments"):
            PinnedExampleSource(
                "bad-assignment",
                self._binary_rows([]),
                {"game": ("vegan", "")},
                1,
            )

        one = self._binary_rows([])
        first = PinnedExampleSource("one", one, {}, 1)
        with self.assertRaisesRegex(ValueError, "namespace"):
            pinned_multi_source_example_factory(
                (first, PinnedExampleSource("one", self._binary_rows([]), {}, 1))
            )
        with self.assertRaisesRegex(ValueError, "handle"):
            pinned_multi_source_example_factory(
                (first, PinnedExampleSource("two", one, {}, 1))
            )

    def test_shuffle_and_batching_are_deterministic_and_bounded(self) -> None:
        values = range(100)
        first = list(
            deterministic_shuffle_buffer(values, seed=91, buffer_size=7)
        )
        second = list(
            deterministic_shuffle_buffer(values, seed=91, buffer_size=7)
        )
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(values))
        self.assertNotEqual(first, list(values))
        batches = list(iter_batches(first, 9))
        self.assertEqual(max(map(len, batches)), 9)
        self.assertEqual(sum(map(len, batches)), 100)


if __name__ == "__main__":
    unittest.main()
