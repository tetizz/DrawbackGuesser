from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from drawback_ml.evaluator_schedule_contract import expected_balanced_slots


class EvaluatorScheduleContractTests(unittest.TestCase):
    def test_matches_typescript_balanced_schedule_vector(self) -> None:
        slots = expected_balanced_slots(
            root_seed=20260724,
            split="train",
            seeds=(
                3602057890,
                612476940,
                3139080135,
                4060177737,
                1447226103,
                2108878359,
                1193876134,
                2959693938,
            ),
            rule_ids=("vegan", "checkers", "truant", "pacman"),
            agent_ids=("random-legal", "greedy-material"),
        )
        self.assertEqual(
            [
                (
                    slot.white_rule_id,
                    slot.black_rule_id,
                    slot.white_agent_id,
                    slot.black_agent_id,
                )
                for slot in slots[:4]
            ],
            [
                ("truant", "pacman", "random-legal", "random-legal"),
                ("pacman", "checkers", "greedy-material", "greedy-material"),
                ("checkers", "vegan", "random-legal", "random-legal"),
                ("vegan", "truant", "greedy-material", "greedy-material"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
