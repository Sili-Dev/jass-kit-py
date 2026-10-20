import unittest

import numpy as np

from jass.agents.mcts.rollout import copy_state, random_rollout
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber
from .positions import random_state


class CheckedGameSim(GameSim):
    """Simulation that fails on invalid cards."""
    def action_play_card(self, card: int) -> None:
        assert card in np.flatnonzero(self.rule.get_valid_cards_from_state(self.state)), 'invalid card'
        super().action_play_card(card)


def sim_for(state, sim_class=GameSim):
    sim = sim_class(rule=RuleSchieber())
    sim._state = copy_state(state)
    return sim


class RolloutTestCase(unittest.TestCase):

    def setUp(self):
        self.rule = RuleSchieber()

    def test_copy_keeps_trick_view(self):
        state = random_state(3, 13)
        self.assertEqual(state.nr_tricks, 3)
        self.assertEqual(state.nr_cards_in_trick, 1)
        sim = sim_for(state)
        card = int(np.flatnonzero(self.rule.get_valid_cards_from_state(sim.state))[0])
        sim.action_play_card(card)
        self.assertEqual(sim.state.tricks[3, 1], card)
        self.assertEqual(state.tricks[3, 1], -1)

    def test_rollout_finishes_with_157(self):
        for seed in range(50):
            sim = sim_for(random_state(seed, seed % 36))
            random_rollout(sim, self.rule, np.random.default_rng(seed))
            self.assertTrue(sim.is_done())
            self.assertEqual(sim.state.points.sum(), 157)

    def test_rollout_plays_valid_cards(self):
        for seed in range(20):
            sim = sim_for(random_state(seed, 0), CheckedGameSim)
            random_rollout(sim, self.rule, np.random.default_rng(seed))
            self.assertTrue(sim.is_done())

    def test_rollout_reproducible(self):
        state = random_state(8, 5)
        sims = [sim_for(state) for _ in range(2)]
        for sim in sims:
            random_rollout(sim, self.rule, np.random.default_rng(5))
        np.testing.assert_array_equal(sims[0].state.tricks, sims[1].state.tricks)


if __name__ == '__main__':
    unittest.main()
