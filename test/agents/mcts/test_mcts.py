import time
import unittest

import numpy as np

from jass.agents.mcts.mcts import MCTS, best_card
from jass.agents.mcts.rollout import copy_state
from jass.game.const import team
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber
from .positions import random_state

RULE = RuleSchieber()


def _final_points(state) -> np.ndarray:
    """Final points of both teams when every player maximises the points of their own team."""
    if state.nr_played_cards == 36:
        return state.points.copy()
    best = None
    for card in np.flatnonzero(RULE.get_valid_cards_from_state(state)):
        points = _final_points(_after(state, card))
        if best is None or points[team[state.player]] > best[team[state.player]]:
            best = points
    return best


def _after(state, card):
    sim = GameSim(rule=RULE)
    sim._state = copy_state(state)
    sim.action_play_card(int(card))
    return sim.state


def exact_values(state) -> dict:
    """For each valid card, the final points of the moving player's team under optimal play."""
    return {int(card): int(_final_points(_after(state, card))[team[state.player]])
            for card in np.flatnonzero(RULE.get_valid_cards_from_state(state))}


class MCTSTestCase(unittest.TestCase):

    def search(self, state, seed=0, **budget):
        return MCTS(RULE, np.random.default_rng(seed)).search(state, **budget)

    def test_requires_a_budget(self):
        with self.assertRaises(ValueError):
            self.search(random_state(1, 0))

    def test_visits_add_up(self):
        state = random_state(1, 0)
        visits = self.search(state, iterations=300)
        self.assertEqual(sum(visits.values()), 300)
        self.assertEqual(sorted(visits), sorted(np.flatnonzero(RULE.get_valid_cards_from_state(state)).tolist()))

    def test_finds_optimal_card_in_endgame(self):
        for seed in range(20):
            state = random_state(seed, 28 + seed % 4)
            values = exact_values(state)
            card = best_card(self.search(state, seed=seed, iterations=3000))
            self.assertEqual(values[card], max(values.values()), 'seed {}: {} chose {}'.format(seed, values, card))

    def test_reproducible(self):
        state = random_state(4, 6)
        self.assertEqual(self.search(state, seed=11, iterations=500), self.search(state, seed=11, iterations=500))

    def test_does_not_change_state(self):
        state = random_state(5, 9)
        tricks = state.tricks.copy()
        hands = state.hands.copy()
        self.search(state, iterations=200)
        np.testing.assert_array_equal(state.tricks, tricks)
        np.testing.assert_array_equal(state.hands, hands)

    def test_deadline(self):
        start = time.perf_counter()
        visits = self.search(random_state(2, 0), deadline=start + 0.2)
        self.assertLess(time.perf_counter() - start, 0.4)
        self.assertGreaterEqual(sum(visits.values()), 1)

    def test_at_least_one_iteration_after_deadline(self):
        visits = self.search(random_state(2, 0), deadline=time.perf_counter() - 1.0)
        self.assertEqual(sum(visits.values()), 1)

    def test_deadline_stops_before_iterations(self):
        start = time.perf_counter()
        visits = self.search(random_state(2, 0), iterations=10 ** 6, deadline=start + 0.2)
        self.assertLess(time.perf_counter() - start, 0.4)
        self.assertLess(sum(visits.values()), 10 ** 6)

    def test_best_card_tie(self):
        self.assertEqual(best_card({5: 10, 3: 10, 7: 2}), 3)


if __name__ == '__main__':
    unittest.main()
