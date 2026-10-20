import unittest

import numpy as np

from jass.arena.dealing_card_seeded_strategy import DealingCardSeededStrategy


class DealingCardSeededStrategyTestCase(unittest.TestCase):

    def test_same_seed_same_deal(self):
        first = DealingCardSeededStrategy(42)
        second = DealingCardSeededStrategy(42)
        for game_nr in range(10):
            np.testing.assert_array_equal(first.deal_cards(game_nr), second.deal_cards(game_nr))

    def test_deal_is_valid(self):
        strategy = DealingCardSeededStrategy(42)
        for game_nr in range(10):
            hands = strategy.deal_cards(game_nr)
            np.testing.assert_array_equal(hands.sum(axis=1), np.full(4, 9))
            np.testing.assert_array_equal(hands.sum(axis=0), np.ones(36))

    def test_games_differ(self):
        strategy = DealingCardSeededStrategy(42)
        self.assertFalse(np.array_equal(strategy.deal_cards(0), strategy.deal_cards(1)))

    def test_independent_of_order(self):
        strategy = DealingCardSeededStrategy(42)
        later = strategy.deal_cards(5)
        np.testing.assert_array_equal(DealingCardSeededStrategy(42).deal_cards(5), later)


if __name__ == '__main__':
    unittest.main()
