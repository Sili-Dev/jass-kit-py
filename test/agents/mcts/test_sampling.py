import unittest

import numpy as np

from jass.agents.mcts.sampling import known_voids, sample_hands
from jass.game.const import NORTH, WEST, HEARTS, DIAMONDS, SPADES, OBE_ABE, DJ, HJ, card_ids, color_of_card, \
    color_masks, color_offset
from jass.game.game_observation import GameObservation
from jass.game.game_util import get_cards_encoded_from_str
from .positions import random_observation


def reference_voids(obs: GameObservation) -> np.ndarray:
    """Copy of the former AgentRuleBasedSchieber._voids, to check that the shared helper behaves the same."""
    voids = np.zeros((4, 4), dtype=bool)
    for i in range(obs.nr_tricks + 1):
        first_player = obs.trick_first_player[i]
        trick = obs.tricks[i]
        if first_player < 0 or trick[0] < 0:
            continue
        lead_color = color_of_card[trick[0]]
        for j in range(1, 4):
            if trick[j] < 0:
                break
            color = color_of_card[trick[j]]
            if color != lead_color and color != obs.trump:
                voids[(first_player - j) % 4, lead_color] = True
    return voids


def trick_observation(trump, trick, hand=('CA', 'CK', 'CQ', 'CJ', 'C10', 'C9', 'C8', 'C7', 'C6')):
    """Observation for south, in the first trick led by north (north, west, south, ...)."""
    obs = GameObservation()
    obs.dealer = WEST
    obs.trump = trump
    obs.forehand = 1
    obs.declared_trump = NORTH
    obs.hand = get_cards_encoded_from_str(list(hand))
    obs.trick_first_player[0] = NORTH
    for i, card in enumerate(trick):
        obs.tricks[0, i] = card_ids[card]
    obs.nr_cards_in_trick = len(trick)
    obs.nr_played_cards = len(trick)
    obs.current_trick = obs.tricks[0]
    obs.player = (NORTH - len(trick)) % 4
    obs.player_view = obs.player
    return obs


class KnownVoidsTestCase(unittest.TestCase):

    def test_same_as_rule_based_voids(self):
        for seed in range(100):
            obs = random_observation(seed, seed % 36)
            np.testing.assert_array_equal(known_voids(obs), reference_voids(obs))

    def test_not_following_marks_void(self):
        obs = trick_observation(HEARTS, ['DA', 'SA'])
        self.assertTrue(known_voids(obs)[WEST, DIAMONDS])
        self.assertFalse(known_voids(obs)[WEST, SPADES])

    def test_trumping_does_not_mark_void(self):
        obs = trick_observation(HEARTS, ['DA', 'H6'])
        self.assertFalse(known_voids(obs)[WEST, DIAMONDS])


def cards_left(obs: GameObservation) -> np.ndarray:
    """Number of cards each player still holds."""
    left = np.full(4, 9)
    for i in range(obs.nr_tricks + 1):
        for j in range(4):
            if obs.tricks[i, j] >= 0:
                left[(obs.trick_first_player[i] - j) % 4] -= 1
    return left


def played_cards(obs: GameObservation) -> np.ndarray:
    played = np.zeros(36, dtype=np.int32)
    played[obs.tricks[obs.tricks >= 0]] = 1
    return played


class SampleHandsTestCase(unittest.TestCase):

    def samples(self, nr=200, max_attempts=50):
        for seed in range(nr):
            obs = random_observation(seed, seed % 36)
            yield obs, sample_hands(obs, np.random.default_rng(seed), max_attempts=max_attempts)

    def assert_valid_deal(self, obs, hands):
        np.testing.assert_array_equal(hands.sum(axis=1), cards_left(obs))
        np.testing.assert_array_equal(hands.sum(axis=0), 1 - played_cards(obs))
        np.testing.assert_array_equal(hands[obs.player], obs.hand)

    def test_hand_sizes(self):
        for obs, hands in self.samples():
            np.testing.assert_array_equal(hands.sum(axis=1), cards_left(obs))

    def test_partition(self):
        for obs, hands in self.samples():
            np.testing.assert_array_equal(hands.sum(axis=0), 1 - played_cards(obs))

    def test_own_hand_unchanged(self):
        for obs, hands in self.samples():
            np.testing.assert_array_equal(hands[obs.player], obs.hand)

    def test_voids_respected(self):
        for obs, hands in self.samples():
            voids = known_voids(obs)
            for p in range(4):
                if p == obs.player:
                    continue
                for c in range(4):
                    if not voids[p, c]:
                        continue
                    forbidden = color_masks[c].copy()
                    if c == obs.trump:
                        forbidden[color_offset[c] + DJ] = 0
                    self.assertEqual((hands[p] * forbidden).sum(), 0)

    def test_jack_exception(self):
        obs = trick_observation(HEARTS, ['H6', 'DA'])
        self.assertTrue(known_voids(obs)[WEST, HEARTS])
        received = [sample_hands(obs, np.random.default_rng(seed))[WEST, HJ] for seed in range(200)]
        self.assertIn(1, received)

    def test_no_jack_exception_without_trump_colour(self):
        obs = trick_observation(OBE_ABE, ['H6', 'DA'])
        self.assertTrue(known_voids(obs)[WEST, HEARTS])
        for seed in range(200):
            self.assertEqual(sample_hands(obs, np.random.default_rng(seed))[WEST, HJ], 0)

    def test_reproducible(self):
        obs = random_observation(3, 10)
        np.testing.assert_array_equal(sample_hands(obs, np.random.default_rng(7)),
                                      sample_hands(obs, np.random.default_rng(7)))

    def test_fallback_ignores_voids(self):
        for obs, hands in self.samples(nr=50, max_attempts=0):
            self.assert_valid_deal(obs, hands)


if __name__ == '__main__':
    unittest.main()
