import unittest

import numpy as np

from jass.agents.trump.features import (obs_to_trump_features, label_to_class, class_to_action, encode_hands,
                                        trump_features, FEATURE_SETS, NR_FEATURES, FOREHAND_INDEX, PUSH_CLASS)
from jass.game.const import PUSH, HEARTS, UNE_UFE, card_ids
from jass.game.game_observation import GameObservation
from jass.game.game_util import get_cards_encoded_from_str


def obs_with(hand, forehand):
    obs = GameObservation()
    obs.hand = get_cards_encoded_from_str(hand)
    obs.forehand = forehand
    return obs


class TrumpFeaturesTestCase(unittest.TestCase):
    HAND = ['DA', 'DK', 'HJ', 'H9', 'SA', 'S6', 'CQ', 'C10', 'C7']

    def test_cards_are_one_hot_in_const_order(self):
        features = obs_to_trump_features(obs_with(self.HAND, forehand=-1))
        self.assertEqual(NR_FEATURES, len(features))
        self.assertEqual(9, features[:36].sum())
        for card in self.HAND:
            self.assertEqual(1.0, features[card_ids[card]])

    def test_forehand_flag_is_set_only_when_push_is_allowed(self):
        self.assertEqual(1.0, obs_to_trump_features(obs_with(self.HAND, forehand=-1))[FOREHAND_INDEX])
        self.assertEqual(0.0, obs_to_trump_features(obs_with(self.HAND, forehand=0))[FOREHAND_INDEX])

    def test_label_mapping_round_trips(self):
        self.assertEqual(HEARTS, label_to_class(HEARTS))
        self.assertEqual(UNE_UFE, label_to_class(UNE_UFE))
        self.assertEqual(PUSH_CLASS, label_to_class(6))
        self.assertEqual(PUSH_CLASS, label_to_class(10))
        self.assertEqual(PUSH, class_to_action(PUSH_CLASS))
        self.assertEqual([0, 1, 2, 3, 4, 5], [class_to_action(c) for c in range(6)])

    def test_cards_plus_suit_features(self):
        f = obs_to_trump_features(obs_with(self.HAND, forehand=-1), feature_set='cards+suit')
        self.assertEqual(FEATURE_SETS['cards+suit'], len(f))
        self.assertEqual(53, len(f))
        np.testing.assert_array_equal(obs_to_trump_features(obs_with(self.HAND, forehand=-1)), f[:37])
        # per suit D, H, S, C: count/9, jack, nine, ace
        np.testing.assert_allclose([2 / 9, 0, 0, 1, 2 / 9, 1, 1, 0, 2 / 9, 0, 0, 1, 3 / 9, 0, 0, 0], f[37:],
                                   rtol=1e-6)
        self.assertEqual(np.float32, f.dtype)

    def test_encode_hands_matches_single_encoding(self):
        single = obs_to_trump_features(obs_with(self.HAND, forehand=0), feature_set='cards+suit')
        batch = encode_hands(np.stack([single[:37], single[:37]]), 'cards+suit')
        np.testing.assert_array_equal(np.stack([single, single]), batch)
        np.testing.assert_array_equal(single[None, :37], encode_hands(single[None, :37], 'cards'))

    def test_unknown_feature_set_raises(self):
        with self.assertRaises(ValueError):
            obs_to_trump_features(obs_with(self.HAND, forehand=-1), feature_set='cards+suits')
        with self.assertRaises(ValueError):
            encode_hands(np.zeros((1, 37), dtype=np.float32), 'nope')
        with self.assertRaises(ValueError):
            trump_features(np.zeros(36), True, feature_set='nope')


if __name__ == '__main__':
    unittest.main()
