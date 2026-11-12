import json
import os
import tempfile
import unittest

import numpy as np

from jass.agents.trump.features import PUSH_CLASS, FOREHAND_INDEX
from jass.train.trump.data import load_trump_csv, strong_player_mask


def row(cards, can_push, user, label):
    """Build a csv row; the csv stores the push flag inverted (0 = may push)."""
    hand = [0] * 36
    for c in cards:
        hand[c] = 1
    return ','.join(str(v) for v in hand + [0 if can_push else 1, user, label])


class LoadTrumpCsvTestCase(unittest.TestCase):
    NINE = list(range(9))

    def setUp(self):
        rows = [
            row(self.NINE, 1, 11, 1),        # hearts
            row(self.NINE, 1, 12, 10),       # push, old encoding
            row(self.NINE, 0, 13, 6),        # push without forehand -> dropped
            row(list(range(8)), 1, 14, 2),   # only 8 cards -> dropped
            row(self.NINE, 0, 15, 4),        # obe-abe
        ]
        fd, self.path = tempfile.mkstemp(suffix='.csv')
        with os.fdopen(fd, 'w') as f:
            f.write('\n'.join(rows) + '\n')

    def tearDown(self):
        os.remove(self.path)

    def test_loads_and_filters_illegal_rows(self):
        data = load_trump_csv(self.path)
        self.assertEqual((3, 37), data.X.shape)
        np.testing.assert_array_equal([1, PUSH_CLASS, 4], data.y)
        np.testing.assert_array_equal([11, 12, 15], data.users)
        self.assertEqual(np.float32, data.X.dtype)
        np.testing.assert_array_equal([1, 1, 0], data.X[:, FOREHAND_INDEX])

    def test_cards_plus_suit_keeps_forehand_flip(self):
        data = load_trump_csv(self.path, feature_set='cards+suit')
        self.assertEqual((3, 53), data.X.shape)
        self.assertEqual(np.float32, data.X.dtype)
        np.testing.assert_array_equal([1, 1, 0], data.X[:, FOREHAND_INDEX])
        np.testing.assert_array_equal(load_trump_csv(self.path).X, data.X[:, :37])

    def test_unknown_feature_set_raises(self):
        with self.assertRaises(ValueError):
            load_trump_csv(self.path, feature_set='nope')


class StrongPlayerMaskTestCase(unittest.TestCase):
    STATS = [dict(id=0, mean=85.0, std=40.0, nr=5000),
             dict(id=1, mean=85.0, std=40.0, nr=5000),
             dict(id=2, mean=85.0, std=40.0, nr=999),
             dict(id=3, mean=79.4, std=40.0, nr=5000),
             dict(id=4, mean=79.5, std=40.0, nr=1000)]   # id 5 is missing from the stats

    def setUp(self):
        fd, self.stats_path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(self.STATS, f)

    def tearDown(self):
        os.remove(self.stats_path)

    def test_keeps_strong_players_with_enough_games(self):
        users = np.array([0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal([False, True, False, False, True, False],
                                      strong_player_mask(users, self.stats_path))

    def test_keep_user0(self):
        users = np.array([0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal([True, True, False, False, True, False],
                                      strong_player_mask(users, self.stats_path, keep_user0=True))

    def test_thresholds_are_parameters(self):
        users = np.array([1, 2, 3])
        np.testing.assert_array_equal([True, True, True],
                                      strong_player_mask(users, self.stats_path, min_mean=79.0, min_games=900))


if __name__ == '__main__':
    unittest.main()
