import os
import tempfile
import unittest

import numpy as np

from jass.agents.trump.features import FOREHAND_INDEX, NR_CLASSES, NR_FEATURES, PUSH_CLASS
from jass.agents.trump.trump_model import DEFAULT_MODEL_PATH, TrumpModel
from jass.game.const import HEARTS, PUSH, SPADES
from jass.game.game_observation import GameObservation
from jass.game.game_util import get_cards_encoded_from_str

HAND = ['DA', 'DK', 'HJ', 'H9', 'SA', 'S6', 'CQ', 'C10', 'C7']


def obs_with(forehand):
    obs = GameObservation()
    obs.hand = get_cards_encoded_from_str(HAND)
    obs.forehand = forehand
    return obs


def model_preferring(first, second):
    """A model whose logits rank class `first` highest and `second` next, whatever the hand."""
    bias = np.zeros(NR_CLASSES, dtype=np.float32)
    bias[first], bias[second] = 2.0, 1.0
    return TrumpModel.linear(np.zeros((NR_CLASSES, NR_FEATURES), dtype=np.float32), bias)


def two_layer_model():
    rng = np.random.default_rng(1)
    W0, b0 = rng.normal(size=(4, 53)).astype(np.float32), rng.normal(size=4).astype(np.float32)
    W1, b1 = rng.normal(size=(NR_CLASSES, 4)).astype(np.float32), rng.normal(size=NR_CLASSES).astype(np.float32)
    return TrumpModel([(W0, b0), (W1, b1)], feature_set='cards+suit'), (W0, b0, W1, b1)


class TrumpModelTestCase(unittest.TestCase):

    def test_picks_the_highest_logit(self):
        self.assertEqual(SPADES, model_preferring(SPADES, HEARTS).action_trump(obs_with(forehand=-1)))

    def test_push_is_returned_as_game_action(self):
        self.assertEqual(PUSH, model_preferring(PUSH_CLASS, HEARTS).action_trump(obs_with(forehand=-1)))

    def test_never_pushes_after_partner_pushed(self):
        self.assertEqual(HEARTS, model_preferring(PUSH_CLASS, HEARTS).action_trump(obs_with(forehand=0)))

    def test_probabilities_sum_to_one_and_push_is_zero_when_masked(self):
        model = model_preferring(PUSH_CLASS, HEARTS)
        features = np.zeros((2, NR_FEATURES), dtype=np.float32)
        features[0, -1] = 1.0
        proba = model.probabilities(features)
        np.testing.assert_allclose([1.0, 1.0], proba.sum(axis=1), rtol=1e-6)
        self.assertEqual(0.0, proba[1, PUSH_CLASS])

    def test_load_round_trip_and_rejects_wrong_shape(self):
        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        try:
            weights = np.random.default_rng(0).normal(size=(NR_CLASSES, NR_FEATURES)).astype(np.float32)
            np.savez(path, weights=weights, bias=np.zeros(NR_CLASSES), format_version=np.array(1))
            features = np.ones(NR_FEATURES, dtype=np.float32)
            np.testing.assert_allclose(features @ weights.T, TrumpModel.load(path).logits(features), rtol=1e-5)
        finally:
            os.remove(path)
        with self.assertRaises(ValueError):
            TrumpModel.linear(np.zeros((6, NR_FEATURES)), np.zeros(6))

    @unittest.skipUnless(DEFAULT_MODEL_PATH.exists(), 'trained model not present')
    def test_shipped_model_returns_a_legal_action(self):
        model = TrumpModel.load()
        self.assertIn(model.action_trump(obs_with(forehand=-1)), [0, 1, 2, 3, 4, 5, PUSH])
        self.assertIn(model.action_trump(obs_with(forehand=0)), [0, 1, 2, 3, 4, 5])


class DenseTrumpModelTestCase(unittest.TestCase):

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)

    def tearDown(self):
        os.remove(self.path)

    def test_two_layer_forward_pass(self):
        model, (W0, b0, W1, b1) = two_layer_model()
        x = np.random.default_rng(2).random((3, 53)).astype(np.float32)
        x[:, FOREHAND_INDEX] = 1
        expected = np.maximum(x @ W0.T + b0, 0) @ W1.T + b1
        np.testing.assert_allclose(expected, model.logits(x), rtol=1e-5)
        np.testing.assert_allclose(model.logits(x[1]), model.logits(x)[1], rtol=1e-6)
        self.assertEqual(np.float32, model.logits(x.astype(np.float64)).dtype)

    def test_save_load_round_trip_v2(self):
        model, _ = two_layer_model()
        model.save(self.path)
        loaded = TrumpModel.load(self.path)
        self.assertIs(str, type(loaded.feature_set))
        self.assertEqual('cards+suit', loaded.feature_set)
        x = np.ones(53, dtype=np.float32)
        np.testing.assert_array_equal(model.logits(x), loaded.logits(x))

    def test_action_trump_uses_models_feature_set(self):
        model, _ = two_layer_model()
        self.assertIn(model.action_trump(obs_with(forehand=-1)), [0, 1, 2, 3, 4, 5, PUSH])
        self.assertIn(model.action_trump(obs_with(forehand=0)), [0, 1, 2, 3, 4, 5])

    def test_rejects_layers_that_do_not_chain(self):
        W = np.zeros((NR_CLASSES, 37), dtype=np.float32)
        cases = [([(W, np.zeros(NR_CLASSES))], 'cards+suit'),
                 ([(np.zeros((4, 37)), np.zeros(4)), (np.zeros((NR_CLASSES, 5)), np.zeros(NR_CLASSES))], 'cards'),
                 ([(np.zeros((4, 37)), np.zeros(4))], 'cards'),
                 ([(W, np.zeros(NR_CLASSES + 1))], 'cards'),
                 ([], 'cards'),
                 ([(W, np.zeros(NR_CLASSES))], 'nope')]
        for layers, feature_set in cases:
            with self.assertRaises(ValueError):
                TrumpModel(layers, feature_set=feature_set)

    def test_load_reports_missing_keys_as_value_error(self):
        model, (W0, b0, W1, b1) = two_layer_model()
        np.savez(self.path, format_version=np.array(2), nr_layers=np.array(2), W0=W0, b0=b0, W1=W1, b1=b1)
        with self.assertRaises(ValueError):
            TrumpModel.load(self.path)  # no feature_set
        np.savez(self.path, format_version=np.array(1), weights=W1)
        with self.assertRaises(ValueError):
            TrumpModel.load(self.path)  # v1 without bias

    def test_load_rejects_unknown_version(self):
        np.savez(self.path, format_version=np.array(3))
        with self.assertRaises(ValueError):
            TrumpModel.load(self.path)


if __name__ == '__main__':
    unittest.main()
