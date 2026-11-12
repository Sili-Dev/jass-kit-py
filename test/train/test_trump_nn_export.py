import importlib.util
import json
import os
import tempfile
import unittest

import numpy as np

from jass.agents.trump.features import FOREHAND_INDEX, NR_CLASSES

HAS_TF = importlib.util.find_spec('tensorflow') is not None


@unittest.skipUnless(HAS_TF, 'tensorflow not installed (pip install -e .[train-nn])')
class KerasExportTestCase(unittest.TestCase):

    def test_numpy_model_matches_keras(self):
        import tensorflow as tf
        from jass.train.trump.keras_export import build_network, to_trump_model

        tf.keras.utils.set_random_seed(0)
        net = build_network(53, hidden=(16, 8))
        x = np.random.default_rng(0).random((64, 53)).astype(np.float32)
        x[:, FOREHAND_INDEX] = 1  # push allowed, so the numpy model does not mask anything
        y = np.random.default_rng(1).integers(0, NR_CLASSES, 64)
        net.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        net.fit(x, y, epochs=2, verbose=0)

        model = to_trump_model(net, 'cards+suit')
        self.assertEqual('cards+suit', model.feature_set)
        np.testing.assert_allclose(net.predict(x, verbose=0), model.logits(x), atol=1e-5)

    def test_network_has_logits_for_every_class(self):
        from jass.train.trump.keras_export import build_network
        net = build_network(37, hidden=(4,))
        self.assertEqual((None, NR_CLASSES), net.output_shape)

    def test_export_refuses_layers_it_cannot_copy(self):
        from tensorflow import keras
        from jass.train.trump.keras_export import to_trump_model
        net = keras.Sequential([keras.Input(shape=(37,)), keras.layers.Dense(8), keras.layers.BatchNormalization(),
                                keras.layers.Dense(NR_CLASSES)])
        with self.assertRaises(ValueError):
            to_trump_model(net, 'cards')

    def test_export_refuses_activations_the_numpy_model_does_not_apply(self):
        from tensorflow import keras
        from jass.train.trump.keras_export import to_trump_model
        tanh_hidden = keras.Sequential([keras.Input(shape=(37,)), keras.layers.Dense(8, activation='tanh'),
                                        keras.layers.Dense(NR_CLASSES)])
        softmax_out = keras.Sequential([keras.Input(shape=(37,)), keras.layers.Dense(8, activation='relu'),
                                        keras.layers.Dense(NR_CLASSES, activation='softmax')])
        no_bias = keras.Sequential([keras.Input(shape=(37,)), keras.layers.Dense(NR_CLASSES, use_bias=False)])
        for net in [tanh_hidden, softmax_out, no_bias]:
            with self.assertRaises(ValueError):
                to_trump_model(net, 'cards')


def write_random_csv(path, nr_rows, seed=0):
    """Legal rows in the csv layout: 36 cards (9 set), push flag (0 = may push), user, label."""
    rng = np.random.default_rng(seed)
    lines = []
    for i in range(nr_rows):
        hand = np.zeros(36, dtype=int)
        hand[rng.choice(36, 9, replace=False)] = 1
        may_push = rng.random() < 0.5
        label = int(rng.choice([0, 1, 2, 3, 4, 5, 6])) if may_push else int(rng.integers(0, 6))
        lines.append(','.join(str(v) for v in list(hand) + [0 if may_push else 1, 1 + i % 5, label]))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


@unittest.skipUnless(HAS_TF, 'tensorflow not installed (pip install -e .[train-nn])')
class TrainNnTestCase(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.csv = os.path.join(self.dir.name, 'trump.csv')
        self.out = os.path.join(self.dir.name, 'model.npz')
        self.plot = os.path.join(self.dir.name, 'plots', 'model.png')
        write_random_csv(self.csv, 400)

    def tearDown(self):
        self.dir.cleanup()

    def test_trains_saves_model_and_plot(self):
        from jass.agents.trump.trump_model import TrumpModel
        from jass.train.trump.train_nn import main
        main(['--csv', self.csv, '--out', self.out, '--plot', self.plot, '--hidden', '8', '--epochs', '2'])
        self.assertEqual('cards+suit', TrumpModel.load(self.out).feature_set)
        self.assertTrue(os.path.exists(self.plot))

    def test_filter_that_keeps_no_rows_stops_before_training(self):
        from jass.train.trump.train_nn import main
        stats = os.path.join(self.dir.name, 'stats.json')
        with open(stats, 'w') as f:
            json.dump([dict(id=u, mean=78.0, std=40.0, nr=5000) for u in range(1, 6)], f)
        with self.assertRaises(SystemExit):
            main(['--csv', self.csv, '--out', self.out, '--plot', self.plot, '--epochs', '2',
                  '--stats', stats, '--min-player-mean', '99'])
        self.assertFalse(os.path.exists(self.out))


if __name__ == '__main__':
    unittest.main()
