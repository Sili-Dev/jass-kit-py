"""
Numpy-only trump model for the bot, reading the weights exported by jass.train.trump.train (linear, format 1)
or jass.train.trump.train_nn (dense network, format 2).
"""
import os
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from jass.agents.trump.features import FEATURE_SETS, FOREHAND_INDEX, NR_CLASSES, PUSH_CLASS, class_to_action, \
    obs_to_trump_features
from jass.game.game_observation import GameObservation

FORMAT_LINEAR = 1
FORMAT_DENSE = 2

# the model in the source tree; set JASS_TRUMP_MODEL when the package is installed elsewhere (e.g. in docker)
MODEL_PATH_ENV = 'JASS_TRUMP_MODEL'
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / 'models' / 'trump_linear.npz'

Layer = Tuple[np.ndarray, np.ndarray]


def default_model_path() -> Path:
    return Path(os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH))


class TrumpModel:
    """
    Stack of dense layers: x @ W.T + b per layer, ReLU between layers, one logit per class 0..6 at the end.
    A linear model is the case with a single layer.
    """

    def __init__(self, layers: Sequence[Layer], feature_set: str = 'cards'):
        if feature_set not in FEATURE_SETS:
            raise ValueError('Unknown feature set {!r}, expected one of {}'.format(feature_set, sorted(FEATURE_SETS)))
        if not layers:
            raise ValueError('A trump model needs at least one layer')
        expected_inputs = FEATURE_SETS[feature_set]
        self._layers = []
        for i, (weights, bias) in enumerate(layers):
            weights = np.asarray(weights, dtype=np.float32)
            bias = np.asarray(bias, dtype=np.float32)
            if weights.ndim != 2 or weights.shape[1] != expected_inputs or bias.shape != (weights.shape[0],):
                raise ValueError('Layer {}: expected weights (out, {}) and bias (out,), got {} and {}'.format(
                    i, expected_inputs, weights.shape, bias.shape))
            self._layers.append((weights, bias))
            expected_inputs = weights.shape[0]
        if expected_inputs != NR_CLASSES:
            raise ValueError('The last layer must have {} outputs, got {}'.format(NR_CLASSES, expected_inputs))
        self.feature_set = feature_set

    @classmethod
    def linear(cls, weights: np.ndarray, bias: np.ndarray) -> 'TrumpModel':
        """Linear softmax model on the 'cards' features: logits = features @ weights.T + bias."""
        return cls([(weights, bias)], feature_set='cards')

    @classmethod
    def load(cls, path=None) -> 'TrumpModel':
        """Load exported weights from path, by default from default_model_path()."""
        path = default_model_path() if path is None else path
        with np.load(path) as data:
            version = int(data['format_version']) if 'format_version' in data else None
            try:
                if version == FORMAT_LINEAR:
                    return cls.linear(data['weights'], data['bias'])
                if version == FORMAT_DENSE:
                    layers = [(data['W{}'.format(i)], data['b{}'.format(i)]) for i in range(int(data['nr_layers']))]
                    return cls(layers, feature_set=str(data['feature_set']))
            except KeyError as e:
                raise ValueError('Incomplete trump model file {}: {}'.format(path, e)) from e
            raise ValueError('Unsupported trump model format {} in {}'.format(version, path))

    def save(self, path):
        """Write the model in format 2, readable by load()."""
        arrays = {}
        for i, (weights, bias) in enumerate(self._layers):
            arrays['W{}'.format(i)] = weights
            arrays['b{}'.format(i)] = bias
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, format_version=np.array(FORMAT_DENSE), feature_set=np.array(self.feature_set),
                 nr_layers=np.array(len(self._layers)), **arrays)

    def logits(self, features: np.ndarray) -> np.ndarray:
        """Logits for one feature vector or a batch (n, inputs); push is -inf where it is not allowed."""
        features = np.asarray(features, dtype=np.float32)
        x = features
        for i, (weights, bias) in enumerate(self._layers):
            x = x @ weights.T + bias
            if i < len(self._layers) - 1:
                x = np.maximum(x, 0.0)
        not_allowed = features[..., FOREHAND_INDEX] == 0
        x[..., PUSH_CLASS] = np.where(not_allowed, -np.inf, x[..., PUSH_CLASS])
        return x

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = self.logits(features)
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def action_trump(self, obs: GameObservation) -> int:
        """The trump action for the game: 0..5, or PUSH."""
        features = obs_to_trump_features(obs, feature_set=self.feature_set)
        return class_to_action(int(np.argmax(self.logits(features))))
