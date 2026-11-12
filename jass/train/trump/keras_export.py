"""
Keras network for trump selection (exercise 10) and its export to the numpy TrumpModel the bot runs.

Needs tensorflow, which is only a training dependency: pip install -e .[train-nn]
"""
from typing import Sequence

from tensorflow import keras

from jass.agents.trump.features import NR_CLASSES
from jass.agents.trump.trump_model import TrumpModel


def build_network(nr_inputs: int, hidden: Sequence[int] = (128, 64), dropout: float = 0.2) -> keras.Model:
    """Dense ReLU layers with dropout, ending in NR_CLASSES logits (no softmax: train with from_logits=True)."""
    network = keras.Sequential([keras.Input(shape=(nr_inputs,))])
    for units in hidden:
        network.add(keras.layers.Dense(units, activation='relu'))
        if dropout > 0:
            network.add(keras.layers.Dropout(dropout))
    network.add(keras.layers.Dense(NR_CLASSES))
    return network


def to_trump_model(network: keras.Model, feature_set: str) -> TrumpModel:
    """
    Copy the Dense layers in order into a TrumpModel; dropout does nothing at prediction time and is skipped.
    TrumpModel applies ReLU between layers and nothing after the last, so anything else is refused rather than
    exported as a model that computes different logits from the network.
    """
    dense = []
    for layer in network.layers:
        if isinstance(layer, keras.layers.Dense):
            dense.append(layer)
        elif not isinstance(layer, keras.layers.Dropout):
            raise ValueError('Cannot export layer {} of type {}'.format(layer.name, type(layer).__name__))
    layers = []
    for i, layer in enumerate(dense):
        expected = 'linear' if i == len(dense) - 1 else 'relu'
        activation = keras.activations.serialize(layer.activation)
        activation = activation if isinstance(activation, str) else activation.get('config', {}).get('name', '?')
        if activation != expected:
            raise ValueError('Layer {} has activation {}, the numpy model needs {}'.format(
                layer.name, activation, expected))
        if not layer.use_bias:
            raise ValueError('Layer {} has no bias, the numpy model needs one'.format(layer.name))
        kernel, bias = layer.get_weights()
        layers.append((kernel.T, bias))  # keras stores (in, out), TrumpModel (out, in)
    return TrumpModel(layers, feature_set=feature_set)
