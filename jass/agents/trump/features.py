"""
Feature encoding for trump selection, shared by training and by the agent at runtime.

Feature set 'cards' matches the Swisslos trump data set: 36 one-hot cards in the order of jass.game.const
(DA..D6, HA..H6, SA..S6, CA..C6), followed by one forehand flag that is 1 when the player may still push.
Feature set 'cards+suit' appends, for each suit D, H, S, C: card count / 9, has jack, has nine, has ace.
"""
import numpy as np

from jass.game.const import PUSH, PUSH_ALT
from jass.game.game_observation import GameObservation

NR_FEATURES = 37
FOREHAND_INDEX = 36

# number of inputs of each feature set; the first NR_FEATURES are always the 'cards' features
FEATURE_SETS = {'cards': NR_FEATURES, 'cards+suit': NR_FEATURES + 16}

# offsets of a card within its suit, in the order of jass.game.const
_ACE, _JACK, _NINE = 0, 3, 5

# model classes: 0..5 are the trump modes of jass.game.const, 6 is push
NR_CLASSES = 7
PUSH_CLASS = 6


def _check_feature_set(feature_set: str):
    if feature_set not in FEATURE_SETS:
        raise ValueError('Unknown feature set {!r}, expected one of {}'.format(feature_set, sorted(FEATURE_SETS)))


def encode_hands(cards: np.ndarray, feature_set: str) -> np.ndarray:
    """Encode a batch of 'cards' features (n, 37) in the given feature set (n, FEATURE_SETS[feature_set])."""
    _check_feature_set(feature_set)
    cards = np.asarray(cards, dtype=np.float32)
    if feature_set == 'cards':
        return cards
    by_suit = cards[:, :36].reshape(-1, 4, 9)
    suit = np.stack([by_suit.sum(axis=2) / 9.0, by_suit[:, :, _JACK], by_suit[:, :, _NINE], by_suit[:, :, _ACE]],
                    axis=2)
    return np.concatenate([cards, suit.reshape(-1, 16)], axis=1).astype(np.float32)


def trump_features(hand: np.ndarray, can_push: bool, feature_set: str = 'cards') -> np.ndarray:
    """Encode a one-hot hand of 36 cards and the push flag as a float vector of length FEATURE_SETS[feature_set]."""
    _check_feature_set(feature_set)
    features = np.zeros(NR_FEATURES, dtype=np.float32)
    features[:36] = hand
    features[FOREHAND_INDEX] = 1.0 if can_push else 0.0
    return encode_hands(features[None], feature_set)[0]


def obs_to_trump_features(obs: GameObservation, feature_set: str = 'cards') -> np.ndarray:
    """Encode the observation of the player who has to choose trump (forehand == -1 means push is allowed)."""
    return trump_features(obs.hand, can_push=obs.forehand == -1, feature_set=feature_set)


def label_to_class(label: int) -> int:
    """Map a trump label from the data (0..5, 6 or 10 for push) to a model class 0..6."""
    return PUSH_CLASS if label in (PUSH, PUSH_ALT) else int(label)


def class_to_action(cls: int) -> int:
    """Map a model class 0..6 back to the trump action the game expects (push is PUSH = 10)."""
    return PUSH if cls == PUSH_CLASS else int(cls)
