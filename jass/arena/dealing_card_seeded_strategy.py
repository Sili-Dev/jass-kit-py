import numpy as np

from jass.arena.dealing_card_strategy import DealingCardStrategy


class DealingCardSeededStrategy(DealingCardStrategy):
    """
    Deal cards reproducibly: the deal depends only on the seed and the number of the game, so that different
    arenas with the same seed play the same deals, independent of any other random numbers.
    """
    def __init__(self, seed: int):
        self._seed = seed

    def deal_cards(self, game_nr: int = 0, total_nr_games: int = 0) -> np.ndarray:
        rng = np.random.default_rng([self._seed, game_nr])
        hands = np.zeros((4, 36), dtype=np.int32)
        for player, cards in enumerate(rng.permutation(36).reshape(4, 9)):
            hands[player, cards] = 1
        return hands
