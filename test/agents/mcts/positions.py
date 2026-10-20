"""
Reproducible random game positions for the tests of the MCTS agents.
"""
import numpy as np

from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_observation import GameObservation
from jass.game.game_state_util import observation_from_state
from jass.game.rule_schieber import RuleSchieber


def random_state(seed: int, nr_cards_played: int) -> GameState:
    """
    Deal with the given seed, let the forehand declare a random trump and play nr_cards_played random valid cards.
    """
    rng = np.random.default_rng(seed)
    rule = RuleSchieber()
    hands = np.zeros((4, 36), dtype=np.int32)
    for player, cards in enumerate(rng.permutation(36).reshape(4, 9)):
        hands[player, cards] = 1
    sim = GameSim(rule=rule)
    sim.init_from_cards(hands, dealer=seed % 4)
    sim.action_trump(int(rng.integers(0, 6)))
    for _ in range(nr_cards_played):
        valid = np.flatnonzero(rule.get_valid_cards_from_state(sim.state))
        sim.action_play_card(int(rng.choice(valid)))
    return sim.state


def random_observation(seed: int, nr_cards_played: int) -> GameObservation:
    state = random_state(seed, nr_cards_played)
    return observation_from_state(state, state.player)
