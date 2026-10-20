"""
Playing out a round in the simulation, used by the tree search after it leaves the tree.
"""
import copy

import numpy as np

from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber


def copy_state(state: GameState) -> GameState:
    """
    Deep copy of a state. The current trick of a state is a view into its tricks, which a deep copy does not
    preserve, so it is set again on the copy.
    """
    result = copy.deepcopy(state)
    if result.nr_played_cards < 36:
        result.current_trick = result.tricks[result.nr_tricks]
    else:
        result.current_trick = None
    return result


def random_rollout(sim: GameSim, rule: RuleSchieber, rng: np.random.Generator) -> None:
    """
    Play random valid cards until the round is over. Valid cards are calculated directly from the state, without
    creating observations.
    """
    state = sim.state
    while not sim.is_done():
        valid = rule.get_valid_cards(state.hands[state.player], state.current_trick, state.nr_cards_in_trick,
                                     state.trump)
        cards = np.flatnonzero(valid)
        sim.action_play_card(int(cards[rng.integers(len(cards))]))
