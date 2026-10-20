"""
Monte Carlo tree search agent that sees the complete game state (cheating). It is an upper reference point for
agents that only see their observation.
"""
import logging
import math
import time
from typing import Optional

import numpy as np

from jass.agents.agent_cheating import AgentCheating
from jass.agents.agent_rule_based_schieber import AgentRuleBasedSchieber
from jass.agents.mcts.mcts import MCTS, best_card
from jass.game.game_state import GameState
from jass.game.game_state_util import observation_from_state
from jass.game.rule_schieber import RuleSchieber


class AgentCheatingMCTS(AgentCheating):
    """
    Select trump like the rule based agent and play the card found by MCTS on the full game state.
    """
    def __init__(self, time_budget: float = 10.0, iterations: Optional[int] = None, c: float = math.sqrt(2),
                 seed: Optional[int] = None):
        """
        Args:
            time_budget: seconds of search for each card
            iterations: number of iterations for each card, replaces the time budget if given
            c: exploration parameter of UCB1
            seed: seed of the random number generator
        """
        self._logger = logging.getLogger(__name__)
        self._rule = RuleSchieber()
        self._time_budget = time_budget
        self._iterations = iterations
        self._mcts = MCTS(self._rule, np.random.default_rng(seed), c=c)
        self._rule_based = AgentRuleBasedSchieber()

    def action_trump(self, state: GameState) -> int:
        return self._rule_based.action_trump(observation_from_state(state, state.player))

    def action_play_card(self, state: GameState) -> int:
        valid = np.flatnonzero(self._rule.get_valid_cards_from_state(state))
        if len(valid) == 1:
            return int(valid[0])
        if self._iterations is not None:
            visits = self._mcts.search(state, iterations=self._iterations)
        else:
            visits = self._mcts.search(state, deadline=time.perf_counter() + self._time_budget)
        return best_card(visits)
