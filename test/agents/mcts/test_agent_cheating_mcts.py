import unittest
from unittest import mock

import numpy as np

from jass.agents.agent_cheating_random_schieber import AgentCheatingRandomSchieber
from jass.agents.agent_rule_based_schieber import AgentRuleBasedSchieber
from jass.agents.mcts.agent_cheating_mcts import AgentCheatingMCTS
from jass.agents.mcts.mcts import MCTS
from jass.arena.arena import Arena
from jass.game.game_sim import GameSim
from jass.game.game_state_util import observation_from_state
from jass.game.rule_schieber import RuleSchieber
from .positions import random_state


def dealt_state(seed):
    rng = np.random.default_rng(seed)
    hands = np.zeros((4, 36), dtype=np.int32)
    for player, cards in enumerate(rng.permutation(36).reshape(4, 9)):
        hands[player, cards] = 1
    sim = GameSim(rule=RuleSchieber())
    sim.init_from_cards(hands, dealer=seed % 4)
    return sim.state


class AgentCheatingMCTSTestCase(unittest.TestCase):

    def test_trump_like_rule_based(self):
        agent = AgentCheatingMCTS(iterations=10, seed=0)
        rule_based = AgentRuleBasedSchieber()
        for seed in range(20):
            state = dealt_state(seed)
            self.assertEqual(agent.action_trump(state),
                             rule_based.action_trump(observation_from_state(state, state.player)))

    def test_single_valid_card_without_search(self):
        state = random_state(4, 35)
        with mock.patch.object(MCTS, 'search') as search:
            card = AgentCheatingMCTS(iterations=10, seed=0).action_play_card(state)
        search.assert_not_called()
        self.assertEqual(card, int(np.flatnonzero(state.hands[state.player])[0]))

    def test_plays_valid_cards(self):
        agent = AgentCheatingMCTS(iterations=30, seed=0)
        rule = RuleSchieber()
        for seed in range(30):
            state = random_state(seed, seed % 36)
            self.assertIn(agent.action_play_card(state), np.flatnonzero(rule.get_valid_cards_from_state(state)))

    def test_beats_random_in_arena(self):
        np.random.seed(1)
        arena = Arena(nr_games_to_play=20, cheating_mode=True, print_every_x_games=100)
        mcts = AgentCheatingMCTS(iterations=50, seed=1)
        arena.set_players(mcts, AgentCheatingRandomSchieber(), mcts, AgentCheatingRandomSchieber())
        arena.play_all_games()
        self.assertGreater(arena.points_team_0.sum(), arena.points_team_1.sum())


if __name__ == '__main__':
    unittest.main()
