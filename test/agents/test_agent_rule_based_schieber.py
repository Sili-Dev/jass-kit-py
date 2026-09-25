import unittest

from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.agent_rule_based_schieber import AgentRuleBasedSchieber
from jass.arena.arena import Arena
from jass.game.const import PUSH, HEARTS, DIAMONDS, NORTH, EAST, WEST, card_ids
from jass.game.game_observation import GameObservation
from jass.game.game_util import get_cards_encoded_from_str


def observation(hand, trump=-1, forehand=-1, trick=(), first_player=NORTH, player=None):
    obs = GameObservation()
    obs.dealer = EAST
    obs.trump = trump
    obs.forehand = forehand
    obs.declared_trump = NORTH if trump != -1 else -1
    obs.hand = get_cards_encoded_from_str(hand)
    obs.trick_first_player[0] = first_player
    for i, card in enumerate(trick):
        obs.tricks[0, i] = card_ids[card]
    obs.nr_cards_in_trick = len(trick)
    obs.nr_played_cards = len(trick)
    obs.current_trick = obs.tricks[0]
    # play goes counter-clockwise, so the player after first_player has the next lower number
    obs.player = (first_player - len(trick)) % 4 if player is None else player
    obs.player_view = obs.player
    return obs


class AgentRuleBasedSchieberTestCase(unittest.TestCase):

    def setUp(self):
        self.agent = AgentRuleBasedSchieber()

    def test_select_trump_with_jack_and_nine(self):
        obs = observation(['HJ', 'H9', 'HA', 'H10', 'H6', 'DA', 'S7', 'C8', 'C6'])
        self.assertEqual(self.agent.action_trump(obs), HEARTS)

    def test_push_weak_hand(self):
        obs = observation(['D7', 'D8', 'H6', 'H8', 'S7', 'SQ', 'C9', 'C7', 'C10'])
        self.assertEqual(self.agent.action_trump(obs), PUSH)

    def test_no_push_when_pushed_to(self):
        obs = observation(['D7', 'D8', 'H6', 'H8', 'S7', 'SQ', 'C9', 'C7', 'C10'], forehand=0)
        self.assertNotEqual(self.agent.action_trump(obs), PUSH)

    def test_pull_trumps_with_jack(self):
        obs = observation(['DJ', 'D6', 'HA', 'H10', 'S7', 'SQ', 'C9', 'C7', 'C10'], trump=DIAMONDS, forehand=1)
        self.assertEqual(self.agent.action_play_card(obs), card_ids['DJ'])

    def test_give_points_to_partner_when_last(self):
        # north leads, west and south follow, east is last: partner (west) wins with the ace
        obs = observation(['H10', 'H6', 'DJ', 'SA', 'S7', 'C6', 'C7', 'C8', 'C9'], trump=DIAMONDS, forehand=1,
                          trick=['H7', 'HA', 'H8'], first_player=NORTH)
        self.assertEqual(obs.player, EAST)
        self.assertEqual(self.agent.action_play_card(obs), card_ids['H10'])

    def test_win_cheaply_when_last(self):
        obs = observation(['HA', 'HK', 'H6', 'DJ', 'S7', 'C6', 'C7', 'C8', 'C9'], trump=DIAMONDS, forehand=1,
                          trick=['HQ', 'H10', 'H8'], first_player=NORTH)
        self.assertEqual(self.agent.action_play_card(obs), card_ids['HK'])

    def test_discard_low_card_when_losing(self):
        obs = observation(['HQ', 'H6', 'SA', 'S10', 'C6', 'C7', 'C8', 'C9', 'CA'], trump=DIAMONDS, forehand=1,
                          trick=['HA', 'H9', 'H8'], first_player=WEST)
        self.assertEqual(obs.player, NORTH)
        self.assertEqual(self.agent.action_play_card(obs), card_ids['H6'])

    def test_plays_valid_cards_and_beats_random(self):
        arena = Arena(nr_games_to_play=200, check_move_validity=True, print_every_x_games=1000)
        agent = AgentRuleBasedSchieber()
        opponent = AgentRandomSchieber()
        arena.set_players(agent, opponent, agent, opponent)
        arena.play_all_games()
        self.assertEqual(arena.nr_games_played, 200)
        self.assertGreater(arena.points_team_0.sum(), arena.points_team_1.sum())


if __name__ == '__main__':
    unittest.main()
