"""
Rule based agent for the game of jass (Schieber).

The agent selects trump by scoring its hand for each trump and plays cards following the rules of thumb of
a reasonable human player: pull trumps when the own team declared, cash sure winners, give points to the
partner when the partner wins the trick ('schmieren'), win tricks as cheaply as possible and otherwise
discard the least valuable card.

It only uses what the observation contains: its own hand and the cards played so far.
"""
import logging

import numpy as np

from jass.agents.agent import Agent
from jass.game.const import PUSH, OBE_ABE, UNE_UFE, MAX_TRUMP, card_values, color_of_card, offset_of_card, \
    color_masks, partner_player, card_strings
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber

# score of a card for the evaluation of a trump selection, indexed by the offset of the card in its color
# (A, K, Q, J, 10, 9, 8, 7, 6)
TRUMP_SCORE = np.array([15, 10, 7, 25, 6, 19, 5, 5, 5])
NO_TRUMP_SCORE = np.array([9, 7, 5, 2, 1, 0, 0, 0, 0])
OBE_ABE_SCORE = np.array([14, 10, 8, 7, 5, 0, 5, 0, 0])
UNE_UFE_SCORE = np.array([0, 2, 1, 1, 5, 5, 7, 9, 11])

# below this score, the forehand player pushes
PUSH_THRESHOLD = 68

# rank of the cards in the trump color (higher is stronger), indexed by offset: J > 9 > A > K > Q > 10 > 8 > 7 > 6
TRUMP_RANK = np.array([6, 5, 4, 8, 3, 7, 2, 1, 0])

# the strength of trumps is above the strength of the lead color, which is above all other cards
TRUMP_STRENGTH_OFFSET = 20
COLOR_STRENGTH_OFFSET = 10


def evaluate_trumps(hand: np.ndarray) -> np.ndarray:
    """
    Score a hand for each of the six trumps, higher is better.

    Args:
        hand: one-hot encoded hand
    Returns:
        array of 6 scores, indexed by trump
    """
    cards = np.flatnonzero(hand)
    scores = np.zeros(MAX_TRUMP + 1, dtype=np.int32)
    for trump in range(4):
        for card in cards:
            if color_of_card[card] == trump:
                scores[trump] += TRUMP_SCORE[offset_of_card[card]]
            else:
                scores[trump] += NO_TRUMP_SCORE[offset_of_card[card]]
    scores[OBE_ABE] = OBE_ABE_SCORE[offset_of_card[cards]].sum()
    scores[UNE_UFE] = UNE_UFE_SCORE[offset_of_card[cards]].sum()
    return scores


def card_strength(card: int, lead_color: int, trump: int) -> int:
    """
    Strength of a card in a trick, the card with the highest strength wins the trick. Cards that neither follow
    the lead color nor are trump have strength 0.
    """
    color = color_of_card[card]
    offset = offset_of_card[card]
    if trump < 4 and color == trump:
        return TRUMP_STRENGTH_OFFSET + TRUMP_RANK[offset]
    if color == lead_color:
        # offset 0 is the ace, which is the lowest card in une-ufe
        return COLOR_STRENGTH_OFFSET + (offset if trump == UNE_UFE else 8 - offset)
    return 0


class AgentRuleBasedSchieber(Agent):
    """
    Play jass (Schieber) following a fixed set of rules.
    """
    def __init__(self, push_threshold: int = PUSH_THRESHOLD):
        self._logger = logging.getLogger(__name__)
        self._rule = RuleSchieber()
        self._push_threshold = push_threshold

    def action_trump(self, obs: GameObservation) -> int:
        """
        Select the trump with the best score for the hand, or push if the hand is weak and pushing is allowed.
        """
        scores = evaluate_trumps(obs.hand)
        trump = int(np.argmax(scores))
        if obs.forehand == -1 and scores[trump] < self._push_threshold:
            self._logger.debug('Push with scores {}'.format(scores))
            return PUSH
        self._logger.debug('Trump {} with scores {}'.format(trump, scores))
        return trump

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Select the card to play according to the rules described in the module.
        """
        valid = np.flatnonzero(self._rule.get_valid_cards_from_obs(obs))
        if len(valid) == 1:
            return int(valid[0])

        if obs.nr_cards_in_trick == 0:
            card = self._lead(obs, valid)
        else:
            card = self._follow(obs, valid)
        self._logger.debug('Played card: {}'.format(card_strings[card]))
        return int(card)

    #
    # leading a trick
    #

    def _lead(self, obs: GameObservation, valid: np.ndarray) -> int:
        trump = obs.trump
        unseen = self._unseen_cards(obs)
        voids = self._voids(obs)
        opponents = [p for p in range(4) if p != obs.player and p != partner_player[obs.player]]

        if trump < 4:
            own_trumps = [c for c in valid if color_of_card[c] == trump]
            unseen_trumps = unseen * color_masks[trump]
            our_team_declared = obs.declared_trump in (obs.player, partner_player[obs.player])
            opponents_may_have_trump = unseen_trumps.sum() > 0 and not all(voids[p, trump] for p in opponents)

            # pull the trumps of the opponents with our highest trump, as long as it can not be beaten
            if our_team_declared and opponents_may_have_trump and own_trumps:
                best = max(own_trumps, key=lambda c: TRUMP_RANK[offset_of_card[c]])
                if self._is_highest(best, trump, trump, unseen):
                    return best

        # cash a card that wins the trick for sure
        winners = [c for c in valid
                   if color_of_card[c] != trump and self._is_highest(c, color_of_card[c], trump, unseen)
                   and not self._may_be_trumped(color_of_card[c], trump, unseen, voids, opponents)]
        if winners:
            return max(winners, key=lambda c: card_values[trump, c])

        # otherwise give away as little as possible, keeping the trumps
        return min(valid, key=lambda c: self._discard_key(c, trump, color_of_card[c], obs.hand))

    #
    # following in a trick
    #

    def _follow(self, obs: GameObservation, valid: np.ndarray) -> int:
        trump = obs.trump
        trick = obs.current_trick[:obs.nr_cards_in_trick]
        first_player = obs.trick_first_player[obs.nr_tricks]
        lead_color = color_of_card[trick[0]]
        is_last = obs.nr_cards_in_trick == 3
        unseen = self._unseen_cards(obs)
        voids = self._voids(obs)

        strengths = [card_strength(c, lead_color, trump) for c in trick]
        winning_index = int(np.argmax(strengths))
        winning_card = trick[winning_index]
        winning_player = (first_player - winning_index) % 4
        partner_wins = winning_player == partner_player[obs.player]
        trick_points = int(card_values[trump, trick].sum())

        # players still to play after us
        still_to_play = [(first_player - i) % 4 for i in range(obs.nr_cards_in_trick + 1, 4)]
        opponents_to_play = [p for p in still_to_play if p != partner_player[obs.player]]

        if partner_wins and (is_last or self._is_safe(winning_card, lead_color, trump, unseen, voids,
                                                      opponents_to_play)):
            # the trick is ours: give points to the partner, but do not waste high trumps
            return max(valid, key=lambda c: self._schmier_key(c, trump))

        beating = [c for c in valid if card_strength(c, lead_color, trump) > strengths[winning_index]]
        if partner_wins:
            # partner might lose the trick, only take over with a card that wins for sure
            beating = [c for c in beating
                       if self._is_safe(c, lead_color, trump, unseen, voids, opponents_to_play)]
            if beating:
                return max(beating, key=lambda c: card_values[trump, c])
            return min(valid, key=lambda c: self._discard_key(c, trump, lead_color, obs.hand))

        if beating:
            if is_last:
                cheapest = min(beating, key=lambda c: (self._trump_cost(c, trump), card_values[trump, c],
                                                       card_strength(c, lead_color, trump)))
                # do not spend a valuable trump on a trick with few points
                if self._trump_cost(cheapest, trump) < 2 or trick_points + card_values[trump, cheapest] >= 10:
                    return cheapest
            else:
                safe = [c for c in beating if self._is_safe(c, lead_color, trump, unseen, voids, opponents_to_play)]
                if safe:
                    return min(safe, key=lambda c: (self._trump_cost(c, trump), card_values[trump, c]))

        return min(valid, key=lambda c: self._discard_key(c, trump, lead_color, obs.hand))

    #
    # helpers
    #

    @staticmethod
    def _unseen_cards(obs: GameObservation) -> np.ndarray:
        """One-hot encoded cards that are neither in the hand nor have been played."""
        unseen = np.ones(36, dtype=np.int32) - obs.hand
        played = obs.tricks[obs.tricks >= 0]
        unseen[played] = 0
        return unseen

    @staticmethod
    def _voids(obs: GameObservation) -> np.ndarray:
        """
        Colors that players are known not to have anymore, as boolean array [player, color]. A player that does
        not follow the lead color has none left, except that trump may always be played (and in the case of a
        trump lead, the jack of trump may be kept).
        """
        voids = np.zeros((4, 4), dtype=bool)
        for i in range(obs.nr_tricks + 1):
            first_player = obs.trick_first_player[i]
            trick = obs.tricks[i]
            if first_player < 0 or trick[0] < 0:
                continue
            lead_color = color_of_card[trick[0]]
            for j in range(1, 4):
                if trick[j] < 0:
                    break
                color = color_of_card[trick[j]]
                if color != lead_color and color != obs.trump:
                    voids[(first_player - j) % 4, lead_color] = True
        return voids

    @staticmethod
    def _is_highest(card: int, lead_color: int, trump: int, unseen: np.ndarray) -> bool:
        """True if no unseen card of the same color is stronger."""
        strength = card_strength(card, lead_color, trump)
        color = color_of_card[card]
        for other in np.flatnonzero(unseen * color_masks[color]):
            if card_strength(other, lead_color, trump) > strength:
                return False
        return True

    @staticmethod
    def _may_be_trumped(color: int, trump: int, unseen: np.ndarray, voids: np.ndarray, players) -> bool:
        """True if one of the players is known to be out of the color and might still hold a trump."""
        if trump >= 4 or color == trump:
            return False
        if (unseen * color_masks[trump]).sum() == 0:
            return False
        return any(voids[p, color] and not voids[p, trump] for p in players)

    def _is_safe(self, card: int, lead_color: int, trump: int, unseen: np.ndarray, voids: np.ndarray,
                 opponents_to_play) -> bool:
        """True if the card can not be beaten by the opponents that still play in this trick, as far as known."""
        if not opponents_to_play:
            return True
        if trump < 4 and color_of_card[card] == trump:
            return self._is_highest(card, trump, trump, unseen) or \
                all(voids[p, trump] for p in opponents_to_play)
        if color_of_card[card] != lead_color:
            return False
        return self._is_highest(card, lead_color, trump, unseen) and \
            not self._may_be_trumped(lead_color, trump, unseen, voids, opponents_to_play)

    @staticmethod
    def _trump_cost(card: int, trump: int) -> int:
        """How much it hurts to spend this card: 0 for non trumps, 1 for small trumps, 2 for jack and nine."""
        if trump >= 4 or color_of_card[card] != trump:
            return 0
        return 2 if TRUMP_RANK[offset_of_card[card]] >= 7 else 1

    @staticmethod
    def _schmier_key(card: int, trump: int):
        """Order for giving points to the partner: many points, but keep the valuable trumps."""
        return -AgentRuleBasedSchieber._trump_cost(card, trump), card_values[trump, card], \
            -card_strength(card, color_of_card[card], trump)

    @staticmethod
    def _discard_key(card: int, trump: int, lead_color: int, hand: np.ndarray):
        """Order for throwing away a card: keep trumps, few points, weak cards, from long colors first."""
        color = color_of_card[card]
        is_trump = trump < 4 and color == trump
        color_length = int((hand * color_masks[color]).sum())
        return is_trump, card_values[trump, card], card_strength(card, color, trump), -color_length
