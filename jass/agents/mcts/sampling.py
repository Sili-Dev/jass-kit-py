"""
Information about the hidden cards of the other players: which colours they are known not to have, and random
distributions of the unseen cards that are consistent with the observation (determinization).
"""
import numpy as np

from jass.game.const import DJ, color_of_card, color_offset
from jass.game.game_observation import GameObservation


def known_voids(obs: GameObservation) -> np.ndarray:
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


def cards_left_per_player(obs: GameObservation) -> np.ndarray:
    """Number of cards each player still holds."""
    left = np.full(4, 9, dtype=np.int32)
    for i in range(obs.nr_tricks + 1):
        first_player = obs.trick_first_player[i]
        for j in range(4):
            if obs.tricks[i, j] < 0:
                break
            left[(first_player - j) % 4] -= 1
    return left


def sample_hands(obs: GameObservation, rng: np.random.Generator, max_attempts: int = 50) -> np.ndarray:
    """
    Deal the cards the observing player has not seen to the other players, so that every player gets the number
    of cards they still hold and nobody gets a card of a color they are known to be out of. A player out of trump
    may still get the jack of trump. If no such deal is found in max_attempts, the voids are ignored.

    Args:
        obs: the observation of the player to move
        rng: random number generator
        max_attempts: number of attempts to find a deal that respects the voids
    Returns:
        one-hot encoded hands of all four players, [4, 36]
    """
    played = obs.tricks[obs.tricks >= 0]
    unseen = np.ones(36, dtype=np.int32) - obs.hand
    unseen[played] = 0
    unseen_cards = np.flatnonzero(unseen)

    slots = cards_left_per_player(obs)
    slots[obs.player] = 0

    voids = known_voids(obs)
    allowed = np.zeros((4, 36), dtype=bool)
    for p in range(4):
        if p != obs.player:
            allowed[p] = ~voids[p, color_of_card]
    if obs.trump < 4:
        allowed[:, color_offset[obs.trump] + DJ] = True
    allowed[obs.player] = False

    hands = np.zeros((4, 36), dtype=np.int32)
    hands[obs.player] = obs.hand

    for _ in range(max_attempts):
        deal = _deal_respecting(unseen_cards, slots, allowed, rng)
        if deal is not None:
            return hands + deal

    # fallback: ignore the voids, hand sizes are still respected
    shuffled = rng.permutation(unseen_cards)
    start = 0
    for p in range(4):
        hands[p, shuffled[start:start + slots[p]]] = 1
        start += slots[p]
    return hands


def _deal_respecting(cards: np.ndarray, slots: np.ndarray, allowed: np.ndarray, rng: np.random.Generator):
    """One attempt to deal the cards, most constrained card first. Returns the hands or None on failure."""
    free = slots.copy()
    deal = np.zeros((4, 36), dtype=np.int32)
    remaining = list(rng.permutation(cards))
    while remaining:
        # the card that the fewest players with free slots may receive, ties in the random order
        options = [np.flatnonzero(allowed[:, c] & (free > 0)) for c in remaining]
        index = min(range(len(remaining)), key=lambda i: len(options[i]))
        if len(options[index]) == 0:
            return None
        player = rng.choice(options[index])
        deal[player, remaining.pop(index)] = 1
        free[player] -= 1
    return deal
