"""
Monte Carlo tree search for the card play of Schieber on a fully known game state (perfect information).

Each node stores the number of visits and the sum of the rewards of both teams, the reward of a simulation being
the points of each team divided by 157. Selection uses UCB1 from the point of view of the team that moves at the
parent node (lecture C2).
"""
import math
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from jass.agents.mcts.rollout import copy_state, random_rollout
from jass.game.const import team
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber

TOTAL_POINTS = 157


class Node:
    """
    Node of the search tree, it corresponds to the state after its card was played.
    """
    def __init__(self, parent: Optional['Node'], card: int, player: int, untried: List[int]):
        self.parent = parent
        self.card = card
        # player to move in this node, -1 if the round is over
        self.player = player
        self.children: Dict[int, 'Node'] = {}
        self.untried = untried
        self.n = 0
        self.w = np.zeros(2, dtype=np.float64)


class MCTS:
    """
    Search the card to play in a fully known game state.
    """
    def __init__(self, rule: RuleSchieber, rng: np.random.Generator, c: float = math.sqrt(2),
                 rollout: Callable[[GameSim, RuleSchieber, np.random.Generator], None] = random_rollout):
        self._rule = rule
        self._rng = rng
        self._c = c
        self._rollout = rollout

    def search(self, state: GameState, iterations: Optional[int] = None,
               deadline: Optional[float] = None) -> Dict[int, int]:
        """
        Run the search from the given state, which is not changed. At least one iteration is always run, then the
        search stops at whichever of the two limits is reached first.

        Args:
            state: the state in which the player to move has to play a card
            iterations: maximal number of iterations
            deadline: time (from time.perf_counter()) after which no new iteration is started
        Returns:
            number of visits for each card that can be played
        """
        if iterations is None and deadline is None:
            raise ValueError('Either iterations or deadline must be given')

        root = self._new_node(None, -1, state)
        sim = GameSim(rule=self._rule)
        nr_iterations = 0
        while nr_iterations == 0 or \
                ((iterations is None or nr_iterations < iterations) and
                 (deadline is None or time.perf_counter() < deadline)):
            sim._state = copy_state(state)
            self._iterate(root, sim)
            nr_iterations += 1
        return {card: child.n for card, child in root.children.items()}

    def _iterate(self, root: Node, sim: GameSim) -> None:
        node = root
        # selection
        while not node.untried and node.children:
            node = self._select(node)
            sim.action_play_card(node.card)
        # expansion
        if node.untried:
            card = node.untried.pop(int(self._rng.integers(len(node.untried))))
            sim.action_play_card(card)
            child = self._new_node(node, card, sim.state)
            node.children[card] = child
            node = child
        # simulation
        self._rollout(sim, self._rule, self._rng)
        reward = sim.state.points / TOTAL_POINTS
        # backpropagation
        while node is not None:
            node.n += 1
            node.w += reward
            node = node.parent

    def _select(self, node: Node) -> Node:
        node_team = team[node.player]
        log_n = math.log(node.n)
        return max(node.children.values(),
                   key=lambda child: child.w[node_team] / child.n + self._c * math.sqrt(log_n / child.n))

    def _new_node(self, parent: Optional[Node], card: int, state: GameState) -> Node:
        if state.nr_played_cards == 36:
            return Node(parent, card, -1, [])
        untried = np.flatnonzero(self._rule.get_valid_cards_from_state(state)).tolist()
        return Node(parent, card, state.player, untried)


def best_card(visits: Dict[int, int]) -> int:
    """The card with the most visits, ties are broken by the lowest card."""
    return min(visits, key=lambda card: (-visits[card], card))
