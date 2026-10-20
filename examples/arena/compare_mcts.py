"""
Compare the determinization MCTS agent with a reference agent on a fixed set of deals.

Every deal is played twice with the teams swapped between the seats, so that the luck of the cards cancels. The
result is the mean point difference per deal with a 95% confidence interval.

    python examples/arena/compare_mcts.py --deals 100 --time-budget 0.2 --opponent rule_based
"""
import argparse
import logging

import numpy as np

from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.agent_rule_based_schieber import AgentRuleBasedSchieber
from jass.agents.mcts.agent_determinization_mcts import AgentDeterminizationMCTS
from jass.arena.arena import Arena
from jass.arena.dealing_card_seeded_strategy import DealingCardSeededStrategy

OPPONENTS = {
    'rule_based': AgentRuleBasedSchieber,
    'random': AgentRandomSchieber,
}


def play(north_south, east_west, deals: int, seed: int) -> Arena:
    arena = Arena(nr_games_to_play=deals, dealing_card_strategy=DealingCardSeededStrategy(seed),
                  print_every_x_games=max(1, deals // 20))
    arena.set_players(north_south, east_west, north_south, east_west)
    arena.play_all_games()
    return arena


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--deals', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--time-budget', type=float, default=0.2)
    parser.add_argument('--opponent', choices=sorted(OPPONENTS), default='rule_based')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    mcts = AgentDeterminizationMCTS(time_budget=args.time_budget, seed=args.seed)
    opponent = OPPONENTS[args.opponent]()

    # both arenas start with the same dealer, so deal i is the same in both
    mcts_first = play(mcts, opponent, args.deals, args.seed)
    mcts_second = play(opponent, mcts, args.deals, args.seed)

    mcts_points = mcts_first.points_team_0 + mcts_second.points_team_1
    opponent_points = mcts_first.points_team_1 + mcts_second.points_team_0
    diff = (mcts_points - opponent_points) / 2
    half_width = 1.96 * diff.std(ddof=1) / np.sqrt(args.deals) if args.deals > 1 else float('nan')

    print('MCTS ({:.2f} s per card) against {}, {} deals played on both sides'.format(
        args.time_budget, args.opponent, args.deals))
    print('Mean point difference per deal: {:+.1f} (95% CI {:+.1f} .. {:+.1f})'.format(
        diff.mean(), diff.mean() - half_width, diff.mean() + half_width))
    print('Share of points: {:.1%}'.format(mcts_points.sum() / (mcts_points.sum() + opponent_points.sum())))


if __name__ == '__main__':
    main()
