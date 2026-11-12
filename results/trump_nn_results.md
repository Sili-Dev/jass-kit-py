# Ex 10: neural trump results (2026-11-12)

## Offline (same stratified 80/20 split, seed 42, 71,965 test rows)

| Model | Inputs | Training rows | Accuracy | Macro-F1 | Epochs |
|---|---|---|---|---|---|
| logistic regression (Ex 8) | cards (37) | 287,860 | 0.6357 | 0.5949 | – |
| network 64 | cards+suit (53) | 287,860 | 0.6635 | 0.6204 | 50 (cap) |
| network 128×64 (`models/trump_nn.npz`) | cards+suit (53) | 287,860 | 0.6672 | 0.6189 | 41 (early stop) |
| network 256×128 | cards+suit (53) | 287,860 | 0.6668 | 0.6217 | 26 (early stop) |
| network 128×64, strong players (`models/trump_nn_strong.npz`) | cards+suit (53) | 49,670 | 0.6527 | 0.6048 | 50 (cap) |

Strong players: mean ≥ 79.5 points and ≥ 1,000 games in `player_all_stat.json`, user 0 excluded. Training curves are
the PNGs next to this file.

## Arena (`examples/arena/evaluate.py`, 600 deals each played from both sides, 100 iterations, seed 2026)

| Agent trump | Opponent | Points per deal | 95% CI | Point share |
|---|---|---|---|---|
| linear | plain MCTS (rule based trump) | −0.30 | −3.01 .. +2.41 | 49.9% |
| network 128×64 | plain MCTS | −0.63 | −3.25 .. +1.99 | 49.8% |
| network 128×64 | linear | −0.81 | −3.29 .. +1.67 | 49.7% |
| network strong | network 128×64 | −1.25 | −3.76 .. +1.25 | 49.6% |

## Conclusion

The network predicts the Swisslos players' trump choice about 3 points of accuracy better than logistic regression,
and capacity beyond one hidden layer of 64 units adds nothing. In play none of the models differs measurably from
the others or from the rule based trump of plain MCTS: every interval contains 0 and is about ±2.5 points wide.
The deployment rule (CI lower bound above 0 against the linear model) is not met, so the bot keeps
`trump_linear.npz`.
