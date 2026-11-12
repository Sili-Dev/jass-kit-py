"""
Train a feed-forward trump network on the Swisslos data (exercise 10) and export it for the bot.

    python -m jass.train.trump.train_nn --csv path/to/2018_10_18_trump.csv --out models/trump_nn.npz
    python -m jass.train.trump.train_nn --csv ... --out models/trump_nn_strong.npz \\
        --stats path/to/player_all_stat.json --min-player-mean 79.5 --min-player-games 1000

The test split is the one of jass.train.trump.train (stratified 80/20, same seed) on all rows; the strong player
filter only removes training rows, so all models are scored on the same test rows. Logistic regression on the
'cards' features (the Ex 8 model) is trained on the same split and printed for reference.

Needs the train-nn extra: pip install -e .[train-nn]
"""
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from jass.agents.trump.features import NR_FEATURES, FEATURE_SETS
from jass.train.trump.data import load_trump_csv, strong_player_mask
from jass.train.trump.train import MODELS, full_proba, mask_push, report

FEATURE_SET = 'cards+suit'
VALIDATION_SHARE = 0.1


def parse_args(argv: Optional[List[str]]) -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description='Train the neural trump model (exercise 10)')
    parser.add_argument('--csv', required=True, type=Path, help='2018_10_18_trump.csv')
    parser.add_argument('--out', type=Path, default=Path('models/trump_nn.npz'), help='exported model')
    parser.add_argument('--plot', type=Path, default=None, help='training curves, default results/<out stem>.png')
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 64], help='hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=50, help='maximum, training stops early on validation loss')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--stats', type=Path, default=None, help='player_all_stat.json, enables the player filter')
    parser.add_argument('--min-player-mean', type=float, default=79.5)
    parser.add_argument('--min-player-games', type=int, default=1000)
    parser.add_argument('--keep-user0', action='store_true', help='keep user 0 (anonymous players) when filtering')
    parser.add_argument('--nrows', type=int, default=None, help='only read the first rows, for quick runs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.plot is None:
        args.plot = Path('results') / (args.out.stem + '.png')
    return args, parser


def plot_history(history: dict, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (loss, acc) = plt.subplots(1, 2, figsize=(10, 4))
    for axis, key, title in [(loss, 'loss', 'Loss'), (acc, 'accuracy', 'Accuracy')]:
        axis.plot(history[key], label='train')
        axis.plot(history['val_' + key], label='validation')
        axis.set_title(title)
        axis.set_xlabel('epoch')
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print('Saved training curves to {}'.format(path))


def main(argv: Optional[List[str]] = None):
    args, parser = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    data = load_trump_csv(args.csv, nrows=args.nrows, feature_set=FEATURE_SET)
    X_train, X_test, y_train, y_test, users_train, _ = train_test_split(
        data.X, data.y, data.users, test_size=0.2, stratify=data.y, random_state=args.seed)
    print('train {}, test {}'.format(len(y_train), len(y_test)))

    if args.stats is not None:
        keep = strong_player_mask(users_train, args.stats, min_mean=args.min_player_mean,
                                  min_games=args.min_player_games, keep_user0=args.keep_user0)
        if not keep.any():
            parser.error('the player filter (mean >= {}, games >= {}, stats {}) keeps no training rows'.format(
                args.min_player_mean, args.min_player_games, args.stats))
        X_train, y_train = X_train[keep], y_train[keep]
        print('player filter keeps {} of {} training rows'.format(keep.sum(), len(keep)))

    start = time.time()
    logreg = MODELS['logreg'](args.seed).fit(X_train[:, :NR_FEATURES], y_train)
    report('logreg', y_test, mask_push(full_proba(logreg, X_test[:, :NR_FEATURES]), X_test), time.time() - start)

    # imported late so that --help and the argument errors above do not wait for tensorflow
    import tensorflow as tf
    from jass.train.trump.keras_export import build_network, to_trump_model

    tf.keras.utils.set_random_seed(args.seed)
    order = np.random.default_rng(args.seed).permutation(len(y_train))
    nr_val = max(1, int(len(order) * VALIDATION_SHARE))
    fit_idx, val_idx = order[:-nr_val], order[-nr_val:]

    network = build_network(FEATURE_SETS[FEATURE_SET], hidden=args.hidden, dropout=args.dropout)
    network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    start = time.time()
    history = network.fit(
        X_train[fit_idx], y_train[fit_idx], validation_data=(X_train[val_idx], y_train[val_idx]),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience,
                                                    restore_best_weights=True)])

    model = to_trump_model(network, FEATURE_SET)
    name = 'nn ' + 'x'.join(str(h) for h in args.hidden)
    report(name, y_test, np.argmax(model.logits(X_test), axis=1), time.time() - start)

    model.save(args.out)
    print('Saved {} model to {}'.format(name, args.out))
    plot_history(history.history, args.plot)


if __name__ == '__main__':
    main()
