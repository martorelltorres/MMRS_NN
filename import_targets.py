#!/usr/bin/env python3
"""Bring the regression targets produced by the simulation campaign into this repository.

Closes the gap between the two repositories: multi_robot_system extracts the campaign into
`<campaign>/{train,test}_targets.csv`, and this script rewrites them in the layout the
regressor already expects, so no step is done by hand.

    python3 import_targets.py --campaign path/to/campaigns/v2_sweep

Reads
    <campaign>/train_targets.csv    w_max per (area, fleet size), training areas
    <campaign>/test_targets.csv     w_max per (area, fleet size), test areas

Writes
    weights/{3,4,5,6}AUV_weights.csv    6 rows each, one per training area
    weights/optimal_test_weights.csv    20 rows, the five test areas x four fleet sizes

The previous files are moved to weights/published/ first: they are the ones behind the
published figures and must stay reproducible.
"""

import argparse
import os
import shutil
import sys

import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(REPO, 'weights')
BACKUP_DIR = os.path.join(WEIGHTS_DIR, 'published')

# Column order the regressor reads, and the 0-10 scale it works on.
COLUMNS = ['area', 'auv_count', 'utility', 'w1', 'w2', 'w3']
WEIGHT_SUM = 10.0

# The weight grid W of Section 3.6, on the 0-10 scale.
OWA_WEIGHT_GRID = {(4, 4, 2), (6, 2, 2), (6, 4, 0), (8, 2, 0), (10, 0, 0)}

EXPECTED_TRAIN_AREAS = 6
EXPECTED_TEST_ROWS = 20
FLEETS = [3, 4, 5, 6]

WARNINGS = []


def warn(message):
    WARNINGS.append(message)
    print('WARNING: %s' % message)


def load_targets(path):
    """Read a targets file and normalize its column names to what the regressor expects."""
    frame = pd.read_csv(path)

    # The extractor reports the mean utility across realizations; older files just say
    # "utility". Accept both so this works before and after the repeated-trials campaign.
    if 'utility' not in frame.columns:
        if 'utility_mean' not in frame.columns:
            raise ValueError('%s has neither utility nor utility_mean' % path)
        frame = frame.rename(columns={'utility_mean': 'utility'})

    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError('%s is missing the columns %s' % (path, missing))

    return frame[COLUMNS].copy()


def rescale(frame):
    """Put the weighting vectors on the 0-10 scale this repository uses.

    The parameter server writes them summing to 1 and the datasets use 10; the two differ by
    a constant factor that cannot change the ordering of the AUV scores.
    """
    frame = frame.copy()
    totals = frame[['w1', 'w2', 'w3']].sum(axis=1)
    needs_scaling = totals < WEIGHT_SUM / 2.0
    if needs_scaling.any():
        frame.loc[needs_scaling, ['w1', 'w2', 'w3']] *= 10.0
    return frame


def check_grid(frame, label):
    """Flag weight vectors outside the grid: the signature of a predicted-weights run."""
    for _, row in frame.iterrows():
        weights = tuple(round(row[c], 6) for c in ('w1', 'w2', 'w3'))
        if weights not in OWA_WEIGHT_GRID:
            warn('%s: area %d / %d AUVs has the weights %s, which are not in the grid W. '
                 'A predicted-weights run may have contaminated that cell of the sweep.'
                 % (label, row.area, row.auv_count, weights))


def backup(paths):
    if not any(os.path.isfile(p) for p in paths):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for path in paths:
        if os.path.isfile(path):
            target = os.path.join(BACKUP_DIR, os.path.basename(path))
            if not os.path.exists(target):
                shutil.copy2(path, target)
    print('previous files kept in %s' % BACKUP_DIR)


def main():
    parser = argparse.ArgumentParser(
        description='Import campaign targets into the regressor layout.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--campaign', required=True,
                        help='folder holding train_targets.csv and test_targets.csv')
    parser.add_argument('--train-prefix', default='train')
    parser.add_argument('--test-prefix', default='test')
    parser.add_argument('--dry-run', action='store_true',
                        help='report what would be written without touching any file')
    args = parser.parse_args()

    train_path = os.path.join(args.campaign, '%s_targets.csv' % args.train_prefix)
    test_path = os.path.join(args.campaign, '%s_targets.csv' % args.test_prefix)
    for path in (train_path, test_path):
        if not os.path.isfile(path):
            print('error: %s not found' % path)
            return 1

    train = rescale(load_targets(train_path))
    test = rescale(load_targets(test_path))

    check_grid(train, 'training targets')
    check_grid(test, 'test targets')

    if len(test) != EXPECTED_TEST_ROWS:
        warn('the test set has %d rows, %d expected (5 areas x 4 fleet sizes)'
             % (len(test), EXPECTED_TEST_ROWS))

    outputs = {}
    for fleet in FLEETS:
        subset = train[train.auv_count == fleet].sort_values('area')
        if len(subset) != EXPECTED_TRAIN_AREAS:
            warn('%d AUVs: %d training areas, %d expected'
                 % (fleet, len(subset), EXPECTED_TRAIN_AREAS))
        outputs[os.path.join(WEIGHTS_DIR, '%dAUV_weights.csv' % fleet)] = subset

    outputs[os.path.join(WEIGHTS_DIR, 'optimal_test_weights.csv')] = \
        test.sort_values(['area', 'auv_count'])

    print('\ntraining areas: %s' % sorted(train.area.unique()))
    print('test areas:     %s' % sorted(test.area.unique()))

    if args.dry_run:
        print('\n--dry-run: nothing written')
        for path, frame in outputs.items():
            print('  would write %s (%d rows)' % (os.path.basename(path), len(frame)))
        return 0

    backup(list(outputs))
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
        print('wrote %s (%d rows)' % (path, len(frame)))

    if WARNINGS:
        print('\n%d warning(s) raised, review them before training:' % len(WARNINGS))
        for message in WARNINGS:
            print('  - %s' % message)

    return 0


if __name__ == '__main__':
    sys.exit(main())
