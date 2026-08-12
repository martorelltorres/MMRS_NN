"""Paired comparison of OWA against the round-robin baseline.

Both campaigns simulate the same six cells over the same eight object layouts, with the same
seeds, so every OWA mission has a round-robin counterpart on an identical scenario and the
comparison can be blocked by realization.

Utility is compared, not latency. Round-robin tends to leave part of a vehicle's buffer
uncollected, and objects that are never delivered do not enter a latency mean, so raw latency
flatters the weaker policy. Equation (6) carries the transmitted counts in the numerator and
does not have that defect.
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
import rosbag
from scipy.stats import wilcoxon

import campaign_paths

CAMPAIGNS = (('v2_dispersion', 'owa'), ('v3_roundrobin', 'roundrobin'))


def metrics(path):
    pri, reg = [], []
    m = {'priority_objects': 0, 'regular_objects': 0, 'travelled_distance': 0.0}
    bag = rosbag.Bag(path)
    for topic, msg, _ in bag.read_messages():
        if topic.endswith('regular_communication_latency'):
            reg.append(sum(getattr(msg, 'comm_latency', ())))
        elif topic.endswith('priority_communication_latency'):
            pri.append(sum(getattr(msg, 'comm_latency', ())))
        elif topic.endswith('asv_travelled_distance'):
            m['travelled_distance'] = msg.travelled_distance
        elif topic.endswith('allocator_data_transmited'):
            m['regular_objects'] = sum(getattr(msg, 'transmitted_regular_objects', ()))
            m['priority_objects'] = sum(getattr(msg, 'transmitted_priority_objects', ()))
    bag.close()
    m['priority_latency'] = float(np.mean(pri)) if pri else 0.0
    m['priority_std'] = float(np.std(pri, ddof=1)) if len(pri) > 1 else 0.0
    m['regular_latency'] = float(np.mean(reg)) if reg else 0.0
    m['regular_std'] = float(np.std(reg, ddof=1)) if len(reg) > 1 else 0.0
    return m


def collect(data_root, refresh=False):
    cache_file = campaign_paths.cache_path(data_root, 'roundrobin_metrics.pkl')
    cache = {}
    if os.path.exists(cache_file) and not refresh:
        cache = pickle.load(open(cache_file, 'rb'))
    rows = []
    for root, agg in CAMPAIGNS:
        base = campaign_paths.resolve(data_root, root)
        for man in sorted(glob.glob(os.path.join(base, '*/*/%s/manifest.csv' % agg))):
            folder = os.path.dirname(man)
            cell = folder.replace('/%s' % agg, '').split('%s/' % root)[1]
            for _, r in pd.read_csv(man).iterrows():
                if r.outcome != 'completed':
                    continue
                if isinstance(r.warnings, str) and r.warnings.strip():
                    continue
                path = os.path.join(folder, 'bagfiles', str(r['bagfile']))
                if not os.path.exists(path):
                    continue
                if path not in cache:
                    cache[path] = metrics(path)
                index = int(r['index'])
                m = dict(cache[path])
                m.update(cell=cell, auvs=int(cell.split('/')[1][0]), policy=agg,
                         cfg=(index // 8 if agg == 'owa' else -1),
                         rep=(index % 8 if agg == 'owa' else index))
                rows.append(m)
    pickle.dump(cache, open(cache_file, 'wb'))
    return pd.DataFrame(rows)


def utility(group, alpha=1.0, beta=0.5, gamma=1.0):
    """Equations (7), (8) and (6) over every policy evaluated in the cell."""
    g = group.copy()
    for c in ['regular_latency', 'priority_latency', 'regular_std', 'priority_std',
              'travelled_distance']:
        lo, hi = g[c].min(), g[c].max()
        g[c + '_n'] = 0.0 if hi - lo == 0 else (g[c] - lo) / (hi - lo)
    for c in ['priority_objects', 'regular_objects']:
        hi = g[c].max()
        g[c + '_n'] = 0.0 if hi == 0 else g[c] / hi
    pri = g['priority_objects_n'] / (1 + np.exp(g['priority_latency_n'] + g['priority_std_n']))
    reg = g['regular_objects_n'] / (1 + np.exp(g['regular_latency_n'] + g['regular_std_n']))
    g['utility'] = alpha * pri + beta * reg + gamma * np.exp(-g['travelled_distance_n'])
    return g


def blocks(frame, cell=None, fleet=None):
    """Matched (OWA, round-robin) utility pairs sharing a cell and a realization."""
    sub = frame if cell is None else frame[frame.cell == cell]
    sub = sub if fleet is None else sub[sub.auvs == fleet]
    pairs = []
    for _, g in sub.groupby('cell'):
        rr = g[g.policy == 'roundrobin'].set_index('rep').utility
        ow = g[(g.policy == 'owa') & (g.cfg == 0)].set_index('rep').utility
        for k in sorted(set(rr.index) & set(ow.index)):
            pairs.append((ow[k], rr[k]))
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def report(a, b, label):
    if len(a) < 6:
        print('  %-26s solo %d bloques, insuficiente' % (label, len(a)))
        return
    p = wilcoxon(a, b).pvalue
    mark = '***' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print('  %-26s n=%2d  OWA gana %2d/%-2d  dif %+6.1f%%  p=%.5f  %s'
          % (label, len(a), (a > b).sum(), len(a),
             100 * (a.mean() - b.mean()) / b.mean(), p, mark))


def main():
    ap = campaign_paths.parser(__doc__)
    ap.add_argument('--refresh', action='store_true', help='ignore the metric cache')
    args = ap.parse_args()

    d = collect(args.data_root, args.refresh)
    print('misiones: OWA %d, round-robin %d\n'
          % (sum(d.policy == 'owa'), sum(d.policy == 'roundrobin')))

    u = pd.concat([utility(g) for _, g in d.groupby('cell')])

    print('=== OWA (4,4,2) frente a round-robin, pareado por disposicion ===\n')
    for cell in sorted(u.cell.unique()):
        report(*blocks(u, cell=cell), label=cell)

    print('\n=== agrupado por tamano de flota ===\n')
    for fleet in sorted(u.auvs.unique()):
        report(*blocks(u, fleet=fleet), label='%d AUVs' % fleet)


if __name__ == '__main__':
    main()
