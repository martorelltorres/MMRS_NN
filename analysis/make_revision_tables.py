"""Build the tables of the revised Sections 5.2 and 5.3 from the baseline and predicted campaigns.

Two results are produced here, both of which the original submission could not report because it
simulated one mission per configuration.

Section 5.2 -- the reliability of Table 2. The Average Prediction Accuracy of Equation (13)
divides by the spread of the five weighting vectors of the grid. The replication campaign shows
that spread is of the same order as the variation induced by moving the objects, so the quotient
is a ratio of two comparable noise terms. This is quantified directly: the same weighting vector
is scored against the same grid on each of the eight object layouts of a cell, and the resulting
APA is tabulated. An APA computed from a single layout, which is what Table 2 reports, cannot be
read as a property of the weighting vector.

Section 5.3 -- the baseline comparison. OWA is compared against round-robin, which serves the
vehicles in turn and ignores link quality, range and pending data. Both campaigns cover the same
six cells over the same eight object layouts with the same seeds, so the contrast is paired and
blocked by layout. Utility is compared rather than latency: round-robin leaves part of a buffer
uncollected, and objects that are never delivered never enter a latency mean, which flatters the
weaker policy. Equation (6) carries the transmitted counts in its numerator and does not have
that defect.

    python3 make_revision_tables.py [--outdir DIR]
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
import rosbag
from scipy.stats import wilcoxon

GRID = [(4, 4, 2), (6, 2, 2), (6, 4, 0), (8, 2, 0), (10, 0, 0)]
import campaign_paths

ALPHA, BETA, GAMMA = 1.0, 0.5, 1.0


# --------------------------------------------------------------------------- data

def raw_metrics(path):
    """Mission-level metrics, matching owa_artm_data_extractor exactly."""
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


def collect(dispersion, roundrobin, cache_file):
    """Both campaigns in one frame, keyed by cell, policy and object layout."""
    cache = pickle.load(open(cache_file, 'rb')) if os.path.exists(cache_file) else {}
    rows = []
    for root, policy in ((dispersion, 'owa'), (roundrobin, 'roundrobin')):
        for man in sorted(glob.glob(os.path.join(root, '*/*/%s/manifest.csv' % policy))):
            folder = os.path.dirname(man)
            cell = folder.replace('/%s' % policy, '').split(os.path.basename(root) + '/')[1]
            for _, r in pd.read_csv(man).iterrows():
                if r.outcome != 'completed':
                    continue
                if isinstance(r.warnings, str) and r.warnings.strip():
                    continue
                path = os.path.join(folder, 'bagfiles', str(r['bagfile']))
                if not os.path.exists(path):
                    continue
                if path not in cache:
                    cache[path] = raw_metrics(path)
                m = dict(cache[path])
                index = int(r['index'])
                m.update(cell=cell, area=int(cell.split('/')[0]), auvs=int(cell.split('/')[1][0]),
                         policy=policy,
                         cfg=(index // 8 if policy == 'owa' else -1),
                         rep=(index % 8 if policy == 'owa' else index))
                rows.append(m)
    pickle.dump(cache, open(cache_file, 'wb'))
    return pd.DataFrame(rows)


def utility(group):
    """Equations (7), (8) and (6) within one normalization group."""
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
    g['utility'] = ALPHA * pri + BETA * reg + GAMMA * np.exp(-g['travelled_distance_n'])
    return g


def fmt_p(p):
    return r'$<$0.001' if p < 0.001 else '%.3f' % p


# --------------------------------------------------------------------------- 5.2

def apa_stability(d):
    """APA of each weighting vector on each object layout, normalised within the layout.

    Every layout carries the five vectors of the grid, so U_Wmin and U_Wmax are recomputed
    from that layout alone. This reproduces the procedure of Table 2 eight times over, changing
    nothing but the placement of the objects.
    """
    owa = d[d.policy == 'owa']
    rows = []
    for (area, auvs), cell in owa.groupby(['area', 'auvs']):
        for rep, layout in cell.groupby('rep'):
            if len(layout) < len(GRID):
                continue
            u = utility(layout)
            lo, hi = u.utility.min(), u.utility.max()
            if hi <= lo:
                continue
            for _, row in u.iterrows():
                rows.append({'area': area, 'auvs': auvs, 'rep': rep, 'cfg': int(row.cfg),
                             'APA': 100 * (row.utility - lo) / (hi - lo), 'spread': hi - lo})
    return pd.DataFrame(rows)


def table_apa_stability(a, path):
    lines = [
        r'\begin{table}[t]',
        r'\caption{Average Prediction Accuracy of Equation~(\ref{eq:APM}) recomputed for each'
        r' weighting vector on each of the eight object layouts of a cell. Nothing changes'
        r' between the columns of a row except the placement of the objects. The metric is'
        r' therefore not a property of the weighting vector, and an APA obtained from a single'
        r' layout carries no information about it.}',
        r'\label{tab:apa_stability}',
        r'\centering',
        r'\begin{tabular}{lrrr}',
        r'\toprule',
        r'$\vec{w}$ & Mean APA (\%) & Range over layouts (\%) & Width (pp) \\',
        r'\midrule',
    ]
    for (area, auvs), cell in a.groupby(['area', 'auvs']):
        lines.append(r'\multicolumn{4}{l}{\textit{%d m$^2$, %d AUVs}} \\' % (area, auvs))
        for cfg, g in cell.groupby('cfg'):
            lines.append(r'(%d,%d,%d) & %.0f & [%.0f, %.0f] & %.0f \\'
                         % (GRID[cfg] + (g.APA.mean(), g.APA.min(), g.APA.max(),
                                         g.APA.max() - g.APA.min())))
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\end{table}']
    open(path, 'w').write('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------- 5.3

def baseline_pairs(u, cell=None, fleet=None):
    """Matched (OWA, round-robin) utility pairs sharing a cell and an object layout."""
    sub = u if cell is None else u[u.cell == cell]
    sub = sub if fleet is None else sub[sub.auvs == fleet]
    a, b = [], []
    for _, g in sub.groupby('cell'):
        rr = g[g.policy == 'roundrobin'].set_index('rep').utility
        ow = g[(g.policy == 'owa') & (g.cfg == 0)].set_index('rep').utility
        for k in sorted(set(rr.index) & set(ow.index)):
            a.append(ow[k])
            b.append(rr[k])
    return np.array(a), np.array(b)


def table_baseline(u, path):
    lines = [
        r'\begin{table}[t]',
        r'\caption{Communication-aware aggregation against the round-robin baseline, paired by'
        r' object layout (Wilcoxon signed-rank, eight blocks per cell). OWA uses the balanced'
        r' vector $(4,4,2)$. The advantage of reasoning about link quality, range and pending'
        r' data appears only when the fleet is large enough to contend for the single surface'
        r' vehicle.}',
        r'\label{tab:baseline}',
        r'\centering',
        r'\begin{tabular}{rrrrl}',
        r'\toprule',
        r'Area (m$^2$) & AUVs & Wins & $\Delta U$ (\%) & $p$ \\',
        r'\midrule',
    ]
    for cell in sorted(u.cell.unique()):
        a, b = baseline_pairs(u, cell=cell)
        area, fleet = cell.split('/')[0], cell.split('/')[1][0]
        lines.append(r'%s & %s & %d/%d & %+.1f & %s \\'
                     % (area, fleet, (a > b).sum(), len(a),
                        100 * (a.mean() - b.mean()) / b.mean(), fmt_p(wilcoxon(a, b).pvalue)))
    lines.append(r'\midrule')
    for fleet in sorted(u.auvs.unique()):
        a, b = baseline_pairs(u, fleet=fleet)
        lines.append(r'\multicolumn{2}{l}{\textit{pooled, %d AUVs}} & %d/%d & %+.1f & %s \\'
                     % (fleet, (a > b).sum(), len(a),
                        100 * (a.mean() - b.mean()) / b.mean(), fmt_p(wilcoxon(a, b).pvalue)))
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(path, 'w').write('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------- text

def summary(d, u, a, out):
    lines = ['Numbers quoted in the running text of Sections 5.2 and 5.3', '=' * 62,
             'OWA missions:        %d' % (d.policy == 'owa').sum(),
             'round-robin missions: %d' % (d.policy == 'roundrobin').sum(), '']

    width = a.groupby(['area', 'auvs', 'cfg']).APA.agg(['min', 'max'])
    full = ((width['min'] < 5) & (width['max'] > 95)).sum()
    lines += ['-- Section 5.2, reliability of the APA metric --',
              'mean width of the APA interval:   %.0f pp of the 0-100 scale'
              % (width['max'] - width['min']).mean(),
              'vectors spanning the full range:  %d of %d' % (full, len(width)),
              'grid spread U_Wmax - U_Wmin:      %.3f mean, %.3f to %.3f'
              % (a.spread.mean(), a.spread.min(), a.spread.max()), '']

    lines += ['-- Section 5.3, baseline comparison --']
    for fleet in sorted(u.auvs.unique()):
        p, q = baseline_pairs(u, fleet=fleet)
        lines.append('%d AUVs: OWA wins %2d/%d, dU %+.1f%%, p = %.5f'
                     % (fleet, (p > q).sum(), len(p),
                        100 * (p.mean() - q.mean()) / q.mean(), wilcoxon(p, q).pvalue))
    for cell in sorted(u.cell.unique()):
        p, q = baseline_pairs(u, cell=cell)
        lines.append('  %-14s OWA wins %d/%d, dU %+.1f%%, p = %.5f'
                     % (cell, (p > q).sum(), len(p),
                        100 * (p.mean() - q.mean()) / q.mean(), wilcoxon(p, q).pvalue))

    text = '\n'.join(lines)
    open(out, 'w').write(text + '\n')
    print(text)


def main():
    ap = campaign_paths.parser(__doc__)
    ap.add_argument('--outdir', default='article_tables')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = collect(campaign_paths.resolve(args.data_root, 'v2_dispersion'),
                campaign_paths.resolve(args.data_root, 'v3_roundrobin'),
                campaign_paths.cache_path(args.data_root, 'roundrobin_metrics.pkl'))
    u = pd.concat([utility(g) for _, g in d.groupby('cell')])
    a = apa_stability(d)

    table_apa_stability(a, os.path.join(args.outdir, 'tab_apa_stability.tex'))
    table_baseline(u, os.path.join(args.outdir, 'tab_baseline.tex'))
    summary(d, u, a, os.path.join(args.outdir, 'text_figures_revision.txt'))
    print('\ntables written to %s' % args.outdir)


if __name__ == '__main__':
    main()
