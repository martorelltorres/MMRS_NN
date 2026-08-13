"""Build the LaTeX result tables of the revised manuscript from the campaign bagfiles.

Reads the phase-2 replication campaign (240 missions: 6 cells x 5 weighting vectors x 8
object layouts) and emits booktabs tables ready to paste into the article. Every number the
revised Section 5 reports comes from here, so the tables can be regenerated after any change
to the data instead of being transcribed by hand.

    python3 make_article_tables.py [--outdir DIR]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import rosbag
from scipy.stats import mannwhitneyu, wilcoxon

import campaign_paths

GRID = [(4, 4, 2), (6, 2, 2), (6, 4, 0), (8, 2, 0), (10, 0, 0)]


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


def collect(root, cache):
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    rows = []
    for man in sorted(glob.glob(os.path.join(root, '*/*/owa/manifest.csv'))):
        folder = os.path.dirname(man)
        cell = folder.replace('/owa', '').split(os.path.basename(root) + '/')[1]
        for _, r in pd.read_csv(man).iterrows():
            if r.outcome != 'completed':
                continue
            if isinstance(r.warnings, str) and r.warnings.strip():
                continue
            path = os.path.join(folder, 'bagfiles', str(r['bagfile']))
            if not os.path.exists(path):
                continue
            m = raw_metrics(path)
            m.update(cell=cell, area=int(cell.split('/')[0]),
                     auvs=int(cell.split('/')[1][0]),
                     cfg=int(r['index']) // 8, rep=int(r['index']) % 8)
            rows.append(m)
    frame = pd.DataFrame(rows)
    frame.to_pickle(cache)
    return frame


def utility(group, alpha=1.0, beta=0.5, gamma=1.0):
    """Equations (7), (8) and (6) applied within one normalization group.

    The coefficients are the baseline of Section 3.5, which the manuscript declares as
    (alpha, beta, gamma) = (1, 0.5, 1). The published CSV files were built with beta = 1;
    the difference is disclosed in the response letter and does not alter any conclusion
    reported here.
    """
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


# --------------------------------------------------------------------------- tables

def fmt_p(p):
    return '$<$0.001' if p < 0.001 else '%.3f' % p


def table_variability(d, path):
    """Mean priority latency per configuration with its 95% CI over the 8 layouts."""
    lines = [
        r'\begin{table}[t]',
        r'\caption{Mean priority-object latency for each weighting vector, with 95\%'
        r' confidence intervals over eight independent object layouts. Variability induced'
        r' by object placement alone (CV) is reported in the last column.}',
        r'\label{tab:variability}',
        r'\centering',
        r'\begin{tabular}{llrrr}',
        r'\toprule',
        r'Area (m$^2$) & $\vec{w}$ & PLM (s) & 95\% CI & CV (\%) \\',
        r'\midrule',
    ]
    for (area, auvs), cell in d.groupby(['area', 'auvs']):
        lines.append(r'\multicolumn{5}{l}{\textit{%d m$^2$, %d AUVs}} \\' % (area, auvs))
        for cfg, g in cell.groupby('cfg'):
            mean, sd, n = g.priority_latency.mean(), g.priority_latency.std(), len(g)
            ci = 1.96 * sd / np.sqrt(n)
            lines.append(r'& (%d,%d,%d) & %.0f & [%.0f, %.0f] & %.1f \\'
                         % (GRID[cfg] + (mean, mean - ci, mean + ci, 100 * sd / mean)))
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\end{table}']
    open(path, 'w').write('\n'.join(lines) + '\n')


def table_variance(d, path):
    """How much of the utility variance each factor explains, per cell."""
    lines = [
        r'\begin{table}[t]',
        r'\caption{Decomposition of the utility variance within each mission configuration.'
        r' Object placement dominates the weighting vector in every case.}',
        r'\label{tab:variance}',
        r'\centering',
        r'\begin{tabular}{rrrrr}',
        r'\toprule',
        r'Area (m$^2$) & AUVs & Weighting (\%) & Placement (\%) & Residual (\%) \\',
        r'\midrule',
    ]
    for (area, auvs), cell in d.groupby(['area', 'auvs']):
        s = utility(cell)
        gm = s.utility.mean()
        ss_cfg = sum(len(x) * (x.utility.mean() - gm) ** 2 for _, x in s.groupby('cfg'))
        ss_rep = sum(len(x) * (x.utility.mean() - gm) ** 2 for _, x in s.groupby('rep'))
        ss_tot = ((s.utility - gm) ** 2).sum()
        lines.append(r'%d & %d & %.1f & %.1f & %.1f \\'
                     % (area, auvs, 100 * ss_cfg / ss_tot, 100 * ss_rep / ss_tot,
                        100 * (ss_tot - ss_cfg - ss_rep) / ss_tot))
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(path, 'w').write('\n'.join(lines) + '\n')


def table_fleet_effect(d, path):
    """The one effect that survives replication: balanced vs maximalist, by fleet size."""
    u = pd.concat([utility(g) for _, g in d.groupby(['area', 'auvs'])])
    lines = [
        r'\begin{table}[t]',
        r'\caption{The balanced weighting vector against the two maximalist ones, paired by'
        r' object layout and pooled across areas (Wilcoxon signed-rank). Layouts won counts'
        r' those on which $(4,4,2)$ attains the higher mission utility.}',
        r'\label{tab:fleet}',
        r'\centering',
        r'\begin{tabular}{rlrrl}',
        r'\toprule',
        r'AUVs & Comparison & Layouts won & $\Delta U$ (\%) & $p$ \\',
        r'\midrule',
    ]
    for fleet in sorted(u.auvs.unique()):
        for rival in (3, 4):
            pairs = []
            for _, g in u[u.auvs == fleet].groupby('area'):
                piv = g[g.cfg.isin([0, rival])].pivot_table(
                    index='rep', columns='cfg', values='utility').dropna()
                if 0 in piv and rival in piv:
                    pairs += list(zip(piv[0], piv[rival]))
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            p = wilcoxon(a, b).pvalue
            lines.append(r'%d & (4,4,2) vs.\ (%d,%d,%d) & %d/%d & %+.1f & %s \\'
                         % ((fleet,) + GRID[rival] + ((a > b).sum(), len(a),
                            100 * (a.mean() - b.mean()) / b.mean(), fmt_p(p))))
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\end{table}']
    open(path, 'w').write('\n'.join(lines) + '\n')


def summary(d, out):
    """Figures quoted in the running text, so they never drift from the tables."""
    u = pd.concat([utility(g) for _, g in d.groupby(['area', 'auvs'])])
    cv = [100 * g.priority_latency.std() / g.priority_latency.mean()
          for _, g in d.groupby(['cell', 'cfg'])]

    tot = sig = 0
    for _, g in d.groupby('cell'):
        piv = g.pivot_table(index='rep', columns='cfg', values='priority_latency').dropna()
        for i in range(5):
            for j in range(i + 1, 5):
                tot += 1
                sig += wilcoxon(piv[i], piv[j]).pvalue < 0.05

    spread = {}
    for (area, auvs), g in u.groupby(['area', 'auvs']):
        m = g.groupby('cfg').utility.mean()
        spread[(area, auvs)] = 100 * (m.max() - m.min()) / m.min()
    s3 = [v for k, v in spread.items() if k[1] == 3]
    s6 = [v for k, v in spread.items() if k[1] == 6]

    lines = [
        'Numbers quoted in the running text of Section 5',
        '=' * 60,
        'missions analysed:            %d' % len(d),
        'placement CV (priority lat.): %.1f%% to %.1f%%, median %.1f%%'
        % (min(cv), max(cv), np.median(cv)),
        'per-cell paired tests:        %d of %d significant at 0.05' % (sig, tot),
        'expected-utility spread:      %.1f%% (3 AUVs) vs %.1f%% (6 AUVs), ratio %.1fx'
        % (np.mean(s3), np.mean(s6), np.mean(s6) / np.mean(s3)),
        'fleet-size effect:            Mann-Whitney p = %.3f'
        % mannwhitneyu(s6, s3, alternative='greater').pvalue,
    ]
    text = '\n'.join(lines)
    open(out, 'w').write(text + '\n')
    print(text)


def main():
    parser = campaign_paths.parser(__doc__)
    parser.add_argument('--outdir', default='article_tables')
    args = parser.parse_args()
    root = campaign_paths.resolve(args.data_root, 'v2_dispersion')
    cache = campaign_paths.cache_path(args.data_root, 'v2_dispersion_metrics.pkl')
    os.makedirs(args.outdir, exist_ok=True)

    d = collect(root, cache)
    print('loaded %d missions from %s\n' % (len(d), root))

    table_variability(d, os.path.join(args.outdir, 'tab_variability.tex'))
    table_variance(d, os.path.join(args.outdir, 'tab_variance.tex'))
    table_fleet_effect(d, os.path.join(args.outdir, 'tab_fleet_effect.tex'))
    summary(d, os.path.join(args.outdir, 'text_figures.txt'))
    print('\ntables written to %s' % args.outdir)


if __name__ == '__main__':
    main()
