"""Rebuild Table 2 and the APA, APE and AUI metrics from the predicted-weight campaign.

Compares, for every test cell, the utility obtained by simulating the weight vector predicted
by the adopted model against the utilities of the five vectors of the grid simulated during
dataset generation:

    APA = (U_wp - U_Wmin) / (U_Wmax - U_Wmin) * 100      Equation (13)
    APE = (1 - U_wp / U_Wmax) * 100                      Equation (14)
    AUI = (U_wp - U_Wmax) / U_Wmax * 100                 Equation (15)

Both campaigns must share the object layout of each cell, otherwise the comparison mixes the
effect of the weighting vector with the effect of object placement. This is checked and the
script refuses to report a cell where the scenarios differ.

Writes the LaTeX table and a CSV with every intermediate quantity.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import rosbag

import campaign_paths

ALPHA, BETA, GAMMA = 1.0, 0.5, 1.0          # Section 3.5 baseline


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
    if m['travelled_distance'] <= 0:
        raise ValueError('ATD is zero: the ASV distance was not recorded')
    return m


def collect(root):
    rows = []
    for man in sorted(glob.glob(os.path.join(root, '*/*/owa/manifest.csv'))):
        folder = os.path.dirname(man)
        cell = folder.replace('/owa', '').split(root.rstrip('/') + '/')[1]
        for _, r in pd.read_csv(man).iterrows():
            if r.outcome != 'completed':
                continue
            path = os.path.join(folder, 'bagfiles', str(r['bagfile']))
            if not os.path.exists(path):
                continue
            m = metrics(path)
            m.update(area=int(cell.split('/')[0]), auvs=int(cell.split('/')[1][0]),
                     scenario=str(r['scenario']),
                     w=(r.get('w1'), r.get('w2'), r.get('w3')))
            rows.append(m)
    return pd.DataFrame(rows)


def utility(group):
    """Equations (7), (8) and (6) within one mission configuration."""
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


def main():
    ap = campaign_paths.parser(__doc__)
    ap.add_argument('--outdir', default='article_tables')
    args = ap.parse_args()
    SWEEP = campaign_paths.resolve(args.data_root, 'v2_sweep')
    PREDICTED = campaign_paths.resolve(args.data_root, 'v4_predicted')
    os.makedirs(args.outdir, exist_ok=True)

    sweep, pred = collect(SWEEP), collect(PREDICTED)
    print('barrido: %d misiones | pesos predichos: %d misiones\n' % (len(sweep), len(pred)))
    if pred.empty:
        raise SystemExit('no hay misiones con pesos predichos todavia')

    rows = []
    for (area, auvs), p in pred.groupby(['area', 'auvs']):
        s = sweep[(sweep.area == area) & (sweep.auvs == auvs)]
        if len(s) < 5:
            print('  aviso: %d/%dAUV tiene %d misiones de barrido, se omite'
                  % (area, auvs, len(s)))
            continue
        if set(s.scenario) != set(p.scenario):
            print('  aviso: %d/%dAUV usa escenarios distintos (%s vs %s), se omite'
                  % (area, auvs, sorted(set(s.scenario)), sorted(set(p.scenario))))
            continue
        # Normalise both campaigns together: U_wp must sit on the same scale as the grid.
        joint = utility(pd.concat([s, p], ignore_index=True))
        u_grid = joint.iloc[:len(s)].utility
        u_pred = joint.iloc[len(s):].utility.iloc[0]
        lo, hi = u_grid.min(), u_grid.max()
        rows.append({'area': area, 'auvs': auvs, 'U_min': lo, 'U_max': hi, 'U_wp': u_pred,
                     'APA': 100 * (u_pred - lo) / (hi - lo) if hi > lo else np.nan,
                     'APE': 100 * (1 - u_pred / hi),
                     'AUI': 100 * (u_pred - hi) / hi})
    d = pd.DataFrame(rows).sort_values(['area', 'auvs'])
    d.to_csv(os.path.join(args.outdir, 'prediction_metrics.csv'), index=False)

    print(d.to_string(index=False, float_format=lambda x: '%.3f' % x))
    print('\n  APA media %.1f%%   APE media %.1f%%   AUI media %+.1f%%'
          % (d.APA.mean(), d.APE.mean(), d.AUI.mean()))
    print('  celdas donde U_wp supera U_Wmax: %d de %d' % ((d.AUI > 0).sum(), len(d)))

    lines = [r'\begin{table}[]', r'\centering',
             r'\caption{Utility obtained with the weight configurations of $\mathcal{W}$ and'
             r' with the predicted weights ($w_{\text{p}}$), for each combination of test area'
             r' and fleet size.}',
             r'\label{tab:utility_comp}', r'\begin{tabular}{c c c c c}', r'\toprule',
             r'Area (a) & AUVs (n) & $U_{\mathcal{W}min}$ & $U_{\mathcal{W}max}$ & $U_{w_{p}}$ \\',
             r'\midrule']
    for area, g in d.groupby('area'):
        lines.append(r'\multirow{%d}{*}{$%dm^2$}' % (len(g), area))
        for _, r in g.iterrows():
            lines.append(r'   & %d  & %.3f & %.3f & %.3f \\' % (r.auvs, r.U_min, r.U_max, r.U_wp))
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\end{table}']
    path = os.path.join(args.outdir, 'tab_utility_comp.tex')
    open(path, 'w').write('\n'.join(lines) + '\n')
    print('  tabla escrita en %s' % path)


if __name__ == '__main__':
    main()
