"""Recompute every quantity quoted in the revised manuscript and check it against the text.

Each check states the campaign it reads, the formula applied and the value the manuscript
claims, then recomputes it from the bagfiles and compares. The purpose is that no number in
Sections 5 and 6, the abstract or the highlights is present without a reproducible derivation,
and that regenerating the tables after any change to the data immediately reveals any figure in
the running text that has drifted.

    python3 verify_article_numbers.py

Exit status is non-zero if any check fails.
"""
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

GRID = [(4, 4, 2), (6, 2, 2), (6, 4, 0), (8, 2, 0), (10, 0, 0)]
import campaign_paths


# Section 3.5 baseline. The manuscript declares (1, 0.5, 1) in Sections 3.4 and 3.5.
ALPHA, BETA, GAMMA = 1.0, 0.5, 1.0

RESULTS = []


def check(section, quantity, claimed, actual, formula, source, tol=0.05):
    """Compare a manuscript figure against its recomputation."""
    if isinstance(claimed, (str, bool)) or claimed is None:
        ok = bool(claimed == actual)
    else:
        ok = abs(claimed - actual) <= tol
    RESULTS.append((ok, section, quantity, claimed, actual, formula, source))
    return ok


# --------------------------------------------------------------------------- data

def raw_metrics(path):
    import rosbag
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


def load(root, policy, cache, dispersion):
    rows = []
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
            idx = int(r['index'])
            m.update(cell=cell, area=int(cell.split('/')[0]), auvs=int(cell.split('/')[1][0]),
                     policy=policy, scenario=str(r['scenario']),
                     cfg=(idx // 8 if policy == 'owa' and root == dispersion else -1),
                     rep=(idx % 8 if policy == 'owa' and root == dispersion else idx))
            rows.append(m)
    return pd.DataFrame(rows)


def utility(group):
    """Equations (7), (8) and (6).

    Min-max normalisation of the five latency/distance metrics and max normalisation of the
    two object counts, applied within the normalisation group, then

        U = alpha*TP_n/(1+exp(PLM_n+PSD_n)) + beta*TR_n/(1+exp(RLM_n+RSD_n)) + gamma*exp(-ATD_n)
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
    g['utility'] = ALPHA * pri + BETA * reg + GAMMA * np.exp(-g['travelled_distance_n'])
    return g


def main():
    ap = campaign_paths.parser(__doc__)
    ap.add_argument('--models', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'),
        help='directory holding the regression results of this repository')
    args = ap.parse_args()
    DISPERSION = campaign_paths.resolve(args.data_root, 'v2_dispersion')
    ROUNDROBIN = campaign_paths.resolve(args.data_root, 'v3_roundrobin')
    SWEEP = campaign_paths.resolve(args.data_root, 'v2_sweep')
    PREDICTED = campaign_paths.resolve(args.data_root, 'v4_predicted')
    CACHE = campaign_paths.cache_path(args.data_root, 'roundrobin_metrics.pkl')
    # The predicted weights are an output of this repository, written by
    # export_predicted_weights.py, not part of the published campaign data.
    WEIGHTS = os.path.join(args.models, 'predicted_weights_rf.csv')
    cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}
    disp = load(DISPERSION, 'owa', cache, DISPERSION)
    rr = load(ROUNDROBIN, 'roundrobin', cache, DISPERSION)
    pickle.dump(cache, open(CACHE, 'wb'))

    S = 'campaign sizes'
    check(S, 'replication missions', 240, len(disp),
          'count of completed, warning-free missions', 'v2_dispersion manifests', tol=0)
    check(S, 'baseline missions', 48, len(rr),
          'count of completed, warning-free missions', 'v3_roundrobin manifests', tol=0)

    # ----------------------------------------------------------------- Section 5.2
    S = '5.2 placement variability'
    cv = [100 * g.priority_latency.std(ddof=1) / g.priority_latency.mean()
          for _, g in disp.groupby(['cell', 'cfg'])]
    check(S, 'min CV of PLM (%)', 16.3, min(cv),
          'sd/mean of priority_latency over the 8 layouts of a configuration', 'v2_dispersion')
    check(S, 'max CV of PLM (%)', 52.7, max(cv),
          'sd/mean of priority_latency over the 8 layouts of a configuration', 'v2_dispersion')
    check(S, 'median CV of PLM (%)', 24.4, float(np.median(cv)),
          'median over the 30 configurations', 'v2_dispersion')

    # Variance decomposition: one-way sums of squares of the cell utilities.
    shares = []
    for _, cell in disp.groupby(['area', 'auvs']):
        s = utility(cell)
        gm = s.utility.mean()
        ss_cfg = sum(len(x) * (x.utility.mean() - gm) ** 2 for _, x in s.groupby('cfg'))
        ss_rep = sum(len(x) * (x.utility.mean() - gm) ** 2 for _, x in s.groupby('rep'))
        ss_tot = ((s.utility - gm) ** 2).sum()
        shares.append((100 * ss_cfg / ss_tot, 100 * ss_rep / ss_tot))
    w = [x[0] for x in shares]
    p = [x[1] for x in shares]
    check(S, 'min placement share (%)', 39.6, min(p),
          'SS_between(layout) / SS_total of the cell utilities', 'v2_dispersion')
    check(S, 'max placement share (%)', 81.4, max(p),
          'SS_between(layout) / SS_total of the cell utilities', 'v2_dispersion')
    check(S, 'min weighting share (%)', 2.6, min(w),
          'SS_between(weighting) / SS_total of the cell utilities', 'v2_dispersion')
    check(S, 'max weighting share (%)', 15.2, max(w),
          'SS_between(weighting) / SS_total of the cell utilities', 'v2_dispersion')
    check(S, 'placement > weighting in every cell', True,
          all(pp > ww for ww, pp in shares), 'per-cell comparison', 'v2_dispersion', tol=0)

    tot = sig = 0
    for _, g in disp.groupby('cell'):
        piv = g.pivot_table(index='rep', columns='cfg', values='priority_latency').dropna()
        for i in range(5):
            for j in range(i + 1, 5):
                tot += 1
                sig += wilcoxon(piv[i], piv[j]).pvalue < 0.05
    check(S, 'paired tests performed', 60, tot, 'C(5,2) pairs x 6 cells', 'v2_dispersion', tol=0)
    check(S, 'significant at 0.05', 1, sig,
          'Wilcoxon signed-rank on PLM, blocked by layout', 'v2_dispersion', tol=0)

    # ----------------------------------------------------------------- APA stability
    S = '5.2 APA stability'
    rows = []
    for (area, auvs), cell in disp.groupby(['area', 'auvs']):
        for rep, layout in cell.groupby('rep'):
            if len(layout) < 5:
                continue
            u = utility(layout)
            lo, hi = u.utility.min(), u.utility.max()
            if hi <= lo:
                continue
            for _, r in u.iterrows():
                rows.append({'area': area, 'auvs': auvs, 'cfg': int(r.cfg),
                             'APA': 100 * (r.utility - lo) / (hi - lo)})
    a = pd.DataFrame(rows)
    wd = a.groupby(['area', 'auvs', 'cfg']).APA.agg(['min', 'max'])
    check(S, 'mean APA interval width (pp)', 92, float((wd['max'] - wd['min']).mean()),
          'Eq.(13) recomputed per layout; width = max-min over the 8 layouts', 'v2_dispersion',
          tol=0.5)
    check(S, 'configurations spanning full scale', 21,
          int(((wd['min'] < 5) & (wd['max'] > 95)).sum()),
          'count of (cell, weighting) with min<5 and max>95', 'v2_dispersion', tol=0)
    check(S, 'configurations examined', 30, len(wd), '6 cells x 5 weighting vectors',
          'v2_dispersion', tol=0)

    # ----------------------------------------------------------------- Section 5.3
    S = '5.3 baseline comparison'
    u = pd.concat([utility(g) for _, g in pd.concat([disp, rr]).groupby('cell')])

    def pairs(cell=None, fleet=None):
        sub = u if cell is None else u[u.cell == cell]
        sub = sub if fleet is None else sub[sub.auvs == fleet]
        A, B = [], []
        for _, g in sub.groupby('cell'):
            x = g[g.policy == 'roundrobin'].set_index('rep').utility
            y = g[(g.policy == 'owa') & (g.cfg == 0)].set_index('rep').utility
            for k in sorted(set(x.index) & set(y.index)):
                A.append(y[k])
                B.append(x[k])
        return np.array(A), np.array(B)

    F = 'utility Eq.(6) normalised per cell; Wilcoxon signed-rank paired by layout'
    A, B = pairs(fleet=6)
    check(S, '6 AUVs dU (%)', 27.1, 100 * (A.mean() - B.mean()) / B.mean(), F, 'both campaigns')
    check(S, '6 AUVs wins', 22, int((A > B).sum()), F, 'both campaigns', tol=0)
    check(S, '6 AUVs blocks', 24, len(A), F, 'both campaigns', tol=0)
    check(S, '6 AUVs p<0.001', True, wilcoxon(A, B).pvalue < 0.001, F, 'both campaigns', tol=0)
    A, B = pairs(fleet=3)
    check(S, '3 AUVs dU (%)', 0.9, 100 * (A.mean() - B.mean()) / B.mean(), F, 'both campaigns')
    check(S, '3 AUVs wins', 12, int((A > B).sum()), F, 'both campaigns', tol=0)
    check(S, '3 AUVs p', 0.768, wilcoxon(A, B).pvalue, F, 'both campaigns', tol=0.001)
    for cell, claim in (('15000/6AUVs', 17.9), ('35000/6AUVs', 22.8), ('55000/6AUVs', 41.5),
                        ('15000/3AUVs', -9.7)):
        A, B = pairs(cell=cell)
        check(S, '%s dU (%%)' % cell, claim, 100 * (A.mean() - B.mean()) / B.mean(), F, 'both')
    A, B = pairs(cell='15000/3AUVs')
    check(S, '15000/3AUVs wins', 0, int((A > B).sum()), F, 'both campaigns', tol=0)
    check(S, '15000/3AUVs p', 0.008, wilcoxon(A, B).pvalue, F, 'both campaigns', tol=0.001)
    A, B = pairs(cell='55000/6AUVs')
    check(S, '55000/6AUVs wins', 8, int((A > B).sum()), F, 'both campaigns', tol=0)

    # ----------------------------------------------------------------- fleet effect
    S = '5.3 weighting within the grid'
    ud = pd.concat([utility(g) for _, g in disp.groupby(['area', 'auvs'])])
    F = 'utility Eq.(6); Wilcoxon paired by layout, pooled over the 3 areas'
    for fleet, rival, dclaim, pclaim in ((6, 3, 14.5, 0.001), (6, 4, 13.9, 0.002)):
        P = []
        for _, g in ud[ud.auvs == fleet].groupby('area'):
            piv = g[g.cfg.isin([0, rival])].pivot_table(
                index='rep', columns='cfg', values='utility').dropna()
            P += list(zip(piv[0], piv[rival]))
        A = np.array([x[0] for x in P])
        B = np.array([x[1] for x in P])
        check(S, '%dAUV (4,4,2) vs %s dU (%%)' % (fleet, GRID[rival]), dclaim,
              100 * (A.mean() - B.mean()) / B.mean(), F, 'v2_dispersion')
        pv = wilcoxon(A, B).pvalue
        check(S, '%dAUV (4,4,2) vs %s p' % (fleet, GRID[rival]), True,
              pv < 0.0015 if pclaim == 0.001 else abs(pv - pclaim) < 0.0006, F,
              'v2_dispersion', tol=0)

    spread = {}
    for (area, auvs), g in ud.groupby(['area', 'auvs']):
        m = g.groupby('cfg').utility.mean()
        spread[(area, auvs)] = 100 * (m.max() - m.min()) / m.min()
    s3 = [v for k, v in spread.items() if k[1] == 3]
    s6 = [v for k, v in spread.items() if k[1] == 6]
    F = '(max-min)/min of the per-weighting mean utility, averaged over the 3 areas'
    check(S, 'spread 3 AUVs (%)', 7.2, float(np.mean(s3)), F, 'v2_dispersion')
    check(S, 'spread 6 AUVs (%)', 15.0, float(np.mean(s6)), F, 'v2_dispersion')
    check(S, 'spread ratio', 2.1, float(np.mean(s6) / np.mean(s3)), F, 'v2_dispersion')
    check(S, 'fleet-size Mann-Whitney p', 0.050,
          mannwhitneyu(s6, s3, alternative='greater').pvalue,
          'one-sided Mann-Whitney on the 3 vs 3 area spreads', 'v2_dispersion', tol=0.001)

    # ----------------------------------------------------------------- Section 5.1
    S = '5.1 predicted weights'
    sweep = load(SWEEP, 'owa', cache, DISPERSION)
    pred = load(PREDICTED, 'owa', cache, DISPERSION)
    pickle.dump(cache, open(CACHE, 'wb'))
    check(S, 'sweep missions', 220, len(sweep), 'completed missions', 'v2_sweep', tol=0)
    check(S, 'predicted missions', 20, len(pred), 'completed missions', 'v4_predicted', tol=0)

    rows = []
    for (area, auvs), pp in pred.groupby(['area', 'auvs']):
        ss = sweep[(sweep.area == area) & (sweep.auvs == auvs)]
        if len(ss) < 5 or set(ss.scenario) != set(pp.scenario):
            continue
        j = utility(pd.concat([ss, pp], ignore_index=True))
        ug = j.iloc[:len(ss)].utility
        up = j.iloc[len(ss):].utility.iloc[0]
        lo, hi = ug.min(), ug.max()
        rows.append({'area': area, 'auvs': auvs,
                     'APA': 100 * (up - lo) / (hi - lo), 'APE': 100 * (1 - up / hi),
                     'AUI': 100 * (up - hi) / hi})
    d = pd.DataFrame(rows)
    F = 'Eqs.(13),(14),(15); U from Eq.(6) normalised over the 5 grid runs plus the predicted one'
    check(S, 'test cells', 20, len(d), 'area x fleet combinations of the test set', 'both', tol=0)
    check(S, 'mean APE (%)', 0.8, d.APE.mean(), F, 'v2_sweep + v4_predicted')
    check(S, 'mean AUI (%)', -0.8, d.AUI.mean(), F, 'v2_sweep + v4_predicted')
    check(S, 'min APA (%)', -9.6, d.APA.min(), F, 'v2_sweep + v4_predicted', tol=0.1)
    check(S, 'max APA (%)', 355.4, d.APA.max(), F, 'v2_sweep + v4_predicted', tol=0.1)
    check(S, 'cells with U_wp > U_Wmax', 8, int((d.AUI > 0).sum()),
          'count of cells with positive AUI', 'v2_sweep + v4_predicted', tol=0)

    # The alternative normalisation scope discussed at the end of Section 5.1: the scale is
    # fixed by the five grid missions and the predicted one is projected onto it, so a value
    # outside the grid range normalises outside [0, 1].
    S = '5.1 alternative normalisation'
    cols = ['regular_latency', 'priority_latency', 'regular_std', 'priority_std',
            'travelled_distance']
    alt = []
    for (area, auvs), pp in pred.groupby(['area', 'auvs']):
        ss = sweep[(sweep.area == area) & (sweep.auvs == auvs)]
        if len(ss) < 5:
            continue
        lohi = {c: (ss[c].min(), ss[c].max()) for c in cols}
        hic = {c: ss[c].max() for c in ['priority_objects', 'regular_objects']}

        def u(r):
            n = {c: (0.0 if lohi[c][1] == lohi[c][0]
                     else (r[c] - lohi[c][0]) / (lohi[c][1] - lohi[c][0])) for c in cols}
            o = {c: (0.0 if hic[c] == 0 else r[c] / hic[c]) for c in hic}
            return (ALPHA * o['priority_objects']
                    / (1 + np.exp(n['priority_latency'] + n['priority_std']))
                    + BETA * o['regular_objects']
                    / (1 + np.exp(n['regular_latency'] + n['regular_std']))
                    + GAMMA * np.exp(-n['travelled_distance']))

        ug = ss.apply(u, axis=1)
        up = u(pp.iloc[0])
        alt.append(100 * (up - ug.min()) / (ug.max() - ug.min()))
    alt = np.array(alt)
    F = 'Eq.(13) with the scale fixed by the 5 grid missions, prediction projected onto it'
    check(S, 'mean APA, grid-only scale (%)', 253, float(alt.mean()), F, 'v2_sweep + v4_predicted',
          tol=0.5)
    check(S, 'max APA, grid-only scale (%)', 2129, float(alt.max()), F, 'v2_sweep + v4_predicted',
          tol=1)
    check(S, 'cells outside [0,100], grid-only', 10, int(((alt < 0) | (alt > 100)).sum()), F,
          'v2_sweep + v4_predicted', tol=0)
    check(S, 'cells outside [0,100], joint group', 9,
          int(((d.APA < 0) | (d.APA > 100)).sum()),
          'Eq.(13) with the 6-mission joint group used in Table 4', 'v2_sweep + v4_predicted',
          tol=0)

    # The worked example of Section 5.1.
    S = '5.1 worked example 35000/6AUVs'
    ss = sweep[(sweep.area == 35000) & (sweep.auvs == 6)]
    pp = pred[(pred.area == 35000) & (pred.auvs == 6)]
    both = pd.concat([ss, pp])
    check(S, 'priority objects identical', True, both.priority_objects.nunique() == 1,
          'all 6 missions of the cell', 'v2_sweep + v4_predicted', tol=0)
    check(S, 'priority objects value', 22, int(both.priority_objects.iloc[0]),
          'transmitted_priority_objects', 'v2_sweep + v4_predicted', tol=0)
    check(S, 'regular objects value', 29, int(both.regular_objects.iloc[0]),
          'transmitted_regular_objects', 'v2_sweep + v4_predicted', tol=0)
    check(S, 'min PLM (s)', 142, both.priority_latency.min(), 'mean priority latency', 'both',
          tol=0.5)
    check(S, 'max PLM (s)', 161, both.priority_latency.max(), 'mean priority latency', 'both',
          tol=0.5)
    check(S, 'min ATD (m)', 1578, both.travelled_distance.min(), 'ASV travelled distance',
          'both', tol=1)
    check(S, 'max ATD (m)', 1672, both.travelled_distance.max(), 'ASV travelled distance',
          'both', tol=1)
    check(S, 'predicted run holds min ATD', True,
          pp.travelled_distance.iloc[0] == both.travelled_distance.min(),
          'comparison against the 5 grid runs', 'both', tol=0)

    # ----------------------------------------------------------------- weight ranges
    S = '5.1 predicted weight ranges'
    w = pd.read_csv(WEIGHTS)
    for col, lo, hi in (('w1', 5.7, 7.4), ('w2', 2.0, 3.3), ('w3', 0.3, 1.5)):
        check(S, '%s min' % col, lo, w[col].min(), 'predicted_weights_rf.csv', 'RF model', tol=0.05)
        check(S, '%s max' % col, hi, w[col].max(), 'predicted_weights_rf.csv', 'RF model', tol=0.05)

    # ----------------------------------------------------------------- supervised set
    S = '5.1 supervised dataset'
    tr = pd.read_csv(os.path.join(SWEEP, 'train_targets.csv'))
    te = pd.read_csv(os.path.join(SWEEP, 'test_targets.csv'))
    check(S, 'training samples', 24, len(tr), '6 areas x 4 fleet sizes', 'v2_sweep', tol=0)
    check(S, 'test samples', 20, len(te), '5 areas x 4 fleet sizes', 'v2_sweep', tol=0)
    check(S, 'one realization per target', True, bool((tr.n_realizations == 1).all()),
          'the target is the argmax over the grid on a single layout', 'v2_sweep', tol=0)
    for col, lo, hi in (('w1', 4, 10), ('w2', 0, 4), ('w3', 0, 2)):
        check(S, 'target %s min' % col, lo, tr[col].min(),
              'argmax weighting vector per training cell', 'v2_sweep', tol=0)
        check(S, 'target %s max' % col, hi, tr[col].max(),
              'argmax weighting vector per training cell', 'v2_sweep', tol=0)

    # ----------------------------------------------------------------- Section 4
    # The regression comparison is produced by the retraining repository, not by the campaigns,
    # so it is read from its result files. Summary figures are pooled over the 20 test
    # configurations, which is the set the significance tests use.
    S = '4 model comparison'
    NN = args.models
    if os.path.isdir(NN):
        per = pd.read_csv(os.path.join(NN, 'owa_model_test_metrics.csv'))
        prd = pd.read_csv(os.path.join(NN, 'owa_model_predictions_on_test.csv'))
        rf = per[per.Model == 'Random Forest']
        check(S, 'RF MAE band low', 1.20, rf.MAE.min(), 'per-fleet MAE of Random Forest',
              'owa_model_test_metrics.csv', tol=0.005)
        check(S, 'RF MAE band high', 1.64, rf.MAE.max(), 'per-fleet MAE of Random Forest',
              'owa_model_test_metrics.csv', tol=0.005)
        check(S, 'RF RMSE band low', 1.44, rf.RMSE.min(), 'per-fleet RMSE of Random Forest',
              'owa_model_test_metrics.csv', tol=0.005)
        check(S, 'RF RMSE band high', 1.83, rf.RMSE.max(), 'per-fleet RMSE of Random Forest',
              'owa_model_test_metrics.csv', tol=0.005)
        six = per[per['AUV Count'] == 6]
        check(S, 'SVR RMSE at 6 AUVs', 2.10,
              float(six[six.Model == 'SVR'].RMSE.iloc[0]), 'per-fleet RMSE',
              'owa_model_test_metrics.csv', tol=0.005)
        check(S, 'SVR is worst RMSE at 6 AUVs', True,
              six.sort_values('RMSE').Model.iloc[-1] == 'SVR', 'per-fleet RMSE ranking',
              'owa_model_test_metrics.csv', tol=0)

        # RMSE follows the definition used to build the per-fleet figures: sklearn's
        # mean_squared_error with squared=False on a multi-output target averages the
        # per-component RMSE. Verified to reproduce all 24 cells of the metrics file.
        def rmse_sk(g):
            return float(np.mean([np.sqrt(((g['true_w%d' % i] - g['pred_w%d' % i]) ** 2).mean())
                                  for i in (1, 2, 3)]))

        recomputed = all(
            abs(per[(per.Model == mod) & (per['AUV Count'] == n)].RMSE.iloc[0] - rmse_sk(g)) < 1e-6
            for (mod, n), g in prd.groupby(['model', 'auv_count']))
        check(S, 'RMSE definition reproduces the metrics file', True, recomputed,
              'mean over the 3 components of the per-component RMSE', 'MMRS_NN/results', tol=0)

        agg = []
        for mod, g in prd.groupby('model'):
            e = np.stack([np.abs(g['true_w%d' % i] - g['pred_w%d' % i]).values for i in (1, 2, 3)])
            agg.append({'model': mod, 'MAE': e.mean(), 'RMSE': rmse_sk(g)})
        agg = pd.DataFrame(agg)
        models = agg[agg.model != 'Mean baseline'].copy()
        F = 'the 20 test configurations, aggregated as in the per-fleet figures'
        check(S, 'lowest MAE is SVR', True,
              models.sort_values('MAE').model.iloc[0] == 'SVR', F, 'MMRS_NN/results', tol=0)
        check(S, 'lowest RMSE is Lasso', True,
              models.sort_values('RMSE').model.iloc[0] == 'Lasso', F, 'MMRS_NN/results', tol=0)
        check(S, 'SVR RMSE rank of five', 4,
              int(models.RMSE.rank().loc[models.model == 'SVR'].iloc[0]), F, 'MMRS_NN', tol=0)
        check(S, 'RF MAE rank of five', 2,
              int(models.MAE.rank().loc[models.model == 'Random Forest'].iloc[0]), F, 'MMRS_NN',
              tol=0)
        check(S, 'RF RMSE rank of five', 2,
              int(models.RMSE.rank().loc[models.model == 'Random Forest'].iloc[0]), F, 'MMRS_NN',
              tol=0)
        worst2 = set()
        for (n, crit) in [(n, c) for n in per['AUV Count'].unique() for c in ('MAE', 'RMSE')]:
            sub = per[per['AUV Count'] == n].sort_values(crit)
            worst2 |= set(sub.Model.iloc[-2:])
        check(S, 'RF never among the two worst', True, 'Random Forest' not in worst2,
              'per fleet size and per criterion', 'MMRS_NN/results', tol=0)
        check(S, 'Lasso reproduces the baseline', True,
              bool(np.allclose(
                  prd[prd.model == 'Lasso'][['pred_w1', 'pred_w2', 'pred_w3']].values,
                  prd[prd.model == 'Mean baseline'][['pred_w1', 'pred_w2', 'pred_w3']].values)),
              'element-wise comparison of the two prediction sets', 'MMRS_NN/results', tol=0)

        # The significance tests pair the 20 test configurations, each summarised by the mean
        # absolute error over its three weight components.
        def per_config(mod):
            g = prd[prd.model == mod].sort_values(['auv_count', 'area'])
            return np.stack([np.abs(g['true_w%d' % i] - g['pred_w%d' % i]).values
                             for i in (1, 2, 3)]).mean(axis=0)

        F = 'Wilcoxon signed-rank on the 20 paired per-configuration mean absolute errors'
        check(S, 'RF vs SVR p', 0.90,
              wilcoxon(per_config('Random Forest'), per_config('SVR')).pvalue, F, 'MMRS_NN',
              tol=0.005)
        check(S, 'RF vs constant p', 0.39,
              wilcoxon(per_config('Random Forest'), per_config('Mean baseline')).pvalue, F,
              'MMRS_NN', tol=0.005)

    # ----------------------------------------------------------------- report
    bad = [r for r in RESULTS if not r[0]]
    print('%-34s %-30s %10s %10s' % ('SECTION', 'QUANTITY', 'CLAIMED', 'RECOMPUTED'))
    print('-' * 92)
    last = None
    for ok, sec, q, c, act, formula, src in RESULTS:
        if sec != last:
            print('\n[%s]' % sec)
            last = sec
        cs = c if isinstance(c, (str, bool)) else '%.3f' % c
        as_ = act if isinstance(act, (str, bool, np.bool_)) else '%.3f' % act
        print('  %s %-30s %10s %10s   %s' % ('ok ' if ok else 'FAIL', q, cs, as_,
                                             '' if ok else '<-- MISMATCH'))
    print('\n%d checks, %d passed, %d failed' % (len(RESULTS), len(RESULTS) - len(bad), len(bad)))
    if bad:
        print('\nFORMULAS AND SOURCES OF THE FAILING CHECKS')
        for ok, sec, q, c, act, formula, src in bad:
            print('  %s / %s\n    formula: %s\n    source:  %s' % (sec, q, formula, src))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
