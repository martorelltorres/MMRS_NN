"""Does regressing the utility surface beat regressing the winning weight vector?

The published formulation learns a mapping (area, fleet) -> w_max, where w_max is the
argmax over the five simulated weighting vectors of a cell. That target has two defects: it
collapses five missions into one label, leaving 24 training samples out of 120 missions, and
the argmax is unstable, since the replication campaign shows the differences between vectors
are mostly placement noise.

The alternative regresses U(area, fleet, w) over every mission and takes the argmax of the
predicted surface at prediction time. It uses five times more training data and a continuous,
directly measured target instead of a discrete unstable one.

Both formulations are evaluated on the same held-out areas and judged by what actually
matters: the utility the operator obtains by deploying the recommended vector.
"""
import glob
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Campaign extractions; override the location with $MRS_DATA.
_DATA = os.environ.get('MRS_DATA', os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'campaigns'))
TRAIN = os.path.join(_DATA, 'v2_sweep', 'train_data.csv')
TEST = os.path.join(_DATA, 'v2_sweep', 'test_data.csv')

SURFACE_GRID = {'svr__C': [0.1, 1, 10, 100],
                'svr__epsilon': [0.01, 0.05, 0.1, 0.3],
                'svr__gamma': ['scale', 0.1, 1.0],
                'svr__kernel': ['rbf', 'poly'],
                'svr__degree': [2, 3]}


def cells(frame):
    """(area, fleet) -> table of the five weighting vectors with their utility."""
    out = {}
    for key, g in frame.groupby(['area', 'auv_count']):
        out[key] = g[['w1', 'w2', 'w3', 'utility']].reset_index(drop=True)
        # Weight vectors are stored on the 0-1 scale in some campaigns and 0-10 in others.
        if out[key][['w1', 'w2', 'w3']].values.sum(axis=1).mean() < 5:
            out[key][['w1', 'w2', 'w3']] *= 10
    return out


def regret(chosen, table):
    """Utility given up by deploying `chosen` instead of the best vector of the cell."""
    best = table.utility.max()
    row = table.iloc[np.argmin(np.abs(table[['w1', 'w2', 'w3']].values - chosen).sum(axis=1))]
    return 100 * (best - row.utility) / best, row.utility


def main():
    tr, te = pd.read_csv(TRAIN), pd.read_csv(TEST)
    tr_cells, te_cells = cells(tr), cells(te)
    print('train: %d misiones en %d celdas | test: %d misiones en %d celdas\n'
          % (len(tr), len(tr_cells), len(te), len(te_cells)))

    # ---- formulation A: predict w_max directly (what the paper does) -------------
    Xa = np.array([[k[1], k[0]] for k in sorted(tr_cells)], dtype=float)
    ya = np.array([tr_cells[k].loc[tr_cells[k].utility.idxmax(), ['w1', 'w2', 'w3']].values
                   for k in sorted(tr_cells)], dtype=float)
    sa = StandardScaler().fit(Xa)
    model_a = MultiOutputRegressor(SVR(kernel='poly', degree=3, C=0.1, epsilon=0.01,
                                       gamma=1.0, max_iter=200000)).fit(sa.transform(Xa), ya)

    # ---- formulation B: regress the utility surface, then take its argmax ---------
    rows = []
    for (area, auvs), t in tr_cells.items():
        for _, r in t.iterrows():
            rows.append([auvs, area, r.w1, r.w2, r.w3, r.utility])
    B = np.array(rows, dtype=float)
    pipe = make_pipeline(StandardScaler(), SVR(max_iter=200000))
    pipe.steps[1] = ('svr', pipe.steps[1][1])
    search = GridSearchCV(pipe, SURFACE_GRID, cv=LeaveOneOut(),
                          scoring='neg_mean_absolute_error', n_jobs=-1)
    search.fit(B[:, :5], B[:, 5])
    model_b = search.best_estimator_
    print('superficie de utilidad: %s   LOO MAE %.4f\n'
          % ({k.replace('svr__', ''): v for k, v in search.best_params_.items()},
             -search.best_score_))

    grid = sorted({tuple(r) for t in tr_cells.values()
                   for r in t[['w1', 'w2', 'w3']].values.tolist()})

    print('  celda        | A: predice w_max        | B: argmax de la superficie')
    print('  ' + '-' * 74)
    reg_a, reg_b, reg_0, hit_a, hit_b = [], [], [], 0, 0
    for key in sorted(te_cells):
        area, auvs = key
        t = te_cells[key]
        best = t.loc[t.utility.idxmax(), ['w1', 'w2', 'w3']].values

        pa = model_a.predict(sa.transform([[auvs, area]]))[0]
        ra, _ = regret(pa, t)

        scores = [model_b.predict([[auvs, area, w[0], w[1], w[2]]])[0] for w in grid]
        pb = np.array(grid[int(np.argmax(scores))], dtype=float)
        rb, _ = regret(pb, t)

        # reference: always deploy the balanced vector, no learning at all
        r0, _ = regret(np.array([4., 4., 2.]), t)

        reg_a.append(ra); reg_b.append(rb); reg_0.append(r0)
        hit_a += np.allclose(np.round(pa), best, atol=1.5)
        hit_b += np.allclose(pb, best)
        print('  %5d/%dAUV  | (%4.1f,%4.1f,%4.1f) regret %5.1f%% | (%2.0f,%2.0f,%2.0f) regret %5.1f%%'
              % (area, auvs, pa[0], pa[1], pa[2], ra, pb[0], pb[1], pb[2], rb))

    print('\n  regret medio   A (w_max directo) : %5.2f%%' % np.mean(reg_a))
    print('  regret medio   B (superficie)    : %5.2f%%' % np.mean(reg_b))
    print('  regret medio   fijar (4,4,2)     : %5.2f%%' % np.mean(reg_0))
    print('  aciertos exactos del argmax      : A %d/%d, B %d/%d'
          % (hit_a, len(te_cells), hit_b, len(te_cells)))

    from scipy.stats import wilcoxon
    print('\n  B vs A            p=%.4f' % wilcoxon(reg_b, reg_a).pvalue)
    print('  B vs fijar (4,4,2) p=%.4f' % wilcoxon(reg_b, reg_0).pvalue)


if __name__ == '__main__':
    main()
