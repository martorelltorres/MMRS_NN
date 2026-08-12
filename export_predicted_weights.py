"""Export the weight vectors predicted for the test cells, ready to be simulated.

Fits the adopted Random Forest configuration on the training areas and writes one row per
test cell in the format run_simulations.py consumes with --mode predicted, namely
area,auvs,w1,w2,w3. The runner renormalises the vector to sum one, so the scale written here
is only a matter of readability.

The model is refitted here rather than reused from regressor.py so that the exported file
depends on nothing but the training targets and the configuration reported in Table 1.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, 'weights')
# Table 1: selected by leave-one-out cross-validation over the 24 training samples.
PARAMS = {'n_estimators': 1000, 'max_depth': 2, 'random_state': 0}


def load(pattern):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(WEIGHTS, pattern)))]
    d = pd.concat(frames, ignore_index=True)
    return d.rename(columns={'auv_count': 'auvs'}) if 'auv_count' in d.columns else d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results', 'predicted_weights_rf.csv'))
    args = ap.parse_args()

    train = load('[3-6]AUV_weights.csv')
    test = load('optimal_test_weights.csv')
    print('entrenamiento: %d muestras, %d areas' % (len(train), train.area.nunique()))
    print('celdas de test: %d\n' % len(test))

    X = train[['auvs', 'area']].values.astype(float)
    y = train[['w1', 'w2', 'w3']].values.astype(float)
    scaler = StandardScaler().fit(X)
    model = MultiOutputRegressor(RandomForestRegressor(**PARAMS)).fit(scaler.transform(X), y)

    cells = test[['area', 'auvs']].drop_duplicates().sort_values(['area', 'auvs'])
    pred = model.predict(scaler.transform(cells[['auvs', 'area']].values.astype(float)))
    pred = np.clip(pred, 0.0, None)

    out = cells.copy()
    out[['w1', 'w2', 'w3']] = np.round(pred, 4)
    out.to_csv(args.output, index=False)

    print('  area   AUVs |    w1     w2     w3  | suma | vector normalizado')
    for _, r in out.iterrows():
        s = r.w1 + r.w2 + r.w3
        n = (r.w1 / s, r.w2 / s, r.w3 / s)
        print('  %6d  %d   | %5.2f  %5.2f  %5.2f | %4.2f | (%.3f, %.3f, %.3f)'
              % (r.area, r.auvs, r.w1, r.w2, r.w3, s, n[0], n[1], n[2]))

    distinct = out[['w1', 'w2', 'w3']].round(3).drop_duplicates()
    print('\n  vectores distintos predichos: %d de %d celdas' % (len(distinct), len(out)))
    print('  escrito %s' % args.output)


if __name__ == '__main__':
    main()
