"""Draw the aggregate model comparison of Section 4.1.

Figures~\\ref{fig:MAE} and~\\ref{fig:RMSE} report both error criteria per fleet size, which is
what shows the alternation between techniques, but it leaves the aggregate ordering to be
reconstructed by eye from eight bars. The statements the section makes are about that
aggregate: which technique attains the lowest mean absolute error, which the lowest root mean
squared error, and why the two do not agree.

The bars carry a zero baseline, since their length is what encodes the magnitude, and the
values are printed at their ends because the differences that matter here are a few hundredths
and the eye should not be asked to resolve them from length alone.

Both criteria are aggregated over the twenty test configurations exactly as the per-fleet
values are, that is, as the mean over the three weight components of the per-component figure,
which is what scikit-learn returns for a multi-output target.

    python3 make_model_figure.py [--outdir DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO, 'results')
ARTICLE = os.environ.get('MMRS_ARTICLE_IMAGES', 'figures')

WIDTH = 6.1
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
})

# Two series, separated in hue and in lightness. Validated in OKLab: dE 26 for normal vision
# and 19 under protanopia, the worst of the three simulated deficiencies.
MAE_COLOUR = '#1f4e79'
RMSE_COLOUR = '#b07aa1'
GRID = '#e4e6e8'

MODELS = ['SVR', 'Random Forest', 'Decision Tree', 'Lasso', 'Polynomial']
LABEL = {'SVR': 'Support Vector Regression', 'Random Forest': 'Random Forest',
         'Decision Tree': 'Decision Tree', 'Lasso': 'Lasso', 'Polynomial': 'Polynomial',
         'Mean baseline': 'Constant predictor'}


def aggregate(frame, model):
    g = frame[frame.model == model].sort_values(['auv_count', 'area'])
    e = np.stack([np.abs(g['true_w%d' % i] - g['pred_w%d' % i]).values for i in (1, 2, 3)])
    rmse = float(np.mean([np.sqrt((e[i] ** 2).mean()) for i in range(3)]))
    return float(e.mean()), rmse


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', default=ARTICLE)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    frame = pd.read_csv(os.path.join(RESULTS, 'owa_model_predictions_on_test.csv'))
    vals = {m: aggregate(frame, m) for m in MODELS + ['Mean baseline']}
    ranked = sorted(MODELS, key=lambda m: vals[m][0])
    # The constant predictor is a reference rather than a candidate, so it sits below a rule
    # instead of being ranked among the techniques.
    rows = ranked + ['Mean baseline']
    y = {m: len(ranked) - i for i, m in enumerate(ranked)}
    y['Mean baseline'] = -0.85

    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    h = 0.34
    for m in rows:
        mae, rmse = vals[m]
        ax.barh(y[m] + h / 2 + 0.012, mae, height=h, color=MAE_COLOUR, zorder=3)
        ax.barh(y[m] - h / 2 - 0.012, rmse, height=h, color=RMSE_COLOUR, zorder=3)
        ax.annotate('%.3f' % mae, xy=(mae, y[m] + h / 2 + 0.012), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=8.5, color='#3d3d3d')
        ax.annotate('%.3f' % rmse, xy=(rmse, y[m] - h / 2 - 0.012), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=8.5, color='#3d3d3d')

    ax.axhline(-0.3, color=GRID, linewidth=0.8, zorder=1)
    ax.set_yticks([y[m] for m in rows])
    ax.set_yticklabels([LABEL[m] for m in rows])
    ax.set_ylim(-1.5, len(ranked) + 0.75)
    ax.set_xlim(0, 2.06)
    ax.set_xlabel('Error over the twenty test configurations')
    # Solid hairlines, one shade off the surface: dashed or dotted grids read as thresholds.
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(axis='y', length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=MAE_COLOUR, label='MAE'),
               plt.Rectangle((0, 0), 1, 1, color=RMSE_COLOUR, label='RMSE')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 1.13), ncol=2,
              frameon=False)

    fig.tight_layout()
    path = os.path.join(args.outdir, 'model_aggregate.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

    print('%-26s %8s %8s' % ('technique', 'MAE', 'RMSE'))
    for m in rows:
        mae, rmse = vals[m]
        print('%-26s %8.3f %8.3f' % (LABEL[m], mae, rmse))
    print('\n  written %s' % path)


if __name__ == '__main__':
    main()
