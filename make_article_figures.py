"""Regenerate the result figures of the manuscript from the retrained models.

Reads the metric files written by regressor.py and writes the figures straight into the
article repository, so the plots and the numbers quoted in the text cannot drift apart.

The constant predictor is drawn alongside the trained models rather than omitted: it is the
reference that says whether any technique is using the mission descriptors at all, and the
comparison is meaningless without it.

    python3 make_article_figures.py [--outdir DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
# Destination of the generated figures; override with --article on the command line.
ARTICLE = os.environ.get('MMRS_ARTICLE_IMAGES', 'figures')

# The manuscript includes these at 0.9\textwidth of a single-column cas-sc page, roughly
# 6.1 in. Drawing them any wider means LaTeX scales them down and the labels shrink with the
# figure, which is what made the first version unreadable. Sizes are chosen so the printed
# text lands close to the body font.
WIDTH = 6.1
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 200,
})

# Fixed order and colours so every figure tells the same visual story.
ORDER = ['Random Forest', 'SVR', 'Decision Tree', 'Polynomial', 'Lasso', 'Mean baseline']
LABEL = {'Random Forest': 'Random Forest', 'SVR': 'SVR', 'Decision Tree': 'Decision Tree',
         'Polynomial': 'Polynomial', 'Lasso': 'Lasso', 'Mean baseline': 'Constant predictor'}
COLOUR = {'Random Forest': '#1f4e79', 'SVR': '#4e79a7', 'Decision Tree': '#76b7b2',
          'Polynomial': '#b07aa1', 'Lasso': '#bab0ac', 'Mean baseline': '#8c8c8c'}


def legend_handles(models):
    """Legend built from the fixed model order, not from the drawing order.

    The bars are sorted by value inside each group, so the order in which they are drawn
    changes from group to group and between figures. Building the legend separately keeps it
    identical everywhere and makes colour the stable identifier of a technique.
    """
    from matplotlib.patches import Patch
    return [Patch(facecolor=COLOUR[m], edgecolor='white', label=LABEL[m],
                  hatch='//' if m == 'Mean baseline' else None) for m in models]


def style(ax, ylabel):
    ax.set_xlabel('Number of AUVs')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def grouped_bars(frame, metric, path, ylabel):
    """One group of bars per fleet size, sorted from lowest to highest error.

    Sorting inside each group instead of keeping a fixed order across groups makes the best
    technique of every configuration readable at a glance, which is the comparison this
    figure exists for. The consequence is that position no longer identifies the model, so
    colour has to, and the legend is built from a fixed order rather than from the drawing
    order to stay identical between the MAE and the RMSE figure.
    """
    fleets = sorted(frame['AUV Count'].unique())
    models = [m for m in ORDER if m in set(frame.Model)]
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))

    for j, fleet in enumerate(fleets):
        values = frame[frame['AUV Count'] == fleet].set_index('Model')[metric]
        for i, m in enumerate(values.reindex(models).sort_values().index):
            pos = j + i * width - 0.4 + width / 2
            ax.bar(pos, values[m], width, color=COLOUR[m],
                   hatch='//' if m == 'Mean baseline' else None,
                   edgecolor='white', linewidth=0.6)

    ax.set_xticks(np.arange(len(fleets)))
    ax.set_xticklabels(fleets)
    style(ax, ylabel)
    ax.legend(handles=legend_handles(models), ncol=3, frameon=False, loc='upper left',
              bbox_to_anchor=(0, 1.24))
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('  escrito %s' % path)


def component_bars(frame, prefix, path, ylabel):
    """Error broken down by weight component, one panel per component."""
    fleets = sorted(frame['AUV Count'].unique())
    models = [m for m in ORDER if m in set(frame['Regression Model'])]
    # Stacked rather than side by side: three panels across 6.1 in leave two inches each,
    # which is not enough for twenty-four bars and a readable axis.
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.4), sharex=True)
    width = 0.8 / len(models)
    for k, ax in enumerate(axes, start=1):
        col = '%s_w%d' % (prefix, k)
        # Sorted within each group, as in the aggregated figures, so the best technique for
        # a given component and fleet size is the leftmost bar. The ordering is computed per
        # panel: a technique that estimates w1 best need not be the best for w3.
        for j, fleet in enumerate(fleets):
            values = frame[frame['AUV Count'] == fleet].set_index('Regression Model')[col]
            for i, m in enumerate(values.reindex(models).sort_values().index):
                pos = j + i * width - 0.4 + width / 2
                ax.bar(pos, values[m], width, color=COLOUR[m],
                       hatch='//' if m == 'Mean baseline' else None,
                       edgecolor='white', linewidth=0.6)
        ax.set_xticks(np.arange(len(fleets)))
        ax.set_xticklabels(fleets)
        # The component label goes inside the axes: as a title it collides with the legend
        # on the top panel, and the y label is written once for the three panels so the
        # three copies do not overlap each other.
        ax.text(0.005, 0.95, '$w_%d$' % k, transform=ax.transAxes, va='top', fontsize=11)
        style(ax, ylabel if k == 2 else '')
        if k < 3:
            ax.set_xlabel('')
    axes[0].legend(handles=legend_handles(models), ncol=3, frameon=False,
                   loc='lower left', bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('  escrito %s' % path)


def surfaces(path):
    """Response surface of the adopted model for each weight component."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler
    import glob

    tr = pd.concat([pd.read_csv(f) for f in
                    sorted(glob.glob(os.path.join(os.path.dirname(RESULTS),
                                                  'weights/[3-6]AUV_weights.csv')))])
    key = 'auv_count' if 'auv_count' in tr.columns else 'auvs'
    X = tr[[key, 'area']].values.astype(float)
    y = tr[['w1', 'w2', 'w3']].values.astype(float)
    scaler = StandardScaler().fit(X)
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=1000, max_depth=2, random_state=0))
    model.fit(scaler.transform(X), y)

    areas = np.linspace(X[:, 1].min(), X[:, 1].max(), 120)
    fleets = np.arange(3, 7)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7), sharey=True)
    for k, ax in enumerate(axes):
        for n in fleets:
            grid = scaler.transform(np.column_stack([np.full_like(areas, n), areas]))
            ax.plot(areas / 1000.0, model.predict(grid)[:, k], label='%d AUVs' % n,
                    linewidth=1.8)
        obs = tr[tr[key].isin(fleets)]
        ax.scatter(obs['area'] / 1000.0, obs['w%d' % (k + 1)], s=18, c='0.25',
                   marker='x', zorder=5, label='training targets' if k == 0 else None)
        # One x label for the three panels: repeating it under each one leaves three copies
        # that overlap into unreadable text at the width this figure is printed.
        ax.set_xlabel('Exploration area ($10^3$ m$^2$)' if k == 1 else '')
        ax.set_title('$w_%d$' % (k + 1), fontsize=11)
        ax.grid(linestyle=':', linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Predicted weight')
    axes[0].legend(ncol=5, frameon=False, fontsize=8, loc='upper left',
                   bbox_to_anchor=(0, 1.30))
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('  escrito %s' % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir', default=ARTICLE)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    metrics = pd.read_csv(os.path.join(RESULTS, 'owa_model_test_metrics.csv'))
    grouped_bars(metrics, 'MAE', os.path.join(args.outdir, 'MAE.pdf'),
                 'Mean absolute error')
    grouped_bars(metrics, 'RMSE', os.path.join(args.outdir, 'RMSE.pdf'),
                 'Root mean squared error')

    # The per-component panels are included as SVG by the manuscript; results.pdf is the
    # analysis-procedure diagram and must not be touched here.
    comp = pd.read_csv(os.path.join(RESULTS, 'owa_model_test_metrics_per_component.csv'))
    component_bars(comp, 'MAE', os.path.join(args.outdir, 'MAE_w.svg'),
                   'Mean absolute error')
    component_bars(comp, 'RMSE', os.path.join(args.outdir, 'RMSE_w.svg'),
                   'Root mean squared error')

    # Figure_4 is the regression-surface figure of Section 5.
    surfaces(os.path.join(args.outdir, 'Figure_4.pdf'))


if __name__ == '__main__':
    main()
