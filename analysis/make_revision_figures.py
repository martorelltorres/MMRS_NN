"""Draw the two figures of Section 5.2 from the replication campaign.

Both replace tables that ran to thirty rows each and that nobody reads as numbers. What the
subsection argues is a shape rather than a set of values, and a shape is what these show.

The variability figure places the five weighting vectors of a cell side by side with the 95%
confidence interval of their mean priority latency over the eight object layouts. The intervals
overlap inside every panel, which is the tabulated claim made visible.

The accuracy figure is the central one. For each weighting vector it draws the eight values the
Average Prediction Accuracy of Equation (13) takes across the eight layouts, spanned from
minimum to maximum. Area, fleet size and weighting vector are fixed along each row; the only
thing that changes between its eight points is where the objects were placed. Most rows cross
the whole scale, which is the argument the subsection makes.

    python3 make_revision_figures.py [--outdir DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import campaign_paths
import make_revision_tables as mrt

# Destination of the generated figures; override with --outdir.
ARTICLE = os.environ.get('MMRS_ARTICLE_IMAGES', 'figures')

# Matching make_article_figures.py: the manuscript includes these at 0.9\textwidth of a
# single-column cas-sc page, and the fonts are sized so the printed labels land near body text.
WIDTH = 6.1
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10.5,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 200,
})

# One series per figure: the weighting vector is named on the y axis, so colour would only
# repeat what the axis already says. A single ink keeps both figures readable in grayscale and
# free of any colour-vision concern.
INK = '#1f4e79'
SOFT = '#9db4c8'
# Solid hairline one shade off the surface: a dotted grid reads as a threshold.
GRID = '#e4e6e8'
AREAS = (15000, 35000, 55000)
FLEETS = (3, 6)


def panel_grid(ylabel, label_y=0.02):
    """Rows are fleet sizes, columns are areas, which is how the campaign is organised.

    `label_y` lifts the shared x label when a legend has to sit underneath it.
    """
    fig, axes = plt.subplots(len(FLEETS), len(AREAS), figsize=(WIDTH, 4.6))
    for row, fleet in enumerate(FLEETS):
        for col, area in enumerate(AREAS):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(r'%s m$^2$' % format(area, ','), pad=6)
            if col == 0:
                ax.set_ylabel('%d AUVs' % fleet)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.set_axisbelow(True)
    fig.supxlabel(ylabel, y=label_y, fontsize=10)
    return fig, axes


def vector_labels(grid):
    return ['(%d,%d,%d)' % g for g in grid]


def figure_variability(d, path):
    """Mean priority latency with its 95% confidence interval over the eight layouts."""
    owa = d[d.policy == 'owa']
    fig, axes = panel_grid('Mean priority-object latency (s)')
    labels = vector_labels(mrt.GRID)
    for row, fleet in enumerate(FLEETS):
        for col, area in enumerate(AREAS):
            ax = axes[row][col]
            cell = owa[(owa.area == area) & (owa.auvs == fleet)]
            ys, means, errs = [], [], []
            for cfg, g in cell.groupby('cfg'):
                n = len(g)
                sd = g.priority_latency.std(ddof=1)
                ys.append(4 - int(cfg))
                means.append(g.priority_latency.mean())
                errs.append(1.96 * sd / np.sqrt(n))
            ax.errorbar(means, ys, xerr=errs, fmt='o', color=INK, ecolor=SOFT,
                        elinewidth=2.0, capsize=3, markersize=5, zorder=3)
            ax.set_yticks(range(5))
            ax.set_yticklabels(labels[::-1] if col == 0 else [])
            ax.set_ylim(-0.7, 4.7)
            ax.grid(axis='x', color=GRID, linewidth=0.8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('  written %s' % path)


def figure_apa_stability(a, path):
    """The APA each weighting vector attains on each of the eight object layouts."""
    fig, axes = panel_grid('Average Prediction Accuracy (%)', label_y=0.105)
    labels = vector_labels(mrt.GRID)
    for row, fleet in enumerate(FLEETS):
        for col, area in enumerate(AREAS):
            ax = axes[row][col]
            cell = a[(a.area == area) & (a.auvs == fleet)]
            for cfg, g in cell.groupby('cfg'):
                y = 4 - int(cfg)
                lo, hi = g.APA.min(), g.APA.max()
                ax.plot([lo, hi], [y, y], color=SOFT, linewidth=3.0,
                        solid_capstyle='round', zorder=2)
                ax.plot(g.APA, [y] * len(g), 'o', color=INK, markersize=3.4,
                        alpha=0.85, zorder=3)
                ax.plot(g.APA.mean(), y, '|', color=INK, markersize=11,
                        markeredgewidth=2.0, zorder=4)
            ax.set_yticks(range(5))
            ax.set_yticklabels(labels[::-1] if col == 0 else [])
            ax.set_ylim(-0.7, 4.7)
            ax.set_xlim(-5, 105)
            ax.set_xticks([0, 50, 100])
            ax.grid(axis='x', color=GRID, linewidth=0.8)
    # Two mark types share the axis, so they are named once for the whole figure.
    handles = [plt.Line2D([], [], color=INK, marker='o', linestyle='none', markersize=3.4,
                          label='one object layout'),
               plt.Line2D([], [], color=INK, marker='|', linestyle='none', markersize=11,
                          markeredgewidth=2.0, label='mean over the eight layouts'),
               plt.Line2D([], [], color=SOFT, linewidth=3.0, label='observed range')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('  written %s' % path)


def main():
    ap = campaign_paths.parser(__doc__)
    ap.add_argument('--outdir', default=ARTICLE)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = mrt.collect(campaign_paths.resolve(args.data_root, 'v2_dispersion'),
                    campaign_paths.resolve(args.data_root, 'v3_roundrobin'),
                    campaign_paths.cache_path(args.data_root, 'roundrobin_metrics.pkl'))
    figure_variability(d, os.path.join(args.outdir, 'variability.pdf'))
    figure_apa_stability(mrt.apa_stability(d),
                         os.path.join(args.outdir, 'apa_stability.pdf'))


if __name__ == '__main__':
    main()
