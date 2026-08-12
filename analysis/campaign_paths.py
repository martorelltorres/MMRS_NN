"""Resolution of the campaign data location, shared by the analysis scripts.

The simulation campaigns are published as ROS bagfiles separately from this repository,
because they run to several hundred megabytes. Every analysis script therefore needs to be
told where they were unpacked. The location is taken, in order of precedence, from the
--data-root option, from the MRS_DATA environment variable, or from a `campaigns` directory
next to this repository.

A campaign directory is expected to hold one sub-directory per campaign, each laid out as

    <campaign>/<area>/<N>AUVs/<policy>/manifest.csv
    <campaign>/<area>/<N>AUVs/<policy>/bagfiles/results_<i>.bag

with `policy` being `owa` or `roundrobin`.
"""
import argparse
import os

ENV_VAR = 'MRS_DATA'
CAMPAIGNS = ('v2_sweep', 'v2_dispersion', 'v3_roundrobin', 'v4_predicted')


def default_root():
    env = os.environ.get(ENV_VAR)
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'campaigns')


def add_argument(parser):
    """Register --data-root on an ArgumentParser."""
    parser.add_argument('--data-root', default=default_root(),
                        help='directory holding the campaign folders '
                             '(default: $%s, else ./campaigns)' % ENV_VAR)
    return parser


def resolve(root, campaign, required=True):
    """Absolute path of one campaign, with a legible error when it is absent."""
    path = os.path.join(root, campaign)
    if required and not os.path.isdir(path):
        raise SystemExit(
            'campaign "%s" not found under %s\n'
            'Point --data-root at the directory where the published bagfiles were unpacked, '
            'or set %s.' % (campaign, root, ENV_VAR))
    return path


def cache_path(root, name):
    """Location of a metric cache, kept beside the campaigns it summarises."""
    return os.path.join(root, name)


def parser(description):
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_argument(ap)
    return ap
