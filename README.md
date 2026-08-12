# MMRS_NN — OWA parameter prediction and campaign analysis

Companion code for *Machine Learning-Based Prediction of OWA Parameters for Communication-Aware
Coordination in Marine Multi-Robot Systems* (Ocean Engineering).

This repository covers two things: the regression study that estimates OWA weighting vectors
from mission descriptors, and the analysis of the simulation campaigns that the manuscript's
results section reports. Every table and every figure quoted in that section is produced here
from the recorded data, and a verification script recomputes each number in the running text and
reports it against the value the manuscript states.

## What you need

```bash
python3 -m venv env
source env/bin/activate
pip install scikit-learn numpy pandas matplotlib seaborn scipy
```

The analysis scripts additionally read ROS bagfiles and therefore need `rosbag`, which comes
with a ROS Noetic installation:

```bash
source /opt/ros/noetic/setup.bash
```

TensorFlow and Keras are needed only by the `nn_*` scripts, which are a neural-network variant
outside the OWA regression study and are not required to reproduce any published result.

## Campaign data

The simulations are the expensive part: a single mission runs in real time and takes between ten
and forty minutes, and the four campaigns together amount to roughly 530 missions. The recorded
bagfiles are published as a separate dataset because of their size.

Unpack them anywhere and point the analysis scripts at that location, either with `--data-root`
or through the `MRS_DATA` environment variable:

```bash
export MRS_DATA=/path/to/campaigns
```

The expected layout is one directory per campaign:

| Campaign | Missions | Contents |
|---|---|---|
| `v2_sweep` | 220 | grid sweep over all areas and fleet sizes; source of the supervised targets |
| `v2_dispersion` | 240 | replication campaign, 6 cells x 5 weighting vectors x 8 object layouts |
| `v3_roundrobin` | 48 | round-robin baseline over the same cells and layouts |
| `v4_predicted` | 20 | the predicted weighting vector simulated on each test cell |

each laid out as `<campaign>/<area>/<N>AUVs/<policy>/{manifest.csv,bagfiles/,params/}`, with
`policy` being `owa` or `roundrobin`.

Regenerating the campaigns themselves, rather than analysing the published ones, is done from the
`multi_robot_system` package of the [MMRS_stack](https://github.com/martorelltorres/MMRS_stack):

```bash
rosrun multi_robot_system run_simulations.py \
    --areas 15000 35000 55000 --auvs 3 6 --realizations 8 --workers 2 \
    --output-root $MRS_DATA/v2_dispersion
```

Cells of six AUVs must be run with `--workers 1`: two concurrent six-AUV missions contend for
the machine and distort the mission durations.

## Reproducing the results

### 1. Import the supervised targets

```bash
python3 import_targets.py --campaign $MRS_DATA/v2_sweep
```

Converts the campaign extraction into `weights/{3,4,5,6}AUV_weights.csv` (six training areas
each) and `weights/optimal_test_weights.csv` (20 rows: five test areas x four fleet sizes). The
previous files are copied to `weights/published/` first, so the originally published figures stay
reproducible. `--dry-run` previews the result.

The script warns when a weighting vector falls outside the grid W, which is the signature of a
predicted-weights run left inside a sweep folder.

### 2. Compare the regression models

```bash
python3 regressor.py
```

Compares Decision Tree, Random Forest, SVR, Polynomial and Lasso regression under leave-one-out
cross-validation, and writes MAE and RMSE, globally and per weight component, to `results/`.
These are the numbers behind the model-comparison figures of the manuscript.

Two points matter when reading the output:

**A single model, not one per fleet size.** Training one model per fleet left `auv_count` with
zero variance inside each subset, so the scaler collapsed it and every model effectively saw a
single feature with six samples. A single model over the 24 samples uses both descriptors;
metrics are still reported per fleet size.

**A constant baseline is always reported.** `Mean baseline` predicts the mean of the training
targets and ignores the inputs entirely. It is not decoration: a model that does not beat it is
not using the mission descriptors at all, and in this dataset none of them does by a significant
margin. Read that comparison before drawing conclusions about any regressor.

### 3. Export the predicted weights

```bash
python3 export_predicted_weights.py
```

Writes `results/predicted_weights_rf.csv`, the weighting vector the adopted model predicts for
each test cell. These are the vectors simulated in the `v4_predicted` campaign.

### 4. Build the tables

```bash
cd analysis
python3 make_article_tables.py       # variability and variance, from v2_dispersion
python3 make_revision_tables.py      # accuracy stability and baseline comparison
python3 compute_prediction_metrics.py # predicted weights against the grid
python3 compare_roundrobin.py        # paired OWA vs round-robin summary
```

Each writes LaTeX tables to `article_tables/` and prints the figures quoted in the running text.
All of them accept `--data-root` and cache their per-bagfile metrics, so a second run is fast.

### 5. Verify every number

```bash
cd analysis
python3 verify_article_numbers.py
```

Recomputes each quantity the manuscript quotes, from the bagfiles, and prints it beside the value
the text claims, together with the formula and the campaign it derives from. It exits non-zero if
any check fails, so a change in the data that would invalidate a sentence is detected rather than
propagated.

```
[5.3 baseline comparison]
  ok  6 AUVs dU (%)                      27.100     27.134
  ok  6 AUVs wins                        22.000     22.000
...
88 checks, 88 passed, 0 failed
```

## Conventions

All utilities follow Equation (6) of the manuscript with the baseline coefficients
`(alpha, beta, gamma) = (1, 0.5, 1)`. Latency and distance metrics are min-max normalised and
object counts are max normalised **within a mission configuration**, that is, within a given
exploration area and fleet size; utilities are never compared across fleet sizes. Note that the
originally published CSV files were built with `beta = 1`; every result reproduced here uses the
declared value of 0.5.

Comparisons between weighting vectors, and between OWA and the baseline, are paired by object
layout and tested with a Wilcoxon signed-rank test blocked by layout.

## Layout

| Path | Contents |
|---|---|
| `import_targets.py` | brings campaign targets into this repository |
| `regressor.py` | model comparison reported in the manuscript |
| `export_predicted_weights.py` | predicted weighting vector per test cell |
| `make_article_figures.py` | regression figures of the manuscript |
| `surface_experiment.py` | utility-surface exploration over the descriptor domain |
| `analysis/` | campaign analysis: tables, baseline comparison and verification |
| `analysis/campaign_paths.py` | resolution of the campaign data location |
| `weights/` | supervised training and test targets |
| `weights/published/` | targets behind the originally published figures |
| `results/` | metrics, predictions and exported weights |
| `data/` | earlier campaign extractions retained for provenance |
| `nn_*.py`, `artm_*` | neural-network and ARTM variants, outside the OWA regression study |
