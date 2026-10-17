# Singapore Rainfall Prediction

This project uses PySpark to analyse Singapore rainfall from 2017 to 2024 and
predict whether rainfall at a station will exceed a threshold in the next
30 minutes.

## Main files

- `utils/preprocess.py`: shared data loading and cleaning functions.
- `task1.py`: Task 1 rainfall pattern analysis.
- `task2.py`: final Task 2 training script.
- `inference.py`: load a trained final model and run one prediction.
- `utils/rainfall_model.py`: final GBT model with neighbor rainfall.
- `utils/model_evaluation.py`: shared evaluation and plotting functions.
- `task2_logistic.py` and `utils/logistic_model.py`: logistic baseline.
- `task2_gbt_baseline.py` and `utils/gbt_baseline_model.py`: GBT baseline without
  neighbor rainfall.
- `plot_model_comparison.py`: plot the preliminary model comparison.
- `task2_history_ablation.py` and `utils/history_ablation_model.py`: remove the
  30-minute history feature, then remove both 15- and 30-minute history.
- `task2_time_ablation.py` and `utils/time_ablation_model.py`: remove month/hour
  features or replace their cyclic encoding with raw numeric values.
- `plot_history_ablation_comparison.py`: recreate the threshold-level history
  ablation comparison from saved results.
- `requirements.txt`: tested Python package versions.

## Environment

The experiments were tested with Python 3.10.22, Java 17, and the package
versions in `requirements.txt`. For example, create an environment and install
the dependencies with:

```powershell
conda create -n dsa5208proj2 python=3.10.22
conda activate dsa5208proj2
python -m pip install -r requirements.txt
```

Java 17 must also be installed and available through `PATH` or `JAVA_HOME`.

## Data

Place `rainfall_2017.csv` through `rainfall_2024.csv` in `data/`. They can be
downloaded from data.gov.sg with:

```powershell
python download_dataset.py
```

The raw CSV files are excluded from Git because they are large. Exact numeric
reproduction requires the same source files used for the saved runs.

## Reproducing the report experiments

All commands below are run from the project root in the configured
environment.

### Task 1 rainfall patterns

Generate the annual, monthly, year-month, hourly, and station summaries and
figures with:

```powershell
python task1.py
```

This writes the underlying CSV summaries to `tables/` and plots to `figures/`.

### Preliminary Task 2 models

Train the logistic baseline using one quarter of eligible station-time
examples:

```powershell
python task2_logistic.py
```

Train the GBT baseline using the selected 1/16 training sample:

```powershell
python task2_gbt_baseline.py
```

The earlier 1/64 GBT uses the same script with a different training modulus:

```powershell
python -c "import task2_gbt_baseline as t; t.TRAIN_SAMPLE_MODULUS=64; t.main()"
```

The preliminary comparison figure can be recreated from the three saved runs
listed in `plot_model_comparison.py`:

```powershell
python plot_model_comparison.py
```

If the models have been retrained under new timestamped directory names,
update the `RUNS` paths in that plotting script first.

## Final model

The final model is a Spark `GBTClassifier`. In addition to station, time,
threshold, and recent rainfall features, it uses the mean current rainfall at
the nearest monitoring stations.

Run it with the tested default settings:

```bash
python task2.py
```

The default settings reproduce the selected configuration:

```text
sample modulus:           4
training sample modulus: 16
nearest stations:         3
number of trees:         20
maximum tree depth:       4
learning rate:            0.1
subsampling rate:         0.8
random seed:              5208
```

Use `python task2.py --help` to see every option. For example:

```bash
python task2.py --max-iter 30 --max-depth 5 --neighbor-count 5 --step-size 0.05
```

The main model options are:

- `--sample-modulus`: sampling applied to all examples. A smaller value uses
  more data; `1` means no sampling.
- `--training-sample-modulus`: additional training selection. It must be a
  multiple of `--sample-modulus`.
- `--neighbor-count`: number of nearest stations used for neighbor rainfall.
- `--max-iter`: number of boosted trees.
- `--max-depth`: maximum tree depth.
- `--step-size`: GBT learning rate.
- `--subsampling-rate`: fraction sampled for each boosting iteration.
- `--seed`: random seed.
- `--thresholds`: rainfall thresholds separated by spaces.
- `--spark-master`, `--driver-memory`, and `--shuffle-partitions`: local or
  cluster Spark settings.
- `--run-name`: optional output directory name.

Each run creates a new directory under `models/` containing the model JSON,
parameters, station-neighbor mapping, metrics, and figures.

### Inference

Predict one station/time/threshold from an existing final-model run with:

```powershell
python inference.py `
  --model-run models/final_gbt_run_20261017_184044 `
  --station-id S06 `
  --prediction-time "2024-11-13 16:30:00" `
  --threshold-mm 1.0
```

The script evaluates the trees stored in `model.json`, so it does not require a
native Hadoop model directory. It loads the relevant annual CSV and
automatically constructs current rainfall, 15- and 30-minute rainfall history,
and mean neighbor rainfall. The prediction time is Singapore local time and
must fall exactly on minute 00 or 30, with complete five-minute readings from
that time through 25 minutes earlier.

## Rainfall-history ablation

Run both history-feature ablations with the tested final-model settings:

```powershell
python task2_history_ablation.py
```

This trains two models on the same examples:

- `remove_30min`: retains current rainfall and the 15-minute history total.
- `remove_15min_30min`: retains current rainfall but removes both history
  totals.

The run directory contains each model's metrics and threshold plot, an overall
metric comparison, and `history_ablation_threshold_comparison_test.png`, which
compares PR-AUC, ROC-AUC, and Brier score across all seven thresholds for the
full model and both ablations.

To recreate only the threshold-level figure from an existing run:

```powershell
python plot_history_ablation_comparison.py `
  models/history_ablation_run_20261017_190306
```

## Month/hour feature ablation

Run both time-feature ablations with:

```powershell
python task2_time_ablation.py
```

The two variants remove all month/hour inputs or replace the month/hour sine
and cosine pairs with raw numeric month and hour. The run automatically saves
an overall comparison and a three-panel threshold comparison against the full
cyclic-time model.

The report uses `models/time_ablation_run_20261017_194655`. Its main output
files are:

```text
time_ablation.png
time_ablation_threshold_comparison_test.png
comparison_results.json
remove_time/
noncyclic_time/
```

The full cyclic model, no-time model, and numeric-time model obtained test
ROC-AUC values of 0.9181, 0.9116, and 0.9082 respectively. Their test PR-AUC
values were nearly identical at 0.6281, while the full cyclic model had the
lowest mean test Brier score.

## Build the report

Copy the final neighbor-ablation figure and history-ablation figure used by the
current report:

```powershell
Copy-Item `
  models/final_gbt_run_20261017_184044/baseline_comparison.png `
  figures/final_model_comparison.png
Copy-Item `
  models/history_ablation_run_20261017_190306/history_ablation_threshold_comparison_test.png `
  figures/history_ablation.png
Copy-Item `
  models/time_ablation_run_20261017_194655/time_ablation_threshold_comparison_test.png `
  figures/time_ablation.png
```

Then compile the LaTeX report twice so that references are resolved:

```powershell
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

## Saved runs used by the current report

The current report results are backed by these run directories:

- `models/logistic_run_20261015_200027`
- `models/gbt_run_20261016_163426`
- `models/gbt_run_20261015_213401`
- `models/final_gbt_run_20261017_184044`
- `models/history_ablation_run_20261017_190306`
- `models/time_ablation_run_20261017_194655`

Generated model directories, figures, tables, and raw rainfall CSV files are
excluded from Git. Include the six run directories above separately if saved
metrics and fitted-model artifacts must accompany the submission.

## Submission checklist

- Replace the member-name and student-ID placeholders near the top of
  `report.tex`, then compile the final `report.pdf`.
- Submit the complete source code, `README.md`, and `requirements.txt`.
- Submit the six saved run directories listed above if the experimental
  artifacts are required.
- Submit `model.json` and `neighbor_mapping.json` from the selected final run;
  these are the fitted-model artifacts used by `inference.py`.
- Do not include the multi-gigabyte rainfall CSV files unless Canvas explicitly
  requests them; they can be obtained with `download_dataset.py`.
