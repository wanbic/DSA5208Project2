# Singapore Rainfall Prediction

This project uses PySpark to analyse Singapore rainfall from 2017 to 2024 and
predict whether rainfall at a station will exceed a threshold in the next
30 minutes.

## Main files

- `preprocess.py`: shared data loading and cleaning functions.
- `task1.py`: Task 1 rainfall pattern analysis.
- `task2.py`: final Task 2 training script.
- `rainfall_model.py`: final GBT model with neighbor rainfall.
- `model_evaluation.py`: shared evaluation and plotting functions.
- `task2_logistic.py` and `logistic_model.py`: logistic baseline.
- `task2_gbt_baseline.py` and `gbt_baseline_model.py`: GBT baseline without
  neighbor rainfall.
- `plot_model_comparison.py`: plot the preliminary model comparison.

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

Each run creates a new directory under `models/` containing the model,
parameters, station-neighbor mapping, metrics, and figures.

## Baselines

Run the earlier models with:

```bash
python task2_logistic.py
python task2_gbt_baseline.py
```

## Data

Place `rainfall_2017.csv` through `rainfall_2024.csv` in `data/`. The raw CSV
files and generated model directories are excluded from Git because they are
large.
