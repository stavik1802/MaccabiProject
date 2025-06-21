# Football Player Physical Stats Prediction

This project predicts football player physical stats (e.g. player load, distance covered) using per-minute match data and PyTorch deep learning models. It supports both single-target and multi-target regression, configurable prediction horizons, and runs on GPU or CPU.

## Directory Structure

```
.
├── data/                  # Store raw and processed data files here
├── models/                # Neural network model architectures
│   ├── lstm.py
│   └── __init__.py
├── utils/                 # Data loading, preprocessing, metrics, config, visualization
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── visualization.py
│   └── config.py
├── scripts/               # Main entry points for training, prediction, evaluation
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── analysis/              # (Optional) Additional analysis scripts
│   ├── filter.py
│   ├── filter_data_all.py
│   ├── filter_time.py
│   ├── find_playing_times.py
│   ├── play_time.py
│   ├── update_data.py
│   └── heatmap.py
├── requirements.txt
├── README.md
```

## Features

- **Single & Multi-Target Regression:** Toggle between predicting a single stat or several at once with a config/CLI flag.
- **Prediction Horizon:** Predict future stats N minutes ahead (e.g. 10 minutes). Configurable.
- **PyTorch Models:** Modular LSTM, BiLSTM, and (optionally) attention-based models.
- **GPU/CPU:** Use `--gpu` flag to enable GPU if available, otherwise CPU.
- **Evaluation:** MAE, RMSE, and R² metrics with explanations and plots.
- **Visualization:** Learning curves, predicted vs. true scatterplots.
- **Extensible:** Modular code for easy extension.

## Usage

**Install dependencies:**
```sh
pip install -r requirements.txt
```

**Train a model:**
```sh
python scripts/train.py --data data/match.csv --target playerload_1s_sum \
    --mode single --horizon 10 --gpu
```
- `--mode single` or `--mode multi` for single/multi-target.
- `--horizon 10` predicts 10 minutes ahead.

**Evaluate a trained model:**
```sh
python scripts/evaluate.py --data data/match.csv --model_path path/to/model.pt \
    --target playerload_1s_sum --mode single --horizon 10
```

**Predict with a trained model:**
```sh
python scripts/predict.py --data data/match.csv --model_path path/to/model.pt \
    --target playerload_1s_sum --mode single --horizon 10
```

## Configurable Arguments

- `--data`: Path to CSV file.
- `--target`: Target stat(s) to predict (comma-separated for multi-target).
- `--mode`: `single` or `multi`.
- `--horizon`: Prediction horizon in minutes.
- `--gpu`: Use GPU if available.

## Metrics Explained

- **MAE (Mean Absolute Error):** Average absolute difference between predictions and true values. Lower is better.
- **RMSE (Root Mean Squared Error):** Square root of the average squared differences; penalizes larger errors more than MAE.
- **R² (Coefficient of Determination):** How much variance is explained by the model (1.0 = perfect, 0 = no better than mean).

Plots help visualize:
- Model fit (predicted vs. true values)
- Training/validation loss curves

## Extending

- Add new models to `models/`.
- Add new metrics or visualizations to `utils/`.
- Add new scripts to `scripts/`.

## Data Assumption

- All features are numeric and the data is already normalized.
- 1 row = 1 minute of stats for a single player.

## Support

Open an issue or pull request for feature requests or bugs.