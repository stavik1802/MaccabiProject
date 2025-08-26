# MaccabiProject

End-to-end tooling for analyzing football (soccer) match data for Maccabi Haifa: preprocessing raw tracking/substitutions data, generating features, training and evaluating models for physical load and predicting ball possession using physical features, and running substitution suggestion application based on physical load model and possession model/what-if experiments.

## Repository layout

``` 
MaccabiProject/
├── analysis/                  # One-off analysis utilities (filtering, play times, heatmaps)
├── experiment-subs/           # Experiments around substitution impact and improved players
├── experiments-physical/      # Physical load prediction experiments
├── experiments-possession/    # Possession estimation experiments and studies
├── model-physical/            # Training and evaluation entrypoints for physical models
├── model-possession/          # Training and evaluation entrypoints for possession models
├── substitution-application/  # Pipeline and evaluation for substitution application
├── utils_physical/            # Preprocessing, feature engineering, metrics, visualization for physical tasks
├── utils-possession/          # Data prep and utilities for possession-related tasks
├── utils-subs/                # Utilities for substitution-related analysis
├── utils-plot/                # Common plotting helpers
├── sub/                       # Substitutions CSV inputs (sample/game files)
├── requirements.txt           # Python dependencies
├── process.sbatch             # Slurm job script example
└── README.md
```

## Requirements

- Python 3.9+
- Optional: CUDA-capable GPU for training with PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data inputs

- Place raw substitutions CSVs under `sub/` (examples included).
- Physical/possession raw data paths are referenced by scripts in `utils_physical/` and `utils-possession/`. Adjust paths/flags as needed within those scripts.

## Common workflows

### 1) Physical load pipeline

- Preprocess and feature engineering (see `utils_physical/pre_proccess.py` and helpers). Example direct run:
- uses as input substitution files and raw GPS data of players
- creates the times of the first and second half 
- creates raw data according to the time each player played in specific game 
- creates json of field coordinates (corners and center)
- creates attacking direction for each half 
- creates a folder for each game contains merged_features_<Player code>.csv 
-each player file contains accumulated physical data, each row represents 15-seconds data accumulated with previous rows 
- Physical features: 
    1. Distance(m)
    2. Speed (m/s)
    3. High-speed running (m)
    4. Very high accelaration count
    5. Avg jerk
    6. Turns count
    7. PlayerLoad
    8. Time spent in walking
    9. Time spent in jogging
    10. Time spent in Running
    11. Time spent in Sprinting
    12. Total sprints
    13. Sprints in attacking direction 
    14. Sprints in defensive direction
    15. Distance in attacking direction
    16. Distance in defensive direction
    17. Time Spent in attacking actions
    18. Time Spent in defensive actions
    19. Time in attacking third of the field
    20. Time in middle third of the field
    21. Time in defending third of the field
```bash
python utils_physical/pre_proccess.py <raw_players_data_folder>
```
- Train physical model:
Model to predict physical state progress of player in game time.
```bash
python model-physical/train_physical.py --help
```
- Evaluate autoregressive TCN or trained models:
```bash
python model-physical/eval_autoregressive_tcn.py --help
```

### 2) Possession pipeline

- Prepare possession data:
- expects event log data from OPTA
- creates possession in windows of five minutes, each window independent.
-- use `utils-possession/shift_chunck_subs.py` to create independent window data from the made in stage ###1 
```bash
python utils-possession/make_train_folder_poss.py --help
python utils-possession/compute_possession.py --help
python utils-possession/shift_chunck_subs.py --help
```
- Train/evaluate possession model:
- expects for each game a folder contains independent physical feature windows and poss.csv contains the possession data.
- Model to predict possession in each 5-minute window using physical features. 
```bash
python model-possession/train_poss.py --help
python model-possession/eval_poss.py --help
```

### 3) Substitution suggesstion 
- suggests given a game and minute what players and what group of player should substituted each player.
- Player groups: 1-Center Backs, 2- Defensive/Center Midfielders, 3-Right Backs/Wingers and Left Backs/Wingers 4- attacking midfielders and Center Forwards. 
- Uses models created for physical state prediction and possession prediction
- Run substitution pipeline and evaluation:
```bash
python substitution-application/subs_pipe.py --help
python substitution-application/subs_eval.py --help
```
- Experiments around improved players:
```bash
python experiment-subs/experiment_improved_player.py --help
python experiment-subs/improved_subs_players.py --help
```

### 4) Experiments and plots

- Physical experiments:
```bash
python experiments-physical/experiment_window_size.py --help
python experiments-physical/experiment_best_feature_pred.py --help
python experiments-physical/experiment_extra_data.py --help
python experiments-physical/scale_feature_best.py --help
```
- Possession experiments:
```bash
python experiments-possession/experiment_grouped.py --help
python experiments-possession/experiment_home_vs_away_grouped.py --help
python experiments-possession/experiment_seasons_grouped.py --help
python experiments-possession/experiment_top_vs_other_group.py --help
python experiments-possession/experiment_weight.py --help
python experiments-possession/experiment_window_poss.py --help
python experiments-possession/feature_importance.py --help
```
- Substitutions experiments:
```bash
python experiment-subs/experiment_improved_player.py --help
python experiment-subs/improved_subs_players.py --help
```
- Plotting utilities:
```bash
python utils-plot/create_bar.py --help
python utils-plot/plot_importance.py --help
python utils-plot/plot_pretty_improved.py --help
python utils-plot/pretty_bar_feat.py --help
```

## Detailed file reference

### analysis/
- `filter.py`: Filter datasets by conditions/time windows.
- `filter_data_all.py`: Apply full filtering pipeline over datasets.
- `filter_time.py`: Time-based filtering utilities.
- `find_playing_times.py`: Derive players' playing-time intervals.
- `heatmap.py`: Generate heatmaps for positional/metric analysis.
- `play_time.py`: Compute and summarize play-time stats.
- `update_data.py`: Update/refresh analysis datasets.

### experiment-subs/
- `experiment_improved_player.py`: Experiment on improved player profiles under substitutions.
- `improved_subs_players.py`: Analyze and compare improved players in substitution contexts.

### experiments-physical/
- `experiment_best_feature_pred.py`: Which features best predict physical metrics.
- `experiment_extra_data.py`: Impact of additional data sources/features.
- `experiment_window_size.py`: Sensitivity to temporal window size.
- `scale_feature_best.py`: Feature scaling strategies comparison.

### experiments-possession/
- `experiment_grouped.py`: Grouped possession experiments across cohorts.
- `experiment_home_vs_away_grouped.py`: Home vs away grouped comparison.
- `experiment_seasons_grouped.py`: Across seasons grouped comparison.
- `experiment_top_vs_other_group.py`: Top vs other teams grouped comparison.
- `experiment_weight.py`: Weighting strategies in possession modeling.
- `experiment_window_poss.py`: Possession window-size sensitivity.
- `feature_importance.py`: Feature importance for possession tasks.
- `home_away.json`: Config/data for home vs away groupings.
- `seasons.json`: Config/data for season cohorts.
- `top_vs_other_teams.json`: Config/data for team cohorts.

### model-physical/
- `train_physical.py`: Train physical-load models.
- `eval_autoregressive_tcn.py`: Evaluate autoregressive TCN baselines/models.
- `saved-model/`: Saved model artifacts (if present).

### model-possession/
- `train_poss.py`: Train possession models.
- `eval_poss.py`: Evaluate possession models.
- `model-saved/`: Saved possession model artifacts (if present).

### substitution-application/
- `subs_pipe.py`: Substitution application pipeline entrypoint.
- `subs_eval.py`: Evaluate substitution application performance.
- `starters.json`: Starters configuration/examples.

### utils_physical/
- `add_merged_data.py`: Merge additional datasets into base physical data.
- `analyze_sub_file.py`: Analyze a single substitutions file.
- `attacking_direction.py`: Infer attacking direction over time.
- `change_to_date_data.py`: Convert time columns to dates.
- `config.py`: Constants/paths for physical utilities.
- `copy_files_from_folders.py`: Copy and organize files from folders.
- `create_games_data.py`: Build per-game data packages.
- `data_loading.py`: Load physical datasets from disk.
- `extract_player_data.py`: Extract per-player data windows.
- `extract_player_halves_data.py`: Extract data by halves per player.
- `feature_columns.py`: Define feature columns for modeling.
- `feature_rename_map.py`: Map for renaming feature columns.
- `filter.py`: General filtering utilities for physical data.
- `filter_data_all.py`: Apply full physical filtering pipeline.
- `filter_halves.py`: Filter to first/second half segments.
- `filter_only_maccabi.py`: Keep only Maccabi Haifa related records.
- `filter_subs_names.py`: Normalize/clean substitutions names.
- `filter_time.py`: Time-based filtering helpers.
- `find_bench.py`: Detect bench area/players.
- `find_field_bounds.py`: Determine field boundaries from data.
- `find_playing_times.py`: Compute playing intervals per player.
- `find_ten_runners.py`: Identify top runners by distance/speed.
- `generate_data.py`: Generate engineered datasets for training.
- `heatmap.py`: Produce heatmaps for physical metrics.
- `infer_field.py`: Infer field coordinates/orientation.
- `list_players.py`: List/aggregate players in datasets.
- `merge.py`: Merge multiple data sources/files.
- `metrics.py`: Metrics for evaluating physical models.
- `minute_features.py`: Compute per-minute features.
- `order.py`: Ordering/indexing utilities.
- `pertubate_dist.py`: Perturb distance for augmentation/tests.
- `play_time.py`: Compute play-time related features.
- `players_lists.py`: Manage curated player lists.
- `pre_proccess.py`: Main preprocessing pipeline (physical).
- `preprocessing.py`: Additional preprocessing helpers.
- `rename_features_names.py`: Unified feature renaming.
- `setup.py`: Setup/bootstrapping utilities.
- `sofa_hunt.py`: Retrieve/parse SofaScore-like data.
- `split.py`: Train/val/test splitting utilities.
- `sub/`: Auxiliary subdirectory used by utils.
- `update_data.py`: Update datasets after new games.
- `visualization.py`: Visualization helpers for physical tasks.

### utils-possession/
- `compute_possession.py`: Compute possession labels/metrics.
- `copy_poss_to_folder.py`: Copy possession data into train folders.
- `create_poss_binary.py`: Create binary possession targets.
- `delete_poss.py`: Clean/remove possession datasets.
- `make_data_for_poss.py`: Build possession-ready datasets.
- `make_ind.py`: Create indices/identifiers for possession data.
- `make_train_folder_poss.py`: Create train folder structure for possession.
- `shift_chunck_subs.py`: Shift and chunk substitutions data for possession.
- `sofa_hunt.py`: Retrieve/parse SofaScore-like possession data.

### utils-subs/
- `independent_avg_player.py`: Build independent average player profiles.
- `make_average_player.py`: Compute average player profiles.
- `minute_60.py`: Substitution analysis at 60-minute mark.

### utils-plot/
- `create_bar.py`: Create bar plots from experiment results.
- `plot_importance.py`: Plot feature importance charts.
- `plot_pretty_improved.py`: Pretty plotting for improved players.
- `pretty_bar_feat.py`: Enhanced feature bar plots.

### Other top-level files
- `requirements.txt`: Python dependencies.
- `process.sbatch`: Example Slurm job submission script.
- `README.md`: This document.

## Configuration notes

- Several utilities expect specific folder structures and filenames. Review constants/paths in:
  - `utils_physical/config.py`, `utils_physical/feature_columns.py`
  - `utils-possession/*`
  - Scripts inside `model-physical/` and `model-possession/`
- Many scripts accept CLI flags. Use `--help` to discover arguments and defaults.

## Running on Slurm (optional)

Use the provided `process.sbatch` as a template. Example:
```bash
sbatch process.sbatch
```
Key lines to update: job name, partition, GPU requirements, conda/env activation, and the Python entrypoint.

## Development

- Python style: basic, no enforced formatter in repo. Optional dev tools are commented in `requirements.txt`.
- Prefer explicit, reproducible runs via CLI flags and fixed seeds where available.

## License

Proprietary/Private. Do not distribute without permission.