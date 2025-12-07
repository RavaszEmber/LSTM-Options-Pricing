Purpose
-------
This file gives concise, repository-specific guidance for AI coding agents (Copilot-style) so you can be productive quickly. It documents the big-picture architecture, developer workflows, naming conventions, and a few concrete examples pointing to code to edit.

Key points (quick skim)
- Project type: PyTorch models for option pricing, with preprocessing in a Jupyter notebook and multiple transformer variants under `transformers/`.
- Data flow: raw text/csv files (DATA_DIR in `Preprocessing.ipynb`) -> run preprocessing notebook -> generates `./data.csv` -> training scripts (`train.py`, `transformers/train.py`) consume `data.csv` -> model artifacts saved in `checkpoints/`.
- Primary model code: `models/mlp.py` (simple MLP) and transformer implementations under `transformers/models/` (see `encoder_only_transformer.py`, `encoder_decoder_transformer.py`, `lstm_transformer.py`).

Environment & deps
- See `pyproject.toml`: Python >= 3.13 and core deps include `pandas`, `ruff`, `torch`.
- Additional packages used in notebooks / scripts but not declared in pyproject: `arch` (GARCH), `scipy`, `sklearn`, `matplotlib`, `numpy`. If running notebooks or preprocessing, ensure these are installed in the environment.

Where to start (files to open)
- Data preprocessing and feature engineering: `Preprocessing.ipynb` — contains `read_dataset`, `preprocess_options_df`, `interpolate_rates`, `create_volatility_features`, and writes `data.csv`.
- Training entry points: `train.py` (root) and `transformers/train.py` + shell runners in `transformers/` (`batch_train.sh`, `train.sh`, `runner.sh`).
- Model definitions: `models/mlp.py`, `transformers/models/*.py` (encoder/decoder/informer/lstm variants).
- Configs: `transformers/configs/*.yaml` — modify hyperparams here for transformer experiments.
- Tests: `tests/test_black_scholes.py` and `tests/test_heston.py` — unit tests for `black_scholes.py` and `heston.py`.

Conventions & patterns to follow
- Notebooks produce canonical datasets: `Preprocessing.ipynb` writes `./data.csv`. Training code expects this CSV in the repo root unless the training script's path is overridden.
- Preprocessing uses a helper `read_dataset(directory, file_pattern, label)` with glob patterns. When adding new data, match the existing glob and format expectations (column names, date parsing). Column-cleaning helpers: `_clean_column_names`, `_drop_columns`.
- Time-to-maturity (MTM) semantics: Many functions rely on `MTM` being months as computed in the notebook. If you change MTM calculation, update all downstream code (interpolation, `get_rate`, rolling-window splits).
- Scaling: `create_rolling_window_split` fits `MinMaxScaler()` on the train split and applies it to val/test. Preserve this behavior when adding features or modifying feature order.
- Checkpoint naming: checkpoints are grouped under `checkpoints/<model_name>/` with filenames like `YYYY_MM.pt` and `*_results.csv`. Keep that folder layout to remain compatible with evaluation scripts.

Tests & quick checks
- Unit tests are under `tests/`. Run them with your environment's pytest (not included in pyproject dependencies):
  - Recommended: use a virtualenv/conda with the versions from `pyproject.toml`, then `pip install pytest` if needed.
  - Run: `python -m pytest -q` from the repo root.
- Small sanity checks: open `data.csv` after preprocessing and confirm columns `QUOTE_DATE`, `EXPIRE_DATE`, `MTM`, `RFR`, `VOL_GG`, `VOL_90D`, `CALL`, `PUT` exist.

Editing guidance (concrete examples)
- To change rate interpolation logic: edit `interpolate_rate` / `get_rate` in `Preprocessing.ipynb` (cells labeled around "Interpolating Risk Free Rate...").
- To change volatility estimation: edit `estimate_volatility` or `estimate_gjr_garch_volatility` in `Preprocessing.ipynb`. Note: the notebook uses `arch.arch_model`; confirm `arch` is present in the environment.
- To add a new transformer config: copy a YAML from `transformers/configs/`, update hyperparameters, and run `transformers/train.py` or the provided `train.sh` in that folder.
- To modify model architecture: edit `models/mlp.py` for MLP changes or `transformers/models/*.py` for transformer variants. Keep input/output tensor shapes compatible with the training data loader.

Integration points & runtime expectations
- The notebooks assume a DATA_DIR path (set in `Preprocessing.ipynb` as `DATA_DIR = "/home/sagemaker-user/LSTM Project/CS7643ProjectData/"`). This is environment-specific; for local runs, set `DATA_DIR` to where raw data sits or place raw files next to the repo and adjust the glob.
- Checkpoints are consumed by evaluation scripts and some training wrappers expect consistent naming and metadata in the sibling `*_results.csv` files.

What the agent should not change without human sign-off
- Global date/MTM handling and scaling strategy. These affect model correctness and test expectations.
- pyproject.toml Python version (>3.13) — changing it may impact CI and reproducibility.

If anything is unclear or missing
- Ask which runtime you're targeting (local dev, SageMaker, Docker). I can expand this file with explicit setup commands, a requirements.txt, or a short developer-runbook (install steps, reproduce notebook runs, run training). Please tell me what you'd like next.

References
- `Preprocessing.ipynb`, `train.py`, `models/mlp.py`, `transformers/`, `transformers/configs/`, `tests/`, `pyproject.toml`, `README.md`
