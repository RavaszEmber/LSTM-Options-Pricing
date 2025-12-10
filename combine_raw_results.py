import pandas as pd
from pathlib import Path

checkpoint_dirs = sorted(Path("checkpoints").glob("*"))

for checkpoint_dir in checkpoint_dirs:
    if not checkpoint_dir.is_dir():
        continue

    raw_results_files = sorted(checkpoint_dir.glob("*_raw_results.csv"))

    if not raw_results_files:
        continue

    dfs = [pd.read_csv(f) for f in raw_results_files]
    combined = pd.concat(dfs, ignore_index=True)

    output_path = checkpoint_dir / "all_raw_results.csv"
    combined.to_csv(output_path, index=False)

    print(f"{checkpoint_dir.name}: Combined {len(raw_results_files)} files, {len(combined)} total rows")