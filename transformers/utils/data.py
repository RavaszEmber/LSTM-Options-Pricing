"""Data loading and preprocessing utilities"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


def add_option_lifecycle_features(df):
    """
    Add features that capture option lifecycle regardless of specific expiration date.

    Args:
        df: DataFrame with options data (must have QUOTE_DATE, EXPIRE_DATE, STRIKE, UNDERLYING_LAST)

    Returns:
        DataFrame with additional features: DTE, DTE_NORMALIZED, MONEYNESS, MONEYNESS_BUCKET, EXPIRY_HORIZON
    """
    # Days to expiration
    df['DTE'] = (df['EXPIRE_DATE'] - df['QUOTE_DATE']).dt.days

    # Normalized time to expiration (0 = expiration, 1 = at listing)
    # Grouped by contract to get max DTE for each option
    df['DTE_NORMALIZED'] = df.groupby(['STRIKE', 'EXPIRE_DATE'])['DTE'].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )

    # Moneyness (how in/out of the money)
    df['MONEYNESS'] = df['STRIKE'] / df['UNDERLYING_LAST']

    # Moneyness buckets for grouping (can adjust thresholds based on your data)
    df['MONEYNESS_BUCKET'] = pd.cut(
        df['MONEYNESS'],
        bins=[0, 0.90, 0.95, 1.00, 1.05, 1.10, np.inf],
        labels=['deep_itm', 'itm', 'atm', 'otm', 'deep_otm', 'far_otm']
    )

    # Expiration horizon buckets
    df['EXPIRY_HORIZON'] = pd.cut(
        df['DTE'],
        bins=[0, 30, 90, 180, 365, np.inf],
        labels=['short', 'medium', 'long', 'leap', 'ultra_leap']
    )

    return df


def encode_categorical_features(df, feature_columns):
    """
    One-hot encode categorical features for model input.

    Args:
        df: DataFrame with categorical columns
        feature_columns: List of feature column names

    Returns:
        Tuple of (df, updated_feature_columns)
    """
    updated_features = feature_columns.copy()

    # One-hot encode EXPIRY_HORIZON if it exists and is in features
    if 'EXPIRY_HORIZON' in df.columns:
        horizon_dummies = pd.get_dummies(df['EXPIRY_HORIZON'], prefix='HORIZON', dtype=float)
        df = pd.concat([df, horizon_dummies], axis=1)

        # If EXPIRY_HORIZON was in features, replace with dummy columns
        if 'EXPIRY_HORIZON' in updated_features:
            idx = updated_features.index('EXPIRY_HORIZON')
            updated_features.pop(idx)
            updated_features.extend(list(horizon_dummies.columns))

    # Encode MONEYNESS_BUCKET as ordinal if used as feature
    if 'MONEYNESS_BUCKET' in updated_features:
        moneyness_map = {
            'deep_itm': 0, 'itm': 1, 'atm': 2, 
            'otm': 3, 'deep_otm': 4, 'far_otm': 5
        }
        df['MONEYNESS_BUCKET_ENCODED'] = df['MONEYNESS_BUCKET'].map(moneyness_map).astype(float)

        # Replace MONEYNESS_BUCKET with encoded version
        idx = updated_features.index('MONEYNESS_BUCKET')
        updated_features[idx] = 'MONEYNESS_BUCKET_ENCODED'

    return df, updated_features


def load_preprocessed_data(filepath, rank):
    """
    Load preprocessed options data and add lifecycle features.

    Args:
        filepath: Path to CSV file
        rank: Process rank for distributed training

    Returns:
        DataFrame with original and lifecycle features
    """
    from .training import is_main_process

    if is_main_process(rank):
        print(f"Loading data from {filepath}...")

    df = pd.read_csv(filepath, parse_dates=['QUOTE_DATE', 'EXPIRE_DATE'])

    # Add lifecycle features
    df = add_option_lifecycle_features(df)

    if is_main_process(rank):
        print(f"Loaded {len(df):,} call option records")
        print(f"Date range: {df['QUOTE_DATE'].min()} to {df['QUOTE_DATE'].max()}")

        # Show distribution of new features
        print(f"\nExpiration Horizon Distribution:")
        horizon_counts = df['EXPIRY_HORIZON'].value_counts().sort_index()
        for horizon, count in horizon_counts.items():
            pct = 100 * count / len(df)
            print(f"  {horizon:>12}: {count:>10,} ({pct:>5.1f}%)")

        print(f"\nMoneyness Distribution:")
        moneyness_counts = df['MONEYNESS_BUCKET'].value_counts().sort_index()
        for bucket, count in moneyness_counts.items():
            pct = 100 * count / len(df)
            print(f"  {bucket:>12}: {count:>10,} ({pct:>5.1f}%)")

        print(f"\nDTE Statistics:")
        print(f"  Min:    {df['DTE'].min():>6} days")
        print(f"  Max:    {df['DTE'].max():>6} days")
        print(f"  Mean:   {df['DTE'].mean():>6.1f} days")
        print(f"  Median: {df['DTE'].median():>6.1f} days")

    return df


def create_option_sequences(
    df: pd.DataFrame,
    features: List[str],
    date_col: str,
    target_col: str,
    seq_len: int,
    pred_len: int = 1,
    label_len: int = 0,
    group_by: List[str] = None,
    min_chain_length: int = None,
    stratify_by_horizon: bool = False,
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[pd.Timestamp], List[dict]]:
    """
    Create sequences WITHOUT data leakage for option pricing.
    Groups by (STRIKE, EXPIRE_DATE) or other columns to keep option contracts separate.

    Args:
        df: DataFrame with options data (already sorted by date)
        features: List of feature column names
        date_col: Name of date column
        target_col: Name of target column
        seq_len: Length of encoder input sequence
        pred_len: Prediction horizon (how many steps ahead to predict)
        label_len: Length of decoder input overlap with encoder
        group_by: Columns to group by. Default: ["STRIKE", "MONEYNESS_BUCKET"]
        min_chain_length: Minimum observations required per group
        stratify_by_horizon: Track sequences by expiration horizon
        verbose: Print detailed statistics

    Returns:
        Tuple of (X_enc_list, X_dec_list, y_list, dates_list, metadata_list)
    """
    X_enc_list = []
    X_dec_list = []
    y_list = []
    dates_list = []
    metadata_list = []

    # Default grouping strategy
    if group_by is None:
        group_by = ["STRIKE", "MONEYNESS_BUCKET"]

    if min_chain_length is None:
        min_chain_length = seq_len + label_len + pred_len

    # Statistics tracking
    stats = {
        'total_groups': 0,
        'skipped_short': 0,
        'skipped_no_sequences': 0,
        'processed_groups': 0,
        'sequences_by_horizon': {}
    }

    # Group by specified columns
    for group_key, df_group in df.groupby(group_by):
        stats['total_groups'] += 1
        df_group = df_group.sort_values(date_col).reset_index(drop=True)

        # Extract features and targets
        feature_vals = df_group[features].to_numpy(dtype=np.float32)
        target_vals = df_group[target_col].to_numpy(dtype=np.float32)
        n_observations = len(df_group)

        # Check minimum length
        if n_observations < min_chain_length:
            stats['skipped_short'] += 1
            continue

        sequences_before = len(X_enc_list)

        # Create sliding windows
        for enc_end_idx in range(seq_len - 1, n_observations - pred_len):
            enc_start_idx = enc_end_idx - (seq_len - 1)

            # Encoder input: historical window [t-seq_len+1, ..., t]
            x_enc = feature_vals[enc_start_idx:enc_end_idx + 1]

            # Decoder input: overlapping + future window
            dec_start_idx = enc_end_idx - label_len + 1
            dec_end_idx = enc_end_idx + pred_len + 1

            # Check bounds
            if dec_end_idx > n_observations:
                break

            x_dec = feature_vals[dec_start_idx:dec_end_idx]

            # Target: future values [t+1, ..., t+pred_len]
            target_start_idx = enc_end_idx + 1
            target_end_idx = enc_end_idx + pred_len + 1
            y = target_vals[target_start_idx:target_end_idx]

            # Validate shapes
            if x_enc.shape[0] != seq_len:
                continue
            if x_dec.shape[0] != (label_len + pred_len):
                continue
            if y.shape[0] != pred_len:
                continue

            X_enc_list.append(x_enc)
            X_dec_list.append(x_dec)
            y_list.append(y)
            dates_list.append(df_group.iloc[enc_end_idx][date_col])

            # Store metadata if stratifying by horizon
            if stratify_by_horizon and 'EXPIRY_HORIZON' in df_group.columns:
                horizon = str(df_group.iloc[enc_end_idx]['EXPIRY_HORIZON'])
                dte = df_group.iloc[enc_end_idx]['DTE']

                metadata_list.append({
                    'group_key': group_key,
                    'horizon': horizon,
                    'dte': dte
                })

                # Track sequences by horizon
                if horizon not in stats['sequences_by_horizon']:
                    stats['sequences_by_horizon'][horizon] = 0
                stats['sequences_by_horizon'][horizon] += 1

        # Check if group produced sequences
        if len(X_enc_list) == sequences_before:
            stats['skipped_no_sequences'] += 1
        else:
            stats['processed_groups'] += 1

    # Print statistics
    if verbose:
        print(f"\n{'='*60}")
        print("Sequential Data Creation Statistics")
        print(f"{'='*60}")
        print(f"Grouping by: {group_by}")
        print(f"Total groups: {stats['total_groups']:,}")
        print(f"  Skipped (too short &lt; {min_chain_length}): {stats['skipped_short']:,} ({100*stats['skipped_short']/max(stats['total_groups'],1):.1f}%)")
        print(f"  Skipped (no valid sequences): {stats['skipped_no_sequences']:,} ({100*stats['skipped_no_sequences']/max(stats['total_groups'],1):.1f}%)")
        print(f"  Processed successfully: {stats['processed_groups']:,} ({100*stats['processed_groups']/max(stats['total_groups'],1):.1f}%)")
        print(f"Total sequences created: {len(X_enc_list):,}")

        if stats['sequences_by_horizon']:
            print(f"\nSequences by Expiration Horizon:")
            for horizon in sorted(stats['sequences_by_horizon'].keys()):
                count = stats['sequences_by_horizon'][horizon]
                pct = 100 * count / len(X_enc_list)
                print(f"  {horizon:>12}: {count:>8,} ({pct:>5.1f}%)")

        if stats['processed_groups'] > 0:
            print(f"\nAvg sequences per group: {len(X_enc_list)/stats['processed_groups']:.1f}")
        print(f"{'='*60}\n")

    return X_enc_list, X_dec_list, y_list, dates_list, metadata_list


def validate_horizon_coverage(train_metadata, val_metadata, test_metadata, rank, skip_on_empty=False):
    """
    Ensure all expiration horizons are represented in train/val/test splits.
    
    Args:
        train_metadata: List of metadata dicts from training data
        val_metadata: List of metadata dicts from validation data
        test_metadata: List of metadata dicts from test data
        rank: Process rank for distributed training
        skip_on_empty: If True, return False instead of raising error for empty test horizons
        
    Returns:
        bool: True if validation passes, False if should skip (when skip_on_empty=True)
    """
    from .training import is_main_process

    if not is_main_process(rank):
        return True

    train_horizons = set(m['horizon'] for m in train_metadata if m and 'horizon' in m)
    val_horizons = set(m['horizon'] for m in val_metadata if m and 'horizon' in m)
    test_horizons = set(m['horizon'] for m in test_metadata if m and 'horizon' in m)

    print(f"\n{'='*60}")
    print("Expiration Horizon Coverage")
    print(f"{'='*60}")
    print(f"Train horizons: {sorted(train_horizons)}")
    print(f"Val horizons:   {sorted(val_horizons)}")
    print(f"Test horizons:  {sorted(test_horizons)}")

    # Check for empty test horizons
    if not test_horizons:
        if skip_on_empty:
            print("WARNING: Test set has no expiration horizons - SKIPPING")
            print(f"{'='*60}\n")
            return False
        else:
            raise ValueError("Test set has no expiration horizons! Increase min_chain_length or check data filtering.")
    
    missing_val = train_horizons - val_horizons
    missing_test = train_horizons - test_horizons

    if missing_val:
        print(f"WARNING: Validation missing horizons: {missing_val}")
    if missing_test:
        print(f"WARNING: Test missing horizons: {missing_test}")
        print(f"Consider reducing min_chain_length or adjusting date ranges")

    if not missing_val and not missing_test:
        print("All horizons represented in train/val/test")

    print(f"{'='*60}\n")
    return True


def ensure_minimum_horizon_coverage(df, min_sequences_per_horizon=10):
    """
    Filter data to ensure minimum sequences per horizon.
    
    Args:
        df: DataFrame with EXPIRY_HORIZON column
        min_sequences_per_horizon: Minimum sequences required per horizon
        
    Returns:
        Filtered DataFrame
    """
    if 'EXPIRY_HORIZON' not in df.columns:
        return df
    
    # Count potential sequences per horizon
    horizon_counts = df['EXPIRY_HORIZON'].value_counts()
    valid_horizons = horizon_counts[horizon_counts >= min_sequences_per_horizon].index
    
    if len(valid_horizons) == 0:
        print(f"WARNING: No horizons have >= {min_sequences_per_horizon} sequences")
        return df
    
    # Filter to valid horizons only
    filtered_df = df[df['EXPIRY_HORIZON'].isin(valid_horizons)].copy()
    
    print(f"Filtered to {len(valid_horizons)} horizons with >= {min_sequences_per_horizon} sequences each")
    print(f"Kept {len(filtered_df):,} / {len(df):,} records ({100*len(filtered_df)/len(df):.1f}%)")
    
    return filtered_df


def create_rolling_window_split(
    df, 
    test_month, 
    test_year, 
    feature_columns, 
    target_column, 
    train_months, 
    val_months, 
    rank,
    sequential=False,
    seq_len=96,
    label_len=48,
    pred_len=1,
    group_by=None,
    min_chain_length=None,
    stratify_by_horizon=False
):
    """
    Create train/val/test splits with proper scaling.

    Args:
        sequential: If True, create sequences grouped by option contracts (no leak)
                   If False, use simple sliding windows (original behavior)
        group_by: Columns to group by when sequential=True
        stratify_by_horizon: Track and validate horizon distribution
    """
    from .training import is_main_process

    # Compute target column from bid-ask midpoint
    df[target_column] = df[['C_BID', 'C_ASK']].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    # random perturb
    df["VOL_RANDOM_PERTURB"] = df["VOL_90D"] * (1 + np.random.normal(0, 0.1, len(df)))

    df["VOL_RANDOM"] = np.random.normal(0, 0.1, len(df))

    # constant
    df["VOL_CONSTANT"] = df["VOL_90D"].mean()

    # Encode categorical features if present
    if sequential and group_by:
        df, feature_columns = encode_categorical_features(df, feature_columns)

    # Define date ranges
    test_start = pd.Timestamp(year=test_year, month=test_month, day=1)
    test_end = test_start + pd.offsets.MonthEnd(0)
    val_start = test_start - pd.offsets.MonthBegin(val_months)
    val_end = test_start - pd.Timedelta(days=1)
    train_start = val_start - pd.offsets.MonthBegin(train_months)
    train_end = val_start - pd.Timedelta(days=1)

    if is_main_process(rank):
        print(f"\n{'='*60}")
        print("Data Split Configuration")
        print(f"{'='*60}")
        print(f"Train: {train_start.date()} to {train_end.date()} ({train_months} months)")
        print(f"Val:   {val_start.date()} to {val_end.date()} ({val_months} month)")
        print(f"Test:  {test_start.date()} to {test_end.date()} (1 month)")
        print(f"{'='*60}\n")

    # Filter data by date ranges
    train_df = df[
        (df['QUOTE_DATE'] >= train_start) & (df['QUOTE_DATE'] <= train_end)
    ].copy()

    val_df = df[
        (df['QUOTE_DATE'] >= val_start) & (df['QUOTE_DATE'] <= val_end)
    ].copy()

    test_df = df[
        (df['QUOTE_DATE'] >= test_start) & (df['QUOTE_DATE'] <= test_end)
    ].copy()
    
    # Ensure minimum horizon coverage if stratifying
    if sequential and stratify_by_horizon:
        if is_main_process(rank):
            print("Ensuring minimum horizon coverage...")
        train_df = ensure_minimum_horizon_coverage(train_df, min_sequences_per_horizon=20)
        val_df = ensure_minimum_horizon_coverage(val_df, min_sequences_per_horizon=5)
        test_df = ensure_minimum_horizon_coverage(test_df, min_sequences_per_horizon=5)

    if sequential:
        # Use leak-free sequential approach grouped by option contracts
        if is_main_process(rank):
            print("Creating sequences with option contract grouping...")
            if group_by:
                print(f"Grouping by: {group_by}")

        # Fit scaler on train features only
        scaler = MinMaxScaler()
        train_features = train_df[feature_columns + [target_column]].values
        scaler.fit(train_features)

        # Create sequences for each split
        train_data = _prepare_sequential_data(
            train_df, feature_columns, target_column, scaler, 
            seq_len, label_len, pred_len, rank, split_name="Train",
            group_by=group_by,
            min_chain_length=min_chain_length,
            stratify_by_horizon=stratify_by_horizon
        )

        val_data = _prepare_sequential_data(
            val_df, feature_columns, target_column, scaler,
            seq_len, label_len, pred_len, rank, split_name="Val",
            group_by=group_by,
            min_chain_length=min_chain_length,
            stratify_by_horizon=stratify_by_horizon
        )

        test_data = _prepare_sequential_data(
            test_df, feature_columns, target_column, scaler,
            seq_len, label_len, pred_len, rank, split_name="Test",
            group_by=group_by,
            min_chain_length=min_chain_length,
            stratify_by_horizon=stratify_by_horizon
        )

        # Validate horizon coverage if stratifying
        if stratify_by_horizon:
            horizon_valid = validate_horizon_coverage(
                train_data.get('metadata', []),
                val_data.get('metadata', []),
                test_data.get('metadata', []),
                rank,
                skip_on_empty=True
            )
            if not horizon_valid:
                # Return None to signal skip
                return None, None, None, scaler

        return train_data, val_data, test_data, scaler

    else:
        # Original behavior: simple time-based split with scaling
        if is_main_process(rank):
            print("Creating simple time-based splits (original method)...")

        train_df = train_df[
            ['QUOTE_DATE'] + feature_columns + [target_column]
        ].set_index('QUOTE_DATE')

        val_df = val_df[
            ['QUOTE_DATE'] + feature_columns + [target_column]
        ].set_index('QUOTE_DATE')

        test_df = test_df[
            ['QUOTE_DATE'] + feature_columns + [target_column]
        ].set_index('QUOTE_DATE')

        # Fit scaler on train data
        scaler = MinMaxScaler()
        scaler.fit(train_df)

        # Scale all splits
        train_scaled = pd.DataFrame(
            scaler.transform(train_df),
            columns=train_df.columns,
            index=train_df.index
        )
        val_scaled = pd.DataFrame(
            scaler.transform(val_df),
            columns=val_df.columns,
            index=val_df.index
        )
        test_scaled = pd.DataFrame(
            scaler.transform(test_df),
            columns=test_df.columns,
            index=test_df.index
        )

        if is_main_process(rank):
            print("Data splits created:")
            print(f"  Train: {len(train_scaled):,} samples")
            print(f"  Val:   {len(val_scaled):,} samples")
            print(f"  Test:  {len(test_scaled):,} samples")

        return train_scaled, val_scaled, test_scaled, scaler


def _prepare_sequential_data(
    df, 
    feature_columns, 
    target_column, 
    scaler, 
    seq_len, 
    label_len, 
    pred_len,
    rank,
    split_name,
    group_by=None,
    min_chain_length=None,
    stratify_by_horizon=False
):
    """Helper function to prepare sequential data with scaling"""
    from .training import is_main_process

    # Scale the data first
    df_scaled = df.copy()
    scaled_values = scaler.transform(df[feature_columns + [target_column]])
    df_scaled[feature_columns + [target_column]] = scaled_values

    # Create sequences
    X_enc, X_dec, y, dates, metadata = create_option_sequences(
        df=df_scaled,
        features=feature_columns,
        date_col='QUOTE_DATE',
        target_col=target_column,
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
        group_by=group_by,
        min_chain_length=min_chain_length,
        stratify_by_horizon=stratify_by_horizon,
        verbose=is_main_process(rank)
    )

    if is_main_process(rank):
        print(f"  {split_name}: {len(X_enc):,} sequences created")
        if len(X_enc) > 0:
            print(f"    Encoder shape: {X_enc[0].shape}")
            print(f"    Decoder shape: {X_dec[0].shape}")
            print(f"    Target shape: {y[0].shape}")

    return {
        'X_enc': X_enc,
        'X_dec': X_dec,
        'y': y,
        'dates': dates,
        'metadata': metadata
    }


class OptionPricingDataset(Dataset):
    """PyTorch Dataset for option pricing with sequence support"""

    def __init__(self, data_df, target_col, seq_len, label_len, pred_len, feature_cols):
        self.data = data_df[feature_cols].values.astype(np.float32)
        self.target = data_df[target_col].values.astype(np.float32)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.sequential = False

    @classmethod
    def from_sequences(cls, data_dict, seq_len, label_len, pred_len):
        """
        Alternative constructor for sequential data (pre-computed sequences)

        Args:
            data_dict: Dictionary with keys 'X_enc', 'X_dec', 'y', 'dates'
        """
        dataset = cls.__new__(cls)
        dataset.X_enc = data_dict['X_enc']
        dataset.X_dec = data_dict['X_dec']
        dataset.y = data_dict['y']
        dataset.dates = data_dict['dates']
        dataset.seq_len = seq_len
        dataset.label_len = label_len
        dataset.pred_len = pred_len
        dataset.sequential = True
        return dataset

    def __len__(self):
        if self.sequential:
            return len(self.X_enc)
        else:
            return len(self.data) - self.seq_len - self.label_len - self.pred_len + 1

    def __getitem__(self, idx):
        if self.sequential:
            # Pre-computed sequences (no leakage)
            return (
                torch.FloatTensor(self.X_enc[idx]),
                torch.FloatTensor(self.X_dec[idx]),
                torch.FloatTensor(self.y[idx])
            )
        else:
            # Original sliding window approach
            s_begin = idx
            s_end = s_begin + self.seq_len
            x_enc = self.data[s_begin:s_end]

            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            x_dec = self.data[r_begin:r_end]

            y = self.target[s_end:s_end + self.pred_len]

            return (
                torch.FloatTensor(x_enc),
                torch.FloatTensor(x_dec),
                torch.FloatTensor(y)
            )