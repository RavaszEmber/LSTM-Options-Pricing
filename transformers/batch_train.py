#!/usr/bin/env python
"""
batch_train.py - Train models for all 12 months and aggregate results
"""

import os
import sys
import argparse
import yaml
import pickle
import json
from pathlib import Path
from datetime import datetime
import subprocess
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Optional import for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Hyperparameter optimization disabled.")

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def optimize_hyperparameters(base_config, output_dir):
    """
    Run hyperparameter optimization on first month.
    
    Returns:
        Dictionary of optimized parameters
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Skipping hyperparameter optimization.")
        return {}
    
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    
    def objective(trial):
        # Create trial config (deep copy to avoid modifying base_config)
        import copy
        config = copy.deepcopy(base_config)
        search_space = config['hyperopt']['search_space']
        
        # Sample hyperparameters
        for param, bounds in search_space.items():
            if param in ['learning_rate', 'weight_decay']:
                config['training'][param] = trial.suggest_float(param, bounds[0], bounds[1], log=True)
            elif param == 'batch_size':
                config['training'][param] = trial.suggest_categorical(param, bounds)
            elif param in ['seq_len', 'label_len']:
                config['data'][param] = trial.suggest_int(param, bounds[0], bounds[1])
            elif param in config['model']:
                if isinstance(bounds, list) and len(bounds) == 2 and isinstance(bounds[0], (int, float)):
                    if isinstance(bounds[0], int):
                        config['model'][param] = trial.suggest_int(param, bounds[0], bounds[1])
                    else:
                        config['model'][param] = trial.suggest_float(param, bounds[0], bounds[1])
                else:
                    config['model'][param] = trial.suggest_categorical(param, bounds)
        
        # Update min_chain_length based on model type
        if config['model']['name'] == 'encoder_only_transformer':
            # Encoder-only doesn't use label_len
            config['data']['min_chain_length'] = config['data']['seq_len'] + config['data']['pred_len']
        else:
            # Standard encoder-decoder calculation
            config['data']['min_chain_length'] = config['data']['seq_len'] + config['data']['label_len'] + config['data']['pred_len']
        
        # Set to first month with reduced epochs for optimization
        config['data']['test_month'] = config['hyperopt']['optimize_on_month']
        config['training']['epochs'] = 30  # Reduced for hyperparameter optimization
        config['training']['early_stopping_patience'] = 10  # Reduced for faster trials
        config['output']['save_dir'] = f"{output_dir}/trial_{trial.number}"
        
        # Save trial config
        trial_config_path = f"{output_dir}/trial_{trial.number}_config.yaml"
        os.makedirs(os.path.dirname(trial_config_path), exist_ok=True)
        with open(trial_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run training
        success = train_single_model(trial_config_path, num_gpus=8)
        if not success[0]:
            return float('inf')
        
        # Extract RMSE
        results = load_results(config['output']['save_dir'])
        if results is None:
            return float('inf')
        
        return results['results']['rmse']
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Optimize with error handling per trial
    n_trials = base_config['hyperopt']['trials']
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    
    def safe_objective(trial):
        try:
            return objective(trial)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e} - SKIPPING")
            return float('inf')
    
    study.optimize(safe_objective, n_trials=n_trials)
    
    # Save best parameters
    best_params = study.best_params
    best_rmse = study.best_value
    
    print(f"Optimization complete. Best RMSE: ${best_rmse:.2f}")
    print(f"Best parameters: {best_params}")
    
    # Save results
    with open(f"{output_dir}/optimization_results.json", 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_rmse': best_rmse,
            'n_trials': len(study.trials)
        }, f, indent=2)
    
    return best_params


def create_monthly_configs(base_config_path, output_dir, year=2023):
    """
    Create 12 monthly config files from base config.
    Includes hyperparameter optimization if enabled.
    
    Args:
        base_config_path: Path to base config file
        output_dir: Directory to save monthly configs
        year: Year for testing
        
    Returns:
        List of (month, config_path) tuples
    """
    base_config = load_config(base_config_path)
    
    os.makedirs(output_dir, exist_ok=True)
    monthly_configs = []
    
    month_names = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    # Run hyperparameter optimization if enabled
    optimized_params = {}
    if base_config.get('hyperopt', {}).get('enabled', False):
        print("="*80)
        print("HYPERPARAMETER OPTIMIZATION ENABLED")
        print("="*80)
        hyperopt_dir = os.path.join(output_dir, 'hyperopt')
        optimized_params = optimize_hyperparameters(base_config, hyperopt_dir)
    
    for month in range(1, 13):
        # Create config for this month
        config = base_config.copy()
        config['data']['test_month'] = month
        config['data']['test_year'] = year
        
        # Apply optimized parameters
        if optimized_params:
            for param, value in optimized_params.items():
                if param in ['learning_rate', 'batch_size', 'weight_decay']:
                    config['training'][param] = value
                elif param in ['seq_len', 'label_len']:
                    config['data'][param] = value
                elif param in config['model']:
                    config['model'][param] = value
            
            # Update min_chain_length based on model type
            if config['model']['name'] == 'encoder_only_transformer':
                # Encoder-only doesn't use label_len
                config['data']['min_chain_length'] = config['data']['seq_len'] + config['data']['pred_len']
            else:
                # Standard encoder-decoder calculation
                config['data']['min_chain_length'] = config['data']['seq_len'] + config['data']['label_len'] + config['data']['pred_len']
        
        # Remove hyperopt section from monthly configs
        if 'hyperopt' in config:
            del config['hyperopt']
        
        # Update output directory
        model_name = config['model']['name']
        config['output']['save_dir'] = f"./results/{model_name}/month_{month:02d}_{month_names[month-1]}"
        
        # Save config
        config_filename = f"config_month_{month:02d}.yaml"
        config_path = os.path.join(output_dir, config_filename)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        monthly_configs.append((month, month_names[month-1], config_path))
        if not optimized_params:  # Only print if not optimizing
            print(f"Created config for {month_names[month-1]} ({month}): {config_path}")
    
    if optimized_params:
        print(f"\nCreated 12 monthly configs with optimized parameters")
        print(f"Optimization results saved to: {output_dir}/hyperopt/optimization_results.json")
    
    return monthly_configs


def train_single_model(config_path, num_gpus=8, master_port=29500):
    """
    Train a single model using torchrun.
    
    Args:
        config_path: Path to config file
        num_gpus: Number of GPUs to use
        master_port: Master port for distributed training
        
    Returns:
        (success: bool, time_elapsed: float)
    """
    start_time = time.time()
    
    cmd = [
        'torchrun',
        f'--nproc_per_node={num_gpus}',
        f'--master_port={master_port}',
        'train.py',
        '--config', config_path
    ]
    
    print(f"{'='*80}")
    print(f"Starting training: {config_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"\nTraining completed successfully in {elapsed:.2f} seconds")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\nTraining failed after {elapsed:.2f} seconds")
        print(f"Error: {e}")
        return False, elapsed


def load_results(results_dir):
    """
    Load training results from a directory.
    
    Args:
        results_dir: Directory containing saved results
        
    Returns:
        Dictionary with model results or None if not found
    """
    # Find the most recent results file
    results_files = list(Path(results_dir).glob('*_results_*.pkl'))
    
    if not results_files:
        print(f"No results found in {results_dir}")
        return None
    
    # Get most recent file
    latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_file, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {latest_file}: {e}")
        return None


def aggregate_results(monthly_configs, base_results_dir):
    """
    Aggregate results from all monthly models.
    
    Args:
        monthly_configs: List of (month, month_name, config_path) tuples
        base_results_dir: Base directory containing monthly results
        
    Returns:
        DataFrame with aggregated results
    """
    all_results = []
    
    for month, month_name, config_path in monthly_configs:
        config = load_config(config_path)
        results_dir = config['output']['save_dir']
        
        print(f"Loading results for {month_name}...")
        results = load_results(results_dir)
        
        if results is None:
            print(f"Skipping {month_name} - no results found")
            continue
        
        # Extract metrics
        row = {
            'month': month,
            'month_name': month_name,
            'rmse': results['results']['rmse'],
            'mae': results['results']['mae'],
            'mse': results['results']['mse'],
            'theil_u1': results['results']['theil_u1'],
            'epochs_trained': results['history']['epochs_trained'],
            'best_val_loss': results['history']['best_val_loss'],
            'final_train_loss': results['history']['train_losses'][-1],
            'training_time': results['history']['training_time'],
            'results_dir': results_dir
        }
        
        # Store predictions and targets for later analysis
        row['predictions'] = results['results']['predictions']
        row['targets'] = results['results']['targets']
        
        all_results.append(row)
    
    df = pd.DataFrame(all_results)
    return df


def plot_aggregate_metrics(df, output_dir):
    """
    Create comprehensive plots of aggregate performance.
    
    Args:
        df: DataFrame with monthly results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Monthly RMSE Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(df['month_name'], df['rmse'], color='steelblue', alpha=0.8, edgecolor='black')
    ax.axhline(y=df['rmse'].mean(), color='red', linestyle='--', 
               label=f'Mean RMSE: ${df["rmse"].mean():.2f}', linewidth=2)
    ax.set_xlabel('Test Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
    ax.set_title('Root Mean Squared Error by Test Month', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_rmse.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Multiple Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RMSE
    axes[0, 0].bar(df['month_name'], df['rmse'], color='steelblue', alpha=0.8, edgecolor='black')
    axes[0, 0].axhline(y=df['rmse'].mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_ylabel('RMSE ($)', fontweight='bold')
    axes[0, 0].set_title('Root Mean Squared Error', fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # MAE
    axes[0, 1].bar(df['month_name'], df['mae'], color='darkorange', alpha=0.8, edgecolor='black')
    axes[0, 1].axhline(y=df['mae'].mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_ylabel('MAE ($)', fontweight='bold')
    axes[0, 1].set_title('Mean Absolute Error', fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Theil U
    axes[1, 0].bar(df['month_name'], df['theil_u1'], color='green', alpha=0.8, edgecolor='black')
    axes[1, 0].axhline(y=df['theil_u1'].mean(), color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_ylabel('Theil U1', fontweight='bold')
    axes[1, 0].set_title('Theil U1 Coefficient', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Training Time
    axes[1, 1].bar(df['month_name'], df['training_time']/60, color='purple', alpha=0.8, edgecolor='black')
    axes[1, 1].axhline(y=(df['training_time']/60).mean(), color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_ylabel('Time (minutes)', fontweight='bold')
    axes[1, 1].set_title('Training Time', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Predictions vs Actuals for All Months (Combined)
    fig, ax = plt.subplots(figsize=(12, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 12))
    
    for idx, row in df.iterrows():
        predictions = row['predictions'].flatten()
        targets = row['targets'].flatten()
        
        # Sample for visualization
        sample_size = min(1000, len(predictions))
        indices = np.random.choice(len(predictions), sample_size, replace=False)
        
        ax.scatter(targets[indices], predictions[indices], 
                  alpha=0.4, s=20, label=row['month_name'], 
                  color=colors[idx])
    
    # Perfect prediction line
    min_val = min(df['targets'].apply(lambda x: x.min()).min(), 
                  df['predictions'].apply(lambda x: x.min()).min())
    max_val = max(df['targets'].apply(lambda x: x.max()).max(), 
                  df['predictions'].apply(lambda x: x.max()).max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Option Price ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Option Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('Predictions vs Actuals - All Months', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_actuals_all.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Error Distribution
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, row in df.iterrows():
        predictions = row['predictions'].flatten()
        targets = row['targets'].flatten()
        errors = predictions - targets
        
        axes[idx].hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[idx].axvline(x=errors.mean(), color='green', linestyle='--', 
                         linewidth=2, label=f'Mean: ${errors.mean():.2f}')
        axes[idx].set_xlabel('Prediction Error ($)', fontweight='bold')
        axes[idx].set_ylabel('Frequency', fontweight='bold')
        axes[idx].set_title(f'{row["month_name"]} - RMSE: ${row["rmse"]:.2f}', fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Box Plot of Errors
    fig, ax = plt.subplots(figsize=(14, 6))
    
    error_data = []
    labels = []
    for idx, row in df.iterrows():
        predictions = row['predictions'].flatten()
        targets = row['targets'].flatten()
        errors = predictions - targets
        error_data.append(errors)
        labels.append(row['month_name'])
    
    bp = ax.boxplot(error_data, labels=labels, patch_artist=True,
                    medianprops=dict(color='red', linewidth=2),
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Test Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Error ($)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Prediction Errors by Month', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def generate_summary_report(df, output_dir):
    """
    Generate comprehensive summary report.
    
    Args:
        df: DataFrame with monthly results
        output_dir: Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80)
        f.write("12-MONTH OPTION PRICING MODEL PERFORMANCE REPORT\n")
        f.write("="*80)
        
        # Overall Statistics
        f.write("AGGREGATE PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean RMSE:        ${df['rmse'].mean():>10.2f}  (±${df['rmse'].std():.2f})\n")
        f.write(f"Mean MAE:         ${df['mae'].mean():>10.2f}  (±${df['mae'].std():.2f})\n")
        f.write(f"Mean Theil U1:    {df['theil_u1'].mean():>11.6f}  (±{df['theil_u1'].std():.6f})\n")
        f.write(f"Total Train Time: {df['training_time'].sum()/3600:>10.2f} hours\n")
        f.write(f"Avg Train Time:   {df['training_time'].mean()/60:>10.2f} minutes per month\n")
        f.write("\n")
        
        # Best/Worst Months
        f.write("BEST AND WORST PERFORMING MONTHS\n")
        f.write("-"*80 + "\n")
        best_rmse = df.loc[df['rmse'].idxmin()]
        worst_rmse = df.loc[df['rmse'].idxmax()]
        f.write(f"Best RMSE:  {best_rmse['month_name']:<12} ${best_rmse['rmse']:.2f}\n")
        f.write(f"Worst RMSE: {worst_rmse['month_name']:<12} ${worst_rmse['rmse']:.2f}\n")
        f.write("\n")
        
        # Monthly Breakdown
        f.write("MONTHLY PERFORMANCE BREAKDOWN")
        f.write("-"*80)
        f.write(f"{'Month':<12} {'RMSE ($)':>10} {'MAE ($)':>10} {'Theil U1':>10} {'Epochs':>8} {'Time (min)':>12}\n")
        f.write("-"*80)
        
        for _, row in df.iterrows():
            f.write(f"{row['month_name']:<12} "
                   f"{row['rmse']:>10.2f} "
                   f"{row['mae']:>10.2f} "
                   f"{row['theil_u1']:>10.6f} "
                   f"{row['epochs_trained']:>8} "
                   f"{row['training_time']/60:>12.2f}\n")
        
        f.write("\n")
        
        # Statistical Tests
        f.write("STATISTICAL ANALYSIS")
        f.write("-"*80)
        
        # Normality test on RMSE
        _, p_value = stats.shapiro(df['rmse'])
        f.write(f"Shapiro-Wilk Test (RMSE normality): p-value = {p_value:.4f}\n")
        if p_value > 0.05:
            f.write("  → RMSE values appear normally distributed\n")
        else:
            f.write("  → RMSE values may not be normally distributed\n")
        
        f.write("\n")
        
        # Coefficient of Variation
        cv_rmse = (df['rmse'].std() / df['rmse'].mean()) * 100
        cv_mae = (df['mae'].std() / df['mae'].mean()) * 100
        f.write(f"Coefficient of Variation (RMSE): {cv_rmse:.2f}%\n")
        f.write(f"Coefficient of Variation (MAE):  {cv_mae:.2f}%\n")
        f.write("\n")
        
        # Percentiles
        f.write("RMSE PERCENTILES")
        f.write("-"*80)
        for percentile in [25, 50, 75, 90, 95]:
            value = np.percentile(df['rmse'], percentile)
            f.write(f"{percentile}th percentile: ${value:.2f}\n")
        
        f.write("="*80)
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        f.write("="*80)
    
    print(f"Summary report saved to {report_path}")
    
    # Also save as JSON for programmatic access
    json_path = os.path.join(output_dir, 'summary_metrics.json')
    summary = {
        'aggregate': {
            'mean_rmse': float(df['rmse'].mean()),
            'std_rmse': float(df['rmse'].std()),
            'mean_mae': float(df['mae'].mean()),
            'std_mae': float(df['mae'].std()),
            'mean_theil_u1': float(df['theil_u1'].mean()),
            'std_theil_u1': float(df['theil_u1'].std()),
            'total_training_time_hours': float(df['training_time'].sum()/3600)
        },
        'monthly': df[['month', 'month_name', 'rmse', 'mae', 'theil_u1', 
                      'epochs_trained', 'training_time']].to_dict('records')
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"JSON summary saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch train models for all 12 months')
    parser.add_argument('--config', type=str, required=True,
                       help='Base config file')
    parser.add_argument('--num_gpus', type=int, default=8,
                       help='Number of GPUs to use')
    parser.add_argument('--master_port', type=int, default=29500,
                       help='Master port for distributed training')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and only generate reports from existing results')
    parser.add_argument('--output_dir', type=str, default='./batch_results',
                       help='Directory for batch results and reports')
    parser.add_argument('--year', type=int, default=2023,
                       help='Year for testing')
    args = parser.parse_args()
    
    print("="*80)
    print("BATCH TRAINING: 12-MONTH OPTION PRICING MODELS")
    print("="*80)
    print(f"Base config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Test year: {args.year}")
    print("="*80)
    
    # Create monthly configs
    configs_dir = os.path.join(args.output_dir, 'configs')
    monthly_configs = create_monthly_configs(args.config, configs_dir, args.year)
    
    if not args.skip_training:
        # Train all models
        training_log = []
        
        for month, month_name, config_path in monthly_configs:
            print(f"\n{'#'*80}")
            print(f"# TRAINING MODEL {month}/12: {month_name.upper()}")
            print(f"{'#'*80}\n")
            
            success, elapsed = train_single_model(
                config_path, 
                args.num_gpus, 
                args.master_port
            )
            
            training_log.append({
                'month': month,
                'month_name': month_name,
                'success': success,
                'time_elapsed': elapsed
            })
            
            if not success:
                print(f"Warning: Training failed for {month_name}")
                print("Continuing with next month...\n")
        
        # Save training log
        log_path = os.path.join(args.output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"Training log saved to {log_path}")
    
    # Aggregate results
    print("="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    
    df = aggregate_results(monthly_configs, args.output_dir)
    
    if df.empty:
        print("❌ No results found to aggregate!")
        return
    
    # Save aggregated results
    csv_path = os.path.join(args.output_dir, 'aggregated_results.csv')
    df_export = df.drop(['predictions', 'targets'], axis=1)  # Don't export arrays to CSV
    df_export.to_csv(csv_path, index=False)
    print(f"Aggregated results saved to {csv_path}")
    
    # Generate plots
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plots_dir = os.path.join(args.output_dir, 'plots')
    plot_aggregate_metrics(df, plots_dir)
    
    # Generate summary report
    print("="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    reports_dir = os.path.join(args.output_dir, 'reports')
    generate_summary_report(df, reports_dir)
    
    print("="*80)
    print("BATCH TRAINING COMPLETE!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"  - Configurations: {configs_dir}")
    print(f"  - Plots: {plots_dir}")
    print(f"  - Reports: {reports_dir}")
    print(f"  - Aggregated CSV: {csv_path}")
    print("="*80)


if __name__ == '__main__':
    main()

