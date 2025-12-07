#!/usr/bin/env python
"""Multi-GPU training script for transformer-based option pricing"""

import argparse
import yaml
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import warnings

# Local imports
from models import get_model
from utils.data import (
    load_preprocessed_data,
    create_rolling_window_split,
    OptionPricingDataset
)
from utils.training import (
    setup_distributed,
    cleanup_distributed,
    set_seed,
    is_main_process,
    train_epoch,
    validate_epoch,
    evaluate_model,
    save_results,
    plot_results
)

warnings.filterwarnings('ignore')


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config, args):
    """Merge config file with command line arguments"""
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.data_path is not None:
        config['data']['path'] = args.data_path
    if args.output_dir is not None:
        config['output']['save_dir'] = args.output_dir
    if args.seed is not None:
        config['hardware']['seed'] = args.seed
    
    return config


def create_dataloaders(train_data, val_data, test_data, config, rank, world_size, sequential=False):
    """Create distributed dataloaders"""
    
    if sequential:
        # Create datasets from pre-computed sequences (no leakage)
        train_dataset = OptionPricingDataset.from_sequences(
            train_data,
            seq_len=config['data']['seq_len'],
            label_len=config['data']['label_len'],
            pred_len=config['data']['pred_len']
        )

        val_dataset = OptionPricingDataset.from_sequences(
            val_data,
            seq_len=config['data']['seq_len'],
            label_len=config['data']['label_len'],
            pred_len=config['data']['pred_len']
        )

        test_dataset = OptionPricingDataset.from_sequences(
            test_data,
            seq_len=config['data']['seq_len'],
            label_len=config['data']['label_len'],
            pred_len=config['data']['pred_len']
        )
    else:
        # Original behavior: create datasets from DataFrames
        train_dataset = OptionPricingDataset(
            data_df=train_data,
            target_col=config['data']['target_column'],
            seq_len=config['data']['seq_len'],
            label_len=config['data']['label_len'],
            pred_len=config['data']['pred_len'],
            feature_cols=config['data']['feature_columns']
        )

        val_dataset = OptionPricingDataset(
            data_df=val_data,
            target_col=config['data']['target_column'],
            seq_len=config['data']['seq_len'],
            label_len=config['data']['label_len'],
            pred_len=config['data']['pred_len'],
            feature_cols=config['data']['feature_columns']
        )

        test_dataset = OptionPricingDataset(
            data_df=test_data,
            target_col=config['data']['target_column'],
            seq_len=config['data']['seq_len'],
            label_len=config['data']['label_len'],
            pred_len=config['data']['pred_len'],
            feature_cols=config['data']['feature_columns']
        )

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config['hardware']['seed']
    ) if world_size > 1 else None

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        sampler=test_sampler,
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    return train_loader, val_loader, test_loader, train_sampler


def train_model(model, train_loader, val_loader, config, device, 
                rank, world_size, train_sampler):
    """Full training loop with early stopping"""
    
    criterion = nn.MSELoss()
    
    # Setup optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")

    # Setup scheduler
    if config['training']['scheduler']['type'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience'],
            verbose=is_main_process(rank),
            min_lr=config['training']['scheduler']['min_lr']
        )
    else:
        scheduler = None

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    if is_main_process(rank):
        model_without_ddp = model.module if isinstance(model, DDP) else model
        print(f"\n{'='*70}")
        print(f"TRAINING STARTED - {config['model']['name'].upper()}")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Model parameters: {sum(p.numel() for p in model_without_ddp.parameters()):,}")
        print(f"Training batches per epoch (per GPU): {len(train_loader)}")
        print(f"Validation batches per epoch (per GPU): {len(val_loader)}")
        print(f"Effective batch size: {config['training']['batch_size'] * world_size}")
        print(f"{'='*70}\n")

    start_time = time.time()

    for epoch in range(config['training']['epochs']):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train and validate
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            rank, world_size, epoch + 1, config['training']['grad_clip']
        )
        val_loss = validate_epoch(
            model, val_loader, criterion, device,
            rank, world_size, epoch + 1
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Step scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early stopping check
        improvement = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            improvement = "✓"

            # Save best model
            if is_main_process(rank):
                model_to_save = model.module if isinstance(model, DDP) else model
                best_model_state = model_to_save.state_dict()
        else:
            patience_counter += 1

        # Print progress
        if is_main_process(rank):
            print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Best Val:   {best_val_loss:.6f}")
            print(f"  Patience:   {patience_counter}/{config['training']['early_stopping_patience']}")
            print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")
            if improvement:
                print(f"  Status:     {improvement} Improvement!")

        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            if is_main_process(rank):
                print(f"\n{'='*70}")
                print(f"Early stopping triggered at epoch {epoch + 1}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"{'='*70}\n")
            break

    # Load best model
    if is_main_process(rank):
        model_to_save = model.module if isinstance(model, DDP) else model
        model_to_save.load_state_dict(best_model_state)

    # Synchronize model across all processes
    if world_size > 1:
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    training_time = time.time() - start_time

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'training_time': training_time
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Transformer for Option Pricing')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data file (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size per GPU (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Maximum number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    args = parser.parse_args()

    # Load and merge config
    config = load_config(args.config)
    config = merge_config_with_args(config, args)

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()

    # Set seed
    set_seed(config['hardware']['seed'])

    # Device setup
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')
        if is_main_process(rank):
            print("WARNING: No GPU available, using CPU")

    if is_main_process(rank):
        print(f"\n{'='*70}")
        print(f"TRANSFORMER-BASED OPTION PRICING - {config['model']['name'].upper()}")
        print(f"{'='*70}")
        print(f"Config: {args.config}")
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local rank: {local_rank}")
        print(f"Device: {device}")
        print(f"Effective batch size: {config['training']['batch_size'] * world_size}")
        print(f"{'='*70}\n")

    # Load data
    data_path = config['data'].get('path', args.data_path)
    if data_path is None:
        raise ValueError("Data path must be specified in config or as argument")
    
    df = load_preprocessed_data(data_path, rank)

    # Get sequential mode settings from config
    sequential = config['data'].get('sequential', False)
    group_by = config['data'].get('group_by', None)
    min_chain_length = config['data'].get('min_chain_length', None)
    stratify_by_horizon = config['data'].get('stratify_by_horizon', False)
    
    if is_main_process(rank):
        if sequential:
            print(f"Mode: Sequential (no data leakage)")
            print(f"Grouping: {group_by if group_by else 'Default'}")
            print(f"Stratify by horizon: {stratify_by_horizon}")
        else:
            print("Mode: Original (simple sliding windows)")

    # Create splits
    train_data, val_data, test_data, scaler = create_rolling_window_split(
        df, 
        config['data']['test_month'], 
        config['data']['test_year'],
        config['data']['feature_columns'], 
        config['data']['target_column'],
        config['data']['train_months'], 
        config['data']['val_months'], 
        rank,
        sequential=sequential,
        seq_len=config['data']['seq_len'],
        label_len=config['data']['label_len'],
        pred_len=config['data']['pred_len'],
        group_by=group_by,
        min_chain_length=min_chain_length,
        stratify_by_horizon=stratify_by_horizon
    )

    # Create dataloaders
    train_loader, val_loader, test_loader, train_sampler = create_dataloaders(
        train_data, val_data, test_data, config, rank, world_size,
        sequential=sequential
    )

    # Determine actual number of features (may have changed due to encoding)
    if sequential:
        # Get from first sequence
        if len(train_data['X_enc']) > 0:
            n_features = train_data['X_enc'][0].shape[1]
        else:
            raise ValueError("No training sequences created!")
    else:
        n_features = len(config['data']['feature_columns'])
    
    if is_main_process(rank):
        print(f"\nModel will use {n_features} input features")

    # Create model
    model_config = config['model'].copy()
    model_config['n_features'] = n_features  # Use actual feature count
    model_name = model_config.pop('name')
    
    model = get_model(model_name, **model_config).to(device)

    # Wrap with DDP for multi-GPU
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    if is_main_process(rank):
        model_without_ddp = model.module if isinstance(model, DDP) else model
        print(f"Model created with {sum(p.numel() for p in model_without_ddp.parameters()):,} parameters")

    # Train model
    history = train_model(
        model, train_loader, val_loader, config, device,
        rank, world_size, train_sampler
    )

    # Evaluate model
    if is_main_process(rank):
        print(f"{'='*70}")
        print("EVALUATION")
        print(f"{'='*70}")

    results = evaluate_model(
        model, test_loader, scaler, device, rank, world_size,
        config['data']['feature_columns'], config['data']['target_column'],
        sequential=sequential
    )

    # Print and save results
    if is_main_process(rank) and results is not None:
        print(f"\nModel: {model_name.upper()}")
        print(f"Training time: {history['training_time']:.2f} seconds")
        print("\nPerformance Metrics:")
        print(f"  RMSE: ${results['rmse']:.2f}")
        print(f"  MAE:  ${results['mae']:.2f}")
        print(f"  MSE:  {results['mse']:.2f}")
        print(f"  Theil U1: {results['theil_u1']:.6f}")

        # Save everything
        save_results(model, config, history, results, rank)
        plot_results(history, results, config, rank)

        print(f"{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")

    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()
