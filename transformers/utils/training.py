"""Training utilities for distributed training"""

import os
import pickle
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def setup_distributed():
    """Initialize distributed training environment"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process(rank):
    """Check if this is the main process"""
    return rank == 0


def train_epoch(model, train_loader, criterion, optimizer, device, 
                rank, world_size, epoch, grad_clip=1.0):
    """Training epoch with multi-GPU support"""
    model.train()
    total_loss = 0
    num_batches = 0

    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
    else:
        pbar = train_loader

    for batch_idx, (x_enc, x_dec, y) in enumerate(pbar):
        x_enc = x_enc.to(device, non_blocking=True)
        x_dec = x_dec.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Skip bad batches
        if torch.isnan(x_enc).any() or torch.isnan(x_dec).any() or torch.isnan(y).any():
            continue
        if torch.isinf(x_enc).any() or torch.isinf(x_dec).any() or torch.isinf(y).any():
            continue

        optimizer.zero_grad(set_to_none=True)

        try:
            output = model(x_enc, x_dec)
            if torch.isnan(output).any():
                continue

            output = output.squeeze()
            y = y.squeeze()
            loss = criterion(output, y)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if is_main_process(rank):
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        except RuntimeError as e:
            if is_main_process(rank):
                print(f"Runtime error at batch {batch_idx}: {str(e)}")
            continue

    if num_batches == 0:
        return float('nan')

    avg_loss = total_loss / num_batches

    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

    return avg_loss


def validate_epoch(model, val_loader, criterion, device, rank, world_size, epoch):
    """Validation epoch with multi-GPU support"""
    model.eval()
    total_loss = 0
    num_batches = 0

    if is_main_process(rank):
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} Validation')
    else:
        pbar = val_loader

    with torch.no_grad():
        for x_enc, x_dec, y in pbar:
            x_enc = x_enc.to(device, non_blocking=True)
            x_dec = x_dec.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            output = model(x_enc, x_dec)
            output = output.squeeze()
            y = y.squeeze()

            loss = criterion(output, y)
            total_loss += loss.item()
            num_batches += 1

            if is_main_process(rank):
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / num_batches

    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

    return avg_loss


def evaluate_model(model, test_loader, scaler, device, rank, world_size, 
                  feature_cols, target_col, sequential=False):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_targets = []

    if is_main_process(rank):
        pbar = tqdm(test_loader, desc='Evaluating')
    else:
        pbar = test_loader

    with torch.no_grad():
        for x_enc, x_dec, y in pbar:
            x_enc = x_enc.to(device, non_blocking=True)
            x_dec = x_dec.to(device, non_blocking=True)

            predictions = model(x_enc, x_dec)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.numpy())

    predictions_scaled = np.concatenate(all_predictions, axis=0).reshape(-1, 1)
    targets_scaled = np.concatenate(all_targets, axis=0).reshape(-1, 1)

    if world_size > 1:
        pred_tensor = torch.from_numpy(predictions_scaled).cuda()
        target_tensor = torch.from_numpy(targets_scaled).cuda()

        local_size = torch.tensor([pred_tensor.size(0)], device=device)
        size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)

        max_size = max([s.item() for s in size_list])

        if pred_tensor.size(0) < max_size:
            padding = torch.zeros(
                max_size - pred_tensor.size(0), 1,
                device=device, dtype=pred_tensor.dtype
            )
            pred_tensor = torch.cat([pred_tensor, padding], dim=0)
            target_tensor = torch.cat([target_tensor, padding], dim=0)

        pred_list = [torch.zeros_like(pred_tensor) for _ in range(world_size)]
        target_list = [torch.zeros_like(target_tensor) for _ in range(world_size)]

        dist.all_gather(pred_list, pred_tensor)
        dist.all_gather(target_list, target_tensor)

        if is_main_process(rank):
            predictions_list = []
            targets_list = []
            for i, size in enumerate(size_list):
                size = size.item()
                predictions_list.append(pred_list[i][:size].cpu().numpy())
                targets_list.append(target_list[i][:size].cpu().numpy())

            predictions_scaled = np.concatenate(predictions_list, axis=0)
            targets_scaled = np.concatenate(targets_list, axis=0)

    if not is_main_process(rank):
        return None

    # Inverse transform - handle sequential vs non-sequential differently
    if sequential:
        # In sequential mode, scaler was fit on [features + target]
        # Target is the last column
        n_cols = scaler.n_features_in_
        target_idx = n_cols - 1

        dummy_scaled = np.zeros((len(predictions_scaled), n_cols))
        dummy_scaled[:, target_idx] = predictions_scaled.flatten()
        predictions = scaler.inverse_transform(dummy_scaled)[:, target_idx].reshape(-1, 1)

        dummy_scaled[:, target_idx] = targets_scaled.flatten()
        targets = scaler.inverse_transform(dummy_scaled)[:, target_idx].reshape(-1, 1)
    else:
        # Non-sequential mode: scaler was fit on [features + target] concatenated
        # The target column is the last column in the scaler
        n_cols = scaler.n_features_in_
        target_idx = n_cols - 1  # Target is always the last column after scaling

        dummy_scaled = np.zeros((len(predictions_scaled), n_cols))
        dummy_scaled[:, target_idx] = predictions_scaled.flatten()
        predictions = scaler.inverse_transform(dummy_scaled)[:, target_idx].reshape(-1, 1)

        dummy_scaled[:, target_idx] = targets_scaled.flatten()
        targets = scaler.inverse_transform(dummy_scaled)[:, target_idx].reshape(-1, 1)

    # Metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    theil_u1 = np.sqrt(mse) / (np.sqrt(np.mean(targets**2)) + 1e-8)

    return {
        'predictions': predictions,
        'targets': targets,
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'theil_u1': theil_u1
    }
    

def save_results(model, config, history, results, rank):
    """Save model and results"""
    if not is_main_process(rank):
        return

    from torch.nn.parallel import DistributedDataParallel as DDP

    os.makedirs(config['output']['save_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['model']['name']

    # Save model
    model_to_save = model.module if isinstance(model, DDP) else model
    model_path = os.path.join(
        config['output']['save_dir'], 
        f'{model_name}_model_{timestamp}.pt'
    )
    torch.save(model_to_save.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Save config
    config_path = os.path.join(
        config['output']['save_dir'], 
        f'{model_name}_config_{timestamp}.pkl'
    )
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    # Save results
    results_path = os.path.join(
        config['output']['save_dir'], 
        f'{model_name}_results_{timestamp}.pkl'
    )
    with open(results_path, 'wb') as f:
        pickle.dump({'history': history, 'results': results}, f)


def plot_results(history, results, config, rank):
    """Generate and save plots"""
    if not is_main_process(rank):
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(config['output']['save_dir'], exist_ok=True)
    model_name = config['model']['name']

    # Training history
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history['train_losses']) + 1)
    ax.plot(epochs, history['train_losses'], label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_losses'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(f'Training History - {model_name.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(config['output']['save_dir'], f'{model_name}_training_history.png'), 
        dpi=300
    )
    plt.close()

    # Predictions
    predictions = results['predictions'].flatten()
    targets = results['targets'].flatten()

    sample_size = min(5000, len(predictions))
    indices = np.random.choice(len(predictions), sample_size, replace=False)
    pred_sample = predictions[indices]
    target_sample = targets[indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(target_sample, pred_sample, alpha=0.5, s=20)
    min_val = min(target_sample.min(), pred_sample.min())
    max_val = max(target_sample.max(), pred_sample.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('Actual Option Price ($)')
    ax1.set_ylabel('Predicted Option Price ($)')
    ax1.set_title(f'Predictions vs Actual - {model_name.upper()}')
    ax1.grid(True, alpha=0.3)

    residuals = predictions - targets
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Prediction Error ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Errors')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(
        os.path.join(config['output']['save_dir'], f'{model_name}_predictions.png'), 
        dpi=300
    )
    plt.close()
