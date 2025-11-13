import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models import mlp

DATA_DIR = "data/"
TEST_YEAR = 2023


def read_preprocessed_data(filename):
    return pd.read_csv(filename, low_memory=False, index_col=0)


def create_rolling_window_split(
    df,
    test_month,
    test_year=2023,
    feature_columns=["STRIKE", "UNDERLYING_LAST", "MTM", "RFR", "VOLATILITY"],
    train_months=8,
    val_months=1,
):
    test_start = pd.Timestamp(year=test_year, month=test_month, day=1)
    test_end = test_start + pd.offsets.MonthEnd(0)

    val_start = test_start - pd.offsets.MonthBegin(val_months)
    val_end = test_start - pd.Timedelta(days=1)

    train_start = val_start - pd.offsets.MonthBegin(train_months)
    train_end = val_start - pd.Timedelta(days=1)

    train_df = (
        df.loc[
            (df["QUOTE_DATE"] >= train_start) & (df["QUOTE_DATE"] <= train_end),
            ["QUOTE_DATE"] + feature_columns,
        ]
        .set_index("QUOTE_DATE")
        .copy()
    )
    val_df = (
        df.loc[
            (df["QUOTE_DATE"] >= val_start) & (df["QUOTE_DATE"] <= val_end),
            ["QUOTE_DATE"] + feature_columns,
        ]
        .set_index("QUOTE_DATE")
        .copy()
    )
    test_df = (
        df.loc[
            (df["QUOTE_DATE"] >= test_start) & (df["QUOTE_DATE"] <= test_end),
            ["QUOTE_DATE"] + feature_columns,
        ]
        .set_index("QUOTE_DATE")
        .copy()
    )

    # Drop rows with NaN values
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()

    scaler = MinMaxScaler()
    scaler.fit(train_df)

    train_scaled = pd.DataFrame(
        scaler.transform(train_df), columns=train_df.columns, index=train_df.index
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_df), columns=val_df.columns, index=val_df.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df), columns=test_df.columns, index=test_df.index
    )

    return train_scaled, val_scaled, test_scaled


def calculate_pimentel_metrics(y_true, y_pred):
    """
    Calculate performance metrics from Pimentel et al. paper.

    Args:
        y_true: actual values (tensor)
        y_pred: predicted values (tensor)

    Returns:
        dict with Theil U1, Bias proportion, Variance proportion, Covariance proportion
    """
    import numpy as np

    # Convert to numpy for calculations
    y_true = y_true.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()

    m = len(y_true)

    # Theil U1 (Equation 27)
    mse = (1 / m) * np.sum((y_true - y_pred) ** 2)
    numerator = np.sqrt(mse)
    denominator = np.sqrt((1 / m) * np.sum(y_true**2))
    theil_u1 = numerator / denominator

    # Mean and std of actual and predicted
    y_mean = y_true.mean()
    y_hat_mean = y_pred.mean()
    sigma_y = y_true.std(ddof=0)
    sigma_y_hat = y_pred.std(ddof=0)

    # Correlation coefficient
    rho = np.corrcoef(y_true, y_pred)[0, 1]

    # Bias proportion (Equation 28) - squared difference in means
    bias_prop = (y_hat_mean - y_mean) ** 2 / mse

    # Variance proportion (Equation 29) - squared difference in standard deviations
    var_prop = (sigma_y_hat - sigma_y) ** 2 / mse

    # Covariance proportion (Equation 30) - unsystematic error
    cov_prop = 2 * (1 - rho) * sigma_y_hat * sigma_y / mse

    return {
        "theil_u1": theil_u1,
        "bias_prop": bias_prop,
        "var_prop": var_prop,
        "cov_prop": cov_prop,
    }


def train(
    train_df,
    val_df,
    model_type,
    device,
    criterion=nn.MSELoss(),
    number_layers=4,
    units_per_layer=32,
    batch_norm_momentum=0.3,
    learning_rate=0.004469,
    weight_decay=0.000425,
    epochs=20,
    patience=3,
):
    X_train = torch.tensor(train_df.drop(columns=["CALL"]).values, dtype=torch.float32)
    X_val = torch.tensor(val_df.drop(columns=["CALL"]).values, dtype=torch.float32)

    y_train = torch.tensor(train_df["CALL"].values.reshape(-1, 1), dtype=torch.float32)
    y_val = torch.tensor(val_df["CALL"].values.reshape(-1, 1), dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    model = model_type(
        input_feature_size=X_train.shape[1],
        output_size=1,
        number_layers=number_layers,
        units_per_layer=units_per_layer,
        momentum=batch_norm_momentum,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    X_val_device = X_val.to(device)
    y_val_device = y_val.to(device)

    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []
    epochs_recorded = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        all_train_preds = []
        all_train_targets = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_train_preds.append(y_pred.detach())
            all_train_targets.append(y_batch.detach())

        avg_train_loss = total_loss / len(train_loader)

        train_preds_tensor = torch.cat(all_train_preds, dim=0)
        train_targets_tensor = torch.cat(all_train_targets, dim=0)
        train_metrics = calculate_pimentel_metrics(
            train_targets_tensor, train_preds_tensor
        )

        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_device)
                val_loss = criterion(val_pred, y_val_device)
                val_metrics = calculate_pimentel_metrics(y_val_device, val_pred)

            train_losses.append(avg_train_loss)
            val_losses.append(val_loss.item())
            epochs_recorded.append(epoch + 1)

            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}"
            )
            print(
                f"  Val Metrics - Bias: {val_metrics['bias_prop']:.4f}, Var: {val_metrics['var_prop']:.4f}, Cov: {val_metrics['cov_prop']:.4f}"
            )
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. Best val loss: {best_val_loss:.4f}"
            )
            break
    return model


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    df = read_preprocessed_data(f"{DATA_DIR}/data.csv")
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])

    # Note: this is the non GG
    training_columns = ["UNDERLYING_LAST", "STRIKE", "MTM", "VOL_90D", "RFR", "CALL"]
    for month in range(1, 13):
        train_df, val_df, test_df = create_rolling_window_split(
            df, test_year=TEST_YEAR, test_month=1, feature_columns=training_columns
        )
        print(f"Length of training set: {len(train_df)}")
        print(f"Length of validation set: {len(val_df)}")
        print(f"Length of test set: {len(test_df)}")

        # Check for NaN values
        print(f"\nNaN values in train_df:")
        print(train_df.isna().sum())
        print(f"\nNaN values in CALL column: {train_df['CALL'].isna().sum()}")

        criterion = nn.MSELoss()
        model = train(train_df, val_df, mlp.PimentelMLP, device, criterion=criterion)

        checkpoint_dir = f"checkpoints/mlp"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = f"{checkpoint_dir}/{TEST_YEAR}_{month:02d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": "PimentelMLP",
                "test_year": TEST_YEAR,
                "test_month": month,
            },
            checkpoint_path,
        )
        print(f"Model saved to {checkpoint_path}")

        X_test = torch.tensor(
            test_df.drop(columns=["CALL"]).values, dtype=torch.float32
        )
        y_test = torch.tensor(
            test_df["CALL"].values.reshape(-1, 1), dtype=torch.float32
        )

        model.eval()
        X_test_device = X_test.to(device)
        y_test_device = y_test.to(device)

        with torch.no_grad():
            test_pred = model(X_test_device)
            test_loss = criterion(test_pred, y_test_device)
            test_mae = torch.mean(torch.abs(test_pred - y_test_device)).item()
            test_metrics = calculate_pimentel_metrics(y_test_device, test_pred)

        eval_results = pd.DataFrame(
            [
                {
                    "test_year": TEST_YEAR,
                    "test_month": month,
                    "model_type": "PimentelMLP",
                    "test_loss_mse": test_loss.item(),
                    "test_rmse": torch.sqrt(test_loss).item(),
                    "test_mae": test_mae,
                    "theil_u1": test_metrics["theil_u1"],
                    "bias_proportion": test_metrics["bias_prop"],
                    "variance_proportion": test_metrics["var_prop"],
                    "covariance_proportion": test_metrics["cov_prop"],
                    "sum_of_proportions": test_metrics["bias_prop"]
                    + test_metrics["var_prop"]
                    + test_metrics["cov_prop"],
                }
            ]
        )

        eval_results_path = f"{checkpoint_dir}/{TEST_YEAR}_{month:02d}_results.csv"
        eval_results.to_csv(eval_results_path, index=False)

        print(eval_results)


if __name__ == "__main__":
    main()
