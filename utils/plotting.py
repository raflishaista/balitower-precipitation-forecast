"""Plotting routines for precipitation forecasts."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_forecast(figsize=(14, 9), save_path="outputs/figures/precipitation_forecast.png",
                  train_df=None, test_df=None, sarima_pred=None, sarima_test_ci=None,
                  xgb_pred=None, lstm_pred=None, fc=None, fc_low=None, fc_high=None,
                  city="Jakarta", lstm_ok=False,
                  sarima_rmse=None, sarima_mae=None, xgb_rmse=None, xgb_mae=None,
                  lstm_rmse=None, lstm_mae=None):
    """Create and save the 2x2 forecast comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Historical + SARIMAX fit vs test
    ax1 = axes[0, 0]
    ax1.plot(train_df.index, train_df["precip"], color="steelblue", label="Train", alpha=0.7)
    ax1.plot(test_df.index, test_df["precip"], color="grey", label="Actual (test)", alpha=0.7)
    test_idx = test_df.index[0]
    ax1.axvline(test_idx, color="grey", linestyle="--", alpha=0.5)
    ax1.plot(test_df.index, sarima_pred, color="darkblue", label="SARIMAX pred", lw=2)
    ax1.fill_between(test_df.index, sarima_test_ci[:, 0], sarima_test_ci[:, 1],
                     color="darkblue", alpha=0.15)
    if lstm_ok:
        ax1.plot(test_df.index[-len(lstm_pred):], lstm_pred, color="purple", label="LSTM pred", lw=2, alpha=0.7)
    ax1.set_title("SARIMAX — Test Period")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlabel("")
    ax1.set_ylabel("Precipitation (mm)")

    # Plot 2: Historical + XGBoost predictions on test
    ax2 = axes[0, 1]
    ax2.plot(train_df.index, train_df["precip"], color="steelblue", label="Train", alpha=0.7)
    ax2.plot(test_df.index, test_df["precip"], color="grey", label="Actual (test)", alpha=0.7)
    ax2.axvline(test_idx, color="grey", linestyle="--", alpha=0.5)
    ax2.plot(test_df.index, xgb_pred, color="darkorange", label="XGBoost pred", lw=2)
    ax2.set_title("XGBoost — Test Period")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_xlabel("")
    ax2.set_ylabel("Precipitation (mm)")

    # Plot 3: 7-day forecast
    ax3 = axes[1, 0]
    ax3.bar(fc["Date"], fc["SARIMAX"], color="steelblue", alpha=0.7, label="SARIMAX")
    ax3.bar([pd.Timestamp(d) + pd.Timedelta("1D") for d in fc["Date"]], fc["XGBoost"],
            color="darkorange", alpha=0.7, label="XGBoost")
    if lstm_ok:
        ax3.bar([pd.Timestamp(d) + pd.Timedelta("2D") for d in fc["Date"]], fc["LSTM"],
                color="purple", alpha=0.7, label="LSTM")
    ax3.fill_between(fc["Date"], fc_low, fc_high, color="steelblue", alpha=0.12,
                     label="SARIMAX 95% CI")
    ax3.set_title(f"7-Day Precipitation Forecast ({city})")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax3.tick_params(axis="x", rotation=45)
    ax3.set_ylabel("Precipitation (mm)")
    ax3.axhline(0, color="black", linewidth=0.5)

    # Plot 4: Model comparison bar chart
    ax4 = axes[1, 1]
    models = ["SARIMAX", "XGBoost"]
    rmse_vals = [sarima_rmse, xgb_rmse]
    mae_vals = [sarima_mae, xgb_mae]
    if lstm_ok:
        models.append("LSTM")
        rmse_vals.append(lstm_rmse)
        mae_vals.append(lstm_mae)
    x = np.arange(len(models))
    w = 0.35
    ax4.bar(x - w/2, rmse_vals, w, label="RMSE", color="steelblue")
    ax4.bar(x + w/2, mae_vals, w, label="MAE", color="darkorange")
    ax4.set_title("Test-Period Model Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend()
    ax4.set_ylabel("Error (mm)")
    for i, v in enumerate(rmse_vals):
        ax4.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {save_path}")
