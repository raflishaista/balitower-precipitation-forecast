"""Top-level precipitation forecast runner.

Usage:
    python run_forecast.py                          # default: Jakarta, 10y, all models
    python run_forecast.py --city "jakarta"
    python run_forecast.py --input inputs_json/foo.json
    python run_forecast.py --no-lstm                # skip deep learning
    python run_forecast.py --years 5
    python run_forecast.py --models sarima xgb hgb ets rf
"""
import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from joblib import dump

warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.precipitation_models import (
    load_precipitation_data, load_from_json, save_forecast_to_json,
    build_features, split_train_test,
    sarima_fit, sarima_test_predictions, sarima_forward_forecast,
    train_xgboost, forecast_xgboost,
    train_histgb, forecast_histgb,
    ets_fit, ets_test_predictions, ets_forward_forecast,
    train_rf, forecast_rf,
    forecast_patchtst, forecast_timesfm, forecast_tabicl,
    _LSTM_AVAILABLE, _LSTM_ERROR,
    _TRANSFORMER_AVAILABLE,
    _TABICL_AVAILABLE,
    evaluate,
    save_pipeline,
    backtest_last_n_days,
)
from utils.plotting import plot_forecast, plot_backtest

FORECAST_HORIZON = 7
DEFAULT_OUTPUT_DIR = "outputs/predictions"
DEFAULT_MODEL_PATH = "data/processed/precipitation_model.joblib"
DEFAULT_FIGURE_PATH = "outputs/figures/precipitation_forecast.png"

# Available model shortcuts
MODEL_REGISTRY = {
    "sarima":   {"train": sarima_fit,            "test": sarima_test_predictions,
                 "forecast": sarima_forward_forecast},
    "xgb":      {"train": train_xgboost,         "test": forecast_xgboost,
                 "forecast": None},  # uses model directly
    "hgb":      {"train": train_histgb,          "test": forecast_histgb,
                 "forecast": None},
    "ets":      {"train": ets_fit,               "test": ets_test_predictions,
                 "forecast": ets_forward_forecast},
    "rf":       {"train": train_rf,              "test": forecast_rf,
                 "forecast": None},
    "lstm":     {"available": "_LSTM_AVAILABLE"},
    "patchtst": {"available": "_TRANSFORMER_AVAILABLE"},
    "timesfm":  {"available": "_TRANSFORMER_AVAILABLE"},
    "tabicl":   {"available": "_TABICL_AVAILABLE"},
}


def _build_fc_dataframe(forecast_dates, model_preds, model_names):
    """Build a unified forecast DataFrame from per-model predictions."""
    fc = pd.DataFrame({"Date": forecast_dates})
    for name, preds in model_preds.items():
        fc[name] = np.round(preds, 2)
    return fc


def _plot_multi_model_comparison(test_df, model_preds, model_metrics,
                                 save_path="outputs/figures/precipitation_forecast.png"):
    """Plot all model predictions vs actual on one chart + bar chart of metrics."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    n_models = len(model_preds)
    colors = plt.cm.tab20c(np.linspace(0, 1, n_models))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: All models on test period (lines)
    ax1 = axes[0, 0]
    ax1.plot(test_df.index, test_df["precip"].values, color="black", lw=2,
             label="Actual", zorder=5)
    for (name, preds), color in zip(model_preds.items(), colors):
        ax1.plot(test_df.index, preds, color=color, lw=1.5, alpha=0.85, label=name)
    ax1.set_title("All Models — Test Period Predictions")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlabel("")
    ax1.set_ylabel("Precipitation (mm)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax1.tick_params(axis="x", rotation=45)
    ax1.axhline(0, color="grey", lw=0.5)

    # Plot 2: RMSE comparison bar chart
    ax2 = axes[0, 1]
    names = list(model_metrics.keys())
    rmse_vals = [model_metrics[n]["RMSE"] for n in names]
    mae_vals = [model_metrics[n]["MAE"] for n in names]
    x_pos = np.arange(len(names))
    w = 0.35
    bars1 = ax2.bar(x_pos - w/2, rmse_vals, w, label="RMSE", color=colors[:len(names)])
    bars2 = ax2.bar(x_pos + w/2, mae_vals, w, label="MAE", color=colors[len(names):2*len(names)])
    ax2.set_title("Test-Period Model Comparison")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=15)
    ax2.legend()
    ax2.set_ylabel("Error (mm)")
    for i, (rmse, mae) in enumerate(zip(rmse_vals, mae_vals)):
        ax2.text(i - w/2, rmse + 0.2, f"{rmse:.1f}", ha="center", fontsize=7)
        ax2.text(i + w/2, mae + 0.2, f"{mae:.1f}", ha="center", fontsize=7)

    # Plot 3: 7-day forward forecast (grouped bar)
    ax3 = axes[1, 0]
    n_days = FORECAST_HORIZON
    x_bar = np.arange(n_days)
    model_keys = list(model_preds.keys())
    for j, name in enumerate(model_keys):
        preds = model_preds[name][-n_days:]
        offset = (j - (len(model_keys)-1)/2) * 0.2
        ax3.bar(x_bar + offset, preds, 0.18, label=name,
                color=colors[j % len(colors)], alpha=0.85)
    ax3.set_title("7-Day Forward Forecast (last 7 test days)")
    ax3.set_xticks(x_bar)
    ax3.set_xticklabels([str(i+1) for i in range(n_days)], rotation=45)
    ax3.legend(fontsize=7)
    ax3.set_ylabel("Precipitation (mm)")
    ax3.axhline(0, color="grey", lw=0.5)

    # Plot 4: Model ranking table
    ax4 = axes[1, 1]
    ax4.axis("off")
    # Sort models by RMSE (best first)
    sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]["RMSE"])
    table_data = []
    for rank, (name, met) in enumerate(sorted_models, 1):
        table_data.append([f"#{rank}", name, f"{met['RMSE']:.2f}", f"{met['MAE']:.2f}"])
    table = ax4.table(cellText=table_data,
                      colLabels=["Rank", "Model", "RMSE", "MAE"],
                      cellLoc="center", loc="center",
                      colWidths=[0.12, 0.25, 0.28, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    # Highlight best model
    for i in range(1, len(table_data)+1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 1:
                cell.set_facecolor("#d4edda")
            elif i == len(table_data):
                cell.set_facecolor("#f8d7da")
    ax4.set_title("Model Ranking (lowest RMSE = best)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {save_path}")


def run(city, years, include_lstm=True, save_json_path=None, models=None):
    """Run full forecast pipeline with all configured models."""
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DEFAULT_FIGURE_PATH), exist_ok=True)

    # Determine which models to run
    if models is None:
        models = ["sarima", "xgb", "hgb", "ets", "rf"]
        if include_lstm and _LSTM_AVAILABLE:
            models.append("lstm")
        if _TRANSFORMER_AVAILABLE:
            models.append("patchtst")
            models.append("timesfm")
        if _TABICL_AVAILABLE:
            models.append("tabicl")

    print(f"\n{'='*60}")
    print(f"  PRECIPITATION FORECAST — {city.upper()}")
    print(f"  {years} years of daily data  |  Horizon: {FORECAST_HORIZON} days")
    print(f"  Models: {', '.join(models)}")
    print(f"{'='*60}\n")

    # 1. Load data
    df, source_file = load_precipitation_data(city, years)
    print(f"  Date range: {df.index[0].date()} - {df.index[-1].date()} ({len(df)} days)\n")

    # 2. Features + split
    feat_df = build_features(df)
    train_df, test_df = split_train_test(feat_df)
    print(f"  Train: {len(train_df)} days | Test: {len(test_df)} days\n")

    # Shared state
    all_model_preds = {}   # name -> test predictions
    all_model_metrics = {} # name -> metrics dict
    all_model_states = {}  # name -> trained model / results
    forecast_preds = {}    # name -> future 7-day forecast

    # ---- Train & test each model ----
    for name in models:
        info = MODEL_REGISTRY.get(name)
        if not info:
            print(f"  [SKIP] Unknown model '{name}'")
            continue
        if "available" in info:
            avail = globals().get(info["available"], False)
            if not avail:
                err_msg = globals().get(f"_{info['available'].replace('_','',1).upper()}_ERROR", "")
                print(f"  [SKIP] {name.upper()} — {err_msg or 'not available'}")
                continue

        print(f"> {name.upper()} ...")
        try:
            if name == "sarima":
                res = info["train"](train_df)
                pred, ci = info["test"](res, test_df)
                metrics = evaluate(test_df["precip"].values, pred, "SARIMAX")
                all_model_states["sarima"] = res
                all_model_preds["sarima"] = pred
                all_model_metrics["sarima"] = metrics
                # Future forecast
                fc_dates = pd.date_range(start=df.index[-1] + timedelta(days=1),
                                         periods=FORECAST_HORIZON, freq="D")
                fc_pred, fc_ci = info["forecast"](res, feat_df, FORECAST_HORIZON)
                forecast_preds["sarima"] = (fc_pred, fc_ci)
                print(f"  {metrics}")

            elif name == "xgb":
                model, feat_cols = info["train"](train_df)
                pred = info["test"](model, feat_cols, test_df)
                metrics = evaluate(test_df["precip"].values, pred, "XGBoost")
                all_model_states["xgb"] = (model, feat_cols)
                all_model_preds["xgb"] = pred
                all_model_metrics["xgb"] = metrics
                # Future forecast
                future_X = feat_df[feat_cols].tail(FORECAST_HORIZON).values
                forecast_preds["xgb"] = np.maximum(model.predict(future_X), 0)
                print(f"  {metrics}")

            elif name == "hgb":
                model, feat_cols = info["train"](train_df)
                pred = info["test"](model, feat_cols, test_df)
                metrics = evaluate(test_df["precip"].values, pred, "HistGB")
                all_model_states["hgb"] = (model, feat_cols)
                all_model_preds["hgb"] = pred
                all_model_metrics["hgb"] = metrics
                future_X = feat_df[feat_cols].tail(FORECAST_HORIZON).values
                forecast_preds["hgb"] = np.maximum(model.predict(future_X), 0)
                print(f"  {metrics}")

            elif name == "ets":
                res = info["train"](train_df)
                pred, ci = info["test"](res, test_df)
                metrics = evaluate(test_df["precip"].values, pred, "ETS")
                all_model_states["ets"] = res
                all_model_preds["ets"] = pred
                all_model_metrics["ets"] = metrics
                fc_pred, fc_ci = info["forecast"](res, FORECAST_HORIZON)
                forecast_preds["ets"] = fc_pred
                print(f"  {metrics}")

            elif name == "rf":
                model, feat_cols = info["train"](train_df)
                pred = info["test"](model, feat_cols, test_df)
                metrics = evaluate(test_df["precip"].values, pred, "RF")
                all_model_states["rf"] = (model, feat_cols)
                all_model_preds["rf"] = pred
                all_model_metrics["rf"] = metrics
                future_X = feat_df[feat_cols].tail(FORECAST_HORIZON).values
                forecast_preds["rf"] = np.maximum(model.predict(future_X), 0)
                print(f"  {metrics}")

            elif name == "lstm":
                from utils.precipitation_models import (
                    prepare_lstm_sequence, train_lstm, forecast_lstm,
                )
                X_all, y_all, scaler = prepare_lstm_sequence(feat_df)
                split = int(len(X_all) * 0.85)
                lstm_model = train_lstm(X_all[:split], y_all[:split], epochs=20, batch=32)
                lstm_pred = forecast_lstm(lstm_model, feat_df["precip"].values, scaler)
                actual_tail = feat_df["precip"].iloc[-len(lstm_pred):].values
                metrics = evaluate(actual_tail, lstm_pred, "LSTM")
                all_model_states["lstm"] = lstm_model
                all_model_preds["lstm"] = lstm_pred
                all_model_metrics["lstm"] = metrics
                print(f"  {metrics}")

            elif name == "patchtst":
                series = train_df["precip"].values.astype(np.float32)
                n_test = len(test_df)
                # Align: predict forward from each training point, collect last n_test predictions
                context_len = 128
                preds_eval, actual_eval = [], []
                for offset in range(n_test):
                    idx = len(series) - n_test + offset
                    if idx < context_len:
                        continue
                    p = forecast_patchtst(series[:idx], horizon=FORECAST_HORIZON)
                    preds_eval.extend(p.tolist())
                    actual_eval.extend(series[idx:idx+FORECAST_HORIZON].tolist())
                preds_eval = np.array(preds_eval)
                actual_eval = np.array(actual_eval)
                # Average over horizon steps for per-day metrics
                n_pts = preds_eval.size // FORECAST_HORIZON
                preds_reshaped = preds_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                actual_reshaped = actual_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                preds_avg = preds_reshaped.mean(axis=1)
                actual_avg = actual_reshaped.mean(axis=1)
                metrics = evaluate(actual_avg, preds_avg, "PatchTST")
                # Forward forecast from end of training
                fc = forecast_patchtst(series, horizon=FORECAST_HORIZON)
                # Pad to match test set length for plotting
                pred_full = np.tile(fc, int(np.ceil(n_test / FORECAST_HORIZON)))[:n_test]
                all_model_preds["patchtst"] = pred_full
                all_model_metrics["patchtst"] = metrics
                print(f"  {metrics}")

            elif name == "timesfm":
                series = train_df["precip"].values.astype(np.float32)
                n_test = len(test_df)
                context_len = 256
                preds_eval, actual_eval = [], []
                for offset in range(n_test):
                    idx = len(series) - n_test + offset
                    if idx < context_len:
                        continue
                    p = forecast_timesfm(series[:idx], horizon=FORECAST_HORIZON)
                    preds_eval.extend(p.tolist())
                    actual_eval.extend(series[idx:idx+FORECAST_HORIZON].tolist())
                preds_eval = np.array(preds_eval)
                actual_eval = np.array(actual_eval)
                n_pts = preds_eval.size // FORECAST_HORIZON
                preds_reshaped = preds_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                actual_reshaped = actual_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                preds_avg = preds_reshaped.mean(axis=1)
                actual_avg = actual_reshaped.mean(axis=1)
                metrics = evaluate(actual_avg, preds_avg, "TimesFM")
                fc = forecast_timesfm(series, horizon=FORECAST_HORIZON)
                pred_full = np.tile(fc, int(np.ceil(n_test / FORECAST_HORIZON)))[:n_test]
                all_model_preds["timesfm"] = pred_full
                all_model_metrics["timesfm"] = metrics
                print(f"  {metrics}")

            elif name == "tabicl":
                from utils.precipitation_models import forecast_tabicl
                series = train_df["precip"].values.astype(np.float32)
                n_test = len(test_df)
                preds_eval, actual_eval = [], []
                # Rolling eval over last n_test points
                for offset in range(n_test):
                    idx = len(series) - n_test + offset
                    if idx < 64:
                        continue
                    df_chunk = train_df.iloc[:idx+1].copy()
                    p = forecast_tabicl(df_chunk, horizon=FORECAST_HORIZON)
                    preds_eval.extend(p.tolist())
                    actual_eval.extend(series[idx:idx+FORECAST_HORIZON].tolist())
                preds_eval = np.array(preds_eval)
                actual_eval = np.array(actual_eval)
                n_pts = preds_eval.size // FORECAST_HORIZON
                preds_reshaped = preds_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                actual_reshaped = actual_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                preds_avg = preds_reshaped.mean(axis=1)
                actual_avg = actual_reshaped.mean(axis=1)
                metrics = evaluate(actual_avg, preds_avg, "TabICL")
                fc = forecast_tabicl(train_df, horizon=FORECAST_HORIZON)
                pred_full = np.tile(fc, int(np.ceil(n_test / FORECAST_HORIZON)))[:n_test]
                all_model_preds["tabicl"] = pred_full
                all_model_metrics["tabicl"] = metrics
                print(f"  {metrics}")

        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_model_metrics:
        print("\n  No models trained successfully. Exiting.")
        return None

    # ---- 7-Day Forward Forecast Table ----
    print("\n--- 7-Day Precipitation Forecast ---\n")
    forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1),
                                   periods=FORECAST_HORIZON, freq="D")
    fc_dict = {"Date": forecast_dates}
    for name in all_model_preds:
        if name in forecast_preds:
            fc_dict[name] = np.round(forecast_preds[name], 2)
        else:
            fc_dict[name] = np.round(all_model_preds[name][-FORECAST_HORIZON:], 2)
    fc = pd.DataFrame(fc_dict)
    print(fc.to_string(index=False))

    # ---- Multi-model comparison plot ----
    _plot_multi_model_comparison(
        test_df, all_model_preds, all_model_metrics,
        save_path=DEFAULT_FIGURE_PATH,
    )

    # ---- Backtest ----
    print("\n  Walking-forward backtest (last 7 days)...")
    bt_result = backtest_last_n_days(df, n=7, offset=14)
    if bt_result is not None:
        bt_dates_str = [d.strftime("%Y-%m-%d") for d in bt_result["dates"]]
        print(f"  Backtest dates: {', '.join(bt_dates_str)}")
        for bt_name in ["sarima", "xgb"]:
            if bt_name in bt_result:
                print(f"  {bt_name.upper()} : {bt_result[f'{bt_name}_metrics']}")
        plot_backtest(bt_result,
                      save_path=os.path.join(
                          DEFAULT_FIGURE_PATH.rsplit("/", 1)[0], "backtest_last7days.png"))

    # ---- Save outputs ----
    json_path = save_json_path or os.path.join(
        DEFAULT_OUTPUT_DIR, f"forecast_{city.replace(' ', '_')}.json"
    )
    import json
    fc_dict = {
        "city": city,
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
        "models": {},
    }
    for name in all_model_preds:
        fc_dict["models"][name] = {
            "predictions_7day": [round(float(v), 2) for v in forecast_preds.get(name,
                                                all_model_preds[name][-FORECAST_HORIZON:])],
            "metrics": all_model_metrics[name],
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fc_dict, f, indent=2)
    print(f"  [Saved] {json_path}")

    print(f"\n  [Done] {city} forecast complete\n")
    return fc


def main():
    parser = argparse.ArgumentParser(description="Precipitation forecast — multi-model ensemble")
    parser.add_argument("--city", default="jakarta", help="City name (default: jakarta)")
    parser.add_argument("--years", type=int, default=10, help="Years of historical data (default: 10)")
    parser.add_argument("--input", default=None, help="Path to inputs_json/*.json (overrides --city)")
    parser.add_argument("--output", default=None, help="Output path for forecast JSON")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM model")
    parser.add_argument("--models", nargs="+",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Models to run (default: sarima xgb hgb ets rf [+lstm/patchtst/timesfm/tabicl if available])")
    parser.add_argument("--backtest-offset", type=int, default=14,
                        help="Days before end to start backtest window (default: 14)")
    args = parser.parse_args()

    if args.input:
        df, city_name = load_from_json(args.input)
        if df is None:
            sys.exit(1)
        feat_df = build_features(df)
        train_df, test_df = split_train_test(feat_df)
        print(f"\n{'='*60}")
        print(f"  PRECIPITATION FORECAST — {city_name.upper()}")
        print(f"  {len(train_df)+len(test_df)} days | Train: {len(train_df)} | Test: {len(test_df)}")
        print(f"{'='*60}\n")

        # Train all requested models
        models = args.models or ["sarima", "xgb", "hgb", "ets", "rf"]
        all_preds = {}
        all_metrics = {}
        all_states = {}
        forecast_preds = {}  # name -> 7-day forward forecast array

        for name in models:
            info = MODEL_REGISTRY.get(name)
            if not info:
                print(f"  [SKIP] Unknown model '{name}'")
                continue
            if "available" in info:
                avail = globals().get(info["available"], False)
                if not avail:
                    print(f"  [SKIP] {name.upper()} — not available")
                    continue
            print(f"> {name.upper()} ...")
            try:
                if name == "sarima":
                    res = info["train"](train_df)
                    pred, ci = info["test"](res, test_df)
                    metrics = evaluate(test_df["precip"].values, pred, "SARIMAX")
                    all_states["sarima"] = res
                    all_preds["sarima"] = pred
                    all_metrics["sarima"] = metrics
                elif name == "xgb":
                    model, feat_cols = info["train"](train_df)
                    pred = info["test"](model, feat_cols, test_df)
                    metrics = evaluate(test_df["precip"].values, pred, "XGBoost")
                    all_states["xgb"] = (model, feat_cols)
                    all_preds["xgb"] = pred
                    all_metrics["xgb"] = metrics
                elif name == "hgb":
                    model, feat_cols = info["train"](train_df)
                    pred = info["test"](model, feat_cols, test_df)
                    metrics = evaluate(test_df["precip"].values, pred, "HistGB")
                    all_states["hgb"] = (model, feat_cols)
                    all_preds["hgb"] = pred
                    all_metrics["hgb"] = metrics
                elif name == "ets":
                    res = info["train"](train_df)
                    pred, ci = info["test"](res, test_df)
                    metrics = evaluate(test_df["precip"].values, pred, "ETS")
                    all_states["ets"] = res
                    all_preds["ets"] = pred
                    all_metrics["ets"] = metrics
                elif name == "rf":
                    model, feat_cols = info["train"](train_df)
                    pred = info["test"](model, feat_cols, test_df)
                    metrics = evaluate(test_df["precip"].values, pred, "RF")
                    all_states["rf"] = (model, feat_cols)
                    all_preds["rf"] = pred
                    all_metrics["rf"] = metrics
                elif name == "patchtst":
                    series = train_df["precip"].values.astype(np.float32)
                    n_test = len(test_df)
                    eval_start = max(0, len(series) - 35)
                    preds_eval, actual_eval = [], []
                    for i in range(eval_start, len(series) - FORECAST_HORIZON):
                        p = forecast_patchtst(series[:i+1], horizon=FORECAST_HORIZON)
                        preds_eval.extend(p.tolist())
                        actual_eval.extend(series[i+1:i+1+FORECAST_HORIZON].tolist())
                    preds_eval = np.array(preds_eval)
                    actual_eval = np.array(actual_eval)
                    n_pts = preds_eval.size // FORECAST_HORIZON
                    preds_reshaped = preds_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                    actual_reshaped = actual_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                    metrics = evaluate(actual_reshaped.mean(axis=1), preds_reshaped.mean(axis=1), "PatchTST")
                    fc = forecast_patchtst(series, horizon=FORECAST_HORIZON)
                    all_preds["patchtst"] = np.tile(fc, int(np.ceil(n_test / FORECAST_HORIZON)))[:n_test]
                    all_metrics["patchtst"] = metrics
                    forecast_preds["patchtst"] = fc
                    print(f"  {metrics}")
                elif name == "timesfm":
                    series = train_df["precip"].values.astype(np.float32)
                    n_test = len(test_df)
                    eval_start = max(0, len(series) - 35)
                    preds_eval, actual_eval = [], []
                    for i in range(eval_start, len(series) - FORECAST_HORIZON):
                        p = forecast_timesfm(series[:i+1], horizon=FORECAST_HORIZON)
                        preds_eval.extend(p.tolist())
                        actual_eval.extend(series[i+1:i+1+FORECAST_HORIZON].tolist())
                    preds_eval = np.array(preds_eval)
                    actual_eval = np.array(actual_eval)
                    n_pts = preds_eval.size // FORECAST_HORIZON
                    preds_reshaped = preds_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                    actual_reshaped = actual_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                    metrics = evaluate(actual_reshaped.mean(axis=1), preds_reshaped.mean(axis=1), "TimesFM")
                    fc = forecast_timesfm(series, horizon=FORECAST_HORIZON)
                    all_preds["timesfm"] = np.tile(fc, int(np.ceil(n_test / FORECAST_HORIZON)))[:n_test]
                    all_metrics["timesfm"] = metrics
                    forecast_preds["timesfm"] = fc
                    print(f"  {metrics}")
                elif name == "tabicl":
                    from utils.precipitation_models import forecast_tabicl
                    series = train_df["precip"].values.astype(np.float32)
                    n_test = len(test_df)
                    eval_start = max(0, len(series) - 35)
                    preds_eval, actual_eval = [], []
                    for i in range(eval_start, len(series) - FORECAST_HORIZON):
                        p = forecast_tabicl(train_df.iloc[:i+1], horizon=FORECAST_HORIZON)
                        preds_eval.extend(p.tolist())
                        actual_eval.extend(series[i+1:i+1+FORECAST_HORIZON].tolist())
                    preds_eval = np.array(preds_eval)
                    actual_eval = np.array(actual_eval)
                    n_pts = preds_eval.size // FORECAST_HORIZON
                    preds_reshaped = preds_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                    actual_reshaped = actual_eval[:n_pts * FORECAST_HORIZON].reshape(n_pts, FORECAST_HORIZON)
                    metrics = evaluate(actual_reshaped.mean(axis=1), preds_reshaped.mean(axis=1), "TabICL")
                    fc = forecast_tabicl(train_df, horizon=FORECAST_HORIZON)
                    all_preds["tabicl"] = np.tile(fc, int(np.ceil(n_test / FORECAST_HORIZON)))[:n_test]
                    all_metrics["tabicl"] = metrics
                    forecast_preds["tabicl"] = fc
                    print(f"  {metrics}")
                else:
                    print(f"  [SKIP] {name} — not implemented in --input mode")
                    continue
                print(f"  {metrics}")
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")

        # Forecast table
        forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1),
                                       periods=FORECAST_HORIZON, freq="D")
        fc_dict = {"Date": forecast_dates}
        for name in all_preds:
            if name in forecast_preds:
                fc_dict[name] = np.round(forecast_preds[name], 2)
            else:
                fc_dict[name] = np.round(all_preds[name][-FORECAST_HORIZON:], 2)
        fc = pd.DataFrame(fc_dict)
        print(f"\n--- 7-Day Precipitation Forecast ---\n")
        print(fc.to_string(index=False))

        # Multi-model plot
        _plot_multi_model_comparison(test_df, all_preds, all_metrics)

        # Backtest
        bt_n = 7
        bt_offset = args.backtest_offset
        print(f"\n  Walking-forward backtest (days -{bt_offset+1} to -{bt_offset+bt_n} from end)...")
        bt_result = backtest_last_n_days(df, n=bt_n, offset=bt_offset)
        if bt_result is not None:
            bt_dates_str = [d.strftime("%Y-%m-%d") for d in bt_result["dates"]]
            print(f"  Backtest dates: {', '.join(bt_dates_str)}")
            for bt_name in ["sarima", "xgb"]:
                if bt_name in bt_result:
                    print(f"  {bt_name.upper()} : {bt_result[f'{bt_name}_metrics']}")
            bt_plot_path = os.path.join(
                DEFAULT_FIGURE_PATH.rsplit("/", 1)[0], "backtest_last7days.png"
            )
            plot_backtest(bt_result, save_path=bt_plot_path)

        # Save JSON
        json_path = args.output or os.path.join(
            DEFAULT_OUTPUT_DIR, f"forecast_{city_name.replace(' ', '_')}.json"
        )
        import json
        # Build 7-day forward forecasts from the last training point
        fc_dict_out = {"city": city_name,
                       "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
                       "models": {}}
        for name in all_preds:
            if name in all_states:
                state = all_states[name]
                try:
                    if name == "sarima":
                        fc = np.round(state.get_forecast(steps=FORECAST_HORIZON).predicted_mean.values[:FORECAST_HORIZON], 2)
                    elif name in ("xgb", "hgb", "rf"):
                        feat_cols = state[1]
                        fc = np.round(np.maximum(state[0].predict(feat_df[feat_cols].tail(FORECAST_HORIZON).values), 0), 2)
                    elif name == "ets":
                        fc = np.round(np.maximum(state.forecast(FORECAST_HORIZON).values[:FORECAST_HORIZON], 0), 2)
                    else:
                        fc = np.array([round(float(v), 2) for v in all_preds[name][-FORECAST_HORIZON:]])
                except Exception:
                    fc = np.array([round(float(v), 2) for v in all_preds[name][-FORECAST_HORIZON:]])
            else:
                fc = np.array([round(float(v), 2) for v in all_preds[name][-FORECAST_HORIZON:]])
            fc_dict_out["models"][name] = {"predictions_7day": [round(float(v), 2) for v in fc],
                                           "metrics": all_metrics[name]}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(fc_dict_out, f, indent=2)
        print(f"  [Saved] {json_path}")
        print()
        return

    run(args.city, args.years, include_lstm=not args.no_lstm, models=args.models)


if __name__ == "__main__":
    main()
