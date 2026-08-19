"""Top-level precipitation forecast runner.

Usage:
    python run_forecast.py                          # default: Jakarta, 10y, all models
    python run_forecast.py --city "jakarta"
    python run_forecast.py --input inputs_json/foo.json
    python run_forecast.py --no-lstm                # skip deep learning
    python run_forecast.py --years 5
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
    _LSTM_AVAILABLE,
    evaluate,
    save_pipeline,
)
from utils.plotting import plot_forecast

FORECAST_HORIZON = 7
DEFAULT_OUTPUT_DIR = "outputs/predictions"
DEFAULT_MODEL_PATH = "data/processed/precipitation_model.joblib"
DEFAULT_FIGURE_PATH = "outputs/figures/precipitation_forecast.png"


def run(city, years, include_lstm=True, save_json_path=None):
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DEFAULT_FIGURE_PATH), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PRECIPITATION FORECAST — {city.upper()}")
    print(f"  {years} years of daily data  |  Horizon: {FORECAST_HORIZON} days")
    print(f"{'='*60}\n")

    # 1. Load data
    df, source_file = load_precipitation_data(city, years)
    print(f"  Date range: {df.index[0].date()} - {df.index[-1].date()} ({len(df)} days)\n")

    # 2. Features
    feat_df = build_features(df)
    train_df, test_df = split_train_test(feat_df)
    print(f"  Train: {len(train_df)} days | Test: {len(test_df)} days\n")

    # -- SARIMAX --
    print("> SARIMAX(1,0,1)(0,1,1,7) ...")
    sarima_res = sarima_fit(train_df)
    sarima_pred, sarima_test_ci = sarima_test_predictions(sarima_res, test_df)
    sarima_metrics = evaluate(test_df["precip"].values, sarima_pred, "SARIMAX")
    print(f"  {sarima_metrics}")

    # -- XGBoost --
    print("> XGBoost ...")
    xgb_model, feat_cols = train_xgboost(train_df)
    xgb_pred = forecast_xgboost(xgb_model, feat_cols, test_df)
    xgb_metrics = evaluate(test_df["precip"].values, xgb_pred, "XGBoost")
    print(f"  {xgb_metrics}")

    # -- LSTM (optional) --
    lstm_metrics = None
    lstm_pred = None
    lstm_ok = False
    if include_lstm and _LSTM_AVAILABLE:
        print("> LSTM ...")
        from utils.precipitation_models import (
            prepare_lstm_sequence, train_lstm, forecast_lstm,
        )
        X_all, y_all, scaler = prepare_lstm_sequence(feat_df)
        split = int(len(X_all) * 0.85)
        X_tr, y_tr = X_all[:split], y_all[:split]
        lstm_model = train_lstm(X_tr, y_tr, epochs=20, batch=32)
        lstm_pred = forecast_lstm(lstm_model, feat_df["precip"].values, scaler)
        actual_tail = feat_df["precip"].iloc[-len(lstm_pred):].values
        lstm_metrics = evaluate(actual_tail, lstm_pred, "LSTM")
        lstm_ok = True
        print(f"  {lstm_metrics}")
    elif include_lstm and not _LSTM_AVAILABLE:
        print(f"  LSTM skipped — {_LSTM_ERROR}")

    # -- 7-day forward forecast --
    print("\n--- 7-Day Precipitation Forecast ---\n")
    forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1),
                                   periods=FORECAST_HORIZON, freq="D")
    sarima_fc, sarima_ci = sarima_forward_forecast(sarima_res, feat_df, FORECAST_HORIZON)
    future_exog_xgb = feat_df[feat_cols].tail(FORECAST_HORIZON).values
    xgb_fc = np.maximum(xgb_model.predict(future_exog_xgb), 0)

    fc = pd.DataFrame({
        "Date": forecast_dates,
        "SARIMAX": np.round(sarima_fc, 2),
        "XGBoost": np.round(xgb_fc, 2),
    })
    if lstm_ok:
        fc["LSTM"] = np.round(lstm_pred, 2)

    fc_low = np.round(sarima_ci[:, 0], 2)
    fc_high = np.round(sarima_ci[:, 1], 2)
    fc["SARIMAX_Lo95"] = fc_low
    fc["SARIMAX_Hi95"] = fc_high

    print(fc.to_string(index=False))

    # -- Plot --
    plot_forecast(
        save_path=DEFAULT_FIGURE_PATH,
        train_df=train_df, test_df=test_df,
        sarima_pred=sarima_pred, sarima_test_ci=sarima_test_ci,
        xgb_pred=xgb_pred, lstm_pred=lstm_pred,
        fc=fc, fc_low=fc_low, fc_high=fc_high,
        city=city, lstm_ok=lstm_ok,
        sarima_rmse=sarima_metrics["RMSE"], sarima_mae=sarima_metrics["MAE"],
        xgb_rmse=xgb_metrics["RMSE"], xgb_mae=xgb_metrics["MAE"],
        lstm_rmse=lstm_metrics["RMSE"] if lstm_metrics else None,
        lstm_mae=lstm_metrics["MAE"] if lstm_metrics else None,
    )

    # Save model + metadata
    meta = {
        "city": city, "years": years, "source": source_file,
        "sarima_order": "(1,0,1)(0,1,1,7)",
        "forecast_horizon": FORECAST_HORIZON,
        "metrics": [sarima_metrics, xgb_metrics] + ([lstm_metrics] if lstm_ok else []),
    }
    save_pipeline(sarima_res, xgb_model, feat_cols, meta, DEFAULT_MODEL_PATH)

    # Save JSON forecast
    json_path = save_json_path or os.path.join(
        DEFAULT_OUTPUT_DIR, f"forecast_{city.replace(' ', '_')}.json"
    )
    save_forecast_to_json(city, forecast_dates, sarima_fc, xgb_fc, sarima_ci,
                          [sarima_metrics, xgb_metrics], json_path)

    print(f"\n  [Done] {city} forecast complete\n")
    return fc


def main():
    parser = argparse.ArgumentParser(description="Precipitation forecast — SARIMAX / XGBoost / LSTM")
    parser.add_argument("--city", default="jakarta", help="City name (default: jakarta)")
    parser.add_argument("--years", type=int, default=10, help="Years of historical data (default: 10)")
    parser.add_argument("--input", default=None, help="Path to inputs_json/*.json (overrides --city)")
    parser.add_argument("--output", default=None, help="Output path for forecast JSON")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM model")
    args = parser.parse_args()

    if args.input:
        df, city_name = load_from_json(args.input)
        if df is None:
            sys.exit(1)
        feat_df = build_features(df)
        train_df, test_df = split_train_test(feat_df)
        print(f"\n{'='*60}")
        print(f"  PRECIPITATION FORECAST - {city_name.upper()}")
        print(f"  {len(train_df)+len(test_df)} days | Train: {len(train_df)} | Test: {len(test_df)}")
        print(f"{'='*60}\n")

        sarima_res = sarima_fit(train_df)
        sarima_pred, sarima_test_ci = sarima_test_predictions(sarima_res, test_df)
        sarima_metrics = evaluate(test_df["precip"].values, sarima_pred, "SARIMAX")

        xgb_model, feat_cols = train_xgboost(train_df)
        xgb_pred = np.maximum(xgb_model.predict(test_df[feat_cols].values), 0)
        xgb_metrics = evaluate(test_df["precip"].values, xgb_pred, "XGBoost")
        print(f"  SARIMAX: {sarima_metrics}")
        print(f"  XGBoost: {xgb_metrics}")

        forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1),
                                       periods=FORECAST_HORIZON, freq="D")
        sarima_fc, sarima_ci = sarima_forward_forecast(sarima_res, feat_df, FORECAST_HORIZON)
        xgb_fc = np.maximum(xgb_model.predict(feat_df[feat_cols].tail(FORECAST_HORIZON).values), 0)

        fc = pd.DataFrame({"Date": forecast_dates, "SARIMAX": np.round(sarima_fc, 2),
                           "XGBoost": np.round(xgb_fc, 2)})
        fc_low = np.round(sarima_ci[:, 0], 2)
        fc_high = np.round(sarima_ci[:, 1], 2)
        fc["SARIMAX_Lo95"] = fc_low
        fc["SARIMAX_Hi95"] = fc_high

        print(f"\n--- 7-Day Precipitation Forecast ---\n")
        print(fc.to_string(index=False))

        json_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR,
                                                 f"forecast_{city_name.replace(' ', '_')}.json")
        save_forecast_to_json(city_name, forecast_dates, sarima_fc, xgb_fc, sarima_ci,
                              [sarima_metrics, xgb_metrics], json_path)

        # Save model too
        meta = {"city": city_name, "source": args.input,
                "metrics": [sarima_metrics, xgb_metrics]}
        save_pipeline(sarima_res, xgb_model, feat_cols, meta, DEFAULT_MODEL_PATH)
        print()
        return

    run(args.city, args.years, include_lstm=not args.no_lstm)


if __name__ == "__main__":
    main()
