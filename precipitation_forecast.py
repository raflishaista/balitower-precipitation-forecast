"""
Precipitation Forecast — ARIMA / SARIMA / XGBoost / LSTM
=========================================================
Trains three models on 10 years of daily precipitation data and
produces a 7-day forecast with comparison plots.

Models:
  SARIMAX  — seasonal ARIMA with exogenous features (day-of-year cycle)
  XGBoost  — gradient-boosted trees with lag/rolling features
  LSTM     — deep sequence model (requires tensorflow; gracefully skipped
             if unavailable or Python ABI incompatible)

Usage:
  python precipitation_forecast.py                        # default: Jakarta, 10y
  python precipitation_forecast.py --city "jakarta"       # by name
  python precipitation_forecast.py --file outputs/foo.xlsx --col Precipitation
  python precipitation_forecast.py --years 5              # shorter history
  python precipitation_forecast.py --no-lstm              # skip deep learning
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import dump, load

warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("seaborn-v0_8-whitegrid")

# -- constants -----------------------------------------------------------------
FORECAST_HORIZON = 7
TRAIN_SPLIT = 0.85  # last 15% held out for validation
LATENT_DIM = 32
LSTM_SEQ_LEN = 30
FORECAST_PATH = "outputs/precipitation_forecast.png"
MODEL_PATH = "outputs/precipitation_model.joblib"

# -- helpers -------------------------------------------------------------------

def load_precipitation_data(city_name="Jakarta", years=10):
    """Load precipitation column from existing Excel or fetch from NASA POWER."""
    # Try existing file first
    import os, glob
    candidates = glob.glob(f"outputs/*{city_name.replace(' ', '_')}*weather*.xlsx") + \
                 glob.glob(f"outputs/*{city_name.lower().replace(' ', '_')}*precipitation*.xlsx")
    if candidates:
        path = sorted(candidates, key=os.path.getmtime, reverse=True)[0]
        df = pd.read_excel(path)
        if "Precipitation" in df.columns:
            col = "Precipitation"
        elif "Total Precipitation (mm)" in df.columns:
            col = "Total Precipitation (mm)"
        else:
            col = [c for c in df.columns if "precip" in c.lower()][0]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").resample("D").sum().interpolate(method="linear").dropna()
        df = df[df.index >= (datetime.now() - timedelta(days=years * 365))]
        print(f"  Loaded {len(df)} days from {path}")
        return df[[col]].rename(columns={col: "precip"}), path

    # Fetch from NASA POWER
    from weather import NASAPowerWeather
    api = NASAPowerWeather()
    from weather import geocode_location, format_display_name
    result = geocode_location(city_name)
    if not result:
        raise ValueError(f"Cannot geocode '{city_name}'")
    lat, lon, display = result
    name = format_display_name(display, city_name)
    start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")
    df = api.fetch(lat, lon, start, end)
    if df is None or "PRECTOTCORR" not in df.columns:
        raise ValueError("Could not fetch precipitation data")
    df = df[["Date", "PRECTOTCORR"]].rename(columns={"Date": "date", "PRECTOTCORR": "precip"})
    df = df.set_index("date").resample("D").sum().interpolate(method="linear").dropna()
    print(f"  Fetched {len(df)} days from NASA POWER for {name}")
    return df, None


def build_features(df, lags=(1, 2, 3, 7, 14, 30), windows=(7, 14, 30)):
    """Add lag, rolling-mean, rolling-min, rolling-max, and cyclical time features."""
    df = df.copy()
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["year"] = df.index.year
    # Cyclical encoding for seasonality
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    for lag in lags:
        df[f"lag_{lag}"] = df["precip"].shift(lag)
    for w in windows:
        df[f"roll_mean_{w}"] = df["precip"].shift(1).rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df["precip"].shift(1).rolling(w, min_periods=1).std()
        df[f"roll_min_{w}"] = df["precip"].shift(1).rolling(w, min_periods=1).min()
    df = df.dropna().reset_index(drop=True)
    return df


def split_train_test(df, split=TRAIN_SPLIT):
    split_idx = int(len(df) * split)
    return df.iloc[:split_idx], df.iloc[split_idx:]


# -- SARIMAX -------------------------------------------------------------------

def sarima_fit(train_df):
    """Fit SARIMAX model and return results + exog arrays for re-use."""
    endog = train_df["precip"]
    exog = train_df[["doy_sin", "doy_cos"]].values
    model = SARIMAX(endog, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7),
                    exogenous=exog, enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=100, method="lbfgs")
    return results


def sarima_test_predictions(results, test_df):
    """Return in-sample predictions aligned to the test set."""
    test_exog = test_df[["doy_sin", "doy_cos"]].values
    pred = results.get_forecast(steps=len(test_df), exog=test_exog)
    preds = pred.predicted_mean.values
    preds = np.maximum(preds, 0)
    return preds, pred.conf_int().values


def sarima_forward_forecast(results, feat_df, horizon=FORECAST_HORIZON):
    """Return N-step ahead point forecast + confidence interval."""
    exog_tail = feat_df[["doy_sin", "doy_cos"]].tail(horizon).values
    pred = results.get_forecast(steps=horizon, exog=exog_tail)
    preds = pred.predicted_mean.values
    preds = np.maximum(preds, 0)
    return preds, pred.conf_int().values


# -- XGBoost -------------------------------------------------------------------

def train_xgboost(train_df):
    import xgboost as xgb
    feature_cols = [c for c in train_df.columns if c not in ("precip", "day_of_year", "month", "year")]
    X, y = train_df[feature_cols].values, train_df["precip"].values
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y)
    return model, feature_cols


def forecast_xgboost(model, feature_cols, test_df, horizon=FORECAST_HORIZON):
    X_test = test_df[feature_cols].values
    preds = np.maximum(model.predict(X_test), 0)  # precipitation cannot be negative
    return preds


# -- LSTM (optional) -----------------------------------------------------------

_LSTM_AVAILABLE = False
_LSTM_ERROR = ""

try:
    import tensorflow as tf
    _LSTM_AVAILABLE = True
except Exception as e:
    _LSTM_ERROR = str(e)


def build_lstm_model(input_dim, latent=LATENT_DIM):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    model = Sequential([
        LSTM(latent, input_shape=(LSTM_SEQ_LEN, input_dim), return_sequences=False),
        Dropout(0.2),
        LSTM(latent // 2),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_lstm_sequence(df, seq_len=LSTM_SEQ_LEN):
    from sklearn.preprocessing import MinMaxScaler
    prec = df["precip"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    prec_scaled = scaler.fit_transform(prec).flatten()

    X, y = [], []
    for i in range(seq_len, len(prec_scaled)):
        X.append(prec_scaled[i - seq_len:i])
        y.append(prec_scaled[i])
    return np.array(X), np.array(y), scaler


def train_lstm(X_train, y_train, epochs=30, batch=32):
    import tensorflow as tf
    model = build_lstm_model(1)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
              validation_split=0.15, verbose=0)
    return model


def forecast_lstm(model, full_series, scaler, horizon=FORECAST_HORIZON):
    last_seq = full_series[-LSTM_SEQ_LEN:].reshape(1, LSTM_SEQ_LEN, 1)
    preds_scaled = np.zeros(horizon)
    current = last_seq
    for i in range(horizon):
        step = model.predict(current, verbose=0)
        preds_scaled[i] = step[0, 0]
        current = np.concatenate([current[:, 1:], step.reshape(1, 1, 1)], axis=1)
    return scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()


# -- evaluation ----------------------------------------------------------------

def evaluate(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4)}


# -- main pipeline -------------------------------------------------------------

def run(city, years, include_lstm=True):
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
    if include_lstm:
        if not _LSTM_AVAILABLE:
            print(f"⚠ LSTM skipped — {type(tf).__name__ if 'tf' in dir() else _LSTM_ERROR}")
        else:
            print("> LSTM ...")
            X_all, y_all, scaler = prepare_lstm_sequence(feat_df)
            split = int(len(X_all) * TRAIN_SPLIT)
            X_tr, y_tr = X_all[:split], y_all[:split]
            lstm_model = train_lstm(X_tr, y_tr, epochs=20, batch=32)
            lstm_pred = forecast_lstm(lstm_model, feat_df["precip"].values, scaler)
            actual_tail = feat_df["precip"].iloc[-len(lstm_pred):].values
            lstm_metrics = evaluate(actual_tail, lstm_pred, "LSTM")
            lstm_ok = True
            print(f"  {lstm_metrics}")

    # -- 7-day forward forecast --
    print("\n--- 7-Day Precipitation Forecast ---\n")
    forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1),
                                   periods=FORECAST_HORIZON, freq="D")
    sarima_fc, sarima_ci = sarima_forward_forecast(sarima_res, feat_df, FORECAST_HORIZON)

    # XGBoost forward forecast
    future_exog_xgb = feat_df[feat_cols].tail(FORECAST_HORIZON).values
    xgb_fc = xgb_model.predict(future_exog_xgb)

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
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

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
    ax1.set_title("SARIMAX & XGBoost — Test Period"); ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlabel(""); ax1.set_ylabel("Precipitation (mm)")

    # Plot 2: Historical + XGBoost predictions on test
    ax2 = axes[0, 1]
    ax2.plot(train_df.index, train_df["precip"], color="steelblue", label="Train", alpha=0.7)
    ax2.plot(test_df.index, test_df["precip"], color="grey", label="Actual (test)", alpha=0.7)
    ax2.axvline(test_idx, color="grey", linestyle="--", alpha=0.5)
    ax2.plot(test_df.index, xgb_pred, color="darkorange", label="XGBoost pred", lw=2)
    ax2.set_title("XGBoost — Test Period"); ax2.legend(loc="upper left", fontsize=8)
    ax2.set_xlabel(""); ax2.set_ylabel("Precipitation (mm)")

    # Plot 3: 7-day forecast
    ax3 = axes[1, 0]
    ax3.bar(fc["Date"], fc["SARIMAX"], color="steelblue", alpha=0.7, label="SARIMAX")
    ax3.bar([pd.Timestamp(d) + pd.Timedelta("1D") for d in fc["Date"]], fc["XGBoost"], color="darkorange", alpha=0.7, label="XGBoost")
    if lstm_ok:
        ax3.bar([pd.Timestamp(d) + pd.Timedelta("2D") for d in fc["Date"]], fc["LSTM"], color="purple", alpha=0.7, label="LSTM")
    ax3.fill_between(fc["Date"], fc_low, fc_high, color="steelblue", alpha=0.12, label="SARIMAX 95% CI")
    ax3.set_title(f"7-Day Precipitation Forecast ({city})")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax3.tick_params(axis="x", rotation=45)
    ax3.set_ylabel("Precipitation (mm)")
    ax3.axhline(0, color="black", linewidth=0.5)

    # Plot 4: Model comparison bar chart
    ax4 = axes[1, 1]
    models = ["SARIMAX", "XGBoost"]
    rmse_vals = [sarima_metrics["RMSE"], xgb_metrics["RMSE"]]
    mae_vals = [sarima_metrics["MAE"], xgb_metrics["MAE"]]
    if lstm_ok:
        models.append("LSTM")
        rmse_vals.append(lstm_metrics["RMSE"])
        mae_vals.append(lstm_metrics["MAE"])
    x = np.arange(len(models))
    w = 0.35
    ax4.bar(x - w/2, rmse_vals, w, label="RMSE", color="steelblue")
    ax4.bar(x + w/2, mae_vals, w, label="MAE", color="darkorange")
    ax4.set_title("Test-Period Model Comparison"); ax4.set_xticks(x)
    ax4.set_xticklabels(models); ax4.legend(); ax4.set_ylabel("Error (mm)")
    for i, v in enumerate(rmse_vals):
        ax4.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FORECAST_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Saved] {FORECAST_PATH}")

    # Save model + metadata for re-use
    meta = {
        "city": city, "years": years, "source": source_file,
        "sarima_order": "(1,0,1)(0,1,1,7)","sarima_period": sarima_res.arfreq,
        "forecast_horizon": FORECAST_HORIZON,
        "metrics": [sarima_metrics, xgb_metrics] + ([lstm_metrics] if lstm_ok else []),
    }
    dump({"model_sarima": sarima_res, "model_xgboost": xgb_model,
          "feature_cols": feat_cols, "meta": meta}, MODEL_PATH)
    print(f"  [Saved] {MODEL_PATH}\n")
    return fc


def main():
    parser = argparse.ArgumentParser(description="Precipitation forecast — SARIMAX / XGBoost / LSTM")
    parser.add_argument("--city", default="jakarta", help="City name (default: jakarta)")
    parser.add_argument("--years", type=int, default=10, help="Years of historical data (default: 10)")
    parser.add_argument("--file", default=None, help="Path to existing Excel file (overrides --city)")
    parser.add_argument("--col", default=None, help="Column name in Excel (default: auto-detect)")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM model")
    args = parser.parse_args()

    if args.file:
        df = pd.read_excel(args.file)
        if args.col:
            col = args.col
        elif "Precipitation" in df.columns:
            col = "Precipitation"
        elif "Total Precipitation (mm)" in df.columns:
            col = "Total Precipitation (mm)"
        else:
            col = [c for c in df.columns if "precip" in c.lower()][0]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").resample("D").sum().interpolate(method="linear").dropna()
        print(f"Loaded {len(df)} days from {args.file}")
        print(f"Date range: {df.index[0].date()} - {df.index[-1].date()}\n")
        feat_df = build_features(df)
        train_df, test_df = split_train_test(feat_df)
        sarima_res = train_sarimax(train_df)
        sarima_pred, sarima_ci = forecast_sarimax(sarima_res, test_df)
        sarima_metrics = evaluate(test_df["precip"].values, sarima_pred, "SARIMAX")
        xgb_model, feat_cols = train_xgboost(train_df)
        xgb_pred = xgb_model.predict(test_df[feat_cols].values)
        xgb_metrics = evaluate(test_df["precip"].values, xgb_pred, "XGBoost")
        print(f"SARIMAX: {sarima_metrics}  XGBoost: {xgb_metrics}\n")

        forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1),
                                       periods=FORECAST_HORIZON, freq="D")
        future_exog = feat_df[["doy_sin", "doy_cos"]].tail(FORECAST_HORIZON).values
        sarima_fc, sarima_ci_fc = forecast_sarimax(sarima_res, feat_df, FORECAST_HORIZON)
        xgb_fc = xgb_model.predict(feat_df[feat_cols].tail(FORECAST_HORIZON).values)
        fc = pd.DataFrame({"Date": forecast_dates, "SARIMAX": np.round(sarima_fc, 2),
                           "XGBoost": np.round(xgb_fc, 2)})
        print(fc.to_string(index=False))
        dump({"meta": {"city": args.file, "source": args.file,
                       "metrics": [sarima_metrics, xgb_metrics]}}, MODEL_PATH)
        print(f"[Saved] {MODEL_PATH}")
        return

    run(args.city, args.years, include_lstm=not args.no_lstm)


if __name__ == "__main__":
    main()
