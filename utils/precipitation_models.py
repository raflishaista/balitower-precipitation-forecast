"""Precipitation forecasting models — SARIMAX / XGBoost / LSTM.

Public API:
    load_precipitation_data(city_name, years)  -> (DataFrame, source_path)
    load_from_json(json_path)                  -> (DataFrame, city_name)
    build_features(df, lags, windows)          -> DataFrame with engineered features
    split_train_test(df, split)               -> (train_df, test_df)
    sarima_fit(train_df)                       -> SARIMAX Results
    sarima_test_predictions(results, test_df)  -> (preds, conf_int)
    sarima_forward_forecast(results, feat_df, horizon) -> (preds, conf_int)
    train_xgboost(train_df)                   -> (model, feature_cols)
    forecast_xgboost(model, cols, test_df)    -> preds
    evaluate(y_true, y_pred, name)            -> dict
    save_pipeline(sarima_res, xgb_model, cols, meta, path)
    save_forecast_to_json(city, dates, sarima_fc, xgb_fc, ci, metrics, path)
"""
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import dump, load

warnings.filterwarnings("ignore", category=FutureWarning)

FORECAST_HORIZON = 7
TRAIN_SPLIT = 0.85
LATENT_DIM = 32
LSTM_SEQ_LEN = 30

_LSTM_AVAILABLE = False
_LSTM_ERROR = ""

try:
    import tensorflow as tf
    _LSTM_AVAILABLE = True
except Exception as e:
    _LSTM_ERROR = str(e)


def load_precipitation_data(city_name="Jakarta", years=10):
    import os, glob
    candidates = glob.glob(f"data/raw/*{city_name.replace(' ', '_')}*weather*.xlsx") + \
                 glob.glob(f"data/raw/*{city_name.lower().replace(' ', '_')}*precipitation*.xlsx") + \
                 glob.glob(f"data/raw/*{city_name.replace(' ', '_')}*precipitation*.xlsx")
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

    from utils.weather_collector import NASAPowerWeather, geocode_location, format_display_name
    api = NASAPowerWeather()
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


def load_from_json(json_path):
    """Load precipitation (and optional weather) data from JSON.
    Supports both legacy schema (just dates+precip_mm) and extended schema
    (with humidity_mm, pressure_kPa, windspeed_ms, temperature_c, etc.).
    Trims trailing -999 (NASA POWER missing) values from ALL columns.
    """
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame({'Date': data['dates'], 'precip': data['precip_mm']})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').asfreq('D')
    # Replace NASA POWER missing-value sentinel (-999) with NaN
    df = df.replace(-999.0, np.nan)
    # Trim trailing NaN rows BEFORE loading exogenous columns (arrays may differ in length)
    df = df.dropna(how='all')
    valid_len = len(df)
    city = data.get('city', json_path)
    # Load optional exogenous weather columns if present
    for col_map in [
        ('humidity', 'humidity'),       # RH2M %
        ('rh2m', 'humidity'),           # NASA POWER code
        ('pressure', 'pressure'),       # PS kPa
        ('ps', 'pressure'),             # NASA POWER code
        ('windspeed', 'windspeed'),     # WS10M m/s
        ('ws10m', 'windspeed'),         # NASA POWER code
        ('temperature', 'temperature'), # T2M C
        ('t2m', 'temperature'),         # NASA POWER code
        ('wind_direction', 'wind_dir'), # WD10M degrees
    ]:
        key, target = col_map
        if key in data:
            vals = data[key]
            # Trim to valid_len so it aligns with df after trailing-NaN removal
            vals = vals[:valid_len]
            df[target] = vals
    # Now replace any remaining -999 in exogenous cols with NaN and interpolate
    exog_cols = [c for c in df.columns if c not in ('precip',)]
    df = df.replace(-999.0, np.nan)
    df[exog_cols] = df[exog_cols].interpolate(method='linear')
    df = df.dropna()
    print(f'  Loaded {len(df)} days from {json_path} (city: {city})')
    return df, city


def save_forecast_to_json(city, forecast_dates, sarima_fc, xgb_fc, sarima_ci,
                          metrics, output_path):
    import json
    result = {
        'city': city,
        'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'sarimax_forecast': [round(float(v), 2) for v in sarima_fc],
        'sarimax_ci_lower': [round(float(v), 2) for v in sarima_ci[:, 0]],
        'sarimax_ci_upper': [round(float(v), 2) for v in sarima_ci[:, 1]],
        'xgboost_forecast': [round(float(v), 2) for v in xgb_fc],
        'test_metrics': metrics,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f'  [Saved] {output_path}')


def build_features(df, lags=(1, 2, 3, 7, 14, 30), windows=(7, 14, 30)):
    """Build feature matrix from precipitation + optional exogenous weather columns.

    Available exogenous columns (auto-detected if present):
      - humidity   : relative humidity % (RH2M)
      - pressure   : surface pressure kPa (PS)
      - windspeed  : 10m wind speed m/s (WS10M)
      - temperature: 2m temperature C (T2M)
      - wind_dir   : wind direction degrees (WD10M)
    """
    df = df.copy()
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    # Precipitation history features
    for lag in lags:
        df[f"lag_{lag}"] = df["precip"].shift(lag)
    for w in windows:
        df[f"roll_mean_{w}"] = df["precip"].shift(1).rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df["precip"].shift(1).rolling(w, min_periods=1).std()
        df[f"roll_min_{w}"] = df["precip"].shift(1).rolling(w, min_periods=1).min()
    # Exogenous weather features (if available)
    _EXOG_MAP = {
        "humidity": None,      # RH2M %
        "pressure": None,      # PS kPa
        "windspeed": None,     # WS10M m/s
        "temperature": None,   # T2M C
        "wind_dir": None,      # WD10M degrees
    }
    # Find which exogenous columns exist in the data
    available_exog = [col for col in _EXOG_MAP if col in df.columns]
    if available_exog:
        # Lagged exogenous features (lag 1 and 7)
        for col in available_exog:
            for lag in (1, 7):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        # Derived interaction features
        if "humidity" in available_exog and "temperature" in available_exog:
            df["temp_humid_product"] = df["temperature"] * df["humidity"] / 100
        if "pressure" in available_exog:
            df["pressure_trend"] = df["pressure"] - df["pressure"].shift(1)
        if "humidity" in available_exog:
            df["humidity_trend"] = df["humidity"] - df["humidity"].shift(1)
        if "wind_dir" in available_exog:
            df["wind_dir_sin"] = np.sin(2 * np.pi * df["wind_dir"] / 360)
            df["wind_dir_cos"] = np.cos(2 * np.pi * df["wind_dir"] / 360)
    df = df.dropna().reset_index(drop=True)
    return df


def split_train_test(df, split=TRAIN_SPLIT):
    split_idx = int(len(df) * split)
    return df.iloc[:split_idx], df.iloc[split_idx:]


# -- SARIMAX -------------------------------------------------------------------

def _sarima_exog_cols(df):
    """Return list of exogenous column names for SARIMAX, ordered consistently."""
    cols = ["doy_sin", "doy_cos"]
    _EXOG_ORDER = ["humidity_lag1", "humidity_lag7", "pressure_lag1", "pressure_lag7",
                   "windspeed_lag1", "windspeed_lag7", "temperature_lag1", "temperature_lag7",
                   "temp_humid_product", "pressure_trend", "humidity_trend",
                   "wind_dir_sin", "wind_dir_cos"]
    cols += [c for c in _EXOG_ORDER if c in df.columns]
    return cols


def sarima_fit(train_df):
    endog = train_df["precip"]
    exog_cols = _sarima_exog_cols(train_df)
    exog = train_df[exog_cols].values if exog_cols else None
    model = SARIMAX(endog, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7),
                    exogenous=exog, enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=100, method="lbfgs")
    results._exog_cols = exog_cols  # store for later use
    return results


def sarima_test_predictions(results, test_df):
    exog_cols = getattr(results, '_exog_cols', ["doy_sin", "doy_cos"])
    test_exog = test_df[exog_cols].values if exog_cols else None
    pred = results.get_forecast(steps=len(test_df), exog=test_exog)
    preds = np.asarray(pred.predicted_mean)[:horizon]
    preds = np.maximum(preds, 0)
    return preds, pred.conf_int().values


def sarima_forward_forecast(results, feat_df, horizon=FORECAST_HORIZON):
    exog_cols = getattr(results, '_exog_cols', ["doy_sin", "doy_cos"])
    exog_tail = feat_df[exog_cols].tail(horizon).values if exog_cols else None
    pred = results.get_forecast(steps=horizon, exog=exog_tail)
    preds = np.asarray(pred.predicted_mean)[:horizon]
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
    preds = np.maximum(model.predict(X_test), 0)
    return preds


# -- LSTM (optional) -----------------------------------------------------------

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


# -- HistGradientBoosting (sklearn, no install needed) --------------------------

def train_histgb(train_df):
    """Train HistGradientBoostingRegressor on feature columns."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    feature_cols = [c for c in train_df.columns if c not in ("precip", "day_of_year", "month", "year")]
    X, y = train_df[feature_cols].values, train_df["precip"].values
    model = HistGradientBoostingRegressor(
        max_iter=200, max_depth=6, learning_rate=0.05,
        random_state=42, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, min_samples_leaf=5,
    )
    model.fit(X, y)
    return model, feature_cols


def forecast_histgb(model, feature_cols, test_df):
    preds = np.maximum(model.predict(test_df[feature_cols].values), 0)
    return preds


# -- ETS / Holt-Winters (statsmodels, no install needed) -----------------------

def ets_fit(train_df):
    """Fit Holt-Winters Exponential Smoothing with additive trend & multiplicative seasonality."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    endog = train_df["precip"]
    # Additive trend, multiplicative seasonal (handles zero/near-zero values safely with additive)
    model = ExponentialSmoothing(
        endog,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated",
    )
    results = model.fit(optimized=True, use_brute=True)
    return results


def ets_test_predictions(results, test_df):
    """ETS Forecast using Holt-Winters .forecast() method (no confidence intervals in statsmodels 0.14)."""
    n = len(test_df)
    preds = np.maximum(results.forecast(n).values, 0)
    # Approximate CI via residual std
    resid_std = float(np.std(results.resid.dropna()))
    ci_lower = preds - 1.96 * resid_std
    ci_upper = preds + 1.96 * resid_std
    return preds, np.column_stack([ci_lower, ci_upper])


def ets_forward_forecast(results, horizon=FORECAST_HORIZON):
    preds = np.maximum(results.forecast(horizon).values, 0)
    resid_std = float(np.std(results.resid.dropna()))
    ci_lower = preds - 1.96 * resid_std
    ci_upper = preds + 1.96 * resid_std
    return preds, np.column_stack([ci_lower, ci_upper])


# -- Random Forest (sklearn, no install needed) ---------------------------------

def train_rf(train_df):
    from sklearn.ensemble import RandomForestRegressor
    feature_cols = [c for c in train_df.columns if c not in ("precip", "day_of_year", "month", "year")]
    X, y = train_df[feature_cols].values, train_df["precip"].values
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
        min_samples_split=5, min_samples_leaf=2,
    )
    model.fit(X, y)
    return model, feature_cols


def forecast_rf(model, feature_cols, test_df):
    preds = np.maximum(model.predict(test_df[feature_cols].values), 0)
    return preds


# -- Transformer models (conditional, requires torch + transformers) ------------

_TRANSFORMER_AVAILABLE = False
_TRANSFORMER_ERROR = ""
_PATCHTST_AVAILABLE = False
_PATCHTST_ERROR = ""
_TIMESFM_AVAILABLE = False
_TIMESFM_ERROR = ""
_TABICL_AVAILABLE = False
_TABICL_ERROR = ""

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from transformers import PatchTSTForPrediction, TimesFm2_5ModelForPrediction
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

if _HAS_TORCH and _HAS_TRANSFORMERS:
    _TRANSFORMER_AVAILABLE = True
else:
    _LSTM_ERROR += f"\n  Transformers skipped: torch={'OK' if _HAS_TORCH else 'MISSING'}, transformers={'OK' if _HAS_TRANSFORMERS else 'MISSING'}"

try:
    import tabicl
    _TABICL_AVAILABLE = True
except ImportError as e:
    _TABICL_ERROR = str(e)


# ---- PatchTST (IBM) — zero-shot pretrained, no fine-tuning needed -------------

_PATCHTST_MODEL_ID = "ibm-research/patchtst-etth1-pretrain"
_PATCHTST_CONTEXT = 128
_PATCHTST_PRED_LEN = 7


def _get_patchtst_model():
    """Load PatchTST for single-channel precipitation forecasting."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(_PATCHTST_MODEL_ID)
    cfg.num_input_channels = 1
    cfg.context_length = _PATCHTST_CONTEXT
    cfg.prediction_length = _PATCHTST_PRED_LEN
    cfg.mask_ratio = 0.0
    model = PatchTSTForPrediction.from_pretrained(_PATCHTST_MODEL_ID, config=cfg)
    model.eval()
    return model


def forecast_patchtst(series, horizon=_PATCHTST_PRED_LEN):
    """Run PatchTST zero-shot forecast on a 1-D precip series.

    Parameters
    ----------
    series : np.ndarray  — daily precipitation values (1-D)
    horizon : int
    Returns
    -------
    np.ndarray of shape (horizon,) with predicted precipitation, clipped >= 0.
    """
    import torch
    x = torch.tensor(series[-_PATCHTST_CONTEXT:].astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        out = _get_patchtst_model()(past_values=x)
    preds = out.prediction_outputs.squeeze().numpy()[:horizon]
    return np.maximum(preds, 0)


# ---- TimesFM 2.5 (Google) — zero-shot pretrained ----------------------------

_TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-transformers"
_TIMESFM_CONTEXT = 256
_TIMESFM_HORIZON = 128


def _get_timesfm_model():
    """Load TimesFM 2.5 for single-series forecasting."""
    model = TimesFm2_5ModelForPrediction.from_pretrained(_TIMESFM_MODEL_ID)
    model.eval()
    return model


def forecast_timesfm(series, horizon=FORECAST_HORIZON, context_len=_TIMESFM_CONTEXT):
    """Run TimesFM zero-shot forecast on a 1-D precip series.

    Parameters
    ----------
    series : np.ndarray  — daily precipitation values (1-D)
    horizon : int — number of future steps (must be <= model's horizon_length)
    context_len : int — how many past steps to use
    Returns
    -------
    np.ndarray of shape (horizon,) with predicted precipitation, clipped >= 0.
    """
    import torch
    x = torch.tensor(series[-context_len:].astype(np.float32), dtype=torch.float32)
    model = _get_timesfm_model()
    with torch.no_grad():
        out = model(past_values=[x])
    preds = out.mean_predictions.squeeze().numpy()[:horizon]
    return np.maximum(preds, 0)


# ---- TabICL (Inria/SODA) — in-context learning forecaster --------------------

def forecast_tabicl(train_df, horizon=FORECAST_HORIZON, context_len=512):
    """Run TabICL zero-shot forecast using in-context learning.

    TabICL treats the historical series as a few-shot context and
    predicts the next `horizon` steps. Requires `pip install tabicl[forecast]`.
    """
    if not _TABICL_AVAILABLE:
        raise RuntimeError(f"TabICL not available: {_TABICL_ERROR}")
    from tabicl import TabICLForecaster
    from tabicl.forecast import TimeSeriesDataFrame
    from tabicl.forecast._preprocessing import build_horizon

    prec = train_df["precip"].values.astype(float)
    idx = pd.date_range(end=train_df.index[-1], periods=len(prec), freq="D")
    df = pd.DataFrame({"item_id": [0] * len(prec), "timestamp": idx, "target": prec})
    ts_df = TimeSeriesDataFrame(df)

    forecaster = TabICLForecaster(max_context_length=context_len)
    future_df = build_horizon(ts_df, prediction_length=horizon)
    result = forecaster.predict(ts_df, future_df)
    preds = result["target"].values
    return np.maximum(preds, 0)


# -- evaluation ----------------------------------------------------------------

def evaluate(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4)}


def backtest_last_n_days(df, n=7, offset=0):
    """Walk-forward backtest: train on data up to *(len - n - offset)*, predict next *n* days.

    Parameters
    ----------
    df : DataFrame with 'precip' column
    n : int — size of backtest window in days (default 7)
    offset : int — how many days before the end to start the window.
        offset=0  → last n days of data
        offset=7  → n days ending 7 days before the end (skips trailing missing data)

    Returns a dict with predicted/actual arrays and metrics for each model.
    Returns None if not enough data.
    """
    trim = n + offset
    if len(df) < 365 + trim:
        return None
    feat = build_features(df)
    train_end = len(feat) - trim
    train_df = feat.iloc[:train_end]
    test_df = feat.iloc[train_end:train_end + n]
    if len(test_df) != n:
        return None
    # SARIMAX
    sarima_res = sarima_fit(train_df)
    sarima_pred, sarima_ci = sarima_test_predictions(sarima_res, test_df)
    sarima_metrics = evaluate(test_df["precip"].values, sarima_pred, "SARIMAX")
    # XGBoost
    xgb_model, feat_cols = train_xgboost(train_df)
    xgb_pred = forecast_xgboost(xgb_model, feat_cols, test_df)
    xgb_metrics = evaluate(test_df["precip"].values, xgb_pred, "XGBoost")
    # Actual values and dates
    actual = test_df["precip"].values
    dates = df.index[train_end:train_end + n]
    return {
        "dates": dates,
        "actual": actual,
        "sarima_pred": sarima_pred,
        "sarima_ci": sarima_ci,
        "sarima_metrics": sarima_metrics,
        "xgb_pred": xgb_pred,
        "xgb_metrics": xgb_metrics,
        "feat_cols": feat_cols,
    }


def save_pipeline(sarima_res, xgb_model, feat_cols, meta, model_path="data/processed/precipitation_model.joblib"):
    dump({"model_sarima": sarima_res, "model_xgboost": xgb_model,
          "feature_cols": feat_cols, "meta": meta}, model_path)
    print(f"  [Saved] {model_path}")
