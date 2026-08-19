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
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame({'Date': data['dates'], 'precip': data['precip_mm']})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').asfreq('D').interpolate(method='linear').dropna()
    city = data.get('city', json_path)
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
    df = df.copy()
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["year"] = df.index.year
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
    endog = train_df["precip"]
    exog = train_df[["doy_sin", "doy_cos"]].values
    model = SARIMAX(endog, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7),
                    exogenous=exog, enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=100, method="lbfgs")
    return results


def sarima_test_predictions(results, test_df):
    test_exog = test_df[["doy_sin", "doy_cos"]].values
    pred = results.get_forecast(steps=len(test_df), exog=test_exog)
    preds = pred.predicted_mean.values
    preds = np.maximum(preds, 0)
    return preds, pred.conf_int().values


def sarima_forward_forecast(results, feat_df, horizon=FORECAST_HORIZON):
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


# -- evaluation ----------------------------------------------------------------

def evaluate(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4)}


def save_pipeline(sarima_res, xgb_model, feat_cols, meta, model_path="data/processed/precipitation_model.joblib"):
    dump({"model_sarima": sarima_res, "model_xgboost": xgb_model,
          "feature_cols": feat_cols, "meta": meta}, model_path)
    print(f"  [Saved] {model_path}")
