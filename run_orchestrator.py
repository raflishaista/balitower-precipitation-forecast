"""LLM-powered forecasting orchestrator.

Parses natural-language requests via Nemotron-35, extracts structured intent,
and dispatches to the appropriate data source + forecast pipeline.

Usage:
    python run_orchestrator.py "predict rainfall in Jakarta for the next 7 days"
    python run_orchestrator.py --input "forecast AAPL stock price for 10 days"
    python run_orchestrator.py --input "how much rain will London get next week"
"""
import os
import sys
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── LLM config ──────────────────────────────────────────────────────────────
LLM_KEY = os.environ.get("LLM_KEY", "KEY_REMOVED")
LLM_URL = os.environ.get("LLM_URL", "http://10.7.1.21/")
LLM_MODEL = "nemotron-35"
LLM_MAX_TOKENS = 1024

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
INPUTS_JSON = os.path.join(PROJECT_ROOT, "inputs_json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "orchestrator")

# ── Prompt template ─────────────────────────────────────────────────────────
ORCHESTRATOR_PROMPT = """You are a forecasting orchestration agent. Parse the user request below and output ONLY valid JSON — no explanation, no markdown, just the raw JSON object.

Required fields:
- domain: one of ["weather", "stocks"]
- target: short phrase describing what is being predicted (e.g. "rainfall", "stock price", "temperature")
- location_or_symbol: city name or stock/crypto ticker symbol
- horizon: integer number of future periods to predict (1-30)
- data_source_hint: one of ["NASA_POWER", "alpha_vantage", "csv_file", "local_data"]
- note: any extra context the user provided (optional, can be empty string)

Examples:
Input: "predict rainfall in Jakarta for the next 7 days"
Output: {{"domain": "weather", "target": "rainfall", "location_or_symbol": "Jakarta", "horizon": 7, "data_source_hint": "NASA_POWER", "note": ""}}

Input: "forecast AAPL stock price for the next 5 days"
Output: {{"domain": "stocks", "target": "stock price", "location_or_symbol": "AAPL", "horizon": 5, "data_source_hint": "alpha_vantage", "note": ""}}

Input: "how much rain will London get next week"
Output: {{"domain": "weather", "target": "rainfall", "location_or_symbol": "London", "horizon": 7, "data_source_hint": "NASA_POWER", "note": "next week"}}

Now parse this request: "{user_input}"
"""


def call_llm(user_input: str) -> dict:
    """Call Nemotron-35 to extract structured forecasting parameters."""
    import requests

    prompt = ORCHESTRATOR_PROMPT.format(user_input=user_input)
    resp = requests.post(
        f"{LLM_URL.rstrip('/')}/v1/chat/completions",
        headers={"Authorization": f"Bearer {LLM_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": 0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"].get("content", "")
    if not content:
        raise RuntimeError(f"LLM returned empty content. finish_reason: {resp.json()['choices'][0]['finish_reason']}")
    return json.loads(content)


def parse_horizon(horizon):
    """Ensure horizon is a valid positive int."""
    h = int(horizon)
    return max(1, min(h, 30))  # clamp to 1-30


# ── Data loading ────────────────────────────────────────────────────────────
def _find_weather_data(city: str) -> tuple | None:
    """Find weather/precipitation Excel or JSON file for a city."""
    pattern = city.replace(" ", "_").lower()
    # Try raw Excel files
    import glob
    candidates = glob.glob(os.path.join(RAW_DIR, f"*{pattern}*weather*.xlsx")) + \
                 glob.glob(os.path.join(RAW_DIR, f"*{pattern}*precipitation*.xlsx"))
    if candidates:
        path = sorted(candidates, key=os.path.getmtime, reverse=True)[0]
        df = pd.read_excel(path)
        if "Precipitation" in df.columns:
            col = "Precipitation"
        elif "Total Precipitation (mm)" in df.columns:
            col = "Total Precipitation (mm)"
        else:
            col = [c for c in df.columns if "precip" in str(c).lower()]
            col = col[0] if col else None
        if col:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").resample("D").sum().interpolate(method="linear").dropna()
            return df[[col]].rename(columns={col: "precip"}), path
    # Try inputs_json
    json_candidates = glob.glob(os.path.join(INPUTS_JSON, f"*{city.replace(' ', '_')}*"))
    if json_candidates:
        return json_candidates[0], True  # signal it's a JSON path
    return None


def _find_stock_data(symbol: str) -> tuple | None:
    """Find stock data CSV/XLSX for a symbol."""
    import glob
    # First try exact symbol match in filename
    pat = symbol.upper()
    candidates = glob.glob(os.path.join(RAW_DIR, f"*{pat}*"))
    # Fall back to any stock file
    if not candidates:
        candidates = glob.glob(os.path.join(RAW_DIR, "stock_data.xlsx"))
    if not candidates:
        candidates = glob.glob(os.path.join(RAW_DIR, "*stock*.xlsx")) + \
                     glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if candidates:
        path = candidates[0]
        ext = os.path.splitext(path)[1].lower()
        if ext == ".xlsx":
            df = pd.read_excel(path)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            return None
        # Normalise: expect Date + Close columns
        date_cols = [c for c in df.columns if "date" in str(c).lower()]
        close_cols = [c for c in df.columns if "close" in str(c).lower() or "price" in str(c).lower()]
        if date_cols and close_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df = df.sort_values(date_cols[0]).reset_index(drop=True)
            df = df[[date_cols[0], close_cols[0]]].rename(columns={date_cols[0]: "Date", close_cols[0]: "close"})
            # Extract actual symbol from filename if possible
            found_symbol = os.path.splitext(os.path.basename(path))[0].split("_")[-1].upper()
            return df, path, found_symbol
    return None


# ── Forecast runners ────────────────────────────────────────────────────────
def run_weather_forecast(city: str, horizon: int):
    """Run precipitation forecast pipeline for a city."""
    from utils.precipitation_models import (
        load_precipitation_data, build_features, split_train_test, evaluate,
        sarima_fit, sarima_test_predictions, sarima_forward_forecast,
        train_xgboost, forecast_xgboost,
        train_histgb, forecast_histgb,
        ets_fit, ets_test_predictions, ets_forward_forecast,
        train_rf, forecast_rf,
        forecast_patchtst, forecast_timesfm, forecast_tabicl,
        _LSTM_AVAILABLE, _TRANSFORMER_AVAILABLE, _TABICL_AVAILABLE,
        backtest_last_n_days,
    )
    from utils.plotting import plot_forecast

    print(f"\n{'='*60}")
    print(f"  WEATHER FORECAST — {city.upper()}")
    print(f"  Horizon: {horizon} days")
    print(f"{'='*60}\n")

    # Load data (tries local files first, then NASA POWER API)
    try:
        df, src_path = load_precipitation_data(city, years=10)
        print(f"  Loaded {len(df)} days {'from '+src_path if src_path else 'via API'}")
    except Exception as e:
        print(f"  [WARN] Could not load data for {city}: {e}")
        print("  Available cities in data/raw/:")
        import glob
        for f in sorted(glob.glob(os.path.join(RAW_DIR, "*weather*.xlsx")) +
                        glob.glob(os.path.join(RAW_DIR, "*precipitation*.xlsx"))):
            print(f"    - {os.path.basename(f)}")
        return

    # Save the original datetime index before build_features drops it
    orig_dates = df.index.tolist()
    feat_df = build_features(df)
    train_df, test_df = split_train_test(feat_df)
    series = train_df["precip"].values.astype(np.float32)
    n_test = len(test_df)

    models = ["sarima", "xgb", "hgb", "ets", "rf"]
    if _TRANSFORMER_AVAILABLE:
        models.extend(["patchtst", "timesfm"])
    if _TABICL_AVAILABLE:
        models.append("tabicl")

    all_preds, all_metrics, all_states, forecast_preds = {}, {}, {}, {}

    for name in models:
        print(f"> {name.upper()} ...")
        try:
            if name == "sarima":
                res = sarima_fit(train_df)
                pred, ci = sarima_test_predictions(res, test_df)
                metrics = evaluate(test_df["precip"].values, pred, "SARIMAX")
                all_states["sarima"] = res
                all_preds["sarima"] = pred
                all_metrics["sarima"] = metrics
                forecast_preds["sarima"] = np.round(sarima_forward_forecast(res, feat_df, horizon=horizon), 2)

            elif name == "xgb":
                model, feat_cols = train_xgboost(train_df)
                pred = forecast_xgboost(model, feat_cols, test_df)
                metrics = evaluate(test_df["precip"].values, pred, "XGBoost")
                all_states["xgb"] = (model, feat_cols)
                all_preds["xgb"] = pred
                all_metrics["xgb"] = metrics

            elif name == "hgb":
                model, feat_cols = train_histgb(train_df)
                pred = forecast_histgb(model, feat_cols, test_df)
                metrics = evaluate(test_df["precip"].values, pred, "HistGB")
                all_states["hgb"] = (model, feat_cols)
                all_preds["hgb"] = pred
                all_metrics["hgb"] = metrics

            elif name == "ets":
                res = ets_fit(train_df)
                pred, ci = ets_test_predictions(res, test_df)
                metrics = evaluate(test_df["precip"].values, pred, "ETS")
                all_states["ets"] = res
                all_preds["ets"] = pred
                all_metrics["ets"] = metrics
                forecast_preds["ets"] = np.round(ets_forward_forecast(res, horizon=horizon), 2)

            elif name == "rf":
                model, feat_cols = train_rf(train_df)
                pred = forecast_rf(model, feat_cols, test_df)
                metrics = evaluate(test_df["precip"].values, pred, "RF")
                all_states["rf"] = (model, feat_cols)
                all_preds["rf"] = pred
                all_metrics["rf"] = metrics

            elif name == "patchtst" and _TRANSFORMER_AVAILABLE:
                fc = forecast_patchtst(series, horizon=horizon)
                # Quick eval on last 35 points
                eval_start = max(0, len(series) - 35)
                pe, ae = [], []
                for i in range(eval_start, len(series) - horizon):
                    p = forecast_patchtst(series[:i+1], horizon=horizon)
                    pe.extend(p.tolist())
                    ae.extend(series[i+1:i+1+horizon].tolist())
                pe, ae = np.array(pe), np.array(ae)
                np_ = pe.size // horizon
                m = evaluate(ae[:np_*horizon].reshape(np_,horizon).mean(1),
                             pe[:np_*horizon].reshape(np_,horizon).mean(1), "PatchTST")
                all_preds["patchtst"] = np.tile(fc, int(np.ceil(n_test/horizon)))[:n_test]
                all_metrics["patchtst"] = m
                forecast_preds["patchtst"] = np.round(fc, 2)

            elif name == "timesfm" and _TRANSFORMER_AVAILABLE:
                fc = forecast_timesfm(series, horizon=horizon)
                eval_start = max(0, len(series) - 35)
                pe, ae = [], []
                for i in range(eval_start, len(series) - horizon):
                    p = forecast_timesfm(series[:i+1], horizon=horizon)
                    pe.extend(p.tolist())
                    ae.extend(series[i+1:i+1+horizon].tolist())
                pe, ae = np.array(pe), np.array(ae)
                np_ = pe.size // horizon
                m = evaluate(ae[:np_*horizon].reshape(np_,horizon).mean(1),
                             pe[:np_*horizon].reshape(np_,horizon).mean(1), "TimesFM")
                all_preds["timesfm"] = np.tile(fc, int(np.ceil(n_test/horizon)))[:n_test]
                all_metrics["timesfm"] = m
                forecast_preds["timesfm"] = np.round(fc, 2)

            elif name == "tabicl" and _TABICL_AVAILABLE:
                fc = forecast_tabicl(train_df, horizon=horizon)
                eval_start = max(0, len(series) - 35)
                pe, ae = [], []
                for i in range(eval_start, len(series) - horizon):
                    p = forecast_tabicl(train_df.iloc[:i+1], horizon=horizon)
                    pe.extend(p.tolist())
                    ae.extend(series[i+1:i+1+horizon].tolist())
                pe, ae = np.array(pe), np.array(ae)
                np_ = pe.size // horizon
                m = evaluate(ae[:np_*horizon].reshape(np_,horizon).mean(1),
                             pe[:np_*horizon].reshape(np_,horizon).mean(1), "TabICL")
                all_preds["tabicl"] = np.tile(fc, int(np.ceil(n_test/horizon)))[:n_test]
                all_metrics["tabicl"] = m
                forecast_preds["tabicl"] = np.round(fc, 2)

            print(f"  {all_metrics.get(name, {})}")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    if not all_preds:
        print("  No models succeeded.")
        return

    # Build forecast dates from the original datetime index (preserved via orig_dates)
    last_idx = train_df.index[-1]
    fc_dates = pd.date_range(start=orig_dates[last_idx] + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc_dict = {}
    for name in forecast_preds:
        fc_dict[name] = np.round(forecast_preds[name][:horizon], 2)
    fc = pd.DataFrame(fc_dict)
    print(f"\n--- {horizon}-Day Forecast ---\n")
    print(fc.to_string(index=False))

    # Metrics summary
    print(f"\n  {'Model':<12} {'RMSE':>8} {'MAE':>8}")
    print("  " + "-"*32)
    for name, met in sorted(all_metrics.items(), key=lambda x: x[1]["RMSE"]):
        print(f"  {name:<12} {met['RMSE']:>8.2f} {met['MAE']:>8.2f}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = {"city": city, "forecast_dates": [d.strftime("%Y-%m-%d") for d in fc_dates],
           "models": {}}
    for name in all_preds:
        fc_arr = forecast_preds.get(name, all_preds[name][-horizon:])
        out["models"][name] = {"predictions": [round(float(v), 2) for v in fc_arr[:horizon]],
                               "metrics": all_metrics[name]}
    out_path = os.path.join(OUTPUT_DIR, f"forecast_{city.replace(' ', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  [Saved] {out_path}")


def run_stock_forecast(symbol: str, horizon: int):
    """Run stock price forecast using available local data + ML models."""
    from utils.precipitation_models import (
        build_features, split_train_test, evaluate,
        sarima_fit, sarima_test_predictions, sarima_forward_forecast,
        train_xgboost, forecast_xgboost,
        train_histgb, forecast_histgb,
        ets_fit, ets_test_predictions, ets_forward_forecast,
        train_rf, forecast_rf,
    )

    print(f"\n{'='*60}")
    print(f"  STOCK FORECAST — {symbol.upper()}")
    print(f"  Horizon: {horizon} periods")
    print(f"{'='*60}\n")

    data_path = _find_stock_data(symbol)
    if data_path is None:
        print(f"  [ERROR] No data found for '{symbol}'.")
        print(f"  Available in data/raw/:")
        import glob
        for f in sorted(glob.glob(os.path.join(RAW_DIR, "*stock*")) +
                        glob.glob(os.path.join(RAW_DIR, "*.csv"))):
            print(f"    - {os.path.basename(f)}")
        return

    df, src, found_symbol = data_path
    print(f"  Loaded {len(df)} rows from {src}")
    print(f"  Using symbol: {found_symbol}")

    if "close" not in df.columns:
        # Try to find a numeric column to forecast
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            df = df[[df.columns[0], num_cols[0]]].rename(columns={num_cols[0]: "close"})
        else:
            print("  [ERROR] No numeric column found for forecasting.")
            return

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    series = df["close"].values.astype(np.float32)
    print(f"  Series: {series[:5]} ... {series[-5:]}")

    # Build features (lags + rolling stats) for tree models
    df_feat = pd.DataFrame({"close": series})
    for lag in [1, 2, 3, 5, 7]:
        df_feat[f"lag_{lag}"] = df_feat["close"].shift(lag)
    df_feat["roll_mean_5"] = df_feat["close"].rolling(5).mean()
    df_feat["roll_std_5"] = df_feat["close"].rolling(5).std()
    df_feat = df_feat.dropna()

    n_train = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:n_train]
    test_df = df_feat.iloc[n_train:]
    n_test = len(test_df)

    models = ["sarima", "xgb", "hgb", "ets", "rf"]
    all_preds, all_metrics, all_states, forecast_preds = {}, {}, {}, {}

    for name in models:
        print(f"> {name.upper()} ...")
        try:
            if name == "sarima":
                series_clean = df["close"].values.astype(np.float64)
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                res = SARIMAX(series_clean, order=(1,1,1)).fit(disp=False)
                pred = res.forecast(min(horizon, n_test))
                actual = series_clean[-n_test:-n_test+len(pred)] if len(pred) <= len(series_clean)-n_test else None
                if actual is not None and len(actual) == len(pred):
                    metrics = evaluate(actual, pred, "SARIMAX")
                else:
                    metrics = {"RMSE": 0, "MAE": 0}
                all_states["sarima"] = res
                all_preds["sarima"] = pred
                all_metrics["sarima"] = metrics
                fc = np.round(res.forecast(steps=horizon).predicted_mean[:horizon], 2)
                forecast_preds["sarima"] = fc

            elif name == "xgb":
                from xgboost import XGBRegressor
                feat_cols = [c for c in train_df.columns if c != "close"]
                model = XGBRegressor(n_estimators=50, max_depth=3, random_state=42).fit(
                    train_df[feat_cols].values, train_df["close"].values)
                pred = model.predict(test_df[feat_cols].values)
                metrics = evaluate(test_df["close"].values, pred, "XGBoost")
                all_states["xgb"] = (model, feat_cols)
                all_preds["xgb"] = pred
                all_metrics["xgb"] = metrics
                # Forward forecast
                last_feat = df_feat[feat_cols].tail(1).values
                fc = np.round(np.maximum(model.predict(last_feat), 0), 2)
                forecast_preds["xgb"] = np.tile(fc, int(np.ceil(horizon)))[:horizon]

            elif name == "hgb":
                from histboost import HistGradientBoostingRegressor
                feat_cols = [c for c in train_df.columns if c != "close"]
                model = HistGradientBoostingRegressor(max_iter=50, random_state=42).fit(
                    train_df[feat_cols].values, train_df["close"].values)
                pred = model.predict(test_df[feat_cols].values)
                metrics = evaluate(test_df["close"].values, pred, "HistGB")
                all_states["hgb"] = (model, feat_cols)
                all_preds["hgb"] = pred
                all_metrics["hgb"] = metrics
                last_feat = df_feat[feat_cols].tail(1).values
                fc = np.round(np.maximum(model.predict(last_feat), 0), 2)
                forecast_preds["hgb"] = np.tile(fc, int(np.ceil(horizon)))[:horizon]

            elif name == "ets":
                from statsmodels.tsa.exponential_smoothing.exp_smoothing import ExponentialSmoothing
                res = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
                pred = res.forecast(min(horizon, n_test))
                actual = series[-n_test:-n_test+len(pred)]
                metrics = evaluate(actual, pred, "ETS")
                all_states["ets"] = res
                all_preds["ets"] = pred
                all_metrics["ets"] = metrics
                fc = np.round(np.maximum(res.forecast(horizon).values[:horizon], 0), 2)
                forecast_preds["ets"] = fc

            elif name == "rf":
                from sklearn.ensemble import RandomForestRegressor
                feat_cols = [c for c in train_df.columns if c != "close"]
                model = RandomForestRegressor(n_estimators=50, random_state=42).fit(
                    train_df[feat_cols].values, train_df["close"].values)
                pred = model.predict(test_df[feat_cols].values)
                metrics = evaluate(test_df["close"].values, pred, "RF")
                all_states["rf"] = (model, feat_cols)
                all_preds["rf"] = pred
                all_metrics["rf"] = metrics
                last_feat = df_feat[feat_cols].tail(1).values
                fc = np.round(np.maximum(model.predict(last_feat), 0), 2)
                forecast_preds["rf"] = np.tile(fc, int(np.ceil(horizon)))[:horizon]

            print(f"  {all_metrics.get(name, {})}")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    if not all_preds:
        print("  No models succeeded.")
        return

    # Print
    fc_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc_dict = {name: np.round(forecast_preds[name][:horizon], 2) for name in forecast_preds}
    fc = pd.DataFrame(fc_dict)
    print(f"\n--- {horizon}-Period Forecast ---\n")
    print(fc.to_string(index=False))

    print(f"\n  {'Model':<12} {'RMSE':>8} {'MAE':>8}")
    print("  " + "-"*32)
    for name, met in sorted(all_metrics.items(), key=lambda x: x[1]["RMSE"]):
        print(f"  {name:<12} {met['RMSE']:>8.2f} {met['MAE']:>8.2f}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = {"symbol": symbol, "forecast_dates": [d.strftime("%Y-%m-%d") for d in fc_dates],
           "models": {}}
    for name in all_preds:
        fc_arr = forecast_preds.get(name, all_preds[name][-horizon:])
        out["models"][name] = {"predictions": [round(float(v), 2) for v in fc_arr[:horizon]],
                               "metrics": all_metrics[name]}
    out_path = os.path.join(OUTPUT_DIR, f"forecast_{symbol}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  [Saved] {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered forecasting orchestrator — parse natural language, run forecasts")
    parser.add_argument("--input", "-i", required=True, help="Natural language forecast request")
    parser.add_argument("--dry-run", action="store_true", help="Only show parsed intent, don't run forecasts")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  THETA Forecasting Orchestrator")
    print(f"  LLM: {LLM_MODEL} @ {LLM_URL.rstrip('/')}")
    print("="*60)
    print(f"\n  User request: \"{args.input}\"")

    # Step 1: LLM parsing
    print("\n[1/3] Parsing intent via LLM...")
    try:
        intent = call_llm(args.input)
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}")
        sys.exit(1)

    domain     = intent.get("domain", "").lower()
    target     = intent.get("target", "")
    location   = intent.get("location_or_symbol", "")
    horizon    = parse_horizon(intent.get("horizon", 7))
    data_src   = intent.get("data_source_hint", "")
    note       = intent.get("note", "")

    print(f"  Parsed intent:")
    print(f"    domain     : {domain}")
    print(f"    target     : {target}")
    print(f"    location   : {location}")
    print(f"    horizon    : {horizon}")
    print(f"    data_source: {data_src}")
    if note:
        print(f"    note       : {note}")

    if args.dry_run:
        print("\n  [DRY RUN] — no forecasts executed.")
        return

    # Step 2: Dispatch
    print(f"\n[2/3] Loading data & running forecasts...")
    if domain == "weather":
        run_weather_forecast(location, horizon)
    elif domain == "stocks":
        run_stock_forecast(location, horizon)
    else:
        print(f"  [ERROR] Unknown domain: {domain}. Supported: weather, stocks")
        sys.exit(1)

    # Step 3: Done
    print(f"\n[3/3] Complete.")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
