"""
Compare baseline (precip-only) vs enhanced (with exogenous) feature sets.
Fetches fresh data from NASA POWER API.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import requests

# ── Fetch data ─────────────────────────────────────────────────────────────────
params = {
    'parameters': 'PRECTOTCORR,T2M,RH2M,PS,WS10M,WD10M,T2M_MAX,T2M_MIN',
    'community': 'RE', 'format': 'JSON',
    'start': '20160821', 'end': '20260819',
    'latitude': '-6.2088', 'longitude': '106.8456',
}
resp = requests.get(
    'https://power.larc.nasa.gov/api/temporal/daily/point',
    params=params, timeout=60,
)
data = resp.json()
raw = data['properties']['parameter']
rows = {}
for code, values in raw.items():
    for date_str, value in values.items():
        if value is None:
            continue
        if date_str not in rows:
            rows[date_str] = {}
        rows[date_str][code] = float(value)

df = pd.DataFrame.from_dict(rows, orient='index')
df.index.name = 'Date'
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').sort_index()
df = df[df['PRECTOTCORR'] > 0].copy()
print(f'Data loaded: {len(df)} days ({df.index[0].date()} - {df.index[-1].date()})')

split = int(len(df) * 0.85)
df_train = df.iloc[:split].copy()
df_full = df.copy()

# ── Feature builders ────────────────────────────────────────────────────────────

def make_features(train_df, extra_cols=None):
    """Build feature matrix. If extra_cols given, append exogenous features aligned to train."""
    f = pd.DataFrame()
    f['doy_sin'] = np.sin(2 * np.pi * train_df.index.dayofyear / 365.25)
    f['doy_cos'] = np.cos(2 * np.pi * train_df.index.dayofyear / 365.25)

    # Precipitation lags & rolling stats (shifted so we don't leak future)
    for lag in [1, 2, 3, 7, 14, 30]:
        f[f'precip_lag{lag}'] = train_df['PRECTOTCORR'].shift(lag).values
    for w in [7, 14, 30]:
        f[f'precip_roll_mean_{w}'] = train_df['PRECOTCORR'].shift(1).rolling(w, min_periods=1).mean().values
        f[f'precip_roll_std_{w}'] = train_df['PRECOTCORR'].shift(1).rolling(w, min_periods=1).std().values

    return f.dropna()


def make_enhanced_features(train_df, full_df):
    """Same as make_features + exogenous features from full_df (aligned by index)."""
    f = make_features(train_df)
    # Exogenous: pull from full_df, same dates as f.index
    align_idx = f.index
    for col in ['RH2M', 'PS', 'WS10M', 'T2M']:
        vals = full_df.loc[align_idx, col].values
        f[col] = vals
        for lag in [1, 7]:
            shifted = full_df[col].shift(lag).reindex(align_idx).values
            f[f'{col}_lag{lag}'] = shifted

    # Derived interactions
    f['temp_humid'] = (full_df.loc[align_idx, 'T2M'] * full_df.loc[align_idx, 'RH2M'] / 100).values
    f['pressure_trend'] = np.diff(full_df['PS'].values)[len(full_df) - len(align_idx):]
    f['humidity_trend'] = np.diff(full_df['RH2M'].values)[len(full_df) - len(align_idx):]
    f['wind_dir_sin'] = np.sin(2 * np.pi * full_df.loc[align_idx, 'WD10M'].values / 360)
    f['wind_dir_cos'] = np.cos(2 * np.pi * full_df.loc[align_idx, 'WD10M'].values / 360)

    return f.dropna()


# ── Build features ────────────────────────────────────────────────────��────────
X_base = make_features(df_train).values
X_enh = make_enhanced_features(df_train, df_full).values
y_base = df['PRECOTCORR'].values[split:].iloc[:len(X_base)].values
y_enh = df['PRECOTCORR'].values[split:].iloc[:len(X_enh)].values
print(f'Baseline features: {X_base.shape[1]} | Enhanced features: {X_enh.shape[1]}')
print(f'Train samples: base={len(X_base)}, enhanced={len(X_enh)}')

# ── Train & evaluate ───────────────────────────────────────────────────────────
XGB_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)

print('\nTraining baseline XGBoost...')
m_base = xgb.XGBRegressor(**XGB_PARAMS)
m_base.fit(X_base, y_base)
p_base = np.maximum(m_base.predict(X_base), 0)
rmse_b = np.sqrt(mean_squared_error(y_base, p_base))
r2_b = r2_score(y_base, p_base)
print(f'  Baseline (precip lags only): RMSE={rmse_b:.2f}  R2={r2_b:.4f}')

print('Training enhanced XGBoost...')
m_enh = xgb.XGBRegressor(**XGB_PARAMS)
m_enh.fit(X_enh, y_enh)
p_enh = np.maximum(m_enh.predict(X_enh), 0)
rmse_e = np.sqrt(mean_squared_error(y_enh, p_enh))
r2_e = r2_score(y_enh, p_enh)
print(f'  Enhanced (+exogenous):       RMSE={rmse_e:.2f}  R2={r2_e:.4f}')

print(f'\nImprovement: RMSE delta={rmse_b - rmse_e:.2f} ({(1 - rmse_e / rmse_b) * 100:.1f}%)')

# ── Feature importance ─────────────────────────────────────────────────────────
feat_names = list(make_enhanced_features(df_train, df_full).columns)
imp = sorted(zip(feat_names, m_enh.feature_importances_), key=lambda x: -x[1])
print('\nTop 25 feature importances (Enhanced):')
for name, iv in imp[:25]:
    print(f'  {name:30s}: {iv:.4f}')

# ── SARIMAX comparison ─────────────────────────────────────────────────────────
from statsmodels.tsa.statespace.sarimax import SARIMAX

print('\nTraining SARIMAX (baseline)...')
# Align y to X indices
y_aligned = df['PRECOTCORR'].values[split:].values[:len(X_base)]
y_train_sarima = y_aligned[:int(len(y_aligned) * 0.85)]
y_test_sarima = y_aligned[int(len(y_aligned) * 0.85):]
train_sarima = df.iloc[:split + int(len(y_aligned) * 0.85)].copy()
test_sarima = df.iloc[split + int(len(y_aligned) * 0.85):].copy()

exog_train = train_sarima[['doy_sin', 'doy_cos']].values if 'doy_sin' in train_sarima.columns else np.column_stack([
    np.sin(2 * np.pi * train_sarima.index.dayofyear / 365.25),
    np.cos(2 * np.pi * train_sarima.index.dayofyear / 365.25),
])
exog_test = np.column_stack([
    np.sin(2 * np.pi * test_sarima.index.dayofyear / 365.25),
    np.cos(2 * np.pi * test_sarima.index.dayofyear / 365.25),
])

sarima = SARIMAX(train_sarima['PRECOTCORR'], order=(1, 0, 1), seasonal_order=(0, 1, 1, 7),
                 exogenous=exog_train, enforce_stationarity=False, enforce_invertibility=False)
sarima_res = sarima.fit(disp=False, maxiter=100, method='lbfgs')
sarima_pred = sarima_res.get_forecast(steps=len(test_sarima), exog=exog_test)
sarima_forecast = np.maximum(sarima_pred.predicted_mean.values, 0)
sarima_y = test_sarima['PRECOTCORR'].values
rmse_s = np.sqrt(mean_squared_error(sarima_y, sarima_forecast))
r2_s = r2_score(sarima_y, sarima_forecast)
print(f'  SARIMAX (baseline):          RMSE={rmse_s:.2f}  R2={r2_s:.4f}')

# Enhanced SARIMAX with exogenous
enh_exog_train = np.column_stack([exog_train, train_sarima[['RH2M', 'PS', 'WS10M', 'T2M']].values])
enh_exog_test = np.column_stack([exog_test, test_sarima[['RH2M', 'PS', 'WS10M', 'T2M']].values])
sarima_e = SARIMAX(train_sarima['PRECOTCORR'], order=(1, 0, 1), seasonal_order=(0, 1, 1, 7),
                   exogenous=enh_exog_train, enforce_stationarity=False, enforce_invertibility=False)
sarima_e_res = sarima_e.fit(disp=False, maxiter=100, method='lbfgs')
sarima_e_pred = sarima_e_res.get_forecast(steps=len(test_sarima), exog=enh_exog_test)
sarima_e_forecast = np.maximum(sarima_e_pred.predicted_mean.values, 0)
rmse_se = np.sqrt(mean_squared_error(sarima_y, sarima_e_forecast))
r2_se = r2_score(sarima_y, sarima_e_forecast)
print(f'  SARIMAX (+exogenous):        RMSE={rmse_se:.2f}  R2={r2_se:.4f}')

print(f'\n=== SUMMARY ===')
print(f'  SARIMAX baseline:             RMSE={rmse_s:.2f}  R2={r2_s:.4f}')
print(f'  SARIMAX + exogenous:          RMSE={rmse_se:.2f}  R2={r2_se:.4f}  ({(1-rmse_se/rmse_s)*100:.1f}%)')
print(f'  XGBoost baseline:             RMSE={rmse_b:.2f}  R2={r2_b:.4f}')
print(f'  XGBoost + exogenous:          RMSE={rmse_e:.2f}  R2={r2_e:.4f}  ({(1-rmse_e/rmse_b)*100:.1f}%)')
