# THETA Feedback & Memory

## Session: 2026-08-14 (Morning) - Final Fixes

### Changes Made
1. Fixed `run_forecast.py` --input mode transformer eval blocks (patchtst, timesfm, tabicl) to properly pad predictions to `n_test` length using `np.tile(fc, ...)[:n_test]` matching the `run()` mode pattern.
2. Fixed `utils/precipitation_models.py` `forecast_tabicl()` to work around TabICL library bug in `build_horizon` (ValueError: cannot insert item_id, already exists) by using manual `build_horizon` + `forecaster.predict()` instead of `predict_df()`.

### Current Status
- All 7 models (sarima, xgb, hgb, ets, rf, patchtst, timesfm, tabicl) now run without errors in both `--input` and default `--city` modes.
- Plots generated correctly with all model predictions aligned to test period length.

### TabICL Notes
- TabICL has a bug in its `build_horizon` function where it calls `train_tsdf.reset_index()` then tries to insert `item_id` column, but `reset_index()` already includes `item_id` in the columns.
- Workaround: Use `build_horizon()` + `forecaster.predict(context, future)` instead of `predict_df()`.
- TabICL evaluation is slow (~6s per call) due to HuggingFace model loading each time. Consider caching the forecaster.

### Model Performance (Jakarta, 10y)
- TimesFM: RMSE 15.28, MAE 10.79 (best)
- TabICL: RMSE 14.89, MAE 11.92
- PatchTST: RMSE 21.28, MAE 16.09
- Sarima/XGBoost/HistGB/ETS/RF also running successfully
