"""
Training pipeline (Step 1+2+3 of the brief).

Fetches (features, targets) from the feature store, trains several models
per forecast horizon (24h/48h/72h), evaluates with RMSE/MAE/R2, keeps the
best model per horizon, and stores everything in the model registry
(models/ directory, or Hopsworks Model Registry if configured).

Run:
    python src/train_pipeline.py
"""
import datetime as dt
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import load_feature_store, to_hourly_grid, describe_gaps

# Raw input features from the feature store
RAW_POLLUTANT_COLS = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
RAW_WEATHER_COLS = ["temp", "humidity", "pressure", "wind_speed", "wind_deg", "clouds"]
RAW_TIME_COLS = ["hour", "day", "day_of_week", "month", "is_weekend"]

# Central 80% interval. Widened/narrowed per prediction for ensembles; see
# `predict_with_interval`.
INTERVAL_Z = 1.2816

MODEL_CANDIDATES = {
    "ridge": lambda: Ridge(alpha=10.0),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        max_features=0.6, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.04,
        subsample=0.8, min_samples_leaf=5, random_state=42
    ),
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rich feature set from the raw feature-store columns.

    1. Cyclical encoding for hour/day_of_week/month (sin+cos)
    2. Multiple lag horizons (1h, 2h, 3h, 6h, 12h, 24h, 48h)
    3. Rolling statistics (mean, std, min, max) at multiple windows
    4. Rate-of-change features at multiple scales
    5. Interaction features (wind x pollutant dispersal, temp x humidity)
    6. Outlier capping on AQI to reduce spike influence

    All lag/rolling operations are positional, so the caller must pass a frame
    on a continuous hourly grid (`utils.to_hourly_grid`). Otherwise `shift(24)`
    means "24 rows ago", which stops being "24 hours ago" at the first gap.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- Cap extreme AQI outliers (>99th percentile) to reduce noise ----
    cap = df["aqi"].quantile(0.99)
    df["aqi_capped"] = df["aqi"].clip(upper=cap)

    # ---- Cyclical time encoding ----
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ---- Lag features (on capped AQI for robustness) ----
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f"aqi_lag_{lag}h"] = df["aqi_capped"].shift(lag)

    # ---- Rolling statistics ----
    for window in [3, 6, 12, 24]:
        rolling = df["aqi_capped"].rolling(window, min_periods=1)
        df[f"aqi_rmean_{window}h"] = rolling.mean()
        df[f"aqi_rstd_{window}h"] = rolling.std().fillna(0)
        df[f"aqi_rmin_{window}h"] = rolling.min()
        df[f"aqi_rmax_{window}h"] = rolling.max()

    # ---- Rate of change at different scales ----
    df["aqi_diff_1h"] = df["aqi_capped"].diff().fillna(0)
    df["aqi_diff_3h"] = df["aqi_capped"].diff(3).fillna(0)
    df["aqi_diff_6h"] = df["aqi_capped"].diff(6).fillna(0)
    df["aqi_diff_24h"] = df["aqi_capped"].diff(24).fillna(0)

    # ---- PM2.5 lags (since it's the main AQI driver) ----
    for lag in [1, 3, 6, 24]:
        df[f"pm25_lag_{lag}h"] = df["pm2_5"].shift(lag)

    # ---- Weather lags (previous conditions influence future AQI) ----
    for lag in [6, 24]:
        df[f"wind_speed_lag_{lag}h"] = df["wind_speed"].shift(lag)
        df[f"humidity_lag_{lag}h"] = df["humidity"].shift(lag)

    # ---- Interaction features ----
    # Wind disperses pollutants — high wind + high PM2.5 = likely to decrease
    df["wind_pm25_interaction"] = df["wind_speed"] * df["pm2_5"]
    # Temperature inversions trap pollution
    df["temp_humidity"] = df["temp"] * df["humidity"]
    # Pressure changes correlate with weather fronts
    df["pressure_diff_6h"] = df["pressure"].diff(6).fillna(0)

    # ---- AQI trend indicator (is it getting better or worse?) ----
    df["aqi_trend_6h"] = (df["aqi_capped"] - df["aqi_rmean_6h"]).fillna(0)
    df["aqi_trend_24h"] = (df["aqi_capped"] - df["aqi_rmean_24h"]).fillna(0)

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return the list of feature columns (everything we engineered minus
    raw identifiers and targets)."""
    exclude = {"timestamp", "aqi", "aqi_raw", "aqi_capped",
               "hour", "day", "day_of_week", "month"}  # replaced by cyclical
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]


def make_targets(df: pd.DataFrame, horizon_hours: int) -> pd.Series:
    """Target = capped AQI `horizon_hours` ahead.

    A negative shift pulls a future value backwards onto the current row, so
    row t carries the AQI observed at t+horizon. That is the correct direction:
    a positive shift would train the model to predict the past. The final
    `horizon` rows get NaN targets and are dropped by the caller.

    This is positional, which is safe only because the caller reindexes onto a
    continuous hourly grid first.
    """
    return df["aqi_capped"].shift(-horizon_hours)


def make_delta_targets(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Models learn the CHANGE from the current AQI, not the absolute level.

    With one year of history, the first expanding-window CV fold trains on
    late-summer air (target max ~85) and is scored on winter smog (max ~201).
    Tree ensembles cannot predict outside the range they were trained on, so
    predicting the level directly produces a negative R2 on that fold no matter
    how good the features are. The change from the current reading is roughly
    stationary across seasons, so the same models transfer.
    """
    return target - current


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _ensemble_spread(model, X_scaled: np.ndarray):
    """Std-dev across independently-fitted ensemble members.

    Only meaningful for bagged ensembles such as RandomForest, where each tree
    is a standalone estimate of the target. Boosted trees are additive
    corrections to one another, so the spread across their estimators is not an
    uncertainty measure and is deliberately not used here.
    """
    if not isinstance(model, RandomForestRegressor):
        return None
    member_preds = np.stack([est.predict(X_scaled) for est in model.estimators_])
    return member_preds.std(axis=0)


def predict_with_interval(bundle: dict, X_scaled: np.ndarray, current_aqi=None):
    """Point forecast plus a rough central interval.

    Width is anchored to `residual_std` — the spread of the model's own
    out-of-sample cross-validation errors, which captures total error rather
    than just model variance. For a RandomForest the width is then modulated by
    how much the individual trees disagree about this particular input, so
    unusual conditions produce visibly wider bands.

    This is a heuristic, not a calibrated prediction interval. It is labelled
    as such in the dashboard.
    """
    model = bundle["model"]
    pred = model.predict(X_scaled)

    if bundle.get("target_mode", "delta") == "delta":
        if current_aqi is None:
            raise ValueError("This model predicts a change in AQI, so the "
                             "current AQI is required to reconstruct a level.")
        pred = np.asarray(current_aqi, dtype=float) + pred
    pred = np.clip(pred, 0, 500)

    residual_std = bundle.get("residual_std")
    if not residual_std:
        return pred, None, None

    sigma = np.full(len(pred), float(residual_std))
    spread = _ensemble_spread(model, X_scaled)
    reference = bundle.get("mean_spread")
    if spread is not None and reference:
        # Relative disagreement, clamped so a single odd input cannot produce
        # an absurdly wide or misleadingly narrow band.
        sigma = sigma * np.clip(spread / reference, 0.5, 2.0)

    z = bundle.get("interval_z", INTERVAL_Z)
    lower = np.clip(pred - z * sigma, 0, 500)
    upper = np.clip(pred + z * sigma, 0, 500)
    return pred, lower, upper


def evaluate_lstm(df: pd.DataFrame, horizon: int, n_splits: int):
    """Train and cross-validate the Keras LSTM candidate for one horizon.

    Uses its own sequence dataset rather than the flattened feature matrix, so
    the evaluated rows are a near-identical but not byte-identical subset of
    the ones the scikit-learn candidates see. Both require roughly 48 hours of
    prior history, so the two row sets differ only at the very start of the
    series; the metrics note in the registry records this.
    """
    import lstm_model as lm

    if not lm.is_available():
        return None

    seq_cols = lm.available_feature_cols(df)
    if len(seq_cols) < 8:
        print(f"  [horizon {horizon}h] lstm: too few raw feature columns "
              f"({len(seq_cols)}) -- skipping.")
        return None

    values, current, target, ends = lm.build_sequence_dataset(df, horizon, seq_cols)
    if len(ends) < 200:
        print(f"  [horizon {horizon}h] lstm: only {len(ends)} complete "
              f"{lm.SEQUENCE_LENGTH}h windows -- skipping.")
        return None

    y_abs = target[ends]
    now = current[ends]
    y_delta = y_abs - now

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    all_preds, all_true, per_fold = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(ends)):
        # Fit the scaler only on rows the training windows can see, then apply
        # it to the whole series before windowing.
        last_train_row = int(ends[train_idx[-1]])
        scaler = StandardScaler().fit(values[:last_train_row + 1])
        scaled = scaler.transform(values).astype("float32")

        X_train = lm.make_windows(scaled, ends[train_idx])
        X_test = lm.make_windows(scaled, ends[test_idx])

        model = lm.LSTMForecaster(n_features=len(seq_cols))
        model.fit(X_train, y_delta[train_idx])
        preds = np.clip(now[test_idx] + model.predict(X_test), 0, 500)

        all_preds.extend(preds)
        all_true.extend(y_abs[test_idx])
        per_fold.append({
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_target_mean": float(y_abs[train_idx].mean()),
            "test_target_mean": float(y_abs[test_idx].mean()),
            **evaluate(y_abs[test_idx], preds),
        })
        print(f"    lstm fold {fold}: RMSE={per_fold[-1]['rmse']:.2f} "
              f"R2={per_fold[-1]['r2']:.3f}", flush=True)

    all_preds, all_true = np.array(all_preds), np.array(all_true)
    metrics = evaluate(all_true, all_preds)

    # Refit on everything for the deployed artifact, matching the other candidates.
    final_scaler = StandardScaler().fit(values[~np.isnan(values).any(axis=1)])
    scaled_all = final_scaler.transform(values).astype("float32")
    final_model = lm.LSTMForecaster(n_features=len(seq_cols))
    final_model.fit(lm.make_windows(scaled_all, ends), y_delta)

    return {
        "metrics": metrics,
        "fold_metrics": per_fold,
        "residual_std": float(np.std(all_true - all_preds)),
        "mean_spread": None,
        "model": final_model,
        "scaler": final_scaler,
        "feature_cols": seq_cols,
        "seq_len": lm.SEQUENCE_LENGTH,
        "is_sequence": True,
    }


def train_for_horizon(df: pd.DataFrame, horizon: int, feature_cols: list):
    data = df.copy()
    data["target"] = make_targets(data, horizon)
    data = data.dropna(subset=feature_cols + ["target", "aqi_capped"])

    if len(data) < 100:
        print(f"  [horizon {horizon}h] not enough rows ({len(data)}) -- skipping. "
              f"Run backfill.py with a larger --days first.")
        return None

    X = data[feature_cols].values
    y = data["target"].values
    # AQI at prediction time, used for a true persistence baseline.
    current_aqi = data["aqi_capped"].values

    # Gap = forecast horizon to avoid train/test target overlap
    n_splits = min(5, max(2, len(data) // 200))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    splits = list(tscv.split(X))

    # Models learn the change from the current reading; see make_delta_targets.
    y_delta = make_delta_targets(y, current_aqi)

    results = {}
    fold_results = {}
    fitted_models = {}
    residual_stats = {}

    for name, factory in MODEL_CANDIDATES.items():
        all_preds, all_true, all_spread = [], [], []
        per_fold = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            # Scaling is fitted per fold. Fitting one scaler on the full
            # dataset up front would leak each test fold's mean and variance
            # into the model that is scored on it.
            fold_scaler = StandardScaler().fit(X[train_idx])
            X_train = fold_scaler.transform(X[train_idx])
            X_test = fold_scaler.transform(X[test_idx])

            model = factory()
            model.fit(X_train, y_delta[train_idx])
            # Reconstruct an absolute AQI so every model, the baseline, and the
            # dashboard are all scored on the same scale.
            preds = np.clip(current_aqi[test_idx] + model.predict(X_test), 0, 500)

            all_preds.extend(preds)
            all_true.extend(y[test_idx])
            per_fold.append({
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_target_mean": float(y[train_idx].mean()),
                "test_target_mean": float(y[test_idx].mean()),
                **evaluate(y[test_idx], preds),
            })

            spread = _ensemble_spread(model, X_test)
            if spread is not None:
                all_spread.extend(spread)

        # Metrics over ALL out-of-sample predictions (more reliable than a
        # per-fold average).
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        metrics = evaluate(all_true, all_preds)
        results[name] = metrics
        fold_results[name] = per_fold

        residual_stats[name] = {
            "residual_std": float(np.std(all_true - all_preds)),
            "mean_spread": float(np.mean(all_spread)) if all_spread else None,
        }

        # Refit on 100% of the data for the artifact we deploy. Note that the
        # metrics above describe the cross-validated models, NOT this one — the
        # deployed model has seen every test fold. That is standard practice
        # (more data, better model) but it does mean the saved metrics are an
        # estimate of generalisation, not a measurement of this artifact.
        final_scaler = StandardScaler().fit(X)
        final_model = factory()
        final_model.fit(final_scaler.transform(X), y_delta)
        fitted_models[name] = (final_model, final_scaler)

        worst = min(per_fold, key=lambda f: f["r2"])
        print(f"  [horizon {horizon}h] {name}: RMSE={metrics['rmse']:.2f} "
              f"MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f} "
              f"(worst fold {worst['fold']}: R2={worst['r2']:.3f})")

    # Deep-learning candidate. Trained separately because it consumes 48-hour
    # sequences rather than one flattened row per prediction.
    lstm_result = None
    try:
        import lstm_model as lm
        if lm.is_available():
            print(f"  [horizon {horizon}h] lstm: training on "
                  f"{lm.SEQUENCE_LENGTH}h sequences...", flush=True)
            lstm_result = evaluate_lstm(df, horizon, n_splits)
        else:
            print(f"  [horizon {horizon}h] tensorflow not installed -- "
                  f"skipping the LSTM candidate.")
    except Exception as exc:
        print(f"  [horizon {horizon}h] lstm failed "
              f"({type(exc).__name__}: {exc}) -- continuing without it.")

    if lstm_result:
        results["lstm"] = lstm_result["metrics"]
        fold_results["lstm"] = lstm_result["fold_metrics"]
        residual_stats["lstm"] = {
            "residual_std": lstm_result["residual_std"],
            "mean_spread": None,
        }
        worst = min(lstm_result["fold_metrics"], key=lambda f: f["r2"])
        print(f"  [horizon {horizon}h] lstm: "
              f"RMSE={lstm_result['metrics']['rmse']:.2f} "
              f"MAE={lstm_result['metrics']['mae']:.2f} "
              f"R2={lstm_result['metrics']['r2']:.3f} "
              f"(worst fold {worst['fold']}: R2={worst['r2']:.3f})")

    # Persistence baseline: "the AQI in `horizon` hours will be what it is
    # now". Evaluated on exactly the same out-of-sample rows as the models.
    naive_true, naive_pred = [], []
    for _, test_idx in splits:
        naive_true.extend(y[test_idx])
        naive_pred.extend(current_aqi[test_idx])
    naive_true, naive_pred = np.array(naive_true), np.array(naive_pred)
    naive_metrics = evaluate(naive_true, naive_pred)
    print(f"  [horizon {horizon}h] persistence_baseline: RMSE={naive_metrics['rmse']:.2f} "
          f"MAE={naive_metrics['mae']:.2f} R2={naive_metrics['r2']:.3f}")

    best_name = min(results, key=lambda n: results[n]["rmse"])
    if best_name == "lstm":
        best_model = lstm_result["model"]
        best_scaler = lstm_result["scaler"]
        best_feature_cols = lstm_result["feature_cols"]
        is_sequence, seq_len = True, lstm_result["seq_len"]
    else:
        best_model, best_scaler = fitted_models[best_name]
        best_feature_cols = feature_cols
        is_sequence, seq_len = False, None
    beats_baseline = results[best_name]["rmse"] < naive_metrics["rmse"]
    verdict = "beats" if beats_baseline else "DOES NOT BEAT"
    print(f"  [horizon {horizon}h] ** BEST -> {best_name} "
          f"(RMSE={results[best_name]['rmse']:.2f}, R2={results[best_name]['r2']:.3f}), "
          f"{verdict} persistence"
          + ("" if beats_baseline else " -- dashboard will serve the baseline"))

    return {
        "horizon": horizon,
        "best_model_name": best_name,
        "model": best_model,
        "scaler": best_scaler,
        "metrics": results,
        "fold_metrics": fold_results,
        "baseline_metrics": naive_metrics,
        "baseline_residual_std": float(np.std(naive_true - naive_pred)),
        "beats_baseline": bool(beats_baseline),
        "feature_cols": best_feature_cols,
        "is_sequence": is_sequence,
        "seq_len": seq_len,
        "n_train_rows": int(len(data)),
        "n_splits": n_splits,
        **residual_stats[best_name],
        "interval_z": INTERVAL_Z,
    }


def save_to_registry(result: dict):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    horizon = result["horizon"]
    trained_at = dt.datetime.now(dt.timezone.utc).isoformat()

    keras_path = os.path.join(config.MODELS_DIR, f"model_{horizon}h.keras")
    if result["is_sequence"]:
        # Keras models do not pickle, so the network is saved alongside the
        # joblib bundle and referenced by path.
        result["model"].save(keras_path)
        serialisable_model = None
    else:
        serialisable_model = result["model"]
        if os.path.exists(keras_path):
            # A previous run deployed the LSTM here; remove the orphan so
            # loading cannot pick up a stale network.
            os.remove(keras_path)

    joblib.dump(
        {"model": serialisable_model, "scaler": result["scaler"],
         "feature_cols": result["feature_cols"],
         "model_name": result["best_model_name"],
         "is_sequence": result["is_sequence"],
         "seq_len": result["seq_len"],
         "keras_path": keras_path if result["is_sequence"] else None,
         "target_mode": "delta",
         "residual_std": result["residual_std"],
         "baseline_residual_std": result["baseline_residual_std"],
         "beats_baseline": result["beats_baseline"],
         "mean_spread": result["mean_spread"],
         "interval_z": result["interval_z"],
         "trained_at": trained_at},
        os.path.join(config.MODELS_DIR, f"model_{horizon}h.joblib"),
        # A 200-tree forest serialises to ~10 MB uncompressed, and the daily
        # workflow commits this file back to the repo. Compression cuts it by
        # roughly an order of magnitude for a negligible load-time cost.
        compress=3,
    )
    summary = {
        "horizon_hours": horizon,
        "deployed_model": result["best_model_name"],
        "model_family": ("deep learning (Keras LSTM)" if result["is_sequence"]
                         else "scikit-learn"),
        "target_mode": "delta (change from current AQI)",
        "selected_metrics": result["metrics"][result["best_model_name"]],
        "persistence_baseline": result["baseline_metrics"],
        "beats_baseline": result["beats_baseline"],
        "serving": (result["best_model_name"] if result["beats_baseline"]
                    else "persistence (baseline outperformed the trained model)"),
        "all_candidates": result["metrics"],
        "fold_metrics": result["fold_metrics"][result["best_model_name"]],
        "residual_std": result["residual_std"],
        "n_train_rows": result["n_train_rows"],
        "n_cv_splits": result["n_splits"],
        "n_features": len(result["feature_cols"]),
        "trained_at": trained_at,
        "note": ("Metrics are out-of-sample from TimeSeriesSplit cross-validation. "
                 "The deployed artifact is refit on 100% of the data, so these "
                 "figures estimate its generalisation rather than measure it. "
                 "Per-fold metrics are included because the earliest expanding-"
                 "window fold trains on a single season and is scored on another, "
                 "which dominates the pooled figures. The LSTM candidate is scored "
                 "on 48-hour sequence windows, a near-identical but not byte-"
                 "identical subset of the rows the scikit-learn candidates use."),
    }
    with open(os.path.join(config.MODELS_DIR, f"metrics_{horizon}h.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _sequence_attributions(result: dict, explain_X, background_X):
    """Per-(timestep, feature) attributions for the LSTM.

    Tries SHAP's GradientExplainer first. Keras 3 support in SHAP is patchy, so
    it falls back to permutation importance — shuffling one feature across all
    timesteps and measuring the RMSE increase — which is model-agnostic and
    always available. Returns (attributions, method_label).
    """
    try:
        import shap
        explainer = shap.GradientExplainer(result["model"].model, background_X)
        values = explainer.shap_values(explain_X)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        if values.ndim == 4:  # (samples, timesteps, features, outputs)
            values = values[..., 0]
        return np.abs(values).mean(axis=0), "SHAP (GradientExplainer)"
    except Exception as exc:
        print(f"  GradientExplainer unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to permutation importance.", flush=True)

    model = result["model"]
    baseline = model.predict(explain_X)
    n_timesteps, n_features = explain_X.shape[1], explain_X.shape[2]
    importance = np.zeros((n_timesteps, n_features))
    rng = np.random.default_rng(42)

    for f in range(n_features):
        permuted = explain_X.copy()
        permuted[:, :, f] = permuted[rng.permutation(len(permuted)), :, f]
        delta = np.sqrt(np.mean((model.predict(permuted) - baseline) ** 2))
        # Permutation is applied to the whole feature series at once, so the
        # score is spread evenly across timesteps rather than resolved per step.
        importance[:, f] = delta / n_timesteps
    return importance, "Permutation importance"


def _explain_sequence_model(result: dict, df: pd.DataFrame, plt):
    """Explainability plot for the LSTM.

    Two panels: which features matter, and how far back in the 48-hour window
    the network actually looks.
    """
    import lstm_model as lm

    horizon = result["horizon"]
    feature_cols = result["feature_cols"]
    seq_len = result["seq_len"]

    values, _, _, ends = lm.build_sequence_dataset(df, horizon, feature_cols, seq_len)
    if len(ends) < 60:
        print("  not enough complete sequences for an explainability plot.", flush=True)
        return

    scaled = result["scaler"].transform(values).astype("float32")
    explain_X = lm.make_windows(scaled, ends[-100:])
    background_X = lm.make_windows(scaled, ends[-400:-100][::4])

    attributions, method = _sequence_attributions(result, explain_X, background_X)

    per_feature = attributions.mean(axis=0)
    per_timestep = attributions.mean(axis=1)
    order = np.argsort(per_feature)[::-1][:20]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.6, 1]})

    ax1.barh([feature_cols[i] for i in order][::-1],
             per_feature[order][::-1], color="#22d3ee")
    ax1.set_xlabel(f"Mean |attribution| ({method})")
    ax1.set_title(f"Feature importance — {horizon}h LSTM")
    ax1.grid(axis="x", alpha=0.25)

    hours_back = np.arange(seq_len - 1, -1, -1)
    ax2.plot(hours_back, per_timestep, color="#a78bfa", linewidth=2)
    ax2.fill_between(hours_back, per_timestep, color="#a78bfa", alpha=0.25)
    ax2.invert_xaxis()
    ax2.set_xlabel("Hours before prediction time")
    ax2.set_ylabel("Mean |attribution|")
    ax2.set_title("How far back the network looks")
    ax2.grid(alpha=0.25)

    out_path = os.path.join(config.MODELS_DIR, f"shap_summary_{horizon}h.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {method} summary saved -> {out_path}", flush=True)


def run_shap_explanation(result: dict, df: pd.DataFrame):
    """Generates a SHAP summary plot for the best model at each horizon.

    The sample is the most recent 100 complete rows. Because the deployed model
    was refit on all data, these rows are in-sample for it: the plot shows
    which features the model relies on, not how well it generalises. It answers
    "what is driving the current forecast", which is what the dashboard claims.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  shap/matplotlib not installed -- skipping explainability plot "
              "(pip install shap matplotlib).", flush=True)
        return

    if result.get("is_sequence"):
        return _explain_sequence_model(result, df, plt)

    feature_cols = result["feature_cols"]
    X = df[feature_cols].dropna().tail(100)
    if X.empty:
        print("  no complete rows available for SHAP -- skipping.", flush=True)
        return

    try:
        X_scaled = result["scaler"].transform(X.values)
        model = result["model"]
        if hasattr(model, "estimators_"):  # tree ensembles
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled, check_additivity=False)
        else:
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False,
                          max_display=20)
        out_path = os.path.join(config.MODELS_DIR, f"shap_summary_{result['horizon']}h.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  SHAP summary saved -> {out_path}", flush=True)
    except Exception as exc:
        # Explainability is a nice-to-have; never fail a training run over it.
        print(f"  SHAP plot failed ({type(exc).__name__}: {exc}) -- continuing.", flush=True)


def run():
    df = load_feature_store()
    if df.empty:
        raise SystemExit("Feature store is empty. Run backfill.py first.")

    gaps = describe_gaps(df)
    if gaps:
        missing = sum(g[2] for g in gaps)
        print(f"Feature store has {len(gaps)} gap(s) totalling {missing} missing hour(s); "
              f"reindexing onto a continuous hourly grid so lag features stay "
              f"anchored to real time.")
    df = to_hourly_grid(df)

    print(f"Loaded {len(df)} feature rows.")
    print("Engineering features...")
    df = engineer_features(df)
    feature_cols = get_feature_cols(df)
    print(f"  {len(feature_cols)} features total.")

    trained = 0
    for horizon in config.HORIZONS:
        print(f"\nTraining models for +{horizon}h forecast...")
        result = train_for_horizon(df, horizon, feature_cols)
        if result is None:
            continue
        save_to_registry(result)
        run_shap_explanation(result, df)
        trained += 1

    print("\n" + "=" * 60)
    if trained:
        print(f"Training complete ({trained}/{len(config.HORIZONS)} horizons). "
              f"Models saved to: {config.MODELS_DIR}")
    else:
        print("No models trained -- the feature store does not have enough history yet.")
    print("=" * 60)


if __name__ == "__main__":
    run()
