# Pearls AQI Predictor — Technical Report

**City:** Karachi (24.8607, 67.0011)
**Forecast horizons:** +24h, +48h, +72h
**Data window:** 16 Aug 2025 – 16 Aug 2026 (8,760 hourly rows, one full year)
**Target scale:** US EPA AQI, 0–500

---

## 1. Problem framing

Predict the US EPA Air Quality Index for a city 24, 48 and 72 hours ahead,
using only data available at prediction time, on a fully serverless stack with
no always-on infrastructure.

The system is three scheduled jobs and two read-only interfaces:

```
OpenWeather (pollution) ─┐
                         ├─▶ feature_pipeline.py ──▶ feature store ──▶ train_pipeline.py ──▶ model registry
Open-Meteo (weather) ────┘        (hourly)           (parquet or          (daily)              (models/)
                                                      Hopsworks)                                   │
                                                                                                   ▼
                                                                              Streamlit dashboard + FastAPI
```

---

## 2. Data

### 2.1 Sources

| Source | Provides | Notes |
|---|---|---|
| OpenWeather Air Pollution | CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃ | Free tier, history back to ~Nov 2020 |
| Open-Meteo Archive | temp, humidity, pressure, wind speed/direction, cloud cover | Free, no key, ~5-day reanalysis lag |

Both are queried in UTC and joined on the hour. The join is an inner merge on a
tz-naive UTC `timestamp` column; hours where either side is missing or null are
dropped rather than imputed.

### 2.2 Computing the AQI

OpenWeather returns raw concentrations plus its own 1–5 index, neither of which
is the 0–500 EPA scale the project reports against. The conversion applies the
EPA's piecewise-linear breakpoint tables per pollutant and takes the
**maximum of the sub-indices**, which is the EPA's definition of the overall AQI.
Gaseous pollutants are converted from µg/m³ to ppb/ppm at 25 °C and 1 atm.

Which pollutant sets the AQI, across the full year:

| Dominant pollutant | Hours | Share |
|---|---|---|
| PM2.5 | 6,336 | 74.6% |
| PM10 | 1,467 | 17.3% |
| O₃ | 693 | 8.2% |

A PM2.5-only conversion would therefore understate the AQI for roughly a
quarter of all hours — which is what an earlier version of this code did.

**Known limitation.** The EPA defines these breakpoints against averaging
windows this data source does not provide (24-hour for particulates, 8-hour for
O₃ and CO). An instantaneous hourly reading is used as a proxy, which biases the
computed AQI upward during short spikes. Fixing this properly requires rolling
the concentrations over their correct windows before the AQI lookup and is the
most defensible next change to the data layer.

### 2.3 Data quality issues found and fixed

**Breakpoint gap producing false maximum readings.** The EPA tables jump from
12.0 to 12.1 µg/m³, 35.4 to 35.5, and so on, because the standard requires the
concentration to be *truncated* to the table's precision before lookup. Without
that step, a PM2.5 reading of 12.03 matches no bucket. The original code fell
through to a hardcoded `return 500.0`, so 32 hours of genuinely clean air were
recorded as "Hazardous, AQI 500":

| Timestamp | PM2.5 (µg/m³) | Recorded AQI | Correct AQI |
|---|---|---|---|
| 2025-08-18 08:00 | 12.03 | 500.0 | 50 (Good) |
| 2025-10-15 14:00 | 35.43 | 500.0 | 100 (Moderate) |
| 2025-12-15 07:00 | 150.43 | 500.0 | 200 (Unhealthy) |

These corrupted both the training targets and the alerting logic.
`src/repair_feature_store.py` recomputes the AQI column in place from the stored
concentrations, which were always correct.

**Genuine extreme values.** After the fix, three consecutive hours on 7 June
2026 still report AQI 500. These are real: PM10 reached 611–647 µg/m³, above the
top of the EPA table (604), during what the wind and cloud data suggest was a
dust event. They are correctly capped rather than discarded.

**Temporal gaps.** The store had 11 gaps of 24 hours each (264 missing hours)
from failed pipeline runs. Because every lag and rolling feature is computed
with a positional `shift()`, a missing hour silently turns `shift(24)` from
"24 hours ago" into "24 rows ago". The store is now reindexed onto a continuous
hourly grid: gaps of up to 3 hours are interpolated and flagged via
`is_imputed`, longer gaps are held as explicit nulls and dropped at training
time. This preserves temporal alignment without fabricating a full day of air
quality data.

---

## 3. Features

Two feature sets, because two model families need different things.

**Tabular models (Ridge, Random Forest, Gradient Boosting) — 64 features.**
History must be flattened into explicit columns:

- Cyclical time encodings (sin/cos for hour, day-of-week, month)
- AQI lags at 1, 2, 3, 6, 12, 24 and 48 hours
- Rolling mean / std / min / max over 3, 6, 12 and 24-hour windows
- Rate-of-change at 1, 3, 6 and 24 hours
- PM2.5 lags (1, 3, 6, 24h) and weather lags (6, 24h)
- Interactions: wind × PM2.5 (dispersal), temp × humidity (inversion),
  6-hour pressure change (fronts)
- AQI outliers capped at the 99th percentile

**Sequence model (LSTM) — 21 features × 48 timesteps.**
Raw per-hour signals only: the eight pollutants, six weather variables, and the
cyclical time encodings. Lag and rolling columns are deliberately excluded
because the 48-hour sequence already contains that information.

### 3.1 Target formulation

All models predict the **change** in AQI from the current reading, not the
absolute level:

```
target = aqi_capped[t + horizon] - aqi_capped[t]
```

This was not the original design, and the reason for the change is the single
most important modelling finding in this project — see §5.1.

---

## 4. Evaluation protocol

- `TimeSeriesSplit` with 5 expanding-window folds
- `gap=horizon` between train and test, so no training row's target overlaps
  the test period
- `StandardScaler` fitted **inside each fold** on training rows only
- Metrics pooled across all out-of-sample predictions rather than averaged
  per fold
- Every model compared against a **persistence baseline** ("the AQI in N hours
  will be what it is now") evaluated on identical rows

The deployed artifact is refit on 100% of the data, so reported metrics
*estimate* its generalisation rather than measure it. This is standard practice
but worth stating explicitly.

---

## 5. Results

### 5.1 Why absolute-level prediction fails

With one year of data starting in August, the first expanding-window fold
trains on late-summer air and is scored on winter smog:

| Fold | Train target mean | Test target mean | Train max | Test max |
|---|---|---|---|---|
| 0 | 43.1 | 128.5 | 85.1 | 201.0 |
| 1 | 85.1 | 118.5 | 201.0 | 217.6 |
| 2 | 96.4 | 88.4 | 217.6 | 201.0 |

Tree ensembles cannot predict outside the range of their training targets. A
Random Forest that has never seen an AQI above 85 physically cannot output 201,
so fold 0 produces a large negative R² regardless of feature quality, and that
fold dominates the pooled metric.

Predicting the change instead of the level is roughly stationary across seasons
and fixes most of this:

| Horizon | Model | R² (absolute) | R² (delta) |
|---|---|---|---|
| 24h | Random Forest | −0.346 | **+0.245** |
| 72h | Random Forest | −0.745 | **+0.074** |
| 24h | Ridge | −0.941 | −0.949 |

Ridge is unaffected because a linear model extrapolates freely — its problem is
different, and it remains the worst candidate at every horizon.

### 5.2 Final model comparison

Pooled out-of-sample RMSE / MAE / R² across all five folds. Lower RMSE is better.

**+24h**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **LSTM (TensorFlow)** | **30.29** | **18.50** | **0.514** |
| Persistence baseline | 30.40 | 18.52 | 0.509 |
| Random Forest | 37.67 | 26.82 | 0.245 |
| Gradient Boosting | 42.52 | 30.06 | 0.038 |
| Ridge | 60.53 | 38.46 | −0.949 |

**+48h**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **LSTM (TensorFlow)** | **38.53** | **25.74** | **0.209** |
| Persistence baseline | 40.13 | 26.33 | 0.139 |
| Random Forest | 43.80 | 32.77 | −0.026 |
| Gradient Boosting | 44.51 | 33.37 | −0.060 |
| Ridge | 65.66 | 46.47 | −1.306 |

**+72h**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Random Forest** | **41.30** | **30.09** | **0.074** |
| Gradient Boosting | 42.98 | 31.53 | −0.002 |
| LSTM (TensorFlow) | 43.14 | 29.39 | −0.011 |
| Persistence baseline | 43.97 | 29.88 | −0.049 |
| Ridge | 64.29 | 46.93 | −1.243 |

**Summary of what is deployed**

| Horizon | Deployed model | RMSE | Baseline RMSE | Improvement |
|---|---|---|---|---|
| +24h | LSTM | 30.29 | 30.40 | 0.4% |
| +48h | LSTM | 38.53 | 40.13 | 4.0% |
| +72h | Random Forest | 41.30 | 43.97 | 6.1% |

### 5.3 Honest reading of these numbers

**Accuracy degrades with horizon, as expected.** R² falls from 0.51 at 24h to
0.07 at 72h. Any claim otherwise would be suspicious.

**The +24h margin over persistence is negligible.** 30.29 vs 30.40 RMSE is a
0.4% improvement and is well within fold-to-fold noise — fold 1 alone ranges
from R² 0.178 to 0.725. The honest statement is that at 24 hours **this system
matches persistence, it does not beat it.** Short-horizon AQI is strongly
autocorrelated, and "tomorrow looks like today" is a genuinely strong baseline
that published forecasting systems also struggle to beat.

**The margin is real at 48h and 72h.** 4–6% improvements are where the models
earn their place, which makes sense: persistence degrades quickly as the horizon
grows, while the models can use weather and seasonal structure.

**Every horizon is verified against the baseline at training time.** If a
trained model loses, `beats_baseline: false` is recorded in the registry and
both the dashboard and the API serve the persistence forecast instead, labelled
as such. Before the delta-target change this was the case at 24h and 48h.

### 5.4 What the models look at

SHAP on the deployed models (GradientExplainer for the LSTMs, TreeExplainer for
the Random Forest) gives a consistent picture at +24h:

1. `dow_sin` / `is_weekend` — weekly traffic rhythm is the strongest signal
2. `pm10` — coarse particulates, the second most common AQI driver
3. `wind_speed` — dispersal
4. `hour_cos` / `hour_sin` — daily rush-hour cycle

The attribution-by-timestep panel shows the LSTM's attention peaking 5–8 hours
before prediction time and decaying steadily further back, with the oldest hours
in the 48-hour window contributing very little. That suggests a shorter window
would lose almost nothing, and is worth testing.

---

## 6. Engineering

### 6.1 Automation

| Workflow | Schedule | Action |
|---|---|---|
| `feature_pipeline.yml` | Hourly (`0 * * * *`) | Fetch, compute features, commit the store |
| `training_pipeline.yml` | Daily (`0 3 * * *`) | Retrain all horizons, commit the registry |

Both declare `permissions: contents: write` (the default `GITHUB_TOKEN` is
read-only on new repositories) and share a `concurrency` group so the hourly and
daily jobs cannot collide when they both push at 03:00 UTC. The commit steps
tolerate a missing `data/` or `models/` directory on first run and retry with a
rebase if another run pushed first.

### 6.2 Reliability

- API calls retry with exponential backoff, distinguishing rate limits (429)
  and server errors from an invalid key (401), which fails immediately
- The feature store is written atomically via a temp file and `os.replace`, so
  an interrupted run cannot truncate the only copy of the history
- Feature-store writes merge column-wise, so a partial row cannot null out
  columns already stored for that hour
- Timestamps are normalised to tz-naive UTC on every read and write, preventing
  the tz-aware/naive mix that silently defeats deduplication
- Missing or null features at prediction time are reported by name rather than
  raising a NaN error into the dashboard

### 6.3 Uncertainty

Forecasts carry an 80% band. Width is anchored to the model's own out-of-sample
CV residual spread — which captures total error, not just model variance — and
for the Random Forest is modulated per prediction by how much the individual
trees disagree about that input, clamped to 0.5–2× the base width.

This is a heuristic, not a calibrated prediction interval, and is labelled as
such in the UI. Proper calibration would use conformal prediction or quantile
regression.

---

## 7. Limitations

1. **One year of data.** Not enough to distinguish seasonal patterns from
   year-specific weather. This is the binding constraint on everything else.
2. **No forecast weather as input.** The single largest missed opportunity.
   The models predict 24–72h ahead using only conditions observed *now*. A
   genuine weather forecast (Open-Meteo's forecast endpoint is free) would give
   the model the future conditions it is implicitly trying to guess.
3. **Instantaneous readings used against averaged-window breakpoints** (§2.2).
4. **Single monitoring point.** OpenWeather interpolates to a lat/lon rather
   than reporting a ground station; a city the size of Karachi has real spatial
   variation this cannot capture.
5. **No event awareness.** Dust storms, crop burning and industrial incidents
   drive the largest AQI excursions and are invisible to these features.
6. **The +24h model only matches persistence** (§5.3).

---

## 8. What would most improve accuracy, in priority order

1. **Add forecast weather as an input feature.** Highest leverage by a wide
   margin, and cheap — the data is free and the pipeline already exists.
2. **Accumulate 2+ years of history.** Fixes the fold-0 seasonal extrapolation
   problem at its root and would let the LSTM use a longer window.
3. **Correct the AQI averaging windows** (24h for PM, 8h for O₃/CO) before the
   breakpoint lookup.
4. **Conformal prediction** for calibrated rather than heuristic intervals.
5. **Hyperparameter search** for the LSTM — the current architecture
   (64→32 units, 48-hour window) was chosen by reasoning, not tuning, and the
   attribution decay curve suggests a shorter window may do as well.
6. **Spatial features** if additional monitoring stations become available.

---

## 9. Reproducing these results

```bash
pip install -r requirements.txt
export OPENWEATHER_API_KEY=your_key

python src/backfill.py --days 90
python src/train_pipeline.py          # ~17 min on CPU, LSTM dominates the time
python src/predict.py

streamlit run app/streamlit_app.py    # dashboard
uvicorn api.main:app --port 8000      # REST API
```

Metrics for every horizon, including per-fold breakdowns and the baseline
comparison, are written to `models/metrics_{horizon}h.json`.
