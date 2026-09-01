# Pearls AQI Predictor — Full Implementation

An end-to-end, serverless AQI forecasting system: hourly feature pipeline →
daily training pipeline → 3-day forecast dashboard, automated with GitHub
Actions.

## Architecture

```
OpenWeather API ──(hourly, GH Actions)──▶ feature_pipeline.py ──▶ feature store (data/features.parquet)
                                                                        │
                                                     (daily, GH Actions)│
                                                                        ▼
                                                              train_pipeline.py
                                                          (Ridge / RF / GBM, SHAP)
                                                                        │
                                                                        ▼
                                                              models/ (model registry)
                                                                        │
                                                                        ▼
                                                        streamlit run app/streamlit_app.py
```

## 1. Setup (≈30–60 min)

```bash
cd aqi_predictor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Get a **free** OpenWeather API key: https://openweathermap.org/api
(Air Pollution API + Current Weather are both free tier.)

```bash
export OPENWEATHER_API_KEY=your_key_here
export AQI_CITY="Karachi"
export AQI_LAT=24.8607
export AQI_LON=67.0011
```

## 2. Backfill historical data (≈10 min to run, do this once)

```bash
python src/backfill.py --days 90
```

OpenWeather's air-pollution history goes back to ~Nov 2020, so you can pull
up to a year; weather history is fetched free from Open-Meteo (no key
needed). More days = better-trained models.

## 3. Train models

```bash
python src/train_pipeline.py
```

This trains Ridge, Random Forest, and Gradient Boosting for each of the
+24h/+48h/+72h horizons, picks the best by RMSE via time-series
cross-validation, and saves it plus a SHAP feature-importance plot to `models/`.

Models predict the **change** from the current AQI rather than the absolute
level. With roughly a year of history, the earliest expanding-window CV fold
trains on one season and is scored on another; tree ensembles cannot predict
outside their training range, so predicting the level directly produces a
negative R² on that fold regardless of feature quality. The change from the
current reading is far closer to stationary across seasons.

Every horizon is also scored against a **persistence baseline** ("the AQI in N
hours will be what it is now"). If the baseline wins, that is recorded in
`models/metrics_*.json` and the dashboard serves the baseline for that horizon
rather than knowingly publishing the worse forecast. Beating persistence at
+24h is genuinely hard and is not expected until the feature store covers
multiple years.

Four model families compete at every horizon: **Ridge** (statistical),
**Random Forest** and **Gradient Boosting** (classical ML), and a **Keras
LSTM** (deep learning) over 48-hour sequences. The LSTM uses a deliberately
smaller, raw feature set — it learns temporal structure from the sequence, so
feeding it the engineered lag columns as well would be redundant.

If TensorFlow is not installed the LSTM candidate is skipped cleanly and the
other three still train.

## 4. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

The theme lives in `.streamlit/config.toml`.

## 4b. Run the REST API

The dashboard is the human interface; FastAPI is the machine interface. Both
read the same feature store and model registry — the prediction logic is not
duplicated.

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive Swagger docs at http://localhost:8000/docs.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, row count, which horizons have models |
| `GET /current` | Latest measured AQI, category, dominant pollutant |
| `GET /forecast` | 3-day forecast with uncertainty bands |
| `GET /history?hours=336` | Recent observed AQI |
| `GET /alerts` | Whether hazardous AQI is expected |
| `GET /metrics` | Per-horizon model metrics and baseline comparison |
| `GET /categories/{aqi}` | Classify an arbitrary AQI value |

## 5. Automate it

Push this repo to GitHub, add `OPENWEATHER_API_KEY` as a repo **secret**,
and `AQI_CITY`/`AQI_LAT`/`AQI_LON` as repo **variables**. The two workflows
in `.github/workflows/` will then run the feature pipeline hourly and
retraining daily, committing updated data/models back to the repo — no
servers required (GitHub Actions' free tier covers this easily).

Both workflows declare `permissions: contents: write`, which the commit-back
step requires — the default `GITHUB_TOKEN` is read-only on new repositories.
They also share a `concurrency` group so the hourly and daily jobs cannot
collide when they both push at 03:00 UTC.

The repo variables are optional: if you leave `AQI_LAT`/`AQI_LON` unset, the
defaults in `config.py` apply. (GitHub expands an undefined `vars.X` to an
empty string, so `config.py` treats blank as absent rather than trying to
parse `""` as a float.)

## Maintenance

```bash
python src/repair_feature_store.py           # dry run
python src/repair_feature_store.py --apply   # rebuild the aqi column in place
```

Recomputes the `aqi` column from the stored pollutant concentrations. Useful
after any change to the EPA conversion, since concentrations are stored raw.

### How AQI is computed

`owm_aqi_to_us_aqi()` returns the US EPA AQI as the **maximum of the
per-pollutant sub-indices** (PM2.5, PM10, O₃, CO, SO₂, NO₂), which is the EPA's
own definition — not PM2.5 alone. Gaseous pollutants are converted from µg/m³
to ppb/ppm at 25 °C and 1 atm before lookup.

Concentrations are truncated to each breakpoint table's precision before
lookup, as the EPA specifies. This matters: the table jumps from 12.0 to 12.1
µg/m³, so without truncation a PM2.5 reading of 12.03 matches no bucket at all.

One honest caveat for any write-up: the EPA defines these breakpoints against
averaging windows this data source does not provide (24h for PM, 8h for O₃ and
CO), so an instantaneous hourly reading is used as a proxy. That biases results
high during short spikes.

## 6. EDA

```bash
python notebooks/eda.py    # writes plots to data/eda/
```

## Swapping in Hopsworks (managed feature store)

The local-parquet backend is the default so the project runs end-to-end with
zero external accounts. To use a managed feature store instead:

1. Create a free project at https://app.hopsworks.ai
2. Account Settings → API keys → New API key
   (scopes: `featurestore`, `project`, `job`)
3. Install and configure:

```bash
pip install hopsworks
export HOPSWORKS_API_KEY=your_key
export HOPSWORKS_PROJECT=your_project_name
```

4. Verify connectivity, then migrate the existing history:

```bash
python src/verify_hopsworks.py             # connectivity + schema check
python src/verify_hopsworks.py --migrate   # upload the local parquet store
```

5. Switch the backend:

```bash
export FEATURE_STORE_BACKEND=hopsworks
```

Every component — feature pipeline, backfill, training, dashboard, API — now
reads and writes Hopsworks with no other code changes. `timestamp` is the
feature group's primary key and event time, so re-sending an hour upserts
rather than duplicating.

For CI, add `HOPSWORKS_API_KEY` as a repository **secret** and
`FEATURE_STORE_BACKEND` / `HOPSWORKS_PROJECT` as repository **variables**;
both workflows already pass them through.

---

## Realistic timeline to "complete and working"

Being honest about scope: this is a full applied ML systems project, not
a one-off script. Here's a grounded estimate, assuming a few focused hours
per day:

| Phase | What | Time |
|---|---|---|
| Setup + feature pipeline live | API keys, run `feature_pipeline.py` manually, confirm hourly data lands correctly | 0.5–1 day |
| Backfill + EDA | Pull 60–90+ days of history, sanity-check for gaps/outliers, look at seasonality | 1 day |
| First training pass | Get `train_pipeline.py` running end-to-end, sanity-check metrics aren't nonsense | 1 day |
| Automation | Wire up GitHub Actions, confirm hourly/daily runs actually commit fresh data | 0.5–1 day |
| Dashboard | Streamlit app, alerts, SHAP plots wired to real models | 1 day |
| **Minimum viable end-to-end system** | | **~4–5 days** |
| Iteration: more data, tune models | Let the hourly pipeline accumulate 30+ days of *real* backfilled data (not just synthesized), retrain, compare | **2–4 weeks of calendar time** (this is data accumulation time, not effort time) |
| Model improvement | Try lag features of different windows, add more pollutant stations, try LSTM/Transformer if data supports it, tune hyperparameters | 2–4 focused days spread across that period |
| Polish + report | Clean dashboard, write up methodology/results/limitations | 1–2 days |
| **Total to a genuinely solid, submittable project** | | **~3–4 weeks**, mixing active work (~8–10 days total effort) with pipeline data-collection time you can't compress |

**On "perfecting" the prediction — an important caveat:** there's no such
thing as a perfect AQI forecast. AQI depends on weather (which is itself
only forecastable with uncertainty), traffic, industrial activity, and
sometimes one-off events (fires, dust storms) that no model sees coming.
Realistic, respectable performance for a 24h-ahead AQI model is typically
**R² in the 0.7–0.9 range** and degrades further at +48h/+72h — expect
noticeably worse accuracy at 72h than 24h, that's expected and worth
stating plainly in your report rather than chasing an unrealistic target.
Reporting that degradation honestly (with RMSE/MAE per horizon, as this
pipeline already computes) is itself part of a strong submission.

### What would most improve accuracy, in priority order
1. **More historical data** — 90 days is a bare minimum; 6–12 months captures seasonal patterns much better.
2. **A real weather forecast API** (not just current conditions) as an *input* feature for future predictions — right now the model predicts from current-time features only, since it has no forecast weather to condition on 24–72h out. Adding OpenWeather's or Open-Meteo's forecast endpoint as an input is the single highest-leverage upgrade.
3. **More monitoring stations / spatial features** if your city has multiple AQI stations.
4. Hyperparameter tuning and trying LSTM/Temporal Fusion Transformer once you have enough data (dozens of thousands of hourly rows) to justify it — with under a few thousand rows, gradient boosting/random forest will usually beat deep learning.

## Deliverables checklist (maps to your brief)
- [x] Feature pipeline (`src/feature_pipeline.py`)
- [x] Backfill script (`src/backfill.py`)
- [x] Training pipeline with RMSE/MAE/R² (`src/train_pipeline.py`)
- [x] Model registry (`models/`)
- [x] CI/CD automation (`.github/workflows/`)
- [x] Interactive dashboard with forecast + alerts (`app/streamlit_app.py`)
- [x] SHAP feature importance (generated during training, shown in dashboard)
- [x] EDA (`notebooks/eda.py`)
- [x] Persistence baseline comparison per horizon (`models/metrics_*.json`)
- [x] Uncertainty bands on the forecast (ensemble spread + CV residual spread)
- [ ] Your write-up: once you've run this for a couple of weeks, write the
      "detailed report" documenting what you tried, real metrics you got,
      and what you'd do differently — I can help draft that once you have
      real numbers to report.
