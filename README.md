# Pearls AQI Predictor

🔗 **Live demo:** https://10pearlsaqi-predictor-ameerhamza.streamlit.app/

An end-to-end, serverless AQI forecasting system: hourly feature pipeline →
daily training pipeline → 3-day forecast dashboard, automated with GitHub
Actions.

## At a glance

| | |
|---|---|
| **What** | 3-day US EPA AQI forecast for Karachi (`+24h` / `+48h` / `+72h`) |
| **Status** | Complete and live. Hourly feature collection and daily retraining run on GitHub Actions; the dashboard is deployed on Streamlit Community Cloud. |
| **Data** | `data/features.parquet`: **8,839** hourly rows, **2025-09-03 00:00 → 2026-09-06 06:00 UTC** (grows with each hourly run) |
| **Deployed models** | Random Forest at every horizon (`beats_baseline=true` vs persistence) |
| **Source of truth** | Local parquet for training and serving. Hopsworks is an optional dual-write replica. |
| **Stack** | Python, scikit-learn, TensorFlow/Keras, Hopsworks, GitHub Actions, Streamlit, FastAPI, OpenWeather, Open-Meteo, SHAP, Git |

## Tech stack

| Layer | Technology | Role in this repo |
|---|---|---|
| Language | Python 3.11 (`runtime.txt`) | All pipelines, dashboard, and API |
| Tabular ML | scikit-learn `1.9.0`, joblib | Ridge, Random Forest, Gradient Boosting; deployed artifacts |
| Deep learning | TensorFlow / Keras (`requirements-train.txt`) | Optional LSTM candidate (12-hour sequences) |
| Feature store | Apache Parquet (PyArrow) | Source of truth for training and serving |
| Managed replica | Hopsworks (`requirements-ingest.txt`) | Optional stream ingest (`stream=True`, Kafka) |
| Explainability | SHAP | Per-horizon feature importance (`models/shap_*`) |
| Human UI | Streamlit + Plotly | Dashboard and Model Analysis pages |
| Machine API | FastAPI + Uvicorn | Same forecasts as the dashboard |
| Data APIs | OpenWeather, Open-Meteo | Pollution + current weather; historical weather archive |
| Automation | GitHub Actions | Hourly features, daily retrain, commit-back |
| Versioning | Git | Feature store and model registry live in the repo |

## Architecture

```
OpenWeather API ──┐
                  ├──(hourly, GH Actions)──▶ feature_pipeline.py ──▶ data/features.parquet  (source of truth)
Open-Meteo API ───┘                                              └──▶ Hopsworks (optional dual-write, stream ingest)
                                                                        │
                                                     (daily, GH Actions)│
                                                                        ▼
                                                              train_pipeline.py
                                                    (Ridge / RF / GBM / LSTM, SHAP)
                                                                        │
                                                                        ▼
                                                              models/ (model registry)
                                                                        │
                    ┌───────────────────────────────────────────────────┴────────────────────────┐
                    ▼                                                                            ▼
     streamlit run app/streamlit_app.py                                           uvicorn api.main:app
```

## 1. Setup

```bash
cd aqi_predictor
python -m venv venv
# Unix / macOS:
source venv/bin/activate
# Windows PowerShell:
# .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` is the serving stack (dashboard, API, parquet feature
store, sklearn models). Streamlit Cloud reads `runtime.txt` and installs
that file only — it does **not** need TensorFlow or Hopsworks. The
deployed models are Random Forest joblibs loaded via `predict.py`.

To train or retrain locally (including the optional Keras LSTM candidate):

```bash
pip install -r requirements-train.txt
```

Hourly GitHub Actions dual-write to Hopsworks also needs:

```bash
pip install -r requirements-ingest.txt
```

Get a **free** OpenWeather API key: https://openweathermap.org/api
(Air Pollution API + Current Weather are both free tier.)

```bash
export OPENWEATHER_API_KEY=your_key_here
export AQI_CITY="Karachi"
export AQI_LAT=24.8607
export AQI_LON=67.0011
```

On Windows PowerShell, use `$env:OPENWEATHER_API_KEY="your_key_here"` (and
the same form for the other variables). `config.py` also loads a project-root
`.env` if present; that file is gitignored.

## 2. Backfill historical data

```bash
python src/backfill.py --days 365
```

OpenWeather's air-pollution history goes back to ~Nov 2020, so you can pull
up to a year; weather history is fetched free from Open-Meteo (no key
needed). More days = better-trained models. The committed store was built
from a 365-day real backfill plus hourly cron rows — do not recreate it
with a synthetic generator. Fetch logs live in `data/PROVENANCE.md`.

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
LSTM** (deep learning) over 12-hour sequences. The LSTM uses a deliberately
smaller, raw feature set — it learns temporal structure from the sequence, so
feeding it the engineered lag columns as well would be redundant.

If TensorFlow is not installed the LSTM candidate is skipped cleanly and the
other three still train. Daily GitHub Actions training installs
`requirements-train.txt` so the LSTM still competes in CI.

On the current registry, Random Forest wins all three horizons and beats
persistence on RMSE:

| Horizon | Deployed model | RMSE | Persistence RMSE | Pooled R² |
|---|---|---:|---:|---:|
| +24h | Random Forest | 27.43 | 29.51 | 0.567 |
| +48h | Random Forest | 34.98 | 39.09 | 0.294 |
| +72h | Random Forest | 39.14 | 42.57 | 0.102 |

## 4. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

The theme lives in `.streamlit/config.toml`. Two pages: **Dashboard**
(current conditions, 3-day forecast, uncertainty bands, health alerts, SHAP
drivers) and **Model Analysis** (history, accuracy vs persistence, SHAP
summary plots).

## 5. Run the REST API

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
| `GET /categories` | The EPA category scale this service reports |
| `GET /categories/{aqi}` | Classify an arbitrary AQI value |

## 6. Automate it

Push this repo to GitHub and configure the following repository settings.
The two workflows in `.github/workflows/` then run the feature pipeline
hourly and retraining daily, committing updated data/models back to the
repo — no servers required (GitHub Actions' free tier covers this easily).

| Name | Type | Used by |
|---|---|---|
| `OPENWEATHER_API_KEY` | **Secret** | Feature pipeline |
| `HOPSWORKS_API_KEY` | **Secret** | Feature pipeline (dual-write) |
| `AQI_CITY` | Variable | Both workflows |
| `AQI_LAT` | Variable | Both workflows |
| `AQI_LON` | Variable | Both workflows |
| `HOPSWORKS_PROJECT` | Variable | Feature pipeline |
| `HOPSWORKS_ENABLED` | Variable | Feature pipeline (`true` to dual-write) |
| `HOPSWORKS_FG_VERSION` | Variable | Feature pipeline (default in code: `2`) |

`feature_pipeline.yml` installs `requirements.txt` plus
`requirements-ingest.txt` and passes the Hopsworks settings through so CI
can stream-ingest into the managed feature group. A Hopsworks failure is
logged and ignored; parquet still commits.

The training workflow reads parquet only. It installs `requirements-train.txt`
and does not require Hopsworks to be enabled.

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

## 7. EDA

```bash
python notebooks/eda.py    # writes plots to data/eda/
```

## Optional Hopsworks dual-write

Parquet (`data/features.parquet`) is **always** the source of truth.
Training (`train_pipeline.py`), serving (`predict.py`, Streamlit, FastAPI),
and the dashboard never read Hopsworks. When Hopsworks is enabled, every
successful parquet write also attempts a best-effort replica insert.

This is dual-write, not a backend switch. `FEATURE_STORE_BACKEND` is
deprecated and ignored if set.

**Why stream ingest.** The default Hopsworks Python writer uses Delta-RS
against HDFS and needs Kerberos. That path fails on native Windows
(`libgssapi_krb5` / RPC abort). Inserts therefore use `stream=True`: rows
are published to Kafka and Hopsworks materializes the offline store
server-side. That was a deliberate choice so the same ingest code works on
a Windows workstation and on Ubuntu CI, without a second write path.

To enable the replica:

1. Create a free project at https://app.hopsworks.ai
2. Account Settings → API keys → New API key
   (scopes: `featurestore`, `project`, `job`)
3. Install the ingest extras and set:

```bash
pip install -r requirements-ingest.txt
export HOPSWORKS_ENABLED=true
export HOPSWORKS_API_KEY=your_key
export HOPSWORKS_PROJECT=your_project_name
export HOPSWORKS_FG_VERSION=2
```

The feature group defaults to `aqi_features` version 2 (`stream=True`).
`timestamp` is the primary key and event time, so re-sending an hour upserts
rather than duplicating. If the Hopsworks write fails, the error is logged
and the pipeline still finishes — parquet has already been updated.

For CI, add `HOPSWORKS_API_KEY` as a repository **secret** and
`HOPSWORKS_PROJECT` / `HOPSWORKS_ENABLED` / `HOPSWORKS_FG_VERSION` as
repository **variables**. The hourly workflow already installs
`requirements-ingest.txt` and passes these through.

## Retrospective

The system described above is implemented, trained on a year of real
OpenWeather + Open-Meteo history, automated, and deployed. The work mixed
pipeline implementation with calendar time for data accumulation: a 90-day
verified pull, a 365-day merge, hourly cron, and a repair that unified the
local and GitHub stores without dropping measured hours
(see `data/PROVENANCE.md`).

There is no such thing as a perfect AQI forecast. AQI depends on weather
(which is itself only forecastable with uncertainty), traffic, industrial
activity, and sometimes one-off events (fires, dust storms) that no model
sees coming. Realistic, respectable performance for a 24h-ahead AQI model is
typically **R² in the 0.7–0.9 range** and degrades further at +48h/+72h —
expect noticeably worse accuracy at 72h than 24h, that's expected and worth
stating plainly in the report rather than chasing an unrealistic target.
Reporting that degradation honestly (with RMSE/MAE per horizon, as this
pipeline already computes) is itself part of a strong submission.

### What would most improve accuracy, in priority order
1. **More historical data** — 90 days is a bare minimum; 6–12 months captures seasonal patterns much better.
2. **A real weather forecast API** (not just current conditions) as an *input* feature for future predictions — right now the model predicts from current-time features only, since it has no forecast weather to condition on 24–72h out. Adding OpenWeather's or Open-Meteo's forecast endpoint as an input is the single highest-leverage upgrade.
3. **More monitoring stations / spatial features** if your city has multiple AQI stations.
4. Hyperparameter tuning and trying LSTM/Temporal Fusion Transformer once you have enough data (dozens of thousands of hourly rows) to justify it — with under a few thousand rows, gradient boosting/random forest will usually beat deep learning.

## Documentation

| Artifact | What it is |
|---|---|
| `report/` | Full formal LaTeX write-up (`report/aqi_report.tex` → `report/aqi_report.pdf`) with figures, tables, and dashboard screenshots. |

Compile the formal report with:

```bash
cd report
pdflatex aqi_report.tex   # run three times for ToC and references
```

## What this repo includes

Hourly feature collection and a one-shot backfill, a training pipeline
that scores Ridge / Random Forest / GBM / LSTM against a persistence
baseline (RMSE, MAE, R²), a committed model registry, GitHub Actions for
hourly features and daily retraining, a Streamlit dashboard with a
3-day forecast, uncertainty bands, alerts, and SHAP, plus EDA scripts
and a FastAPI serving layer over the same parquet store. The write-up
lives in `report/` (full LaTeX write-up and compiled PDF).
