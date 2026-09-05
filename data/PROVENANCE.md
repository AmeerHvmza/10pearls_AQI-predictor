# Feature-store provenance

This file exists so a parquet sitting in `data/` cannot be mistaken for
measured history without a recorded fetch.

## This pull

| Field | Value |
|---|---|
| Generated at (parquet mtime, UTC) | `2026-09-01T09:25:21.564373+00:00` |
| Command | `python src/backfill.py --days 90` |
| Working directory | `d:\UserFiles\Desktop\aqi_predictor` |
| Location | Karachi (`AQI_LAT=24.8607`, `AQI_LON=67.0011`) |
| Secret source | project-root `.env` (`OPENWEATHER_API_KEY`). `.env` is gitignored (`.gitignore` line 8). The key is not recorded here. |
| Pollution API | `https://api.openweathermap.org/data/2.5/air_pollution/history` |
| Weather API | `https://archive-api.open-meteo.com/v1/archive` (no key) |
| Successful pollution HTTP 200s | 3 chunks |
| OpenWeather `list` entries summed | 721 + 697 + 673 = **2091** |
| Successful weather HTTP 200s | 1 |
| Open-Meteo hourly timestamps | **2064** |
| Rows written to `data/features.parquet` | **2055** |
| Timestamp range (UTC) | 2026-06-03 09:00:00 → 2026-08-27 23:00:00 |
| Join note | 82 pollution hours had no weather row (Open-Meteo archive lags ~5 days). Those hours were not written. |

## HTTP GET URLs actually called (API key redacted)

```
https://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8607&lon=67.0011&start=1780477200&end=1783069200&appid=REDACTED
https://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8607&lon=67.0011&start=1783069200&end=1785661200&appid=REDACTED
https://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8607&lon=67.0011&start=1785661200&end=1788253200&appid=REDACTED
https://archive-api.open-meteo.com/v1/archive?latitude=24.8607&longitude=67.0011&start_date=2026-06-03&end_date=2026-08-27&hourly=temperature_2m%2Crelative_humidity_2m%2Csurface_pressure%2Cwind_speed_10m%2Cwind_direction_10m%2Ccloud_cover&timezone=UTC
```

## SHA256

| File | Bytes | SHA256 |
|---|---|---|
| `data/features.parquet` | 110333 | `1e603a0a7084c1fe3a47b9e66e83515edc41990c114ea7591a11ae2412a44074` |
| `data/provenance_first_pollution.json` | 114589 | `694af2a1f20890406b3610b8d14191bd9dfb21773b300f3df1a7eda4345fb963` |
| `data/provenance_first_weather.json` | 109143 | `f55ca72eda1c167e84c31e690ca490937fbe740a08df7c51e88b93caca8f33dc` |

The two `provenance_first_*.json` files are the **unmodified JSON bodies** of
the first successful pollution fetch (OpenWeather history chunk, 721 `list`
items) and the first (only) weather fetch (Open-Meteo archive). They do not
contain the API key.

## Terminal output of this run (verbatim)

```
Note: Open-Meteo's archive lags ~5 days, so weather history stops at 2026-08-27. The hourly feature pipeline fills the remainder going forward.
Fetching 90 days of pollution history...
  pollution 2026-06-03 -> 2026-07-03
HTTP GET https://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8607&lon=67.0011&start=1780477200&end=1783069200&appid=REDACTED
  -> HTTP 200, list entries this chunk: 721
  -> saved first pollution JSON -> D:\UserFiles\Desktop\aqi_predictor\data\provenance_first_pollution.json
  pollution 2026-07-03 -> 2026-08-02
HTTP GET https://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8607&lon=67.0011&start=1783069200&end=1785661200&appid=REDACTED
  -> HTTP 200, list entries this chunk: 697
  pollution 2026-08-02 -> 2026-09-01
HTTP GET https://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8607&lon=67.0011&start=1785661200&end=1788253200&appid=REDACTED
  -> HTTP 200, list entries this chunk: 673
Fetching matching weather history from Open-Meteo (free archive)...
HTTP GET https://archive-api.open-meteo.com/v1/archive?latitude=24.8607&longitude=67.0011&start_date=2026-06-03&end_date=2026-08-27&hourly=temperature_2m%2Crelative_humidity_2m%2Csurface_pressure%2Cwind_speed_10m%2Cwind_direction_10m%2Ccloud_cover&timezone=UTC
  -> HTTP 200, hourly timestamps: 2064
  -> saved first weather JSON -> D:\UserFiles\Desktop\aqi_predictor\data\provenance_first_weather.json
  82 pollution hour(s) had no weather row; 0 more had null weather values. Both were skipped.
Feature store now has 2055 rows (0 duplicate timestamps merged) -> D:\UserFiles\Desktop\aqi_predictor\data\features.parquet
Backfilled 2055 hourly rows (2026-06-03 09:00:00 -> 2026-08-27 23:00:00 UTC).
```

## Entry #2 — restore verified 2055-row store (2026-09-01)

**Why.** `git pull` had replaced the working copy with GitHub commit
`9897269` (`9145` rows, sha256 `dd9267fd…`), which is the unverified
8760-row file (now known to have come from `synthetic_backfill.py`) plus
later Actions hourly rows. That file is superseded. `src/synthetic_backfill.py`
was deleted from the working tree so it cannot generate another year of
fake history.

**Restore.** Recovered `data/features.parquet` from `stash@{0}` only (did
not `git stash pop`, so other stashed edits were not mixed in). Hash
checked **before** any further writes:

| | |
|---|---|
| SHA256 | `1e603a0a7084c1fe3a47b9e66e83515edc41990c114ea7591a11ae2412a44074` |
| Bytes | 110333 |
| Rows | 2055 |
| Range (UTC) | 2026-06-03 09:00:00 → 2026-08-27 23:00:00 |
| MATCH | True vs entry #1 |

**Then** `python src/feature_pipeline.py` with the real key (`.env`).
OpenWeather current pollution + weather. Terminal (verbatim):

```
Feature store now has 2056 rows (0 duplicate timestamps merged) -> D:\UserFiles\Desktop\aqi_predictor\data\features.parquet
  note: 1 gap(s) totalling 107 missing hour(s); lag features are computed on a reindexed hourly grid.
Feature store now has 2163 rows (2056 duplicate timestamps merged) -> D:\UserFiles\Desktop\aqi_predictor\data\features.parquet
[2026-09-01 11:00:00Z] AQI=59.4 (driver: pm2_5, OWM index: 3) written for Karachi
```

The 107 hours between 2026-08-27 23:00 and 2026-09-01 11:00 are **grid
placeholders**, not API history (Open-Meteo archive cannot cover them).
Only `2026-09-01 11:00` is a new measured hour.

| After feature_pipeline | |
|---|---|
| SHA256 | `2c6ca87ecaea8cf20b4cfbfa8a6b76cf7521d4b7084fae3ed7264766bb144e85` |
| Bytes | 113138 |
| Rows | 2163 unique timestamps |
| Range (UTC) | 2026-06-03 09:00:00 → 2026-09-01 11:00:00 |
| Measured AQI hours | 2023 |

LSTM lookback was then changed 48h → 12h and models retrained on this
file. `keras_path` in new joblibs is the relative name `model_{h}h.keras`.

**Forecast.** After retrain, `python src/predict.py` and `GET /forecast`
still return no numeric AQI: the 12h input window ending at 2026-09-01
11:00 still contains NaNs from the unmeasured grid hours immediately
before that latest fetch. Closing that needs ~12 consecutive successful
hourly `feature_pipeline` runs (or a recent weather API that can fill
those hours). No interpolation.

## Entry #3 — 365-day real backfill merged into verified store (2026-09-03)

**Why.** After hourly cron closed the Aug 27 → Sep 3 gap, the store had
**2200** continuous rows (`2026-06-03 09:00` → `2026-09-03 00:00`, sha256
`37c731c71fd9bfaada13a9a0441ce62d042022a04d96a97f33208c80a9375eed`,
135006 bytes). That window is one season. OpenWeather history + Open-Meteo
archive jointly allow a full year, so a larger **real** backfill was run
and **merged** (not replaced).

**Command.** `python src/backfill.py --days 365`

Pollution: 13 OpenWeather history GETs, all HTTP 200. List sizes:
721, 721, 721, 721, 673, 625, 696, 697, 673, 721, 697, 673, 120.
Weather: 1 Open-Meteo archive GET, HTTP 200, **8664** hourly timestamps
(`2025-09-03` → `2026-08-29`; archive lag ~5 days). 97 pollution hours
had no weather row and were skipped.

```
HTTP GET https://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8607&lon=67.0011&start=1756857600&end=1759449600&appid=REDACTED
  -> HTTP 200, list entries: 721
… (11 more pollution chunks, all HTTP 200) …
HTTP GET https://archive-api.open-meteo.com/v1/archive?latitude=24.8607&longitude=67.0011&start_date=2025-09-03&end_date=2026-08-29&hourly=temperature_2m%2Crelative_humidity_2m%2Csurface_pressure%2Cwind_speed_10m%2Cwind_direction_10m%2Ccloud_cover&timezone=UTC
  -> HTTP 200, hourly timestamps: 8664
Feature store now has 8761 rows (2103 duplicate timestamps merged)
Backfilled 8664 hourly rows (2025-09-03 00:00:00 -> 2026-08-29 23:00:00 UTC).
```

2103 overlapping hours were the June–August verified rows. The **97**
post-archive hours (`2026-08-29 23:00` → `2026-09-03 00:00`) from cron
were kept. Arithmetic: `8664 + 2200 − 2103 = 8761`.

| After merge | |
|---|---|
| SHA256 | `d606d790098f9ed4203843a898041c46df2c112dc143d93778fad2386e387679` |
| Bytes | 464214 |
| Unique timestamps | 8761 |
| Measured AQI hours | 8551 |
| Range (UTC) | 2025-09-03 00:00:00 → 2026-09-03 00:00:00 |

This 8761-row file is **not** the deleted `synthetic_backfill.py` 8760-row
file. It is OpenWeather + Open-Meteo history plus cron-filled recent hours.

## Entry #4 — local / GitHub split repaired (2026-09-05)

**What diverged.** After entry #3 the year-long store lived only in the
local working tree and was never pushed. GitHub Actions kept appending
to `origin/main`'s shorter store (the audited 90-day file plus hourly
cron). By 2026-09-05 those were two different files:

| Side | Rows | Range (UTC) | Measured AQI hours |
|---|---:|---|---:|
| Local working tree | 8797 | 2025-09-03 00:00 → 2026-09-04 12:00 | 8585 |
| `origin/main` | 2263 | 2026-06-03 09:00 → 2026-09-05 15:00 | 2263 |

**Merge.** Concatenate, `groupby(timestamp).last()` (local first, origin
last so origin fills local NaN placeholders). No measured hour from
either side was dropped.

| | |
|---|---|
| Timestamps only local | 6561 |
| Timestamps only origin | 27 |
| Overlap | 2236 |
| Real AQI hours only local | 6381 |
| Real AQI hours only origin | 59 |
| Real hours lost (either side) | **0** |
| Union of real hours | 8644 |

Then `to_hourly_grid` + derived lags, and one `feature_pipeline.py` run
wrote **2026-09-05 16:00 UTC** (AQI 47.1, driver pm2_5).

| After repair | |
|---|---|
| SHA256 | `a96a44647884bb36c5cba25daa2315d579aa80518ef1f364b930bd4da0602763` |
| Bytes | 487052 |
| Unique timestamps | 8825 |
| Measured AQI hours | 8705 |
| Range (UTC) | 2025-09-03 00:00:00 → 2026-09-05 16:00:00 |
| Max timestamp step | 1 hour (no gaps) |
