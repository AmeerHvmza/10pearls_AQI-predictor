"""Shared helpers: feature-store I/O and AQI utilities."""
import os
import sys
import tempfile
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Short gaps are bridged by interpolation so that lag features stay aligned to
# real time. Anything longer is left as NaN: fabricating a full day of air
# quality would be worse than dropping those rows during training.
MAX_INTERPOLATE_HOURS = 3


def aqi_category(aqi: float) -> str:
    if aqi is None or pd.isna(aqi):
        return "Unknown"
    aqi = float(aqi)
    if aqi < 0:
        return "Good"
    for lo, hi, label in config.AQI_CATEGORIES:
        if lo <= aqi < hi:
            return label
    return "Hazardous"


def category_color(category: str) -> str:
    return config.AQI_CATEGORY_COLORS.get(category, "#8b95a5")


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Force `timestamp` to tz-naive UTC datetimes.

    Every producer in this project works in UTC, but they disagree on whether
    they attach a tzinfo. Mixing the two turns the column into dtype `object`,
    which makes equality-based dedup fail silently and lets duplicate hours
    accumulate in the store.
    """
    if df.empty or "timestamp" not in df.columns:
        return df
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = ts.dt.tz_localize(None)
    return df.dropna(subset=["timestamp"])


def to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex the feature store onto a continuous hourly UTC grid.

    Lag/rolling features are computed with positional `shift()`, so a missing
    hour would otherwise make `shift(24)` mean "24 rows back" instead of
    "24 hours back" and quietly misalign every downstream feature. Rows added
    here are flagged via `is_imputed` so models and the dashboard can tell
    measured hours from bridged ones.
    """
    if df.empty or "timestamp" not in df.columns:
        return df

    df = normalize_timestamps(df).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    full_index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full_index)
    df.index.name = "timestamp"

    inserted = df["aqi"].isna() if "aqi" in df.columns else pd.Series(False, index=df.index)
    numeric_cols = df.select_dtypes("number").columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time", limit=MAX_INTERPOLATE_HOURS, limit_area="inside"
    )

    df = df.reset_index()
    df["is_imputed"] = (inserted.values & df["aqi"].notna().values).astype(int)

    # Calendar fields are derived, not measured, so recompute them rather than
    # letting interpolation invent fractional hours or months.
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return df


def describe_gaps(df: pd.DataFrame) -> list:
    """Return [(gap_start, gap_end, missing_hours), ...] for reporting."""
    if df.empty or "timestamp" not in df.columns:
        return []
    ts = normalize_timestamps(df)["timestamp"].sort_values()
    deltas = ts.diff()
    gaps = []
    for prev, delta in zip(ts.shift(), deltas):
        if pd.notna(delta) and delta > pd.Timedelta(hours=1):
            missing = int(delta.total_seconds() // 3600) - 1
            gaps.append((prev, prev + delta, missing))
    return gaps


def load_feature_store() -> pd.DataFrame:
    """Read the feature store. Returns an empty frame with the right
    schema if nothing has been written yet."""
    if config.FEATURE_STORE_BACKEND == "hopsworks":
        return normalize_timestamps(_load_from_hopsworks())

    if os.path.exists(config.FEATURES_PATH):
        return normalize_timestamps(pd.read_parquet(config.FEATURES_PATH))
    return pd.DataFrame()


def save_feature_store(df: pd.DataFrame) -> None:
    """Append-and-dedupe write to the feature store."""
    if config.FEATURE_STORE_BACKEND == "hopsworks":
        return _save_to_hopsworks(df)

    if df is None or df.empty:
        print("Nothing to save: received an empty frame.")
        return

    os.makedirs(config.DATA_DIR, exist_ok=True)
    df = normalize_timestamps(df)
    existing = load_feature_store()

    combined = df if existing.empty else pd.concat([existing, df], ignore_index=True)
    before = len(combined)

    # Stable sort keeps existing rows ahead of incoming ones at the same
    # timestamp, and GroupBy.last() takes the last *non-null* value per column.
    # So a newer row wins wherever it has data, but a column it simply doesn't
    # carry falls back to the stored value instead of being nulled out.
    combined = combined.sort_values("timestamp", kind="stable")
    combined = combined.groupby("timestamp", as_index=False, sort=True).last()
    combined = combined.reset_index(drop=True)

    _atomic_write_parquet(combined, config.FEATURES_PATH)

    gaps = describe_gaps(combined)
    missing = sum(g[2] for g in gaps)
    print(f"Feature store now has {len(combined)} rows "
          f"({before - len(combined)} duplicate timestamps merged) -> {config.FEATURES_PATH}")
    if gaps:
        print(f"  note: {len(gaps)} gap(s) totalling {missing} missing hour(s); "
              f"lag features are computed on a reindexed hourly grid.")


def _atomic_write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write via a temp file + replace so an interrupted run can't leave a
    truncated parquet behind (the store is the only copy of the history)."""
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=directory)
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def hopsworks_login():
    """Authenticate against Hopsworks, with actionable errors.

    Kept separate from the read/write helpers so the training pipeline can
    reuse the same session for the Model Registry.
    """
    try:
        import hopsworks
    except ImportError as exc:
        raise SystemExit(
            "The Hopsworks client is not installed.\n"
            "  pip install hopsworks"
        ) from exc

    if not config.HOPSWORKS_API_KEY:
        raise SystemExit(
            "HOPSWORKS_API_KEY is not set.\n"
            "  Create a free project at https://app.hopsworks.ai, then\n"
            "  Account Settings > API keys > New API key (scopes: "
            "featurestore, project, job).\n"
            "  Local:  export HOPSWORKS_API_KEY=your_key\n"
            "  CI:     add it as a repository secret."
        )

    return hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT or None,
    )


def _hopsworks_feature_group(create: bool):
    project = hopsworks_login()
    fs = project.get_feature_store()
    if not create:
        try:
            return fs.get_feature_group(config.HOPSWORKS_FEATURE_GROUP,
                                        version=config.HOPSWORKS_FG_VERSION)
        except Exception:
            # Nothing has been written yet; callers treat this as an empty store.
            return None
    return fs.get_or_create_feature_group(
        name=config.HOPSWORKS_FEATURE_GROUP,
        version=config.HOPSWORKS_FG_VERSION,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly AQI + weather features for 3-day AQI forecasting",
        online_enabled=False,
    )


def _load_from_hopsworks() -> pd.DataFrame:
    """Optional Hopsworks backend. Requires `pip install hopsworks`
    and HOPSWORKS_API_KEY / HOPSWORKS_PROJECT env vars."""
    fg = _hopsworks_feature_group(create=False)
    if fg is None:
        print(f"Feature group '{config.HOPSWORKS_FEATURE_GROUP}' does not exist "
              f"yet; treating the store as empty.")
        return pd.DataFrame()
    return fg.read()


def _save_to_hopsworks(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print("Nothing to save: received an empty frame.")
        return

    df = normalize_timestamps(df)
    # Hopsworks lower-cases feature names and rejects most punctuation, so
    # normalise here rather than letting the insert fail server-side.
    df = df.rename(columns={c: c.lower().replace(".", "_").replace("-", "_")
                            for c in df.columns})

    fg = _hopsworks_feature_group(create=True)
    # `timestamp` is the primary key, so an insert upserts: re-sending an hour
    # that already exists updates it instead of duplicating it.
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Wrote {len(df)} rows to Hopsworks feature group "
          f"'{config.HOPSWORKS_FEATURE_GROUP}' v{config.HOPSWORKS_FG_VERSION}.")
