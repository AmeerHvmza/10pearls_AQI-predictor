"""
Exploratory Data Analysis for the AQI feature store.
Run as a script (or paste into a Jupyter notebook cell by cell):

    python notebooks/eda.py
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from utils import load_feature_store

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

df = load_feature_store()
if df.empty:
    raise SystemExit("Feature store is empty — run backfill.py first.")

print(df.describe())

# AQI over time
plt.figure(figsize=(12, 4))
plt.plot(df["timestamp"], df["aqi"])
plt.title("AQI over time")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "aqi_timeseries.png"))
plt.close()

# Hourly seasonality
plt.figure(figsize=(8, 4))
sns.boxplot(x="hour", y="aqi", data=df)
plt.title("AQI distribution by hour of day")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "aqi_by_hour.png"))
plt.close()

# Correlation heatmap
numeric_cols = df.select_dtypes("number").columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"))
plt.close()

print(f"EDA plots written to {OUT_DIR}")
