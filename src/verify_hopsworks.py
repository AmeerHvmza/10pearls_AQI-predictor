"""
End-to-end check of the Hopsworks feature store backend.

The local parquet backend is the default so the project runs with zero
external accounts. This script proves the Hopsworks path works before you
switch the pipelines over to it.

Setup (one time, free tier):
  1. Create a project at https://app.hopsworks.ai
  2. Account Settings > API keys > New API key
     Scopes needed: featurestore, project, job
  3. Set the environment variables:
       export HOPSWORKS_API_KEY=your_key
       export HOPSWORKS_PROJECT=your_project_name
     PowerShell:
       $env:HOPSWORKS_API_KEY="your_key"
       $env:HOPSWORKS_PROJECT="your_project_name"
  4. pip install hopsworks

Run:
    python src/verify_hopsworks.py              # connectivity + schema check
    python src/verify_hopsworks.py --migrate    # also upload the local parquet

Once this passes, set FEATURE_STORE_BACKEND=hopsworks and every pipeline
(feature, backfill, training, dashboard, API) reads and writes Hopsworks
instead of the local file — no other code changes.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import hopsworks_login, normalize_timestamps


def _step(n, text):
    print(f"\n[{n}] {text}")


def run(migrate: bool):
    print("=" * 68)
    print("Hopsworks feature store verification")
    print("=" * 68)

    _step(1, "Checking configuration")
    print(f"  FEATURE_STORE_BACKEND = {config.FEATURE_STORE_BACKEND}")
    print(f"  HOPSWORKS_PROJECT     = {config.HOPSWORKS_PROJECT or '(default project)'}")
    print(f"  HOPSWORKS_API_KEY     = {'set' if config.HOPSWORKS_API_KEY else 'MISSING'}")
    print(f"  Feature group         = {config.HOPSWORKS_FEATURE_GROUP} "
          f"v{config.HOPSWORKS_FG_VERSION}")

    _step(2, "Authenticating")
    project = hopsworks_login()
    print(f"  Connected to project: {project.name}")

    _step(3, "Opening the feature store")
    fs = project.get_feature_store()
    print(f"  Feature store: {fs.name}")

    _step(4, "Looking for the feature group")
    try:
        fg = fs.get_feature_group(config.HOPSWORKS_FEATURE_GROUP,
                                  version=config.HOPSWORKS_FG_VERSION)
        print(f"  Found. Description: {fg.description}")
        print("  Schema:")
        for feature in fg.schema:
            print(f"    - {feature.name}: {feature.type}")
    except Exception as exc:
        fg = None
        print(f"  Not created yet ({type(exc).__name__}). This is expected on a "
              f"first run; it will be created on the first write.")

    if fg is not None:
        _step(5, "Reading rows back")
        df = fg.read()
        print(f"  Read {len(df)} rows.")
        if not df.empty and "timestamp" in df.columns:
            print(f"  Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    if not migrate:
        print("\n" + "=" * 68)
        print("Connectivity OK. Re-run with --migrate to upload the local "
              "parquet store,\nor set FEATURE_STORE_BACKEND=hopsworks to use it "
              "from here on.")
        print("=" * 68)
        return

    _step(6, "Migrating the local parquet store to Hopsworks")
    if not os.path.exists(config.FEATURES_PATH):
        raise SystemExit(f"No local store at {config.FEATURES_PATH} to migrate.")

    local = normalize_timestamps(pd.read_parquet(config.FEATURES_PATH))
    print(f"  Local store has {len(local)} rows.")

    # Import here so the module-level backend switch does not affect the read
    # above, which must come from the local file.
    original_backend = config.FEATURE_STORE_BACKEND
    config.FEATURE_STORE_BACKEND = "hopsworks"
    try:
        from utils import _save_to_hopsworks
        _save_to_hopsworks(local)
    finally:
        config.FEATURE_STORE_BACKEND = original_backend

    print("\n" + "=" * 68)
    print("Migration complete. Set FEATURE_STORE_BACKEND=hopsworks "
          "(and add\nHOPSWORKS_API_KEY as a GitHub secret) to run the pipelines "
          "against it.")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--migrate", action="store_true",
                        help="Upload the local parquet feature store to Hopsworks.")
    args = parser.parse_args()
    run(args.migrate)
