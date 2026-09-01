"""
Regenerate explainability plots from the model registry, without retraining.

`train_pipeline.py` writes these automatically at the end of each run. This
script is for the case where the plots are missing or stale but the models are
current — for example when a training run happened on a machine where `shap`
could not be imported.

Run:
    python src/explain.py            # all horizons
    python src/explain.py --horizon 24
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import load_feature_store, to_hourly_grid
from train_pipeline import engineer_features, run_shap_explanation
from predict import load_model


def run(horizons):
    df = load_feature_store()
    if df.empty:
        raise SystemExit("Feature store is empty.")
    df = engineer_features(to_hourly_grid(df))

    for horizon in horizons:
        bundle, load_error = load_model(horizon)
        if bundle is None:
            print(f"[{horizon}h] {load_error} -- skipping.")
            continue

        print(f"[{horizon}h] explaining {bundle.get('model_name')} "
              f"({'sequence' if bundle.get('is_sequence') else 'tabular'} model)")
        # run_shap_explanation expects the shape train_pipeline builds, so
        # adapt the saved bundle to it.
        result = {
            "horizon": horizon,
            "model": bundle["model"],
            "scaler": bundle["scaler"],
            "feature_cols": bundle["feature_cols"],
            "is_sequence": bundle.get("is_sequence", False),
            "seq_len": bundle.get("seq_len"),
        }
        run_shap_explanation(result, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=None,
                        help="Only regenerate this horizon (default: all).")
    args = parser.parse_args()
    run([args.horizon] if args.horizon else config.HORIZONS)
