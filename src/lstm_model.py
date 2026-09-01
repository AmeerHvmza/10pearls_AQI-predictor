"""
TensorFlow/Keras LSTM forecaster.

The scikit-learn candidates consume one engineered row per prediction, with
history flattened into explicit lag and rolling columns. The LSTM instead
consumes a raw 48-hour sequence and learns the temporal structure itself, so
it uses a deliberately smaller feature set with no lag columns — feeding it
the engineered lags as well would just duplicate what the sequence already
contains.

Like the other candidates it predicts the CHANGE from the current AQI, not the
absolute level, so all models remain directly comparable.

TensorFlow is an optional import: `is_available()` lets the training pipeline
skip this candidate cleanly on machines where TensorFlow is not installed.
"""
import os
import numpy as np

SEQUENCE_LENGTH = 48  # hours of history fed to the network

# Raw per-hour signals only. Anything derived from a window (lags, rolling
# means, diffs) is excluded because the sequence already carries it.
SEQUENCE_FEATURE_COLS = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "temp", "humidity", "pressure", "wind_speed", "wind_deg", "clouds",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend",
]

DEFAULT_EPOCHS = int(os.getenv("AQI_LSTM_EPOCHS", "25"))
DEFAULT_BATCH_SIZE = int(os.getenv("AQI_LSTM_BATCH", "64"))


def is_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


def _keras():
    """Import Keras lazily and quietly."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.keras.utils.set_random_seed(42)
    return tf


def available_feature_cols(df) -> list:
    return [c for c in SEQUENCE_FEATURE_COLS if c in df.columns]


def build_sequence_dataset(df, horizon: int, feature_cols: list,
                           seq_len: int = SEQUENCE_LENGTH):
    """Turn the hourly grid into (windows, targets, anchors, end_positions).

    Window `i` covers rows [end-seq_len+1, end] and predicts the AQI at
    end+horizon. Only windows that are complete (no NaN anywhere in the window,
    a known target, and a known current AQI) are returned.

    `end_positions` are row offsets into `df`, so the caller can keep the
    chronological ordering that time-series cross-validation depends on.
    """
    values = df[feature_cols].to_numpy(dtype="float32")
    current = df["aqi_capped"].to_numpy(dtype="float32")
    target = df["aqi_capped"].shift(-horizon).to_numpy(dtype="float32")

    n_rows = len(df)
    row_ok = ~np.isnan(values).any(axis=1)
    # A window is usable only if every row inside it is complete. A rolling
    # sum over the per-row flags gives that in one pass.
    cumulative = np.concatenate([[0], np.cumsum(row_ok)])
    ends = np.arange(seq_len - 1, n_rows)
    window_complete = (cumulative[ends + 1] - cumulative[ends + 1 - seq_len]) == seq_len

    usable = window_complete & ~np.isnan(target[ends]) & ~np.isnan(current[ends])
    end_positions = ends[usable]
    return values, current, target, end_positions


def make_windows(values: np.ndarray, end_positions: np.ndarray,
                 seq_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Materialise the 3D (samples, timesteps, features) array."""
    offsets = np.arange(-seq_len + 1, 1)
    index = end_positions[:, None] + offsets[None, :]
    return values[index]


class LSTMForecaster:
    """Keras LSTM with a scikit-learn-shaped fit/predict surface."""

    def __init__(self, n_features: int, seq_len: int = SEQUENCE_LENGTH,
                 epochs: int = DEFAULT_EPOCHS,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        self.n_features = n_features
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None

    def _build(self):
        tf = _keras()
        layers = tf.keras.layers
        model = tf.keras.Sequential([
            layers.Input(shape=(self.seq_len, self.n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="huber",  # less sensitive to AQI spikes than MSE
            metrics=["mae"],
        )
        return model

    def fit(self, X, y, verbose: int = 0):
        tf = _keras()
        self.model = self._build()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        ]
        # Validation split must be the TAIL of the training data, not a random
        # sample, or the model validates against hours it trained on.
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            shuffle=False,
            callbacks=callbacks,
            verbose=verbose,
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(X, verbose=0).ravel()

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str, n_features: int, seq_len: int = SEQUENCE_LENGTH):
        tf = _keras()
        obj = cls(n_features=n_features, seq_len=seq_len)
        obj.model = tf.keras.models.load_model(path, compile=False)
        return obj
