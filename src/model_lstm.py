"""
Demand model: Long Short Term Memory network (Chapter 3.4.1).

TensorFlow / Keras, per 3.4. Household demand is sequential with strong daily
and weekly periodicity tied to human routine, and the LSTM is designed to learn
that temporal order (Hochreiter & Schmidhuber, 1997; Kong et al., 2019).

Scalers are fitted on the TRAINING SPLIT ONLY and then applied. Fitting on the
full series before splitting would leak future statistics backwards.
"""
from __future__ import annotations
import json
import numpy as np

from . import config as C


class Scaler:
    """Explicit mu/sigma so the absence of leakage is auditable."""
    def fit(self, X):
        f = X.reshape(-1, X.shape[-1]) if X.ndim == 3 else X.reshape(-1, 1)
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-8
        return self
    def transform(self, X):  return (X - self.mu) / self.sd
    def inverse(self, X):    return X * self.sd + self.mu
    def fit_transform(self, X): return self.fit(X).transform(X)


class DemandLSTM:
    def __init__(self, seed=C.SEED):
        self.seed = seed
        self.model = None
        self.xs, self.ys = Scaler(), Scaler()
        self.history = None

    def _build(self, n_feat):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.seed)
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(C.SEQ_LEN, n_feat)),
            tf.keras.layers.LSTM(C.LSTM_UNITS, return_sequences=True),
            tf.keras.layers.Dropout(C.LSTM_DROPOUT),
            tf.keras.layers.LSTM(C.LSTM_UNITS_2),
            tf.keras.layers.Dropout(C.LSTM_DROPOUT),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(C.LSTM_LR),
                  loss="mse", metrics=["mae"])
        return m

    def fit(self, Xtr, ytr, Xva, yva, epochs=C.LSTM_EPOCHS, verbose=0):
        import tensorflow as tf
        Xtr_s = self.xs.fit_transform(Xtr)
        Xva_s = self.xs.transform(Xva)
        ytr_s = self.ys.fit_transform(ytr.reshape(-1, 1)).ravel()
        yva_s = self.ys.transform(yva.reshape(-1, 1)).ravel()
        self.model = self._build(Xtr.shape[-1])
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                patience=C.LSTM_PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                patience=3, factor=0.5, min_lr=1e-5),
        ]
        h = self.model.fit(Xtr_s, ytr_s, validation_data=(Xva_s, yva_s),
                           epochs=epochs, batch_size=C.LSTM_BATCH,
                           callbacks=cbs, verbose=verbose, shuffle=False)
        self.history = {k: [float(x) for x in v] for k, v in h.history.items()}
        return self

    def predict(self, X):
        p = self.model.predict(self.xs.transform(X), verbose=0).ravel()
        return self.ys.inverse(p.reshape(-1, 1)).ravel()

    def save(self, name="lstm_demand"):
        self.model.save(C.MODELS / f"{name}.keras")
        (C.MODELS / f"{name}_scalers.json").write_text(json.dumps({
            "x_mu": self.xs.mu.tolist(), "x_sd": self.xs.sd.tolist(),
            "y_mu": self.ys.mu.tolist(), "y_sd": self.ys.sd.tolist()}))
