import json

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import MODEL_DIR
from .preprocessing import make_features


def koreksi_prior(p: np.ndarray, skala: float) -> np.ndarray:
    """Bagi odds dengan skala bobot kelas. Ini DUGAAN, verifikasi dengan tests/compare.py."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    odds = p / (1 - p) / skala
    return odds / (1 + odds)


class Predictor:
    def __init__(self):
        with open(MODEL_DIR / "ensemble.json", encoding="utf-8") as f:
            cfg = json.load(f)

        # Kunci JSON berupa teks ("36.0"); ubah jadi angka agar cocok saat map()
        self.freq_maps = {
            col: {float(k): float(v) for k, v in tabel.items()}
            for col, tabel in cfg["freq"].items()
        }

        self.feat_xgb = cfg["features"]["xgb"]
        self.scale_xgb = cfg["prior_scale"]["xgb"]
        # Pakai threshold khusus XGBoost kalau sudah dihitung, kalau belum pakai yang lama
        self.threshold = cfg.get("threshold_xgb", cfg["threshold"])
        self.versions = cfg.get("versions", {})

        n = cfg["n_seeds"]
        self.xgb_models = []
        for i in range(n):
            m = xgb.XGBClassifier()
            m.load_model(str(MODEL_DIR / f"xgb_{i}.ubj"))
            self.xgb_models.append(m)

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        X = make_features(raw, self.freq_maps)

        Xx = X[self.feat_xgb].copy()
        # SESUAIKAN dengan cara training. Cek dulu: booster.feature_types
        Xx["City_Type"] = Xx["City_Type"].astype("category")
        p_xgb = np.mean([m.predict_proba(Xx)[:, 1] for m in self.xgb_models], axis=0)

        return koreksi_prior(p_xgb, self.scale_xgb)

    def predict(self, raw: pd.DataFrame):
        p = self.predict_proba(raw)
        return p, (p >= self.threshold).astype(int)