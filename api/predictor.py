import json

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

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

        self.feat_cat = cfg["features"]["cat"]
        self.feat_xgb = cfg["features"]["xgb"]
        self.w_xgb = cfg["weight_xgb"]
        self.threshold = cfg["threshold"]
        self.scale = cfg["prior_scale"]
        self.versions = cfg.get("versions", {})

        n = cfg["n_seeds"]

        self.cat_models = []
        for i in range(n):
            m = CatBoostClassifier()
            m.load_model(str(MODEL_DIR / f"cat_{i}.cbm"))
            self.cat_models.append(m)

        self.xgb_models = []
        for i in range(n):
            m = xgb.XGBClassifier()
            m.load_model(str(MODEL_DIR / f"xgb_{i}.ubj"))
            self.xgb_models.append(m)

        idx = self.cat_models[0].get_cat_feature_indices()
        self.cat_cols = [self.feat_cat[i] for i in idx]

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        X = make_features(raw, self.freq_maps)

        # CatBoost: 35 fitur, kolom kategorikal harus berupa teks
        Xc = X[self.feat_cat].copy()
        for c in self.cat_cols:
            Xc[c] = Xc[c].astype(str)
        p_cat = np.mean([m.predict_proba(Xc)[:, 1] for m in self.cat_models], axis=0)

        # XGBoost: hanya 16 fitur terpilih
        Xx = X[self.feat_xgb].copy()
        # SESUAIKAN dengan cara training. Cek dulu: booster.feature_types
        Xx["City_Type"] = Xx["City_Type"].astype("category")
        p_xgb = np.mean([m.predict_proba(Xx)[:, 1] for m in self.xgb_models], axis=0)

        p_cat = koreksi_prior(p_cat, self.scale["cat"])
        p_xgb = koreksi_prior(p_xgb, self.scale["xgb"])

        return self.w_xgb * p_xgb + (1 - self.w_xgb) * p_cat

    def predict(self, raw: pd.DataFrame):
        p = self.predict_proba(raw)
        return p, (p >= self.threshold).astype(int)