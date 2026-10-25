import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_ORIGINS
from .predictor import Predictor
from .schemas import BatchIn, InputData, PredictionOut

state: dict = {}


def get_predictor() -> Predictor:
    """Muat model saat pertama kali dibutuhkan, lalu simpan untuk request berikutnya."""
    if "predictor" not in state:
        try:
            state["predictor"] = Predictor()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model gagal dimuat: {e}")
    return state["predictor"]


app = FastAPI(title="EV Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    p = get_predictor()
    return {
        "threshold": p.threshold,
        "n_models": len(p.xgb_models),
        "trained_with": p.versions,
    }


def _build(p: Predictor, probs, labels):
    return [
        PredictionOut(probability=float(a), prediction=int(b), threshold=p.threshold)
        for a, b in zip(probs, labels)
    ]


@app.post("/predict", response_model=PredictionOut)
def predict(data: InputData):
    p = get_predictor()
    try:
        probs, labels = p.predict(pd.DataFrame([data.model_dump()]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memprediksi: {e}")
    return _build(p, probs, labels)[0]


@app.post("/predict/batch", response_model=list[PredictionOut])
def predict_batch(batch: BatchIn):
    p = get_predictor()
    try:
        probs, labels = p.predict(pd.DataFrame([i.model_dump() for i in batch.items]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memprediksi: {e}")
    return _build(p, probs, labels)