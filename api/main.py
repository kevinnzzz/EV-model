from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_ORIGINS
from .predictor import Predictor
from .schemas import BatchIn, InputData, PredictionOut

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor()   # model dimuat SEKALI saat server menyala
    yield
    state.clear()


app = FastAPI(title="EV Prediction API", version="1.0.0", lifespan=lifespan)

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
    p: Predictor = state["predictor"]
    return {
        "threshold": p.threshold,
        "weight_xgb": p.w_xgb,
        "n_models_per_type": len(p.cat_models),
        "trained_with": p.versions,
    }


def _build(p: Predictor, probs, labels):
    return [
        PredictionOut(probability=float(a), prediction=int(b), threshold=p.threshold)
        for a, b in zip(probs, labels)
    ]


@app.post("/predict", response_model=PredictionOut)
def predict(data: InputData):
    p: Predictor = state["predictor"]
    try:
        probs, labels = p.predict(pd.DataFrame([data.model_dump()]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memprediksi: {e}")
    return _build(p, probs, labels)[0]


@app.post("/predict/batch", response_model=list[PredictionOut])
def predict_batch(batch: BatchIn):
    p: Predictor = state["predictor"]
    try:
        probs, labels = p.predict(pd.DataFrame([i.model_dump() for i in batch.items]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memprediksi: {e}")
    return _build(p, probs, labels)