from __future__ import annotations

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from ..models.bert_classifier import BertClassifier, ModelConfig
from ..training.train import build_tokenizer

app = FastAPI(title="bert-ts-classifier")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label_id: int
    probabilities: list[float]


@app.get("/health")
async def health():  # pragma: no cover
    return {"status": "ok"}


@app.on_event("startup")
async def load_model():  # pragma: no cover
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    mcfg = ModelConfig()
    app.state.model = BertClassifier(mcfg).to(device).eval()
    app.state.tokenizer_fn = build_tokenizer(mcfg.model_name)
    app.state.device = device


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):  # pragma: no cover
    toks = app.state.tokenizer_fn([req.text], max_len=256)
    for k in toks:
        toks[k] = toks[k].to(app.state.device)
    with torch.inference_mode():
        logits = app.state.model(**toks)["logits"].softmax(-1).cpu()[0]
    label_id = int(logits.argmax().item())
    return PredictResponse(label_id=label_id, probabilities=[float(x) for x in logits.tolist()])
