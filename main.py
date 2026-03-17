from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

MODEL_DIR = "specialty_model_v2"
CONF_THRESH = 0.4

app = FastAPI(title="Medical Specialty Recommendation API")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

class RecommendRequest(BaseModel):
    text: str = Field(..., min_length=3)
    k: int = Field(1, ge=1, le=10)

def topk(text: str, k: int = 3):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=-1).squeeze(0)

    k = min(k, probs.shape[0])
    values, indices = torch.topk(probs, k=k)

    preds = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        label = model.config.id2label.get(idx, str(idx))
        preds.append((label, float(score)))
    return preds

def recommend(text: str, k: int = 1):
    predictions = topk(text, k)
    top1_label, top1_score = predictions[0]

    if top1_score < CONF_THRESH:
        return {
            "status": "uncertain",
            "message": "Please describe your symptoms more clearly (e.g., pain, fever, cough, rash).",
            "top": [{"label": top1_label, "confidence": top1_score}],
            "threshold": CONF_THRESH,
        }

    return {
        "status": "ok",
        "top": [{"label": l, "confidence": s} for (l, s) in predictions],
        "threshold": CONF_THRESH,
    }

@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    return {
        "input": req.text,
        **recommend(req.text, req.k),
    }