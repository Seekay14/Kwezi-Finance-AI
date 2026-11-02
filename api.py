from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "nlp_triage_model.pkl"
PAGE_PATH = ROOT / "index.html"

if not MODEL_PATH.exists():
    raise RuntimeError("Model file is missing. Run python train.py before starting the API.")

model = joblib.load(MODEL_PATH)
app = FastAPI(title="Kwezi Finance Triage API", version="1.0.0")

class CustomerMessage(BaseModel):
    text: str = Field(min_length=1, max_length=5000)

@app.get("/", include_in_schema=False)
def home():
    if not PAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="Website page not found.")
    return FileResponse(PAGE_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/classify")
def classify(message: CustomerMessage):
    text = message.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter a message to classify.")

    try:
        prediction = str(model.predict([text])[0])
        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities)) * 100
    except Exception:
        raise HTTPException(status_code=500, detail="The model could not classify this message.")

    if confidence < 80.0:
        priority = "HUMAN REVIEW"
        queue = "Human Supervisor (recommended)"
        reply = "This query needs review by a Kwezi Finance team member."
    elif prediction == "Fraud":
        priority = "HIGH"
        queue = "Security Team (recommended)"
        reply = "This query has been flagged for urgent security review."
    elif prediction == "Billing":
        priority = "MEDIUM"
        queue = "Billing Team (recommended)"
        reply = "This query has been classified for billing review."
    elif prediction == "Account Access":
        priority = "MEDIUM"
        queue = "Account Support (recommended)"
        reply = "This query has been classified for account support."
    else:
        priority = "LOW"
        queue = "Customer Care (recommended)"
        reply = "This query has been classified for customer care."

    return {
        "ai_category": prediction,
        "priority_level": priority,
        "recommended_queue": queue,
        "confidence_score": f"{confidence:.2f}%",
        "softbot_reply": reply,
    }

