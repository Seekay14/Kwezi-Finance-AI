# app.py — Kwezi Finance FastAPI Backend
# Run with:  python -m uvicorn app:app --reload

import os
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(
    title="Kwezi Finance AI Triage API",
    description="NLP-based customer support & fraud triage system",
    version="1.0.0",
)

# --- LOAD MODEL ON STARTUP ---
MODEL_PATH = "nlp_triage_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"   Classes: {list(model.classes_)}")
else:
    print(f"⚠️  Model not found at {MODEL_PATH}. Run train.py first.")


# --- REQUEST SCHEMA ---
class Query(BaseModel):
    text: str


# --- SERVE THE FRONTEND AT "/" ---
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if not os.path.exists("index.html"):
        return "<h1>index.html not found</h1><p>Place index.html in the same folder as app.py.</p>"
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# --- HEALTH CHECK ---
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


# --- CLASSIFY ENDPOINT ---
@app.post("/classify")
async def classify(query: Query):
    if model is None:
        return {"error": "Model not loaded. Please run train.py first."}

    text = query.text.strip()
    if not text:
        return {"error": "Empty message."}

    # 1. AI Prediction
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities) * 100)

    # 2. Priority + softbot reply
    if prediction == "Fraud":
        priority = "HIGH 🔴"
        priority_class = "HIGH"
        reply = "🚨 We have identified this as a Fraud issue. A security agent is calling you now."
    elif prediction == "Billing":
        priority = "MEDIUM 🟡"
        priority_class = "MEDIUM"
        reply = "💳 I have routed this to our Billing Department. They will review your fees shortly."
    elif prediction == "Account Access":
        priority = "MEDIUM 🟡"
        priority_class = "MEDIUM"
        reply = "🔒 I have flagged this for our Account Security team to unlock your profile."
    else:
        priority = "LOW 🟢"
        priority_class = "LOW"
        reply = "ℹ️ Thank you. A general consultant will respond to your query soon."

    # 3. Confidence safety net (< 80% → human review, per documentation)
    if confidence < 80.0:
        priority = "HUMAN REVIEW ⚠️"
        priority_class = "UNKNOWN"
        reply = (
            "Our AI is not entirely sure about this query. "
            "It has been escalated to a human supervisor for review."
        )

    # 4. Response (keys match what index.html expects)
    return {
        "ai_category": prediction,
        "priority_level": priority,
        "priority_class": priority_class,
        "confidence_score": f"{confidence:.2f}%",
        "softbot_reply": reply,
        "raw_probabilities": {
            cls: round(float(prob), 4)
            for cls, prob in zip(model.classes_.tolist(), probabilities.tolist())
        },
    }
