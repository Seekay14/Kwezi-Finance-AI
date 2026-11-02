import os
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # <-- ADD THIS

app = FastAPI(title="Kwezi Finance AI Triage API")

# --- ALLOW CORS (So file:/// can connect) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any file or website
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "nlp_triage_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    print(f"⚠️  Model not found at {MODEL_PATH}. Run train.py first.")

class Query(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if not os.path.exists("index.html"):
        return "<h1>index.html not found</h1>"
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/classify")
async def classify(query: Query):
    if model is None:
        return {"error": "Model not loaded. Please run train.py first."}

    text = query.text.strip()
    if not text:
        return {"error": "Empty message."}

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities) * 100)

    if prediction == "Fraud":
        priority, priority_class = "HIGH 🔴", "HIGH"
        reply = "🚨 We have identified this as a Fraud issue. A security agent is calling you now."
    elif prediction == "Billing":
        priority, priority_class = "MEDIUM 🟡", "MEDIUM"
        reply = "💳 I have routed this to our Billing Department."
    elif prediction == "Account Access":
        priority, priority_class = "MEDIUM 🟡", "MEDIUM"
        reply = "🔒 I have flagged this for our Account Security team."
    else:
        priority, priority_class = "LOW 🟢", "LOW"
        reply = "ℹ️ Thank you. A general consultant will respond soon."

    if confidence < 80.0:
        priority, priority_class = "HUMAN REVIEW ⚠️", "UNKNOWN"
        reply = "Our AI is not entirely sure about this query. Escalated to a human supervisor."

    return {
        "ai_category": prediction,
        "priority_level": priority,
        "priority_class": priority_class,
        "confidence_score": f"{confidence:.2f}%",
        "softbot_reply": reply
    }
