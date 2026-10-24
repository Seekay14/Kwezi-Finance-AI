from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Kwezi Finance NLP API", description="AI Customer Triage")

# Load the trained NLP model
if not os.path.exists('nlp_triage_model.pkl'):
    raise FileNotFoundError("Model file not found. Please run train.py first.")

model = joblib.load('nlp_triage_model.pkl')


# Define the expected input structure
class CustomerMessage(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Kwezi Finance Triage API. Go to /docs to test."}


@app.post("/classify")
def classify_message(message: CustomerMessage):
    try:
        # Make prediction
        prediction = model.predict([message.text])[0]

        # Determine Priority based on Category
        if prediction == "Fraud":
            priority = "HIGH"
            reply = "🚨 We have identified this as a Fraud issue. A security agent is calling you now."
        elif prediction == "Billing":
            priority = "MEDIUM"
            reply = "💳 I have routed this to our Billing Department. They will review your fees shortly."
        elif prediction == "Account Access":
            priority = "MEDIUM"
            reply = "🔒 I have flagged this for our Account Security team to unlock your profile."
        else:
            priority = "LOW"
            reply = "ℹ️ Thank you. A general consultant will respond to your query soon."

        return {
            "original_message": message.text,
            "ai_category": prediction,
            "priority_level": priority,
            "softbot_reply": reply
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
