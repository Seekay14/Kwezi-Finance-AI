from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="Kwezi Finance NLP API",
    description="AI Customer Triage System",
    version="1.0.0"
)

# Load the trained NLP model
model_path = 'nlp_triage_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")

model = joblib.load(model_path)


# Define the expected input structure from the user
class CustomerMessage(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "Welcome to the Kwezi Finance Triage API. Go to /docs to test the model."
    }


@app.post("/classify")
def classify_message(message: CustomerMessage):
    try:
        # 1. Get the AI prediction
        prediction = model.predict([message.text])[0]

        # 2. Calculate Confidence Score
        probabilities = model.predict_proba([message.text])[0]
        confidence = max(probabilities) * 100

        # 3. Human-in-the-Loop Rule (From Section 4 of your PDF)
        # If the AI is less than 80% sure, route to a human supervisor
        if confidence < 80.0:
            return {
                "original_message": message.text,
                "ai_category": "Needs Human Review",
                "priority_level": "UNKNOWN",
                "confidence_score": f"{confidence:.2f}%",
                "softbot_reply": "⚠️ The AI is not fully confident about this query. I am forwarding this to a human supervisor to review manually."
            }

        # 4. Determine Priority and Softbot Reply (Matching Section 5 of your PDF)
        if prediction == "Fraud":
            priority = "HIGH"
            reply = "Thank you. I have detected that this is a Fraud case and have forwarded it to our Security Team. Someone will get back to you shortly."
        elif prediction == "Billing":
            priority = "MEDIUM"
            reply = "Thank you. I have detected that this is a Billing case and have forwarded it to our Finance Team. Someone will get back to you shortly."
        elif prediction == "Account Access":
            priority = "MEDIUM"
            reply = "Thank you. I have detected that this is an Account Access case and have forwarded it to our Support Team. Someone will get back to you shortly."
        else:
            priority = "LOW"
            reply = "Thank you. I have detected that this is a General inquiry and have forwarded it to our Customer Care Team. Someone will get back to you shortly."

        # 5. Return the successful response
        return {
            "original_message": message.text,
            "ai_category": prediction,
            "priority_level": priority,
            "confidence_score": f"{confidence:.2f}%",
            "softbot_reply": reply
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))