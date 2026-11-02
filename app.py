from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "nlp_triage_model.pkl"

st.set_page_config(page_title="Kwezi Finance AI", page_icon="🏦", layout="centered")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Kwezi Finance")
st.caption("AI-assisted customer query triage · Student prototype")
st.write("Classify a sample banking query and see the model's suggested priority and support queue.")
st.info("Use sample text only. Do not enter real customer names, account numbers, card details, or other private information.")

if model is None:
    st.error("Model not found. Run python train.py from the project folder, then restart this app.")
    st.stop()

message = st.text_area(
    "Customer message",
    height=130,
    max_chars=5000,
    placeholder="Example: I see a transaction on my account that I did not make.",
)

if st.button("Analyse message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Enter a message to classify.")
    else:
        try:
            predicted = str(model.predict([message.strip()])[0])
            scores = model.predict_proba([message.strip()])[0]
            confidence = float(max(scores)) * 100
        except Exception:
            st.error("The model could not classify this message.")
            st.stop()

        if confidence < 80.0:
            priority = "HUMAN REVIEW"
            queue = "Human Supervisor (recommended)"
            reply = "This query needs review by a Kwezi Finance team member."
        elif predicted == "Fraud":
            priority, queue = "HIGH", "Security Team (recommended)"
            reply = "This query has been flagged for urgent security review."
        elif predicted == "Billing":
            priority, queue = "MEDIUM", "Billing Team (recommended)"
            reply = "This query has been classified for billing review."
        elif predicted == "Account Access":
            priority, queue = "MEDIUM", "Account Support (recommended)"
            reply = "This query has been classified for account support."
        else:
            priority, queue = "LOW", "Customer Care (recommended)"
            reply = "This query has been classified for customer care."

        st.subheader("Triage result")
        col1, col2, col3 = st.columns(3)
        col1.metric("Category", predicted)
        col2.metric("Priority", priority)
        col3.metric("Model score", f"{confidence:.2f}%")
        st.write(f"**Suggested queue:** {queue}")
        st.success(reply)
        st.caption("This is a recommendation only. The prototype does not create a ticket or contact a team.")
        with st.expander("Technical details"):
            st.write({str(label): f"{float(score) * 100:.2f}%" for label, score in zip(model.classes_, scores)})
