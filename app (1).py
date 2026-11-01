import streamlit as st
import joblib
import os
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Kwezi Finance AI", page_icon="🏦", layout="wide")


# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    if not os.path.exists('nlp_triage_model.pkl'):
        st.error("Model file not found. Please run train.py first.")
        return None
    return joblib.load('nlp_triage_model.pkl')


model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.title("Kwezi Finance")
    st.markdown("### AI Customer Triage System")
    st.info(
        "This system uses Natural Language Processing (NLP) to automatically categorize and prioritize customer support queries.")
    st.markdown("---")
    st.markdown("**Categories:**")
    st.markdown("- 🚨 Fraud/Security")
    st.markdown("- 💳 Billing/Fees")
    st.markdown("- 🔒 Account Access")
    st.markdown("- ℹ️ General Inquiry")

# --- MAIN INTERFACE ---
st.title("🏦 Customer Support Triage Portal")
st.markdown("Paste a customer's message below to automatically route it to the correct department.")

# Input area
user_input = st.text_area("Customer Message:", height=150,
                          placeholder="e.g., I just got an SMS saying R5000 was withdrawn but I didn't do it!")

if st.button("Analyze & Route Message", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter a customer message first.")
    elif model is not None:
        # 1. AI Prediction
        prediction = model.predict([user_input])[0]

        # 2. Confidence Score (Fixing the missing logic)
        probabilities = model.predict_proba([user_input])[0]
        confidence = max(probabilities) * 100

        # 3. Determine Priority and Softbot Reply
        if prediction == "Fraud":
            priority = "HIGH 🔴"
            reply = "🚨 We have identified this as a Fraud issue. A security agent is calling you now."
            color = "red"
        elif prediction == "Billing":
            priority = "MEDIUM 🟡"
            reply = "💳 I have routed this to our Billing Department. They will review your fees shortly."
            color = "orange"
        elif prediction == "Account Access":
            priority = "MEDIUM 🟡"
            reply = "🔒 I have flagged this for our Account Security team to unlock your profile."
            color = "orange"
        else:
            priority = "LOW 🟢"
            reply = "ℹ️ Thank you. A general consultant will respond to your query soon."
            color = "green"

        # 4. Confidence Score Safety Net (From your documentation)
        if confidence < 80.0:
            priority = "HUMAN REVIEW ⚠️"
            reply = "Our AI is not entirely sure about this query. It has been escalated to a human supervisor for review."
            color = "blue"

        # --- DISPLAY RESULTS ---
        st.divider()
        st.subheader("AI Analysis Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Detected Category", prediction)
        col2.metric("Priority Level", priority)
        col3.metric("AI Confidence", f"{confidence:.2f}%")

        st.markdown(f"**Automated Softbot Reply:**")
        st.info(reply)

        # Show technical details
        with st.expander("View Technical Details (For IT Staff)"):
            st.write(f"**Original Message:** {user_input}")
            st.write(f"**Raw Probability Distribution:** {dict(zip(model.classes_, probabilities))}")