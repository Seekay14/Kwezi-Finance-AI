import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("--- 1. Generating Simulated Customer Messages ---")

# We create a dataset of typical bank customer queries
data = {
    'Message': [
        "I just got an SMS saying R5000 was withdrawn but I didn't do it!",
        "Someone stole my bank card at the ATM, please block it.",
        "There is a suspicious login attempt on my account from another country.",
        "I lost my phone and my banking app is on it, I need to block my profile.",
        "Why was I charged a monthly service fee? I thought my account was free.",
        "I need a copy of my bank statement for the last 3 months for a loan application.",
        "The ATM ate my card and didn't give me my money.",
        "How do I update my cellphone number on the app?",
        "I am trying to log in but it says my password is incorrect, please help.",
        "My account is locked because I entered the wrong PIN too many times.",
        "What are the trading hours for the branch in Vanderbijlpark?",
        "Can I open a savings account for my child online?",
        "I need to speak to a consultant about a home loan.",
        "There is an unauthorized debit order on my account, I want it reversed immediately.",
        "Someone is using my identity to open accounts, this is fraud!",
        "Please unblock my internet banking, I am locked out.",
        "Why did the ATM charge me R50 for a withdrawal?",
        "How do I get a new SIM card linked to my bank account?",
        "I want to cancel my credit card.",
        "A hacker just changed my banking password, help!"
    ],
    # 0: Fraud/Security, 1: Billing/Fees, 2: Account Access/Lost Card, 3: General Inquiry
    'Category': [
        "Fraud", "Fraud", "Fraud", "Fraud",
        "Billing", "Billing", "Billing",
        "Account Access", "Account Access", "Account Access", "Account Access",
        "General", "General",
        "Fraud", "Fraud", "Account Access",
        "Billing",
        "Account Access",
        "General",
        "Fraud"
    ]
}

df = pd.DataFrame(data)

# Save this dataset to CSV so you can show it in your documentation
df.to_csv('customer_queries.csv', index=False)
print(f"Dataset created with {len(df)} sample messages.")

print("\n--- 2. Training the NLP Model ---")

# Split data into Features (Text) and Target (Category)
X = df['Message']
y = df['Category']

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We use a Pipeline: It automatically converts text to numbers (TF-IDF), then trains the model.
# Multinomial Naive Bayes is perfect for text classification.
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')), # Removes common words like "the", "is"
    ('classifier', MultinomialNB()) # The AI model
])

# Train the model
model_pipeline.fit(X_train, y_train)

print("\n--- 3. Model Evaluation ---")
# Test the model on unseen data
y_pred = model_pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained pipeline model
joblib.dump(model_pipeline, 'nlp_triage_model.pkl')
print("\nSuccess! Model saved as 'nlp_triage_model.pkl'")
