import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("--- 1. Loading customer_queries.csv ---")
df = pd.read_csv("customer_queries.csv")
print(f"Loaded {len(df)} messages.")
print(df["Category"].value_counts())

print("\n--- 2. Training the NLP Model ---")
X, y = df["Message"], df["Category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("classifier", MultinomialNB(alpha=0.1)),
])
model_pipeline.fit(X_train, y_train)

print("\n--- 3. Model Evaluation ---")
y_pred = model_pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Retrain on ALL data before saving so the final model uses every example
model_pipeline.fit(X, y)
joblib.dump(model_pipeline, "nlp_triage_model.pkl")
print("\nSuccess! Model saved as 'nlp_triage_model.pkl'")
