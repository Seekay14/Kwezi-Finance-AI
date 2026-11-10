# train.py
# Train the Kwezi Finance model using customer_queries.csv

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_FOLDER = Path(__file__).resolve().parent
CSV_PATH = PROJECT_FOLDER / "customer_queries.csv"
MODEL_PATH = PROJECT_FOLDER / "nlp_triage_model.pkl"

# Check that the CSV exists
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {CSV_PATH.name}. Put it in the same folder as train.py."
    )

# Read the data from the CSV
df = pd.read_csv(CSV_PATH)

# Match column names without depending on capitalization or extra spaces
column_names = {column.strip().lower(): column for column in df.columns}

if "message" not in column_names or "category" not in column_names:
    raise ValueError(
        "The CSV must have columns named Message and Category.\n"
        f"Columns found: {list(df.columns)}"
    )

# Keep and clean the two columns the model needs
df = df[[column_names["message"], column_names["category"]]].copy()
df.columns = ["Message", "Category"]

df = df.dropna(subset=["Message", "Category"])
df["Message"] = df["Message"].astype(str).str.strip()
df["Category"] = df["Category"].astype(str).str.strip()

# Remove rows with a blank message or category
df = df[(df["Message"] != "") & (df["Category"] != "")]

if df.empty:
    raise ValueError("The CSV has no usable message and category rows.")

# Make category capitalization consistent with the API
allowed_categories = {
    "fraud": "Fraud",
    "billing": "Billing",
    "account access": "Account Access",
    "general": "General",
}

df["Category"] = df["Category"].str.lower().map(allowed_categories)

if df["Category"].isna().any():
    bad_categories = df.loc[df["Category"].isna(), "Category"].tolist()
    raise ValueError(
        "The CSV contains an unsupported category. Use Fraud, Billing, "
        "Account Access, or General."
    )

# Show how many examples are available for each category
print("Messages loaded:", len(df))
print("\nExamples per category:")
print(df["Category"].value_counts())

# Stratified splitting requires at least two examples per category
category_counts = df["Category"].value_counts()
if len(category_counts) < 2 or category_counts.min() < 2:
    raise ValueError(
        "Add at least two labelled messages for every category before training."
    )

X = df["Message"]
y = df["Category"]

# Keep examples from each category in both the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Convert text into features, then train the classifier
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB()),
])

model.fit(X_train, y_train)

# Evaluate on the held-out test examples
predictions = model.predict(X_test)

print("\nModel evaluation")
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model beside this script for app.py to load
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
