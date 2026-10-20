Kwezi Finance: AI-Driven Customer Support & Fraud Triage System
Module: AI for Business Analysis with Python (Web Development 3.2)
Institution: Vaal University of Technology (VUT)
Due Date: 02 November 2026

 Project Overview
Kwezi Finance is an AI-powered triage system designed for the finance sector. Retail banks receive thousands of customer queries daily, and manually sorting them is slow and costly. This Python-based solution uses Natural Language Processing (NLP) to automatically read incoming customer messages, categorize them (Fraud, Billing, Account Access, or General), and assign a priority level so that urgent fraud reports are escalated to human agents instantly.

 How it Works
Data Ingestion: A simulated dataset of customer messages (customer_queries.csv) is used.

NLP Processing: The text is cleaned and converted into numerical data using TF-IDF (Term Frequency-Inverse Document Frequency).

Machine Learning Model: A Multinomial Naive Bayes classifier (via scikit-learn) is trained to predict the category of the message.

API Deployment: A FastAPI server is deployed. When a message is sent to the /classify endpoint, the AI predicts the category and generates a "Softbot" reply for the customer.

 Repository Structure
train.py: The Python script that generates the dataset, preprocesses the text, trains the Machine Learning model, and saves the model as nlp_triage_model.pkl.

app.py: The FastAPI backend that serves the trained model. It listens for incoming text and returns the AI's classification, priority level, and a Softbot reply.

nlp_triage_model.pkl: The trained machine learning model (generated after running train.py).

customer_queries.csv: The dataset used to train the AI.

 How to Run the Project
Install the required dependencies:

bash
pip install pandas numpy scikit-learn joblib fastapi uvicorn pydantic
Train the model:

bash
python train.py
Start the API server:

bash
uvicorn app:app --reload
Open your browser and go to http://127.0.0.1:8000/docs to test the AI via the interactive Swagger UI.

👥 Group Members
Manqoba Vakala
