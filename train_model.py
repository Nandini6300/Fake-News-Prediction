# train_model.py

import pandas as pd
import string
import re
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# Load Dataset
# ----------------------------
print("Loading datasets...")

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0   # Fake
true["label"] = 1   # Real

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------
# Text Preprocessing Function
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text...")

data["text"] = data["text"].apply(clean_text)

# ----------------------------
# Features and Labels
# ----------------------------
X = data["text"]
y = data["label"]

# ----------------------------
# Split Dataset
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# TF-IDF Vectorization
# ----------------------------
print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ----------------------------
# Train Model
# ----------------------------
print("Training model...")

model = PassiveAggressiveClassifier(max_iter=1000)

model.fit(X_train, y_train)

# ----------------------------
# Predictions
# ----------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("==============================\n")

print("Classification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ----------------------------
# Save Model
# ----------------------------
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("models/vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel saved successfully!")
print("Location: models/model.pkl")
print("Vectorizer saved successfully!")
print("Location: models/vectorizer.pkl")