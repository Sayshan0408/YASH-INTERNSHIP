# AI Classifier - DeepSource Demo Project
# Dataset Source: UCI Machine Learning Repository (Official)
# URL: https://archive.ics.uci.edu/ml/datasets/iris
# The Iris dataset is one of the most famous datasets in AI/ML history
# It was introduced by statistician Ronald Fisher in 1936

import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------------------
# BAD PRACTICE 1: Hardcoded secret keys (Secrets Analyzer catches)
# ----------------------------------------------------------------
API_KEY = "sk-1234567890abcdef1234567890abcdef"
DB_PASSWORD = "admin@123"
GITHUB_TOKEN = "ghp_abcdefghijklmnop1234567890ABCDEF"

# ----------------------------------------------------------------
# BAD PRACTICE 2: Unused imports
# ----------------------------------------------------------------
import random
import datetime
import sys

# ----------------------------------------------------------------
# Load Real Dataset from UCI (via sklearn which hosts UCI datasets)
# Source: https://archive.ics.uci.edu/ml/datasets/iris
# ----------------------------------------------------------------
# Dataset Info:
# - 150 samples of iris flowers
# - 3 classes: Setosa, Versicolor, Virginica
# - 4 features: sepal length, sepal width, petal length, petal width
# - Collected by: R.A. Fisher, 1936
# ----------------------------------------------------------------

# BAD PRACTICE 3: No docstring on function
def load_dataset():
    iris = load_iris()
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df["species"] = iris.target
    df["species_name"] = df["species"].map({
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    })
    return df, iris

# BAD PRACTICE 4: Bare except clause (Anti-pattern)
def preprocess_data(df, iris):
    try:
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    except:
        print("Error in preprocessing")
        return None, None, None, None

# BAD PRACTICE 5: Mutable default argument (Bug Risk)
def store_results(result, history=[]):
    history.append(result)
    return history

# BAD PRACTICE 6: Using == to compare with None
def train_model(X_train, y_train):
    if X_train == None:
        return None
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# BAD PRACTICE 7: Division without zero check (Bug Risk)
def calculate_error_rate(correct, total):
    return (total - correct) / total

# BAD PRACTICE 8: Shadowing built-in names
def get_summary(list, input):
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)

# BAD PRACTICE 9: SQL Injection risk (Security)
def fetch_model_from_db(model_name):
    query = "SELECT * FROM models WHERE name = '" + model_name + "'"
    return query

# BAD PRACTICE 10: Print instead of logging
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Model Accuracy: " + str(round(accuracy * 100, 2)) + "%")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Setosa", "Versicolor", "Virginica"]
    ))
    return accuracy, predictions

def predict_flower(model, sepal_length, sepal_width, petal_length, petal_width):
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data).max()
    return species_map[prediction[0]], round(confidence * 100, 2)

if __name__ == "__main__":
    print("=" * 60)
    print("   IRIS FLOWER CLASSIFIER - DeepSource Demo")
    print("   Dataset: UCI Machine Learning Repository")
    print("   URL: https://archive.ics.uci.edu/ml/datasets/iris")
    print("=" * 60)

    # Load dataset
    df, iris = load_dataset()
    print(f"\nDataset loaded: {len(df)} samples, {len(df.columns)} columns")
    print(f"Classes: {df['species_name'].unique()}")
    print("\nFirst 5 rows of dataset:")
    print(df.head())

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df, iris)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # Train
    model = train_model(X_train, y_train)
    print("\nModel trained successfully!")

    # Evaluate
    accuracy, predictions = evaluate_model(model, X_test, y_test)

    # Demo predictions
    print("\n--- Sample Predictions ---")
    samples = [
        (5.1, 3.5, 1.4, 0.2, "Expected: Setosa"),
        (6.3, 3.3, 4.7, 1.6, "Expected: Versicolor"),
        (7.2, 3.6, 6.1, 2.5, "Expected: Virginica"),
    ]

    for sl, sw, pl, pw, expected in samples:
        flower, confidence = predict_flower(model, sl, sw, pl, pw)
        print(f"Input: [{sl}, {sw}, {pl}, {pw}]")
        print(f"Predicted: {flower} ({confidence}% confidence) | {expected}")
        print("-" * 40)
