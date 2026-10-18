
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
import joblib

DATA_PATH = "citizen-grievances/citizen-grievances.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading required columns...")
df = pd.read_csv(
    DATA_PATH,
    usecols=["Category", "Sub Category"],
    dtype=str
)

df = df.dropna(subset=["Category", "Sub Category"])
df["Category"] = df["Category"].str.strip()
df["Sub Category"] = df["Sub Category"].str.strip()
df = df[(df["Category"] != "") & (df["Sub Category"] != "")]

print(f"Usable records: {len(df):,}")
print(f"Categories: {df['Category'].nunique()}")

# Save category distribution for the dashboard
counts = df["Category"].value_counts()
counts.rename_axis("Category").reset_index(name="Count").to_csv(
    f"{OUTPUT_DIR}/category_distribution.csv", index=False
)

# Plot category distribution
plt.figure(figsize=(12, 8))
counts.sort_values().plot(kind="barh")
plt.title("BBMP Citizen Grievances by Category")
plt.xlabel("Number of grievances")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/category_distribution.png", dpi=150)
plt.close()

# Predict Category from Sub Category text
X = df["Sub Category"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train):,}")
print(f"Test records: {len(X_test):,}")
print("Training TF-IDF + Logistic Regression...")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

print("\n=== MODEL RESULTS ===")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Macro F1:   {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

report = classification_report(
    y_test, predictions, output_dict=True, zero_division=0
)

with open(f"{OUTPUT_DIR}/classification_report.json", "w") as file:
    json.dump(report, file, indent=2)

with open(f"{OUTPUT_DIR}/metrics.json", "w") as file:
    json.dump({
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_records": len(X_train),
        "test_records": len(X_test),
        "categories": int(df["Category"].nunique())
    }, file, indent=2)

# Confusion matrix for the most common 12 categories
top_categories = counts.head(12).index.tolist()
mask = y_test.isin(top_categories)

fig, ax = plt.subplots(figsize=(13, 11))
ConfusionMatrixDisplay.from_predictions(
    y_test[mask],
    model.predict(X_test[mask]),
    labels=top_categories,
    xticks_rotation=90,
    normalize="true",
    values_format=".2f",
    cmap="Blues",
    ax=ax
)
ax.set_title("Normalized Confusion Matrix — Top 12 Categories")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

joblib.dump(model, f"{MODEL_DIR}/grievance_classifier.joblib")

print("\nSaved model: models/grievance_classifier.joblib")
print("Saved charts and metrics in outputs/")
print("\nSample predictions:")
for text in [
    "Street Light Not Working",
    "Garbage vehicle not arrived",
    "Potholes",
    "Stray dog related complaints"
]:
    print(f"{text} -> {model.predict([text])[0]}")

print("\nTraining complete.")
