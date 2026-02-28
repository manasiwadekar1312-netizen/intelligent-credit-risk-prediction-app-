# ===============================
# CREDIT RISK PREDICTION PROJECT
# ===============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# STEP 1: LOAD DATA
# ===============================

df = pd.read_csv("loan_data.csv")

# Clean column names (very important)
df.columns = df.columns.str.strip()

print("Columns in dataset:")
print(df.columns)

# ===============================
# STEP 2: DROP UNNECESSARY COLUMN
# ===============================

if "loan_id" in df.columns:
    df = df.drop("loan_id", axis=1)

# ===============================
# STEP 3: FEATURE ENGINEERING
# ===============================

# Create income to loan ratio
df["income_loan_ratio"] = df["income_annum"] / (df["loan_amount"] + 1)

# ===============================
# STEP 4: ENCODE CATEGORICAL DATA
# ===============================

le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = le.fit_transform(df[column])

# ===============================
# STEP 5: DEFINE FEATURES & TARGET
# ===============================

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# ===============================
# STEP 6: TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# STEP 7: FEATURE SCALING
# ===============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# STEP 8: MODEL TRAINING
# ===============================

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

# ===============================
# STEP 9: MODEL EVALUATION
# ===============================

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# STEP 10: SAVE MODEL
# ===============================

joblib.dump(model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and Scaler saved successfully.")
