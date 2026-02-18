# ========================================
# CREDIT CARD FRAUD DETECTION - IMBALANCE HANDLING
# Dataset: credit_card_fraud_dataset.csv
# ========================================

# ----------------------------
# 0. Import Libraries
# ----------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# 1. Load Dataset
# ----------------------------
data = pd.read_csv("credit_card_fraud_dataset.csv")

print("Dataset Shape:", data.shape)
print(data.head())


# ----------------------------
# 2. Drop Date Column (Not Numeric)
# ----------------------------
data = data.drop("TransactionDate", axis=1)


# ----------------------------
# 3. Encode Categorical Columns
# ----------------------------
le = LabelEncoder()

data["TransactionType"] = le.fit_transform(data["TransactionType"])
data["Location"] = le.fit_transform(data["Location"])


# ----------------------------
# 4. Check Class Distribution
# ----------------------------
print("\nFraud Class Distribution:")
print(data["IsFraud"].value_counts())

# Visual 1: Class Distribution
plt.figure()
sns.countplot(x="IsFraud", data=data)
plt.title("Class Distribution (0 = Legitimate, 1 = Fraud)")
plt.show()


# ----------------------------
# 5. Feature and Target Split
# ----------------------------
X = data.drop("IsFraud", axis=1)
y = data["IsFraud"]


# ----------------------------
# 6. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ----------------------------
# 7. Model Before Balancing
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- BEFORE SMOTE ---")
cm_before = confusion_matrix(y_test, y_pred)
print(cm_before)
print(classification_report(y_test, y_pred))

# Visual 2: Confusion Matrix (Before SMOTE)
plt.figure()
sns.heatmap(cm_before, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Before SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ----------------------------
# 8. Apply SMOTE
# ----------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE Distribution:")
print(pd.Series(y_resampled).value_counts())

# Visual 3: Distribution After SMOTE
plt.figure()
sns.countplot(x=y_resampled)
plt.title("Balanced Distribution After SMOTE")
plt.show()


# ----------------------------
# 9. Model After Balancing
# ----------------------------
model_smote = LogisticRegression(max_iter=1000)
model_smote.fit(X_resampled, y_resampled)

y_pred_smote = model_smote.predict(X_test)

print("\n--- AFTER SMOTE ---")
cm_after = confusion_matrix(y_test, y_pred_smote)
print(cm_after)
print(classification_report(y_test, y_pred_smote))

# Visual 4: Confusion Matrix (After SMOTE)
plt.figure()
sns.heatmap(cm_after, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - After SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ----------------------------
# 10. Pie Chart Visualization
# ----------------------------
labels = ["Legitimate", "Fraud"]
sizes = data["IsFraud"].value_counts()

plt.figure()
plt.pie(sizes, labels=labels, autopct="%1.1f%%")
plt.title("Fraud vs Legitimate Transactions")
plt.show()


# ----------------------------
# 11. Feature Importance
# ----------------------------
importance = model_smote.coef_[0]

plt.figure()
plt.bar(X.columns, importance)
plt.xticks(rotation=45)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.show()
