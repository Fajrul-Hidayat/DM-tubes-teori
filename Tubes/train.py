from ucimlrepo import fetch_ucirepo
  
# fetch dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
  
# data (as pandas dataframes) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 
  
# metadata 
print(rice_cammeo_and_osmancik.metadata) 
  
# variable information 
print(rice_cammeo_and_osmancik.variables) 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# ===============================
# DATA PREPARATION
# ===============================
df = pd.concat([X, y], axis=1)
df.head()

X = df.drop(columns=['Class'])

# ⬇⬇⬇ LABEL STRING (UNTUK SVM)
y_str = df['Class']


# ===============================
# SPLIT DATA (SATU KALI SAJA)
# ===============================
X_train, X_test, y_train_str, y_test_str = train_test_split(
    X, y_str,
    test_size=0.2,
    random_state=42,
    stratify=y_str
)


# ===============================
# SVM (TANPA ENCODE LABEL)
# ===============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42
)

svm_model.fit(X_train_scaled, y_train_str)
y_pred_svm = svm_model.predict(X_test_scaled)

print("=== SVM Classification Report ===")
print(classification_report(y_test_str, y_pred_svm))
print("Accuracy:", accuracy_score(y_test_str, y_pred_svm))


# ===============================
# XGBOOST (ENCODE LABEL KHUSUS)
# ===============================
label_map = {'Cammeo': 0, 'Osmancik': 1}

y_train_xgb = y_train_str.map(label_map)
y_test_xgb = y_test_str.map(label_map)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train_xgb)

y_pred_xgb = xgb_model.predict(X_test)

print("=== XGBoost Classification Report ===")
print(classification_report(y_test_xgb, y_pred_xgb))
print("Accuracy:", accuracy_score(y_test_xgb, y_pred_xgb))


# ===============================
# CONFUSION MATRIX
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(12,4))

# SVM
sns.heatmap(
    confusion_matrix(y_test_str, y_pred_svm),
    annot=True, fmt='d', cmap='Blues',
    ax=axes[0]
)
axes[0].set_title("Confusion Matrix - SVM")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# XGBoost
sns.heatmap(
    confusion_matrix(y_test_xgb, y_pred_xgb),
    annot=True, fmt='d', cmap='Greens',
    ax=axes[1]
)
axes[1].set_title("Confusion Matrix - XGBoost")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.show()


# ===============================
# SAVE MODEL
# ===============================
joblib.dump(scaler, "scaler_svm_rice.pkl")
joblib.dump(svm_model, "svm_rice_model.pkl")
joblib.dump(xgb_model, "xgboost_rice_model.pkl")
