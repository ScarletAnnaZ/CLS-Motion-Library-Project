import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILE = os.path.join(BASE_DIR, 'output','features', 'full_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, 'rf_model.pkl')
# SCALER_FILE = os.path.join(MODEL_DIR, 'scaler_rf.pkl')

# read data
df = pd.read_csv(FEATURE_FILE)
X = df.drop(columns=["Label"]).values
y = df["Label"].values

# feature standardization (optional)
scaler = StandardScaler()
X = scaler.fit_transform(X)
# joblib.dump(scaler, SCALER_FILE)

# split the training test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train Random Forest 
rf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf, MODEL_FILE)
print(f"✅ Random Forest model saved to {MODEL_FILE}")

# model evaluation report
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

#  confusion matrix
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix- Random Fores')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

