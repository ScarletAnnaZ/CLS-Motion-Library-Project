# === test file for lightgbm（Not accepted）===

import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILE = os.path.join(BASE_DIR, 'output', 'features', 'full_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')

# load feature data
df = pd.read_csv(FEATURE_FILE)
X = df.drop(columns=["Label"]).values
y = df["Label"].values

# Feature standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# save scaler
'''
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(scaler, scaler_path)
'''

# Divide the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# save the model
joblib.dump(model, MODEL_FILE)
print(f"✅ LightGBM model saved to {MODEL_FILE}")

# Evaluation model 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# confusion matrix
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix - LightGBM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
