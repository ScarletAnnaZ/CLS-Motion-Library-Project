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

# ==== 路径设置 ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILE = os.path.join(BASE_DIR, 'output', 'features', 'full_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, 'rf_model.pkl')
# SCALER_FILE = os.path.join(MODEL_DIR, 'scaler_rf.pkl')

# ==== 读取数据 ====
df = pd.read_csv(FEATURE_FILE)
X = df.drop(columns=["Label"]).values
y = df["Label"].values

# ==== 特征标准化（可选） ====
scaler = StandardScaler()
X = scaler.fit_transform(X)
# joblib.dump(scaler, SCALER_FILE)

# ==== 拆分训练测试集 ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 训练 Random Forest ====
rf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# ==== 保存模型 ====
joblib.dump(rf, MODEL_FILE)
print(f"✅ Random Forest model saved to {MODEL_FILE}")

# ==== 模型评估 ====
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ==== 混淆矩阵可视化 ====
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
