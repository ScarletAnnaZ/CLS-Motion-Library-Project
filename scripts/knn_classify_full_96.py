import os
import pandas as pd
import joblib
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILE = os.path.join(BASE_DIR, 'output', 'features', 'full_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(FEATURE_FILE)

# Extract the feature matrix X and the label y
X = df.drop(columns=[ 'Label']).values
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['Label'].values 


# split train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Training set: {X_train.shape}, Test set: {X_test.shape}")

# create and train KNN model
best_knn = KNeighborsClassifier(n_neighbors=5)
best_knn.fit(X_train, y_train)

from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from itertools import product
import random

'''
# 随机采样超参数组合
n_iter = 5
param_combinations = list(product(
    [5, 10, 15],                     # n_neighbors
    ['uniform', 'distance'],        # weights
    [1, 2]                           # p
))

random.shuffle(param_combinations)
param_combinations = param_combinations[:n_iter]

best_score = -1
best_model = None
best_params = {}

print(" Searching best hyperparameters with progress:\n")

for n_neighbors, weights, p in tqdm(param_combinations):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    score = cross_val_score(knn, X_train, y_train, cv=3).mean()
    print(f"Params: n={n_neighbors}, w={weights}, p={p} → CV Score = {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_knn = knn
        best_params = {'n_neighbors': n_neighbors, 'weights': weights, 'p': p}

print("\n🌟 Best hyperparameters:", best_params)
print(f"✅ Best cross-validation accuracy: {best_score:.4f}")

'''

# Train best model
# best_knn.fit(X_train, y_train)


#！！store the model
joblib.dump(best_knn, os.path.join(MODEL_DIR, 'knn_model.pkl'))
print(f"✅ KNN model saved to {MODEL_DIR}/knn_model.pkl")

# ========== 评估 ==========
y_pred = best_knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")

# predict
# y_pred = knn.predict(X_test)

# output
# accuracy = accuracy_score(y_test, y_pred)
#print(f"✅ Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

#  confusion matrix
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
