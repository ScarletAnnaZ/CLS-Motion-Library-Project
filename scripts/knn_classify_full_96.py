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

from tqdm import tqdm
from sklearn.model_selection import _search


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

# create and train KNN model
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)


#########
# 超参数搜索空间 
param_dist = {
    'n_neighbors': randint(3, 15),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 曼哈顿 vs 欧几里得
}

# 搜索最佳参数 
knn = KNeighborsClassifier()
rand_search = RandomizedSearchCV(knn,
                                 param_distributions=param_dist,
                                 n_iter=5,
                                 cv=5,
                                 verbose=1,
                                 n_jobs=-1)

rand_search.fit(X_train, y_train)

best_knn = rand_search.best_estimator_
print("🌟 Best hyperparameters:", rand_search.best_params_)


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
print("📊 Classification Report:\n")
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
