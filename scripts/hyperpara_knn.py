import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from bayes_opt import BayesianOptimization
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# path setting
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILE = os.path.join(BASE_DIR, 'output', 'features', 'full_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, 'knn_bayes.pkl')

# load the data
df = pd.read_csv(FEATURE_FILE)
X = df.drop(columns=["Label"]).values
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Label"].values)  # Encode string labels to integers

# y = df["Label"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function
def knn_cv(n_neighbors, p, weights_flag):
    n_neighbors = int(round(n_neighbors))
    p = int(round(p))
    weights = 'uniform' if int(round(weights_flag)) == 0 else 'distance'

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p
    )
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score

# Define the search space
pbounds = {
    'n_neighbors': (1, 30),
    'p': (1, 2),
    'weights_flag': (0, 1)
}

# Bayesian Optimization
optimizer = BayesianOptimization(
    f=knn_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# progress function
init_points = 5
n_iter = 10

def logger(res):
    i = len(optimizer.res)
    print(f" Round {i}/{init_points + n_iter} - score: {res['target']:.4f}")
    print(f"   ➤ Params: {res['params']}")
    print(f"   ⭐ Best score so far: {optimizer.max['target']:.4f}")

# optimization
optimizer.maximize(
    init_points=init_points,
    n_iter=n_iter
)

# train the model with best parameters
best_params = optimizer.max['params']
best_knn = KNeighborsClassifier(
    n_neighbors=int(round(best_params['n_neighbors'])),
    p=int(round(best_params['p'])),
    weights='uniform' if int(round(best_params['weights_flag'])) == 0 else 'distance'
)
best_knn.fit(X_train, y_train)

# save the model
joblib.dump(best_knn, MODEL_FILE)
print(f"✅ Best KNN model saved to {MODEL_FILE}")

# evaluation model
y_pred = best_knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc:.4f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred, labels=best_knn.classes_)
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=best_knn.classes_, yticklabels=best_knn.classes_)
plt.title("Confusion Matrix - KNN (Bayesian Optimized)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
