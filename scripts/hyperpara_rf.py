import os
import pandas as pd
import joblib
import numpy as np

from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# load and process data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILE = os.path.join(BASE_DIR, 'output', 'features', 'full_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, 'rf_hyper.pkl')

df = pd.read_csv(FEATURE_FILE)
X = df.drop(columns=["Label"]).values
y = df["Label"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define objective function
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    # Cast to int
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score

# Define search space
pbounds = {
    'n_estimators': (50, 300),
    'max_depth': (10, 50),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
}

# Run optimization
optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)


# Define the progress callback function to see the progress
init_points = 5
n_iter = 10

def logger(res):
    i = len(optimizer.res)
    print(f"📍 Round {i}/{init_points + n_iter} - score: {res['target']:.4f}")
    print(f"   ➤ Params: {res['params']}")
    print(f"   ⭐ Best score so far: {optimizer.max['target']:.4f}")

optimizer.maximize(
    init_points=init_points,   # initial random exploration
    n_iter=n_iter      # iterations of Bayesian optimization
)

# Best result
print("✅ Best result:")
print(optimizer.max)

#  refit the model using best parameters
best_params = optimizer.max['params']
best_model = RandomForestClassifier(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']),
    min_samples_split=int(best_params['min_samples_split']),
    min_samples_leaf=int(best_params['min_samples_leaf']),
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)


# model save
best_rf = best_model

joblib.dump(best_rf, MODEL_FILE)
print(f"✅ Random Forest model saved to {MODEL_FILE}")

# predict
y_pred = best_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# confusion matrix
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred, labels=best_rf.classes_)
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=best_rf.classes_, yticklabels=best_rf.classes_)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

