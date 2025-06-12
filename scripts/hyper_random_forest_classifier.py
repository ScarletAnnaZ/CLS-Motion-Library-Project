import os
import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
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

MODEL_FILE = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
# SCALER_FILE = os.path.join(MODEL_DIR, 'scaler_rf.pkl')

df = pd.read_csv(FEATURE_FILE)
X = df.drop(columns=["Label"]).values
y = df["Label"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
# joblib.dump(scaler, SCALER_FILE)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# best_rf = RandomForestClassifier(n_estimators=100, max_depth=18, class_weight='balanced', random_state=42)

print("🚀 Start training Random Forest...")
# best_rf.fit(X_train, y_train)


from tqdm import tqdm
from sklearn.model_selection import _search


# 为 cross-validation 加进度条
tqdm.pandas()

original_fit_and_score = _search._fit_and_score

def fit_and_score_with_progress(*args, **kwargs):
    tqdm.write(f"🔄 Starting training: {fit_and_score_with_progress.counter + 1}")
    fit_and_score_with_progress.counter += 1
    return original_fit_and_score(*args, **kwargs)

fit_and_score_with_progress.counter = 0
_search._fit_and_score = fit_and_score_with_progress


'''hyperparameter'''

param_dist = {'n_estimators': randint(10,200),
              'max_depth': randint(10,50)}


# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)


# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('🌟Best hyperparameters:',  rand_search.best_params_)


# model save
joblib.dump(best_rf, MODEL_FILE)
print(f"✅ Random Forest model saved to {MODEL_FILE}")

# predict
y_pred = best_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

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

