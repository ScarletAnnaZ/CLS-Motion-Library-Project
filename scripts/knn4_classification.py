import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILE = os.path.join(BASE_DIR, 'output', 'features', 'knn4_features.csv')
print("🔍 Looking for:", FEATURE_FILE)

# read features data
df = pd.read_csv(FEATURE_FILE)

# extract feature X and Y
X = df[['Root_Y_mean', 'Head_Z_std', 'LeftKnee_Y_std', 'RightElbow_X_mean']].values
y = df['Label'].values

# split to train and test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# initial and train
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# predict
y_pred = knn.predict(X_test)

# print
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
