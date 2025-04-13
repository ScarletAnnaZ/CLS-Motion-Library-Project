import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'features')
FEATURE_FILE = os.path.join(OUTPUT_DIR, 'small_features.csv')

df = pd.read_csv(FEATURE_FILE)

# Extraction feature matrix X 和 label y
X = df.drop(columns=['Label']).values
y = df['Label'].values

#80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creat KNN 
knn = KNeighborsClassifier(n_neighbors=5)  #  set n_neighbors as 5
knn.fit(X_train, y_train)

# predict for test
y_pred = knn.predict(X_test)

# calculate classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

print(classification_report(y_test, y_pred))

# plot
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
