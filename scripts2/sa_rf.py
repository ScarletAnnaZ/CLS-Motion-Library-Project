import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, 'output2')
window_sizes = [10, 30, 60, 100, 120, 240, 600]
SAMPLE_SIZE = 1685

# result cellection
results_knn = []
results_rf = []

for win in window_sizes:
    file_path = os.path.join(FEATURE_DIR, f'features_{win}.csv')
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        continue

    df = pd.read_csv(file_path)

    if 'Label' not in df.columns:
        print(f"❌ Missing Label column: {file_path}")
        continue

    # Hierarchical sampling
    if len(df) > SAMPLE_SIZE:
        try:
            df, _ = train_test_split(df, train_size=SAMPLE_SIZE, stratify=df['Label'], random_state=42)
        except ValueError:
            print(f"⚠️ Window {win}: Stratified sampling failed, fallback to random sampling")
            df = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        print(f"⚠️ Window {win}: Only {len(df)} samples, using all")

    # features + labels
    X = df.drop(columns=['Label']).values
    y = df['Label'].values
    X = StandardScaler().fit_transform(X)

    # split train and test set
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        print(f"⚠️ Window {win}: stratify split failed, using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    # train KNN 
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    results_knn.append((win, acc_knn))

    # train RF 
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    results_rf.append((win, acc_rf))

    print(f"✅ Window {win}: KNN Accuracy = {acc_knn:.4f}, RF Accuracy = {acc_rf:.4f}")

# transfer the result as DataFrame 
df_knn = pd.DataFrame(results_knn, columns=['window_size', 'knn_accuracy']).sort_values('window_size')
df_rf = pd.DataFrame(results_rf, columns=['window_size', 'rf_accuracy']).sort_values('window_size')
df_merged = pd.merge(df_knn, df_rf, on='window_size')

# visualization

plt.figure(figsize=(10, 6))
plt.plot(df_rf['window_size'], df_rf['rf_accuracy'], marker='o')
plt.title('RF Accuracy vs Frame Size')
plt.xlabel('Frame Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_merged['window_size'], df_merged['knn_accuracy'], marker='o', label='KNN')
plt.plot(df_merged['window_size'], df_merged['rf_accuracy'], marker='s', label='Random Forest')
plt.title('Accuracy Comparison: KNN vs Random Forest')
plt.xlabel('Window Size (frames)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# print 
print("\n📊 Combined Accuracy Table:")
print(df_merged)
