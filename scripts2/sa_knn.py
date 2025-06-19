import os
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, 'output2') # change the local path
window_sizes = [10, 30, 60, 100, 120, 240, 600]
results = []


for win in window_sizes:
    file_path = os.path.join(FEATURE_DIR, f'features_{win}.csv')
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        continue
    
    df = pd.read_csv(file_path)
    # 统一采样数量（比如最多使用 1000 个 segment）
    SAMPLE_SIZE = 1000
    if len(df) > SAMPLE_SIZE:
       df = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
       print(f"⚠️ Window {win}: 数据量只有 {len(df)}，不足 {SAMPLE_SIZE}，保留全部")

    
    if 'Label' not in df.columns:
        print(f"❌ Lack Label: {file_path}")
        continue

    X = df.drop(columns=['Label']).values
    y = df['Label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        print(f"⚠️The Label distribution is uneven. Skip stratify: features_{win}.csv")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
    
    print(f"Window {win} Label quantity distribution:\n{pd.Series(y).value_counts()}\n")


    #X_train, X_test, y_train, y_test = train_test_split(
     #   X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    
    acc = accuracy_score(y_test, y_pred)
    results.append((win, acc ))
    print(f"✅ Window {win}: Accuracy = {acc:.4f}")
    print ()

# Convert to a DataFrame and visualize
if results:
    results_df = pd.DataFrame(results, columns=['window_size', 'accuracy']).sort_values('window_size')

    # print
    print("\n📊 Results of sensitivity analysis:")
    print(results_df)

    # visualize
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['window_size'], results_df['accuracy'], marker='o')
    plt.title('KNN Accuracy vs Frame Size')
    plt.xlabel('Frame Size')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No feature files were successfully read. Check whether the path and file name are correct.")
