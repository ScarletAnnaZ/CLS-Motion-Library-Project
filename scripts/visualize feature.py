import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'features')
FEATURE_FILE = os.path.join(OUTPUT_DIR, 'small_features.csv')

# 加载数据
df = pd.read_csv(FEATURE_FILE)

# 查看数据的基本信息
print(df.info())
print(df.describe())

# 查看标签的分布情况
label_counts = df['Label'].value_counts()
print(label_counts)

# 可视化标签分布
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Label')
plt.title('Label Distribution')
plt.xlabel('Action Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 随机选择几列特征进行分布可视化 (比如前5列)
feature_columns = df.columns[:5]

# 绘制每个类别下每个特征的分布情况
plt.figure(figsize=(15, 8))
for i, feature in enumerate(feature_columns):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data=df, x=feature, hue='Label', kde=True, palette='Set2', bins=30)
    plt.title(f'Distribution of Feature {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 绘制每个特征的分布图
for feature in feature_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, hue='Label', kde=True, palette='Set2', bins=30)
    plt.title(f'Distribution of Feature: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()