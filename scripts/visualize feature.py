import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'features')
FEATURE_FILE = os.path.join(OUTPUT_DIR, 'small_features.csv')

df = pd.read_csv(FEATURE_FILE)

# View the basic information of the data
print(df.info())
print(df.describe())

# View the basic information of the data
label_counts = df['Label'].value_counts()
print(label_counts)

# Visual label distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Label')
plt.title('Label Distribution')
plt.xlabel('Action Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

feature_columns = df.columns[:5]

# Draw the distribution of each feature under each category
plt.figure(figsize=(15, 8))
for i, feature in enumerate(feature_columns):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data=df, x=feature, hue='Label', kde=True, palette='Set2', bins=30)
    plt.title(f'Distribution of Feature {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Draw the distribution map of each feature
for feature in feature_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, hue='Label', kde=True, palette='Set2', bins=30)
    plt.title(f'Distribution of Feature: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()