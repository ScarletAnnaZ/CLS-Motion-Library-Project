import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
LABELS_FILE = os.path.join(OUTPUT_DIR, 'standardized_labels.json')

def load_labels(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_labels(labels_dict):
    # Fetch labeled data
    categories = [v['category'] for v in labels_dict.values()]
    descriptions = [v['description'] for v in labels_dict.values()]
    
    # create DataFrame
    df = pd.DataFrame({
        'category': categories,
        'description': descriptions
    })
    
    # label categories‘ number
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['Label', 'Count']
    # description counts
    description_counts = df['description'].value_counts().reset_index()
    description_counts.columns = ['Label', 'Count']
    
    # CSV
    category_counts.to_csv(os.path.join(OUTPUT_DIR, 'category_counts.csv'), index=False)
    description_counts.to_csv(os.path.join(OUTPUT_DIR, 'description_counts.csv'), index=False)
    
    print("✅ Category counts and description counts saved as CSV files.")
    
    # plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Count', y='Label', data=category_counts)
    plt.title('Top categories by Frequency')
    plt.xlabel('Count')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.show()
    

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Count', y='Label', data=description_counts.head(20))
    plt.title('Top 20 Descriptions by Frequency')
    plt.xlabel('Count')
    plt.ylabel('Description')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    labels_dict = load_labels(LABELS_FILE)
    
    # 分析标签
    analyze_labels(labels_dict)
