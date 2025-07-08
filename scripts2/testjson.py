import os
import json
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_FILE = os.path.join(BASE_DIR, 'output2', 'dance_labels.json')

# read JSON file 
with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# extract category list 
categories = [item.get('category', 'Unknown') for item in data.values()]

# statistical distribution
category_counts = pd.Series(categories).value_counts().sort_values(ascending=False)

# print statistical result
print("📊 Statistics of the number of each category:")
print(category_counts)

# visualization
category_counts.plot(kind='barh', figsize=(10, 8))
plt.gca().invert_yaxis()
plt.title('Most Frequent Categories')
plt.xlabel('Count')
plt.ylabel('Category')
plt.tight_layout()
plt.show()

# optional：save as CSV file
#output_csv = os.path.join(BASE_DIR, 'output', 'category_distribution.csv')
#category_counts.to_csv(output_csv, header=['count'])
#print(f"✅ The classification statistics have been saved to:{output_csv}")
