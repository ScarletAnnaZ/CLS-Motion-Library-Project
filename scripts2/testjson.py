import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ==== 设置路径（根据需要修改） ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_FILE = os.path.join(BASE_DIR, 'output2', 'dance_related_labels.json')

# ==== 读取 JSON 文件 ====
with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ==== 提取 category 列表 ====
categories = [item.get('category', 'Unknown') for item in data.values()]

# ==== 统计分布 ====
category_counts = pd.Series(categories).value_counts().sort_values(ascending=False)

# ==== 打印统计结果 ====
print("📊 各 category 的数量统计：")
print(category_counts)

# ==== 可视化前 N 个分类 ====
category_counts.plot(kind='barh', figsize=(10, 8))
plt.gca().invert_yaxis()
plt.title('Most Frequent Categories')
plt.xlabel('Count')
plt.ylabel('Category')
plt.tight_layout()
plt.show()

# ==== 可选：保存为 CSV 文件 ====
#output_csv = os.path.join(BASE_DIR, 'output', 'category_distribution.csv')
#category_counts.to_csv(output_csv, header=['count'])
#print(f"✅ 分类统计已保存到: {output_csv}")
