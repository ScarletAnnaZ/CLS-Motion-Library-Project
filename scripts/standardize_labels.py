import pandas as pd
import os
import json
import re  # 用于正则表达式处理

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
EXCEL_PATH = os.path.join(BASE_DIR, 'cmu-mocap-index-spreadsheet.xls')

def load_excel_labels(excel_path):
    df = pd.read_excel(excel_path, sheet_name=0, skiprows=10, usecols=[0, 1, 2], names=['motion_id', 'description', 'category'])
    
    # remove na value
    df.dropna(subset=['motion_id', 'description', 'category'], inplace=True)
    
    # top 5 check
    print("Preview of the Excel file:")
    print(df.head())
    
    return df

def standardize_category_label(category):
    """
    从 category 标签中提取括号内的内容。
    如果不存在括号内容，则保留原标签。
    """
    match = re.search(r'\((.*?)\)', category)
    if match:
        return match.group(1).strip()  # 提取括号内的内容并去除两端空格
    return category.strip()  # 如果不存在括号内容，则返回原始标签

def save_labels_as_json(labels_df, output_path):
    labels_dict = {}

    for index, row in labels_df.iterrows():
        motion_id = row['motion_id']
        description = row['description']
        category = standardize_category_label(row['category'])
        
        labels_dict[motion_id] = {
            "description": description,
            "category": category  # 使用标准化后的标签
        }

    # 保存为 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels_dict, f, indent=4, ensure_ascii=False)

    print(f"✅ Labels saved as JSON to {output_path}")

if __name__ == "__main__":
    # 加载 Excel 标签
    labels_df = load_excel_labels(EXCEL_PATH)
    
    # 保存为 JSON 文件
    json_output_path = os.path.join(OUTPUT_DIR, 'standardized_labels.json')
    save_labels_as_json(labels_df, json_output_path)
