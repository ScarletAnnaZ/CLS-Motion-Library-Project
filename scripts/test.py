import pandas as pd
import os
import json
import re  # 用于正则表达式处理

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
EXCEL_PATH = os.path.join(BASE_DIR, 'cmu-mocap-index-spreadsheet.xls')

def load_excel_labels(excel_path):
    # 读取 Excel 文件
    df = pd.read_excel(excel_path, sheet_name=0, skiprows=10, usecols=[0, 1, 2], names=['motion_id', 'description', 'category'])
    
    # 移除空值
    df.dropna(subset=['motion_id', 'description', 'category'], inplace=True)
    
    # 打印前五行检查
    print("Preview of the Excel file:")
    print(df.head())
    
    return df

def standardize_category_label(category):
    """
    从 category 标签中提取括号内的内容，并拆分成多个标签
    """
    match = re.search(r'\((.*?)\)', category)
    if match:
        category_content = match.group(1).strip()  # 提取括号内的内容
    else:
        category_content = category.strip()  # 如果不存在括号内容，则使用原始标签

    # 将标签拆分为列表，以逗号或斜杠为分隔符，并去除多余空格
    tags = [tag.strip().lower() for tag in re.split(r'[,/]', category_content)]
    tags = list(set(tags))  # 去除重复标签
    tags = [tag for tag in tags if tag]  # 移除空白标签
    
    return tags

def save_labels_as_json(labels_df, output_path):
    labels_dict = {}

    for index, row in labels_df.iterrows():
        motion_id = row['motion_id']
        description = row['description']
        categories = standardize_category_label(row['category'])
        
        labels_dict[motion_id] = {
            "description": description,
            "categories": categories  # 使用标准化并拆分后的标签列表
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
