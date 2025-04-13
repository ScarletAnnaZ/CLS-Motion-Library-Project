import os
import json

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
STANDARD_LABELS_FILE = os.path.join(OUTPUT_DIR, 'tag_library.json')
LABELS_FILE = os.path.join(OUTPUT_DIR, 'standardized_labels.json')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'normalized_labels.json')

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_labels(labels_dict, standard_labels):
    normalized_dict = {}
    
    for motion_id, label_info in labels_dict.items():
        description = label_info["description"]
        original_category = label_info["category"]
        
        # 初始化一个空的标准标签列表
        normalized_categories = set()
        
        # 遍历标签库进行匹配
        for standard_label, equivalent_labels in standard_labels.items():
            for label in equivalent_labels:
                if label.lower() in original_category.lower():
                    normalized_categories.add(standard_label)
        
        # 如果没有匹配到标准标签，则保留原标签
        if not normalized_categories:
            normalized_categories.add(original_category)
        
        normalized_dict[motion_id] = {
            "description": description,
            "categories": list(normalized_categories)
        }
    
    return normalized_dict

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Normalized labels saved to {file_path}")

if __name__ == "__main__":
    # 加载文件
    labels_dict = load_json(LABELS_FILE)
    standard_labels = load_json(STANDARD_LABELS_FILE)
    
    # 进行标签规范化
    normalized_labels = normalize_labels(labels_dict, standard_labels)
    
    # 保存结果
    save_json(normalized_labels, OUTPUT_FILE)
