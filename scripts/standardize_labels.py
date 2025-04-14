import pandas as pd
import os
import json
import re  

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
    extracte the content of the parentheses,
    if no parentheses exist, the original label retained
    """
    match = re.search(r'\((.*?)\)', category)
    if match:
        return match.group(1).strip()  # Extract the contents of the parentheses and remove the Spaces at both ends，
    return category.strip()  # if no parentheses, back to original labels

def save_labels_as_json(labels_df, output_path):
    labels_dict = {}

    for index, row in labels_df.iterrows():
        motion_id = row['motion_id']
        description = row['description']
        category = standardize_category_label(row['category'])
        
        labels_dict[motion_id] = {
            "description": description,
            "category": category  
        }

    # store as JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels_dict, f, indent=4, ensure_ascii=False)

    print(f"✅ Labels saved as JSON to {output_path}")

if __name__ == "__main__":
    # Excel labels
    labels_df = load_excel_labels(EXCEL_PATH)
    
    # as json 
    json_output_path = os.path.join(OUTPUT_DIR, 'standardized_labels.json')
    save_labels_as_json(labels_df, json_output_path)
