import pandas as pd
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

excel_path = os.path.join(BASE_DIR, 'cmu-mocap-index-spreadsheet.xls')

df = pd.read_excel(excel_path)

print(df.head())


excel_path = os.path.join(BASE_DIR, 'cmu-mocap-index-spreadsheet.xls')
output_json = os.path.join(BASE_DIR, 'output', 'motion_labels.json')
output_csv = os.path.join(BASE_DIR, 'output', 'motion_labels.csv')

#  Excel（skip top 10 row）
df = pd.read_excel(excel_path, skiprows=10)

# keep the key column
df = df[['MOTION', 'DESCRIPTION from CMU web database', 'SUBJECT from CMU web database']]
df.columns = ['motion_id', 'long_label', 'subject_raw']

# Extract short label ( extarct the first key words from SUBJECT ）
df['short_label'] = df['subject_raw'].str.extract(r'Subject #[0-9]+ \((.*?)\)', expand=False)
df['short_label'] = df['short_label'].str.split(',').str[0].str.strip().str.lower()

# build dictionary 
motion_dict = {}
for _, row in df.iterrows():
    motion_id = str(row['motion_id']).strip()
    short_label = str(row['short_label']).strip()
    long_label = str(row['long_label']).strip()
    if motion_id and short_label and long_label:
        motion_dict[motion_id] = {
            'short_label': short_label,
            'long_label': long_label
        }

# store as JSON
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(motion_dict, f, indent=4, ensure_ascii=False)
print(f"motion_labels.json saved to: {output_json}")

# store as CSV
df[['motion_id', 'short_label', 'long_label']].to_csv(output_csv, index=False)
print(f"motion_labels.csv saved to: {output_csv}")
