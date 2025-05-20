import os
import json
import numpy as np
import pandas as pd
from bvh import Bvh

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'output', 'processed600_bvh')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'features')
LABELS_FILE = os.path.join(BASE_DIR, 'output', 'standardized_labels.json')
SUMMARY_FILE = os.path.join(BASE_DIR, 'output', 'bvh_summary.csv')  
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'full_features.csv')
TARGET_FRAMES = 600

def read_bvh(filepath):
    with open(filepath, 'r') as file:
        data = file.read()
        return Bvh(data)

def extract_features(bvh_data):
    return np.array(bvh_data.frames, dtype=float)

def load_labels(labels_file):
    with open(labels_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_file(file_path, label):
    bvh_data = read_bvh(file_path)
    features = extract_features(bvh_data)
    labels = [label] * len(features)
    return features, labels

def save_features(features_list, labels_list, output_file):
    all_features = np.vstack(features_list)
    all_labels = np.concatenate(labels_list)

    df = pd.DataFrame(all_features)
    df['Label'] = all_labels

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Features saved to {output_file}")

def process_all_files(summary_file, labels_dict, processed_dir, output_file):
    df_summary = pd.read_csv(summary_file)
    features_list = []
    labels_list = []

    for _, row in df_summary.iterrows():
        full_motion_id = row['File']
        motion_id = os.path.basename(full_motion_id).replace('.bvh', '')

        if motion_id in labels_dict:
            label = labels_dict[motion_id]["category"]
            rel_path = os.path.join(processed_dir, full_motion_id)

            if os.path.exists(rel_path):
                try:
                    features, labels = process_file(rel_path, label)
                    features_list.append(features)
                    labels_list.append(labels)
                    print(f"✅ Processed: {motion_id}")
                except Exception as e:
                    print(f"❌ Failed {motion_id}: {e}")
            else:
                print(f"❌ File not found: {rel_path}")
        else:
            print(f"❌ Label not found for {motion_id}")

    save_features(features_list, labels_list, output_file)

if __name__ == "__main__":
    labels_dict = load_labels(LABELS_FILE)
    process_all_files(SUMMARY_FILE, labels_dict, PROCESSED_DIR, OUTPUT_FILE)
