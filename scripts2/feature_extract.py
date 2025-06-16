from bvh import Bvh
import numpy as np
import os
import pandas as pd

# 设置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'output2', 'segments_120')

OUTPUT_CSV = os.path.join(BASE_DIR, 'output2', 'dance_features.csv')

def extract_features_from_bvh(file_path):
    with open(file_path, 'r') as f:
        bvh_data = Bvh(f.read())

    motion = np.array(bvh_data.frames, dtype=float)  # shape: (n_frames, n_channels)
    
    # 提取统计特征：mean, std, max, min
    stats = []
    stats.extend(np.mean(motion, axis=0))
    stats.extend(np.std(motion, axis=0))
    stats.extend(np.max(motion, axis=0))
    stats.extend(np.min(motion, axis=0))
    
    return stats

def batch_extract_features(data_dir, output_csv):
    records = []

    for fname in os.listdir(data_dir):
        if fname.endswith('.bvh'):
            fpath = os.path.join(data_dir, fname)
            try:
                features = extract_features_from_bvh(fpath)
                records.append([fname] + features)
            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")
    
    # 保存为 DataFrame
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"✅ Feature extraction complete. Saved to: {output_csv}")
