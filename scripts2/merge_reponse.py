import os
from bvh import Bvh
import pandas as pd

# === 路径设置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENT_DIR = os.path.join(BASE_DIR, 'output2', 'segments_600')
SELECTION_FILE = os.path.join(BASE_DIR, 'output2', 'predicted_segment_responses.csv')
OUTPUT_BVH = os.path.join(BASE_DIR, 'output2', 'merged_response1.bvh')

# === 读取选择结果 ===
df = pd.read_csv(SELECTION_FILE)
file_list = df["Selected Motion"].tolist()

# === 合并 BVH 数据 ===
all_frames = []
frame_time = None
skeleton_text = None

for idx, fname in enumerate(file_list):
    if fname == "no_motion_found":
        print(f"⚠️ Skipping: no_motion_found")
        continue

    try:
        motion_id = fname.replace(".bvh", "")
        
        motion_path = os.path.join(SEGMENT_DIR, fname)

        if not os.path.exists(motion_path):
            print(f"⚠️ Skipping missing file: {motion_path}")
            continue

        with open(motion_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        bvh = Bvh(raw)

        if idx == 0:
            skeleton_text = raw.split("MOTION")[0]
            frame_time = bvh.frame_time

        all_frames.extend(bvh.frames)

    except Exception as e:
        print(f"❌ Error loading {fname}: {e}")
        continue

# === 保存合并后的 BVH 文件 ===
if all_frames:
    os.makedirs(os.path.dirname(OUTPUT_BVH), exist_ok=True)
    with open(OUTPUT_BVH, 'w', encoding='utf-8') as fout:
        fout.write(skeleton_text)
        fout.write("MOTION\n")
        fout.write(f"Frames: {len(all_frames)}\n")
        fout.write(f"Frame Time: {frame_time:.6f}\n")
        for frame in all_frames:
            fout.write(" ".join(frame) + "\n")
    print(f"\n✅ Merged response saved to: {OUTPUT_BVH}")
else:
    print("❌ No valid motions to merge.")
