#!/usr/bin/env python3
# merge_bvh_sequence.py

import os
import random
from bvh import Bvh
from generate_agent_responses import get_file_list   # <- 新增
                                         
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BVH_DIR     = os.path.join(BASE_DIR,'output','processed600_bvh')
OUTPUT_BVH  = os.path.join(BASE_DIR, 'output', 'merged_sequence.bvh')

random.seed(42)

# 调用 generate_agent_responses 里的函数拿到 file_list
file_list = get_file_list()

# merge BVH 
all_frames    = []
frame_time    = None
skeleton_text = None

for idx, fname in enumerate(file_list):
    # 从文件名中提取主体编号，比如 "28_18.bvh" → folder "28"
    subject = fname.split('_')[0]
    bvh_path = os.path.join(BVH_DIR, subject, fname)

    print(f"Reading {bvh_path}")
    
    with open(bvh_path, 'r', encoding='utf-8') as f:
            raw = f.read()
    bvh = Bvh(raw)
    if idx == 0:
            skeleton_text = raw.split("MOTION")[0]
            frame_time    = bvh.frame_time
    all_frames.extend(bvh.frames)

print(f"\nTotal frames merged: {len(all_frames)}")

os.makedirs(os.path.dirname(OUTPUT_BVH), exist_ok=True)
with open(OUTPUT_BVH, 'w', encoding='utf-8') as fout:
    fout.write(skeleton_text)
    fout.write("MOTION\n")
    fout.write(f"Frames: {len(all_frames)}\n")
    fout.write(f"Frame Time: {frame_time:.6f}\n")
    for frame in all_frames:
        fout.write(" ".join(frame) + "\n")

print(f"\n✅ Merged BVH saved to {OUTPUT_BVH}")
