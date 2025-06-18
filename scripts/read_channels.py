import os
import pandas as pd
import numpy as np
from bvh import Bvh
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#BVH_FILE = os.path.join(BASE_DIR, 'input_AKOB', '1stmay', 'Take 2020-05-01 11.26.00_FB_mirror,follow,drones_follow.bvh')
BVH_FILE = os.path.join(BASE_DIR, 'output2','segments_120','05_01_seg0.bvh')
output_path =  os.path.join(BASE_DIR,'output','features','extract_joint_channels.json')
#"/Users/anzhai/motion-library-project/output/features/extract_joint_channels.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(BVH_FILE, "r") as f:
    bvh = Bvh(f.read())
print("CHANNEL NUMBER = ", len(bvh.get_joints()))  # 总关节数
print("Number of features per frame = ", len(bvh.frames[0]))  # 每帧的 feature 数


joint_channels = []

for joint in bvh.get_joints():
    joint_name = joint.name
    try:
        channels = bvh.joint_channels(joint_name)
        for ch in channels:
            joint_channels.append(f"{joint_name}_{ch}")
    except:
        continue

print(f"Total channels() = {len(joint_channels)}")
for i, ch in enumerate(joint_channels):
    print(f"{i+1:02d}: {ch}")
    
'''
# 保存 joint_channels 列表为 JSON 文件
with open(output_path, "w") as f:
    json.dump(joint_channels, f, indent=2) 

print(f"\n✅ Channel list saved to {output_path}")'''