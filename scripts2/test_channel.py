import os
import json
from bvh import Bvh

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_BVH = os.path.join(BASE_DIR, "input_AKOB", '1stmay',"Take 2020-05-01 11.26.00_FB_mirror,follow,drones_follow.bvh")  # 替换为实际路径
CHANNEL_JSON = os.path.join(BASE_DIR, "output", "features", "extract_joint_channels.json")

# Load the list of training channels
with open(CHANNEL_JSON, "r") as f:
    required_channels = json.load(f)

# Parse BVH files
with open(INPUT_BVH, "r") as f:
    bvh_data = Bvh(f.read())

input_channels = []
for joint in bvh_data.get_joints():
    joint_name = joint.name
    try:
        channels = bvh_data.joint_channels(joint_name)
        for ch in channels:
            input_channels.append(f"{joint_name}_{ch}")
    except:
        continue

# Output all the channels in the input BVH file
print("\n📦 Input BVH contains the following channels:")
for i, ch in enumerate(input_channels, 1):
    print(f"{i:03d}. {ch}")

# Check for missing and existing channels
missing_channels = [ch for ch in required_channels if ch not in input_channels]
existing_channels = [ch for ch in required_channels if ch in input_channels]

print(f"✅ Total required channels: {len(required_channels)}")
print(f"✅ Found in input: {len(existing_channels)}")
print(f"⚠️ Missing in input: {len(missing_channels)}")

if missing_channels:
    print("\n❌ Missing channels:")
    for ch in missing_channels:
        print("  -", ch)
else:
    print("\n🎉 All required channels are present!")
