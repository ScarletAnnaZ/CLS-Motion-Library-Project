import os
import pandas as pd
import numpy as np
from bvh import Bvh

with open("/Users/anzhai/motion-library-project/data/02/02_01.bvh", "r") as f:
    bvh = Bvh(f.read())

print("CHANNEL NUMBER = ", len(bvh.get_joints()))  # 总关节数
print("每帧数据维度 = ", len(bvh.frames[0]))  # 每帧的 feature 数
