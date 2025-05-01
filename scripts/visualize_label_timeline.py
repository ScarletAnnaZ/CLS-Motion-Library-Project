import os
import pandas as pd
import matplotlib.pyplot as plt

# 路径设置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENT_CSV = os.path.join(BASE_DIR, 'output', 'akob_segment_labels.csv')

# 读取段落标签数据
df = pd.read_csv(SEGMENT_CSV)

# 提取时间范围（起止秒数）
df['start'] = df['Time Segment'].apply(lambda x: float(x.split('-')[0]))
df['end'] = df['Time Segment'].apply(lambda x: float(x.split('-')[1].split()[0]))
df['duration'] = df['end'] - df['start']

# 可视化
fig, ax = plt.subplots(figsize=(14, 4))
y = 1

for idx, row in df.iterrows():
    ax.barh(y=y, width=row['duration'], left=row['start'], height=0.5,
            label=row['Dominant Label'] if idx == 0 or row['Dominant Label'] != df.loc[idx-1, 'Dominant Label'] else "",
            color=None)
    ax.text(row['start'] + row['duration']/2, y,
            row['Dominant Label'], va='center', ha='center', fontsize=8)

ax.set_xlim(0, df['end'].max())
ax.set_ylim(0.5, 1.5)
ax.set_xlabel('Time (seconds)')
ax.set_title('Predicted Action Timeline (Segment-wise)')
ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()
