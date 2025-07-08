import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SUMMARY_FILE = os.path.join(OUTPUT_DIR, 'bvh_summary.csv')

def load_summary(file_path):
    return pd.read_csv(file_path)

def add_labels(ax):
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # only show non-zero values
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, color='black')

def visualize_summary(df):
    # set Seaborn 
    sns.set(style="whitegrid")
    
    # Frames distribution

    plt.figure(figsize=(12, 6))
    ax = sns.histplot(df['Frames'], bins=50, kde=True, color='skyblue')
    add_labels(ax)
    plt.title('Distribution of Frames in BVH Files')
    plt.xlabel('Frames')
    plt.ylabel('Count')
    plt.show()
    
    # Frame time distribution
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(df['Frame Time'], bins=50, kde=True, color='orange')
    add_labels(ax)
    plt.title('Distribution of Frame Time in BVH Files')
    plt.xlabel('Frame Time (seconds)')
    plt.ylabel('Count')
    plt.show()
    
    # total duration time distribution
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(df['Total Duration (seconds)'], bins=50, kde=True, color='green')
    add_labels(ax)
    plt.title('Distribution of Total Duration in BVH Files')
    plt.xlabel('Total Duration (seconds)')
    plt.ylabel('Count')
    plt.show()
    
    # Frames & Total Duration 
    plt.figure(figsize=(12, 6))
    ax = sns.scatterplot(data=df, x='Frames', y='Total Duration (seconds)', hue='Frame Time', palette='cool')
    plt.title('Frames vs Total Duration')
    plt.xlabel('Frames')
    plt.ylabel('Total Duration (seconds)')
    plt.show()

if __name__ == "__main__":
    # read CSV 
    df = load_summary(SUMMARY_FILE)
    
    visualize_summary(df)
