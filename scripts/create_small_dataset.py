import os
import pandas as pd
import numpy as np

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SUMMARY_FILE = os.path.join(OUTPUT_DIR, 'bvh_summary.csv')
SMALL_SUMMARY_FILE = os.path.join(OUTPUT_DIR, 'small_bvh_summary.csv')

# Select the number of samples
SAMPLE_SIZE = 20

def create_small_dataset(summary_file, output_file, sample_size):
    # CSV 
    df = pd.read_csv(summary_file)
    
    # Select a specified number of samples at random
    small_df = df.sample(n=sample_size, random_state=42)  # set Seed 
    
    #  new CSV 
    small_df.to_csv(output_file, index=False)
    print(f"✅ Small dataset saved to {output_file}")

if __name__ == "__main__":
    create_small_dataset(SUMMARY_FILE, SMALL_SUMMARY_FILE, SAMPLE_SIZE)
