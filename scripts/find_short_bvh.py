import os
import pandas as pd
from bvh import Bvh


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'short_bvh_files.csv')

TARGET_FRAMES = 600

def read_bvh(filepath):
    with open(filepath, 'r') as file:
        data = file.read()
        return Bvh(data)

def find_short_bvh_files(data_dir, output_file):
    short_files = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.bvh'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, data_dir)

                try:
                    # read BVH 
                    bvh_data = read_bvh(file_path)
                    
                    # get frames
                    frame_count = len(bvh_data.frames)
                    
                    if frame_count < TARGET_FRAMES:
                        short_files.append([relative_path, frame_count])
                        print(f"✅ Found short file: {relative_path} with {frame_count} frames")
                
                except Exception as e:
                    print(f"❌ Failed to read {relative_path}: {e}")

    # CSV
    df = pd.DataFrame(short_files, columns=['File', 'Frames'])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_file}")

if __name__ == "__main__":
    find_short_bvh_files(DATA_DIR, OUTPUT_FILE)
