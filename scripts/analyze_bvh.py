import os
import csv
from bvh import Bvh

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'bvh_summary.csv')

def read_bvh(filepath):
    with open(filepath, 'r') as file:
        data = file.read()
        return Bvh(data)

def analyze_bvh_files(data_dir, output_file):
    summary_data = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.bvh'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, data_dir)

                try:
                    # read file
                    bvh_data = read_bvh(input_path)
                    
                    # 
                    frames = len(bvh_data.frames)
                    frame_time = float(bvh_data.frame_time)
                    total_duration = frames * frame_time  # total duration wime(seconds)

                    summary_data.append([relative_path, frames, frame_time, total_duration])
                    print(f"✅ Processed: {relative_path}")

                except Exception as e:
                    print(f"❌ Failed to process {relative_path}: {e}")
    
    # store as CSV 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['File', 'Frames', 'Frame Time', 'Total Duration (seconds)'])
        csv_writer.writerows(summary_data)
    
    print(f"Analysis complete! Results saved to: {output_file}")

if __name__ == "__main__":
    analyze_bvh_files(DATA_DIR, OUTPUT_FILE)
