import os
from bvh import Bvh

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'data_clean', 'dance_motion_library')
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'output2')
SEGMENT_SIZES = [10, 30, 60, 100]


def read_bvh_file(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
    return Bvh(data)


def write_bvh_segment(bvh_obj, header_lines, segment_frames, output_path):
    with open(output_path, 'w') as f:
        for line in header_lines:
            f.write(line)
        f.write(f"Frames: {len(segment_frames)}\n")
        f.write(f"Frame Time: {bvh_obj.frame_time}\n")
        for frame in segment_frames:
            f.write(" ".join(frame) + "\n")


def extract_segments(bvh_path, segment_size):
    bvh_obj = read_bvh_file(bvh_path)
    motion_frames = bvh_obj.frames

    if len(motion_frames) < segment_size:
        return []  # Skip files too short

    segments = []
    for start in range(0, len(motion_frames) - segment_size + 1, segment_size):
        segment = motion_frames[start:start + segment_size]
        segments.append(segment)

    return segments, bvh_obj


def extract_bvh_header(filepath):
    header = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("Frames:"):
                break
            header.append(line)
    return header


def process_all_segments():
    for seg_size in SEGMENT_SIZES:
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"segments_{seg_size}")
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(INPUT_DIR):
            if filename.endswith(".bvh"):
                input_path = os.path.join(INPUT_DIR, filename)
                try:
                    segments, bvh_obj = extract_segments(input_path, seg_size)
                    header = extract_bvh_header(input_path)

                    for idx, seg in enumerate(segments):
                        base_name = os.path.splitext(filename)[0]
                        output_name = f"{base_name}_seg{idx}.bvh"
                        output_path = os.path.join(output_dir, output_name)
                        write_bvh_segment(bvh_obj, header, seg, output_path)

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")


if __name__ == "__main__":
    process_all_segments()
    print("✅ Segmenting completed.")
