"""
Script that reads the JSON file, extracts the keyframe from each video,
and saves it in the "keyframe" folder with the same name as the video file.
"""

import os
import json
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor

def extract_keyframe(task):
    video_name, keyframe_index, video_dir, keyframe_dir = task
    video_file = os.path.join(video_dir, f"{video_name}.mp4")
    output_file = os.path.join(keyframe_dir, f"{video_name}.jpg")
    
    command = [
        'ffmpeg',
        '-i', video_file,
        '-vf', f"select=eq(n\\,{keyframe_index})", # Select the frame by index
        '-vsync', 'vfr',
        '-q:v', '2',
        output_file
    ]
    
    subprocess.run(command)
    return f"Extracted keyframe for {video_name}"

def extract_keyframes(json_path, data_dir, split=1, max_workers=4):
    video_dir = os.path.join(data_dir, f"unbalanced_train_{split}")
    keyframe_dir = os.path.join(data_dir, f"keyframe_split_{split}")

    with open(json_path, 'r') as f:
        keyframes = json.load(f)

    if not os.path.exists(keyframe_dir):
        os.makedirs(keyframe_dir)

    tasks = [(video_name, keyframe_index, video_dir, keyframe_dir) for video_name, keyframe_index in keyframes.items()]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(extract_keyframe, tasks):
            print(result)

    print("Keyframe extraction completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keyframes from videos based on JSON file.")
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file containing keyframe indices.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory containing videos and where keyframes will be saved.')
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel workers to use for extraction.')

    args = parser.parse_args()

    extract_keyframes(args.json_path, args.data_dir, args.split, args.max_workers)



