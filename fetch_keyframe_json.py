"""
Script that reads the JSON file, extracts the keyframe from each video,
and saves it in the "keyframe" folder with the same name as the video file.

Usage: 
python3 fetch_keyframe_json.py --json_dir /backup/data3/shivam/audioset_processed/ --data_dir /backup/data3/shivam/audio-visual-dataset/ --split 1  --max_workers 20
"""

import os
import json
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor

def extract_keyframe(task):
    """
    Extracts a keyframe from a video using ffmpeg.
    
    Parameters:
    - task (tuple): A tuple containing video name, keyframe index, video directory, and keyframe directory.
    
    Returns:
    - str: A message indicating the extraction status of the keyframe.
    """
    video_name, keyframe_index, video_dir, keyframe_dir = task
    video_file = os.path.join(video_dir, f"{video_name}.mp4")
    output_file = os.path.join(keyframe_dir, f"{video_name}.jpg")
    
    command = [
        'ffmpeg',
        '-i', video_file,
        '-vf', f"select=eq(n\\,{keyframe_index})",  # Select the frame by index
        '-vsync', 'vfr',
        '-q:v', '2',
        output_file
    ]
    
    subprocess.run(command)
    return f"Extracted keyframe for {video_name}"

def extract_keyframes(json_dir, data_dir, split=1, max_workers=4):
    """
    Extracts keyframes from videos based on indices provided in a JSON file.
    
    Parameters:
    - json_dir (str): Path to the directory containing keyframe json files.
    - data_dir (str): Path to the data directory containing videos.
    - split (int): Split number for directory structure.
    - max_workers (int): Maximum number of parallel workers to use for extraction.
    """
    video_dir = os.path.join(data_dir, f"unbalanced_train_{split}")
    keyframe_dir = os.path.join(data_dir, f"keyframe_split_{split}")

    # Load keyframe indices from JSON file
    with open(os.path.join(json_dir, f"video_frames_split{split}.json"), 'r') as f:
        keyframes = json.load(f)
    
    os.makedirs(keyframe_dir, exist_ok=True)

    # Prepare tasks for parallel processing
    tasks = [(video_name, keyframe_index, video_dir, keyframe_dir) for video_name, keyframe_index in keyframes.items()]

    # Extract keyframes in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(extract_keyframe, tasks):
            print(result)

    print("Keyframe extraction completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keyframes from videos based on JSON file.")
    parser.add_argument('--json_dir', type=str, required=True, help='Path to the JSON dir containing json files.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory containing videos and where keyframes will be saved.')
    parser.add_argument('--split', type=int, required=True, help='Split number for directory structure.')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel workers to use for extraction.')

    args = parser.parse_args()
    extract_keyframes(args.json_dir, args.data_dir, args.split, args.max_workers)
