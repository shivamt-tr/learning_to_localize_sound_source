#!/bin/bash

# Define the list of splits
splits=(1 3 4 5 6 7 8 9 10)

# Loop through each split and run the Python script
for split in "${splits[@]}"
do
    python3 fetch_keyframe_json.py --json_dir /backup/data3/shivam/audioset_processed/ --data_dir /backup/data3/shivam/audio-visual-dataset/ --split $split --max_workers 20
done
