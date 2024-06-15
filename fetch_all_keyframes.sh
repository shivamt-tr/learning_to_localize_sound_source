#!/bin/bash

# Define the list of splits
# splits=(3 10)

# Loop through each split and run the Python script
# for split in "${splits[@]}"
# do
#     python3 fetch_keyframe_json.py --json_dir /backup/data3/shivam/audioset_processed/ --data_dir /backup/data3/shivam/audio-visual-dataset/ --split $split --max_workers 20
# done

python3 fetch_keyframe_json.py --json_dir /backup/data3/shivam/audioset_processed/ --data_dir /data4/shivam/ --split 7 --max_workers 20
python3 fetch_keyframe_json.py --json_dir /backup/data3/shivam/audioset_processed/ --data_dir /data4/shivam/ --split 8 --max_workers 20