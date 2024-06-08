import os
import random
import shutil

def copy_random_videos(src_folder, dest_folder, num_videos=50):
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # List all mp4 files in the source folder
    video_files = [f for f in os.listdir(src_folder) if f.endswith('.png')]

    # Check if there are enough videos in the source folder
    if len(video_files) < num_videos:
        raise ValueError(f"Not enough videos in the source folder. Found {len(video_files)} videos, but need {num_videos}.")

    # Randomly select num_videos videos
    selected_videos = random.sample(video_files, num_videos)

    # Copy the selected videos to the destination folder
    for video in selected_videos:
        src_path = os.path.join(src_folder, video)
        dest_path = os.path.join(dest_folder, video)
        shutil.copy(src_path, dest_path)
        print(f"Copied {video} to {dest_folder}")


def copy_files_with_keyword(source_folder, destination_folder, keyword):
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Iterate over all files in the source folder
    for file_name in os.listdir(source_folder):
        # Check if the keyword is in the file name
        if keyword in file_name:
            source_path = os.path.join(source_folder, file_name)
            destination_path = os.path.join(destination_folder, file_name)
            
            # Copy the file to the destination folder
            shutil.copy(source_path, destination_path)
            print(f"Copied: {file_name}")


if __name__ == "__main__":
    src_folder = "/data4/shivam/dataset/vggsound_processed/images"
    dest_folder = "/data4/shivam/samples_vggsound"
    # copy_random_videos(src_folder, dest_folder, num_videos=1000)
    copy_files_with_keyword(src_folder, dest_folder, keyword="sobbing")
