import os
import random
import shutil

def copy_random_videos(src_folder, dest_folder, num_videos=50):
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # List all mp4 files in the source folder
    video_files = [f for f in os.listdir(src_folder) if f.endswith('.mp4')]

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

if __name__ == "__main__":
    src_folder = "/shika_backup/data3/shivam/audio-visual-dataset/unbalanced_train_4/"
    dest_folder = "/shika_data4/shivam/video_samples"
    copy_random_videos(src_folder, dest_folder, num_videos=50)
