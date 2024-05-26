import os
import cv2
import time
import imageio
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

sample_rate = 22050  # for SoundNet model input
target_length=220672


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'resized_video'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'audio'), exist_ok=True)

    error_count = 0
    pbar = tqdm(total=len(os.listdir(args.data_dir)), initial=0, desc="Processing")
    for video_file in os.listdir(args.data_dir):
        if video_file.endswith(".mp4"):
            try:
                video_path = os.path.join(args.data_dir, video_file)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                
                # resize video frames
                fps = cap.get(cv2.CAP_PROP_FPS)
                writer = imageio.get_writer(os.path.join(args.save_dir, "resized_video", f"{video_name}.wav" ), mode='I', fps=fps)
                cap = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
                    writer.append_data(np.array(frame))
                cap.release()
                writer.close()

                # extract audio (the range of values is [-1, 1])
                audio, _ = librosa.load(video_path, sr=sample_rate, mono=True)          # audio = (220672,), 22.05kHz and mono for input to SoundNet
                np.clip(audio, -1, 1, out=audio)                                        # clip the samples to be in the range [-1, 1]
                audio = np.tile(audio, 10)[:target_length]                              # repeat the audio samples and select a fixed size to maintain consistency across different length audio

                # if audio length is less than the target length, pad with zeros
                if audio.shape[0] < target_length:
                    audio = np.pad(audio, (0, target_length - audio.shape[0]), 'constant')

                sf.write(os.path.join(args.save_dir, "audio", f"{video_name}.wav"), audio, sample_rate)
            except Exception as e:
                print("Error in processing video: ", e)
                error_count += 1
        pbar.set_description(f"Errors: {error_count}")
        pbar.update(1)

    print(f"Total number of files: {len(os.listdir(args.data_dir))}, and {error_count} files could not be processed")