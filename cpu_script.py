import os
import cv2
import librosa
import numpy as np
from tqdm import tqdm
from PIL import Image
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


sample_rate = 22050
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_transforms = transforms.Compose([transforms.Resize((320, 320)),   # AVModel expects 320x320 input
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])


def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(np.array(frame))
    cap.release()
    
    return frames


def extract_audio(video_path, target_length=220672):

    try:
        # extract audio (the range of values is [-1, 1])
        audio, _ = librosa.load(video_path, sr=sample_rate, mono=True)          # audio = (220672,), 22.05kHz and mono for input to SoundNet
        np.clip(audio, -1, 1, out=audio)                                        # clip the samples to be in the range [-1, 1]
        audio = np.tile(audio, 10)[:target_length]                              # repeat the audio samples and select a fixed size to maintain consistency across different length audio

        # if audio length is less than the target length, pad with zeros
        if audio.shape[0] < target_length:
            audio = np.pad(audio, (0, target_length - audio.shape[0]), 'constant')
        
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return None

    return audio


if __name__ == "__main__":

    video_folder = '/shika_backup/data3/shivam/audio-visual-dataset/unbalanced_train_1'
    save_at = '/shika_backup/data3/shivam/audioset_test/'
    
    os.makedirs(save_at, exist_ok=True)
    os.makedirs(os.path.join(save_at, 'frames_array'), exist_ok=True)
    os.makedirs(os.path.join(save_at, 'audio'), exist_ok=True)

    error_count = 0
    pbar = tqdm(total=len(os.listdir(video_folder)), initial=0, desc="Processing")
    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_folder, video_file)
            try:
                frames = extract_frames(video_path)                     # frames=250x3x320x320,
                audio = extract_audio(video_path)                       # audio=(220672,)
                if audio is None:
                    error_count += 1
                    continue
                # save frames array as .npy and audio as .wav
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                np.save(os.path.join(save_at, "frames_array", f"{video_name}.npy"), frames)
                sf.write(os.path.join(save_at, "audio", f"{video_name}.wav"), audio, sample_rate)
            except Exception as e:
                print("Error in processing video: ", e)
                error_count += 1
        pbar.set_description(f"Errors: {error_count}")
        pbar.update(1)

    print(f"Total number of files: {len(os.listdir(video_folder))}, and {error_count} files could not be processed")