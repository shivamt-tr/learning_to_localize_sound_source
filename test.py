import os
import cv2
import json
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import warnings
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from network import AVModel
from soundnet import SoundNet

warnings.filterwarnings("ignore")

SAMPLE_RATE = 22050
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 16
NUM_WORKERS = 8

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

class VideoDataset(Dataset):
    def __init__(self, video_dir, audio_dir):
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.video_dir = video_dir
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)
        audio_path = os.path.join(self.audio_dir, os.path.splitext(video_file)[0] + ".wav")
        frames = extract_frames(video_path)
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        return frames, audio, video_file

def extract_frames(video_path):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: List of PIL Image objects representing video frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))  # Save each frame as a PIL image
    cap.release()
    return frames

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of PIL Images.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        tuple: Collated batch.
    """
    frames = [item[0] for item in batch]
    audios = [item[1] for item in batch]
    video_files = [item[2] for item in batch]
    return frames, audios, video_files

def get_keyframe(frames, audio, device, model, soundnet):
    """
    Find the keyframe index based on audio-visual correspondence.

    Args:
        frames (list): List of video frames (PIL Images).
        audio (numpy.ndarray): Audio signal.
        device (torch.device): Device to run the models on (CPU or CUDA).
        model (AVModel): Audio-Visual model.
        soundnet (SoundNet): SoundNet model.

    Returns:
        PIL.Image: Keyframe image.
        int: Index of the keyframe.
    """
    # Preprocess audio for SoundNet
    audio_input = torch.from_numpy(audio * 256).float().view(1, 1, -1, 1).to(device)
    audio_soundnet_features = soundnet(audio_input).squeeze().view(1, -1)

    # Initialize variables to find the key frame
    highest_corr = float('-inf')
    key_frame = None
    frame_idx = None

    # Process audio embedding
    with torch.no_grad():
        audio_embedding = model.audio_embedding_model(audio_soundnet_features)
        h = model.sound_fc1(audio_embedding)
        h = F.relu(h)
        h = model.sound_fc2(h)
        h = F.normalize(h, p=2, dim=1).unsqueeze(2)

    # Process frames in batches
    num_frames = len(frames)
    for batch_start in range(0, num_frames, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_frames)
        batch_frames = frames[batch_start:batch_end]

        # Transform frames and convert to tensor
        batch_tensors = torch.stack([image_transforms(frame) for frame in batch_frames]).to(device)

        # Process frame embeddings
        with torch.no_grad():
            vis_embeddings = model.vision_embedding_model(batch_tensors)

        # Normalize vision embeddings
        normalized_vis_embeddings = F.normalize(vis_embeddings, p=2, dim=1)
        reshaped_vis_embeddings = normalized_vis_embeddings.view(normalized_vis_embeddings.size(0), 512, 400).permute(0, 2, 1)

        # Compute attention maps
        att_maps = torch.matmul(reshaped_vis_embeddings, h)
        att_maps = F.relu(att_maps).squeeze()
        att_maps = model.att_softmax(att_maps)

        vis_embeddings = vis_embeddings.view(vis_embeddings.size(0), 512, 400)

        # Compute correspondence scores for the batch
        for i, (vis_embedding, att_map) in enumerate(zip(vis_embeddings, att_maps)):
            z = torch.matmul(vis_embedding, att_map).squeeze()
            corr_score = torch.sum(z).item()

            # Update the key frame if a higher correspondence score is found
            if corr_score > highest_corr:
                highest_corr = corr_score
                key_frame = batch_frames[i]
                frame_idx = batch_start + i

    return key_frame, frame_idx

def process_videos(args):

    video_dir = os.path.join(args.data_dir, "resized_video")
    audio_dir = os.path.join(args.data_dir, "audio")

    dataset = VideoDataset(video_dir, audio_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AVModel().to(device).eval()
    model.load_state_dict(torch.load("sound_localization_latest.pth"))

    soundnet = SoundNet().to(device).eval()
    soundnet.load_state_dict(torch.load("soundnet8_final.pth"))

    video_keyframes = dict()
    save_frequency = 10000
    error_count = 0
    pbar = tqdm(total=len(dataloader), desc="Processing")

    for idx, (frames, audio, video_file) in enumerate(dataloader):
        video_file = video_file[0]  # Unpack the single element in the batch
        if audio is None:
            error_count += 1
            continue
        try:
            key_frame, frame_idx = get_keyframe(frames[0], audio[0], device, model, soundnet)
            video_name = os.path.splitext(video_file)[0]
            video_keyframes[video_name] = frame_idx
            if (idx + 1) % save_frequency == 0:
                with open(os.path.join(args.data_dir, 'video_frames.json'), 'a') as json_file:
                    json.dump(video_keyframes, json_file)
                video_keyframes.clear()
        except Exception as e:
            print(f"Error in processing video {video_file}: {e}")
            error_count += 1
        pbar.set_description(f"Errors: {error_count}")
        pbar.update(1)

    pbar.close()

    if video_keyframes:  # Save any remaining keyframes
        with open(os.path.join(args.data_dir, 'video_frames.json'), 'a') as json_file:
            json.dump(video_keyframes, json_file)

    print(f"Total number of files: {len(dataloader)}, and {error_count} files could not be processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    process_videos(args)
