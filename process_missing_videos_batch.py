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
from network import AVModel
from soundnet import SoundNet

warnings.filterwarnings("ignore")

SAMPLE_RATE = 22050
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 4

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def extract_frames(video_path):
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

def get_keyframe(frames, audio, device, model, soundnet):
    audio_input = torch.from_numpy(audio * 256).float().view(1, 1, -1, 1).to(device)
    audio_soundnet_features = soundnet(audio_input).squeeze().view(1, -1)

    highest_corr = float('-inf')
    key_frame = None
    frame_idx = None

    with torch.no_grad():
        audio_embedding = model.audio_embedding_model(audio_soundnet_features)
        h = model.sound_fc1(audio_embedding)
        h = F.relu(h)
        h = model.sound_fc2(h)
        h = F.normalize(h, p=2, dim=1).unsqueeze(2)

    num_frames = len(frames)
    for batch_start in range(0, num_frames, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_frames)
        batch_frames = frames[batch_start:batch_end]

        batch_tensors = torch.stack([image_transforms(frame) for frame in batch_frames]).to(device)

        with torch.no_grad():
            vis_embeddings = model.vision_embedding_model(batch_tensors)

        normalized_vis_embeddings = F.normalize(vis_embeddings, p=2, dim=1)
        reshaped_vis_embeddings = normalized_vis_embeddings.view(normalized_vis_embeddings.size(0), 512, 400).permute(0, 2, 1)

        att_maps = torch.matmul(reshaped_vis_embeddings, h)
        att_maps = F.relu(att_maps).squeeze()
        att_maps = model.att_softmax(att_maps)

        vis_embeddings = vis_embeddings.view(vis_embeddings.size(0), 512, 400)

        for i, (vis_embedding, att_map) in enumerate(zip(vis_embeddings, att_maps)):
            z = torch.matmul(vis_embedding, att_map).squeeze()
            corr_score = torch.sum(z).item()

            if corr_score > highest_corr:
                highest_corr = corr_score
                key_frame = batch_frames[i]
                frame_idx = batch_start + i

    return key_frame, frame_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    video_dir = os.path.join(args.data_dir, "resized_video")
    audio_dir = os.path.join(args.data_dir, "audio")
    os.makedirs(os.path.join(args.data_dir, 'keyframe'), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AVModel().to(device).eval()
    model.load_state_dict(torch.load("sound_localization_latest.pth"))

    soundnet = SoundNet().to(device).eval()
    soundnet.load_state_dict(torch.load("soundnet8_final.pth"))

    video_keyframes = dict()
    save_frequency = 10000
    error_count = 0

    json_path = os.path.join(args.data_dir, 'video_frames.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as json_file:
                video_keyframes = json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
            video_keyframes = {}

    processed_videos = set(video_keyframes.keys())

    pbar = tqdm(total=len(os.listdir(video_dir)), desc="Processing")

    for idx, video_file in enumerate(os.listdir(video_dir)):
        if video_file.endswith(".mp4"):
            video_name = os.path.splitext(video_file)[0]
            if video_name in processed_videos:
                pbar.update(1)
                continue

            video_path = os.path.join(video_dir, video_file)
            audio_path = os.path.join(audio_dir, video_name + ".wav")
            try:
                frames = extract_frames(video_path)
                audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                if audio is None:
                    error_count += 1
                    continue
                key_frame, frame_idx = get_keyframe(frames, audio, device, model, soundnet)

                video_keyframes[video_name] = frame_idx

                if (idx + 1) % save_frequency == 0:
                    try:
                        with open(json_path, 'r') as json_file:
                            existing_data = json.load(json_file)
                    except json.JSONDecodeError as e:
                        print(f"Error loading JSON during save: {e}")
                        existing_data = {}
                    existing_data.update(video_keyframes)
                    with open(json_path, 'w') as json_file:
                        json.dump(existing_data, json_file)
                    video_keyframes.clear()
            except Exception as e:
                print(f"Error in processing video {video_file}: {e}")
                error_count += 1
            pbar.set_description(f"Errors: {error_count}")
            pbar.update(1)

    pbar.close()

    if video_keyframes:
        try:
            with open(json_path, 'r') as json_file:
                existing_data = json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON during final save: {e}")
            existing_data = {}
        existing_data.update(video_keyframes)
        with open(json_path, 'w') as json_file:
            json.dump(existing_data, json_file)

    print(f"Total number of files: {len(os.listdir(video_dir))}, and {error_count} files could not be processed")
