import os
import cv2
import json
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from multiprocessing import Pool, cpu_count

from network import AVModel
from soundnet import SoundNet

sample_rate = 22050  # for SoundNet model input
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))  # save each frame as a PIL
    cap.release()
    return frames


def get_keyframe(frames, audio, device, model, soundnet, batch_size=32):
    # preprocess audio for soundnet
    audio_input = audio * 256  # make range [-256, 256]
    audio_input = torch.from_numpy(audio_input).float().view(1, 1, -1, 1)  # batch, channels=1, dim, 1 = 1, 1, 220672, 1
    audio_input = audio_input.to(device)

    # AVModel uses the output features of "object" branch of Conv8 layer from SoundNet model
    audio_soundnet_features = soundnet(audio_input)  # 1x1000x1x1
    audio_soundnet_features = audio_soundnet_features.squeeze().view(1, -1)  # 1x1000

    # get key_frame
    highest_corr = float('-inf')
    key_frame = None
    frame_idx = None

    # process audio embedding through AudioNet
    with torch.no_grad():
        audio_embedding = model.audio_embedding_model(audio_soundnet_features)
        h = model.sound_fc1(audio_embedding)
        h = F.relu(h)
        h = model.sound_fc2(h)

        # normalize audio embedding
        h = F.normalize(h, p=2, dim=1)
        h = h.unsqueeze(2)  # 1x512x1

    # batch processing of frames
    num_frames = len(frames)
    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        batch_frames = frames[batch_start:batch_end]

        transformed_frames = torch.stack([image_transforms(frame) for frame in batch_frames]).to(device)  # batch_size x 3 x 320 x 320

        # process frame embedding
        with torch.no_grad():
            vis_embedding = model.vision_embedding_model(transformed_frames)  # batch_size x 512 x 20 x 20

        # normalize vision embedding across channel dimension
        normalized_vis_embedding = F.normalize(vis_embedding, p=2, dim=1)  # batch_size x 512 x 20 x 20
        reshaped_vis_embedding = normalized_vis_embedding.view(normalized_vis_embedding.size(0), 512, 400)
        reshaped_vis_embedding = reshaped_vis_embedding.permute(0, 2, 1)  # batch_size x 400 x 512

        # compute attention map
        att_map = torch.matmul(reshaped_vis_embedding, h)  # batch_size x 400 x 1
        att_map = torch.squeeze(att_map)  # batch_size x 400
        att_map = F.relu(att_map)
        att_map = model.att_softmax(att_map)  # batch_size x 400

        vis_embedding = vis_embedding.view(vis_embedding.size(0), 512, 400)  # batch_size x 512 x 400
        z = torch.matmul(vis_embedding, att_map.unsqueeze(2))  # batch_size x 512 x 1
        z = torch.squeeze(z, 2)  # batch_size x 512

        # compute the correspondence score
        corr_scores = torch.sum(z, dim=1)  # batch_size

        max_score, max_idx = torch.max(corr_scores, dim=0)
        if max_score.item() > highest_corr:
            highest_corr = max_score.item()
            key_frame = batch_frames[max_idx.item()]
            frame_idx = batch_start + max_idx.item()

    return key_frame, frame_idx


def process_video(video_file, args):
    video_dir = os.path.join(args.data_dir, "resized_video")
    audio_dir = os.path.join(args.data_dir, "audio")
    video_path = os.path.join(video_dir, video_file)
    audio_path = os.path.join(audio_dir, os.path.splitext(video_file)[0] + ".wav")

    try:
        frames = extract_frames(video_path)  # frames=250x3x320x320,
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)  # audio = (220672,), 22.05kHz and mono for input to SoundNet
        if audio is None:
            return video_file, None
        key_frame, frame_idx = get_keyframe(frames, audio, args.device, args.model, args.soundnet)
        return video_file, frame_idx
    except Exception as e:
        print(f"Error in processing video {video_file}: {e}")
        return video_file, None


def main(args):
    video_dir = os.path.join(args.data_dir, "resized_video")
    os.makedirs(os.path.join(args.data_dir, 'keyframe'), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AVModel().to(device).eval()
    model.load_state_dict(torch.load("sound_localization_latest.pth"))

    soundnet = SoundNet().to(device).eval()
    soundnet.load_state_dict(torch.load("soundnet8_final.pth"))

    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    args.device = device
    args.model = model
    args.soundnet = soundnet

    video_keyframes = dict()
    error_count = 0

    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(lambda video_file: process_video(video_file, args), video_files),
                            total=len(video_files), desc="Processing"))

    for video_file, frame_idx in results:
        if frame_idx is not None:
            video_keyframes[os.path.splitext(video_file)[0]] = frame_idx
        else:
            error_count += 1

    with open(os.path.join(args.data_dir, 'video_frames.json'), 'w') as json_file:
        json.dump(video_keyframes, json_file)

    print(f"Total number of files: {len(video_files)}, and {error_count} files could not be processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args)
