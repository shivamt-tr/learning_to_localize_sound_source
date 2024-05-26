import os
import cv2
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import soundfile as sf
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from network import AVModel
from soundnet import SoundNet


sample_rate = 22050  # for SoundNet model input
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
        frames.append(Image.fromarray(frame))                                  # save each frame as a PIL
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


def get_keyframe(frames, audio, device, model, soundnet):
    
    # preprocess audio for soundnet
    audio_input = audio*256                                                 # make range [-256, 256]
    # audio_input = audio * (2 ** -23)                                      # make range [-256, 256]. what does this mean? why soundnet does it?
    # print(audio_input.shape, max(audio_input), min(audio_input))
    audio_input = torch.from_numpy(audio_input).float().view(1, 1, -1, 1)   # batch, channels=1, dim, 1 = 1, 1, 220172, 1

    audio_input = audio_input.to(device)

    # AVModel uses the output features of "object" branch of Conv8 layer from SoundNet model 
    audio_soundnet_features = soundnet(audio_input)                         # 1x1000x1x1
    audio_soundnet_features = audio_soundnet_features.squeeze().view(1, -1) # 1x1000

    # get key_frame
    highest_corr = float('-inf')
    key_frame = None
    frame_idx = None

    # process audio embedding through AudioNet
    with torch.no_grad():
        audio_embedding = model.audio_embedding_model(audio_soundnet_features)   
    
        # compute audio embedding and attention map
        h = model.sound_fc1(audio_embedding)
        h = F.relu(h)
        h = model.sound_fc2(h)

        # normalize audio embedding
        h = F.normalize(h, p=2, dim=1)
        h = h.unsqueeze(2)                                                    # 1x512x1

    for i, frame in enumerate(frames):

        transformed_frame = image_transforms(frame).unsqueeze(0).to(device)   # 1x3x320x320

        # process frame embedding
        with torch.no_grad():
            vis_embedding = model.vision_embedding_model(transformed_frame)   # 1x512x20x20
        
        # normalize vision embedding across channel dimension
        normalized_vis_embedding = F.normalize(vis_embedding, p=2, dim=1)     # 1x512x20x20
        reshaped_vis_embedding = normalized_vis_embedding.view(normalized_vis_embedding.size(0), 512, 400)
        reshaped_vis_embedding = reshaped_vis_embedding.permute(0, 2, 1)      # 1x400x512
        
        # compute attention map
        att_map = torch.matmul(reshaped_vis_embedding, h)                     # 1x400x1
        att_map = torch.squeeze(att_map)                                      # 400
        att_map = F.relu(att_map)
        att_map = model.att_softmax(att_map)                                  # 400
        # att_map = att_map.unsqueeze(2)
        
        vis_embedding = vis_embedding.view(vis_embedding.size(0), 512, 400)   # 1x512x400
        z = torch.matmul(vis_embedding, att_map)                              # 1x512
        z = torch.squeeze(z)                                                  # 512

        # compute the correspondence score
        corr_score = torch.sum(z).item()
        
        if corr_score > highest_corr:
            highest_corr = corr_score
            key_frame = frame
            frame_idx = i

    # print(f"{frame_idx+1} frame selected out of 250 frames")

    return key_frame


def process_video(video_file):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AVModel().to(device).eval()
    model.load_state_dict(torch.load("sound_localization_latest.pth"))

    soundnet = SoundNet().to(device).eval()
    soundnet.load_state_dict(torch.load("soundnet8_final.pth"))

    frames = extract_frames(video_file)     # frames=250x3x320x320
    audio = extract_audio(video_file)       # audio=(220672,)
    if audio is None:
        return None, None

    key_frame = get_keyframe(frames, audio, device, model, soundnet)
    return key_frame, audio


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'audio'), exist_ok=True)

    video_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".mp4")]

    def process_and_save(video_file):
        try:
            key_frame, audio = process_video(video_file)
            if key_frame is None or audio is None:
                return False
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            key_frame.save(os.path.join(args.save_dir, "images", f"{video_name}.png"))
            sf.write(os.path.join(args.save_dir, "audio", f"{video_name}.wav"), audio, sample_rate)
            return True
        except Exception as e:
            print(f"Error in processing video: {e}")
            return False

    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_and_save, video_files), total=len(video_files)))

    error_count = len(video_files) - sum(results)
    print(f"Total number of files: {len(video_files)}, and {error_count} files could not be processed")