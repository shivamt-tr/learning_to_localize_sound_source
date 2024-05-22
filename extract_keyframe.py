import os
import cv2
import librosa
from tqdm import tqdm
from PIL import Image
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from network import AVModel
from soundnet import SoundNet


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
        frames.append(Image.fromarray(frame))
    cap.release()
    
    return frames


def process_video(video_file, device, model, soundnet):

    # extract audio
    audio, _ = librosa.load(video_file, sr=22000, mono=True)     # audio = (220172,), 22kHz used in SoundNet

    # preprocess audio for soundnet
    audio_input = audio * (2 ** -23)
    audio_input = torch.from_numpy(audio_input).float().view(1, 1, -1, 1)   # batch, channels, length, extra = 1, 1, 220172, 1
    audio_input = audio_input.to(device)
    
    # AVModel uses the output features of "object" branch of Conv8 layer from SoundNet model 
    audio_soundnet_features = soundnet(audio_input)     # 1x1000x1x1
    audio_soundnet_features = audio_soundnet_features.squeeze().view(1, -1)   # 1x1000

    # frames=250x3x320x320, # audio=(22012,)
    frames = extract_frames(video_file)

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
        h = h.unsqueeze(2)               # 1x512x1

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

    return key_frame, audio


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_folder = '/shika_backup/data3/shivam/audio-visual-dataset/unbalanced_train_1'
    save_at = '/shika_backup/data3/shivam/audioset/'
    os.makedirs(save_at, exist_ok=True)
    os.makedirs(os.path.join(save_at, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_at, 'audio'), exist_ok=True)

    model = AVModel().to(device).eval()
    model.load_state_dict(torch.load("sound_localization_latest.pth"))

    soundnet = SoundNet().to(device).eval()
    soundnet.load_state_dict(torch.load("soundnet8_final.pth"))

    error_count = 0
    pbar = tqdm(total=len(os.listdir(video_folder)), initial=0, desc="Processing")
    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4"):
            try:
                key_frame, audio = process_video(os.path.join(video_folder, video_file),
                                                 device,
                                                 model,
                                                 soundnet)
        
                # save key-frame and audio
                video_basename = os.path.basename(video_file)
                video_name = os.path.splitext(video_basename)[0]
                key_frame.save(os.path.join(save_at, "images", f"{video_name}.png"))
                sf.write(os.path.join(save_at, "audio", f"{video_name}.wav"), audio, 22000)
            except Exception as e:
                print("Error in processing video: ", e)
                error_count += 1
        pbar.set_description(f"Errors: {error_count}")
        pbar.update(1)

    print(f"Total number of files: {len(os.listdir(video_folder))}, and {error_count} files could not be processed")