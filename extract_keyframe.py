import cv2
import librosa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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


def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def extract_frames_and_audio(video_path):

    # extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    frames = torch.cat([image_transforms(f).unsqueeze(0) for f in frames])    # 250x3x320x320

    fps = cap.get(cv2.CAP_PROP_FPS)   # 25

    # extract audio
    audio, sr = librosa.load(video_path, sr=22000, mono=True)     # audio = (220172,), 22kHz used in SoundNet
    audio = audio * (2 ** -23)
    audio = torch.from_numpy(audio).float().view(1, 1, -1, 1)     # batch, channels, length, extra = 1, 1, 220172, 1
    
    return frames, audio, sr, fps


def get_keyframe(model, frames, audio):

    highest_corr = float('-inf')
    key_frame = None
    frame_idx = None

    # process audio embedding through AudioNet
    with torch.no_grad():
        audio_embedding = model.audio_embedding_model(audio)   
    
        # compute audio embedding and attention map
        h = model.sound_fc1(audio_embedding)
        h = F.relu(h)
        h = model.sound_fc2(h)

        # normalize audio embedding
        h = F.normalize(h, p=2, dim=1)
        h = h.unsqueeze(2)               # 1x512x1

    for i, frame in enumerate(frames):

        # process frame embedding
        with torch.no_grad():
            vis_embedding = model.vision_embedding_model(frame).unsqueeze(0)                   # 1x512x20x20

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

    print(f"{frame_idx+1} frame selected out of 250 frames")
    # print(torch.max(key_frame), torch.min(key_frame))
    key_frame = unnormalize(key_frame, mean, std)  # Unnormalize the tensor
    key_frame = key_frame.permute(1, 2, 0).cpu().numpy()
    key_frame = np.clip(key_frame * 255, 0, 255).astype(np.uint8)
    key_frame = Image.fromarray(key_frame)

    return key_frame


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AVModel().to(device).eval()
    model.load_state_dict(torch.load("sound_localization_latest.pth"))

    soundnet = SoundNet().to(device).eval()
    soundnet.load_state_dict(torch.load("soundnet8_final.pth"))

    video_path = 'video_samples/-0sDdBQt3Gc_"Railroad car, train wagon, Rail transport, Train, Vehicle".mp4'
    
    # sr=44100, fps=25, frames=250x3x320x320, #audio=441344
    frames, audio, sr, fps = extract_frames_and_audio(video_path)
    frames, audio = frames.to(device), audio.to(device)
    
    # AVModel uses the output features of "object" branch of Conv8 layer from SoundNet model 
    audio_soundnet_features = soundnet(audio)     # 1x1000x1x1
    audio_soundnet_features = audio_soundnet_features.squeeze().view(1, -1)   # 1x1000

    key_frame_image = get_keyframe(model, frames, audio_soundnet_features)
    key_frame_image.save("keyframe.png")
