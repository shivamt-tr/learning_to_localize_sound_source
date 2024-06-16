import os
import cv2
import nltk
import time
import torch
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count, set_start_method

from nltk.corpus import wordnet
from facenet_pytorch import MTCNN

from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

nltk.download('wordnet')

def expand_keywords(keywords):
    expanded_keywords = set(keywords)
    for keyword in keywords:
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                expanded_keywords.add(lemma.name())
    return list(expanded_keywords)

def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def detect_and_save_faces_from_video(video_files, input_folder, output_folder, keywords, padding_ratio=0.25,
                                     blur_threshold=100.0, min_face_size=64, min_prob=0.95,
                                     outscale=4, denoise_strength=0.5, tile=0, tile_pad=10, pre_pad=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=min_face_size)
    esrmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='RealESRGAN_x4plus.pth',
        dni_weight=None,
        model=esrmodel,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=False,
        device=device)
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler
    )
    expanded_keywords = expand_keywords(keywords)
    os.makedirs(output_folder, exist_ok=True)
    for filename in tqdm(video_files, desc="Processing Videos"):
        if any(keyword in filename for keyword in expanded_keywords):
            video_path = os.path.join(input_folder, filename)
            video = cv2.VideoCapture(video_path)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, 24, dtype=int)
            for frame_count in frame_indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = video.read()
                if not ret:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
                if boxes is not None:
                    for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                        if prob < min_prob or landmark is None or len(landmark) != 5:
                            continue
                        x1, y1, x2, y2 = [int(b) for b in box]
                        width = x2 - x1
                        height = y2 - y1
                        padding_w = int(width * padding_ratio)
                        padding_h = int(height * padding_ratio)
                        x1 = max(0, x1 - padding_w)
                        y1 = max(0, y1 - padding_h)
                        x2 = min(frame.shape[1], x2 + padding_w)
                        y2 = min(frame.shape[0], y2 + padding_h)
                        face_with_surrounding = frame[y1:y2, x1:x2]
                        if not is_blurry(face_with_surrounding, blur_threshold):
                            face_with_surrounding = cv2.cvtColor(face_with_surrounding, cv2.COLOR_BGR2RGB)
                            _, _, face_sr = face_enhancer.enhance(face_with_surrounding, has_aligned=False, only_center_face=False, paste_back=True)
                            enhanced_face = cv2.cvtColor(face_sr, cv2.COLOR_RGB2BGR)
                            resized_face = cv2.resize(enhanced_face, (256, 256))
                            face_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_frame{frame_count}_face_{i}.jpg")
                            cv2.imwrite(face_filename, resized_face)
            video.release()

def process_videos(video_files_chunk, input_folder, output_folder, keywords, args):
    detect_and_save_faces_from_video(
        video_files=video_files_chunk,
        input_folder=input_folder,
        output_folder=output_folder,
        keywords=keywords,
        padding_ratio=args.padding_ratio,
        blur_threshold=args.blur_threshold,
        min_face_size=args.min_face_size,
        min_prob=args.min_prob,
        outscale=args.outscale,
        denoise_strength=args.denoise_strength,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
    )

def main():
    parser = argparse.ArgumentParser(description="Detect and save faces from videos with enhancement.")
    parser.add_argument('--input_folder', type=str, default='/shika_data4/shivam/keyframe_split_remaining', help='Path to the input folder containing videos.')
    parser.add_argument('--output_folder', type=str, default='/shika_data4/shivam/keyframe_face_remaining', help='Path to the output folder to save the processed images.')
    parser.add_argument('--padding_ratio', type=float, default=0.5, help='Padding ratio around the detected face.')
    parser.add_argument('--blur_threshold', type=float, default=100.0, help='Threshold to determine if an image is blurry.')
    parser.add_argument('--min_face_size', type=int, default=64, help='Minimum size of the face to detect.')
    parser.add_argument('--min_prob', type=float, default=0.95, help='Minimum probability threshold for face detection.')
    parser.add_argument('--outscale', type=int, default=4, help='Output scale for the face enhancement.')
    parser.add_argument('--denoise_strength', type=float, default=0.5, help='Denoise strength for the face enhancement.')
    parser.add_argument('--tile', type=int, default=0, help='Tile size for the face enhancement.')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding size for the face enhancement.')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size for the face enhancement.')
    args = parser.parse_args()

    # Set multiprocessing start method to 'spawn'
    set_start_method('spawn', force=True)

    keywords = ["laugh", "giggling", "sob", "sobbing", "nose", "nose blow",
                "laughter", "snore", "snoring", "smile", "sneeze", "cry",
                "cough", "scream", "explode", "explosion", "burp", "burping",
                "eructation", "hiccup", "sing", "choir", "whistle", "whistling",
                "whoop", "chatter", "whimper", "breathing", "grunt", "grunting",
                "throat", "sigh", "gurgling", "humming", "beatboxing"]

    input_folder = args.input_folder
    output_folder = args.output_folder

    video_files = os.listdir(input_folder)

    num_processes = 2
    chunk_size = len(video_files) // num_processes

    video_chunks = [video_files[i:i + chunk_size] for i in range(0, len(video_files), chunk_size)]

    with Pool(processes=num_processes) as pool:
        pool.starmap(process_videos, [(chunk, input_folder, output_folder, keywords, args) for chunk in video_chunks])

if __name__ == '__main__':
    main()
