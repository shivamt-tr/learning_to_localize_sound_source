"""
CUDA_VISIBLE_DEVICES=0 python3 extract_face_video.py --input_folder /shika_backup/data3/shivam/audio-visual-dataset/unbalanced_train_1/ --output_folder /shika_backup/data3/shivam/audio-visual-dataset/all_face_1

Reference: 
1. https://research.google.com/audioset/dataset/index.html
2. https://github.com/xinntao/Real-ESRGAN/
3. https://github.com/Tinglok/avstyle/blob/main/dataset/Into-the-Wild/video2jpg.py
4. https://www.kaggle.com/code/timesler/guide-to-mtcnn-in-facenet-pytorch
"""

import os
import cv2
import nltk
import torch
import numpy as np
from tqdm import tqdm
import argparse

from nltk.corpus import wordnet
from facenet_pytorch import MTCNN

from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


nltk.download('wordnet')


def expand_keywords(keywords):
    """
    Expands a list of keywords by including their synonyms.

    This function takes a list of keywords and finds their synonyms using WordNet,
    a lexical database for the English language. It returns a list of the original
    keywords along with their synonyms.

    Parameters:
    keywords (list of str): A list of keywords to be expanded.

    Returns:
    list of str: A list containing the original keywords and their synonyms.
    """
    # Convert the list of keywords to a set to avoid duplicate entries
    expanded_keywords = set(keywords)
    
    # Iterate through each keyword in the list
    for keyword in keywords:
        # Get all the synsets (synonym sets) for the current keyword
        for syn in wordnet.synsets(keyword):
            # Iterate through each lemma (word form) in the synset
            for lemma in syn.lemmas():
                # Add the lemma's name (synonym) to the set of expanded keywords
                expanded_keywords.add(lemma.name())
    
    # Convert the set back to a list before returning
    return list(expanded_keywords)


def is_blurry(image, threshold=100.0):
    """
    Determines if an image is blurry based on the variance of its Laplacian.

    This function converts the input image to grayscale, calculates the variance
    of the Laplacian (a measure of edge detection), and compares it to a specified
    threshold to determine if the image is blurry.

    Parameters:
    image (numpy.ndarray): The input image in BGR format.
    threshold (float): The threshold for the variance of the Laplacian. Images with
                       variance below this threshold are considered blurry. Default is 100.0.

    Returns:
    bool: True if the image is blurry, False otherwise.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Laplacian of the grayscale image and then compute the variance
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Determine if the variance is below the threshold
    return variance < threshold


def enhance_face_with_realesrgan(face, model_name='RealESRGAN_x4plus', outscale=4, denoise_strength=0.5, tile=0, tile_pad=10, pre_pad=0, gpu_id=None):
    """
    Enhance the resolution of a face image using Real-ESRGAN and optionally GFPGAN for face enhancement.

    Parameters:
    face (ndarray): Input face image in BGR format.
    model_name (str): The name of the model to use for enhancement. Default is 'RealESRGAN_x4plus'.
    outscale (int): The final upscaling factor. Default is 4.
    denoise_strength (float): Denoise strength, used only for the 'realesr-general-x4v3' model. Default is 0.5.
    tile (int): Tile size for processing to avoid memory issues. Default is 0 (no tiling).
    tile_pad (int): Tile padding size. Default is 10.
    pre_pad (int): Pre padding size at each border. Default is 0.

    Returns:
    ndarray: Enhanced face image in BGR format.
    """

    # Remove file extension from model name
    model_name = model_name.split('.')[0]

    # Select the appropriate model and its configuration
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # Determine the model path and download if not available
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join('weights'), progress=True, file_name=None)

    # Use dni_weight to control the denoise strength for 'realesr-general-x4v3' model
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # Initialize the Real-ESRGAN upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=False,  # uses fp16 precision when set to True
        gpu_id=gpu_id
    )
    
    # Initialize the GFPGAN face enhancer
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler
    )

    # Convert the face image to RGB format
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Enhance the face using GFPGAN
    _, _, face_sr = face_enhancer.enhance(face, has_aligned=False, only_center_face=False, paste_back=True)
    # face_sr, _ = upsampler.enhance(face, outscale=outscale)

    # Convert the enhanced face back to BGR format
    face_sr = cv2.cvtColor(face_sr, cv2.COLOR_RGB2BGR)
    
    return face_sr


def detect_and_save_faces_from_video(input_folder, output_folder, keywords, device, padding_ratio=0.25,
                                     blur_threshold=100.0, min_face_size=64, min_prob=0.95, model_name='RealESRGAN_x4plus',
                                     outscale=4, denoise_strength=0.5, tile=0, tile_pad=10, pre_pad=0, gpu_id=None):
    """
    Detects faces in video frames, enhances the faces using Real-ESRGAN, and saves the enhanced faces as images.

    Parameters:
    input_folder (str): Directory containing input videos.
    output_folder (str): Directory to save enhanced face images.
    keywords (list): List of keywords to filter the videos.
    device (torch.device): Device to run the MTCNN face detector.
    padding_ratio (float): Ratio of padding to add around the detected face. Default is 0.20.
    blur_threshold (float): Threshold for detecting blurry images. Default is 100.0.
    min_face_size (int): Minimum face size for detection. Default is 64.
    min_prob (float): Minimum probability threshold for face detection. Default is 0.95.
    model_name (str): Model name for Real-ESRGAN. Default is 'RealESRGAN_x4plus'.
    outscale (int): Upscale factor for Real-ESRGAN. Default is 4.
    denoise_strength (float): Denoise strength for Real-ESRGAN. Default is 0.5.
    tile (int): Tile size for Real-ESRGAN. Default is 0.
    tile_pad (int): Tile padding for Real-ESRGAN. Default is 10.
    pre_pad (int): Pre padding for Real-ESRGAN. Default is 0.

    Returns:
    None
    """
    
    # Initialize the MTCNN face detector
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=min_face_size)
    
    # Expand keywords using WordNet
    expanded_keywords = expand_keywords(keywords)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the video files in the input folder
    for filename in tqdm(os.listdir(input_folder), desc="Processing Videos"):
        if any(keyword in filename for keyword in expanded_keywords):
            video_path = os.path.join(input_folder, filename)
            video = cv2.VideoCapture(video_path)
            
            # Get the total number of frames in the video
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, 24, dtype=int)  # 24 evenly spaced frames

            for frame_count in frame_indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = video.read()
                if not ret:
                    continue

                # Convert the frame to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces and landmarks
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

                        # Add padding to the bounding box
                        x1 = max(0, x1 - padding_w)
                        y1 = max(0, y1 - padding_h)
                        x2 = min(frame.shape[1], x2 + padding_w)
                        y2 = min(frame.shape[0], y2 + padding_h)

                        face_with_surrounding = frame[y1:y2, x1:x2]

                        # Check if the face is not blurry
                        if not is_blurry(face_with_surrounding, blur_threshold):
                            # Enhance the face using Real-ESRGAN
                            enhanced_face = enhance_face_with_realesrgan(face_with_surrounding, model_name, outscale, denoise_strength, tile, tile_pad, pre_pad, gpu_id)

                            # Resize the enhanced face to 256x256
                            resized_face = cv2.resize(enhanced_face, (256, 256))

                            # Save the enhanced face image
                            face_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_frame{frame_count}_face_{i}.jpg")
                            cv2.imwrite(face_filename, resized_face)
    
    video.release()


def main():
    parser = argparse.ArgumentParser(description="Detect and save faces from videos with enhancement.")
    parser.add_argument('--input_folder', type=str, default='/shika_data4/shivam/keyframe_split_remaining', help='Path to the input folder containing videos.')
    parser.add_argument('--output_folder', type=str, default='/shika_data4/shivam/keyframe_face_remaining', help='Path to the output folder to save the processed images.')
    parser.add_argument('--padding_ratio', type=float, default=0.25, help='Padding ratio around the detected face.')
    parser.add_argument('--blur_threshold', type=float, default=100.0, help='Threshold to determine if an image is blurry.')
    parser.add_argument('--min_face_size', type=int, default=64, help='Minimum size of the face to detect.')
    parser.add_argument('--min_prob', type=float, default=0.95, help='Minimum probability threshold for face detection.')
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus', help='Name of the model for face enhancement.')
    parser.add_argument('--outscale', type=int, default=4, help='Output scale for the face enhancement.')
    parser.add_argument('--denoise_strength', type=float, default=0.5, help='Denoise strength for the face enhancement.')
    parser.add_argument('--tile', type=int, default=0, help='Tile size for the face enhancement.')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding size for the face enhancement.')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size for the face enhancement.')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keywords = ["laugh", "giggling", "sob", "sobbing", "nose", "nose blow",
                "laughter", "snore", "snoring", "smile", "sneeze", "cry",
                "cough", "scream", "explode", "explosion", "burp", "burping",
                "eructation", "hiccup", "sing", "choir", "whistle", "whistling",
                "whoop", "chatter", "whimper", "breathing", "grunt", "grunting",
                "throat", "sigh", "gurgling", "humming", "beatboxing", ]
    
    detect_and_save_faces_from_video(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        keywords=keywords,
        device=device,
        padding_ratio=args.padding_ratio,
        blur_threshold=args.blur_threshold,
        min_face_size=args.min_face_size,
        min_prob=args.min_prob,
        model_name=args.model_name,
        outscale=args.outscale,
        denoise_strength=args.denoise_strength,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        gpu_id=args.gpu
    )


if __name__ == '__main__':
    main()