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
import time
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
# from face_parsing import BiSeNet

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


# def segment_face(face, segmenter, device):
#     """
#     Segments the face region from the input face image using a segmentation model.

#     Parameters:
#     face (ndarray): Input face image in BGR format.
#     segmenter (torch.nn.Module): Pre-trained face segmentation model.
#     device (torch.device): Device to run the segmentation model on.

#     Returns:
#     tuple: Bounding box (x1, y1, x2, y2) of the segmented face region.
#     """
#     face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#     face_rgb = cv2.resize(face_rgb, (512, 512))
#     face_rgb = torch.from_numpy(face_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
#     face_rgb = face_rgb.to(device)

#     with torch.no_grad():
#         out = segmenter(face_rgb)[0]
#     parsing = out.squeeze(0).cpu().numpy().argmax(0)
    
#     # Extract the bounding box of the face region
#     face_mask = (parsing > 0).astype(np.uint8)
#     contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) == 0:
#         return None, None, None, None  # Return invalid coordinates if no contours are found
#     x, y, w, h = cv2.boundingRect(contours[0])

#     return x, y, x + w, y + h


def detect_and_save_faces_from_video(input_folder, output_folder, keywords, device, padding_ratio=0.25,
                                     blur_threshold=100.0, min_face_size=64, min_prob=0.95,
                                     outscale=4, denoise_strength=0.5, tile=0, tile_pad=10, pre_pad=0):
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

    # Initialize the face enhancement model
    esrmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # Initialize the Real-ESRGAN upsampler
    upsampler = RealESRGANer(
        scale=4,
        model_path='RealESRGAN_x4plus.pth',
        dni_weight=None,
        model=esrmodel,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=False,            # uses fp16 precision when set to True
        device=device)  
    
    # Initialize the GFPGAN face enhancer
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler
    )

    # Initialize the face segmentation model
    # segmenter = BiSeNet(n_classes=19).to(device).eval()
    # segmenter.load_state_dict(torch.load('79999_iter.pth'), strict=False)
    
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
                # tic = time.time()
                boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
                # print("mt", time.time() - tic)
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
                            # Segment the face to include the full region
                            # seg_x1, seg_y1, seg_x2, seg_y2 = segment_face(face_with_surrounding, segmenter, device)
                            # if seg_x1 is None or seg_y1 is None or seg_x2 is None or seg_y2 is None:
                            #     continue
                            
                            # segmented_face = face_with_surrounding[seg_y1:seg_y2, seg_x1:seg_x2]
                            # if segmented_face.size == 0:
                            #     continue

                            # Segment the face to include the full region
                            # seg_x1, seg_y1, seg_x2, seg_y2 = segment_face(face_with_surrounding, segmenter, device)
                            # if seg_x1 is None or seg_y1 is None or seg_x2 is None or seg_y2 is None:
                            #     continue
                            
                            # segmented_face = face_with_surrounding[seg_y1:seg_y2, seg_x1:seg_x2]
                            # if segmented_face.size == 0:
                            #     continue

                            # Enhance the face using Real-ESRGAN
                            
                            # Convert the face image to RGB format
                            face_with_surrounding = cv2.cvtColor(face_with_surrounding, cv2.COLOR_BGR2RGB)
                            
                            # Enhance the face using GFPGAN
                            # tic = time.time()
                            _, _, face_sr = face_enhancer.enhance(face_with_surrounding, has_aligned=False, only_center_face=False, paste_back=True)
                            # print("enh", time.time() - tic)

                            # Convert the enhanced face back to BGR format
                            enhanced_face = cv2.cvtColor(face_sr, cv2.COLOR_RGB2BGR)
                            
                            # Convert the face image to RGB format
                            face_with_surrounding = cv2.cvtColor(face_with_surrounding, cv2.COLOR_BGR2RGB)
                            
                            # Enhance the face using GFPGAN
                            # tic = time.time()
                            _, _, face_sr = face_enhancer.enhance(face_with_surrounding, has_aligned=False, only_center_face=False, paste_back=True)
                            # print("enh", time.time() - tic)

                            # Convert the enhanced face back to BGR format
                            enhanced_face = cv2.cvtColor(face_sr, cv2.COLOR_RGB2BGR)

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
    parser.add_argument('--padding_ratio', type=float, default=0.5, help='Padding ratio around the detected face.')
    parser.add_argument('--padding_ratio', type=float, default=0.5, help='Padding ratio around the detected face.')
    parser.add_argument('--blur_threshold', type=float, default=100.0, help='Threshold to determine if an image is blurry.')
    parser.add_argument('--min_face_size', type=int, default=64, help='Minimum size of the face to detect.')
    parser.add_argument('--min_prob', type=float, default=0.95, help='Minimum probability threshold for face detection.')
    parser.add_argument('--outscale', type=int, default=4, help='Output scale for the face enhancement.')
    parser.add_argument('--denoise_strength', type=float, default=0.5, help='Denoise strength for the face enhancement.')
    # if you encounter cuda out of memory issues, set tile to a small number, this will ensure that image is processed in tiles
    parser.add_argument('--tile', type=int, default=4, help='Tile size for the face enhancement.')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding size for the face enhancement.')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size for the face enhancement.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        outscale=args.outscale,
        denoise_strength=args.denoise_strength,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
    )


if __name__ == '__main__':
    main()

