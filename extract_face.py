import os
import cv2
import nltk
import torch
import numpy as np

from nltk.corpus import wordnet
from facenet_pytorch import MTCNN

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

def enhance_face_with_realesrgan(face, model_name='RealESRGAN_x4plus', outscale=4, denoise_strength=0.5, tile=0, tile_pad=10, pre_pad=0, face_enhance=False, fp32=False, gpu_id=None):
    model_name = model_name.split('.')[0]
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

    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join('weights'), progress=True, file_name=None)

    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_sr, _ = upsampler.enhance(face, outscale=outscale)
    face_sr = cv2.cvtColor(face_sr, cv2.COLOR_RGB2BGR)
    return face_sr

def detect_and_save_faces(input_folder, output_folder, keywords, device, padding_ratio=0.15, blur_threshold=50.0, min_face_size=64, model_name='RealESRGAN_x4plus', outscale=4, denoise_strength=0.5, tile=0, tile_pad=10, pre_pad=0, face_enhance=False, fp32=False, gpu_id=None):
    mtcnn = MTCNN(keep_all=True, device=device)
    expanded_keywords = expand_keywords(keywords)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if any(keyword in filename for keyword in expanded_keywords):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(img_rgb)

            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(b) for b in box]

                    width = x2 - x1
                    height = y2 - y1
                    padding_w = int(width * padding_ratio)
                    padding_h = int(height * padding_ratio)

                    # Add padding to the bounding box
                    x1 = max(0, x1 - padding_w)
                    y1 = max(0, y1 - padding_h)
                    x2 = min(img.shape[1], x2 + padding_w)
                    y2 = min(img.shape[0], y2 + padding_h)

                    face_with_surrounding = img[y1:y2, x1:x2]

                    if not is_blurry(face_with_surrounding, blur_threshold) and width >= min_face_size and height >= min_face_size:
                        enhanced_face = enhance_face_with_realesrgan(face_with_surrounding, model_name, outscale, denoise_strength, tile, tile_pad, pre_pad, face_enhance, fp32, gpu_id)

                        resized_face = cv2.resize(enhanced_face, (256, 256))

                        face_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_face_{i}.jpg")
                        cv2.imwrite(face_filename, resized_face)


input_folder = '/shika_data4/shivam/dataset/vggsound_processed/images'
output_folder = '/shika_data4/shivam/learning_to_localize_sound_source/faces'
keywords = ["laugh", "giggling", "sob", "sobbing", "nose", "smile", "sneeze", "cry", "cough", "scream", "explode", "explosion"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detect_and_save_faces(input_folder, output_folder, keywords, device)
