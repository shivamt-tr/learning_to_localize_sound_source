"""
Audio Preprocessing Script with Keyword Filtering and Parallel Processing

This script processes audio files in a specified folder by converting them into mel-spectrograms.
The mel-spectrograms are then saved as PyTorch tensors. The script includes functionality to 
filter audio files based on specified keywords and their synonyms using WordNet.

Usage: python preprocess_audio_vqvae.py
"""

import os
import librosa
import numpy as np
import cv2
import torch
import random
from tqdm import tqdm
from nltk.corpus import wordnet
from multiprocessing import Pool, cpu_count

import nltk
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


def audio_to_mel(audio_path, sr=22050):
    """
    Preprocesses an audio file as a mel-spectrogram for VQ-VAE2 audio encoder.

    Args:
        audio_path (str): Path to the audio sample.

    Returns:
        torch.Tensor: A mel-spectrogram tensor (BxCxHxW = 1x1x512x128)
        H: Number of frequency bins in the resized mel spectrogram
        W: Number of time frames in the resized mel spectrogram

    Notes:
        Steps to compute mel-spectrogram and convert to fixed-time length and resolution
        1. Loads the audio file.
        2. Computes the mel spectrogram.
        3. Converts the power spectrogram to the decibel scale.
        4. Adjusts the mel spectrogram to a fixed time length.
        5. Resizes the mel spectrogram to a fixed resolution.
        6. Converts and reshapes the inputs for PyTorch.

    Reference:
        1. https://github.com/kuai-lab/sound-guided-semantic-image-manipulation
        2. https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
    """
    
    n_mels = 128
    time_length = 432  # or 864
    resize_resolution = 512

    # Load audio as a time series numpy array
    y, sr = librosa.load(audio_path, sr=sr)   # y: (220672,)

    # Compute mel spectrogram, convert to decibel scale, and normalize to [0, 1]
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)        # 128x432
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max) / 80.0 + 1      # 128x432

    # Initialize a zero matrix for padding or cropping
    zero = np.zeros((n_mels, time_length))
    h, w = mel_spectrogram.shape
    
    # Adjust mel spectrogram to have a fixed time length
    if w >= time_length:
        j = random.randint(0, w - time_length)
        mel_spectrogram = mel_spectrogram[:, j:j + time_length]
    else:
        zero[:, :w] = mel_spectrogram[:, :w]
        mel_spectrogram = zero
    
    # Resize mel spectrogram to a fixed time frame of 512
    resized_spectrogram = cv2.resize(mel_spectrogram, (resize_resolution, n_mels))      # 128x512

    # Convert and reshape the input for PyTorch
    audio_input = torch.from_numpy(resized_spectrogram).unsqueeze(0).float()   # CxHxW = 1x128x512

    return audio_input


def process_file(args):
    """
    Processes an individual audio file if its name contains any expanded keywords.
    
    Args:
        args (tuple): A tuple containing:
            file_name (str): The name of the audio file.
            folder_path (str): The path to the folder containing the audio files.
            output_folder (str): The path to the folder where the processed tensors will be saved.
            sr (int): The sample rate for audio processing.
            expanded_keywords (list): The list of expanded keywords to check in the filename.
    
    Returns:
        None
    """
    file_name, folder_path, output_folder, sr, expanded_keywords = args
    
    # Check if any expanded keyword is in the file name
    if any(keyword in file_name for keyword in expanded_keywords):
        audio_path = os.path.join(folder_path, file_name)
        mel_tensor = audio_to_mel(audio_path, sr=sr)
        
        # Construct the output path and save the tensor
        output_path = os.path.join(output_folder, file_name.replace('.wav', '.pt'))
        torch.save(mel_tensor, output_path)


def preprocess_and_save(folder_path, output_folder, keywords, sr=22050):
    """
    Preprocesses audio files in a folder, filtering by keywords and saving mel-spectrograms.

    Args:
        folder_path (str): Path to the folder containing audio files.
        output_folder (str): Path to the folder to save precomputed tensors.
        keywords (list of str): List of keywords to filter filenames.
        sr (int): Sample rate for the audio files.
    """
    # Expand the list of keywords using WordNet
    expanded_keywords = expand_keywords(keywords)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all .wav files in the input folder
    files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.wav')]
    
    # Create a list of tasks for parallel processing
    tasks = [(file_name, folder_path, output_folder, sr, expanded_keywords) for file_name in files]
    
    # Process files in parallel using multiprocessing
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_file, tasks), total=len(tasks)))


input_folder = '/backup/data3/shivam/audio-visual-dataset/audio_balanced/audio/'
output_folder = '/backup/data3/shivam/audio-visual-dataset/balanced_precomputed_mel'
keywords = ["laugh", "giggling", "sob", "sobbing", "nose", "nose blow",
            "laughter", "snore", "snoring", "smile", "sneeze", "cry",
            "cough", "scream", "explode", "explosion", "burp", "burping",
            "eructation", "hiccup", "sing", "choir", "whistle", "whistling",
            "whoop", "chatter", "whimper", "breathing", "grunt", "grunting",
            "throat", "sigh", "gurgling", "humming", "beatboxing", ]
    
preprocess_and_save(input_folder, output_folder, keywords)