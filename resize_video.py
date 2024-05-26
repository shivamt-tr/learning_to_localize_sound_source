import os
import cv2
import imageio
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import soundfile as sf
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("ignore")

sample_rate = 22050  # for SoundNet model input
target_length = 220672


def process_video(args):

    video_file, data_dir, save_dir = args
    video_path = os.path.join(data_dir, video_file)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    try:
        # resize video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Error opening video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = imageio.get_writer(os.path.join(save_dir, "resized_video", f"{video_name}.mp4"), mode='I', fps=fps)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
            writer.append_data(np.array(frame))

        cap.release()
        writer.close()

        # extract audio (the range of values is [-1, 1])
        audio, _ = librosa.load(video_path, sr=sample_rate, mono=True)  # audio = (220672,), 22.05kHz and mono for input to SoundNet
        np.clip(audio, -1, 1, out=audio)  # clip the samples to be in the range [-1, 1]
        audio = np.tile(audio, 10)[:target_length]  # repeat the audio samples and select a fixed size to maintain consistency across different length audio

        # if audio length is less than the target length, pad with zeros
        if audio.shape[0] < target_length:
            audio = np.pad(audio, (0, target_length - audio.shape[0]), 'constant')

        sf.write(os.path.join(save_dir, "audio", f"{video_name}.wav"), audio, sample_rate)
    except Exception as e:
        return f"Error in processing video: {video_path}, Error: {e}"

    return None


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'resized_video'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'audio'), exist_ok=True)

    video_files = [f for f in os.listdir(args.data_dir) if f.endswith(".mp4")]
    pool_args = [(video_file, args.data_dir, args.save_dir) for video_file in video_files]

    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_video, pool_args), total=len(video_files), desc="Processing"))

    error_count = sum(1 for result in results if result is not None)
    for result in results:
        if result is not None:
            print(result)

    print(f"Total number of files: {len(video_files)}, and {error_count} files could not be processed")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    args = parser.parse_args()

    main(args)