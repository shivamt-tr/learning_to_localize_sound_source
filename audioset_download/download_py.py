import csv
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import yt_dlp as youtube_dl
import subprocess
from pathlib import Path


def download_and_process_video(row, outdir, ffmpeg_path, faster):
    ytid, start, end, label = row
    filename = f"{ytid}_{label}.mp4"
    filepath = Path(outdir) / filename

    if filepath.exists():
        print(f"{filename} already exists. Skipping...")
        return

    try:
        # Define download options
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' if faster > 0 else 'best',
            'outtmpl': str(filepath),
            'quiet': True,
            'no_warnings': True,
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'http://www.youtube.com/watch?v={ytid}'])

        # Cut the video
        start_time = float(start)
        duration = float(end) - start_time if end else 10  # Default duration is 10 seconds

        output_path = str(filepath) + '.tmp.mp4'
        if faster <= 1:
            cmd = [ffmpeg_path, '-ss', str(start_time), '-i', str(filepath), '-t', str(duration),
                   '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                   '-c:a', 'aac', '-b:a', '128k', '-strict', 'experimental', output_path]
        else:
            cmd = [ffmpeg_path, '-ss', str(start_time), '-i', str(filepath), '-t', str(duration),
                   '-c', 'copy', output_path]

        subprocess.run(cmd, check=True)
        os.rename(output_path, str(filepath))
        print(f"Processed {filename}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")

def main(csv_file, outdir, ffmpeg_path, njobs=4, faster=1):
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    with ThreadPoolExecutor(max_workers=njobs) as executor:
        futures = [executor.submit(download_and_process_video, row, outdir, ffmpeg_path, faster) for row in rows]
        for future in futures:
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and process videos with custom ffmpeg path.')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file containing video info')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for processed videos')
    parser.add_argument('--ffmpeg_path', type=str, required=True)
    # parser.add_argument('--ffprobe_path', type=str, required=True)

    args = parser.parse_args()

    njobs = 20  # Adjust based on your system
    faster = 1  # Adjust based on your preference

    main(args.csv_file, args.outdir, args.ffmpeg_path, njobs, faster)
