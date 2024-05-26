import cv2
import time
import imageio
import numpy as np

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)
    writer = imageio.get_writer("sample.mp4", mode='I', fps=25)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
        writer.append_data(np.array(frame))
        # frames.append(frame)
    cap.release()
    writer.close()
    
tic = time.time()
extract_frames('/data/video_samples/Cw4jVLXshd8_"Vehicle, Car".mp4')
print(time.time() - tic)