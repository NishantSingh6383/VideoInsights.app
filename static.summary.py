#simplesummary
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def extract_frames(video_path, frame_rate, max_frames):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if count % frame_rate == 0:
            frames.append(image)
            if len(frames) >= max_frames:
                break
        count += 1
    vidcap.release()
    return frames

def apply_kmeans(frames, k):
    X = np.array(frames).reshape(-1, frames[0].shape[0]*frames[0].shape[1]*frames[0].shape[2])
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.predict(X)
    key_frames = []
    for i in range(k):
        index = np.where(labels == i)[0]
        if len(index) > 0:
            key_frames.append(frames[index[0]])
    return np.array(key_frames)

def copy_key_frames(key_frames, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, frame in enumerate(key_frames):
        cv2.imwrite(os.path.join(output_path, f"frame_{i}.jpg"), frame)

def summarize_video(video_path, frame_rate, k, max_frames, output_path):
    frames = extract_frames(video_path, frame_rate, max_frames)
    print(f"Total frames extracted: {len(frames)}")
    if len(frames) < max_frames:
        print(f"Warning: Number of frames extracted ({len(frames)}) is less than the requested number of frames ({max_frames}). Using only {len(frames)} frames.")
    key_frames = apply_kmeans(frames, k)
    copy_key_frames(key_frames, output_path)
    print(f"Video summarization completed. {len(key_frames)} key frames saved in {output_path}")

summarize_video('v21.mpg', 10, 5, 100, 'C:/Users/admin/OneDrive/Desktop/VideoSum/statsumm')
