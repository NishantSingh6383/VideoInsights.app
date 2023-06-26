import cv2
import numpy as np
import os

def extract_frames(video_path, frame_rate):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames[::frame_rate]

def extract_histograms(frames):
    histograms = []
    for frame in frames:
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    return histograms

def apply_kmeans(histograms, k):
    samples = np.array(histograms)
    samples = samples.reshape(-1, samples.shape[-1])
    _, labels, centers = cv2.kmeans(samples, k, None, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), attempts=10, flags=cv2.KMEANS_PP_CENTERS)
    key_frame_indices = []
    for i in range(k):
        index = np.where(labels == i)[0]
        if len(index) > 0:
            key_frame_indices.append(index[0])
    return key_frame_indices

def save_frames(frames, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, frame in enumerate(frames):
        cv2.imwrite(output_path + '/frame{}.jpg'.format(i), frame)

def summarize_video(video_path, frame_rate, k, max_frames, output_path):
    frames = extract_frames(video_path, frame_rate)[:max_frames]
    if len(frames) < max_frames:
        print(f"Warning: Number of frames extracted ({len(frames)}) is less than the requested number of frames ({max_frames}). Using only {len(frames)} frames.")
        max_frames = len(frames)
    histograms = extract_histograms(frames)
    key_frame_indices = apply_kmeans(histograms, k)
    key_frames = [frames[i] for i in key_frame_indices]
    save_frames(key_frames, output_path)
    print(f"Video summarization completed. {len(key_frames)} key frames saved in {output_path}")

video_path = 'v21.mpg'
frame_rate = 3
k = 20
max_frames = 1000
output_path = 'C:/Users/admin/OneDrive/Desktop/VideoSum/colorsummary.static'

summarize_video(video_path, frame_rate, k, max_frames, output_path)
