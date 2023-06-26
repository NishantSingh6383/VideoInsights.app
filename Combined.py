import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def extract_frames(video_path, frame_rate):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if count % frame_rate == 0:
            frames.append(image)
        count += 1
    vidcap.release()
    return np.array(frames)

def extract_motion(frames):
    motion_features = []
    prev_frame = frames[0]
    for frame in frames[1:]:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_feature = np.mean(magnitude)
        motion_features.append(motion_feature)
        prev_frame = frame
    return np.array(motion_features)

def apply_kmeans_motion(frames, motion_features, k):
    features = np.array(motion_features).reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
    labels = kmeans.labels_
    key_frames = []
    for i in range(k):
        index = np.where(labels == i)[0]
        if len(index) > 0:
            key_frames.append(frames[index[0]])
    return np.array(key_frames)

def extract_histograms(frames):
    histograms = []
    for frame in frames:
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    return histograms

def apply_kmeans_color(histograms, k):
    samples = np.array(histograms)
    samples = samples.reshape(-1, samples.shape[-1])
    kmeans = KMeans(n_clusters=k, random_state=42).fit(samples)
    labels = kmeans.labels_
    key_frame_indices = []
    for i in range(k):
        index = np.where(labels == i)[0]
        if len(index) > 0:
            key_frame_indices.append(index[0])
    return np.array(key_frame_indices)

def detect_events(motion_features, threshold):
    events = []
    event_start = 0
    event_end = 0
    for i in range(len(motion_features)):
        if motion_features[i] > threshold:
            if event_start == 0:
                event_start = i
            event_end = i
        else:
            if event_start != 0:
                events.append((event_start, event_end))
                event_start = 0
                event_end = 0
    if event_start != 0:
        events.append((event_start, event_end))
    return events

def extract_event_frames(frames, events):
    event_frames = []
    for event in events:
        start_frame = event[0]
        end_frame = event[1]
        for i in range(start_frame, end_frame+1):
            event_frames.append(frames[i])
    return np.array(event_frames)

def create_event_summary(event_frames, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, frame in enumerate(event_frames):
        cv2.imwrite(os.path.join(output_path, f"frame_{i}.jpg"), frame)

    print(f"Video summarization completed. {len(event_frames)} event frames saved in {output_path}")

def summarize_video(video_path, frame_rate, k_motion, k_color, threshold, max_frames, output_path):
    frames = extract_frames(video_path, frame_rate)[:max_frames]
    motion_features = extract_motion(frames)
    key_frames_motion = apply_kmeans_motion(frames, motion_features, k_motion)
    histograms = extract_histograms(frames)
    key_frame_indices_color = apply_kmeans_color(histograms, k_color)
    key_frames_color = frames[key_frame_indices_color]
    events = detect_events(motion_features, threshold)
    event_frames = extract_event_frames(frames, events)
    create_event_summary(event_frames, output_path)

    print(f"Video summarization completed. Key frames based on motion: {len(key_frames_motion)}. Key frames based on color: {len(key_frames_color)}. Event frames: {len(event_frames)}.")

if __name__ == '__main__':
    video_path = 'v21.mpg'
    frame_rate = 10  # Extract one frame every n seconds
    k_motion = 5  # Number of motion-based key frames
    k_color = 10  # Number of color-based key frames
    threshold = 5  # Motion threshold to detect events
    max_frames = 1000  # Maximum number of frames to process
    output_path = 'C:/Users/admin/OneDrive/Desktop/VideoSum/static_combined_summary'

    summarize_video(video_path, frame_rate, k_motion, k_color, threshold, max_frames, output_path)

