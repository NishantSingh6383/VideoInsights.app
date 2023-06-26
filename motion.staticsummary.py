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

def apply_kmeans(frames, motion_features, k):
    features = np.array(motion_features).reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
    labels = kmeans.labels_
    key_frames = []
    for i in range(k):
        index = np.where(labels == i)[0]
        if len(index) > 0:
            key_frames.append(frames[index[0]])
    return np.array(key_frames)

def create_static_summary(key_frames, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, frame in enumerate(key_frames):
        cv2.imwrite(os.path.join(output_path, f"frame_{i}.jpg"), frame)

    print(f"Video summarization completed. {len(key_frames)} key frames saved in {output_path}")

def summarize_video(video_path, frame_rate, k, max_frames, output_path):
    frames = extract_frames(video_path, frame_rate)
    if len(frames) < max_frames:
        print(f"Warning: Number of frames extracted ({len(frames)}) is less than the requested number of frames ({max_frames}). Using only {len(frames)} frames.")
        max_frames = len(frames)
    motion_features = extract_motion(frames[:max_frames])
    key_frames = apply_kmeans(frames[:max_frames], motion_features, k)
    create_static_summary(key_frames, output_path)

if __name__ == '__main__':
    video_path = 'v21.mpg'
    frame_rate = 10  # Extract one frame every n seconds
    k = 5  # Number of clusters to create
    max_frames = 100  # Maximum number of frames to use for clustering
    output_path = 'C:/Users/admin/OneDrive/Desktop/VideoSum/motionstatic_summary'

    summarize_video(video_path, frame_rate, k, max_frames, output_path)
