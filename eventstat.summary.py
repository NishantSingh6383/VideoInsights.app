import cv2
import numpy as np
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

def summarize_video(video_path, frame_rate, threshold, output_path):
    frames = extract_frames(video_path, frame_rate)
    motion_features = extract_motion(frames)
    events = detect_events(motion_features, threshold)
    event_frames = extract_event_frames(frames, events)
    create_event_summary(event_frames, output_path)

if __name__ == '__main__':
    video_path = 'vid00.mp4'
    frame_rate = 10  # Extract one frame every n seconds
    threshold = 5  # Motion threshold to detect events
    output_path = 'C:/Users/admin/OneDrive/Desktop/VideoSum/eventstat_summary'

    summarize_video(video_path, frame_rate, threshold, output_path)
