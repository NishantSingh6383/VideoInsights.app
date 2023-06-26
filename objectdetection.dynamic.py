import cv2
import numpy as np
import os

# Load YOLOv4 model
net = cv2.dnn.readNet("C:/Users/admin/OneDrive/Desktop/pro1/darknet/yolov4.weights","C:/Users/admin/OneDrive/Desktop/pro1/darknet/yolov4.cfg.cfg" )

def detect_objects(frame):
    # Detect objects in the frame using YOLOv4
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Extract relevant information from the detected objects
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    return class_ids, confidences, boxes

def save_video_with_objects(frames, output_path, frame_rate):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change codec as needed
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    for frame in frames:
        class_ids, _, _ = detect_objects(frame)
        if len(class_ids) > 0:
            video_writer.write(frame)

    video_writer.release()

def extract_frames(video_path, frame_rate):
    frames = []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    desired_frames = int(total_frames / frame_rate)

    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        if i % frame_rate == 0:
            frames.append(frame)
        if len(frames) == desired_frames:
            break

    video.release()
    return frames

def summarize_video(video_path, frame_rate, output_path):
    frames = extract_frames(video_path, frame_rate)

    # Create the output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_video_with_objects(frames, output_path, frame_rate)
    print(f"Video summarization completed. {len(frames)} frames processed.")

# Example usage
video_path = 'v21.mpg'
frame_rate = 20  # Extract one frame every 20 seconds
output_path = "C:/Users/admin/OneDrive/Desktop/VideoSum/dynamicobject.summary.mp4"
summarize_video(video_path, frame_rate, output_path)