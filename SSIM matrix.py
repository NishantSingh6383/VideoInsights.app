import os
import cv2
from skimage.metrics import structural_similarity as compare_ssim


def calculate_frame_similarity(frame1_path, frame2_path):
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    ssim = compare_ssim(frame1_gray, frame2_gray)
    return ssim


def compare_outputs(folder1, folder2):
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    num_files = min(len(files1), len(files2))
    similarity_sum = 0

    for i in range(num_files):
        image1_path = os.path.join(folder1, files1[i])
        image2_path = os.path.join(folder2, files2[i])
        similarity = calculate_frame_similarity(image1_path, image2_path)
        similarity_sum += similarity

    average_similarity = similarity_sum / num_files
    return average_similarity


output_folder_event = "C:/Users/admin/OneDrive/Desktop/VideoSum/eventstat_summary"
output_folder_motion ="C:/Users/admin/OneDrive/Desktop/VideoSum/motionstatic_summary"
output_folder_color ="C:/Users/admin/OneDrive/Desktop/VideoSum/colorsummary.static"



# Compare event and motion techniques
similarity_event_motion = compare_outputs(output_folder_event, output_folder_motion)
print("Similarity between event and motion techniques:", similarity_event_motion)

# Compare event and color techniques
similarity_event_color = compare_outputs(output_folder_event, output_folder_color)
print("Similarity between event and color techniques:", similarity_event_color)


similarity_motion_color = compare_outputs(output_folder_motion, output_folder_color)
print("Similarity between motion and color techniques:", similarity_motion_color)


if similarity_event_motion > similarity_event_color and similarity_event_motion > similarity_motion_color:
    print("Event technique is better.")
elif similarity_event_color > similarity_event_motion and similarity_event_color > similarity_motion_color:
    print("Color technique is better.")
else:
    print("Motion technique is better.")