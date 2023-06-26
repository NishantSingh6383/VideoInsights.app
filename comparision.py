import os

# Provide the paths to the output directories of the three techniques
color_output_path = "C:/Users/admin/OneDrive/Desktop/VideoSum/colorsummary.static"
motion_output_path = "C:/Users/admin/OneDrive/Desktop/VideoSum/motionstatic_summary"
event_output_path = "C:/Users/admin/OneDrive/Desktop/VideoSum/eventstat_summary"

# Count the number of frames in each output folder
color_frames = len(os.listdir(color_output_path))
motion_frames = len(os.listdir(motion_output_path))
event_frames = len(os.listdir(event_output_path))

# Calculate the ratios
total_frames = color_frames + motion_frames + event_frames
color_ratio = color_frames / total_frames
motion_ratio = motion_frames / total_frames
event_ratio = event_frames / total_frames

print("color_ratio : ", color_ratio)
print("motion_ratio : ", motion_ratio)
print("event_ratio : ", event_ratio)

# Compare the ratios
if color_ratio > motion_ratio and color_ratio > event_ratio:
    print("Color Technique is more effective.")
elif motion_ratio > color_ratio and motion_ratio > event_ratio:
    print("Motion Technique is more effective.")
elif event_ratio > color_ratio and event_ratio > motion_ratio:
    print("Event Technique is more effective.")
else:
    print("Multiple techniques have the same effectiveness.")