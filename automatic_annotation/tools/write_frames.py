import cv2
import os

# Define paths
video_path = os.path.join("videos", "Fog-video.mp4")
output_dir = os.path.join("output_frames")

# Read video
cam = cv2.VideoCapture(video_path)

# Get video FPS and duration
fps = cam.get(cv2.CAP_PROP_FPS)
total_frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)

# Safe duration calculation
duration_secs = total_frames / max(fps, 1)

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Saving frames to:", os.path.abspath(output_dir))

# Configuration
frame_interval = 3  # seconds → 1 frame every 3 seconds

frame_index = 0
current_time = 0

# Traverse ENTIRE video
while current_time < duration_secs:
    cam.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
    ret, frame = cam.read()
    if not ret:
        break

    filename = os.path.join(output_dir, f'frame{frame_index}.jpg')
    print(f'Creating... {filename}')
    cv2.imwrite(filename, frame)

    frame_index += 1
    current_time += frame_interval

# Release resources
cam.release()
cv2.destroyAllWindows()
