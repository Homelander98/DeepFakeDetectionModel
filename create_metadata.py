import os
import csv

# Define paths to real and fake videos
real_videos_dir = "data/Celeb-real"
fake_videos_dir = "data/Celeb-synthesis"

# Output metadata file
metadata_file = "data/dataset.csv"

# Create metadata
with open(metadata_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["video_path", "label"])  # Write header

    # Add real videos (files like 'id0_0000.mp4')
    for video_name in os.listdir(real_videos_dir):
        if video_name.endswith(".mp4") and "_" in video_name:  # Ensure it's a video file
            video_path = os.path.join(real_videos_dir, video_name)
            writer.writerow([video_path, 0])  # Label 0 for real videos

    # Add fake videos (files like 'id0_id1_0000.mp4')
    for video_name in os.listdir(fake_videos_dir):
        if video_name.endswith(".mp4") and "_" in video_name:  # Ensure it's a video file
            video_path = os.path.join(fake_videos_dir, video_name)
            writer.writerow([video_path, 1])  # Label 1 for fake videos

print(f"Metadata file created: {metadata_file}")