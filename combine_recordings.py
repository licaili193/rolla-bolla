import cv2
import os
import re
import numpy as np
import math

def combine_videos_in_grid(folder_path, output_folder, max_dimension=1280):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all mp4 files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    # Sort files by episode number
    video_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))
    print(f"Found {len(video_files)} videos to combine.")

    # Determine grid size
    num_videos = len(video_files)
    grid_cols = math.ceil(math.sqrt(num_videos))
    grid_rows = math.ceil(num_videos / grid_cols)

    # Load videos and prepare for concatenation
    clips = []
    max_width = 0
    max_height = 0
    for filename in video_files:
        cap = cv2.VideoCapture(os.path.join(folder_path, filename))
        if not cap.isOpened():
            print(f"Error opening video file: {filename}")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        clips.append((cap, width, height))
    print("Loaded all videos.")

    # Adjust the dimensions based on the grid size
    final_width = max_width * grid_cols
    final_height = max_height * grid_rows

    # Calculate scaling factor to constrain dimensions
    scaling_factor = min(max_dimension / final_width, max_dimension / final_height)

    # Adjust the dimensions based on the scaling factor
    final_width = int(final_width * scaling_factor)
    final_height = int(final_height * scaling_factor)

    # Write text and concatenate frames
    print("Combining videos...")
    frame_count = 0
    while True:
        grid_frames = []
        for row in range(grid_rows):
            row_frames = []
            for col in range(grid_cols):
                idx = row * grid_cols + col
                if idx < len(clips):
                    cap, width, height = clips[idx]
                    ret, frame = cap.read()
                    if not ret:
                        for cap, _, _ in clips:
                            cap.release()
                        print("Combination complete.")
                        return
                    # Add text to the frame
                    if idx == len(clips) - 1:  # Check if it's the last clip
                        text = "recording_final"
                    else:
                        episode_number = re.search(r'(\d+)', video_files[idx]).group()
                        text = f"Episode {episode_number}"
                    cv2.putText(frame, text, (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # Resize frame based on the scaling factor
                    resized_frame = cv2.resize(frame, (int(width * scaling_factor), int(height * scaling_factor)))
                    row_frames.append(resized_frame)
                else:
                    # Fill empty spaces in the grid with white frames
                    row_frames.append(np.ones((int(max_height * scaling_factor), int(max_width * scaling_factor), 3), dtype=np.uint8) * 255)
            # Concatenate frames horizontally to form a row
            row_frame = np.concatenate(row_frames, axis=1)
            grid_frames.append(row_frame)
        # Concatenate rows vertically to form the grid
        final_frame = np.concatenate(grid_frames, axis=0)
        cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count + 1:05d}.png'), final_frame)
        print(f"Saved frame {frame_count + 1}")
        frame_count += 1

if __name__ == '__main__':
    folder_path = 'recordings'
    output_folder = 'combined_frames'
    combine_videos_in_grid(folder_path, output_folder)
