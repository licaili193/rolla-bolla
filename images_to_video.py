import cv2
import os
import re

import os
import re
import cv2

def images_to_video(folder_path, output_video, frame_rate=30):
    """
    Convert a sequence of PNG images in a folder to a video.

    Args:
        folder_path (str): The path to the folder containing the PNG images.
        output_video (str): The path to the output video file.
        frame_rate (int, optional): The frame rate of the output video. Defaults to 30.

    Returns:
        None
    """
    # Get all PNG files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # Sort files by frame number
    image_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))

    # Get the dimensions of the first image
    first_image = cv2.imread(os.path.join(folder_path, image_files[0]))
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Read and write each image to the video
    for filename in image_files:
        img = cv2.imread(os.path.join(folder_path, filename))
        out.write(img)
        print(f"Added {filename} to the video.")

    # Release the VideoWriter object
    out.release()
    print("Video creation complete.")

if __name__ == '__main__':
    folder_path = 'combined_frames'
    output_video = 'output_video.mp4'
    frame_rate = 50  # Adjust as needed
    images_to_video(folder_path, output_video, frame_rate)
