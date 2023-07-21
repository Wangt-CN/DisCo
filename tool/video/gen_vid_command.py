import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Video')
    parser.add_argument('--img_path', default=None, type=str, help='image frame path')
    args = parser.parse_args()

    video_img_source_path = args.img_path
    fps = 30
    size = (256, 256)
    codec = cv2.VideoWriter_fourcc(*"mp4v")

    # Initialize video writer
    video_writer = cv2.VideoWriter("{}_vid_output.mp4".format(video_img_source_path), codec, fps, size)
    folder_path = video_img_source_path
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    # Filter the list to only include image filesd
    image_files = [file for file in files if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
    num_frm = len(image_files)

    for image_fname_ in image_files:
        image_fname = '{}/{}'.format(folder_path, image_fname_)
        frame = cv2.imread(image_fname)
        # Write frames to video
        video_writer.write(frame)

    # Release resources
    video_writer.release()