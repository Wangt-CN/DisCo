import argparse
import re
from pathlib import Path

import cv2
from tqdm import tqdm


def assemble_frames_to_videos(input_folder_path, output_folder_path, fps=30):
    # Define the two formats
    format1 = r"^(TiktokDance_\d+_)(\d+)(\D+\.\w+)$"
    format2 = r"^(TiktokDance_\d+_\d+_1x1_)(\d+)(\D+\.\w+)$"
    format3 = r"^(S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}_)(\d{4})(\D*\.\w+)$"

    # Create a Path object
    input_path = Path(input_folder_path)
    assert input_path.exists(), f"Input folder '{input_folder_path}' does not exist"
    output_path = Path(output_folder_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # Use a dictionary to group frame file paths by video names
    frames_dict = {}

    # Iterate over image files in the directory
    for img_file in input_path.glob("*"):
        # Use the regex that matches the file name
        if re.match(format1, img_file.name):
            format_regex = format1
        elif re.match(format2, img_file.name):
            format_regex = format2
        elif re.match(format3, img_file.name):
            format_regex = format3
        else:
            print(f"File name '{img_file.name}' does not match any format")
            continue  # skip if the file name does not match either format

        match = re.match(format_regex, img_file.name)
        video_name = match.group(1)[:-1]  # Remove the trailing underscore
        frame_index = int(match.group(2))

        # Add the frame file path to the corresponding group
        if video_name not in frames_dict:
            frames_dict[video_name] = []
        frames_dict[video_name].append((frame_index, img_file))
    
    print(f"Found {len(frames_dict)} videos")

    # Iterate over groups to create videos
    for video_name, frames in tqdm(frames_dict.items(), desc="Assembling videos"):
        # Sort frames by index
        frames.sort(key=lambda x: x[0])

        # Load the first frame to get the frame size
        frame0 = cv2.imread(str(frames[0][1]))
        frame_size = (frame0.shape[1], frame0.shape[0])

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )  # 'mp4v' is a codec used for .mp4 videos
        output_video_path = output_path / f"{video_name}.mp4"
        video_out = cv2.VideoWriter(str(output_video_path), fourcc, fps, frame_size)

        # Write frames to the video
        for _, img_file in tqdm(
            frames, desc=f"Writing frames for video '{video_name}'"
        ):
            frame = cv2.imread(str(img_file))
            video_out.write(frame)

        # Release the VideoWriter
        video_out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble frames into videos")
    parser.add_argument(
        "-i",
        "--input_folder",
        required=True,
        type=str,
        help="Input folder path containing frames",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        default="",
        type=str,
        help="Output folder path for videos",
    )
    parser.add_argument(
        "--fps", default=30, type=int, help="Frames per second for the output videos"
    )

    args = parser.parse_args()

    args.input_folder = args.input_folder.rstrip("/")
    if args.output_folder == "":
        args.output_folder = args.input_folder + "_videos"

    assemble_frames_to_videos(args.input_folder, args.output_folder, args.fps)
