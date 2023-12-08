import argparse
import re
from pathlib import Path

import cv2
from tqdm import tqdm


def assemble_frames_to_videos(input_folder_path, output_folder_path, fps=30, interval=8):
    # Define the two formats
    folder_format1 = r"^(TiktokDance_\d+_)(\d+)(\D+)$"
    folder_format2 = r"^(TiktokDance_\d+_\d+_1x1_)(\d+)(\D+)$"
    
    frame_format1 = r"^(TiktokDance_\d+_)(\d+)(\D*\.\w+)$"
    frame_format2 = r"^(TiktokDance_\d+_\d+_1x1_)(\d+)(\D*\.\w+)$"

    # Create a Path object
    input_path = Path(input_folder_path)
    assert input_path.exists(), f"Input folder '{input_folder_path}' does not exist"
    output_path = Path(output_folder_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # Use a dictionary to group frame file paths by video names
    frames_dict = {}
    subfolders = sorted(list(input_path.glob("*")))

    for i, folder in tqdm(enumerate(subfolders), desc="reading folders", total=len(subfolders)):
        if not folder.is_dir():
            print(f"ignoring non-folder file {folder.name}")
            continue
        if re.match(folder_format1, folder.name):
            folder_format_regex = folder_format1
            match1 = True
        elif re.match(folder_format2, folder.name):
            folder_format_regex = folder_format2
            match1 = False
        else:
            print(f"Folder name '{folder.name}' does not match any format")
            continue

        match = re.match(folder_format_regex, folder.name)
        video_name = match.group(1)[:-1]  # Remove the trailing underscore
        folder_index = int(match.group(2))

        end_of_video = False
        if i == len(subfolders) - 1:
            end_of_video = True
        elif match1 and subfolders[i + 1].name[:18] != folder.name[:18]:
            end_of_video = True
        elif not match1 and subfolders[i + 1].name[:23] != folder.name[:23]:
            end_of_video = True

        if match1 and (folder_index - 1) % interval != 0 and not end_of_video:  # assume folder_index starts from 1 if match1
            print(f"skipping folder {folder.name}")
            continue
        if not match1 and folder_index % interval != 0 and not end_of_video:  # assume folder_index starts from 0 if not match1
            print(f"skipping folder {folder.name}")
            continue

        if video_name not in frames_dict:
            frames_dict[video_name] = []
        for frame in folder.glob("*"):
            if re.match(frame_format1, frame.name):
                frame_format_regex = frame_format1
            elif re.match(frame_format2, frame.name):
                frame_format_regex = frame_format2
            else:
                print(f"Frame name '{frame.name}' does not match any format")
                continue

            match = re.match(frame_format_regex, frame.name)
            frame_index = int(match.group(2))
            if frame_index >= len(frames_dict[video_name]):
                frames_dict[video_name].append((frame_index, frame))
            else:
                frames_dict[video_name][frame_index] = (frame_index, frame)
    
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
    parser.add_argument("--interval", default=7, type=int, help="Interval between subfolders")

    args = parser.parse_args()

    args.input_folder = args.input_folder.rstrip("/")
    if args.output_folder == "":
        args.output_folder = args.input_folder + "_videos"

    assemble_frames_to_videos(args.input_folder, args.output_folder, args.fps, args.interval)
