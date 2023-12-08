import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


def copy_files(input_folder, output_folder, use_symlink=False, selected_frame_idx: int = 0):
    print("Copying files from {} to {}".format(input_folder, output_folder))
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # Ensure output directory exists
    output_folder.mkdir(parents=True, exist_ok=True)
    # try:
    #     print(f"Copying files from {input_folder.absolute()} to {output_folder.absolute()}")
    # except Exception as e:
    #     print(e)

    # Collect all subfolders
    subfolders = list(input_folder.iterdir())

    # Collect and copy or link files from each subfolder
    for subfolder in tqdm(subfolders, desc="Copying files", unit="subfolder"):
        if subfolder.is_dir():
            file_list = sorted(list(subfolder.iterdir()))
            if selected_frame_idx != -1:  # -1 means use all frames
                selected_frame_idx = max(min(selected_frame_idx, len(file_list) - 1), 0)

            for idx, file in enumerate(file_list):
                if idx != selected_frame_idx:
                    continue

                # Skip files of 0 size
                if file.stat().st_size > 0:
                    if selected_frame_idx == -1:  # -1 means use all frames
                        output_file_path = output_folder / (subfolder.name + "-" + file.name)
                    else:
                        output_file_path = output_folder / file.name

                    if use_symlink:
                        if not output_file_path.exists():  # Check if file doesn't exist
                            output_file_path.symlink_to(file.resolve())  # Using absolute path here
                    else:
                        shutil.copy2(file, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="Input folder path containing subfolders with files",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="",
        help="Output folder path where files will be copied or linked",
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        default=False,
        help="If set, creates symbolic links instead of copying files",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="The frame index to be copied from each subfolder",
    )

    args = parser.parse_args()

    # Remove trailing slash from input_folder if it exists
    args.input_folder = args.input_folder.rstrip("/")

    # Set default output folder if not provided
    if args.output_folder == "":
        args.output_folder = args.input_folder + f"_{args.frame_idx}nd_frames"

    copy_files(args.input_folder, args.output_folder, args.soft, args.frame_idx)
