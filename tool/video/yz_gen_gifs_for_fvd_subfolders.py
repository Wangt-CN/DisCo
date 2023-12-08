import argparse
import threading
from pathlib import Path

import imageio
from tqdm import tqdm


def process_subfolder(subfolder, output_folder, fps, pbar, semaphore, format):
    # Decrease the semaphore count
    with semaphore:
        # Initialize empty frames list
        frames = []

        # Read the frames in alphabetical order
        png_files = sorted(Path(subfolder).glob("*.png"))
        jpg_files = sorted(Path(subfolder).glob("*.jpg"))
        frames_path = png_files + jpg_files

        for frame_path in frames_path:
            try:
                frames.append(imageio.imread(str(frame_path)))
            except Exception as e:
                print(f"Couldn't read file {frame_path} because: {str(e)}")
                continue

        # Skip this subfolder if no frames could be read
        if not frames:
            print(f"No frames could be read from {subfolder.name}")
            return

        # Create output video
        output_path = Path(output_folder, f"{subfolder.name}.{format}")
        imageio.mimwrite(output_path, frames, fps=fps)

        pbar.set_description(
            f"Processed {subfolder.name}, which contains {len(frames)} frames"
        )
        pbar.update(1)


def create_gif_from_folder(input_folder, output_folder, fps, num_workers, format):
    # Ensure the output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Collect subfolders
    subfolders = [sf for sf in Path(input_folder).iterdir() if sf.is_dir()]

    # Initialize tqdm progress bar
    pbar = tqdm(total=len(subfolders), desc="Creating GIFs")

    # Create a thread for each subfolder
    threads = []
    semaphore = threading.Semaphore(num_workers)
    for subfolder in subfolders:
        thread = threading.Thread(
            target=process_subfolder,
            args=(subfolder, output_folder, fps, pbar, semaphore, format),
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="Input folder path containing subfolders with frames",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default=None,
        help="Output folder path to save generated gifs. If not specified, '_gif' is appended to the input folder name.",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        default=3,
        help="Frames per second for the generated GIFs",
    )
    parser.add_argument(
        "-n", "--num_workers", type=int, default=16, help="Number of worker threads"
    )
    parser.add_argument("--format", type=str, default="mp4", choices=["gif", "mp4"])
    args = parser.parse_args()

    # If output folder is not specified, append '_gif' to the input folder name
    args.input_folder = args.input_folder.rstrip("/")
    if args.output_folder is None:
        args.output_folder = str(Path(args.input_folder).resolve()) + f"_{args.format}"

    create_gif_from_folder(
        args.input_folder, args.output_folder, args.fps, args.num_workers, args.format
    )
