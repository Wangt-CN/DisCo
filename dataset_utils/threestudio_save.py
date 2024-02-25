import os
from PIL import Image
import cv2
import imageio
import argparse
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    return args
       
def save_img_sequence(
    img_dir,
    save_path,
    save_format="gif",
    fps=25,
) -> str:
    assert save_format in ["gif", "mp4"]
    imgs = os.listdir(img_dir)
    imgs = [Image.open(os.path.join(img_dir, f)) for f in imgs]

    if save_format == "gif":
        imgs = [np.array(i) for i in imgs]
        imageio.mimsave(save_path, imgs, fps=fps, palettesize=256)
    elif save_format == "mp4":
        imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
        imageio.mimsave(save_path, imgs, fps=fps)
    return save_path

if __name__ == "__main__":
    args = parse_args()
    save_img_sequence(args.img_dir, args.save_path)