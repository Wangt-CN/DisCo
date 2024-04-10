from skimage.metrics import structural_similarity as ssim_eval
import numpy as np
import cv2
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms
import lpips
from math import log10, sqrt
  

def psnr_eval(original, compressed):

    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def load_image(path):
    if os.path.exists(path):
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
    else:
        image = None
    return image


def l1_eval(imageA, imageB):
    err = np.absolute((imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def compute_ssim_l1_psnr(gen_inst_name_full, gt_inst_name_full, mode):
    gen_inst_name_full = sorted(gen_inst_name_full)
    gt_inst_name_full = sorted(gt_inst_name_full)

    scores = []
    for gen_path, gt_path in tqdm(zip(gen_inst_name_full, gt_inst_name_full)):   
        gen_filename = os.path.splitext(
                os.path.basename(gen_path))[0]
        gt_filename = os.path.splitext(
                os.path.basename(gt_path))[0]

        assert gen_filename == gt_filename, f"file mismatch: {gen_filename} vs {gt_filename}"

        image1 = cv2.imread(gen_path)
        image2 = cv2.imread(gt_path)
        if image1 is None:
            print(f"Failed to load image from {gen_path}")
            # Check if gen_path is a symlink
            if os.path.islink(gen_path):
                target_path = os.readlink(gen_path)
                print(f"The symbolic link {gen_path} points to {target_path}")
            else:
                print(f"{gen_path} is not a symbolic link.")
            return None

        # Resize images to the same dimensions, if needed
        if image1.shape != image2.shape:
            image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    
        if mode == 'L1':
            score = l1_eval(image1, image2)
        elif mode == 'PSNR':
            score = psnr_eval(image1, image2)
        else:
            # ssim_value = ssim(image1, image2, multichannel=True)
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

            # Compute SSIM between two images
            (score, _) = ssim_eval(image1_gray, image2_gray, full=True)
        scores.append(score)

    score_ave = np.mean(scores)
    return score_ave


def compute_lpips(gen_inst_name_full, gt_inst_name_full):
    gen_inst_name_full = sorted(gen_inst_name_full)
    gt_inst_name_full = sorted(gt_inst_name_full)
    convert_tensor = transforms.ToTensor()
    loss_fn_vgg = lpips.LPIPS(net='vgg')

    scores = []
    for gen_path, gt_path in tqdm(zip(gen_inst_name_full, gt_inst_name_full)):   
        gen_filename = os.path.splitext(
                os.path.basename(gen_path))[0]
        gt_filename = os.path.splitext(
                os.path.basename(gt_path))[0]

        assert gen_filename == gt_filename, 'file mismatch'

        image1 = convert_tensor(load_image(gen_path)).unsqueeze(0)
        image2 = convert_tensor(load_image(gt_path)).unsqueeze(0)
    
        score = loss_fn_vgg(image1, image2).item()
        
        scores.append(score)

    score_ave = np.mean(scores)
    return score_ave