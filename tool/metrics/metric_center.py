# -*- encoding: utf-8 -*-
'''
@Time        :2021/10/13 15:32:28
@Author      :Yongfei Liu
@Email       :liuyf3@shanghaitech.edu.cn


## support FID-K(Gassian Blur), IS, FID-Img(from video to image), FID-ViD(3DRN50), FVD(Inception3D)
'''
import sys
from pathlib import Path

import os
import random
from tqdm import tqdm
from glob import glob
import torch
import numpy as np
from scipy import linalg
from PIL import Image
import os.path as osp
import torch.nn.functional as F
import glob
import json
import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
from tool.metrics.utils import ResizeDataset, EXTENSIONS, ResizeDatasetBlur, ResizeDatasetInceptionScoreBlur, ResizeDatasetFIDVideoBlur
from tool.metrics.utils import DatasetFVDVideoResize, DatasetFVDVideoFromFramesResize
from tool.metrics.features import build_feature_extractor, get_reference_statistics
from tool.metrics.resize import build_resizer
from tool.metrics.ssim_l1_lpips_psnr import compute_ssim_l1_psnr, compute_lpips
import json

"""
Compute the FID score given the mu, sigma of two sets
"""


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Danica J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


"""
Compute the KID score given the sets of features
"""


def kernel_distance(feats1, feats2, num_subsets=100, max_subset_size=1000):
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feats2[np.random.choice(feats2.shape[0], m, replace=False)]
        y = feats1[np.random.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)


"""
Compute the inception features for a batch of images
"""


def get_batch_features(batch, model, device):
    if model is None:
        return batch.numpy()
    with torch.no_grad():
        feat = model(batch.to(device))
    return feat.detach().cpu().numpy()


"""
Compute the inception features for a list of files
"""


def get_files_features(l_files, batch_size, num_workers, model=None, device=torch.device("cuda"), mode="clean", custom_fn_resize=None, description=""):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)

    # build resizing function based on options
    if custom_fn_resize is not None:
        fn_resize = custom_fn_resize
    else:
        fn_resize = build_resizer(mode)

    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, fn_resize=fn_resize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # collect all inception features
    l_feats = []
    for batch in tqdm(dataloader, desc=description):
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Compute the inception features for a folder of features
"""


def get_folder_features(fdir, batch_size, num_workers, model=None, num=None,
                        shuffle=False, seed=0, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description=""):
    # get all relevant files in the dataset
    # = sorted([file for ext in EXTENSIONS for file in glob(os.path.join(fdir, f"**/*.{ext}"), recursive=True)])
    files = []
    for ext in EXTENSIONS:
        for file in glob(os.path.join(fdir, f"**/*.{ext}"), recursive=True):
            try:
                image = Image.open(str(file))
                image.verify()
                image.close()
                files.append(file)
            except:
                print('{} can not open'.format(file))
                continue
    files = sorted(files)

    print(f"Found {len(files)} images in the folder {fdir}")
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    np_feats = get_files_features(files, model, batch_size=batch_size, num_workers=num_workers, device=device,
                                  mode=mode,
                                  custom_fn_resize=custom_fn_resize,
                                  description=description)
    return np_feats


"""
Compute the FID score given the inception features stack
"""


def fid_from_feats(feats1, feats2):
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)


"""
Computes the FID score for a folder of images for a specific dataset 
and a specific resolution
"""


def fid_folder(fdir, dataset_name, dataset_res, dataset_split, batch_size, num_workers, model=None, mode="clean",
               device=torch.device("cuda")):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res,
                                                 mode=mode, seed=0, split=dataset_split)
    fbname = os.path.basename(fdir)
    # get all inception features for folder images
    np_feats = get_folder_features(fdir, model, batch_size=batch_size, num_workers=num_workers,
                                   device=device,
                                   mode=mode, description=f"FID {fbname} : ")
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


"""
Compute the FID stats from a generator model
"""


def get_model_features(G, model, batch_size, mode="clean", z_dim=512,
                       num_gen=50_000,
                       device=torch.device("cuda"), desc="FID model: "):
    fn_resize = build_resizer(mode)
    # Generate test features
    num_iters = int(np.ceil(num_gen / batch_size))
    l_feats = []
    for idx in tqdm(range(num_iters), desc=desc):
        with torch.no_grad():
            z_batch = torch.randn((batch_size, z_dim)).to(device)
            # generated image is in range [0,255]
            img_batch = G(z_batch)
            # split into individual batches for resizing if needed
            if mode != "legacy_tensorflow":
                resized_batch = torch.zeros(batch_size, 3, 299, 299)
                for idx in range(batch_size):
                    curr_img = img_batch[idx]
                    img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
                    img_resize = fn_resize(img_np)
                    resized_batch[idx] = torch.tensor(
                        img_resize.transpose((2, 0, 1)))
            else:
                resized_batch = img_batch
            feat = get_batch_features(resized_batch, model, device)
        l_feats.append(feat)
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Computes the FID score for a generator model for a specific dataset 
and a specific resolution
"""


def fid_model(G, dataset_name, dataset_res, dataset_split, batch_size, num_workers,
              model=None, z_dim=512, num_gen=50_000,
              mode="clean", 
              device=torch.device("cuda")):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res,
                                                 mode=mode, seed=0, split=dataset_split)
    # build resizing function based on options
    fn_resize = build_resizer(mode)

    # Generate test features
    np_feats = get_model_features(G, model, batch_size=batch_size, mode=mode,
                                  z_dim=z_dim, num_gen=num_gen,
                                  device=device)

    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


"""
Computes the FID score between the two given folders
"""


def compare_folders(fdir1, fdir2, feat_model, mode, batch_size, num_workers,
                    device=torch.device("cuda")):
    # get all inception features for the first folder
    fbname1 = os.path.basename(fdir1)
    np_feats1 = get_folder_features(fdir1, feat_model, batch_size=batch_size, num_workers=num_workers,
                                    device=device, mode=mode,
                                    description=f"FID {fbname1} : ")
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = get_folder_features(fdir2, feat_model, batch_size=batch_size, num_workers=num_workers,
                                    device=device, mode=mode,
                                    description=f"FID {fbname2} : ")
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def compute_kid(batch_size, num_workers, fdir1=None, fdir2=None, gen=None,
                mode="clean", 
                device=torch.device("cuda"), dataset_name="FFHQ",
                dataset_res=1024, dataset_split="train", num_gen=50_000, z_dim=512):
    # build the feature extractor based on the mode
    feat_model = build_feature_extractor(mode, device)

    # if both dirs are specified, compute KID between folders
    if fdir1 is not None and fdir2 is not None:
        print("compute KID between two folders")
        # get all inception features for the first folder
        fbname1 = os.path.basename(fdir1)
        np_feats1 = get_folder_features(fdir1, None, batch_size=batch_size, num_workers=num_workers,
                                        device=device, mode=mode,
                                        description=f"KID {fbname1} : ")
        # get all inception features for the second folder
        fbname2 = os.path.basename(fdir2)
        np_feats2 = get_folder_features(fdir2, None, batch_size=batch_size, num_workers=num_workers,
                                        device=device, mode=mode,
                                        description=f"KID {fbname2} : ")
        score = kernel_distance(np_feats1, np_feats2)
        return score

    # compute kid of a folder
    elif fdir1 is not None and fdir2 is None:
        print(f"compute KID of a folder with {dataset_name} statistics")
        # define the model if it is not specified
        model = build_feature_extractor(mode, device)
        ref_feats = get_reference_statistics(dataset_name, dataset_res,
                                             mode=mode, seed=0, split=dataset_split, metric="KID")
        fbname = os.path.basename(fdir1)
        # get all inception features for folder images
        np_feats = get_folder_features(fdir1, model, batch_size=batch_size, num_workers=num_workers,
                                       device=device,
                                       mode=mode, description=f"KID {fbname} : ")
        score = kernel_distance(ref_feats, np_feats)
        return score

    # compute fid for a generator
    elif gen is not None:
        print(
            f"compute KID of a model with {dataset_name}-{dataset_res} statistics")
        # define the model if it is not specified
        model = build_feature_extractor(mode, device)
        ref_feats = get_reference_statistics(dataset_name, dataset_res,
                                             mode=mode, seed=0, split=dataset_split, metric="KID")
        # build resizing function based on options
        fn_resize = build_resizer(mode)
        # Generate test features
        np_feats = get_model_features(gen, model, batch_size=batch_size, mode=mode,
                                      z_dim=z_dim, num_gen=num_gen, desc="KID model: ",
                                      device=device)
        score = kernel_distance(ref_feats, np_feats)
        return score

    else:
        raise ValueError(f"invalid combination of directories and models entered")


def compute_fid(batch_size, num_workers, fdir1=None, fdir2=None, gen=None,
                mode="clean",
                device=torch.device("cuda"), dataset_name="FFHQ",
                dataset_res=1024, dataset_split="train", num_gen=50_000, z_dim=512):
    # build the feature extractor based on the mode
    feat_model = build_feature_extractor(mode, device)

    # if both dirs are specified, compute FID between folders
    if fdir1 is not None and fdir2 is not None:
        print("compute FID between two folders")
        score = compare_folders(fdir1, fdir2, feat_model, batch_size=batch_size, num_workers=num_workers,
                                mode=mode, 
                                device=device)
        return score

    # compute fid of a folder
    elif fdir1 is not None and fdir2 is None:
        print(f"compute FID of a folder with {dataset_name} statistics")
        score = fid_folder(fdir1, dataset_name, dataset_res, dataset_split, batch_size=batch_size, num_workers=num_workers,
                           model=feat_model, mode=mode,
                           device=device)
        return score

    # compute fid for a generator
    elif gen is not None:
        print(f"compute FID of a model with {dataset_name}-{dataset_res} statistics")
        score = fid_model(gen, dataset_name, dataset_res, dataset_split, batch_size=batch_size, num_workers=num_workers,
                          model=feat_model, z_dim=z_dim, num_gen=num_gen,
                          mode=mode,
                          device=device)
        return score

    else:
        raise ValueError("invalid combination of directories and models entered")


def get_files_features_blur(l_files, batch_size, num_workers, model=None, device=torch.device("cuda"), mode="clean", custom_fn_resize=None, description=""):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)

    # build resizing function based on options
    if custom_fn_resize is not None:
        fn_resize = custom_fn_resize
    else:
        fn_resize = build_resizer(mode)

    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, fn_resize=fn_resize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # collect all inception features
    l_feats = []
    for batch in tqdm(dataloader, desc=description):
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


def compute_net_prediction(dataset, feat_model, batch_size, num_workers):

    # fn_resize = build_resizer(mode='clean')

    # wrap the images in a dataloader for parallelizing the resize operation
    # dataset = ResizeDatasetBlur(inst_name_full, fn_resize=fn_resize, )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # collect all inception features
    l_feats = []
    for batch in tqdm(dataloader, desc='start load the feature'):
        l_feats.append(get_batch_features(batch, feat_model, torch.device('cuda')))
    np_feats = np.concatenate(l_feats)
    return np_feats


def compute_net_vid_prediction(dataset, feat_model, batch_size, num_workers):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # collect all inception features
    l_feats = []
    num_frame = 0
    batch_assemble = []
    for batch in tqdm(dataloader, desc='start load the feature'):
        batch = batch.squeeze(0)
        num_frame += batch.shape[0]
        batch_assemble.append(batch)
        if num_frame >= batch_size:
            batch_s = torch.cat(batch_assemble, dim=0)
            l_feats.append(get_batch_features(batch_s, feat_model, torch.device('cuda')))
            batch_assemble = []
            num_frame = 0
        else:
            batch_assemble.append(batch)

    np_feats = np.concatenate(l_feats)
    return np_feats


def compute_3d_video_prediction(dataset, feat_model, batch_size, num_workers):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    assert batch_size == 1
    # collect all inception features
    l_feats = []

    for batch in tqdm(dataloader, desc='start load the feature'):

        batch = batch.squeeze(0)
        l_feats.append(get_batch_features(batch, feat_model, torch.device('cuda')).mean(0, keepdims=True))  # one video is split into multiple segment, we need average every segments

    np_feats = np.concatenate(l_feats)
    return np_feats


def compute_fid_folder_scores(gen_inst_name_full, gt_inst_name_full, feat_model, is_blur_img, gaussian_blur_kernel, batch_size, num_workers):

    fn_resize = build_resizer(mode='clean')

    dataset_gen = ResizeDatasetBlur(gen_inst_name_full, fn_resize=fn_resize, is_blur=is_blur_img, gaussian_kernel=gaussian_blur_kernel)
    np_feats_gen = compute_net_prediction(dataset_gen, feat_model, batch_size=batch_size, num_workers=num_workers)

    dataset_gt = ResizeDatasetBlur(gt_inst_name_full, fn_resize=fn_resize, is_blur=is_blur_img, gaussian_kernel=gaussian_blur_kernel)
    np_feats_gt = compute_net_prediction(dataset_gt, feat_model, batch_size=batch_size, num_workers=num_workers)

    fid_score = fid_from_feats(feats1=np_feats_gen, feats2=np_feats_gt)

    return fid_score


def compute_video_fid_folder_scores(gen_inst_name_full, gt_inst_name_full, feat_model, is_blur_img, gaussian_blur_kernel, sample_frames, is_sample_frames, batch_size, num_workers):

    fn_resize = build_resizer(mode='clean')

    dataset_gen = ResizeDatasetFIDVideoBlur(gen_inst_name_full, fn_resize=fn_resize, is_blur=is_blur_img, gaussian_kernel=gaussian_blur_kernel,
                                            is_sample_frames=is_sample_frames, sample_frames=sample_frames)
    np_feats_gen = compute_net_vid_prediction(dataset_gen, feat_model, batch_size=batch_size, num_workers = num_workers)

    dataset_gt = ResizeDatasetFIDVideoBlur(gt_inst_name_full, fn_resize=fn_resize, is_blur=is_blur_img, gaussian_kernel=gaussian_blur_kernel,
                                           is_sample_frames=is_sample_frames, sample_frames=sample_frames)
    np_feats_gt = compute_net_vid_prediction(dataset_gt, feat_model, batch_size=batch_size, num_workers = num_workers)
    fid_score = fid_from_feats(feats1=np_feats_gen, feats2=np_feats_gt)

    return fid_score


def compute_inception_scores(gen_inst_name_full, feat_model, is_blur_img, gaussian_blur_kernel, batch_size, num_workers, num_splits=1):

    fn_resize = build_resizer(mode='clean')

    dataset_gen = ResizeDatasetInceptionScoreBlur(gen_inst_name_full, fn_resize=fn_resize, is_blur=is_blur_img, gaussian_kernel=gaussian_blur_kernel)
    np_pred_logit_gen = compute_net_prediction(dataset_gen, feat_model, batch_size=batch_size, num_workers=num_workers)

    preds = F.softmax(torch.as_tensor(np_pred_logit_gen), dim=1)
    N = preds.shape[0]
    split_scores = []
    for k in range(num_splits):
        part = preds[k * (N // num_splits): (k+1) * (N // num_splits), :]
        pmean = part.mean(0, keepdim=True)
        scores = (part * (part.log() - pmean.log())).sum(1).numpy()
        split_scores.append(np.exp(np.mean(scores)))
        # part = part.numpy()
        # pmean = pmean.numpy()
        # kl = part * (np.log(part) - np.log(pmean))
        # kl = np.mean(np.sum(kl, 1))
        # split_scores.append(np.exp(kl))

    return np.mean(split_scores), np.std(split_scores)


def compute_fid_video_scores(gen_inst_name_full, gt_inst_name_full, feat_model, mode, sample_duration, batch_size, num_workers):

    if mode == "FVD-3DRN50":
        sample_size = 112
    elif mode == "FVD-3DInception":
        sample_size = 224
    elif mode == 'MAE':
        sample_size = 224
    else:
        raise NotImplementedError
    
    if Path(gen_inst_name_full[0]).suffix in [".mp4", ".gif"]:
        dataset_gen = DatasetFVDVideoResize(gen_inst_name_full, sample_duration, mode, sample_size)
    else:
        dataset_gen = DatasetFVDVideoFromFramesResize(gen_inst_name_full, sample_duration, mode, sample_size)

    np_feats_gen = compute_3d_video_prediction(dataset_gen, feat_model, batch_size=batch_size, num_workers=num_workers)

    if Path(gt_inst_name_full[0]).suffix in [".mp4", ".gif"]:
        dataset_gt = DatasetFVDVideoResize(gt_inst_name_full, sample_duration, mode, sample_size)
    else:
        dataset_gt = DatasetFVDVideoFromFramesResize(gt_inst_name_full, sample_duration, mode, sample_size)

    np_feats_gt = compute_3d_video_prediction(dataset_gt, feat_model, batch_size=batch_size, num_workers=num_workers)
    if mode == 'MAE':
        fid_score = np.abs((np_feats_gen - np_feats_gt)).mean()
    else:
        fid_score = fid_from_feats(feats1=np_feats_gen, feats2=np_feats_gt)

    return fid_score

def evaluation_visual_creation(
        gen_inst_names_full, batch_size, num_workers,
        root_dir, gt_inst_names_full=None, metric="FID",
        gaussian_blur_kernel=0, num_splits=1, is_sample_frame=False,
        number_sample_frames=1, sample_duration=16):
    """
        gen_inst_path: the generated image/video path,  "/dataspace/MSCOCO/train2014"
        gen_inst_names: list of image/video name ['xxx.jpg', 'xxx.jpg']
        gt_inst_path: the groundtruth image path
        gt_inst_names: the groundtruth image names

        gaussian_blur_kernel: the parameter to control the 

    """

    device = torch.device('cuda')

    assert gaussian_blur_kernel >= 0
    if metric == "FID":
        is_blur_img = True
        if gaussian_blur_kernel == 0:
            is_blur_img = False

        feat_model = build_feature_extractor(mode='clean', root_dir=root_dir, device=device)
        # print(f"evaluation_gen_path:{gen_inst_path}")
        print(f'start evluation {metric} over {len(gen_inst_names_full)} generated Image/Video and {len(gt_inst_names_full)} gt Image')
        FID_score = compute_fid_folder_scores(
            gen_inst_names_full, gt_inst_names_full,
            feat_model, batch_size=batch_size, num_workers=num_workers,
            is_blur_img=is_blur_img, gaussian_blur_kernel=gaussian_blur_kernel)
        
        print(f"The FID_score {FID_score}, blur_image:{is_blur_img}, gaussian_kernel_size:{gaussian_blur_kernel}")
        return {metric: FID_score}

    elif metric == "FID-Img":
        is_blur_img = False
        gaussian_blur_kernel = 0
        feat_model = build_feature_extractor(mode='clean', root_dir=root_dir, device=device)
        print(f'start evluation FID-Img over {len(gen_inst_names_full)} generated Image/Video and {len(gt_inst_names_full)} gt Image')
        FID_score = compute_video_fid_folder_scores(
            gen_inst_names_full, gt_inst_names_full, feat_model,
            is_blur_img, gaussian_blur_kernel, batch_size=batch_size,
            num_workers=num_workers, is_sample_frames=is_sample_frame, 
            sample_frames=number_sample_frames)

        print(f"The FID-Img:{FID_score}, is_blur:{is_blur_img}, gaussian_kernel:{gaussian_blur_kernel}, is_sample:{is_sample_frame}, num_sam_frame:{number_sample_frames}")
        return {metric: FID_score}
                
    elif metric == "IS":

        is_blur_img = True
        if gaussian_blur_kernel == 0:
            is_blur_img = False

        feat_model = build_feature_extractor(mode='clean_InceptionScore', root_dir=root_dir, device=device)
        print(f'start evluation inception scores over {len(gen_inst_names_full)} generated Image')
        IS_score_mean, IS_score_std = compute_inception_scores(
            gen_inst_name_full=gen_inst_names_full, batch_size=batch_size,
            num_workers=num_workers, feat_model=feat_model, is_blur_img=is_blur_img,
            gaussian_blur_kernel=gaussian_blur_kernel, num_splits=num_splits)

        print(f"The inception_score mean/std:{IS_score_mean}/{IS_score_std}={IS_score_mean/IS_score_std}, num_splits:{num_splits}, blur_image:{is_blur_img}, gaussian_kernel_size:{gaussian_blur_kernel}")
        return {'IS_score_mean': IS_score_mean, 'IS_score_std': IS_score_std}

    elif metric in ['FVD-3DRN50', "FVD-3DInception"]:
        
        print(f'start evluation {metric} over {len(gen_inst_names_full)} generated Image/Video and {len(gt_inst_names_full)} gt Image/Video')
        feat_model = build_feature_extractor(mode=metric, root_dir=root_dir, device=device, sample_duration=sample_duration)
        FVD_score = compute_fid_video_scores(
            gen_inst_names_full, gt_inst_names_full, feat_model,
            mode=metric, sample_duration=sample_duration,
            batch_size=batch_size, num_workers=num_workers)

        print(f"The {metric} {FVD_score}, duration:{sample_duration}")
        return {metric: FVD_score}
    
    elif metric == 'MAE':
        print(f'start evluation {metric} over {len(gen_inst_names_full)} generated Image/Video and {len(gt_inst_names_full)} gt Image/Video')
        FVD_score = compute_fid_video_scores(
            gen_inst_names_full, gt_inst_names_full, None,
            mode=metric, sample_duration=sample_duration,
            batch_size=batch_size, num_workers=num_workers)
        print(f"The MAE_score {FVD_score}, duration:{sample_duration}")
        return {metric: FVD_score}
    elif metric in ['SSIM', 'L1', 'PSNR']:
        
        print(f'start evluation {metric} over {len(gen_inst_names_full)} generated Image/Video and {len(gt_inst_names_full)} gt Image/Video')
        ssim_l1_score = compute_ssim_l1_psnr(
            gen_inst_names_full, gt_inst_names_full,
            mode=metric)

        print(f"The {metric} {ssim_l1_score}")
        return {metric: ssim_l1_score}
    elif metric == 'LPIPS':
        
        print(f'start evluation {metric} over {len(gen_inst_names_full)} generated Image/Video and {len(gt_inst_names_full)} gt Image/Video')
        lpips_score = compute_lpips(
            gen_inst_names_full, gt_inst_names_full)

        print(f"The {metric} {lpips_score}")
        return {metric: lpips_score}
    else:
        print("we are not support the metric you specified, please contact me")
        raise NotImplementedError 


def get_all_eval_scores(
        root_dir, path_gen, path_gt, metrics,
        batch_size=1, blur=0, number_sample_frames=8,
        num_gen=None, num_gt=None, num_splits=1, sample_duration=8,
        num_workers=16):
    number_sample_frames = min(sample_duration, number_sample_frames)
    if root_dir not in path_gen:
        path_gen = os.path.join(root_dir, path_gen)
    if root_dir not in path_gt:
        path_gt = os.path.join(root_dir, path_gt)
    gen_inst_names_v, gt_inst_names_v = [], []
    gen_inst_names_i, gt_inst_names_i = [], []
    res_all = {}
    type2metric = {'fid-img': 'FID-Img', 'fid': 'FID', 'mae': 'MAE', 
                    'fid-vid': "FVD-3DRN50", 'fvd': "FVD-3DInception", 'is': 'IS', 
                    'l1': 'L1', 'ssim': 'SSIM', 'lpips': 'LPIPS', 'psnr': 'PSNR'}
    empty_gen_folder_v, empty_gt_folder_v = False, False
    empty_gen_folder_i, empty_gt_folder_i = False, False
    for type in metrics:
        if type in ['fid-img', 'fvd', 'fid-vid', 'MAE']:
            if empty_gen_folder_v or empty_gt_folder_v:
                print(f"Empty gen/gt folder {path_gen}, {path_gt}")
                break
            if len(gen_inst_names_v) == 0:
                gen_inst_names_v = glob.glob(f"{path_gen}/*.gif") + glob.glob(f"{path_gen}/*.mp4") + glob.glob(f"{path_gen}/*.png") + glob.glob(f"{path_gen}/*.jpg") 
                gt_inst_names_v = glob.glob(f"{path_gt}/*.gif") + glob.glob(f"{path_gt}/*.mp4") + glob.glob(f"{path_gt}/*.png") + glob.glob(f"{path_gt}/*.jpg")
                gen_inst_names_v = [osp.basename(gen_inst) for gen_inst in gen_inst_names_v]
                gen_inst_names_v.sort()
                gt_inst_names_v = [osp.basename(gt_inst) for gt_inst in gt_inst_names_v]
                gt_inst_names_v.sort()
                if num_gen is not None:
                    gen_inst_names_v = gen_inst_names_v[:num_gen]
                if num_gt is not None:
                    gt_inst_names_v = gt_inst_names_v[:num_gt]
                empty_gen_folder_v = len(gen_inst_names_v) == 0
                empty_gt_folder_v = len(gt_inst_names_v) == 0
                
                if not empty_gen_folder_v:
                    gen_inst_names_full_v = []
                    for gen_name in tqdm(gen_inst_names_v, desc=f'load_gen for {metrics}'):
                        gen_inst_names_full_v.append(osp.join(path_gen, gen_name))
                if not empty_gt_folder_v:
                    gt_inst_names_full_v = []
                    for gt_name in tqdm(gt_inst_names_v, desc=f'load_gt for {metrics}'):
                        gt_inst_names_full_v.append(osp.join(path_gt, gt_name))
        elif type in ['fid', 'is', 'ssim', 'l1', 'lpips', 'psnr']:
            if empty_gen_folder_i or empty_gt_folder_i:
                break
            if len(gen_inst_names_i) == 0:
                gen_inst_names_i = glob.glob(f"{path_gen}/*.png")
                gt_inst_names_i = glob.glob(f"{path_gt}/*.png")
                gen_inst_names_i += glob.glob(f"{path_gen}/*.jpg")
                gt_inst_names_i += glob.glob(f"{path_gt}/*.jpg")
                gen_inst_names_i = [osp.basename(gen_inst) for gen_inst in gen_inst_names_i]
                gen_inst_names_i.sort()
                gt_inst_names_i = [osp.basename(gt_inst) for gt_inst in gt_inst_names_i]
                gt_inst_names_i.sort()
                if num_gen is not None:
                    gen_inst_names_i = gen_inst_names_i[:num_gen]
                if num_gt is not None:
                    gt_inst_names_i = gt_inst_names_i[:num_gt]
                empty_gen_folder_i = len(gen_inst_names_i) == 0
                empty_gt_folder_i = len(gt_inst_names_i) == 0
                
                if not empty_gen_folder_i:
                    gen_inst_names_full_i = []
                    for gen_name in tqdm(gen_inst_names_i, desc=f'load_gen for {metrics}'):
                        gen_inst_names_full_i.append(osp.join(path_gen, gen_name))
                if not empty_gt_folder_i:
                    gt_inst_names_full_i = []
                    for gt_name in tqdm(gt_inst_names_i, desc=f'load_gt for {metrics}'):
                        gt_inst_names_full_i.append(osp.join(path_gt, gt_name))
        elif type in ['clean-fid']:
            print(f'skipping caching file path for {type}')
        else:
            print(f"we are not support the metric you specified: {type}")
            break

        # when calculating the FID, you need to specify:
        # path_gen, gen_inst_names, path_gt, gt_inst_names,
        # gaussian_blur_kernel = 0 means that we do not blur the images
        # batch_size, the default value is 128, which is acceptable, you can scale it up.
        if type == 'fid':
            if empty_gen_folder_i or empty_gt_folder_i:
                print(f"Empty gen/gt folder {path_gen}, {path_gt}")
                break
            fid_batch_size = min(batch_size, min(len(gen_inst_names_i), len(gt_inst_names_i)))
            res = evaluation_visual_creation(
                gen_inst_names_full_i, fid_batch_size,
                num_workers, root_dir, gt_inst_names_full_i,
                metric=type2metric[type], gaussian_blur_kernel=blur,
                num_splits=-1,
                is_sample_frame=False, number_sample_frames=1, sample_duration=-1)
        elif type == 'clean-fid':
            from cleanfid import fid
            score = fid.compute_fid(path_gen, path_gt)
            res = {'clean-fid': score}

        # when calculating the Inception_score, you need to specify:
        # path_gen, gen_inst_names,
        # gaussian_blur_kernel = 0 means that we do not blur the images, the DALL-E also calcuate the IS-1, IS-2, ..., IS-K, you can change it.
        # batch_size, the default value is 128, which is acceptable, you can scale it up.
        # num_split, default is 1, cogview sets num_split=1, but DM-GAN sets num_split=10, you can change it as you want.
        elif type == 'is':
            if empty_gen_folder_i:
                print(f"Empty gen folder {path_gen}")
                break
            is_batch_size = min(batch_size, len(gen_inst_names_i))
            res = evaluation_visual_creation(
                gen_inst_names_full_i, is_batch_size,
                num_workers, root_dir, None, None, metric=type2metric[type],
                gaussian_blur_kernel=blur, num_splits=num_splits,
                is_sample_frame=False, number_sample_frames=1,
                sample_duration=-1)

        # when calculating the FID-img, you need to specify:
        # path_gen, gen_inst_names, path_gt, gt_inst_names,
        # gaussian_blur_kernel = 0 means that we do not blur the images, FID-1, FID-2, ..., FID-K, you can change it.
        # batch_size, the default value is 128, which is acceptable, you can scale it up.
        #  is_sample_frame, if False, we calcuate all frames decoded from videos. num_sample_frames, the parameters you need to specify when you want to sample frames from video
        elif type in ['fid-img', 'fid-vid', 'fvd', 'mae']:
            if empty_gen_folder_v or empty_gt_folder_v:
                print(f"Empty gen/gt folder {path_gen}, {path_gt}")
                break
            v_batch_size = min(batch_size, min(
                    len(gen_inst_names_v), len(gt_inst_names_v)))
            res = evaluation_visual_creation(
                gen_inst_names_full_v, v_batch_size,
                num_workers, root_dir, gt_inst_names_full_v,
                metric=type2metric[type], gaussian_blur_kernel=blur, num_splits=-1,
                is_sample_frame=False, number_sample_frames=number_sample_frames, sample_duration=sample_duration)
        elif type in ['ssim', 'l1', 'psnr']:
            if empty_gen_folder_i or empty_gt_folder_i:
                print(f"Empty gen/gt folder {path_gen}, {path_gt}")
                break
            res = evaluation_visual_creation(
                gen_inst_names_full_i, 1,
                0, root_dir, gt_inst_names_full_i,
                metric=type2metric[type], gaussian_blur_kernel=0, num_splits=-1,
                is_sample_frame=False, number_sample_frames=1, sample_duration=-1)
        elif type in ['lpips']:
            if empty_gen_folder_i or empty_gt_folder_i:
                print(f"Empty gen/gt folder {path_gen}, {path_gt}")
                break
            lpips_batch_size = min(batch_size, min(
                    len(gen_inst_names_i), len(gt_inst_names_i)))
            res = evaluation_visual_creation(
                gen_inst_names_full_i, lpips_batch_size,
                num_workers, root_dir, gt_inst_names_full_i,
                metric=type2metric[type], gaussian_blur_kernel=blur,
                num_splits=-1,
                is_sample_frame=False, number_sample_frames=1, sample_duration=-1)
        res_all.update(res)
    return res_all

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="/f_ndata/G")
    parser.add_argument('--path_gen', type=str, default="sdm/SDM_KLF8_S512_MSCOCO/eval_visu/pred")
    parser.add_argument('--path_gt', type=str, default="dataset/mscoco/val2017")
    parser.add_argument('--num_gen', type=int, default=None)
    parser.add_argument('--num_gt', type=int, default=None)
    parser.add_argument('--type', type=str, default='fid', nargs="+")
    parser.add_argument('--blur', type=int, default=0)
    parser.add_argument('--num_splits', type=int, default=1)
    parser.add_argument('--number_sample_frames', type=int, default=8)
    parser.add_argument('--sample_duration', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--write_metric_to', type=str, default=None)

    args = parser.parse_args()

    res_all = get_all_eval_scores(
        root_dir=args.root_dir,
        path_gen=args.path_gen, path_gt=args.path_gt,
        metrics=args.type, batch_size=args.batch_size,
        number_sample_frames=args.number_sample_frames, num_gen=args.num_gen,
        num_gt=args.num_gt, num_splits=args.num_splits,
        sample_duration=args.sample_duration, num_workers=args.num_workers)
    print(res_all)
    if args.write_metric_to is not None:
        json.dump(res_all, open(args.write_metric_to, "w"))
