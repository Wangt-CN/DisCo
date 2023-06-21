"""
helpers for extractign features from image
"""
import os
from random import sample
import numpy as np
import torch
import torch.nn as nn
from tool.metrics.downloads_helper import check_download_url
from tool.metrics.inception_pytorch import InceptionV3
from tool.metrics.inception_torchscript import InceptionV3W
import torchvision.models as models
import tool.metrics.resnet3d as resnet3d
from tool.metrics.inception3d import InceptionI3d

"""
Build a feature extractor for each of the modes
"""


def build_feature_extractor(mode, root_dir, device=torch.device("cuda"), sample_duration=16):

    if mode == "clean":
        feat_model = InceptionV3W("/tmp", download=True, resize_inside=False).to(device)
        feat_model = feat_model.eval()
        
    elif mode == "clean_InceptionScore":
        # we need the softmax function after the calcualtion
        feat_model = models.inception_v3(pretrained=True, progress=True)
        feat_model = feat_model.to(device).eval()
    elif mode == "FVD-3DRN50":
        feat_model = resnet3d.resnet50(num_classes=400, shortcut_type="B", sample_size=112, sample_duration=sample_duration, last_fc=False)
        model_data = torch.load(os.path.join(root_dir, "eval_fvd/resnet-50-kinetics.pth"), map_location='cpu')
        model_state_new = {}
        for key, value in model_data['state_dict'].items():
            key_new = key.replace('module.', '')
            model_state_new[key_new] = value

        feat_model.load_state_dict(model_state_new)
        feat_model = feat_model.to(device).eval()
    elif mode == "FVD-3DInception":  # based on logits to calcuate, which can be found the description from latent vision transformer
        feat_model = InceptionI3d(400, in_channels=3)
        feat_model.load_state_dict(torch.load(os.path.join(root_dir, "eval_fvd/i3d_pretrained_400.pt")))
        feat_model = feat_model.to(device).eval()
    else:
        raise NotImplementedError
    return feat_model


"""
Load precomputed reference statistics for commonly used datasets
"""


def get_reference_statistics(name, res, mode="clean", seed=0, split="test", metric="FID"):
    base_url = "https://www.cs.cmu.edu/~clean-fid/stats/"
    if split == "custom":
        res = "na"
    if metric == "FID":
        rel_path = (f"{name}_{mode}_{split}_{res}.npz").lower()
        url = f"{base_url}/{rel_path}"
        mod_path = os.path.dirname(metrics.__file__)
        stats_folder = os.path.join(mod_path, "stats")
        fpath = check_download_url(local_folder=stats_folder, url=url)
        stats = np.load(fpath)
        mu, sigma = stats["mu"], stats["sigma"]
        return mu, sigma
    elif metric == "KID":
        rel_path = (f"{name}_{mode}_{split}_{res}_kid.npz").lower()
        url = f"{base_url}/{rel_path}"
        mod_path = os.path.dirname(metrics.__file__)
        stats_folder = os.path.join(mod_path, "stats")
        fpath = check_download_url(local_folder=stats_folder, url=url)
        stats = np.load(fpath)
        return stats["feats"]
