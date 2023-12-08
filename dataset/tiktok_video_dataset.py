import re

import numpy as np
import torch
from PIL import Image

from config import *
from dataset.tsv_cond_dataset import TsvCondImgCompositeDataset


class BaseDataset(TsvCondImgCompositeDataset):
    def __init__(self, args, yaml_file, split="train", preprocesser=None):
        self.img_size = getattr(args, "img_full_size", args.img_size)
        self.basic_root_dir = BasicArgs.root_dir
        self.max_video_len = args.max_video_len
        assert self.max_video_len == 1
        self.fps = args.fps
        self.dataset = "TiktokDance-Image"
        self.preprocesser = preprocesser
        if not hasattr(args, "ref_mode"):
            args.ref_mode = "first"

        super().__init__(
            args, yaml_file, split=split, size_frame=args.max_video_len, tokzr=None
        )
        self.eval_sample_interval = args.eval_sample_interval
        self.train_sample_interval = args.train_sample_interval
        self.img_key_dict = {key: i for i, key in enumerate(self.image_keys)}
        self.data_dir = args.data_dir
        self.img_ratio = (
            (1.0, 1.0)
            if not hasattr(self.args, "img_ratio") or self.args.img_ratio is None
            else self.args.img_ratio
        )
        self.img_scale = (
            (1.0, 1.0)
            if not split == "train"
            else getattr(self.args, "img_scale", (0.9, 1.0))
        )  # val set should keep scale=1.0 to avoid the random crop
        print(
            f"Current Data: {split}; Use image scale: {self.img_scale}; Use image ratio: {self.img_ratio}"
        )

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        try:
            self.ref_transform = transforms.Compose(
                [  # follow CLIP transform
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(
                        (224, 224),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=False,
                    ),
                    transforms.Normalize(
                        [0.48145466, 0.4578275, 0.40821073],
                        [0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
            self.ref_transform_mask = transforms.Compose(
                [  # follow CLIP transform
                    transforms.RandomResizedCrop(
                        (224, 224),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=False,
                    ),
                    transforms.ToTensor(),
                ]
            )
        except:
            print("### Current pt version not support antialias, thus remove it! ###")
            self.ref_transform = transforms.Compose(
                [  # follow CLIP transform
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(
                        (224, 224),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.Normalize(
                        [0.48145466, 0.4578275, 0.40821073],
                        [0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
            self.ref_transform_mask = transforms.Compose(
                [  # follow CLIP transform
                    transforms.RandomResizedCrop(
                        (224, 224),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                ]
            )
        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

    def add_mask_to_img(self, img, mask, img_key):  # pil, pil
        if not img.size == mask.size:
            # import pdb; pdb.set_trace()
            # print(f'Reference image ({img_key}) size ({img.size}) is different from the mask size ({mask.size}), therefore try to resize the mask')
            mask = mask.resize(img.size)  # resize the mask
        mask_array = np.array(mask)
        img_array = np.array(img)
        mask_array[mask_array < 127.5] = 0
        mask_array[mask_array > 127.5] = 1
        return Image.fromarray(img_array * mask_array), Image.fromarray(
            img_array * (1 - mask_array)
        )  # foreground, background

    def augmentation(self, frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame)

    def _get_frame_idx_seq(self, start_img_key, nframes: int, frame_interval: int):
        format1 = r"^(.*TiktokDance_\d+_)(\d+)(\.\w+)$"
        format2 = r"^(TiktokDance_\d+_\d+_1x1_)(\d+)(\.\w+)$"
        format3 = r"^(.*_TiktokDance_[a-z0-9]+_\d+_)(\d+)(\.\w+)$"
        format4 = r"^(?!.*TiktokDance)([^_]+_)(\d+)$"  # NTU dataset
        match1 = re.match(format1, start_img_key)
        match2 = re.match(format2, start_img_key)
        match3 = re.match(format3, start_img_key)
        match4 = re.match(format4, start_img_key)
        if match1:
            prefix, start_frame_idx, suffix = match1.groups()
        elif match2:
            prefix, start_frame_idx, suffix = match2.groups()
        elif match3:
            prefix, start_frame_idx, suffix = match3.groups()
        elif match4:
            prefix, start_frame_idx = match4.groups()
            suffix = ""
        else:
            raise IndexError(f"failed to parse {start_img_key}")

        start_frame_idx_int = int(start_frame_idx)
        img_idx_seq = []
        for i in range(nframes):
            frame_idx = str(start_frame_idx_int + i * frame_interval).zfill(
                len(start_frame_idx)
            )
            img_key = prefix + frame_idx + suffix
            if img_key not in self.img_key_dict:
                return False, None
            img_idx_seq.append(self.img_key_dict[img_key])
        return True, img_idx_seq

    def get_metadata(self, idx):
        while True:
            img_idx, cap_idx = self.get_image_cap_index(idx)
            status, frame_idx_seq = self._get_frame_idx_seq(
                self.image_keys[img_idx], self.args.nframes, self.args.frame_interval
            )
            if status:
                break
            idx -= 1

        frame_seq = []
        img_key_seq = []
        for frame_idx in frame_idx_seq:
            img_key = self.image_keys[frame_idx]
            frame, _ = self.get_visual_data(frame_idx)
            frame_seq.append(frame)
            img_key_seq.append(img_key)

        pose_img_seq = []
        for frame_idx in frame_idx_seq:
            pose_img = self.get_cond(frame_idx, "poses")
            pose_img_seq.append(pose_img)

        # preparing outputs
        meta_data = {}
        meta_data["img_seq"] = frame_seq
        meta_data["img_key_seq"] = img_key_seq
        meta_data["pose_img_seq"] = pose_img_seq

        ref_img_idx = self.get_reference_frame_idx(img_idx)
        meta_data["ref_img_key"] = self.image_keys[ref_img_idx]
        meta_data["ref_img"], _ = self.get_visual_data(ref_img_idx)
        meta_data["ref_mask"] = self.get_cond(ref_img_idx, "masks")
        meta_data["bg_ref_img_key"] = meta_data["ref_img_key"]
        meta_data["bg_ref_img"] = meta_data["ref_img"]
        meta_data["bg_ref_mask"] = meta_data["ref_mask"]

        return meta_data

    def get_reference_frame_idx(self, img_idx):
        def _get_first_frame_img_key(prefix, frame_idx, suffix):
            new_frame_idx = str(1).zfill(len(frame_idx))
            result = prefix + new_frame_idx + suffix
            if result in self.img_key_dict.keys():
                return result
            else:
                keys = [key for key in self.img_key_dict.keys() if prefix in key]
                keys = sorted(keys)
                return keys[0]

        img_key = self.image_keys[img_idx]
        format1 = r"^(.*TiktokDance_\d+_)(\d+)(\.\w+)$"
        format2 = r"^(TiktokDance_\d+_\d+_1x1_)(\d+)(\.\w+)$"
        format3 = r"^(.*_TiktokDance_[a-z0-9]+_\d+_)(\d+)(\.\w+)$"
        format4 = r"^(?!.*TiktokDance)([^_]+_)(\d+)$"  # NTU dataset
        match1 = re.match(format1, img_key)
        match2 = re.match(format2, img_key)
        match3 = re.match(format3, img_key)
        match4 = re.match(format4, img_key)
        if match1:
            prefix, frame_idx, suffix = match1.groups()
            # print(f"{img_key} matched to match1, {prefix}, {frame_idx}, {suffix}")
        elif match2:
            prefix, frame_idx, suffix = match2.groups()
            # print(f"{img_key} matched to match2, {prefix}, {frame_idx}, {suffix}")
        elif match3:
            prefix, frame_idx, suffix = match3.groups()
            # print(f"{img_key} matched to match3, {prefix}, {frame_idx}, {suffix}")
        elif match4:
            prefix, frame_idx = match4.groups()
            suffix = ""
            # print(f"{img_key} matched to match4, {prefix}, {frame_idx}")
        else:
            print("failed to match the image key: ", img_key)
            return super().get_reference_frame_idx(img_idx)

        try:
            if self.args.ref_mode == "first" or self.split != "train":
                new_img_key = _get_first_frame_img_key(prefix, frame_idx, suffix)
            else:
                valid_img_keys = [x for x in self.img_key_dict.keys() if prefix in x]
                if self.args.ref_mode == "random":
                    new_img_key = random.choice(valid_img_keys)
                elif self.args.ref_mode == "random_sparse":
                    new_img_key = random.choice(valid_img_keys[::30])
                elif self.args.ref_mode == "random_sparse_part":
                    if random.random() < 0.2:
                        new_img_key = random.choice(valid_img_keys[::30])
                    else:
                        new_img_key = _get_first_frame_img_key(
                            prefix, frame_idx, suffix
                        )
        except:
            new_img_key = _get_first_frame_img_key(prefix, frame_idx, suffix)
        return self.img_key_dict[new_img_key]

    def get_visual_data(self, img_idx):
        try:
            row = self.get_row_from_tsv(self.visual_tsv, img_idx)
            return self.str2img(row[-1]), False
        except Exception as e:
            raise ValueError(f"{e}, in get_visual_data()")

    def __len__(self):
        if self.split == "train":
            if getattr(self.args, "max_train_samples", None):
                return min(self.args.max_train_samples, super().__len__())
            else:
                return int(super().__len__() // self.train_sample_interval)
        else:
            if getattr(self.args, "max_eval_samples", None):
                return min(self.args.max_eval_samples, super().__len__())
            else:
                return int(super().__len__() // self.eval_sample_interval)

    def __getitem__(self, idx):
        if self.split == "train":
            idx = int(idx * self.train_sample_interval)
            idx = idx + random.randint(0, self.train_sample_interval - 1)
            idx = min(idx, len(self) - 1)
        elif self.split == "eval":
            idx = int(idx * self.eval_sample_interval)

        raw_data = self.get_metadata(idx)
        img_seq = raw_data["img_seq"]
        pose_img_seq = raw_data["pose_img_seq"]
        fg_img = raw_data["ref_img"]
        bg_img = raw_data["bg_ref_img"]

        state = torch.get_rng_state()
        fg_state = state
        aug_img_seq = []
        for img in img_seq:
            img = self.augmentation(img, self.transform, state)
            aug_img_seq.append(img)
        aug_img_seq = torch.stack(aug_img_seq, dim=1)
        aug_pose_img_seq = []
        for pose_img in pose_img_seq:
            pose_img = self.augmentation(pose_img, self.cond_transform, state)
            aug_pose_img_seq.append(pose_img)
        aug_pose_img_seq = torch.stack(aug_pose_img_seq, dim=1)
        if getattr(self.args, "refer_clip_preprocess", None):
            raise NotImplementedError
            fg_img = self.preprocesser(fg_img).pixel_values[0]  # use clip preprocess
        else:
            fg_img = self.augmentation(fg_img, self.ref_transform, fg_state)
        bg_img = self.augmentation(bg_img, self.transform, state)

        if self.args.combine_use_mask:  # True
            ref_mask = raw_data["ref_mask"]
            bg_ref_mask = raw_data["bg_ref_mask"]
            assert not getattr(
                self.args, "refer_clip_preprocess", None
            )  # mask not support the CLIP process

            ### first resize mask to the img size
            ref_mask = ref_mask.resize(raw_data["ref_img"].size)
            bg_ref_mask = bg_ref_mask.resize(raw_data["bg_ref_img"].size)

            ref_mask = self.augmentation(ref_mask, self.ref_transform_mask, fg_state)
            bg_ref_mask = self.augmentation(
                bg_ref_mask, self.cond_transform, state
            )  # controlnet path input

            # apply the mask
            fg_img = fg_img * ref_mask  # foreground
            bg_img = bg_img * (1 - bg_ref_mask)  # background

        # caption = raw_data["caption"]
        outputs = {
            "img_key_seq": ";".join(raw_data["img_key_seq"]),
            # "input_text": caption,
            "label_img_seq": aug_img_seq,  # ground truth
            "cond_img_seq": aug_pose_img_seq,  # pose
            "reference_img": fg_img,  # foreground
            "reference_img_controlnet": bg_img,  # background
        }

        return outputs
