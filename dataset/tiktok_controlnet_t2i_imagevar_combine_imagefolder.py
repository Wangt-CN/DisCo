from config import *
import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, imagefolder_path, imagefolder_mask_path, imagefolder_pose_path, split='val', preprocesser=None):
        self.args = args
        self.img_size = getattr(args, 'img_full_size', args.img_size)
        self.max_video_len = 1  ## Todo, now it is image-based dataloader
        self.size_frame = 1  ## Todo
        self.is_composite = False
        self.basic_root_dir = BasicArgs.root_dir
        self.preprocesser = preprocesser
        self.split = split
        if not hasattr(args, "ref_mode"):
            args.ref_mode = "first"

        self.data_dir = args.data_dir
        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(
            #     self.img_size,
            #     scale=(0.9, 1.0), ratio=(1., 1.),
            #     interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.ref_transform = transforms.Compose([ # follow CLIP transform
            transforms.ToTensor(),
            # transforms.RandomResizedCrop(
            #     (224, 224),
            #     scale=(0.9, 1.0), ratio=(1., 1.),
            #     interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711]),
        ])
        self.ref_transform_mask = transforms.Compose([ # follow CLIP transform
            # transforms.RandomResizedCrop(
            #     (224, 224),
            #     scale=(0.9, 1.0), ratio=(1., 1.),
            #     interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
            transforms.ToTensor(),
        ])
        self.cond_transform = transforms.Compose([
            # transforms.RandomResizedCrop(
            #     self.img_size,
            #     scale=(0.9, 1.0), ratio=(1., 1.),
            #     interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.imagefolder_path = imagefolder_path
        self.imagefolder_mask_path = imagefolder_mask_path
        self.imagefolder_pose_path = imagefolder_pose_path
        self.folder_image_list = os.listdir(imagefolder_path)


    def add_mask_to_img(self, img, mask, img_key): #pil, pil
        if not img.size == mask.size:
            # import pdb; pdb.set_trace()
            # print(f'Reference image ({img_key}) size ({img.size}) is different from the mask size ({mask.size}), therefore try to resize the mask')
            mask = mask.resize(img.size) # resize the mask
        mask_array = np.array(mask)
        img_array = np.array(img)
        mask_array[mask_array < 127.5] = 0
        mask_array[mask_array > 127.5] = 1
        return Image.fromarray(img_array * mask_array), Image.fromarray(img_array * (1-mask_array)) # foreground, background

    def augmentation(self, frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame)
    
    def get_metadata(self, idx):
        img_idx, cap_idx = self.get_image_cap_index(idx)
        img_key = self.image_keys[img_idx]
        (caption_sample, tag, start,
         end, _) = self.get_caption_and_timeinfo_wrapper(
            img_idx, cap_idx)
        # get image or video frames
        # frames: (T, C, H, W),  is_video: binary tag
        frames, is_video = self.get_visual_data(img_idx)

        if isinstance(caption_sample, dict):
            caption = caption_sample["caption"]
        else:
            caption = caption_sample
            caption_sample = None

        # preparing outputs
        meta_data = {}
        meta_data['caption'] = caption  # raw text data, not tokenized
        meta_data['img_key'] = img_key
        meta_data['is_video'] = is_video  # True: video data, False: image data
        meta_data['pose_img'] = self.get_cond(img_idx, 'poses')
        if self.args.combine_use_mask:
            meta_data['mask_img'] = self.get_cond(img_idx, 'masks')
        ref_img_idx = self.get_reference_frame_idx(img_idx)
        meta_data['ref_img_key'] = self.image_keys[ref_img_idx]
        meta_data['reference_img'], _ = self.get_visual_data(ref_img_idx)
        if self.args.combine_use_mask:
            meta_data['mask_img_ref'] = self.get_cond(ref_img_idx, 'masks')
        meta_data['img'] = frames
        return meta_data
    
    def get_visual_data(self, img_idx):
        try:
            row = self.get_row_from_tsv(self.visual_tsv, img_idx)
            return self.str2img(row[-1]), False
        except Exception as e:
            raise ValueError(
                    f"{e}, in get_visual_data()")

    def __len__(self):
        return len(self.folder_image_list)

    def __getitem__(self, idx):
        image_name = self.folder_image_list[idx]
        image_path = os.path.join(self.imagefolder_path, image_name)
        image_mask_path = os.path.join(self.imagefolder_mask_path, ''.join(image_name.split('.')[:-1])+'-person.png')
        image_pose_path = os.path.join(self.imagefolder_pose_path, ''.join(image_name.split('.')[:-1])+'-pose.png')
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # add mask
        try: # if the background image has human mask
            image_mask = Image.open(image_mask_path)
            if not image_mask.mode == "RGB":
                image_mask = image_mask.convert("RGB")
            state = torch.get_rng_state()

            if self.args.eval_visu_changefore:
                image_aug = self.augmentation(image, self.ref_transform, state)
                image_mask_aug = self.augmentation(image_mask, self.ref_transform_mask, state)
                image_mask_aug[image_mask_aug > 0] = 1.
                image_foreground = image_aug * image_mask_aug
                output = {'reference_img':image_foreground, 'background_mask':1-image_mask_aug}
            else: # default to change the background
                image_aug = self.augmentation(image, self.transform, state)
                image_mask_aug = self.augmentation(image_mask, self.cond_transform, state)
                image_mask_aug[image_mask_aug > 0] = 1.
                image_background = image_aug * (1 - image_mask_aug)
                output = {'reference_img_controlnet': image_background, 'background_mask':1-image_mask_aug}

        except: # no mask provided, use ref mask in the following agent.py
            state = torch.get_rng_state()
            if self.args.eval_visu_changefore:
                image_aug = self.augmentation(image, self.ref_transform, state)  # full image
                image_mask_aug = torch.zeros_like(image_aug)
                output = {'reference_img':image_aug, 'background_mask':image_mask_aug}
            else:
                image_aug = self.augmentation(image, self.transform, state)  # full image
                image_mask_aug = torch.zeros_like(image_aug)
                output = {'reference_img_controlnet': image_aug, 'background_mask':image_mask_aug}


        if self.args.eval_visu_changepose:
            try:  # if the background image has human mask
                image_pose = Image.open(image_pose_path)
                if not image_pose.mode == "RGB":
                    image_pose = image_pose.convert("RGB")
                skeleton_img = self.augmentation(image_pose, self.cond_transform, state)
            except: #no pose, use the ref pose in the following agent.py
                skeleton_img = torch.zeros_like(image_background)
            output['cond_imgs'] = skeleton_img

        return output