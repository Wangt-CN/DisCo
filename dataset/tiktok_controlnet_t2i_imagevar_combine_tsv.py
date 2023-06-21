from config import *
import torch
import numpy as np
from PIL import Image
from dataset.tsv_cond_dataset import TsvCondImgCompositeDataset


class BaseDataset(TsvCondImgCompositeDataset):
    def __init__(self, args, yaml_file, split='train', preprocesser=None):
        self.img_size = getattr(args, 'img_full_size', args.img_size)
        self.basic_root_dir = BasicArgs.root_dir
        self.max_video_len = args.max_video_len
        assert self.max_video_len == 1
        self.fps = args.fps
        self.dataset = "TiktokDance-Image"
        self.preprocesser = preprocesser
        if not hasattr(args, "ref_mode"):
            args.ref_mode = "first"
        
        super().__init__(
            args, yaml_file, split=split,
            size_frame=args.max_video_len, tokzr=None)
        self.data_dir = args.data_dir
        self.img_ratio = (1., 1.) if not hasattr(self.args, 'img_ratio') or self.args.img_ratio is None else self.args.img_ratio
        self.img_scale = (1., 1.) if not split=='train' else getattr(self.args, 'img_scale', (0.9, 1.0)) # val set should keep scale=1.0 to avoid the random crop
        print(f'Current Data: {split}; Use image scale: {self.img_scale}; Use image ratio: {self.img_ratio}')

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.img_size,
                scale=self.img_scale, ratio=self.img_ratio,
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        try:
            self.ref_transform = transforms.Compose([ # follow CLIP transform
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    (224, 224),
                    scale=self.img_scale, ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                     [0.26862954, 0.26130258, 0.27577711]),
            ])
            self.ref_transform_mask = transforms.Compose([  # follow CLIP transform
                transforms.RandomResizedCrop(
                    (224, 224),
                    scale=self.img_scale, ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                transforms.ToTensor(),
            ])
        except:
            print('### Current pt version not support antialias, thus remove it! ###')
            self.ref_transform = transforms.Compose([ # follow CLIP transform
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    (224, 224),
                    scale=self.img_scale, ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                     [0.26862954, 0.26130258, 0.27577711]),
            ])
            self.ref_transform_mask = transforms.Compose([ # follow CLIP transform
                transforms.RandomResizedCrop(
                    (224, 224),
                    scale=self.img_scale, ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
        self.cond_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.img_size,
                scale=self.img_scale, ratio=self.img_ratio,
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

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
        if self.split == 'train':
            if getattr(self.args, 'max_train_samples', None):
                return min(self.args.max_train_samples, super().__len__())
            else:
                return super().__len__()
        else:
            if getattr(self.args, 'max_eval_samples', None):
                return min(self.args.max_eval_samples, super().__len__())
            else:
                return super().__len__()

    def __getitem__(self, idx):
        # try:
        raw_data = self.get_metadata(idx)
        # except Exception as e:
        #     print(e, self.yaml_file)
        img = raw_data['img']
        skeleton_img = raw_data['pose_img']
        reference_img = raw_data['reference_img']
        img_key = raw_data['img_key']
        ref_img_key = raw_data['ref_img_key']

        reference_img_controlnet = reference_img
        state = torch.get_rng_state()
        img = self.augmentation(img, self.transform, state)
        skeleton_img = self.augmentation(skeleton_img, self.cond_transform, state)
        reference_img_controlnet = self.augmentation(reference_img_controlnet, self.transform, state) # controlnet path input
        if getattr(self.args, 'refer_clip_preprocess', None):
            reference_img = self.preprocesser(reference_img).pixel_values[0]  # use clip preprocess
        else:
            reference_img = self.augmentation(reference_img, self.ref_transform, state)

        reference_img_vae = reference_img_controlnet
        if self.args.combine_use_mask:
            mask_img_ref = raw_data['mask_img_ref']
            assert not getattr(self.args, 'refer_clip_preprocess', None) # mask not support the CLIP process

            ### first resize mask to the img size
            mask_img_ref = mask_img_ref.resize(raw_data['reference_img'].size)

            reference_img_mask = self.augmentation(mask_img_ref, self.ref_transform_mask, state)
            reference_img_controlnet_mask = self.augmentation(mask_img_ref, self.cond_transform, state)  # controlnet path input

            # apply the mask
            reference_img = reference_img * reference_img_mask# foreground
            reference_img_vae = reference_img_vae * reference_img_controlnet_mask # foreground, but for vae
            reference_img_controlnet = reference_img_controlnet * (1 - reference_img_controlnet_mask)# background

        caption = raw_data['caption']
        outputs = {'img_key':img_key, 'input_text': caption, 'label_imgs': img, 'cond_imgs': skeleton_img, 'reference_img': reference_img, 'reference_img_controlnet':reference_img_controlnet, 'reference_img_vae':reference_img_vae}
        if self.args.combine_use_mask:
            outputs['background_mask'] = (1 - reference_img_mask)
            outputs['background_mask_controlnet'] = (1 - reference_img_controlnet_mask)

        return outputs
