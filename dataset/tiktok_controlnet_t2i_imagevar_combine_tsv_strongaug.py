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
        self.split = split

        super().__init__(
            args, yaml_file, split=split,
            size_frame=args.max_video_len, tokzr=None)
        self.data_dir = args.data_dir

        self.random_square_height = transforms.Lambda(lambda img: transforms.functional.crop(img, top=int(torch.randint(0, img.height - img.width, (1,)).item()), left=0, height=img.width, width=img.width))
        self.random_square_width = transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=int(torch.randint(0, img.width - img.height, (1,)).item()), height=img.height, width=img.height))

        min_crop_scale = 0.5 if self.args.strong_aug_stage2 and split=='train' else 0.9 # for stronger augmentation (only use in train)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.img_size,
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.cond_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.img_size,
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.ref_transform = transforms.Compose([ # follow CLIP transform
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                (224, 224),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711]),
        ])

        self.ref_transform_mask = transforms.Compose([ # follow CLIP transform
            transforms.RandomResizedCrop(
                (224, 224),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BICUBIC),
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

    def augmentation(self, frame, transform1, transform2=None, state=None):
        if state is not None:
            torch.set_rng_state(state)
        frame_transform1 = transform1(frame) if transform1 is not None else frame
        if transform2 is None:
            return frame_transform1
        else:
            return transform2(frame_transform1)
    
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

        # first check the size of the ref image
        ref_img_size = raw_data['reference_img'].size
        if self.args.strong_rand_stage2 and self.split=='train':
            if ref_img_size[0] > ref_img_size[1]: # width > height
                transform1 = self.random_square_width
            elif ref_img_size[0] < ref_img_size[1]:
                transform1 = self.random_square_height
            else:
                transform1 = None
        else:
            transform1 = None

        reference_img_controlnet = reference_img
        state = torch.get_rng_state()
        img = self.augmentation(img, transform1, self.transform, state)
        skeleton_img = self.augmentation(skeleton_img, transform1, self.cond_transform, state)
        reference_img_controlnet = self.augmentation(reference_img_controlnet, transform1, self.transform, state) # controlnet path input
        if getattr(self.args, 'refer_clip_preprocess', None):
            reference_img = self.preprocesser(reference_img).pixel_values[0]  # use clip preprocess
        else:
            reference_img = self.augmentation(reference_img, transform1, self.ref_transform, state)

        reference_img_vae = reference_img_controlnet
        if self.args.combine_use_mask:
            mask_img_ref = raw_data['mask_img_ref']
            ### first resize mask to the img size
            mask_img_ref = mask_img_ref.resize(ref_img_size)

            assert not getattr(self.args, 'refer_clip_preprocess', None) # mask not support the CLIP process
            reference_img_mask = self.augmentation(mask_img_ref, transform1, self.ref_transform_mask, state)
            reference_img_controlnet_mask = self.augmentation(mask_img_ref, transform1, self.cond_transform, state)  # controlnet path input

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
