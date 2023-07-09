from config import *
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os, math, re, json
import numpy as np
from tqdm import tqdm
import PIL
from PIL import Image
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import random

training_templates_smallest = [
    'photo of a sks {}',
]

reg_templates_smallest = [
    'photo of a {}',
]
coco_joints_name = ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', 'Left Wrist',
            'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle', 'Pelvis', 'Neck']


class BaseDataset(Dataset):
    def __init__(self, args, yaml_file, split='train', preprocesser=None):
        self.dataset = "tiktok"
        self.args = args
        self.split = split
        self.is_train = split == "train"
        self.is_composite = False
        self.on_memory = getattr(args, 'on_memory', False)
        self.img_size = getattr(args, 'img_full_size', args.img_size)
        self.max_video_len = 1 ## Todo, now it is image-based dataloader
        self.size_frame = 1 ## Todo
        self.yaml_file = yaml_file
        self.stickwidth = 4
        self.preprocesser = preprocesser
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.img_size,
                scale=(1.0, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.ref_transform = transforms.Compose([ # follow CLIP transform
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                (224, 224),
                scale=(1.0, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711]),
        ])

        self.ref_transform_mask = transforms.Compose([  # follow CLIP transform
            transforms.RandomResizedCrop(
                (224, 224),
                scale=(1.0, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        self.cond_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.img_size,
                scale=(1.0, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.total_num_videos = 340
        self.image_path = '{}/{}'
        self.ref_image_path = '{:05d}/images/{:04d}.png'
        # self.anno_pose_path = '{}/openpose_json/{:04d}.jpg.json'
        self.anno_pose_path = '{:05d}/openpose_json/{:04d}.png.json'
        self.anno_path = 'GIT/{:05d}/labels/{:04d}.txt'
        self.ref_mask_path = '{}/groundsam/{}.mask.jpg'

        self.ref_image_path_web = '{}/{}'
        self.anno_pose_path_web = '{}/openpose_json/{}.json'

        self.image_paths_list = []
        self.ref_image_paths_list = []
        self.pose_image_paths_list = []
        self.anno_list = []
        self.anno_pose_list = []
        self.mask_list = []
        self.file_name_id = []

        assert split == 'val'

        # for specific choose the video
        ref_fg_folder = './demo_data/fg'
        ref_bg_folder = './demo_data/bg'
        ref_pose_folder = './demo_data/pose'
        bg_list = os.listdir(os.path.join(ref_bg_folder, 'images'))


        ref_fg_files_list = os.listdir(os.path.join(ref_fg_folder, 'images'))
        # Kevin Ver: always chooose the 1st frame as the referece image
        # TODO: WT Revision: if t = n, then reference frm could be between frame(0)~frame(n-1)
        for ref_img_name in ref_fg_files_list:
            ref_image_path = os.path.join(ref_fg_folder, 'images', ref_img_name)
            ref_mask_path = os.path.join(ref_fg_folder, 'masks', ref_img_name)

            pose_files_list = os.listdir(ref_pose_folder)
            for pose_img_name in pose_files_list:
                pose_image_path = os.path.join(ref_pose_folder, pose_img_name)

                self.pose_image_paths_list.append(pose_image_path) # actually have no gt file, just use the target pose file
                self.ref_image_paths_list.append(ref_image_path)
                self.mask_list.append(ref_mask_path)
                self.file_name_id.append(f'ref{ref_img_name}--pose{pose_img_name}')

        self.bgref_ref_image_paths_list = []
        self.bgref_mask_list = []
        for ref_bg_name in bg_list:
            ref_image_fname = os.path.join(ref_bg_folder, 'images', ref_bg_name)
            ref_mask_fname = os.path.join(ref_bg_folder, 'masks', ref_bg_name)

            self.bgref_ref_image_paths_list.append(ref_image_fname)
            self.bgref_mask_list.append(ref_mask_fname)
        self.bgref_num_images = len(self.bgref_ref_image_paths_list)


        self.num_images = len(self.ref_image_paths_list)
        self._length = self.num_images 
        print('number of samples:',self._length)


    def __len__(self):
        if self.split == 'train':
            if getattr(self.args, 'max_train_samples', None):
                return min(self.args.max_train_samples, self._length)
            else:
                return self._length
        else:
            if getattr(self.args, 'max_eval_samples', None):
                return min(self.args.max_eval_samples, self._length)
            else:
                return self._length

    def normalize_mask(self, mask):
        mask[mask>=0.001] = 1
        mask[mask<0.001] = 0
        return mask

    # draw the body keypoint and lims
    def draw_bodypose(self, canvas, pose):
        canvas = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        canvas = np.zeros_like(canvas)

        for i in range(18):
            x, y = pose[i][0:2]
            if x>=0 and y>=0: 
                cv2.circle(canvas, (int(x), int(y)), 4, self.colors[i], thickness=-1)
                # cv2.putText(canvas, '%d'%(i), (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        for limb_idx in range(17):
            cur_canvas = canvas.copy()
            index_a = self.limbSeq[limb_idx][0]-1
            index_b = self.limbSeq[limb_idx][1]-1

            if pose[index_a][0]<0 or pose[index_b][0]<0 or pose[index_a][1]<0 or pose[index_b][1]<0:
                continue

            Y = [pose[index_a][0], pose[index_b][0]]
            X = [pose[index_a][1], pose[index_b][1]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), self.stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[limb_idx])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        # Convert color space from BGR to RGB
        # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        # Create PIL image object from numpy array
        canvas = Image.fromarray(canvas)
        return canvas


    def coco2openpose(self, img, coco_keypoints):

        # coco keypoints: [x1,y1,v1,...,xk,yk,vk]       (k=17)
        #     ['Nose', Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
        #      'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
        # openpose keypoints: [y1,...,yk], [x1,...xk]   (k=18, with Neck)
        #     ['Nose' (0), *'Neck'* (1), 'Rsho' (2), 'Relb' (3), 'Rwri' (4), 'Lsho' (5), 'Lelb' (6), 'Lwri' (7),'Rhip' (8),
        #      'Rkne' (9), 'Rank' (10), 'Lhip' (11), 'Lkne' (12), 'Lank' (13), 'Reye' (14), 'Leye' (15), 'Rear' (16), 'Lear' (17)]

        openpose_keypoints = [
            coco_keypoints[0], # Nose (0)
            list((np.asarray(coco_keypoints[5]) + np.asarray(coco_keypoints[6]))/2), # Neck (1)
            coco_keypoints[6], # RShoulder (2)
            coco_keypoints[8], # RElbow (3)
            coco_keypoints[10], # RWrist (4)
            coco_keypoints[5], # LShoulder (5)
            coco_keypoints[7], # LElbow (6)
            coco_keypoints[9], # LWrist (7)
            coco_keypoints[12], # RHip (8)
            coco_keypoints[14], # RKnee (9)
            coco_keypoints[16], # RAnkle (10)
            coco_keypoints[11], # LHip (11)
            coco_keypoints[13], # LKnee (12)
            coco_keypoints[15], # LAnkle (13)
            coco_keypoints[2], # REye (14)
            coco_keypoints[1], # LEye (15)
            coco_keypoints[4], # REar (16)
            coco_keypoints[3], # LEar (17)
        ] 
        return self.draw_bodypose(img, openpose_keypoints)

    def load_image(self, path):
        if os.path.exists(path):
            image = Image.open(path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
        else:
            image = None
        return image

    def load_mask(self, path):
        try:
            if os.path.exists(path):
                img = self.load_image(path)
                img = np.asarray(img)
                bk = img[:, :, :] == [68, 0, 83]
                fg = (bk == False)
                fg = fg * 255.0
                mask = fg.astype(np.uint8)
                ipl_mask = Image.fromarray(mask)
            else:
                ipl_mask = None
        except Exception as e:
            print(e)
            ipl_mask = None
        return ipl_mask

    def load_mask_tiktok(self, path):
        if os.path.exists(path):
            image = Image.open(path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
        else:
            image = None
        return image

    def load_openpose(self, anno_pose_path, ref_image):
        # Load detected openpose keypoint json file
        pose_without_visibletag = []
        f = open(anno_pose_path, 'r')
        d = json.load(f)
        # if there is a valid openpose skeleton, load it
        if len(d) > 0:
            for j in range(17):
                x = d[0]['keypoints'][j][0]
                y = d[0]['keypoints'][j][1]
                pose_without_visibletag.append([x, y])
        else:  # if there is not valid openpose skeleton, add a dummy one
            for j in range(17):
                x = -1
                y = -1
                pose_without_visibletag.append([x, y])

                # convert coordinates to skeleton image
        skeleton_img = self.coco2openpose(ref_image, pose_without_visibletag)
        return skeleton_img

    def get_img_txt_pair(self, idx):
        img_path = self.ref_image_paths_list[idx % self.num_images]
        ref_img_path = self.ref_image_paths_list[idx % self.num_images]
        ref_mask_path = self.mask_list[idx % self.num_images]
        anno_pose_path = self.pose_image_paths_list[idx % self.num_images]
        img_key =  self.file_name_id[idx % self.num_images]

        ref_mask = self.load_mask_tiktok(ref_mask_path)
        ref_image = Image.open(ref_img_path)
        if not ref_image.mode == "RGB":
            ref_image = ref_image.convert("RGB")
        ref_mask = ref_mask.resize(ref_image.size) # resize the mask to img
        img = Image.open(img_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")

        # Load detected openpose keypoint json file
        pose_without_visibletag = []
        f = open(anno_pose_path,'r')
        d = json.load(f)
        # if there is a valid openpose skeleton, load it
        if len(d)>0:
            for j in range(17):
                x = d[0]['keypoints'][j][0]
                y = d[0]['keypoints'][j][1]
                pose_without_visibletag.append([x,y])
        else: # if there is not valid openpose skeleton, add a dummy one
            for j in range(17):
                x = -1
                y = -1
                pose_without_visibletag.append([x,y])      

        # convert coordinates to skeleton image
        skeleton_img = self.coco2openpose(ref_image, pose_without_visibletag)

        # preparing outputs
        meta_data = {}
        meta_data['img'] = img
        meta_data['img_key'] = img_key
        meta_data['is_video'] = False
        meta_data['skeleton_img'] = skeleton_img
        meta_data['reference_img'] = ref_image
        meta_data['ref_mask'] = ref_mask
        return meta_data

    def augmentation(self, frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame)

    def __getitem__(self, idx):
        raw_data = self.get_img_txt_pair(idx)
        img = raw_data['img']
        skeleton_img = raw_data['skeleton_img']
        reference_img = raw_data['reference_img']
        img_key = raw_data['img_key']


        ### random sample background
        ref_bg_idx = random.choice(list(range(0, self.bgref_num_images)))
        ref_bg_img_path = self.bgref_ref_image_paths_list[ref_bg_idx]
        ref_bg_ref_mask_path = os.path.join(self.args.web_data_root, self.bgref_mask_list[ref_bg_idx])
        ref_bg_image = Image.open(os.path.join(self.args.web_data_root, ref_bg_img_path))
        ref_bg_ref_mask = self.load_mask_tiktok(ref_bg_ref_mask_path)
        if not ref_bg_image.mode == "RGB":
            ref_bg_image = ref_bg_image.convert("RGB")
        ref_bg_ref_mask = ref_bg_ref_mask.resize(ref_bg_image.size)  # resize the mask to img

        img_key = img_key + '_{}'.format(ref_bg_img_path.split('/')[-1])


        reference_img_controlnet = reference_img
        state = torch.get_rng_state()
        img = self.augmentation(img, self.transform, state)
        skeleton_img = self.augmentation(skeleton_img, self.cond_transform, state)
        reference_img_controlnet = self.augmentation(reference_img_controlnet, self.transform, state)
        ref_bg_reference_img_controlnet = self.augmentation(ref_bg_image, self.transform, state)

        reference_img_vae = reference_img_controlnet
        if getattr(self.args, 'refer_clip_preprocess', None):
            reference_img = self.preprocesser(reference_img).pixel_values[0] # use clip preprocess
        else:
            reference_img = self.augmentation(reference_img, self.ref_transform)

        if self.args.combine_use_mask:
            mask_img_ref = raw_data['ref_mask']
            assert not getattr(self.args, 'refer_clip_preprocess', None) # mask not support the CLIP process

            # ### first resize mask to the img size
            mask_img_ref = mask_img_ref.resize(raw_data['reference_img'].size)

            reference_img_mask = self.augmentation(mask_img_ref, self.ref_transform_mask, state)
            reference_img_controlnet_mask = self.augmentation(mask_img_ref, self.cond_transform, state)  # controlnet path input
            ref_bg_reference_img_controlnet_mask = self.augmentation(ref_bg_ref_mask, self.cond_transform, state)  # controlnet path input

            # linshi wangtan
            reference_img_mask = self.normalize_mask(reference_img_mask)
            reference_img_controlnet_mask = self.normalize_mask(reference_img_controlnet_mask)
            ref_bg_reference_img_controlnet_mask = self.normalize_mask(ref_bg_reference_img_controlnet_mask)

            # apply the mask
            reference_img = reference_img * reference_img_mask# foreground
            reference_img_vae = reference_img_vae * reference_img_controlnet_mask # foreground, but for vae
            # reference_img_controlnet = reference_img_controlnet * (1 - reference_img_controlnet_mask)# background
            reference_img_controlnet = ref_bg_reference_img_controlnet * (1 - ref_bg_reference_img_controlnet_mask)  # background

        outputs = {'img_key':img_key, 'label_imgs': img, 'cond_imgs': skeleton_img, 'reference_img': reference_img, 'reference_img_controlnet':reference_img_controlnet, 'reference_img_vae':reference_img_vae}
        if self.args.combine_use_mask:
            outputs['background_mask'] = (1 - reference_img_mask)
            outputs['background_mask_controlnet'] = (1 - reference_img_controlnet_mask)
        outputs['save_filename'] = self.file_name_id[idx % self.num_images]

        return outputs


    def preprocess_input(self, reference_img, fg_mask, ref_bg_image, bg_mask, skeleton_img):
        fg_mask = fg_mask.resize(reference_img.size)


        reference_img_controlnet = reference_img
        state = torch.get_rng_state()
        img = self.augmentation(reference_img, self.transform, state)
        skeleton_img = self.augmentation(skeleton_img, self.cond_transform, state)
        reference_img_controlnet = self.augmentation(reference_img_controlnet, self.transform, state)
        ref_bg_reference_img_controlnet = self.augmentation(ref_bg_image, self.transform, state)
        reference_img_vae = reference_img_controlnet
        reference_img = self.augmentation(reference_img, self.ref_transform)


        reference_fg_mask = self.augmentation(fg_mask, self.ref_transform_mask, state)
        reference_fg_controlnet_mask = self.augmentation(fg_mask, self.cond_transform, state)  # controlnet path input
        ref_bg_reference_img_controlnet_mask = self.augmentation(bg_mask, self.cond_transform, state)  # controlnet path input

        # linshi wangtan
        reference_img_mask = self.normalize_mask(reference_fg_mask)
        reference_img_controlnet_mask = self.normalize_mask(reference_fg_controlnet_mask)
        ref_bg_reference_img_controlnet_mask = self.normalize_mask(ref_bg_reference_img_controlnet_mask)

        # apply the mask
        reference_img = reference_img * reference_img_mask  # foreground
        reference_img_vae = reference_img_vae * reference_img_controlnet_mask  # foreground, but for vae
        reference_img_controlnet = ref_bg_reference_img_controlnet * (1 - ref_bg_reference_img_controlnet_mask)  # background
        outputs = {'label_imgs': img.unsqueeze(0), 'cond_imgs': skeleton_img.unsqueeze(0), 'reference_img': reference_img.unsqueeze(0), 'reference_img_controlnet':reference_img_controlnet.unsqueeze(0), 'reference_img_vae':reference_img_vae.unsqueeze(0)}
        return outputs


    def preprocess_masked_input(self, reference_img_masked, ref_bg_image_masked, skeleton_img):
        reference_img = reference_img_masked
        ref_bg_image = ref_bg_image_masked

        def pil2binary_fg(img):
            xx = np.array(img.convert('L'))
            xx[xx > 0] = 255
            xx[xx < 255] = 0
            return xx
        def pil2binary_bg(img):
            xx = np.array(img.convert('L'))
            xx[xx == 0] = 255
            xx[xx < 255] = 0
            return xx

        fg_mask = Image.fromarray(pil2binary_fg(reference_img)).convert('RGB')
        bg_mask = Image.fromarray(pil2binary_bg(ref_bg_image)).convert('RGB')

        fg_mask = fg_mask.resize(reference_img.size)

        reference_img_controlnet = reference_img
        state = torch.get_rng_state()
        img = self.augmentation(reference_img, self.transform, state)
        skeleton_img = self.augmentation(skeleton_img, self.cond_transform, state)
        reference_img_controlnet = self.augmentation(reference_img_controlnet, self.transform, state)
        ref_bg_reference_img_controlnet = self.augmentation(ref_bg_image, self.transform, state)
        reference_img_vae = reference_img_controlnet
        reference_img = self.augmentation(reference_img, self.ref_transform)


        reference_fg_mask = self.augmentation(fg_mask, self.ref_transform_mask, state)
        reference_fg_controlnet_mask = self.augmentation(fg_mask, self.cond_transform, state)  # controlnet path input
        ref_bg_reference_img_controlnet_mask = self.augmentation(bg_mask, self.cond_transform, state)  # controlnet path input

        # linshi wangtan
        reference_img_mask = self.normalize_mask(reference_fg_mask)
        reference_img_controlnet_mask = self.normalize_mask(reference_fg_controlnet_mask)
        ref_bg_reference_img_controlnet_mask = self.normalize_mask(ref_bg_reference_img_controlnet_mask)

        # apply the mask
        reference_img = reference_img * reference_img_mask  # foreground
        reference_img_vae = reference_img_vae * reference_img_controlnet_mask  # foreground, but for vae
        reference_img_controlnet = ref_bg_reference_img_controlnet * (1 - ref_bg_reference_img_controlnet_mask)  # background
        outputs = {'label_imgs': img.unsqueeze(0), 'cond_imgs': skeleton_img.unsqueeze(0), 'reference_img': reference_img.unsqueeze(0), 'reference_img_controlnet':reference_img_controlnet.unsqueeze(0), 'reference_img_vae':reference_img_vae.unsqueeze(0)}
        return outputs