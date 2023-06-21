from config import *
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os, math, re, json, copy
import numpy as np
from tqdm import tqdm
import random
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import code

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
        self.pose_normalize = getattr(args, 'pose_normalize', False)
        self.normalize_by_1st_frm = getattr(args, 'normalize_by_1st_frm', False)
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

        self.random_square_height = transforms.Lambda(lambda img: transforms.functional.crop(img, top=int(torch.randint(0, img.height - img.width, (1,)).item()), left=0, height=img.width, width=img.width))
        self.random_square_width = transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=int(torch.randint(0, img.width - img.height, (1,)).item()), height=img.height, width=img.height))

        self.fix_square_height = transforms.Lambda(lambda img: transforms.functional.crop(img, top=int((img.height - img.width)/4), left=0, height=img.width, width=img.width))
        min_crop_scale = 0.5 if self.args.strong_aug_stage1 and split=='train' else 1.0 # for stronger augmentation (only use in train)
        # min_crop_scale = 0.5
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.img_size,
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
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

        self.ref_transform_mask = transforms.Compose([  # follow CLIP transform
            transforms.RandomResizedCrop(
                (224, 224),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        self.cond_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.img_size,
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.total_num_videos = 340
        self.anno_path = 'GIT/{:05d}/labels/{:04d}.txt'
        self.image_path = '{:05d}/images/{:04d}.png'
        self.anno_pose_path = '{:05d}/openpose_json/{:04d}.png.json'
        self.ref_mask_path = '{:05d}/masks/{:04d}.png'

        self.image_path_web = '{}/{}'
        self.ref_image_path_web = '{}/{}'
        self.anno_pose_path_web = '{}/openpose_json/{}.json'
        self.ref_mask_path_web = '{}/groundsam/{}.mask.jpg'

        self.image_paths_list = []
        self.ref_image_paths_list = []
        self.ref_pose_paths_list = []
        self.anno_list = []
        self.anno_pose_list = []
        self.anno_init_pose_list = []
        self.mask_list = []

        ft_video_idx = getattr(args, 'ft_idx', '001_1.57_2.17_1x1') # default elon
        if split == 'train': # for training video
            # video_idx = ['001_1.57_2.17_1x1'] # elon mask 2
            # video_idx = ['007_7.36_7.44_1x1'] # 007
            # video_idx = ['001_1.57_2.17_9x16', '001_11.46_11.54_9x16', '001_5.37_5.44_9x16', '001_8.14_8.27_9x16']  # elon mask 1+2+3+4
            video_idx = [ft_video_idx] # 007
            dataset_prefix = self.args.web_data_root
        else: # for pose video
            video_idx = [335, 137]
            # ref_video_idx = '001_1.57_2.17_1x1' # 007
            # ref_video_idx = '007_7.36_7.44_1x1' # 007
            ref_video_idx = ft_video_idx # 007
            dataset_prefix = self.args.tiktok_data_root

        ### use specific image video frame to ft, sample in the fly
        for vid in video_idx:

            if isinstance(vid, str): # web image data
                assert split == 'train'
                folder_path = os.path.join(args.web_data_root, '{}').format(vid)
                if os.path.exists(os.path.join(folder_path, 'image_list.txt')):
                    image_files_idx = list(open(os.path.join(folder_path, 'image_list.txt')))
                    image_files_idx = [file.strip() for file in image_files_idx]
                else:
                    image_files_idx = os.listdir(folder_path)
                    image_files_idx = [file for file in image_files_idx if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
            else:
                assert split == 'val'
                folder_path = os.path.join(args.tiktok_data_root, '{:05d}/images/').format(vid)
                files = os.listdir(folder_path)
                image_files_idx = [file for file in files if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]

                folder_path_ref = os.path.join(args.web_data_root, '{}').format(ref_video_idx)
                if os.path.exists(os.path.join(folder_path_ref, 'image_list.txt')):
                    image_files_idx_ref = list(open(os.path.join(folder_path_ref, 'image_list.txt')))
                    image_files_idx_ref = [file.strip() for file in image_files_idx_ref]
                else:
                    image_files_idx_ref = os.listdir(folder_path_ref)
                    image_files_idx_ref = [file for file in image_files_idx_ref if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]

            ### if use specific number of frames as training sample
            ft_img_num = getattr(self.args, 'ft_img_num', 0) # default value
            ft_one_ref_image = getattr(self.args, 'ft_one_ref_image', True) # default value

            if ft_img_num>0: # randomly sample n image as training sample
                random.seed(self.args.seed)
                frm_list = random.sample(image_files_idx, ft_img_num)
                frm_list.sort()
                print('Random Sample Frames: {}'.format(frm_list))
            else:
                frm_list = image_files_idx

            # FT: [Kevin Ver] always chooose the 1st frame as the referece image
            # TODO: WT Revision: if t = n, then reference frm could be between frame(0)~frame(n-1)
            if ft_one_ref_image: # only use 1st frame as the reference
                if split == 'train':
                    while len(frm_list) < 128:
                        frm_list *= 2 # enlarge the dataset to fix the dataloader bug
                    for fid in frm_list:
                        image_fname = os.path.join(dataset_prefix, self.image_path_web.format(vid, fid))
                        ref_image_fname = os.path.join(dataset_prefix, self.image_path_web.format(vid, frm_list[0]))
                        ref_mask_fname = os.path.join(dataset_prefix, self.ref_mask_path_web.format(vid, frm_list[0]))  # only choose 1st frame as ref frame
                        anno_pose_fname = os.path.join(dataset_prefix, self.anno_pose_path_web.format(vid, fid))

                        # anno_fname = self.anno_path.format(vid, fid)
                        anno_fname = os.path.join(dataset_prefix, vid + '__' + fid)
                        self.image_paths_list.append(image_fname)
                        self.ref_image_paths_list.append(ref_image_fname)
                        self.anno_pose_list.append(anno_pose_fname)
                        self.anno_list.append(anno_fname)
                        self.mask_list.append(ref_mask_fname)

                else: # use tiktok pose for dancing
                    ref_image_fname = os.path.join(self.args.web_data_root, self.image_path_web.format(ref_video_idx, image_files_idx_ref[0]))
                    ref_mask_fname = os.path.join(self.args.web_data_root, self.ref_mask_path_web.format(ref_video_idx, image_files_idx_ref[0]))  # only choose 1st frame as ref frame
                    ref_pose_fname = os.path.join(self.args.web_data_root, self.anno_pose_path_web.format(ref_video_idx, image_files_idx_ref[0]))

                    refpose_files = os.listdir(os.path.join(args.tiktok_data_root, '{:05d}/images/').format(vid))
                    refpose_image_files = [file for file in refpose_files if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
                    refpose_image_files_len = len(refpose_image_files)
                    for fid in range(1, refpose_image_files_len):
                        image_fname = os.path.join(dataset_prefix, self.image_path.format(vid, fid % refpose_image_files_len))
                        anno_pose_fname = os.path.join(dataset_prefix, self.anno_pose_path.format(vid, fid % refpose_image_files_len))  # change to the next frame's pose
                        anno_fname = os.path.join(dataset_prefix, self.anno_path.format(vid, fid % refpose_image_files_len))
                        anno_init_pose_fname =  os.path.join(dataset_prefix, self.anno_pose_path.format(vid, 1))
                        self.image_paths_list.append(image_fname)
                        self.ref_image_paths_list.append(ref_image_fname)
                        self.ref_pose_paths_list.append(ref_pose_fname)
                        self.anno_pose_list.append(anno_pose_fname)
                        self.anno_list.append(anno_fname)
                        self.anno_init_pose_list.append(anno_init_pose_fname)
                        self.mask_list.append(ref_mask_fname)

                    # random sort the input list for visualization
                    random_idx = list(range(len(self.image_paths_list)))
                    random.shuffle(random_idx)
                    self.image_paths_list =  [self.image_paths_list[i] for i in random_idx]
                    self.ref_image_paths_list = [self.ref_image_paths_list[i] for i in random_idx]
                    self.ref_pose_paths_list = [self.ref_pose_paths_list[i] for i in random_idx]

                    self.anno_pose_list = [self.anno_pose_list[i] for i in random_idx]
                    self.anno_list = [self.anno_list[i] for i in random_idx]
                    self.mask_list = [self.mask_list[i] for i in random_idx]
                    self.anno_init_pose_list = [self.anno_init_pose_list[i] for i in random_idx]


            else:
                if split == 'train':
                    while len(frm_list) < 12:
                        frm_list *= 2 # enlarge the dataset to fix the dataloader bug
                    for fid in frm_list:
                        # ref image use random sample
                        ref_resample_flag = True
                        while ref_resample_flag:
                            ref_fid = random.choice(frm_list)
                            if os.path.exists(os.path.join(dataset_prefix, self.ref_mask_path_web.format(vid, ref_fid))):
                                ref_resample_flag = False

                        image_fname = os.path.join(dataset_prefix, self.image_path_web.format(vid, fid))
                        ref_image_fname = os.path.join(dataset_prefix, self.image_path_web.format(vid, ref_fid))
                        anno_pose_fname = os.path.join(dataset_prefix, self.anno_pose_path_web.format(vid, fid))

                        # anno_fname = self.anno_path.format(vid, fid)
                        anno_fname = os.path.join(dataset_prefix, vid + '__' + fid)
                        ref_mask_fname = os.path.join(dataset_prefix, self.ref_mask_path_web.format(vid, ref_fid))  # only choose 1st frame as ref frame
                        self.image_paths_list.append(image_fname)
                        self.ref_image_paths_list.append(ref_image_fname)
                        self.anno_pose_list.append(anno_pose_fname)
                        self.anno_list.append(anno_fname)
                        self.mask_list.append(ref_mask_fname)


                else: # use tiktok pose for dancing, randomly choose one frame as ref
                    # ref image use random sample
                    ref_resample_flag = True
                    while ref_resample_flag:
                        ref_fid = random.choice(image_files_idx_ref)
                        if os.path.exists(os.path.join(self.args.web_data_root, self.ref_mask_path_web.format(ref_video_idx, ref_fid))):
                            ref_resample_flag = False

                    ref_image_fname = os.path.join(self.args.web_data_root, self.image_path_web.format(ref_video_idx, ref_fid))
                    ref_mask_fname = os.path.join(self.args.web_data_root, self.ref_mask_path_web.format(ref_video_idx, ref_fid))  # only choose 1st frame as ref frame
                    ref_pose_fname = os.path.join(self.args.web_data_root, self.anno_pose_path_web.format(ref_video_idx, ref_fid))

                    refpose_files = os.listdir(os.path.join(args.tiktok_data_root, '{:05d}/images/').format(vid))
                    refpose_image_files = [file for file in refpose_files if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
                    refpose_image_files_len = len(refpose_image_files)
                    for fid in range(1, refpose_image_files_len):
                        image_fname = os.path.join(dataset_prefix, self.image_path.format(vid, fid % refpose_image_files_len))
                        anno_pose_fname = os.path.join(dataset_prefix, self.anno_pose_path.format(vid, fid % refpose_image_files_len))  # change to the next frame's pose
                        anno_fname = os.path.join(dataset_prefix, self.anno_path.format(vid, fid % refpose_image_files_len))
                        anno_init_pose_fname =  os.path.join(dataset_prefix, self.anno_pose_path.format(vid, 1))
                        self.image_paths_list.append(image_fname)
                        self.ref_image_paths_list.append(ref_image_fname)
                        self.ref_pose_paths_list.append(ref_pose_fname)
                        self.anno_pose_list.append(anno_pose_fname)
                        self.anno_list.append(anno_fname)
                        self.anno_init_pose_list.append(anno_init_pose_fname)
                        self.mask_list.append(ref_mask_fname)

                    # random sort the input list for visualization
                    random_idx = list(range(len(self.image_paths_list)))
                    random.shuffle(random_idx)
                    self.image_paths_list =  [self.image_paths_list[i] for i in random_idx]
                    self.ref_image_paths_list = [self.ref_image_paths_list[i] for i in random_idx]
                    self.ref_pose_paths_list = [self.ref_pose_paths_list[i] for i in random_idx]

                    self.anno_pose_list = [self.anno_pose_list[i] for i in random_idx]
                    self.anno_list = [self.anno_list[i] for i in random_idx]
                    self.mask_list = [self.mask_list[i] for i in random_idx]
                    self.anno_init_pose_list = [self.anno_init_pose_list[i] for i in random_idx]


        self.num_images = len(self.image_paths_list)
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

    def load_pose_ann(self, anno_pose_path):
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
        return pose_without_visibletag

    def get_valid_jnt_index(self, image, pose):
        img_width, img_height = image.size
        valid_pose_index = np.asarray(
            [item[0] > -1 and item[0] < img_width and item[1] > -1 and item[1] < img_height for item in pose])
        # when compute affine transform, skip joints on the arms and legs
        valid_pose_index[10] = False  # RWrist
        valid_pose_index[9] = False  # LWrist
        valid_pose_index[16] = False  # RAnkle
        valid_pose_index[15] = False  # LAnkle
        # add Neck to joint set
        if valid_pose_index[5] == True and valid_pose_index[6] == True:
            pose.append(list((np.asarray(pose[5]) + np.asarray(pose[6])) / 2))
            valid_pose_index = np.append(valid_pose_index, True)
        else:
            pose.append([-1, -1])
            valid_pose_index = np.append(valid_pose_index, False)
            # add Pelvis to joint set
        if valid_pose_index[11] == True and valid_pose_index[12] == True:
            pose.append(list((np.asarray(pose[11]) + np.asarray(pose[12])) / 2))
            valid_pose_index = np.append(valid_pose_index, True)
        else:
            pose.append([-1, -1])
            valid_pose_index = np.append(valid_pose_index, False)
        return valid_pose_index, pose

    def pose_norm(self, image, input_ref_pose, input_init_pose, input_pose):
        ref_pose = copy.deepcopy(input_ref_pose)
        init_pose = copy.deepcopy(input_init_pose)
        pose = copy.deepcopy(input_pose)
        # compute transformation between ref_pose and init_pose, and apply transforms to pose
        # get index for the valid joints
        valid_ref_pose, ref_pose = self.get_valid_jnt_index(image, ref_pose)  # reference pose is the pose of reference image
        valid_init_pose, init_pose = self.get_valid_jnt_index(image, init_pose)  # init pose is the initial target pose (1st frame)

        # get common valid joints
        common_valid_pose = valid_ref_pose * valid_init_pose
        # if limited joints available, skip normalization
        if np.sum(common_valid_pose * 1) < 5:
            t_x, t_y = 0, 0
            s_x, s_y = 1, 1
        else:
            np_ref_pose = np.asarray(ref_pose)
            np_init_pose = np.asarray(init_pose)
            np_ref_pose_set = np_ref_pose[common_valid_pose]
            np_init_pose_set = np_init_pose[common_valid_pose]
            # get affine trans matrix: transform from init pose to ref pose
            affine_matrix, _ = cv2.estimateAffinePartial2D(np_init_pose_set, np_ref_pose_set)
            t_x = affine_matrix[0, 2]
            t_y = affine_matrix[1, 2]
            s_x = np.sign(affine_matrix[0, 0]) * np.sqrt(
                affine_matrix[0, 0] * affine_matrix[0, 0] + affine_matrix[0, 1] * affine_matrix[0, 1])
            s_y = np.sign(affine_matrix[1, 1]) * np.sqrt(
                affine_matrix[1, 0] * affine_matrix[1, 0] + affine_matrix[1, 1] * affine_matrix[1, 1])

        # apply scale + translate to the given pose
        for i in range(len(pose)):
            if pose[i][0] >= 0 or pose[i][1] >= 0:
                pose[i][0] = int(pose[i][0] * s_x + t_x)
                pose[i][1] = int(pose[i][1] * s_y + t_y)
        return pose

    def normalize_mask(self, mask):
        mask[mask>=0.001] = 1
        mask[mask<0.001] = 0
        return mask


    def get_img_txt_pair(self, idx):

        # img_path = os.path.join(self.args.tiktok_data_root, self.image_paths_list[idx % self.num_images])
        # ref_img_path = os.path.join(self.args.tiktok_data_root, self.ref_image_paths_list[idx % self.num_images])
        # anno_pose_path = os.path.join(self.args.tiktok_data_root, self.anno_pose_list[idx % self.num_images])
        # anno_path = os.path.join(self.args.tiktok_data_root, self.anno_list[idx % self.num_images])
        # ref_mask_path = os.path.join(self.args.tiktok_data_root, self.mask_list[idx % self.num_images])

        img_path = self.image_paths_list[idx % self.num_images]
        ref_img_path = self.ref_image_paths_list[idx % self.num_images]

        anno_pose_path = self.anno_pose_list[idx % self.num_images]
        anno_path = self.anno_list[idx % self.num_images]
        ref_mask_path = self.mask_list[idx % self.num_images]

        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        if 'youtube' in ref_mask_path:
            ref_mask = self.load_mask(ref_mask_path)
        else:
            ref_mask = Image.open(ref_mask_path).convert('RGB')

        ref_image = Image.open(ref_img_path)
        if not ref_image.mode == "RGB":
            ref_image = ref_image.convert("RGB")

        ref_original_width, ref_original_height = ref_image.size
        ref_mask = ref_mask.resize(ref_image.size) # resize the mask to img
        if self.pose_normalize and self.split is not 'train':
            ref_image = ref_image.resize(image.size)
        ref_new_width, ref_new_height = ref_image.size

        if 'youtube' in anno_pose_path:
            img_key = self.anno_list[idx % self.num_images]
        else:
            anno = list(open(anno_path))
            img_key = json.loads(anno[0].strip())['image_key']
        """
        example:
        {"num_region": 6, "image_key": "TiktokDance_00001_0002.png", "image_split": "00001", "image_read_error": false}
        {"box_id": 0, "class_name": "aerosol_can", "norm_bbox": [0.5, 0.5, 1.0, 1.0], "conf": 0.0, "region_caption": "a woman with an orange dress with butterflies on her shirt.", "caption_conf": 0.9404542168542169}
        {"box_id": 1, "class_name": "person", "norm_bbox": [0.46692365407943726, 0.4977584183216095, 0.9338473081588745, 0.995516836643219], "conf": 0.912740170955658, "region_caption": "a woman with an orange dress with butterflies on her shirt.", "caption_conf": 0.9404542168542169}
        {"box_id": 2, "class_name": "butterfly", "norm_bbox": [0.2368704378604889, 0.5088028907775879, 0.1444256454706192, 0.04199704900383949], "conf": 0.8738771677017212, "region_caption": "a brown butterfly sitting on an orange background.", "caption_conf": 0.9297735554473283}
        {"box_id": 3, "class_name": "butterfly", "norm_bbox": [0.6688584089279175, 0.5137135982513428, 0.11311062425374985, 0.05455022677779198], "conf": 0.8287128806114197, "region_caption": "a brown butterfly sitting on an orange wall.", "caption_conf": 0.9264783379302365}
        {"box_id": 4, "class_name": "blouse", "norm_bbox": [0.4692786931991577, 0.6465241312980652, 0.9283269643783569, 0.6027728319168091], "conf": 0.6851752400398254, "region_caption": "a woman wearing an orange shirt with butterflies on it.", "caption_conf": 0.9978814544264754}
        {"box_id": 5, "class_name": "short_pants", "norm_bbox": [0.44008955359458923, 0.8769687414169312, 0.8799525499343872, 0.2431662678718567], "conf": 0.6741859316825867, "region_caption": "a person wearing an orange shirt and grey sweatpants.", "caption_conf": 0.9731313580907464}
        """

        # Load detected openpose keypoint json file
        pose_xy = self.load_pose_ann(anno_pose_path)
        if self.pose_normalize and self.split is not 'train':
            ref_pose_path = self.ref_pose_paths_list[idx]
            anno_init_pose_path = self.anno_init_pose_list[idx]
            ref_pose_xy = self.load_pose_ann(ref_pose_path)
            for xy in ref_pose_xy:
                if xy[0]!=-1 and xy[1]!=-1:
                    xy[0] = float(xy[0])/ref_original_width*ref_new_width
                    xy[1] = float(xy[1])/ref_original_height*ref_new_height


            if self.normalize_by_1st_frm:
                init_pose_xy = self.load_pose_ann(anno_init_pose_path)
                pose_xy = self.pose_norm(image, ref_pose_xy, init_pose_xy, pose_xy)
            else:
                pose_xy = self.pose_norm(image, ref_pose_xy, pose_xy, pose_xy)


        # convert coordinates to skeleton image
        #skeleton_img = self.coco2openpose(image, pose_without_visibletag)
        skeleton_img = self.coco2openpose(image, pose_xy)

        # preparing outputs
        meta_data = {}
        meta_data['img_key'] = img_key
        meta_data['is_video'] = False
        meta_data['skeleton_img'] = skeleton_img
        meta_data['reference_img'] = ref_image
        meta_data['img'] = image
        meta_data['ref_mask'] = ref_mask
        return meta_data

    def augmentation(self, frame, transform1, transform2=None, state=None):
        if state is not None:
            torch.set_rng_state(state)
        frame_transform1 = transform1(frame) if transform1 is not None else frame
        if transform2 is None:
            return frame_transform1
        else:
            return transform2(frame_transform1)


    def __getitem__(self, idx):
        raw_data = self.get_img_txt_pair(idx)

        img = raw_data['img']
        skeleton_img = raw_data['skeleton_img']
        reference_img = raw_data['reference_img']
        img_key = raw_data['img_key']

        # first check the size of the ref image
        ref_img_size = raw_data['reference_img'].size
        if self.args.strong_rand_stage2 and self.split == 'train':
            if ref_img_size[0] > ref_img_size[1]: # width > height
                transform1 = self.random_square_width
            elif ref_img_size[0] < ref_img_size[1]:
                transform1 = self.random_square_height
            else:
                transform1 = None
        else:
            transform1 = self.fix_square_height


        reference_img_controlnet = reference_img
        state = torch.get_rng_state()
        img = self.augmentation(img, transform1, self.transform, state)

        skeleton_img = self.augmentation(skeleton_img, transform1, self.cond_transform, state)
        reference_img_controlnet = self.augmentation(reference_img_controlnet, transform1, self.transform, state)

        reference_img_vae = reference_img_controlnet
        if getattr(self.args, 'refer_clip_preprocess', None):
            reference_img = self.preprocesser(reference_img).pixel_values[0] # use clip preprocess
        else:
            reference_img = self.augmentation(reference_img, transform1, self.ref_transform, state)

        if self.args.combine_use_mask:
            mask_img_ref = raw_data['ref_mask']
            assert not getattr(self.args, 'refer_clip_preprocess', None) # mask not support the CLIP process

            # ### first resize mask to the img size
            mask_img_ref = mask_img_ref.resize(ref_img_size)

            reference_img_mask = self.augmentation(mask_img_ref, transform1, self.ref_transform_mask, state)
            reference_img_controlnet_mask = self.augmentation(mask_img_ref, transform1, self.cond_transform, state)  # controlnet path input

            # linshi wangtan
            reference_img_mask = self.normalize_mask(reference_img_mask)
            reference_img_controlnet_mask = self.normalize_mask(reference_img_controlnet_mask)

            # apply the mask
            reference_img = reference_img * reference_img_mask# foreground
            reference_img_vae = reference_img_vae * reference_img_controlnet_mask # foreground, but for vae
            reference_img_controlnet = reference_img_controlnet * (1 - reference_img_controlnet_mask)# background


        outputs = {'img_key':img_key, 'label_imgs': img, 'cond_imgs': skeleton_img, 'reference_img': reference_img, 'reference_img_controlnet':reference_img_controlnet, 'reference_img_vae':reference_img_vae}
        if self.args.combine_use_mask:
            outputs['background_mask'] = (1 - reference_img_mask)
            outputs['background_mask_controlnet'] = (1 - reference_img_controlnet_mask)

        return outputs

